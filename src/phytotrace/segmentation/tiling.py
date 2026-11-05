"""High-resolution overlapping image tiling and mask stitching for localized leaf-miner damage."""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def generate_overlapping_tiles(
    image_shape: Tuple[int, int],  # (height, width)
    tile_size: Tuple[int, int] = (640, 640),
    overlap: int = 128,
) -> List[Tuple[int, int, int, int]]:
    """Compute bounding box coordinates (xmin, ymin, xmax, ymax) for overlapping tiles.
    
    Ensures complete canvas coverage with no gaps along right and bottom borders.
    """
    height, width = image_shape[:2]
    tile_h, tile_w = tile_size

    if height <= tile_h and width <= tile_w:
        return [(0, 0, width, height)]

    step_y = max(1, tile_h - overlap)
    step_x = max(1, tile_w - overlap)

    y_starts = list(range(0, max(1, height - tile_h + 1), step_y))
    if not y_starts or y_starts[-1] + tile_h < height:
        y_starts.append(max(0, height - tile_h))

    x_starts = list(range(0, max(1, width - tile_w + 1), step_x))
    if not x_starts or x_starts[-1] + tile_w < width:
        x_starts.append(max(0, width - tile_w))

    boxes: List[Tuple[int, int, int, int]] = []
    for ys in y_starts:
        for xs in x_starts:
            ye = min(height, ys + tile_h)
            xe = min(width, xs + tile_w)
            boxes.append((xs, ys, xe, ye))

    # Deduplicate in case edges overlap identically
    unique_boxes = sorted(list(set(boxes)), key=lambda b: (b[1], b[0]))
    return unique_boxes


def extract_tile_crops(
    image: np.ndarray,
    tile_boxes: List[Tuple[int, int, int, int]],
) -> List[np.ndarray]:
    """Extract sub-image crops corresponding to tile bounding boxes."""
    crops = []
    for xmin, ymin, xmax, ymax in tile_boxes:
        crop = image[ymin:ymax, xmin:xmax]
        crops.append(crop)
    return crops


def stitch_tile_masks(
    canvas_shape: Tuple[int, int],  # (height, width)
    tile_masks: List[np.ndarray],
    tile_boxes: List[Tuple[int, int, int, int]],
    blend_mode: str = "max",
) -> np.ndarray:
    """Stitch tile-level binary or probability masks back onto the full canvas.
    
    Args:
        canvas_shape: (height, width) of the original full-resolution photograph.
        tile_masks: List of 2D uint8 or float32 masks for each tile.
        tile_boxes: List of (xmin, ymin, xmax, ymax) coordinate tuples.
        blend_mode: 'max' for union/max pooling, 'average' for linear blending.
        
    Returns:
        Full-size stitched mask matching canvas_shape.
    """
    height, width = canvas_shape[:2]
    is_float = len(tile_masks) > 0 and np.issubdtype(tile_masks[0].dtype, np.floating)

    if is_float:
        canvas = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32) if blend_mode == "average" else None

        for mask, (xmin, ymin, xmax, ymax) in zip(tile_masks, tile_boxes):
            th, tw = ymax - ymin, xmax - xmin
            if mask.shape[:2] != (th, tw):
                mask_res = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_LINEAR)
            else:
                mask_res = mask

            if blend_mode == "max":
                canvas[ymin:ymax, xmin:xmax] = np.maximum(canvas[ymin:ymax, xmin:xmax], mask_res)
            else:
                canvas[ymin:ymax, xmin:xmax] += mask_res
                weight_map[ymin:ymax, xmin:xmax] += 1.0

        if blend_mode == "average" and weight_map is not None:
            canvas = np.divide(canvas, np.maximum(weight_map, 1e-6))
        return canvas

    else:
        # Binary uint8 mask
        canvas = np.zeros((height, width), dtype=np.uint8)
        for mask, (xmin, ymin, xmax, ymax) in zip(tile_masks, tile_boxes):
            th, tw = ymax - ymin, xmax - xmin
            if mask.shape[:2] != (th, tw):
                mask_res = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            else:
                mask_res = mask
            canvas[ymin:ymax, xmin:xmax] = cv2.bitwise_or(canvas[ymin:ymax, xmin:xmax], mask_res)
        return canvas


def compare_tiling_vs_resizing(
    image_shape: Tuple[int, int],  # (height, width)
    tile_size: Tuple[int, int] = (640, 640),
    overlap: int = 128,
) -> Dict[str, Any]:
    """Analytical evaluation comparing whole-image downsampling vs. overlapping tile inference.
    
    Provides concrete trade-off figures between spatial fidelity for narrow leaf mines
    and inference latency on Apple Silicon MPS.
    """
    h, w = image_shape[:2]
    tiles = generate_overlapping_tiles((h, w), tile_size=tile_size, overlap=overlap)
    num_tiles = len(tiles)

    # Scale factor when downsampling to tile_size (e.g., 640x640)
    scale_y = tile_size[0] / max(1, h)
    scale_x = tile_size[1] / max(1, w)
    full_resize_scale = min(scale_x, scale_y)

    # Biological mine dimensions (approx 1.0mm - 2.0mm serpentine mine on a 150mm field of view)
    # At native 3024x4032, a fine 1mm mine occupies ~15-20 pixels.
    native_mine_width_px = 18.0
    downsampled_mine_width_px = round(native_mine_width_px * full_resize_scale, 2)
    tiled_mine_width_px = native_mine_width_px  # 1.0x native resolution

    return {
        "native_dimensions": f"{w}x{h} ({round(w * h / 1e6, 2)} MP)",
        "full_resize_target": f"{tile_size[1]}x{tile_size[0]}",
        "full_resize_scale_factor": round(full_resize_scale, 4),
        "full_resize_effective_resolution_pct": round(full_resize_scale * 100.0, 2),
        "downsampled_mine_feature_width_px": downsampled_mine_width_px,
        "downsampled_feature_status": (
            "SUB_PIXEL_RISK (Lost or blurred into veins during downsampling)" if downsampled_mine_width_px < 3.5
            else "DEGRADED (Fine frass line lost)" if downsampled_mine_width_px < 7.0
            else "RETAINED"
        ),
        "tiling_mode_tile_count": num_tiles,
        "tiling_mode_tile_size": f"{tile_size[1]}x{tile_size[0]}",
        "tiling_mode_overlap_px": overlap,
        "tiling_effective_resolution_pct": 100.0,
        "tiled_mine_feature_width_px": tiled_mine_width_px,
        "tiled_feature_status": "FULL_FIDELITY (Preserves serpentine frass & margins)",
        "mps_estimated_latency_ratio": f"{num_tiles}x vs 1x (parallelizable on Apple MPS)",
        "recommendation": (
            "Use overlapping tiling (640x640, 128px overlap) for high-accuracy damage quantification and early mine screening; "
            "use single-image resize for fast mobile screening of advanced blotch mines."
        ),
    }

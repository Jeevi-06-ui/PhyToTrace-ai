"""Leaf-mining damage segmentation module based on lightweight YOLO segmentation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from threading import Lock

from phytotrace.segmentation.tiling import (
    extract_tile_crops,
    generate_overlapping_tiles,
    stitch_tile_masks,
)
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.mine_segmenter")


class MineSegmenter:
    """Wrapper around lightweight YOLO segmentation model for leaf-miner damage localization."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.10,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._prediction_lock = Lock()
        self.is_trained = False
        
        if weights_path:
            p = resolve_path(weights_path)
            if p.is_file():
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(str(p))
                    if self.model.task != "segment" or self.model.names != {0: "mine_damage"}:
                        raise ValueError("Expected a single-class mine_damage segmentation checkpoint")
                    self.is_trained = True
                    logger.info("Loaded trained YOLO mine segmenter from %s", p)
                except Exception as e:
                    logger.warning("Could not load YOLO model from %s: %s", p, e)
            else:
                logger.warning("YOLO weights file does not exist at %s", p)

    def predict(
        self,
        image_bgr: np.ndarray,
        leaf_mask: Optional[np.ndarray] = None,
        use_tiling: bool = False,
        tile_size: Tuple[int, int] = (640, 640),
        tile_overlap: int = 128,
        *, diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Detect and segment leaf miner mine damage regions.
        
        Args:
            image_bgr: Input BGR image array.
            leaf_mask: Optional binary leaf mask to intersect with damage regions.
            use_tiling: Whether to use high-resolution overlapping tiling for small mines.
            tile_size: Dimensions of tiles when use_tiling=True.
            tile_overlap: Pixel overlap between adjacent tiles.
            
        Returns:
            Dict containing:
                'damage_detected': bool
                'regions': List of dicts (region_id, bbox, polygon, confidence, area_pixels)
                'combined_mask': uint8 array (union of all mine masks)
                'total_mined_pixels': int
                'inference_mode': str ('tiled' or 'full_image')
                'status': str
        """
        h, w = image_bgr.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        regions = []

        if not self.is_trained or self.model is None:
            return {
                "damage_detected": None,
                "regions": [],
                "combined_mask": combined_mask,
                "total_mined_pixels": 0,
                "inference_mode": "none",
                "status": "model_not_trained",
            }

        # ---------------------------------------------------------------------
        # TILING MODE: For high-resolution preservation of narrow serpentine mines
        # ---------------------------------------------------------------------
        if use_tiling and (h > tile_size[0] or w > tile_size[1]):
            tile_boxes = generate_overlapping_tiles((h, w), tile_size=tile_size, overlap=tile_overlap)
            tile_crops = extract_tile_crops(image_bgr, tile_boxes)

            tile_masks = []
            reg_idx = 1

            for crop, (xmin, ymin, xmax, ymax) in zip(tile_crops, tile_boxes):
                ch, cw = crop.shape[:2]
                t_mask = np.zeros((ch, cw), dtype=np.uint8)

                results = self._predict_native(
                    source=crop,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                )

                if results and len(results) > 0 and results[0].masks is not None:
                    res = results[0]
                    masks_data = res.masks.data.cpu().numpy()
                    boxes_data = res.boxes.xyxy.cpu().numpy()
                    conf_data = res.boxes.conf.cpu().numpy()

                    for i in range(len(masks_data)):
                        m = (masks_data[i] > 0.5).astype(np.uint8) * 255
                        m_res = m
                        t_mask = cv2.bitwise_or(t_mask, m_res)

                        # Map bounding box back to original canvas
                        bx1, by1, bx2, by2 = boxes_data[i]
                        orig_bbox = [int(bx1 + xmin), int(by1 + ymin), int(bx2 + xmin), int(by2 + ymin)]

                        # Map polygon back to original canvas
                        contours, _ = cv2.findContours(m_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        poly = []
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            poly = [[int(pt[0][0] + xmin), int(pt[0][1] + ymin)] for pt in c]

                        regions.append({
                            "region_id": f"R_{reg_idx:02d}",
                            "bbox": orig_bbox,
                            "confidence": float(round(conf_data[i], 4)),
                            "area_pixels": int(np.count_nonzero(m_res)),
                            "polygon": poly,
                        })
                        reg_idx += 1

                tile_masks.append(t_mask)

            combined_mask = stitch_tile_masks((h, w), tile_masks, tile_boxes, blend_mode="max")
            if leaf_mask is not None:
                combined_mask = cv2.bitwise_and(combined_mask, leaf_mask)

            total_mined = int(np.count_nonzero(combined_mask))
            return {
                "damage_detected": len(regions) > 0,
                "regions": regions,
                "combined_mask": combined_mask,
                "total_mined_pixels": total_mined,
                "inference_mode": f"tiled ({len(tile_boxes)} tiles)",
                "status": "success",
            }

        # ---------------------------------------------------------------------
        # STANDARD FULL-IMAGE MODE: Single forward pass
        # ---------------------------------------------------------------------
        results = self._predict_native(
            source=image_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        candidates = []
        rejected_candidates = []
        raw_masks = {}
        if results:
            result = results[0]
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    bbox = [int(v) for v in box.xyxy[0].tolist()]
                    confidence = float(box.conf.item())
                    class_id = int(box.cls.item())
                    candidate_id = f"RAW_{i+1:02d}"
                    mask = (result.masks.data[i].cpu().numpy() > 0).astype(np.uint8) * 255 if result.masks is not None else None
                    if mask is not None and mask.shape != (h, w):
                        raise ValueError("YOLO mask does not match original image dimensions")

                    area = cv2.countNonZero(mask) if mask is not None else 0
                    clipped = cv2.bitwise_and(mask, leaf_mask) if mask is not None and leaf_mask is not None else mask
                    overlap = cv2.countNonZero(clipped) if clipped is not None and leaf_mask is not None else None
                    containment = overlap / area if area and overlap is not None else None
                    box_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (w * h)
                    reasons = []
                    if not area:
                        reasons.append("EMPTY_MASK")
                    if box_ratio > 0.35:
                        reasons.append("EXCEEDS_FRAME_RATIO")
                    if containment is not None and containment < 0.58:
                        reasons.append("POOR_LEAF_CONTAINMENT")
                    # Measure the size floor on the model's 640-pixel long edge.
                    effective_area = (overlap if overlap is not None else area) * (640 / max(w, h)) ** 2
                    if area and effective_area < 1:
                        reasons.append("BELOW_MODEL_PIXEL_SIZE")
                    polygons = self._polygons(clipped) if area else []
                    if area and not polygons:
                        reasons.append("NO_POLYGON")
                    
                    x_c, y_c, bw_c, bh_c = cv2.boundingRect(clipped) if (area and overlap) else (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                    in_leaf_bbox = [int(x_c), int(y_c), int(x_c + bw_c), int(y_c + bh_c)]

                    candidate = {
                        "candidate_id": candidate_id,
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "bbox": bbox,
                        "in_leaf_bbox": in_leaf_bbox,
                        "confidence": confidence,
                        "raw_pixels": area,
                        "in_leaf_pixels": overlap,
                        "leaf_containment_pct": round(100 * containment, 2) if containment is not None else None,
                        "raw_polygons": self._polygons(mask) if area else [],
                        "clipped_polygons": polygons,
                        "reasons": reasons,
                        "decision": "rejected" if reasons else "passed_geometry",
                    }
                    if diagnostics and mask is not None:
                        raw_masks[candidate_id] = mask
                    candidates.append(candidate)
                    if reasons:
                        rejected_candidates.append(candidate)
                        continue

                    cv2.bitwise_or(combined_mask, clipped, dst=combined_mask)
                    x, y, bw, bh = cv2.boundingRect(clipped)
                    regions.append({
                        "region_id": f"R_{len(regions)+1:02d}",
                        "candidate_id": candidate_id,
                        "bbox": [x, y, x + bw, y + bh],
                        "confidence": confidence,
                        "area_pixels": cv2.countNonZero(clipped),
                        "polygon": polygons[0],
                        "polygons": polygons,
                    })

        return {
            "damage_detected": bool(regions),
            "regions": regions,
            "candidates": candidates,
            "rejected_candidates": rejected_candidates,
            "raw_candidates_count": len(candidates),
            "accepted_candidates_count": len(regions),
            "rejected_candidates_count": len(rejected_candidates),
            "combined_mask": combined_mask,
            "total_mined_pixels": cv2.countNonZero(combined_mask),
            "inference_mode": "full_image",
            "status": "success",
            **({"raw_masks": raw_masks} if diagnostics else {}),
        }

    def _predict_native(self, **kwargs):
        # Ultralytics owns the letterbox transform and removes its padding.
        with self._prediction_lock:
            imgsz = kwargs.pop("imgsz", None)
            if imgsz is None:
                src = kwargs.get("source")
                if isinstance(src, np.ndarray):
                    sh, sw = src.shape[:2]
                    imgsz = min(1280, max(640, int(np.ceil(max(sh, sw) / 32) * 32)))
                else:
                    imgsz = 640
            return self.model.predict(retina_masks=True, imgsz=imgsz, **kwargs)

    @staticmethod
    def _polygons(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1, 2).tolist() for c in sorted(contours, key=cv2.contourArea, reverse=True)
                if len(c) >= 3 and cv2.contourArea(c) > 0]

"""I/O and rasterization utilities for CVAT polygon annotations and binary masks."""

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import cv2
import numpy as np

from phytotrace.segmentation.schemas import (
    AnnotationStatus,
    ImageSegmentationCatalogRecord,
    PolygonAnnotation,
    SegmentationClass,
)
from phytotrace.segmentation.validators import validate_polygon_geometry
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.segmentation.io")

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class MatchResolutionStatus(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"


@dataclass
class MatchResult:
    """Detailed outcome of filename resolution against candidate pool."""
    query: str
    matched_target: Optional[str]
    status: MatchResolutionStatus
    candidates: List[str]
    message: str = ""


def resolve_filename_match(
    query_name: str,
    candidate_pool: Sequence[str],
) -> MatchResult:
    """Safely and deterministically match a query filename against a candidate pool.
    
    Resolution hierarchy:
    1. Tries exact basename match first.
    2. If that fails, tries Unicode-safe case-insensitive basename matching using casefold().
    3. If that fails, tries stem-equivalence matching for standard image extensions (.jpg vs .jpeg).
    
    Integrity rules:
    - Requires exactly one matching candidate entry.
    - Rejects ambiguous collisions (reporting all conflicting candidate names).
    - Rejects genuinely unmatched filenames.
    - Preserves original filenames and image IDs.
    - Does not rename, move, delete or modify any photographs.
    - Does not silently discard annotations.
    """
    if not candidate_pool:
        return MatchResult(
            query=query_name,
            matched_target=None,
            status=MatchResolutionStatus.UNMATCHED,
            candidates=[],
            message=f"Candidate pool is empty; cannot resolve '{query_name}'.",
        )

    q_base = Path(query_name).name

    # 1. Exact basename match
    exact_matches = [c for c in candidate_pool if Path(c).name == q_base]
    if len(exact_matches) == 1:
        return MatchResult(
            query=query_name,
            matched_target=exact_matches[0],
            status=MatchResolutionStatus.EXACT,
            candidates=exact_matches,
            message=f"Exact basename match: '{q_base}' == '{exact_matches[0]}'.",
        )
    elif len(exact_matches) > 1:
        return MatchResult(
            query=query_name,
            matched_target=None,
            status=MatchResolutionStatus.AMBIGUOUS,
            candidates=exact_matches,
            message=f"Ambiguous exact matches: '{q_base}' matches multiple identical candidates: {exact_matches}.",
        )

    # 2. Unicode-safe case-insensitive basename matching using casefold()
    q_folded = q_base.casefold()
    casefold_matches = [c for c in candidate_pool if Path(c).name.casefold() == q_folded]
    if len(casefold_matches) == 1:
        return MatchResult(
            query=query_name,
            matched_target=casefold_matches[0],
            status=MatchResolutionStatus.NORMALIZED,
            candidates=casefold_matches,
            message=f"Unicode casefold normalized match: '{q_base}' -> '{casefold_matches[0]}'.",
        )
    elif len(casefold_matches) > 1:
        return MatchResult(
            query=query_name,
            matched_target=None,
            status=MatchResolutionStatus.AMBIGUOUS,
            candidates=casefold_matches,
            message=f"Ambiguous casefold collision: '{q_base}' matches multiple candidates: {casefold_matches}.",
        )

    # 3. Stem-equivalence matching for image extensions (.jpg vs .jpeg, .JPG vs .JPEG)
    q_path = Path(q_base)
    q_suffix_folded = q_path.suffix.casefold()
    if q_suffix_folded in IMAGE_EXTENSIONS:
        q_stem_folded = q_path.stem.casefold()
        stem_matches = [
            c for c in candidate_pool
            if Path(c).stem.casefold() == q_stem_folded and Path(c).suffix.casefold() in IMAGE_EXTENSIONS
        ]
        if len(stem_matches) == 1:
            return MatchResult(
                query=query_name,
                matched_target=stem_matches[0],
                status=MatchResolutionStatus.NORMALIZED,
                candidates=stem_matches,
                message=f"Normalized stem-equivalence match: '{q_base}' -> '{stem_matches[0]}'.",
            )
        elif len(stem_matches) > 1:
            return MatchResult(
                query=query_name,
                matched_target=None,
                status=MatchResolutionStatus.AMBIGUOUS,
                candidates=stem_matches,
                message=f"Ambiguous stem collision: '{q_base}' matches multiple candidates: {stem_matches}.",
            )

    # 4. Genuinely unmatched
    return MatchResult(
        query=query_name,
        matched_target=None,
        status=MatchResolutionStatus.UNMATCHED,
        candidates=[],
        message=f"Genuinely unmatched: no candidate matching '{q_base}'.",
    )


def find_coco_annotation_entry(
    parsed_annotations: Dict[str, Dict[str, Any]],
    target_filename: str,
) -> Tuple[Optional[Dict[str, Any]], MatchResult]:
    """Look up image annotations in parsed COCO dict using resolve_filename_match.
    
    Returns:
        (annotation_entry_dict_or_None, MatchResult)
    """
    match_res = resolve_filename_match(target_filename, list(parsed_annotations.keys()))
    if match_res.status in (MatchResolutionStatus.EXACT, MatchResolutionStatus.NORMALIZED):
        return parsed_annotations.get(match_res.matched_target), match_res
    return None, match_res


def polygons_to_binary_mask(
    polygons: List[PolygonAnnotation],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize a list of polygon annotations into a binary uint8 mask (0 or 255).
    
    If polygons is empty, returns an all-zero binary mask of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if not poly.points or len(poly.points) < 3:
            continue
        # Clip coordinates to valid image dimensions to prevent OpenCV segmentation faults
        pts = np.array(
            [[np.clip(pt[0], 0, width - 1), np.clip(pt[1], 0, height - 1)] for pt in poly.points],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask


def binary_mask_to_polygons(
    mask: np.ndarray,
    min_area: float = 10.0,
    epsilon: float = 1.5,
) -> List[List[Tuple[float, float]]]:
    """Convert binary mask into simplified polygon contours."""
    if mask is None or np.sum(mask) == 0:
        return []

    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
        if len(approx) >= 3:
            pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            polygons.append(pts)

    return polygons


def parse_coco_segmentation_json(
    coco_json_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Parse standard CVAT-exported COCO 1.0 JSON into polygon annotations per image filename.
    
    Returns:
        Dict mapping image_filename -> {
            'leaf': [PolygonAnnotation, ...],
            'mine_damage': [PolygonAnnotation, ...],
            'image_info': {'id': int, 'width': int, 'height': int},
            'present_in_coco': bool
        }
    """
    coco_p = resolve_path(coco_json_path)
    if not coco_p.is_file():
        raise FileNotFoundError(f"COCO annotations file not found at {coco_p}")

    with open(coco_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Map category IDs
    cat_map: Dict[int, SegmentationClass] = {}
    for cat in data.get("categories", []):
        cname = str(cat.get("name", "")).strip().lower()
        if "leaf" in cname and "mine" not in cname:
            cat_map[cat["id"]] = SegmentationClass.LEAF
        elif "mine" in cname:
            cat_map[cat["id"]] = SegmentationClass.MINE_DAMAGE
        else:
            logger.warning("Unrecognized segmentation category '%s'; treating as mine_damage.", cname)
            cat_map[cat["id"]] = SegmentationClass.MINE_DAMAGE

    # 2. Map image IDs to filenames and dimensions (every image in data['images'] is tracked)
    img_map: Dict[int, dict] = {}
    results: Dict[str, Dict[str, Any]] = {}

    for img in data.get("images", []):
        fname = Path(img.get("file_name", "")).name
        img_info = {
            "id": int(img.get("id", 0)),
            "filename": fname,
            "width": int(img.get("width", 0)),
            "height": int(img.get("height", 0)),
        }
        img_map[img_info["id"]] = img_info
        # Initialize entry for every registered image in COCO
        results[fname] = {
            "leaf": [],
            "mine_damage": [],
            "image_info": img_info,
            "present_in_coco": True,
        }

    # 3. Parse annotations
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in img_map:
            logger.warning("Annotation references unknown image_id %s; skipping.", img_id)
            continue

        fname = img_map[img_id]["filename"]
        cat_id = ann.get("category_id")
        seg_class = cat_map.get(cat_id, SegmentationClass.MINE_DAMAGE)

        seg = ann.get("segmentation", [])
        # In COCO 1.0, polygon segmentation is a list of coordinate lists: [[x1, y1, x2, y2, ...]]
        # In RLE, it's a dict with 'counts'; flag and skip or warn
        if isinstance(seg, dict):
            logger.warning("RLE segmentation format not supported for %s; use CVAT Polygon COCO export.", fname)
            continue

        for poly_coords in seg:
            if not isinstance(poly_coords, list) or len(poly_coords) < 6:
                continue

            pts: List[Tuple[float, float]] = []
            for i in range(0, len(poly_coords) - 1, 2):
                try:
                    px = float(poly_coords[i])
                    py = float(poly_coords[i + 1])
                    pts.append((px, py))
                except (ValueError, TypeError):
                    continue

            if len(pts) >= 3:
                poly_ann = PolygonAnnotation(
                    points=pts,
                    label=seg_class,
                    area_pixels=float(ann.get("area", 0.0)),
                    bbox=tuple(ann.get("bbox", (0, 0, 0, 0))),
                    attributes=ann.get("attributes", {}),
                )
                if seg_class == SegmentationClass.LEAF:
                    results[fname]["leaf"].append(poly_ann)
                else:
                    results[fname]["mine_damage"].append(poly_ann)

    return results

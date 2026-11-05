"""Quality validation checks for leaf and mine segmentation annotations."""

from typing import Dict, List, Tuple
import cv2
import numpy as np

from phytotrace.segmentation.schemas import PolygonAnnotation, SegmentationClass


def validate_polygon_geometry(
    polygon: PolygonAnnotation,
    image_width: int,
    image_height: int,
) -> Tuple[bool, List[str]]:
    """Validate that a polygon is topologically valid and bounded within image frame."""
    errors = []
    if len(polygon.points) < 3:
        errors.append(f"Polygon has fewer than 3 vertices ({len(polygon.points)}).")
        return False, errors

    pts = np.array(polygon.points, dtype=np.float32)
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    if min_x < 0 or min_y < 0 or max_x > image_width or max_y > image_height:
        errors.append(
            f"Polygon extends outside image bounds [0, {image_width}] x [0, {image_height}]: "
            f"bbox=[{min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f}]."
        )

    # Check non-zero contour area
    area = cv2.contourArea(pts.astype(np.int32))
    if area < 5.0:
        errors.append(f"Polygon area is degenerate ({area:.1f} pixels < 5 px).")

    return len(errors) == 0, errors


def validate_mine_within_leaf_mask(
    mine_mask: np.ndarray,
    leaf_mask: np.ndarray,
    max_allowed_outside_ratio: float = 0.05,
) -> Tuple[bool, float, List[str]]:
    """Verify that mine damage pixels reside within the leaf canopy boundary.
    
    Returns:
        (is_valid, outside_ratio, error_messages)
    """
    errors = []
    mine_pixels = np.sum(mine_mask > 0)
    if mine_pixels == 0:
        return True, 0.0, []

    leaf_pixels = np.sum(leaf_mask > 0)
    if leaf_pixels == 0:
        errors.append("Leaf mask is empty while mine mask contains damage pixels.")
        return False, 1.0, errors

    # Pixels where mine is present but leaf is NOT present
    outside_mask = (mine_mask > 0) & (leaf_mask == 0)
    outside_pixels = np.sum(outside_mask)
    outside_ratio = float(outside_pixels / mine_pixels)

    if outside_ratio > max_allowed_outside_ratio:
        errors.append(
            f"Damage annotation extends outside leaf boundary: {outside_pixels}/{mine_pixels} px "
            f"({outside_ratio*100:.1f}% > {max_allowed_outside_ratio*100:.1f}% allowed)."
        )

    return len(errors) == 0, round(outside_ratio, 4), errors

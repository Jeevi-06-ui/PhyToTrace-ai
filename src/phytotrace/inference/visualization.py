"""Explainable visualization and overlay generation for PhytoTrace AI."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Standardized color scheme (BGR format for OpenCV)
COLOR_LEAF_BGR = (87, 139, 46)        # Forest green
COLOR_TUTA_BGR = (0, 165, 255)        # Amber
COLOR_LIRIOMYZA_BGR = (209, 206, 0)   # Cyan / Turquoise
COLOR_UNCERTAIN_BGR = (80, 127, 255)   # Coral
COLOR_GRAY_BGR = (128, 128, 128)      # Neutral gray


def draw_damage_overlay(
    image_bgr: np.ndarray,
    leaf_mask: Optional[np.ndarray],
    regions: List[Dict[str, any]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Create an explainable visual overlay depicting segmented leaf and regional damage.
    
    Args:
        image_bgr: Original input BGR image.
        leaf_mask: Binary leaf mask (uint8).
        regions: List of detected region dictionaries with 'polygon', 'region_id', 'evidence'.
        alpha: Blending transparency.
        
    Returns:
        Annotated BGR image with translucent overlays, contours, and region tags.
    """
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    # 1. Subtle green contour for leaf boundary
    if leaf_mask is not None and np.count_nonzero(leaf_mask) > 0:
        leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, leaf_contours, -1, COLOR_LEAF_BGR, thickness=2)

    # 2. Draw damage regions with class-specific evidence colors
    for r in regions:
        poly_pts = r.get("polygon", [])
        if not poly_pts:
            # Fallback to bbox if polygon empty
            bbox = r.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                poly_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if poly_pts:
            pts_arr = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
            
            # Select color based on pest evidence
            evidence = r.get("pest_evidence", "uncertain")
            if evidence == "tuta":
                col = COLOR_TUTA_BGR
            elif evidence == "liriomyza":
                col = COLOR_LIRIOMYZA_BGR
            elif evidence == "mixed":
                col = (180, 105, 255) # Pink / Magenta
            else:
                col = COLOR_UNCERTAIN_BGR

            # Fill translucent mask
            cv2.fillPoly(overlay, [pts_arr], col)
            # Outline boundary
            cv2.polylines(overlay, [pts_arr], isClosed=True, color=col, thickness=2)

            # Label region ID
            reg_id = r.get("region_id", "R")
            if len(poly_pts) > 0:
                cx, cy = poly_pts[0]
                cv2.putText(
                    overlay,
                    reg_id,
                    (cx, max(15, cy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    reg_id,
                    (cx, max(15, cy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    # Blend overlay with original
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)
    return blended


def create_side_by_side_comparison(
    original_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    decision_summary: str = "",
    damage_percentage: Optional[float] = None,
    model_version: str = "PhytoTrace v0.1.0",
) -> np.ndarray:
    """Generate a side-by-side comparison board with header banner and diagnostics."""
    h1, w1 = original_bgr.shape[:2]
    h2, w2 = annotated_bgr.shape[:2]
    
    # Normalize heights
    target_h = max(h1, h2, 480)
    orig_resized = cv2.resize(original_bgr, (int(w1 * (target_h / h1)), target_h))
    ann_resized = cv2.resize(annotated_bgr, (int(w2 * (target_h / h2)), target_h))
    
    # Combined width
    combined_body = np.hstack([orig_resized, ann_resized])
    comb_w = combined_body.shape[1]
    
    # Create top info header bar
    header_h = 70
    header = np.zeros((header_h, comb_w, 3), dtype=np.uint8)
    header[:] = (26, 36, 26)  # Dark forest green theme
    
    # Add title and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        header,
        f"PhytoTrace AI: {decision_summary.upper()}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 255, 230),
        2,
        cv2.LINE_AA,
    )
    
    dmg_text = f"Damage: {damage_percentage:.1f}%" if damage_percentage is not None else "Damage: N/A"
    sub_text = f"{model_version} | {dmg_text} | Processed: {timestamp}"
    cv2.putText(
        header,
        sub_text,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 210, 180),
        1,
        cv2.LINE_AA,
    )

    full_canvas = np.vstack([header, combined_body])
    return full_canvas


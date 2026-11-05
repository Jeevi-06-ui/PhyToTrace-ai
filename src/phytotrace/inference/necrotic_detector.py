"""Tomato foliage necrotic and brown blotch damage detector.
Identifies brown, dark, and desiccated mining spots and lesions strictly within the leaf canopy.
"""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def detect_necrotic_leaf_spots(
    image_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    min_area: int = 35,
    max_leaf_ratio: float = 0.20,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Detect necrotic, brown, and dark blotch damage spots strictly inside the leaf canopy.

    In tomato agronomy, Tuta absoluta blotch mines and advanced Liriomyza mines cause
    cellular mesophyll necrosis, turning leaf tissue yellowish-brown, rust, or dark brown
    with frass deposits. This detector isolates these distinct chrominance deviations
    while ensuring zero background or healthy leaf false positives.

    Args:
        image_bgr: Input BGR image.
        leaf_mask: Binary leaf canopy mask (255 inside leaf, 0 outside).
        min_area: Minimum pixel area for a qualifying spot.
        max_leaf_ratio: Maximum fraction of total leaf area for a single spot.

    Returns:
        Tuple of (spot_binary_mask, spot_regions_list)
    """
    h, w = image_bgr.shape[:2]
    spot_mask = np.zeros((h, w), dtype=np.uint8)
    spot_regions: List[Dict[str, Any]] = []

    if leaf_mask is None:
        return spot_mask, spot_regions

    leaf_pixels = cv2.countNonZero(leaf_mask)
    if leaf_pixels < 250:
        return spot_mask, spot_regions

    # 1. Erode leaf boundary adaptively so table/soil/background boundary pixels cannot be mistaken for spots
    k_size = max(7, min(25, int(max(h, w) * 0.015)))
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    inner_leaf = cv2.erode(leaf_mask, k_erode)
    if cv2.countNonZero(inner_leaf) < 150:
        inner_leaf = leaf_mask

    # 2. Extract color spaces
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a_chan = lab[:, :, 1].astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    # 3. Baseline green stats inside this specific leaf
    green_a = a_chan[inner_leaf > 0]
    if len(green_a) == 0:
        return spot_mask, spot_regions

    med_a = float(np.median(green_a))

    # 4. Necrotic / Brown spot conditions:
    # - Healthy tomato foliage is green: a* ~ 95-118, G >> R (R/G < 0.85).
    # - Necrotic / brown spots on leaf:
    #   a) a* increases significantly above healthy lamina (a* >= max(128.0, med_a + 14.0))
    #   b) Red channel exceeds Green (r >= 1.08 * g)
    #   c) Hue in warm brown/yellow range (hue <= 26.0)
    #   d) High saturation of desiccated necrotic lesion (sat >= 70.0)
    #   e) Strictly inside inner leaf
    brown_spot_cond = (
        (a_chan >= max(128.0, med_a + 14.0)) &
        (r >= 1.08 * g) &
        (hue <= 26.0) &
        (sat >= 70.0) &
        (inner_leaf > 0)
    )

    raw_spots = brown_spot_cond.astype(np.uint8) * 255

    # 5. Morphological filtering: remove single-pixel salt noise and close spot fragments
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_spots, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    # 6. Extract spot contours and filter by realistic lesion morphology
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = leaf_pixels * max_leaf_ratio
    green_ref = (g > r + 6.0) & (a_chan < 125.0) & (leaf_mask > 0)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < max(min_area, leaf_pixels * 0.0002) or area > max_area:
            continue

        # Check aspect ratio to reject thin petioles / stems
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 3.0:
            continue

        # Check green neighborhood: dilate contour and verify perimeter touches green leaf blade
        spot_single = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(spot_single, [cnt], -1, 255, thickness=cv2.FILLED)
        dilated = cv2.dilate(spot_single, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        perimeter_ring = cv2.subtract(dilated, spot_single)

        green_count = cv2.countNonZero(cv2.bitwise_and(perimeter_ring, perimeter_ring, mask=green_ref.astype(np.uint8) * 255))
        ring_total = max(1, cv2.countNonZero(perimeter_ring))
        green_ratio = green_count / ring_total
        if green_ratio < 0.30:
            continue

        # Draw filled spot on mask
        cv2.drawContours(spot_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Simplify polygon for smooth SVG rendering
        approx = cv2.approxPolyDP(cnt, 0.012 * cv2.arcLength(cnt, True), True)
        poly = approx.reshape(-1, 2).tolist() if len(approx) >= 3 else cnt.reshape(-1, 2).tolist()

        # Compute spot circularity to distinguish blotch vs irregular mine
        peri = max(1.0, float(cv2.arcLength(cnt, True)))
        circ = (4.0 * np.pi * area) / (peri * peri)
        tuta_score = float(np.clip(0.60 + 0.30 * np.clip(circ / 0.30, 0.1, 1.0), 0.65, 0.92))

        spot_regions.append({
            "spot_id": f"SPOT_{len(spot_regions)+1:02d}",
            "bbox": [int(bx), int(by), int(bx + bw), int(by + bh)],
            "area_pixels": int(area),
            "polygon": poly,
            "polygons": [poly],
            "damage_type": "blotch_mine",
            "pest_evidence": "tuta",
            "confidence": round(tuta_score, 3),
            "is_necrotic_spot": True,
        })

    return spot_mask, spot_regions

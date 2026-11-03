"""Damage severity calculation and leaf area percentage assessment."""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


def compute_damage_severity(
    leaf_mask: np.ndarray,
    mined_mask: np.ndarray,
    min_leaf_pixels: int = 500,
) -> Dict[str, any]:
    """Compute the percentage of visible leaf area exhibiting mining lesions.
    
    Formula:
        visible_damage_percentage = 100.0 * (mined_leaf_pixels / visible_leaf_pixels)
        
    Guarantees:
    - Mines are strictly intersected with the leaf mask so background artifacts are excluded.
    - Uses union of mine masks to prevent double-counting overlapping detections.
    - Guards against division by zero and tiny unrepresentative leaf crops.
    - Notes clearly that visible damage does NOT equate directly to economic yield loss.
    
    Args:
        leaf_mask: Binary uint8 array (0 or 255) for visible leaf surface.
        mined_mask: Binary uint8 array (0 or 255) for union of mined lesions.
        min_leaf_pixels: Minimum visible leaf pixel count for valid calculation.
        
    Returns:
        Dict with pixel counts, percentage, severity tier, and caveats.
    """
    if leaf_mask is None or leaf_mask.size == 0:
        return {
            "leaf_pixels": 0,
            "mined_pixels": 0,
            "damage_percentage": None,
            "severity_tier": "unknown",
            "is_valid": False,
            "warning": "Leaf mask is empty or not provided.",
        }

    # Intersect mined mask strictly with leaf mask
    clean_leaf_binary = (leaf_mask > 0).astype(np.uint8)
    clean_mined_binary = (mined_mask > 0).astype(np.uint8) if (mined_mask is not None and mined_mask.size > 0) else np.zeros_like(clean_leaf_binary)
    
    mined_on_leaf = cv2.bitwise_and(clean_mined_binary, clean_leaf_binary)
    
    leaf_pixels = int(np.count_nonzero(clean_leaf_binary))
    mined_pixels = int(np.count_nonzero(mined_on_leaf))
    
    if leaf_pixels < min_leaf_pixels:
        return {
            "leaf_pixels": leaf_pixels,
            "mined_pixels": mined_pixels,
            "damage_percentage": None,
            "severity_tier": "insufficient_leaf_area",
            "is_valid": False,
            "warning": f"Visible leaf area ({leaf_pixels} px) is below minimum threshold ({min_leaf_pixels} px) for valid severity assessment.",
        }
        
    percentage = (mined_pixels / leaf_pixels) * 100.0
    percentage = float(round(min(100.0, max(0.0, percentage)), 2))
    
    # Agronomic severity categorization (EPPO/ICAR benchmark ranges)
    if percentage == 0.0:
        tier = "none"
    elif percentage < 5.0:
        tier = "low_infestation"
    elif percentage < 20.0:
        tier = "moderate_infestation"
    else:
        tier = "severe_infestation"
        
    return {
        "leaf_pixels": leaf_pixels,
        "mined_pixels": mined_pixels,
        "damage_percentage": percentage,
        "severity_tier": tier,
        "is_valid": True,
        "warning": None,
        "agronomic_caveat": "Damage percentage measures visible surface injury only; it does not model whole-plant physiological loss or yield reduction.",
    }


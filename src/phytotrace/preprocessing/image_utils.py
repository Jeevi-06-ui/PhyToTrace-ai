"""Image utilities: resizing, letterboxing, cropping, and color conversion."""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    fill_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image with constant aspect ratio and symmetric padding.
    
    Args:
        image: Input image array (H, W, C).
        target_size: Target (width, height).
        fill_color: Background padding color.
        
    Returns:
        (padded_image, scale_ratio, (pad_left, pad_top))
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=fill_color,
    )
    return padded, scale, (pad_left, pad_top)


def crop_region_with_context(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
    context_margin_ratio: float = 0.15,
) -> np.ndarray:
    """Crop a bounding box region from an image with surrounding contextual margin.
    
    Args:
        image: Full image numpy array.
        bbox: (x1, y1, x2, y2) coordinates.
        context_margin_ratio: Contextual expansion ratio.
        
    Returns:
        Cropped numpy array.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    margin_x = int(box_w * context_margin_ratio)
    margin_y = int(box_h * context_margin_ratio)
    
    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(w, x2 + margin_x)
    crop_y2 = min(h, y2 + margin_y)
    
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return crop


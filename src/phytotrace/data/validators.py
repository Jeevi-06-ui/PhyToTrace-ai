"""Validation utilities for images and dataset integrity."""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.validators")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def check_and_fix_exif_orientation(image_path: Path) -> Optional[np.ndarray]:
    """Load image with PIL to handle EXIF orientation properly, then convert to BGR numpy array.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        BGR numpy array (OpenCV format) or None if corrupt.
    """
    try:
        with Image.open(image_path) as pil_img:
            # Transpose orientation tag if present
            pil_img = ImageOps.exif_transpose(pil_img)
            # Convert to RGB mode (handles RGBA, P, etc.)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            rgb_arr = np.array(pil_img)
            # Convert RGB to BGR for OpenCV
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            return bgr_arr
    except Exception as e:
        logger.error("Failed to read image %s: %s", image_path, e)
        return None


def validate_single_image(
    image_path: Path,
    min_width: int = 256,
    min_height: int = 256,
    max_size_mb: float = 25.0,
) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    """Validate a single image file for format, corruption, and dimensions.
    
    Returns:
        (is_valid, reason, (width, height))
    """
    if not image_path.exists():
        return False, "File does not exist", None
        
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported extension: {image_path.suffix}", None
        
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File size {file_size_mb:.2f}MB exceeds limit of {max_size_mb}MB", None
        
    img_bgr = check_and_fix_exif_orientation(image_path)
    if img_bgr is None:
        return False, "Corrupted image file / cannot decode", None
        
    h, w = img_bgr.shape[:2]
    if w < min_width or h < min_height:
        return False, f"Dimensions ({w}x{h}) smaller than required ({min_width}x{min_height})", (w, h)
        
    return True, "Valid", (w, h)


def quarantine_corrupted_image(image_path: Path, quarantine_dir: Path, reason: str) -> Path:
    """Safely move a corrupted or invalid image to a quarantine folder without deleting."""
    quarantine_dir = resolve_path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = quarantine_dir / image_path.name
    # Append index if destination already exists
    idx = 1
    while dest_path.exists():
        dest_path = quarantine_dir / f"{image_path.stem}_{idx}{image_path.suffix}"
        idx += 1
        
    shutil.move(str(image_path), str(dest_path))
    
    # Write reason note
    reason_file = dest_path.with_suffix(dest_path.suffix + ".reason.txt")
    with open(reason_file, "w", encoding="utf-8") as f:
        f.write(f"Original path: {image_path}\nReason: {reason}\n")
        
    logger.warning("Quarantined corrupt/invalid image %s -> %s (Reason: %s)", image_path, dest_path, reason)
    return dest_path


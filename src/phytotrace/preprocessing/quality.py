"""Image quality assessment and input validation for PhytoTrace AI."""

from typing import Dict, Tuple
import cv2
import numpy as np


def assess_image_quality(
    image_bgr: np.ndarray,
    blur_threshold: float = 100.0,
    min_brightness: float = 30.0,
    max_brightness: float = 235.0,
    min_dimension: int = 256,
) -> Dict[str, any]:
    """Evaluate image quality metrics to decide if image is suitable for diagnostic analysis.
    
    Args:
        image_bgr: BGR numpy image array.
        blur_threshold: Laplacian variance below this is flagged as blurry.
        min_brightness: Average luminance below this is flagged as underexposed.
        max_brightness: Average luminance above this is flagged as overexposed.
        min_dimension: Minimum acceptable height and width.
        
    Returns:
        Dict with quality score, metrics, pass/fail status, and diagnostic messages.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "is_acceptable": False,
            "quality_status": "corrupt_or_empty",
            "reasons": ["Image array is empty or corrupt."],
            "blur_metric": 0.0,
            "mean_brightness": 0.0,
            "resolution": (0, 0),
        }
        
    h, w = image_bgr.shape[:2]
    reasons = []
    
    # 1. Dimension check
    if h < min_dimension or w < min_dimension:
        reasons.append(f"Image resolution ({w}x{h}) is below minimum threshold ({min_dimension}x{min_dimension}).")
        
    # 2. Blur assessment via Laplacian variance
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = laplacian_var < blur_threshold
    if is_blurry:
        reasons.append(f"Image appears blurry (Laplacian variance: {laplacian_var:.1f} < threshold {blur_threshold:.1f}).")
        
    # 3. Brightness / exposure assessment
    # Convert to LAB or YUV to get luminance
    mean_brightness = float(np.mean(gray))
    if mean_brightness < min_brightness:
        reasons.append(f"Image is severely underexposed (Mean brightness: {mean_brightness:.1f} < {min_brightness:.1f}).")
    elif mean_brightness > max_brightness:
        reasons.append(f"Image is severely overexposed/glared (Mean brightness: {mean_brightness:.1f} > {max_brightness:.1f}).")
        
    is_acceptable = len(reasons) == 0
    
    if is_acceptable:
        status = "acceptable"
    elif is_blurry and (mean_brightness < min_brightness or mean_brightness > max_brightness):
        status = "poor_lighting_and_blur"
    elif is_blurry:
        status = "insufficient_sharpness"
    else:
        status = "poor_exposure"
        
    return {
        "is_acceptable": is_acceptable,
        "quality_status": status,
        "reasons": reasons,
        "blur_metric": round(laplacian_var, 2),
        "mean_brightness": round(mean_brightness, 2),
        "resolution": (w, h),
    }


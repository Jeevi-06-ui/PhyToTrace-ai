"""Leaf segmentation module to separate leaf canopy from complex background."""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.leaf_segmenter")


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2 block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LightweightUNet(nn.Module):
    """Memory-efficient U-Net for binary leaf surface segmentation on Apple Silicon."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.out_conv(d1)
        return logits


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, float, int, int]:
    """Resize image preserving aspect ratio with neutral padding.
    
    Returns:
        (canvas, scale, dx, dy)
    """
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    return canvas, scale, dx, dy


class LeafSegmenter:
    """High-level leaf segmenter combining trained neural model with heuristic color fallback."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = LightweightUNet(in_channels=3, out_channels=1).to(self.device)
        self.is_trained = False
        
        if model_path:
            p = resolve_path(model_path)
            if p.is_file():
                try:
                    state = torch.load(p, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state)
                    self.model.eval()
                    self.is_trained = True
                    logger.info("Loaded trained LeafSegmenter from %s", p)
                except Exception as e:
                    logger.warning("Could not load leaf segmenter weights from %s: %s", p, e)
            else:
                logger.info("Leaf segmenter weights not found at %s. Operating in heuristic mode.", p)
        self.model.eval()

    def segment_leaf_heuristic(self, image_bgr: np.ndarray) -> np.ndarray:
        """Heuristic leaf segmentation using LAB/HSV color space, hole filling, and morphological filtering.
        Used as a robust baseline and diagnostic benchmark for leaf boundaries.
        """
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        
        # In HSV, vegetation green is roughly Hue 20 to 95, with sufficient Saturation and Value
        lower_green = np.array([20, 28, 25])
        upper_green = np.array([95, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # In LAB, green corresponds to negative/low values in 'a' channel (a < 127)
        l, a, b = cv2.split(lab)
        lab_mask = (a < 127).astype(np.uint8) * 255
        
        # Combined color mask
        combined = cv2.bitwise_and(hsv_mask, lab_mask)
        
        # Morphological opening to remove soil noise and background specks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        
        # Hole filling using external contours: fills interior mine gaps, pale veins and highlights
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(opened)
        min_component_area = (h * w) * 0.005  # At least 0.5% of total frame
        
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_component_area:
                cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
                
        # Morphological closing to seal small leaflet gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        clean_mask = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close)
                
        return clean_mask

    def segment(self, image_bgr: np.ndarray, threshold: float = 0.50, *, diagnostics: bool = False) -> Dict[str, any]:
        """Generate binary leaf mask from BGR image with confidence-aware fallback.
        
        Returns:
            Dict containing:
                'mask': uint8 array (0 or 255) of same size as input
                'leaf_pixel_count': int
                'leaf_area_ratio': float (fraction of total frame)
                'method': 'unet' if calibrated trained model loaded, else 'heuristic_fallback'
        """
        h_orig, w_orig = image_bgr.shape[:2]
        diagnostic_telemetry = {
            "logits_min": None,
            "logits_max": None,
            "logits_mean": None,
            "logits_std": None,
            "probs_min": None,
            "probs_max": None,
            "probs_mean": None,
            "probs_std": None,
            "prob_spread": None,
            "probs_gt_half_count": None,
            "status": "NOT_TRAINED",
        }
        
        probs = None
        if self.is_trained:
            # Match the checkpoint's training transform: RGB, [0, 1], direct resize.
            rgb = cv2.cvtColor(cv2.resize(image_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                raw_logits = logits.squeeze().cpu().numpy()

            p_min, p_max, p_mean, p_std = float(probs.min()), float(probs.max()), float(probs.mean()), float(probs.std())
            l_min, l_max, l_mean, l_std = float(raw_logits.min()), float(raw_logits.max()), float(raw_logits.mean()), float(raw_logits.std())
            prob_spread = p_max - p_min
            gt_half = int(np.count_nonzero(probs > threshold))

            diagnostic_telemetry = {
                "logits_min": round(l_min, 4),
                "logits_max": round(l_max, 4),
                "logits_mean": round(l_mean, 4),
                "logits_std": round(l_std, 4),
                "probs_min": round(p_min, 4),
                "probs_max": round(p_max, 4),
                "probs_mean": round(p_mean, 4),
                "probs_std": round(p_std, 4),
                "prob_spread": round(prob_spread, 4),
                "probs_gt_half_count": gt_half,
                "status": "PROBABILITY_SPREAD_PASSED" if (prob_spread >= 0.25 and p_std >= 0.04) else "UNRELIABLE_FALLBACK",
            }

            # This detects flat outputs. Calibration requires separate validation.
            if prob_spread >= 0.25 and p_std >= 0.04:
                raw_unet = self._threshold_mask(probs, (w_orig, h_orig), threshold)
                mask = self._fill_internal_holes(raw_unet)
                method = "unet"
            else:
                logger.info(
                    "U-Net probability distribution is uncalibrated (spread=%.3f, std=%.3f). "
                    "Falling back to adaptive color-space heuristic leaf segmentation.",
                    prob_spread,
                    p_std,
                )
                mask = self.segment_leaf_heuristic(image_bgr)
                method = "heuristic_fallback"
        else:
            mask = self.segment_leaf_heuristic(image_bgr)
            method = "heuristic"
            diagnostic_telemetry["status"] = "HEURISTIC_ONLY"
            
        leaf_pixels = int(np.count_nonzero(mask))
        total_pixels = h_orig * w_orig
        leaf_ratio = round(leaf_pixels / max(1, total_pixels), 4)
        
        result = {
            "mask": mask,
            "leaf_pixel_count": leaf_pixels,
            "leaf_area_ratio": leaf_ratio,
            "method": method,
            "diagnostics": diagnostic_telemetry,
        }
        if diagnostics and probs is not None:
            result["probability_map"] = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            result["unet_mask"] = self._threshold_mask(probs, (w_orig, h_orig), threshold)
        return result

    @staticmethod
    def _threshold_mask(probs, size, threshold):
        return (cv2.resize(probs, size, interpolation=cv2.INTER_LINEAR) > threshold).astype(np.uint8) * 255

    @staticmethod
    def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
        """Fill internal damage holes inside leaf contours to ensure mining lesions remain within leaf boundary."""
        if mask is None or np.count_nonzero(mask) == 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        if contours and hierarchy is not None:
            for i, cnt in enumerate(contours):
                # Parent contours (external boundary)
                if hierarchy[0][i][3] == -1 and cv2.contourArea(cnt) > 80:
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
        else:
            filled = closed
        return filled

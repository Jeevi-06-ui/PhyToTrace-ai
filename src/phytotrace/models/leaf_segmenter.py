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
        """Heuristic leaf segmentation using LAB/HSV color space and morphological filtering.
        Used as a robust baseline and diagnostic benchmark for leaf boundaries.
        """
        # Convert to HSV and Lab
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        
        # In HSV, vegetation green is roughly Hue 25 to 88, with sufficient Saturation and Value
        lower_green = np.array([22, 35, 30])
        upper_green = np.array([95, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # In LAB, green corresponds to negative values in 'a' channel (a < 125)
        l, a, b = cv2.split(lab)
        lab_mask = (a < 126).astype(np.uint8) * 255
        
        # Combined color mask
        combined = cv2.bitwise_and(hsv_mask, lab_mask)
        
        # Morphological opening to remove soil noise, then closing to seal mine gaps
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        # Keep significant connected components (remove isolated specks)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
        clean_mask = np.zeros_like(closed)
        h, w = closed.shape
        min_component_area = (h * w) * 0.005  # At least 0.5% of the total frame
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                clean_mask[labels == i] = 255
                
        # If color mask was completely empty (e.g. non-green lighting or brown leaf), fallback to Otsu
        if np.count_nonzero(clean_mask) == 0:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, clean_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        return clean_mask

    def segment(self, image_bgr: np.ndarray, threshold: float = 0.50) -> Dict[str, any]:
        """Generate binary leaf mask from BGR image.
        
        Returns:
            Dict containing:
                'mask': uint8 array (0 or 255) of same size as input
                'leaf_pixel_count': int
                'leaf_area_ratio': float (fraction of total frame)
                'method': 'unet' if trained model loaded, else 'heuristic'
        """
        h_orig, w_orig = image_bgr.shape[:2]
        
        if self.is_trained:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (256, 256))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            binary_256 = (probs > threshold).astype(np.uint8) * 255
            mask = cv2.resize(binary_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            method = "unet"
        else:
            mask = self.segment_leaf_heuristic(image_bgr)
            method = "heuristic"
            
        leaf_pixels = int(np.count_nonzero(mask))
        total_pixels = h_orig * w_orig
        leaf_ratio = round(leaf_pixels / max(1, total_pixels), 4)
        
        return {
            "mask": mask,
            "leaf_pixel_count": leaf_pixels,
            "leaf_area_ratio": leaf_ratio,
            "method": method,
        }

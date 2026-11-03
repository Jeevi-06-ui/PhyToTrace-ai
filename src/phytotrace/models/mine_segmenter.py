"""Leaf-mining damage segmentation module based on lightweight YOLO segmentation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.mine_segmenter")


class MineSegmenter:
    """Wrapper around lightweight YOLO segmentation model for leaf-miner damage localization."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.is_trained = False
        
        if weights_path:
            p = resolve_path(weights_path)
            if p.is_file():
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(str(p))
                    self.is_trained = True
                    logger.info("Loaded trained YOLO mine segmenter from %s", p)
                except Exception as e:
                    logger.warning("Could not load YOLO model from %s: %s", p, e)
            else:
                logger.warning("YOLO weights file does not exist at %s", p)

    def predict(
        self,
        image_bgr: np.ndarray,
        leaf_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """Detect and segment leaf miner mine damage regions.
        
        Args:
            image_bgr: Input BGR image array.
            leaf_mask: Optional binary leaf mask to intersect with damage regions.
            
        Returns:
            Dict containing:
                'damage_detected': bool
                'regions': List of dicts (region_id, bbox, polygon, confidence, area_pixels)
                'combined_mask': uint8 array (union of all mine masks)
                'total_mined_pixels': int
        """
        h, w = image_bgr.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        regions = []

        if not self.is_trained or self.model is None:
            # Model not trained / weights missing
            return {
                "damage_detected": None,
                "regions": [],
                "combined_mask": combined_mask,
                "total_mined_pixels": 0,
                "status": "model_not_trained",
            }

        # Run YOLO inference
        results = self.model.predict(
            source=image_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return {
                "damage_detected": False,
                "regions": [],
                "combined_mask": combined_mask,
                "total_mined_pixels": 0,
                "status": "success",
            }

        result = results[0]
        if result.masks is not None and result.boxes is not None:
            masks_data = result.masks.data.cpu().numpy()  # (N, H_mask, W_mask)
            boxes_data = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            conf_data = result.boxes.conf.cpu().numpy()    # (N,)

            for i in range(len(masks_data)):
                # Resize mask to original image dimensions
                m = (masks_data[i] > 0.5).astype(np.uint8) * 255
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

                # Intersect with leaf mask if provided
                if leaf_mask is not None:
                    m_resized = cv2.bitwise_and(m_resized, leaf_mask)

                region_pixels = int(np.count_nonzero(m_resized))
                if region_pixels == 0:
                    continue

                # Add to union mask
                combined_mask = cv2.bitwise_or(combined_mask, m_resized)

                # Extract polygon contour
                contours, _ = cv2.findContours(m_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygon = []
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    polygon = c.reshape(-1, 2).tolist()

                bbox = [int(v) for v in boxes_data[i]]
                regions.append({
                    "region_id": f"R_{i+1:02d}",
                    "bbox": bbox,
                    "confidence": float(round(conf_data[i], 4)),
                    "area_pixels": region_pixels,
                    "polygon": polygon,
                })

        total_mined = int(np.count_nonzero(combined_mask))
        return {
            "damage_detected": len(regions) > 0,
            "regions": regions,
            "combined_mask": combined_mask,
            "total_mined_pixels": total_mined,
            "status": "success",
        }


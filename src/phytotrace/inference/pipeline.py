"""Integrated multi-stage inference pipeline for PhytoTrace AI."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from PIL import Image

from phytotrace.config import load_yaml_config
from phytotrace.data.validators import check_and_fix_exif_orientation
from phytotrace.device import get_device
from phytotrace.inference.severity import compute_damage_severity
from phytotrace.inference.visualization import draw_damage_overlay, create_side_by_side_comparison
from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.models.decision_engine import PhytoTraceDecisionEngine
from phytotrace.models.leaf_segmenter import LeafSegmenter
from phytotrace.models.mine_segmenter import MineSegmenter
from phytotrace.models.species_classifier import SpeciesClassifierInferenceWrapper
from phytotrace.preprocessing.image_utils import crop_region_with_context
from phytotrace.preprocessing.quality import assess_image_quality
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.pipeline")


class InferencePipeline:
    """Orchestrates end-to-end multi-stage inference for tomato leaf mining analysis."""

    def __init__(self, config_name: str = "inference.yaml"):
        self.config = load_yaml_config(config_name).get("pipeline", {})
        device_pref = self.config.get("device", "auto")
        self.device = get_device(device_pref)
        self.model_version = "PhytoTrace-v0.1.0-alpha"

        # 1. Quality threshold settings
        qc_cfg = self.config.get("quality_check", {})
        self.blur_threshold = qc_cfg.get("blur_threshold", 100.0)
        self.brightness_min = qc_cfg.get("brightness_min", 30.0)
        self.brightness_max = qc_cfg.get("brightness_max", 235.0)

        # 2. Leaf Segmenter
        leaf_cfg = self.config.get("leaf_segmenter", {})
        leaf_weights = leaf_cfg.get("weights_path") if leaf_cfg.get("enabled", True) else None
        self.leaf_segmenter = LeafSegmenter(model_path=leaf_weights, device=str(self.device))

        # 3. Mine Segmenter (YOLO)
        mine_cfg = self.config.get("mine_segmenter", {})
        mine_weights = mine_cfg.get("weights_path") if mine_cfg.get("enabled", True) else None
        mine_conf = mine_cfg.get("confidence_threshold", 0.25)
        self.mine_segmenter = MineSegmenter(
            weights_path=mine_weights,
            confidence_threshold=mine_conf,
            device=str(self.device),
        )

        # 4. Species Evidence Classifier
        sp_cfg = self.config.get("species_classifier", {})
        sp_weights = sp_cfg.get("weights_path") if sp_cfg.get("enabled", True) else None
        self.species_wrapper = SpeciesClassifierInferenceWrapper(weights_path=sp_weights, device=str(self.device))

        # 5. Baseline 3-Class Classifier
        base_cfg = self.config.get("baseline", {})
        self.baseline_model = None
        if base_cfg.get("enabled", True) and base_cfg.get("weights_path"):
            bpath = resolve_path(base_cfg.get("weights_path"))
            if bpath.is_file():
                try:
                    self.baseline_model = MobileNetV3Baseline(num_classes=3, pretrained=False).to(self.device)
                    state = torch.load(bpath, map_location=self.device, weights_only=True)
                    self.baseline_model.load_state_dict(state)
                    self.baseline_model.eval()
                    logger.info("Loaded baseline classifier from %s", bpath)
                except Exception as e:
                    logger.warning("Could not load baseline classifier from %s: %s", bpath, e)

        # 6. Decision Engine
        self.decision_engine = PhytoTraceDecisionEngine(
            tuta_threshold=sp_cfg.get("tuta_threshold", 0.60),
            liriomyza_threshold=sp_cfg.get("liriomyza_threshold", 0.60),
        )

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        image_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """Run full multi-stage inspection on a single tomato leaf photograph.
        
        Args:
            image_input: File path or BGR numpy array.
            image_id: Optional tracking identifier.
            
        Returns:
            Structured diagnostic dictionary.
        """
        start_time = time.time()
        resolved_id = image_id or f"IMG_{int(time.time()*1000)}"

        # 1. Image Loading
        if isinstance(image_input, (str, Path)):
            p = resolve_path(image_input)
            img_bgr = check_and_fix_exif_orientation(p)
            if img_bgr is None:
                return self._error_response(resolved_id, "Corrupted image file or unsupported format", start_time)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            return self._error_response(resolved_id, "Unsupported image input type", start_time)

        # 2. Quality Control Stage
        qc_result = assess_image_quality(
            image_bgr=img_bgr,
            blur_threshold=self.blur_threshold,
            min_brightness=self.brightness_min,
            max_brightness=self.brightness_max,
        )

        if not qc_result["is_acceptable"]:
            decision_info = self.decision_engine.evaluate(
                quality_assessment=qc_result,
                leaf_ratio=0.0,
                damage_detected=None,
                damage_percentage=None,
                tuta_evidence=None,
                liriomyza_evidence=None,
            )
            return self._build_response(
                image_id=resolved_id,
                qc_result=qc_result,
                leaf_pixels=None,
                damage_detected=None,
                damage_pixels=None,
                damage_pct=None,
                regions=[],
                tuta_ev=None,
                liriomyza_ev=None,
                mixed_status="not_evaluated",
                decision_info=decision_info,
                annotated_bgr=img_bgr,
                start_time=start_time,
            )

        # 3. Leaf Surface Segmentation
        leaf_seg = self.leaf_segmenter.segment(img_bgr)
        leaf_mask = leaf_seg["mask"]
        leaf_pixels = leaf_seg["leaf_pixel_count"]
        leaf_ratio = leaf_seg["leaf_area_ratio"]

        # 4. Baseline Classifier Screening (if weights available)
        base_class = None
        base_conf = None
        if self.baseline_model is not None:
            try:
                rgb_224 = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (224, 224))
                t = torch.from_numpy(rgb_224).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                t = t.to(self.device)
                with torch.no_grad():
                    probs = self.baseline_model.predict_probabilities(t).squeeze().cpu().numpy()
                idx = int(np.argmax(probs))
                base_class = MobileNetV3Baseline.CLASS_NAMES[idx]
                base_conf = float(round(probs[idx], 4))
            except Exception as e:
                logger.warning("Baseline forward pass failed: %s", e)

        # 5. Mine Damage Localization & Segmentation
        mine_res = self.mine_segmenter.predict(img_bgr, leaf_mask=leaf_mask)
        damage_detected = mine_res["damage_detected"]
        raw_regions = mine_res["regions"]
        combined_mine_mask = mine_res["combined_mask"]

        # 6. Damage Percentage Calculation
        severity = compute_damage_severity(leaf_mask, combined_mine_mask)
        damage_pct = severity.get("damage_percentage")
        damage_pixels = severity.get("mined_pixels")

        # 7. Pest Evidence Classification for Detected Regions
        processed_regions = []
        region_tuta_scores = []
        region_liriomyza_scores = []

        for r in raw_regions:
            bbox = r["bbox"]
            crop_bgr = crop_region_with_context(img_bgr, bbox)
            crop_rgb = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (224, 224))
            t_crop = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
            ev_res = self.species_wrapper.evaluate_region_tensor(t_crop)
            t_score = ev_res.get("tuta_evidence")
            l_score = ev_res.get("liriomyza_evidence")

            pest_ev_tag = "uncertain"
            if t_score is not None and l_score is not None:
                region_tuta_scores.append(t_score)
                region_liriomyza_scores.append(l_score)
                if t_score >= 0.60 and l_score >= 0.60:
                    pest_ev_tag = "mixed"
                elif t_score >= 0.60:
                    pest_ev_tag = "tuta"
                elif l_score >= 0.60:
                    pest_ev_tag = "liriomyza"

            processed_regions.append({
                "region_id": r["region_id"],
                "bbox": r["bbox"],
                "area_pixels": r["area_pixels"],
                "confidence": r["confidence"],
                "polygon": r["polygon"],
                "pest_evidence": pest_ev_tag,
                "tuta_score": t_score,
                "liriomyza_score": l_score,
            })

        # Aggregate leaf-level evidence
        mean_tuta = float(round(np.mean(region_tuta_scores), 4)) if region_tuta_scores else None
        mean_liriomyza = float(round(np.mean(region_liriomyza_scores), 4)) if region_liriomyza_scores else None

        # Mixed infestation status
        if mean_tuta is not None and mean_liriomyza is not None:
            if mean_tuta >= 0.60 and mean_liriomyza >= 0.60:
                mixed_status = "mixed_suspected"
            else:
                mixed_status = "single_pest"
        else:
            mixed_status = "not_evaluated"

        # 8. Decision Engine Synthesis
        decision_info = self.decision_engine.evaluate(
            quality_assessment=qc_result,
            leaf_ratio=leaf_ratio,
            damage_detected=damage_detected,
            damage_percentage=damage_pct,
            tuta_evidence=mean_tuta,
            liriomyza_evidence=mean_liriomyza,
            baseline_class=base_class,
            baseline_confidence=base_conf,
        )

        # 9. Visualization & Overlay Generation
        annotated_bgr = draw_damage_overlay(
            image_bgr=img_bgr,
            leaf_mask=leaf_mask,
            regions=processed_regions,
        )

        return self._build_response(
            image_id=resolved_id,
            qc_result=qc_result,
            leaf_pixels=leaf_pixels,
            damage_detected=damage_detected,
            damage_pixels=damage_pixels,
            damage_pct=damage_pct,
            regions=processed_regions,
            tuta_ev=mean_tuta,
            liriomyza_ev=mean_liriomyza,
            mixed_status=mixed_status,
            decision_info=decision_info,
            annotated_bgr=annotated_bgr,
            start_time=start_time,
        )

    def _build_response(
        self,
        image_id: str,
        qc_result: dict,
        leaf_pixels: Optional[int],
        damage_detected: Optional[bool],
        damage_pixels: Optional[int],
        damage_pct: Optional[float],
        regions: list,
        tuta_ev: Optional[float],
        liriomyza_ev: Optional[float],
        mixed_status: str,
        decision_info: dict,
        annotated_bgr: np.ndarray,
        start_time: float,
    ) -> Dict[str, any]:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "image_id": image_id,
            "model_version": self.model_version,
            "quality_status": qc_result.get("quality_status", "unknown"),
            "damage_detected": damage_detected,
            "damage_regions": regions,
            "leaf_area_pixels": leaf_pixels,
            "damage_area_pixels": damage_pixels,
            "damage_percentage": damage_pct,
            "tuta_evidence": tuta_ev,
            "liriomyza_evidence": liriomyza_ev,
            "mixed_infestation_status": mixed_status,
            "uncertainty_status": decision_info.get("uncertainty_status", "unknown"),
            "decision": decision_info.get("decision", "unsupported_image"),
            "explanation": decision_info.get("explanation", ""),
            "safety_notice": decision_info.get("safety_notice", ""),
            "limitations": decision_info.get("limitations", []),
            "processing_time_ms": elapsed_ms,
            "_annotated_image": annotated_bgr,
        }

    def _error_response(self, image_id: str, reason: str, start_time: float) -> Dict[str, any]:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "image_id": image_id,
            "model_version": self.model_version,
            "quality_status": "error",
            "damage_detected": None,
            "damage_regions": [],
            "leaf_area_pixels": None,
            "damage_area_pixels": None,
            "damage_percentage": None,
            "tuta_evidence": None,
            "liriomyza_evidence": None,
            "mixed_infestation_status": "not_evaluated",
            "uncertainty_status": "error",
            "decision": "unsupported_image",
            "explanation": f"Processing halted: {reason}",
            "safety_notice": PhytoTraceDecisionEngine.SAFETY_NOTICE,
            "limitations": [reason],
            "processing_time_ms": elapsed_ms,
            "_annotated_image": None,
        }


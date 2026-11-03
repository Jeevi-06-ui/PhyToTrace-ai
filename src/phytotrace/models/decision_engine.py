"""Uncertainty-aware multi-stage decision engine for PhytoTrace AI."""

from typing import Dict, List, Optional, Tuple


class PhytoTraceDecisionEngine:
    """Decision engine integrating quality, segmentation, detection, and pest-evidence
    into conservative, agronomically defensible outcomes.
    """

    ALLOWED_DECISIONS = {
        "no_visible_mine",
        "mine_detected_species_uncertain",
        "possible_tuta",
        "possible_liriomyza",
        "possible_mixed",
        "other_damage",
        "image_quality_insufficient",
        "unsupported_image",
    }

    CONSERVATIVE_FALLBACK_TEXT = (
        "Visible leaf-mining damage may be present, but the pest identity cannot be determined "
        "reliably from this image alone. Please obtain expert stereomicroscopic confirmation or a sharp close-up."
    )

    SAFETY_NOTICE = (
        "Advisory Notice: This decision is an explainable machine learning aid. Do not apply chemical pesticides "
        "or alter IPM schedules without secondary field scout verification."
    )

    def __init__(
        self,
        tuta_threshold: float = 0.60,
        liriomyza_threshold: float = 0.60,
        uncertainty_margin: float = 0.15,
        min_leaf_ratio: float = 0.05,
    ):
        self.tuta_threshold = tuta_threshold
        self.liriomyza_threshold = liriomyza_threshold
        self.uncertainty_margin = uncertainty_margin
        self.min_leaf_ratio = min_leaf_ratio

    def evaluate(
        self,
        quality_assessment: Dict[str, any],
        leaf_ratio: float,
        damage_detected: Optional[bool],
        damage_percentage: Optional[float],
        tuta_evidence: Optional[float],
        liriomyza_evidence: Optional[float],
        baseline_class: Optional[str] = None,
        baseline_confidence: Optional[float] = None,
    ) -> Dict[str, any]:
        """Synthesize all pipeline stages into an explainable decision with uncertainty flag.
        
        Returns:
            Dict containing:
                'decision': str (one of ALLOWED_DECISIONS)
                'uncertainty_status': str ('calibrated_certain', 'uncertain_abstention', 'data_incomplete')
                'explanation': str
                'safety_notice': str
                'limitations': List[str]
        """
        limitations: List[str] = []

        # 1. Image Quality Stage
        if not quality_assessment.get("is_acceptable", True):
            return {
                "decision": "image_quality_insufficient",
                "uncertainty_status": "quality_rejected",
                "explanation": f"Input image failed quality threshold: {', '.join(quality_assessment.get('reasons', []))}",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": quality_assessment.get("reasons", []),
            }

        # 2. Leaf Surface Presence Stage
        if leaf_ratio < self.min_leaf_ratio:
            limitations.append(f"Leaf occupies only {leaf_ratio*100:.1f}% of image (minimum required: {self.min_leaf_ratio*100:.1f}%).")
            return {
                "decision": "unsupported_image",
                "uncertainty_status": "insufficient_leaf_surface",
                "explanation": "Image does not appear to contain a sufficiently framed tomato leaf.",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }

        # 3. Check Baseline / Damage Presence
        # If baseline classifier is available and predicts healthy with high confidence and no mines found
        if baseline_class == "healthy" and (damage_detected is False or damage_detected is None):
            return {
                "decision": "no_visible_mine",
                "uncertainty_status": "healthy_leaf",
                "explanation": "No leaf-mining or pathogen damage symptoms detected.",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }

        if baseline_class == "other_damage" and (damage_detected is False or damage_detected is None):
            return {
                "decision": "other_damage",
                "uncertainty_status": "non_miner_damage",
                "explanation": "Visible damage is present on the leaf, but does not resemble serpentine or blotch leaf mining.",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }

        # If no damage detected by segmentation
        if damage_detected is False:
            return {
                "decision": "no_visible_mine",
                "uncertainty_status": "no_mines_segmented",
                "explanation": "Leaf segmentation identified leaf surface, but no leaf-mining lesions were localized.",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }

        # 4. Leaf-mining damage is present (damage_detected is True or baseline indicates miner)
        # Now evaluate species evidence if available
        if tuta_evidence is None and liriomyza_evidence is None:
            # Species classifier not yet trained or evaluated on this region
            limitations.append("Species evidence classifier model is not trained/active; only damage presence is confirmed.")
            return {
                "decision": "mine_detected_species_uncertain",
                "uncertainty_status": "species_model_unavailable",
                "explanation": self.CONSERVATIVE_FALLBACK_TEXT,
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }

        # Evaluate evidence scores
        tuta_score = tuta_evidence or 0.0
        liriomyza_score = liriomyza_evidence or 0.0

        tuta_pos = tuta_score >= self.tuta_threshold
        liriomyza_pos = liriomyza_score >= self.liriomyza_threshold

        tuta_ambiguous = (self.tuta_threshold - self.uncertainty_margin) <= tuta_score < self.tuta_threshold
        liriomyza_ambiguous = (self.liriomyza_threshold - self.uncertainty_margin) <= liriomyza_score < self.liriomyza_threshold

        if tuta_ambiguous or liriomyza_ambiguous:
            limitations.append(f"Evidence scores near decision boundary (Tuta: {tuta_score:.2f}, Liriomyza: {liriomyza_score:.2f}).")
            return {
                "decision": "mine_detected_species_uncertain",
                "uncertainty_status": "evidence_boundary_abstention",
                "explanation": self.CONSERVATIVE_FALLBACK_TEXT,
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }

        if tuta_pos and liriomyza_pos:
            return {
                "decision": "possible_mixed",
                "uncertainty_status": "mixed_evidence_detected",
                "explanation": "Regional evidence indicates features consistent with both Tuta and Liriomyza. Expert inspection strongly recommended.",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }
        elif tuta_pos:
            return {
                "decision": "possible_tuta",
                "uncertainty_status": "tuta_evidence_dominant",
                "explanation": f"Evidence score for Tuta-associated blotch/broad mine ({tuta_score:.2f}) meets detection threshold.",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }
        elif liriomyza_pos:
            return {
                "decision": "possible_liriomyza",
                "uncertainty_status": "liriomyza_evidence_dominant",
                "explanation": f"Evidence score for Liriomyza-associated serpentine mine ({liriomyza_score:.2f}) meets detection threshold.",
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }
        else:
            # Low evidence for both specific pests despite mining lesion
            return {
                "decision": "mine_detected_species_uncertain",
                "uncertainty_status": "insufficient_species_evidence",
                "explanation": self.CONSERVATIVE_FALLBACK_TEXT,
                "safety_notice": self.SAFETY_NOTICE,
                "limitations": limitations,
            }


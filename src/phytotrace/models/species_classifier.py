"""Region-based pest-evidence classifier for Tuta and Liriomyza."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.species_classifier")


class PestEvidenceClassifier(nn.Module):
    """Multi-label evidence classifier outputting two independent logits:
    Logit 0: Tuta-associated mining evidence
    Logit 1: Liriomyza-associated mining evidence
    """

    TARGET_NAMES = ["tuta_evidence", "liriomyza_evidence"]

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        in_features = base.classifier[0].in_features
        # 2 independent outputs
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, 3, 224, 224).
            
        Returns:
            Logits of shape (B, 2).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def predict_evidence_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to obtain independent evidence scores in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE loss that ignores unlabelled or unverified species targets.
    
    Prevents unverified records from being falsely penalized as negatives.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        loss_unreduced = self.bce(logits, targets)
        masked_loss = loss_unreduced * valid_mask
        denom = valid_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / denom


class SpeciesClassifierInferenceWrapper:
    """Inference wrapper handling trained weights or clean DATA_NOT_READY status."""

    def __init__(self, weights_path: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.is_trained = False
        
        if weights_path:
            p = resolve_path(weights_path)
            if p.is_file():
                try:
                    self.model = PestEvidenceClassifier(pretrained=False).to(self.device)
                    state = torch.load(p, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state)
                    self.model.eval()
                    self.is_trained = True
                    logger.info("Loaded trained Species Evidence Classifier from %s", p)
                except Exception as e:
                    logger.warning("Could not load species model from %s: %s", p, e)
            else:
                logger.info("Species model weights not found at %s. Marking species classification DATA_NOT_READY.", p)

    def evaluate_region_tensor(self, tensor_bgr_crop: torch.Tensor) -> Dict[str, any]:
        """Evaluate evidence scores on a single region crop."""
        if not self.is_trained or self.model is None:
            return {
                "tuta_evidence": None,
                "liriomyza_evidence": None,
                "status": "DATA_NOT_READY",
                "message": "Species evidence model requires verified expert annotations and has not been trained.",
            }

        with torch.no_grad():
            tensor = tensor_bgr_crop.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            scores = self.model.predict_evidence_scores(tensor).squeeze(0).cpu().numpy()
            
        return {
            "tuta_evidence": float(round(scores[0], 4)),
            "liriomyza_evidence": float(round(scores[1], 4)),
            "status": "evaluated",
            "message": None,
        }


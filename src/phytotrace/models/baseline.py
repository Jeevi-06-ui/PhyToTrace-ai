"""MobileNetV3-Small 3-class baseline image classifier for leaf damage screening."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from phytotrace.utils.logging import get_logger

logger = get_logger("phytotrace.baseline")


class MobileNetV3Baseline(nn.Module):
    """MobileNetV3-Small adapted for 3-class leaf damage screening:
    Class 0: healthy
    Class 1: leaf_miner_damage
    Class 2: other_damage
    """

    CLASS_NAMES = ["healthy", "leaf_miner_damage", "other_damage"]

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        # Feature extractor
        self.features = base.features
        self.avgpool = base.avgpool

        # Custom classification head
        in_features = base.classifier[0].in_features  # 576 for mobilenet_v3_small
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze feature extractor parameters for initial head training."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("Froze MobileNetV3 backbone layers.")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None) -> None:
        """Unfreeze backbone parameters for fine-tuning.
        
        Args:
            unfreeze_last_n_blocks: If specified, unfreezes only the last N inverted residual blocks.
                                   If None, unfreezes all feature layers.
        """
        if unfreeze_last_n_blocks is None:
            for param in self.features.parameters():
                param.requires_grad = True
            logger.info("Unfroze all MobileNetV3 backbone layers.")
        else:
            total_blocks = len(self.features)
            for i, block in enumerate(self.features):
                requires_grad = i >= (total_blocks - unfreeze_last_n_blocks)
                for param in block.parameters():
                    param.requires_grad = requires_grad
            logger.info("Unfroze last %d layers of MobileNetV3 backbone.", unfreeze_last_n_blocks)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


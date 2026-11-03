"""Training pipeline for lightweight leaf segmentation U-Net."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phytotrace.config import load_yaml_config
from phytotrace.data.dataset import TomatoLeafSegmentationDataset
from phytotrace.device import get_device
from phytotrace.evaluation.metrics import compute_binary_segmentation_metrics
from phytotrace.models.leaf_segmenter import LightweightUNet
from phytotrace.training.checkpointing import save_checkpoint
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.train_leaf_segmenter")


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Soft Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss.mean()


def train_leaf_segmenter_pipeline(
    image_paths: List[Path],
    mask_paths: List[Path],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 0.001,
    save_dir: str = "models/checkpoints/leaf_segmenter",
) -> Dict[str, any]:
    """Train the lightweight U-Net for leaf canopy segmentation."""
    device = get_device("auto")
    logger.info("Using device: %s for leaf segmentation training.", device)

    if not image_paths or not mask_paths or len(image_paths) != len(mask_paths):
        raise ValueError("Must provide non-empty equal lists of image and mask paths.")

    # Split 80/20 train/val
    split_idx = max(1, int(len(image_paths) * 0.8))
    train_imgs, val_imgs = image_paths[:split_idx], image_paths[split_idx:]
    train_masks, val_masks = mask_paths[:split_idx], mask_paths[split_idx:]

    train_ds = TomatoLeafSegmentationDataset(train_imgs, train_masks)
    val_ds = TomatoLeafSegmentationDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LightweightUNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = resolve_path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    best_ckpt = save_path / "leaf_unet_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        avg_train_loss = train_loss / len(train_ds)

        # Validation
        model.eval()
        val_dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                targets = y.cpu().numpy()
                for p, t in zip(probs, targets):
                    metrics = compute_binary_segmentation_metrics(p > 0.5, t > 0.5)
                    val_dices.append(metrics["dice"])

        avg_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        logger.info("Epoch %d/%d | Train Loss: %.4f | Val Dice: %.4f", epoch, epochs, avg_train_loss, avg_val_dice)

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), best_ckpt)

    return {"status": "trained", "best_val_dice": best_dice, "checkpoint": str(best_ckpt)}


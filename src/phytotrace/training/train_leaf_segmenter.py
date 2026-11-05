"""Training pipeline for lightweight leaf segmentation U-Net."""

from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phytotrace.data.dataset import TomatoLeafSegmentationDataset
from phytotrace.device import get_device
from phytotrace.evaluation.metrics import compute_binary_segmentation_metrics
from phytotrace.models.leaf_segmenter import LightweightUNet
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
    val_image_paths: List[Path],
    val_mask_paths: List[Path],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 0.001,
    save_dir: str = "models/checkpoints/leaf_experiment",
) -> Dict[str, any]:
    """Train the lightweight U-Net for leaf canopy segmentation."""
    device = get_device("auto")
    logger.info("Using device: %s for leaf segmentation training.", device)

    if set(map(resolve_path, image_paths)) & set(map(resolve_path, val_image_paths)):
        raise ValueError("Training and validation photographs overlap.")
    torch.manual_seed(42)
    train_ds = TomatoLeafSegmentationDataset(image_paths, mask_paths, augment=True)
    val_ds = TomatoLeafSegmentationDataset(val_image_paths, val_mask_paths)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LightweightUNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = resolve_path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    optimizer_steps = 0
    best_dice = -1.0
    best_iou = 0.0
    best_ckpt = save_path / "leaf_unet_best.pt"

    if best_ckpt.exists():
        raise FileExistsError(f"Choose a new run directory: {best_ckpt} already exists")

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
            optimizer_steps += 1
            train_loss += loss.item() * len(x)

        avg_train_loss = train_loss / len(train_ds)

        # Validation
        model.eval()
        val_dices = []
        val_ious = []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                targets = y.cpu().numpy()
                for p, t in zip(probs, targets):
                    metrics = compute_binary_segmentation_metrics(p > 0.5, t > 0.5)
                    val_dices.append(metrics["dice"])
                    val_ious.append(metrics["iou"])

        avg_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        avg_val_iou = float(np.mean(val_ious))
        logger.info(
            "Epoch %d/%d | Steps: %d | Train BCE/Dice: %.4f | Val BCE/Dice: %.4f | Dice: %.4f | IoU: %.4f",
            epoch, epochs, optimizer_steps, avg_train_loss, val_loss / len(val_ds), avg_val_dice, avg_val_iou,
        )

        if avg_val_dice > best_dice:
            best_dice, best_iou = avg_val_dice, avg_val_iou
            torch.save(model.state_dict(), best_ckpt)

    return {"status": "trained", "best_val_dice": best_dice, "best_val_iou": best_iou,
            "optimizer_steps": optimizer_steps, "checkpoint": str(best_ckpt)}


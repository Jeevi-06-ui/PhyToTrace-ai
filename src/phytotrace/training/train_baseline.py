"""Training pipeline for MobileNetV3-Small 3-class baseline classifier."""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phytotrace.config import load_yaml_config
from phytotrace.data.augmentation import get_baseline_transforms
from phytotrace.data.dataset import TomatoLeafClassificationDataset
from phytotrace.device import get_device
from phytotrace.evaluation.metrics import compute_classification_metrics
from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.training.checkpointing import save_checkpoint
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path
from phytotrace.utils.seeding import seed_everything

logger = get_logger("phytotrace.train_baseline")


def train_baseline_pipeline(
    config_path: str = "baseline.yaml",
    splits_manifest_path: str = "data/splits/splits.json",
    metadata_csv_path: str = "data/metadata/metadata.csv",
    image_dir: str = "data/raw",
    override_epochs: Optional[int] = None,
) -> Dict[str, any]:
    """Execute end-to-end training for MobileNetV3-Small baseline.
    
    Returns:
        Experiment result summary dictionary.
    """
    cfg = load_yaml_config(config_path)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    
    seed = train_cfg.get("seed", 42)
    seed_everything(seed)
    
    device = get_device(train_cfg.get("device", "auto"))
    logger.info("Using device: %s for baseline training.", device)

    # 1. Load splits and metadata
    split_p = resolve_path(splits_manifest_path)
    meta_p = resolve_path(metadata_csv_path)

    if not split_p.is_file() or not meta_p.is_file():
        raise FileNotFoundError(
            f"Dataset splits or metadata missing. Splits: {split_p}, Metadata: {meta_p}. "
            "Please run 'scripts/ingest_images.py' and 'scripts/make_splits.py' first."
        )

    with open(split_p, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    meta_df = pd.read_csv(meta_p, dtype=str).fillna("")
    
    train_ids = set(split_manifest.get("train", []))
    val_ids = set(split_manifest.get("val", []))
    
    train_df = meta_df[meta_df["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df = meta_df[meta_df["image_id"].isin(val_ids)].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Train or Validation set is empty! Train count: {len(train_df)}, Val count: {len(val_df)}. "
            "Cannot train without samples in both partitions."
        )

    # 2. Build datasets and dataloaders
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 0)  # macOS DataLoader safe default

    train_ds = TomatoLeafClassificationDataset(
        df=train_df,
        image_dir=resolve_path(image_dir),
        transforms=get_baseline_transforms(is_training=True),
    )
    val_ds = TomatoLeafClassificationDataset(
        df=val_df,
        image_dir=resolve_path(image_dir),
        transforms=get_baseline_transforms(is_training=False),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. Model instantiation
    model = MobileNetV3Baseline(
        num_classes=model_cfg.get("num_classes", 3),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    
    # 4. Phase 1: Train classification head
    head_epochs = train_cfg.get("head_epochs", 5) if override_epochs is None else 1
    model.freeze_backbone()
    optimizer_head = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=train_cfg.get("lr_head", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    logger.info("Starting Phase 1: Training classification head for %d epochs...", head_epochs)
    for epoch in range(1, head_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_head.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), train_cfg.get("gradient_clip_norm", 1.0))
            optimizer_head.step()
            train_loss += loss.item() * len(x)
            
        avg_train_loss = train_loss / len(train_ds)
        logger.info("Phase 1 - Epoch %d/%d | Train Loss: %.4f", epoch, head_epochs, avg_train_loss)

    # 5. Phase 2: Fine-tune backbone
    finetune_epochs = train_cfg.get("finetune_epochs", 20) if override_epochs is None else 1
    model.unfreeze_backbone(unfreeze_last_n_blocks=3)
    optimizer_ft = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr_finetune", 0.0001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs)

    best_val_f1 = -1.0
    best_checkpoint_path = None
    save_dir = resolve_path(cfg.get("checkpoints", {}).get("save_dir", "models/checkpoints/baseline"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    patience = train_cfg.get("early_stopping_patience", 6)
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    logger.info("Starting Phase 2: Fine-tuning backbone for %d epochs...", finetune_epochs)
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_ft.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clip_norm", 1.0))
            optimizer_ft.step()
            train_loss += loss.item() * len(x)

        scheduler.step()
        avg_train_loss = train_loss / len(train_ds)

        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss = criterion(out, y)
                val_loss += v_loss.item() * len(x)
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_ds)
        val_metrics = compute_classification_metrics(
            np.array(all_targets),
            np.array(all_preds),
            MobileNetV3Baseline.CLASS_NAMES,
        )
        val_macro_f1 = val_metrics.get("macro_f1", 0.0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        logger.info(
            "Phase 2 - Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Macro F1: %.4f",
            epoch, finetune_epochs, avg_train_loss, avg_val_loss, val_macro_f1,
        )

        # Checkpoint if best
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            best_checkpoint_path = save_dir / "baseline_mobilenetv3_best.pt"
            save_checkpoint(
                state_dict=model.state_dict(),
                epoch=epoch,
                metrics=val_metrics,
                save_path=best_checkpoint_path,
                is_best=True,
                extra_metadata={"config": cfg},
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # 6. Save experiment report
    exp_id = f"exp_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = resolve_path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    exp_manifest = {
        "experiment_id": exp_id,
        "date": datetime.now().isoformat(),
        "device": str(device),
        "best_checkpoint": str(best_checkpoint_path),
        "best_val_macro_f1": best_val_f1,
        "history": history,
        "config": cfg,
    }
    with open(exp_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(exp_manifest, f, indent=2)

    logger.info("Training pipeline finished. Experiment logged at %s", exp_dir)
    return exp_manifest


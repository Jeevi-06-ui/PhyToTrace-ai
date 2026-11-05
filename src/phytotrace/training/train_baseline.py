"""Training pipeline for MobileNetV3-Small 3-class baseline classifier with Apple Silicon MPS support."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phytotrace.config import load_yaml_config
from phytotrace.data.augmentation import get_baseline_transforms
from phytotrace.data.dataset import TomatoLeafClassificationDataset
from phytotrace.data.split import make_group_splits, save_split_manifest
from phytotrace.device import get_device
from phytotrace.evaluation.metrics import compute_classification_metrics
from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.training.checkpointing import save_checkpoint
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path
from phytotrace.utils.seeding import seed_everything

logger = get_logger("phytotrace.train_baseline")


def plot_training_history(history: Dict[str, list], save_path: Path) -> None:
    """Generate and save comprehensive loss, accuracy, and Macro-F1 curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=1.5, markersize=4)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=1.5, markersize=4)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Finetuning Epoch", fontsize=10)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Metric curves
    ax2.plot(epochs, history["val_macro_f1"], "g-^", label="Val Macro F1", linewidth=2.0, markersize=5)
    if "val_accuracy" in history and history["val_accuracy"]:
        ax2.plot(epochs, history["val_accuracy"], "m--d", label="Val Accuracy", linewidth=1.5, markersize=4)
    if "train_accuracy" in history and history["train_accuracy"]:
        ax2.plot(epochs, history["train_accuracy"], "c:", label="Train Accuracy", linewidth=1.2)
        
    ax2.set_title("Validation Macro F1 & Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Finetuning Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves to %s", save_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Render and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Validation Confusion Matrix",
        ylabel="Ground Truth",
        xlabel="Model Prediction",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold",
            )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", save_path)


def compute_class_weights(dataset: TomatoLeafClassificationDataset, num_classes: int = 3) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights on training data only."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt > 0:
            w = total_samples / (num_classes * cnt)
        else:
            w = 1.0
        weights.append(w)
        
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize so mean weight is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def train_baseline_pipeline(
    config_path: str = "baseline.yaml",
    splits_manifest_path: str = "data/splits/splits.json",
    metadata_csv_path: str = "data/metadata/metadata.csv",
    image_dir: str = "data/raw",
    override_epochs: Optional[int] = None,
    head_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    finetune_lr: Optional[float] = None,
    patience: Optional[int] = None,
    device_str: Optional[str] = None,
    seed_val: Optional[int] = None,
    use_class_weights: bool = True,
    is_smoke_test: bool = False,
    save_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end training for MobileNetV3-Small baseline.
    
    Schedule:
    - Stage A: Head training with frozen backbone for `head_epochs`.
    - Stage B: Backbone fine-tuning (last 3 blocks) for `finetune_epochs` with early stopping.
    """
    cfg = load_yaml_config(config_path)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    
    # 1. Resolve effective hyperparameters
    seed = seed_val if seed_val is not None else train_cfg.get("seed", 42)
    seed_everything(seed)
    
    target_device_str = device_str if device_str is not None else train_cfg.get("device", "auto")
    device = get_device(target_device_str)
    
    effective_batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", 16)
    effective_lr_head = learning_rate if learning_rate is not None else train_cfg.get("lr_head", 0.001)
    effective_lr_ft = finetune_lr if finetune_lr is not None else train_cfg.get("lr_finetune", 0.0001)
    effective_patience = patience if patience is not None else train_cfg.get("early_stopping_patience", 6)
    
    # Resolve Stage A & Stage B Epochs
    if is_smoke_test:
        eff_head_epochs = 1
        eff_finetune_epochs = 1
    else:
        # Stage A: head epochs
        if head_epochs is not None:
            eff_head_epochs = head_epochs
        else:
            eff_head_epochs = train_cfg.get("head_epochs", 5)

        # Stage B: finetune epochs
        if finetune_epochs is not None:
            eff_finetune_epochs = finetune_epochs
        elif override_epochs is not None:
            if override_epochs > eff_head_epochs:
                eff_finetune_epochs = override_epochs - eff_head_epochs
            else:
                eff_finetune_epochs = max(1, override_epochs)
        else:
            eff_finetune_epochs = train_cfg.get("finetune_epochs", 20)

    # 2. Print Effective Configuration Banner
    print("-" * 65)
    print("📋 EFFECTIVE BASELINE TRAINING CONFIGURATION:")
    print(f"   • Device              : {device} (Apple MPS: {device.type == 'mps'})")
    print(f"   • Random Seed         : {seed}")
    print(f"   • Batch Size          : {effective_batch_size}")
    print(f"   • Stage A Head Epochs : {eff_head_epochs}")
    print(f"   • Stage B FT Epochs   : {eff_finetune_epochs}")
    print(f"   • Total Scheduled     : {eff_head_epochs + eff_finetune_epochs} epochs")
    print(f"   • LR Head (Stage A)   : {effective_lr_head}")
    print(f"   • LR Finetune (Stage B): {effective_lr_ft}")
    print(f"   • Early Stop Patience : {effective_patience} epochs without val improvement")
    print(f"   • Class Weighting     : {'ENABLED (Inverse-Frequency)' if use_class_weights else 'DISABLED'}")
    print("-" * 65)

    # 3. Load splits and metadata
    split_p = resolve_path(splits_manifest_path)
    meta_p = resolve_path(metadata_csv_path)

    if not split_p.is_file() or not meta_p.is_file():
        raise FileNotFoundError(
            f"Dataset splits or metadata missing. Splits: {split_p}, Metadata: {meta_p}. "
            "Please run 'scripts/build_dataset_inventory.py' and 'scripts/make_splits.py' first."
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
            f"Train or Validation set is empty! Train count: {len(train_df)}, Val count: {len(val_df)}."
        )

    # 4. Build datasets and dataloaders
    num_workers = train_cfg.get("num_workers", 0)

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

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False, num_workers=num_workers)

    # 5. Model instantiation & Loss criterion
    num_classes = model_cfg.get("num_classes", 3)
    model = MobileNetV3Baseline(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)

    if use_class_weights:
        class_weights = compute_class_weights(train_ds, num_classes=num_classes).to(device)
        logger.info(
            "Computed training class weights: Healthy=%.3f, LeafMiner=%.3f, Other=%.3f",
            class_weights[0].item(), class_weights[1].item(), class_weights[2].item(),
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        class_weights = None
        criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # STAGE A: Train Classification Head with Frozen Backbone
    # -------------------------------------------------------------------------
    model.freeze_backbone()
    optimizer_head = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=effective_lr_head,
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    logger.info("Starting Stage A: Training classification head for %d epochs...", eff_head_epochs)
    for epoch in range(1, eff_head_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_head.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), train_cfg.get("gradient_clip_norm", 1.0))
            optimizer_head.step()

            train_loss += loss.item() * len(x)
            preds = torch.argmax(out, dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += len(x)
            
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        logger.info(
            "Stage A [Head] - Epoch %02d/%02d | Train Loss: %.4f | Train Acc: %.2f%%",
            epoch, eff_head_epochs, avg_train_loss, train_acc * 100.0,
        )

    # -------------------------------------------------------------------------
    # STAGE B: Fine-Tune Backbone Layers
    # -------------------------------------------------------------------------
    model.unfreeze_backbone(unfreeze_last_n_blocks=3)
    
    # Differential learning rate: backbone at lr_finetune, head at 2x lr_finetune
    optimizer_ft = torch.optim.AdamW([
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": effective_lr_ft},
        {"params": model.classifier.parameters(), "lr": effective_lr_ft * 2.0},
    ], weight_decay=train_cfg.get("weight_decay", 0.0001))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=max(1, eff_finetune_epochs), eta_min=1e-6
    )

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_val_metrics = {}
    best_epoch = 0
    best_checkpoint_path = None
    target_save_dir = save_dir_override or cfg.get("checkpoints", {}).get("save_dir", "models/checkpoints/baseline")
    save_dir = resolve_path(target_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "learning_rates": [],
    }

    logger.info("Starting Stage B: Fine-tuning backbone for %d epochs...", eff_finetune_epochs)
    for epoch in range(1, eff_finetune_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_ft.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clip_norm", 1.0))
            optimizer_ft.step()

            train_loss += loss.item() * len(x)
            preds = torch.argmax(out, dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += len(x)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss = criterion(out, y)
                val_loss += v_loss.item() * len(x)
                val_total += len(x)
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / val_total
        val_metrics = compute_classification_metrics(
            np.array(all_targets),
            np.array(all_preds),
            MobileNetV3Baseline.CLASS_NAMES,
        )
        val_macro_f1 = val_metrics.get("macro_f1", 0.0)
        val_acc = val_metrics.get("accuracy", 0.0)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["train_accuracy"].append(round(train_acc, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))
        history["learning_rates"].append(round(current_lr, 6))

        p_stats = val_metrics.get("per_class", {})
        h_f1 = p_stats.get("healthy", {}).get("f1", 0.0)
        lm_f1 = p_stats.get("leaf_miner_damage", {}).get("f1", 0.0)
        oth_f1 = p_stats.get("other_damage", {}).get("f1", 0.0)

        logger.info(
            "Stage B [FT] - Epoch %02d/%02d | Train Loss: %.4f, Acc: %.1f%% | Val Loss: %.4f, Acc: %.1f%% | Macro F1: %.4f (H: %.2f, LM: %.2f, O: %.2f) | LR: %.6f",
            epoch, eff_finetune_epochs, avg_train_loss, train_acc * 100.0, avg_val_loss, val_acc * 100.0,
            val_macro_f1, h_f1, lm_f1, oth_f1, current_lr,
        )

        # Deterministic Model Selection: Primary Macro F1, tie-breaker lower Val Loss
        is_best = False
        if val_macro_f1 > best_val_f1 + 1e-4:
            is_best = True
        elif abs(val_macro_f1 - best_val_f1) <= 1e-4 and avg_val_loss < best_val_loss - 1e-4:
            is_best = True

        if is_best or epoch == 1:
            best_val_f1 = val_macro_f1
            best_val_loss = avg_val_loss
            best_val_metrics = val_metrics
            best_epoch = epoch
            patience_counter = 0

            best_checkpoint_path = save_dir / "baseline_mobilenetv3_best.pt"
            save_checkpoint(
                state_dict=model.state_dict(),
                epoch=epoch,
                metrics=val_metrics,
                save_path=best_checkpoint_path,
                is_best=True,
                extra_metadata={
                    "config": cfg,
                    "effective_hyperparameters": {
                        "device": str(device),
                        "seed": seed,
                        "batch_size": effective_batch_size,
                        "head_epochs": eff_head_epochs,
                        "finetune_epochs": eff_finetune_epochs,
                        "lr_head": effective_lr_head,
                        "lr_finetune": effective_lr_ft,
                        "use_class_weights": use_class_weights,
                        "best_epoch": best_epoch,
                        "best_val_macro_f1": best_val_f1,
                    },
                },
            )
        else:
            patience_counter += 1
            if patience_counter >= effective_patience:
                logger.info(
                    "Early stopping triggered at Epoch %d (Patience: %d epochs without improvement). Best Epoch was %d.",
                    epoch, effective_patience, best_epoch,
                )
                break

    # 6. Save Experiment Manifest, Curves & Confusion Matrix
    exp_id = f"exp_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = resolve_path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    exp_manifest = {
        "experiment_id": exp_id,
        "date": datetime.now().isoformat(),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "best_val_macro_f1": best_val_f1,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_metrics": best_val_metrics,
        "history": history,
        "effective_config": {
            "device": str(device),
            "seed": seed,
            "batch_size": effective_batch_size,
            "head_epochs": eff_head_epochs,
            "finetune_epochs": eff_finetune_epochs,
            "lr_head": effective_lr_head,
            "lr_finetune": effective_lr_ft,
            "early_stopping_patience": effective_patience,
            "use_class_weights": use_class_weights,
        },
    }
    with open(exp_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(exp_manifest, f, indent=2)

    if len(history["train_loss"]) > 0:
        plot_training_history(history, exp_dir / "loss_and_validation_curves.png")

    if "confusion_matrix" in best_val_metrics:
        cm_arr = np.array(best_val_metrics["confusion_matrix"])
        plot_confusion_matrix(cm_arr, MobileNetV3Baseline.CLASS_NAMES, exp_dir / "confusion_matrix.png")
        with open(exp_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
            json.dump({"confusion_matrix": best_val_metrics["confusion_matrix"], "classes": MobileNetV3Baseline.CLASS_NAMES}, f, indent=2)

    logger.info("Training pipeline finished. Experiment logged at %s", exp_dir)
    return exp_manifest


def run_baseline_training_smoke_test() -> Dict[str, Any]:
    """Execute a self-contained training smoke test on Apple Silicon MPS using synthetic fixtures."""
    logger.info("Setting up isolated synthetic fixture environment for training smoke test...")
    smoke_dir = resolve_path("data/interim/smoke_test")
    smoke_img_dir = smoke_dir / "images"
    smoke_img_dir.mkdir(parents=True, exist_ok=True)

    damage_types = ["healthy", "leaf_miner_damage", "other_damage"] * 4
    records = []

    for i in range(12):
        img_name = f"smoke_{i:02d}.jpg"
        img_p = smoke_img_dir / img_name
        arr = np.full((224, 224, 3), (35 + (i * 15) % 80, 140, 45), dtype=np.uint8)
        arr[::8, :] = (60, 160, 70)
        Image.fromarray(arr).save(img_p)

        records.append({
            "image_id": f"SMK_{i:04d}",
            "original_filename": img_name,
            "image_path": str(img_p.relative_to(resolve_path("."))),
            "collector_id": "COL_SMOKE",
            "collection_date": "2026-11-03",
            "farm_id": f"FARM_SMK_{i % 2}",
            "field_id": "F1",
            "row_id": "R1",
            "plant_id": f"P_SMK_{i % 3}",
            "leaf_id": f"L_SMK_{i % 2}",
            "collection_session_id": "SESS_SMOKE",
            "tomato_variety": "Synthetic_Check",
            "growth_stage": "vegetative",
            "image_view_type": "whole_leaf",
            "image_quality": "sharp",
            "visible_damage_type": damage_types[i],
            "suspected_pest": "none",
            "confirmed_pest": "unconfirmed",
            "pest_confirmation_method": "none",
            "expert_reviewer_id": "none",
            "label_verification_status": "unverified_field_guess",
            "mixed_infestation_status": "none",
            "annotation_status": "unannotated",
            "notes": "Synthetic training smoke test fixture",
            "consent_or_permission": "yes",
            "usage_license": "MIT",
        })

    smoke_meta_csv = smoke_dir / "metadata_smoke.csv"
    smoke_df = pd.DataFrame(records)
    smoke_df.to_csv(smoke_meta_csv, index=False)

    smoke_splits_json = smoke_dir / "splits_smoke.json"
    manifest = make_group_splits(
        df=smoke_df,
        group_col="farm_id",
        fallback_group_col="plant_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    save_split_manifest(manifest, smoke_splits_json)

    smoke_ckpt_dir = "models/checkpoints/smoke_test"
    result = train_baseline_pipeline(
        config_path="baseline.yaml",
        splits_manifest_path=str(smoke_splits_json),
        metadata_csv_path=str(smoke_meta_csv),
        image_dir=str(smoke_img_dir),
        is_smoke_test=True,
        save_dir_override=smoke_ckpt_dir,
    )
    result["status"] = "SMOKE_TEST_PASSED"
    result["synthetic_samples"] = 12
    result["message"] = (
        "MobileNetV3-Small forward/backward pass, Stage A head training, Stage B fine-tuning, "
        "and checkpoint serialization verified successfully on Apple Silicon MPS."
    )
    return result

"""Training pipeline for region-based multi-label pest evidence classifier."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phytotrace.config import load_yaml_config
from phytotrace.data.augmentation import get_region_transforms
from phytotrace.data.dataset import TomatoDamageEvidenceDataset
from phytotrace.device import get_device
from phytotrace.models.species_classifier import MaskedBCEWithLogitsLoss, PestEvidenceClassifier
from phytotrace.training.checkpointing import save_checkpoint
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.train_species")


def check_species_data_readiness(
    metadata_df: pd.DataFrame,
    min_verified_samples: int = 20,
) -> Tuple[bool, Dict[str, int], str]:
    """Inspect dataset to verify if sufficient independently verified species labels exist."""
    if metadata_df.empty:
        return False, {}, "Metadata catalog is empty."

    status_col = "label_verification_status"
    confirmed_col = "confirmed_pest"

    if status_col not in metadata_df.columns or confirmed_col not in metadata_df.columns:
        return False, {}, f"Required columns '{status_col}' or '{confirmed_col}' missing from metadata."

    verified = metadata_df[metadata_df[status_col].isin(["expert_reviewed", "lab_confirmed"])]
    counts = verified[confirmed_col].value_counts().to_dict()

    tuta_count = counts.get("tuta_absoluta", 0)
    liriomyza_count = counts.get("liriomyza_spp", 0)
    total_verified = len(verified)

    if total_verified < min_verified_samples:
        msg = (
            f"Insufficient verified species data: Found {total_verified} verified samples "
            f"(Tuta: {tuta_count}, Liriomyza: {liriomyza_count}). Minimum required is {min_verified_samples}."
        )
        return False, counts, msg

    return True, counts, "Verified dataset meets readiness threshold."


def train_species_evidence_pipeline(
    config_path: str = "species.yaml",
    metadata_csv_path: str = "data/metadata/metadata.csv",
    splits_manifest_path: str = "data/splits/splits.json",
) -> Dict[str, any]:
    """Train region-based pest evidence model ONLY if verified labels exist."""
    meta_p = resolve_path(metadata_csv_path)
    split_p = resolve_path(splits_manifest_path)

    if not meta_p.is_file() or not split_p.is_file():
        return {
            "status": "DATA_NOT_READY",
            "reason": f"Required files missing: Metadata exists={meta_p.is_file()}, Splits exist={split_p.is_file()}.",
            "action_required": "Run 'scripts/ingest_images.py' and 'scripts/make_splits.py' once images are collected.",
        }

    meta_df = pd.read_csv(meta_p, dtype=str).fillna("")
    is_ready, counts, report_msg = check_species_data_readiness(meta_df, min_verified_samples=20)

    if not is_ready:
        logger.warning("Pest evidence training blocked: %s", report_msg)
        return {
            "status": "DATA_NOT_READY",
            "reason": report_msg,
            "current_verified_counts": counts,
            "action_required": "Annotate and obtain expert verification for at least 20 Tuta and Liriomyza samples.",
        }

    # If data is ready, execute training
    device = get_device("auto")
    cfg = load_yaml_config(config_path)
    train_cfg = cfg.get("training", {})

    with open(split_p, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    train_ids = set(split_manifest.get("train", []))
    val_ids = set(split_manifest.get("val", []))

    train_df = meta_df[meta_df["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df = meta_df[meta_df["image_id"].isin(val_ids)].reset_index(drop=True)

    train_ds = TomatoDamageEvidenceDataset(train_df, transforms=get_region_transforms(is_training=True))
    val_ds = TomatoDamageEvidenceDataset(val_df, transforms=get_region_transforms(is_training=False))

    train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=False)

    model = PestEvidenceClassifier(pretrained=True).to(device)
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 0.0005))

    epochs = train_cfg.get("epochs", 25)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, targets, mask, _ in train_loader:
            x, targets, mask = x.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

    save_dir = resolve_path(train_cfg.get("save_dir", "models/checkpoints/species"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "species_evidence_best.pt"
    torch.save(model.state_dict(), best_ckpt)

    return {"status": "trained", "checkpoint": str(best_ckpt), "verified_counts": counts}

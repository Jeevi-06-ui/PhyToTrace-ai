#!/usr/bin/env python3
"""PhytoTrace AI — Phase 3 Segmentation Training CLI & Strict Readiness Gate.

Enforces zero-fabrication and scientific data integrity:
- Prevents training on synthetic/empty masks.
- Enforces minimum verified sample threshold.
- Strictly isolates the frozen held-out test partition.
- Utilizes Apple Metal Performance Shaders (MPS).
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import yaml

from phytotrace.device import get_device
from phytotrace.training.train_leaf_segmenter import train_leaf_segmenter_pipeline
from phytotrace.training.train_mine_segmenter import train_mine_segmenter_pipeline
from phytotrace.utils.paths import resolve_path


def reviewed_records(catalog_path, splits_path):
    catalog = pd.read_csv(catalog_path, dtype=str).fillna("")
    reviewers = pd.read_csv(catalog_path.with_name("reviewer_decisions.csv"), dtype=str).fillna("")
    reviewers = reviewers.rename(columns={"decision": "human_decision"})
    reviewed = catalog.merge(reviewers[["image_id", "human_decision", "split_partition"]],
                             on="image_id", validate="one_to_one")
    reviewed = reviewed[
        reviewed["ready_for_training"].str.lower().isin(["true", "1"])
        & reviewed["human_decision"].isin(["confirmed_mine", "confirmed_negative"])
        & reviewed["human_decision"].eq(reviewed["reviewer_decision"])
    ]
    splits = json.loads(splits_path.read_text())
    ids = [image_id for name in ("train", "val", "test") for image_id in splits[name]]
    if len(set(ids)) != len(ids):
        raise ValueError("Frozen split contains overlapping or duplicate image IDs")
    for name in ("train", "val", "test"):
        if not reviewed.loc[reviewed["split_partition"] == name, "image_id"].isin(splits[name]).all():
            raise ValueError(f"Reviewer decisions disagree with the frozen {name} split")
    return catalog, reviewed, splits


def audit_training_readiness(
    catalog_path: Path,
    splits_path: Path,
    min_train_samples: int = 10,
) -> dict:
    """Evaluate whether verified annotations meet the scientific gate criteria."""
    if not catalog_path.is_file():
        return {
            "ready": False,
            "reason": f"Segmentation catalog not found at {catalog_path}. Run scripts/audit_segmentation_readiness.py first.",
            "train_verified_count": 0,
            "total_verified_count": 0,
        }

    cat_df, ready_df, splits = reviewed_records(catalog_path, splits_path)
    train_ids, val_ids, test_ids = (set(splits[name]) for name in ("train", "val", "test"))

    train_ready = ready_df[ready_df["image_id"].isin(train_ids)]
    val_ready = ready_df[ready_df["image_id"].isin(val_ids)]
    test_ready = ready_df[ready_df["image_id"].isin(test_ids)]

    leaf_completed = (cat_df["leaf_mask_status"] == "completed").sum()
    mine_completed = (cat_df["mine_annotation_status"] == "completed").sum()

    is_ready = len(train_ready) >= min_train_samples and len(val_ready) > 0

    return {
        "ready": is_ready,
        "total_cataloged": len(cat_df),
        "total_verified_ready": len(ready_df),
        "train_ready_count": len(train_ready),
        "val_ready_count": len(val_ready),
        "test_ready_count": len(test_ready),
        "leaf_masks_completed": int(leaf_completed),
        "mine_masks_completed": int(mine_completed),
        "min_train_required": min_train_samples,
        "reason": (
            "Verified human annotations meet the gate threshold."
            if is_ready else
            f"Insufficient verified training annotations ({len(train_ready)} / {min_train_samples} required). "
            "Zero synthetic masks permitted. Genuine human CVAT annotations required."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="PhytoTrace Phase 3 Segmentation Training CLI with Gate Enforcement.")
    parser.add_argument("--task", type=str, choices=["leaf", "mine", "check_gate"], default="check_gate", help="Segmentation task to train")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml", help="Path to segmentation YAML configuration")
    parser.add_argument("--data_yaml", type=str, default="data/annotations/yolo_dataset.yaml", help="Path to YOLO dataset YAML configuration")
    parser.add_argument("--run-name", default="mine_experiment", help="YOLO output run name")
    parser.add_argument("--save-dir", default="models/checkpoints/leaf_experiment", help="New U-Net run directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--force", action="store_true", help="Bypass minimum training sample gate (NOT recommended)")
    args = parser.parse_args()

    cfg_p = resolve_path(args.config)
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    catalog_p = resolve_path(cfg["data"]["catalog_csv"])
    splits_p = resolve_path(cfg["data"]["splits_json"])
    min_train = cfg.get("training_gate", {}).get("minimum_verified_train_samples", 10)

    print("=" * 70)
    print("🌿 PHYTOTRACE AI — PHASE 3 SEGMENTATION TRAINING GATE")
    print("=" * 70)

    gate_status = audit_training_readiness(catalog_p, splits_p, min_train_samples=min_train)

    print(f"• Total Cataloged Images         : {gate_status.get('total_cataloged', 0)}")
    print(f"• Verified Leaf Masks Completed   : {gate_status.get('leaf_masks_completed', 0)}")
    print(f"• Verified Mine Masks Completed   : {gate_status.get('mine_masks_completed', 0)}")
    print(f"• Training Set Verified Samples   : {gate_status.get('train_ready_count', 0)} / {min_train} required")
    print(f"• Validation Set Verified Samples : {gate_status.get('val_ready_count', 0)}")
    print(f"• Held-Out Test Isolated Samples  : {gate_status.get('test_ready_count', 0)} (Preserved & Frozen)")
    print(f"• Gate Status                     : {'🟢 UNLOCKED' if gate_status['ready'] else '🔴 LOCKED'}")
    print(f"• Audit Message                   : {gate_status['reason']}")
    print("-" * 70)

    if args.task == "check_gate":
        if not gate_status["ready"]:
            print("\n📌 ACTION REQUIRED:")
            print("1. Annotate the 18 pilot images or all 95 images in CVAT (docs/CVAT_ANNOTATION_GUIDE.md).")
            print("2. Export COCO 1.0 JSON (instances_default.json).")
            print("3. Ingest annotations via: .venv/bin/python scripts/import_segmentation_annotations.py --coco_json <path>")
            print("4. Re-run this script to start model training.")
            sys.exit(0)
        else:
            print("✅ All gates passed. You may proceed with --task leaf or --task mine.")
            sys.exit(0)

    if not gate_status["ready"] and not args.force:
        print("\n❌ TRAINING ABORTED BY SCIENTIFIC INTEGRITY GATE:")
        print(f"   {gate_status['reason']}")
        print("   PhytoTrace prohibits training on synthetic or fabricated masks.")
        sys.exit(1)

    # Device configuration
    device = get_device("auto")
    print(f"Using Hardware Accelerator: {device}")

    epochs = args.epochs or cfg.get("training", {}).get("epochs", 30)

    if args.task == "leaf":
        print(f"\n🚀 Launching Leaf Surface U-Net Training for {epochs} epochs...")
        _, reviewed, splits = reviewed_records(catalog_p, splits_p)
        reviewed = reviewed[reviewed["leaf_mask_status"] == "completed"]
        train = reviewed[reviewed["image_id"].isin(splits["train"])]
        val = reviewed[reviewed["image_id"].isin(splits["val"])]
        if len(train) < min_train or val.empty:
            raise ValueError("The U-Net requires reviewed leaf masks in both frozen training and validation splits.")
        res = train_leaf_segmenter_pipeline(
            [resolve_path(f"data/raw/original_photos/{name}") for name in train["original_filename"]],
            list(train["leaf_mask_path"]),
            [resolve_path(f"data/raw/original_photos/{name}") for name in val["original_filename"]],
            list(val["leaf_mask_path"]),
            epochs=epochs,
            batch_size=cfg.get("training", {}).get("batch_size", 8),
            lr=cfg.get("training", {}).get("lr", 0.0005),
            save_dir=args.save_dir,
        )
        print(f"Training Result: {res}")

    elif args.task == "mine":
        _, reviewed, splits = reviewed_records(catalog_p, splits_p)
        for partition in ("train", "val"):
            decisions = set(reviewed.loc[reviewed["image_id"].isin(splits[partition]), "human_decision"])
            if not {"confirmed_mine", "confirmed_negative"} <= decisions:
                sys.exit(f"YOLO recovery requires reviewed positives and negatives in frozen {partition}.")
        print(f"\n🚀 Launching YOLO Mine Instance Segmentation for {epochs} epochs...")
        # Ultralytics training wrapper
        res = train_mine_segmenter_pipeline(
            data_yaml_path=args.data_yaml,
            epochs=epochs,
            batch_size=cfg.get("training", {}).get("batch_size", 4),
            imgsz=cfg.get("model", {}).get("input_size", [640, 640])[0],
            pretrained_weights="yolov8n-seg.pt",
            run_name=args.run_name,
        )
        print(f"Training Result: {res}")
        if res["status"] != "trained":
            sys.exit(1)


if __name__ == "__main__":
    main()

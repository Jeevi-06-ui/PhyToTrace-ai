#!/usr/bin/env python3
"""PhytoTrace AI — Dataset Validation & Training Readiness Auditor."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import pandas as pd
from PIL import Image

from phytotrace.data.metadata import MetadataManager
from phytotrace.utils.paths import resolve_path

VALID_DAMAGE_TYPES = {
    "none", "healthy",
    "leaf_miner_mine", "leaf_miner_damage",
    "chewing", "blight", "punctures", "curling", "other", "other_damage",
}

MIN_SAMPLES_PER_CLASS_FOR_TRAINING = 10


def audit_dataset_readiness(
    metadata_path: Path,
    image_dir: Path,
) -> Dict[str, any]:
    """Perform a rigorous, non-destructive audit of dataset readiness for Phase 2 training.
    
    Returns:
        Structured audit dictionary with readiness status: 'READY_FOR_TRAINING' or 'INSUFFICIENT_DATA'.
    """
    report = {
        "status": "INSUFFICIENT_DATA",
        "total_records": 0,
        "valid_images_on_disk": 0,
        "missing_images_on_disk": 0,
        "unreadable_corrupt_images": 0,
        "duplicate_image_ids": 0,
        "class_distribution": {},
        "mapped_3class_distribution": {
            "healthy": 0,
            "leaf_miner_damage": 0,
            "other_damage": 0,
        },
        "independent_farms": 0,
        "independent_plants": 0,
        "schema_errors": [],
        "reasons_for_block": [],
        "action_required": "",
    }

    if not metadata_path.is_file():
        report["reasons_for_block"].append(f"Metadata file does not exist at {metadata_path}.")
        report["action_required"] = "Ingest student images using 'scripts/ingest_images.py' or create 'data/metadata/metadata.csv'."
        return report

    mgr = MetadataManager(metadata_path)
    df = mgr.df
    report["total_records"] = len(df)

    if df.empty:
        report["reasons_for_block"].append("Metadata catalog contains 0 records.")
        report["action_required"] = "Collect tomato leaf photographs and run 'scripts/ingest_images.py'."
        return report

    # 1. Schema Validation
    is_valid_schema, schema_errors = mgr.validate_all_records()
    if not is_valid_schema:
        report["schema_errors"] = schema_errors[:10]
        report["reasons_for_block"].append(f"{len(schema_errors)} schema validation errors encountered.")

    # 2. Duplicate image IDs
    if "image_id" in df.columns:
        dup_ids = df[df.duplicated(subset=["image_id"], keep=False)]["image_id"].unique().tolist()
        report["duplicate_image_ids"] = len(dup_ids)
        if dup_ids:
            report["reasons_for_block"].append(f"Found {len(dup_ids)} duplicated image IDs: {dup_ids[:5]}.")

    # 3. File existence and integrity checks
    valid_count = 0
    missing_count = 0
    corrupt_count = 0

    for idx, row in df.iterrows():
        rel_path = str(row.get("image_path", ""))
        img_id = str(row.get("image_id", f"row_{idx}"))
        full_path = resolve_path(rel_path)

        if not full_path.is_file():
            # Check image_dir fallback
            candidate = image_dir / Path(rel_path).name
            if candidate.is_file():
                full_path = candidate
            else:
                missing_count += 1
                continue

        # Try opening image with PIL and OpenCV
        try:
            with Image.open(full_path) as pil_img:
                pil_img.verify()
            cv_img = cv2.imread(str(full_path))
            if cv_img is None or cv_img.size == 0:
                corrupt_count += 1
            else:
                valid_count += 1
        except Exception:
            corrupt_count += 1

    report["valid_images_on_disk"] = valid_count
    report["missing_images_on_disk"] = missing_count
    report["unreadable_corrupt_images"] = corrupt_count

    if missing_count > 0:
        report["reasons_for_block"].append(f"{missing_count} metadata records point to missing files on disk.")
    if corrupt_count > 0:
        report["reasons_for_block"].append(f"{corrupt_count} images could not be opened or decoded.")

    # 4. Class Distribution & Vocabulary
    raw_class_counts = df["visible_damage_type"].value_counts().to_dict() if "visible_damage_type" in df.columns else {}
    report["class_distribution"] = raw_class_counts

    # Map to 3 baseline classes
    mapped_counts = {"healthy": 0, "leaf_miner_damage": 0, "other_damage": 0}
    for raw_label, count in raw_class_counts.items():
        clean_lbl = str(raw_label).lower().strip()
        if clean_lbl in {"healthy", "none"}:
            mapped_counts["healthy"] += count
        elif clean_lbl in {"leaf_miner_mine", "leaf_miner_damage"}:
            mapped_counts["leaf_miner_damage"] += count
        else:
            mapped_counts["other_damage"] += count
    report["mapped_3class_distribution"] = mapped_counts

    # Check minimum samples per class
    missing_or_sparse_classes = []
    for cls_name, count in mapped_counts.items():
        if count < MIN_SAMPLES_PER_CLASS_FOR_TRAINING:
            missing_or_sparse_classes.append(f"'{cls_name}' (has {count}, needs >={MIN_SAMPLES_PER_CLASS_FOR_TRAINING})")

    if missing_or_sparse_classes:
        report["reasons_for_block"].append(
            f"Insufficient class balance for 3-class baseline: {', '.join(missing_or_sparse_classes)}."
        )

    # 5. Independent Farms & Plants
    farms = set(df["farm_id"].dropna().unique()) - {"", "unknown", "none"} if "farm_id" in df.columns else set()
    plants = set(df["plant_id"].dropna().unique()) - {"", "unknown", "none"} if "plant_id" in df.columns else set()
    report["independent_farms"] = len(farms)
    report["independent_plants"] = len(plants)

    if len(plants) < 3 and valid_count > 0:
        report["reasons_for_block"].append(
            f"Fewer than 3 independent plants recorded ({len(plants)}). Need multiple plants to prevent split leakage."
        )

    # Final Readiness Decision
    if not report["reasons_for_block"] and valid_count >= (MIN_SAMPLES_PER_CLASS_FOR_TRAINING * 3):
        report["status"] = "READY_FOR_TRAINING"
        report["action_required"] = "Proceed with 'scripts/make_splits.py' and 'scripts/train_baseline.py'."
    else:
        report["status"] = "INSUFFICIENT_DATA"
        report["action_required"] = (
            "Collect and ingest real tomato leaf photographs for all 3 classes (healthy, leaf_miner_damage, other_damage) "
            "into 'data/raw' using 'scripts/ingest_images.py'."
        )

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate PhytoTrace dataset and evaluate Phase 2 training readiness.")
    parser.add_argument("--metadata_path", type=str, default="data/metadata/metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--image_dir", type=str, default="data/raw", help="Path to raw image directory")
    args = parser.parse_args()

    meta_path = resolve_path(args.metadata_path)
    img_dir = resolve_path(args.image_dir)

    print("=" * 65)
    print("🌿 PHYTOTRACE AI — PHASE 2 DATASET & TRAINING READINESS AUDIT")
    print("=" * 65)

    report = audit_dataset_readiness(meta_path, img_dir)

    print(f"Readiness Status       : {report['status']}")
    print(f"Total Metadata Records : {report['total_records']}")
    print(f"Valid Decodable Images : {report['valid_images_on_disk']}")
    print(f"Missing Files on Disk  : {report['missing_images_on_disk']}")
    print(f"Corrupt Image Files    : {report['unreadable_corrupt_images']}")
    print(f"Duplicate Image IDs    : {report['duplicate_image_ids']}")
    print(f"Independent Farms      : {report['independent_farms']}")
    print(f"Independent Plants     : {report['independent_plants']}")

    print("-" * 65)
    print("3-Class Baseline Target Distribution:")
    for cls_name, count in report["mapped_3class_distribution"].items():
        print(f"  • {cls_name:<18}: {count} samples")

    if report["reasons_for_block"]:
        print("-" * 65)
        print("Readiness Blockers (Why training cannot begin yet):")
        for r in report["reasons_for_block"]:
            print(f"  ⚠️  {r}")

    print("-" * 65)
    print(f"Action Required: {report['action_required']}")
    print("=" * 65)

    # Save to report artifact
    out_report = resolve_path("reports/dataset_readiness_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed readiness report written to {out_report}")


if __name__ == "__main__":
    main()

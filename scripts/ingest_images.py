#!/usr/bin/env python3
"""PhytoTrace AI — Image Ingestion and Quality Quarantine Script."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import cv2
import pandas as pd
from PIL import Image

from phytotrace.data.deduplicate import compute_dhash, compute_sha256, detect_duplicates
from phytotrace.data.metadata import REQUIRED_COLUMNS
from phytotrace.data.schemas import (
    AnnotationStatus,
    ConfirmedPest,
    GrowthStage,
    ImageMetadataRecord,
    ImageQuality,
    ImageViewType,
    LabelVerificationStatus,
    MixedInfestationStatus,
    PestConfirmationMethod,
    SuspectedPest,
    VisibleDamageType,
)
from phytotrace.data.validators import check_and_fix_exif_orientation, quarantine_corrupted_image, validate_single_image
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("scripts.ingest_images")


def ingest_directory(
    source_dir: Path,
    target_raw_dir: Path = Path("data/raw"),
    metadata_csv: Path = Path("data/metadata/metadata.csv"),
    quarantine_dir: Path = Path("data/interim/quarantine"),
    collector_id: str = "COL_01",
    farm_id: str = "FARM_01",
) -> None:
    source_dir = resolve_path(source_dir)
    target_raw_dir = resolve_path(target_raw_dir)
    metadata_csv = resolve_path(metadata_csv)
    quarantine_dir = resolve_path(quarantine_dir)

    target_raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if present
    if metadata_csv.is_file():
        df_meta = pd.read_csv(metadata_csv, dtype=str).fillna("")
    else:
        df_meta = pd.DataFrame(columns=REQUIRED_COLUMNS)

    existing_shas = set()
    existing_ids = set(df_meta["image_id"].tolist()) if not df_meta.empty else set()
    
    # Pre-index existing file hashes
    for existing_file in target_raw_dir.glob("*.*"):
        if existing_file.is_file() and existing_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                existing_shas.add(compute_sha256(existing_file))
            except Exception:
                pass

    next_id_num = len(existing_ids) + 1
    new_records = []
    quarantined = []
    duplicates = []

    image_candidates = [
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    print(f"Found {len(image_candidates)} image candidates in {source_dir}...")

    for src_file in sorted(image_candidates):
        is_valid, reason, dims = validate_single_image(src_file)
        if not is_valid:
            q_dest = quarantine_corrupted_image(src_file, quarantine_dir, reason)
            quarantined.append({"source": str(src_file), "reason": reason, "quarantine_path": str(q_dest)})
            continue

        sha = compute_sha256(src_file)
        if sha in existing_shas:
            duplicates.append({"source": str(src_file), "sha256": sha, "reason": "Exact duplicate of existing image"})
            continue

        # Safe non-destructive import: copy and standardize
        img_id = f"IMG_{next_id_num:04d}"
        dest_filename = f"{img_id}{src_file.suffix.lower()}"
        dest_path = target_raw_dir / dest_filename

        shutil.copy2(str(src_file), str(dest_path))
        existing_shas.add(sha)
        next_id_num += 1

        # Create metadata record
        rel_path = f"data/raw/{dest_filename}"
        record = ImageMetadataRecord(
            image_id=img_id,
            original_filename=src_file.name,
            image_path=rel_path,
            collector_id=collector_id,
            collection_date=datetime.now().strftime("%Y-%m-%d"),
            farm_id=farm_id,
            field_id="F1",
            row_id="R1",
            plant_id=f"P_{farm_id}_{next_id_num % 10:02d}",
            leaf_id=f"L_{next_id_num % 5:02d}",
            collection_session_id=f"SESS_{datetime.now().strftime('%Y%m%d')}",
            tomato_variety="unknown",
            growth_stage=GrowthStage.UNKNOWN,
            image_view_type=ImageViewType.WHOLE_LEAF,
            image_quality=ImageQuality.SHARP,
            visible_damage_type=VisibleDamageType.NONE,
            suspected_pest=SuspectedPest.UNKNOWN,
            confirmed_pest=ConfirmedPest.UNCONFIRMED,
            pest_confirmation_method=PestConfirmationMethod.NONE,
            expert_reviewer_id="none",
            label_verification_status=LabelVerificationStatus.UNVERIFIED_FIELD_GUESS,
            mixed_infestation_status=MixedInfestationStatus.NOT_EVALUATED,
            annotation_status=AnnotationStatus.UNANNOTATED,
            notes="",
            consent_or_permission="yes",
            usage_license="CC-BY-4.0",
        )
        new_records.append(record.model_dump())

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_combined = pd.concat([df_meta, df_new], ignore_index=True)
        df_combined.to_csv(metadata_csv, index=False)
        print(f"✅ Ingested {len(new_records)} new images into {target_raw_dir} and updated {metadata_csv}.")
    else:
        print("No new valid images ingested.")

    # Save Ingestion Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_candidates": len(image_candidates),
        "ingested_count": len(new_records),
        "quarantined": quarantined,
        "duplicates": duplicates,
    }
    report_path = resolve_path("reports/ingestion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Ingestion audit report saved to {report_path}.")


def main():
    parser = argparse.ArgumentParser(description="Ingest smartphone tomato leaf images into PhytoTrace.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing raw camera images")
    parser.add_argument("--collector_id", type=str, default="COL_01", help="Collector pseudonym ID")
    parser.add_argument("--farm_id", type=str, default="FARM_01", help="Farm / site ID")
    args = parser.parse_args()

    ingest_directory(
        source_dir=Path(args.source_dir),
        collector_id=args.collector_id,
        farm_id=args.farm_id,
    )


if __name__ == "__main__":
    main()


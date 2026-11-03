#!/usr/bin/env python3
"""PhytoTrace AI — Dataset and Metadata Validation Script."""

import argparse
import json
from pathlib import Path
import pandas as pd

from phytotrace.data.metadata import MetadataManager
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Validate PhytoTrace dataset and metadata integrity.")
    parser.add_argument("--metadata_path", type=str, default="data/metadata/metadata.csv", help="Path to metadata.csv")
    args = parser.parse_args()

    meta_path = resolve_path(args.metadata_path)
    print("=" * 60)
    print("🌿 PHYTOTRACE AI — DATASET VALIDATION REPORT")
    print("=" * 60)

    if not meta_path.is_file():
        print(f"❌ Error: Metadata file does not exist at {meta_path}")
        print("Tip: Run 'scripts/ingest_images.py' or copy 'data/metadata/metadata_template.csv'.")
        return

    mgr = MetadataManager(meta_path)
    df = mgr.df
    print(f"Total metadata records: {len(df)}")

    # 1. Schema Validation
    is_valid, errors = mgr.validate_all_records()
    if is_valid:
        print("✅ Schema Conformity: All records strictly adhere to Pydantic schema.")
    else:
        print(f"❌ Schema Violations ({len(errors)} found):")
        for err in errors[:5]:
            print(f"   - {err}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more.")

    # 2. File Existence Check
    missing_files = []
    for idx, row in df.iterrows():
        rel_p = str(row.get("image_path", ""))
        full_p = resolve_path(rel_p)
        if not full_p.is_file():
            missing_files.append((row.get("image_id", "unknown"), rel_p))

    if missing_files:
        print(f"❌ Missing Files on Disk ({len(missing_files)} records have missing files):")
        for img_id, p in missing_files[:5]:
            print(f"   - ID {img_id}: {p} not found")
    else:
        print("✅ Referential Integrity: 100% of metadata image files exist on disk.")

    # 3. Class & Verification Distributions
    if not df.empty:
        print("-" * 60)
        print("Visible Damage Type Distribution:")
        print(df["visible_damage_type"].value_counts().to_string())

        print("-" * 60)
        print("Label Verification Status Distribution:")
        print(df["label_verification_status"].value_counts().to_string())

        print("-" * 60)
        print("Confirmed Pest Distribution (Scientifically Verified):")
        print(df["confirmed_pest"].value_counts().to_string())

        print("-" * 60)
        print("Farm / Site Distribution:")
        print(df["farm_id"].value_counts().to_string())

    print("=" * 60)


if __name__ == "__main__":
    main()


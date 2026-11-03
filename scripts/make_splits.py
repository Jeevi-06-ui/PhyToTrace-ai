#!/usr/bin/env python3
"""PhytoTrace AI — Group-Aware Dataset Splitting Script."""

import argparse
from pathlib import Path
import pandas as pd

from phytotrace.data.split import make_group_splits, save_split_manifest, verify_no_split_leakage
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Generate group-aware train/val/test splits without leakage.")
    parser.add_argument("--metadata_path", type=str, default="data/metadata/metadata.csv", help="Metadata CSV path")
    parser.add_argument("--out_manifest", type=str, default="data/splits/splits.json", help="Destination JSON path")
    parser.add_argument("--group_col", type=str, default="farm_id", help="Primary grouping column")
    parser.add_argument("--fallback_group_col", type=str, default="plant_id", help="Secondary grouping column")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_p = resolve_path(args.metadata_path)
    out_p = resolve_path(args.out_manifest)

    print("=" * 60)
    print("🌿 PHYTOTRACE AI — GROUP-AWARE SPLIT GENERATOR")
    print("=" * 60)

    if not meta_p.is_file():
        print(f"❌ Error: Metadata file does not exist at {meta_p}")
        return

    df = pd.read_csv(meta_p, dtype=str).fillna("")
    print(f"Loaded {len(df)} records from {meta_p}.")

    manifest = make_group_splits(
        df=df,
        group_col=args.group_col,
        fallback_group_col=args.fallback_group_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Verify no split leakage
    no_leakage, errors = verify_no_split_leakage(manifest, df, [args.group_col, args.fallback_group_col, "leaf_id"])
    if no_leakage:
        print("✅ Data Leakage Verification Passed: Zero image ID or group key overlap across partitions.")
    else:
        print(f"❌ LEAKAGE DETECTED ({len(errors)} violations):")
        for e in errors:
            print(f"   - {e}")
        return

    save_split_manifest(manifest, out_p)
    print(f"Saved split manifest to {out_p}.")
    print(f"Train samples: {len(manifest['train'])}")
    print(f"Val samples  : {len(manifest['val'])}")
    print(f"Test samples : {len(manifest['test'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()


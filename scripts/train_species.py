#!/usr/bin/env python3
"""PhytoTrace AI — Pest-Evidence Multi-Label Classifier Training CLI."""

import argparse
from phytotrace.training.train_species import train_species_evidence_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train region-based pest evidence classifier.")
    parser.add_argument("--config", type=str, default="species.yaml", help="Configuration file name")
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata.csv", help="Metadata CSV path")
    parser.add_argument("--splits", type=str, default="data/splits/splits.json", help="Splits JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print("🌿 PHYTOTRACE AI — PEST-EVIDENCE CLASSIFIER TRAINING")
    print("=" * 60)

    res = train_species_evidence_pipeline(
        config_path=args.config,
        metadata_csv_path=args.metadata,
        splits_manifest_path=args.splits,
    )

    if res.get("status") == "DATA_NOT_READY":
        print(f"🔒 STATUS: DATA_NOT_READY")
        print(f"Reason : {res.get('reason')}")
        print(f"Action : {res.get('action_required')}")
    elif res.get("status") == "trained":
        print(f"✅ STATUS: TRAINED")
        print(f"Checkpoint: {res.get('checkpoint')}")
    else:
        print(f"Result: {res}")


if __name__ == "__main__":
    main()


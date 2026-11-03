#!/usr/bin/env python3
"""PhytoTrace AI — Baseline Classifier Training CLI."""

import argparse
from phytotrace.training.train_baseline import train_baseline_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small 3-class baseline image classifier.")
    parser.add_argument("--config", type=str, default="baseline.yaml", help="Configuration file name or path")
    parser.add_argument("--splits", type=str, default="data/splits/splits.json", help="Path to split manifest")
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--image_dir", type=str, default="data/raw", help="Path to raw image directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override finetuning epochs for smoke tests")
    args = parser.parse_args()

    print("=" * 60)
    print("🌿 PHYTOTRACE AI — BASELINE CLASSIFIER TRAINING")
    print("=" * 60)

    try:
        manifest = train_baseline_pipeline(
            config_path=args.config,
            splits_manifest_path=args.splits,
            metadata_csv_path=args.metadata,
            image_dir=args.image_dir,
            override_epochs=args.epochs,
        )
        print("✅ Training complete!")
        print(f"Best Validation Macro F1: {manifest.get('best_val_macro_f1', 0.0):.4f}")
        print(f"Saved Checkpoint: {manifest.get('best_checkpoint')}")
    except Exception as e:
        print(f"❌ Training aborted or blocked: {e}")


if __name__ == "__main__":
    main()


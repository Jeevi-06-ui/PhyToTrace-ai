#!/usr/bin/env python3
"""PhytoTrace AI — Baseline Classifier Training CLI with Explicit Training Controls."""

import argparse
import json
from pathlib import Path

from phytotrace.training.train_baseline import run_baseline_training_smoke_test, train_baseline_pipeline
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small 3-class baseline image classifier.")
    parser.add_argument("--config", type=str, default="baseline.yaml", help="Configuration file name or path")
    parser.add_argument("--splits", type=str, default="data/splits/splits.json", help="Path to split manifest")
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--image_dir", type=str, default="data/raw", help="Path to raw image directory")
    
    # Explicit Training Controls (Task 2)
    parser.add_argument("--head-epochs", "--head_epochs", type=int, default=None, help="Stage A head epochs (default: 5)")
    parser.add_argument("--finetune-epochs", "--finetune_epochs", type=int, default=None, help="Stage B backbone fine-tuning epochs (default: 20)")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs or fallback Stage B fine-tuning epochs")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None, help="Training batch size (default: 16)")
    parser.add_argument("--learning-rate", "--learning_rate", "--lr", type=float, default=None, help="Stage A head learning rate (default: 0.001)")
    parser.add_argument("--finetune-lr", "--finetune_lr", type=float, default=None, help="Stage B backbone learning rate (default: 0.0001)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience in epochs (default: 6)")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('mps', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--class-weights", "--class_weights", dest="use_class_weights", action="store_true", default=True, help="Use inverse class frequency weighting in CrossEntropyLoss")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false", help="Disable class weighting")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run an automated forward/backward smoke test on isolated synthetic fixtures without requiring real data",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("🌿 PHYTOTRACE AI — BASELINE CLASSIFIER TRAINING PIPELINE")
    print("=" * 65)

    if args.smoke_test:
        print("Executing automated training smoke test on Apple Silicon MPS...")
        try:
            res = run_baseline_training_smoke_test()
            print("=" * 65)
            print("✅ SMOKE TEST PASSED!")
            print(f"Status           : {res['status']}")
            print(f"Device Tested    : {res['device']}")
            print(f"Checkpoint Saved : {res['best_checkpoint']}")
            print(f"Validation F1    : {res['best_val_macro_f1']:.4f}")
            print(f"Confirmation     : {res['message']}")
            print("=" * 65)
            return
        except Exception as e:
            print(f"❌ Smoke test failed: {e}")
            return

    # Real data training path
    meta_p = resolve_path(args.metadata)
    split_p = resolve_path(args.splits)

    if not meta_p.is_file() or not split_p.is_file():
        print("❌ STATUS: INSUFFICIENT_DATA")
        print(f"Reason: Metadata or split files not found.")
        print(f"  - Metadata file: {meta_p} (Exists: {meta_p.is_file()})")
        print(f"  - Splits file  : {split_p} (Exists: {split_p.is_file()})")
        print("-" * 65)
        print("Action Required:")
        print("  1. Ingest real tomato leaf images: python scripts/ingest_images.py")
        print("  2. Validate dataset readiness  : python scripts/validate_dataset.py")
        print("  3. Create group-aware splits   : python scripts/make_splits.py")
        print("  (Tip: Run 'python scripts/train_baseline.py --smoke_test' to verify pipeline mechanics)")
        print("=" * 65)
        return

    try:
        manifest = train_baseline_pipeline(
            config_path=args.config,
            splits_manifest_path=args.splits,
            metadata_csv_path=args.metadata,
            image_dir=args.image_dir,
            override_epochs=args.epochs,
            head_epochs=args.head_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            finetune_lr=args.finetune_lr,
            patience=args.patience,
            device_str=args.device,
            seed_val=args.seed,
            use_class_weights=args.use_class_weights,
        )
        print("=" * 65)
        print("✅ TRAINING COMPLETE!")
        print(f"Experiment ID           : {manifest.get('experiment_id')}")
        print(f"Best Validation Epoch   : {manifest.get('best_epoch')}")
        print(f"Best Validation Macro F1: {manifest.get('best_val_macro_f1', 0.0):.4f}")
        print(f"Best Validation Loss    : {manifest.get('best_val_loss', 0.0):.4f}")
        print(f"Best Model Checkpoint   : {manifest.get('best_checkpoint')}")
        print("=" * 65)
    except Exception as e:
        print(f"❌ Training aborted or blocked: {e}")
        raise e


if __name__ == "__main__":
    main()

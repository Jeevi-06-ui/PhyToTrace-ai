#!/usr/bin/env python3
"""PhytoTrace AI — Segmentation Training CLI."""

import argparse
from pathlib import Path

from phytotrace.training.train_leaf_segmenter import train_leaf_segmenter_pipeline
from phytotrace.training.train_mine_segmenter import train_mine_segmenter_pipeline
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Train leaf surface or YOLO mine segmentation model.")
    parser.add_argument("--task", type=str, choices=["leaf", "mine"], required=True, help="Segmentation task to train")
    parser.add_argument("--data_yaml", type=str, default="data/annotations/yolo_dataset.yaml", help="Path for YOLO data yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🌿 PHYTOTRACE AI — {args.task.upper()} SEGMENTATION TRAINING")
    print("=" * 60)

    if args.task == "leaf":
        print("Searching for mask annotations in data/annotations/leaf_masks...")
        mask_dir = resolve_path("data/annotations/leaf_masks")
        img_dir = resolve_path("data/raw")
        if not mask_dir.is_dir() or len(list(mask_dir.glob("*.png"))) == 0:
            print("❌ DATA_NOT_READY: No leaf mask annotations found in data/annotations/leaf_masks.")
            print("Tip: Provide binary leaf masks (.png) matching images in data/raw.")
            return

        img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        mask_paths = [mask_dir / f"{p.stem}.png" for p in img_paths if (mask_dir / f"{p.stem}.png").is_file()]
        matching_imgs = [p for p in img_paths if (mask_dir / f"{p.stem}.png").is_file()]

        res = train_leaf_segmenter_pipeline(matching_imgs, mask_paths, epochs=args.epochs)
        print(f"Status: {res}")

    elif args.task == "mine":
        res = train_mine_segmenter_pipeline(data_yaml_path=args.data_yaml, epochs=args.epochs)
        print(f"Status: {res.get('status')}")
        if res.get("status") == "DATA_NOT_READY":
            print(f"Reason: {res.get('message')}")


if __name__ == "__main__":
    main()


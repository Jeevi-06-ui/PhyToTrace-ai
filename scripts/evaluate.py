#!/usr/bin/env python3
"""PhytoTrace AI — Held-Out Test Evaluation and Metric Reporting."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phytotrace.data.augmentation import get_baseline_transforms
from phytotrace.data.dataset import TomatoLeafClassificationDataset
from phytotrace.device import get_device
from phytotrace.evaluation.metrics import compute_classification_metrics
from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.utils.paths import resolve_path


def evaluate_baseline_on_test(
    checkpoint_path: Path,
    splits_path: Path,
    metadata_path: Path,
    image_dir: Path,
) -> dict:
    device = get_device("auto")
    if not checkpoint_path.is_file():
        return {
            "status": "NOT_EVALUATED",
            "reason": f"Baseline checkpoint not found at {checkpoint_path}. Model has not been trained.",
        }

    with open(splits_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_ids = set(manifest.get("test", []))
    meta_df = pd.read_csv(metadata_path, dtype=str).fillna("")
    test_df = meta_df[meta_df["image_id"].isin(test_ids)].reset_index(drop=True)

    if test_df.empty:
        return {
            "status": "NOT_EVALUATED",
            "reason": "Test partition in split manifest contains 0 samples.",
        }

    test_ds = TomatoLeafClassificationDataset(
        df=test_df,
        image_dir=image_dir,
        transforms=get_baseline_transforms(is_training=False),
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = MobileNetV3Baseline(num_classes=3, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    metrics = compute_classification_metrics(
        np.array(all_targets),
        np.array(all_preds),
        MobileNetV3Baseline.CLASS_NAMES,
    )
    metrics["status"] = "EVALUATED"
    metrics["test_samples"] = len(test_df)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhytoTrace models on held-out test split.")
    parser.add_argument("--baseline_checkpoint", type=str, default="models/checkpoints/baseline/baseline_mobilenetv3_best.pt")
    parser.add_argument("--splits", type=str, default="data/splits/splits.json")
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata.csv")
    parser.add_argument("--image_dir", type=str, default="data/raw")
    args = parser.parse_args()

    print("=" * 60)
    print("🌿 PHYTOTRACE AI — INDEPENDENT TEST EVALUATION")
    print("=" * 60)

    report = {
        "baseline_classifier": evaluate_baseline_on_test(
            checkpoint_path=resolve_path(args.baseline_checkpoint),
            splits_path=resolve_path(args.splits),
            metadata_path=resolve_path(args.metadata),
            image_dir=resolve_path(args.image_dir),
        ),
        "leaf_segmenter": {
            "status": "NOT_EVALUATED",
            "reason": "Leaf segmentation evaluation requires ground-truth test masks.",
        },
        "mine_segmenter": {
            "status": "NOT_EVALUATED",
            "reason": "Mine segmentation evaluation requires ground-truth test polygons.",
        },
        "species_evidence_classifier": {
            "status": "NOT_EVALUATED",
            "reason": "Species evidence model requires verified expert species test set.",
        },
    }

    print(json.dumps(report, indent=2))
    report_p = resolve_path("reports/evaluation_report.json")
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {report_p}.")
    print("=" * 60)


if __name__ == "__main__":
    main()


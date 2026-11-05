#!/usr/bin/env python3
"""PhytoTrace AI — Systematic Generalization Evaluation Suite.

Evaluates the upgraded hybrid segmentation & validation pipeline across:
- Held-out frozen test partition (strictly preserved)
- Training & validation samples
- Confirmed human negatives
- Real-world smartphone photo
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.utils.paths import resolve_path


def evaluate():
    catalog_p = resolve_path("data/segmentation/catalog.csv")
    splits_p = resolve_path("data/splits/splits.json")
    out_dir = resolve_path("reports/debug_segmentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_p) as f:
        splits = json.load(f)

    test_ids = set(splits.get("test", []))
    val_ids = set(splits.get("val", []))
    train_ids = set(splits.get("train", []))

    cat_df = pd.read_csv(catalog_p, dtype=str).fillna("")

    pipe = InferencePipeline("inference.yaml")

    results = []

    # 1. Evaluate Pilot & Test Catalog Samples
    print("=" * 80)
    print("PHYTOTRACE AI — GENERALIZATION EVALUATION")
    print("=" * 80)

    for idx, row in cat_df.iterrows():
        img_id = row["image_id"]
        filename = row["original_filename"]
        img_p = resolve_path(f"data/raw/original_photos/{filename}")
        if not img_p.exists():
            continue

        # Determine partition
        if img_id in test_ids:
            partition = "HELD_OUT_TEST"
        elif img_id in val_ids:
            partition = "VALIDATION"
        elif img_id in train_ids:
            partition = "TRAIN"
        else:
            partition = "UNASSIGNED"

        # GT leaf and mine
        l_p = resolve_path(row["leaf_mask_path"]) if row.get("leaf_mask_path") else None
        m_p = resolve_path(row["mine_mask_path"]) if row.get("mine_mask_path") else None

        gt_leaf_px = 0
        if l_p and l_p.exists():
            l_mat = cv2.imread(str(l_p), cv2.IMREAD_GRAYSCALE)
            if l_mat is not None:
                gt_leaf_px = int(np.count_nonzero(l_mat))

        gt_mine_px = 0
        if m_p and m_p.exists():
            m_mat = cv2.imread(str(m_p), cv2.IMREAD_GRAYSCALE)
            if m_mat is not None:
                gt_mine_px = int(np.count_nonzero(m_mat))

        gt_is_positive = gt_mine_px > 0

        # Run pipeline
        pred = pipe.predict(str(img_p), image_id=f"EVAL_{img_id}")

        pred_damage = pred.get("damage_detected", False)
        pred_dmg_pct = pred.get("damage_percentage", 0.0) or 0.0
        pred_leaf_px = pred.get("leaf_area_pixels", 0) or 0
        pred_regions = len(pred.get("damage_regions", []))
        decision = pred.get("decision", "")

        results.append({
            "image_id": img_id,
            "filename": filename,
            "partition": partition,
            "gt_leaf_pixels": gt_leaf_px,
            "gt_mine_pixels": gt_mine_px,
            "gt_positive": gt_is_positive,
            "pred_leaf_pixels": pred_leaf_px,
            "pred_damage_detected": pred_damage,
            "pred_damage_percentage": pred_dmg_pct,
            "pred_regions_count": pred_regions,
            "decision": decision,
        })

    # 2. Evaluate Smartphone Photo
    phone_p = resolve_path("data/inspections/raw/INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg")
    pred_phone = pipe.predict(str(phone_p), image_id="EVAL_SMARTPHONE")
    results.append({
        "image_id": "SMARTPHONE_01",
        "filename": "WhatsApp Image 2026-11-05 at 02.45.33.jpeg",
        "partition": "FIELD_SMARTPHONE",
        "gt_leaf_pixels": 0,
        "gt_mine_pixels": 0,
        "gt_positive": False,
        "pred_leaf_pixels": pred_phone.get("leaf_area_pixels", 0) or 0,
        "pred_damage_detected": pred_phone.get("damage_detected", False),
        "pred_damage_percentage": pred_phone.get("damage_percentage", 0.0) or 0.0,
        "pred_regions_count": len(pred_phone.get("damage_regions", [])),
        "decision": pred_phone.get("decision", ""),
    })

    eval_df = pd.DataFrame(results)
    eval_csv_path = out_dir / "generalization_evaluation_results.csv"
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"Saved evaluation results to {eval_csv_path}")

    # Summary by partition
    for part in ["HELD_OUT_TEST", "VALIDATION", "TRAIN", "FIELD_SMARTPHONE"]:
        sub = eval_df[eval_df["partition"] == part]
        if sub.empty:
            continue
        print(f"\n--- PARTITION: {part} (n={len(sub)}) ---")
        for _, r in sub.head(8).iterrows():
            print(
                f"  {r['image_id']:14s} | GT Pos: {str(r['gt_positive']):5s} | "
                f"Pred Pos: {str(r['pred_damage_detected']):5s} | Dmg: {r['pred_damage_percentage']:5.2f}% | "
                f"Regions: {r['pred_regions_count']} | LeafPx: {r['pred_leaf_pixels']:,} | {r['decision']}"
            )

    # Check false positives on known negatives
    neg_sub = eval_df[eval_df["gt_positive"] == False]
    fp_count = (neg_sub["pred_damage_detected"] == True).sum()
    print(f"\nNegative Samples False Positive Count: {fp_count} / {len(neg_sub)} ({fp_count/max(1,len(neg_sub))*100:.1f}%)")


if __name__ == "__main__":
    evaluate()


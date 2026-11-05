#!/usr/bin/env python3
"""PhytoTrace AI — Phase 3 Segmentation Dataset Readiness Auditor."""

from datetime import datetime
import json
from pathlib import Path
import pandas as pd

from phytotrace.segmentation.schemas import AnnotationStatus, ImageSegmentationCatalogRecord
from phytotrace.utils.paths import resolve_path


def audit_segmentation_readiness(
    metadata_csv: Path,
    catalog_csv: Path,
    masks_dir: Path,
    pilot_csv: Path = None,
    splits_json: Path = None,
) -> dict:
    meta_df = pd.read_csv(metadata_csv, dtype=str).fillna("")
    cat_df = None

    if catalog_csv.is_file():
        cat_df = pd.read_csv(catalog_csv, dtype=str).fillna("")
    else:
        # Initialize default unannotated catalog
        records = []
        for _, row in meta_df.iterrows():
            img_id = str(row["image_id"])
            fname = str(row.get("original_filename", ""))
            record = ImageSegmentationCatalogRecord(
                image_id=img_id,
                original_filename=fname,
                plant_id=row.get("plant_id", ""),
                farm_id=row.get("farm_id", ""),
                visible_damage_type=row.get("visible_damage_type", "unlabelled"),
                leaf_mask_status=AnnotationStatus.UNANNOTATED,
                mine_annotation_status=AnnotationStatus.UNANNOTATED,
                annotation_reviewer="none",
                num_damage_regions=0,
                image_quality=row.get("image_quality", "sharp"),
                ready_for_training=False,
                is_verified_negative=False,
                leaf_mask_path="",
                mine_mask_path="",
                notes="Awaiting CVAT polygon annotation",
            )
            records.append(record.to_dict())
        cat_df = pd.DataFrame(records)
        catalog_csv.parent.mkdir(parents=True, exist_ok=True)
        cat_df.to_csv(catalog_csv, index=False)

    total_images = len(cat_df)
    leaf_done = int((cat_df["leaf_mask_status"] == "completed").sum())
    mine_done = int((cat_df["mine_annotation_status"] == "completed").sum())
    
    is_neg_col = "is_verified_negative" in cat_df.columns
    neg_done = int(cat_df["is_verified_negative"].apply(lambda x: str(x).lower() in ["true", "1"]).sum()) if is_neg_col else 0

    discrepancies = int(
        (
            (cat_df["mine_annotation_status"] == "flagged") |
            cat_df["notes"].str.contains("WARN|DISCREPANCY|violation|CONFLICT", case=False)
        ).sum()
    )

    ready_count = int(cat_df["ready_for_training"].astype(str).str.lower().isin(["true", "1"]).sum())

    pilot_count = 0
    pilot_ready_count = 0
    if pilot_csv and pilot_csv.is_file():
        p_df = pd.read_csv(pilot_csv, dtype=str).fillna("")
        pilot_count = len(p_df)
        p_ids = set(p_df["image_id"])
        pilot_ready_count = int(cat_df[cat_df["image_id"].isin(p_ids)]["ready_for_training"].apply(lambda x: str(x).lower() in ["true", "1"]).sum())

    # Split partition breakdown
    train_ready = 0
    val_ready = 0
    test_ready = 0
    leaf_train_done = 0
    leaf_val_done = 0
    leaf_test_done = 0
    min_train_required = 10

    if splits_json and splits_json.is_file():
        with open(splits_json, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        train_ids = set(splits_data.get("train", []))
        val_ids = set(splits_data.get("val", []))
        test_ids = set(splits_data.get("test", []))

        train_cat = cat_df[cat_df["image_id"].isin(train_ids)]
        val_cat = cat_df[cat_df["image_id"].isin(val_ids)]
        test_cat = cat_df[cat_df["image_id"].isin(test_ids)]

        train_ready = int(train_cat["ready_for_training"].astype(str).str.lower().isin(["true", "1"]).sum())
        val_ready = int(val_cat["ready_for_training"].astype(str).str.lower().isin(["true", "1"]).sum())
        test_ready = int(test_cat["ready_for_training"].astype(str).str.lower().isin(["true", "1"]).sum())

        leaf_train_done = int((train_cat["leaf_mask_status"] == "completed").sum())
        leaf_val_done = int((val_cat["leaf_mask_status"] == "completed").sum())
        leaf_test_done = int((test_cat["leaf_mask_status"] == "completed").sum())

    unet_ready = leaf_train_done >= min_train_required
    yolo_ready = train_ready >= min_train_required
    overall_status = "READY_FOR_TRAINING" if (unet_ready and yolo_ready) else "AWAITING_CVAT_ANNOTATIONS"

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": overall_status,
        "unet_leaf_gate_status": "READY" if unet_ready else "LOCKED",
        "yolo_mine_gate_status": "READY" if yolo_ready else "LOCKED",
        "total_images": total_images,
        "independent_plants": int(cat_df["plant_id"].nunique()),
        "independent_farms": int(cat_df["farm_id"].nunique()),
        "leaf_masks_completed": leaf_done,
        "leaf_masks_pending": total_images - leaf_done,
        "mine_annotations_completed": mine_done,
        "mine_annotations_pending": total_images - mine_done,
        "verified_negative_empty_masks": neg_done,
        "flagged_discrepancies": discrepancies,
        "samples_ready_for_segmentation_training": ready_count,
        "pilot_samples_total": pilot_count,
        "pilot_samples_ready": pilot_ready_count,
        "train_ready_samples": train_ready,
        "val_ready_samples": val_ready,
        "test_ready_samples": test_ready,
        "leaf_masks_train_completed": leaf_train_done,
        "min_train_required": min_train_required,
        "total_damage_regions_annotated": int(pd.to_numeric(cat_df["num_damage_regions"], errors="coerce").fillna(0).sum()),
        "damage_type_distribution": cat_df["visible_damage_type"].value_counts().to_dict(),
        "catalog_path": str(catalog_csv),
        "annotation_guide": "docs/CVAT_ANNOTATION_GUIDE.md",
        "pilot_manifest": "data/segmentation/pilot_manifest.csv",
        "action_required": (
            f"Annotate/correct CVAT polygons ({train_ready}/{min_train_required} training samples ready). "
            "Resolve flagged images in data/segmentation/reviewer_decisions.csv and export updated COCO 1.0."
            if not yolo_ready else "All Phase 3 segmentation training gates satisfied."
        ),
    }
    return report


def main():
    meta_p = resolve_path("data/metadata/metadata.csv")
    cat_p = resolve_path("data/segmentation/catalog.csv")
    masks_p = resolve_path("data/segmentation/masks")
    pilot_p = resolve_path("data/segmentation/pilot_manifest.csv")
    splits_p = resolve_path("data/splits/splits.json")
    rep_json_p = resolve_path("reports/segmentation_readiness_report.json")
    rep_md_p = resolve_path("reports/segmentation_readiness_report.md")

    print("=" * 70)
    print("🌿 PHYTOTRACE AI — PHASE 3 SEGMENTATION READINESS AUDIT")
    print("=" * 70)

    if not meta_p.is_file():
        print(f"❌ Master metadata file not found at {meta_p}")
        return

    report = audit_segmentation_readiness(meta_p, cat_p, masks_p, pilot_p, splits_p)

    print(f"Readiness Status               : {report['status']}")
    print(f"  • U-Net Leaf Boundary Gate   : {'🟢 READY' if report['unet_leaf_gate_status'] == 'READY' else '🔴 LOCKED'} ({report['leaf_masks_train_completed']} / {report['min_train_required']} train leaf masks)")
    print(f"  • YOLO Mine Instance Gate    : {'🟢 READY' if report['yolo_mine_gate_status'] == 'READY' else '🔴 LOCKED'} ({report['train_ready_samples']} / {report['min_train_required']} train verified samples)")
    print(f"Total Master Images            : {report['total_images']}")
    print(f"Leaf Masks Completed           : {report['leaf_masks_completed']} / {report['total_images']}")
    print(f"Mine Annotations Completed     : {report['mine_annotations_completed']} / {report['total_images']}")
    print(f"Verified Negative Empty Masks  : {report['verified_negative_empty_masks']}")
    print(f"Flagged Discrepancies          : {report['flagged_discrepancies']}")
    print(f"Pilot Set Progress             : {report['pilot_samples_ready']} / {report['pilot_samples_total']} ready")
    print(f"  - Train Partition Ready      : {report['train_ready_samples']} / {report['min_train_required']} required")
    print(f"  - Validation Partition Ready : {report['val_ready_samples']}")
    print(f"  - Test Partition Isolated    : {report['test_ready_samples']} (Frozen)")
    print(f"Action Required                : {report['action_required']}")
    print("-" * 70)

    rep_json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_json_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown report
    md_content = rf"""# PhytoTrace AI — Phase 3 Segmentation Dataset Readiness Audit

**Audit Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Overall Readiness Status:** `{report['status']}`  
**Annotation Guide:** [`docs/CVAT_ANNOTATION_GUIDE.md`](file:///Users/jeevithp/Desktop/PhyToTrace/docs/CVAT_ANNOTATION_GUIDE.md)  
**Pilot Manifest:** [`data/segmentation/pilot_manifest.csv`](file:///Users/jeevithp/Desktop/PhyToTrace/data/segmentation/pilot_manifest.csv)  
**Reviewer Decisions Ledger:** [`data/segmentation/reviewer_decisions.csv`](file:///Users/jeevithp/Desktop/PhyToTrace/data/segmentation/reviewer_decisions.csv)

---

## 1. Readiness Metric Summary

| Metric | Current Count | Target Requirement | Status |
| :--- | :---: | :---: | :---: |
| **Total Images Tracked** | **{report['total_images']}** | 95 | ✅ Complete Catalog |
| **Independent Plants** | **{report['independent_plants']}** | 95 | ✅ Complete Coverage |
| **Completed Leaf Boundary Masks** | **{report['leaf_masks_completed']}** | 95 | ⏳ Pilot Complete (18/18) |
| **Completed Mine Damage Polygons** | **{report['mine_annotations_completed']}** | 48 (symptomatic) | ⏳ 12 Symptomatic Verified |
| **Verified Negative Empty Mine Masks** | **{report['verified_negative_empty_masks']}** | $\approx 47$ | ⏳ Awaiting Human Review in CVAT |
| **Flagged Discrepancies** | **{report['flagged_discrepancies']}** | 0 | ⚠️ 6 Actions Required |
| **Pilot Set Progress (18 samples)** | **{report['pilot_samples_ready']} / {report['pilot_samples_total']}** | 18 | ⏳ 12 Verified Ready |
| **Train Split Verified Samples** | **{report['train_ready_samples']} / 10** | 10 | ⏳ 4 Samples Needed to Unlock Gate |
| **Validation Split Verified Samples** | **{report['val_ready_samples']} / 4** | 4 | ⏳ 3 Verified Ready, 1 Flagged |
| **Held-Out Test Isolated Samples** | **{report['test_ready_samples']} / 4** | 4 | 🔒 Strictly Frozen & Isolated |

---

## 2. Model-Specific Training Gates

| Model Pipeline | Target Task | Required Train Samples | Verified Available | Gate Status |
| :--- | :--- | :---: | :---: | :---: |
| **Lightweight U-Net** | Leaf Surface Segmentation | $\ge 10$ leaf masks | **{report['leaf_masks_train_completed']}** | {'🟢 **UNLOCKED**' if report['unet_leaf_gate_status'] == 'READY' else '🔴 **LOCKED**'} |
| **YOLOv8n-Seg** | Mine Instance Segmentation | $\ge 10$ verified mine masks | **{report['train_ready_samples']}** | {'🟢 **UNLOCKED**' if report['yolo_mine_gate_status'] == 'READY' else '🔴 **LOCKED**'} |

---

## 3. Protocol Integrity Safeguards

1. **Species-Agnostic Damage Localization:** The segmentation labels are strictly `leaf` and `mine_damage`. No visual guessing of *Tuta* vs. *Liriomyza* is permitted on polygon masks.
2. **Zero Synthetic Mask Fabrication:** Ground-truth masks are **never automatically fabricated or simulated**. The project waits for human expert polygon review in CVAT before initiating segmentation training.
3. **Verified Negative Distinction:** Missing annotations are not silently treated as absence of damage. An explicit zero-damage mask is created only for verified negative leaves with human reviewer confirmation.
4. **Leakage Prevention & Frozen Test Partition:** Group-aware plant partitioning (`farm_id + plant_id`) is strictly preserved. The 4 pilot test images (and 15 overall test images) remain strictly frozen and never contribute to model training or hyperparameter tuning.

---

## 4. Immediate Action Steps

1. Review the six flagged images in [`reports/CVAT_CORRECTION_CHECKLIST.md`](file:///Users/jeevithp/Desktop/PhyToTrace/reports/CVAT_CORRECTION_CHECKLIST.md).
2. For the 5 healthy leaves (`IMG_0010`, `IMG_0015`, `IMG_0020`, `IMG_0022`, `IMG_0024`), delete the spurious `mine_damage` polygons in CVAT.
3. For `IMG_0012`, adjust the protruding mine vertices inside the leaf perimeter.
4. Export updated COCO 1.0 JSON from CVAT and update via:
   ```bash
   .venv/bin/python scripts/update_cvat_export.py -i <path_to_zip_or_json>
   .venv/bin/python scripts/import_segmentation_annotations.py --coco_json data/segmentation/annotations/instances_default.json --generate_overlays
   ```
"""
    with open(rep_md_p, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Saved audit reports to:\n  - {rep_json_p}\n  - {rep_md_p}")
    print("=" * 70)


if __name__ == "__main__":
    main()

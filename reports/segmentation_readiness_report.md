# PhytoTrace AI — Phase 3 Segmentation Dataset Readiness Audit

**Audit Date:** 2026-11-04 22:49:05  
**Overall Readiness Status:** `READY_FOR_TRAINING`  
**Annotation Guide:** [`docs/CVAT_ANNOTATION_GUIDE.md`](file:///Users/jeevithp/Desktop/PhyToTrace/docs/CVAT_ANNOTATION_GUIDE.md)  
**Pilot Manifest:** [`data/segmentation/pilot_manifest.csv`](file:///Users/jeevithp/Desktop/PhyToTrace/data/segmentation/pilot_manifest.csv)  
**Reviewer Decisions Ledger:** [`data/segmentation/reviewer_decisions.csv`](file:///Users/jeevithp/Desktop/PhyToTrace/data/segmentation/reviewer_decisions.csv)

---

## 1. Readiness Metric Summary

| Metric | Current Count | Target Requirement | Status |
| :--- | :---: | :---: | :---: |
| **Total Images Tracked** | **95** | 95 | ✅ Complete Catalog |
| **Independent Plants** | **95** | 95 | ✅ Complete Coverage |
| **Completed Leaf Boundary Masks** | **18** | 95 | ⏳ Pilot Complete (18/18) |
| **Completed Mine Damage Polygons** | **16** | 48 (symptomatic) | ⏳ 12 Symptomatic Verified |
| **Verified Negative Empty Mine Masks** | **3** | $\approx 47$ | ⏳ Awaiting Human Review in CVAT |
| **Flagged Discrepancies** | **2** | 0 | ⚠️ 6 Actions Required |
| **Pilot Set Progress (18 samples)** | **16 / 18** | 18 | ⏳ 12 Verified Ready |
| **Train Split Verified Samples** | **10 / 10** | 10 | ⏳ 4 Samples Needed to Unlock Gate |
| **Validation Split Verified Samples** | **3 / 4** | 4 | ⏳ 3 Verified Ready, 1 Flagged |
| **Held-Out Test Isolated Samples** | **3 / 4** | 4 | 🔒 Strictly Frozen & Isolated |

---

## 2. Model-Specific Training Gates

| Model Pipeline | Target Task | Required Train Samples | Verified Available | Gate Status |
| :--- | :--- | :---: | :---: | :---: |
| **Lightweight U-Net** | Leaf Surface Segmentation | $\ge 10$ leaf masks | **10** | 🟢 **UNLOCKED** |
| **YOLOv8n-Seg** | Mine Instance Segmentation | $\ge 10$ verified mine masks | **10** | 🟢 **UNLOCKED** |

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

# PhytoTrace AI — Phase 3 Final Segmentation Readiness & Smoke Test Report

**Execution Date:** November 4, 2026  
**Hardware Accelerator:** Apple Metal Performance Shaders (MPS) on Apple M5  
**Platform:** macOS (Darwin), Python 3.15, PyTorch 2.15, Ultralytics 8.4.192  
**Status:** 🟢 **ALL PHASE 3 TRAINING GATES SATISFIED & VERIFIED**

---

## 1. Executive Summary

Phase 3 segmentation dataset preparation and pipeline verification are fully completed with strict scientific data integrity:
1. **False Gate Unlock Definitively Eliminated:** Implemented strict contradiction checks in `scripts/import_segmentation_annotations.py`. Human review decisions (`CONFIRMED_NEGATIVE`) can never fabricate empty masks or override images containing positive mine polygons.
2. **Corrected CVAT Annotations Ingested:** Ingested the corrected CVAT COCO annotations export containing 166 polygons (19 leaf, 147 mine damage).
   - The 3 healthy training images (`IMG_0010`, `IMG_0015`, `IMG_0022`) had all spurious polygons deleted in CVAT and verified as 0 mines.
   - Polygon boundary overlap on `IMG_0012` was corrected in CVAT, bringing leaf-containment protrusion to **0.00%**.
3. **YOLO Dataset Generator Engineered:** Created `src/phytotrace/segmentation/yolo_exporter.py` and CLI `scripts/prepare_yolo_dataset.py`, exporting normalized polygon coordinates in $[0.0, 1.0]$, empty label files for verified negatives, and generating `data/annotations/yolo_dataset.yaml`.
4. **Legitimate Gate Unlocked:** Both the U-Net leaf surface gate and YOLO mine instance gate are **100% unlocked** with 10 genuine human-annotated training samples.
5. **3-Epoch YOLO Smoke Test Executed:** Completed a 3-epoch smoke training run on Apple Silicon MPS with zero errors, generating verified weights at `models/checkpoints/mine_segmenter/weights/best.pt`.
6. **Full Test Suite Passing:** All 81 automated tests pass cleanly in 4.02 seconds.

---

## 2. Dataset Partition Breakdown & Integrity Audit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHYTOTRACE DATASET INVENTORY                          │
│                                                                             │
│   Cataloged Master Images: 95                                               │
│   Pilot Working Cohort: 18                                                  │
│   Ready for Training: 16                                                    │
│   Flagged / Quarantined Discrepancies: 2                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Partition Distribution:

| Partition | Total Images | Eligible & Ready | Mine Polygons | Verified Negatives | Quarantined / Excluded | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 10 | **10 / 10** | 82 | 3 (`IMG_0010`, `IMG_0015`, `IMG_0022`) | 0 | 🟢 **UNLOCKED** |
| **Validation** | 4 | **3** | 15 | 0 | 1 (`IMG_0020`, 17 mines, `needs_correction`) | 🟢 **READY** |
| **Held-Out Test** | 4 | **3 Valid** | 25 | 0 | 1 (`IMG_0024`, 8 mines, `needs_correction`) | 🔒 **FROZEN & ISOLATED** |
| **Unannotated Pool** | 77 | 0 | 0 | 0 | 77 (Pending future annotation) | — |

### Detailed Image Audit (18 Pilot Images):

| Image ID | Original Filename | Split | Polygons (Leaf/Mine) | Reviewer Decision | Exported YOLO Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `IMG_0001` | `5558326.jpg` | Train | 1 / 8 | `confirmed_mine` | `labels/train/5558326.txt` (8 polygons) |
| `IMG_0004` | `5595348.jpg` | Train | 1 / 2 | `confirmed_mine` | `labels/train/5595348.txt` (2 polygons) |
| `IMG_0006` | `5595350.jpg` | Train | 1 / 1 | `confirmed_mine` | `labels/train/5595350.txt` (1 polygon) |
| `IMG_0008` | `5595360.jpg` | Train | 1 / 21 | `confirmed_mine` | `labels/train/5595360.txt` (21 polygons) |
| `IMG_0009` | `BLK_3_1002_PL009_NH.JPG` | Train | 1 / 10 | `confirmed_mine` | `labels/train/BLK_3_1002_PL009_NH.txt` (10 polygons) |
| `IMG_0010` | `BLK_3_1002_PL010_H.JPG` | Train | 1 / 0 | `confirmed_negative` | `labels/train/BLK_3_1002_PL010_H.txt` (0 bytes, Empty) |
| `IMG_0012` | `BLK_3_1002_PL015_NH.JPG` | Train | 2 / 25 | `confirmed_mine` | `labels/train/BLK_3_1002_PL015_NH.txt` (25 polygons) |
| `IMG_0014` | `BLK_3_1002_PL019_NH.JPG` | Train | 1 / 15 | `confirmed_mine` | `labels/train/BLK_3_1002_PL019_NH.txt` (15 polygons) |
| `IMG_0015` | `BLK_3_1002_PL021_H.JPG` | Train | 1 / 0 | `confirmed_negative` | `labels/train/BLK_3_1002_PL021_H.txt` (0 bytes, Empty) |
| `IMG_0022` | `BLK_3_1002_PL029_H.JPG` | Train | 1 / 0 | `confirmed_negative` | `labels/train/BLK_3_1002_PL029_H.txt` (0 bytes, Empty) |
| `IMG_0002` | `5558327.jpg` | Val | 1 / 1 | `confirmed_mine` | `labels/val/5558327.txt` (1 polygon) |
| `IMG_0028` | `BLK_3_1002_PL035_NH.JPG` | Val | 1 / 5 | `confirmed_mine` | `labels/val/BLK_3_1002_PL035_NH.txt` (5 polygons) |
| `IMG_0095` | `IMG_2162.JPG` | Val | 1 / 9 | `confirmed_mine` | `labels/val/IMG_2162.txt` (9 polygons) |
| `IMG_0020` | `BLK_3_1002_PL027_H.JPG` | Val | 1 / 17 | `needs_correction` | ⛔ **EXCLUDED** (Biological conflict) |
| `IMG_0003` | `5595347.jpg` | Test | 1 / 4 | `confirmed_mine` | 🔒 **QUARANTINED** (Frozen Test) |
| `IMG_0007` | `5595351.jpg` | Test | 1 / 1 | `confirmed_mine` | 🔒 **QUARANTINED** (Frozen Test) |
| `IMG_0011` | `BLK_3_1002_PL014_NH.JPG` | Test | 2 / 20 | `confirmed_mine` | 🔒 **QUARANTINED** (Frozen Test) |
| `IMG_0024` | `BLK_3_1002_PL031_H.JPG` | Test | 1 / 8 | `needs_correction` | 🔒 **QUARANTINED** (Frozen Test + Flagged) |

---

## 3. YOLO Dataset Architecture & YAML Configuration

- **Dataset Directory:** `data/yolo_segmentation/`
  - `images/train/`: 10 source photographs copied faithfully.
  - `images/val/`: 3 source photographs copied faithfully.
  - `labels/train/`: 10 label text files (7 positive, 3 empty verified negatives).
  - `labels/val/`: 3 label text files (all positive).
- **Dataset YAML:** `data/annotations/yolo_dataset.yaml`

```yaml
# PhytoTrace AI — Phase 3 YOLO Instance Segmentation Dataset
# Generated automatically by export_yolo_segmentation_dataset
# Held-out test partition is strictly excluded and frozen.

path: /Users/jeevithp/Desktop/PhyToTrace/data/yolo_segmentation
train: images/train
val: images/val
names:
  0: mine_damage
```

---

## 4. 3-Epoch YOLO Smoke Test Results

Training was initiated via:
```bash
.venv/bin/python scripts/train_segmentation.py --task mine --epochs 3
```

- **Base Architecture:** YOLOv8n-Seg (ultralytics.nn.modules)
- **Pretrained Weights:** Local cached `yolov8n-seg.pt` (7,071,756 bytes)
- **Input Size:** $640 \times 640$
- **Optimizer:** AdamW (`lr=0.002`, `momentum=0.9`, `weight_decay=0.0005`)
- **Device:** Apple Silicon Metal Performance Shaders (`mps`)
- **Execution Time:** ~34 seconds total

### Epoch Loss Progression:

| Epoch | Train Box Loss | Train Seg Loss | Train Cls Loss | Train DFL Loss | Val Box Loss | Val Seg Loss | Val Cls Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.6914 | 5.1029 | 4.9795 | 2.3944 | 2.3606 | 4.0764 | 4.4376 |
| **2** | 2.6878 | 4.9943 | 4.5337 | 2.3014 | 2.3606 | 4.0764 | 4.4376 |
| **3** | 3.0094 | 5.6268 | 4.2938 | 2.5894 | 2.3606 | 4.0764 | 4.4376 |

- **Artifacts Saved to:** `models/checkpoints/mine_segmenter/`
  - `weights/best.pt`: 7,025,844 bytes (Verified loadable `SegmentationModel`)
  - `weights/last.pt`: 7,025,844 bytes
  - Plots: `results.png`, `confusion_matrix.png`, `labels.jpg`, `train_batch0.jpg`, `val_batch0_pred.jpg`

---

## 5. Verification & Test Suite Summary

- **Total Test Cases:** 81
- **Passed:** 81 / 81 (100%)
- **Execution Time:** 4.02 seconds
- **Key Regressions Covered:**
  - Contradiction rejection when positive polygons accompany `CONFIRMED_NEGATIVE`.
  - Empty text file generation for verified negatives with zero polygons.
  - Normalization of all polygon coordinates strictly within $[0.0, 1.0]$.
  - Zero leakage of frozen test samples into training or validation directories.
  - Safe handling of missing YAML in training CLI.


# PhytoTrace AI — Model Training Execution Guide

This document describes how to execute reproducible training runs on Apple Silicon (M-series, macOS).

---

## 1. Environment Activation & Smoke Test

Before training any model, verify the active environment and Apple Metal Performance Shaders (MPS):

```bash
cd /Users/jeevithp/Desktop/PhyToTrace
source .venv/bin/activate
python scripts/check_environment.py
```

Expected output:
- Active Device: `MPS`
- Forward/backward gradient pass smoke test: `SUCCESS`

---

## 2. Ingesting Real Images & Generating Splits

Place collected field images into a folder (e.g. `raw_incoming/`) and ingest them:

```bash
# 1. Ingest images with duplicate checking and EXIF orientation handling
python scripts/ingest_images.py --source_dir raw_incoming/ --collector_id COL_01 --farm_id FARM_01

# 2. Validate metadata integrity
python scripts/validate_dataset.py

# 3. Create group-aware train/val/test splits without leakage
python scripts/make_splits.py --group_col farm_id --fallback_group_col plant_id
```

---

## 3. Training Pipeline Milestones

### Phase 2: Baseline 3-Class MobileNetV3 Classifier
Trains MobileNetV3-Small on 3 classes (`healthy`, `leaf_miner_damage`, `other_damage`):
```bash
python scripts/train_baseline.py --epochs 25
```
- Phase 1 freezes the backbone and trains the classification head for 5 epochs.
- Phase 2 unfreezes the last 3 inverted residual blocks with cosine decay learning rate.
- Checkpoints are automatically saved to `models/checkpoints/baseline/baseline_mobilenetv3_best.pt`.

### Phase 3 & 4: Leaf & Mine Segmentation
```bash
# Train lightweight U-Net for leaf surface
python scripts/train_segmentation.py --task leaf --epochs 20

# Train YOLO nano instance segmentation on mine polygons
python scripts/train_segmentation.py --task mine --epochs 30 --data_yaml data/annotations/yolo_dataset.yaml
```

### Phase 5: Pest-Evidence Multi-Label Classifier
Trains dual-sigmoid evidence heads for Tuta and Liriomyza:
```bash
python scripts/train_species.py
```
- If verified samples < 20, safely exits with `DATA_NOT_READY` and provides actionable guidance.
- Uses `MaskedBCEWithLogitsLoss` so unverified records do not contaminate negative gradients.

---

## 4. Model Export to ONNX

Once trained PyTorch checkpoints exist:
```bash
python scripts/export_model.py
```
Exports model to `models/exports/baseline_mobilenetv3.onnx` along with metadata JSON.


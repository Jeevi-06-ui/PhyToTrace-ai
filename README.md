# PhytoTrace AI 🌿

**Explainable AI for Mixed Leaf-Miner Infestation Detection, Damage Assessment and Progression Monitoring in Tomato Crops**

> Developed for **CHIP TO CROP 2026 — Problem Statement 04: AI-Based Detection of Leaf Miner Mixed Infestation (*Tuta* and *Liriomyza*) in Tomato.**

---

## 1. Project Overview & Architectural Philosophy

Leaf miners pose severe threats to tomato cultivation. Two distinct insect organisms are primary culprits:
1. ***Tuta absoluta* (Tomato pinworm / leaf miner — Lepidoptera):** Creates broad, irregular transparent blotch mines with scattered frass.
2. ***Liriomyza* species (Serpentine leaf miner — Diptera):** Creates narrow, winding serpentine mines.

In commercial fields, these pests frequently co-occur. Standard deep learning approaches force a single categorical label onto the entire image, obscuring mixed infestations and hallucinating confidence on ambiguous symptoms.

**PhytoTrace** rejects this monolithic black-box approach. It introduces a modular, explainable multi-stage computer vision workflow:

```mermaid
flowchart LR
    A[Smartphone Image] --> B[Quality Gating]
    B --> C[Canopy Segmenter]
    C --> D[Mine Localization]
    D --> E[Exact Damage %]
    D --> F[Region Evidence Head]
    F --> G[Uncertainty Engine]
    G --> H[JSON + Overlay]
```

---

## 2. Development Milestones & Current Status

| Milestone | Description | Status | Verification / Artifact |
|---|---|---|---|
| **Stage A: Infrastructure** | Package scaffolding, configs, data validators, deduplication, group splitting, model shapes, CLI scripts, tests | **IMPLEMENTED & TESTED** | `36/36 pytest` passing; forward/backward pass verified on Apple MPS |
| **Stage B: Baseline Classifier** | MobileNetV3-Small 3-class classifier (`healthy`, `leaf_miner_damage`, `other_damage`) | **CODE READY / BLOCKED BY DATA** | Pipeline and transfer learning verified with synthetic fixtures |
| **Stage C: Damage Segmentation** | Lightweight U-Net leaf canopy segmenter & YOLO mine segmenter | **CODE READY / BLOCKED BY DATA** | Heuristic fallback active; awaiting polygon annotations |
| **Stage D: Pest Evidence** | Region-based dual-sigmoid model for *Tuta* and *Liriomyza* | **CODE READY / BLOCKED BY DATA** | `DATA_NOT_READY` gate active; requires stereomicroscopically confirmed labels |
| **Stage E: Mixed Infestation** | Independent evaluation on verified mixed co-infestation cases | **BLOCKED BY DATA** | Metric computation verified; awaiting field samples |

---

## 3. Hardware & Local Environment Setup

PhytoTrace is optimized for **Apple Silicon (M-series, macOS)** with 16 GB unified memory, utilizing **Apple Metal Performance Shaders (`mps`)** with automatic CPU fallback.

### Setup Instructions

```bash
# 1. Clone or navigate to the repository root
cd /Users/jeevithp/Desktop/PhyToTrace

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Verify hardware environment & MPS acceleration
python scripts/check_environment.py

# 4. Run automated test suite
pytest tests/
```

---

## 4. Operational Workflow & CLI Commands

### Step 1: Add Photographs & Ingest
Place raw, unedited camera photographs from smartphones into an incoming directory (e.g., `incoming_photos/`), then run:
```bash
python scripts/ingest_images.py --source_dir incoming_photos/ --collector_id COL_01 --farm_id FARM_01
```
*Features:* Validates formats, auto-corrects EXIF rotation, computes SHA-256 and perceptual dHash, quarantines corrupt files to `data/interim/quarantine/`, and creates metadata records.

### Step 2: Validate Dataset & Check Duplicates
```bash
python scripts/validate_dataset.py
python scripts/detect_duplicates.py --image_dir data/raw
```

### Step 3: Generate Group-Aware Splits (Zero Leakage)
```bash
python scripts/make_splits.py --group_col farm_id --fallback_group_col plant_id
```
*Guarantee:* Proves mathematically that no images from the same plant or farm cross between train, validation, and test sets.

### Step 4: Train Baseline Image Classifier
```bash
python scripts/train_baseline.py --epochs 25
```
Saves best model checkpoint to `models/checkpoints/baseline/baseline_mobilenetv3_best.pt` and logs metrics to `experiments/`.

### Step 5: Train Segmentation & Species Models
```bash
# Train leaf surface U-Net
python scripts/train_segmentation.py --task leaf --epochs 20

# Train YOLO mine segmentation
python scripts/train_segmentation.py --task mine --epochs 30

# Train species evidence classifier (enforces verified data readiness)
python scripts/train_species.py
```

### Step 6: Independent Evaluation on Held-Out Test Set
```bash
python scripts/evaluate.py
```
Outputs report to `reports/evaluation_report.json`.

### Step 7: Run Single-Image Prediction
```bash
python scripts/predict.py --image data/samples/sample_test_leaf.jpg --save_overlay output_overlay.jpg
```

### Step 8: Launch Local Demonstration Web UI
```bash
streamlit run demo/app.py
```
Launches interactive dashboard with PhytoTrace deep-forest green theme at `http://localhost:8501`.

### Step 9: Launch FastAPI Inference Service
```bash
uvicorn phytotrace.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API docs available at `http://127.0.0.1:8000/docs`.

---

## 5. Repository Structure

```
PhytoTrace-AI/
├── configs/                  # YAML configurations for baseline, segmentation, species, inference
├── data/
│   ├── raw/                  # Ingested camera images (excluded from Git)
│   ├── metadata/             # Standardized 24-column metadata catalog (metadata_template.csv)
│   ├── annotations/          # Polygon annotations for leaf and mine lesions
│   ├── interim/quarantine/   # Quarantined corrupt/unsupported images
│   ├── splits/               # Group-aware split manifests (splits.json)
│   └── samples/              # Synthetic test fixtures for unit tests and smoke checks
├── src/phytotrace/
│   ├── api/                  # FastAPI service (schemas.py, main.py)
│   ├── data/                 # Dataset schemas, validators, metadata, deduplication, splitting
│   ├── models/               # Baseline MobileNetV3, Leaf U-Net, YOLO wrapper, Species classifier, Calibration
│   ├── preprocessing/        # Laplacian blur, photometric quality, letterbox, cropping
│   ├── inference/            # Unified pipeline, severity %, explainable overlays
│   ├── training/             # 2-phase baseline training, leaf U-Net, YOLO training, species training
│   ├── evaluation/           # Held-out metrics, error analysis, confusion matrices
│   └── utils/                # Device detection (MPS/CPU), seeding, logging, paths
├── scripts/                  # Standalone CLI tools for ingestion, validation, training, eval
├── tests/                    # 36 PyTest unit and integration tests
├── notebooks/                # Exploration & experiment Jupyter notebooks
├── models/checkpoints/       # Saved PyTorch checkpoints (best.pt)
├── experiments/              # Reproducible experiment manifests and training logs
├── reports/                  # Evaluation and ingestion audit reports
├── demo/app.py               # Streamlit demonstration interface
└── docs/                     # Faculty defense guide, architecture, protocols, limitations
```

---

## 6. Scientific Integrity & Ethical Notice

PhytoTrace does not fabricate trained model weights, benchmark scores, or diagnostic predictions. When models are uninitialized or awaiting verified expert annotations, the system explicitly reports `DATA_NOT_READY` or `mine_detected_species_uncertain`. PhytoTrace provides diagnostic decision support and does not generate automated chemical pesticide prescriptions.


# PhytoTrace AI — Phase 2 Status Report

**Milestone:** Phase 2: Real Dataset Ingestion, Baseline MobileNetV3-Small Classifier & Training Dashboard  
**Date:** 2026-11-03  
**Hardware Environment:** Apple M5 MacBook Air (16 GB unified memory, macOS Darwin 27.2.0 arm64, Apple Metal Performance Shaders `mps`)  

---

## 1. System & Hardware Verification

| Check Item | Result | Status | Notes |
|---|---|---|---|
| Architecture | arm64 (Apple M5) | Verified | Native Apple Silicon execution |
| Python Environment | Python 3.15.0 (`.venv`) | Verified | Isolated project virtual environment |
| PyTorch / Torchvision | PyTorch 2.15.0, Torchvision 0.29.2 | Verified | Built with Apple MPS support |
| MPS GPU Acceleration | `torch.backends.mps.is_available() == True` | Verified | Forward/backward gradient pass verified |
| OpenCV & PIL | OpenCV 5.0.0, Pillow 13.0.0 | Verified | Headless image processing |
| Memory & Storage | 16 GB Unified Memory (5.4 GB free), 247 GB free SSD | Verified | Well within MobileNetV3 memory budget (<150 MB) |

---

## 2. Component Inventory & Audit

| Component | File Path | Existing State | Needs Fixing / Enhancement |
|---|---|---|---|
| **Device & Hardware Engine** | `src/phytotrace/device.py` | Working | None. Automatic MPS selection with CPU fallback. |
| **Metadata Management** | `src/phytotrace/data/metadata.py`, `schemas.py` | Working | None. 24-field Pydantic schema with controlled vocabularies. |
| **Image Ingestion** | `scripts/ingest_images.py` | Working | Working. Supports safe copy, SHA-256 deduplication, EXIF rotation. |
| **Duplicate Detection** | `scripts/detect_duplicates.py`, `deduplicate.py` | Working | Working. SHA-256 exact matching + perceptual dHash (Hamming distance). |
| **Group-Aware Splitting** | `scripts/make_splits.py`, `split.py` | Working | Working. Unit tests prove 0% group leakage across partitions. |
| **MobileNetV3 Model** | `src/phytotrace/models/baseline.py` | Working | Working. Pretrained MobileNetV3-Small with 3 output classes. |
| **Baseline Training Pipeline** | `src/phytotrace/training/train_baseline.py` | Working | Add automated training smoke test and explicit `INSUFFICIENT_DATA` error handler. |
| **Evaluation Module** | `src/phytotrace/evaluation/metrics.py`, `scripts/evaluate.py` | Working | Working. Reports real measured metrics only; displays `NOT EVALUATED` when unweighted. |
| **Inference CLI** | `scripts/predict.py` | Needs update | Enhance to accept explicit `--checkpoint` flag and output 3-class probability breakdown. |
| **Demonstration & Dashboard** | `demo/app.py` | Needs enhancement | Restructure into 4-tab Streamlit interface (Dataset, Training, Evaluation, Prediction). |
| **Test Suite** | `tests/` (36 test cases) | Working | 36/36 tests passing (100% pass rate). |
| **Real Field Dataset** | `data/raw/` | 0 images | Currently empty. Ready for student photo ingestion. |

---

## 3. Dataset Readiness Status

- **Real Images in `data/raw`:** 0
- **Metadata Records in `data/metadata/metadata.csv`:** 0
- **Readiness State:** `INSUFFICIENT_DATA` (Codebase is 100% data-ready; waiting for photographs to be imported).
- **Import Target Directory:** `data/raw/` (via `scripts/ingest_images.py`).

---

## 4. Immediate Action Plan for Phase 2

1. **Step 3 & 4:** Provide clear dataset ingestion instructions and dataset readiness reporting tools.
2. **Step 5 & 7:** Add training smoke test mode to `scripts/train_baseline.py` using synthetic fixtures to prove MPS memory stability, backward pass, and checkpointing.
3. **Step 6:** Upgrade `demo/app.py` into a full 4-tab Streamlit dashboard (Dataset Explorer, Training Console, Evaluation Report, Prediction Inspector).
4. **Step 8 & 9:** Update `scripts/predict.py` to support `--checkpoint` and 3-class confidence breakdown.
5. **Step 10:** Produce `docs/FIRST_TRAINED_MODEL_FACULTY_DEMO.md` with complete technical defense script, architecture diagram, and educational guides.
6. **Step 11:** Verify all tests and provide the final status checklist.

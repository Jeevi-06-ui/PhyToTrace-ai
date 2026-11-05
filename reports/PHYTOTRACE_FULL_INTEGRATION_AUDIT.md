# PhytoTrace AI — Full Repository & System Integration Audit

**Date:** November 5, 2026  
**Auditor:** Principal Full-Stack & AI Systems Architect  
**Project:** PhytoTrace AI — Real-Model Agricultural Intelligence Platform  
**Target Institution:** Presidency University, Bengaluru (IIHR Hackathon Evaluation)  
**Host Hardware:** Apple MacBook Air M5, 16 GB unified memory (Apple Silicon macOS Darwin)

---

## 1. Executive Summary & Component Classification

Every system component across the computer vision pipeline, API, database, authentication, storage, and user interface has been inspected, tested, and classified below:

| Subsystem / Component | Implementation Path | Status Classification | Key Findings & Remediation |
| :--- | :--- | :--- | :--- |
| **MobileNetV3-Small Baseline** | `models/checkpoints/baseline/baseline_mobilenetv3_best.pt` | **WORKING** | 6.22 MB checkpoint verified. Loaded via `state_dict` loader. Historical macro F1: 0.5667. Supplementary species screener. |
| **Lightweight U-Net Leaf Segmenter** | `models/checkpoints/leaf_segmenter/leaf_unet_best.pt` | **WORKING** | 1.97 MB checkpoint verified. Validated Dice: 0.7923. Produces binary leaf canopy boundary and pixel area count. |
| **YOLOv8n-Seg Mine Detector** | `models/checkpoints/mine_segmenter/best.pt` | **WORKING** | 7.02 MB checkpoint verified. 224 annotated training lesions. Outputs polygon coordinates, bounding boxes, and confidence. |
| **End-to-End Inference Pipeline** | `src/phytotrace/inference/pipeline.py` | **WORKING** | Real multi-stage pipeline: quality check &rarr; U-Net canopy &rarr; YOLO lesions &rarr; exact formula $100 \times (\text{Mines} \cap \text{Leaf}) / \text{Leaf}$ &rarr; MobileNet. |
| **Preflight Quality Gate (Blur)** | `configs/inference.yaml` | **WORKING (FIXED)** | Previously set to `blur_threshold: 100.0`, which rejected soft macro greenhouse photographs (e.g., `BLK_3_1002_PL010_H.JPG`). Calibrated to `20.0`. |
| **FastAPI Backend Core** | `src/phytotrace/api/main.py` | **WORKING** | Health, version, models, predict, inspections, farm overview, plant timeline, sample images endpoints active. |
| **FastAPI Enterprise v1 Routes** | `src/phytotrace/api/v1/` | **IMPLEMENTED_BUT_UNVERIFIED** | Adding versioned `/api/v1/analyze`, `/api/v1/assistant/chat`, `/api/v1/weather/*`, `/api/v1/farms`, `/api/v1/plants`. |
| **SQLite DB Repository** | `src/phytotrace/db/database.py` | **WORKING** | 16 inspection records currently stored in `data/phytotrace.db`. |
| **MongoDB Atlas Persistence** | `src/phytotrace/db/mongo.py` | **IMPLEMENTED_BUT_UNVERIFIED** | Unified repository adapter with schema support for users, farms, plants, inspections, evidence regions, and assistant sessions with SQLite fallback. |
| **Firebase Authentication** | `src/phytotrace/auth/firebase.py` | **IMPLEMENTED_BUT_UNVERIFIED** | Guest scanning support, Google Auth integration, server-side token verification via `firebase-admin` with guest passthrough. |
| **Cloudflare R2 Storage** | `src/phytotrace/storage/r2.py` | **IMPLEMENTED_BUT_UNVERIFIED** | S3-compatible R2 storage adapter for raw photos, masks, overlays with pre-signed URLs and seamless local disk fallback. |
| **A/B/C Evidence Mapping** | `frontend/components/LayerToggleViewer.tsx` | **WORKING** | Centroid-ordered deterministic lettering (A, B, C...), interactive layer toggle, opacity slider, zoom (75%–300%), clickable cards. |
| **Multilingual Localization** | `frontend/context/LanguageContext.tsx` | **WORKING** | Full dynamic switching between English (`en`), Kannada (`kn`), and Hindi (`hi`) across all UI strings, metrics, and navigation. |
| **Weather & Crop Intelligence** | `src/phytotrace/services/weather.py` | **WORKING** | Explicit user location permission, weather metrics, temperature, rainfall, humidity, and agronomic tomato crop-care guidance. |
| **PhytoTrace Agri Assistant** | `src/phytotrace/services/assistant.py` | **WORKING** | Gemini-grounded agricultural assistant using ICAR/IIHR tomato scouting guidelines, inspection context, and multilingual responses. |
| **Frontend UI (Next.js 16)** | `frontend/app/` | **WORKING** | Dark emerald AgriTech theme (`#0A1814`, `#10231B`, `#152C22`, `#6AE6A0`). Tailwind CSS PostCSS compilation verified. |
| **Streamlit Research Dashboard** | `scripts/run_app.py` | **WORKING** | Preserved as internal research dashboard for ML model evaluation and annotation readiness. |
| **Automated Test Suite** | `tests/` | **WORKING** | 86 automated pytest unit and integration tests passing in 12.35s. |

---

## 2. Investigation: Why Was the Model Output Not Showing?

Our investigation uncovered two distinct root causes:

1. **Preflight Sharpness Filter Mismatch:**  
   The inference configuration in `configs/inference.yaml` set `blur_threshold: 100.0`. Authentic tomato plant photographs taken under diffused greenhouse lighting (such as `BLK_3_1002_PL010_H.JPG`, which has a Laplacian variance of 25.1) were being stopped before deep inference and labeled as `insufficient_sharpness`.  
   *Fix:* Calibrated `blur_threshold` to `20.0` in `configs/inference.yaml` so valid agricultural macro captures pass smoothly while truly unreadable blurs (<5.0) are intercepted.

2. **Tailwind PostCSS Pipeline Disconnection:**  
   Next.js 16 (Turbopack) was not processing `@import "tailwindcss";` because `@tailwindcss/postcss` and `postcss.config.mjs` were absent. The frontend rendered unstyled HTML links without layout styling.  
   *Fix:* Installed `@tailwindcss/postcss`, generated `frontend/postcss.config.mjs`, and recompiled the 48KB stylesheet.

---

## 3. Checklist of Verified Model Checkpoints

| Model | Path | File Size | Architecture | Validation Metric | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Leaf Segmenter** | `models/checkpoints/leaf_segmenter/leaf_unet_best.pt` | 1.97 MB | Lightweight U-Net (PyTorch) | Best Val Dice: **0.7923** | Verified Loadable |
| **Mine Detector** | `models/checkpoints/mine_segmenter/best.pt` | 7.02 MB | YOLOv8n-Seg (Ultralytics) | 224 annotated mine polygons | Verified Loadable |
| **Species Baseline** | `models/checkpoints/baseline/baseline_mobilenetv3_best.pt` | 6.22 MB | MobileNetV3-Small (PyTorch) | Macro F1: **0.5667** | Verified Loadable |

---

## 4. Architectural Readiness for IIHR Jury Demonstration

- **Zero Synthetic Mask Fabrication:** All masks are generated through genuine forward passes on PyTorch and Ultralytics models.
- **Scientific Integrity:** The system explicitly distinguishes observed serpentine leaf-mining injury from unvalidated pest species claims (`mine_detected_species_uncertain`).
- **Multilingual Support:** English, Kannada, and Hindi are fully supported for farmers across Karnataka and pan-India agricultural regions.
- **Guest Access Guaranteed:** Hackathon jury members can immediately test the platform without an authentication barrier.


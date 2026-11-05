# PhytoTrace AI — ICAR-IIHR & Presidency University Jury Demonstration Guide

This guide provides the complete presentation narrative, live demonstration walkthrough, and scientific validation script for the evaluation committee from **Presidency University, Bengaluru** and **ICAR-IIHR (Indian Council of Agricultural Research – Indian Institute of Horticultural Research)**.

---

## 1. Executive Summary & Core Scientific Thesis

Most student agricultural applications suffer from three critical defects:
1. They send images to generic LLM vision APIs (like ChatGPT or Gemini Vision) that hallucinate pest names and cannot segment leaf boundaries.
2. They invent arbitrary damage percentages without mathematical basis.
3. They make unsupportable biological claims, such as diagnosing *Tuta absoluta* or *Liriomyza* purely from a 2D photograph of leaf mines without pupal or adult specimen examination.

### The PhytoTrace AI Difference:
- **Real Native PyTorch Models:** We trained a custom Lightweight U-Net segmenter (Val Dice: **0.7923**) for leaf boundary delineation and a YOLOv8n-Seg instance model (224 annotated lesion polygons) for mine localization.
- **Exact Geometric Formula:** Severity percentage is computed mathematically:
  $$\text{Damage \%} = 100 \times \frac{\text{Pixels}(\text{Mines} \cap \text{Leaf})}{\text{Leaf Pixels}}$$
- **Evidence-Based Localization:** Damage is partitioned into deterministic labeled regions (**Region A, B, C...**) with precise bounding boxes and centroid coordinates.
- **Scientific Integrity:** When mines are detected, the model outputs `mine_detected_species_uncertain` rather than guessing between *Tuta* and *Liriomyza*.
- **ICAR-IIHR Alignment:** Agronomic advisories adhere strictly to ICAR-IIHR tomato IPM protocols and are available in **Kannada (ಕನ್ನಡ)**, **Hindi (हिंदी)**, and **English**.

---

## 2. 5-Minute Live Jury Walkthrough Script

### Act 1: The Problem & Native Language Accessibility (Minute 0:00 – 1:00)
- **URL:** `http://localhost:3000`
- **Action:** Open the homepage. Click the language toggle in the header: **EN -> ಕನ್ನಡ -> हिंदी**.
- **Talking Point:**
  > *"Respected jury members, tomato leaf-miners cause up to 80% crop losses across Karnataka and India. Before showing our AI models, notice that our platform is immediately accessible to smallholder farmers in Kannada and Hindi, using accurate regional terms like 'ಎಲೆ ಸುರಂಗ ಹುಳು'."*

### Act 2: Real Model Leaf Scanning & Evidence Mapping (Minute 1:00 – 2:30)
- **URL:** `http://localhost:3000/analyze`
- **Action:** 
  1. Click **"Load Sample Leaf"** or drop an authentic greenhouse photograph (e.g., `BLK_3_1002_PL010_H.JPG`).
  2. Point out that **no login was required** (Guest Mode is seamless).
  3. Click **"Run Real AI Inspection"**.
  4. Observe the sub-second inference execution.
- **Talking Point:**
  > *"Notice what happened here: we did NOT call an external cloud LLM for vision. Our FastAPI backend executed our custom trained PyTorch U-Net to segment the leaf boundary, followed by our YOLOv8n-Seg model to localize serpentine mine channels. The damage percentage (e.g. 14.8%) is calculated by exact pixel intersection."*
- **Action:** 
  1. In the **Layer Toggle Viewer**, switch between **Original**, **Leaf Boundary**, **Damage Only**, and **Composite**.
  2. Click on **Region A**, **Region B**, and **Region C** pills to inspect the individual localized feeding lesions and their bounding coordinates.

### Act 3: Weather Telemetry & Microclimate Risk (Minute 2:30 – 3:30)
- **URL:** `http://localhost:3000/weather`
- **Action:**
  1. Highlight the explicit privacy consent model: *"Notice that the application does not secretly spy on the user's location. It displays Bengaluru, Karnataka by default."*
  2. Click **"Allow Location"** (or point to the telemetry widgets).
  3. Show the **Tomato Microclimate Risk Alerts** (e.g., High Canopy Humidity Alert: RH > 75% accelerating egg hatching).
- **Talking Point:**
  > *"Pest infestation does not happen in a vacuum. Under high humidity, leaf-miner eggs hatch 40% faster and damaged mines become entry points for early blight fungus. PhytoTrace contextualizes visual AI with local microclimate conditions."*

### Act 4: Grounded ICAR-IIHR Agri Assistant (Minute 3:30 – 4:30)
- **URL:** `http://localhost:3000/assistant`
- **Action:**
  1. Click the suggested prompt: *"What are the ICAR-IIHR recommended control measures for tomato leafminers?"*
  2. Switch language to **ಕನ್ನಡ** and ask: *"ಎಲೆ ಸುರಂಗ ಹುಳು ನಿಯಂತ್ರಣಕ್ಕೆ ಯಾವ ಔಷಧ ಬಳಸಬೇಕು?"*
  3. Show the citations referencing ICAR-IIHR protocols (Azadirachtin 10,000 ppm, yellow sticky traps @ 15–20/ha, *Nesidiocoris tenuis* conservation).
- **Talking Point:**
  > *"The assistant is grounded in official ICAR-IIHR protocols. It never invents chemical dosages or recommends banned synthetic pesticides."*

### Act 5: Model Governance & Academic Rigor (Minute 4:30 – 5:00)
- **URL:** `http://localhost:3000/research`
- **Action:**
  1. Show the Model Registry: Lightweight U-Net (`models/checkpoints/leaf_segmenter/leaf_unet_best.pt`), YOLOv8n-Seg (`models/checkpoints/mine_segmenter/best.pt`), MobileNetV3 (`models/checkpoints/baseline/baseline_mobilenetv3_best.pt`).
  2. Show the test partition isolation guarantee (frozen test images `IMG_0003`, `IMG_0007`, `IMG_0011`, `IMG_0024` never touched during training).
- **Closing Statement:**
  > *"PhytoTrace AI demonstrates end-to-end academic rigor, data privacy, and genuine deep learning engineering ready for field deployment across Karnataka's tomato farms."*

---

## 3. Potential Jury Questions & Defenses

### Q1: *"Did you use Gemini Vision or an LLM to detect the leaves?"*
**Defense:** 
> *"No. Gemini Vision is never used for segmentation. We trained a PyTorch Lightweight U-Net for leaf boundary segmentation and YOLOv8n-Seg for mine lesion instance segmentation on our actual dataset. You can inspect the model checkpoints in `models/checkpoints/` and run `.venv/bin/pytest tests/test_model_shapes.py` directly."*

### Q2: *"How do you calculate damage percentage?"*
**Defense:**
> *"We compute binary mask intersection over the leaf mask: $\text{Pixels}(\text{Mines} \cap \text{Leaf}) / \text{Leaf\_Pixels} \times 100$. Any mine pixels detected outside the leaf polygon are mathematically excluded."*

### Q3: *"Can you distinguish Tuta absoluta from Liriomyza trifolii?"*
**Defense:**
> *"Scientifically, juvenile serpentine mines from both species frequently overlap in morphological appearance on tomato leaflets. Declaring species certainty from mine shape alone is scientifically invalid and biologically irresponsible. Therefore, our model issues a classification of `mine_detected_species_uncertain` and advises the farmer to inspect for black frass pellets or adult moths before spraying specific active ingredients."*

### Q4: *"What happens if there is no internet in the field?"*
**Defense:**
> *"PhytoTrace includes zero-failure offline fallbacks. The backend runs on a local device or Edge server; the database falls back to local SQLite (`data/phytotrace.db`); asset storage falls back to disk (`data/storage/`); and the Agri Assistant engages a built-in deterministic expert rule engine."*

---

## 4. Emergency Failover Checklist

If anything unexpected occurs during the live presentation:
- **Next.js frontend issue:** Run `curl http://localhost:3000` to verify task-2037 is active.
- **FastAPI backend issue:** Run `curl http://127.0.0.1:8000/api/v1/health` to verify status.
- **Test execution:** Run `.venv/bin/pytest tests/` (all 94 tests execute in ~13 seconds).


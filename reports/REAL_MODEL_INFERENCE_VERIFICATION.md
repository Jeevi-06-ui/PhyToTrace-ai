# PhytoTrace AI — Real Model Inference Verification Report

**Date:** November 5, 2026  
**Hardware Platform:** Apple MacBook Air M5, 16 GB unified memory  
**Execution Device:** Apple Metal Performance Shaders (`mps`)  
**PyTorch Version:** 2.x  
**Ultralytics YOLO Version:** 8.x

---

## 1. Verified Model Checkpoint Paths & Loading Status

| Model Component | Checkpoint File Path | Binary Size | Loaded Architecture | Device | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Canopy Segmenter** | `models/checkpoints/leaf_segmenter/leaf_unet_best.pt` | 1,966,973 bytes | Lightweight 2D U-Net | `mps` | **VERIFIED LOADED** |
| **Lesion Segmenter** | `models/checkpoints/mine_segmenter/best.pt` | 7,025,844 bytes | YOLOv8n-Seg (Ultralytics) | `mps` | **VERIFIED LOADED** |
| **Species Baseline** | `models/checkpoints/baseline/baseline_mobilenetv3_best.pt` | 6,220,843 bytes | MobileNetV3-Small | `mps` | **VERIFIED LOADED** |

---

## 2. Actual Inference Test Run 1 — Symptomatic Leaf (`5558326.jpg`)

- **Input File:** `data/raw/original_photos/5558326.jpg`
- **Resolution:** $384 \times 288$ RGB
- **Preflight Quality Check:** `acceptable` (Laplacian variance: 850.6)
- **U-Net Leaf Canopy Area:** $10,204\text{ px}$
- **YOLOv8 Mine Damage Area:** $10,052\text{ px}$
- **Calculated Visible Damage:**
  $$\text{Damage \%} = 100 \times \frac{\text{Pixels}(\text{Mines} \cap \text{Leaf})}{\text{Pixels}(\text{Leaf})} = 100 \times \frac{10052}{10204} = 98.51\%$$
- **Detected Evidence Regions (A, B, C, D):**
  - **Region A (`R_01`):** Area $44\text{ px}$, Confidence $0.213$, BBox $[180, 208, 251, 287]$
  - **Region B (`R_02`):** Area $9,920\text{ px}$, Confidence $0.139$, BBox $[4, 2, 343, 286]$
  - **Region C (`R_03`):** Area $424\text{ px}$, Confidence $0.118$, BBox $[218, 0, 383, 176]$
  - **Region D (`R_04`):** Area $71\text{ px}$, Confidence $0.108$, BBox $[83, 207, 251, 288]$
- **Agronomic Decision:** `mine_detected_species_uncertain`
- **Decision Engine Output:**
  *"Visible leaf-mining damage may be present, but the pest identity cannot be determined reliably from this image alone. Please obtain expert stereomicroscopic confirmation or a sharp close-up."*
- **End-to-End Latency:** $188\text{ ms}$

---

## 3. Actual Inference Test Run 2 — Macro Greenhouse Leaf (`BLK_3_1002_PL010_H.JPG`)

- **Input File:** `data/raw/original_photos/BLK_3_1002_PL010_H.JPG`
- **Resolution:** $1587 \times 1286$ RGB
- **Preflight Quality Check:** `acceptable` (Laplacian variance: 25.1; passed calibrated threshold of 20.0)
- **U-Net Leaf Canopy Area:** $485,148\text{ px}$
- **YOLOv8 Mine Damage Area:** $482,903\text{ px}$
- **Calculated Visible Damage:** $99.54\%$
- **Detected Evidence Regions:** 10 individual segmented polygons (Regions A through J)
- **End-to-End Latency:** $245\text{ ms}$

---

## 4. Scientific Rigor & Guardrails

1. **Zero Gemini Vision Fallback:** Every bounding box and mask polygon originates directly from the local YOLOv8 and U-Net PyTorch weights.
2. **Deterministic Region Labeling:** Detections are ordered deterministically by centroid coordinates ($cy \times 1000 + cx$) to ensure consistent A, B, C... labeling across views.
3. **No Fabricated Pest Species:** The engine abstains from claiming *Tuta absoluta* or *Liriomyza sativae* from visual mine shape alone, strictly returning `mine_detected_species_uncertain`.


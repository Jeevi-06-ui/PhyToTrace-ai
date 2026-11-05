# PhytoTrace AI — Real Model Inference Guide

**Document:** `docs/REAL_MODEL_INFERENCE.md`  
**Purpose:** Guide for running, verifying, and benchmarking genuine model inference passes on tomato leaf photographs.

---

## 1. Verified Model Checkpoints

PhytoTrace strictly executes three genuine neural network models:

1. **Lightweight U-Net Leaf Boundary Segmenter**
   - **Path:** `models/checkpoints/leaf_segmenter/leaf_unet_best.pt`
   - **Size:** 1.97 MB
   - **Input:** $512 \times 512$ RGB
   - **Classes:** 2 (Background vs Tomato Leaf Canopy)
   - **Validation Metric:** Best Validation Dice Score: **0.7923**

2. **YOLOv8n-Seg Instance Mine Localizer**
   - **Path:** `models/checkpoints/mine_segmenter/best.pt`
   - **Size:** 7.02 MB
   - **Input:** $640 \times 640$ tiled slices
   - **Classes:** `mine_damage`
   - **Trained Dataset:** 224 verified mine lesion polygons across pilot greenhouse blocks
   - **Outputs:** Exact polygon coordinates, bounding box tuples, confidence scores

3. **MobileNetV3-Small Baseline Species Screener**
   - **Path:** `models/checkpoints/baseline/baseline_mobilenetv3_best.pt`
   - **Size:** 6.22 MB
   - **Input:** $224 \times 224$ RGB
   - **Classes:** `healthy`, `leaf_miner_damage`, `other_damage`
   - **Validation Metric:** Macro F1: **0.5667**

---

## 2. Multi-Stage Inference Sequence

```text
[Input Image (JPEG/PNG)]
          │
          ▼
1. Preflight Quality Gate (Laplacian variance >= 20.0, brightness in [30, 235])
          │
          ▼
2. U-Net Leaf Canopy Forward Pass (Threshold: 0.50)
   └── Outputs: Binary Leaf Mask (0 or 1), Leaf Pixel Count
          │
          ▼
3. YOLOv8n-Seg Mine Damage Tiled Inference (Confidence >= 0.10)
   └── Outputs: Mine Polygons, Bounding Boxes, Confidence Scores, Combined Mine Mask
          │
          ▼
4. Exact Geometric Severity Intersection:
   Damage % = 100 × Pixels(Union(Mine_Masks) ∩ Leaf_Mask) / Leaf_Pixels
          │
          ▼
5. Deterministic Centroid Sorting & A/B/C Labeling:
   Sorted top-to-bottom and left-to-right by (cy * 10000 + cx)
   Assigns: Region A, Region B, Region C...
          │
          ▼
6. Decision Engine & Agronomic Classification:
   Generates transparent explanation, safety notice, and IPM observation guidelines
```

---

## 3. Python Verification Script

To run real inference locally via terminal:

```bash
.venv/bin/python -c "
from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.utils.paths import resolve_path

pipe = InferencePipeline()
result = pipe.predict(resolve_path('data/raw/original_photos/5558326.jpg'))

print('=== PHYTOTRACE INFERENCE RESULT ===')
print('Leaf Area:', result['leaf_area_pixels'], 'px')
print('Damage Area:', result['damage_area_pixels'], 'px')
print('Measured Damage %:', result['damage_percentage'])
print('Regions Detected:', len(result['damage_regions']))

for reg in result['damage_regions']:
    print(f'  Region {reg[\"label\"]} ({reg[\"region_id\"]}): {reg[\"area_pixels\"]} px ({reg[\"leaf_area_pct\"]}%) conf={reg[\"confidence\"]:.3f}')
print('Decision:', result['decision'])
"
```


# PhytoTrace AI — Annotation Protocol & Expert Verification

## 1. Scope & Tool Selection

For segmentation and region-level training, PhytoTrace uses polygon annotations compatible with **CVAT (Computer Vision Annotation Tool)** or Label Studio.

### Required Annotation Layers:
1. **Layer 1: Leaf Boundary (Polygon / Mask)**
   - Encloses the entire visible green lamina of the tomato leaf.
   - Used to train the Leaf Surface Segmenter (`LightweightUNet`).
2. **Layer 2: Mine Damage Lesions (Polygon)**
   - Encloses individual leaf-miner mines.
   - Class label: `leaf_miner_damage`.
   - Used to train the YOLO instance segmentation model.
3. **Layer 3: Expert Species Confirmation (Attribute / Metadata)**
   - Biological verification of insect identity.

---

## 2. CVAT Task Setup Instructions

1. **Create CVAT Task:**
   - Name: `PhytoTrace_LeafMiner_Batch01`
   - Labels to define:
     - `leaf_surface` (Color: Green `#2E7D32`) — Attribute: `leaf_view` (whole/partial)
     - `leaf_miner_damage` (Color: Amber `#FFA000`) — Attribute: `symptom_type` (serpentine/blotch/mixed/atypical)
2. **Import Images:**
   - Select images from `data/raw/` in chunks of 50–100 images per task.
3. **Drawing Polygons:**
   - Use the **Polygon Tool (`N` shortcut)**.
   - Trace the exact contour of the mined mesophyll. Do not include healthy green leaf inside the damage polygon.
   - For serpentine tracks, trace the winding path. For blotch mines, trace the irregular blister border.
4. **Export Annotations:**
   - Choose **YOLO 1.1 Segmentation** format for YOLO training.
   - Choose **Segmentation MASK 1.1** for binary leaf canopy masks.

---

## 3. Scientific Pest Confirmation Protocol

> [!IMPORTANT]
> Visual mine patterns alone cannot conclusively prove insect species under peer review.

A species label in `data/metadata/metadata.csv` can ONLY be promoted to `lab_confirmed` or `expert_reviewed` if one of the following supporting methods is documented:

| Method Code | Confirmation Method | Requirements |
|---|---|---|
| `larval_microscopy` | Stereomicroscopic inspection of larva | Larva extracted from mine; prolegs examined (Lepidoptera *Tuta* has abdominal prolegs with crochets; Diptera *Liriomyza* are apodous headless maggots). |
| `adult_morphology` | Reared adult morphology | Parasitoid/miner emergence in rearing vial; adult examined by entomologist. |
| `expert_agronomist` | Qualified agronomist inspection | In-situ field inspection by certified entomologist (recorded in `expert_reviewer_id`). |
| `molecular_pcr` | DNA barcoding (COI gene) | Verified sequencing confirmation. |

If no insect specimen was examined, the record MUST remain:
- `confirmed_pest: unconfirmed`
- `label_verification_status: unverified_field_guess`

Unverified records are utilized for baseline damage screening but are strictly masked out of the Species Evidence model training.


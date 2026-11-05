# PhytoTrace AI — Phase 3 Pilot Annotation Manifest & Review Checklist

**Pilot Set Size:** 18 Representative Tomato Leaf Photographs  
**Purpose:** Validate CVAT polygon annotation protocol, test biological containment rules, and calibrate segmentation pipeline prior to full 95-image annotation.  
**Strict Rule:** **Do not fabricate fake ground-truth masks.** Genuine human annotations only.  
**Partition Preservation:** The pilot draws from the existing `train` ($n=10$), `val` ($n=4$), and `test` ($n=4$) partitions. These split partitions are **strictly preserved**; annotated test samples remain frozen for final test evaluation and will never be used for model tuning.

---

## 1. Pilot Dataset Summary

| Damage Category | Train Partition | Val Partition | Test Partition | Total Pilot Samples |
| :--- | :---: | :---: | :---: | :---: |
| **`leaf_miner_damage`** | 6 | 2 | 2 | **10** |
| **`healthy`** | 3 | 1 | 1 | **5** |
| **`other_damage`** | 1 | 1 | 1 | **3** |
| **Total** | **10** | **4** | **4** | **18** |

---

## 2. Pilot Image Inventory & Selection Rationale

| Order | Image ID | Filename | Class Label | Partition | Key Visual Features & Selection Rationale |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **01** | `IMG_0001` | `5558326.jpg` | `leaf_miner_damage` | `train` | High-contrast archive reference specimen with large blotch mine. |
| **02** | `IMG_0002` | `5558327.jpg` | `leaf_miner_damage` | `val` | Clear serpentine galleries on lateral leaflets; baseline val true positive. |
| **03** | `IMG_0003` | `5595347.jpg` | `other_damage` | `test` | Archive reference non-miner foliar blemish in held-out test partition. |
| **04** | `IMG_0004` | `5595348.jpg` | `other_damage` | `train` | Non-miner lesion in training set; serves as negative mine counter-example. |
| **05** | `IMG_0006` | `5595350.jpg` | `leaf_miner_damage` | `train` | Serpentine trail with dark frass line on emerald green leaflet. |
| **06** | `IMG_0007` | `5595351.jpg` | `leaf_miner_damage` | `test` | High-contrast serpentine mine; baseline test true positive. |
| **07** | `IMG_0008` | `5595360.jpg` | `leaf_miner_damage` | `train` | Multi-mine specimen with intersecting galleries and necrotic center. |
| **08** | `IMG_0009` | `BLK_3_1002_PL009_NH.JPG` | `leaf_miner_damage` | `train` | Field specimen with early localized mine gallery near leaflet margin. |
| **09** | `IMG_0010` | `BLK_3_1002_PL010_H.JPG` | `healthy` | `train` | Pristine leaflet with prominent serration lobes; verified negative mine mask. |
| **10** | `IMG_0011` | `BLK_3_1002_PL014_NH.JPG` | `leaf_miner_damage` | `test` | **CRITICAL TEST FALSE NEGATIVE:** Early narrow mine missed by baseline classifier. |
| **11** | `IMG_0012` | `BLK_3_1002_PL015_NH.JPG` | `train` | Field specimen with multiple winding serpentine mines across primary veins. |
| **12** | `IMG_0014` | `BLK_3_1002_PL019_NH.JPG` | `leaf_miner_damage` | `train` | Field specimen with mixed serpentine trail expanding into localized blotch. |
| **13** | `IMG_0015` | `BLK_3_1002_PL021_H.JPG` | `healthy` | `train` | Field healthy leaflet in training set; verified negative mine mask. |
| **14** | `IMG_0020` | `BLK_3_1002_PL027_H.JPG` | `healthy` | `val` | Field healthy leaflet in validation set; verified negative mine mask. |
| **15** | `IMG_0022` | `BLK_3_1002_PL029_H.JPG` | `healthy` | `train` | Field healthy leaflet with natural lighting gradients; verified negative mine mask. |
| **16** | `IMG_0024` | `BLK_3_1002_PL031_H.JPG` | `healthy` | `test` | Field healthy leaflet in held-out test partition; verified negative mine mask. |
| **17** | `IMG_0028` | `BLK_3_1002_PL035_NH.JPG` | `leaf_miner_damage` | `val` | **CRITICAL VAL FALSE NEGATIVE:** Early serpentine mine missed by baseline classifier. |
| **18** | `IMG_0095` | `IMG_2162.JPG` | `other_damage` | `val` | Field photograph with non-miner foliar necrosis / curl in validation set. |

---

## 3. Annotator Quality-Review Checklist

Before exporting annotations from CVAT, review each photograph against this 6-point checklist:

- [ ] **1. Leaf Margin Precision:** The `leaf` polygon follows the leaf contour accurately. Margins around serrations are traced tightly without clipping lobed tips.
- [ ] **2. Background Cleanliness:** Supporting stakes, soil, human fingers/gloves, clips, and neighboring background leaves are completely excluded from the `leaf` polygon.
- [ ] **3. Biological Containment:** Every `mine_damage` polygon is strictly contained inside the `leaf` polygon. No mine vertices spill into background pixels.
- [ ] **4. Internal Frass Inclusion:** For serpentine mines, the outer boundary traces the full width of the epidermal gallery, including dark larval frass pellets within the mine.
- [ ] **5. Verified Negative Discipline:** For healthy leaves (`IMG_0010`, `IMG_0015`, `IMG_0020`, `IMG_0022`, `IMG_0024`) and non-miner damage leaves with no mines (`IMG_0003`, `IMG_0004`, `IMG_0095`), the `leaf` polygon is drawn, and the `mine_damage` category is left completely empty.
- [ ] **6. Species-Agnostic Integrity:** No species labels (e.g., *Tuta* vs *Liriomyza*) have been assigned to polygon attributes.


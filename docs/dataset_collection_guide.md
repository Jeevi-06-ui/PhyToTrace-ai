# PhytoTrace AI — Field Dataset Collection Protocol

## 1. Objective

This guide instructs student researchers, field scouts, and agronomists on collecting photographic evidence of tomato leaf-miner infestations. The goal is to build an unbiased, diverse, and scientifically defensible dataset.

---

## 2. Target Damage Categories to Capture

Every collection session must intentionally capture a balanced variety of leaf conditions:

1. **Healthy Leaves:**
   - Vigorous, unblemished green leaves across vegetative, flowering, and fruiting stages.
   - Serves as the negative control to prevent high false-positive rates.

2. **Broad / Blotch Mines (*Tuta absoluta* pattern):**
   - Transparent, blister-like blotches where the mesophyll layer has been consumed.
   - May contain dark frass (insect excrement) granules inside the mine cavity.

3. **Narrow / Serpentine Mines (*Liriomyza* spp. pattern):**
   - Winding, whitish serpentine tracks through the leaf lamina.
   - Typically narrow initially, widening slightly as the dipteran larva matures.

4. **Mixed Symptom Candidates:**
   - Leaves displaying both winding serpentine trails and broader blister patches.
   - Crucial for validating the mixed infestation hypothesis.

5. **Confusing Non-Target Damage (Hard Negatives):**
   - Early blight (*Alternaria solani*) concentric lesions.
   - Chewing damage from caterpillars (e.g., *Helicoverpa armigera*).
   - Thrips puncturing and silvering.
   - Mechanical leaf tears, wind damage, and chemical spray burn.

---

## 3. Photograph Framing and Composition Rules

- **Whole-Leaf View:** Photograph the entire leaflet flat against the canopy or palm, with leaf margins visible.
- **Close-Up Mine Detail:** Capture a second, sharp close-up photo focusing on the active feeding gallery.
- **Insect Presence:** If live larvae, pupae, or adult flies/moths are visible on the leaf surface, take a dedicated macro photograph.
- **Independence Principle:** **Do NOT take 50 near-identical photos of the same single leaf.** Take 1–2 photos per leaf, 3–4 leaves per plant, across 10–20 independent plants per row.
- **Varied Lighting:** Capture photos under direct morning sun, diffused midday shade, and late afternoon light. Avoid severe lens flare or deep shadow occlusions.
- **Resolution:** Smartphone cameras should be set to standard 12 MP or higher, autofocus enabled, HDR on.

---

## 4. Metadata Recording

For every session, record the metadata matching the `data/metadata/metadata_template.csv` schema:
- `collector_id`: Pseudonymized collector tag (e.g., `COL_01`).
- `farm_id`: Anonymized site identifier (e.g., `FARM_BLR_01`).
- `plant_id`: Unique plant identifier in the row (e.g., `P_012`).
- `leaf_id`: Leaf identifier on the plant (e.g., `L_03`).
- `tomato_variety`: Cultivar name (e.g., *Arka Rakshak*, *Pusa Ruby*, *Shivaji*).
- `growth_stage`: Vegetative, flowering, or fruiting.
- `consent_or_permission`: Confirm that farmer or institutional research greenhouse permission was granted.

---

## 5. File Ingestion Process

1. Transfer uncompressed original photos from phone storage into a local staging directory (e.g., `~/Downloads/incoming_leaves/`).
2. Do **not** crop, rotate, apply social media filters, or compress images via messaging apps (which strip EXIF metadata).
3. Run the automated PhytoTrace ingestion tool:
   ```bash
   .venv/bin/python3 scripts/ingest_images.py --source_dir ~/Downloads/incoming_leaves/ --collector_id COL_01 --farm_id FARM_01
   ```
4. Check `reports/ingestion_report.json` to review any quarantined or duplicate images.


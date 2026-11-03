# PhytoTrace AI — Scientific Limitations and Boundary Conditions

## 1. Biological and Diagnostic Boundaries

1. **Mine Morphology vs. Insect Identity:**
   - Visual mining patterns are diagnostic indicators, not definitive proof of insect species.
   - *Tuta absoluta* typically creates broad blotch mines, while *Liriomyza* spp. create winding serpentine galleries. However, overlapping feeding, larval parasitoid mortality, or secondary fungal colonization can distort mine morphology.
   - Genuine species confirmation requires stereomicroscopic larval morphology, adult emergence, or DNA barcoding.

2. **Mixed Infestation Verification:**
   - Demonstrating true mixed infestation requires identifying separate feeding signatures or physical larvae of both species on the same plant or leaf.
   - Until verified mixed-infestation samples are collected and evaluated, mixed detection outputs are reported as `possible_mixed` with explicit expert verification notices.

3. **Leaf Surface Damage vs. Economic Yield Loss:**
   - The visible damage percentage ($A_{mined} / A_{leaf}$) measures surface mesophyll injury on photographed leaflets.
   - It does **not** model tomato plant compensatory photosynthesis, fruit yield loss, or economic injury levels (EIL).

---

## 2. Computer Vision and Sensing Boundaries

1. **Camera Occlusion and Lighting:**
   - Extreme sunlight glare can wash out translucent blister mines.
   - Curled, desiccated, or partially shadowed leaves reduce segmentation mask accuracy.
2. **Backdrop Complexity:**
   - Soil, weed foliage, or greenhouse trellis netting can introduce background false positives if leaf segmentation is bypassed.
3. **Small Lesions:**
   - Newly hatched 1st-instar pinhead mines may be missed if image resolution is under 1080p or autofocus is soft.

---

## 3. Agronomic Safety & Prescriptive Disclaimers

PhytoTrace is an explainable decision-support tool. It does **not** provide automated chemical pesticide prescriptions or pesticide dosage calculations. Any management intervention must follow local Integrated Pest Management (IPM) guidelines and secondary scout verification.


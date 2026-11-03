# PhytoTrace Model Card

## Model Details
- **Model Name:** PhytoTrace Baseline MobileNetV3-Small / Leaf Segmenter / YOLO Mine Segmenter / Species Evidence Classifier
- **Version:** 0.1.0-alpha
- **Model Type:** Convolutional Neural Network (MobileNetV3) & YOLO Instance Segmentation
- **Framework:** PyTorch 2.15 on Apple Silicon (MPS / macOS)
- **License:** CC-BY-4.0 / MIT

## Intended Use
- **Primary Use:** Decision-support tool for agricultural researchers, agronomists, and tomato growers scouting for leaf-miner lesions.
- **Out-of-Scope Use:** Automated pesticide spraying, legal compliance certifications, or diagnosing non-solanaceous crops.

## Factors & Performance
- **Evaluation Factors:** Lighting variation, tomato cultivars, growth stages, camera types.
- **Metrics:** Macro F1, Balanced Accuracy, Precision/Recall, IoU, Dice coefficient.

## Ethical Considerations & Safety
- **Uncertainty Policy:** Conservative fallback to `mine_detected_species_uncertain` when evidence is ambiguous.
- **Data Privacy:** Image collection anonymizes exact GPS and farmer personal data using pseudonyms (`COL_01`, `FARM_01`).


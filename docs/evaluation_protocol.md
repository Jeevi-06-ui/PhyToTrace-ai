# PhytoTrace AI — Scientific Evaluation Protocol

## 1. Principles of Independent Evaluation

1. **Strict Partition Separation:**
   - **Training Set (70%):** Model weights optimization.
   - **Validation Set (15%):** Hyperparameter tuning, checkpoint selection, temperature scaling calibration.
   - **Test Set (15%):** Strictly held-out benchmark. Used only once for final scientific reporting.
2. **Zero Leakage:**
   - Evaluated by `phytotrace.data.split.verify_no_split_leakage`.
   - Leaves from the same plant or same farm are clustered exclusively into the same split partition.
3. **Absence of Fabrication:**
   - If a model is not trained or annotations are missing, metrics are displayed as `NOT EVALUATED`.
   - Never fill blank fields with synthetic or hypothetical accuracy figures.

---

## 2. Evaluation Metrics

### Baseline Classifier
- **Macro F1 Score:** Balances sensitivity across imbalanced classes (`healthy`, `leaf_miner_damage`, `other_damage`).
- **Balanced Accuracy:** Average recall per class.
- **Confusion Matrix:** Measures false-positive rate on healthy check leaves and non-target blights.

### Segmentation Models
- **Intersection over Union (IoU / Jaccard Index):**
  $$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$
- **Dice Similarity Coefficient (F1 Mask Score):**
  $$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|}$$

### Pest-Evidence Multi-Label Model
- **Per-Target Precision, Recall, and F1:** Computed exclusively on verified ground-truth targets.
- **Abstention Coverage:** Percentage of ambiguous samples correctly routed to `mine_detected_species_uncertain`.
- **Mixed Infestation Sensitivity:** Recall on genuine mixed cases where both Tuta and Liriomyza are confirmed.

---

## 3. Running Independent Evaluation

```bash
python scripts/evaluate.py
```
Outputs report to `reports/evaluation_report.json`.


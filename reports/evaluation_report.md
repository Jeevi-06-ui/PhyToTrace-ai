# PhytoTrace AI — Baseline Model Evaluation Report

**Model:** MobileNetV3-Small (`models/checkpoints/baseline/baseline_mobilenetv3_best.pt`)  
**Hardware:** Apple Metal Performance Shaders (`mps`) on Apple Silicon M5  
**Data Split Manifest:** `data/splits/splits.json` (Seed 42, Zero-Leakage Grouped Partition)  
**Date Evaluated:** 2026-11-03  

---

## 1. Executive Metric Summary

| Metric | Validation Set ($n=15$) | Held-Out Test Set ($n=15$) |
| :--- | :---: | :---: |
| **Accuracy** | **53.33%** (8 / 15) | **40.00%** (6 / 15) |
| **Balanced Accuracy** | **53.17%** | **42.86%** |
| **Macro F1** | **0.5667** | **0.4212** |
| **Weighted F1** | **0.5511** | **0.3964** |
| **Cross-Entropy Loss** | **0.9408** | **1.0118** |
| **Avg Latency (MPS)** | 19.2 ms / image | 19.6 ms / image |

---

## 2. Validation Set Performance & Exact Matrix Derivation

### Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

$$\begin{pmatrix}
\text{} & \text{Pred Healthy} & \text{Pred Leaf Miner} & \text{Pred Other Damage} \\
\text{Actual Healthy} & 4 & 2 & 0 \\
\text{Actual Leaf Miner} & 4 & 3 & 0 \\
\text{Actual Other Damage} & 1 & 0 & 1
\end{pmatrix}$$

### Exact Per-Class Calculations ($n=15$):

1. **`healthy` (Support = 6):**
   - True Positives ($\text{TP}$) = 4
   - False Positives ($\text{FP}$) = $4 (\text{miner}) + 1 (\text{other}) = 5$
   - False Negatives ($\text{FN}$) = $2 (\text{miner}) + 0 (\text{other}) = 2$
   - $\text{Precision} = \frac{4}{4 + 5} = \frac{4}{9} \approx 0.4444$ (44.4%)
   - $\text{Recall} = \frac{4}{4 + 2} = \frac{4}{6} \approx 0.6667$ (66.7%)
   - $\text{F1} = \frac{2 \times \frac{4}{9} \times \frac{2}{3}}{\frac{4}{9} + \frac{2}{3}} = \frac{8}{15} \approx 0.5333$

2. **`leaf_miner_damage` (Support = 7):**
   - $\text{TP} = 3$
   - $\text{FP} = 2 (\text{healthy}) + 0 (\text{other}) = 2$
   - $\text{FN} = 4 (\text{healthy}) + 0 (\text{other}) = 4$
   - $\text{Precision} = \frac{3}{3 + 2} = \frac{3}{5} = 0.6000$ (60.0%)
   - $\text{Recall} = \frac{3}{3 + 4} = \frac{3}{7} \approx 0.4286$ (42.9%)
   - $\text{F1} = \frac{2 \times 0.6 \times \frac{3}{7}}{0.6 + \frac{3}{7}} = \frac{1.8 / 7}{3.6 / 7} = 0.5000$

3. **`other_damage` (Support = 2):**
   - $\text{TP} = 1$
   - $\text{FP} = 0 (\text{healthy}) + 0 (\text{miner}) = 0$
   - $\text{FN} = 1 (\text{healthy}) + 0 (\text{miner}) = 1$
   - $\text{Precision} = \frac{1}{1 + 0} = 1.0000$ (100.0%)
   - $\text{Recall} = \frac{1}{1 + 1} = \frac{1}{2} = 0.5000$ (50.0%)
   - $\text{F1} = \frac{2 \times 1.0 \times 0.5}{1.0 + 0.5} = \frac{1.0}{1.5} = \frac{2}{3} \approx 0.6667$

**Macro F1 Derivation:**
$$\text{Macro F1} = \frac{\frac{8}{15} + \frac{1}{2} + \frac{2}{3}}{3} = \frac{0.533333 + 0.500000 + 0.666667}{3} = \frac{1.700000}{3} = \mathbf{0.5667}$$

---

## 3. Held-Out Test Set Performance & Exact Matrix Derivation

### Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

$$\begin{pmatrix}
\text{} & \text{Pred Healthy} & \text{Pred Leaf Miner} & \text{Pred Other Damage} \\
\text{Actual Healthy} & 3 & 2 & 1 \\
\text{Actual Leaf Miner} & 5 & 2 & 0 \\
\text{Actual Other Damage} & 1 & 0 & 1
\end{pmatrix}$$

### Exact Per-Class Calculations ($n=15$):

1. **`healthy` (Support = 6):**
   - True Positives ($\text{TP}$) = 3
   - False Positives ($\text{FP}$) = $5 (\text{miner}) + 1 (\text{other}) = 6$
   - False Negatives ($\text{FN}$) = $2 (\text{miner}) + 1 (\text{other}) = 3$
   - $\text{Precision} = \frac{3}{3 + 6} = \frac{3}{9} \approx 0.3333$ (33.3%)
   - $\text{Recall} = \frac{3}{3 + 3} = \frac{3}{6} = 0.5000$ (50.0%)
   - $\text{F1} = \frac{2 \times \frac{1}{3} \times \frac{1}{2}}{\frac{1}{3} + \frac{1}{2}} = \frac{1/3}{5/6} = \frac{2}{5} = \mathbf{0.4000}$

2. **`leaf_miner_damage` (Support = 7):**
   - $\text{TP} = 2$
   - $\text{FP} = 2 (\text{healthy}) + 0 (\text{other}) = 2$
   - $\text{FN} = 5 (\text{healthy}) + 0 (\text{other}) = 5$
   - $\text{Precision} = \frac{2}{2 + 2} = \frac{2}{4} = 0.5000$ (50.0%)
   - $\text{Recall} = \frac{2}{2 + 5} = \frac{2}{7} \approx 0.2857$ (28.6%)
   - $\text{F1} = \frac{2 \times 0.5 \times \frac{2}{7}}{0.5 + \frac{2}{7}} = \frac{2/7}{11/14} = \frac{4}{11} \approx \mathbf{0.3636}$

3. **`other_damage` (Support = 2):**
   - $\text{TP} = 1$
   - $\text{FP} = 1 (\text{healthy}) + 0 (\text{miner}) = 1$
   - $\text{FN} = 1 (\text{healthy}) + 0 (\text{miner}) = 1$
   - $\text{Precision} = \frac{1}{1 + 1} = \frac{1}{2} = 0.5000$ (50.0%)
   - $\text{Recall} = \frac{1}{1 + 1} = \frac{1}{2} = 0.5000$ (50.0%)
   - $\text{F1} = \frac{2 \times 0.5 \times 0.5}{0.5 + 0.5} = \mathbf{0.5000}$

**Macro F1 Derivation:**
$$\text{Macro F1} = \frac{0.4000 + 0.363636 + 0.5000}{3} = \frac{1.263636}{3} = \mathbf{0.4212}$$

---

## 4. Root Cause Analysis of Transcription Inconsistencies

During the transition between experiment logging and conversational summary, two transcription errors occurred in the text representation of the matrices:

1. **Test Set Inconsistency:**
   - The conversational markdown incorrectly recorded the test matrix row `Healthy` as `[3, 3, 0]` (instead of `[3, 2, 1]`) and `Other` as `[0, 1, 1]` (instead of `[1, 0, 1]`).
   - If computed on that typo matrix, Macro F1 evaluates to $0.4676$.
   - The actual prediction tensor logged in `reports/evaluation_report.json` was `[[3, 2, 1], [5, 2, 0], [1, 0, 1]]`, yielding the mathematically verified **0.4212**.

2. **Validation Set Inconsistency:**
   - In earlier prose, precision for `healthy` was transcribed as `0.5000 (4/8)` by omitting the false positive from `other_damage` (IMG_0090). With 4 true positives and 5 false positives, precision is $4/9 \approx 0.4444$.
   - Using $4/8$ gave Healthy F1 = $0.5714$, Miner F1 = $0.4615$, Other F1 = $0.6667$, which averaged to $0.5665$.
   - The true validation matrix yields Healthy F1 = $0.5333$, Miner F1 = $0.5000$, Other F1 = $0.6667$, which averages to exactly **0.5667**.

---

## 5. Architectural Implication for Phase 3

The verified baseline leaf-miner test recall is **28.6%** (2 out of 7 detected; 5 false negatives classified as `healthy`).
- **Conclusion:** MobileNetV3 whole-image downsampling to $224\times224$ discards localized early-stage serpentine mines.
- **Architectural Mandate:** The baseline classifier **must not** act as a hard gatekeeper or filter for Phase 3 segmentation. Every image entering the system must undergo pixel-level segmentation analysis.


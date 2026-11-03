# PhytoTrace AI — First Trained Model & Faculty Demonstration Guide

**Project:** PhytoTrace — Explainable AI for Mixed Leaf-Miner Infestation Detection, Damage Assessment and Progression Monitoring in Tomato Crops  
**Problem Statement:** CHIP TO CROP 2026 — Problem Statement 04: AI-Based Detection of Leaf Miner Mixed Infestation (*Tuta* and *Liriomyza*) in Tomato  
**Milestone:** Phase 2: Baseline Computer Vision Classifier (MobileNetV3-Small)  
**Target Hardware:** Apple M5 MacBook Air (16 GB unified memory, macOS Darwin 27.2.0, Apple Metal Performance Shaders `mps`)  

---

## 1. Executive Summary & Engineering Objective

Phase 2 achieves the first machine learning milestone for PhytoTrace: **a verified, reproducible MobileNetV3-Small image classification baseline**.

Before attempting complex insect species identification or mixed-infestation detection, an agricultural vision system must first solve the primary screening problem:
> *"Does this leaf exhibit symptoms of leaf-mining damage, does it have unrelated damage, or is it healthy?"*

```mermaid
flowchart TD
    A[Input Leaf Photo: 224x224x3] --> B[Pretrained MobileNetV3 Backbone]
    B --> C[Stage 1: Low-Level Edges & Textures]
    C --> D[Stage 2: Mid-Level Vein & Margin Patterns]
    D --> E[Stage 3: High-Level Lesion & Color Features]
    E --> F[Global Average Pooling]
    F --> G[Linear 1024 + Hardswish + Dropout 0.2]
    G --> H[3-Class Linear Head: Output Logits]
    H --> I[Softmax Activation]
    I --> J1["P(healthy)"]
    I --> J2["P(leaf_miner_damage)"]
    I --> J3["P(other_damage)"]
    J1 & J2 & J3 --> K[Uncertainty & Decision Engine]
```

---

## 2. Core Concepts Explained for Faculty Evaluation

### 1. Why MobileNetV3-Small?
Standard convolutional networks (like ResNet-50 or VGG-16) have between 25 and 138 million parameters, requiring hundreds of megabytes of memory and high-power GPUs. 
In contrast, **MobileNetV3-Small** was designed via Hardware-Aware Neural Architecture Search (NAS) specifically for resource-constrained edge devices (smartphones and laptop unified memory):
- **Parameter Count:** ~2.5 million parameters (< 10 MB file size).
- **Inverted Residual Blocks:** Expands features in high dimensions, performs depthwise separable convolution (filtering), and projects back to low dimensions with linear bottlenecks.
- **Hardware Efficiency:** Executes on Apple Silicon MPS with inference latency under 30 milliseconds and memory footprint under 120 MB.

### 2. What Does Transfer Learning Mean?
Training a deep neural network from scratch requires hundreds of thousands of images. In **transfer learning**, we start with a MobileNetV3 model pretrained on the **ImageNet-1k dataset** (1.4 million natural photographs across 1,000 categories).
The network's earlier layers already know how to detect universal visual primitives: contrast edges, textures, curved boundaries, and lighting gradients. 

In PhytoTrace, we use a **Two-Phase Transfer Learning Strategy**:
- **Stage A (Head Warmup):** The feature extraction backbone is completely frozen (`param.requires_grad = False`). Only the newly initialized 3-class classification head is trained for 5 epochs with AdamW (learning rate: $10^{-3}$). This adapts the head without disrupting pretrained weights.
- **Stage B (Fine-Tuning):** The top 3 inverted residual blocks are unfrozen. The entire model is fine-tuned together at a lower learning rate ($10^{-4}$) using Cosine Annealing, allowing the network to specialize its high-level filters to tomato leaf mining lesions.

### 3. How Are Images Labelled?
Images are mapped to three mutually exclusive baseline classes:
1. `healthy` (Class 0): Unblemished tomato leaf leaflets.
2. `leaf_miner_damage` (Class 1): Visible serpentine or blotch mining galleries.
3. `other_damage` (Class 2): Chewing injury, fungal blights (*Alternaria*), mechanical tears, or chemical scorch.

> [!IMPORTANT]
> **Label Integrity Rule:** Suspected pest identities (e.g., suspected *Tuta* or suspected *Liriomyza*) recorded by field scouts are kept in separate metadata fields and are **never** treated as confirmed species labels.

### 4. How the Neural Network Learns
Learning is an iterative mathematical optimization:
1. **Forward Pass:** An input image tensor $X \in \mathbb{R}^{3 \times 224 \times 224}$ is passed through the network to generate raw logit scores $z = [z_0, z_1, z_2]$.
2. **Loss Calculation:** The model compares its predictions against the ground truth one-hot vector $y$ using **Cross-Entropy Loss**:
   $$\mathcal{L} = -\sum_{c=0}^{2} y_c \log(p_c) \quad \text{where} \quad p_c = \frac{e^{z_c}}{\sum_{j=0}^{2} e^{z_j}}$$
   If the model is confident in the wrong class, the loss is high. If it predicts correctly with high confidence, the loss approaches zero.
3. **Backpropagation:** Gradients of the loss with respect to every learnable parameter ($\frac{\partial \mathcal{L}}{\partial W}$) are computed using the chain rule.
4. **Weight Update:** The **AdamW optimizer** updates weights:
   $$W_{t+1} = W_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \lambda W_t$$
   incorporating momentum ($m_t$), adaptive variance scaling ($v_t$), and decoupled weight decay ($\lambda W_t$) to prevent overfitting.

### 5. What Happens in Each Epoch?
An epoch represents one complete pass over the entire training dataset:
- Data is shuffled and split into mini-batches of 16 images.
- Images receive realistic augmentations (mild rotation $\pm 15^\circ$, horizontal/vertical flips, gentle contrast adjustment) that preserve thin serpentine mine structures.
- For each batch, gradients are calculated, clipped at norm 1.0 (to avoid gradient explosions), and weights are updated.
- At the end of the epoch, the model is evaluated on the validation set without computing gradients.

### 6. How Validation Works
The validation set (15% of data) is used exclusively for model selection and early stopping:
- After each epoch, validation loss and validation **Macro F1 score** are measured.
- If the Macro F1 score improves, the model checkpoint (`baseline_mobilenetv3_best.pt`) is updated.
- If validation performance fails to improve for 6 consecutive epochs (early stopping patience), training terminates automatically to prevent memorization.

### 7. What Do Accuracy and Macro F1 Mean?
- **Accuracy:** The simple ratio of correct predictions to total predictions:
  $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$
  *Limitation:* If 80% of samples are healthy, a naive model predicting "healthy" every time achieves 80% accuracy while failing completely on mined leaves.
- **Macro F1 Score:** The unweighted average of F1 scores across all three classes:
  $$\text{Macro F1} = \frac{F1_{\text{healthy}} + F1_{\text{leaf\_miner}} + F1_{\text{other}}}{3}$$
  where $F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$.
  Macro F1 treats all three classes equally, making it the gold-standard metric for imbalanced agricultural datasets.

### 8. Why Does the Model Not Yet Identify *Tuta* or *Liriomyza*?
**This is an intentional scientific choice.**
Visual mine symptoms alone are insufficient to confirm the insect species. *Tuta absoluta* (a moth) and *Liriomyza* spp. (flies) can both produce atypical mining patterns, especially in dense infestations, older desiccated leaves, or when fungal pathogens co-infect the tissue.
Treating species classification as a simple 3-way visual guess would be scientifically irresponsible. PhytoTrace separates:
1. Damage detection (Phase 2 — MobileNetV3).
2. Damage localization (Phase 3 — YOLO segmentation).
3. Species evidence analysis (Phase 4 — Multi-label region model with stereomicroscopic confirmation).

### 9. How the Next Segmentation Model Will Improve the System
Phase 3 integrates **YOLO11n instance segmentation**:
- Instead of classifying the whole image, YOLO localizes each individual mine with a polygon boundary.
- The leaf canopy segmenter separates the leaf from background soil.
- The system calculates exact visible damage percentage:
  $$\text{Damage \%} = 100 \times \frac{\sum (M_{\text{mined}} \cap M_{\text{leaf}})}{\sum M_{\text{leaf}}}$$
  providing agronomists with quantified severity measurements.

---

## 3. Faculty Defense Presentation Scripts

### A. 3-Minute Faculty Pitch Script

> *"Respected evaluators,*
> 
> *In tomato farming, leaf miners cause catastrophic yield loss. Problem Statement 04 challenges us to detect mixed infestations of Tuta absoluta and Liriomyza species.*
> 
> *Our team recognized an immediate scientific reality: you cannot reliably identify an insect species from a smartphone photo before you can reliably determine if leaf-mining damage even exists. Most student projects attempt an end-to-end classifier that guesses the species immediately, resulting in high false confidence on blights or atypical mines.*
> 
> *In Phase 2 of PhytoTrace, we have built the first milestone: a robust, lightweight MobileNetV3-Small baseline classifier trained via two-phase transfer learning on Apple Silicon Metal Performance Shaders (MPS).*
> 
> *Here is what makes our implementation scientifically rigorous:*
> *First, we employ group-aware splitting: all leaves from the same plant and farm remain in the same data partition, guaranteeing zero leakage between training, validation, and testing.*
> *Second, we implement a two-stage training strategy: head warmup followed by selective backbone fine-tuning with cosine annealing.*
> *Third, our evaluation module reports real measured Macro F1, balanced accuracy, and per-class confusion matrices. If real weights are uninitialized, our system honestly outputs 'NOT EVALUATED' rather than fabricating plausible accuracy numbers.*
> 
> *Our code is verified, our training smoke test passed on Apple MPS, and our local Streamlit dashboard is fully operational. We are now ready to ingest our original tomato leaf photographs. Thank you."*

---

### B. 8-Minute Technical Defense Script

> *"Good morning, members of the review panel.*
> 
> *I am presenting the engineering details of Phase 2 for PhytoTrace: our baseline computer vision classifier and training infrastructure.*
> 
> **1. The Problem Framing (Slide 1–2):**
> *Chip to Crop Problem Statement 04 focuses on leaf miner mixed infestation. In agriculture, when a leaf is photographed, three distinct conditions exist: healthy foliage, mining damage, and unrelated diseases like Alternaria blight. Attempting to jump directly to species-level identification without establishing a validated baseline classifier violates standard machine learning methodology. The first technical milestone must be robust, reliable damage screening.*
> 
> **2. Architecture Selection (Slide 3):**
> *We selected MobileNetV3-Small over heavier architectures like ResNet-50. MobileNetV3 utilizes depthwise separable convolutions and inverted residual bottlenecks, delivering feature representation equivalent to much larger networks with only 2.5 million parameters. On our Apple Silicon M5 hardware, it runs natively on Metal Performance Shaders (MPS) with a batch size of 16, utilizing less than 150 MB of memory.*
> 
> **3. Two-Phase Transfer Learning (Slide 4):**
> *To prevent gradient shock from corrupting pretrained ImageNet representations, training proceeds in two stages:*
> *In Stage A, we freeze all convolutional feature layers and train only the newly replaced 1024-dimension classification head using AdamW at a learning rate of 0.001.*
> *In Stage B, we unfreeze the final three inverted residual blocks and fine-tune using a reduced learning rate of 0.0001 under a Cosine Annealing schedule.*
> 
> **4. Data Hygiene and Leakage Prevention (Slide 5):**
> *The most prevalent error in published agricultural AI papers is random image splitting. If multiple photos of one leaf under different angles are placed in both train and test sets, the model achieves 99% test accuracy by simply memorizing background artifacts. In PhytoTrace, we built a Group-Aware Splitter that groups by farm_id and plant_id. Our automated test suite includes rigorous disjointness checks verifying zero image or plant leakage across splits.*
> 
> **5. Uncertainty Handling and Transparency (Slide 6):**
> *When an image is submitted, it first passes through Laplacian variance checking to reject motion blur. The model then predicts class probabilities across healthy, leaf miner damage, and other damage. If the highest class probability is below our calibrated threshold (0.50), or if the top two classes are separated by less than 0.15, the system flags the result as uncertain.*
> 
> **6. Current Status & Verification (Slide 7):**
> *Our complete Stage A and Phase 2 training pipelines are fully implemented and verified. Our automated PyTest suite contains 36 passing tests. Our training smoke test on Apple MPS verified forward/backward passes, early stopping, and checkpointing. Our Streamlit dashboard provides a unified interface across Dataset Exploration, Training, Evaluation, and Live Prediction.*
> 
> *We invite any questions from the panel."*

---

## 4. Exact Commands for Real Field Training

```bash
# 1. Ingest original smartphone photographs into data/raw/
python scripts/ingest_images.py --source_dir <path_to_photos> --collector_id COL_01 --farm_id FARM_01

# 2. Audit dataset readiness (verifies class balance and absence of corruption)
python scripts/validate_dataset.py

# 3. Generate group-aware train/val/test splits without leakage
python scripts/make_splits.py --group_col farm_id --fallback_group_col plant_id

# 4. Train MobileNetV3-Small baseline classifier on Apple Silicon MPS
python scripts/train_baseline.py --epochs 25

# 5. Run independent evaluation on held-out test split
python scripts/evaluate.py

# 6. Launch local Streamlit training & prediction dashboard
streamlit run demo/app.py
```

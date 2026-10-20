# Lab 8: Comparative Study of Attention Mechanisms and Insertion Positions in Pretrained ResNet50

This laboratory investigates the integration of channel attention, spatial attention, sequential channel-spatial attention, self-attention, and advanced attention variants (ECA, CBAM, Coordinate Attention)n depths, kernel sizes, and structural configurations.

---

## Dataset Overview & Preprocessing

* **Dataset**: TensorFlow Flowers (`tf_flowers`).
* **Class Volume**: 3,670 natural RGB images across 5 classes (`dandelion`: 898, `daisy`: 633, `tulips`: 799, `sunflowers`: 699, `roses`: 641).
* **Stratified Dataset Split**: Mutually disjoint 70% Training (2,569 images), 15% Validation (550 images), and 15% Testing (551 images) partitions generated with seed 42. Test distribution: 135 dandelion, 95 daisy, 120 tulips, 105 sunflowers, and 96 roses.
* **Preprocessing & Augmentation**: Inputs resized to $224 \times 224 \times 3$ and normalized using ImageNet ResNet50 pre-processing routines. Training augmentation includes random horizontal flipping, random rotation ($\pm 18^\circ$), random zoom ($10\%$), random translation ($10\%$), and random contrast adjustments ($10\%$).

---

## Backbone Architecture & Insertion Positions

The backbone architecture employs ResNet50 pre-trained on ImageNet. As spatial resolution decreases across stages, feature depth increases, transitioning representations from generic visual features to high-level semantic abstractions:

| ResNet50 Stage | Candidate Insertion Position | Spatial Resolution | Feature Channels ($C$) | Spatial Grid ($N = HW$) | Self-Attention Matrix Entries ($N \times N$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stem + MaxPool** | — | $56 \times 56$ | 64 | 3,136 | — |
| **`conv2_x`** | **$P_1$** | $56 \times 56$ | 256 | 3,136 | 9,834,496 (37.52 MB) |
| **`conv3_x`** | **$P_2$** | $28 \times 28$ | 512 | 784 | 614,656 (2.34 MB) |
| **`conv4_x`** | **$P_3$** | $14 \times 14$ | 1024 | 196 | 38,416 (0.15 MB) |
| **`conv5_x`** | **$P_4$** | $7 \times 7$ | 2048 | 49 | 2,401 (0.01 MB) |

The spatial grid size grows quadratically for self-attention memory allocations, expanding from 2,401 entries at $P_4$ to $9.8 \times 10^6$ entries at $P_1$.

---

## Attention Mechanisms & Formulations

### 1. Channel Attention (Squeeze-and-Excitation / SE)
Global average pooling collapses spatial dimensions to generate channel descriptor $z_c = \frac{1}{HW}\sum_{i=1}^H \sum_{j=1}^W F(i,j,c)$. Gating is performed via two dense layers with reduction ratio $r=16$:

$$M_c(F) = \sigma(W_2 \, \delta(W_1 z)), \quad F_c = F \odot M_c(F) \text{}$$

### 2. Spatial Attention
Calculates average and maximum pooling descriptors along feature channels $F_{\text{avg}} = \text{Mean}_c(F)$ and $F_{\text{max}} = \text{Max}_c(F)$. Concatenated tensors $F_s = [F_{\text{avg}} \parallel F_{\text{max}}]$ are passed through a $k \times k$ convolution:

$$M_s(F) = \sigma(\text{Conv}_{k \times k}(F_s)), \quad F'_s = F \odot M_s(F) \text{}$$

### 3. Sequential Channel-Spatial Attention
Applies channel gating followed by spatial gating:

$$F_c = M_c(F) \odot F, \quad F_{cs} = M_s(F_c) \odot F_c \text{}$$

### 4. Self-Attention
Flattens spatial dimensions to $X \in \mathbb{R}^{N \times C}$ ($N=HW$). Linear transformations produce queries $Q=XW_Q$, keys $K=XW_K$ ($d=C/8$), and values $V=XW_V$ ($C/2$):

$$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right), \quad Y = A V, \quad F_{\text{SA}} = F + \gamma \, \text{reshape}(Y) \text{}$$

Where $\gamma$ is a learnable scalar parameter initialized to zero.

### 5. Additional Mechanisms
* **Efficient Channel Attention (ECA)**: Implements unreduced 1D adaptive convolution over pooled channel vectors.
* **CBAM**: Dual-pooling shared MLP channel attention combined sequentially with $7 \times 7$ spatial attention.
* **Coordinate Attention (CoordAtt)**: Direction-aware $1D$ spatial pooling along height and width axes with h-swish gated $1 \times 1$ convolutions.

---

## Experimental Setup & Training Protocol

* **Transfer Learning Strategy**: Two-phase Adam optimization with batch size 32:
  1. **Head Phase** (5 epochs, $\eta = 10^{-3}$): ResNet50 backbone frozen; only inserted attention modules and classifier head trained.
  2. **Fine-Tuning Phase** (5 epochs, $\eta = 10^{-4}$): Unfreezes `conv5_x` block while keeping Batch Normalization layers frozen.
* **Loss Function**: Categorical Cross-Entropy Loss.
* **Model Selection**: The checkpoint achieving the highest validation accuracy across both phases is evaluated on the unobserved 551-image test set.

---

## Performance Results & Ablation Analysis

### 1. Comprehensive Performance Comparison

| Model Name | Insertion Position | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Total Parameters | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline ResNet50** | None | 0.9274 | 0.9263 | 0.9315 | 0.9268 | 23,597,957 | 4.03 ms/img |
| **Channel / SE** | $P_3$ | 0.9165 | 0.9133 | 0.9165 | 0.9141 | 23,730,117 | 4.06 ms/img |
| **Spatial ($7 \times 7$)** | $P_3$ | 0.9347 | 0.9349 | 0.9349 | 0.9347 | 23,598,056 | 4.07 ms/img |
| **Channel + Spatial ($k=7$)** | $P_1$ | 0.9165 | 0.9206 | 0.9133 | 0.9158 | 23,606,520 | 4.13 ms/img |
| **Channel + Spatial ($k=7$)** | $P_2$ | 0.9365 | 0.9353 | 0.9365 | 0.9357 | 23,631,368 | 4.07 ms/img |
| **Channel + Spatial ($k=7$)** | $P_3$ | 0.9328 | 0.9370 | 0.9296 | 0.9323 | 23,730,216 | 4.08 ms/img |
| **Channel + Spatial ($k=7$)** | $P_4$ | 0.9220 | 0.9249 | 0.9213 | 0.9203 | 24,124,520 | 4.07 ms/img |
| **Channel + Spatial ($k=3$)** | $P_3$ | 0.9238 | 0.9246 | 0.9252 | 0.9243 | 23,730,136 | 4.11 ms/img |
| **Channel + Spatial ($k=5$)** | $P_3$ | **0.9456** | **0.9458** | **0.9442** | **0.9447** | 23,730,168 | 4.07 ms/img |
| **Multi-Stage ($C \to S$)** | $P_3 + P_4$ | 0.9238 | 0.9246 | 0.9252 | 0.9226 | 24,256,779 | 4.11 ms/img |
| **Multi-Stage ($C \to S$)** | $P_2 + P_3 + P_4$ | 0.9292 | 0.9297 | 0.9283 | 0.9287 | 24,290,190 | 4.15 ms/img |
| **Self-Attention** | $P_3$ | 0.9256 | 0.9246 | 0.9229 | 0.9233 | 24,908,678 | 4.26 ms/img |
| **Combined ($C+S+\text{SA}$)** | $P_3$ | 0.9383 | 0.9381 | 0.9393 | 0.9382 | 25,040,937 | 4.30 ms/img |
| **ECA** | $P_3$ | 0.9165 | 0.9177 | 0.9173 | 0.9150 | 23,597,962 | 4.05 ms/img |
| **CBAM** | $P_3$ | 0.9365 | 0.9348 | 0.9401 | 0.9367 | 23,730,216 | 4.08 ms/img |
| **Coordinate Attention** | $P_3$ | 0.9238 | 0.9226 | 0.9234 | 0.9228 | 23,698,469 | 4.08 ms/img |

*All accuracy variations across models fall within the $95\%$ confidence interval ($\pm 0.022$) of the test set.*

---

## Analytical Findings & Qualitative Interpretability

### 1. Spatial Depth Impact
Intermediate feature insertion ($P_2$ at `conv3_x` with accuracy 0.9365, $P_3$ at `conv4_x` with accuracy 0.9328) outperforms early spatial placement ($P_1$ at `conv2_x`, accuracy 0.9165) and deep semantic placement ($P_4$ at `conv5_x`, accuracy 0.9220). Multi-stage attention insertion ($P_3+P_4$ and $P_2+P_3+P_4$) increases computational parameters without yielding performance improvements.

### 2. Parameter Complexity vs. Performance
No monotonic correlation exists between model size and classification accuracy (Spearman rank correlation $\rho = +0.26, p = 0.31$). Heavy modules such as Self-Attention (+1.31M parameters) and Combined Attention (+1.44M parameters) increase latency by up to $0.27\text{ ms/img}$ without surpassing lightweight alternatives like $5 \times 5$ spatial gating.

### 3. Internal Attention Maps vs. Grad-CAM
* **Internal Spatial Maps**: Generated spatial maps exhibit near-uniform activation ($\approx 1.0$) with slight attenuation along image borders, indicating that spatial attention acts as a soft boundary gate rather than an object segmenter.
* **Self-Attention Degeneracy**: Learned scalar weight $\gamma$ converges to $-0.0236$, causing the self-attention matrix to collapse to a single spatial coordinate per image with an average row entropy of 0.04 nats (compared to uniform 5.28 nats).
* **Grad-CAM Localization**: For correctly classified samples, Grad-CAM feature focus remains centered on flower heads across baseline and attention-enhanced architectures. Mass focus proxy statistics show comparable compactness between baseline ($0.560$) and spatial gating ($0.575 - 0.582$).

---

## Execution Guide

```bash
jupyter notebook Lab-8.ipynb
```

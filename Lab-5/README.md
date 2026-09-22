# Lab 5: Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

This laboratory systematically evaluates weight initialization methods, regularization techniques, first-order optimization algorithms, structural hyperparameters, transfer learning paradigms, and 5-fold cross-validation using MobileNet V2 on the Oxford-IIIT Pet dataset.

---

## Directory Structure

```text
Lab-5/
├── latex/
│   ├── report.tex
│   ├── sample_images.eps
│   ├── weight_init_loss.eps
│   ├── weight_init_acc.eps
│   ├── regularization_acc.eps
│   ├── regularization_loss.eps
│   ├── bn_comparison.eps
│   ├── optimizer_loss.eps
│   ├── optimizer_acc.eps
│   ├── hyperparam_lr.eps
│   ├── hyperparam_batch.eps
│   ├── hyperparam_dropout.eps
│   ├── transfer_vs_finetune_acc.eps
│   ├── transfer_vs_finetune_loss.eps
│   ├── cv_accuracy.eps
│   ├── confusion_matrix.eps
│   └── misclassified_samples.eps
├── Lab-5.ipynb
└── Readme.md
```

---

## Dataset Overview

* **Dataset**: Oxford-IIIT Pet Dataset.
* **Class Count**: 37 breeds of cats and dogs.
* **Input Resolution**: Preprocessed and resized to RGB tensors of shape $224 \times 224 \times 3$.
* **Data Protocol**: Normalized according to ImageNet pre-training specifications with independent training, validation, and untouched test partitions.

---

## Architecture: MobileNet V2

The backbone employs MobileNet V2, which leverages depthwise separable convolutions ($3 \times 3$ depthwise followed by $1 \times 1$ pointwise), inverted residual blocks, linear bottlenecks, Batch Normalization, and ReLU6 activations. The network classification head is structured as:

$$\text{Input } (224 \times 224 \times 3) \longrightarrow \text{MobileNet V2 Backbone} \longrightarrow \text{GlobalAveragePooling2D} \longrightarrow \text{Dense} \longrightarrow \text{Softmax } (37 \text{ classes}) \text{}$$

---

## Experimental Modules & Analysis

### 1. Weight Initialization
Evaluated over 7 epochs using the Adam optimizer with a frozen MobileNet V2 base:
* **Zero Initialization**: Fails completely ($1.36\%$ validation accuracy, $71.3\,\text{s}$) because identical neuron activations prevent symmetry breaking.
* **Random Normal**: Reaches $91.02\%$ validation accuracy ($60.7\,\text{s}$).
* **Xavier / Glorot**: Reaches $90.93\%$ validation accuracy ($60.7\,\text{s}$).
* **He Initialization**: Reaches $90.65\%$ validation accuracy ($60.7\,\text{s}$).

### 2. Regularization & Overfitting
Trained for 10 epochs using He initialization and Adam:
* **No Regularization**: Attains $91.20\%$ validation accuracy with near $100\%$ training accuracy, presenting the widest generalization gap.
* **L2 Regularization**: Attains $89.84\%$ validation accuracy.
* **Dropout**: Attains $90.93\%$ validation accuracy, maintaining the narrowest margin between training and validation trajectories.
* **Batch Normalization**: Attains $90.02\%$ validation accuracy, providing smoother validation loss tracking.

### 3. Batch Normalization Mechanics
Mini-batch normalization stabilizes internal covariate shift across mini-batch inputs $x_1, \dots, x_m$ with learnable parameters $\gamma$ and $\beta$:

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2, \quad \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta \text{}$$

* **Verification**: On sample vector $x = [2, 4, 6, 8]$, $\mu_B = 5$, $\sigma_B^2 = 5$, yielding normalized outputs $\hat{x} \approx [-1.342, -0.447, 0.447, 1.342]$.

### 4. Optimization Algorithms
Benchmarked across 10 epochs with a frozen base:

| Optimizer | Final Loss | Best Validation Accuracy | Epochs to Converge | Wall Time |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | 1.6521 | 77.13% | 10 | 78.7 s |
| **Momentum** | 0.3176 | 91.20% | 6 | 79.8 s |
| **RMSProp** | 0.0551 | 90.47% | 6 | 79.6 s |
| **Adam** | **0.0599** | **91.38%** | **4** | 80.5 s |

### 5. CNN Hyperparameter Sensitivity
Isolated one-factor-at-a-time hyperparameter studies:
* **Learning Rate**: $\eta = 10^{-3} \implies 91.11\%$ validation accuracy; $\eta = 10^{-4} \implies 89.11\%$.
* **Batch Size**: $16 \implies 91.65\%$; $32 \implies 91.20\%$; $64 \implies 91.92\%$.
* **Dropout Rate**: $0.0 \implies 92.01\%$; $0.25 \implies 91.74\%$; $0.5 \implies 91.20\%$.

### 6. Transfer Learning vs. Fine-Tuning
* **Feature Extraction** (Frozen base, train classification head): Validated at $91.02\%$ accuracy in $79.4\,\text{s}$.
* **Fine-Tuning** (Last 30 base layers unfrozen, $\eta = 10^{-5}$): Validated at $89.84\%$ accuracy in $101.1\,\text{s}$. Fine-tuning requires lower learning rates to prevent catastrophic forgetting of pre-trained generic visual representations.

---

## 5-Fold Cross-Validation & Model Selection

Cross-validation performed across candidate configurations on training data:

| Configuration | F1 | F2 | F3 | F4 | F5 | $\text{Mean} \pm \text{SD}$ (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1: Baseline (Glorot, No Reg, Adam)** | 92.13 | 90.57 | 89.31 | 89.60 | 90.86 | **$90.49 \pm 1.00$** |
| **C2: He + Dropout + Adam** | 91.35 | 91.06 | 89.80 | 89.41 | 89.49 | $90.22 \pm 0.82$ |
| **C3: He + BatchNorm + Adam** | 90.57 | 90.57 | 90.38 | 88.53 | 89.79 | $89.97 \pm 0.77$ |
| **C4: He + Dropout + RMSProp** | 92.52 | 90.48 | 88.05 | 88.63 | 89.59 | $89.85 \pm 1.57$ |

Selected configuration **C1** exhibits the highest cross-validation mean accuracy with controlled variance.

---

## Final Independent Test Evaluation

Retrained C1 on full training data (12 epochs) and tested on the untouched partition:

* **Test Accuracy**: 91.21%
* **Weighted Precision**: 0.9150
* **Weighted Recall**: 0.9121
* **Weighted F1-Score**: 0.9119
* **Total Parameters**: 2,426,725
* **Training Time**: 94.0 s
* **Confusion Matrix Insights**: Keeshond, Newfoundland, Havanese, Pug, and Scottish Terrier achieved 100% recall. Most frequent confusion occurred between Birman and Ragdoll (8 misclassifications) due to shared phenotypic coat patterns.

---

## Execution Guide

```bash
jupyter notebook Lab-5.ipynb
```

# Lab 4: Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

This laboratory implements a two-phase transfer learning and deep fine-tuning pipeline on the CIFAR-10 image classification benchmark using the VGG16 architecture pre-trained on ImageNet

---

## Dataset Overview

* **Dataset**: CIFAR-10
* **Number of Classes**: 10 (`Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, `Truck`)
* **Training Samples**: 50,000 images ($32 \times 32 \times 3$)
* **Testing Samples**: 10,000 images ($32 \times 32 \times 3$)
* **Preprocessing**: Pixel normalization to range $[0, 1]$ and one-hot target encoding.

---

## Workflow & Implementation Details

### Task 1: Dataset Preparation & Inspection
* Extracted and normalized training and testing tensors.
* Generated sample visualizations across all 10 target classes and exported to `cifar10_samples.eps`.

### Task 2: Architecture & Transfer Learning Setup
* **Base Model**: VGG16 (ImageNet weights, top classification layer excluded, input shape $32 \times 32 \times 3$).
* **Freezing Strategy**: Frozen all base convolutional layers (`trainable = False`).
* **Classification Head**:
  * `GlobalAveragePooling2D()`
  * `Dense(256, activation='relu')`
  * `Dropout(0.3)`
  * `Dense(10, activation='softmax')`
* **Parameters**: $14,714,688$ non-trainable base parameters and $133,898$ trainable classifier parameters.

### Task 3: Phase 1 Feature Extractor Training
* Trained the top classification layers for 10 epochs using Adam optimizer ($\text{learning rate} = 10^{-3}$) with categorical cross-entropy loss.
* Baseline accuracy reached $\sim 65.36\%$ on training and $61.36\%$ on validation data.

### Task 4: Phase 2 Fine-Tuning
* Unfroze the final convolutional block of VGG16 (`block5_conv1`, `block5_conv2`, `block5_conv3`, `block5_pool`) while retaining frozen weights for earlier blocks.
* Recompiled with a lower learning rate ($\text{Adam}, \text{learning rate} = 10^{-4}$).
* Trained for an additional 10 epochs (total 20 epochs), boosting validation accuracy to $\sim 72.44\%$.
* Exported metric progression curves to `accuracy_curves.eps` and `loss_curves.eps`.

### Task 5: Model Evaluation
* Evaluated predictions on the unobserved 10,000 test set images.
* Computed quantitative performance metrics: Test Accuracy, Weighted Precision, Weighted Recall, Weighted F1-score, and Confusion Matrix.

---

## Experimental Results

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **0.7119** |
| **Weighted Precision** | **0.7105** |
| **Weighted Recall** | **0.7119** |
| **Weighted F1-Score** | **0.7064** |
| **Test Loss** | **1.4025** |

---

## Execution Guide

```bash
# Run notebook in environment
jupyter notebook Lab-1.ipynb
```

# Lab 3: Implementation of Convolutional Neural Networks (CNNs) for Image Classification

This laboratory investigates the fundamental operations of Convolutional Neural Networks (CNNs), including kernel scaling, stride, padding, pooling strategies, convolutional feature map extraction, and deep classification on CIFAR-10 in PyTorch.

---

## Dataset Overview

* **Dataset**: CIFAR-10 (via `torchvision.datasets`)
* **Target Classes**: 10 (`plane`, `car`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`)
* **Volume**: 50,000 training images, 10,000 testing images ($3 \times 32 \times 32$ tensor format)
* **Distribution**: Perfectly balanced dataset with 5,000 samples per class in the training split.

---

## Workflow & Implementation Details

### Task 1: Dataset Exploration & Distribution Analysis
* Loaded tensors via PyTorch `DataLoader` with batch size 32.
* Rendered 10 class samples (`cifar_sample_images.eps`) and visualized class balance via bar chart (`cifar_class_dist.eps`).

### Task 2: Convolutional Mechanics & Feature Map Visualization
* **Kernel Dimension Comparison**: Evaluated spatial receptive fields across $3 \times 3$, $5 \times 5$, and $7 \times 7$ kernels.
* **Stride & Padding Analysis**: Evaluated spatial resolution preservation (`padding='same'`) versus boundary reduction (`padding=0`).
* **Feature Map Inspection**: Extracted and plotted all 8 activation feature maps from the initial convolutional layer using the `viridis` colormap (`feature_maps.eps`).
* **Downsampling Analysis**: Compared spatial dimensionality reductions between `MaxPool2d(2, 2)` and `AvgPool2d(2, 2)` ($30 \times 30 \to 15 \times 15$).

### Task 3: 2-Stage CNN Architecture
* **Feature Extractor**:
  * $\text{Conv2D}(3 \to 32, \text{kernel}=3, \text{padding}=1) \to \text{ReLU} \to \text{MaxPool2D}(2, 2)$
  * $\text{Conv2D}(32 \to 64, \text{kernel}=3, \text{padding}=1) \to \text{ReLU} \to \text{MaxPool2D}(2, 2)$
* **Classifier Head**:
  * $\text{Flatten} \to \text{Linear}(64 \times 8 \times 8 \to 10)$

### Task 4: Training & Validation
* Trained for 20 epochs using Adam optimizer ($\text{lr}=0.001$) and `CrossEntropyLoss`.
* Recorded training and validation loss and accuracy trajectories (`train_val_accuracy.eps`, `train_val_loss.eps`).

### Task 5: Evaluation & Metrics
* Evaluated test set performance, generating macro-averaged classification metrics and confusion matrix heatmap.

---

## Performance Summary

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **0.6984** |
| **Macro Precision** | **0.7006** |
| **Macro Recall** | **0.6984** |
| **Macro F1-Score** | **0.6984** |

---

## Execution Guide

```bash
jupyter notebook Lab-2.ipynb
```

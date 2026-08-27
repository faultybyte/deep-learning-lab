# Lab 2: Implementation of a Multi-Layer Perceptron (MLP) for Multi-Class Image Classification

This laboratory implements a Multi-Layer Perceptron (MLP) classification pipeline on the Fashion-MNIST dataset, featuring automated hyperparameter search using `RandomizedSearchCV` with 5-fold cross-validation.

---

## Dataset Overview

* **Dataset**: Fashion-MNIST
* **Target Classes**: 10 (`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`)
* **Samples**: 60,000 training images, 10,000 testing images ($28 \times 28$ grayscale)
* **Preprocessing**: 
  * Flattened input dimensions: $28 \times 28 \to 784$
  * Min-Max Normalization: scaled pixel values to $[0, 1]$
  * Categorical One-Hot Encoding: 10-dimensional binary vectors

---

## Workflow & Implementation Details

### Task 1 & 2: Dataset Exploration & Preprocessing
* Plotted sample images (`mnist_sample_images.eps`) and verified uniform class representation with 6,000 images per category (`mnist_class_distribution.eps`).
* Flattened arrays to shape `(N, 784)` and normalized training/test partitions.

### Task 3 & 4: Baseline MLP Construction & Training
* **Architecture**: Input (784) $\to$ Dense (128, ReLU) $\to$ Dense (64, ReLU) $\to$ Dense (10, Softmax).
* **Parameters**: 109,386 trainable parameters.
* **Training**: 20 epochs with Adam optimizer, batch size 32, and 20% validation split.
* **Convergence**: Plotted training vs. validation accuracy and loss (`accuracy_curve.eps`, `loss_curve.eps`).

### Task 5: Baseline Evaluation
* Baseline Model Test Performance: Accuracy = 84.97%, Precision = 85.78%, Recall = 84.97%, F1-score = 84.79%.

### Task 6: Hyperparameter Search via RandomizedSearchCV
Integrated `KerasClassifier` from SciKeras with scikit-learn's `RandomizedSearchCV` across a 5-fold cross-validation scheme:
* **Search Space**:
  * Hidden Layers: `[1, 2, 3]`
  * Hidden Neurons: `[32, 64, 128, 256]`
  * Learning Rates: `[0.1, 0.01, 0.001]`
  * Optimizers: `['sgd', 'adam', 'rmsprop']`
  * Activations: `['relu', 'tanh', 'sigmoid']`
  * Dropout Rates: `[0.0, 0.2, 0.5]`
  * Batch Sizes: `[16, 32, 64, 128]`
  * Epochs: `[10, 20, 30]`

---

## Key Optimization Results

| Model Configuration | Best Parameters | CV Score / Test Accuracy |
| :--- | :--- | :--- |
| **Best Hyperparameters** | Optimizer: `adam`, LR: `0.001`, Neurons: `256`, Layers: `1`, Activation: `tanh`, Dropout: `0.2`, Batch Size: `64`, Epochs: `20` | **CV Accuracy: 0.8896** |
| **Optimized Test Model** | Evaluated on 10,000 unobserved test samples | **Test Accuracy: 0.8726** |

---

## Execution Guide

```bash
pip install scikeras scikit-learn==1.5.2
jupyter notebook Lab-3.ipynb
```

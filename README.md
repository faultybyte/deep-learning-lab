# Deep Learning Lab

A modular repository containing end-to-end implementations of deep learning workloads, ranging from exploratory data analysis and tabular classification to custom convolutional architectures, hyperparameter optimization, and deep transfer learning.

---

## Repository Structure

```text
.
├── Readme.md
├── Lab-1/
│   ├── latex/
│   │   ├── report.tex
│   │   ├── cifar10_samples.eps
│   │   ├── accuracy_curves.eps
│   │   ├── loss_curves.eps
│   │   └── confusion_matrix.eps
│   ├── Lab-1.ipynb
│   └── Readme.md
├── Lab-2/
│   ├── latex/
│   │   ├── report.tex
│   │   ├── cifar_sample_images.eps
│   │   ├── cifar_class_dist.eps
│   │   ├── feature_maps.eps
│   │   ├── train_val_accuracy.eps
│   │   ├── train_val_loss.eps
│   │   └── confusion_matrix.eps
│   ├── Lab-2.ipynb
│   └── Readme.md
├── Lab-3/
│   ├── latex/
│   │   ├── report.tex
│   │   ├── mnist_sample_images.eps
│   │   ├── mnist_class_distribution.eps
│   │   ├── accuracy_curve.eps
│   │   ├── loss_curve.eps
│   │   └── confusion_matrix.eps
│   ├── Lab-3.ipynb
│   └── Readme.md
└── Lab-4/
    ├── latex/
    │   ├── report.tex
    │   ├── banknote_histograms.eps
    │   ├── banknote_heatmap.eps
    │   └── banknote_pairplots.eps
    ├── Lab-4.ipynb
    └── Readme.md
```

---

## Global Environment & Dependencies

Install the core dependencies across all laboratory modules:

```bash
# Core machine learning, computer vision, and visualization libraries
pip install numpy pandas matplotlib seaborn scikit-learn

# Deep learning backends
pip install tensorflow keras torch torchvision

# Hyperparameter optimization wrappers
pip install scikeras
```

---

## Global Execution Guidelines

1. **Jupyter Notebooks**: Execute `.ipynb` files sequentially within a GPU-enabled environment (e.g., CUDA-compatible local workstation or Google Colab).
2. **Publication-Quality Figures**: All plots are configured with serif typography (`Times New Roman`), bold axis labels ($14\,\text{pt}$), and exported to high-resolution vector format (`.eps`, $600\,\text{dpi}$) for seamless inclusion in LaTeX documentation.
3. **Reproducibility**: Global random seeds (`42`) are enforced across Python `random`, NumPy, PyTorch, and TensorFlow backends.

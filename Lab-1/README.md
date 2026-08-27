# Lab 1: Implementation of a Single Layer Perceptron for Binary Classification

This laboratory focuses on exploratory data analysis (EDA), statistical profiling, distribution evaluation, and feature correlation analysis on the Banknote Authentication Dataset.

---

## Dataset Overview

* **Dataset Source**: Banknote Authentication Dataset (`data_banknote_authentication.txt`)
* **Total Instances**: 1,372 records
* **Missing Values**: 0 null/missing entries across all attributes
* **Target Variable**: `Class` (Binary: 0 for Genuine Banknotes, 1 for Forged Banknotes)
* **Continuous Features (Wavelet Transformed Data)**:
  1. `Variance` (Variance of Wavelet Transformed image)
  2. `Skewness` (Skewness of Wavelet Transformed image)
  3. `Curtosis` (Curtosis of Wavelet Transformed image)
  4. `Entropy` (Entropy of image)

---

## Summary Statistics

| Feature | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Variance** | 0.4337 | 2.8428 | -7.0421 | -1.7730 | 0.4962 | 2.8215 | 6.8248 |
| **Skewness** | 1.9224 | 5.8690 | -13.7731 | -1.7082 | 2.3197 | 6.8146 | 12.9516 |
| **Curtosis** | 1.3976 | 4.3100 | -5.2861 | -1.5750 | 0.6166 | 3.1793 | 17.9274 |
| **Entropy** | -1.1917 | 2.1010 | -8.5482 | -2.4135 | -0.5867 | 0.3948 | 2.4495 |
| **Class** | 0.4446 | 0.4971 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

---

## Workflow & Exploratory Findings

### Task 1: Data Audit & Statistical Summary
* Verified data integrity with 0 missing values and verified data distributions across classes ($55.54\%$ Class 0, $44.46\%$ Class 1).

### Task 2: Univariate Distribution Analysis
* Plotted feature histograms with Kernel Density Estimation (KDE) overlays in `banknote_histograms.eps`.
* `Variance` exhibits bimodal characteristics with high separability between genuine and forged notes.

### Task 3: Correlation & Covariance Heatmap
* Generated full Pearson correlation coefficient heatmap exported to `banknote_heatmap.eps`.
* Identified strong negative linear correlation between `Curtosis` and `Skewness` ($r = -0.53$).

### Task 4: Pairwise Feature Interaction Analysis
* Rendered complete $4 \times 4$ pairplot grid with class coloring (`Set1` palette) and exported to `banknote_pairplots.eps`.
* `Variance` vs. `Skewness` pairwise combination provides the highest linear decision boundary separation between banknote authenticity classes.

---

## Execution Guide

```bash
jupyter notebook Lab-4.ipynb
```

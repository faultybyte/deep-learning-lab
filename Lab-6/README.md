# Lab 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

This laboratory explores recurrent neural network architectures—Vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)—for multi-channel sensory sequence classification, temporal horizon variations, CNN-recurrent video understanding, and sequence-to-sequence autoencoding.

## Dataset Overview

### 1. UCI Human Activity Recognition (HAR)
* **Input Representation**: Three-dimensional tensor $X \in \mathbb{R}^{N \times T \times F}$ with $T = 128$ time steps and $F = 9$ inertial signal channels (3-axis body acceleration, 3-axis gyroscope, 3-axis total acceleration).
* **Classes**: 6 categories (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`).
* **Partitioning**: $N = 3,000$ windows across 21 subjects partitioned subject-wise into 70% train (2,147 windows), 15% validation (386 windows), and 15% test (467 windows).

### 2. UCF101 Video Classification Subset
* **Classes**: 5 action classes (`Basketball`, `BaseballPitch`, `BabyCrawling`, `BenchPress`, `BandMarching`) with 40 videos per category (200 videos total).
* **Sequence Format**: 10 uniformly sampled frames resized to $224 \times 224 \times 3$.

---

## Model Architectures & Formulations

* **Vanilla RNN**:
  $$h_t = \tanh(W_x x_t + W_h h_{t-1} + b_h) \text{}$$
  Suffers from vanishing gradients during Backpropagation Through Time (BPTT) as repeated Jacobian multiplication drives $\left\vert{}\frac{\partial h_t}{\partial h_{t-k}}\right\vert{} \to 0$.

* **LSTM (3 Gates, Dual State)**:
  $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \text{}$$
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c), \quad h_t = o_t \odot \tanh(C_t) \text{}$$

* **GRU (2 Gates, Single State)**:
  $$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z), \quad r_t = \sigma(W_r [h_{t-1}, x_t] + b_r) \text{}$$
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h) \text{}$$

---

## Experimental Results

### 1. Recurrent Architecture Comparison (UCI HAR)
Trained for 30 epochs with 32 recurrent units, Adam optimizer ($\eta = 10^{-3}$), and batch size 32:

| Model | Parameters | Training Time | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SimpleRNN** | 1,974 | 30.6 s | 72.16% | 72.80% | 74.42% | 73.03% |
| **LSTM** | 6,006 | 25.5 s | 94.86% | 95.45% | 94.93% | 94.84% |
| **GRU** | 4,758 | 26.3 s | **98.29%** | **98.42%** | **98.23%** | **98.29%** |

*Inference*: GRU yields the highest classification performance while requiring ~21% fewer parameters than LSTM. Vanilla RNN performance collapses on length-128 sequences due to vanishing gradients.

### 2. Impact of Sequence Length ($T$)
Macro F1 scores over sequence truncations $T \in \{32, 64, 128\}$:

| Sequence Length ($T$) | SimpleRNN F1 (%) | LSTM F1 (%) | GRU F1 (%) |
| :--- | :--- | :--- | :--- |
| **32 Steps** | 81.97 | 87.91 | 94.08 |
| **64 Steps** | 82.01 | 93.22 | 97.55 |
| **128 Steps** | 73.03 | **94.84** | **98.29** |

*Inference*: Gated models capitalize on longer temporal context, whereas plain RNN degradation is pronounced at $T = 128$ (-8.9 points from $T=32$).

### 3. Video Understanding via CNN-Recurrent Pipeline
Spatial features ($D = 1,280$) extracted from frozen ImageNet MobileNet V2 and fed sequentially to recurrent classifiers (40 epochs):

| Pipeline Model | Trainable Parameters | Training Time | Test Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **CNN-LSTM** | 169,285 | 19.87 s | 93.94% | 93.81% |
| **CNN-GRU** | 127,365 | 9.69 s | 93.94% | 93.81% |

### 4. Sequence-to-Sequence Inversion
Encoder-decoder network trained with teacher forcing to reverse 6-digit sequences over alphabet $\{1, \dots, 9\}$ (LSTM, 128 units, embedding dimension 32, 166,794 parameters):

* **Token Accuracy**: 99.99%
* **Sequence Accuracy**: 99.94%
* **Train / Val Loss**: 0.0003 / 0.0004
* **Length Generalization (6 In $\to$ 3 Out)**: 100.0% Token and Sequence accuracy.

---

## Execution Guide

```bash
jupyter notebook Lab-6.ipynb
```

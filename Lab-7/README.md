# Lab 7: End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders

This laboratory investigates representation learning, image reconstruction, noise reduction, and generative modeling across four distinct autoencoder paradigms: Fully Connected Autoencoders (FC-AE), Convolutional Autoencoders (CAE), Convolutional Denoising Autoencoders (CDAE), and Variational Autoencoders (VAE) on the MNIST dataset.

---

## Directory Structure

```text
Lab-7/
├── latex/
│   ├── report.tex
│   ├── fcae_reconstruction.eps
│   ├── fcae_loss.eps
│   ├── cae_comparison.eps
│   ├── cdae_denoising.eps
│   ├── cdae_noise_metrics.eps
│   ├── vae_latent_space.eps
│   ├── vae_generated.eps
│   ├── vae_interpolation.eps
│   ├── vae_loss.eps
│   ├── error_distribution.eps
│   ├── high_error_samples.eps
│   └── latent_dim_study.eps
├── Lab-7.ipynb
└── Readme.md
```

---

## Dataset Overview & Preprocessing

* **Dataset**: MNIST Handwritten Digit Dataset.
* **Input Resolution**: Grayscale images of spatial dimensions $28 \times 28 \times 1$.
* **Data Subset**: 10,000 training images and 2,000 test images.
* **Target Mapping**: Self-supervised reconstruction where the input image serves as its own target ($x \to \hat{x}$).
* **Preprocessing**: Pixel intensity values normalized from $[0, 255]$ to $[0, 1]$.
* **Tensors**:
  * FC-AE: Flattened vector format $x \in \mathbb{R}^{784}$.
  * Spatial Models (CAE, CDAE, VAE): 4D tensor format $X \in \mathbb{R}^{B \times 28 \times 28 \times 1}$.

---

## Model Architectures & Formulations

### 1. Fully Connected Autoencoder (FC-AE)
* **Encoder**: $\text{Input } (784) \longrightarrow \text{Dense}(128, \text{ReLU}) \longrightarrow \text{Dense}(32, \text{ReLU}) \longrightarrow \text{Dense}(16, \text{ReLU})$.
* **Bottleneck**: Latent vector dimension $d_z = 16$.
* **Decoder**: $\text{Dense}(32, \text{ReLU}) \longrightarrow \text{Dense}(128, \text{ReLU}) \longrightarrow \text{Dense}(784, \text{Sigmoid})$.
* **Loss**: Binary Cross-Entropy (BCE).

### 2. Convolutional Autoencoder (CAE)
Preserves 2D spatial feature hierarchies using convolutional and upsampling layers:
* **Encoder**: $\text{Conv2D}(32, 3 \times 3, \text{ReLU}) \longrightarrow \text{MaxPool2D}(2 \times 2) \longrightarrow \text{Conv2D}(64, 3 \times 3, \text{ReLU}) \longrightarrow \text{MaxPool2D}(2 \times 2) \longrightarrow \text{Conv2D}(64, 3 \times 3, \text{ReLU})$.
* **Latent Feature Map**: $7 \times 7 \times 64$.
* **Decoder**: $\text{UpSampling2D}(2 \times 2) \longrightarrow \text{Conv2D}(32, 3 \times 3, \text{ReLU}) \longrightarrow \text{UpSampling2D}(2 \times 2) \longrightarrow \text{Conv2D}(1, 3 \times 3, \text{Sigmoid})$.

### 3. Denoising Convolutional Autoencoder (CDAE)
Trained to map corrupted inputs $\tilde{x}$ to clean original targets $x$ ($\tilde{x} \to z \to x$):
* **Gaussian Corruption**: Additive Gaussian noise $\tilde{x} = \text{clip}(x + n, 0, 1)$ where $n \sim \mathcal{N}(0, \sigma^2)$ with $\sigma \in \{0.1, 0.2, 0.3\}$.
* **Salt-and-Pepper Corruption**: Random pixel replacements with probabilities $p \in \{0.05, 0.10, 0.20\}$.

### 4. Variational Autoencoder (VAE)
Maps inputs to a stochastic latent distribution $q_\phi(z\vert{}x) = \mathcal{N}(\mu(x), \text{diag}(\sigma^2(x)))$. Uses a 2D latent space ($z \in \mathbb{R}^2$).
* **Reparameterization Trick**: Enables backpropagation through stochastic nodes by decoupling randomness:

  $$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \text{}$$

* **Loss Function**: Combined Reconstruction Loss and Kullback-Leibler (KL) Divergence penalty:

  $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{KL}} \text{}$$

  $$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q_\phi(z\vert{}x) \parallel \mathcal{N}(0, I)) = -\frac{1}{2} \sum_{j=1}^{d_z} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right) \text{}$$

---

## Performance Evaluation & Comparative Results

### 1. Consolidated Quantitative Comparison

| Model | Test MSE | Test MAE | Mean SSIM | Trainable Parameters | Inference Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fully Connected AE** | 0.020998 | 0.055382 | 0.751880 | 211,040 | 0.2192 s |
| **Convolutional AE** | **0.002506** | **0.014868** | **0.974898** | **74,497** | **0.2186 s** |
| **Denoising CAE ($\sigma=0.2$)** | 0.004799 | 0.021508 | 0.943831 | 74,497 | 0.2205 s |
| **Variational AE ($z \in \mathbb{R}^2$)** | 0.043855 | 0.102490 | 0.485874 | 284,933 | 0.4418 s |

* **VAE Loss Breakdown**: Reconstruction Loss = $150.0196$, KL Loss = $5.5630$, Total VAE Loss = $155.5825$.

### 2. CDAE Noise Sensitivity Analysis
Evaluated across varying Gaussian noise standard deviations $\sigma$ on the test partition:

| Noise Level ($\sigma$) | Test MSE | Test MAE | Mean SSIM |
| :--- | :--- | :--- | :--- |
| **$\sigma = 0.1$** | 0.003813 | 0.018184 | 0.958773 |
| **$\sigma = 0.2$** | 0.004775 | 0.021443 | 0.944068 |
| **$\sigma = 0.3$** | 0.007161 | 0.029384 | 0.868043 |

---

## Experimental Insights & Visual Analyses

* **Spatial Conservation**: The Convolutional AE achieves an $\sim 8\times$ reduction in MSE ($0.0210 \to 0.0025$) and increases SSIM from $0.752$ to $0.975$ over the FC-AE while using $65\%$ fewer parameters.
* **Latent Space Structure ($z \in \mathbb{R}^2$)**: Visualizing encoded 2D VAE test vectors reveals well-separated clusters for digits 0, 1, 6, and 7. Central overlap occurs among digits 2, 3, 5, and 8 due to spatial bottleneck constraints.
* **Latent Interpolation**: Linear traversal $z(\alpha) = (1-\alpha)z_A + \alpha z_B$ for $\alpha \in [0, 1]$ yields smooth morphing between digits without structural discontinuities, confirming space smoothness.
* **High Reconstruction Error Samples**: Largest reconstruction errors occur on atypical stroke styles (e.g., looped 2s, filled 8s, crossed 7s) where the 2D bottleneck forces convergence toward population averages.

---

## Additional Studies

1. **FC-AE Latent Dimension Scaling**:

   | Latent Dimension ($d_z$) | Test MSE | Test SSIM |
   | :--- | :--- | :--- |
   | **$d_z = 2$** | 0.047703 | 0.424548 |
   | **$d_z = 8$** | 0.029990 | 0.652452 |
   | **$d_z = 16$** | **0.020998** | **0.751880** |
   | **$d_z = 32$** | 0.021501 | 0.742331 |

2. **Transposed Convolution vs. Nearest-Neighbor UpSampling**:
   * `UpSampling2D` + `Conv2D`: MSE = 0.002506, SSIM = 0.974898.
   * `Conv2DTranspose` (Learnable Upsampling): MSE = **0.000593**, SSIM = **0.993310**.

3. **$\beta$-VAE Loss Weighting**:
   * Standard VAE ($\beta = 1.0$): Reconstruction Loss = 150.13, KL Loss = 5.56.
   * Constrained $\beta$-VAE ($\beta = 5.0$): Reconstruction Loss = 155.58, KL Loss = 3.46. Higher $\beta$ enforces stronger prior alignment at the expense of pixel reconstruction fidelity.

---

## Execution Guide

```bash
jupyter notebook Lab-7.ipynb
```

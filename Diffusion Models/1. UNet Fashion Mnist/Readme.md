# DDPM on Fashion-MNIST using a U-Net

A PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** trained on the **Fashion-MNIST** dataset. The model uses a compact **U-Net** with residual blocks, sinusoidal timestep embeddings, and a self-attention bottleneck to learn the reverse diffusion process.

---

## Overview

Diffusion models generate images by learning how to gradually remove Gaussian noise from an image.

During training:

1. A clean image is sampled from the dataset.
2. A random diffusion timestep is selected.
3. Noise corresponding to that timestep is added to the image.
4. The U-Net predicts the noise that was added.
5. The prediction is compared with the true noise using Mean Squared Error (MSE).

During inference:

1. Generation starts from pure Gaussian noise.
2. The model repeatedly predicts and removes noise.
3. After all reverse diffusion steps, a new image is generated.

---

# Architecture

The project follows the original DDPM architecture proposed by Ho et al. (2020).

```
Input Image (32×32)
        │
        ▼
Input Convolution
        │
        ▼
──────────── Encoder ────────────

Residual Block
        │
        ▼
Downsample (32 → 16)

Residual Block
        │
        ▼
Downsample (16 → 8)

────────── Bottleneck ───────────

Residual Block
        │
        ▼
Self Attention
        │
        ▼
Residual Block

──────────── Decoder ────────────

Upsample (8 → 16)
        │
Skip Connection
        ▼
Residual Block

Upsample (16 → 32)
        │
Skip Connection
        ▼
Residual Block

GroupNorm
        │
SiLU
        │
Output Convolution
        ▼
Predicted Noise
```

---

# Model Components

### Sinusoidal Time Embeddings

The diffusion timestep is converted into a high-dimensional sinusoidal embedding. This embedding allows the network to understand the current noise level.

---

### Time MLP

The sinusoidal embedding is projected through a small MLP before being injected into every residual block.

---

### Residual Blocks

Each residual block consists of:

* Group Normalization
* SiLU activation
* Convolution
* Time embedding injection
* Dropout
* Second convolution
* Residual (skip) connection

These blocks allow the network to learn deep representations while maintaining stable gradient flow.

---

### Downsampling

Spatial resolution is reduced using stride-2 convolutions.

```
32×32
   ↓
16×16
   ↓
8×8
```

---

### Self-Attention Bottleneck

A single-head self-attention layer is applied at the bottleneck (8×8 resolution).

Using attention only at low resolution significantly reduces computation while allowing every spatial location to interact with every other location.

---

### Upsampling

Nearest-neighbor interpolation followed by convolution restores the original image resolution.

```
8×8
   ↑
16×16
   ↑
32×32
```

---

### Skip Connections

Feature maps from the encoder are concatenated with decoder features at matching resolutions.

These skip connections preserve fine spatial details that would otherwise be lost during downsampling.

---

# Diffusion Process

The project implements the original DDPM formulation.

### Forward Process

Noise is added according to

```
x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε
```

where

* x₀ is the clean image
* ε is Gaussian noise
* ᾱₜ is the cumulative product of diffusion coefficients

---

### Training Objective

The network predicts the added noise.

Loss:

```
L = MSE( predicted_noise , true_noise )
```

The model never predicts the clean image directly.

---

### Reverse Process

Starting from Gaussian noise, the model iteratively predicts the noise present at each timestep and removes it until a clean image is produced.

---

# Dataset

* Fashion-MNIST
* Original size: 28×28
* Images padded to 32×32
* Pixel values normalized from [0,1] to [-1,1]

---

# Hyperparameters

| Parameter       |              Value |
| --------------- | -----------------: |
| Epochs          |                 30 |
| Batch Size      |                128 |
| Learning Rate   |               2e-4 |
| Diffusion Steps |               1000 |
| Base Channels   |                 64 |
| Optimizer       |              AdamW |
| Loss            | Mean Squared Error |

---

# Project Structure

```
.
├── model.py          # U-Net architecture
├── diffusion.py      # DDPM forward and reverse processes
├── train.py          # Training script
├── checkpoints/      # Saved model weights
├── samples/          # Generated sample images
└── data/             # Fashion-MNIST dataset
```

---

# Training

Run the training script:

```bash
python train.py
```

The script will:

* Download Fashion-MNIST automatically.
* Train the DDPM.
* Save generated sample grids every few epochs.
* Save model checkpoints periodically.

---

# Output

During training:

```
samples/
    epoch_2.png
    epoch_4.png
    ...
```

Model checkpoints:

```
checkpoints/
    ddpm_epoch_5.pt
    ddpm_epoch_10.pt
    ...
```

---

# Future Improvements

Possible extensions include:

* Class-conditional generation
* Classifier-Free Guidance (CFG)
* Multi-head self-attention
* Deeper U-Net architecture
* Cosine beta schedule
* Exponential Moving Average (EMA)
* Mixed precision (AMP)
* Latent diffusion
* Diffusion Transformers (DiT)

---

# References

* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
* Nichol, Alexander Quinn, and Prafulla Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.
* Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.

---

# License

This project is intended for educational and research purposes.

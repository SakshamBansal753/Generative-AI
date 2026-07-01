"""
Generate new Fashion-MNIST-style images from a trained DDPM checkpoint.

Usage:
    python sample.py
    (just edit the CONFIG values below directly, no command-line flags)
"""

import torch
from torchvision.utils import save_image

from model import UNet
from Diffusion import Diffusion


# ============== EDIT THESE VALUES DIRECTLY ==============
CHECKPOINT_PATH = "checkpoints/ddpm_epoch_30.pt"
NUM_SAMPLES = 16
OUTPUT_FILE = "generated.png"
# ==========================================================


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = UNet(in_ch=1, base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = Diffusion(timesteps=ckpt["timesteps"])

    print(f"Sampling {NUM_SAMPLES} images ({ckpt['timesteps']} denoising steps, this takes a bit)...")
    samples = diffusion.sample(model, shape=(NUM_SAMPLES, 1, 32, 32), device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1] for saving

    nrow = int(NUM_SAMPLES ** 0.5) or 1
    save_image(samples, OUTPUT_FILE, nrow=nrow)
    print(f"Saved generated image grid to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
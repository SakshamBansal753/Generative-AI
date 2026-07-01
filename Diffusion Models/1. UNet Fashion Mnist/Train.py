import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import UNet
from Diffusion import Diffusion
from tqdm import tqdm

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
TIMESTEPS = 1000          # number of diffusion steps (more = better quality, slower)
BASE_CHANNELS = 64        # model size: lower (e.g. 32) if you run out of GPU memory
SAMPLE_EVERY = 2          # save a sample image grid every N epochs
SAVE_EVERY = 5            # save a model checkpoint every N epochs
DATA_DIR = "./data"
OUT_DIR = "."

def get_data_loader(data_dir="./data", batch_size=128):
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

     
    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)
    dataloader = get_data_loader( DATA_DIR,BATCH_SIZE)
    model = UNet(in_ch=1, base_ch=BASE_CHANNELS).to(device)
    diffusion = Diffusion(timesteps=TIMESTEPS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx,(x,y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=device).long()
            loss = diffusion.training_loss(model, x, t, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # optional gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")   
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch}/{EPOCHS}] completed. Average Loss: {avg_epoch_loss:.4f}")

        if epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
           samples=diffusion.sample(model, shape=(64, 1, 28, 28), device=device)
           samples=(samples.clamp_(-1, 1) + 1) / 2  # denormalize to [0, 1]
           save_image(samples, os.path.join(OUT_DIR, "samples", f"sample_epoch_{epoch}.png"), nrow=8)   
           print(f"Saved sample images for epoch {epoch}.")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(OUT_DIR, "checkpoints", f"ddpm_epoch_{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "base_ch": BASE_CHANNELS,
                "timesteps": TIMESTEPS,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        print("Training complete.")
 
 
if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage

# ================================================================
# CONFIG
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "."
UT_HAR_DIR = os.path.join(ROOT_DIR, "UT_HAR")
DATA_DIR = os.path.join(UT_HAR_DIR, "data")
LABEL_DIR = os.path.join(UT_HAR_DIR, "label")

TARGET_CLASS = 0

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
LATENT_DIM = 512

os.makedirs("autoencoders", exist_ok=True)
os.makedirs("latents", exist_ok=True)
os.makedirs("reconstructions", exist_ok=True)

# ================================================================
# ARGPARSE
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-class", "--class_id", type=int, default=0,
        help="UT-HAR class to train autoencoder on (0–6)"
    )
    return parser.parse_args()

# ================================================================
# DATA LOADING
# ================================================================
def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f"X_{name}.csv"), allow_pickle=True)
    y = np.load(os.path.join(LABEL_DIR, f"y_{name}.csv"), allow_pickle=True)
    y = np.asarray(y).reshape(-1)
    return X, y

def load_ut_har_class(cls):
    X_train, y_train = load_split("train")
    mask = (y_train == cls)
    Xc = X_train[mask]
    print(f"Loaded class {cls}: {Xc.shape[0]} samples")

    mn, mx = Xc.min(), Xc.max()
    Xc = 2.0 * (Xc - mn) / max(mx - mn, 1e-8) - 1.0
    return Xc.astype(np.float32), (float(mn), float(mx))

class CSIDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0)  # (1,250,90)

# ================================================================
# AUTOENCODER
# ================================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.SiLU(),  # 125×45
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(), # 63×23
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(), # 32×12
        )
        self.flat_dim = 64 * 32 * 12
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.flat_dim = 64 * 32 * 12
        self.fc = nn.Linear(latent_dim, self.flat_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(), # 64x24
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.SiLU(), # 128x48
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),            # 256x96
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 32, 12)
        x = self.deconv(h)
        return F.interpolate(x, size=(250, 90), mode="bilinear", align_corners=False)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# ================================================================
# TRAINING
# ================================================================
def train_autoencoder():
    Xc, _ = load_ut_har_class(TARGET_CLASS)
    dataset = CSIDataset(Xc)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Autoencoder(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training autoencoder...")

    for epoch in range(1, EPOCHS + 1):
        running = 0
        for x in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = F.mse_loss(recon, x)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        print(f"[Epoch {epoch}/{EPOCHS}] Loss = {running / len(loader):.6f}")

    save_path = f"autoencoders/autoencoder_class_{TARGET_CLASS}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")

    return model, dataset

# ================================================================
# SAVE LATENTS
# ================================================================
def save_latents(encoder, dataset):
    print("Saving latents...")
    encoder.eval()

    latents = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            z = encoder(x)
            latents.append(z.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    npy_path = f"latents/latents_class_{TARGET_CLASS}.npy"
    np.save(npy_path, latents)

    print(f"Saved latents: {npy_path} (shape={latents.shape})")
    return latents

# ================================================================
# VISUALIZE RECONSTRUCTION
# ================================================================
def visualize_reconstruction(model, dataset):
    print("Rendering reconstruction example...")

    model.eval()
    x = dataset[0].unsqueeze(0).to(DEVICE)
    recon, _ = model(x)

    real = x[0, 0].cpu().numpy()
    fake = recon[0, 0].detach().cpu().numpy()

    # normalize
    real = (real - real.min()) / (real.max() - real.min() + 1e-8)
    fake = (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)

    real = ndimage.rotate(real, 90, reshape=True)
    fake = ndimage.rotate(fake, 90, reshape=True)

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Class {TARGET_CLASS} Reconstruction")

    plt.subplot(1,2,1)
    plt.imshow(real, cmap="viridis")
    plt.title("Real")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(fake, cmap="viridis")
    plt.title("Reconstructed")
    plt.axis("off")

    out_path = f"reconstructions/recon_class_{TARGET_CLASS}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()      # <-- Requested
    plt.close()

    print(f"Saved: {out_path}")

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    TARGET_CLASS = args.class_id
    print(f"---- Training autoencoder for class {TARGET_CLASS} ----")

    model, dataset = train_autoencoder()
    save_latents(model.encoder, dataset)
    visualize_reconstruction(model, dataset)

import os
import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import matplotlib.pyplot as plt
from scipy import ndimage
import time

# ============================================================
# CLI ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--class", "-c", type=int, default=0)
args = parser.parse_args()
TARGET_CLASS = args.__dict__["class"]

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 32
EPOCHS = 4000
BATCH_SIZE = 128
T = 1000
LR = 2e-4
METRIC_EVERY = 10
METRIC_BATCH = 32

ROOT = "."
UT = os.path.join(ROOT, "UT_HAR")
DATA_DIR = os.path.join(UT, "data")
LABEL_DIR = os.path.join(UT, "label")

# ============================================================
# LOAD UT-HAR (for viz)
# ============================================================
def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f"X_{name}.csv"), allow_pickle=True)
    y = np.load(os.path.join(LABEL_DIR, f"y_{name}.csv"), allow_pickle=True)
    return X.astype(np.float32), y.astype(int)

def load_ut_har_class(c):
    X, y = load_split("train")
    mask = (y == c)
    Xc = X[mask]
    print(f"Loaded UT-HAR class {c}: {Xc.shape[0]} samples")

    mn, mx = Xc.min(), Xc.max()
    Xc = 2 * (Xc - mn) / (mx - mn + 1e-8) - 1
    return Xc.astype(np.float32)

# ============================================================
# LATENT DATASET
# ============================================================
class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, latents):
        self.z = torch.from_numpy(latents).float()

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return self.z[idx]

# ============================================================
# DIFFUSION
# ============================================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-8, 0.999)

def sinusoidal_embedding(t, dim):
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1), math.log(10000), half, device=device))
    a = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class MLP(nn.Module):
    def __init__(self, latent_dim, time_dim=128, hidden=1024):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_dim = time_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)

class LatentDiffusion(nn.Module):
    def __init__(self, model, latent_dim, timesteps=T, device="cuda"):
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.device = device

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = torch.cat([torch.ones(1, device=device), ac[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("ac", ac)
        self.register_buffer("ac_prev", ac_prev)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_om", torch.sqrt(1 - ac))
        self.register_buffer("posterior_variance",
                             torch.clamp(betas * (1 - ac_prev) / (1 - ac), 1e-20))

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return self.sqrt_ac[t].view(-1, 1) * x0 + self.sqrt_om[t].view(-1, 1) * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        eps = self.model(x, t)
        x0 = (x - self.sqrt_om[t].view(-1, 1) * eps) / (self.sqrt_ac[t].view(-1, 1) + 1e-12)

        if (t == 0).all():
            return x0

        bet = self.betas[t].view(-1, 1)
        ac = self.ac[t].view(-1, 1)
        ac_prev = self.ac_prev[t].view(-1, 1)

        mean = (
            bet * torch.sqrt(ac_prev) / (1 - ac) * x0
            + (1 - ac_prev) * torch.sqrt(1 - bet) / (1 - ac) * x
        )
        noise = torch.randn_like(x)
        var = self.posterior_variance[t].view(-1, 1)

        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n):
        x = torch.randn((n, self.latent_dim), device=self.device)
        for t in reversed(range(self.timesteps)):
            tb = torch.full((n,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, tb)
        return x

# ============================================================
# METRICS
# ============================================================
def compute_psnr(x, y):
    mse = max(F.mse_loss(x, y).item(), 1e-10)
    return 20 * math.log10(2.0) - 10 * math.log10(mse)

@torch.no_grad()
def evaluate_fast_metrics(diffusion, decoder, lpips_model, real_z):
    fake_z = diffusion.sample(real_z.shape[0])
    real_im = decoder(real_z).clamp(-1, 1)
    fake_im = decoder(fake_z).clamp(-1, 1)

    mse = F.mse_loss(fake_im, real_im).item()
    psnr = compute_psnr(fake_im, real_im)
    lp = lpips_model(real_im.repeat(1,3,1,1), fake_im.repeat(1,3,1,1)).mean().item()

    return mse, psnr, lp

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"=== LATENT DIFFUSION for UT-HAR class {TARGET_CLASS} (PCA) ===")

    # For visualization only
    real_csi = load_ut_har_class(TARGET_CLASS)

    # Load PCA model
    pca_path = f"pca_models/pca_class_{TARGET_CLASS}.pkl"
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    print(f"Loaded PCA model → {pca_path}")

    # Load PCA latents
    latents_path = f"pca_latents/latents_class_{TARGET_CLASS}.npy"
    latents = np.load(latents_path)
    print(f"Loaded latents: {latents_path}, shape = {latents.shape}")

    dataset = LatentDataset(latents)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, drop_last=True)

    # Diffusion model
    net = MLP(LATENT_DIM).to(DEVICE)
    diffusion = LatentDiffusion(net, LATENT_DIM, T, DEVICE).to(DEVICE)

    lpips_model = lpips.LPIPS(net="vgg").to(DEVICE)
    optim = torch.optim.Adam(diffusion.parameters(), lr=LR)

    # Decoder function using PCA inverse
    def decode(z):
        arr = z.cpu().numpy()
        flat = pca.inverse_transform(arr)
        out = flat.reshape(-1, 1, 250, 90)
        return torch.from_numpy(out).float().to(DEVICE)

    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        diffusion.train()
        total = 0

        for z in loader:
            z = z.to(DEVICE)
            t = torch.randint(0, T, (z.size(0),), dtype=torch.long, device=DEVICE)

            loss = diffusion.p_losses(z, t)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()

        print(f"[Epoch {epoch}/{EPOCHS}] loss={total/len(loader):.6f}")

        # ---- Visualization ----
        with torch.no_grad():
            idx = np.random.randint(0, min(len(real_csi), len(latents)))

            real = real_csi[idx]
            recon = decode(torch.from_numpy(latents[idx:idx+1]).float().to(DEVICE))[0,0].cpu().numpy()
            fake = decode(diffusion.sample(1))[0,0].cpu().numpy()

        def norm_rot(x):
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            return ndimage.rotate(x, 90)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(norm_rot(real), cmap="viridis"); plt.title("Real"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(norm_rot(recon), cmap="viridis"); plt.title("PCA Recon"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(norm_rot(fake), cmap="viridis"); plt.title("Generated"); plt.axis("off")
        plt.show()

        # ---- Metrics ----
        if epoch % METRIC_EVERY == 0:
            idxs = np.random.choice(latents.shape[0], METRIC_BATCH, replace=False)
            real_batch_z = torch.from_numpy(latents[idxs]).float().to(DEVICE)

            t0 = time.time()
            mse, psnr, lp = evaluate_fast_metrics(diffusion, decode, lpips_model, real_batch_z)
            print(f"  > MSE={mse:.6f} PSNR={psnr:.3f} LPIPS={lp:.4f}")

    # Save synthetic CSI
    with torch.no_grad():
        synth_z = diffusion.sample(latents.shape[0])
        synth_csi = decode(synth_z).cpu().numpy()

    np.save(f"pca_synthetic_csi_class_{TARGET_CLASS}.npy", synth_csi)
    print("Saved:", f"pca_synthetic_csi_class_{TARGET_CLASS}.npy")

if __name__ == "__main__":
    main()

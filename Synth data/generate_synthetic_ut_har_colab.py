import os
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from scipy import ndimage
import matplotlib.pyplot as plt


# ================================================================
# CMD ARGUMENTS
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-class", "--class_id", type=int, default=0)
args = parser.parse_args()
TARGET_CLASS = args.class_id


# ================================================================
# CONFIG
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "."
UT_HAR_DIR = os.path.join(ROOT_DIR, "UT_HAR")
DATA_DIR = os.path.join(UT_HAR_DIR, "data")
LABEL_DIR = os.path.join(UT_HAR_DIR, "label")

LATENT_DIM = 512
EPOCHS = 4000
T = 1000
BATCH_SIZE = 64
LR = 2e-4
METRIC_EVERY = 10
METRIC_BATCH = 32


# ================================================================
# LOAD UT-HAR CLASS (FOR VISUALIZATION ONLY)
# ================================================================
def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f"X_{name}.csv"))
    y = np.load(os.path.join(LABEL_DIR, f"y_{name}.csv"))
    return X, y.reshape(-1)


def load_ut_har_class(cls):
    X_train, y_train = load_split("train")
    mask = (y_train == cls)
    Xc = X_train[mask]
    print(f"Loaded UT-HAR class {cls}: {Xc.shape[0]} samples")

    # normalize to [-1, 1]
    mn, mx = Xc.min(), Xc.max()
    Xc = 2 * (Xc - mn) / max(mx - mn, 1e-8) - 1
    return Xc.astype(np.float32)


# ================================================================
# AUTOENCODER (COPY OF autoencoder_full.py)
# ================================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.SiLU(),   # 125x45
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),  # 63x23
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),  # 32x12
        )
        self.flat_dim = 64 * 32 * 12
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.flat_dim = 64 * 32 * 12
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 32, 12)
        x = self.deconv(h)
        return F.interpolate(x, size=(250, 90), mode="bilinear")


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)


def load_autoencoder(path):
    ae = Autoencoder(LATENT_DIM).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    ae.load_state_dict(state)
    ae.eval()
    print(f"Loaded autoencoder weights: {path}")
    return ae


# ================================================================
# LATENT DATASET
# ================================================================
class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, latents):
        self.z = torch.from_numpy(latents).float()

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return self.z[idx]


# ================================================================
# DIFFUSION (on LATENTS)
# ================================================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_c = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_c /= alphas_c[0].clone()
    betas = 1 - (alphas_c[1:] / alphas_c[:-1])
    return betas.clamp(1e-8, 0.999)


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freq = torch.exp(torch.linspace(math.log(1), math.log(10000), half, device=t.device))
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class LatentMLP(nn.Module):
    def __init__(self, dim=LATENT_DIM, tdim=128, hidden=1024):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim), nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + tdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.tdim = tdim

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.tdim)
        t_emb = self.time_mlp(t_emb)
        return self.net(torch.cat([x, t_emb], dim=-1))


class Diffusion(nn.Module):
    def __init__(self, model, dim=LATENT_DIM, timesteps=T):
        super().__init__()
        self.model = model
        betas = cosine_beta_schedule(timesteps).to(DEVICE)
        alphas = 1 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = torch.cat([torch.ones(1, device=DEVICE), ac[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("ac", ac)
        self.register_buffer("ac_prev", ac_prev)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_om", torch.sqrt(1 - ac))

        self.timesteps = timesteps
        self.dim = dim

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_ac[t].unsqueeze(-1) * x0 + self.sqrt_om[t].unsqueeze(-1) * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred = self.model(x_noisy, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        eps = self.model(x, t)
        sqrt_ac = self.sqrt_ac[t].unsqueeze(-1)
        sqrt_om = self.sqrt_om[t].unsqueeze(-1)

        x0 = (x - sqrt_om * eps) / (sqrt_ac + 1e-12)

        if (t == 0).all():
            return x0

        beta_t = self.betas[t].unsqueeze(-1)
        ac_t = self.ac[t].unsqueeze(-1)
        ac_prev_t = self.ac_prev[t].unsqueeze(-1)

        mean = beta_t * torch.sqrt(ac_prev_t) / (1 - ac_t) * x0 + \
               (1 - ac_prev_t) * torch.sqrt(1 - beta_t) / (1 - ac_t) * x

        noise = torch.randn_like(x)
        var = beta_t * (1 - ac_prev_t) / (1 - ac_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n):
        x = torch.randn((n, self.dim), device=DEVICE)
        for t in reversed(range(self.timesteps)):
            tt = torch.full((n,), t, dtype=torch.long, device=DEVICE)
            x = self.p_sample(x, tt)
        return x


# ================================================================
# METRICS
# ================================================================
def compute_psnr(x, y):
    mse = F.mse_loss(x, y).item()
    mse = max(mse, 1e-10)
    return 20 * math.log10(2) - 10 * math.log10(mse)


@torch.no_grad()
def evaluate_fast_metrics(diffusion, decoder, lpips_model, real_latents):
    fake_z = diffusion.sample(real_latents.shape[0])
    real = decoder(real_latents).clamp(-1, 1)
    fake = decoder(fake_z).clamp(-1, 1)

    mse = F.mse_loss(fake, real).item()
    psnr = compute_psnr(fake, real)

    real_img = real.repeat(1, 3, 1, 1)
    fake_img = fake.repeat(1, 3, 1, 1)
    lp = lpips_model(real_img, fake_img).mean().item()

    return mse, psnr, lp


# ================================================================
# MAIN
# ================================================================
def main():
    print(f"=== LATENT DIFFUSION for UT-HAR class {TARGET_CLASS} ===")

    # Load UT-HAR for visualization only
    real_csi = load_ut_har_class(TARGET_CLASS)

    # Load latents
    lat_path = f"latents/latents_class_{TARGET_CLASS}.npy"
    latents = np.load(lat_path)
    print(f"Loaded latents: {lat_path}, shape={latents.shape}")

    ds = LatentDataset(latents)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Load autoencoder
    ae_path = f"autoencoders/autoencoder_class_{TARGET_CLASS}.pth"
    autoencoder = load_autoencoder(ae_path)
    decoder = autoencoder.decoder

    # Diffusion model
    model = LatentMLP().to(DEVICE)
    diffusion = Diffusion(model).to(DEVICE)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=LR)
    lpips_model = lpips.LPIPS(net="vgg").to(DEVICE)

    # Training
    for epoch in range(1, EPOCHS + 1):
        diffusion.train()
        total = 0
        for z in loader:
            z = z.to(DEVICE)
            t = torch.randint(0, T, (z.size(0),), device=DEVICE)

            loss = diffusion.p_losses(z, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"[Epoch {epoch}/{EPOCHS}] loss={total/len(loader):.6f}")

        # VISUALIZATION
        with torch.no_grad():
            idx = np.random.randint(0, latents.shape[0])
            real_im = real_csi[idx]

            z_real = torch.from_numpy(latents[idx:idx+1]).float().to(DEVICE)
            recon = decoder(z_real)[0,0].cpu().numpy()

            fake_z = diffusion.sample(1)
            fake = decoder(fake_z)[0,0].cpu().numpy()

        def norm(x):
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            return ndimage.rotate(x, 90)

        plt.figure(figsize=(12,4))
        plt.suptitle(f"Epoch {epoch}")

        plt.subplot(1,3,1)
        plt.imshow(norm(real_im), cmap="viridis")
        plt.title("Real")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(norm(recon), cmap="viridis")
        plt.title("Recon")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(norm(fake), cmap="viridis")
        plt.title("Generated")
        plt.axis("off")

        plt.show()
        plt.close()

        # METRICS
        if epoch % METRIC_EVERY == 0:
            idxs = np.random.choice(latents.shape[0], METRIC_BATCH, replace=False)
            batch = torch.from_numpy(latents[idxs]).float().to(DEVICE)
            mse, psnr, lp = evaluate_fast_metrics(diffusion, decoder, lpips_model, batch)
            print(f"  > MSE={mse:.6f}  PSNR={psnr:.3f}  LPIPS={lp:.4f}")

    # Save final synthetic CSI
    with torch.no_grad():
        synth_z = diffusion.sample(latents.shape[0])
        synth_csi = decoder(synth_z).cpu().numpy()

    np.save(f"synthetic_class_{TARGET_CLASS}.npy", synth_csi)
    print("Saved:", f"synthetic_class_{TARGET_CLASS}.npy")


if __name__ == "__main__":
    main()

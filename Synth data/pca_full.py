import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.decomposition import PCA

# ============================================================
# CLI ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--class", "-c", type=int, default=0)
args = parser.parse_args()
TARGET_CLASS = args.__dict__["class"]

# ============================================================
# PATHS
# ============================================================
ROOT = "."
UT = os.path.join(ROOT, "UT_HAR")
DATA_DIR = os.path.join(UT, "data")
LABEL_DIR = os.path.join(UT, "label")

os.makedirs("pca_models", exist_ok=True)
os.makedirs("pca_latents", exist_ok=True)
os.makedirs("pca_reconstructions", exist_ok=True)

LATENT_DIM = 32   # You can change (128/256/512)

# ============================================================
# LOAD UT-HAR DATA
# ============================================================
def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f"X_{name}.csv"), allow_pickle=True)
    y = np.load(os.path.join(LABEL_DIR, f"y_{name}.csv"), allow_pickle=True)
    return X.astype(np.float32), y.astype(int)

def load_ut_har_class(c):
    X, y = load_split("train")
    mask = (y == c)
    Xc = X[mask]
    print(f"Loaded class {c}: {Xc.shape[0]} samples")

    # normalize [-1,1]
    mn, mx = Xc.min(), Xc.max()
    Xc = 2 * (Xc - mn) / (mx - mn + 1e-8) - 1
    return Xc.astype(np.float32)

# ============================================================
# MAIN
# ============================================================
def main():
    Xc = load_ut_har_class(TARGET_CLASS)
    N = Xc.shape[0]

    # flatten
    X_flat = Xc.reshape(N, -1)

    print("Fitting PCA...")
    pca = PCA(n_components=LATENT_DIM)
    latents = pca.fit_transform(X_flat)

    # save PCA model
    pca_path = f"pca_models/pca_class_{TARGET_CLASS}.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model → {pca_path}")

    # save latents
    lat_path = f"pca_latents/latents_class_{TARGET_CLASS}.npy"
    np.save(lat_path, latents)
    print(f"Saved PCA latents → {lat_path}, shape={latents.shape}")

    # reconstruct example
    print("Rendering reconstruction example...")
    recon = pca.inverse_transform(latents[0])  # (22500,)
    recon = recon.reshape(250, 90)
    real = Xc[0]

    def norm_rot(x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return ndimage.rotate(x, 90, reshape=True)

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"PCA Reconstruction (class {TARGET_CLASS})")

    plt.subplot(1, 2, 1)
    plt.imshow(norm_rot(real), cmap="viridis")
    plt.title("Real")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(norm_rot(recon), cmap="viridis")
    plt.title("PCA Reconstruction")
    plt.axis("off")

    out_path = f"pca_reconstructions/recon_class_{TARGET_CLASS}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
import argparse
import asyncio
from util import load_data_n_model
from csi_server import start_csi_server, csi_broadcast
from csi_server import send_label
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from scipy import ndimage


# ============================================================
#   LABELS
# ============================================================
class_names = [
    "Lie Down",
    "Fall Down",
    "Walking",
    "Pick Up",
    "Run",
    "Sit Down",
    "Stand Up"
]


# ============================================================
#   LOG FILE
# ============================================================
def log_activity(label: str):
    os.makedirs("logs", exist_ok=True)
    with open("logs/activity_log.txt", "a", encoding="utf-8") as f:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{stamp}: {label}\n")


# ============================================================
#   CSI to Base64 PNG  (old-style visualization)
# ============================================================
def csi_to_base64(tensor):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage
    import base64, io

    # Convert tensor (1,250,90) → (250,90)
    csi = tensor.detach().cpu().numpy()
    if csi.ndim == 3:
        csi = csi[0]

    # (1) NORMALIZATION — EXACT NOTEBOOK STYLE
    csi = (csi - np.min(csi)) / (np.max(csi) - np.min(csi) + 1e-8)

    # (2) ROTATE EXACTLY LIKE NOTEBOOK (90°, preserve shape)
    csi = ndimage.rotate(csi, 90, reshape=True)

    # (3) CREATE FIGURE EXACTLY LIKE YOUR NOTEBOOK
    #     WIDE aspect ratio, no smoothing, jet colormap
    plt.figure(figsize=(12, 3), dpi=100)      # EXACT SHAPE STYLE
    plt.imshow(csi, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.axis("off")

    # (4) SAVE WITHOUT CROPPING
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)

    # (5) RETURN BASE64 FOR BROWSER
    return base64.b64encode(buf.getvalue()).decode("utf-8")




# ============================================================
#   SEND LABEL + IMAGE
# ============================================================
async def send_label_with_image(label, img_b64):
    import json
    payload = json.dumps({
        "label": label,
        "image": img_b64
    })
    await csi_broadcast(payload)


# ============================================================
#   WEIGHTS UTILS
# ============================================================
WEIGHTS_ROOT = "weights"


def get_latest_weights(model_name):
    """
    Returns path to latest weights file or None
    """
    if not os.path.exists(WEIGHTS_ROOT):
        return None

    model_dir = os.path.join(WEIGHTS_ROOT, model_name)
    if not os.path.exists(model_dir):
        return None

    runs = sorted(os.listdir(model_dir))
    if not runs:
        return None

    latest_run = runs[-1]
    weights_path = os.path.join(model_dir, latest_run, "model.pt")

    return weights_path if os.path.exists(weights_path) else None


def save_weights(model, model_name):
    """
    Saves model weights to weights/<model_name>/<timestamp>/model.pt
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(WEIGHTS_ROOT, model_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), path)

    print(f"💾 Weights saved to {path}")
    return path


def load_weights_if_available(model, model_name, device):
    """
    Loads latest weights if they exist
    """
    weights_path = get_latest_weights(model_name)

    if weights_path is None:
        print("🆕 No saved weights found — training from scratch.")
        return False

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✅ Loaded weights from {weights_path}")
    return True


# ============================================================
#   TRAIN (ASYNC)
# ============================================================
async def train_async(model, loader, num_epochs, lr, criterion, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"\n===== TRAIN EPOCH {epoch+1} =====")

        model.train()
        total_loss = 0
        total_acc = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # stats
            total_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(1)
            total_acc += (pred == labels).float().mean().item()

            # label
            pred_idx = pred[0].item()
            label_str = class_names[pred_idx]

            # log file write
            log_activity(label_str)

            await send_label(label_str)    # <-- send backend notification

            # CSI Base64 image
            img_b64 = csi_to_base64(inputs[0])

            # send to browser
            await send_label_with_image(label_str, img_b64)

            await asyncio.sleep(0)

        print(f"[TRAIN] Loss={total_loss/len(loader.dataset):.4f}  "
              f"Acc={total_acc/len(loader):.4f}")


# ============================================================
#   TEST (ASYNC)
# ============================================================
async def test_async(model, loader, criterion, device):
    print("\n===== TESTING =====")

    total_loss = 0
    total_acc = 0
    model.eval()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).long()

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # batch stats (unchanged logic)
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        total_acc += (preds == labels).float().mean().item()

        # 🔽 shuffle only the displayed samples
        indices = torch.randperm(inputs.size(0))

        for i in indices:
            i = i.item()

            pred_idx = preds[i].item()
            label_str = "" + class_names[pred_idx]

            log_activity(label_str)

            await send_label(label_str)

            img_b64 = csi_to_base64(inputs[i])
            await send_label_with_image(label_str, img_b64)

            await asyncio.sleep(1)



# ============================================================
#   MAIN
# ============================================================
async def async_main():
    print("🔵 Starting CSI server…")
    await start_csi_server()
    print("🟢 CSI servers are running.")

    root = "./Data/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print("using dataset:", args.dataset)
    print("using model:", args.model)

    train_loader, test_loader, model, train_epoch = load_data_n_model(
        args.dataset, args.model, root
    )

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights_loaded = load_weights_if_available(
        model,
        args.model,
        device
    )

    if weights_loaded:
        print("🔍 Testing loaded weights before training...")
        await test_async(model, test_loader, criterion, device)
    else:
        await train_async(model, train_loader, train_epoch, 1e-3, criterion, device)
        save_weights(model, args.model)
        await test_async(model, test_loader, criterion, device)

    print("🎉 Training + Testing Finished.")


if __name__ == "__main__":
    asyncio.run(async_main())

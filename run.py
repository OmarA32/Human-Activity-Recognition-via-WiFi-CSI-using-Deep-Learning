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

        total_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(1)
        total_acc += (pred == labels).float().mean().item()

        pred_idx = pred[0].item()
        label_str = "[TEST] " + class_names[pred_idx]

        log_activity(label_str)

        await send_label(label_str)    # <-- send backend notification

        # In test, send CSI image
        img_b64 = csi_to_base64(inputs[0])
        await send_label_with_image(label_str, img_b64)

        await asyncio.sleep(0)

    print(f"[TEST] Loss={total_loss/len(loader.dataset):.4f}  "
          f"Acc={total_acc/len(loader):.4f}")


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

    await train_async(model, train_loader, train_epoch, 1e-3, criterion, device)
    await test_async(model, test_loader, criterion, device)

    print("🎉 Training + Testing Finished.")


if __name__ == "__main__":
    asyncio.run(async_main())

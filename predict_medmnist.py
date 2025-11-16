import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torchvision.transforms as T
from medmnist import ChestMNIST, INFO

from ViT import ThaVit  # your ViT model


# -----------------------------
# Config (must match training)
# -----------------------------
IMAGE_SIZE  = 128
PATCH_SIZE  = 8
NUM_CLASSES = 14
DIM         = 256
DEPTH       = 8
HEADS       = 8
DIM_HEAD    = 64
MLP_DIM     = DIM * 4

# ChestMNIST normalization (what you used in train_medmnist.py)
mean = (0.5,)
std  = (0.5,)


# Helper: show grayscale image
def show(img, upscale=224):
    """
    Show a ChestMNIST image tensor in 'real' grayscale.

    - img: tensor (1, H, W)
    - unnormalizes using mean/std
    - upscales for visibility
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()

    # (1, H, W) -> (H, W)
    img = img.squeeze(0).numpy()

    # unnormalize: x * std + mean
    img = img * std[0] + mean[0]
    img = np.clip(img, 0, 1)

    # to uint8
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")

    # upscale
    if upscale is not None and upscale > max(pil_img.size):
        pil_img = pil_img.resize((upscale, upscale), resample=Image.NEAREST)

    plt.imshow(pil_img, cmap="gray")
    plt.axis("off")
    plt.show()


# Device & model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = ThaVit(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    dim=DIM,
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD,
    mlp_dim=MLP_DIM,
    channels=1,      # IMPORTANT: ChestMNIST is grayscale
    attn_drop=0.0,
    proj_drop=0.1,
    mlp_drop=0.1,
).to(device)

ckpt = torch.load("vit_chestmnist_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()


# Dataset (use test or val split)
test_tf = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_set = ChestMNIST(root="medmnist",
                      split="val",
                      size=128,
                      download=True,
                      transform=test_tf)

# class names from medmnist INFO
info = INFO["chestmnist"]
label_dict = info["label"]     # e.g. {"0": "Atelectasis", ...}
class_names = [label_dict[str(i)] for i in range(NUM_CLASSES)]

# Pick random example and predict
idx = random.randint(0, len(test_set) - 1)
img, label = test_set[idx]   # img: (1, 28, 28), label: multi-label vector

# label can be numpy or tensor, make it 1D numpy
label = np.array(label).reshape(-1)

print("True Labels:")
any_pos = False
for i, v in enumerate(label):
    if v == 1:
        print(f" {class_names[i]}")
        any_pos = True

if not any_pos:
    print("No Disease")

show(img)

# Prepare batch
x = img.unsqueeze(0).to(device)   # (1, 1, 28, 28)

with torch.no_grad():
    logits = model(x)             # (1, 14)
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # (14,)

# Show top-k predicted labels
top_k = 5
top_indices = probs.argsort()[::-1][:top_k]

print("\nTop predictions:")
for i in top_indices:
    print(f" {class_names[i]} : prob={probs[i]:.3f}")

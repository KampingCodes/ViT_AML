import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torchvision.transforms as T
from medmnist import ChestMNIST, INFO

from ViT import ThaVit  # ViT model


# Config
IMAGE_SIZE  = 128
PATCH_SIZE  = 8
NUM_CLASSES = 14
DIM         = 256
DEPTH       = 8
HEADS       = 8
DIM_HEAD    = 64
MLP_DIM     = DIM * 4

# ChestMNIST normalization
mean = (0.5,)
std  = (0.5,)

# show image
def show(img, upscale=224):

    img = img.squeeze(0).numpy()
    img = img * std[0] + mean[0]
    img = np.clip(img, 0, 1)

    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")

    if upscale is not None and upscale > max(pil_img.size):
        pil_img = pil_img.resize((upscale, upscale), resample=Image.NEAREST)

    plt.imshow(pil_img, cmap="gray")
    plt.axis("off")
    plt.show()

# Load model
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
    channels=1,
    attn_drop=0.0,
    proj_drop=0.1,
    mlp_drop=0.1,
).to(device)

best_model = torch.load("vit_chestmnist_best.pt", map_location=device)
model.load_state_dict(best_model["model"])
model.eval()

# Dataset
test_tf = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_set = ChestMNIST(root="medmnist",
                      split="val",
                      size=128,
                      download=True,
                      transform=test_tf)

# class names
info = INFO["chestmnist"]
label_dict = info["label"]
class_names = [label_dict[str(i)] for i in range(NUM_CLASSES)]

def sample_positive_case(dataset):
    """Return (img, label, idx) but only for samples that contain at least 1 disease."""
    while True:
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        label = np.array(label).reshape(-1)
        if label.sum() >= 1:  # must have at least 1 disease
            return img, label, idx


img, label, idx = sample_positive_case(test_set)

print(f"\nSelected image index: {idx}")

# Print true labels
print("\nTrue Labels:")
for i, v in enumerate(label):
    if v == 1:
        print(f"  {class_names[i]}")

show(img)

# Predict
x = img.unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

top_k = 5
top_indices = probs.argsort()[::-1][:top_k]

print("\nTop Predictions:")
for i in top_indices:
    print(f"  {class_names[i]} : prob={probs[i]:.3f}")

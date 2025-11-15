import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from PIL import Image  # NEW

from ViT import ThaVit  # your model


# Config
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CLASSES = 100
DIM = 256
DEPTH = 8
HEADS = 8
DIM_HEAD = 64
MLP_DIM = DIM * 4

# CIFAR-100 normalization
mean = (0.5071, 0.4867, 0.4408)
std  = (0.2675, 0.2565, 0.2761)


# Helper: Show image (unnormalize + upscale for clarity)
def show(img, upscale=256):
    """
    Show a CIFAR-100 image tensor in 'real' RGB space.

    - img: tensor (C, H, W)
    - unnormalizes using CIFAR-100 mean/std
    - upscales to `upscale` x `upscale` with NEAREST so it looks sharp
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()

    # (C, H, W) -> (H, W, C)
    img = img.numpy().transpose(1, 2, 0)

    # unnormalize: x * std + mean  (broadcasted over channels)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)

    # to uint8 for PIL / display
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    # upscale for visibility (CIFAR is 32x32)
    if upscale is not None and upscale > max(pil_img.size):
        pil_img = pil_img.resize((upscale, upscale), resample=Image.NEAREST)

    plt.imshow(pil_img)
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
).to(device)

best_vit = torch.load("vit_best.pt", map_location=device)
model.load_state_dict(best_vit["model"])
model.eval()


# Load CIFAR-100 test set
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std)
])

test_set = CIFAR100(root="data", train=False, download=True, transform=test_tf)
class_names = test_set.classes

# Pick random image
idx = random.randint(0, len(test_set) - 1)
img, label = test_set[idx]

print(f"Image index: {idx}")
print(f"True Class: {label} ({class_names[label]})")

show(img)  # now unnormalized + upscaled


# Prepare input batch
x = img.unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item()

print(f"Predicted: {pred} ({class_names[pred]})")

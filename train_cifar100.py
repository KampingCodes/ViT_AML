import os
import math
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ViT import ThaVit


# config
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 50
WEIGHT_DECAY = 5e-2
NUM_WORKERS = 0
SEED = 1337
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CLASSES = 100
DIM = 256
DEPTH = 8
HEADS = 8
DIM_HEAD = 64
MLP_DIM = DIM * 4
ATTN_DROP = 0.0
PROJ_DROP = 0.1
MLP_DROP = 0.1
LABEL_SMOOTH = 0.1

def seed_everything(seed=SEED):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

# -----------------------------
# data
# -----------------------------
def make_loaders():
    # CIFAR-100 stats
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = CIFAR100(root="data", download=True, train=True, transform=train_tf)
    test_set = CIFAR100(root="data", download=True, train=False, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


# training / eval
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        steps += 1

    return total_loss / steps, total_acc / steps

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        steps += 1
    return total_loss / steps, total_acc / steps


# main
def main():
    seed_everything()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders()

    model = ThaVit(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_dim=MLP_DIM,
        channels=3,
        attn_drop=ATTN_DROP,
        proj_drop=PROJ_DROP,
        mlp_drop=MLP_DROP,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

    warmup_epochs = 10
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # cosine over the remaining steps
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict()}, "vit_best.pt")
            print(f"✓ saved checkpoint (val_acc={best_val_acc*100:.2f}%)")

    print("Done.")

if __name__ == "__main__":
    main()

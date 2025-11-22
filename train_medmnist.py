import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from medmnist import ChestMNIST
from sklearn.metrics import roc_auc_score

from ViT import ThaVit


# config
BATCH_SIZE    = 64
LR            = 3e-4
EPOCHS        = 30
WEIGHT_DECAY  = 5e-2
NUM_WORKERS   = 0
SEED          = 1337

# ChestMNIST - 128x128 grayscale, multi-label with 14 classes
IMAGE_SIZE    = 128
PATCH_SIZE    = 8
NUM_CLASSES   = 14
DIM           = 256
DEPTH         = 8
HEADS         = 8
DIM_HEAD      = 64
MLP_DIM       = DIM * 4
ATTN_DROP     = 0.0
PROJ_DROP     = 0.1
MLP_DROP      = 0.1


# utils
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


def accuracy_multilabel(logits, y, threshold=0.5):
    """
    Simple multi-label accuracy:
    - Apply sigmoid
    - Threshold at 0.5
    - Compare to ground truth and average over all labels and batch
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == y).float().mean().item()
    return correct


# data
def make_loaders():
    # grayscale normalization
    mean = (0.5,)
    std  = (0.5,)

    train_tf = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = ChestMNIST(root="medmnist", split="train", size=128,
                           download=True, transform=train_tf)
    val_set   = ChestMNIST(root="medmnist", split="test", size=128,
                           download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader

def topk_hit_rate(probs, labels, k=5):
    """
    probs:  (N, C) numpy array, sigmoid outputs in [0,1]
    labels: (N, C) numpy array, 0/1 ground truth (multi-label)

    Returns:
        fraction of samples where at least one true positive label
        appears in the top-k predicted labels.
    """
    N, C = probs.shape
    hits = 0
    valid = 0

    for i in range(N):
        true_pos = np.where(labels[i] > 0.5)[0]

        # no positive labels -> skip from metric
        if true_pos.size == 0:
            continue

        # indices of top-k probabilities
        topk_idx = np.argpartition(probs[i], -k)[-k:]

        if np.intersect1d(true_pos, topk_idx).size > 0:
            hits += 1

        valid += 1

    if valid == 0:
        return float("nan")

    return hits / valid


# training / eval
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss, steps = 0.0, 0
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)

        # y: ensure float, shape (B, num_labels)
        y = torch.as_tensor(y, device=device).float()
        if y.ndim > 2:
            y = y.view(y.size(0), -1)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1

        # store for AUC computation
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    avg_loss = total_loss / steps

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()

    try:
        train_auc = roc_auc_score(labels, probs)
    except ValueError:
        # happens if some class is all 0 or all 1 in this split
        train_auc = float("nan")

    return avg_loss, train_auc

@torch.no_grad()
def evaluate(model, loader, device, k=5):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss, steps = 0.0, 0
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)

        y = torch.as_tensor(y, device=device).float()
        if y.ndim > 2:
            y = y.view(y.size(0), -1)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        steps += 1

        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    avg_loss = total_loss / steps

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()

    # macro AUC
    try:
        val_auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        val_auc = float("nan")

    # top-K hit rate
    val_topk = topk_hit_rate(probs, labels, k=k)

    return avg_loss, val_auc, val_topk


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
        channels=1,                  # ChestMNIST is grayscale
        attn_drop=ATTN_DROP,
        proj_drop=PROJ_DROP,
        mlp_drop=MLP_DROP,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 10
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_auc = 0.0
    for epoch in range(EPOCHS):
        tr_loss, tr_auc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        va_loss, va_auc, top5 = evaluate(model, val_loader, device, k=5)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss {tr_loss:.4f} AUC {tr_auc:.4f} | "
            f"val loss {va_loss:.4f} AUC {va_auc:.4f} | "
            f"top-5 hit {top5 * 100:.2f}%"
        )

        score_for_ckpt = va_auc
        if not math.isnan(score_for_ckpt) and score_for_ckpt > best_val_auc:
            best_val_auc = score_for_ckpt
            torch.save({"model": model.state_dict()}, "vit_chestmnist_best.pt")
            print(f"✓ saved checkpoint (val_auc={best_val_auc:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()

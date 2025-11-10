import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

# constants
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10
DECORR_LOSS_WEIGHT = 1e-1
TRACK_EXPERIMENT_ONLINE = False

# helpers
def exists(v):
    return v is not None

# data
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(
    root = 'data',
    download = True,
    train = True,
    transform = transform
)

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# model
from ViT import ThaVit
v = ThaVit(
    dim = 128,
    num_classes = 10,
    image_size = 32,
    patch_size = 4,
    depth = 6,
    heads = 8,
    dim_head = 64,
    mlp_dim = 128 * 4,
)

# optim
from torch.optim import Adam
optim = Adam(v.parameters(), lr = LEARNING_RATE)

# prepare
from accelerate import Accelerator
accelerator = Accelerator()
print(accelerator.device)
vit, optim, dataloader = accelerator.prepare(v, optim, dataloader)

# training loop
for epoch in range(EPOCHS):
    accelerator.print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
    for images, labels in dataloader:
        logits = vit(images)
        loss = F.cross_entropy(logits, labels)
        accelerator.print(f'loss: {loss.item():3f}')
        accelerator.backward(loss)
        optim.step()
        optim.zero_grad()

torch.save(accelerator.unwrap_model(vit).state_dict(), "vit.pth")
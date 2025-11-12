import torch
from ViT import ThaVit
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(
    root = 'data',
    download = False,
    train = False,
    transform = transform
)

dataloader = DataLoader(dataset, 32, True)

v = ThaVit(
    dim=128,
    num_classes=10,
    image_size=32,
    patch_size=4,
    depth=6,
    heads=8,
    dim_head=64,
    mlp_dim=128 * 4,
)

v.load_state_dict(torch.load('ViT.pth', map_location = torch.device('cpu')))
v.eval()

correct = 0
total = 0

for images, labels in dataloader:
    outputs = v(images)
    predicted = torch.argmax(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total} %')

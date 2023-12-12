import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ds_path='./imgs'
dataset = datasets.ImageFolder(ds_path)
print(f'N images: {len(dataset)}')
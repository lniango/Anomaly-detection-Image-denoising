import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch import nn, optim
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

mnist_dataset_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
mnist_dataset_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
    mnist_dataset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    mnist_dataset_test, batch_size=5, shuffle=False)

"""
# Take a batch
images, labels = next(iter(train_loader))
# make a Grid
grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2)
# MNIST -> 1 channel
plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0).squeeze(), cmap="gray")
plt.title("Sample of MNIST images")
plt.axis("off")
plt.show()
"""
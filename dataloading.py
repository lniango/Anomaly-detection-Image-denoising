import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Add gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std  = std
        
    def __call__(self, x):
        return x + torch.randn(x.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1),
    #AddGaussianNoise(mean=0, std=1)
])


mnist_dataset_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
mnist_dataset_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

batch_size_train = 128
batch_size_test = 5

train_loader = torch.utils.data.DataLoader(
    mnist_dataset_train, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    mnist_dataset_test, batch_size=batch_size_test, shuffle=False)

#for img, label in train_loader:
#    print(f"Shape of img : {img.shape} | Shape of label : {label.shape}")

"""
mnist_dataset_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
mnist_dataset_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)


# Data loading
batch_size_train = 128
batch_size_test = 5

train_loader = torch.utils.data.DataLoader(
    mnist_dataset_train, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    mnist_dataset_test, batch_size=batch_size_test, shuffle=False)

# Take a batch
images, labels = next(iter(test_loader))
# make a Grid
grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2)
# MNIST -> 1 channel
plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0).squeeze().clip(min=0.0, max=1.0), cmap="gray")
plt.title("Sample of MNIST images")
plt.axis("off")
plt.show()
"""
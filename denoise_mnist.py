import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch import nn, optim
from dataloading import mnist_dataset_test, mnist_dataset_train
from model import Denoiser 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data loading
batch_size_train = 128
batch_size_test = 5

train_loader = torch.utils.data.DataLoader(
    mnist_dataset_train, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    mnist_dataset_test, batch_size=batch_size_test, shuffle=False)

# Define the model
model = Denoiser(latent_dim=8)
model.to(device)

# optimizer & loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
start_epoch = 0
last_epoch = 10
for epoch in range(start_epoch, last_epoch):
    model.train()
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        
        pred = model(img)
        loss = criterion(pred, label)
from datetime import datetime
import time

import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch import nn, optim
from dataloading import mnist_dataset_test, mnist_dataset_train
from model import Denoiser 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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

# Writer
start = time.time()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/denoise_trainer_{timestamp}')

def add_noise(x, mean=0, std=1):
    return x + torch.randn(x.size()) * std + mean

# Training loop
def train():
    start_epoch = 0
    last_epoch = 51
    for epoch in tqdm(range(start_epoch, last_epoch), total=last_epoch+1, desc="Training loop"):
        model.train()
        training_loss = 0
        val_loss = 0

        for batch_idx, (img, label) in enumerate(train_loader):
            # Add noise
            noisy_img = add_noise(img) # Add noise in the image
            noisy_img = noisy_img.to(device)
            label = label.to(device)

            # Zero gradients for every batch
            optimizer.zero_grad()
            
            pred = model(noisy_img)
            #print(f"Shape prediction: {pred.shape} | target shape : {label.shape}")
            loss = criterion(pred, img)
            training_loss += loss

            # Adjust learning weights
            optimizer.step()
            
            # Display images - Tensorboard
            if batch_idx == 0:
                #writer.add_image("Train batch", img[0], global_step=len(train_loader)//16)
                writer.add_image("Noisy Train batch", noisy_img[0], global_step=len(train_loader)//16)
                writer.add_image("Pred on train set", pred[0], global_step=len(train_loader)//16)
            
        training_loss += training_loss / len(train_loader)
        writer.add_scalar('Training Loss', training_loss, epoch + 1)

        # Evaluation
        if epoch % 1 == 0:
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()
            with torch.no_grad():
                for batch_idx_val, (val_img, labelv) in enumerate(test_loader):
                    noisy_val = add_noise(val_img) # Add noise
                    
                    noisy_val = noisy_val.to(device)
                    labelv = labelv.to(device)
                    predv = model(noisy_val)
                    lossv = criterion(predv, val_img)
                    val_loss += lossv
                    
                    if batch_idx_val == 0:
                        # Display images - Tensorboard
                        #writer.add_image("Val batch", val_img[0], global_step=len(test_loader)//4)
                        writer.add_image("Noisy Val batch", noisy_val[0], global_step=len(test_loader)//4)
                        writer.add_image("Pred on val set", predv[0], global_step=len(test_loader)//4)

        val_loss += val_loss / len(test_loader)
        writer.add_scalar('Validation loss', val_loss, epoch + 1)
        

        writer.flush()

        # Save model
        if epoch % 10 == 0:
            #model_path = f'models/model_{timestamp}_{epoch}'
            model_path = f'models/model_{epoch}'
            torch.save(model.state_dict(), model_path)
        
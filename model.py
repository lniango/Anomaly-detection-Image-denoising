import torch
import torch.utils.data
import torchvision
import torch.nn as nn

class Denoiser(nn.Module):
    # https://arxiv.org/pdf/2101.07937
    def __init__(self, latent_dim):
        super().__init__()
        
        self.latent_dim = latent_dim
        # Encoding
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1) #Input : 1 channel - MNIST dataset
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, latent_dim, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        #self.conv3 = nn.Conv2d(8, latent_dim, 3, stride=1, padding=1)
        
        # Decoding
        #self.deconv3 = nn.ConvTranspose2d(latent_dim, 8, 3)
        self.deconv2 = nn.ConvTranspose2d(latent_dim, 16, 3)
        self.deconv1 = nn.ConvTranspose2d(16, 1, 3)
        self.sigmoid = nn.Sigmoid()
    
    def encoder(self, x, latent_dim):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))
        return x
        
    def decoder(self, latent_dim):
        x = self.maxpool2(self.relu(self.decon2(x)))
        x = self.sigmoid(self.deconv1(x))
        return x
 
    def forward(self, x):
        y = self.encoder(x)
        x_hat = self.decoder(y)
        return x_hat
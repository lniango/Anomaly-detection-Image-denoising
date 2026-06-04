import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch import nn, optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Define the model

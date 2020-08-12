import torch
import torch.nn as nn
import numpy as np

from dataset import ToyDataset
from torch.utils.data import DataLoader

dataset = ToyDataset('data.csv')

train_loader = DataLoader(dataset=dataset,
                         batch_size=32, 
                         shuffle=True, 
                         num_workers=2)

epochs = 1

for epoch in range(epochs):
    for i, x in enumerate(train_loader):
        
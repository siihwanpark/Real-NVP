import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, path):
        self.data = torch.from_numpy(np.loadtxt(path, delimiter=',', dtype=np.float32))
        
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]
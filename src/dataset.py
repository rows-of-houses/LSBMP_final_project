from torch.utils.data import Dataset
import numpy as np
import torch

class NumpyCsvDataSet(Dataset):
    def __init__(self, csv_file, indeces):
        self.csv_file = csv_file
        self.data = torch.from_numpy(np.loadtxt(csv_file, delimiter=',', dtype=float))
        self.data = self.data[indeces]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
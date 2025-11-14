import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .basedataset import BaseDataset


class TransformerDataset(BaseDataset):
    def __init__(self, data_path, split='train'):
        super().__init__(data_path, split)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        if sample.dim() == 1:
            sample = sample.unsqueeze(-1)
        
        return sample

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class BaseDataset(Dataset):
    def __init__(self, data_path, split='train'):

        self.split = split
        self.data_path = Path(data_path)
        self.data = self._load_data()
    
    def _load_data(self):
        file_dict = {
            'train': 'train_data.npy',
            'eval': 'evaluation_data.npy',
            'bias_correction': 'bias_correction_data.npy',
            'test': 'test_data.npy'
        }
        filename = file_dict.get(self.split)
        if filename is None:
            raise ValueError(f"Invalid split name: {self.split}. Please select from {list(file_dict.keys())}.")
        
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}. You may need to run the preprocessor script.")
        
        data = np.load(filepath)
        return torch.from_numpy(data).float()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def get_data_shape(self):
        return self.data.shape


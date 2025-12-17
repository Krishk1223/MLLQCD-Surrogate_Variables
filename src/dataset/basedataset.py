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
            'train_feature': 'train_data_X.npy',
            'train_target': 'train_data_y.npy',
            'eval_feature': 'evaluation_data_X.npy',
            'eval_target': 'evaluation_data_y.npy',
            'bias_correction_feature': 'bias_correction_data_X.npy',
            'bias_correction_target': 'bias_correction_data_y.npy',
            'test_feature': 'test_data_X.npy',
            'test_target': 'test_data_y.npy'
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


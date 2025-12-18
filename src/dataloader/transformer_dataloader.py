import os
import torch
from torch.utils.data import DataLoader
from src.dataset.transformer_dataset import TransformerDataset


def TransformerDataLoader(data_path, config, split='train'):
    #pin memory for faster gpu transfer if cuda available:
    pin_memory = torch.cuda.is_available()

    #config params:
    batch_size = config['training']['batch_size']
    workers = int(config['training']['num_workers'])

    #loaders:
    if split == 'train':
        train_dataset = TransformerDataset(data_path, split='train') # train dataset instance
        train_loader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle=True,
            num_workers = workers,
            pin_memory = pin_memory
        )
        return train_loader #returns only train loader

    elif split == 'eval':
        eval_dataset = TransformerDataset(data_path, split='eval') # eval dataset instance
        eval_loader = DataLoader(
            eval_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers = workers,
            pin_memory = pin_memory
        )
        return eval_loader #returns only eval loader

    elif split == 'bias':
        bias_dataset = TransformerDataset(data_path, split='bias') # bias correction dataset instance
        bias_loader = DataLoader(
            bias_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory
        )

    elif split == 'test':
        test_dataset = TransformerDataset(data_path, split='test') # test dataset instance
        test_loader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = workers,
            pin_memory = pin_memory
        )
        return test_loader #returns only test loader
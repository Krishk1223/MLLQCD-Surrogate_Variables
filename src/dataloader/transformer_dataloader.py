import os
import torch
from torch.utils.data import DataLoader
from src.dataset.transformer_dataset import TransformerDataset


def TransformerDataLoader(data_path, config):
    #create datasets:
    train_dataset = TransformerDataset(data_path, split='train')
    eval_dataset = TransformerDataset(data_path, split='eval')
    test_dataset = TransformerDataset(data_path, split='test')

    #pin memory for faster gpu transfer if cuda available:
    pin_memory = torch.cuda.is_available()

    #config params:
    batch_size = config['training']['batch_size']
    workers = int(config['training']['num_workers'])

    #loaders:
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers = workers,
        pin_memory = pin_memory
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size = batch_size,
        shuffle=False,
        num_workers = workers,
        pin_memory = pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        pin_memory = pin_memory
    )
    return train_loader, eval_loader, test_loader
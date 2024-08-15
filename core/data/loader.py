import logging

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from core.utils import comm

from .datasets import build_datasets, build_dataset

def build_train_dataloader(cfg: DictConfig):
    dataset = build_datasets(cfg.datasets.train)
    
    sampler = DistributedSampler(dataset) # if is_distributed else None
    loader = DataLoader(
        dataset, 
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        sampler=sampler,
        # shuffle=True,
        drop_last=True, 
    )

    return sampler, loader

def build_test_dataloader(cfg, dataset_name: str):
    # return None 
    dataset = build_dataset(dataset_name) 
    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        drop_last=False,
    )


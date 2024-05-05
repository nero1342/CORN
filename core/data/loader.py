import logging
from omegaconf import DictConfig 

import torch
from torch.utils.data import Dataset, DataLoader

from core.utils import comm
from .datasets import build_datasets

def build_train_dataloader(cfg: DictConfig):
    dataset = build_datasets(cfg.dataset.train)
    
    sampler = DistributedSampler(dataset) # if is_distributed else None
    loader = DataLoader(
        dataset, 
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        sampler=sampler,
        shuffle=True,
        drop_last=True, 
    )

    return loader
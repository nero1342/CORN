import logging

import numpy as np
import torch

from datasets import load_dataset

from .build import DATASET_REGISTRY
from .base import BaseDataset

logger = logging.getLogger(__name__)

@DATASET_REGISTRY.register()
class ClassificationDataset(BaseDataset):
    def __init__(self, is_train: bool = True):
        data = load_dataset('fcakyon/pokemon-classification', 'full')

        if is_train:
            self.data = data['train']
        else:
            self.data = data['test']
        
        id_maps = set(x['labels'] for x in self.data) 
        self.id_maps = {v:i for i, v in enumerate(id_maps)}
        logger.info(f"Loaded {len(self.data)} images!")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        img = self._read_image(x['image'], read_mode='pillow')

        return {
            "image": torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))) / 255.,
            "class": self.id_maps[x['labels']]
        }

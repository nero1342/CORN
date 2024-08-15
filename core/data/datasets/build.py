from typing import Dict
import torch
import logging 
from torch.utils.data import Dataset
import copy 

from omegaconf import DictConfig
from fvcore.common.registry import Registry 

from core.data.builtin import PREDEFINED_DATASETS
logger = logging.getLogger(__name__)

DATASET_REGISTRY = Registry("Dataset")

def build_dataset(dataset_name: str) -> Dataset:
    assert dataset_name in PREDEFINED_DATASETS, f"Dataset {dataset_name} is not defined, check file builtin.py"
    assert "dataset" in PREDEFINED_DATASETS[dataset_name], f"Dataset {dataset_name} do not contain dataset field, check file builtin.py"
    
    cfg = copy.deepcopy(PREDEFINED_DATASETS[dataset_name]["dataset"])
    dataset_type = cfg.pop('_target_')
    logger.info(f"Instantiating dataset <{dataset_name}> : <{dataset_type}>")
    dataset = DATASET_REGISTRY.get(dataset_type)(**cfg)

    return dataset

def build_datasets(dataset_names):
    datasets = [build_dataset(dataset_name) for dataset_name in dataset_names]

    if len(datasets) == 1:
       return datasets[0]
    
    return ConcatDataset(datasets)
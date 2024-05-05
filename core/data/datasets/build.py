# import torch
# import logging 
# from torch.utils.data import Dataset

# logger = logging.getLogger(__name__)

# def build_dataset(name):

# from omegaconf import DictConfig
# from fvcore.common.registry import Registry 

# import torch
# from torch import nn
# import logging 

# logger = logging.getLogger(__name__)

# BACKBONE_REGISTRY = Registry("Backbone")

# def build_dataset(dataset_name: DictConfig) -> Dataset:
#     dataset_name = cfg.pop('_target_')
#     logger.info(f"Instantiating backbone <{backbone_name}>")
#     backbone = BACKBONE_REGISTRY.get(backbone_name)(**cfg)
    
#     return backbone

# def build_datasets(cfg):
#     dataset_names = cfg.datasets.train
#     datasets = []
#     for dataset_name in dataset_names:

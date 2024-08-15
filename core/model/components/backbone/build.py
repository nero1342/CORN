import logging

import torch
from fvcore.common.registry import Registry
from omegaconf import DictConfig
from torch import nn

logger = logging.getLogger(__name__)

BACKBONE_REGISTRY = Registry("Backbone")

def build_backbone(cfg: DictConfig) -> nn.Module:
    backbone_name = cfg.pop('_target_')
    logger.info(f"Instantiating backbone <{backbone_name}>")
    backbone = BACKBONE_REGISTRY.get(backbone_name)(**cfg)
    
    return backbone
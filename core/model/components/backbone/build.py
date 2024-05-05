from omegaconf import DictConfig
from fvcore.common.registry import Registry 

import torch
from torch import nn
import logging 

logger = logging.getLogger(__name__)

BACKBONE_REGISTRY = Registry("Backbone")

def build_backbone(cfg: DictConfig) -> nn.Module:
    backbone_name = cfg.pop('_target_')
    logger.info(f"Instantiating backbone <{backbone_name}>")
    backbone = BACKBONE_REGISTRY.get(backbone_name)(**cfg)
    
    return backbone
import logging

import torch
from fvcore.common.registry import Registry
from omegaconf import DictConfig
from torch import nn

logger = logging.getLogger(__name__)

CRITERION_REGISTRY = Registry("Criterion")

def build_criterion(cfg: DictConfig) -> nn.Module:
    criterion_name = cfg.pop('_target_')
    logger.info(f"Instantiating criterion <{criterion_name}>")
    criterion = CRITERION_REGISTRY.get(criterion_name)(**cfg)
    return criterion
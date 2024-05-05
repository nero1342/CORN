from omegaconf import DictConfig
from fvcore.common.registry import Registry 

import torch
from torch import nn
import logging 

logger = logging.getLogger(__name__)

ARCH_REGISTRY = Registry("Arch")
ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

def build_model(cfg: DictConfig) -> nn.Module:
    arch_name = cfg.pop('_target_')
    logger.info(f"Instantiated meta-architecture <{arch_name}>")
    model = ARCH_REGISTRY.get(arch_name)(**cfg)
    
    return model
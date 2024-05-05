import logging
import torch
from torch import nn

from omegaconf import DictConfig

from . import ARCH_REGISTRY
from core.model.components.backbone import build_backbone
from core.model.criterion import build_criterion

logger = logging.getLogger(__name__)

@ARCH_REGISTRY.register()
class Classifier(nn.Module):
    def __init__(self, backbone: DictConfig, criterion: DictConfig):
        super().__init__()

        self.backbone = build_backbone(backbone)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, 10)

        self.criterion = build_criterion(criterion)

    def forward(self, data):
        images = data["image"]
        classes = data["class"]
        
        x = self.backbone(images)["res5"]
        x = self.pool(x).flatten(1)
        x = self.fc(x)

        if self.training:
            # Loss
            losses = self.criterion(x, classes)
            return {"losses": losses }
        else:
            return x

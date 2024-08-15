import logging

import torch
from omegaconf import DictConfig
from torch import nn

from core.model.components.backbone import build_backbone
from core.model.criterion import build_criterion

from . import ARCH_REGISTRY

logger = logging.getLogger(__name__)

@ARCH_REGISTRY.register()
class Classifier(nn.Module):
    def __init__(self, backbone: DictConfig, criterion: DictConfig, num_classes: int):
        super().__init__()

        self.backbone = build_backbone(backbone)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, num_classes)

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

            losses = self.criterion(x, classes)
            pred = x.argmax(dim=1)
            return {
                "losses": losses,
                "pred": pred
            }

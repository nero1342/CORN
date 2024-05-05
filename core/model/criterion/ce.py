import torch 
from torch import nn
import torch.nn.functional as F 

from .build import CRITERION_REGISTRY

@CRITERION_REGISTRY.register()
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, outputs):
        return F.cross_entropy(inputs, outputs)
    
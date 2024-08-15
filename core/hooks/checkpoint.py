import logging

import torch
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch import nn

from .base import HookBase

logger = logging.getLogger(__name__)

class PeriodCheckpointer(HookBase, PeriodicCheckpointer):
    def after_step(self):
        cur_iter = self.storage.iter
        self.step(cur_iter)
        
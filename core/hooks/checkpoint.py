import logging
import torch 
from torch import nn
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from core.utils.events import get_event_storage

from .base import HookBase
logger = logging.getLogger(__name__)

class PeriodCheckpointer(HookBase, PeriodicCheckpointer):
    pass 
    # def __init__(self, checkpointer: Checkpointer, period: int):
    #     super().__init__()

    def after_step(self):
        storage = get_event_storage()
        cur_iter = storage.iter
        self.step(cur_iter)
            
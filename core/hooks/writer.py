import logging
from typing import List

from core.writer import EventWriter

from .base import HookBase

logger = logging.getLogger(__name__)

class PeriodWriter(HookBase):
    def __init__(self, writers: List[EventWriter], period: int = 20):
        super().__init__()
        # logger.info("Initialized PeriodWriter.")
        self.writers = writers
        self.period = period 

    def after_step(self):
        cur_iter = self.storage.iter
        if (cur_iter + 1) % self.period == 0:
            for writer in self.writers:
                writer.write()
    
    def after_train(self):
        for writer in self.writers:
            writer.write()
            writer.close()
    
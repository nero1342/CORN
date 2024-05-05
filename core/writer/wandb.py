from functools import cached_property
from core.utils.events import get_event_storage
import wandb 
import hydra

from . import EventWriter, WRITER_REGISTRY

@WRITER_REGISTRY.register()
class WandBWriter(EventWriter):
    """
    Write all scalars to wanndb server.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        self._writer_args = {"log_dir": log_dir, **kwargs}
        self._last_write = -1

        wandb.init(dir = log_dir, **kwargs)

    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write

        log_dict = {}
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                log_dict[k] = v
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write
        wandb.log(log_dict, new_last_write)

    def close(self):
        if "_writer" in self.__dict__:
            self._writer.close()

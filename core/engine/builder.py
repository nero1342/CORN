import logging

import torch
from fvcore.common.checkpoint import Checkpointer
from omegaconf import DictConfig
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from core.model.arch import build_model
from core.model.utils.parameter_groups import get_parameter_groups
from core.utils import comm
from core.utils.events import EventStorage
from core.utils.model_summary import model_summary

logger = logging.getLogger(__name__)


class TrainerBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        model = self.build_model(cfg.model)
        self.optimizer = self.build_optimizer(cfg.solver, model)
        self.scheduler = self.build_scheduler(cfg.solver, self.optimizer)

        logger.info(model_summary(model))
        self.model = create_ddp_model(model)

        self.checkpointer = Checkpointer(self.model, cfg.paths.output_dir)
        
    def resume_or_load(self, resume: bool = False):
        raise NotImplementedError

    @classmethod
    def build_model(cls, cfg: DictConfig):
        logger.info("Building model...")
        model = build_model(cfg).cuda()
        return model

    @classmethod
    def build_optimizer(
        cls, cfg: DictConfig, model: torch.nn.Module
    ) -> optim.Optimizer:
        param_groups = get_parameter_groups(cfg, model)

        optimizer_type = cfg.optimizer
        if optimizer_type == "SGD":
            optimizer = optim.SGD(param_groups, lr=cfg.base_lr, momentum=cfg.momentum)
        elif optimizer_type == "AdamW":
            optimizer = optim.AdamW(param_groups, lr=cfg.base_lr)
        else:
            raise NotImplementedError(f"No optimizer type {optimizer_type}")

        return optimizer

    @classmethod
    def build_scheduler(
        cls, cfg: DictConfig, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        if cfg.lr_scheduler == "poly":
            total_iterations = cfg.max_iter
            return optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: (1 - (x / total_iterations)) ** 0.9
            )
        elif cfg.lr_scheduler == "step":
            return optim.lr_scheduler.MultiStepLR(
                optimizer, cfg.steps, cfg.scheduler_gamma
            )
        else:
            raise NotImplementedError


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model

    local_rank = comm.get_rank()
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import \
            default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp

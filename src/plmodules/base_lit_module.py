#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from torch import optim
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.opt.schedulers.warmup_cosine_decay_scheduler import WarmupCosineDecayScheduler


class BaseLitModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self.net = hydra.utils.instantiate(cfg.model)
        # compile the network only once, skip if it’s already been compiled.
        self._net_compiled = False

        sdpa_map = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
        }

        self.sdpa_backends = [sdpa_map[backend] for backend in self.cfg.accelerate.sdpa]

    def _model_forward(self, *args, **kwargs):
        with sdpa_kernel(self.sdpa_backends):
            return self.net(*args, **kwargs)

    def setup(self, stage: str) -> None:
        if self.cfg.accelerate.compile and stage == "fit" and torch.__version__ >= "2.0.0" and not self._net_compiled:
            self.net = torch.compile(self.net)
            self._net_compiled = True

    def get_lr_scheduler(self, optimizer):
        scheduler = WarmupCosineDecayScheduler(
            optimizer=optimizer,
            warmup=int(self.cfg.opt.warmup_percent * self.cfg.trainer.max_steps // 100),
            max_iters=int(self.cfg.opt.decay_percent * self.cfg.trainer.max_steps // 100),
        )
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

    def get_optimizer(self):
        if self.cfg.opt.optimizer == "AdamW":
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=float(self.cfg.opt.peak_lr),
                weight_decay=float(self.cfg.opt.weight_decay),
            )
        elif self.cfg.opt.optimizer == "Muon":
            from src.opt.optimizers.muon import Muon

            muon_params, adamw_params = Muon.split_muon_adamw_params(self.net)

            optimizer = Muon(
                lr=float(self.cfg.opt.peak_lr),
                wd=float(self.cfg.opt.weight_decay),
                muon_params=muon_params,
                adamw_params=adamw_params,
            )
        else:
            raise ValueError(f"Optimizer {self.cfg.opt.optimizer} not supported")
        return optimizer

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        lr_scheduler = self.get_lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

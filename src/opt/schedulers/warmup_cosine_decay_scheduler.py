#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import numpy as np
from torch import optim


class WarmupCosineDecayScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr_factor = epoch * 1.0 / max(self.warmup, 1)
        else:
            progress = (epoch - self.warmup) / (self.max_num_iters - self.warmup)
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return lr_factor

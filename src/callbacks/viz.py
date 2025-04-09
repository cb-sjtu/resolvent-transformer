import os
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
from PIL import Image

import src.utils.custom_utils as cu

from . import viz_utils as vu


class Viz(L.Callback):
    def __init__(
        self,
        dirpath: str,
        valid_max_batches_local: int,  # save batches in local machine
        valid_max_batches_log: int,  # log batches to remote wandb
        test_max_batches_local: int,  # save batches in local machine
        test_max_batches_log: int,  # log batches to remote wandb
    ):
        super().__init__()
        self.dirpath = dirpath
        self.valid_max_batches_local = valid_max_batches_local
        self.valid_max_batches_log = valid_max_batches_log
        self.test_max_batches_local = test_max_batches_local
        self.test_max_batches_log = test_max_batches_log

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image:
        img = vu.merge_images([[None]])  # merge a list of list of matplotlib plots or PIL images
        plt.close("all")
        return img  # PIL image

    def on_validation_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            for dataloader_idx in range(len(pl_module.cfg.data.valid)):
                dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
                dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
                os.makedirs(dirpath, exist_ok=True)
        if torch.distributed.is_initialized():  # only for distributed training
            torch.distributed.barrier()  # wait for all processes to finish

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx >= self.valid_max_batches_local:
            return

        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        img = self.get_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
        img.save(dirpath / f"{batch_idx}_rank{trainer.local_rank}.png")  # save image in all processes

        if batch_idx >= self.valid_max_batches_log:
            return

        img = vu.fig_to_wandb(img)
        for logger in trainer.loggers:
            try:  # noqa: SIM105
                logger.log_image(key=f"{dataset_name}", images=[img], step=trainer.global_step)
            except:  # noqa: E722
                pass

    def on_test_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            for dataloader_idx in range(len(pl_module.cfg.data.test)):
                dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
                dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
                os.makedirs(dirpath, exist_ok=True)
        if torch.distributed.is_initialized():  # only for distributed training
            torch.distributed.barrier()  # wait for all processes to finish

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx >= self.test_max_batches_local:
            return

        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        img = self.get_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
        img.save(dirpath / f"{batch_idx}_rank{trainer.local_rank}.png")  # save image in all processes

        if batch_idx >= self.valid_max_batches_log:
            return

        img = vu.fig_to_wandb(img)
        for logger in trainer.loggers:
            try:  # noqa: SIM105
                logger.log_image(key=f"{dataset_name}", images=[img], step=trainer.global_step)
            except:  # noqa: E722
                pass

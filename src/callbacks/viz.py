import os
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import src.utils.custom_utils as cu

from . import viz_utils as vu


class Viz(L.Callback):
    def __init__(
        self,
        valid_max_batches_local: int,  # save batches in local machine
        valid_max_batches_log: int,  # log batches to remote wandb
        test_max_batches_local: int,  # save batches in local machine
        test_max_batches_log: int,  # log batches to remote wandb
    ):
        self.valid_max_batches_local = valid_max_batches_local
        self.valid_max_batches_log = valid_max_batches_log
        self.test_max_batches_local = test_max_batches_local
        self.test_max_batches_log = test_max_batches_log

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image:
        img = vu.merge_images([[None]])  # merge plots of all meshes in one line, PIL image
        plt.close("all")
        return img  # PIL image

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        # create a directory for valid viz, rank 0 only to avoid race condition
        for dataloader_idx in range(len(pl_module.cfg.data.valid)):
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
            dirpath = (
                Path(pl_module.cfg.paths.output_dir) / "viz" / "valid" / f"step_{trainer.global_step}" / dataset_name
            )
            os.makedirs(dirpath, exist_ok=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx >= self.valid_max_batches_local:
            return

        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        img = self.get_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        dirpath = Path(pl_module.cfg.paths.output_dir) / "viz" / "valid" / f"step_{trainer.global_step}" / dataset_name
        img.save(dirpath / f"{batch_idx}.png")

        if batch_idx >= self.valid_max_batches_log:
            return

        img = vu.fig_to_wandb(img)
        for logger in trainer.loggers:
            try:  # noqa: SIM105
                logger.log_image(key=f"{dataset_name}", images=[img], step=trainer.global_step)
            except:  # noqa: E722
                pass

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        # create a directory for test viz, rank 0 only to avoid race condition
        for dataloader_idx in range(len(pl_module.cfg.data.test)):
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
            dirpath = (
                Path(pl_module.cfg.paths.output_dir) / "viz" / "test" / f"step_{trainer.global_step}" / dataset_name
            )
            os.makedirs(dirpath, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx >= self.test_max_batches_local:
            return

        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        img = self.get_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        dirpath = Path(pl_module.cfg.paths.output_dir) / "viz" / "test" / f"step_{trainer.global_step}" / dataset_name
        img.save(dirpath / f"{batch_idx}.png")

        if batch_idx >= self.valid_max_batches_log:
            return

        img = vu.fig_to_wandb(img)
        for logger in trainer.loggers:
            try:  # noqa: SIM105
                logger.log_image(key=f"{dataset_name}", images=[img], step=trainer.global_step)
            except:  # noqa: E722
                pass

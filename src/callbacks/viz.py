import os
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import loggers
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

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
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

        self.log_image(trainer, img, key=f"{dataset_name}")

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

        self.log_image(trainer, img, key=f"{dataset_name}")

    def log_image(
        self,
        trainer: L.Trainer,
        img: Image.Image,
        key: str,
        artifact_file: str = None,
    ):
        for logger in trainer.loggers:
            if isinstance(logger, loggers.WandbLogger):
                logger.log_image(key=key, images=[img], step=trainer.global_step)
            elif isinstance(logger, loggers.TensorBoardLogger):
                # do whatever the tensorboard logger supports
                pass
            elif isinstance(logger, loggers.MLFlowLogger):
                # do whatever the mlflow logger supports
                # Note that the mlflow_logger is essentially an instance of mlflow.Client
                # https://github.com/Lightning-AI/pytorch-lightning/issues/3964#issuecomment-705348121
                # See https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.log_image
                # for API
                # example usage:
                # logger.experiment.log_image(
                #     run_id=logger.run_id,
                #     image=img,
                #     key=key,
                #     artifact_file=artifact_file,
                #     step=trainer.global_step,
                # )
                pass

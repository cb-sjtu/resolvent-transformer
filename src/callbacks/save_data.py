import os
from pathlib import Path

import lightning as L
import torch

import src.utils.custom_utils as cu


class SaveData(L.Callback):
    def __init__(
        self,
        dirpath: str,
        print_lv_local: int,
        print_lv_log: int,
        train_max_batches_local: int,
        train_max_batches_log: int,
        valid_max_batches_local: int,
        valid_max_batches_log: int,
        test_max_batches_local: int,
        test_max_batches_log: int,
    ):
        self.dirpath = dirpath
        self.print_lv_local = print_lv_local
        self.print_lv_log = print_lv_log
        self.train_max_batches_local = train_max_batches_local
        self.train_max_batches_log = train_max_batches_log
        self.valid_max_batches_local = valid_max_batches_local
        self.valid_max_batches_log = valid_max_batches_log
        self.test_max_batches_local = test_max_batches_local
        self.test_max_batches_log = test_max_batches_log

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.is_global_zero:
            dirpath = Path(self.dirpath) / "train"
            os.makedirs(dirpath, exist_ok=True)
        if torch.distributed.is_initialized():  # only for distributed training
            torch.distributed.barrier()  # wait for all processes to finish

    def on_validation_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            for dataloader_idx in range(len(pl_module.cfg.data.valid)):
                dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
                dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
                os.makedirs(dirpath, exist_ok=True)
        if torch.distributed.is_initialized():  # only for distributed training
            torch.distributed.barrier()  # wait for all processes to finish

    def on_test_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            for dataloader_idx in range(len(pl_module.cfg.data.test)):
                dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
                dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
                os.makedirs(dirpath, exist_ok=True)
        if torch.distributed.is_initialized():  # only for distributed training
            torch.distributed.barrier()  # wait for all processes to finish

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        data, label = batch["data"], batch["label"]
        if batch_idx < self.train_max_batches_log:
            pl_module.print(f"===== Train Batch # {batch_idx} =====")
            pl_module.print(data.get_print_info(print_lv=self.print_lv_log))
            pl_module.print(label.get_print_info(print_lv=self.print_lv_log))

        if batch_idx < self.train_max_batches_local:
            # save to file, append to file end
            with open(Path(self.dirpath) / "train" / f"rank_{trainer.local_rank}.txt", "a") as f:
                f.write(f"===== Train Batch # {batch_idx} =====\n")
                f.write(data.get_print_info(print_lv=self.print_lv_local))
                f.write(label.get_print_info(print_lv=self.print_lv_local))
                f.write("\n")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        data, label = batch["data"], batch["label"]
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        if batch_idx < self.valid_max_batches_log:
            pl_module.print(f"===== Valid Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====")
            pl_module.print(data.get_print_info(print_lv=self.print_lv_log))
            pl_module.print(label.get_print_info(print_lv=self.print_lv_log))

        if batch_idx < self.valid_max_batches_local:
            # save to file, append to file end
            with open(
                (
                    Path(self.dirpath)
                    / "valid"
                    / f"step_{trainer.global_step}"
                    / dataset_name
                    / f"rank_{trainer.local_rank}.txt"
                ),
                "a",
            ) as f:
                f.write(f"===== Valid Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====\n")
                f.write(data.get_print_info(print_lv=self.print_lv_local))
                f.write(label.get_print_info(print_lv=self.print_lv_local))
                f.write("\n")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        data, label = batch["data"], batch["label"]
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        if batch_idx < self.test_max_batches_log:
            pl_module.print(f"===== Test Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====")
            pl_module.print(data.get_print_info(print_lv=self.print_lv_log))
            pl_module.print(label.get_print_info(print_lv=self.print_lv_log))

        if batch_idx < self.test_max_batches_local:
            # save to file, append to file end
            with open(
                (
                    Path(self.dirpath)
                    / "test"
                    / f"step_{trainer.global_step}"
                    / dataset_name
                    / f"rank_{trainer.local_rank}.txt"
                ),
                "a",
            ) as f:
                f.write(f"===== Test Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====\n")
                f.write(data.get_print_info(print_lv=self.print_lv_local))
                f.write(label.get_print_info(print_lv=self.print_lv_local))
                f.write("\n")

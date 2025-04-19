from pathlib import Path

import lightning as L

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

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        dirpath = Path(self.dirpath) / "train"
        dirpath.mkdir(parents=True, exist_ok=True)

        if batch_idx < self.train_max_batches_log:
            pl_module.print(f"===== Train Batch # {batch_idx} =====")
            for key, value in batch.items():
                pl_module.print(value.get_print_info(print_lv=self.print_lv_log, info=key))
            pl_module.print("", flush=True)  # add a newline after each batch

        if batch_idx < self.train_max_batches_local:
            filename = Path(self.dirpath) / "train" / f"rank_{trainer.global_rank}.txt"
            with open(filename, "a") as f:  # save to file, append to file end
                f.write(f"===== Train Batch # {batch_idx} =====\n")
                for key, value in batch.items():
                    f.write(value.get_print_info(print_lv=self.print_lv_local, info=key))
                    f.write("\n")
                f.write("\n")  # add a newline after each batch

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
        dirpath.mkdir(parents=True, exist_ok=True)

        if batch_idx < self.valid_max_batches_log:
            pl_module.print(f"===== Valid Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====")
            for key, value in batch.items():
                pl_module.print(value.get_print_info(print_lv=self.print_lv_log, info=key))
            pl_module.print("", flush=True)  # add a newline after each batch

        if batch_idx < self.valid_max_batches_local:
            filename = (
                Path(self.dirpath)
                / "valid"
                / f"step_{trainer.global_step}"
                / dataset_name
                / f"rank_{trainer.global_rank}.txt"
            )
            with open(filename, "a") as f:  # save to file, append to file end
                f.write(f"===== Valid Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====\n")
                for key, value in batch.items():
                    f.write(value.get_print_info(print_lv=self.print_lv_local, info=key))
                    f.write("\n")
                f.write("\n")  # add a newline after each batch

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
        dirpath.mkdir(parents=True, exist_ok=True)

        if batch_idx < self.test_max_batches_log:
            pl_module.print(f"===== Test Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====")
            for key, value in batch.items():
                pl_module.print(value.get_print_info(print_lv=self.print_lv_log, info=key))
            pl_module.print("", flush=True)  # add a newline after each batch

        if batch_idx < self.test_max_batches_local:
            filename = (
                Path(self.dirpath)
                / "test"
                / f"step_{trainer.global_step}"
                / dataset_name
                / f"rank_{trainer.global_rank}.txt"
            )
            with open(filename, "a") as f:  # save to file, append to file end
                f.write(f"===== Test Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx} =====\n")
                for key, value in batch.items():
                    f.write(value.get_print_info(print_lv=self.print_lv_local, info=key))
                    f.write("\n")
                f.write("\n")  # add a newline after each batch

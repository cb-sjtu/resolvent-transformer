import os
from pathlib import Path

import lightning as L

import src.utils.custom_utils as cu


class SaveMetric(L.Callback):
    def __init__(
        self,
        dirpath: str,
    ) -> None:
        super().__init__()
        self.dirpath = dirpath

    def on_validation_start(self, trainer, pl_module):
        self.valid_outputs = {}
        if trainer.is_global_zero:
            for dataloader_idx in range(len(pl_module.cfg.data.valid)):
                dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
                valid_dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
                os.makedirs(valid_dirpath, exist_ok=True)

    def on_test_start(self, trainer, pl_module):
        self.test_outputs = {}
        if trainer.is_global_zero:
            for dataloader_idx in range(len(pl_module.cfg.data.test)):
                dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
                test_dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
                os.makedirs(test_dirpath, exist_ok=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs: dict, batch, batch_idx, dataloader_idx=0):
        """Cache valid batch outputs. Only save metrics since they are smaller than preds and errors."""
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        if dataset_name not in self.valid_outputs:
            self.valid_outputs[dataset_name] = []
        self.valid_outputs[dataset_name].append(outputs["metrics"])

    def on_test_batch_end(self, trainer, pl_module, outputs: dict, batch, batch_idx, dataloader_idx=0):
        """Cache test batch outputs. Only save metrics since they are smaller than preds and errors."""
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        if dataset_name not in self.test_outputs:
            self.test_outputs[dataset_name] = []
        self.test_outputs[dataset_name].append(outputs["metrics"])

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save valid outputs for each process."""
        for dataloader_idx in range(len(pl_module.cfg.data.valid)):
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
            valid_dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
            self._save_outputs(valid_dirpath, self.valid_outputs[dataset_name], trainer.local_rank)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save test outputs for each process."""
        for dataloader_idx in range(len(pl_module.cfg.data.test)):
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
            test_dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
            self._save_outputs(test_dirpath, self.test_outputs[dataset_name], trainer.local_rank)

    def _save_outputs(self, dirpath: Path, outputs: list[dict], rank: int) -> None:
        if len(outputs) == 0:
            return  # in case of no valid_step or test_step
        # save the outputs for each process
        for key in outputs[0]:
            file_key = key.replace("/", "_")
            full_path = dirpath / f"{file_key}_rank{rank}.txt"
            with open(full_path, "w") as f:
                for out in outputs:
                    tensor = out[key].detach().cpu().numpy()
                    if tensor.ndim == 0:  # scalar, sometimes metrics are not sample-wise
                        f.write(str(tensor) + "\n")
                    else:  # (bs, ...)
                        for t in tensor:  # one line per sample
                            f.write(" ".join(map(str, t.flatten())) + "\n")

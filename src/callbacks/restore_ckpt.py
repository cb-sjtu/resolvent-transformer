import os
from pathlib import Path

import lightning as L
import parse


class RestoreCheckpoint(L.Callback):
    def __init__(self, ckpt_root: str, filename_fmt: str) -> None:
        super().__init__()

        if ckpt_root is None:
            raise ValueError(
                "Please specify valid ckpt_root by adding `callbacks.restore_ckpt.ckpt_root=/path/to/ckpt_root`"
            )

        ckpt_filenames = os.listdir(ckpt_root)

        self.step_to_ckpt = {}
        for filename in ckpt_filenames:
            print(filename)
            parse_result = parse.parse(filename_fmt, filename)
            if parse_result is not None and "step" in parse_result.named:
                step = parse_result.named["step"]
                self.step_to_ckpt[step] = Path(ckpt_root) / filename
                print(step, Path(ckpt_root) / filename)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pl_module.print(f"{trainer.global_step=}")
        if trainer.global_step in self.step_to_ckpt:
            pl_module.restore_ckpt(self.step_to_ckpt[trainer.global_step])
            pl_module.print(f"Restored checkpoint at step {trainer.global_step}")

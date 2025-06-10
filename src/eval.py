#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rich
import rootutils
import torch
import torch._dynamo
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule validset and testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.accelerate.dynamo_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = cfg.accelerate.dynamo_cache_size_limit

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision(cfg.accelerate.fp32_matmul_precision)

    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    torch.backends.cuda.matmul.allow_tf32 = cfg.accelerate.fp32_matmul_precision != "highest"
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # pass the whole config to the datamodule and plmodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)(cfg=cfg)

    log.info(f"Instantiating model <{cfg.plmodule._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.plmodule)(cfg=cfg)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # Set max_epochs=0 to skip any training and only restore the model state from checkpoint during fit()
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, max_epochs=0)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info(f"Restoring model state from checkpoint {cfg.ckpt_path} ...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Running validation...")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    if cfg.get("test"):
        log.info("Running test...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    return trainer.callback_metrics, object_dict


def prepare_eval_cfg(cfg: DictConfig) -> Path:
    """
    - Copy the training config and checkpoints into the eval output directory.
    - Return the path to the copied eval config file.
    """
    train_dir = Path(cfg.train_dir)
    train_cfg_path = train_dir / ".hydra" / "config.yaml"
    train_ckpt_dir = train_dir / "checkpoints"

    if not train_cfg_path.is_file() or not train_ckpt_dir.is_dir():
        log.error("Training config or ckpt dir missing")
        sys.exit(1)

    # Copy config and checkpoints
    shutil.copy2(train_cfg_path, cfg.paths.train_conf)
    shutil.copytree(train_ckpt_dir, cfg.paths.ckpt_dir, dirs_exist_ok=True)
    log.info(f"Copied train config → {cfg.paths.train_conf}")
    log.info(f"Copied checkpoints → {cfg.paths.ckpt_dir}")

    # Load the training config, replace trainer and model in eval config
    train_cfg = OmegaConf.load(cfg.paths.train_conf)
    cfg.trainer = train_cfg.trainer
    cfg.plmodule = train_cfg.plmodule
    cfg.model = train_cfg.model
    cfg.seed = train_cfg.seed
    # Only run validation and testing, no need to load optimizer
    # cfg.opt = train_cfg.opt

    return cfg


# if eval_custom.yaml exists, use it as default config file
# otherwise, need to specify config file in command line
config_file_name = "eval_custom.yaml" if os.path.exists("./configs/eval_custom.yaml") else None


@hydra.main(version_base="1.3", config_path="../configs/", config_name=config_file_name)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Copy and prepare evaluation cfg
    cfg = prepare_eval_cfg(cfg)

    all_metrics: dict[str, dict[str, Any]] = {}
    all_objects: dict[str, dict[str, Any]] = {}

    for ckpt_path in Path(cfg.paths.ckpt_dir).glob("*.ckpt"):
        log.info(f"Evaluating {ckpt_path.name} ...")
        # Overwrite ckpt_path to cfg so evaluate(cfg) can access it, since @task_wrapper accepts only one argument
        cfg.ckpt_path = ckpt_path
        metrics, objs = evaluate(cfg)
        all_metrics[ckpt_path.name] = metrics
        all_objects[ckpt_path.name] = objs

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=all_metrics, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.operator_model.data import OperatorDataModule
from src.operator_model.operator_lit_module import OperatorLitModule
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    torch._dynamo.config.cache_size_limit = cfg.dynamo_cache_size_limit

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Dummy data module setup
    #########################################################
    log.info("Preparing data module...")
    log.info("Using dummy data generated")

    # Create data module with default random data generation
    datamodule = OperatorDataModule(cfg=cfg)
    #########################################################

    log.info("Creating model...")
    model = OperatorLitModule(cfg=cfg, compile=cfg.get("compile", False))

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # callbacks.append(
    #     ModelCheckpoint(
    #         monitor="valid/loss",
    #         filename="epoch_{epoch:03d}",
    #         save_top_k=3,
    #         mode="min",
    #         save_last=True,
    #     )
    # )

    # callbacks.append(LearningRateMonitor(logging_interval="step"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("train", True):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test", False):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(config_path="../../configs", config_name="operator/train", version_base="1.3")
def main(cfg: DictConfig) -> float:
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

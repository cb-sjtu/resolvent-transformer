import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import Logger, CSVLogger
from omegaconf import DictConfig, OmegaConf

from src.operator_model.operator_lit_module import OperatorLitModule
from src.operator_model.data import OperatorDataModule, OperatorData
from src.utils import ( 
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch._dynamo.config.cache_size_limit = cfg.dynamo_cache_size_limit

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    #dummy data
    #########################################################
    log.info("loading data...")
    if cfg.get("generate_synthetic_data", False):
        num_functions = 1000
        f_seq_len = cfg.model.get("f_seq_len", 100)
        g_seq_len = cfg.model.get("g_seq_len", 50)
        f_input_dim = cfg.model.f_input_dim
        g_input_dim = cfg.model.g_input_dim
        g_output_dim = cfg.model.g_output_dim
        
        train_data = []
        for _ in range(num_functions):
            train_data.append(
                OperatorData(
                    f_samples=torch.randn(1, f_seq_len, f_input_dim),
                    g_inputs=torch.randn(1, g_seq_len, g_input_dim),
                    g_targets=torch.randn(1, g_seq_len, g_output_dim)
                )
            )
        
        val_data = []
        for _ in range(num_functions//5):
            val_data.append(
                OperatorData(
                    f_samples=torch.randn(1, f_seq_len, f_input_dim),
                    g_inputs=torch.randn(1, g_seq_len, g_input_dim),
                    g_targets=torch.randn(1, g_seq_len, g_output_dim)
                )
            )
        
        datamodule = OperatorDataModule(
            cfg=cfg,
            train_data=train_data,
            val_data=val_data
        )
    else:
        log.warning("loading data from file, set generate_synthetic_data=True.")
        datamodule = None
    #########################################################

    log.info("creating model...")
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

    # trainer = L.Trainer(
    #     accelerator=cfg.trainer.get("accelerator", "auto"),
    #     devices=cfg.trainer.get("devices", "auto"),
    #     precision=cfg.trainer.get("precision", 32),
    #     max_epochs=cfg.trainer.get("max_epochs", 100),
    #     max_steps=cfg.trainer.get("max_steps", -1),
    #     log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
    #     callbacks=callbacks,
    #     logger=logger,
    #     deterministic=cfg.get("deterministic", False),
    # )
    
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
    
    #train the model
    metric_dict, _ = train(cfg)
    
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main() 
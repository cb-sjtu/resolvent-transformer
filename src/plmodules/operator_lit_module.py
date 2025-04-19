import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.custom_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class OperatorLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self.net = hydra.utils.instantiate(cfg.model)

        self.train_metrics = MeanMetric()

        # Use MetricCollection to group metrics
        self.metric_names = [
            "loss",  # total loss
            "error",  # error
            # add more metrics here
        ]

        self.valid_metrics = torch.nn.ModuleList(
            [
                MetricCollection({k: MeanMetric() for k in self.metric_names})
                for _ in range(len(self.cfg.data.valid))  # initialize metrics for each valid_loader
            ]
        )

    def network_inference(self, data: PyTree):
        outputs = self._model_forward(memory=data["f_samples"], query=data["g_inputs"])
        return outputs

    def _loss_function(self, batch: PyTree) -> torch.Tensor:
        pred = self.network_inference(batch["data"])
        return F.mse_loss(pred, batch["label"])

    def get_pred(self, data: PyTree) -> torch.Tensor:
        return self.network_inference(data)

    def get_error(self, batch: PyTree) -> torch.Tensor:
        pred = self.get_pred(batch["data"])
        return torch.abs(pred - batch["label"])

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics:
            metrics.reset()

    def training_step(self, batch: PyTree, batch_idx: int) -> torch.Tensor:
        loss = self._loss_function(batch)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss

    ############ validation #############
    def validation_step(self, batch: PyTree, batch_idx: int, dataloader_idx: int = 0) -> dict:
        loss = self._loss_function(batch)
        errors = self.get_error(batch)
        preds = self.get_pred(batch["data"])

        # TODO: suggest using sample-wise metrics, i.e. each of shape (batch, ...)
        metrics = {"loss": loss.mean(), "error": errors.mean()}

        for metric_name in self.metric_names:
            self.valid_metrics[dataloader_idx][metric_name].update(metrics[metric_name])

        valid_name = cu.get_dataset_name(self.cfg.data.valid, dataloader_idx)

        for metric_name in self.metric_names:
            self.log(
                f"{valid_name}/{metric_name}",
                self.valid_metrics[dataloader_idx][metric_name],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        return {"preds": preds, "errors": errors, "metrics": metrics}

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        pass

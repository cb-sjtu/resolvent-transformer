import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import MeanMetric, MetricCollection

from src.datasets.data_utils import BaseLabelData, OperatorData
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

    def network_inference(self, data: OperatorData):
        outputs = self._model_forward(memory=data.f_samples, query=data.g_inputs)
        return outputs

    def _loss_function(self, data: OperatorData, label: BaseLabelData):
        pred = self.network_inference(data)
        return F.mse_loss(pred, label.label)

    def get_pred(self, data: OperatorData) -> torch.Tensor:
        return self.network_inference(data)

    def get_error(self, data: OperatorData, label: BaseLabelData) -> torch.Tensor:
        pred = self.get_pred(data)
        return torch.abs(pred - label.label)

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics:
            metrics.reset()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        data, label = batch["data"], batch["label"]
        loss = self._loss_function(data, label)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss

    ############ validation #############
    def validation_step(self, batch: OperatorData, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        data, label = batch["data"], batch["label"]
        loss = self._loss_function(data, label)
        error = self.get_error(data, label)

        metrics = {"loss": loss.mean(), "error": error.mean()}

        for metric_name in self.metric_names:
            self.valid_metrics[dataloader_idx][metric_name].update(metrics[metric_name])

        valid_key = list(self.cfg.data.valid.keys())[dataloader_idx]
        valid_name = self.cfg.data.valid[valid_key].name

        for metric_name in self.metric_names:
            self.log(
                f"{valid_name}/{metric_name}",
                self.valid_metrics[dataloader_idx][metric_name],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        return metrics

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        pass

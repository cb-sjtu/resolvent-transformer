import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import MeanMetric, MetricCollection

from src.datasets.data_utils import BaseLabelData, ViconData
from src.models import Vicon
from src.plmodules.base_lit_module import BaseLitModule


class ViconLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # Initialize the Vicon model with the configuration
        self.net = Vicon(cfg.model)

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

    def _prompt_engineering(self, x: torch.Tensor):
        batch_size, seq_len, dim, H, W = x.shape
        x_reshaped = x.view(batch_size * seq_len, dim, H, W)

        mean = x_reshaped.mean(dim=(0, 2, 3), keepdim=True)  # Mean across batch, H, W -> (1, dim, 1, 1)
        std = x_reshaped.std(dim=(0, 2, 3), keepdim=True) + 1e-5  # Std across batch, H, W -> (1, dim, 1, 1)

        x_normalized = (x_reshaped - mean) / std
        x_normalized = x_normalized.view(batch_size, seq_len, dim, H, W)

        return x_normalized, mean, std

    def network_inference(self, data: ViconData):
        dummy_label = torch.zeros_like(data.demo_qoi[:, -1:, :, :, :])
        qoi = data.demo_qoi
        cond = torch.cat((data.demo_cond, data.quest_cond), dim=1)
        cond_norm, cond_mean, cond_std = self._prompt_engineering(cond)
        qoi_norm, qoi_mean, qoi_std = self._prompt_engineering(qoi)
        qoi_norm = torch.cat((qoi_norm, dummy_label), dim=1)

        outputs = self._model_forward(cond_norm, qoi_norm)

        # denormalize the predicted QoI using the mean and std of the QoI
        denormalized_outputs = {}
        for key, tensor in outputs.items():
            batch_size, seq_len, dim, H, W = tensor.shape
            mean_broadcast = qoi_mean.view(1, 1, dim, 1, 1)
            std_broadcast = qoi_std.view(1, 1, dim, 1, 1)
            denormalized_outputs[key] = tensor * std_broadcast + mean_broadcast

        return denormalized_outputs

    def _get_ground_truth_all(self, data: ViconData, label: BaseLabelData):
        qoi = data.demo_qoi
        ground_truth = torch.cat((qoi, label.label), dim=1)
        return ground_truth

    def _get_pred_all(self, outputs: dict):
        demo_pred = outputs["demo_pred"]
        quest_pred = outputs["quest_pred"]
        all_pred = torch.cat([demo_pred, quest_pred], dim=1)
        return all_pred

    def _get_pred_quest(self, outputs: dict):
        quest_pred = outputs["quest_pred"]
        return quest_pred

    def _loss_function(self, pred: torch.Tensor, target: torch.Tensor):
        return F.mse_loss(pred, target)

    def _loss_all(self, batch: dict, short_num_min=1) -> torch.Tensor:
        # used for training
        data = batch["data"]
        label = batch["label"]
        outputs = self.network_inference(data)
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(data, label)
        loss = self._loss_function(all_pred, all_ground_truth)

        return loss

    def _error_all(self, batch: dict, short_num_min=1) -> torch.Tensor:
        data = batch["data"]
        label = batch["label"]
        outputs = self.network_inference(data)
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(data, label)
        error = all_pred - all_ground_truth
        return error

    def _loss_quest(self, batch: dict, short_num_min=1) -> torch.Tensor:
        data = batch["data"]
        label = batch["label"]
        outputs = self.network_inference(data)
        quest_pred = self._get_pred_quest(outputs)
        loss = self._loss_function(quest_pred, label)
        return loss

    def _error_quest(self, batch: dict, short_num_min=1) -> torch.Tensor:
        data = batch["data"]
        label = batch["label"]
        outputs = self.network_inference(data)
        quest_pred = self._get_pred_quest(outputs)
        error = quest_pred - label
        return error

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics:
            metrics.reset()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._loss_all(batch)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss

    ############ validation #############
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self._loss_all(batch)
        error = self._error_all(batch)

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

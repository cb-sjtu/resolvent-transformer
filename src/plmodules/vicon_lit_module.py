import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.custom_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class ViconLitModule(BaseLitModule):
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

    def _prompt_normalization(self, x: torch.Tensor):
        mean = x.mean(dim=(1, 3, 4), keepdim=True)  # Mean across seq, H, W -> (batch_size, 1, dim, 1, 1)
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-5  # Std across seq, H, W -> (batch_size, 1, dim, 1, 1)

        x_normalized = (x - mean) / std

        return x_normalized, mean, std

    def network_inference(self, data: PyTree):
        dummy_label = torch.zeros_like(data["demo_qoi"][:, -1:, :, :, :])
        qoi = data["demo_qoi"]
        cond = torch.cat((data["demo_cond"], data["quest_cond"]), dim=1)
        cond_norm, cond_mean, cond_std = self._prompt_normalization(cond)
        qoi_norm, qoi_mean, qoi_std = self._prompt_normalization(qoi)
        qoi_norm = torch.cat((qoi_norm, dummy_label), dim=1)

        outputs = self._model_forward(cond_norm, qoi_norm)
        # denormalize the predicted QoI using the mean and std of the QoI
        denormalized_outputs = {}
        for key, tensor in outputs.items():
            denormalized_outputs[key] = tensor * qoi_std + qoi_mean

        return denormalized_outputs

    def _get_ground_truth_all(self, batch: PyTree):
        qoi = batch["data"]["demo_qoi"]
        ground_truth = torch.cat((qoi, batch["label"]), dim=1)
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

    def _loss_all(self, batch: PyTree) -> torch.Tensor:
        # used for training
        outputs = self.network_inference(batch["data"])
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(batch)
        loss = self._loss_function(all_pred, all_ground_truth)

        return loss

    def _error_all(self, batch: PyTree) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.network_inference(batch["data"])
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(batch)
        error = all_pred - all_ground_truth
        return all_pred, error

    def _loss_quest(self, batch: PyTree) -> torch.Tensor:
        data = batch["data"]
        label = batch["label"]
        outputs = self.network_inference(data)
        quest_pred = self._get_pred_quest(outputs)
        loss = self._loss_function(quest_pred, label)
        return loss

    def _error_quest(self, batch: PyTree) -> torch.Tensor:
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
        preds, errors = self._error_all(batch)

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

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        pass

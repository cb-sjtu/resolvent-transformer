import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchmetrics import MeanMetric, MetricCollection

from src.data.datasets.dummy_icon import IconData
from src.models.components.vicon import Vicon
from src.opt import WarmupCosineDecayScheduler


class IconLitModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # Initialize the Vicon model with the configuration
        self.net = Vicon(cfg.model)

        sdpa_map = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
        }

        self.sdpa_backends = [sdpa_map[backend] for backend in self.cfg.sdpa]

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

    def _model_forward(self, cond_features, qoi_features):
        with sdpa_kernel(self.sdpa_backends):
            model_input = IconData(cond_features=cond_features, qoi_features=qoi_features)

            outputs = self.net(model_input)

            return outputs  # dict: demo_pred, quest_pred

    def network_inference(self, data):
        if isinstance(data, IconData):
            dummy_label = torch.zeros_like(data.qoi_features[:, -1, :, :, :])
            qoi_features = torch.cat((data.qoi_features, dummy_label), dim=1)

        # add prompt engineering here
        outputs = self._model_forward(data.cond_features, qoi_features)
        return outputs

    def _get_ground_truth_all(self, data: IconData, label):
        qoi_features = data.qoi_features
        qoi_features[:, -1, :, :, :] = label
        return qoi_features

    def _get_pred_all(self, outputs: dict):
        demo_pred = outputs["demo_pred"]
        quest_pred = outputs["quest_pred"]
        all_pred = torch.cat([demo_pred, quest_pred], dim=1)
        return all_pred

    def _get_pred_quest(self, outputs: dict):
        quest_pred = outputs["quest_pred"]
        return quest_pred

    def _loss_function(self, pred, target):
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

    def restore_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, weights_only=False)
        # print(ckpt['state_dict'].keys())
        state_dict = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}  # 移除键中前缀 'net.'
        self.net.load_state_dict(state_dict)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit" and torch.__version__ >= "2.0.0":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=float(self.cfg.opt.peak_lr),
            weight_decay=float(self.cfg.opt.weight_decay),
        )

        scheduler = WarmupCosineDecayScheduler(
            optimizer=optimizer,
            warmup=int(self.cfg.opt.warmup_percent * self.cfg.trainer.max_steps // 100),
            max_iters=int(self.cfg.opt.decay_percent * self.cfg.trainer.max_steps // 100),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

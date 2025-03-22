import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchmetrics import MeanMetric, MetricCollection

from src.operator_model.data import OperatorData
from src.operator_model.model import OperatorTransformer
from src.opt import WarmupCosineDecayScheduler


class OperatorLitModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # self.net = hydra.utils.instantiate(cfg.model)

        self.net = OperatorTransformer(cfg=cfg)

        sdpa_map = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
        }

        self.sdpa_backends = [sdpa_map[backend] for backend in self.cfg.sdpa]

        self.train_metrics = MeanMetric()

        valid_data_count = max(1, len(self.cfg.data.valid) if hasattr(self.cfg.data, "valid") else 1)

        # Use MetricCollection to group metrics
        self.valid_metrics = torch.nn.ModuleList(
            [
                MetricCollection(
                    {
                        "loss": MeanMetric(),
                        "error": MeanMetric(),
                    }
                )
                for _ in range(valid_data_count)
            ]
        )

    def _model_forward(self, f_samples, g_inputs):
        with sdpa_kernel(self.sdpa_backends):
            return self.net(f_samples, g_inputs)

    def _loss_function(self, pred, target):
        return F.mse_loss(pred, target)

    def _loss_operator(self, batch: OperatorData) -> torch.Tensor:
        f_samples = batch.f_samples
        g_inputs = batch.g_inputs
        g_targets = batch.g_targets

        g_outputs = self._model_forward(f_samples, g_inputs)

        loss = self._loss_function(g_outputs, g_targets)

        return loss

    def get_pred(self, batch: OperatorData) -> torch.Tensor:
        f_samples = batch.f_samples
        g_inputs = batch.g_inputs

        return self._model_forward(f_samples, g_inputs)

    def get_error(self, batch: OperatorData) -> torch.Tensor:
        g_outputs = self.get_pred(batch)

        g_targets = batch.g_targets

        return torch.abs(g_outputs - g_targets)

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics:
            metrics.reset()

    def training_step(self, batch: OperatorData, batch_idx: int) -> torch.Tensor:
        loss = self._loss_operator(batch)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss

    ############ validation #############
    def validation_step(self, batch: OperatorData, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self._loss_operator(batch)
        error = self.get_error(batch)

        idx = min(dataloader_idx, len(self.valid_metrics) - 1)
        self.valid_metrics[idx]["loss"].update(loss.mean().item())
        self.valid_metrics[idx]["error"].update(error.mean().item())

        prefix = "valid"
        if hasattr(self.cfg.data, "valid") and self.cfg.data.valid:
            valid_keys = list(self.cfg.data.valid.keys())
            if valid_keys and dataloader_idx < len(valid_keys):
                prefix = f"valid_{valid_keys[dataloader_idx]}"

        self.log(f"{prefix}/loss", self.valid_metrics[idx]["loss"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/error", self.valid_metrics[idx]["error"], on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self._loss_operator(batch)
        error = self.get_error(batch)

        self.log("test/loss", loss.mean(), on_step=False, on_epoch=True)
        self.log("test/error", error.mean(), on_step=False, on_epoch=True)

        return loss

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

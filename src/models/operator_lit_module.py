import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchmetrics import MeanMetric, MetricCollection

from src.data.data_utils import BaseLabelData, OperatorData
from src.models.components.encoder_decoder import EncoderDecoder
from src.opt import WarmupCosineDecayScheduler


class OperatorLitModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # self.net = hydra.utils.instantiate(cfg.model)

        self.net = EncoderDecoder(cfg=cfg)

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

    def _model_forward(self, *args, **kwargs):
        with sdpa_kernel(self.sdpa_backends):
            return self.net(*args, **kwargs)

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

    def restore_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, weights_only=False)
        # print(ckpt['state_dict'].keys())
        state_dict = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
        self.net.load_state_dict(state_dict)

    def setup(self, stage: str) -> None:
        if self.cfg.model.compile and stage == "fit" and torch.__version__ >= "2.0.0":
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

import lightning as L
import torch
from omegaconf import DictConfig
from torch import optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchmetrics import MeanMetric, MetricCollection

import src.data.data_utils as du
from src.models.components import ICE_EncoderDecoder
from src.opt import WarmupCosineDecayScheduler


class IceLitModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # you can also use hydra to instantiate the model
        # self.net = hydra.utils.instantiate(cfg.model)
        self.net = ICE_EncoderDecoder(cfg)

        sdpa_map = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
        }

        self.sdpa_backends = [sdpa_map[backend] for backend in self.cfg.sdpa]

        self.train_metrics = MeanMetric()

        # Use MetricCollection to group metrics
        self.valid_metrics = torch.nn.ModuleList(
            [
                MetricCollection(
                    {
                        "loss": MeanMetric(),
                        "error": MeanMetric(),
                    }
                )
                for _ in range(len(self.cfg.data.valid))  # initialize metrics for each valid_loader
            ]
        )

    def _model_forward(self, *args, **kwargs):
        """
        simple model forward wrapped with sdpa_kernel
        """
        with sdpa_kernel(self.sdpa_backends):
            return self.net(*args, **kwargs)

    def _loss_eqn(self, data: du.DataEqn) -> torch.Tensor:
        """
        get the loss
        """
        pass

    def _loss_meshlist(self, data: du.DataMeshList) -> torch.Tensor:
        """
        get the loss
        """
        pass

    def get_pred(self, data: du.DataMeshList):
        """
        get the prediction
        """
        pass

    def get_error(self, data: du.DataMeshList) -> tuple[torch.Tensor, ...]:
        """
        get the error collection, including
        """
        pass

    ############ training #############

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metrics in self.valid_metrics:
            metrics.reset()

    def training_step(self, batch: du.DataEqn | du.DataMeshList, batch_idx) -> torch.Tensor:
        if isinstance(batch, du.DataMeshList):
            loss = self._loss_meshlist(batch)
        elif isinstance(batch, du.DataEqn):
            loss = self._loss_eqn(batch)
        else:
            raise ValueError(f"Invalid batch type: {type(batch)}")

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=False)
        return loss

    ############ validation #############
    def validation_step(self, batch: du.DataMeshList, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if isinstance(batch, du.DataMeshList):
            return self.validation_step_meshlist(batch, batch_idx, dataloader_idx)
        else:
            raise ValueError(f"Invalid batch type: {type(batch)}")

    def validation_step_meshlist(self, batch: du.DataMeshList, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self._loss_meshlist(batch)
        error = self.get_error(batch)

        self.valid_metrics[dataloader_idx]["loss"].update(loss.mean().item())
        self.valid_metrics[dataloader_idx]["error"].update(error.mean().item())

        valid_name = list(self.cfg.data.valid.keys())[dataloader_idx]

        self.log(f"valid_{valid_name}/loss", self.valid_metrics[dataloader_idx]["loss"], on_step=False, on_epoch=True)
        self.log(f"valid_{valid_name}/error", self.valid_metrics[dataloader_idx]["error"], on_step=False, on_epoch=True)

        return loss

    def restore_ckpt(self, ckpt_path: str) -> None:
        """
        restore the model from a checkpoint
        """
        ckpt = torch.load(ckpt_path, weights_only=False)
        # print(ckpt['state_dict'].keys())
        state_dict = {k[4:]: v for k, v in ckpt["state_dict"].items()}  # remove leading 'net.' in the key
        self.net.load_state_dict(state_dict)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.cfg.opt.peak_lr,
            weight_decay=self.cfg.opt.weight_decay,
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

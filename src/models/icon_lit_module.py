import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchmetrics import MeanMetric, MetricCollection

from src.data.datasets.dummy_icon import IconData, IconDataset
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

    def _model_forward(self, demo_cond_features, demo_qoi_features, quest_cond_features):
        with sdpa_kernel(self.sdpa_backends):
            # Prepare data for Vicon model
            bs = demo_cond_features.shape[0]

            # quest_cond = quest_cond_features.squeeze(1) if quest_cond_features.shape[1] == 1 else quest_cond_features

            # Create the demo pairs (cond, qoi)
            # demo_pairs = demo_cond_features.shape[1]  # Number of demonstration pairs
            c_mask = torch.ones(bs, 1, device=demo_cond_features.device)  # Condition mask, all active

            # Format input as expected by Vicon model: (init, end, c_mask)
            # Where init contains the condition features and end contains the QoI features
            model_input = (demo_cond_features, demo_qoi_features, c_mask)

            # Call the model
            predicted_qoi = self.net(model_input)  # (bs, pairs, 3, 128, 128)

            # Return the predicted QoI for the query condition (taking the last prediction)
            # Shape will be (bs, 1, 3, 128, 128)
            return predicted_qoi[:, -1:, :, :, :]

    def _loss_function(self, pred, target):
        return F.mse_loss(pred, target)

    def _loss_vicon(self, batch) -> torch.Tensor:
        """Calculate loss for Vicon data.

        Args:
            batch: Either a dict with tensors, a ViconData instance, or a ViconDataset

        Returns:
            Loss tensor
        """
        if isinstance(batch, dict):
            demo_cond_features = batch["demo_cond_features"]
            demo_qoi_features = batch["demo_qoi_features"]
            quest_cond_features = batch["quest_cond_features"]
            quest_qoi_features = batch["quest_qoi_features"]
        elif isinstance(batch, IconData):
            demo_cond_features = batch.demo_cond_features
            demo_qoi_features = batch.demo_qoi_features
            quest_cond_features = batch.quest_cond_features
            quest_qoi_features = batch.quest_qoi_features
        elif isinstance(batch, IconDataset):
            # Concatenate data from all samples in the batch
            demo_cond_features = torch.cat([sample.demo_cond_features for sample in batch.samples], dim=0)
            demo_qoi_features = torch.cat([sample.demo_qoi_features for sample in batch.samples], dim=0)
            quest_cond_features = torch.cat([sample.quest_cond_features for sample in batch.samples], dim=0)
            quest_qoi_features = torch.cat([sample.quest_qoi_features for sample in batch.samples], dim=0)
        else:
            raise ValueError(f"unsupported batch type: {type(batch)}")

        # Forward pass through the model
        predicted_qoi = self._model_forward(
            demo_cond_features=demo_cond_features,
            demo_qoi_features=demo_qoi_features,
            quest_cond_features=quest_cond_features,
        )

        # Calculate loss
        loss = self._loss_function(predicted_qoi, quest_qoi_features)

        return loss

    def get_pred(self, batch):
        """Get model predictions for a batch.

        Args:
            batch: Either a dict with tensors, a ViconData instance, or a ViconDataset

        Returns:
            Model predictions
        """
        if isinstance(batch, dict):
            demo_cond_features = batch["demo_cond_features"]
            demo_qoi_features = batch["demo_qoi_features"]
            quest_cond_features = batch["quest_cond_features"]
        elif isinstance(batch, IconData):
            demo_cond_features = batch.demo_cond_features
            demo_qoi_features = batch.demo_qoi_features
            quest_cond_features = batch.quest_cond_features
        elif isinstance(batch, IconDataset):
            # Concatenate data from all samples in the batch
            demo_cond_features = torch.cat([sample.demo_cond_features for sample in batch.samples], dim=0)
            demo_qoi_features = torch.cat([sample.demo_qoi_features for sample in batch.samples], dim=0)
            quest_cond_features = torch.cat([sample.quest_cond_features for sample in batch.samples], dim=0)
        else:
            raise ValueError(f"unsupported batch type: {type(batch)}")

        return self._model_forward(
            demo_cond_features=demo_cond_features,
            demo_qoi_features=demo_qoi_features,
            quest_cond_features=quest_cond_features,
        )

    def get_error(self, batch) -> torch.Tensor:
        """Calculate absolute error between predictions and targets.

        Args:
            batch: Either a dict with tensors, a ViconData instance, or a ViconDataset

        Returns:
            Absolute error tensor
        """
        # Get model predictions
        predicted_qoi = self.get_pred(batch)

        # Get target values
        if isinstance(batch, dict):
            quest_qoi_features = batch["quest_qoi_features"]
        elif isinstance(batch, IconData):
            quest_qoi_features = batch.quest_qoi_features
        elif isinstance(batch, IconDataset):
            quest_qoi_features = torch.cat([sample.quest_qoi_features for sample in batch.samples], dim=0)
        else:
            raise ValueError(f"unsupported batch type: {type(batch)}")

        return torch.abs(predicted_qoi - quest_qoi_features)

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics:
            metrics.reset()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._loss_vicon(batch)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss

    ############ validation #############
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self._loss_vicon(batch)
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
        loss = self._loss_vicon(batch)
        error = self.get_error(batch)

        self.log("test/loss", loss.mean(), on_step=False, on_epoch=True)
        self.log("test/error", error.mean(), on_step=False, on_epoch=True)

        return loss

    def restore_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, weights_only=False)
        state_dict = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}  # Remove 'net.' prefix
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

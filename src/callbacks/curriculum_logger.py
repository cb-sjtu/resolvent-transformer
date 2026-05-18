"""Callback to log curriculum learning parameters to wandb."""

import lightning as L


class CurriculumLogger(L.Callback):
    """Logs curriculum learning parameters (teacher forcing ratio, k-steps) to wandb during training."""

    def __init__(self):
        super().__init__()

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int
    ):
        """Log curriculum parameters at the start of each training batch."""

        # Check if the module has curriculum learning capabilities
        if not (
            hasattr(pl_module, "get_teacher_forcing_ratio")
            and hasattr(pl_module, "get_current_k_steps")
        ):
            return

        # Get current epoch
        current_epoch = pl_module.current_epoch

        # Get curriculum parameters
        teacher_forcing_ratio = pl_module.get_teacher_forcing_ratio(current_epoch)
        k_steps = pl_module.get_current_k_steps(current_epoch)

        # Log to wandb via the module's logger
        if trainer.logger is not None and hasattr(trainer.logger, "log_metrics"):
            # Log curriculum parameters
            metrics = {
                "curriculum/teacher_forcing_ratio": teacher_forcing_ratio,
                "curriculum/k_steps": float(k_steps),
                "curriculum/epoch": float(current_epoch),
                "curriculum/global_step": float(trainer.global_step),
            }

            # Also log scheduled sampling configuration info (only once per epoch)
            if batch_idx == 0:
                if hasattr(pl_module, "ss_enabled") and pl_module.ss_enabled:
                    metrics.update(
                        {
                            "curriculum/ss_schedule_type": pl_module.ss_schedule_type,
                            "curriculum/ss_decay_epochs": float(
                                pl_module.ss_decay_epochs
                            ),
                            "curriculum/ss_initial_ratio": pl_module.ss_initial_ratio,
                            "curriculum/ss_final_ratio": pl_module.ss_final_ratio,
                        }
                    )

                if hasattr(pl_module, "kr_enabled") and pl_module.kr_enabled:
                    metrics.update(
                        {
                            "curriculum/kr_max_k": float(pl_module.kr_max_k),
                            "curriculum/kr_rollout_weight": pl_module.kr_rollout_weight,
                            "curriculum/kr_single_weight": pl_module.kr_single_weight,
                        }
                    )

            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log curriculum parameters at the start of each epoch."""

        # Check if the module has curriculum learning capabilities
        if not (
            hasattr(pl_module, "get_teacher_forcing_ratio")
            and hasattr(pl_module, "get_current_k_steps")
        ):
            return

        current_epoch = pl_module.current_epoch

        # Get curriculum parameters
        teacher_forcing_ratio = pl_module.get_teacher_forcing_ratio(current_epoch)
        k_steps = pl_module.get_current_k_steps(current_epoch)

        # Log epoch-level curriculum info
        if trainer.logger is not None and hasattr(trainer.logger, "log_metrics"):
            metrics = {
                "curriculum_epoch/teacher_forcing_ratio": teacher_forcing_ratio,
                "curriculum_epoch/k_steps": float(k_steps),
                "curriculum_epoch/epoch": float(current_epoch),
            }

            trainer.logger.log_metrics(metrics, step=trainer.global_step)

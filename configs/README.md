# Configuration Files

This directory contains all configuration files for the project, organized using Hydra framework.

## Structure

- `accelerate/` - Configuration for distributed training acceleration
- `callbacks/` - Lightning callbacks configuration (e.g., model summary, checkpointing)
- `data/` - Dataset and dataloader configurations
- `datamodule/` - PyTorch Lightning datamodule configurations
- `experiment/` - Experiment-specific configurations aiming to overwrite only specified parameters
- `extras/` - Additional utility configurations
- `hydra/` - Hydra framework specific configurations
- `logger/` - Logging configurations (e.g., TensorBoard, WandB)
- `loss/` - Loss function related configurations
- `model/` - Model configurations
- `opt/` - Optimizer configurations
- `paths/` - Path configurations
- `plmodule/` - PyTorch Lightning module configurations
- `trainer/` - Trainer configurations
...
...
- `train_nop.yaml` - Training configuration for Neural Operator models
- `train_nop_rollout.yaml` - Training configuration for NOP models with rollout
- `train_vicon.yaml` - Training configuration for Vicon models
...

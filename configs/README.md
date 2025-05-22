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
- `train_custom.yaml` - Training configuration for machine-specific paths
...

## Configuration logic

The order of configuration overrides is determined by the `defaults` list in the main configuration files (e.g. `train_nop.yaml`, `train_nop_rollout.yaml`). Later configurations override earlier ones. The typical order is:

1. Base configuration (`_self_`)
2. Data configuration
3. Model configuration
...
...

Each item in the `defaults` list is 'key: value' pair to configure the corresponding component, for easier understanding, meaning that this item is instantiated from the class/function path specified by `_target_` from configs/key/value.yaml.

### Configuration Inheritance

The configuration system uses a hierarchical structure. For example:

1. In main config (e.g., `train_nop.yaml`), you might have:
```yaml
defaults:
  - callbacks: default  # This points to configs/callbacks/default.yaml
```

2. Then in `configs/callbacks/default.yaml`, you might have:
```yaml
defaults:
  - model_summary  # This points to configs/callbacks/model_summary.yaml
  - checkpoint    # This points to configs/callbacks/checkpoint.yaml
```


In each specific yaml file, the common structure is:

```yaml
name:
  _target_: <xxx_path.YourClassName>  # Fully qualified class/function path
  <arg1>: <value1>                    # Configuration parameter 1
  <arg2>: <value2>                    # Configuration parameter 2
  ...
```

If you have `train_custom.yaml`, it will override the log and data paths

## Best Practices

1. Keep core configurations untouched when forking the repository
2. Use experiment configs for version control of hyperparameters
3. Use `train_custom.yaml` for machine-specific path configurations

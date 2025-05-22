# Project Pipeline

This document describes the workflow for adding new components to the project.

## Adding New Components

### 1. Model and Lightning Module
To add a new model, you typically need to:
1. Create your model implementation in `src/models/your_model_name/`
2. Create your Lightning Module in `src/plmodules/your_plmodule_name.py`

For detailed model structure, please refer to `src/models/README.md`.

### 2. Dataset and DataModule
If you need to add or modify datasets:
1. Add your dataset implementation in `src/datasets/`. You need to return pytrees as the data structure in this project.
See `src/datasets/README.md` for more details.
2. Add your DataModule in `src/datamodules/` (In most cases, you don't need to modify this. base_datamodule.py is enough.)

### 3. Callbacks
To add custom callbacks:
1. Add your callback implementation in `src/callbacks/your_callback_name.py`

For detailed callback structure, please refer to `src/callbacks/README.md`.

## Configuration

After implementing your components, you need to:

1. Create a new configuration file in `configs/train_your_model_name.yaml`. For the first time, you can mimic the format of `configs/train_nop_rollout.yaml`
2. Modify the `defaults` list to include your components
3. Add or modify the corresponding configuration files in the appropriate directories:
   - `configs/model/` for model configurations
   - `configs/plmodule/` for Lightning Module configurations
   - `configs/datamodule/` for DataModule configurations
   - `configs/callbacks/` for callback configurations

For detailed configuration structure and inheritance, please refer to `configs/README.md`.

## Training Pipeline

The main training script `src/train.py` typically doesn't need modification. It handles:

1. **Environment Setup**:
   - Project root directory setup via rootutils
   - Python path configuration
   - Environment variables loading

2. **PyTorch Configuration**:
   - Dynamo cache size limit setting
   - Float32 matrix multiplication precision configuration
   - TF32 settings for CUDA and cuDNN
   - Random seed initialization

3. **Component Instantiation**:
   - DataModule instantiation with full config
   - Model (LightningModule) instantiation with full config
   - Callbacks instantiation
   - Loggers instantiation
   - Trainer instantiation

4. **Training Process**:
   - Hyperparameter logging
   - Model training with checkpoint support
   - Model testing
   - Metric collection and merging

5. **Hydra Integration**:
   - Configuration management
   - Custom config file support (train_custom.yaml)
   - Metric optimization for hyperparameter tuning

The script is designed to be a core component that should remain unchanged in most cases, as it provides a standardized training pipeline for all models in the project.

## Example Workflow

1. **Adding a New Model**:
   ```python
   # src/models/your_model/your_model.py
   class YourModel(nn.Module):
       def __init__(self, **kwargs):
           super().__init__()
           # Your model implementation
   ```

2. **Creating a Lightning Module**:
   ```python
   # src/plmodules/your_plmodule.py
   class YourLitModule(BaseLitModule):
       def __init__(self, cfg: DictConfig):
           super().__init__(cfg)
           # Your module implementation
   ```

3. **Configuration Setup**:
   ```yaml
   # configs/train_your_model.yaml
   defaults:
     - _self_
     - model: your_model
     - plmodule: your_plmodule
     - datamodule: your_datamodule
     - callbacks: your_callbacks
     - ...
   ```

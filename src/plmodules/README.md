# PyTorch Lightning Modules

This directory contains PyTorch Lightning module implementations for the project. These modules handle the training, validation, and testing logic for different types of models.

## Structure

- `base_lit_module.py` - Base Lightning module with common functionality
- `nop_lit_module.py` - Neural Operator module
- `nop_rollout_lit_module.py` - Neural Operator module with rollout capability
- `vicon_lit_module.py` - Vicon-based module
- Your own module.py

## BaseLitModule

The `BaseLitModule` class extends PyTorch Lightning's `LightningModule` and provides core functionality:

- Model instantiation from config
- Optimizer and scheduler configuration
- Model compilation support
- SDPA (Scaled Dot-Product Attention) backend configuration

## Creating a New Lightning Module

When creating a new Lightning Module, you are suggested to inherit from `BaseLitModule` and override several methods.

If you have complicated logic (for example, multiple forward passes, mutiple optimizers, etc.), you are suggested to directly inherit from `LightningModule` and override the methods you need.

### Module Structure (if you inherit from `BaseLitModule`)

```python
class YourLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        # 1. Initialize your model
        # 2. Set up metrics
        # 3. Define any additional attributes
```

#### Model Initialization
The model is automatically initialized in `BaseLitModule` using:
```python
self.net = hydra.utils.instantiate(cfg.model)
```

#### Metrics Setup
```python
# Training metrics
self.train_metrics = MetricCollection({
    "loss": MeanMetric(),
    # Add your custom metrics
})

# Validation/Test metrics
self.metric_names = [
    "your_metric_1",
    "your_metric_2",
]
self.valid_metrics = torch.nn.ModuleList([
    MetricCollection({k: MeanMetric() for k in self.metric_names})
    for _ in range(len(self.cfg.data.valid))
])
```

#### Training Logic
```python
def training_step(self, batch: PyTree, batch_idx: int) -> torch.Tensor:
    # 1. Forward pass
    # 2. Calculate loss
    # 3. Update metrics
    # 4. Log results
    loss = self._loss_function(batch)
    self.train_metrics["loss"](loss)
    self.log("train/loss", self.train_metrics["loss"], on_step=True)
    return loss.mean()
```

- `on_step=True`: Logs the metric value immediately after each training step (batch completion).

#### Evaluation Logic
```python
def eval_step(self, batch: PyTree, batch_idx: int, stage: Literal["valid", "test"], dataloader_idx: int = 0):
    # 1. Get predictions
    # 2. Calculate errors
    # 3. Compute metrics
    # 4. Log results
    preds = self._get_predictions(batch)
    errors = self._calculate_errors(preds, batch)
    metrics = self._compute_metrics(errors)

    # Log metrics
    for metric_name in eval_metrics[dataloader_idx]:
        self.log(f"{dataset_name}/{metric_name}",
                eval_metrics[dataloader_idx][metric_name],
                on_step=False, on_epoch=True)

    return {"preds": preds, "errors": errors, "metrics": metrics}
```

- `on_epoch=True`: Accumulates values across all steps in an epoch (default reduction via `torch.stack` + mean), then logs at epoch end.

### Optional Components

#### Custom Processing Steps
If your model requires special processing, implement these methods:
```python
def _preprocess(self, data: PyTree) -> PyTree:
    """Custom preprocessing logic"""
    return data

def _postprocess(self, output: PyTree) -> PyTree:
    """Custom postprocessing logic"""
    return output
```

#### Custom Loss Function
```python
def _loss_function(self, batch: PyTree) -> torch.Tensor:
    """Custom loss calculation"""
    pred = self._model_forward(batch["data"])
    return your_loss_function(pred, batch["label"])
```

#### Custom Optimizer Configuration
```python
def configure_optimizers(self):
    """Override if you need custom optimizer setup"""
    return super().configure_optimizers()
```

## Return of validation_step and test_step

`validation_step` and `test_step` should return a pytree containing these top-level keys:
- `preds`: A pytree containing the model's predictions
- `errors`: A pytree containing detailed prediction errors (e.g., image format errors)
- `metrics`: A pytree containing evaluation metrics. It is strongly suggested that each metric maintain the batch dimension with shape `(batch_size, xxx)`, for example, `(batch_size, 2, 2)`. While `xxx` can have multiple dimensions, the flattened size should remain reasonably small.

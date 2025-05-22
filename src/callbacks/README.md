# Callbacks

This directory contains PyTorch Lightning callbacks for the project.

For more details, please refer to the [PyTorch Lightning callbacks documentation](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html).

## Creating a New Callback

When creating a new callback, you need to inherit from `lightning.Callback` and implement the necessary hooks.

If you want to add visualization, you can inherit from `Viz` in `viz.py` and override the `get_image` method.

### Basic Structure

```python
from lightning import Callback

class YourCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize your callback parameters

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Your validation batch end logic

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Your test batch end logic
```

### Other Available Hooks

Common hooks you might want to implement:

```python
def on_train_start(self, trainer, pl_module):
    """Called when training starts"""
    pass

def on_train_end(self, trainer, pl_module):
    """Called when training ends"""
    pass

def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
    """Called before validation batch"""
    pass

def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    """Called after validation batch"""
    pass

def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
    """Called before test batch"""
    pass

def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    """Called after test batch"""
    pass
```

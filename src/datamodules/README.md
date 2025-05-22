# DataModules

This directory contains the PyTorch Lightning DataModule implementations for the project. The DataModule is responsible for handling data loading, preparation, and management in a standardized way.

## Structure

- `base_datamodule.py`
- `dataloader_utils.py`
-  your own datamodule.py

## BaseDataModule

The `BaseDataModule` class extends PyTorch Lightning's `LightningDataModule` and provides a standardized way to handle data loading and preparation.

### Customization

You can customize the DataModule by overriding these methods:
- `get_train_dataset_from_cfg()`
- `get_valid_test_dataset_from_cfg()`
- `get_train_collate_fn()`
- `get_valid_test_collate_fn()`

## Dataloader Utilities

The `dataloader_utils.py` file provides utility functions and classes for data loading:

- `CycleLoader`: A utility class that cycles through multiple DataLoaders
...

# Guideline

This document describes the workflow for adding new components to the project.

## Reusing Existing Components

This repository already contains a lot of components, including models, lightning modules, datasets, callbacks, etc. You should try to leverage them by reusing or inheriting from them in your project. Additionally, they serve as practical implementation references for your work.

## Adding New Components

### Model and Lightning Module
- Create new model architectures in `src/models/your_model_folder/your_model_file.py`.
- Create Lightning Module for new training and evaluation pipelines in `src/plmodules/your_plmodule_file.py`

Please refer to `src/models/README.md` and `src/plmodules/README.md` for more details.

### Dataset and DataModule
- Create new torch datasets in `src/datasets/your_dataset_folder/your_dataset_file.py`.
- Create new lightning data modules and dataloaders in `src/datamodules/your_datamodule_file.py`.

Please refer to `src/datasets/README.md` and `src/datamodules/README.md` for more details.

### Callbacks
- Create new callbacks in `src/callbacks/your_callback_name.py`.

Please refer to `src/callbacks/README.md` for more details.

## Configuration

After implementing your components, you need to:

1. Add configuration files in the subdirectories of `configs/`.
2. Create a new configuration file `configs/train_project_name.yaml`. For reference, you can mimic the format of `configs/train_nop_rollout.yaml`, and replace the configs with ones you need.

Please refer to `configs/README.md` for more details.

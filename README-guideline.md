# Guideline

This document describes the workflow for adding new components to the project.

## Reusing Existing Components

This repository already contains a lot of components, including models, lightning modules, datasets, callbacks, etc. You should try to leverage them by reusing or inheriting from them in your project. Additionally, they serve as practical implementation references. Read them before implementing your own.

Do not modify files inherited from the core repository. These files start with a header in the beginning of the file:

```python
#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################
```

Other files are not part of the core repository and are free to modify. Do not add the header in your newly created files, so other developers can easily identify them.

## Implementing New Components
When implementing new components, you generally need to implement both .py files and corresponding .yaml configuration files.

### Model

- Create new model architectures in `src/models/your_model_folder/your_model_file.py`.
- Create corresponding configuration files in `configs/model/`.

Please refer to `src/models/README.md` for more details.

### Lightning Module

- Create new lightning modules for training and evaluation in `src/plmodules/your_plmodule_file.py`.
- Create corresponding configuration files in `configs/plmodule/`.

Please refer to `src/plmodules/README.md` for more details.

### Dataset

- Create new torch datasets in `src/datasets/your_dataset_folder/your_dataset_file.py`.
- Create corresponding configuration files in `configs/data/your_dataset_folder/`. In this folder, create `train/`, `valid/`, and `test/` subfolders. For each training dataset, you should create a corresponding yaml file in `train/` subfolder. You can have multiple yaml files for multiple training datasets. Similarly for validation and testing datasets. In the end, you should also create a main dataset yaml file `configs/data/your_dataset_folder/your_dataset.yaml` to list all training, validation, and testing datasets to be used in the project.

Please refer to `src/datasets/README.md` for more details.

### DataModule

- Create new lightning data modules and dataloaders in `src/datamodules/your_datamodule_file.py`.
- Create corresponding configuration files in `configs/datamodule/`.

### Callbacks

- Create new callbacks in `src/callbacks/your_callback_name.py`.
- Create corresponding configuration files in `configs/callback/`. You need to create a new yaml file `single_callback_name.yaml` for each new callback, and another yaml file `many_callbacks_project_name.yaml` to list all callbacks to be used in the project. Please refer to `configs/callbacks/README.md` for more details.

### Main Configuration

- Create a new configuration file `configs/train_project_name.yaml` as the main configuration file for the project. This file will indicate what configurations to be used in the project, including models, datasets, callbacks, etc.

Please refer to `configs/README.md` for more details.

### Running Scripts

- Create new scripts in `scripts_project_name/` directory. Please see the scripts in `scripts_core/` as references.

## Example Workflow

Let's take the `nop_rollout` (neural operator with rollout) project as an example. This project demonstrates how to implement a complete neural operator training pipeline with rollout validation.

### Model

We use 1D FNO model, which is implemented in `src/models/nop/fno.py`. The corresponding configuration is in `configs/model/fno1d.yaml`.

### Lightning Module

The training and validation logic is implemented in `src/plmodules/nop_rollout_lit_module.py`, which inherits from `BaseLitModule`. The configuration is in `configs/plmodule/nop_rollout.yaml`.

### Dataset

We use the Kuramoto-Shivashinsky (KS) equation simulation dataset. The dataset is implemented in `src/datasets/ks/ks.py`. The corresponding configurations are in `configs/data/ks/` folder. Note that we put the training, validation, and testing configurations in the `train/`, `valid/`, and `test/` subfolders, and create a main dataset yaml file `configs/data/ks/ks.yaml` to list all training, validation, and testing datasets to be used in the project.

In this project, we use two validation datasets: `ks_short` and `ks_long`, for validating the model's performance on short-term and long-term predictions, respectively. Therefore we have two corresponding validation configurations in `configs/data/ks/valid/` folder, and listed them in the main dataset yaml file `configs/data/ks/ks.yaml`.

### Callbacks

We implement two callbacks in `src/callbacks/viz_rollout_error.py ` and `src/callbacks/viz_rollout_1d.py` to visualize the rollout error and the rollout trajectory, respectively. Notably, we inherit from the `Viz` class in `src/callbacks/viz.py` for visualization.

We created two corresponding configuration files in `configs/callbacks/` folder: `viz_rollout_error.yaml` and `viz_rollout_1d.yaml`, and listed them in the main callback yaml file `configs/callbacks/many_callbacks_nop_rollout.yaml`, together with other existing callback configurations.

### Main Configuration

The main configuration file is `configs/train_nop_rollout.yaml`.

### Running Scripts

We didn't implement the running script, but here are two simple examples:

- CPU short training for debugging
```bash
#!/bin/sh
uv sync --extra cpu
export TORCH_COMPILE_DISABLE=1
uv run python src/train.py --config-name=train_nop_rollout trainer=cpu trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5
echo "Done"
```

- GPU full training
```bash
uv sync --extra cu126
uv run python src/train.py --config-name=train_nop_rollout
echo "Done"
```

## Testing

### Unit Tests

You can add unit tests code in the same file as the code to be tested, under the `if __name__ == "__main__":` block. Alternatively, you can add individual test files `test_your_component.py` for complicated tests, but it is suggested to keep the test code in the same folder as the code to be tested.

You may have difficulty in importing local modules. An easy way is to run the test code in the root directory as follows:
```bash
uv sync --extra cpu # for cpu tests
# uv sync --extra cu126 # for gpu tests, see other options in README-icon-core.md
uv run python -m src.subfolder.test_your_component
```
Here `uv run` ensures running in the virtual environment.

### End-to-End Tests

Refer to `README-icon-core.md` and the scripts in `scripts_core/` for how to run code end-to-end.

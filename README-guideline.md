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

## Adding New Components

### Model and Lightning Module

- Create new model architectures in `src/models/your_model_folder/your_model_file.py`.
- Create Lightning Module for new training and evaluation pipelines in `src/plmodules/your_plmodule_file.py`

Please refer to `src/models/README.md` and `src/plmodules/README.md` for more details.

### Dataset and DataModule

- Create new torch datasets in `src/datasets/your_dataset_folder/your_dataset_file.py`.
- Create new lightning data modules and dataloaders in `src/datamodules/your_datamodule_file.py`.

Please refer to `src/datasets/README.md` for more details.

### Callbacks

- Create new callbacks in `src/callbacks/your_callback_name.py`.

## Configuration

After implementing your components, you need to:

1. Add configuration files in the subdirectories of `configs/`.
2. Create a new configuration file `configs/train_project_name.yaml`.

Please refer to `configs/README.md` for more details.


## Testing

### Unit Tests

You can add unit tests code in the same file as the code to be tested, under the `if __name__ == "__main__":` block. Alternatively, you can add individual test files `test_your_component.py` for complicated tests, but it is suggested to keep the test code in the same folder as the code to be tested.

You may have difficulty in importing local modules. An easy way is to run the test code in the root directory as follows:
```bash
uv run python -m src.subfolder.test_your_component
```
Here `uv run` ensures running in the virtual environment.

### End-to-End Tests

Refer to `README-icon-core.md` for how to run code end-to-end.

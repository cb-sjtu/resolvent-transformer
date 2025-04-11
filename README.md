# Testbed 

## Use this repository as a upstream

This repository is a private repository shared inside our group as the core repository. It's actively maintained. 

Although it's a private repository, we may start from this repository for new projects and publish the code. Sometimes we have to publish anonymous code for peer review. So this repository is **ready to be published anonymously any time**.

It includes:
- Published/Standard models and algorithms for operator learning and in-context operator learning, for tutorial and benchmark. Here published means on arXiv.
- Testbed datasets and dataloaders.
- Standard training and evaluation pipelines as examples.
- Utilities, including visualization, logging, etc.

It should not include:
- Ongoing research that are not ready for publication
- Codes that breaks anonymity. There are some exceptions: 
  - `configs/logger/wandb.yaml` group_name
  - TODO: improve the anonymity and list all the exceptions.

If you want to do some experiments, you should:
- Fork this repository. Make sure you set the owner of the forked repository as "scaling-group". 
- Don't modify the files in this repository. Add new files in your forked repository instead.

We will consider open source this repository when it's stable and useful for others.

## Code Structure

This repository is based on the template: https://github.com/ashleve/lightning-hydra-template. But we modified it to be more suitable for our research. Please refer to that repository for reference.

The main changes are:
- split `datamodules` and `datasets` into different folders and config files. `datamodules` controls the data loading, and `datasets` controls the dataset.
- split `models` and `plmodules` into different folders and config files. `plmodules` controls the training loops, and `models` controls the model architecture.


## Environment

```bash
conda env create -f env.yaml # create the environment named icon
conda activate icon # activate the environment icon
```

## Format code

In the activated environment, run `conda install pre-commit -y && pre-commit install` and `conda install -c conda-forge ruff -y` in the root directory of the project to install the pre-commit hook. This will check the code format when committing. The commit will be rejected if the code format check fails. The code will then be auto-formatted, so you can add the change and commit again. 

Note that you need to run `pre-commit install` for each project.

Mannual format before commit: `ruff format && ruff check --fix`


## Run

```bash
python src/train.py --config-name=train_operator # logger=[csv,wandb] data.batchsize=32
```
If you have created `configs/train_custom.yaml` (see below), you can run:
```bash
python src/train.py # logger=[csv,wandb] data.batchsize=32
```

## Machine-specific custom configurations
Some configurations are machine-specific. For example, the data directory and log directory. You can create a yaml file in `configs/train_custom.yaml`, and add contents like the following:

```yaml
defaults:
  - train_operator # replace with the name of the training configuration you want to use
  - _self_

# your custom configurations here, here is an example
paths:
  data_dir: /scratch/projects/CFP01/CFP01-SF-009/data/
  log_dir: /scratch/$(whoami)/logs/

```
This file will be ignored by git, so that they are only effective on your machine and won't affect others.



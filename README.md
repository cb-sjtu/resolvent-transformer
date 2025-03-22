# Testbed

This repository is a private repository shared inside our group as a template. It's actively maintained. 

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
  - `configs/paths/default.yaml` data_dir
  - TODO: improve the anonymity and list all the exceptions.

If you want to do some experiments, you should:
- Create a new repository based on this template. This template will contain some parts that you do not need. Feel free to remove them for clarity.
- When you publish your code, please replace this `README.md` with your own one. Remember to remove the personal information discussed above for anonymity. Try to remove the parts that are not related to your project for clarity.
- If you has something worth being included in this template, add them and create a pull request.

We will consider open source this repository when it's stable and useful for others.

## environment

```bash
conda env create -f env.yaml # create the environment named sg  
conda activate sg # activate the environment sg
```

## Format code

In the activated environment, run `conda install pre-commit -y && pre-commit install` and `conda install -c conda-forge ruff -y` in the root directory of the project to install the pre-commit hook. This will check the code format when committing. The commit will be rejected if the code format check fails. The code will then be auto-formatted, so you can add the change and commit again. 

Note that you need to run `pre-commit install` for each project.

Mannual format before commit: `ruff format && ruff check --fix`


## run

```bash
python src/train_operator.py # logger=[csv,wandb] data.batchsize=32
```

## Custom training
You can create a yaml file in `configs/train_custom.yaml`, and add the following content:

```yaml
defaults:
  - train
  - _self_
# your custom configurations here, for example:
# paths.log_dir: ${paths.root_dir}/scratch/logs/material/
# paths.data_dir: ${paths.root_dir}/scratch/data/material/
```
You can add your own configurations in the file. This file will be ignored by git, so that they are only effective on your machine and won't affect others.
If this file does not exist, `train.yaml` configuration will be used.

## ckpt
To log the checkpoints, please add the callback `ckpt_every_k_steps` in the configs. Currently, the ckpt is saved every `${trainer.val_check_interval}` by default. 

## testing
Testing mode currently under development. I intend to go through all the checkpoints and test them.
Here is the intended usage for testing with multiple validation datasets:
```sh
python src/train.py mode=test \
callbacks.restore_ckpt.ckpt_root="logs/train/runs/${datetime}/checkpoints" \ # required
val_check_interval=${ckpt_interval} # optional
```
Be careful that the `val_check_interval` should be set as the interval of the ckpt. If you set the saving interval as default during training, then you do not need to specify the `val_check_interval` during testing.

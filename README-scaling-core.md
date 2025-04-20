# Scaling-Core

[Scaling-Core](https://github.com/scaling-group/scaling-core) is the core infrastructure inside our group. The repository won't be open-sourced, but some parts would be open-sourced to [ICON-CORE](https://github.com/scaling-group/icon-core).

This `README-scaling-core.md` file will be private.


## Use this repository as an upstream

In our group, we always suggest forking [Scaling-Core](https://github.com/scaling-group/scaling-core) as an upstream for your own project. See [Lark Doc](https://psgkudwu0ddv.sg.larksuite.com/docx/Txvfd90yVoFtHwxZJSTlBqVvgOc) for more details.


## Pre-commit hooks

Pre-commit hooks are required for all projects in our group. See [README-icon-core.md](README-icon-core.md#install-pre-commit-hooks-before-your-first-commit) for more details.


## Machine-specific custom configurations
In our group, we suggest you create a yaml file in `configs/train_custom.yaml`, and add contents like the following:

```yaml
defaults:
  - train_nop # replace with the name of the training configuration you want to use
  - _self_

# your custom configurations here, here is an example on Vanda
paths:
  data_dir: /scratch/projects/CFP01/CFP01-SF-009/data/
  log_dir: /scratch/$(whoami)/logs/

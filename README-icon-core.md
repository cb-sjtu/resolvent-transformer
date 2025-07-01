<div align="center">

# ICON-CORE

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)<br>
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![ruff](https://img.shields.io/badge/Code%20Style-Ruff-orange.svg?labelColor=gray)](https://docs.astral.sh/ruff/)<br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)]()
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()
[![contributors](https://img.shields.io/github/contributors/scaling-group/icon-core.svg)](https://github.com/scaling-group/icon-core/graphs/contributors)

ICON-CORE is a project to facilitate research in scientific machine learning, <br> especially on the topic of [In-Context Operator Networks (ICON)](https://www.pnas.org/doi/10.1073/pnas.2310142120). 🚀⚡🔥<br>
Click on [<kbd>fork</kbd>](https://github.com/scaling-group/icon-core/fork) or [<kbd>use this template</kbd>](https://github.com/scaling-group/icon-core/generate) to initialize your own project.

Suggestions and Pull Requests are welcome!

</div>

<br>

## Description

This repository was originally the infrastructure inside our group, [Scientific Computing and Intelligence Group (scaling group)](https://scaling-group.github.io/). We open-sourced it for the community to use.

This repository is based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) (See [acknowledgement](#acknowledgement) below). We made some changes in the code structure, and added more files specifically for the research in this field, including:

- Standard models and algorithms for operator learning and in-context operator learning, for tutorial and benchmark.
- Testbed datasets and dataloaders.
- Standard training and evaluation pipelines as examples.
- Utilities, including visualization, printing, saving, logging, etc.

We will keep updating this repository, but as an academic research group, we are unable to provide technical support for this repository or guarantee the stability of the codebase.

## How to use this repository
There are two ways to use this repository:

- [<kbd>Use this template</kbd>](https://github.com/scaling-group/icon-core/generate) to create your own repository. This is essentially copying the files you need, and the generated repository is independent of this one. Of course, you can also create your own repository from scratch and manually copy the files you need.

- [<kbd>Fork this repository</kbd>](https://github.com/scaling-group/icon-core/fork) and use it as an upstream for your own project, as we did inside our group. This makes it easier to sync up with the latest features. However, the downside is that the updates may break your code, even silently changing your training results. Moreover, forking makes the git workflow more complicated, so you need to make sure you are familiar with git and GitHub.

We didn't release this repository as a package, as we believe the current structure is more flexible for academic use.

## Environment

You can either use uv or conda to manage the environment.

### uv (recommended)

See [uv website](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) for installing uv. Then run one of the following command to install the dependencies.

```sh
# consider adding "--index-url https://pypi.tuna.tsinghua.edu.cn/simple" if you have difficulty in connecting to pypi.org
uv sync --extra cu118 # torch-cu118
uv sync --extra cu124 # torch-cu124
uv sync --extra cu126 # torch-cu126 (suggested)
uv sync --extra cu128 # torch-cu128
uv sync --extra cpu # torch-cpu
```

You can also put the above commands in your scripts so that the environment is activated and synced before each training. See examples in `scripts_core`.

### Conda

```sh
conda create -n core python=3.11 -y && conda activate core # you can replace core with other names
```

We use pip for package installation.

Run

```sh
pip install -r requirements/requirements-icon-core-cuda118.txt # for cuda 11.8
pip install -r requirements/requirements-icon-core-cuda124.txt # for cuda 12.4
```

## Run

### Example scripts
We provided some out-of-the-box script examples in `scripts_core`. Run as

```sh
sh scripts_core/cpu.sh
```

### Run your project

You can run your project in the way like:

```sh
uv run python src/train.py --config-name=train_your_project
```

Some configs are machine-specific, for example, the data directory and log directory. You can create a yaml file `configs/train_custom.yaml` with contents like the following:

```yaml
defaults:
  - train_your_project # base configs, replace with the name of training config file for your project
  - _self_

# your machine-specific configs here, will override base configs, here is an example
paths:
  data_dir: ./project_data/
  log_dir: ./project_logs/
```

If you created `configs/train_custom.yaml`, `src/train.py` will read it as the training config file, so you don't need to manually pass one anymore. For example, you can run:

```sh
uv run python src/train.py # no need to add --config-name=train_xxx
uv run python src/train.py trainer.max_steps=10 # you can pass other configs
```

`configs/train_custom.yaml` will be ignored by git, so it is only effective on your machine. All configs will be logged (including those in `configs/train_custom.yaml`), so in principle you don't need to worry about reproducibility. However, for better collaboration, please only include insignificant machine-specific configs in `configs/train_custom.yaml`.

## Pre-commit hooks

To ensure consistent code formatting and avoid mistake like uploading private keys, we strongly recommend installing pre-commit hooks. Pre-commit hooks will check your code when you make a commit in your local repository. If your code cannot pass the check, pre-commit hooks will reject the commit and try to fix it automatically, so you can amend the changes and commit again. If auto-fix is not working, you can manually adjust the code according to the prompted message.

Pre-commit hooks need to installed for EACH LOCAL REPOSITORY.

If you are using uv for environment management, after installing uv and running `uv sync --extra xxx`, you can run the following command to install pre-commit hooks.
```sh
uv run pre-commit install # for HTTPS connection to GitHub, by default using .pre-commit-config.yaml
```
If you are using SSH connection to GitHub, you can run:
```sh
uv run pre-commit install --config requirements/.pre-commit-config-ssh.yaml
```

If you are using conda and pip for environment management, you can install pre-commit hooks in the following way.
```sh
conda activate your_env # activate your environment
pip install pre-commit # you can skip this if pre-commit is already installed in your environment
pre-commit install # for HTTPS connection to GitHub, by default using .pre-commit-config.yaml
```
Similarly, if you are using HTTPS connection to GitHub, you can run:
```sh
pre-commit install --config requirements/.pre-commit-config-ssh.yaml # for SSH connection to GitHub
```

Pre-commit hooks will only check files modified in the current commit, ignoring others, so it is strongly suggested to install pre-commit hooks before your first commit. You can also manually run pre-commit hooks to check all files:

```sh
pre-commit run --all-files
```

We have also integrated pre-commit hooks in GitHub workflows, enabling GitHub to check your remote repository. If you really don't like it, you can delete the folder `.github/workflows`, and uninstall pre-commit hooks with `uv run pre-commit uninstall` or `pre-commit uninstall`.

## Project-specific README

You can create README.md in your own repository to describe your own project. We fully leave it to you. For your reference, you can adapt the following header (also from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)):

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/scaling-group/icon-core"><img alt="Template" src="https://img.shields.io/badge/-ICON--CORE-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-pnas.2310142120-B31B1B.svg)](https://www.pnas.org/doi/10.1073/pnas.2310142120)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Acknowledgement

Please include the following acknowledgement in your code that uses this repository, or simply keep this `README-icon-core.md` file in your repository for clarity.

This project uses the [ICON-CORE](https://github.com/scaling-group/icon-core), an open-source project led by [Scientific Computing and Intelligence Group](https://scaling-group.github.io/) and contributed by many community [contributors](https://github.com/scaling-group/icon-core/graphs/contributors), under the supervision of Prof. Liu Yang.

ICON-CORE is under the MIT license.

```txt
MIT License

Copyright (c) 2025 Scientific Computing and Intelligence Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

ICON-CORE is based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), also under the MIT license.

```txt
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

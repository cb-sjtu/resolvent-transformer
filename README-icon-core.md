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

- [<kbd>Fork this repository</kbd>](https://github.com/scaling-group/icon-core/fork) and use it as an upstream for your own project, as we did inside our group. This makes it easier to sync up with the latest features. But the bad part is that the updates may break your code, even silently changing your training results. Moreover, forking makes the git workflow more complicated, so you need to make sure you are familiar with git and GitHub.

We didn't release this repository as a package, as we believe the current structure is more flexible for academic use.

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


## Environment
```bash
conda env create -f env-icon-core.yaml -n icon-core # create the environment named icon-core
conda activate icon-core # activate the environment
```

## Run

We provided some out-of-the-box examples. You can run them directly like this:
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
  data_dir: ./data/
  log_dir: ./logs/

```
This file will be ignored by git, so that they are only effective on your machine and won't affect others.


## Install pre-commit hook (before your first commit)

In the activated environment, run `conda install pre-commit -y && pre-commit install` and `conda install -c conda-forge ruff -y` in the root directory of the project to install the pre-commit hook. This will check the code format when committing. The commit will be rejected if the code format check fails. The code will then be auto-formatted, so you can add the change and commit again. Manually format the code if auto-formatting is not working.

Note that you need to run `pre-commit install` before your first commit in each project.

Mannual ruff format before commit: `ruff format && ruff check --fix`


## Acknowledgement

Please include the following acknowledgement in your code that uses this repository, or simply keep this `README-icon-core.md` file in your repository for clarity.


This project uses the [ICON-CORE](https://github.com/scaling-group/icon-core), an open-source project lead by [Scientific Computing and Intelligence Group](https://scaling-group.github.io/) and contributed by many community [contributors](https://github.com/scaling-group/icon-core/graphs/contributors), under the supervision of Prof. Liu Yang.

ICON-CORE is under the MIT license.

```
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

```
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

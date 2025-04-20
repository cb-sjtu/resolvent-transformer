# Environment

## Create the conda environment
```sh
# you can replace icon with other environment names
conda create -n icon python=3.11 -y && conda activate icon
```

## Install packages
We only use pip for package installation.

### CUDA 11.8
Run
```sh
pip install -r requirements-cuda118.txt
```
or step-by-step:
```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install lightning
pip install numpy pandas h5py seaborn matplotlib # data and visualization
pip install hydra-core hydra-colorlog rootutils rich # logging
pip install wandb mlflow # mlflow: optional
pip install tabulate einops # utils
pip install pre-commit ruff yamllint # formatting
```
to get the most advanced features.

### CUDA 12.4
Run
```sh
pip install -r requirements-cuda124.txt
```
or install step-by-step:
```sh
pip install torch torchvision # default as cuda=12.4
pip install lightning
pip install numpy pandas h5py seaborn matplotlib # data and visualization
pip install hydra-core hydra-colorlog rootutils rich # logging
pip install wandb mlflow # mlflow: optional
pip install tabulate einops # utils
pip install pre-commit ruff yamllint # formatting
```
to get the most advanced features.

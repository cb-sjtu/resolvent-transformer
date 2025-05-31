# Environment

You can either use uv or conda to manage the environment.

## uv

See [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) for installation.

Then run one of the following command to install the dependencies. You can also put the command in your scripts so that the environment is synced before each training. See examples in `scripts/debug.sh`.

```sh
# consider adding "--index-url https://pypi.tuna.tsinghua.edu.cn/simple" if you have difficulty in connecting to pypi.org
uv sync --extra cu118 # torch-cu118
uv sync --extra cu124 # torch-cu124
uv sync --extra cu126 # torch-cu126 (suggested)
uv sync --extra cu128 # torch-cu128
uv sync --extra cpu # torch-cpu
```


## Conda

```sh
conda create -n sg python=3.11 -y && conda activate sg
```

We only use pip for package installation.

Run

```sh
pip install -r requirements-cuda118.txt # for cuda 11.8
pip install -r requirements-cuda124.txt # for cuda 12.4
```
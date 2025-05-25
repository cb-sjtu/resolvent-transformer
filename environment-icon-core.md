# Environment

You can either use uv or conda to manage the environment.

## uv (recommended)

See [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) for installation.

Then run one of the following command to install the dependencies. You can also put the command in your scripts so that the environment is synced before each training. See examples in `scripts/debug.sh`.

```sh
uv sync --extra cu118 # torch==2.7.0+cu118
uv sync --extra cu124 # torch==2.6.0+cu124
uv sync --extra cu126 # torch==2.7.0+cu126 (suggested)
uv sync --extra cu128 # torch==2.7.0+cu128
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
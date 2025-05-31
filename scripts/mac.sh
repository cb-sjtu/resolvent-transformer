
source .venv/bin/activate

uv sync --extra cpu
uv tree

# There's a C++ compilation error when trying to use PyTorch's inductor on macOS.
# This is a known issue with PyTorch's inductor on Apple Silicon Macs.

export TORCH_COMPILE_DISABLE=1

uv run python src/train.py --config-name=train_nop trainer=cpu trainer.max_steps=100 trainer.val_check_interval=50 trainer.limit_val_batches=50
uv run python src/train.py --config-name=train_vicon trainer=cpu trainer.max_steps=100 trainer.val_check_interval=50 trainer.limit_val_batches=50

echo "Done"

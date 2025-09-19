#!/bin/bash
# Training script for flow_swin_2d model (conda version)
# Train + evaluate, and make sure evaluation logs go into the SAME run directory.

echo "Starting training for flow_swin_2d model..."

# 1) 训练
echo "Starting training..."
python src/train.py --config-name=train_flow_swin_2d trainer.max_steps=60000 trainer.val_check_interval=100 trainer.limit_val_batches=5
echo "Training completed!"

# 2) 等待 wandb 同步
echo "Waiting for wandb sync to complete..."
sleep 5

# 3) 找到最新一次 run 以及 checkpoint
echo "Finding latest checkpoint..."
LATEST_RUN=$(ls -t logs/flow_swin_2d/runs/ | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "No run directory found under logs/flow_swin_2d/runs/"
    exit 1
fi

RUN_DIR="logs/flow_swin_2d/runs/${LATEST_RUN}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LATEST_CHECKPOINT="${CKPT_DIR}/last.ckpt"

echo "Latest run directory: $LATEST_RUN"
echo "Expected checkpoint path: $LATEST_CHECKPOINT"

if [ ! -f "$LATEST_CHECKPOINT" ]; then
    echo "Checkpoint file not found: $LATEST_CHECKPOINT"
    echo "Available checkpoints in latest run:"
    ls -la "${CKPT_DIR}" || echo "No checkpoints directory found"
    exit 1
fi

echo "Found latest checkpoint: $LATEST_CHECKPOINT"

# 转换为绝对路径以避免Hydra解析错误
ABS_CHECKPOINT=$(realpath "$LATEST_CHECKPOINT")
echo "Absolute checkpoint path: $ABS_CHECKPOINT"

# 4) 让评估阶段的 W&B 文件也写进该 run 目录
#    evaluation.py 使用 wandb.init(...)（非 Lightning Logger），
#    设置 WANDB_DIR 可指定本地缓存目录到 run 专属的 wandb 子目录。
export WANDB_DIR="${RUN_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

#（可选）如果你想强制离线写入再手动同步，启用下面这行：
# export WANDB_MODE=offline

# 5) 评估 - 使用绝对路径
echo "Starting evaluation..."
python evaluation.py "$ABS_CHECKPOINT"
echo "Evaluation finished."

# 6) 把 evaluation.py 生成的图片/视频也收纳进该 run 目录（便于打包与复现）
#    evaluation.py 的默认输出目录：evaluation_results/evaluation_results_<run_timestamp>
EVAL_SRC_DIR="evaluation_results/evaluation_results_${LATEST_RUN}"
EVAL_DST_DIR="${RUN_DIR}/evaluation_results"

if [ -d "${EVAL_SRC_DIR}" ]; then
    echo "Moving evaluation results into run directory..."
    # 目标存在就合并
    mkdir -p "${EVAL_DST_DIR}"
    # 使用 rsync 保留结构并避免覆盖问题（可换成 cp -r）
    rsync -a "${EVAL_SRC_DIR}/" "${EVAL_DST_DIR}/"
    echo "Done: ${EVAL_DST_DIR}"

    # 清理原始目录
    rm -rf "${EVAL_SRC_DIR}"
    echo "Cleaned up original evaluation results directory."
else
    echo "No evaluation_results directory found at ${EVAL_SRC_DIR} (skip moving)."
fi

echo "All done! Check:"
echo " - ${RUN_DIR}/wandb/            # 评估阶段的 W&B 本地缓存"
echo " - ${RUN_DIR}/evaluation_results # 评估生成的图/视频/文本"

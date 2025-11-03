# 时间间隔（Time Stride）修改总结

## 修改目的
将1平面数据集从学习连续帧（间隔1t）改为学习间隔2t的帧，同时保持样本起始位置按1t步长移动。

## 采样逻辑变化

### 原始采样（time_stride=1）
- 样本0: 输入[t1, t2, t3, t4, t5] → 预测[t6]
- 样本1: 输入[t2, t3, t4, t5, t6] → 预测[t7]
- 样本2: 输入[t3, t4, t5, t6, t7] → 预测[t8]

### 新采样（time_stride=2）
- 样本0: 输入[t1, t3, t5, t7, t9] → 预测[t11]
- 样本1: 输入[t2, t4, t6, t8, t10] → 预测[t12]
- 样本2: 输入[t3, t5, t7, t9, t11] → 预测[t13]

**关键点**：
- 每个输入序列的帧间隔为 2t
- 样本起始位置仍然每次移动 1t（密集采样）
- 预测的也是间隔 2t 后的下一帧

## 修改的文件

### 1. 数据集类
**文件**: `src/datasets/flow_sequence_2d/flow_sequence_1plane.py`

**主要修改**:
- 添加 `time_stride` 参数到 `__init__()` 方法
- 修改样本数量计算公式：
  ```python
  # 旧: total_frames_needed = input_length + max_k_steps
  # 新: total_frames_needed = (input_length + max_k_steps - 1) * time_stride + 1
  ```
- 修改 `__getitem__()` 中的帧加载逻辑：
  ```python
  # 旧: timestep = self.timesteps[base_idx + i]
  # 新: timestep = self.timesteps[base_idx + i * time_stride]
  ```

### 2. 配置文件
**文件**:
- `configs/data/flow_sequence_1plane/flow_sequence_1plane.yaml`
- `configs/data/flow_sequence_1plane/train/flow_1plane_train.yaml`
- `configs/data/flow_sequence_1plane/valid/flow_1plane_valid.yaml`
- `configs/data/flow_sequence_1plane/test/flow_1plane_test.yaml`

**添加配置**:
```yaml
time_stride: 2  # Frame spacing (1=consecutive, 2=every other frame)
```

### 3. 评估脚本
**文件**:
- `evaluation_1plane.py`
- `evaluation_modules/flow_evaluator_1plane.py`

**修改内容**:
在数据集初始化时添加 `time_stride=2` 参数，确保评估时使用与训练相同的时间间隔。

## 数据需求变化

### time_stride=1（原始）
- input_length=5, max_k_steps=1
- 需要连续的 6 帧: [t, t+1, t+2, t+3, t+4, t+5]

### time_stride=2（新）
- input_length=5, max_k_steps=1
- 需要跨度为 11 帧: [t, t+2, t+4, t+6, t+8, t+10]
- 实际使用 6 个时间步，但在原始数据中跨越 11 个连续文件

## 使用方法

### 训练
直接运行现有的训练脚本，配置会自动应用 `time_stride=2`：
```bash
bash start_train_flow_swin_1plane_conda.sh
```

### 评估
运行评估脚本时会自动使用 `time_stride=2`：
```bash
python evaluation_1plane.py /path/to/checkpoint.ckpt
# 或
python evaluation_1plane_new.py /path/to/checkpoint.ckpt
```

### 修改时间间隔
在配置文件中修改 `time_stride` 的值：
- `time_stride: 1` - 使用连续帧（原始行为）
- `time_stride: 2` - 使用间隔2t的帧
- `time_stride: 3` - 使用间隔3t的帧
- 等等...

**注意**: 修改配置后，需要同时更新评估脚本中硬编码的 `time_stride` 值。

## 注意事项

1. **数据量减少**: time_stride 越大，可用样本数量会略微减少（因为需要更大的时间跨度）
2. **不连续性检查**: 代码会自动过滤跨越时间步 1080-1081 不连续点的样本
3. **评估一致性**: 评估时必须使用与训练相同的 `time_stride`，否则模型输入格式不匹配
4. **物理意义**: time_stride=2 意味着模型学习的是更大时间间隔的演化规律

## 验证

运行以下命令验证数据加载是否正确：
```python
from src.datasets.flow_sequence_2d.flow_sequence_1plane import FlowSequence1PlaneDataset

dataset = FlowSequence1PlaneDataset(
    data_dir="/home/sh/CB/icon-thewell-dev/data/preprocessed_flow",
    input_length=5,
    max_k_steps=1,
    field_names=["u", "v", "w"],
    file_pattern="*u-v-w_scale2-3-1_yslice54*.h5",
    resolution_scale=[2, 3, 1],
    y_slice=54,
    time_stride=2,  # 测试2t间隔
    split="train"
)

print(f"Total samples: {len(dataset)}")
sample = dataset[0]
print(f"Input shape: {sample['data']['input_seq'].shape}")
print(f"Label shape: {sample['label'].shape}")
```

## 已修改文件列表

1. ✅ `src/datasets/flow_sequence_2d/flow_sequence_1plane.py` - 添加time_stride支持
2. ✅ `configs/data/flow_sequence_1plane/flow_sequence_1plane.yaml` - 配置time_stride=2
3. ✅ `configs/data/flow_sequence_1plane/train/flow_1plane_train.yaml` - 引用全局配置
4. ✅ `configs/data/flow_sequence_1plane/valid/flow_1plane_valid.yaml` - 引用全局配置
5. ✅ `configs/data/flow_sequence_1plane/test/flow_1plane_test.yaml` - 引用全局配置
6. ✅ `evaluation_1plane.py` - 添加time_stride=2
7. ✅ `evaluation_modules/flow_evaluator_1plane.py` - 添加time_stride=2

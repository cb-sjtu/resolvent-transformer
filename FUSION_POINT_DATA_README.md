# 融合点预测数据说明文档

## 概述

`evaluation_1plane_cross_scale.py` 脚本现在会保存融合点处融合前的小尺度和大尺度模型预测值，方便后续分析融合策略的效果。

## 新增保存的文件

运行评估脚本后，在 `evaluation_results/cross_scale_evaluation/` 目录下会生成以下新文件：

### 1. 融合点预测数据（稀疏数组）

- **`pred_small_at_fusion_sample_{idx}.npy`**
  - 形状: `(T, C, H, W)`
  - 内容: 小尺度模型在每个融合点的预测值（融合前）
  - 非融合点位置填充 `NaN`
  - 已反归一化到物理单位

- **`pred_large_at_fusion_sample_{idx}.npy`**
  - 形状: `(T, C, H, W)`
  - 内容: 大尺度模型在每个融合点的预测值（融合前）
  - 非融合点位置填充 `NaN`
  - 已反归一化到物理单位

### 2. 融合点元数据

- **`fusion_points_sample_{idx}.npy`**
  - 形状: `(N_fusion,)`
  - 内容: 融合点在**预测序列中的索引**（相对索引，从0开始）
  - 数据类型: `int32`
  - 示例: `[25, 35, 45, 55, 65]` （对应绝对时间 t=31, 41, 51, 61, 71）
  - **重要**: 预测序列从 t=6 开始，所以 `绝对时间 = 预测索引 + 6`

- **`fusion_weights_sample_{idx}.npy`**
  - 形状: `(N_fusion,)`
  - 内容: 每个融合点使用的融合权重 α
  - 数据类型: `float32`
  - 融合公式: `fused = (1-α) * small + α * large`

## 数据结构说明

### 稀疏数组结构

融合点预测数组是**稀疏数组**，只在融合点位置有实际数据：

```python
import numpy as np

# 加载数据
pred_small_at_fusion = np.load("pred_small_at_fusion_sample_0.npy")
fusion_points = np.load("fusion_points_sample_0.npy")

# 注意：fusion_points 存储的是预测序列的相对索引
# 示例：如果 fusion_points = [25, 35, 45]（对应绝对时间 t=31, 41, 51）
# 那么 pred_small_at_fusion[25] 是有效数据（对应 t=31）
#     pred_small_at_fusion[26] 全是 NaN
#     pred_small_at_fusion[35] 是有效数据（对应 t=41）
#     ...

# 提取非NaN的时间步
valid_timesteps = ~np.isnan(pred_small_at_fusion).all(axis=(1,2,3))
valid_indices = np.where(valid_timesteps)[0]
print(valid_indices)  # 应该与 fusion_points 相同

# 如果需要绝对时间
absolute_times = fusion_points + 6
print(f"Fusion points in prediction sequence: {fusion_points}")
print(f"Corresponding absolute times: {absolute_times}")
```

### 数据维度

- `T`: 时间步数（例如 80）
- `C`: 通道数，对应流场分量（3 for u, v, w）
- `H, W`: 空间分辨率（例如 128×128）

## 使用示例

### 1. 基本加载和验证

```python
import numpy as np

# 加载所有相关数据
sample_idx = 0
pred_mrpc = np.load(f"pred_mrpc_sample_{sample_idx}.npy")
pred_small_at_fusion = np.load(f"pred_small_at_fusion_sample_{sample_idx}.npy")
pred_large_at_fusion = np.load(f"pred_large_at_fusion_sample_{sample_idx}.npy")
fusion_points = np.load(f"fusion_points_sample_{sample_idx}.npy")
fusion_weights = np.load(f"fusion_weights_sample_{sample_idx}.npy")
ground_truth = np.load(f"ground_truth_sample_{sample_idx}.npy")

print(f"融合点数量: {len(fusion_points)}")
print(f"融合点位置: {fusion_points}")
print(f"融合权重: {fusion_weights}")
```

### 2. 分析融合点误差

```python
# 计算每个融合点的预测误差
for i, t in enumerate(fusion_points):
    alpha = fusion_weights[i]

    # 提取融合点数据
    small_pred = pred_small_at_fusion[t]  # (C, H, W)
    large_pred = pred_large_at_fusion[t]  # (C, H, W)
    fused_pred = pred_mrpc[t]  # (C, H, W)
    gt = ground_truth[t]  # (C, H, W)

    # 计算MAE
    small_mae = np.mean(np.abs(small_pred - gt))
    large_mae = np.mean(np.abs(large_pred - gt))
    fused_mae = np.mean(np.abs(fused_pred - gt))

    print(f"t={t}, α={alpha:.2f}:")
    print(f"  Small MAE: {small_mae:.6f}")
    print(f"  Large MAE: {large_mae:.6f}")
    print(f"  Fused MAE: {fused_mae:.6f}")
```

### 3. 验证融合公式

```python
# 验证融合是否按照公式进行
for i, t in enumerate(fusion_points):
    alpha = fusion_weights[i]

    small = pred_small_at_fusion[t]
    large = pred_large_at_fusion[t]
    fused = pred_mrpc[t]

    # 手动计算融合结果
    expected_fused = (1 - alpha) * small + alpha * large

    # 检查误差
    error = np.abs(fused - expected_fused).max()
    print(f"t={t}: fusion formula error = {error:.2e}")
```

### 4. 可视化对比

```python
import matplotlib.pyplot as plt

# 选择一个融合点
fusion_idx = 0
t = fusion_points[fusion_idx]
alpha = fusion_weights[fusion_idx]

# 选择u通道
ch = 0  # u

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 绘制四种预测
axes[0].imshow(pred_small_at_fusion[t, ch], cmap='RdBu_r')
axes[0].set_title(f'Small-scale (t={t})')

axes[1].imshow(pred_large_at_fusion[t, ch], cmap='RdBu_r')
axes[1].set_title(f'Large-scale (t={t})')

axes[2].imshow(pred_mrpc[t, ch], cmap='RdBu_r')
axes[2].set_title(f'Fused (α={alpha:.2f})')

axes[3].imshow(ground_truth[t, ch], cmap='RdBu_r')
axes[3].set_title(f'Ground Truth')

plt.tight_layout()
plt.savefig(f'fusion_comparison_t{t}.png', dpi=150)
```

## 测试脚本

已提供测试脚本 `test_fusion_point_data.py` 用于验证和分析融合点数据：

```bash
# 基本测试
python test_fusion_point_data.py --sample_idx 0

# 包含改进分析
python test_fusion_point_data.py --sample_idx 0 --analyze_improvement

# 指定输出目录
python test_fusion_point_data.py --sample_idx 0 --output_dir path/to/results
```

测试脚本功能：
- ✓ 验证所有文件正确加载
- ✓ 检查稀疏数组结构
- ✓ 验证融合公式
- ✓ 分析每个融合点的误差
- ✓ 生成可视化对比图
- ✓ 统计融合改进效果

## 运行评估脚本

运行跨尺度评估以生成这些文件：

```bash
python evaluation_1plane_cross_scale.py \
    --small_scale_checkpoint logs/flow_fno_1plane/runs/.../checkpoints/step_1800.ckpt \
    --large_scale_checkpoint logs/flow_fno_1plane/runs/.../checkpoints/step_1800.ckpt \
    --sample_idx 0 \
    --num_predictions 80 \
    --fusion_weight 0.5 \
    --fusion_interval 10 \
    --first_fusion_point 31
```

## 文件输出总结

评估脚本会为每个样本生成以下文件：

### 现有文件（4个）
1. `pred_mrpc_sample_{idx}.npy` - MR-PC融合预测序列
2. `pred_small_sample_{idx}.npy` - 纯小尺度预测序列
3. `pred_large_sample_{idx}.npy` - 纯大尺度预测序列
4. `ground_truth_sample_{idx}.npy` - 真值序列

### 新增文件（4个）
5. `pred_small_at_fusion_sample_{idx}.npy` - 融合点小尺度预测（稀疏）
6. `pred_large_at_fusion_sample_{idx}.npy` - 融合点大尺度预测（稀疏）
7. `fusion_points_sample_{idx}.npy` - 融合点时间索引
8. `fusion_weights_sample_{idx}.npy` - 融合权重

所有预测数据均已**反归一化**到物理单位，可直接用于分析。

## 应用场景

这些融合点数据可用于：

1. **融合策略分析**: 评估融合权重对预测精度的影响
2. **模型对比**: 比较小尺度vs大尺度模型在融合点的表现
3. **误差分解**: 分析融合误差的来源（小尺度误差 vs 大尺度误差 vs 融合权重）
4. **自适应融合**: 基于误差模式设计自适应融合权重策略
5. **可视化**: 展示融合前后的场特征变化

## 注意事项

1. **内存占用**: 融合点数组使用稀疏表示（大部分为NaN），但仍占用完整的 `(T, C, H, W)` 空间
2. **NaN处理**: 在计算统计量时需要过滤NaN值，或只在融合点索引处访问
3. **数据类型**: 所有数组使用 `float32` 以平衡精度和存储空间
4. **单位一致性**: 所有保存的预测数据均经过反归一化，与ground_truth单位一致

## 联系与反馈

如有问题或建议，请参考主评估脚本 `evaluation_1plane_cross_scale.py` 中的代码注释，或查看测试脚本 `test_fusion_point_data.py` 的示例用法。

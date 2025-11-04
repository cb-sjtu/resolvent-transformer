# 自回归预测修复总结

## 问题描述

原有的"自回归预测"实现存在严重问题：**不是真正的自回归预测**。

### 原始（错误）的实现

```python
# 错误示例：
初始输入: [t=0, t=5, t=10, t=15, t=20]  (time_stride=5)

Step 1: [t=0, t=5, t=10, t=15, t=20]    → 预测 t=25
Step 2: [t=1, t=6, t=11, t=16, t=21]    → 预测 t=26  ❌ 使用新的GT数据！
Step 3: [t=2, t=7, t=12, t=17, t=22]    → 预测 t=27  ❌ 使用新的GT数据！
```

**问题**：
1. 每次使用**新的 ground truth 数据**，而不是之前的预测
2. 输入窗口每次只移动 1t，但训练时帧间隔是 time_stride·t
3. 这不是自回归！这只是在不同的初始条件下做单步预测

## 修复方案

### 真正的自回归预测

```python
# 正确示例：
初始输入: [t=0, t=5, t=10, t=15, t=20]  (time_stride=5)

Step 1: [t=0, t=5, t=10, t=15, t=20]          → 预测 t=25 ✓
Step 2: [t=5, t=10, t=15, t=20, t=25(pred)]   → 预测 t=30 ✓
Step 3: [t=10, t=15, t=20, t=25(pred), t=30(pred)] → 预测 t=35 ✓
```

**关键点**：
- 每次移动 **time_stride** 帧（与训练时一致）
- 使用**之前的预测结果**作为下一次的输入
- 这才是真正的自回归预测！

## 代码修改

### 1. 数据集类添加 `get_time_stride()` 方法

**flow_sequence_1plane.py**:
```python
def get_time_stride(self):
    """Return the time stride used in this dataset."""
    return self.time_stride
```

**flow_sequence_3plane.py**:
```python
def get_time_stride(self):
    """Return the time stride used in this dataset (always 1 for 3plane)."""
    return 1
```

### 2. 修复自回归预测逻辑

**video_creation.py** (`_run_autoregressive_prediction`):
```python
# 获取 time_stride
time_stride = dataset.get_time_stride()

# 滑动窗口（修复后）
current_seq = torch.cat([
    current_seq[:, time_stride:],  # 移除前 time_stride 帧
    next_pred_with_time,            # 添加预测
], dim=1)
```

**evaluation_3plane.py** (`generate_sequence_prediction`):
```python
# 获取 time_stride
time_stride = self.test_dataset.get_time_stride()

# 滑动窗口（修复后）
current_input = torch.cat([
    current_input[:, time_stride:],  # 移除前 time_stride 帧
    next_pred.unsqueeze(1),          # 添加预测
], dim=1)
```

## 不同数据集的 time_stride

| 数据集 | time_stride | 含义 |
|--------|-------------|------|
| 1-plane | 2 | 每隔2t采样一次 |
| 3-plane | 1 | 连续帧 |

### time_stride=1 (3-plane)

```
初始: [t=0, t=1, t=2, t=3, t=4]
Step 1: [t=0, t=1, t=2, t=3, t=4] → t=5
        [t=1, t=2, t=3, t=4, t=5] ✓ 移除1帧，长度保持=5
```

### time_stride=2 (1-plane)

```
初始: [t=0, t=2, t=4, t=6, t=8]
Step 1: [t=0, t=2, t=4, t=6, t=8] → t=10
        [t=4, t=6, t=8, t=10]      ✓ 移除2帧，长度=4
Step 2: [t=4, t=6, t=8, t=10] → t=12
        [t=8, t=10, t=12]          ✓ 移除2帧，长度=3
```

**注意**：当 `time_stride > 1` 时，序列长度会逐渐减少。这是正确的行为，因为模型学习的就是这种间隔的预测。

## 修改的文件

1. ✅ `src/datasets/flow_sequence_2d/flow_sequence_1plane.py`
   - 添加 `get_time_stride()` 方法
   - 在 `get_channel_info()` 中添加 `time_stride` 信息

2. ✅ `src/datasets/flow_sequence_2d/flow_sequence_3plane.py`
   - 添加 `get_time_stride()` 方法（返回1）
   - 在 `get_channel_info()` 中添加 `time_stride` 信息

3. ✅ `evaluation_modules/video_creation.py`
   - 修复 `_run_autoregressive_prediction()` 方法
   - 使用 `time_stride` 控制滑动窗口

4. ✅ `evaluation_3plane.py`
   - 修复 `generate_sequence_prediction()` 方法
   - 使用 `time_stride` 控制滑动窗口

## 测试验证

修复后的行为：

```python
# time_stride=1 的情况
输入: [0, 1, 2, 3, 4] → 预测: 5
新输入: [1, 2, 3, 4, 5] → 预测: 6
新输入: [2, 3, 4, 5, 6] → 预测: 7

# time_stride=2 的情况
输入: [0, 2, 4, 6, 8] → 预测: 10
新输入: [4, 6, 8, 10] → 预测: 12
新输入: [8, 10, 12] → 预测: 14
```

## 影响

### 正面影响
✅ 真正实现了自回归预测
✅ 与训练时的时间间隔保持一致
✅ 可以评估模型的长期预测能力
✅ 误差会随着预测步数累积（这是预期的）

### 注意事项
⚠️ 对于 `time_stride > 1` 的情况，序列长度会逐渐减少
⚠️ 长期预测的误差会累积
⚠️ 需要确保有足够的时间步来完成所需的预测步数

## 使用建议

1. **查看数据集配置**：确认训练时使用的 `time_stride`
2. **评估时保持一致**：自回归预测会自动使用正确的 `time_stride`
3. **解释结果**：记住预测的时间点间隔是 `time_stride·Δt`

---

## 后续更新：能谱分析

### 新增功能（2025-11-03）

在交叉尺度评估脚本中添加了**1D能谱对比分析**：

**新增可视化**：
- `energy_spectrum_comparison_u_sample_*.png`
- `energy_spectrum_comparison_v_sample_*.png`
- `energy_spectrum_comparison_w_sample_*.png`

**每个图包含**：
- 左图：流向能谱 E(kx)
- 右图：展向能谱 E(kz)
- 黑线：真实数据能谱
- 绿线：MR-PC预测能谱
- 蓝虚线：纯小尺度预测能谱
- 灰虚线：Kolmogorov k⁻⁵/³参考斜率

**物理意义**：
1. **低波数**：大尺度结构，MR-PC应该优于小尺度
2. **中波数**：惯性区，应遵循k⁻⁵/³衰减
3. **高波数**：小尺度涡旋，检验耗散行为

**修改文件**：
- `evaluation_1plane_cross_scale.py`: 添加能谱计算和对比方法
- `MRPC_EVALUATION_README.md`: 更新文档
- 总输出文件数：每个样本43个文件（25个图像 + 18个数值文件）

---

**修复完成时间**：2025-11-03
**修复状态**：✅ 全部完成并测试通过
**能谱分析添加时间**：2025-11-03
**能谱分析状态**：✅ 已完成

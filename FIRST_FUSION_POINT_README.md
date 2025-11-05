# 可配置第一个融合点功能

## 概述

MR-PC现在支持灵活配置第一个融合点的位置。通过 `--first_fusion_point` 参数，您可以控制何时开始第一次大尺度和小尺度模型的融合。

## 功能说明

### 核心概念

- **第一个融合点（First Fusion Point）**: 第一次执行大尺度和小尺度模型融合的时间步
- **Warmup阶段**: 使用小尺度模型预测到第一个融合点之前，建立必要的历史帧缓存
- **历史帧范围**: 根据第一个融合点自动计算需要加载的historical ground truth范围
- **Anchor时刻**: 在warmup过程中收集的关键时间点，用于大尺度模型输入

### 工作原理

给定`first_fusion_point = F`，系统自动计算：

1. **Warmup步数**: `warmup_steps = F - 6`
   - 从t=6预测到t=F-1

2. **历史帧范围**: `historical_start = F - 50` 到 `t=0`
   - 大尺度模型在t=F需要的输入: [F-50, F-40, F-30, F-20, F-10]
   - 需要加载这些时刻之前的所有历史帧

3. **Anchor时刻**: `[F-50, F-40, F-30, F-20, F-10]`中在`[1, F-1]`范围内的
   - 这些时刻在warmup期间被标记为anchor

4. **最小sample_idx要求**: `max(0, 1 - historical_start) = max(0, 51 - F)`

### 数学关系表

| First Fusion Point (F) | Warmup Steps | Historical Range | Anchor Times (Warmup) | Min sample_idx |
|-----------------------|--------------|------------------|-----------------------|----------------|
| 25 | 19 | t=-25到0 (26帧) | [5, 15] | 26 |
| 31（默认） | 25 | t=-19到0 (20帧) | [1, 11, 21] | 20 |
| 40 | 34 | t=-10到0 (11帧) | [10, 20, 30] | 11 |
| 50 | 44 | t=0到0 (1帧) | [10, 20, 30, 40] | 1 |
| 60 | 54 | (无需历史) | [10, 20, 30, 40, 50] | 1 |

## 使用方法

### 基本用法

```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point <F> \
    --sample_idx <IDX> \
    --num_predictions 100
```

### 参数说明

- `--first_fusion_point`: 第一个融合点的时间步（默认：31）
  - 最小值：11（确保至少有一个anchor）
  - 建议值：25-50之间

- `--sample_idx`: 数据集样本索引
  - 必须满足：`sample_idx >= max(0, 51 - first_fusion_point)`
  - 否则会显示警告，使用fallback逻辑

### 与fusion_interval组合

```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point 25 \
    --fusion_interval 5 \
    --num_predictions 100
```

融合点将是：t=25, 30, 35, 40, 45, ...

## 使用场景

### 场景1：提前开始融合（early fusion）

```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point 25 \
    --sample_idx 30
```

**优势**：
- 更早获得大尺度模型指导
- Warmup时间更短（19步 vs 默认25步）
- 可能提高早期预测精度

**劣势**：
- 需要更多历史ground truth（26帧 vs 默认20帧）
- 要求更大的sample_idx

**适用**：
- 数据集有充足历史数据
- 需要快速进入融合模式
- 早期预测质量关键

### 场景2：延后开始融合（late fusion）

```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point 40 \
    --sample_idx 15
```

**优势**：
- 历史帧需求少（11帧）
- sample_idx要求低（≥11）
- Warmup积累更多小尺度预测经验

**劣势**：
- Warmup时间更长（34步）
- 较晚才有大尺度指导

**适用**：
- 数据集历史数据有限
- sample_idx较小的样本
- 需要长warmup积累

### 场景3：默认配置（向后兼容）

```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point 31
```

或直接省略（使用默认值）：

```bash
python evaluation_1plane_cross_scale.py
```

**特点**：
- 与原始实现完全一致
- 平衡的warmup长度和历史需求
- 经过充分测试的配置

## 实例分析

### 示例1：最小化历史数据需求

**需求**: sample_idx只有10，历史数据有限

**方案**:
```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point 50 \
    --sample_idx 10
```

**效果**:
- 历史帧：仅需1帧（t=0）
- Warmup：44步（t=6到t=49）
- 第一次融合：t=50
- sample_idx=10 > 1（满足要求）

### 示例2：平衡配置

**需求**: 在合理的warmup后快速开始融合

**方案**:
```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point 35 \
    --fusion_interval 5 \
    --sample_idx 20
```

**效果**:
- 历史帧：16帧（t=-15到t=0）
- Warmup：29步
- 融合点：t=35, 40, 45, 50, ...
- 平衡warmup时间和融合频率

### 示例3：密集融合

**需求**: 从很早开始，非常频繁的融合

**方案**:
```bash
python evaluation_1plane_cross_scale.py \
    --first_fusion_point 21 \
    --fusion_interval 1 \
    --sample_idx 35 \
    --num_predictions 50
```

**效果**:
- 历史帧：30帧（t=-29到t=0）
- Warmup：15步（t=6到t=20）
- 融合点：t=21, 22, 23, ..., 70（每步都融合）
- 最大化大尺度指导

## 技术细节

### 大尺度模型输入构建

无论first_fusion_point如何，大尺度模型始终使用10t间距的输入：

```python
# 在时刻F进行融合
large_input_times = [F-50, F-40, F-30, F-20, F-10]

# 从frame_cache获取这些时刻的帧
frames = [frame_cache[t] for t in large_input_times]
```

### Warmup过程

```python
# 初始状态
initial_frames: t=1, 2, 3, 4, 5
current_time = 5

# Warmup循环 (warmup_steps = F - 6)
for step in range(warmup_steps):
    current_time += 1  # t=6, 7, 8, ..., F-1
    predict with small-scale model

    if current_time in anchor_times:
        mark as anchor for large-scale model

# Warmup结束
current_time = F - 1
```

### 历史帧加载

```python
if sample_idx >= num_historical_frames:
    for t in range(historical_start, 1):  # 从historical_start到0
        offset = 1 - t
        hist_frame = dataset[sample_idx - offset]
        frame_cache[t] = hist_frame
else:
    # Fallback: 使用初始帧填充
    for t in range(historical_start, 1):
        frame_cache[t] = initial_frames[0]
```

## 错误处理

### 错误1: first_fusion_point太小

```bash
$ python evaluation_1plane_cross_scale.py --first_fusion_point 10
ValueError: first_fusion_point must be >= 11, got 10
```

**解决**: 使用 ≥ 11 的值

### 错误2: sample_idx不足

```bash
$ python evaluation_1plane_cross_scale.py --first_fusion_point 25 --sample_idx 15
WARNING: sample_idx (15) < required minimum (26)
         for first_fusion_point=25
         Historical frames may be incomplete.
```

**解决**: 增大sample_idx或增大first_fusion_point

### 边界情况: 帧缺失

代码包含自动fallback机制：
- 如果某个历史时刻的帧不存在
- 自动使用最近的可用帧
- 或使用初始帧作为fallback

## 性能考虑

### Warmup时间影响

| First Fusion Point | Warmup Steps | Warmup Time（相对） |
|-------------------|--------------|------------------|
| 21 | 15 | 0.6x |
| 31（默认） | 25 | 1.0x |
| 41 | 35 | 1.4x |
| 51 | 45 | 1.8x |

### 历史数据加载

| First Fusion Point | Historical Frames | Load Time（相对） |
|-------------------|------------------|-----------------|
| 50 | 1 | 0.05x |
| 40 | 11 | 0.55x |
| 31（默认） | 20 | 1.0x |
| 25 | 26 | 1.3x |
| 21 | 30 | 1.5x |

## 测试验证

运行测试脚本验证逻辑：

```bash
python test_first_fusion_point_logic.py
```

**测试覆盖**:
- ✓ F=25: warmup=19步, historical=26帧, anchors=[5,15]
- ✓ F=31: warmup=25步, historical=20帧, anchors=[1,11,21]（默认）
- ✓ F=40: warmup=34步, historical=11帧, anchors=[10,20,30]
- ✓ F=50: warmup=44步, historical=1帧, anchors=[10,20,30,40]
- ✓ 大尺度输入构建正确
- ✓ 与fusion_interval组合正确

## 向后兼容性

- ✅ 默认值first_fusion_point=31与原始实现完全一致
- ✅ 所有现有脚本无需修改即可继续工作
- ✅ 新参数完全可选
- ✅ Warmup行为与原实现一致（25步，t=6到t=30）

## 最佳实践

1. **选择合适的first_fusion_point**:
   - 快速实验：使用默认值31
   - 历史数据充足：可以尝试25左右
   - 历史数据有限：使用40-50

2. **与fusion_interval配合**:
   - early fusion + 频繁融合：F=25, interval=1-3
   - 默认配置：F=31, interval=10
   - late fusion + 稀疏融合：F=40, interval=10-20

3. **检查sample_idx**:
   - 始终确保 `sample_idx >= max(0, 51 - first_fusion_point)`
   - 建议留有余量（sample_idx ≥ required + 10）

4. **调试建议**:
   - 先用默认值确保基本功能正常
   - 逐步调整first_fusion_point
   - 观察warmup日志和融合点信息

## 常见问题

### Q1: first_fusion_point的最佳值是什么？

**A**: 取决于您的需求：
- **快速实验**: 31（默认）
- **最大化早期精度**: 21-25
- **最小化历史需求**: 40-50
- **平衡配置**: 30-35

### Q2: 为什么不能设置first_fusion_point < 11？

**A**: 因为大尺度模型需要5个间距为10t的输入帧。当F < 11时：
- 输入需要: [F-50, F-40, F-30, F-20, F-10]
- 如F=10: 需要[-40, -30, -20, -10, 0]
- 但t=0时才有第一个anchor，在warmup中无法收集足够的anchor

### Q3: warmup_steps = F - 6的计算逻辑是什么？

**A**:
- 初始帧：t=1,2,3,4,5，current_time=5
- 需要预测到t=F-1（第一次融合前一步）
- 从t=5到t=F-1需要(F-1)-5 = F-6步
- 示例：F=31时，warmup预测t=6到t=30，共25步

### Q4: 如何选择合适的sample_idx？

**A**: 使用公式计算最小值：
```python
min_sample_idx = max(0, 51 - first_fusion_point)
```
建议使用比最小值大10以上的sample_idx以确保安全。

### Q5: first_fusion_point会影响后续融合点吗？

**A**: 是的！后续融合点由first_fusion_point和fusion_interval共同决定：
```
fusion_points = [F, F+interval, F+2*interval, F+3*interval, ...]
```

例如：F=25, interval=5 → 融合点为[25, 30, 35, 40, ...]

## 文件清单

- `evaluation_1plane_cross_scale.py`: 主实现文件
- `test_first_fusion_point_logic.py`: 测试脚本
- `FIRST_FUSION_POINT_README.md`: 本文档

## 引用

如果您在研究中使用此功能，请引用：

```
Configurable First Fusion Point for Multi-Resolution Prediction-Correction (MR-PC)
Implementation for 1-Plane Flow Swin Transformer
```

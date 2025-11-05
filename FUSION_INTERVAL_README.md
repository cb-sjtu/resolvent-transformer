# Flexible Fusion Interval Feature

## 概述

MR-PC（Multi-Resolution Prediction-Correction）现在支持灵活的融合间隔配置。您可以通过`--fusion_interval`参数控制大尺度和小尺度模型融合的频率。

## 功能说明

### 核心概念

- **融合间隔（Fusion Interval）**: 两次连续融合之间的时间步数
- **融合点（Fusion Point）**: 执行大尺度和小尺度模型融合的时间步
- **帧缓存（Frame Cache）**: 存储所有历史帧和预测帧，用于构建大尺度模型的输入

### 工作原理

1. **历史帧加载**: 从数据集加载t=-19到t=0的所有历史ground truth帧
2. **Warmup阶段**: 小尺度模型预测t=6到t=30（25步）
3. **主循环**:
   - 在每个融合点（t=31, 31+interval, 31+2×interval, ...）:
     - 小尺度模型预测当前时刻
     - 大尺度模型使用帧缓存中的历史帧（10t间距）进行预测
     - 融合两个预测结果
   - 在融合点之间：小尺度模型单独预测（interval-1步）

### 大尺度模型输入构建

无论融合间隔如何，大尺度模型始终使用**10t间距**的输入：

```python
# 在时刻t进行融合
large_input_times = [t-50, t-40, t-30, t-20, t-10]
```

这些帧从帧缓存中获取，确保大尺度模型的输入符合其训练时的假设。

## 使用方法

### 命令行参数

```bash
python evaluation_1plane_cross_scale.py \
    --sample_idx 0 \
    --num_predictions 100 \
    --fusion_weight 0.8 \
    --fusion_interval <INTERVAL>
```

### 参数说明

- `--fusion_interval`: 融合间隔（默认值：10）
  - `1`: 每个时间步都融合（最频繁）
  - `2`: 每2个时间步融合一次
  - `5`: 每5个时间步融合一次
  - `10`: 每10个时间步融合一次（默认，向后兼容）

## 示例

### 示例1：默认融合间隔（每10t融合）

```bash
python evaluation_1plane_cross_scale.py \
    --fusion_interval 10 \
    --num_predictions 100
```

**融合点**: t=31, 41, 51, 61, 71, 81, 91, 101, 111, 121
**总融合次数**: 10次
**小尺度独立预测**: 90步

### 示例2：每步融合（最大稳定性）

```bash
python evaluation_1plane_cross_scale.py \
    --fusion_interval 1 \
    --num_predictions 50
```

**融合点**: t=31, 32, 33, ..., 80
**总融合次数**: 50次
**小尺度独立预测**: 0步
**特点**: 每步都有大尺度模型的指导，理论上长期稳定性最好

### 示例3：每5步融合（平衡）

```bash
python evaluation_1plane_cross_scale.py \
    --fusion_interval 5 \
    --num_predictions 100
```

**融合点**: t=31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126
**总融合次数**: 20次
**小尺度独立预测**: 80步
**特点**: 在性能和计算成本之间取得平衡

### 示例4：生成视频

```bash
python evaluation_1plane_cross_scale.py \
    --fusion_interval 2 \
    --num_predictions 50 \
    --generate_videos \
    --video_fps 10
```

## 技术细节

### 帧缓存结构

```python
frame_cache = {
    -19: tensor(...),  # 历史GT
    -18: tensor(...),
    ...
    0: tensor(...),
    1: tensor(...),    # 初始帧
    ...
    30: tensor(...),   # Warmup结束
    31: tensor(...),   # 第一次融合
    ...
}
```

### 融合时间点计算

```python
# Warmup结束后，从t=30开始
current_time = 30

while len(predictions) < num_predictions:
    # 融合点
    current_time += 1  # t=31, 32, 33, ...

    # 执行融合
    perform_fusion(current_time)

    # 小尺度继续预测 (fusion_interval - 1) 步
    for _ in range(fusion_interval - 1):
        current_time += 1
        small_scale_predict(current_time)
```

### 大尺度输入选择

```python
def construct_large_scale_input(current_time, frame_cache):
    """构建大尺度模型输入（10t间距）"""
    times = [
        current_time - 50,
        current_time - 40,
        current_time - 30,
        current_time - 20,
        current_time - 10,
    ]

    frames = [frame_cache[t] for t in times]
    return stack(frames)
```

## 性能影响

### 计算成本

| Fusion Interval | 大尺度调用次数（100步） | 相对成本 |
|----------------|---------------------|---------|
| 1              | 100次               | 10x     |
| 2              | 50次                | 5x      |
| 5              | 20次                | 2x      |
| 10（默认）      | 10次                | 1x      |

### 内存使用

帧缓存大小：
- 历史帧：20帧（t=-19到t=0）
- 初始帧：5帧（t=1到t=5）
- Warmup：25帧（t=6到t=30）
- 预测帧：num_predictions帧

**总计**: 50 + num_predictions 帧

对于256×256分辨率，3通道，float32：
- 每帧：256 × 256 × 3 × 4 bytes ≈ 0.75 MB
- 150帧（num_predictions=100）：约112.5 MB

## 预期效果

### Fusion Interval = 1
- **优点**: 最频繁的大尺度指导，理论上长期稳定性最好
- **缺点**: 计算成本最高（10倍）
- **适用场景**: 需要最高精度的长期预测

### Fusion Interval = 5
- **优点**: 平衡性能和成本
- **缺点**: 介于两者之间
- **适用场景**: 实际应用中的推荐选择

### Fusion Interval = 10（默认）
- **优点**: 计算效率最高
- **缺点**: 融合频率最低
- **适用场景**: 快速实验、资源受限环境

## 测试验证

运行测试脚本验证逻辑：

```bash
python test_fusion_interval_logic.py
```

**测试覆盖**:
- ✓ fusion_interval=10: 融合点正确
- ✓ fusion_interval=5: 融合点正确
- ✓ fusion_interval=1: 融合点正确
- ✓ fusion_interval=2: 融合点正确
- ✓ 帧缓存包含所有必需帧

## 向后兼容性

- 默认值`fusion_interval=10`与原始实现完全一致
- 所有现有脚本和配置无需修改即可继续工作
- 新功能完全可选

## 注意事项

1. **最小融合间隔**: 建议≥1，理论上可以支持任意正整数
2. **历史帧要求**: sample_idx必须≥20，以确保有足够的历史ground truth
3. **大尺度模型假设**: 大尺度模型始终使用10t间距输入，这是其训练时的假设
4. **边界情况**: 代码包含回退逻辑，当帧缓存中缺少某个时间点的帧时，会使用最近的可用帧

## 实现文件

- **主文件**: `evaluation_1plane_cross_scale.py`
- **测试脚本**: `test_fusion_interval_logic.py`
- **文档**: `FUSION_INTERVAL_README.md`（本文件）

## 常见问题

### Q1: fusion_interval=1时，为什么不是从t=1开始融合？

**A**: Warmup阶段（t=1到t=30）用于建立初始状态和收集anchor帧。第一次融合发生在t=31，此时我们有足够的历史信息供大尺度模型使用。

### Q2: 大尺度模型的输入为什么总是10t间距？

**A**: 大尺度模型是用10t间距的数据训练的。改变输入间距会违反模型假设，导致性能下降。融合间隔只影响融合的频率，不影响大尺度模型的输入构建。

### Q3: fusion_interval=1时计算会慢多少？

**A**: 理论上慢10倍（相比默认的fusion_interval=10），因为每步都需要调用大尺度模型。实际速度取决于硬件和模型大小。

### Q4: 如何选择合适的fusion_interval？

**A**:
- **快速实验**: 10（默认）
- **生产环境**: 5
- **最高精度**: 1或2
- **资源受限**: 10或更大

### Q5: 帧缓存会不会导致内存溢出？

**A**: 对于典型的预测长度（100-200步），内存占用可控（<500MB）。如果需要预测更长序列（>1000步），可能需要考虑内存优化。

## 引用

如果您在研究中使用此功能，请引用：

```
Multi-Resolution Prediction-Correction (MR-PC) with Flexible Fusion Intervals
Implementation for 1-Plane Flow Swin Transformer
```

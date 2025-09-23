# 3平面4通道训练监测系统

## 🎯 功能概述

为3平面4通道Flow Swin Transformer训练添加了完整的per-channel metrics监测，支持每个通道、每个物理字段、每个平面的独立loss和相对误差监测。

## 📊 监测指标

### 1. **Per-Channel Metrics (12个通道)**
每个通道独立计算以下指标：
- **MSE Loss**: `{channel_name}_mse`
- **MAE Loss**: `{channel_name}_mae`
- **相对误差**: `{channel_name}_rel_error`

**通道命名**:
```
plane0_u_y29, plane0_v_y29, plane0_w_y29, plane0_p_y29,
plane1_u_y54, plane1_v_y54, plane1_w_y54, plane1_p_y54,
plane2_u_y75, plane2_v_y75, plane2_w_y75, plane2_p_y75
```

### 2. **Field-wise Averages (4个物理字段)**
跨所有平面的字段平均：
- **速度字段**: `field_u_avg_mse`, `field_u_avg_rel_error`
- **速度字段**: `field_v_avg_mse`, `field_v_avg_rel_error`
- **速度字段**: `field_w_avg_mse`, `field_w_avg_rel_error`
- **压力字段**: `field_p_avg_mse`, `field_p_avg_rel_error`

### 3. **Plane-wise Averages (3个y平面)**
跨所有字段的平面平均：
- **Plane 0**: `plane0_y29_avg_mse`, `plane0_y29_avg_rel_error`
- **Plane 1**: `plane1_y54_avg_mse`, `plane1_y54_avg_rel_error`
- **Plane 2**: `plane2_y75_avg_mse`, `plane2_y75_avg_rel_error`

## 🔧 技术实现

### 核心功能
1. **`compute_per_channel_metrics()`**: 计算所有per-channel指标
2. **Training监测**: 每100步记录detailed metrics，每步记录summary metrics
3. **Validation监测**: 每个epoch记录所有metrics
4. **WandB集成**: 自动上传到wandB dashboard

### 代码位置
- **主要实现**: `src/plmodules/flow_swin_2d_lit_module.py`
- **配置启用**: `configs/train_flow_swin_3plane.yaml`

## 📈 WandB监测面板

### Training Metrics
```
train/plane0_u_y29_mse
train/plane0_v_y29_mse
...
train/field_u_avg_rel_error
train/field_v_avg_rel_error
train/field_w_avg_rel_error
train/field_p_avg_rel_error
```

### Validation Metrics
```
val/plane0_u_y29_mse
val/plane0_v_y29_mse
...
val/field_u_avg_rel_error
val/field_v_avg_rel_error
val/field_w_avg_rel_error
val/field_p_avg_rel_error
```

## 🚀 使用方法

### 1. **启用监测**
配置文件已默认启用：
```yaml
# configs/train_flow_swin_3plane.yaml
enable_per_channel_metrics: true
```

### 2. **开始训练**
```bash
./start_train_flow_swin_3plane_conda.sh
```

### 3. **监测训练**
在WandB dashboard中查看：
- **Per-channel trends**: 每个通道的学习曲线
- **Field comparisons**: 不同物理字段的学习效果对比
- **Plane analysis**: 不同y平面的预测精度对比

## 💡 监测策略

### Training期间
- **高频监测**: Field averages每步记录
- **详细监测**: 所有metrics每100步记录
- **目的**: 避免log spam，同时保持重要信息可见

### Validation期间
- **完整监测**: 所有metrics每epoch记录
- **目的**: 深入分析模型性能

## 🎯 分析建议

### 1. **收敛诊断**
- 观察不同通道的收敛速度
- 识别学习困难的通道
- 对比不同物理字段的学习效果

### 2. **性能对比**
- **速度字段** (u,v,w): 通常更易学习
- **压力字段** (p): 可能需要更多训练
- **不同平面**: y=54平面可能最具代表性

### 3. **调优指导**
- 基于per-channel性能调整loss权重
- 针对性改进困难通道的预测
- 验证模型在不同物理量上的generalization

## 🔍 故障排除

### 如果metrics显示异常
1. 检查数据归一化是否正确
2. 验证通道映射是否匹配数据
3. 确认模型输出通道数=12

### 如果WandB不显示metrics
1. 确认 `enable_per_channel_metrics: true`
2. 检查WandB连接状态
3. 验证Lightning logging配置

## 📋 输出示例

Training log示例：
```
train/field_u_avg_rel_error: 0.125
train/field_v_avg_rel_error: 0.089
train/field_w_avg_rel_error: 0.156
train/field_p_avg_rel_error: 0.078
```

这套监测系统为3平面4通道的复杂流场预测提供了全面、精细的训练监控能力！

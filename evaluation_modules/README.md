# 模块化评估系统 (Modular Evaluation System)

这是一个重构后的流场预测模型评估系统，具有模块化设计和强大的时间序列监测功能。

## 📁 项目结构

```
evaluation_modules/
├── __init__.py                 # 模块初始化
├── base_evaluator.py          # 基础评估器类
├── flow_evaluator.py          # 流场模型评估器实现
├── time_series_monitor.py     # ⭐ 时间序列点监测（新功能）
├── metrics.py                 # 评估指标计算
├── visualization.py           # 可视化功能
├── video_creation.py          # 视频生成
├── utils.py                   # 工具函数
└── README.md                  # 本文档

# 主要脚本
evaluation_new.py               # 新的主评估脚本
```

## 🚀 主要功能

### ⭐ 时间序列点监测 (新功能)
- **可配置监测点**: 在计算域中选择任意位置的监测点
- **多组件监测**: 同时监测 u, v, w 速度分量和速度大小
- **多模式支持**: 支持训练/测试、自回归/teacher forcing 模式
- **自动可视化**: 生成时间序列曲线图
- **数据导出**: 保存为 CSV 格式便于后续分析

### 📊 综合评估指标
- 基础指标：MSE, MAE, RMSE
- 相对误差：传统相对误差、RMS归一化相对误差、范围归一化相对误差
- 速度大小指标：专门针对流场速度大小的评估

### 🎨 多模态可视化
- 单帧对比图：预测 vs 真实值
- 多步预测演化图
- 误差分布可视化
- 时间序列曲线图

### 🎬 视频生成
- 预测vs真实值对比视频
- 误差演化动画
- 支持 MP4/GIF 格式

## 🔧 使用方法

### 基础用法

```bash
# 使用默认配置
python evaluation_new.py /path/to/checkpoint.ckpt

# 自定义监测点
python evaluation_new.py /path/to/checkpoint.ckpt --custom-points

# 保存预测结果
python evaluation_new.py /path/to/checkpoint.ckpt --save-predictions

# 完整配置
python evaluation_new.py /path/to/checkpoint.ckpt \
    --custom-points \
    --save-predictions \
    --num-samples 5 \
    --num-future-steps 20 \
    --output-dir my_evaluation_results
```

### 自定义监测点

在 `evaluation_new.py` 中修改 `create_custom_monitor_points()` 函数：

```python
def create_custom_monitor_points():
    custom_points = [
        (50, 50),    # 位置1: z=50, x=50
        (100, 100),  # 位置2: z=100, x=100
        (150, 150),  # 位置3: z=150, x=150
        # 添加更多点...
    ]
    return custom_points
```

### 编程接口使用

```python
from evaluation_modules import FlowModelEvaluator

# 创建评估器
evaluator = FlowModelEvaluator(
    checkpoint_path="path/to/checkpoint.ckpt",
    model_config=config,
    monitor_points=[(64, 64), (128, 128), (192, 192)]  # 自定义点
)

# 加载模型和数据
evaluator.load_model_and_datasets()

# 评估单个样本
evaluator.evaluate_sample(sample_idx=0, num_future=15)

# 生成时间序列分析
evaluator.create_time_series_summary()
```

## 📈 输出结果

运行评估后，会在指定的输出目录生成：

```
evaluation_results/
├── plots/                      # 静态可视化图像
│   ├── comparison_*.png        # 单帧对比图
│   └── multi_step_*.png        # 多步预测图
├── videos/                     # 视频文件
│   ├── autoregressive_*.mp4    # 自回归预测视频
│   └── teacher_forcing_*.mp4   # Teacher forcing视频
├── time_series_plots/          # ⭐ 时间序列图像
│   ├── time_series_point_*.png # 单点时间序列
│   └── time_series_all_*.png   # 所有点综合图
├── time_series_data/           # ⭐ 时间序列数据
│   ├── time_series_test_ar.csv # 自回归模式数据
│   └── time_series_test_tf.csv # Teacher forcing数据
├── predictions/                # 预测结果(可选)
│   ├── test/ar/               # 自回归预测H5文件
│   └── test/tf/               # Teacher forcing预测H5文件
└── time_series_report.md       # ⭐ 监测报告
```

## 📊 时间序列数据格式

CSV文件包含以下列：
- `timestep`: 时间步
- `u_point0_z64_x64`: 第0个监测点(64,64)的u分量
- `v_point0_z64_x64`: 第0个监测点(64,64)的v分量
- `w_point0_z64_x64`: 第0个监测点(64,64)的w分量
- `mag_point0_z64_x64`: 第0个监测点(64,64)的速度大小
- ...（为每个监测点重复）

## 🔧 模块说明

### TimeSeriesMonitor 类
```python
# 创建监测器
monitor = TimeSeriesMonitor(monitor_points=[(50,50), (100,100)])

# 记录时间步数据
monitor.record_timestep(data_tensor, split="test", mode="ar", timestep=0)

# 生成所有图像
monitor.generate_all_plots(output_dir)

# 保存CSV数据
monitor.save_data_csv(output_dir)
```

### FlowMetrics 类
```python
metrics = FlowMetrics()

# 计算综合指标
results = metrics.compute_comprehensive_metrics(pred, target, ["u", "v", "w"])

# 格式化输出
summary = metrics.format_metrics_summary(results)
```

### FlowVisualizer 类
```python
visualizer = FlowVisualizer(output_dir)

# 创建对比图
plot_path = visualizer.plot_single_frame_comparison(pred, target, 0, 0)

# 创建多步预测图
plot_path = visualizer.plot_multi_step_prediction(predictions, ground_truth, 0)
```

## 🔄 从旧系统迁移

如果你之前使用 `evaluation.py`，迁移到新系统很简单：

### 旧用法：
```bash
python evaluation.py checkpoint.ckpt --save-predictions
```

### 新用法：
```bash
python evaluation_new.py checkpoint.ckpt --save-predictions --custom-points
```

主要区别：
1. **更多功能**: 新系统增加了时间序列监测
2. **更好的模块化**: 代码组织更清晰，易于维护和扩展
3. **可配置监测点**: 可以自由选择感兴趣的监测位置
4. **更丰富的输出**: 更多的可视化和数据导出选项

## 🚨 注意事项

1. **内存使用**: 监测大量点可能增加内存使用，建议监测点数量控制在20个以内
2. **文件大小**: 启用预测保存会产生大量H5文件，确保有足够磁盘空间
3. **依赖项**: 确保安装了所有必要的包：matplotlib, torch, h5py, pandas等

## 🤝 扩展开发

系统采用模块化设计，易于扩展：

1. **添加新的监测功能**: 在 `time_series_monitor.py` 中扩展
2. **增加评估指标**: 在 `metrics.py` 中添加新的计算方法
3. **创建新的可视化**: 在 `visualization.py` 中实现
4. **支持新的模型**: 继承 `BaseFlowEvaluator` 类

## 📝 更新日志

- **v2.0**: 完全重构，添加时间序列监测功能
- **v1.0**: 原始 evaluation.py 版本

---

有问题或建议请联系开发团队！

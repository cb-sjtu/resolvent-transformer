# Flow Data Visualization

这个文件夹包含了用于可视化预处理流场数据的脚本和输出结果。

## 目录结构

```
visualization/
├── scripts/               # 可视化脚本
│   └── create_flow_videos.py  # 创建流场演化视频的主脚本
├── videos/               # 输出的视频文件将保存在这里
└── README.md             # 本文档
```

## 功能特性

### create_flow_videos.py

这个脚本可以从预处理的流场数据创建时间演化视频，包括：

1. **组合视频** (`flow_combined_evolution.mp4`): 显示 u, v, w 三个速度分量和速度大小的 2x2 网格视图
2. **单通道视频**: 为每个速度分量创建独立视频
   - `flow_u_evolution.mp4`: u 分量演化
   - `flow_v_evolution.mp4`: v 分量演化
   - `flow_w_evolution.mp4`: w 分量演化
3. **速度大小视频** (`flow_magnitude_evolution.mp4`): 只显示速度大小的演化

## 使用方法

### 基本用法

```bash
cd visualization/scripts
python create_flow_videos.py
```

### 高级选项

```bash
python create_flow_videos.py \
    --data_dir /path/to/preprocessed_flow \
    --output_dir /path/to/output/videos \
    --max_frames 50 \
    --fps 15 \
    --fields u v w \
    --percentiles 2 98
```

### 参数说明

- `--data_dir`: 包含预处理流场数据 H5 文件的目录 (默认: `../../data/preprocessed_flow`)
- `--output_dir`: 视频输出目录 (默认: `../videos`)
- `--max_frames`: 处理的最大帧数，0表示处理所有帧 (默认: 100)
- `--fps`: 输出视频的帧率 (默认: 10)
- `--fields`: 要可视化的场名称 (默认: ["u", "v", "w"])
- `--percentiles`: 颜色条范围的百分位数 (默认: [1, 99])

## 依赖项

脚本需要以下Python包：

- `numpy`: 数值计算
- `matplotlib`: 绘图和动画
- `h5py`: HDF5文件读取
- `tqdm`: 进度条显示

## 输出格式

### 视频格式
- 优先输出 MP4 格式 (需要 ffmpeg)
- 如果MP4失败，会自动降级到 GIF 格式 (需要 pillow)

### 视频内容
- **颜色映射**:
  - u, v, w 分量使用 'viridis' 色彩映射
  - 速度大小使用 'plasma' 色彩映射
- **颜色范围**: 基于数据的1-99百分位数确定，确保所有帧的颜色一致性
- **时间信息**: 每帧显示当前时间步信息

## 数据要求

脚本期望输入数据为：

1. **文件格式**: HDF5 (.h5) 文件
2. **文件命名**: 遵循 `u-v-w_scale*_yslice*_*.h5` 模式
3. **数据结构**:
   - 数据集名称: `"data"`
   - 数据形状: `(C, H, W)` 其中 C≥3 (u, v, w 三个通道)
4. **文件顺序**: 文件按时间顺序命名和排序

## 故障排除

### 常见问题

1. **没有找到匹配文件**
   - 检查 `--data_dir` 路径是否正确
   - 确认文件名符合预期的命名模式

2. **视频保存失败**
   - 安装 ffmpeg: `sudo apt install ffmpeg` (Ubuntu/Debian)
   - 或使用 GIF 格式作为备选

3. **内存不足**
   - 减少 `--max_frames` 参数
   - 处理较小的数据集子集

### 性能优化

- 对于大型数据集，建议先使用较小的 `--max_frames` 值进行测试
- 可以调整 `--percentiles` 来获得更好的可视化效果
- 增加 `--fps` 值可以获得更流畅的动画，但文件会更大

## 示例输出

运行脚本后，你将在输出目录中看到：

```
videos/
├── flow_combined_evolution.mp4    # 组合视图 (2x2网格)
├── flow_u_evolution.mp4           # u分量独立视频
├── flow_v_evolution.mp4           # v分量独立视频
├── flow_w_evolution.mp4           # w分量独立视频
└── flow_magnitude_evolution.mp4   # 速度大小视频
```

每个视频都显示了流场随时间的演化，带有适当的颜色条和时间步信息。

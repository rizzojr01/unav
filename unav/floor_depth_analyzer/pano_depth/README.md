# Panorama Depth Estimation & 3D Reconstruction

全景图深度估计方法研究，将深度图转换为3D点云/俯视图(floor plan)进行可视化对比。

## 目录结构

```
pano_depth/
├── pano_to_3d.py              # 核心：全景深度图 → 3D点云/mesh转换
├── visualize_3d.py            # Open3D可视化 & floor plan生成
├── compare_three_models.py    # 对比 GT vs 我们训练 vs 论文预训练
├── compare_with_gt.py         # GT vs 预测对比
├── compare_depth_methods.py   # 多方法对比
└── README.md
```

## 相关外部目录

```
unav/tmp/
├── Depth-Anywhere/            # Depth-Anywhere代码库
│   ├── checkpoints/           # 论文预训练模型
│   │   └── UniFuse/
│   │       ├── UniFuse.pth                  # 原始预训练
│   │       ├── UniFuse_st3d.pth             # Structured3D微调
│   │       └── UniFuse_SpatialAudioGen.pth  # Depth-Anywhere论文模型
│   ├── ckpts/                 # 我们的训练checkpoint
│   │   └── UniFuse/unifuse_st3d_sf3d/ckpt_*.pth
│   └── run_unifuse_inference.py  # 独立推理脚本
└── depth_comparison/          # 对比结果输出
    ├── pred_depth.npy
    ├── pred_depth_paper.npy
    └── three_model_comparison.png
```

## 核心发现

### 深度估计对比结果 (Structured3D scene_00000)

| 模型 | 深度范围 | MAE |
|------|---------|-----|
| Ground Truth | 0.0 - 6.2m | - |
| DA_Retrained (st3d_sf3d) | 0.6 - 4.8m | 1.16m |
| DA_Original (SpatialAudioGen) | 0.5 - 4.4m | 1.45m |
| DA3 (cubemap stitch) | 0.0 - 2.3m | 0.77m* |

*DA3 MAE低是因为scale alignment后深度被压缩，不代表真实性能更好

**关键发现：**
1. DA_Retrained在ST3D+SF3D上重新训练后MAE更低(1.16m vs 1.45m)
2. 所有方法都**低估最大深度** (预测~2-5m vs GT 6.2m)
3. Floor plan重建出现**弯曲畸变** - 直墙变曲线，因为远端深度被低估
4. DA3深度范围被严重压缩，floor plan比例失真

## 快速使用

### 1. 运行Depth-Anywhere推理

```bash
cd /home/unav/Desktop/unav/unav/tmp/Depth-Anywhere

# 使用我们训练的模型
python run_unifuse_inference.py \
  --rgb <RGB_PATH> \
  --ckpt ckpts/UniFuse/unifuse_st3d_sf3d/ckpt_100.pth \
  --output output.npy

# 使用论文预训练模型
python run_unifuse_inference.py \
  --rgb <RGB_PATH> \
  --ckpt checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth \
  --output output_paper.npy
```

### 2. 深度图转3D点云

```bash
cd /home/unav/Desktop/unav
python -m unav.floor_depth_analyzer.pano_depth.pano_to_3d \
  --rgb <RGB_PATH> \
  --depth <DEPTH_NPY_OR_PNG> \
  --output output_3d/
```

### 3. 生成Floor Plan对比

```bash
cd /home/unav/Desktop/unav
python -m unav.floor_depth_analyzer.pano_depth.compare_three_models
```

## 核心转换逻辑

```python
def equirectangular_to_3d(depth, rgb):
    """全景深度图 → 3D点云

    坐标系: Y-up, -Z forward (OpenGL)

    转换公式:
    theta = (u/W - 0.5) * 2*pi   # 经度 [-pi, pi]
    phi = (0.5 - v/H) * pi        # 纬度 [pi/2, -pi/2]

    x = depth * cos(phi) * sin(theta)
    y = depth * sin(phi)
    z = -depth * cos(phi) * cos(theta)
    """
```

## 数据集位置

```
/mnt/data/floorplan-reconstruction/public_data/
├── stru3d/panorama/Structured3D/     # Structured3D (有GT depth)
├── stanford2d3d_full/                 # Stanford2D3D (已完成~450GB)
├── 360monodepth_mp3d/                 # 360MonoDepth MP3D (下载中)
└── structured3d_panorama_full/        # Structured3D完整版 (下载中)
```

### Structured3D 数据格式
- RGB: `scene_XXXXX/2D_rendering/XXXXXX/panorama/full/rgb_rawlight.png`
- Depth: `scene_XXXXX/2D_rendering/XXXXXX/panorama/full/depth.png` (uint16, mm)

## 推理注意事项

1. 需要从Depth-Anywhere目录运行(避免相对导入问题)
2. `face_w=256` 不是 `cube_length`
3. checkpoint格式: 论文的是直接state_dict, 我们训练的包在`{'model': ...}`里

## tmux下载进度

```bash
tmux ls
# download_360mono - 360MonoDepth (~4GB / 88GB)
# download_stanford - Stanford2D3D (已完成 ~450GB)
# download_stru3d - Structured3D (~35GB / 200GB)
```

## 评估指标

| 指标 | 含义 | 方向 |
|------|------|------|
| MRE | Mean Relative Error | 越低越好 |
| MAE | Mean Absolute Error (m) | 越低越好 |
| Abs_Rel | \|pred-gt\|/gt 的均值 | 越低越好 |
| Sq_Rel | (pred-gt)²/gt 的均值 | 越低越好 |
| RMS | 均方根误差 | 越低越好 |
| a1/a2/a3 | max(pred/gt, gt/pred) < 1.25^n 的比例 | 越高越好 |

## 下一步计划

1. 等待数据集下载完成
2. 在更多样本上验证深度估计质量
3. 研究深度低估问题 - 可能需要scale alignment
4. 尝试其他模型 (BiFuseV2, HoHoNet, EGFormer)

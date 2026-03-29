# 整体架构

HDpainter 采用**两阶段**生成式建模流程，将高噪声的 Visium HD 空间转录组数据提升至接近 Xenium 的质量。

## 阶段划分

```
Stage 1 (已实现)  VisiumVAE
                  将基因表达从 n_genes 维压缩到 latent_dim=50 维的潜空间
                  ↓
Stage 2 (待实现)  Latent Diffusion Model (LDM)
                  在潜空间对降级表达进行去噪，以细胞核 mask 为条件
```

## 代码目录结构

```
code/
├── src/                        # 核心模块（Jieke）
│   ├── __init__.py
│   ├── dataset.py              # 数据集 & DataLoader
│   └── models/
│       ├── __init__.py
│       └── vae.py              # VisiumVAE + vae_loss
│
├── scripts/                    # 入口脚本（Jieke）
│   ├── train_vae.py            # VAE 训练脚本
│   └── slurm_train_vae.sh      # SLURM 批处理作业
│
├── preprocess/                 # 数据预处理笔记本（Zhuofan）
│   ├── dataset_process.ipynb       # Xenium → 稀疏 tile
│   ├── degrade_process.ipynb       # 合成降级流程
│   ├── Systhesis_process.ipynb     # 数据合成
│   ├── stpainter_process.ipynb     # stPainter 插补
│   └── reference_visium_process.ipynb
│
├── dataset_loader.ipynb        # 数据加载演示（Zhuofan）
├── evaluate_on_sys.ipynb       # 评估 & 结果重构（Zhuofan）
│
├── docs/                       # ← 本文档目录
├── pyproject.toml              # uv 项目配置
└── logs/                       # SLURM 输出日志
```

## 数据流

```
原始数据 (AnnData .h5ad)
      │
      ▼ [preprocess/dataset_process.ipynb]
稀疏 tile (.pt 文件, 512×512 sliding window)
每个 tile 含:
   input_expr     [H, W, C] — 降级表达 (Visium HD 输入)
   input_nuclei   [H, W, 1] — 细胞核 mask
   target_expr    [H, W, C] — 真值表达 (Xenium)
   target_cell_id [H, W, 1] — 完整细胞 mask (真值)
      │
      ▼ [src/dataset.py → SpatialTranscriptomicsDataset]
PyTorch DataLoader  shape: [B, C, H, W]
      │
      ▼ [src/models/vae.py → VisiumVAE]  (Stage 1)
潜向量 z  [B, latent_dim, H, W]
      │
      ▼ [src/models/ldm.py → LDM]        (Stage 2, 待实现)
去噪后的潜向量
      │
      ▼ VAE Decoder
重建表达 [B, C, H, W]
```

## 模块实现状态

| 模块 | 文件 | 状态 |
|------|------|------|
| Dataset / DataLoader | `src/dataset.py` | ✅ 完成 |
| VAE 模型 | `src/models/vae.py` | ✅ 完成 |
| VAE 训练脚本 | `scripts/train_vae.py` | ✅ 完成 |
| SLURM 提交脚本 | `scripts/slurm_train_vae.sh` | ✅ 完成 |
| Latent Diffusion Model | `src/models/ldm.py` | ⏳ 待实现 |
| LDM 训练脚本 | `scripts/train_ldm.py` | ⏳ 待实现 |
| 推理脚本 | `scripts/infer.py` | ⏳ 待实现 |

## 依赖环境

| 包 | 用途 |
|----|------|
| `torch` (CUDA 12.4 build) | 深度学习框架 |
| `torchvision` | 图像处理工具 |
| `numpy`, `scipy` | 数值计算 |
| `scanpy`, `anndata` | 空间转录组数据处理 |
| `tqdm` | 进度条 |
| `ipykernel` | Jupyter 支持 |

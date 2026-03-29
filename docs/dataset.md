# 数据集与 DataLoader

**文件**: `src/dataset.py`

## `SpatialTranscriptomicsDataset`

### 功能

加载经预处理生成的稀疏 `.pt` tile 文件，自动检测基因数量，并以 `[C, H, W]` 格式返回张量。

### 构造

```python
dataset = SpatialTranscriptomicsDataset(data_dir="/path/to/tiles/")
```

`data_dir` 中应包含若干 `.pt` 文件（由 `preprocess/dataset_process.ipynb` 生成）。  
构造时会自动从第一个 tile 推断 `n_genes`，并赋值到 `dataset.n_genes`。

### 每个 tile 的结构

每个 `.pt` 文件是一个字典，包含以下稀疏 COO 张量（原始形状 `[H, W, C]`）：

| 键 | 原始形状 | 含义 |
|----|----------|------|
| `input_expr` | `[512, 512, C]` | 降级后的 Visium HD 表达（模型输入） |
| `input_nuclei` | `[512, 512, 1]` | 细胞核 mask |
| `target_expr` | `[512, 512, C]` | 清洁表达（来自 Xenium，真值） |
| `target_cell_id` | `[512, 512, 1]` | 完整细胞 mask（真值，用于计算 mask 损失） |

### `__getitem__` 返回

`Dataset` 在 `__getitem__` 中自动完成：
1. `to_dense()` — 稀疏 COO → 稠密张量
2. `.permute(2, 0, 1)` — `[H, W, C]` → `[C, H, W]`（适配 PyTorch 卷积网络）
3. `.float()` 转换（仅对 `input_nuclei` 和 `target_cell_id`）

返回字典 `dict[str, torch.Tensor]`，键同上。

### DataLoader 构建

```python
from src.dataset import build_dataloader

loader = build_dataloader(
    data_dir="path/to/tiles/",
    batch_size=4,
    shuffle=True,
    num_workers=8,
)
```

`build_dataloader` 内部硬编码以下配置：
- `pin_memory=True` — 加速 GPU 数据传输
- `multiprocessing_context="spawn"` — 避免 CUDA 上下文在 DataLoader worker fork 时出现的错误（当 `num_workers > 0` 时自动启用）

> **注意**: 在提交 SLURM 作业时，`num_workers` 应与 `--cpus-per-task` 保持一致（默认建议 4-8）。

## 错误处理

| 情形 | 抛出异常 |
|------|----------|
| `data_dir` 中找不到 `.pt` 文件 | `FileNotFoundError` |

## 数据目录结构（示例）

```
data/
├── NSCLC/           ← 肺癌（最小数据集，建议初始测试）
│   ├── tile_0000.pt
│   ├── tile_0001.pt
│   └── ...
├── PRAD/            ← 前列腺癌
└── CESC/            ← 宫颈癌（最大数据集）
```

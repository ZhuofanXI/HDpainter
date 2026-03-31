# LDM 训练流程

**入口文件**: `scripts/train_ldm.py`  
**SLURM 脚本**: `scripts/slurm_train_ldm.sh`  
**模型定义**: `src/models/ldm.py`

---

## 1. 数据流向

```
SVD_CESC/tile_XXXX.pt        ← build_dataloader (src/dataset.py)
         │
         ▼
z_0    = target_expr  [B, 64, H, W]   — 干净 SVD 特征（目标）
z_cond = input_expr   [B, 64, H, W]   — 退化 SVD 特征（条件）
cell_id= target_cell_id [B, 1, H, W]  — 细胞实例掩码（用于 loss masking）
```

> **注意**：输入的 `.pt` tile 中，SVD 特征已经过 `StandardScaler` 归一化（均值≈0，标准差≈0.6），`dataset.py` 不做额外变换。

---

## 2. 噪声调度（Noise Schedule）

采用线性 beta 调度（`make_linear_schedule`）：

| 超参数 | 值 |
|--------|-----|
| `T` | 1000（总扩散步数） |
| `beta_start` | `1e-4` |
| `beta_end` | `0.02` |

前向扩散公式：

$$z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

---

## 3. 模型：`ConditionalUNet`（`src/models/ldm.py`）

### 架构概览

| 组件 | 说明 |
|------|------|
| **输入** | `[Z_t, Z_cond]` 在通道维拼接 → 128 通道 |
| **编码器** | 4 级下采样，通道数 `[64, 128, 256, 512]`（由 `base_ch=64`, `ch_mult=[1,2,4,8]` 配置） |
| **每级** | `num_res_blocks=2` 个 `ResBlock`（GroupNorm + SiLU + Conv） |
| **瓶颈** | `ResBlock → SelfAttentionBlock（8头）→ ResBlock` |
| **解码器** | 对称上采样，skip connection 拼接 |
| **噪声头** | `1×1 Conv → 64 通道 ε̂`，用于 Huber diffusion loss |
| **边界头** | 先由 ε̂ 估计 Z̃_0，再经 `3×3 Conv → SiLU → 1×1 Conv → 1 通道` 输出边界 logits |

### GroupNorm

所有 `ResBlock` 使用 `num_groups=32`，不使用 BatchNorm（对小 batch size 不稳定）。

### 参数量（默认配置）

默认超参数下约 **248M 参数**（视 `base_ch` 和 `ch_mult` 而定）。运行训练脚本时会打印实际参数量。

---

## 4. 损失函数

所有损失**仅在 `target_cell_id ≠ 0` 的细胞区域计算**（背景区域不纳入梯度）。

### 4.1 扩散损失（Diffusion Loss）

$$\mathcal{L}_{\text{diff}} = \frac{\sum_{i \in \text{cell}} \text{Huber}(\hat{\varepsilon}_i,\varepsilon_i)}{|\text{cell pixels}| \times 64}$$

使用 PyTorch 的 `F.huber_loss`，对 64 个特征通道的细胞像素取均值。

### 4.2 边界损失（Boundary Loss，时间门控）

$$\mathcal{L}_{\text{bound}} = \text{BCE}(\hat{B}, B_{\text{gt}}) + \text{Dice}(\hat{B}, B_{\text{gt}})$$

边界真值 $B_{\text{gt}}$ 由 `cell_id_to_boundary()` 从实例掩码实时生成（4-连通邻域差异）。

**时间门控权重**（`lambda2`）：

$$\lambda_2(t) = \sigma\big((t_{\text{thresh}} - t) \times \text{slope}\big)$$

- 当 $t > t_{\text{thresh}}=200$ 时，$\lambda_2 \approx 0$（仅做扩散去噪）
- 当 $t \ll 200$ 时，$\lambda_2 \approx 1$（激活边界损失）

### 4.3 总损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda_2(t) \cdot \mathcal{L}_{\text{bound}}$$

---

## 5. 训练配置

### 优化器

| 项目 | 配置 |
|------|------|
| 优化器 | `AdamW`，`lr=1e-4`，`weight_decay=1e-4` |
| 混合精度 | `torch.cuda.amp.GradScaler` + `autocast` |
| 梯度裁剪 | `clip_grad_norm_`，最大范数 = 1.0 |

### 检查点

- 每 epoch 覆盖保存 `checkpoints/ldm_cesc/latest.pt`
- 每 `--save_every`（默认10）个 epoch 保存命名检查点 `epoch_XXXX.pt`
- 支持从 `latest.pt` 自动续训（若文件存在则自动加载）

---

## 6. CLI 超参数

```bash
uv run python -u scripts/train_ldm.py \
    --data_dir  /ibex/user/wuj0c/Projects/RNA/HDpainter/data/SVD_CESC \
    --ckpt_dir  /ibex/user/wuj0c/Projects/RNA/HDpainter/checkpoints/ldm_cesc \
    --epochs        200   \
    --batch_size    2     \
    --num_workers   8     \
    --lr            1e-4  \
    --save_every    10    \
    --T             1000  \
    --t_thresh      200   \
    --base_ch       64    \
    --ch_mult       1 2 4 8 \
    --num_res_blocks 2
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | — | SVD 投影 tile 目录（`.pt` 文件） |
| `--ckpt_dir` | — | 检查点输出目录 |
| `--epochs` | 200 | 总训练轮数 |
| `--batch_size` | 2 | 每 GPU batch size（A100，512×512 tile） |
| `--num_workers` | 4 | DataLoader 工作进程数，建议与 SLURM `--cpus-per-task` 一致 |
| `--lr` | 1e-4 | 学习率 |
| `--save_every` | 10 | 每 N epoch 保存命名检查点 |
| `--T` | 1000 | 扩散总步数 |
| `--t_thresh` | 200 | 边界损失激活阈值（低于此步数激活） |
| `--base_ch` | 64 | U-Net 基础通道数 |
| `--ch_mult` | 1 2 4 8 | 各级通道倍率 |
| `--num_res_blocks` | 2 | 每级 ResBlock 数量 |

---

## 7. SLURM 提交（IBEX）

```bash
cd /ibex/user/wuj0c/Projects/RNA/HDpainter/code
sbatch scripts/slurm_train_ldm.sh
```

**资源配置**（`slurm_train_ldm.sh`）：

| 配置项 | 值 |
|--------|-----|
| GPU | 1 × A100（`--constraint=a100`） |
| CPU | 8 核（`--cpus-per-task=8`，与 `num_workers=8` 一致） |
| 内存 | 128 GB |
| 时间限制 | 24 小时 |
| 日志 | `logs/ldm_<JOBID>.out` / `.err` |

**数据路径**（脚本内硬编码，按需修改）：

```bash
DATA=$PROJ/data/SVD_CESC       # 输入 SVD tile 目录
CKPT=$PROJ/checkpoints/ldm_cesc  # 检查点输出目录
```

> **提交前检查**：SLURM 脚本会自动调用 `uv run python -c "import torch"` 验证环境。若报错，请先在登录节点执行 `uv sync`。

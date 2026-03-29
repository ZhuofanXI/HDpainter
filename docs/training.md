# 训练文档

## VAE 训练（Stage 1）

**脚本**: `scripts/train_vae.py`  
**SLURM**: `scripts/slurm_train_vae.sh`

### 完整超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | *必填* | `.pt` tile 目录路径 |
| `--ckpt_dir` | *必填* | checkpoint 保存目录 |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `2` | batch 大小（显存受限时调低） |
| `--lr` | `1e-4` | Adam 学习率 |
| `--latent_dim` | `50` | 潜空间维度 |
| `--kl_weight` | `1e-5` | KL 散度损失权重 |
| `--num_workers` | `4` | DataLoader 并行 worker 数 |
| `--save_every` | `10` | 每 N epoch 保存一次 checkpoint |
| `--resume` | `None` | 从指定 checkpoint 恢复训练 |

### 快速运行（本地测试）

```bash
cd /ibex/user/wuj0c/Projects/RNA/HDpainter/code
uv run python scripts/train_vae.py \
    --data_dir  ../data/NSCLC \
    --ckpt_dir  ../checkpoints/vae_nsclc \
    --epochs    100 \
    --batch_size 2
```

### SLURM 提交（Ibex A100）

```bash
cd /ibex/user/wuj0c/Projects/RNA/HDpainter/code
sbatch scripts/slurm_train_vae.sh
```

**默认 SLURM 资源配置**：

| 资源 | 配置 |
|------|------|
| GPU | 1× A100 (`--constraint=a100`) |
| 时间 | 24 小时 |
| CPU | 8 核 |
| 内存 | 64 GB |
| 日志 | `logs/vae_{job_id}.out` / `.err` |

**SLURM 默认超参数**（可在脚本中编辑）：

```bash
--data_dir   ../data/NSCLC
--ckpt_dir   ../checkpoints/vae_nsclc
--epochs     100
--batch_size 4
--lr         1e-4
--latent_dim 50
--kl_weight  1e-5
--num_workers 8
--save_every 10
```

### 恢复训练

```bash
uv run python scripts/train_vae.py \
    --data_dir  ../data/NSCLC \
    --ckpt_dir  ../checkpoints/vae_nsclc \
    --resume    ../checkpoints/vae_nsclc/vae_epoch0050.pt
```

### Checkpoint 格式

每个 `.pt` checkpoint 包含以下键：

```python
{
    "epoch":           int,             # 当前 epoch（从 1 开始）
    "model_state":     OrderedDict,     # model.state_dict()
    "optimizer_state": OrderedDict,     # optimizer.state_dict()（仅中间 checkpoint）
    "n_genes":         int,             # 基因数量
    "latent_dim":      int,             # 潜空间维度
}
```

**保存规则**：
- 每 `save_every` epoch：`vae_epoch{NNNN}.pt`（含 optimizer 状态，可恢复训练）
- 训练结束：`vae_final.pt`（不含 optimizer 状态）

### 训练细节

- **优化器**: Adam，`lr=1e-4`
- **混合精度**: `torch.amp.GradScaler` + `autocast`（自动检测 CUDA/CPU device）
- **损失**: 见 [vae.md](vae.md) — 仅在细胞像素上计算的 masked MSE + KL 散度
- **训练目标**: 当前 VAE 在 `target_expr`（Xenium 真值表达）上训练，学习其潜空间表示；Stage 2 的 LDM 将在此潜空间内进行条件去噪

> **注意**: Stage 1 仅使用 `target_expr` 和 `target_cell_id`（mask），不使用 `input_expr` 和 `input_nuclei`。后两者将在 Stage 2（LDM）中作为条件/退化输入。

### 控制台输出示例

```
Device: cuda
Tiles: 1024, n_genes: 392
Epoch    1/100 | loss=0.1234 | recon=0.1233 | kl=0.004567
Epoch   10/100 | loss=0.0456 | recon=0.0455 | kl=0.001234
  -> Saved ../checkpoints/vae_nsclc/vae_epoch0010.pt
...
Training complete. Final checkpoint: ../checkpoints/vae_nsclc/vae_final.pt
```

---

## Stage 2（待实现）

Latent Diffusion Model (LDM) 将以 VAE 潜空间为基础，在 `z` 空间内进行条件去噪。

规划中的训练脚本：`scripts/train_ldm.py`（待实现）

**预期条件信号**：
- `input_expr` — 降级表达（主退化输入）
- `input_nuclei` — 细胞核 mask（几何条件）

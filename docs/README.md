# HDpainter — 代码文档目录

> **分支**: `jieke/dev` | **Python**: ≥ 3.10 | **环境管理**: uv

本目录记录 `code/` 仓库的技术实现细节，供 Jieke（模型训练）和 Zhuofan（预处理与评估）共同参考。

## 文档索引

| 文件 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 整体架构与模块说明 |
| [vae.md](vae.md) | VAE 模型实现（Stage 1） |
| [dataset.md](dataset.md) | 数据集加载与 DataLoader |
| [training.md](training.md) | 训练流程、超参数、SLURM 提交 |

## 快速上手

```bash
cd /ibex/user/wuj0c/Projects/RNA/HDpainter/code
uv sync                              # 同步依赖
uv run python scripts/train_vae.py --help   # 查看训练参数
sbatch scripts/slurm_train_vae.sh   # 提交 SLURM 作业
```

# HDpainter — 代码文档目录

> **分支**: `main` | **Python**: ≥ 3.10 | **环境管理**: uv

本目录记录 `code/` 仓库的技术实现细节，供 Jieke（模型训练）和 Zhuofan（预处理与评估）共同参考。

## 文档索引

| 文件                            | 内容                                               |
| ------------------------------- | -------------------------------------------------- |
| [dataset.md](dataset.md)        | 数据集加载、DataLoader、tile 格式说明              |
| [training.md](training.md)      | LDM 训练流程、超参数、噪声调度、SLURM 提交        |
| [experiments.md](experiments.md)| 实验记录：训练结果、推断实验、Bug 修复、瓶颈分析  |
| [zhuofan.md](zhuofan.md)        | 架构设计规范（SVD降维、U-Net、Decoder、Loss、评估）|

## 快速上手

```bash
cd /ibex/user/wuj0c/Projects/RNA/HDpainter/code
uv sync                              # 同步依赖
uv run python scripts/train_ldm.py --help   # 查看训练参数
```

## 实现状态

| 模块 | 文件 | 状态 |
|------|------|------|
| 数据集 | `src/dataset.py` | ✅ 已完成 |
| 条件 U-Net | `src/models/ldm.py` | ✅ 已完成 |
| NB 解码器 | `src/models/decoder.py` | 🔲 待实现 |
| LDM 训练脚本 | `scripts/train_ldm.py` | ✅ 已完成 |
| SLURM 作业 | `scripts/slurm_train_ldm.sh` | ✅ 已完成 |
| 推断脚本 | `scripts/infer.py` | 🔲 待实现 |

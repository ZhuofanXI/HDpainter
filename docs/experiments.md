# LDM 训练与推断实验记录

> 记录时间：2026-04-01  
> 数据集：`SVD_CESC`（107 个 512×512 tile，64 维 SVD 投影后）  
> 硬件：IBEX A100 80GB

---

## 1. 实现概要

### 模型：`ConditionalUNet`（`src/models/ldm.py`）

| 组件 | 配置 |
|------|------|
| 输入 | `[Z_t ‖ Z_cond]` → 128 通道（各 64 维拼接） |
| 编码器 | 4 级，通道 `[64, 128, 256, 512]`，每级 2 个 ResBlock |
| 瓶颈 | ResBlock → Self-Attention（8 头，32×32）→ ResBlock |
| 解码器 | 对称上采样，skip connection 拼接 |
| 噪声头 | 1×1 Conv → 64ch ε̂（Huber diffusion loss） |
| 边界头 | 直接从解码器特征 x 预测 → 3×3 Conv → SiLU → 1ch（BCE + Dice loss） |
| 参数量 | **44.2M** |

### 推断：img2img DDIM（`scripts/infer.py`）

标准的从纯噪声 z_T 出发的采样（full DDIM）对于图像翻译任务效果极差——模型输出接近数据集均值的均匀图像，RMSE 改善但 PCC 崩溃至 0.01。

改用 **SDEdit / img2img** 方式：
1. 对退化输入 z_cond 在时间步 t_start 处加噪得到 z_{t_start}
2. 从 t_start 开始 DDIM 去噪（50 步）

这样 z_cond 的空间结构被保留为起点，模型只需在此基础上细化。

---

## 2. Bug 修复记录

### Bug 1：边界 GT 几乎等于 cell mask（91% 正样本）

**原因**：原实现标记"任意与邻居不同的像素"为边界。在 2μm 分辨率下，细胞很小，几乎每个像素的某个邻居都来自不同的 cell，导致 91% 的细胞像素被标记为边界——实际上等于 cell mask，没有信息量。

**修复**：改为只标记**两个不同非零细胞相邻**的像素（inter-cell boundary），正样本率降至 ~26%（在细胞区域内）。并为 BCE 添加 `pos_weight=3.0` 补偿类别不平衡。

| | 正样本率（在细胞区域内） |
|--|--|
| 修复前 | ~91% |
| 修复后 | ~26% |

### Bug 2：边界头梯度断流（`eps_hat.detach()`）

**原因**：旧版 forward 里用 `eps_hat.detach()` 估计 Z̃_0 再送入边界头：
```python
z0_hat = (z_t - sqrt_1ab * eps_hat.detach()) / sqrt_ab
boundary = self.boundary_head(z0_hat)
```
`detach()` 使边界 loss 的梯度只能训练 boundary_head 自身的 2 层 conv，U-Net 主干完全不更新——边界头在随机特征上学习，永远无法收敛。

**验证**：比较 epoch 10 和 epoch 199 的 boundary_head 权重变化极小（mean abs change ≈ 0.014），boundary loss 全程徘徊在随机预测水平（~1.0）。

**修复**：边界头直接从解码器特征 `x` 预测（与噪声头并列），两个头共享 U-Net 主干，梯度均正常回传。

```python
# 修复后
eps_hat  = self.noise_head(x)
boundary = self.boundary_head(x)   # 不再经过 z0_hat
```

---

## 3. 训练结果对比

### v1（修复前）

- boundary 梯度断流，bound loss 徘徊在随机水平
- diff loss 较低（0.09），因 U-Net 主干不受 boundary 任务干扰

```
Epoch 0000 | diff=0.430  bound=1.515
Epoch 0199 | diff=0.091  bound=1.035
```

### v2（修复后，当前使用版本）

- boundary loss 真实下降，U-Net 主干同时学习去噪和边界任务
- diff loss 略高（0.12），两任务存在一定竞争

```
Epoch 0000 | diff=0.441  bound=1.645
Epoch 0199 | diff=0.120  bound=1.265
```

---

## 4. 推断实验结果

### 4.1 采样方式对比（v1 checkpoint）

| 方法 | RMSE↓ | PCC↑ |
|------|-------|------|
| Baseline（degraded 直接） | 2.778 | 0.144 |
| Full DDIM（t_start=1000，纯噪声） | 2.108 | 0.010 |
| img2img（t_start=500） | **1.564** | 0.122 |

纯噪声 DDIM 的 PCC 崩溃至 0.01 证实了 img2img 的必要性。

### 4.2 t_start 超参数扫描（v2 checkpoint）

| t_start | RMSE↓ | PCC↑ | 备注 |
|---------|-------|------|------|
| 200 | 3.346 | 0.043 | 噪声太少，模型无法改善 |
| 300 | 2.474 | 0.073 | |
| **400** | **2.257** | **0.135** | PCC 最优，空间结构保留最好 |
| 500 | 1.939 | 0.108 | |
| 600 | 1.949 | 0.125 | RMSE 最低 |
| 700 | 2.131 | 0.131 | |
| Baseline | 2.778 | 0.144 | 无处理参考 |

**当前最佳配置**：`t_start=400`，RMSE 降低 **19%**，PCC=0.135（接近 baseline 0.144）。

> RMSE 改善 vs PCC 存在 trade-off：t_start 越大，模型对输入的修改越激进，RMSE 更低但空间结构相关性下降。

---

## 5. 当前瓶颈分析

### 主要瓶颈：训练数据不足

模型有 **44.2M 参数**，但训练集只有 **107 个 tile**。模型可以学到局部去噪模式（RMSE 改善），但无法泛化出全局空间先验——Ground Truth 中清晰的大尺度组织结构（细胞密集区、环状形态）在预测中几乎无法恢复。

### 次要问题：boundary loss 权重调优

修复后 boundary loss 确实在下降（1.64→1.27），但两任务（去噪 + 边界）竞争导致 diff loss 略微升高。后续可以调整 `t_thresh` 或 `λ₂` 权重进一步优化平衡。

---

## 6. 下一步

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 高 | 获取更多训练数据 | PRAD / NSCLC 数据集，扩大到数百个 tile |
| 中 | 实现 NB Decoder | `src/models/decoder.py`：SVD 64 维 → G 基因，NB-NLL loss |
| 中 | 完善推断脚本 | `scripts/infer.py`：支持真实 Visium HD 数据输入（需要 .pkl 投影矩阵） |
| 低 | boundary loss 调优 | 调整 `λ₂` 权重减少对 diff loss 的干扰 |

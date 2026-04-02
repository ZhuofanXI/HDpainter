# LDM 训练与推断实验记录

> 最近更新：2026-04-02  
> 数据集：`SVD_CESC`（107 tile）+ `SVD_NSCLC`（小）+ `SVD_PRAD`（中），共 230 tile  
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

## 5. 多数据集训练实验（v3）

### 5.1 背景

为解决 v2 数据不足问题（107 tile），引入 NSCLC 和 PRAD 数据集，共 **230 tile**，对每个数据集分别计算 SVD 投影后的标准差并进行归一化：

| 数据集 | tile 数 | 有效 std | scale 因子 |
|--------|---------|----------|-----------|
| SVD_CESC | 107 | 1.03 | 0.97 |
| SVD_NSCLC | ~60 | 0.42 | 2.38 |
| SVD_PRAD | ~63 | 0.34 | 2.94 |

Scale 因子在 `src/dataset.py` 的 `_DATASET_SCALE` 字典中硬编码，由 `_infer_scale()` 根据目录名自动读取。

### 5.2 NaN 崩溃（epoch 284）

训练到 epoch 284 时梯度爆炸，loss 变为 NaN。

**根本原因**：PRAD 数据在 ×2.94 缩放后存在极端离群值（`max_abs ≈ 186`，对比 CESC p999=11.8）。这些离群值导致 AMP 溢出，进而梯度爆炸。

**修复**：在 `src/dataset.py` 加入 `.clamp(-10, 10)`（缩放后），裁去 10σ 以外的 SVD 异常值：

```python
input_expr  = (tile["input_expr"].to_dense().permute(2, 0, 1) * self.scale).clamp(-10, 10)
target_expr = (tile["target_expr"].to_dense().permute(2, 0, 1) * self.scale).clamp(-10, 10)
```

训练从 `epoch_0249.pt` 恢复（回滚至崩溃前最近的 named checkpoint）。

### 5.3 训练日志（v3，epoch 0–199 摘录）

```
Epoch 0174 | diff=0.101  bound=1.048
Epoch 0190 | diff=0.085  bound=1.040
Epoch 0199 | diff=0.089  bound=1.035   ← Training complete（首次完整 200 epoch）
```

训练从 epoch 250 恢复后于 epoch 499 完成（checkpoint `epoch_0499.pt` 已保存）。

### 5.4 v3 推断结果（CESC 测试集，20 tile）

对 CESC 测试集评估了 epoch_0249 和 epoch_0499，并进行 t_start 超参扫描：

#### epoch_0249 最优（t_start 扫描）

| t_start | RMSE↓ | PCC↑ | vs Baseline |
|---------|-------|------|------------|
| 100 | 2.317 | 0.135 | RMSE -3%，PCC 持平 |
| 150 | 2.280 | 0.134 | RMSE -4% |
| 200 | 2.257 | 0.132 | RMSE -5% |
| 250 | 2.237 | 0.129 | RMSE -6% |
| 300 | 2.216 | 0.125 | RMSE -7%，PCC -7% |
| **Baseline** | **2.394** | **0.135** | 参考 |

#### epoch_0499（模型停滞，无效）

epochs 0299/0399/0499 推断结果完全相同（RMSE=2.5037，PCC=0.1294），说明训练从 epoch 250 恢复后权重几乎未更新——模型陷入停滞。

---

## 6. 当前瓶颈分析

### 问题 1：多数据集训练后性能不及预期

- v3 epoch_0249 的最佳 RMSE 改善仅 **3–7%**，显著低于 v2 的 19%（但 v2 用了不同 tile 集，不可直接比较）
- **RMSE 和 PCC 存在 trade-off**：t_start 越大，RMSE 越好但 PCC 越差（模型削减了空间异质性）
- 推断时选 **t_start=100** 可保持 PCC 基本不变（0.135 vs 0.135）同时轻微改善 RMSE

### 问题 2：epoch 250+ 训练停滞

- `epoch_0299 / epoch_0399 / epoch_0499` 推断输出完全相同
- 可能原因：NaN 崩溃后恢复时 optimizer 动量状态异常，或 clamp 后 loss 过低导致梯度不足
- `latest.pt` 指向 epoch 266（非 epoch 499），说明存在一个独立的中途取消的短训练覆写了 latest

### 问题 3：boundary loss 持续高位

boundary loss 全程徘徊在 ~1.04–1.07，远未收敛，说明边界检测任务对当前数据量和训练轮数来说仍难以学习。

---

## 7. 下一步建议

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 高 | 诊断训练停滞原因 | 检查 epoch 250 后 loss 曲线；必要时降低 lr 重新从 epoch_0249 开始 fine-tune |
| 高 | 实现 NB Decoder | `src/models/decoder.py`：SVD 64 维 → G 基因，NB-NLL loss |
| 中 | 推断脚本支持真实数据 | `scripts/infer.py`：支持真实 Visium HD 数据（需要 .pkl 投影矩阵） |
| 低 | boundary loss 调优 | 减小 `λ₂` 或 `t_thresh`，降低边界任务对 diff loss 的干扰 |

# VAE 模型文档

**文件**: `src/models/vae.py`  
**实现**: `VisiumVAE` 类 + `vae_loss` 函数

## 设计原则

VAE 专为空间转录组数据设计，核心约束：**仅使用 1×1 卷积（相当于逐 bin 的 MLP）**，严格保持空间独立性——任意两个相邻 bin 之间不发生信息混合。这对保留 Xenium 的高精度空间分辨率至关重要。

## `VisiumVAE`

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_genes` | `int` | — | 基因数量（从数据集自动推断） |
| `latent_dim` | `int` | `50` | 潜空间维度 |

### 张量形状

| 部位 | 形状 |
|------|------|
| 输入/输出 | `[B, n_genes, H, W]` |
| 编码器隐层 | `[B, 256, H, W]` |
| `mu`, `logvar` | `[B, latent_dim, H, W]` |
| 潜向量 `z` | `[B, latent_dim, H, W]` |

### 网络结构

```
Encoder
  Conv2d(n_genes → 1024, k=1) → SiLU
  Conv2d(1024 → 256,   k=1) → SiLU
  ├── fc_mu:     Conv2d(256 → latent_dim, k=1)
  └── fc_logvar: Conv2d(256 → latent_dim, k=1)

Reparameterization
  训练时: z = mu + randn * exp(0.5 * logvar)
  推理时: z = mu

Decoder
  Conv2d(latent_dim → 256,    k=1) → SiLU
  Conv2d(256 → 1024,          k=1) → SiLU
  Conv2d(1024 → n_genes,      k=1)
```

### 前向方法

```python
recon, mu, logvar = model(x)
# x:      [B, n_genes, H, W]
# recon:  [B, n_genes, H, W]
# mu:     [B, latent_dim, H, W]
# logvar: [B, latent_dim, H, W]
```

也可单独调用：

```python
mu, logvar = model.encode(x)
z = model.reparameterize(mu, logvar)
recon = model.decode(z)
```

## `vae_loss`

### 签名

```python
vae_loss(
    recon:     torch.Tensor,  # [B, C, H, W] 重建表达
    target:    torch.Tensor,  # [B, C, H, W] 真值表达
    mu:        torch.Tensor,  # [B, latent_dim, H, W]
    logvar:    torch.Tensor,  # [B, latent_dim, H, W]
    mask:      torch.Tensor,  # [B, 1, H, W]  二值 mask (1=细胞, 0=背景)
    kl_weight: float = 1e-5,  # KL 项权重
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (total, recon_loss, kl_loss)
```

### 损失计算

**重建损失（仅对细胞像素）**
$$\mathcal{L}_{recon} = \frac{\sum_{b,c,h,w} (\hat{x} - x)^2 \cdot m_{b,c,h,w}}{\sum m}$$

**KL 散度（仅对细胞像素）**
$$\mathcal{L}_{KL} = -\frac{1}{2} \frac{\sum_{b,l,h,w} (1 + \log\sigma^2 - \mu^2 - \sigma^2) \cdot m_{b,l,h,w}}{\sum m}$$

**总损失**
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda_{KL} \cdot \mathcal{L}_{KL}, \quad \lambda_{KL} = 10^{-5}$$

> **设计说明**: KL 权重极小（`1e-5`）是为避免后验坍塌（posterior collapse）。mask 限制损失仅在有细胞的像素上计算，防止背景噪声主导训练目标。

## 用法示例

```python
from src.models.vae import VisiumVAE, vae_loss

model = VisiumVAE(n_genes=392, latent_dim=50).to(device)
recon, mu, logvar = model(target_expr)                    # [B, 392, H, W]
mask = (target_cell_id > 0).float()                       # [B, 1, H, W]
loss, recon_l, kl_l = vae_loss(recon, target_expr, mu, logvar, mask)
```

### HDpainter: 基于跨平台扩散模型以提高空间转录组分辨率

本部分仅仅记录低维空间的扩散模型和后续decoder部分内容，其它信息记录在另一份markdown中。

做个简单的流程解释：我在上游的数据处理包括：Xenium转化为Visium HD格式数据，一方面制作mask，使用stpainter进行基因插补并在细胞区域进行重分配；另一方面制作模型输入，使用高斯模糊和Gamma-Poisson分布来模拟HD数据的数据分布。然后进行滑动窗口切片，以降低图像边缘残缺细胞对模型预测的影响。

前面都是预处理，此时得到的数据是稀疏矩阵格式的pytorch张量，如果直接使用VAE来进行降维，那么输入VAE的矩阵数据大小在7G每个tile，而且对全局特征的把握很容易缺失。所以我查找了其它论文和常见工具中的不需要去中心化的适合稀疏矩阵特点的降维方法，然后发现在scanpy等组学“金标准”的工具中，使用的有TruncatedSVD, 所以我使用该方法对切tile前的整个大的样本进行全局的计算和降维，并保存了pkl文件用于后续的升维重建。

那么现在已经完成了降维，就需要直接输入扩散模型进行训练了。第三节主要从数学角度论述TruncatedSVD对我们数据的处理过程，在第四节是对扩散模型部分的详细规划和要求。



#### 第三节：使用TruncatedSVD进行降维和产生密集矩阵

考虑到稀疏矩阵的数据特点, 我们采用Truncated SVD来进行数据降维。 Truncated SVD无需对数据进行去均值中心化，保留了矩阵的稀疏性,处理阶段保存的 .pkl 文件核心包含了降维投影矩阵（右奇异向量 $V_k$）和特征标准化参数。在后续Decoder阶段，只需将64维特征与投影矩阵的转置相乘, 即可完成线性升维。

这种线性升维能过滤高频噪声，只保留数据中最核心的全局结构和主要方差，但会丢失部分非线性的细粒度生物学信号。因此，后续我们采用MLP结合输入时的mask，即单细胞水平的高质量表达标签（Ground Truth）进行监督训练，使用非线性解码器来弥补缺失信号。

##### 1. 截断奇异值分解 (Truncated SVD) 的数学原理

输入稀疏基因表达矩阵为 $X \in \mathbb{R}^{N \times C}$，其中 $N$ 是细胞/空间位点（Bins）的数量，$C$ 是基因的数量。

标准的奇异值分解会将矩阵 $X$ 分解为三个矩阵的乘积：

$X = U \Sigma V^T$

其中：

- $U \in \mathbb{R}^{N \times N}$ 是左奇异向量矩阵。
- $\Sigma \in \mathbb{R}^{N \times C}$ 是对角矩阵，对角线上的元素为奇异值（按从大到小排列）。
- $V^T \in \mathbb{R}^{C \times C}$ 是右奇异向量矩阵。

在 **TruncatedSVD** 中，为了实现降维，我们只保留前 $k$ 个最大的奇异值及其对应的特征向量（在你的代码中，$k = 64$）。此时，原始矩阵被近似表示为：

$X \approx X_k = U_k \Sigma_k V_k^T$

其中：

- $U_k \in \mathbb{R}^{N \times k}$
- $\Sigma_k \in \mathbb{R}^{k \times k}$
- $V_k^T \in \mathbb{R}^{k \times C}$

##### 2. 降维投影 (`fit_transform`)

在代码的 `X_svd = svd.fit_transform(X_sparse)` 这一步，模型计算了数据在低维空间中的坐标（即主成分得分）。

数学上，这相当于将原始高维矩阵 $X$ 投影到由右奇异向量 $V_k$ 张成的 $k$ 维子空间中：

$X_{svd} = X V_k$

由于 $X \approx U_k \Sigma_k V_k^T$，且 $V_k^T V_k = I$，上述投影也可以等价表示为：

$X_{svd} = U_k \Sigma_k$

此时，输出的 $X_{svd}$ 是一个稠密矩阵，维度从 $(N \times C)$ 成功降维到了 $(N \times 64)$。

##### 3. 低维空间标准化 (`StandardScaler`)

在得到 64 维的特征矩阵 $X_{svd}$ 后，代码使用了 `StandardScaler` 对这 64 个特征通道进行独立的标准化（Z-score 归一化）。

设 $Z = X_{svd}$，对于第 $j$ 个特征维度（$j \in [1, 64]$），首先计算该维度的均值 $\mu_j$ 和标准差 $\sigma_j$：

$\mu_j = \frac{1}{N} \sum_{i=1}^{N} Z_{i,j}$

$\sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (Z_{i,j} - \mu_j)^2}$

然后对每个元素进行缩放，得到最终的归一化输出 $\tilde{Z}$（即代码中的 `X_svd_norm`）：

$\tilde{Z}_{i,j} = \frac{Z_{i,j} - \mu_j}{\sigma_j}$

##### 4. 对新数据进行映射 (`transform_degraded`)   [4和5系后续MLP的decoder阶段的数学原理]

在模块 2 中，当你输入退化后的新数据 $X_{new}$（例如降采样后的数据）时，模型会复用在基准数据上学到的投影矩阵 $V_k$ 以及均值 $\mu$ 和标准差 $\sigma$。

**第一步：稀疏矩阵直接投影**

$X_{new\_svd} = X_{new} V_k$

**第二步：使用已保存的参数进行标准化**

$\tilde{Z}_{new\_i,j} = \frac{(X_{new\_svd})_{i,j} - \mu_j}{\sigma_j}$

通过上述数学过程，`TruncatedSVD` 巧妙地利用了矩阵乘法 $X V_k$，在**不破坏输入矩阵 $X$ 稀疏性**（即不需要减去均值将 0 变成非 0）的前提下，提取了数据中方差最大的 64 个正交方向，高效地完成了从数万维基因空间到 64 维潜在空间的特征压缩。



#### 四、模型架构与loss function设置

写在最前面：所有的loss计算一定**只围绕mask标记的细胞区域进行计算**！千万不要计算背景区域。即：

Key: target_cell_id  | Shape: [512, 512, 1]             | Dtype: torch.int32     | Layout: torch.sparse_coo

这个cell_id≠0的部分，针对这部分计算loss。因为我们的模型后续在decoder和evaluation中都是仅考虑细胞区域内的bin的得分，那为什么我们还要输入背景区域呢？这是为了让模型学习Visium HD的数据表达谱的空间结构和分布特征，本质上是为了后续能更好的应用在真实Visium HD数据上做推断，确保模型的实用性。

**多任务联合损失构建 (Multi-task Loss Formulation)** - 确保模型在去噪表达谱的同时，能够根据实例掩码的拓扑结构准确推断细胞边界。因为这部分是我自己去查阅过，所以写在前面，模型的loss可以直接按下文建议设置：

- **表达谱去噪损失:** $\mathcal{L}_{diff} = \mathbb{E}_{t, Z, \epsilon} \left[ \left\| \epsilon - \hat{\epsilon} \right\|^2_2 \right]$。
- **边界拓扑损失:** 约束预测边界与真实边界的差异：$\mathcal{L}_{bound} = \mathcal{L}_{fn}(\hat{B}, B)$。
- **总目标函数:** $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{diff} + \lambda_2 \mathcal{L}_{bound}$。最终推断时，逆扩散采样得到 $\hat{Z}_0$，再由 VAE 解码器生成最终的单细胞分辨率表达谱：$\hat{Y} = D_\psi(\hat{Z}_0)$。

​	推断考虑在前n步都仅仅计算表达谱去噪loss，仅在最后的几步计算边界拓扑损失。目前推荐表达谱去噪损失可以使用**Huber loss（pytorch标准库可直接调用）**，边界拓扑损失可使用**BCE&DICE**的联合损失。在最后几步计算损失中可以逐步提高边界拓扑损失的比例。

关于模型的具体架构，我在凝练我的数据特征、参考论文、生物属性要求后，让大模型帮我写了一份参考建议：

在完成了将空间转录组数据通过滑动窗口切片（Sliding Window）转化为类似图像的张量 $(B, C, H, W)$，并结合 TruncatedSVD 降维到 64 维稠密空间，这完美契合了计算机视觉中隐潜在扩散模型（Latent Diffusion Models, LDM）的标准范式。

针对你在低维空间（64维）下的扩散模型架构设计，我结合目前生成式 AI 和空间组学的前沿研究（如 Stable Diffusion, Palette, DiT 等），为你提供以下几个核心的架构设计建议：

##### 1. 整体架构：条件引导的 U-Net (Conditional U-Net)

由于你的任务本质上是**“图像到图像的翻译”**（从退化的、低分辨率的 64 维特征图，还原为高质量的 64 维特征图），你的扩散模型不能是无条件的，必须以“退化数据”作为条件（Condition）。

- **输入拼接 (Concatenation)**：最直接且有效的条件注入方式是在通道维度（Channel Dimension）进行拼接。
  - 设当前时间步 $t$ 的加噪潜变量为 $Z_t \in \mathbb{R}^{B \times 64 \times H \times W}$。
  - 设退化数据的潜变量为 $Z_{cond} \in \mathbb{R}^{B \times 64 \times H \times W}$。
  - **模型输入**：将两者拼接，输入通道数为 128，即 $X_{in} = \text{Concat}(Z_t, Z_{cond}) \in \mathbb{R}^{B \times 128 \times H \times W}$。
- **网络骨干 (Backbone)**：建议使用带有残差块（ResNet Blocks）和自注意力机制（Self-Attention）的 U-Net。Visium HD 的 bin 非常小（2μm），细胞通常跨越多个 bin，U-Net 的下采样/上采样结构能提供足够大的**感受野（Receptive Field）**，帮助模型理解细胞的整体形态和组织结构。

##### 2. 多任务输出头设计 (Multi-task Heads)

你的 Readme 中提到需要联合计算表达谱去噪损失 $\mathcal{L}_{diff}$ 和 边界拓扑损失 $\mathcal{L}_{bound}$。在标准的扩散模型中，网络通常只预测噪声 $\epsilon$，但这不利于直接计算边界损失。

**建议架构设计：**

- **主干输出**：U-Net 的最后一层输出一个 64 通道的特征图，代表预测的噪声 $\hat{\epsilon}$（或者直接预测 $\hat{Z}_0$）。
- **边界预测分支 (Boundary Head)**：在 U-Net 的解码器末端（或者利用预测出的 $\hat{Z}_0$），外接一个轻量级的卷积模块（例如两层 $3 \times 3$ 卷积），输出单通道的边界概率图 $\hat{B} \in \mathbb{R}^{B \times 1 \times H \times W} $。
- **数学实现**：如果模型预测的是噪声 $\hat{\epsilon}$，你需要利用扩散模型的公式在每一步动态计算出当前对原始无噪声数据 $\hat{Z}_0$ 的估计：$\hat{Z}_0 = \frac{Z_t - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$，然后将这个 $\hat{Z}_0$送入边界预测分支得到 $\hat{B}$，再去计算 BCE & DICE 联合损失。

##### 3. 时间步依赖的损失权重 (Time-dependent Loss Weighting)

你的 Readme 中有一个非常精彩的设定：“仅在最后几步计算边界拓扑损失”。这在扩散模型中极其合理，因为在 $t$ 较大（高噪声）时，图像完全是混沌的，强行约束边界会导致模型崩溃。

**建议设计动态权重函数 $\lambda_2(t)$：**
 你可以将总损失函数重写为带有时间步 $t$ 衰减的格式：

$\mathcal{L}_{total} = \mathcal{L}_{diff} + \lambda_2(t) \mathcal{L}_{bound}$

其中 $\lambda_2(t)$ 可以设计为一个**Sigmoid 退火函数**或**阶跃函数**。例如，假设总步数为 $T=1000$：

- 当 $t > 200$ 时，$\lambda_2(t) = 0$（只专心去噪）。
- 当 $t \le 200$ 时，$\lambda_2(t)$ 从 0 平滑过渡到 1（在图像逐渐清晰时，开始雕刻细胞边界细节）。

##### 4. 引入空间注意力机制 (Spatial Attention)

空间转录组与普通自然图像不同，组织切片中相隔较远的两个区域可能属于同一种细胞类型（比如肿瘤微环境中的免疫细胞浸润）。

- **建议**：在 U-Net 的最底层（Bottleneck，即分辨率最低、特征最抽象的地方）加入 **Transformer 块（Self-Attention）**。
- **作用**：卷积层擅长提取局部纹理（细胞边缘），而自注意力机制擅长捕捉全局上下文（Global Context）。这能让模型在修复某个 bin 的基因表达时，参考整个切片（Tile）中相似细胞的特征，极大提升降噪的准确性。

##### 5. 归一化与激活函数的细节

- **GroupNorm**：在扩散模型中，由于显存限制，Batch Size 通常较小。绝对不要使用 BatchNorm，请使用 `GroupNorm`（通常设 `num_groups=32`），它对小 Batch Size 更加稳定。
- **SiLU (Swish) 激活函数**：在 U-Net 中，使用 SiLU（$x \cdot \sigma(x)$）替代 ReLU。SiLU 在扩散模型中被广泛证明能提供更平滑的梯度和更好的生成质量。

##### 总结：数据流向图 (供你在脑海中或PPT中构建)

1. **输入**：`Degraded_Z` (64通道) + `Noisy_Z_t` (64通道) + `Time_Step_Embedding`
2. **网络**：`Conditional U-Net` (包含 ResNet Blocks + GroupNorm + SiLU + Self-Attention)
3. **输出**：
   - **分支1**：预测噪声 $\hat{\epsilon}$ $\rightarrow$ 计算 Huber Loss ($\mathcal{L}_{diff}$)
   - **分支2**：根据 $\hat{\epsilon}$ 推导 $\hat{Z}_0$ $\rightarrow$ 经过 `Boundary Head` $\rightarrow$ 预测边界 $\hat{B}$ $\rightarrow$ 计算 BCE+DICE ($\mathcal{L}_{bound}$，仅在 $t$ 较小时激活)
4. **推理阶段**：经过 $T$ 步去噪得到最终的干净 $\hat{Z}_0$，直接送入你设计的 `NegativeBinomialDecoder`，输出最终的 $\mu$ 和 $\theta$。



#### 五、解码器和数据升维

- **Step 1: 为什么需要专门的解码器？** - [Reason] 线性逆变换（SVD Inverse）假设数据是连续的高斯分布，会导致重构的基因表达量出现负数或非整数。而真实的单细胞/空间转录组原始计数（Raw Counts）是高度稀疏、离散且存在过度离散化（Overdispersion）的，符合负二项分布（Negative Binomial, NB）或零膨胀负二项分布（ZINB）。
- **Step 2: 核心架构设计** - [Reason] 扩散模型负责在 64 维的连续潜空间（Latent Space）中生成去噪后的特征图。我们将这个 64 维特征作为输入，通过一个浅层神经网络（如 1x1 卷积）映射回原始的 $G$ 维基因空间（例如 20,000 维）。
- **Step 3: 参数化输出** - [Reason] 神经网络不直接输出基因的绝对表达量，而是输出负二项分布的两个关键参数：均值（$\mu$）和离散度（$\theta$）。为了保证这两个参数严格为正，通常在输出层使用 `Softplus` 或 `Exp` 激活函数。
- **Step 4: 损失函数设计** - [Reason] 在训练该解码器时，不使用 MSE Loss，而是使用负二项分布的负对数似然（Negative Log-Likelihood, NLL）作为 Loss。目标是最大化真实原始计数（Raw Counts）在该分布下的概率。

##### 代码骨架参考

这是一个附加在扩散模型之后的独立解码器模块，但是我还没有进行仔细的检查和适配，放在这里是方便后续喂给大模型进行骨架参考。目前不能直接使用。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NegativeBinomialDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_genes=target_gene):
        """
        latent_dim: 扩散模型输出的特征维度 (64)
        num_genes: 原始表达谱的基因总数 (G)
        """
        super().__init__()
        # 使用 1x1 卷积在空间维度上逐像素进行特征映射
        self.mean_decoder = nn.Conv2d(latent_dim, num_genes, kernel_size=1)
        self.disp_decoder = nn.Conv2d(latent_dim, num_genes, kernel_size=1)

    def forward(self, z):
        """
        z: 扩散模型生成的潜变量特征图, shape: (B, 64, H, W)
        返回: 
            mu: 基因表达均值 (B, G, H, W)
            theta: 基因表达离散度 (B, G, H, W)
        """
        # 必须保证 mu 和 theta 严格大于 0
        mu = F.softplus(self.mean_decoder(z)) + 1e-6
        theta = F.softplus(self.disp_decoder(z)) + 1e-6
        return mu, theta

def negative_binomial_loss(y_true, mu, theta, eps=1e-8):
    """
    计算负二项分布的负对数似然损失 (NLL Loss)
    y_true: 真实的原始基因计数 (Raw Counts), shape: (B, G, H, W)
    mu: 预测的均值
    theta: 预测的离散度
    """
    # 避免数值不稳定
    y_true = y_true.float()
    
    # 负二项分布的对数似然公式实现
    t1 = torch.lgamma(theta + y_true) - torch.lgamma(theta) - torch.lgamma(y_true + 1)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = y_true * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    
    log_lik = t1 + t2 + t3
    
    # 返回负对数似然的平均值
    return -torch.mean(log_lik)
```

**使用逻辑：**

1. 扩散模型在 64 维空间训练完毕。
2. 冻结扩散模型，提取其生成的 64 维特征图 `z`。
3. 将 `z` 输入 `NegativeBinomialDecoder`，预测 `mu` 和 `theta`。
4. 使用原始的 target_gene维 Raw Counts 作为 `y_true`，计算 `negative_binomial_loss` 并反向传播更新解码器。
5. 推理时直接使用输出的 `mu` 作为重构的高精度基因表达谱。



#### 六、模型评估

**第一阶段：配对合成数据测试（In silico Validation）**
 利用构建的 `cdata` (输入) 和 `adata` (Ground Truth) 进行严格的定量测试。使用我们人为设置的mask进行模型性能评估，这一步主要用于证明模型在理想/受控条件下的绝对性能。

**第二阶段：横向基准对比（Benchmarking）**
 引入你在文档中提到的 SOTA（State-of-the-Art）工具：`bin2cell`, `enact`, `BIDCell`, `HERGAST`, `cellpose`。在相同的合成数据集上，证明 HDpainter 在分割和降噪上的优越性。

**第三阶段：真实 Visium HD 数据推断与生物学验证（Biological Validation）**
 在真实的 Visium HD 数据上（此时没有绝对的 Ground Truth），通过单细胞测序（scRNA-seq）比对、空间结构解析、信噪比提升等下游生物学分析，证明模型的实际应用价值。


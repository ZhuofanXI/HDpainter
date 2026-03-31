"""
Conditional U-Net for latent diffusion on 64-dim SVD-projected spatial tiles.

Architecture (zhuofan.md §4):
  - Input: Z_t (latent_dim ch) + Z_cond (latent_dim ch) concatenated → 2*latent_dim ch
  - Backbone: U-Net with ResBlocks (GroupNorm + SiLU), downsampling pyramid
  - Bottleneck: Self-Attention transformer block
  - Dual output heads:
      * noise head  → eps_hat  (latent_dim ch)  used for Huber diffusion loss
      * boundary head → boundary logits (1 ch)   used for BCE+Dice loss (low-noise steps only)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time embedding ────────────────────────────────────────────────────────────

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional embedding for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / (half - 1)
    )
    args = timesteps[:, None].float() * freqs[None]  # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


class TimeEmbedding(nn.Module):
    def __init__(self, base_dim: int, emb_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.net = nn.Sequential(
            nn.Linear(base_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.base_dim))  # (B, emb_dim)


# ── Building blocks ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv → +time_emb → GroupNorm → SiLU → Conv → +skip."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions (used at bottleneck)."""

    def __init__(self, ch: int, num_heads: int = 8, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ── Conditional U-Net ─────────────────────────────────────────────────────────

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net denoiser for latent diffusion.

    Args:
        latent_dim:     SVD feature dimension (default 64).
        base_ch:        Base channel width; multiplied by ch_mult at each level.
        ch_mult:        Channel multipliers per encoder level.
        num_res_blocks: ResBlocks per encoder/decoder level.
        num_heads:      Attention heads in bottleneck Self-Attention.
        T:              Total diffusion timesteps (used only for reference).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        base_ch: int = 64,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_heads: int = 8,
        T: int = 1000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.T = T
        num_groups = 32
        emb_dim = base_ch * 4

        self.time_emb = TimeEmbedding(base_ch, emb_dim)

        # Channel sizes at each spatial resolution level
        chs = [base_ch * m for m in ch_mult]  # e.g. [64, 128, 256, 512]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.in_conv = nn.Conv2d(latent_dim * 2, chs[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()  # list of ModuleList (one per level)
        self.downsamples = nn.ModuleList()
        enc_out_chs = []  # record output channel count at each encoder level

        in_ch = chs[0]
        for i, out_ch in enumerate(chs):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(in_ch, out_ch, emb_dim, num_groups))
                in_ch = out_ch
            enc_out_chs.append(in_ch)
            self.enc_blocks.append(level_blocks)
            if i < len(chs) - 1:
                self.downsamples.append(Downsample(in_ch))

        self._enc_out_chs = enc_out_chs

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            ResBlock(in_ch, in_ch, emb_dim, num_groups),
            SelfAttentionBlock(in_ch, num_heads, num_groups),
            ResBlock(in_ch, in_ch, emb_dim, num_groups),
        ])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(len(chs))):
            out_ch = chs[i]
            skip_ch = enc_out_chs[i]
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                block_in = (in_ch + skip_ch) if j == 0 else out_ch
                level_blocks.append(ResBlock(block_in, out_ch, emb_dim, num_groups))
            in_ch = out_ch
            self.dec_blocks.append(level_blocks)
            if i > 0:
                self.upsamples.append(Upsample(in_ch))

        # ── Output heads ─────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(num_groups, in_ch)

        # Head 1: predict noise eps
        self.noise_head = nn.Conv2d(in_ch, latent_dim, 1)

        # Head 2: predict cell boundary from estimated Z_0
        self.boundary_head = nn.Sequential(
            nn.Conv2d(latent_dim, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(
        self,
        z_t: torch.Tensor,         # (B, latent_dim, H, W) noisy latent at step t
        z_cond: torch.Tensor,       # (B, latent_dim, H, W) degraded condition
        t: torch.Tensor,            # (B,) int timestep indices
        alpha_bar_t: torch.Tensor,  # (B,) float ᾱ_t for Z_0 estimation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            eps_hat:  predicted noise,           (B, latent_dim, H, W)
            boundary: cell boundary logits,      (B, 1, H, W)
        """
        t_emb = self.time_emb(t)  # (B, emb_dim)

        x = self.in_conv(torch.cat([z_t, z_cond], dim=1))

        # Encoder — store skip tensors
        skips = []
        down_idx = 0
        for i, level_blocks in enumerate(self.enc_blocks):
            for block in level_blocks:
                x = block(x, t_emb)
            skips.append(x)
            if i < len(self.enc_blocks) - 1:
                x = self.downsamples[down_idx](x)
                down_idx += 1

        # Bottleneck
        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t_emb)

        # Decoder — merge skip connections
        up_idx = 0
        for i, level_blocks in enumerate(self.dec_blocks):
            skip = skips[-(i + 1)]
            for j, block in enumerate(level_blocks):
                if j == 0:
                    x = block(torch.cat([x, skip], dim=1), t_emb)
                else:
                    x = block(x, t_emb)
            if i < len(self.dec_blocks) - 1:
                x = self.upsamples[up_idx](x)
                up_idx += 1

        x = F.silu(self.out_norm(x))

        # Noise prediction
        eps_hat = self.noise_head(x)

        # Estimate clean Z_0 from current prediction, then predict boundary
        # Z_0_hat = (Z_t - sqrt(1 - ᾱ_t) * eps_hat) / sqrt(ᾱ_t)
        sqrt_ab = alpha_bar_t.sqrt()[:, None, None, None]
        sqrt_1ab = (1.0 - alpha_bar_t).sqrt()[:, None, None, None]
        z0_hat = (z_t - sqrt_1ab * eps_hat.detach()) / sqrt_ab.clamp(min=1e-8)

        boundary = self.boundary_head(z0_hat)

        return eps_hat, boundary

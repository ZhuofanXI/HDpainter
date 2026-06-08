from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_average_pool(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    weight = mask.clamp_min(0.0)
    numer = (feat * weight).sum(dim=(-1, -2))
    denom = weight.sum(dim=(-1, -2)).clamp_min(eps)
    return numer / denom


def masked_ring_pool(feat: torch.Tensor, seed_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if seed_mask.ndim == 3:
        seed_mask = seed_mask.unsqueeze(1)
    outer = F.max_pool2d(seed_mask.float(), kernel_size=9, stride=1, padding=4)
    inner = F.max_pool2d(seed_mask.float(), kernel_size=5, stride=1, padding=2)
    ring = (outer - inner).clamp_min(0.0)
    return masked_average_pool(feat, ring, eps=eps)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        return self.shortcut(x) + h


class ConditionAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.cond_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, spatial_tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            self.spatial_norm(spatial_tokens),
            self.cond_norm(cond_tokens),
            self.cond_norm(cond_tokens),
            need_weights=False,
        )
        spatial_tokens = spatial_tokens + attn_out
        spatial_tokens = spatial_tokens + self.ffn(self.ffn_norm(spatial_tokens))
        return spatial_tokens


class SizeLatentClosedRegionModel(nn.Module):
    """Legacy v1 model kept as the canonical project model.

    This file intentionally preserves the original v1 architecture because it
    remains the most reliable checkpoint family for inference/post-process.
    Some later training code expects a few newer output keys, so they are
    returned as compatibility placeholders.
    """

    def __init__(
        self,
        expr_channels: int,
        latent_dim: int,
        cond_dim: int,
        canvas_size: int = 33,
        boundary_samples: int = 64,
        attention_layers: int = 2,
        attention_heads: int = 4,
        scale_radii: tuple[float, ...] = (7.0, 10.0, 13.0, 16.0),
        canvas_margin: float = 1.5,
        min_radius: float = 0.05,
        max_radius: float | None = None,
        logit_sharpness: float = 12.0,
        boundary_residual_scale: float = 0.30,
    ) -> None:
        super().__init__()
        self.expr_channels = int(expr_channels)
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.canvas_size = int(canvas_size)
        self.boundary_samples = int(boundary_samples)
        self.num_scales = int(len(scale_radii))
        self.canvas_margin = float(canvas_margin)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius) if max_radius is not None else float((canvas_size - 1) / 2.0)
        self.logit_sharpness = float(logit_sharpness)
        self.boundary_residual_scale = float(boundary_residual_scale)  # kept for signature compatibility
        self.feature_dim = 96
        self.num_condition_tokens = 3

        self.context_encoder = nn.Sequential(
            nn.Linear(self.cond_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.size_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, 1),
        )
        self.expr_stem = ConvBlock(self.expr_channels, 48)
        self.condition_map_proj = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=1),
            nn.SiLU(),
        )
        self.scale_encoder = nn.Sequential(
            ConvBlock(48, 64),
            ConvBlock(64, self.feature_dim),
            ConvBlock(self.feature_dim, self.feature_dim),
        )
        self.scale_gate_head = nn.Sequential(
            nn.Linear(self.feature_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.cond_to_tokens = nn.Linear(self.feature_dim, self.num_condition_tokens * self.feature_dim)
        self.attn_blocks = nn.ModuleList(
            [ConditionAttentionBlock(self.feature_dim, attention_heads) for _ in range(int(attention_layers))]
        )
        self.decoder = nn.Sequential(
            ConvBlock(self.feature_dim, self.feature_dim),
            ConvBlock(self.feature_dim, self.feature_dim),
        )
        self.shape_head = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 128),
            nn.SiLU(),
            nn.Linear(128, self.boundary_samples),
        )
        self.area_delta_head = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.latent_proj_head = nn.Sequential(
            nn.Linear(self.expr_channels, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.register_buffer("scale_radii", torch.tensor(scale_radii, dtype=torch.float32), persistent=False)
        self.register_buffer("canvas_circle_mask", self._build_canvas_circle_mask(self.canvas_size), persistent=False)
        angle_map, dist_map = self._build_polar_maps(self.canvas_size)
        self.register_buffer("angle_map", angle_map, persistent=False)
        self.register_buffer("dist_map", dist_map, persistent=False)

    @staticmethod
    def _build_canvas_circle_mask(canvas_size: int) -> torch.Tensor:
        coords = torch.arange(canvas_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (float(canvas_size) - 1.0) / 2.0
        rr = torch.sqrt((yy - center).pow(2) + (xx - center).pow(2))
        return (rr <= center + 0.5).float().unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _build_polar_maps(canvas_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.arange(canvas_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (float(canvas_size) - 1.0) / 2.0
        dy = yy - center
        dx = xx - center
        angle = torch.remainder(torch.atan2(dy, dx) + (2.0 * math.pi), 2.0 * math.pi)
        dist = torch.sqrt(dx.pow(2) + dy.pow(2))
        return angle.unsqueeze(0), dist.unsqueeze(0)

    def compute_canvas_radius_from_area(self, pred_area: torch.Tensor) -> torch.Tensor:
        return torch.ceil(torch.sqrt(pred_area.clamp_min(1e-6) / math.pi)) + self.canvas_margin

    def render_closed_mask_from_radii(self, boundary_radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = boundary_radius.shape[0]
        angle_frac = self.angle_map.to(device=boundary_radius.device, dtype=boundary_radius.dtype) / (2.0 * math.pi)
        anchor_pos = angle_frac * float(self.boundary_samples)
        left_idx = torch.floor(anchor_pos).long() % self.boundary_samples
        right_idx = (left_idx + 1) % self.boundary_samples
        frac = anchor_pos - left_idx.to(anchor_pos.dtype)
        batch_idx = torch.arange(batch_size, device=boundary_radius.device).view(-1, 1, 1)
        left_val = boundary_radius[batch_idx, left_idx]
        right_val = boundary_radius[batch_idx, right_idx]
        radius_map = left_val * (1.0 - frac) + right_val * frac
        dist_map = self.dist_map.to(device=boundary_radius.device, dtype=boundary_radius.dtype)
        logits = self.logit_sharpness * (radius_map - dist_map)
        circle_mask = self.canvas_circle_mask.to(device=boundary_radius.device, dtype=boundary_radius.dtype)
        logits = torch.where(circle_mask > 0.5, logits.unsqueeze(1), torch.full_like(logits.unsqueeze(1), -12.0))
        prob = torch.sigmoid(logits)
        return logits, prob

    def aggregate_expr_with_mask(
        self,
        expr_crops: torch.Tensor,
        scale_weights: torch.Tensor,
        mask_prob: torch.Tensor,
    ) -> torch.Tensor:
        fused_expr = (expr_crops * scale_weights[:, :, None, None, None]).sum(dim=1)
        pooled_expr = masked_average_pool(fused_expr, mask_prob)
        return self.latent_proj_head(pooled_expr)

    def forward(
        self,
        expr_crops: torch.Tensor,
        seed_mask: torch.Tensor,
        neighbor_seed_map: torch.Tensor,
        neighbor_distance_map: torch.Tensor,
        cond_vec: torch.Tensor,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if expr_crops.ndim != 5:
            raise ValueError(f"expr_crops must have shape [B,S,C,H,W], got {tuple(expr_crops.shape)}")

        batch_size, num_scales, _, height, width = expr_crops.shape
        context_vec = self.context_encoder(cond_vec)
        pred_log_area = self.size_head(context_vec)
        pred_area = pred_log_area.exp()
        canvas_radius_pref = self.compute_canvas_radius_from_area(pred_area)

        condition_maps = torch.cat([seed_mask, neighbor_seed_map, neighbor_distance_map], dim=1)
        cond_map_proj = self.condition_map_proj(condition_maps)
        scale_feat_list: list[torch.Tensor] = []
        for scale_idx in range(num_scales):
            expr_scale = expr_crops[:, scale_idx]
            feat = self.expr_stem(expr_scale)
            feat = self.scale_encoder(feat + cond_map_proj)
            scale_feat_list.append(feat)
        scale_feat_maps = torch.stack(scale_feat_list, dim=1)

        gate_input = torch.cat([context_vec, torch.log1p(canvas_radius_pref)], dim=1)
        large_gate = torch.sigmoid(self.scale_gate_head(gate_input))
        small_weights = torch.full_like(scale_feat_maps[:, :2, 0, 0, 0], 0.5) * (1.0 - large_gate)
        large_weights = torch.full_like(scale_feat_maps[:, 2:, 0, 0, 0], 0.5) * large_gate
        scale_weights = torch.cat([small_weights, large_weights], dim=1)
        fused_feat = (scale_feat_maps * scale_weights[:, :, None, None, None]).sum(dim=1)

        spatial_tokens = fused_feat.flatten(2).transpose(1, 2)
        cond_tokens = self.cond_to_tokens(context_vec).view(batch_size, self.num_condition_tokens, self.feature_dim)
        for block in self.attn_blocks:
            spatial_tokens = block(spatial_tokens, cond_tokens)
        fused_feat = spatial_tokens.transpose(1, 2).reshape(batch_size, self.feature_dim, height, width)
        decoded_feat = self.decoder(fused_feat)

        global_pool = decoded_feat.mean(dim=(-1, -2))
        seed_pool = masked_average_pool(decoded_feat, seed_mask)
        ring_pool = masked_ring_pool(decoded_feat, seed_mask)
        shape_features = torch.cat([global_pool, seed_pool, ring_pool], dim=1)
        shape_logits = self.shape_head(shape_features)
        shape_centered = shape_logits - shape_logits.mean(dim=1, keepdim=True)
        shape_scale = torch.exp(shape_centered).clamp_min(1e-4)

        area_delta = self.area_delta_head(shape_features).clamp(min=-0.8, max=0.8)
        mask_area_target = pred_area * torch.exp(area_delta)
        delta_theta = (2.0 * math.pi) / float(self.boundary_samples)
        base_area = 0.5 * delta_theta * shape_scale.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6)
        area_scale = torch.sqrt(mask_area_target / base_area)
        boundary_radius = (area_scale * shape_scale).clamp(min=self.min_radius, max=self.max_radius)

        mask_logits, mask_prob = self.render_closed_mask_from_radii(boundary_radius)
        mask_area = mask_prob.sum(dim=(-1, -2, -3), keepdim=False).unsqueeze(1).clamp_min(1.0)
        latent_from_mask = self.aggregate_expr_with_mask(expr_crops, scale_weights, mask_prob)

        # Keep newer keys as compatibility placeholders so later training /
        # inference utilities do not break while the core model stays v1.
        return {
            "pred_log_area": pred_log_area,
            "pred_area": pred_area,
            "canvas_radius_pref": canvas_radius_pref,
            "routing_canvas_radius_pref": canvas_radius_pref,
            "routing_view_radius_pref": canvas_radius_pref,
            "scale_weights": scale_weights,
            "base_scale_weights": scale_weights,
            "area_delta": area_delta,
            "mask_area": mask_area,
            "boundary_radius": boundary_radius,
            "boundary_confidence_logits": torch.zeros_like(boundary_radius),
            "boundary_confidence": torch.ones_like(boundary_radius),
            "shape_residual_delta": torch.zeros_like(boundary_radius),
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
            "latent_from_mask": latent_from_mask,
        }

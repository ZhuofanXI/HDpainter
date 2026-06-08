from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

from dataset import (
    SpatialTranscriptomicsDataset,
    build_center_seed_mask,
    build_condition_vector,
    build_coord_maps,
    build_multiscale_crop_sizes,
    build_multiscale_expr_crops,
    build_neighbor_maps,
    build_tile_split_indices,
    tile_collate_fn,
)
from model import SizeLatentClosedRegionModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the v1 closed-region baseline model."
    )
    parser.add_argument("--run-name", type=str, default="size_latent_closed_region_hybrid_iou_v1")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--chunk-manifest", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", type=str, default="auto", choices=["auto", "random", "region"])
    parser.add_argument("--min-nuc", type=int, default=10)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--canvas-size", type=int, default=33)
    parser.add_argument("--neighbor-k", type=int, default=4)
    parser.add_argument("--aggregate-radius", type=int, default=5)
    parser.add_argument("--boundary-samples", type=int, default=64)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--canvas-margin", type=float, default=1.5)
    parser.add_argument("--size-weight", type=float, default=0.12)
    parser.add_argument("--latent-weight", type=float, default=0.12)
    parser.add_argument("--size-mask-couple-weight", type=float, default=0.05)
    parser.add_argument("--mask-size-couple-weight", type=float, default=0.05)
    parser.add_argument("--mask-area-weight", type=float, default=0.12)
    parser.add_argument("--scale-supervision-weight", type=float, default=0.05)
    parser.add_argument("--quality-sigmoid-k", type=float, default=8.0)
    parser.add_argument("--quality-threshold", type=float, default=0.65)
    parser.add_argument(
        "--instance-batch-limit",
        type=int,
        default=128,
        help="Maximum number of expanded instances processed per forward/backward micro-batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated CUDA ids, e.g. '0' or '0,1'.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(raw_value: str) -> list[int]:
    text = str(raw_value).strip()
    if not text:
        return []
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate gpu ids are not allowed: {raw_value}")
    return ids


def is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def detect_instance_chunk_h5(path: str | Path) -> bool:
    data_path = Path(path)
    if not data_path.exists():
        return False
    try:
        with h5py.File(data_path, "r") as fr:
            return str(fr.attrs.get("dataset_format", "")).strip() == "instance_chunk_h5_v1"
    except Exception:
        return False


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DataParallel) else model


def resolve_base_dataset(dataset: torch.utils.data.Dataset) -> SpatialTranscriptomicsDataset:
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    if not isinstance(current, SpatialTranscriptomicsDataset):
        raise TypeError(f"Expected SpatialTranscriptomicsDataset, got {type(current)!r}")
    return current


def infer_model_dims(dataset: torch.utils.data.Dataset) -> tuple[int, int, int]:
    base_dataset = resolve_base_dataset(dataset)
    expr_channels = int(base_dataset.expr_channels)
    latent_dim = int(base_dataset.latent_dim)
    cond_dim = int(base_dataset.seed_nmf_dim * 2 + 4)
    return expr_channels, latent_dim, cond_dim


def build_datasets(args: argparse.Namespace) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    dataset_kwargs = dict(
        data_dir=args.data_dir,
        min_nuc=args.min_nuc,
        seed_size=args.seed_size,
        canvas_size=args.canvas_size,
        neighbor_k=args.neighbor_k,
        aggregate_radius=args.aggregate_radius,
        chunk_manifest=args.chunk_manifest or None,
    )
    if detect_instance_chunk_h5(args.data_dir):
        train_dataset = SpatialTranscriptomicsDataset(**dataset_kwargs, chunk_split="train")
        val_dataset = SpatialTranscriptomicsDataset(**dataset_kwargs, chunk_split="val")
        return train_dataset, val_dataset

    base_dataset = SpatialTranscriptomicsDataset(**dataset_kwargs, chunk_split="full")
    train_indices, val_indices = build_tile_split_indices(
        dataset=base_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    return Subset(base_dataset, train_indices), Subset(base_dataset, val_indices)


def flatten_size_bin_ids(size_bin_value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(size_bin_value, torch.Tensor):
        return size_bin_value.reshape(-1).long()
    if isinstance(size_bin_value, list):
        if not size_bin_value:
            return torch.zeros((0,), dtype=torch.long)
        flattened = [tensor.reshape(-1).long() for tensor in size_bin_value if isinstance(tensor, torch.Tensor)]
        if not flattened:
            return torch.zeros((0,), dtype=torch.long)
        return torch.cat(flattened, dim=0)
    raise TypeError(f"Unsupported instance_size_bin_ids type: {type(size_bin_value)!r}")


def build_size_aware_sampler(
    dataset: torch.utils.data.Dataset,
    seed: int,
    mid_sample_boost: float,
    large_sample_boost: float,
    large_count_gain: float,
) -> WeightedRandomSampler:
    weights = torch.ones(len(dataset), dtype=torch.float32)
    for sample_idx in range(len(dataset)):
        item = dataset[sample_idx]
        size_bins = flatten_size_bin_ids(item["instance_size_bin_ids"])
        if size_bins.numel() == 0:
            continue

        mid_count = int((size_bins == 1).sum().item())
        large_count = int((size_bins >= 2).sum().item())
        sample_weight = 1.0
        if mid_count > 0:
            sample_weight *= float(mid_sample_boost)
        if large_count > 0:
            sample_weight *= float(large_sample_boost)
            sample_weight *= 1.0 + float(large_count_gain) * max(0, large_count - 1)
        weights[sample_idx] = float(max(sample_weight, 1e-3))

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(int(seed) + 17)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=sampler_generator,
    )


def compute_scale_supervision_weight_for_epoch(epoch: int, args: argparse.Namespace) -> float:
    base_weight = float(args.scale_supervision_weight)
    if not bool(args.scale_supervision_schedule):
        return base_weight

    early_weight = float(args.scale_supervision_weight_early)
    late_weight = float(args.scale_supervision_weight_late)
    early_epochs = max(1, int(args.scale_supervision_early_epochs))

    if epoch >= early_epochs:
        return late_weight

    if early_epochs == 1:
        return late_weight

    progress = float(epoch - 1) / float(early_epochs - 1)
    return (1.0 - progress) * early_weight + progress * late_weight


def compute_scale_teacher_alpha_for_epoch(epoch: int, args: argparse.Namespace) -> float:
    if not bool(args.scale_teacher_warmup):
        return 0.0
    warmup_epochs = max(0, int(args.scale_teacher_warmup_epochs))
    if warmup_epochs <= 0:
        return 0.0
    if epoch > warmup_epochs:
        return 0.0
    if warmup_epochs == 1:
        return 1.0
    progress = float(epoch - 1) / float(warmup_epochs - 1)
    return max(0.0, 1.0 - progress)


def build_instance_batch_from_tile_batch(
    batch: dict[str, torch.Tensor | list[torch.Tensor] | list[list[str]]],
    canvas_size: int,
    seed_radius: int,
) -> dict[str, torch.Tensor]:
    tile_input = batch["tile_input"]  # type: ignore[assignment]
    instance_centers_yx = batch["instance_centers_yx"]  # type: ignore[assignment]
    instance_seed_nmfs = batch["instance_seed_nmfs"]  # type: ignore[assignment]
    instance_nucleus_areas = batch["instance_nucleus_areas"]  # type: ignore[assignment]
    instance_neighbor_seed_nmfs = batch["instance_neighbor_seed_nmfs"]  # type: ignore[assignment]
    instance_neighbor_nucleus_areas = batch["instance_neighbor_nucleus_areas"]  # type: ignore[assignment]
    instance_neighbor_positions = batch["instance_neighbor_positions"]  # type: ignore[assignment]
    instance_neighbor_valid = batch["instance_neighbor_valid"]  # type: ignore[assignment]
    instance_mask_targets = batch["instance_mask_targets"]  # type: ignore[assignment]
    instance_mask_areas = batch["instance_mask_areas"]  # type: ignore[assignment]
    instance_boundary_radius_targets = batch["instance_boundary_radius_targets"]  # type: ignore[assignment]
    instance_latent_targets = batch["instance_latent_targets"]  # type: ignore[assignment]
    instance_size_bin_ids = batch["instance_size_bin_ids"]  # type: ignore[assignment]

    if not isinstance(tile_input, torch.Tensor):
        raise TypeError("Expected tile_input to be a tensor.")

    crop_sizes = build_multiscale_crop_sizes()
    seed_template = build_center_seed_mask(canvas_size, seed_radius)
    coord_y, coord_x = build_coord_maps(canvas_size)

    expr_crops: list[torch.Tensor] = []
    seed_masks: list[torch.Tensor] = []
    neighbor_seed_maps: list[torch.Tensor] = []
    neighbor_distance_maps: list[torch.Tensor] = []
    cond_vecs: list[torch.Tensor] = []
    area_targets: list[torch.Tensor] = []
    latent_targets: list[torch.Tensor] = []
    mask_targets: list[torch.Tensor] = []
    size_bin_ids: list[torch.Tensor] = []
    radius_targets: list[torch.Tensor] = []

    tile_count = len(instance_centers_yx)
    for tile_idx in range(tile_count):
        centers = instance_centers_yx[tile_idx]
        if not isinstance(centers, torch.Tensor) or centers.numel() == 0:
            continue

        tile_expr = tile_input[tile_idx].float()
        seed_nmfs = instance_seed_nmfs[tile_idx]
        nucleus_areas = instance_nucleus_areas[tile_idx]
        neighbor_seed_nmfs = instance_neighbor_seed_nmfs[tile_idx]
        neighbor_nucleus_areas = instance_neighbor_nucleus_areas[tile_idx]
        neighbor_positions = instance_neighbor_positions[tile_idx]
        neighbor_valid = instance_neighbor_valid[tile_idx]
        mask_target_tile = instance_mask_targets[tile_idx]
        mask_area_tile = instance_mask_areas[tile_idx]
        latent_target_tile = instance_latent_targets[tile_idx]
        radius_target_tile = instance_boundary_radius_targets[tile_idx]
        size_bin_tile = instance_size_bin_ids[tile_idx]

        instance_count = int(centers.shape[0])
        for inst_idx in range(instance_count):
            center_y = float(centers[inst_idx, 0].item())
            center_x = float(centers[inst_idx, 1].item())
            expr_multiscale = build_multiscale_expr_crops(
                tile_expr,
                center_y,
                center_x,
                crop_sizes,
                canvas_size,
            )
            nseed_map, ndist_map = build_neighbor_maps(
                neighbor_positions[inst_idx],
                neighbor_valid[inst_idx],
                coord_y,
                coord_x,
                canvas_size,
                seed_radius,
            )
            cond_vec = build_condition_vector(
                seed_nmfs[inst_idx],
                nucleus_areas[inst_idx],
                neighbor_seed_nmfs[inst_idx],
                neighbor_nucleus_areas[inst_idx],
                neighbor_positions[inst_idx],
                neighbor_valid[inst_idx],
            )

            expr_crops.append(expr_multiscale)
            seed_masks.append(seed_template)
            neighbor_seed_maps.append(nseed_map)
            neighbor_distance_maps.append(ndist_map)
            cond_vecs.append(cond_vec)
            area_targets.append(mask_area_tile[inst_idx].view(1).float())
            latent_targets.append(latent_target_tile[inst_idx].float())
            mask_targets.append(mask_target_tile[inst_idx].float())
            size_bin_ids.append(size_bin_tile[inst_idx].view(1).long())
            radius_targets.append(radius_target_tile[inst_idx].float())

    if not expr_crops:
        raise ValueError("The current batch does not contain any valid instances.")

    return {
        "expr_crops": torch.stack(expr_crops, dim=0),
        "seed_mask": torch.stack(seed_masks, dim=0),
        "neighbor_seed_map": torch.stack(neighbor_seed_maps, dim=0),
        "neighbor_distance_map": torch.stack(neighbor_distance_maps, dim=0),
        "cond_vec": torch.stack(cond_vecs, dim=0),
        "area_target": torch.stack(area_targets, dim=0),
        "latent_target": torch.stack(latent_targets, dim=0),
        "mask_target": torch.stack(mask_targets, dim=0),
        "size_bin_id": torch.cat(size_bin_ids, dim=0),
        "boundary_radius_target": torch.stack(radius_targets, dim=0),
    }


def compute_latent_quality(
    latent_pred: torch.Tensor,
    latent_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_similarity = F.cosine_similarity(latent_pred, latent_target, dim=1)
    pred_pos = latent_pred.clamp_min(0.0)
    target_pos = latent_target.clamp_min(0.0)
    latent_completeness = (
        torch.minimum(pred_pos, target_pos).sum(dim=1)
        / target_pos.sum(dim=1).clamp_min(1e-6)
    )
    quality_score = 0.3 * latent_similarity + 0.7 * latent_completeness
    return latent_similarity, latent_completeness, quality_score


class SizeLatentCoupledLoss(nn.Module):
    def __init__(
        self,
        size_weight: float = 0.12,
        latent_weight: float = 0.12,
        size_mask_couple_weight: float = 0.05,
        mask_size_couple_weight: float = 0.05,
        mask_area_weight: float = 0.12,
        scale_supervision_weight: float = 0.05,
        radius_weight: float = 0.05,
        boundary_confidence_weight: float = 0.00,
        band_weight: float = 0.03,
        quality_sigmoid_k: float = 8.0,
        quality_threshold: float = 0.65,
        canvas_margin: float = 1.5,
        scale_radii: tuple[float, ...] = (7.0, 10.0, 13.0, 16.0),
    ) -> None:
        super().__init__()
        self.size_weight = float(size_weight)
        self.latent_weight = float(latent_weight)
        self.size_mask_couple_weight = float(size_mask_couple_weight)
        self.mask_size_couple_weight = float(mask_size_couple_weight)
        self.mask_area_weight = float(mask_area_weight)
        self.scale_supervision_weight = float(scale_supervision_weight)
        self.radius_weight = float(radius_weight)
        self.boundary_confidence_weight = float(boundary_confidence_weight)
        self.band_weight = float(band_weight)
        self.quality_sigmoid_k = float(quality_sigmoid_k)
        self.quality_threshold = float(quality_threshold)
        self.canvas_margin = float(canvas_margin)
        self.register_buffer("scale_radii", torch.tensor(scale_radii, dtype=torch.float32), persistent=False)

    @staticmethod
    def mask_iou_per_instance(mask_prob: torch.Tensor, mask_target: torch.Tensor) -> torch.Tensor:
        mask_bin = (mask_prob >= 0.5).float()
        inter = (mask_bin * mask_target).sum(dim=(-1, -2, -3))
        union = ((mask_bin + mask_target) > 0).float().sum(dim=(-1, -2, -3)).clamp_min(1e-6)
        return inter / union

    @staticmethod
    def soft_dice_loss_per_instance(mask_prob: torch.Tensor, mask_target: torch.Tensor) -> torch.Tensor:
        inter = (mask_prob * mask_target).sum(dim=(-1, -2, -3))
        denom = mask_prob.sum(dim=(-1, -2, -3)) + mask_target.sum(dim=(-1, -2, -3))
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        return 1.0 - dice

    @staticmethod
    def tversky_loss_per_instance(
        mask_prob: torch.Tensor,
        mask_target: torch.Tensor,
        alpha: float = 0.45,
        beta: float = 0.55,
    ) -> torch.Tensor:
        inter = (mask_prob * mask_target).sum(dim=(-1, -2, -3))
        false_pos = (mask_prob * (1.0 - mask_target)).sum(dim=(-1, -2, -3))
        false_neg = ((1.0 - mask_prob) * mask_target).sum(dim=(-1, -2, -3))
        tversky = (inter + 1e-6) / (inter + alpha * false_pos + beta * false_neg + 1e-6)
        return 1.0 - tversky

    @staticmethod
    def boundary_band_map(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        if mask.ndim != 4:
            raise ValueError(f"boundary_band_map expects [B,1,H,W], got {tuple(mask.shape)}")
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
        return (dilated - eroded).clamp(0.0, 1.0)

    @staticmethod
    def build_target_scale_probs_from_area(
        gt_area: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        area_anchors = torch.tensor([16.0, 32.0, 56.0, 96.0], device=device, dtype=dtype)
        scale_profiles = torch.tensor(
            [
                [0.62, 0.26, 0.09, 0.03],
                [0.32, 0.40, 0.20, 0.08],
                [0.14, 0.32, 0.33, 0.21],
                [0.05, 0.15, 0.36, 0.44],
            ],
            device=device,
            dtype=dtype,
        )

        area = gt_area.to(device=device, dtype=dtype)
        target = torch.zeros((area.shape[0], 4), device=device, dtype=dtype)

        low_mask = area <= area_anchors[0]
        high_mask = area >= area_anchors[-1]
        target[low_mask] = scale_profiles[0]
        target[high_mask] = scale_profiles[-1]

        for idx in range(len(area_anchors)):
            equal_mask = area == area_anchors[idx]
            if bool(equal_mask.any()):
                target[equal_mask] = scale_profiles[idx]

        for idx in range(len(area_anchors) - 1):
            left = area_anchors[idx]
            right = area_anchors[idx + 1]
            seg_mask = (area > left) & (area < right)
            if not bool(seg_mask.any()):
                continue
            t = ((area[seg_mask] - left) / (right - left)).unsqueeze(1)
            target[seg_mask] = (1.0 - t) * scale_profiles[idx] + t * scale_profiles[idx + 1]

        row_sum = target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return target / row_sum

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        area_target: torch.Tensor,
        latent_target: torch.Tensor,
        mask_target: torch.Tensor,
        size_bin_id: torch.Tensor,
        boundary_radius_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        pred_log_area = outputs["pred_log_area"].float()
        pred_area = outputs["pred_area"].float().squeeze(1)
        mask_area = outputs["mask_area"].float().squeeze(1)
        latent_from_mask = outputs["latent_from_mask"].float()
        mask_prob = outputs["mask_prob"].float()
        scale_weights = outputs["scale_weights"].float()
        base_scale_weights = outputs["base_scale_weights"].float()
        canvas_radius_pref = outputs["canvas_radius_pref"].float().squeeze(1)
        routing_view_radius_pref = outputs["routing_view_radius_pref"].float().squeeze(1)
        boundary_radius = outputs["boundary_radius"].float()
        boundary_confidence_logits = outputs["boundary_confidence_logits"].float()
        boundary_confidence = outputs["boundary_confidence"].float()
        shape_residual_delta = outputs["shape_residual_delta"].float()
        mask_target = mask_target.float()
        boundary_radius_target = boundary_radius_target.float()

        dice_loss_per = self.soft_dice_loss_per_instance(mask_prob, mask_target)
        tversky_loss_per = self.tversky_loss_per_instance(mask_prob, mask_target)
        seg_loss_per = 0.6 * dice_loss_per + 0.4 * tversky_loss_per
        pred_band = self.boundary_band_map(mask_prob)
        target_band = self.boundary_band_map(mask_target)
        band_loss_per = self.soft_dice_loss_per_instance(pred_band, target_band)

        log_area_target = torch.log(area_target.float().clamp_min(1.0))
        size_gt_per = F.smooth_l1_loss(
            pred_log_area,
            log_area_target,
            reduction="none",
        ).squeeze(1)

        latent_similarity, latent_completeness, quality_score = compute_latent_quality(latent_from_mask, latent_target.float())
        latent_loss_per = 0.3 * (1.0 - latent_similarity) + 0.7 * (1.0 - latent_completeness)

        radius_abs_error = torch.abs(boundary_radius - boundary_radius_target)
        radius_loss_per_angle = F.smooth_l1_loss(
            boundary_radius,
            boundary_radius_target,
            reduction="none",
        )
        confidence_target = torch.exp(-radius_abs_error.detach() / 2.0).clamp(0.0, 1.0)
        boundary_confidence_loss_per = F.binary_cross_entropy_with_logits(
            boundary_confidence_logits,
            confidence_target,
            reduction="none",
        ).mean(dim=1)

        T = torch.sigmoid(
            self.quality_sigmoid_k * (quality_score.detach() - self.quality_threshold)
        )
        log_pred_area = pred_log_area.squeeze(1)
        log_mask_area = torch.log(mask_area.clamp_min(1.0))

        size_from_mask_per = T * F.smooth_l1_loss(
            log_pred_area,
            log_mask_area.detach(),
            reduction="none",
        )
        mask_from_size_per = (1.0 - T) * F.smooth_l1_loss(
            log_mask_area,
            log_pred_area.detach(),
            reduction="none",
        )

        gt_area = area_target.float().squeeze(1)
        small_mask = (gt_area < 24.0).float()
        mid_mask = ((gt_area >= 24.0) & (gt_area < 64.0)).float()
        large_mask = (gt_area >= 64.0).float()

        radius_weight_map = (
            1.0 * small_mask
            + 1.1 * mid_mask
            + 1.35 * large_mask
        )

        size_gt_weight_map = (
            1.0 * small_mask
            + 1.2 * mid_mask
            + 2.0 * large_mask
        )
        size_from_mask_weight_map = (
            1.0 * small_mask
            + 1.0 * mid_mask
            + 1.2 * large_mask
        )
        mask_from_size_weight_map = (
            1.0 * small_mask
            + 1.4 * mid_mask
            + 3.0 * large_mask
        )

        size_gt_per = size_gt_per * size_gt_weight_map
        size_from_mask_per = size_from_mask_per * size_from_mask_weight_map
        mask_from_size_per = mask_from_size_per * mask_from_size_weight_map
        radius_loss_per = radius_loss_per_angle.mean(dim=1) * radius_weight_map
        boundary_confidence_loss_per = boundary_confidence_loss_per * radius_weight_map

        log_gt_area = torch.log(gt_area.clamp_min(1.0))
        log_mask_minus_gt = log_mask_area - log_gt_area
        mask_area_over = F.relu(log_mask_minus_gt)
        mask_area_under = F.relu(-log_mask_minus_gt)
        continuous_radius = torch.sqrt(gt_area.clamp_min(1.0) / math.pi) + self.canvas_margin
        size_progress = torch.sigmoid((continuous_radius - 6.5) / 1.5)
        over_weight_map = 0.80 + (1.55 - 0.80) * (1.0 - size_progress)
        under_weight_map = 0.72 + (2.10 - 0.72) * size_progress
        mask_area_calib_per = over_weight_map * mask_area_over + under_weight_map * mask_area_under

        target_scale_probs = self.build_target_scale_probs_from_area(
            gt_area,
            device=scale_weights.device,
            dtype=scale_weights.dtype,
        ).detach()
        scale_supervision_per = -(target_scale_probs * torch.log(scale_weights.clamp_min(1e-6))).sum(dim=1)
        scale_entropy_per = -(scale_weights.clamp_min(1e-6) * torch.log(scale_weights.clamp_min(1e-6))).sum(dim=1)

        total_per = (
            seg_loss_per
            + self.band_weight * band_loss_per
            + self.radius_weight * radius_loss_per
            + self.boundary_confidence_weight * boundary_confidence_loss_per
            + self.latent_weight * latent_loss_per
            + self.size_weight * size_gt_per
            + self.mask_area_weight * mask_area_calib_per
            + self.scale_supervision_weight * scale_supervision_per
            + self.size_mask_couple_weight * size_from_mask_per
            + self.mask_size_couple_weight * mask_from_size_per
        )
        total = total_per.mean()

        pred_area_ratio = pred_area / gt_area.clamp_min(1.0)
        mask_area_ratio = mask_area / gt_area.clamp_min(1.0)
        mask_iou_per = self.mask_iou_per_instance(mask_prob, mask_target.float())

        def _group_mean(values: torch.Tensor, selector: torch.Tensor) -> float:
            denom = selector.sum().clamp_min(1.0)
            return float(((values * selector).sum() / denom).detach().item())

        metrics = {
            "loss": float(total.detach().item()),
            "seg_loss": float(seg_loss_per.mean().detach().item()),
            "band_loss": float(band_loss_per.mean().detach().item()),
            "dice_loss": float(dice_loss_per.mean().detach().item()),
            "tversky_loss": float(tversky_loss_per.mean().detach().item()),
            "radius_loss": float(radius_loss_per.mean().detach().item()),
            "boundary_confidence_loss": float(boundary_confidence_loss_per.mean().detach().item()),
            "latent_loss": float(latent_loss_per.mean().detach().item()),
            "size_loss": float(size_gt_per.mean().detach().item()),
            "mask_area_calib_loss": float(mask_area_calib_per.mean().detach().item()),
            "scale_supervision_loss": float(scale_supervision_per.mean().detach().item()),
            "scale_entropy": float(scale_entropy_per.mean().detach().item()),
            "size_from_mask_loss": float(size_from_mask_per.mean().detach().item()),
            "mask_from_size_loss": float(mask_from_size_per.mean().detach().item()),
            "latent_similarity": float(latent_similarity.mean().detach().item()),
            "latent_completeness": float(latent_completeness.mean().detach().item()),
            "boundary_confidence_mean": float(boundary_confidence.mean().detach().item()),
            "boundary_residual_abs_mean": float(shape_residual_delta.abs().mean().detach().item()),
            "mask_iou": float(mask_iou_per.mean().detach().item()),
            "area_ratio": float(mask_area_ratio.mean().detach().item()),
            "pred_area_ratio": float(pred_area_ratio.mean().detach().item()),
            "mask_area_ratio": float(mask_area_ratio.mean().detach().item()),
            "pred_area_mean": float(pred_area.mean().detach().item()),
            "mask_area_mean": float(mask_area.mean().detach().item()),
            "gt_area_mean": float(gt_area.mean().detach().item()),
            "canvas_radius_pref_mean": float(canvas_radius_pref.mean().detach().item()),
            "routing_view_radius_pref_mean": float(routing_view_radius_pref.mean().detach().item()),
            "base_scale_w_0": float(base_scale_weights[:, 0].mean().detach().item()),
            "base_scale_w_1": float(base_scale_weights[:, 1].mean().detach().item()),
            "base_scale_w_2": float(base_scale_weights[:, 2].mean().detach().item()),
            "base_scale_w_3": float(base_scale_weights[:, 3].mean().detach().item()),
            "small_iou": _group_mean(mask_iou_per, small_mask),
            "mid_iou": _group_mean(mask_iou_per, mid_mask),
            "large_iou": _group_mean(mask_iou_per, large_mask),
            "small_latent_similarity": _group_mean(latent_similarity, small_mask),
            "mid_latent_similarity": _group_mean(latent_similarity, mid_mask),
            "large_latent_similarity": _group_mean(latent_similarity, large_mask),
            "small_latent_completeness": _group_mean(latent_completeness, small_mask),
            "mid_latent_completeness": _group_mean(latent_completeness, mid_mask),
            "large_latent_completeness": _group_mean(latent_completeness, large_mask),
            "small_area_ratio": _group_mean(mask_area_ratio, small_mask),
            "mid_area_ratio": _group_mean(mask_area_ratio, mid_mask),
            "large_area_ratio": _group_mean(mask_area_ratio, large_mask),
            "small_pred_area_ratio": _group_mean(pred_area_ratio, small_mask),
            "mid_pred_area_ratio": _group_mean(pred_area_ratio, mid_mask),
            "large_pred_area_ratio": _group_mean(pred_area_ratio, large_mask),
            "small_mask_area_ratio": _group_mean(mask_area_ratio, small_mask),
            "mid_mask_area_ratio": _group_mean(mask_area_ratio, mid_mask),
            "large_mask_area_ratio": _group_mean(mask_area_ratio, large_mask),
        }
        scale_mean = scale_weights.mean(dim=0)
        for idx in range(scale_weights.shape[1]):
            metrics[f"scale_w_{idx}"] = float(scale_mean[idx].detach().item())

        aux = {
            "mask_iou": mask_iou_per.detach(),
            "area_ratio": mask_area_ratio.detach(),
            "pred_area_ratio": pred_area_ratio.detach(),
            "mask_area_ratio": mask_area_ratio.detach(),
            "pred_area": pred_area.detach(),
            "mask_area": mask_area.detach(),
            "gt_area": gt_area.detach(),
            "latent_similarity": latent_similarity.detach(),
            "latent_completeness": latent_completeness.detach(),
            "scale_weights": scale_weights.detach(),
            "canvas_radius_pref": canvas_radius_pref.detach(),
            "routing_view_radius_pref": routing_view_radius_pref.detach(),
            "size_bin_id": size_bin_id.detach(),
        }
        return total, metrics, aux


def run_step(
    batch: dict[str, torch.Tensor | list[torch.Tensor] | list[list[str]]],
    model: nn.Module,
    loss_fn: SizeLatentCoupledLoss,
    device: str,
    scaler: GradScaler,
    canvas_size: int,
    seed_radius: int,
    instance_batch_limit: int,
    canvas_margin: float,
    teacher_scale_alpha: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    collect_stats: bool = False,
) -> tuple[float, dict[str, float], dict[str, np.ndarray] | None]:
    instance_batch = build_instance_batch_from_tile_batch(batch, canvas_size, seed_radius)
    total_instances = int(instance_batch["expr_crops"].shape[0])
    if total_instances <= 0:
        raise ValueError("Expanded instance batch is empty.")

    chunk_size = max(1, int(instance_batch_limit))
    use_amp = is_cuda_device(device)

    metric_weighted_sums: dict[str, float] = {}
    stats_chunks: list[dict[str, np.ndarray]] = []
    total_loss_value = 0.0

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for start in range(0, total_instances, chunk_size):
        end = min(start + chunk_size, total_instances)
        chunk_weight = float(end - start) / float(total_instances)

        expr_crops = instance_batch["expr_crops"][start:end].to(device, non_blocking=True)
        seed_mask = instance_batch["seed_mask"][start:end].to(device, non_blocking=True)
        neighbor_seed_map = instance_batch["neighbor_seed_map"][start:end].to(device, non_blocking=True)
        neighbor_distance_map = instance_batch["neighbor_distance_map"][start:end].to(device, non_blocking=True)
        cond_vec = instance_batch["cond_vec"][start:end].to(device, non_blocking=True)
        area_target = instance_batch["area_target"][start:end].to(device, non_blocking=True)
        latent_target = instance_batch["latent_target"][start:end].to(device, non_blocking=True)
        mask_target = instance_batch["mask_target"][start:end].to(device, non_blocking=True)
        size_bin_id = instance_batch["size_bin_id"][start:end].to(device, non_blocking=True)
        boundary_radius_target = instance_batch["boundary_radius_target"][start:end].to(device, non_blocking=True)
        teacher_canvas_radius_pref = None
        if optimizer is not None and teacher_scale_alpha > 0.0:
            teacher_canvas_radius_pref = (
                torch.ceil(torch.sqrt(area_target.float().clamp_min(1.0) / math.pi)) + float(canvas_margin)
            )

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(
                expr_crops=expr_crops,
                seed_mask=seed_mask,
                neighbor_seed_map=neighbor_seed_map,
                neighbor_distance_map=neighbor_distance_map,
                cond_vec=cond_vec,
                teacher_canvas_radius_pref=teacher_canvas_radius_pref,
                teacher_scale_alpha=teacher_scale_alpha if teacher_canvas_radius_pref is not None else None,
            )
            loss, metrics, aux = loss_fn(
                outputs=outputs,
                area_target=area_target,
                latent_target=latent_target,
                mask_target=mask_target,
                size_bin_id=size_bin_id,
                boundary_radius_target=boundary_radius_target,
            )

        if optimizer is not None:
            scaler.scale(loss * chunk_weight).backward()

        chunk_count = int(end - start)
        total_loss_value += float(loss.detach().item()) * chunk_count
        for key, value in metrics.items():
            metric_weighted_sums[key] = metric_weighted_sums.get(key, 0.0) + float(value) * chunk_count

        if collect_stats:
            stats_chunks.append(
                {
                    "mask_iou": aux["mask_iou"].cpu().numpy(),
                    "area_ratio": aux["area_ratio"].cpu().numpy(),
                    "pred_area_ratio": aux["pred_area_ratio"].cpu().numpy(),
                    "mask_area_ratio": aux["mask_area_ratio"].cpu().numpy(),
                    "pred_area": aux["pred_area"].cpu().numpy(),
                    "mask_area": aux["mask_area"].cpu().numpy(),
                    "gt_area": aux["gt_area"].cpu().numpy(),
                    "latent_similarity": aux["latent_similarity"].cpu().numpy(),
                    "latent_completeness": aux["latent_completeness"].cpu().numpy(),
                    "scale_weights": aux["scale_weights"].cpu().numpy(),
                    "canvas_radius_pref": aux["canvas_radius_pref"].cpu().numpy(),
                    "routing_view_radius_pref": aux["routing_view_radius_pref"].cpu().numpy(),
                    "size_bin_id": aux["size_bin_id"].cpu().numpy().astype(np.int64, copy=False),
                }
            )

        del expr_crops, seed_mask, neighbor_seed_map, neighbor_distance_map, cond_vec
        del area_target, latent_target, mask_target, size_bin_id, outputs, loss, aux

    if optimizer is not None:
        scaler.step(optimizer)
        scaler.update()

    metrics = {key: value / float(total_instances) for key, value in metric_weighted_sums.items()}
    metrics["n_instances"] = total_instances
    stats = merge_stats(stats_chunks) if collect_stats else None
    return total_loss_value / float(total_instances), metrics, stats


def merge_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
    if not items:
        return None
    keys = set().union(*(item.keys() for item in items))
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        if values:
            merged[key] = np.concatenate(values, axis=0)
    return merged


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SizeLatentCoupledLoss,
    device: str,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    epochs: int,
    mode: str,
    log_every: int,
    canvas_size: int,
    seed_radius: int,
    instance_batch_limit: int,
    canvas_margin: float,
    teacher_scale_alpha: float = 0.0,
    collect_stats: bool = False,
) -> tuple[float, dict[str, float], dict[str, np.ndarray] | None]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_instances = 0
    metric_sums: dict[str, float] = {}
    stats_items: list[dict[str, np.ndarray]] = []

    progress = tqdm(loader, desc=f"{mode} {epoch:02d}/{epochs:02d}")
    for step_idx, batch in enumerate(progress, start=1):
        if is_train:
            step_loss, metrics, stats = run_step(
                batch=batch,
                model=model,
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                canvas_size=canvas_size,
                seed_radius=seed_radius,
                instance_batch_limit=instance_batch_limit,
                canvas_margin=canvas_margin,
                teacher_scale_alpha=teacher_scale_alpha,
                optimizer=optimizer,
                collect_stats=collect_stats,
            )
        else:
            with torch.no_grad():
                step_loss, metrics, stats = run_step(
                    batch=batch,
                    model=model,
                    loss_fn=loss_fn,
                    device=device,
                    scaler=scaler,
                    canvas_size=canvas_size,
                    seed_radius=seed_radius,
                    instance_batch_limit=instance_batch_limit,
                    canvas_margin=canvas_margin,
                    teacher_scale_alpha=0.0,
                    optimizer=None,
                    collect_stats=collect_stats,
                )

        n_instances = int(metrics.pop("n_instances"))
        total_loss += step_loss * n_instances
        total_instances += n_instances
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * n_instances
        if stats is not None:
            stats_items.append(stats)

        if step_idx % max(1, log_every) == 0 or step_idx == len(loader):
            couple_value = metrics["size_from_mask_loss"] + metrics["mask_from_size_loss"]
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                couple=f"{couple_value:.4f}",
                iou=f"{metrics['mask_iou']:.4f}",
                area=f"{metrics['mask_area_mean']:.1f}/{metrics['gt_area_mean']:.1f}",
            )

    denom = max(1, total_instances)
    epoch_metrics = {key: value / denom for key, value in metric_sums.items()}
    epoch_loss = total_loss / denom
    return epoch_loss, epoch_metrics, merge_stats(stats_items)


def build_epoch_analysis_row(
    epoch: int,
    split: str,
    stats: dict[str, np.ndarray] | None,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"epoch": int(epoch), "split": str(split)}
    if not stats:
        prefixes = ("overall", "small", "mid", "large")
        for prefix in prefixes:
            row[f"{prefix}_count"] = 0
            for key in (
                "mask_iou",
                "area_ratio",
                "pred_area_ratio",
                "mask_area_ratio",
                "pred_area",
                "mask_area",
                "gt_area",
                "latent_similarity",
                "latent_completeness",
                "canvas_radius_pref",
                "routing_view_radius_pref",
            ):
                row[f"{prefix}_{key}"] = 0.0
            for scale_idx in range(4):
                row[f"{prefix}_scale_w_{scale_idx}"] = 0.0
        return row

    gt_area = stats["gt_area"].astype(np.float32, copy=False)
    size_masks = {
        "overall": np.ones_like(gt_area, dtype=bool),
        "small": gt_area < 24.0,
        "mid": (gt_area >= 24.0) & (gt_area < 64.0),
        "large": gt_area >= 64.0,
    }
    for prefix, selector in size_masks.items():
        count = int(selector.sum())
        row[f"{prefix}_count"] = count
        if count == 0:
            for key in (
                "mask_iou",
                "area_ratio",
                "pred_area_ratio",
                "mask_area_ratio",
                "pred_area",
                "mask_area",
                "gt_area",
                "latent_similarity",
                "latent_completeness",
                "canvas_radius_pref",
                "routing_view_radius_pref",
            ):
                row[f"{prefix}_{key}"] = 0.0
            for scale_idx in range(stats["scale_weights"].shape[1]):
                row[f"{prefix}_scale_w_{scale_idx}"] = 0.0
            continue
        row[f"{prefix}_mask_iou"] = float(stats["mask_iou"][selector].mean())
        row[f"{prefix}_area_ratio"] = float(stats["area_ratio"][selector].mean())
        row[f"{prefix}_pred_area_ratio"] = float(stats["pred_area_ratio"][selector].mean())
        row[f"{prefix}_mask_area_ratio"] = float(stats["mask_area_ratio"][selector].mean())
        row[f"{prefix}_pred_area"] = float(stats["pred_area"][selector].mean())
        row[f"{prefix}_mask_area"] = float(stats["mask_area"][selector].mean())
        row[f"{prefix}_gt_area"] = float(stats["gt_area"][selector].mean())
        row[f"{prefix}_latent_similarity"] = float(stats["latent_similarity"][selector].mean())
        row[f"{prefix}_latent_completeness"] = float(stats["latent_completeness"][selector].mean())
        row[f"{prefix}_canvas_radius_pref"] = float(stats["canvas_radius_pref"][selector].mean())
        routing_values = stats.get("routing_view_radius_pref")
        row[f"{prefix}_routing_view_radius_pref"] = (
            float(routing_values[selector].mean()) if routing_values is not None else 0.0
        )
        scale_values = stats["scale_weights"][selector]
        for scale_idx in range(scale_values.shape[1]):
            row[f"{prefix}_scale_w_{scale_idx}"] = float(scale_values[:, scale_idx].mean())
    return row


def write_analysis_summary(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_size_bin_summary(stats: dict[str, np.ndarray] | None, output_path: Path) -> None:
    if not stats:
        return
    routing_view_radius_pref = stats.get("routing_view_radius_pref")
    size_bin_ids = stats["size_bin_id"].astype(np.int64, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(
            [
                "size_bin_id",
                "count",
                "mean_mask_iou",
                "mean_area_ratio",
                "mean_pred_area_ratio",
                "mean_mask_area_ratio",
                "mean_pred_area",
                "mean_mask_area",
                "mean_gt_area",
                "mean_latent_similarity",
                "mean_latent_completeness",
                "mean_canvas_radius_pref",
                "mean_routing_view_radius_pref",
                "mean_scale_w_0",
                "mean_scale_w_1",
                "mean_scale_w_2",
                "mean_scale_w_3",
            ]
        )
        for bin_id in sorted(np.unique(size_bin_ids).tolist()):
            selector = size_bin_ids == int(bin_id)
            scale_values = stats["scale_weights"][selector]
            writer.writerow(
                [
                    int(bin_id),
                    int(selector.sum()),
                    float(stats["mask_iou"][selector].mean()) if selector.any() else 0.0,
                    float(stats["area_ratio"][selector].mean()) if selector.any() else 0.0,
                    float(stats["pred_area_ratio"][selector].mean()) if selector.any() else 0.0,
                    float(stats["mask_area_ratio"][selector].mean()) if selector.any() else 0.0,
                    float(stats["pred_area"][selector].mean()) if selector.any() else 0.0,
                    float(stats["mask_area"][selector].mean()) if selector.any() else 0.0,
                    float(stats["gt_area"][selector].mean()) if selector.any() else 0.0,
                    float(stats["latent_similarity"][selector].mean()) if selector.any() else 0.0,
                    float(stats["latent_completeness"][selector].mean()) if selector.any() else 0.0,
                    float(stats["canvas_radius_pref"][selector].mean()) if selector.any() else 0.0,
                    float(routing_view_radius_pref[selector].mean())
                    if selector.any() and routing_view_radius_pref is not None
                    else 0.0,
                    float(scale_values[:, 0].mean()) if selector.any() else 0.0,
                    float(scale_values[:, 1].mean()) if selector.any() else 0.0,
                    float(scale_values[:, 2].mean()) if selector.any() else 0.0,
                    float(scale_values[:, 3].mean()) if selector.any() else 0.0,
                ]
            )


def save_core_visualizations(stats: dict[str, np.ndarray] | None, output_dir: Path, epoch: int, split: str) -> None:
    if not stats:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_iou = stats["mask_iou"].astype(np.float32, copy=False)
    area_ratio = stats["area_ratio"].astype(np.float32, copy=False)
    pred_area_ratio = stats["pred_area_ratio"].astype(np.float32, copy=False)
    mask_area_ratio = stats["mask_area_ratio"].astype(np.float32, copy=False)
    pred_area = stats["pred_area"].astype(np.float32, copy=False)
    mask_area = stats["mask_area"].astype(np.float32, copy=False)
    gt_area = stats["gt_area"].astype(np.float32, copy=False)
    latent_similarity = stats["latent_similarity"].astype(np.float32, copy=False)
    latent_completeness = stats["latent_completeness"].astype(np.float32, copy=False)
    scale_weights = stats["scale_weights"].astype(np.float32, copy=False)
    size_bin_ids = stats["size_bin_id"].astype(np.int64, copy=False)

    def _save_hist(values: np.ndarray, title: str, xlabel: str, filename: str, color: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=30, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    _save_hist(mask_iou, f"{split} epoch {epoch:03d} IoU distribution", "Mask IoU", f"epoch_{epoch:03d}_{split}_iou_distribution.png", "#2f6db3")
    _save_hist(mask_area_ratio, f"{split} epoch {epoch:03d} mask area ratio distribution", "Mask area / GT area", f"epoch_{epoch:03d}_{split}_area_ratio_distribution.png", "#d07c28")
    _save_hist(pred_area_ratio, f"{split} epoch {epoch:03d} predicted area ratio distribution", "Pred area / GT area", f"epoch_{epoch:03d}_{split}_pred_area_ratio_distribution.png", "#c44e52")
    _save_hist(latent_similarity, f"{split} epoch {epoch:03d} latent similarity", "Cosine similarity", f"epoch_{epoch:03d}_{split}_latent_similarity_distribution.png", "#2c8f6b")
    _save_hist(latent_completeness, f"{split} epoch {epoch:03d} latent completeness", "Completeness", f"epoch_{epoch:03d}_{split}_latent_completeness_distribution.png", "#914d9c")

    fig, ax = plt.subplots(figsize=(6, 4))
    sample_count = min(5000, mask_area.size)
    if mask_area.size > sample_count:
        pick = np.random.default_rng(0).choice(mask_area.size, size=sample_count, replace=False)
        scatter_pred = mask_area[pick]
        scatter_gt = gt_area[pick]
    else:
        scatter_pred = mask_area
        scatter_gt = gt_area
    ax.scatter(scatter_gt, scatter_pred, s=8, alpha=0.35, color="#2c8f6b")
    gt_max = float(scatter_gt.max()) if scatter_gt.size > 0 else 1.0
    pred_max = float(scatter_pred.max()) if scatter_pred.size > 0 else 1.0
    lim_max = max(gt_max, pred_max, 1.0)
    ax.plot([0.0, lim_max], [0.0, lim_max], linestyle="--", color="#666666", linewidth=1.0)
    ax.set_xlim(0.0, lim_max)
    ax.set_ylim(0.0, lim_max)
    ax.set_title(f"{split} epoch {epoch:03d} mask area vs gt area")
    ax.set_xlabel("GT area")
    ax.set_ylabel("Mask area")
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:03d}_{split}_pred_vs_gt_area_scatter.png", dpi=160)
    plt.close(fig)

    unique_bins = sorted(np.unique(size_bin_ids).tolist())
    mean_iou = [
        float(mask_iou[size_bin_ids == int(bin_id)].mean()) if np.any(size_bin_ids == int(bin_id)) else 0.0
        for bin_id in unique_bins
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(int(bin_id)) for bin_id in unique_bins], mean_iou, color="#7a64c9", alpha=0.9)
    ax.set_title(f"{split} epoch {epoch:03d} size-bin IoU")
    ax.set_xlabel("Size bin id")
    ax.set_ylabel("Mean IoU")
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:03d}_{split}_size_bin_iou_bar.png", dpi=160)
    plt.close(fig)

    gt_area_for_group = gt_area
    group_selectors = {
        "small": gt_area_for_group < 24.0,
        "mid": (gt_area_for_group >= 24.0) & (gt_area_for_group < 64.0),
        "large": gt_area_for_group >= 64.0,
    }
    group_means = []
    group_labels = []
    for label, selector in group_selectors.items():
        if selector.any():
            group_means.append(scale_weights[selector].mean(axis=0))
        else:
            group_means.append(np.zeros((scale_weights.shape[1],), dtype=np.float32))
        group_labels.append(label)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(group_labels))
    width = 0.18
    for scale_idx in range(scale_weights.shape[1]):
        vals = [group_means[group_idx][scale_idx] for group_idx in range(len(group_labels))]
        ax.bar(x + (scale_idx - 1.5) * width, vals, width=width, label=f"s{scale_idx}")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean scale weight")
    ax.set_title(f"{split} epoch {epoch:03d} scale weights by size")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:03d}_{split}_scale_weights_by_size.png", dpi=160)
    plt.close(fig)


def save_checkpoint(run_dir: Path, model: nn.Module, optimizer: Adam, epoch: int, save_latest: bool) -> None:
    state = {
        "epoch": int(epoch),
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / f"epoch_{epoch:03d}.pt")
    if save_latest:
        torch.save(state, checkpoint_dir / "latest.pt")


def write_epoch_metrics(rows: list[dict[str, float | int]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_training_summary(best_row: dict[str, float | int], final_row: dict[str, float | int], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(
            [
                "split",
                "epoch",
                "loss",
                "seg_loss",
                "latent_loss",
                "size_loss",
                "mask_iou",
                "latent_similarity",
                "latent_completeness",
                "area_ratio",
            ]
        )
        writer.writerow(
            [
                "best_val",
                best_row["epoch"],
                best_row["val_loss"],
                best_row["val_seg_loss"],
                best_row["val_latent_loss"],
                best_row["val_size_loss"],
                best_row["val_mask_iou"],
                best_row["val_latent_similarity"],
                best_row["val_latent_completeness"],
                best_row["val_area_ratio"],
            ]
        )
        writer.writerow(
            [
                "final_val",
                final_row["epoch"],
                final_row["val_loss"],
                final_row["val_seg_loss"],
                final_row["val_latent_loss"],
                final_row["val_size_loss"],
                final_row["val_mask_iou"],
                final_row["val_latent_similarity"],
                final_row["val_latent_completeness"],
                final_row["val_area_ratio"],
            ]
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if torch.cuda.is_available():
        if gpu_ids:
            primary_gpu = gpu_ids[0]
            torch.cuda.set_device(primary_gpu)
            device = f"cuda:{primary_gpu}"
        else:
            device = "cuda"
    else:
        device = "cpu"

    script_dir = Path(__file__).resolve().parent
    run_dir = script_dir / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_datasets(args)
    expr_channels, latent_dim, cond_dim = infer_model_dims(train_dataset)

    model = SizeLatentClosedRegionModel(
        expr_channels=expr_channels,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        canvas_size=args.canvas_size,
        boundary_samples=args.boundary_samples,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        canvas_margin=args.canvas_margin,
        boundary_residual_scale=args.boundary_residual_scale,
    )
    if is_cuda_device(device):
        model = model.to(device)
    if is_cuda_device(device) and len(gpu_ids) > 1:
        model = DataParallel(model, device_ids=gpu_ids)

    train_sampler = None
    if args.size_aware_sampler:
        train_sampler = build_size_aware_sampler(
            dataset=train_dataset,
            seed=args.seed,
            mid_sample_boost=args.mid_sample_boost,
            large_sample_boost=args.large_sample_boost,
            large_count_gain=args.large_count_gain,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device(device),
        collate_fn=tile_collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device(device),
        collate_fn=tile_collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=is_cuda_device(device))
    loss_fn = SizeLatentCoupledLoss(
        size_weight=args.size_weight,
        latent_weight=args.latent_weight,
        size_mask_couple_weight=args.size_mask_couple_weight,
        mask_size_couple_weight=args.mask_size_couple_weight,
        mask_area_weight=args.mask_area_weight,
        scale_supervision_weight=args.scale_supervision_weight,
        radius_weight=args.radius_weight,
        boundary_confidence_weight=args.boundary_confidence_weight,
        band_weight=args.band_weight,
        quality_sigmoid_k=args.quality_sigmoid_k,
        quality_threshold=args.quality_threshold,
        canvas_margin=args.canvas_margin,
    )

    if train_sampler is not None:
        print(
            "train sampler: size-aware "
            f"(mid_boost={args.mid_sample_boost:.2f}, "
            f"large_boost={args.large_sample_boost:.2f}, "
            f"large_count_gain={args.large_count_gain:.2f})"
        )
    else:
        print("train sampler: shuffle")
    if args.scale_supervision_schedule:
        print(
            "scale supervision schedule: "
            f"early={args.scale_supervision_weight_early:.3f} -> "
            f"late={args.scale_supervision_weight_late:.3f} over "
            f"{args.scale_supervision_early_epochs} epochs"
        )
    else:
        print(f"scale supervision schedule: disabled (fixed={args.scale_supervision_weight:.3f})")
    if args.scale_teacher_warmup:
        print(f"scale teacher warmup: enabled over {args.scale_teacher_warmup_epochs} epochs")
    else:
        print("scale teacher warmup: disabled")

    epoch_rows: list[dict[str, float | int]] = []
    analysis_rows: list[dict[str, float | int | str]] = []
    best_row: dict[str, float | int] | None = None

    for epoch in range(1, args.epochs + 1):
        current_scale_supervision_weight = compute_scale_supervision_weight_for_epoch(epoch, args)
        current_scale_teacher_alpha = compute_scale_teacher_alpha_for_epoch(epoch, args)
        loss_fn.scale_supervision_weight = float(current_scale_supervision_weight)
        train_loss, train_metrics, train_stats = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            optimizer=optimizer,
            epoch=epoch,
            epochs=args.epochs,
            mode="train",
            log_every=args.log_every,
            canvas_size=args.canvas_size,
            seed_radius=args.seed_size,
            instance_batch_limit=args.instance_batch_limit,
            canvas_margin=args.canvas_margin,
            teacher_scale_alpha=current_scale_teacher_alpha,
            collect_stats=True,
        )
        val_loss, val_metrics, val_stats = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            optimizer=None,
            epoch=epoch,
            epochs=args.epochs,
            mode="val",
            log_every=args.log_every,
            canvas_size=args.canvas_size,
            seed_radius=args.seed_size,
            instance_batch_limit=args.instance_batch_limit,
            canvas_margin=args.canvas_margin,
            teacher_scale_alpha=0.0,
            collect_stats=True,
        )

        row: dict[str, float | int] = {
            "epoch": epoch,
            "scale_supervision_weight": current_scale_supervision_weight,
            "scale_teacher_alpha": current_scale_teacher_alpha,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_seg_loss": train_metrics["seg_loss"],
            "val_seg_loss": val_metrics["seg_loss"],
            "train_band_loss": train_metrics["band_loss"],
            "val_band_loss": val_metrics["band_loss"],
            "train_radius_loss": train_metrics["radius_loss"],
            "val_radius_loss": val_metrics["radius_loss"],
            "train_boundary_confidence_loss": train_metrics["boundary_confidence_loss"],
            "val_boundary_confidence_loss": val_metrics["boundary_confidence_loss"],
            "train_latent_loss": train_metrics["latent_loss"],
            "val_latent_loss": val_metrics["latent_loss"],
            "train_size_loss": train_metrics["size_loss"],
            "val_size_loss": val_metrics["size_loss"],
            "train_mask_area_calib_loss": train_metrics["mask_area_calib_loss"],
            "val_mask_area_calib_loss": val_metrics["mask_area_calib_loss"],
            "train_scale_supervision_loss": train_metrics["scale_supervision_loss"],
            "val_scale_supervision_loss": val_metrics["scale_supervision_loss"],
            "train_scale_entropy": train_metrics["scale_entropy"],
            "val_scale_entropy": val_metrics["scale_entropy"],
            "train_base_scale_w_0": train_metrics["base_scale_w_0"],
            "val_base_scale_w_0": val_metrics["base_scale_w_0"],
            "train_base_scale_w_1": train_metrics["base_scale_w_1"],
            "val_base_scale_w_1": val_metrics["base_scale_w_1"],
            "train_base_scale_w_2": train_metrics["base_scale_w_2"],
            "val_base_scale_w_2": val_metrics["base_scale_w_2"],
            "train_base_scale_w_3": train_metrics["base_scale_w_3"],
            "val_base_scale_w_3": val_metrics["base_scale_w_3"],
            "train_size_from_mask_loss": train_metrics["size_from_mask_loss"],
            "val_size_from_mask_loss": val_metrics["size_from_mask_loss"],
            "train_mask_from_size_loss": train_metrics["mask_from_size_loss"],
            "val_mask_from_size_loss": val_metrics["mask_from_size_loss"],
            "train_mask_iou": train_metrics["mask_iou"],
            "val_mask_iou": val_metrics["mask_iou"],
            "train_latent_similarity": train_metrics["latent_similarity"],
            "val_latent_similarity": val_metrics["latent_similarity"],
            "train_latent_completeness": train_metrics["latent_completeness"],
            "val_latent_completeness": val_metrics["latent_completeness"],
            "train_area_ratio": train_metrics["area_ratio"],
            "val_area_ratio": val_metrics["area_ratio"],
            "train_pred_area_ratio": train_metrics["pred_area_ratio"],
            "val_pred_area_ratio": val_metrics["pred_area_ratio"],
            "train_mask_area_ratio": train_metrics["mask_area_ratio"],
            "val_mask_area_ratio": val_metrics["mask_area_ratio"],
            "train_pred_area_mean": train_metrics["pred_area_mean"],
            "val_pred_area_mean": val_metrics["pred_area_mean"],
            "train_mask_area_mean": train_metrics["mask_area_mean"],
            "val_mask_area_mean": val_metrics["mask_area_mean"],
            "train_gt_area_mean": train_metrics["gt_area_mean"],
            "val_gt_area_mean": val_metrics["gt_area_mean"],
            "train_small_iou": train_metrics["small_iou"],
            "val_small_iou": val_metrics["small_iou"],
            "train_mid_iou": train_metrics["mid_iou"],
            "val_mid_iou": val_metrics["mid_iou"],
            "train_large_iou": train_metrics["large_iou"],
            "val_large_iou": val_metrics["large_iou"],
        }
        for scale_idx in range(4):
            row[f"train_scale_w_{scale_idx}"] = train_metrics[f"scale_w_{scale_idx}"]
            row[f"val_scale_w_{scale_idx}"] = val_metrics[f"scale_w_{scale_idx}"]

        epoch_rows.append(row)
        write_epoch_metrics(epoch_rows, run_dir / "epoch_metrics.csv")

        analysis_rows.append(build_epoch_analysis_row(epoch, "train", train_stats))
        analysis_rows.append(build_epoch_analysis_row(epoch, "val", val_stats))
        write_analysis_summary(analysis_rows, analysis_dir / "epoch_analysis_summary.csv")

        save_size_bin_summary(val_stats, analysis_dir / f"epoch_{epoch:03d}_val_size_bin_summary.csv")
        save_core_visualizations(val_stats, analysis_dir, epoch, "val")
        save_checkpoint(run_dir, model, optimizer, epoch, args.save_latest)

        if best_row is None or float(row["val_loss"]) < float(best_row["val_loss"]):
            best_row = dict(row)

        couple_train = train_metrics["size_from_mask_loss"] + train_metrics["mask_from_size_loss"]
        couple_val = val_metrics["size_from_mask_loss"] + val_metrics["mask_from_size_loss"]
        print(
            f"epoch {epoch:03d} | "
            f"scale_sup={current_scale_supervision_weight:.3f} | "
            f"scale_teacher={current_scale_teacher_alpha:.3f} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_iou={train_metrics['mask_iou']:.4f} val_iou={val_metrics['mask_iou']:.4f} | "
            f"train_sim={train_metrics['latent_similarity']:.4f} val_sim={val_metrics['latent_similarity']:.4f} | "
            f"train_comp={train_metrics['latent_completeness']:.4f} val_comp={val_metrics['latent_completeness']:.4f} | "
            f"train_area(mask/pred/gt)={train_metrics['mask_area_mean']:.1f}/{train_metrics['pred_area_mean']:.1f}/{train_metrics['gt_area_mean']:.1f} "
            f"val_area(mask/pred/gt)={val_metrics['mask_area_mean']:.1f}/{val_metrics['pred_area_mean']:.1f}/{val_metrics['gt_area_mean']:.1f}"
        )

    final_row = epoch_rows[-1]
    write_training_summary(best_row or final_row, final_row, run_dir / "training_summary.csv")


if __name__ == "__main__":
    main()

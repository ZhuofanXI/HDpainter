from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize current-model real-HD inference results on selected tiles."
    )
    parser.add_argument("--sample", choices=["COAD", "PRAD", "NSCLC"], help="Sample name used to infer current batch output paths.")
    parser.add_argument("--label-col", choices=["stardist_id", "cellpose_id"], help="Nucleus label column used by infer_hd.py.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/root/autodl-tmp/HDpainter1/model/runs"),
        help="Base model runs directory used when --sample and --label-col are supplied.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/HDpainter1/inference/infer_visiualization"),
        help="Base visualization directory used when --output-dir is omitted.",
    )
    parser.add_argument("--input-h5", type=Path, help="Inference-compatible H5 from build_real_hd_nmf_infer_h5.py")
    parser.add_argument("--pred-h5", type=Path, help="Prediction H5 from infer_hd.py")
    parser.add_argument("--output-dir", type=Path, help="Directory to save rendered PNGs")
    parser.add_argument("--num-tiles", type=int, default=3, help="Number of tiles to render")
    parser.add_argument("--min-instances", type=int, default=300, help="Minimum instance count for candidate tiles")
    parser.add_argument("--max-instances", type=int, default=600, help="Maximum instance count for candidate tiles")
    parser.add_argument("--alpha-cell", type=float, default=0.30, help="Fill alpha for predicted cell overlays")
    parser.add_argument(
        "--threshold-scan-values",
        type=str,
        default="",
        help="Optional comma-separated thresholds to visualize from pred-h5 threshold_scan results. "
        "If omitted and pred-h5 contains threshold_scan, all stored thresholds are used.",
    )
    return parser.parse_args()


def enhance_grayscale(counts_2d: np.ndarray) -> np.ndarray:
    image = counts_2d.astype(np.float32, copy=False)
    lo = float(np.percentile(image, 5.0))
    hi = float(np.percentile(image, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image


def choose_tile_indices(instance_counts: np.ndarray, num_tiles: int, min_instances: int, max_instances: int) -> list[int]:
    candidate = np.flatnonzero((instance_counts >= min_instances) & (instance_counts <= max_instances))
    if candidate.size == 0:
        positive = np.flatnonzero(instance_counts > 0)
        if positive.size == 0:
            raise ValueError(
                f"No tiles have assigned instances. Observed range: "
                f"[{instance_counts.min()}, {instance_counts.max()}]."
            )
        ranked = positive[np.argsort(instance_counts[positive])[::-1]]
        return ranked[: min(num_tiles, ranked.size)].tolist()
    if candidate.size <= num_tiles:
        return candidate.tolist()
    take = np.linspace(0, candidate.size - 1, num=num_tiles, dtype=np.int32)
    return candidate[take].tolist()



def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    if (args.input_h5 is None or args.pred_h5 is None) and (not args.sample or not args.label_col):
        raise ValueError("Provide either --input-h5/--pred-h5 or both --sample and --label-col.")

    if args.sample and args.label_col:
        run_name = f"synthHD_{args.sample}_nmf48_v1"
        infer_dir = args.runs_dir / run_name / "reference_hd_inference"
        prefix = f"{args.sample}_reference_hd_{run_name}_{args.label_col}"
        if args.input_h5 is None:
            args.input_h5 = infer_dir / f"{prefix}_nmf48_infer.h5"
        if args.pred_h5 is None:
            args.pred_h5 = infer_dir / f"{prefix}_pred.h5"
        if args.output_dir is None:
            args.output_dir = args.output_root / args.sample / args.label_col

    if args.output_dir is None:
        raise ValueError("Provide --output-dir, or use --sample and --label-col so it can be inferred.")

    for label, value in [("input_h5", args.input_h5), ("pred_h5", args.pred_h5)]:
        if value is None or not value.exists():
            raise FileNotFoundError(f"Missing {label}: {value}")
    return args


def parse_threshold_scan_values(raw: str) -> list[float]:
    if not raw.strip():
        return []
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Threshold-scan value must be in (0, 1), got {value}.")
        values.append(value)
    return sorted(set(values))


def format_threshold_tag(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):03d}"


def find_largest_instance(
    pred_mask_areas: np.ndarray,
    instance_offsets: np.ndarray,
) -> tuple[int, int, int, float]:
    if pred_mask_areas.size == 0:
        raise ValueError("pred_mask_area_pool is empty; cannot locate the largest predicted instance.")
    global_instance_idx = int(np.argmax(pred_mask_areas))
    largest_area = float(pred_mask_areas[global_instance_idx])
    tile_idx = int(np.searchsorted(instance_offsets, global_instance_idx, side="right") - 1)
    local_instance_idx = int(global_instance_idx - instance_offsets[tile_idx])
    return tile_idx, global_instance_idx, local_instance_idx, largest_area


def build_palette(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    hues = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float32)
    sat = np.full_like(hues, 0.80)
    val = np.full_like(hues, 1.00)
    hsv = np.stack([hues, sat, val], axis=1)
    return hsv_to_rgb(hsv).astype(np.float32, copy=False)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    out = np.zeros((hsv.shape[0], 3), dtype=np.float32)
    choices = (
        np.stack([v, t, p], axis=1),
        np.stack([q, v, p], axis=1),
        np.stack([p, v, t], axis=1),
        np.stack([p, q, v], axis=1),
        np.stack([t, p, v], axis=1),
        np.stack([v, p, q], axis=1),
    )
    for idx in range(6):
        mask = i_mod == idx
        if np.any(mask):
            out[mask] = choices[idx][mask]
    return out


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    mask_bool = mask.astype(bool, copy=False)
    up = np.pad(mask_bool[1:, :], ((0, 1), (0, 0)), constant_values=False)
    down = np.pad(mask_bool[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    left = np.pad(mask_bool[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    right = np.pad(mask_bool[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    interior = mask_bool & up & down & left & right
    return mask_bool & (~interior)


def alpha_blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if not np.any(mask):
        return
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color


def draw_outline(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
    boundary = mask_boundary(mask)
    if np.any(boundary):
        rgb[boundary] = color


def render_base_with_nuclei(base: np.ndarray, nucleus_centers: np.ndarray) -> np.ndarray:
    rgb = np.repeat(base[..., None], 3, axis=2)
    if nucleus_centers.size > 0:
        yy = np.clip(np.rint(nucleus_centers[:, 0]).astype(np.int32), 0, base.shape[0] - 1)
        xx = np.clip(np.rint(nucleus_centers[:, 1]).astype(np.int32), 0, base.shape[1] - 1)
        rgb[yy, xx] = np.array([1.0, 0.15, 0.15], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def render_overlay(base: np.ndarray, masks: np.ndarray, colors: np.ndarray, alpha: float) -> np.ndarray:
    rgb = np.repeat(base[..., None], 3, axis=2)
    for idx in range(masks.shape[0]):
        mask = masks[idx] > 0
        if not np.any(mask):
            continue
        color = colors[idx]
        alpha_blend(rgb, mask, color, alpha=alpha)
        draw_outline(rgb, mask, color=np.clip(color * 0.85, 0.0, 1.0))
    return np.clip(rgb, 0.0, 1.0)


def reconstruct_pred_masks(
    mask_crops: np.ndarray,
    centers: np.ndarray,
    tile_size: int,
    canvas_size: int,
) -> np.ndarray:
    out = np.zeros((mask_crops.shape[0], tile_size, tile_size), dtype=np.float32)
    half = canvas_size // 2
    for idx, crop in enumerate(mask_crops):
        cy = int(round(float(centers[idx, 0])))
        cx = int(round(float(centers[idx, 1])))
        y0 = cy - half
        x0 = cx - half
        y1 = y0 + canvas_size
        x1 = x0 + canvas_size

        src_y0 = max(0, -y0)
        src_x0 = max(0, -x0)
        src_y1 = canvas_size - max(0, y1 - tile_size)
        src_x1 = canvas_size - max(0, x1 - tile_size)

        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(tile_size, y1)
        dst_x1 = min(tile_size, x1)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            continue
        out[idx, dst_y0:dst_y1, dst_x0:dst_x1] = crop[src_y0:src_y1, src_x0:src_x1]
    return out


def build_global_masks_for_tile(
    assigned_map: np.ndarray,
    global_instance_ids: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    if global_instance_ids.size == 0:
        return np.zeros((0, patch_size, patch_size), dtype=np.uint8)
    masks = np.zeros((global_instance_ids.size, patch_size, patch_size), dtype=np.uint8)
    for idx, global_id in enumerate(global_instance_ids.tolist()):
        masks[idx] = (assigned_map == (global_id + 1)).astype(np.uint8)
    return masks


def compute_assigned_instance_sizes(
    tile_assigned_maps: h5py.Dataset,
    n_instances: int,
) -> np.ndarray:
    sizes = np.zeros(n_instances, dtype=np.int64)
    for tile_idx in range(tile_assigned_maps.shape[0]):
        assigned = np.asarray(tile_assigned_maps[tile_idx], dtype=np.int64)
        valid = assigned[assigned > 0] - 1
        if valid.size == 0:
            continue
        binc = np.bincount(valid, minlength=n_instances)
        sizes[: binc.shape[0]] += binc[:n_instances]
    return sizes


def save_size_distribution_figure(
    save_path: Path,
    assigned_sizes: np.ndarray,
) -> None:
    positive_sizes = assigned_sizes[assigned_sizes > 0]
    if positive_sizes.size == 0:
        raise ValueError("No assigned cell bins found; cannot render size distribution.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=170)

    n_bins = int(np.clip(np.sqrt(positive_sizes.size), 20, 80))
    axes[0].hist(positive_sizes, bins=n_bins, color="#3B82F6", alpha=0.85, edgecolor="white")
    axes[0].set_title("Predicted Cell Size Distribution")
    axes[0].set_xlabel("Assigned bin count per cell")
    axes[0].set_ylabel("Cell count")

    sorted_sizes = np.sort(positive_sizes)
    cdf = np.arange(1, sorted_sizes.size + 1, dtype=np.float64) / sorted_sizes.size
    axes[1].plot(sorted_sizes, cdf, color="#EF4444", linewidth=2.0)
    axes[1].set_title("Predicted Cell Size ECDF")
    axes[1].set_xlabel("Assigned bin count per cell")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_ylim(0.0, 1.0)

    summary = (
        f"n={positive_sizes.size}\n"
        f"mean={positive_sizes.mean():.1f}\n"
        f"median={np.median(positive_sizes):.1f}\n"
        f"p90={np.percentile(positive_sizes, 90):.1f}\n"
        f"max={positive_sizes.max():.1f}"
    )
    axes[1].text(
        0.98,
        0.04,
        summary,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_size_distribution_csv(
    save_path: Path,
    assigned_sizes: np.ndarray,
) -> None:
    positive_sizes = assigned_sizes[assigned_sizes > 0]
    total_instances = int(assigned_sizes.shape[0])
    active_ge1 = int(np.sum(assigned_sizes >= 1))
    active_ge3 = int(np.sum(assigned_sizes >= 3))
    active_ge5 = int(np.sum(assigned_sizes >= 5))
    if positive_sizes.size == 0:
        rows = [
            {"metric": "total_candidate_instances", "value": total_instances},
            {"metric": "active_cells_ge1_bins", "value": active_ge1},
            {"metric": "active_cells_ge3_bins", "value": active_ge3},
            {"metric": "active_cells_ge5_bins", "value": active_ge5},
            {"metric": "active_frac_ge1_bins", "value": 0.0 if total_instances == 0 else active_ge1 / total_instances},
            {"metric": "active_frac_ge3_bins", "value": 0.0 if total_instances == 0 else active_ge3 / total_instances},
            {"metric": "active_frac_ge5_bins", "value": 0.0 if total_instances == 0 else active_ge5 / total_instances},
        ]
    else:
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        rows = [
            {"metric": "total_candidate_instances", "value": total_instances},
            {"metric": "active_cells_ge1_bins", "value": active_ge1},
            {"metric": "active_cells_ge3_bins", "value": active_ge3},
            {"metric": "active_cells_ge5_bins", "value": active_ge5},
            {"metric": "active_frac_ge1_bins", "value": 0.0 if total_instances == 0 else active_ge1 / total_instances},
            {"metric": "active_frac_ge3_bins", "value": 0.0 if total_instances == 0 else active_ge3 / total_instances},
            {"metric": "active_frac_ge5_bins", "value": 0.0 if total_instances == 0 else active_ge5 / total_instances},
            {"metric": "n_cells", "value": int(positive_sizes.size)},
            {"metric": "mean_bins", "value": float(np.mean(positive_sizes))},
            {"metric": "median_bins", "value": float(np.median(positive_sizes))},
            {"metric": "min_bins", "value": int(np.min(positive_sizes))},
            {"metric": "max_bins", "value": int(np.max(positive_sizes))},
        ]
        rows.extend(
            {"metric": f"p{p}_bins", "value": float(np.percentile(positive_sizes, p))}
            for p in percentiles
        )

    with save_path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def summarize_assigned_sizes(assigned_sizes: np.ndarray) -> dict[str, float]:
    positive_sizes = assigned_sizes[assigned_sizes > 0]
    total_instances = float(assigned_sizes.shape[0])
    active_ge1 = float(np.sum(assigned_sizes >= 1))
    active_ge3 = float(np.sum(assigned_sizes >= 3))
    active_ge5 = float(np.sum(assigned_sizes >= 5))
    if positive_sizes.size == 0:
        return {
            "total_candidate_instances": total_instances,
            "active_cells_ge1_bins": active_ge1,
            "active_cells_ge3_bins": active_ge3,
            "active_cells_ge5_bins": active_ge5,
            "active_frac_ge1_bins": 0.0 if total_instances == 0 else active_ge1 / total_instances,
            "active_frac_ge3_bins": 0.0 if total_instances == 0 else active_ge3 / total_instances,
            "active_frac_ge5_bins": 0.0 if total_instances == 0 else active_ge5 / total_instances,
            "n_cells": 0.0,
            "mean_bins": 0.0,
            "median_bins": 0.0,
            "min_bins": 0.0,
            "max_bins": 0.0,
            "p5_bins": 0.0,
            "p10_bins": 0.0,
            "p25_bins": 0.0,
            "p50_bins": 0.0,
            "p75_bins": 0.0,
            "p90_bins": 0.0,
            "p95_bins": 0.0,
            "p99_bins": 0.0,
        }
    return {
        "total_candidate_instances": total_instances,
        "active_cells_ge1_bins": active_ge1,
        "active_cells_ge3_bins": active_ge3,
        "active_cells_ge5_bins": active_ge5,
        "active_frac_ge1_bins": 0.0 if total_instances == 0 else active_ge1 / total_instances,
        "active_frac_ge3_bins": 0.0 if total_instances == 0 else active_ge3 / total_instances,
        "active_frac_ge5_bins": 0.0 if total_instances == 0 else active_ge5 / total_instances,
        "n_cells": float(positive_sizes.size),
        "mean_bins": float(np.mean(positive_sizes)),
        "median_bins": float(np.median(positive_sizes)),
        "min_bins": float(np.min(positive_sizes)),
        "max_bins": float(np.max(positive_sizes)),
        "p5_bins": float(np.percentile(positive_sizes, 5)),
        "p10_bins": float(np.percentile(positive_sizes, 10)),
        "p25_bins": float(np.percentile(positive_sizes, 25)),
        "p50_bins": float(np.percentile(positive_sizes, 50)),
        "p75_bins": float(np.percentile(positive_sizes, 75)),
        "p90_bins": float(np.percentile(positive_sizes, 90)),
        "p95_bins": float(np.percentile(positive_sizes, 95)),
        "p99_bins": float(np.percentile(positive_sizes, 99)),
    }


def save_threshold_scan_comparison(
    save_csv_path: Path,
    save_png_path: Path,
    threshold_to_sizes: dict[float, np.ndarray],
) -> None:
    rows: list[dict[str, float | str]] = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=170)

    palette = plt.cm.viridis(np.linspace(0.12, 0.88, num=max(len(threshold_to_sizes), 1)))
    for color, threshold in zip(palette, sorted(threshold_to_sizes.keys())):
        sizes = threshold_to_sizes[threshold]
        stats = summarize_assigned_sizes(sizes)
        rows.append({"threshold": f"{threshold:.2f}", **stats})

        positive_sizes = sizes[sizes > 0]
        if positive_sizes.size == 0:
            continue
        sorted_sizes = np.sort(positive_sizes)
        cdf = np.arange(1, sorted_sizes.size + 1, dtype=np.float64) / sorted_sizes.size
        axes[0].plot(sorted_sizes, cdf, label=f"{threshold:.2f}", color=color, linewidth=2.0)
        axes[1].plot(
            sorted_sizes,
            np.log10(np.arange(sorted_sizes.size, 0, -1, dtype=np.float64)),
            label=f"{threshold:.2f}",
            color=color,
            linewidth=1.8,
        )

    axes[0].set_title("Cell-size ECDF by threshold")
    axes[0].set_xlabel("Assigned bin count per cell")
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(title="threshold")

    axes[1].set_title("Cell-size tail by threshold")
    axes[1].set_xlabel("Assigned bin count per cell")
    axes[1].set_ylabel("log10(#cells >= size)")
    axes[1].legend(title="threshold")

    active_lines = []
    for threshold in sorted(threshold_to_sizes.keys()):
        stats = summarize_assigned_sizes(threshold_to_sizes[threshold])
        active_lines.append(
            f"{threshold:.2f}: >=1 {int(stats['active_cells_ge1_bins'])}, "
            f">=3 {int(stats['active_cells_ge3_bins'])}, >=5 {int(stats['active_cells_ge5_bins'])}"
        )
    axes[1].text(
        0.98,
        0.04,
        "\n".join(active_lines),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(save_png_path, bbox_inches="tight")
    plt.close(fig)

    with save_csv_path.open("w", newline="", encoding="utf-8") as fw:
        fieldnames = [
            "threshold",
            "total_candidate_instances",
            "active_cells_ge1_bins",
            "active_cells_ge3_bins",
            "active_cells_ge5_bins",
            "active_frac_ge1_bins",
            "active_frac_ge3_bins",
            "active_frac_ge5_bins",
            "n_cells",
            "mean_bins",
            "median_bins",
            "min_bins",
            "max_bins",
            "p5_bins",
            "p10_bins",
            "p25_bins",
            "p50_bins",
            "p75_bins",
            "p90_bins",
            "p95_bins",
            "p99_bins",
        ]
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_tile_figure(
    save_path: Path,
    nuclei_rgb: np.ndarray,
    prob_rgb: np.ndarray,
    assigned_rgb: np.ndarray,
    tile_index: int,
    instance_count: int,
    core_nucleus_count: int,
    largest_rgb: np.ndarray | None = None,
    largest_title: str | None = None,
) -> None:
    ncols = 4 if largest_rgb is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), dpi=170)
    axes[0].imshow(nuclei_rgb)
    axes[0].set_title(f"Tile {tile_index} | counts + nuclei\ninstances={instance_count} core_nuclei={core_nucleus_count}")
    axes[0].axis("off")

    axes[1].imshow(prob_rgb)
    axes[1].set_title("counts + local predicted masks")
    axes[1].axis("off")

    axes[2].imshow(assigned_rgb)
    axes[2].set_title("counts + final assigned instances")
    axes[2].axis("off")

    if largest_rgb is not None:
        axes[3].imshow(largest_rgb)
        axes[3].set_title(largest_title or "largest predicted cell")
        axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = resolve_paths(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer_visiualize] input_h5={args.input_h5}")
    print(f"[infer_visiualize] pred_h5={args.pred_h5}")
    print(f"[infer_visiualize] output_dir={args.output_dir}")
    requested_thresholds = parse_threshold_scan_values(args.threshold_scan_values)

    with h5py.File(args.input_h5, "r") as fin, h5py.File(args.pred_h5, "r") as fpred:
        n_tiles = int(fin.attrs["n_samples"])
        pred_n_tiles = int(fpred.attrs["n_samples"])
        if pred_n_tiles != n_tiles:
            raise ValueError(f"Tile count mismatch: input_h5 has {n_tiles}, pred_h5 has {pred_n_tiles}")

        patch_size = int(fin.attrs["patch_size"])
        canvas_size = int(fpred.attrs["canvas_size"])
        instance_offsets = fpred["instance_offsets"][:]
        pred_mask_areas = fpred["pred_mask_area_pool"][:].reshape(-1)
        largest_tile_idx, largest_global_instance_idx, largest_local_instance_idx, largest_area = find_largest_instance(
            pred_mask_areas=pred_mask_areas,
            instance_offsets=instance_offsets,
        )
        nucleus_centers_all = fpred["nucleus_centers_yx_pool"]
        mask_prob_crops_all = fpred["pred_mask_prob_crop_pool"]
        core_nucleus_count_ds = fin["core_nucleus_count"] if "core_nucleus_count" in fin else None
        top_level_assign_maps = fpred["tile_assigned_instance_map"]

        threshold_map_groups: list[tuple[float, h5py.Dataset, h5py.Dataset]] = []
        if "threshold_scan" in fpred:
            scan_root = fpred["threshold_scan"]
            if requested_thresholds:
                thresholds_to_use = requested_thresholds
            else:
                thresholds_to_use = sorted(float(scan_root[key].attrs["threshold"]) for key in scan_root.keys())
            for threshold in thresholds_to_use:
                tag = format_threshold_tag(threshold)
                if tag not in scan_root:
                    raise KeyError(f"Requested threshold {threshold:.2f} not found in pred-h5 threshold_scan/{tag}.")
                grp = scan_root[tag]
                threshold_map_groups.append(
                    (
                        float(grp.attrs["threshold"]),
                        grp["tile_assigned_instance_map"],
                        grp["tile_assigned_score_map"],
                    )
                )

        if not threshold_map_groups:
            threshold_map_groups = [
                (
                    float(fpred.attrs.get("assign_score_threshold", 0.50)),
                    top_level_assign_maps,
                    fpred["tile_assigned_score_map"],
                )
            ]

        reference_threshold, reference_assign_maps, _ = threshold_map_groups[0]
        reference_instance_counts = np.array(
            [int(np.unique(reference_assign_maps[tile_idx][reference_assign_maps[tile_idx] > 0]).size) for tile_idx in range(n_tiles)],
            dtype=np.int32,
        )
        chosen_tiles = choose_tile_indices(reference_instance_counts, args.num_tiles, args.min_instances, args.max_instances)

        threshold_to_sizes: dict[float, np.ndarray] = {}

        for threshold, assign_maps_ds, _score_maps_ds in threshold_map_groups:
            threshold_dir = args.output_dir if len(threshold_map_groups) == 1 else args.output_dir / f"threshold_{threshold:.2f}"
            threshold_dir.mkdir(parents=True, exist_ok=True)
            assigned_sizes = compute_assigned_instance_sizes(
                tile_assigned_maps=assign_maps_ds,
                n_instances=pred_mask_areas.shape[0],
            )
            threshold_to_sizes[threshold] = assigned_sizes

            summary_rows = []
            for rank, tile_idx in enumerate(chosen_tiles, start=1):
                start = int(instance_offsets[tile_idx])
                end = int(instance_offsets[tile_idx + 1])
                count = end - start

                counts_map = fin["x_counts"][tile_idx, 0] if "x_counts" in fin else fin["x_low"][tile_idx].sum(axis=0)
                base = enhance_grayscale(counts_map)
                nucleus_centers = nucleus_centers_all[start:end]
                mask_prob_crops = mask_prob_crops_all[start:end]
                assigned_map = assign_maps_ds[tile_idx]
                global_instance_ids = np.arange(start, end, dtype=np.int32)

                palette = build_palette(max(count, 1))
                local_binary = (mask_prob_crops >= threshold).astype(np.uint8)
                local_binary_full = reconstruct_pred_masks(
                    mask_crops=local_binary,
                    centers=nucleus_centers,
                    tile_size=patch_size,
                    canvas_size=canvas_size,
                )
                assigned_full = build_global_masks_for_tile(
                    assigned_map=assigned_map,
                    global_instance_ids=global_instance_ids,
                    patch_size=patch_size,
                )

                nuclei_rgb = render_base_with_nuclei(base, nucleus_centers)
                prob_rgb = render_overlay(base, local_binary_full, palette, alpha=args.alpha_cell)
                assigned_rgb = render_overlay(base, assigned_full, palette, alpha=args.alpha_cell)

                core_nucleus_count = int(core_nucleus_count_ds[tile_idx]) if core_nucleus_count_ds is not None else count
                save_path = threshold_dir / f"tile_{tile_idx:04d}_rank{rank}.png"
                save_tile_figure(
                    save_path=save_path,
                    nuclei_rgb=nuclei_rgb,
                    prob_rgb=prob_rgb,
                    assigned_rgb=assigned_rgb,
                    tile_index=tile_idx,
                    instance_count=count,
                    core_nucleus_count=core_nucleus_count,
                )

                summary_rows.append(
                    {
                        "tile_idx": int(tile_idx),
                        "rank": int(rank),
                        "mode": "ranked",
                        "threshold": f"{threshold:.2f}",
                        "instance_count": int(count),
                        "core_nucleus_count": int(core_nucleus_count),
                        "largest_global_instance_idx": "",
                        "largest_instance_area": "",
                        "save_path": str(save_path),
                    }
                )

            start = int(instance_offsets[largest_tile_idx])
            end = int(instance_offsets[largest_tile_idx + 1])
            count = end - start

            counts_map = fin["x_counts"][largest_tile_idx, 0] if "x_counts" in fin else fin["x_low"][largest_tile_idx].sum(axis=0)
            base = enhance_grayscale(counts_map)
            nucleus_centers = nucleus_centers_all[start:end]
            mask_prob_crops = mask_prob_crops_all[start:end]
            assigned_map = assign_maps_ds[largest_tile_idx]
            global_instance_ids = np.arange(start, end, dtype=np.int32)

            palette = build_palette(max(count, 1))
            local_binary = (mask_prob_crops >= threshold).astype(np.uint8)
            local_binary_full = reconstruct_pred_masks(
                mask_crops=local_binary,
                centers=nucleus_centers,
                tile_size=patch_size,
                canvas_size=canvas_size,
            )
            assigned_full = build_global_masks_for_tile(
                assigned_map=assigned_map,
                global_instance_ids=global_instance_ids,
                patch_size=patch_size,
            )

            largest_focus_mask = assigned_full[largest_local_instance_idx : largest_local_instance_idx + 1]
            largest_focus_rgb = render_overlay(
                base,
                largest_focus_mask,
                np.array([[1.0, 0.92, 0.10]], dtype=np.float32),
                alpha=min(0.55, args.alpha_cell + 0.20),
            )

            nuclei_rgb = render_base_with_nuclei(base, nucleus_centers)
            prob_rgb = render_overlay(base, local_binary_full, palette, alpha=args.alpha_cell)
            assigned_rgb = render_overlay(base, assigned_full, palette, alpha=args.alpha_cell)

            core_nucleus_count = int(core_nucleus_count_ds[largest_tile_idx]) if core_nucleus_count_ds is not None else count
            save_path = threshold_dir / f"tile_{largest_tile_idx:04d}_largest_cell.png"
            save_tile_figure(
                save_path=save_path,
                nuclei_rgb=nuclei_rgb,
                prob_rgb=prob_rgb,
                assigned_rgb=assigned_rgb,
                tile_index=int(largest_tile_idx),
                instance_count=count,
                core_nucleus_count=core_nucleus_count,
                largest_rgb=largest_focus_rgb,
                largest_title=f"largest predicted cell\ninstance={largest_global_instance_idx} area={largest_area:.1f} thr={threshold:.2f}",
            )

            summary_rows.append(
                {
                    "tile_idx": int(largest_tile_idx),
                    "rank": "",
                    "mode": "largest_cell",
                    "threshold": f"{threshold:.2f}",
                    "instance_count": int(count),
                    "core_nucleus_count": int(core_nucleus_count),
                    "largest_global_instance_idx": int(largest_global_instance_idx),
                    "largest_instance_area": float(largest_area),
                    "save_path": str(save_path),
                }
            )

            summary_path = threshold_dir / "summary.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as fw:
                writer = csv.DictWriter(
                    fw,
                    fieldnames=[
                        "tile_idx",
                        "rank",
                        "mode",
                        "threshold",
                        "instance_count",
                        "core_nucleus_count",
                        "largest_global_instance_idx",
                        "largest_instance_area",
                        "save_path",
                    ],
                )
                writer.writeheader()
                writer.writerows(summary_rows)

            save_size_distribution_figure(
                save_path=threshold_dir / "pred_cell_size_distribution.png",
                assigned_sizes=assigned_sizes,
            )
            save_size_distribution_csv(
                save_path=threshold_dir / "pred_cell_size_distribution_summary.csv",
                assigned_sizes=assigned_sizes,
            )

        if len(threshold_map_groups) > 1:
            save_threshold_scan_comparison(
                save_csv_path=args.output_dir / "threshold_scan_size_summary.csv",
                save_png_path=args.output_dir / "threshold_scan_size_comparison.png",
                threshold_to_sizes=threshold_to_sizes,
            )

    print(f"Saved current-model visualization outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

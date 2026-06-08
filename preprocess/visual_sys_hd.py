from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCALAR_METRICS = [
    "instance_mask_areas_pool",
    "mask_refine_raw_area_pool",
    "mask_refine_refined_area_pool",
    "mask_refine_area_change_ratio_pool",
    "mask_refine_iou_raw_refined_pool",
    "mask_refine_raw_target_similarity_pool",
    "mask_refine_refined_target_similarity_pool",
    "mask_refine_raw_target_completeness_pool",
    "mask_refine_refined_target_completeness_pool",
    "mask_refine_latent_sum_delta_ratio_pool",
    "mask_refine_latent_global_abs_delta_ratio_pool",
    "mask_refine_latent_global_loss_ratio_pool",
    "mask_refine_latent_max_module_loss_ratio_pool",
    "mask_refine_raw_n_components_pool",
    "mask_refine_refined_n_components_pool",
    "mask_refine_raw_n_holes_pool",
    "mask_refine_refined_n_holes_pool",
    "mask_refine_raw_hole_frac_pool",
    "mask_refine_refined_hole_frac_pool",
    "mask_refine_matched_swap_ratio_pool",
    "mask_refine_unmatched_add_count_pool",
    "mask_refine_unmatched_remove_count_pool",
    "instance_size_bin_ids_pool",
    "instance_refined_kind_pool",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize sampled synthetic HD training tiles and summarize instance-chunk "
            "mask-refinement statistics from a *.maskrefined.h5 file."
        )
    )
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-tiles", type=int, default=12)
    parser.add_argument("--select-mode", choices=["random", "top", "spread"], default="spread")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-stat-instances",
        type=int,
        default=300000,
        help="Maximum instances per split loaded for scalar metric summaries. Use 0 for all.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--write-npy", action="store_true", help="Also save sampled tile arrays as .npy files.")
    return parser.parse_args()


def dataset_key(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def require_keys(fr: h5py.File, keys: list[str]) -> None:
    missing = [key for key in keys if key not in fr]
    if missing:
        raise KeyError(f"Missing required datasets: {missing}")


def read_attrs(fr: h5py.File) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for key, value in fr.attrs.items():
        if isinstance(value, np.generic):
            attrs[key] = value.item()
        elif isinstance(value, np.ndarray):
            attrs[key] = value.tolist()
        elif isinstance(value, bytes):
            attrs[key] = value.decode("utf-8", errors="replace")
        else:
            attrs[key] = value
    return attrs


def normalize01(arr: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr[finite]))
        hi = float(np.max(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def tile_background(tile_input: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile_input, dtype=np.float32)
    if tile.ndim == 3:
        summed = np.log1p(np.maximum(tile, 0.0).sum(axis=0))
    elif tile.ndim == 2:
        summed = np.log1p(np.maximum(tile, 0.0))
    else:
        raise ValueError(f"Unsupported tile_input shape: {tile.shape}")
    return normalize01(summed)


def tile_rgb(tile_input: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile_input, dtype=np.float32)
    if tile.ndim != 3:
        bg = normalize01(tile)
        return np.dstack([bg, bg, bg])
    channels = []
    for idx in range(min(3, tile.shape[0])):
        channels.append(normalize01(np.log1p(np.maximum(tile[idx], 0.0))))
    while len(channels) < 3:
        channels.append(np.zeros_like(channels[0]))
    return np.dstack(channels[:3])


def crop_to_full(mask_local: np.ndarray, center_y: float, center_x: float, tile_size: int) -> tuple[slice, slice, slice, slice]:
    canvas_h, canvas_w = int(mask_local.shape[-2]), int(mask_local.shape[-1])
    half_h = canvas_h // 2
    half_w = canvas_w // 2
    cy = int(round(float(center_y)))
    cx = int(round(float(center_x)))
    y0 = cy - half_h
    x0 = cx - half_w
    y1 = y0 + canvas_h
    x1 = x0 + canvas_w

    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    src_y1 = canvas_h - max(0, y1 - tile_size)
    src_x1 = canvas_w - max(0, x1 - tile_size)

    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(tile_size, y1)
    dst_x1 = min(tile_size, x1)

    return (
        slice(src_y0, src_y1),
        slice(src_x0, src_x1),
        slice(dst_y0, dst_y1),
        slice(dst_x0, dst_x1),
    )


def boundary(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool, copy=False)
    up = np.pad(mask_bool[1:, :], ((0, 1), (0, 0)), constant_values=False)
    down = np.pad(mask_bool[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    left = np.pad(mask_bool[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    right = np.pad(mask_bool[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    interior = mask_bool & up & down & left & right
    return mask_bool & (~interior)


def reconstruct_union(local_masks: np.ndarray, centers_yx: np.ndarray, tile_size: int) -> np.ndarray:
    masks = np.asarray(local_masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.shape[0] == 0:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)
    if masks.shape[-2:] == (tile_size, tile_size):
        return (masks.max(axis=0) > 0).astype(np.uint8)
    union = np.zeros((tile_size, tile_size), dtype=np.uint8)
    for idx in range(masks.shape[0]):
        src_y, src_x, dst_y, dst_x = crop_to_full(masks[idx], centers_yx[idx, 0], centers_yx[idx, 1], tile_size)
        if dst_y.stop <= dst_y.start or dst_x.stop <= dst_x.start:
            continue
        union[dst_y, dst_x] = np.maximum(union[dst_y, dst_x], (masks[idx][src_y, src_x] > 0).astype(np.uint8))
    return union


def local_iou(raw_masks: np.ndarray, refined_masks: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_masks)
    refined = np.asarray(refined_masks)
    if raw.ndim == 4:
        raw = raw[:, 0]
    if refined.ndim == 4:
        refined = refined[:, 0]
    raw = raw > 0
    refined = refined > 0
    inter = np.logical_and(raw, refined).sum(axis=(1, 2)).astype(np.float32)
    union = np.logical_or(raw, refined).sum(axis=(1, 2)).astype(np.float32)
    return inter / np.maximum(union, 1.0)


def tile_instance_slices(fr: h5py.File, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ptr = np.asarray(fr[dataset_key(prefix, "instance_tile_ptr_pool")], dtype=np.int64)
    tile_indices = np.asarray(fr[dataset_key(prefix, "tile_index_pool")], dtype=np.int64)
    n_tiles = int(fr[dataset_key(prefix, "tile_input_pool")].shape[0])
    order = np.argsort(ptr, kind="stable")
    sorted_ptr = ptr[order]
    counts = np.bincount(sorted_ptr, minlength=n_tiles).astype(np.int64)
    offsets = np.zeros(n_tiles + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return order, offsets, tile_indices


def choose_tiles(counts: np.ndarray, num_tiles: int, mode: str, seed: int) -> np.ndarray:
    eligible = np.flatnonzero(counts > 0)
    if eligible.size == 0:
        return np.asarray([], dtype=np.int64)
    num_tiles = min(int(num_tiles), int(eligible.size))
    if mode == "top":
        return eligible[np.argsort(counts[eligible])[::-1][:num_tiles]].astype(np.int64)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(eligible, size=num_tiles, replace=False).astype(np.int64))
    quantiles = np.linspace(0, eligible.size - 1, num_tiles)
    spatial_order = eligible[np.argsort(eligible)]
    return np.unique(spatial_order[np.round(quantiles).astype(int)]).astype(np.int64)


def overlay_union(ax: plt.Axes, bg: np.ndarray, mask: np.ndarray, color: str, title: str, alpha: float) -> None:
    ax.imshow(bg, cmap="gray", interpolation="nearest")
    cmap = matplotlib.colors.ListedColormap([color])
    ax.imshow(np.ma.masked_where(mask <= 0, mask), cmap=cmap, alpha=alpha, interpolation="nearest")
    if np.any(mask > 0):
        ax.contour(boundary(mask), levels=[0.5], colors=[color], linewidths=0.6)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def visualize_tiles(fr: h5py.File, args: argparse.Namespace, prefix: str, out_dir: Path) -> list[dict[str, object]]:
    required = [
        dataset_key(prefix, "tile_input_pool"),
        dataset_key(prefix, "tile_index_pool"),
        dataset_key(prefix, "instance_tile_ptr_pool"),
        dataset_key(prefix, "instance_centers_yx_pool"),
        dataset_key(prefix, "instance_mask_targets_pool"),
        dataset_key(prefix, "instance_mask_targets_refined_pool"),
    ]
    require_keys(fr, required)

    order, offsets, tile_indices = tile_instance_slices(fr, prefix)
    counts = np.diff(offsets)
    selected_tiles = choose_tiles(counts, args.num_tiles, args.select_mode, args.seed)
    if selected_tiles.size == 0:
        raise RuntimeError(f"No {prefix} tiles with instances were found.")

    tile_pool = fr[dataset_key(prefix, "tile_input_pool")]
    raw_masks = fr[dataset_key(prefix, "instance_mask_targets_pool")]
    refined_masks = fr[dataset_key(prefix, "instance_mask_targets_refined_pool")]
    centers = fr[dataset_key(prefix, "instance_centers_yx_pool")]

    rows: list[dict[str, object]] = []
    for rank, tile_pos in enumerate(selected_tiles.tolist(), start=1):
        inst_order = order[offsets[tile_pos] : offsets[tile_pos + 1]]
        tile = np.asarray(tile_pool[tile_pos], dtype=np.float32)
        tile_size = int(tile.shape[-1])
        bg = tile_background(tile)
        rgb = tile_rgb(tile)
        raw_local = np.asarray(raw_masks[inst_order])
        refined_local = np.asarray(refined_masks[inst_order])
        center_local = np.asarray(centers[inst_order], dtype=np.float32)

        raw_union = reconstruct_union(raw_local, center_local, tile_size)
        refined_union = reconstruct_union(refined_local, center_local, tile_size)
        overlap = np.logical_and(raw_union > 0, refined_union > 0)
        raw_only = np.logical_and(raw_union > 0, refined_union == 0)
        refined_only = np.logical_and(raw_union == 0, refined_union > 0)
        union_iou = float(overlap.sum() / max(np.logical_or(raw_union > 0, refined_union > 0).sum(), 1))
        instance_ious = local_iou(raw_local, refined_local)

        diff_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
        diff_rgb[..., 0] = raw_only.astype(np.float32)
        diff_rgb[..., 1] = overlap.astype(np.float32)
        diff_rgb[..., 2] = refined_only.astype(np.float32)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes[0, 0].imshow(bg, cmap="gray", interpolation="nearest")
        axes[0, 0].set_title("log1p summed NMF tile", fontsize=10)
        axes[0, 0].axis("off")
        axes[0, 1].imshow(rgb, interpolation="nearest")
        axes[0, 1].set_title("first 3 NMF channels", fontsize=10)
        axes[0, 1].axis("off")
        overlay_union(axes[0, 2], bg, raw_union, "lime", "raw mask union", args.alpha)
        overlay_union(axes[1, 0], bg, refined_union, "magenta", "refined mask union", args.alpha)
        axes[1, 1].imshow(bg, cmap="gray", interpolation="nearest")
        axes[1, 1].imshow(diff_rgb, alpha=0.75, interpolation="nearest")
        axes[1, 1].set_title("diff: red raw, green overlap, blue refined", fontsize=10)
        axes[1, 1].axis("off")
        axes[1, 2].imshow(bg, cmap="gray", interpolation="nearest")
        if np.any(raw_union > 0):
            axes[1, 2].contour(boundary(raw_union), levels=[0.5], colors=["lime"], linewidths=0.6)
        if np.any(refined_union > 0):
            axes[1, 2].contour(boundary(refined_union), levels=[0.5], colors=["magenta"], linewidths=0.6)
        axes[1, 2].set_title("boundaries", fontsize=10)
        axes[1, 2].axis("off")
        fig.suptitle(
            f"{prefix} tile_pool={tile_pos} source_tile={int(tile_indices[tile_pos])} "
            f"instances={inst_order.size} union_iou={union_iou:.3f} "
            f"mean_instance_iou={float(np.mean(instance_ious)):.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        png_path = out_dir / f"{prefix}_tile_{tile_pos:05d}_source_{int(tile_indices[tile_pos]):05d}.png"
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        if args.write_npy:
            np.save(out_dir / f"{prefix}_tile_{tile_pos:05d}_raw_union.npy", raw_union)
            np.save(out_dir / f"{prefix}_tile_{tile_pos:05d}_refined_union.npy", refined_union)

        rows.append(
            {
                "split": prefix,
                "rank": rank,
                "tile_pool_index": int(tile_pos),
                "source_tile_index": int(tile_indices[tile_pos]),
                "n_instances": int(inst_order.size),
                "raw_union_area": int(raw_union.sum()),
                "refined_union_area": int(refined_union.sum()),
                "union_iou": union_iou,
                "mean_instance_iou": float(np.mean(instance_ious)),
                "median_instance_iou": float(np.median(instance_ious)),
                "figure": str(png_path),
            }
        )
    return rows


def sample_indices(n: int, max_items: int, seed: int) -> slice:
    del seed
    if max_items <= 0 or n <= max_items:
        return slice(None)
    return slice(0, int(max_items))


def sample_size(selector: slice, n: int) -> int:
    start, stop, step = selector.indices(n)
    if step <= 0:
        return 0
    return max(0, (stop - start + step - 1) // step)


def summarize_array(arr: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(arr)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def collect_split_stats(fr: h5py.File, prefix: str, max_items: int, seed: int) -> dict[str, object]:
    stats: dict[str, object] = {}
    tile_ptr_key = dataset_key(prefix, "instance_tile_ptr_pool")
    if tile_ptr_key in fr:
        ptr = np.asarray(fr[tile_ptr_key], dtype=np.int64)
        n_tiles = int(fr[dataset_key(prefix, "tile_input_pool")].shape[0])
        counts = np.bincount(ptr, minlength=n_tiles)
        stats["tile_instance_count"] = summarize_array(counts)
        stats["n_tiles"] = int(n_tiles)
        stats["n_instances"] = int(ptr.size)
    n_instances = int(fr[dataset_key(prefix, "instance_mask_targets_pool")].shape[0])
    idx = sample_indices(n_instances, max_items, seed)
    stats["metric_sample_size"] = int(sample_size(idx, n_instances))
    metric_stats: dict[str, object] = {}
    categorical_counts: dict[str, dict[str, int]] = {}
    for suffix in SCALAR_METRICS:
        key = dataset_key(prefix, suffix)
        if key not in fr:
            continue
        values = np.asarray(fr[key][idx])
        if values.ndim != 1:
            continue
        if suffix.endswith("kind_pool") or suffix.endswith("size_bin_ids_pool"):
            unique, counts = np.unique(values.astype(np.int64), return_counts=True)
            categorical_counts[suffix] = {str(int(k)): int(v) for k, v in zip(unique, counts, strict=False)}
        else:
            metric_stats[suffix] = summarize_array(values.astype(np.float64))
    stats["metrics"] = metric_stats
    stats["categorical_counts"] = categorical_counts
    return stats


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, split_stats in summary.get("splits", {}).items():
        if not isinstance(split_stats, dict):
            continue
        metrics = split_stats.get("metrics", {})
        if isinstance(metrics, dict):
            for metric, values in metrics.items():
                if not isinstance(values, dict):
                    continue
                row = {"split": split, "metric": metric}
                row.update(values)
                rows.append(row)
        tile_count = split_stats.get("tile_instance_count")
        if isinstance(tile_count, dict):
            row = {"split": split, "metric": "tile_instance_count"}
            row.update(tile_count)
            rows.append(row)
    return rows


def plot_hist(ax: plt.Axes, values: np.ndarray, title: str, xlabel: str, bins: int = 60) -> None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title)
        return
    ax.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)


def plot_statistics(fr: h5py.File, out_dir: Path, max_items: int, seed: int) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plot_specs = [
        ("train", "mask_refine_iou_raw_refined_pool", "train raw/refined IoU"),
        ("val", "mask_refine_iou_raw_refined_pool", "val raw/refined IoU"),
        ("train", "mask_refine_area_change_ratio_pool", "train area change ratio"),
        ("val", "mask_refine_area_change_ratio_pool", "val area change ratio"),
        ("train", "mask_refine_refined_target_similarity_pool", "train refined target similarity"),
        ("val", "mask_refine_refined_target_similarity_pool", "val refined target similarity"),
        ("train", "mask_refine_latent_global_loss_ratio_pool", "train latent global loss ratio"),
        ("val", "mask_refine_latent_global_loss_ratio_pool", "val latent global loss ratio"),
    ]
    axes_flat = axes.ravel()
    for ax, (prefix, suffix, title) in zip(axes_flat, plot_specs, strict=False):
        key = dataset_key(prefix, suffix)
        if key not in fr:
            ax.text(0.5, 0.5, f"missing {key}", ha="center", va="center")
            ax.set_title(title)
            continue
        n = int(fr[key].shape[0])
        idx = sample_indices(n, max_items, seed + (0 if prefix == "train" else 1))
        values = np.asarray(fr[key][idx], dtype=np.float32)
        plot_hist(ax, values, title, suffix)

    ax = axes_flat[-1]
    for prefix, color in [("train", "#4C78A8"), ("val", "#F58518")]:
        ptr_key = dataset_key(prefix, "instance_tile_ptr_pool")
        tile_key = dataset_key(prefix, "tile_input_pool")
        if ptr_key not in fr or tile_key not in fr:
            continue
        ptr = np.asarray(fr[ptr_key], dtype=np.int64)
        counts = np.bincount(ptr, minlength=int(fr[tile_key].shape[0]))
        ax.hist(counts, bins=40, alpha=0.65, label=prefix, color=color)
    ax.set_title("instances per tile", fontsize=10)
    ax.set_xlabel("instances")
    ax.set_ylabel("tile count")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "stat_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = args.output_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as fr:
        attrs = read_attrs(fr)
        selected_rows = visualize_tiles(fr, args, args.split, tile_dir)
        summary = {
            "input_h5": str(args.input_h5),
            "attrs": attrs,
            "visualized_split": args.split,
            "num_visualized_tiles": len(selected_rows),
            "splits": {
                "train": collect_split_stats(fr, "train", args.max_stat_instances, args.seed),
                "val": collect_split_stats(fr, "val", args.max_stat_instances, args.seed + 1),
            },
        }
        plot_statistics(fr, args.output_dir, args.max_stat_instances, args.seed)

    write_csv_rows(args.output_dir / "selected_tiles.csv", selected_rows)
    write_csv_rows(args.output_dir / "metric_summary.csv", flatten_metric_rows(summary))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[visual_sys_hd] wrote tile figures: {tile_dir}")
    print(f"[visual_sys_hd] wrote statistics: {args.output_dir}")


if __name__ == "__main__":
    main()

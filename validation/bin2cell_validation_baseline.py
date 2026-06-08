from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import anndata as ad
import bin2cell as b2c
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bin2cell-style nucleus expansion baseline on HDpainter validation tiles "
            "and compare expanded masks with validation instance targets."
        )
    )
    parser.add_argument("--sample", action="append", choices=("COAD", "PRAD", "NSCLC"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/OV/batch_preprocess"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/HDpainter1/validation/bin2cell_baseline"))
    parser.add_argument("--target-key", default="val_instance_mask_targets_refined_pool")
    parser.add_argument("--max-bin-distance", type=int, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--subset-pca", action="store_true", default=True)
    parser.add_argument("--no-subset-pca", dest="subset_pca", action="store_false")
    parser.add_argument("--min-target-area", type=float, default=1.0)
    parser.add_argument("--max-tiles", type=int, default=0, help="Debug limit; 0 means all validation tiles.")
    return parser.parse_args()


def crop_center(mask: np.ndarray, center_yx: np.ndarray, size: int) -> np.ndarray:
    half = size // 2
    cy = int(round(float(center_yx[0])))
    cx = int(round(float(center_yx[1])))
    out = np.zeros((size, size), dtype=mask.dtype)
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size
    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y1, mask.shape[0])
    src_x1 = min(x1, mask.shape[1])
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = mask[
        src_y0:src_y1, src_x0:src_x1
    ]
    return out


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return np.nan
    return float(np.logical_and(pred, target).sum() / union)


def size_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size_bin, sub in df.groupby("size_bin_id", dropna=False):
        rows.append(
            {
                "size_bin_id": int(size_bin),
                "count": int(len(sub)),
                "mean_iou": float(sub["iou"].mean()),
                "median_iou": float(sub["iou"].median()),
                "mean_area_ratio": float(sub["area_ratio"].mean()),
                "mean_pred_area": float(sub["pred_area"].mean()),
                "mean_target_area": float(sub["target_area"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("size_bin_id")


def build_val_to_source_mapping(fr: h5py.File, src: h5py.File) -> np.ndarray:
    val_tile_index = fr["val_tile_index_pool"][:].astype(np.int64)
    val_tile_ptr = fr["val_instance_tile_ptr_pool"][:].astype(np.int64)
    source_offsets = src["instance_offsets"][:].astype(np.int64)
    counters: dict[int, int] = defaultdict(int)
    source_indices = np.empty(val_tile_ptr.shape[0], dtype=np.int64)
    for i, tile_ptr in enumerate(val_tile_ptr):
        source_tile = int(val_tile_index[int(tile_ptr)])
        local = counters[source_tile]
        source_indices[i] = int(source_offsets[source_tile] + local)
        counters[source_tile] += 1
    return source_indices


def label_tile_from_nuclei(src: h5py.File, source_tile: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = src["instance_offsets"][:].astype(np.int64)
    start = int(offsets[source_tile])
    end = int(offsets[source_tile + 1])
    masks = src["nucleus_mask_pool"][start:end]
    label_img = np.zeros(masks.shape[1:], dtype=np.int32)
    # Assign larger nuclei first; later smaller labels fill unassigned pixels only.
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    order = np.argsort(-areas)
    for local_idx in order:
        label = int(local_idx) + 1
        mask = masks[int(local_idx)] > 0
        label_img[(label_img == 0) & mask] = label
    return label_img, np.arange(start, end, dtype=np.int64)


def expand_tile(src: h5py.File, source_tile: int, max_bin_distance: int, k: int, subset_pca: bool) -> np.ndarray:
    label_img, _ = label_tile_from_nuclei(src, source_tile)
    h, w = label_img.shape
    rr, cc = np.indices((h, w), dtype=np.int32)
    obs = pd.DataFrame(
        {
            "array_row": rr.reshape(-1),
            "array_col": cc.reshape(-1),
            "labels": label_img.reshape(-1).astype(np.int64),
        }
    )
    if "x_low" in src:
        x = src["x_low"][source_tile].reshape(src["x_low"].shape[1], -1).T.astype(np.float32)
    elif "tile_input" in src:
        x = src["tile_input"][source_tile].reshape(src["tile_input"].shape[1], -1).T.astype(np.float32)
    else:
        x = sparse.csr_matrix((h * w, 1), dtype=np.float32)
    adata = ad.AnnData(X=x, obs=obs)
    b2c.expand_labels(
        adata,
        labels_key="labels",
        expanded_labels_key="labels_expanded",
        algorithm="max_bin_distance",
        max_bin_distance=int(max_bin_distance),
        k=int(k),
        subset_pca=bool(subset_pca),
    )
    return adata.obs["labels_expanded"].to_numpy(dtype=np.int32).reshape(h, w)


def evaluate_sample(sample: str, h5_path: Path, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    with h5py.File(h5_path, "r") as fr:
        source_h5 = fr.attrs["source_h5"]
        source_h5 = Path(source_h5.decode() if isinstance(source_h5, bytes) else source_h5)
        canvas_size = int(fr.attrs.get("canvas_size", 33))
        target_key = str(args.target_key)
        if target_key not in fr:
            raise KeyError(f"{h5_path} missing {target_key}")
        with h5py.File(source_h5, "r") as src:
            source_indices = build_val_to_source_mapping(fr, src)
            val_tile_index = fr["val_tile_index_pool"][:].astype(np.int64)
            val_tile_ptr = fr["val_instance_tile_ptr_pool"][:].astype(np.int64)
            source_offsets = src["instance_offsets"][:].astype(np.int64)
            tile_order = list(dict.fromkeys(int(x) for x in val_tile_index.tolist()))
            if args.max_tiles > 0:
                tile_order = tile_order[: int(args.max_tiles)]
            tile_set = set(tile_order)
            expanded_cache: dict[int, np.ndarray] = {}
            val_indices = [idx for idx, tile_ptr in enumerate(val_tile_ptr) if int(val_tile_index[int(tile_ptr)]) in tile_set]
            for n, val_i in enumerate(val_indices, start=1):
                source_i = int(source_indices[val_i])
                source_tile = int(val_tile_index[int(val_tile_ptr[val_i])])
                if source_tile not in expanded_cache:
                    print(f"[{sample}] expand tile {source_tile} ({len(expanded_cache)+1}/{len(tile_order)})", flush=True)
                    expanded_cache[source_tile] = expand_tile(
                        src,
                        source_tile=source_tile,
                        max_bin_distance=int(args.max_bin_distance),
                        k=int(args.k),
                        subset_pca=bool(args.subset_pca),
                    )
                local_label = int(source_i - int(source_offsets[source_tile]) + 1)
                expanded_label = expanded_cache[source_tile] == local_label
                center = src["nucleus_centers_yx_pool"][source_i]
                pred_crop = crop_center(expanded_label, center, canvas_size)
                target = np.asarray(fr[target_key][val_i, 0], dtype=bool)
                target_area = float(target.sum())
                if target_area < float(args.min_target_area):
                    continue
                pred_area = float(pred_crop.sum())
                rows.append(
                    {
                        "sample": sample,
                        "val_instance_index": int(val_i),
                        "source_tile": source_tile,
                        "source_instance_index": source_i,
                        "size_bin_id": int(fr["val_instance_size_bin_ids_pool"][val_i]),
                        "target_area": target_area,
                        "pred_area": pred_area,
                        "area_ratio": pred_area / max(target_area, 1.0),
                        "iou": iou(pred_crop, target),
                    }
                )
    df = pd.DataFrame(rows)
    sample_dir = out_dir / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(sample_dir / "bin2cell_val_instance_iou.csv", index=False)
    size_df = size_bin_summary(df)
    size_df.to_csv(sample_dir / "bin2cell_val_size_bin_summary.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "sample": sample,
                "count": int(len(df)),
                "mean_iou": float(df["iou"].mean()),
                "median_iou": float(df["iou"].median()),
                "mean_area_ratio": float(df["area_ratio"].mean()),
                "median_area_ratio": float(df["area_ratio"].median()),
            }
        ]
    )
    summary.to_csv(sample_dir / "bin2cell_val_summary.csv", index=False)
    save_plots(df, size_df, sample_dir, sample)
    return summary


def save_plots(df: pd.DataFrame, size_df: pd.DataFrame, out_dir: Path, sample: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(df["iou"].dropna(), bins=80, color="#4c78a8", alpha=0.85)
    plt.xlabel("IoU")
    plt.ylabel("instances")
    plt.title(f"{sample} bin2cell validation IoU")
    plt.tight_layout()
    plt.savefig(out_dir / "bin2cell_val_iou_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(size_df["size_bin_id"].astype(str), size_df["mean_iou"], color="#59a14f", alpha=0.9)
    plt.xlabel("size bin")
    plt.ylabel("mean IoU")
    plt.title(f"{sample} bin2cell IoU by size bin")
    plt.tight_layout()
    plt.savefig(out_dir / "bin2cell_val_size_bin_iou.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    samples = args.sample or ["COAD", "PRAD", "NSCLC"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sample in samples:
        h5_path = (
            args.data_root
            / sample
            / f"regularize_train_tiles_degraded_{sample}_nmf48.instchunk512_train.maskrefined.h5"
        )
        print(f"[sample] {sample}: {h5_path}", flush=True)
        summaries.append(evaluate_sample(sample, h5_path, args.output_dir, args))
    all_summary = pd.concat(summaries, axis=0, ignore_index=True)
    all_summary.to_csv(args.output_dir / "bin2cell_val_summary_all.csv", index=False)
    print(f"summary={args.output_dir / 'bin2cell_val_summary_all.csv'}")


if __name__ == "__main__":
    main()

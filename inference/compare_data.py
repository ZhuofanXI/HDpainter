from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


@dataclass
class DatasetSpec:
    name: str
    path: Path
    layer: str | None


@dataclass
class MatrixStats:
    row_sum: np.ndarray
    row_nnz: np.ndarray
    col_sum: np.ndarray
    col_nnz: np.ndarray
    total_sum: float
    total_nnz: int
    shape: tuple[int, int]


def parse_dataset(value: str) -> DatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must use NAME=PATH or NAME=PATH:LAYER")
    name, rest = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty.")
    layer = None
    path_text = rest
    if ":" in rest:
        candidate_path, candidate_layer = rest.rsplit(":", 1)
        if candidate_layer and not candidate_layer.startswith(("/", "\\")):
            path_text = candidate_path
            layer = None if candidate_layer in {"X", "None", "none"} else candidate_layer
    path = Path(path_text)
    return DatasetSpec(name=name, path=path, layer=layer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pseudo-cell data quality across bin2cell, HDpainter direct segmentation, "
            "and GNN post-processed results."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        help="Dataset as NAME=PATH or NAME=PATH:LAYER. Use layer X/None for adata.X.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--spatial-grid", type=int, default=40)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--sample-cells", type=int, default=3000)
    parser.add_argument("--sample-genes", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2024)
    parser.add_argument(
        "--reconstruction-dataset",
        default="",
        help="Dataset name to evaluate reconstruction against its own reference layer.",
    )
    parser.add_argument("--reconstruction-layer", default="GNN_ReX")
    parser.add_argument("--reconstruction-reference-layer", default="lognorm")
    return parser.parse_args()


def open_backed(path: Path) -> ad.AnnData:
    return ad.read_h5ad(path, backed="r")


def list_layers(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        if "layers" not in f:
            return []
        return list(f["layers"].keys())


def choose_layer(path: Path, requested: str | None) -> str | None:
    if requested is not None:
        return requested
    layers = list_layers(path)
    if "GNN_ReX" in layers:
        return "GNN_ReX"
    if "counts" in layers:
        return "counts"
    if "lognorm" in layers:
        return "lognorm"
    return None


def matrix_node(f: h5py.File, layer: str | None):
    if layer is None:
        return f["X"]
    if "layers" not in f or layer not in f["layers"]:
        available = list(f["layers"].keys()) if "layers" in f else []
        raise KeyError(f"Layer '{layer}' not found. Available layers: {available}")
    return f["layers"][layer]


def node_shape(node) -> tuple[int, int]:
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]), int(node.shape[1])
    if "shape" in node.attrs:
        shape = tuple(int(x) for x in node.attrs["shape"])
        return shape[0], shape[1]
    if "shape" in node:
        shape = tuple(int(x) for x in node["shape"][()])
        return shape[0], shape[1]
    raise ValueError("Cannot infer matrix shape from h5ad node.")


def is_sparse_node(node) -> bool:
    return isinstance(node, h5py.Group) and {"data", "indices", "indptr"}.issubset(node.keys())


def compute_matrix_stats(path: Path, layer: str | None, chunk_rows: int) -> MatrixStats:
    with h5py.File(path, "r") as f:
        node = matrix_node(f, layer)
        n_obs, n_vars = node_shape(node)
        row_sum = np.zeros(n_obs, dtype=np.float64)
        row_nnz = np.zeros(n_obs, dtype=np.int32)
        col_sum = np.zeros(n_vars, dtype=np.float64)
        col_nnz = np.zeros(n_vars, dtype=np.int64)
        total_nnz = 0

        if is_sparse_node(node):
            data_ds = node["data"]
            indices_ds = node["indices"]
            indptr = node["indptr"][:]
            for start_row in range(0, n_obs, int(chunk_rows)):
                end_row = min(start_row + int(chunk_rows), n_obs)
                start_ptr = int(indptr[start_row])
                end_ptr = int(indptr[end_row])
                data = np.asarray(data_ds[start_ptr:end_ptr], dtype=np.float64)
                indices = np.asarray(indices_ds[start_ptr:end_ptr], dtype=np.int64)
                local_indptr = indptr[start_row : end_row + 1] - start_ptr
                counts = np.diff(local_indptr).astype(np.int64)
                row_nnz[start_row:end_row] = counts.astype(np.int32)
                if data.size:
                    row_ids = np.repeat(np.arange(end_row - start_row), counts)
                    np.add.at(row_sum[start_row:end_row], row_ids, data)
                    col_sum += np.bincount(indices, weights=data, minlength=n_vars)
                    positive = data > 0
                    total_nnz += int(np.count_nonzero(positive))
                    if np.any(positive):
                        col_nnz += np.bincount(indices[positive], minlength=n_vars)
        else:
            for start_row in range(0, n_obs, int(chunk_rows)):
                end_row = min(start_row + int(chunk_rows), n_obs)
                block = np.asarray(node[start_row:end_row, :], dtype=np.float32)
                positive = block > 0
                row_sum[start_row:end_row] = block.sum(axis=1, dtype=np.float64)
                row_nnz[start_row:end_row] = positive.sum(axis=1, dtype=np.int32)
                col_sum += block.sum(axis=0, dtype=np.float64)
                col_nnz += positive.sum(axis=0, dtype=np.int64)
                total_nnz += int(np.count_nonzero(positive))

    return MatrixStats(
        row_sum=row_sum,
        row_nnz=row_nnz,
        col_sum=col_sum,
        col_nnz=col_nnz,
        total_sum=float(row_sum.sum()),
        total_nnz=int(total_nnz),
        shape=(n_obs, n_vars),
    )


def read_obs_var(path: Path) -> tuple[pd.DataFrame, pd.Index, pd.Index, dict[str, tuple[int, ...]]]:
    backed = open_backed(path)
    obs = backed.obs.copy()
    obs_names = backed.obs_names.copy()
    var_names = backed.var_names.copy()
    obsm_shapes = {key: tuple(value.shape) for key, value in backed.obsm.items()}
    backed.file.close()
    return obs, obs_names, var_names, obsm_shapes


def robust_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p05": float(np.percentile(finite, 5)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
        f"{prefix}_max": float(np.max(finite)),
    }


def get_spatial(obs: pd.DataFrame, path: Path, spatial_key: str) -> np.ndarray | None:
    backed = open_backed(path)
    coords = None
    if spatial_key in backed.obsm:
        coords = np.asarray(backed.obsm[spatial_key][:], dtype=np.float32)
    backed.file.close()
    if coords is None and {"centroid_x", "centroid_y"}.issubset(obs.columns):
        coords = obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    if coords is None and {"array_col", "array_row"}.issubset(obs.columns):
        coords = obs[["array_col", "array_row"]].to_numpy(dtype=np.float32)
    if coords is not None and coords.ndim == 2 and coords.shape[1] == 2:
        return coords
    return None


def save_histograms(metric_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("matrix_sum", "Per-cell matrix sum"),
        ("matrix_nnz", "Detected genes per cell"),
        ("n_bins", "Bins per cell"),
        ("area", "Cell area"),
        ("matrix_sum_per_bin", "Matrix sum per bin"),
        ("matrix_nnz_per_bin", "Detected genes per bin"),
        ("mean_pred_score", "Mean prediction score"),
    ]
    for metric, title in metrics:
        if metric not in metric_df.columns:
            continue
        plt.figure(figsize=(8, 5))
        for name, sub in metric_df.groupby("dataset"):
            values = sub[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            hi = np.percentile(values, 99)
            values = values[values <= hi] if np.isfinite(hi) else values
            plt.hist(values, bins=80, alpha=0.35, density=True, label=name)
        plt.title(title)
        plt.xlabel(metric)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_distribution.png", dpi=180)
        plt.close()


def spatial_heatmaps(dataset_records: dict[str, dict], out_dir: Path, grid: int) -> None:
    for name, rec in dataset_records.items():
        coords = rec["coords"]
        if coords is None:
            continue
        x = coords[:, 0]
        y = coords[:, 1]
        weights = rec["stats"].row_sum
        cell_grid, xedges, yedges = np.histogram2d(x, y, bins=int(grid))
        signal_grid, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=weights)
        for label, arr in [("cell_density", cell_grid), ("signal_density", signal_grid)]:
            plt.figure(figsize=(6, 5))
            plt.imshow(arr.T, origin="lower", cmap="viridis", aspect="equal")
            plt.colorbar(label=label)
            plt.title(f"{name} {label}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{name}_{label}_spatial_heatmap.png", dpi=180)
            plt.close()


def pairwise_gene_metrics(dataset_records: dict[str, dict]) -> pd.DataFrame:
    rows = []
    names = list(dataset_records)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            lrec = dataset_records[left]
            rrec = dataset_records[right]
            common, li, ri = np.intersect1d(lrec["var_names"], rrec["var_names"], return_indices=True)
            if common.size < 2:
                continue
            l_mean = lrec["stats"].col_sum[li] / max(lrec["stats"].shape[0], 1)
            r_mean = rrec["stats"].col_sum[ri] / max(rrec["stats"].shape[0], 1)
            l_detect = lrec["stats"].col_nnz[li] / max(lrec["stats"].shape[0], 1)
            r_detect = rrec["stats"].col_nnz[ri] / max(rrec["stats"].shape[0], 1)
            l_prob = lrec["stats"].col_sum[li].astype(float)
            r_prob = rrec["stats"].col_sum[ri].astype(float)
            if l_prob.sum() > 0:
                l_prob = l_prob / l_prob.sum()
            if r_prob.sum() > 0:
                r_prob = r_prob / r_prob.sum()
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "common_genes": int(common.size),
                    "gene_mean_pearson": safe_corr(l_mean, r_mean, "pearson"),
                    "gene_mean_spearman": safe_corr(l_mean, r_mean, "spearman"),
                    "gene_detection_pearson": safe_corr(l_detect, r_detect, "pearson"),
                    "gene_detection_spearman": safe_corr(l_detect, r_detect, "spearman"),
                    "gene_total_js_distance": float(jensenshannon(l_prob, r_prob)) if l_prob.sum() and r_prob.sum() else np.nan,
                    "top_100_gene_overlap": top_overlap(l_mean, r_mean, 100),
                    "top_500_gene_overlap": top_overlap(l_mean, r_mean, 500),
                }
            )
    return pd.DataFrame(rows)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 2:
        return np.nan
    x = x[keep]
    y = y[keep]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def top_overlap(x: np.ndarray, y: np.ndarray, k: int) -> float:
    k_eff = min(int(k), x.size, y.size)
    if k_eff <= 0:
        return np.nan
    left = set(np.argpartition(x, -k_eff)[-k_eff:])
    right = set(np.argpartition(y, -k_eff)[-k_eff:])
    return float(len(left & right) / k_eff)


def pairwise_cell_metrics(dataset_records: dict[str, dict], cell_id_col: str) -> pd.DataFrame:
    rows = []
    names = list(dataset_records)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            lrec = dataset_records[left]
            rrec = dataset_records[right]
            l_ids = cell_ids(lrec["obs"], lrec["obs_names"], cell_id_col)
            r_ids = cell_ids(rrec["obs"], rrec["obs_names"], cell_id_col)
            common, li, ri = np.intersect1d(l_ids, r_ids, return_indices=True)
            if common.size < 2:
                rows.append({"left": left, "right": right, "common_cells": int(common.size)})
                continue
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "common_cells": int(common.size),
                    "matrix_sum_pearson": safe_corr(lrec["stats"].row_sum[li], rrec["stats"].row_sum[ri], "pearson"),
                    "matrix_sum_spearman": safe_corr(lrec["stats"].row_sum[li], rrec["stats"].row_sum[ri], "spearman"),
                    "matrix_nnz_pearson": safe_corr(lrec["stats"].row_nnz[li], rrec["stats"].row_nnz[ri], "pearson"),
                    "matrix_nnz_spearman": safe_corr(lrec["stats"].row_nnz[li], rrec["stats"].row_nnz[ri], "spearman"),
                }
            )
    return pd.DataFrame(rows)


def cell_ids(obs: pd.DataFrame, obs_names: pd.Index, cell_id_col: str) -> np.ndarray:
    if cell_id_col in obs.columns:
        return obs[cell_id_col].astype(str).to_numpy()
    return obs_names.astype(str).to_numpy()


def gene_scatter_plots(dataset_records: dict[str, dict], out_dir: Path) -> None:
    names = list(dataset_records)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            lrec = dataset_records[left]
            rrec = dataset_records[right]
            common, li, ri = np.intersect1d(lrec["var_names"], rrec["var_names"], return_indices=True)
            if common.size < 2:
                continue
            l_mean = lrec["stats"].col_sum[li] / max(lrec["stats"].shape[0], 1)
            r_mean = rrec["stats"].col_sum[ri] / max(rrec["stats"].shape[0], 1)
            l_det = lrec["stats"].col_nnz[li] / max(lrec["stats"].shape[0], 1)
            r_det = rrec["stats"].col_nnz[ri] / max(rrec["stats"].shape[0], 1)
            for metric, x, y in [("gene_mean", l_mean, r_mean), ("gene_detection", l_det, r_det)]:
                plt.figure(figsize=(5, 5))
                plt.scatter(np.log1p(x), np.log1p(y), s=3, alpha=0.35)
                plt.xlabel(f"{left} log1p {metric}")
                plt.ylabel(f"{right} log1p {metric}")
                plt.title(f"{left} vs {right} {metric}")
                plt.tight_layout()
                plt.savefig(out_dir / f"{left}_vs_{right}_{metric}_scatter.png", dpi=180)
                plt.close()


def sample_layer_matrix(path: Path, layer: str, cells: np.ndarray, genes: np.ndarray) -> np.ndarray:
    backed = ad.read_h5ad(path, backed="r")
    sub = backed[cells, genes].to_memory()
    backed.file.close()
    if layer == "X":
        x = sub.X
    else:
        x = sub.layers[layer]
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def reconstruction_metrics(
    spec: DatasetSpec,
    rec: dict,
    out_dir: Path,
    layer: str,
    reference_layer: str,
    sample_cells: int,
    sample_genes: int,
    seed: int,
) -> pd.DataFrame:
    layers = list_layers(spec.path)
    if layer not in layers or reference_layer not in layers:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    n_cells, n_genes = rec["stats"].shape
    cells = np.sort(rng.choice(n_cells, size=min(sample_cells, n_cells), replace=False))
    genes = np.sort(rng.choice(n_genes, size=min(sample_genes, n_genes), replace=False))
    pred = sample_layer_matrix(spec.path, layer, cells, genes)
    ref = sample_layer_matrix(spec.path, reference_layer, cells, genes)
    diff = pred - ref
    row = {
        "dataset": spec.name,
        "layer": layer,
        "reference_layer": reference_layer,
        "sample_cells": int(cells.size),
        "sample_genes": int(genes.size),
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "pearson_flat": safe_corr(ref.reshape(-1), pred.reshape(-1), "pearson"),
        "spearman_flat": safe_corr(ref.reshape(-1), pred.reshape(-1), "spearman"),
    }
    plt.figure(figsize=(5, 5))
    idx = np.random.default_rng(int(seed) + 1).choice(ref.size, size=min(150000, ref.size), replace=False)
    plt.scatter(ref.reshape(-1)[idx], pred.reshape(-1)[idx], s=1, alpha=0.15)
    plt.xlabel(reference_layer)
    plt.ylabel(layer)
    plt.title(f"{spec.name} reconstruction sample")
    plt.tight_layout()
    plt.savefig(out_dir / f"{spec.name}_reconstruction_scatter.png", dpi=180)
    plt.close()
    return pd.DataFrame([row])


def build_per_cell_frame(name: str, obs: pd.DataFrame, stats: MatrixStats) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dataset": name,
            "matrix_sum": stats.row_sum,
            "matrix_nnz": stats.row_nnz,
        },
        index=obs.index,
    )
    for col in ["n_bins", "area", "total_counts", "n_genes_by_counts", "mean_pred_score", "max_pred_score"]:
        if col in obs.columns:
            frame[col] = pd.to_numeric(obs[col], errors="coerce").to_numpy()
    if "n_bins" in frame.columns:
        denom = frame["n_bins"].replace(0, np.nan)
        frame["matrix_sum_per_bin"] = frame["matrix_sum"] / denom
        frame["matrix_nnz_per_bin"] = frame["matrix_nnz"] / denom
    return frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dataset_records: dict[str, dict] = {}
    per_cell_frames = []
    summary_rows = []
    resolved_specs = []

    for spec in args.dataset:
        layer = choose_layer(spec.path, spec.layer)
        resolved_specs.append(DatasetSpec(spec.name, spec.path, layer))
        print(f"[load] {spec.name}: path={spec.path} layer={layer or 'X'}", flush=True)
        obs, obs_names, var_names, obsm_shapes = read_obs_var(spec.path)
        stats = compute_matrix_stats(spec.path, layer=layer, chunk_rows=int(args.chunk_rows))
        coords = get_spatial(obs, spec.path, spatial_key=str(args.spatial_key))
        per_cell = build_per_cell_frame(spec.name, obs, stats)
        per_cell_frames.append(per_cell)
        detected_genes = int(np.count_nonzero(stats.col_nnz > 0))
        density = float(stats.total_nnz / max(stats.shape[0] * stats.shape[1], 1))
        row = {
            "dataset": spec.name,
            "path": str(spec.path),
            "layer": layer or "X",
            "n_cells": int(stats.shape[0]),
            "n_genes": int(stats.shape[1]),
            "detected_genes": detected_genes,
            "matrix_total_sum": float(stats.total_sum),
            "matrix_density": density,
            "obsm_shapes": json.dumps({k: list(v) for k, v in obsm_shapes.items()}, ensure_ascii=True),
        }
        row.update(robust_summary(stats.row_sum, "cell_matrix_sum"))
        row.update(robust_summary(stats.row_nnz, "cell_detected_genes"))
        for col in ["n_bins", "area", "mean_pred_score", "max_pred_score"]:
            if col in per_cell.columns:
                row.update(robust_summary(per_cell[col].to_numpy(dtype=float), col))
        summary_rows.append(row)
        dataset_records[spec.name] = {
            "spec": DatasetSpec(spec.name, spec.path, layer),
            "obs": obs,
            "obs_names": obs_names,
            "var_names": var_names,
            "stats": stats,
            "coords": coords,
            "per_cell": per_cell,
        }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    per_cell_df = pd.concat(per_cell_frames, axis=0)
    per_cell_df.to_csv(args.output_dir / "per_cell_metrics.csv")

    gene_rows = []
    for name, rec in dataset_records.items():
        stats = rec["stats"]
        gene_rows.append(
            pd.DataFrame(
                {
                    "dataset": name,
                    "gene": rec["var_names"].astype(str),
                    "gene_sum": stats.col_sum,
                    "gene_detected_cells": stats.col_nnz,
                    "gene_mean": stats.col_sum / max(stats.shape[0], 1),
                    "gene_detection_rate": stats.col_nnz / max(stats.shape[0], 1),
                }
            )
        )
    pd.concat(gene_rows, axis=0).to_csv(args.output_dir / "per_gene_metrics.csv", index=False)

    gene_pairwise = pairwise_gene_metrics(dataset_records)
    gene_pairwise.to_csv(args.output_dir / "pairwise_gene_metrics.csv", index=False)
    cell_pairwise = pairwise_cell_metrics(dataset_records, cell_id_col=str(args.cell_id_col))
    cell_pairwise.to_csv(args.output_dir / "pairwise_cell_metrics.csv", index=False)

    save_histograms(per_cell_df, plot_dir)
    spatial_heatmaps(dataset_records, plot_dir, grid=int(args.spatial_grid))
    gene_scatter_plots(dataset_records, plot_dir)

    if args.reconstruction_dataset:
        spec_map = {spec.name: spec for spec in resolved_specs}
        if args.reconstruction_dataset not in spec_map:
            raise ValueError(f"Unknown reconstruction dataset: {args.reconstruction_dataset}")
        recon_df = reconstruction_metrics(
            spec=spec_map[args.reconstruction_dataset],
            rec=dataset_records[args.reconstruction_dataset],
            out_dir=plot_dir,
            layer=str(args.reconstruction_layer),
            reference_layer=str(args.reconstruction_reference_layer),
            sample_cells=int(args.sample_cells),
            sample_genes=int(args.sample_genes),
            seed=int(args.random_seed),
        )
        recon_df.to_csv(args.output_dir / "reconstruction_metrics.csv", index=False)

    print(f"summary={args.output_dir / 'summary_metrics.csv'}")
    print(f"plots={plot_dir}")


if __name__ == "__main__":
    main()

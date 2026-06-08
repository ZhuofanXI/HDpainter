from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


DEFAULT_MARKERS = {
    "epithelial_tumor": ["EPCAM", "KRT8", "KRT18", "KRT19", "PAX8", "MUC16", "MSLN", "CLDN3", "CLDN4"],
    "stromal_fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1", "FAP", "PDGFRA"],
    "endothelial": ["PECAM1", "VWF", "KDR", "ENG", "FLT1"],
    "t_cell": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A", "CD8B"],
    "b_cell": ["MS4A1", "CD79A", "CD79B", "BANK1"],
    "nk_cell": ["NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"],
    "myeloid": ["LYZ", "LST1", "FCGR3A", "CD68", "C1QA", "C1QB", "MS4A7"],
    "pericyte_smooth_muscle": ["ACTA2", "RGS5", "MYH11", "TAGLN", "MCAM"],
}


def parse_dataset(value: str) -> tuple[str, Path, str | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must use NAME=PATH or NAME=PATH:LAYER")
    name, rest = value.split("=", 1)
    layer = None
    path_text = rest
    if ":" in rest:
        candidate_path, candidate_layer = rest.rsplit(":", 1)
        if candidate_layer and not candidate_layer.startswith(("/", "\\")):
            path_text = candidate_path
            layer = None if candidate_layer in {"X", "None", "none"} else candidate_layer
    return name, Path(path_text), layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VisiumHD pseudo-cell signal quality with spatial held-out recovery "
            "and marker spatial-quality metrics."
        )
    )
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--heldout-cells", type=int, default=6000)
    parser.add_argument("--heldout-genes", type=int, default=1200)
    parser.add_argument("--heldout-positive", type=int, default=120000)
    parser.add_argument("--heldout-zero", type=int, default=120000)
    parser.add_argument("--marker-cells", type=int, default=30000)
    parser.add_argument("--spatial-k", type=int, default=8)
    parser.add_argument("--marker-file", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=2024)
    return parser.parse_args()


def dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def open_adata(path: Path) -> ad.AnnData:
    return ad.read_h5ad(path, backed="r")


def get_gene_indices(adata: ad.AnnData, genes: list[str]) -> tuple[list[str], np.ndarray]:
    var_upper = pd.Series(np.arange(adata.n_vars), index=pd.Index(adata.var_names.astype(str)).str.upper())
    found = []
    idx = []
    for gene in genes:
        key = gene.upper()
        if key in var_upper.index:
            loc = var_upper.loc[key]
            if isinstance(loc, pd.Series):
                loc = int(loc.iloc[0])
            found.append(str(adata.var_names[int(loc)]))
            idx.append(int(loc))
    return found, np.asarray(idx, dtype=np.int64)


def spatial_coords(adata: ad.AnnData, spatial_key: str) -> np.ndarray:
    if spatial_key in adata.obsm:
        return np.asarray(adata.obsm[spatial_key][:], dtype=np.float32)
    if {"centroid_x", "centroid_y"}.issubset(adata.obs.columns):
        return adata.obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    if {"array_col", "array_row"}.issubset(adata.obs.columns):
        return adata.obs[["array_col", "array_row"]].to_numpy(dtype=np.float32)
    raise ValueError("No spatial coordinates found.")


def full_library_size(adata: ad.AnnData) -> np.ndarray:
    if "total_counts" in adata.obs.columns:
        lib = pd.to_numeric(adata.obs["total_counts"], errors="coerce").to_numpy(dtype=np.float32)
        if np.all(np.isfinite(lib)) and np.nanmax(lib) > 0:
            return lib
    x = adata.layers["counts"] if "counts" in adata.layers else adata.X
    sums = np.asarray(x.sum(axis=1)).reshape(-1).astype(np.float32)
    return sums


def read_matrix(
    adata: ad.AnnData,
    rows: np.ndarray,
    cols: np.ndarray,
    layer: str | None,
    target_sum: float,
) -> np.ndarray:
    sub = adata[rows, cols].to_memory()
    if layer is None:
        x = sub.X
        return dense(x).astype(np.float32, copy=False)
    x = sub.layers[layer]
    x = dense(x).astype(np.float32, copy=False)
    if layer == "counts":
        lib = full_library_size(adata)[rows].astype(np.float32)
        scale = np.divide(float(target_sum), lib, out=np.zeros_like(lib), where=lib > 0)
        x = x * scale[:, None]
        x = np.log1p(x).astype(np.float32, copy=False)
    return x


def choose_genes_for_heldout(datasets, n_genes: int, rng: np.random.Generator) -> list[str]:
    common = None
    for _, adata, _ in datasets:
        genes = pd.Index(adata.var_names.astype(str))
        common = genes if common is None else common.intersection(genes)
    if common is None or len(common) == 0:
        raise ValueError("No common genes across datasets.")
    common = np.asarray(common, dtype=object)
    size = min(int(n_genes), common.size)
    return list(rng.choice(common, size=size, replace=False))


def sample_entries(
    x: np.ndarray,
    n_positive: int,
    n_zero: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(x.reshape(-1) > 0)
    zero = np.flatnonzero(x.reshape(-1) <= 0)
    picks = []
    labels = []
    if positive.size:
        n = min(int(n_positive), positive.size)
        picks.append(rng.choice(positive, size=n, replace=False))
        labels.extend(["positive"] * n)
    if zero.size and n_zero > 0:
        n = min(int(n_zero), zero.size)
        picks.append(rng.choice(zero, size=n, replace=False))
        labels.extend(["zero"] * n)
    if not picks:
        raise ValueError("No held-out entries could be sampled.")
    return np.concatenate(picks), np.asarray(labels, dtype=object)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    if y_true.size == 0:
        return {f"{prefix}_n": 0}
    diff = y_pred - y_true
    row = {
        f"{prefix}_n": int(y_true.size),
        f"{prefix}_mse": float(np.mean(diff * diff)),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_bias": float(np.mean(diff)),
    }
    if y_true.size > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        row[f"{prefix}_pearson"] = float(pearsonr(y_true, y_pred).statistic)
        row[f"{prefix}_spearman"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        row[f"{prefix}_pearson"] = np.nan
        row[f"{prefix}_spearman"] = np.nan
    return row


def heldout_recovery(
    name: str,
    adata: ad.AnnData,
    layer: str | None,
    genes: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, float]:
    coords = spatial_coords(adata, args.spatial_key)
    n_cells = adata.n_obs
    sample_cells = np.sort(rng.choice(n_cells, size=min(args.heldout_cells, n_cells), replace=False))
    _, gene_idx = get_gene_indices(adata, genes)
    if gene_idx.size == 0:
        raise ValueError(f"{name}: no held-out genes found.")

    nn = NearestNeighbors(n_neighbors=min(args.spatial_k + 1, n_cells), metric="euclidean")
    nn.fit(coords)
    neighbor_idx = nn.kneighbors(coords[sample_cells], return_distance=False)[:, 1:]
    union_rows = np.unique(np.concatenate([sample_cells, neighbor_idx.reshape(-1)]))
    row_pos = pd.Series(np.arange(union_rows.size), index=union_rows)
    x_union = read_matrix(adata, union_rows, gene_idx, layer=layer, target_sum=float(args.target_sum))
    sample_pos = row_pos.loc[sample_cells].to_numpy()
    neighbor_pos = row_pos.loc[neighbor_idx.reshape(-1)].to_numpy().reshape(neighbor_idx.shape)
    x_sample = x_union[sample_pos]
    pred_sample = x_union[neighbor_pos].mean(axis=1)

    flat_idx, labels = sample_entries(
        x_sample,
        n_positive=int(args.heldout_positive),
        n_zero=int(args.heldout_zero),
        rng=rng,
    )
    y_true = x_sample.reshape(-1)[flat_idx]
    y_pred = pred_sample.reshape(-1)[flat_idx]
    rows = flat_idx // x_sample.shape[1]
    cols = flat_idx % x_sample.shape[1]
    heldout_df = pd.DataFrame(
        {
            "dataset": name,
            "cell_index": sample_cells[rows],
            "gene": np.asarray(genes, dtype=object)[cols],
            "entry_type": labels,
            "true": y_true,
            "pred_neighbor_mean": y_pred,
        }
    )
    heldout_df.to_csv(out_dir / f"{name}_heldout_entries.csv", index=False)

    row = {"dataset": name, "layer": layer or "X", "heldout_cells": int(sample_cells.size), "heldout_genes": int(gene_idx.size)}
    row.update(metric_row(y_true, y_pred, "all"))
    for label in ["positive", "zero"]:
        mask = labels == label
        row.update(metric_row(y_true[mask], y_pred[mask], label))

    plt.figure(figsize=(5, 5))
    plot_n = min(120000, y_true.size)
    plot_idx = rng.choice(y_true.size, size=plot_n, replace=False)
    plt.scatter(y_true[plot_idx], y_pred[plot_idx], s=1, alpha=0.15)
    plt.xlabel("held-out true signal")
    plt.ylabel("spatial-neighbor prediction")
    plt.title(f"{name} held-out recovery")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_heldout_recovery_scatter.png", dpi=180)
    plt.close()
    return row


def load_marker_sets(marker_file: Path | None) -> dict[str, list[str]]:
    if marker_file is None:
        return DEFAULT_MARKERS
    markers: dict[str, list[str]] = {}
    text = marker_file.read_text(encoding="utf-8").splitlines()
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            group, gene = [x.strip() for x in line.split(",", 1)]
        elif "\t" in line:
            group, gene = [x.strip() for x in line.split("\t", 1)]
        else:
            group, gene = "custom", line
        markers.setdefault(group, []).append(gene)
    return markers


def moran_i(x: np.ndarray, neighbors: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    z = x - np.mean(x)
    denom = np.sum(z * z)
    if denom <= 0:
        return np.nan
    neighbor_z = z[neighbors]
    num = np.sum(z[:, None] * neighbor_z)
    n = x.size
    w = neighbors.size
    return float((n / w) * (num / denom))


def marker_spatial_quality(
    name: str,
    adata: ad.AnnData,
    layer: str | None,
    marker_sets: dict[str, list[str]],
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_dir: Path,
) -> pd.DataFrame:
    coords_all = spatial_coords(adata, args.spatial_key)
    n_cells = adata.n_obs
    sample_cells = np.sort(rng.choice(n_cells, size=min(args.marker_cells, n_cells), replace=False))
    coords = coords_all[sample_cells]
    nn = NearestNeighbors(n_neighbors=min(args.spatial_k + 1, sample_cells.size), metric="euclidean")
    nn.fit(coords)
    neighbors = nn.kneighbors(coords, return_distance=False)[:, 1:]

    rows = []
    for group, genes in marker_sets.items():
        found, gene_idx = get_gene_indices(adata, genes)
        if gene_idx.size == 0:
            continue
        x = read_matrix(adata, sample_cells, gene_idx, layer=layer, target_sum=float(args.target_sum))
        score = x.mean(axis=1)
        neighbor_score = score[neighbors].mean(axis=1)
        high_cut = np.percentile(score, 90)
        high = score >= high_cut
        top_edge_fraction = float(np.mean(high[:, None] & high[neighbors]))
        expected = float(np.mean(high) ** 2)
        enrichment = top_edge_fraction / expected if expected > 0 else np.nan
        row = {
            "dataset": name,
            "layer": layer or "X",
            "marker_group": group,
            "n_markers_found": int(len(found)),
            "markers_found": ";".join(found),
            "score_mean": float(np.mean(score)),
            "score_median": float(np.median(score)),
            "score_p90": float(high_cut),
            "score_top10_mean": float(np.mean(score[high])) if np.any(high) else np.nan,
            "score_background_mean": float(np.mean(score[~high])) if np.any(~high) else np.nan,
            "moran_i": moran_i(score, neighbors),
            "neighbor_corr_pearson": metric_corr(score, neighbor_score, "pearson"),
            "neighbor_corr_spearman": metric_corr(score, neighbor_score, "spearman"),
            "top10_neighbor_enrichment": float(enrichment),
        }
        rows.append(row)

        plt.figure(figsize=(6, 5))
        plt.scatter(coords[:, 0], coords[:, 1], c=score, s=2, cmap="viridis")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(label=f"{group} marker score")
        plt.title(f"{name} {group}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_{group}_marker_spatial.png", dpi=180)
        plt.close()
    return pd.DataFrame(rows)


def metric_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return np.nan
    x = x[keep]
    y = y[keep]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(int(args.random_seed))

    opened = []
    for name, path, layer in args.dataset:
        print(f"[open] {name}: {path} layer={layer or 'X'}", flush=True)
        opened.append((name, open_adata(path), layer))

    heldout_genes = choose_genes_for_heldout(opened, n_genes=int(args.heldout_genes), rng=rng)
    (args.output_dir / "heldout_genes.txt").write_text("\n".join(heldout_genes), encoding="utf-8")
    marker_sets = load_marker_sets(args.marker_file)

    heldout_rows = []
    marker_tables = []
    for name, adata_obj, layer in opened:
        print(f"[heldout] {name}", flush=True)
        heldout_rows.append(heldout_recovery(name, adata_obj, layer, heldout_genes, args, rng, plot_dir))
        print(f"[markers] {name}", flush=True)
        marker_tables.append(marker_spatial_quality(name, adata_obj, layer, marker_sets, args, rng, plot_dir))

    heldout_df = pd.DataFrame(heldout_rows)
    marker_df = pd.concat(marker_tables, axis=0, ignore_index=True) if marker_tables else pd.DataFrame()
    heldout_df.to_csv(args.output_dir / "heldout_reconstruction_metrics.csv", index=False)
    marker_df.to_csv(args.output_dir / "marker_spatial_quality.csv", index=False)

    if not marker_df.empty:
        pivot = marker_df.pivot_table(
            index="marker_group",
            columns="dataset",
            values=["moran_i", "neighbor_corr_pearson", "top10_neighbor_enrichment"],
            aggfunc="mean",
        )
        pivot.to_csv(args.output_dir / "marker_spatial_quality_pivot.csv")

    for _, adata_obj, _ in opened:
        adata_obj.file.close()

    print(f"heldout={args.output_dir / 'heldout_reconstruction_metrics.csv'}")
    print(f"markers={args.output_dir / 'marker_spatial_quality.csv'}")
    print(f"plots={plot_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a segmentation-derived spatial pseudo-single-cell AnnData "
            "for HERGAST-like graph post-processing."
        )
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA when the downstream signal model uses HVG/all-gene features.",
    )
    parser.add_argument("--min-counts", type=float, default=20.0)
    parser.add_argument("--min-genes", type=int, default=15)
    parser.add_argument("--min-bins", type=int, default=9)
    parser.add_argument(
        "--use-highly-variable",
        action="store_true",
        help="Use a simple variance-based top-gene filter before PCA.",
    )
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument(
        "--signal-n-top-genes",
        type=int,
        default=0,
        help="Number of HVGs for signal_process.py. 0 means mark all genes as HVG.",
    )
    return parser.parse_args()


def _to_csr_float32(x) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(x, dtype=np.float32))


def _as_dense_float32(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def sparse_column_variance(x: sp.csr_matrix) -> np.ndarray:
    x = x.tocsr().astype(np.float32)
    n_obs = max(int(x.shape[0]), 1)
    mean = np.asarray(x.sum(axis=0)).reshape(-1) / float(n_obs)
    mean_sq = np.asarray(x.power(2).sum(axis=0)).reshape(-1) / float(n_obs)
    return np.maximum(mean_sq - mean * mean, 0.0).astype(np.float32, copy=False)


def filter_cells(adata: ad.AnnData, min_counts: float, min_genes: int, min_bins: int) -> ad.AnnData:
    x_counts = _to_csr_float32(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    total_counts = np.asarray(x_counts.sum(axis=1)).reshape(-1)
    n_genes = np.asarray((x_counts > 0).sum(axis=1)).reshape(-1)

    keep = np.ones(adata.n_obs, dtype=bool)
    if min_counts > 0:
        keep &= total_counts >= float(min_counts)
    if min_genes > 0:
        keep &= n_genes >= int(min_genes)
    if min_bins > 1:
        if "n_bins" not in adata.obs.columns:
            raise ValueError("--min-bins requires obs['n_bins'] in the input h5ad.")
        keep &= adata.obs["n_bins"].to_numpy(dtype=np.int64) >= int(min_bins)

    if not np.any(keep):
        raise ValueError("All cells were filtered out. Relax min-counts/min-genes/min-bins.")
    return adata[keep].copy()


def validate_cell_level_input(adata: ad.AnnData, input_h5ad: Path) -> None:
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{input_h5ad} is empty: shape=({adata.n_obs}, {adata.n_vars}).")

    if "n_bins" not in adata.obs.columns:
        raise ValueError(
            f"{input_h5ad} does not look like a cell-level h5ad: missing obs['n_bins']. "
            "Run infer_hd.py through the aggregate stage first, or pass *_pred_cell_level.h5ad."
        )

    if "cell_id" not in adata.obs.columns and adata.obs.index.name != "cell_id":
        raise ValueError(
            f"{input_h5ad} does not look like a cell-level h5ad: missing obs['cell_id'] "
            "and obs index is not named 'cell_id'."
        )

    if not (
        "spatial" in adata.obsm
        or {"centroid_x", "centroid_y"}.issubset(adata.obs.columns)
        or {"array_col", "array_row"}.issubset(adata.obs.columns)
    ):
        raise ValueError(
            f"{input_h5ad} is missing spatial coordinates. Expected obsm['spatial'], "
            "obs['centroid_x'/'centroid_y'], or obs['array_col'/'array_row']."
        )

    if {"array_row", "array_col", "cell_id"}.issubset(adata.obs.columns):
        n_bins_unique = pd.to_numeric(adata.obs["n_bins"], errors="coerce").nunique(dropna=True)
        if n_bins_unique <= 2 and int(adata.obs["n_bins"].min()) <= 1:
            raise ValueError(
                f"{input_h5ad} looks like a bin-level h5ad rather than a cell-level aggregate: "
                "obs['n_bins'] has almost no variation. Pass *_pred_cell_level.h5ad."
            )


def normalize_log1p_counts(adata: ad.AnnData, target_sum: float) -> None:
    counts = _to_csr_float32(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    adata.layers["counts"] = counts.copy()

    library_size = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.float32)
    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size, dtype=np.float32),
        where=library_size > 0,
    )
    x_norm = counts.multiply(scale[:, None]).tocsr()
    x_lognorm = x_norm.copy()
    x_lognorm.data = np.log1p(x_lognorm.data).astype(np.float32, copy=False)
    adata.X = x_lognorm
    adata.layers["lognorm"] = x_lognorm.copy()
    adata.obs["total_counts"] = library_size
    adata.obs["n_genes_by_counts"] = np.asarray((counts > 0).sum(axis=1)).reshape(-1).astype(np.int32)


def select_pca_matrix(adata: ad.AnnData, use_highly_variable: bool, n_top_genes: int) -> tuple[np.ndarray, np.ndarray]:
    x = _as_dense_float32(adata.X)
    if not use_highly_variable:
        return x, np.ones(adata.n_vars, dtype=bool)

    n_top = min(int(n_top_genes), adata.n_vars)
    if n_top <= 0 or n_top >= adata.n_vars:
        return x, np.ones(adata.n_vars, dtype=bool)

    variances = np.var(x, axis=0)
    top_idx = np.argpartition(variances, -n_top)[-n_top:]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[top_idx] = True
    adata.var["highly_variable"] = mask
    return x[:, mask], mask


def compute_signal_hvg(adata: ad.AnnData, n_top_genes: int) -> np.ndarray:
    n_top = int(n_top_genes)
    if n_top <= 0 or n_top >= adata.n_vars:
        mask = np.ones(adata.n_vars, dtype=bool)
    else:
        x = _to_csr_float32(adata.X)
        variances = sparse_column_variance(x)
        top_idx = np.argpartition(variances, -n_top)[-n_top:]
        mask = np.zeros(adata.n_vars, dtype=bool)
        mask[top_idx] = True
    adata.var["highly_variable"] = mask
    adata.var["highly_variable_signal"] = mask
    return mask


def compute_pca(adata: ad.AnnData, n_pca: int, use_highly_variable: bool, n_top_genes: int) -> None:
    x, hv_mask = select_pca_matrix(adata, use_highly_variable=use_highly_variable, n_top_genes=n_top_genes)
    if "highly_variable" not in adata.var.columns:
        adata.var["highly_variable"] = hv_mask

    n_components = min(int(n_pca), max(1, adata.n_obs - 1), x.shape[1])
    if n_components <= 0:
        raise ValueError("Cannot compute PCA with the current matrix shape.")

    x_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    z = pca.fit_transform(x_scaled).astype(np.float32, copy=False)
    adata.obsm["X_pca"] = z
    adata.uns["pca"] = {
        "variance": pca.explained_variance_.astype(np.float32),
        "variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "n_components": int(n_components),
        "use_highly_variable": bool(use_highly_variable),
    }


def ensure_spatial(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    elif {"centroid_x", "centroid_y"}.issubset(adata.obs.columns):
        coords = adata.obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    elif {"array_col", "array_row"}.issubset(adata.obs.columns):
        coords = adata.obs[["array_col", "array_row"]].to_numpy(dtype=np.float32)
    else:
        raise ValueError("Input h5ad must contain obsm['spatial'] or centroid/array coordinate columns.")

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Spatial coordinates must have shape [n_cells, 2], got {coords.shape}.")
    adata.obsm["spatial"] = coords
    return coords


def main() -> None:
    args = parse_args()

    print("[1/5] loading cell-level h5ad...")
    adata = ad.read_h5ad(args.input_h5ad)
    validate_cell_level_input(adata, args.input_h5ad)
    if "counts" not in adata.layers:
        adata.layers["counts"] = _to_csr_float32(adata.X)

    print("[2/5] filtering and normalizing...")
    adata = filter_cells(
        adata,
        min_counts=float(args.min_counts),
        min_genes=int(args.min_genes),
        min_bins=int(args.min_bins),
    )
    normalize_log1p_counts(adata, target_sum=float(args.target_sum))

    print("[3/5] computing PCA and signal HVGs...")
    if args.skip_pca:
        print("      skipping PCA by request")
    else:
        compute_pca(
            adata,
            n_pca=int(args.n_pca),
            use_highly_variable=bool(args.use_highly_variable),
            n_top_genes=int(args.n_top_genes),
        )
    signal_hv_mask = compute_signal_hvg(adata, n_top_genes=int(args.signal_n_top_genes))

    print("[4/5] preparing spatial coordinates for tile-level graph construction...")
    ensure_spatial(adata)
    adata.uns["post_process"] = {
        "source_h5ad": str(args.input_h5ad),
        "target_sum": float(args.target_sum),
        "n_pca": int(args.n_pca),
        "min_counts": float(args.min_counts),
        "min_genes": int(args.min_genes),
        "min_bins": int(args.min_bins),
        "skip_pca": bool(args.skip_pca),
        "signal_n_top_genes": int(args.signal_n_top_genes),
        "n_signal_hvg": int(np.sum(signal_hv_mask)),
        "graph_scope": "signal_process.py rebuilds spatial/expression graphs inside each DIC tile",
    }
    adata.uns["signal_process_input"] = {
        "status": "prepared",
        "node_unit": "segmentation-derived pseudo-cell",
        "feature_key": "HVG" if int(args.signal_n_top_genes) != 0 else "all_genes_marked_as_HVG",
        "n_signal_hvg": int(np.sum(signal_hv_mask)),
        "spatial_key": "spatial",
        "graph_construction": "deferred_to_signal_process_tile_batches",
        "model_entrypoint": "signal_process.py",
    }

    print("[5/5] writing output...")
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_h5ad)
    print(f"output_h5ad={args.output_h5ad}")
    print(f"shape=({adata.n_obs}, {adata.n_vars}) n_signal_hvg={int(np.sum(signal_hv_mask))}")


if __name__ == "__main__":
    main()

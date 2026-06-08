from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/HDpainter1/inference/data_check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check pseudo-cell QC metrics before post_process.py filtering. "
            "Plots per-cell total counts, detected genes, and aggregated bin counts."
        )
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _counts_matrix(adata: ad.AnnData):
    return adata.layers["counts"] if "counts" in adata.layers else adata.X


def _metric_arrays(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _counts_matrix(adata)
    if sp.issparse(x):
        x = x.tocsr()
        total_counts = np.asarray(x.sum(axis=1)).reshape(-1)
        n_genes = np.asarray((x > 0).sum(axis=1)).reshape(-1)
    else:
        x = np.asarray(x)
        total_counts = np.asarray(x.sum(axis=1)).reshape(-1)
        n_genes = np.asarray((x > 0).sum(axis=1)).reshape(-1)

    if "n_bins" not in adata.obs.columns:
        raise ValueError("Input h5ad must contain obs['n_bins'] to plot bin-count distribution.")
    n_bins = pd.to_numeric(adata.obs["n_bins"], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(n_bins)):
        raise ValueError("obs['n_bins'] contains non-finite values.")

    return total_counts.astype(np.float64), n_genes.astype(np.float64), n_bins.astype(np.float64)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _hist_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot plot empty metric array.")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        return np.array([lo - 0.5, hi + 0.5], dtype=np.float64)
    return np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)


def plot_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    threshold: float,
    output_path: Path,
    n_bins: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(values, bins=_hist_edges(values, n_bins=n_bins), color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color="#D55E00", linestyle="--", linewidth=1.6, label=f"default threshold = {threshold:g}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of cells")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input_h5ad)
    total_counts, n_genes, n_bins = _metric_arrays(adata)

    prefix = args.prefix.strip()
    if not prefix:
        prefix = args.input_h5ad.stem
    output_dir = args.output_dir / prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_histogram(
        total_counts,
        title="Per-cell total counts distribution",
        xlabel="Total counts per cell",
        threshold=0.0,
        output_path=output_dir / "counts_distribution.png",
        n_bins=int(args.bins),
        dpi=int(args.dpi),
    )
    plot_histogram(
        n_genes,
        title="Per-cell detected genes distribution",
        xlabel="Detected genes per cell",
        threshold=0.0,
        output_path=output_dir / "genes_distribution.png",
        n_bins=int(args.bins),
        dpi=int(args.dpi),
    )
    plot_histogram(
        n_bins,
        title="Per-cell aggregated bins distribution",
        xlabel="Aggregated bins per cell",
        threshold=1.0,
        output_path=output_dir / "bins_distribution.png",
        n_bins=int(args.bins),
        dpi=int(args.dpi),
    )

    summary = pd.DataFrame(
        [
            {"metric": "total_counts", **_summary(total_counts)},
            {"metric": "n_genes_by_counts", **_summary(n_genes)},
            {"metric": "n_bins", **_summary(n_bins)},
        ]
    )
    summary.to_csv(output_dir / "metric_summary.csv", index=False)
    pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "total_counts": total_counts,
            "n_genes_by_counts": n_genes,
            "n_bins": n_bins,
        }
    ).to_csv(output_dir / "per_cell_qc_metrics.csv", index=False)

    print(f"output_dir={output_dir}")
    print(f"n_cells={adata.n_obs} n_genes={adata.n_vars}")
    print(f"counts_plot={output_dir / 'counts_distribution.png'}")
    print(f"genes_plot={output_dir / 'genes_distribution.png'}")
    print(f"bins_plot={output_dir / 'bins_distribution.png'}")


if __name__ == "__main__":
    main()

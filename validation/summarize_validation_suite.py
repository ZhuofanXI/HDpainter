from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize validation outputs under the HDpainter validation directory.")
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/HDpainter1"))
    parser.add_argument("--ov-root", type=Path, default=Path("/root/autodl-tmp/OV"))
    parser.add_argument("--validation-dir", type=Path, default=Path("/root/autodl-tmp/HDpainter1/validation"))
    parser.add_argument("--epoch", type=int, default=12)
    return parser.parse_args()


def sample_from_run_name(run_name: str) -> str:
    parts = run_name.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "synthHD" else run_name


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    headers = [str(x) for x in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def collect_hdpainter_training(project_root: Path, epoch: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    curve_rows = []
    size_rows = []
    for run_dir in sorted((project_root / "model" / "runs").glob("synthHD_*_nmf48_v1")):
        sample = sample_from_run_name(run_dir.name)
        path = run_dir / "training_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "sample", sample)
            df.insert(1, "run_name", run_dir.name)
            summary_rows.append(df)
        path = run_dir / "analysis" / "epoch_analysis_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "sample", sample)
            df.insert(1, "run_name", run_dir.name)
            curve_rows.append(df)
        path = run_dir / "analysis" / f"epoch_{epoch:03d}_val_size_bin_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "sample", sample)
            df.insert(1, "run_name", run_dir.name)
            df.insert(2, "epoch", int(epoch))
            size_rows.append(df)
    return (
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame(),
        pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame(),
        pd.concat(size_rows, ignore_index=True) if size_rows else pd.DataFrame(),
    )


def collect_bin2cell_baseline(validation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = validation_dir / "bin2cell_baseline"
    summary_path = base / "bin2cell_val_summary_all.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    rows = []
    for path in sorted(base.glob("*/bin2cell_val_size_bin_summary.csv")):
        sample = path.parent.name
        df = pd.read_csv(path)
        df.insert(0, "sample", sample)
        rows.append(df)
    size = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return summary, size


def collect_ov_signal(ov_root: Path) -> dict[str, pd.DataFrame]:
    base = ov_root / "hergast_like_signal_epoch012"
    paths = {
        "ov_compare_summary": base / "compare_ov_e50" / "summary_metrics.csv",
        "ov_pairwise_gene": base / "compare_ov_e50" / "pairwise_gene_metrics.csv",
        "ov_pairwise_cell": base / "compare_ov_e50" / "pairwise_cell_metrics.csv",
        "ov_signal_heldout": base / "eval_quality_e50" / "heldout_reconstruction_metrics.csv",
        "ov_marker_quality": base / "eval_quality_e50" / "marker_spatial_quality.csv",
    }
    return {key: pd.read_csv(path) if path.exists() else pd.DataFrame() for key, path in paths.items()}


def save_figures(
    out_dir: Path,
    training_summary: pd.DataFrame,
    curves: pd.DataFrame,
    hdp_size: pd.DataFrame,
    b2c_summary: pd.DataFrame,
    b2c_size: pd.DataFrame,
    ov_tables: dict[str, pd.DataFrame],
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    best = training_summary[training_summary["split"].astype(str).eq("best_val")].copy() if not training_summary.empty else pd.DataFrame()
    if not best.empty:
        merged = best[["sample", "mask_iou"]].rename(columns={"mask_iou": "HDpainter"})
        if not b2c_summary.empty:
            merged = merged.merge(b2c_summary[["sample", "mean_iou"]].rename(columns={"mean_iou": "bin2cell"}), on="sample", how="left")
        ax = merged.set_index("sample").plot(kind="bar", figsize=(7, 4))
        ax.set_ylabel("Validation IoU")
        ax.set_title("HDpainter vs bin2cell validation baseline")
        plt.tight_layout()
        plt.savefig(fig_dir / "hdpainter_vs_bin2cell_val_iou.png", dpi=180)
        plt.close()

    if not hdp_size.empty and not b2c_size.empty:
        for sample in sorted(set(hdp_size["sample"]).intersection(set(b2c_size["sample"]))):
            h = hdp_size[hdp_size["sample"].eq(sample)]
            b = b2c_size[b2c_size["sample"].eq(sample)]
            plt.figure(figsize=(7, 4))
            plt.plot(h["size_bin_id"], h["mean_mask_iou"], marker="o", label="HDpainter")
            plt.plot(b["size_bin_id"], b["mean_iou"], marker="o", label="bin2cell expansion")
            plt.xlabel("size bin")
            plt.ylabel("mean IoU")
            plt.title(f"{sample} validation IoU by size bin")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{sample}_hdpainter_vs_bin2cell_size_bin_iou.png", dpi=180)
            plt.close()

    if not curves.empty:
        val = curves[curves["split"].astype(str).eq("val")]
        for metric in ["overall_mask_iou", "overall_latent_similarity", "overall_latent_completeness", "overall_area_ratio"]:
            if metric not in val.columns:
                continue
            plt.figure(figsize=(6, 4))
            for sample, sub in val.groupby("sample"):
                plt.plot(sub["epoch"], sub[metric], marker="o", label=sample)
            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"validation_curve_{metric}.png", dpi=180)
            plt.close()

    heldout = ov_tables.get("ov_signal_heldout", pd.DataFrame())
    if not heldout.empty:
        for metric in ["positive_mse", "positive_pearson", "all_mse", "all_pearson", "all_spearman"]:
            if metric not in heldout.columns:
                continue
            plt.figure(figsize=(7, 4))
            plt.bar(heldout["dataset"], heldout[metric])
            plt.xticks(rotation=20, ha="right")
            plt.ylabel(metric)
            plt.title(f"OV held-out recovery {metric}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"ov_heldout_{metric}.png", dpi=180)
            plt.close()

    marker = ov_tables.get("ov_marker_quality", pd.DataFrame())
    if not marker.empty:
        means = marker.groupby("dataset", as_index=False)[
            ["moran_i", "neighbor_corr_pearson", "neighbor_corr_spearman", "top10_neighbor_enrichment"]
        ].mean()
        for metric in ["moran_i", "neighbor_corr_pearson", "neighbor_corr_spearman", "top10_neighbor_enrichment"]:
            plt.figure(figsize=(7, 4))
            plt.bar(means["dataset"], means[metric])
            plt.xticks(rotation=20, ha="right")
            plt.ylabel(metric)
            plt.title(f"OV marker spatial quality {metric}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"ov_marker_{metric}.png", dpi=180)
            plt.close()


def write_report(
    out_dir: Path,
    training_summary: pd.DataFrame,
    hdp_size: pd.DataFrame,
    b2c_summary: pd.DataFrame,
    b2c_size: pd.DataFrame,
    ov_tables: dict[str, pd.DataFrame],
) -> None:
    lines = ["# HDpainter Validation Suite", ""]
    best = training_summary[training_summary["split"].astype(str).eq("best_val")].copy() if not training_summary.empty else pd.DataFrame()
    lines.extend(["## Validation Tile Segmentation", ""])
    if best.empty:
        lines.append("No HDpainter validation summary found.")
    else:
        cols = ["sample", "epoch", "mask_iou", "latent_similarity", "latent_completeness", "area_ratio"]
        lines.append(markdown_table(best[cols]))

    lines.extend(["", "## bin2cell Expansion Baseline On Same Validation Tiles", ""])
    if b2c_summary.empty:
        lines.append("bin2cell baseline has not completed yet.")
    else:
        lines.append(markdown_table(b2c_summary))

    lines.extend(["", "## HDpainter vs bin2cell Validation IoU", ""])
    if not best.empty and not b2c_summary.empty:
        comp = best[["sample", "mask_iou"]].merge(
            b2c_summary[["sample", "mean_iou", "median_iou", "mean_area_ratio"]],
            on="sample",
            how="left",
        )
        comp = comp.rename(
            columns={
                "mask_iou": "hdpainter_mean_iou",
                "mean_iou": "bin2cell_mean_iou",
                "median_iou": "bin2cell_median_iou",
                "mean_area_ratio": "bin2cell_mean_area_ratio",
            }
        )
        comp["delta_mean_iou"] = comp["hdpainter_mean_iou"] - comp["bin2cell_mean_iou"]
        lines.append(markdown_table(comp))

    lines.extend(["", "## OV Signal Quality: Held-Out Spatial Recovery", ""])
    heldout = ov_tables.get("ov_signal_heldout", pd.DataFrame())
    if heldout.empty:
        lines.append("No OV held-out signal-quality table found.")
    else:
        cols = ["dataset", "layer", "positive_mse", "positive_pearson", "all_mse", "all_pearson", "all_spearman"]
        lines.append(markdown_table(heldout[[c for c in cols if c in heldout.columns]]))

    lines.extend(["", "## OV Signal Quality: Marker Spatial Metrics", ""])
    marker = ov_tables.get("ov_marker_quality", pd.DataFrame())
    if marker.empty:
        lines.append("No OV marker spatial-quality table found.")
    else:
        marker_mean = marker.groupby("dataset", as_index=False).agg(
            moran_i_mean=("moran_i", "mean"),
            neighbor_corr_pearson_mean=("neighbor_corr_pearson", "mean"),
            neighbor_corr_spearman_mean=("neighbor_corr_spearman", "mean"),
            top10_neighbor_enrichment_mean=("top10_neighbor_enrichment", "mean"),
        )
        lines.append(markdown_table(marker_mean))

    lines.extend(["", "## Notes", ""])
    lines.append("- bin2cell baseline uses bin2cell.expand_labels default max_bin_distance=2, k=4 on validation tile nucleus labels.")
    lines.append("- HDpainter IoU is model validation IoU against the same 33x33 instance target masks.")
    lines.append("- COAD/PRAD/NSCLC GNN signal-quality evaluation still requires generating their GNN e50 h5ad outputs.")
    (out_dir / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.validation_dir / "evaluation_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_summary, curves, hdp_size = collect_hdpainter_training(args.project_root, epoch=int(args.epoch))
    b2c_summary, b2c_size = collect_bin2cell_baseline(args.validation_dir)
    ov_tables = collect_ov_signal(args.ov_root)

    training_summary.to_csv(out_dir / "hdpainter_segmentation_training_summary.csv", index=False)
    curves.to_csv(out_dir / "hdpainter_segmentation_epoch_curves.csv", index=False)
    hdp_size.to_csv(out_dir / "hdpainter_segmentation_size_bin_summary.csv", index=False)
    b2c_summary.to_csv(out_dir / "bin2cell_validation_summary.csv", index=False)
    b2c_size.to_csv(out_dir / "bin2cell_validation_size_bin_summary.csv", index=False)
    for key, df in ov_tables.items():
        df.to_csv(out_dir / f"{key}.csv", index=False)

    save_figures(out_dir, training_summary, curves, hdp_size, b2c_summary, b2c_size, ov_tables)
    write_report(out_dir, training_summary, hdp_size, b2c_summary, b2c_size, ov_tables)

    ov_plot_src = args.ov_root / "hergast_like_signal_epoch012" / "eval_quality_e50" / "plots"
    ov_plot_dst = out_dir / "ov_signal_quality_plots"
    if ov_plot_src.exists():
        if ov_plot_dst.exists():
            shutil.rmtree(ov_plot_dst)
        shutil.copytree(ov_plot_src, ov_plot_dst)

    print(f"summary={out_dir / 'validation_summary.md'}")
    print(f"figures={out_dir / 'figures'}")


if __name__ == "__main__":
    main()

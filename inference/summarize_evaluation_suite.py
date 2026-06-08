from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HDpainter segmentation and signal-quality evaluations.")
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/HDpainter1"))
    parser.add_argument("--ov-root", type=Path, default=Path("/root/autodl-tmp/OV"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/HDpainter1/inference/evaluation_suite"))
    parser.add_argument("--epoch", type=int, default=12)
    return parser.parse_args()


def sample_from_run_name(run_name: str) -> str:
    parts = run_name.split("_")
    if len(parts) >= 2 and parts[0] == "synthHD":
        return parts[1]
    return run_name


def collect_training_summary(project_root: Path, epoch: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    curve_rows = []
    size_rows = []
    for run_dir in sorted((project_root / "model" / "runs").glob("synthHD_*_nmf48_v1")):
        sample = sample_from_run_name(run_dir.name)
        train_summary = run_dir / "training_summary.csv"
        if train_summary.exists():
            df = pd.read_csv(train_summary)
            df.insert(0, "sample", sample)
            df.insert(1, "run_name", run_dir.name)
            summary_rows.append(df)

        epoch_summary = run_dir / "analysis" / "epoch_analysis_summary.csv"
        if epoch_summary.exists():
            df = pd.read_csv(epoch_summary)
            df.insert(0, "sample", sample)
            df.insert(1, "run_name", run_dir.name)
            curve_rows.append(df)

        size_summary = run_dir / "analysis" / f"epoch_{epoch:03d}_val_size_bin_summary.csv"
        if size_summary.exists():
            df = pd.read_csv(size_summary)
            df.insert(0, "sample", sample)
            df.insert(1, "run_name", run_dir.name)
            df.insert(2, "epoch", int(epoch))
            size_rows.append(df)

    return (
        pd.concat(summary_rows, axis=0, ignore_index=True) if summary_rows else pd.DataFrame(),
        pd.concat(curve_rows, axis=0, ignore_index=True) if curve_rows else pd.DataFrame(),
        pd.concat(size_rows, axis=0, ignore_index=True) if size_rows else pd.DataFrame(),
    )


def collect_reference_inference(project_root: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted((project_root / "model" / "runs").glob("synthHD_*_nmf48_v1/reference_hd_inference/*_instance_analysis/instance_quality_summary.csv")):
        run_dir = csv_path.parents[2]
        sample = sample_from_run_name(run_dir.name)
        method = "stardist_id" if "stardist_id" in csv_path.as_posix() else "cellpose_id" if "cellpose_id" in csv_path.as_posix() else "unknown"
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df.insert(0, "sample", sample)
        df.insert(1, "run_name", run_dir.name)
        df.insert(2, "reference_method", method)
        df.insert(3, "source_csv", str(csv_path))
        rows.append(df)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def collect_ov_signal(ov_root: Path) -> dict[str, pd.DataFrame]:
    base = ov_root / "hergast_like_signal_epoch012"
    paths = {
        "ov_compare_summary": base / "compare_ov_e50" / "summary_metrics.csv",
        "ov_pairwise_gene": base / "compare_ov_e50" / "pairwise_gene_metrics.csv",
        "ov_pairwise_cell": base / "compare_ov_e50" / "pairwise_cell_metrics.csv",
        "ov_signal_heldout": base / "eval_quality_e50" / "heldout_reconstruction_metrics.csv",
        "ov_marker_quality": base / "eval_quality_e50" / "marker_spatial_quality.csv",
    }
    out = {}
    for key, path in paths.items():
        out[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def save_figures(training_summary: pd.DataFrame, curve: pd.DataFrame, size_bins: pd.DataFrame, ov_tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    best = training_summary[training_summary["split"].astype(str).eq("best_val")].copy() if not training_summary.empty else pd.DataFrame()
    if not best.empty:
        for metric, ylabel in [
            ("mask_iou", "Validation mask IoU"),
            ("latent_similarity", "Validation latent similarity"),
            ("latent_completeness", "Validation latent completeness"),
            ("area_ratio", "Mask / GT area"),
        ]:
            plt.figure(figsize=(6, 4))
            plt.bar(best["sample"], best[metric])
            plt.ylabel(ylabel)
            plt.xlabel("sample")
            plt.title(f"Best validation {metric}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"best_val_{metric}.png", dpi=180)
            plt.close()

    if not curve.empty:
        val = curve[curve["split"].astype(str).eq("val")].copy()
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

    if not size_bins.empty:
        for metric in ["mean_mask_iou", "mean_area_ratio", "mean_latent_similarity", "mean_latent_completeness"]:
            if metric not in size_bins.columns:
                continue
            plt.figure(figsize=(7, 4))
            for sample, sub in size_bins.groupby("sample"):
                plt.plot(sub["size_bin_id"], sub[metric], marker="o", label=sample)
            plt.xlabel("size bin")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"size_bin_{metric}.png", dpi=180)
            plt.close()

    heldout = ov_tables.get("ov_signal_heldout", pd.DataFrame())
    if not heldout.empty:
        plot_cols = ["positive_mse", "positive_pearson", "all_mse", "all_pearson", "all_spearman"]
        for metric in plot_cols:
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


def write_report(training_summary: pd.DataFrame, size_bins: pd.DataFrame, ov_tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
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

    lines = [
        "# HDpainter Evaluation Suite Summary",
        "",
        "## Segmentation Validation",
        "",
    ]
    best = training_summary[training_summary["split"].astype(str).eq("best_val")].copy() if not training_summary.empty else pd.DataFrame()
    if best.empty:
        lines.append("No training validation summary was found.")
    else:
        cols = ["sample", "epoch", "mask_iou", "latent_similarity", "latent_completeness", "area_ratio"]
        lines.append(markdown_table(best[cols]))

    lines.extend(["", "## Size-Bin Validation At Final Epoch", ""])
    if size_bins.empty:
        lines.append("No size-bin validation summaries were found.")
    else:
        agg = size_bins.groupby("sample", as_index=False).agg(
            total_val_instances=("count", "sum"),
            mean_size_bin_iou=("mean_mask_iou", "mean"),
            min_size_bin_iou=("mean_mask_iou", "min"),
            mean_area_ratio=("mean_area_ratio", "mean"),
            mean_latent_similarity=("mean_latent_similarity", "mean"),
            mean_latent_completeness=("mean_latent_completeness", "mean"),
        )
        lines.append(markdown_table(agg))

    lines.extend(["", "## OV Signal Quality: Held-Out Spatial Recovery", ""])
    heldout = ov_tables.get("ov_signal_heldout", pd.DataFrame())
    if heldout.empty:
        lines.append("No OV held-out signal-quality table was found.")
    else:
        cols = [
            "dataset",
            "layer",
            "positive_mse",
            "positive_pearson",
            "positive_spearman",
            "all_mse",
            "all_pearson",
            "all_spearman",
        ]
        lines.append(markdown_table(heldout[[c for c in cols if c in heldout.columns]]))

    lines.extend(["", "## OV Signal Quality: Marker Spatial Metrics", ""])
    marker = ov_tables.get("ov_marker_quality", pd.DataFrame())
    if marker.empty:
        lines.append("No OV marker spatial-quality table was found.")
    else:
        marker_mean = marker.groupby("dataset", as_index=False).agg(
            moran_i_mean=("moran_i", "mean"),
            neighbor_corr_pearson_mean=("neighbor_corr_pearson", "mean"),
            neighbor_corr_spearman_mean=("neighbor_corr_spearman", "mean"),
            top10_neighbor_enrichment_mean=("top10_neighbor_enrichment", "mean"),
        )
        lines.append(markdown_table(marker_mean))

    lines.extend(["", "## Current Automation Status", ""])
    lines.append("- Xenium validation tile segmentation metrics are available for COAD, PRAD, and NSCLC.")
    lines.append("- OV has complete bin2cell vs HDpainter direct vs HDpainter+GNN e50 signal-quality evaluation.")
    lines.append("- COAD/PRAD/NSCLC GNN signal-quality evaluation requires generating their post_process + signal_process outputs first.")
    (out_dir / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_summary, curve, size_bins = collect_training_summary(args.project_root, epoch=int(args.epoch))
    reference_inference = collect_reference_inference(args.project_root)
    ov_tables = collect_ov_signal(args.ov_root)

    training_summary.to_csv(args.output_dir / "segmentation_training_summary.csv", index=False)
    curve.to_csv(args.output_dir / "segmentation_epoch_curves.csv", index=False)
    size_bins.to_csv(args.output_dir / "segmentation_size_bin_summary.csv", index=False)
    reference_inference.to_csv(args.output_dir / "reference_hd_instance_quality_summary.csv", index=False)
    for key, df in ov_tables.items():
        df.to_csv(args.output_dir / f"{key}.csv", index=False)

    save_figures(training_summary, curve, size_bins, ov_tables, args.output_dir)
    write_report(training_summary, size_bins, ov_tables, args.output_dir)

    ov_plot_src = args.ov_root / "hergast_like_signal_epoch012" / "eval_quality_e50" / "plots"
    ov_plot_dst = args.output_dir / "ov_signal_quality_plots"
    if ov_plot_src.exists() and not ov_plot_dst.exists():
        shutil.copytree(ov_plot_src, ov_plot_dst)

    print(f"summary={args.output_dir / 'evaluation_summary.md'}")
    print(f"figures={args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()

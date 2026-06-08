from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

try:
    from infer_utils import (
        aggregate_to_cell_h5ad,
        build_infer_h5,
        configure_warnings,
        log,
        output_paths,
        parse_nucleus_label_cols,
        postprocess_cell_h5ad,
        prepare_h5ad,
        run_model_inference,
        signal_process_cell_h5ad,
        script_paths,
        should_stop,
        writeback_predictions,
    )
except ModuleNotFoundError as exc:
    if exc.name != "infer_utils":
        raise
    from .infer_utils import (
        aggregate_to_cell_h5ad,
        build_infer_h5,
        configure_warnings,
        log,
        output_paths,
        parse_nucleus_label_cols,
        postprocess_cell_h5ad,
        prepare_h5ad,
        run_model_inference,
        signal_process_cell_h5ad,
        script_paths,
        should_stop,
        writeback_predictions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command HDPainter real-HD inference pipeline: prepare nucleus-segmented h5ad, "
            "build inference H5, run model inference, write predictions to h5ad, aggregate to "
            "pseudo-single-cell h5ad, and run post-process graph/PCA preparation."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path("/root/autodl-tmp/OV/nucleus_segment/refer_OV_hd_nucleus_segmented.h5ad"),
        help="Bin-level h5ad produced by nucleus_segment.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained HDPainter checkpoint, e.g. model/runs/.../checkpoints/epoch_012.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/OV/hdpainter_inference"),
        help="Directory for all intermediate and final outputs.",
    )
    parser.add_argument("--run-prefix", type=str, default="refer_OV_hd")
    parser.add_argument(
        "--nucleus-label-col",
        type=str,
        default="",
        help="Deprecated single-column alias. If set, overrides --nucleus-label-cols.",
    )
    parser.add_argument(
        "--nucleus-label-cols",
        type=str,
        default="stardist_id,cellpose_id",
        help=(
            "Comma-separated nucleus label columns from nucleus_segment.py. "
            "Default runs one full inference pipeline for stardist_id and one for cellpose_id."
        ),
    )
    parser.add_argument(
        "--source-image-path",
        type=Path,
        default=None,
        help=(
            "Full-resolution H&E image used to auto-run nucleus_segment.py when requested "
            "nucleus label columns are missing from --input-h5ad."
        ),
    )
    parser.add_argument("--segmentation-out-dir", type=Path, default=None)
    parser.add_argument("--bin2cell-path", type=Path, default=None)
    parser.add_argument("--segmentation-library-id", type=str, default="Visium_HD")
    parser.add_argument("--segmentation-mpp", type=float, default=0.4)
    parser.add_argument("--segmentation-buffer", type=int, default=150)
    parser.add_argument("--segmentation-no-crop", action="store_true")
    parser.add_argument("--stardist-model", type=str, default="2D_versatile_he")
    parser.add_argument("--stardist-block-size", type=int, default=4096)
    parser.add_argument("--stardist-min-overlap", type=int, default=128)
    parser.add_argument("--stardist-context", type=int, default=128)
    parser.add_argument("--stardist-prob-thresh", type=float, default=None)
    parser.add_argument("--stardist-nms-thresh", type=float, default=None)
    parser.add_argument(
        "--nucleus-filter-min-bins",
        type=int,
        default=4,
        help="Before inference, remove nucleus labels covering fewer than this many bins.",
    )
    parser.add_argument(
        "--no-filter-small-nuclei",
        action="store_true",
        help="Disable pre-inference filtering of small nucleus labels.",
    )
    parser.add_argument("--basis-h5ad", type=Path, default=Path("/root/autodl-tmp/OV/raw_sys_cell_nmf48.h5ad"))
    parser.add_argument("--basis-varm-key", type=str, default="NMF_H_48")
    parser.add_argument("--module-csv", type=Path, default=None)
    parser.add_argument(
        "--selected-genes-path",
        type=Path,
        default=None,
        help=(
            "Optional gene list used by older build_real_hd_nmf_infer_h5.py versions. "
            "If omitted and needed, it is exported from --basis-h5ad or from --module-csv columns."
        ),
    )
    parser.add_argument("--sample-type", type=str, default="OV")
    parser.add_argument("--sample-type-id", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--min-nucleus-bins", type=int, default=4)
    parser.add_argument("--mu-iters", type=int, default=50)
    parser.add_argument("--nmf-batch-rows", type=int, default=1_600_000)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--canvas-size", type=int, default=33)
    parser.add_argument("--seed-radius", type=int, default=5)
    parser.add_argument("--neighbor-k", type=int, default=4)
    parser.add_argument("--aggregate-radius", type=int, default=5)
    parser.add_argument("--boundary-samples", type=int, default=64)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--canvas-margin", type=float, default=1.5)
    parser.add_argument("--instance-batch-limit", type=int, default=128)
    parser.add_argument("--assign-score-threshold", type=float, default=0.50)
    parser.add_argument(
        "--threshold-scan-values",
        type=str,
        default="",
        help="Optional comma-separated thresholds for embedded model inference.",
    )

    parser.add_argument("--cell-id-col", type=str, default="cell_id")
    parser.add_argument("--score-col", type=str, default="pred_score")
    parser.add_argument("--tile-col", type=str, default="pred_tile_idx")
    parser.add_argument("--coord-weight-col", type=str, default="n_counts_adjusted")
    parser.add_argument("--aggregate-min-bins", type=int, default=1)
    parser.add_argument("--aggregate-min-mean-score", type=float, default=None)

    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument("--post-min-counts", type=float, default=20.0)
    parser.add_argument("--post-min-genes", type=int, default=15)
    parser.add_argument("--post-min-bins", type=int, default=9)
    parser.add_argument("--use-highly-variable", action="store_true")
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument(
        "--signal-n-top-genes",
        type=int,
        default=0,
        help="Number of HVGs used for signal_process.py. 0 means all genes.",
    )
    parser.add_argument("--skip-signal-process", action="store_true")
    parser.add_argument("--signal-dim-reduction", choices=("PCA", "HVG", "all"), default="HVG")
    parser.add_argument("--signal-hidden-dims", type=int, nargs=2, default=(100, 32), metavar=("HIDDEN", "OUT"))
    parser.add_argument("--signal-graph-input-dim", type=int, default=64)
    parser.add_argument("--signal-epochs", type=int, default=200)
    parser.add_argument("--signal-lr", type=float, default=1e-3)
    parser.add_argument("--signal-att-drop", type=float, default=0.3)
    parser.add_argument("--signal-weight-decay", type=float, default=1e-4)
    parser.add_argument("--signal-gradient-clipping", type=float, default=5.0)
    parser.add_argument("--signal-device-idx", type=int, default=0)
    parser.add_argument("--signal-center-msg", choices=("out", "in"), default="out")
    parser.add_argument("--signal-batch-data", dest="signal_batch_data", action="store_true", default=True)
    parser.add_argument("--no-signal-batch-data", dest="signal_batch_data", action="store_false")
    parser.add_argument("--signal-num-batch-x", type=int, default=4)
    parser.add_argument("--signal-num-batch-y", type=int, default=4)
    parser.add_argument("--signal-batch-spatial-k", type=int, default=4)
    parser.add_argument("--signal-batch-expression-k", type=int, default=3)
    parser.add_argument("--signal-key-added", type=str, default="HERGAST")
    parser.add_argument("--signal-no-reconstruction", action="store_true")
    parser.add_argument("--signal-run-leiden", action="store_true")
    parser.add_argument("--signal-leiden-resolution", type=float, default=0.3)
    parser.add_argument("--signal-random-seed", type=int, default=2024)
    parser.add_argument("--signal-save-model", type=Path, default=None)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all stages even if outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full output plans and subprocess commands. By default the pipeline prints compact stage logs.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show non-critical Python warnings from AnnData/pandas. Hidden by default to keep long runs readable.",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default="signal",
        choices=["prepare", "build", "infer", "writeback", "aggregate", "postprocess", "signal"],
        help="Run until this stage, inclusive.",
    )
    return parser.parse_args()


def label_has_positive_values(adata: ad.AnnData, key: str) -> bool:
    if key not in adata.obs.columns:
        return False
    values = pd.to_numeric(adata.obs[key], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return bool(np.any(values > 0))


def missing_nucleus_label_cols(input_h5ad: Path, nucleus_label_cols: list[str]) -> list[str]:
    if not input_h5ad.exists():
        raise FileNotFoundError(f"Input h5ad does not exist: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad)
    try:
        return [col for col in nucleus_label_cols if not label_has_positive_values(adata, col)]
    finally:
        del adata


def ensure_nucleus_labels_available(args: argparse.Namespace, nucleus_label_cols: list[str]) -> bool:
    missing = missing_nucleus_label_cols(args.input_h5ad, nucleus_label_cols)
    if not missing:
        log(
            "Input h5ad already contains requested nucleus labels; skipping nucleus segmentation: "
            + ", ".join(nucleus_label_cols)
        )
        return True

    if args.source_image_path is None:
        raise ValueError(
            f"{args.input_h5ad} is missing requested nucleus label columns with positive labels: {missing}. "
            "Pass --source-image-path so infer_hd.py can run nucleus_segment.py first, "
            "or provide an input h5ad that already contains those columns."
        )
    if not args.source_image_path.exists():
        raise FileNotFoundError(args.source_image_path)

    segmentation_out_dir = args.segmentation_out_dir or (args.input_h5ad.parent / "nucleus_segment_autorun")
    script = Path(__file__).resolve().parent / "nucleus_segment.py"
    cmd = [
        sys.executable,
        str(script),
        "--input-h5ad",
        str(args.input_h5ad),
        "--source-image-path",
        str(args.source_image_path),
        "--out-dir",
        str(segmentation_out_dir),
        "--output-h5ad",
        str(args.input_h5ad),
        "--library-id",
        str(args.segmentation_library_id),
        "--mpp",
        str(args.segmentation_mpp),
        "--buffer",
        str(args.segmentation_buffer),
        "--stardist-model",
        str(args.stardist_model),
        "--stardist-block-size",
        str(args.stardist_block_size),
        "--stardist-min-overlap",
        str(args.stardist_min_overlap),
        "--stardist-context",
        str(args.stardist_context),
    ]
    if args.bin2cell_path is not None:
        cmd.extend(["--bin2cell-path", str(args.bin2cell_path)])
    if args.segmentation_no_crop:
        cmd.append("--no-crop")
    if args.stardist_prob_thresh is not None:
        cmd.extend(["--stardist-prob-thresh", str(args.stardist_prob_thresh)])
    if args.stardist_nms_thresh is not None:
        cmd.extend(["--stardist-nms-thresh", str(args.stardist_nms_thresh)])

    log(
        "Missing nucleus labels; running nucleus_segment.py before inference: "
        + ", ".join(missing)
    )
    log(" ".join(cmd))
    if args.dry_run:
        log("Dry run requested; stopping before prepare/build because labels are not present yet.")
        return False
    subprocess.run(cmd, check=True)

    still_missing = missing_nucleus_label_cols(args.input_h5ad, nucleus_label_cols)
    if still_missing:
        raise ValueError(
            f"nucleus_segment.py finished but these requested label columns are still missing/empty: {still_missing}"
        )
    log("Nucleus label preparation complete; proceeding with inference.")
    return True


def main() -> None:
    args = parse_args()
    configure_warnings(show_warnings=bool(args.show_warnings))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scripts = script_paths()
    nucleus_label_cols = parse_nucleus_label_cols(args)
    log(f"Nucleus label columns to run: {', '.join(nucleus_label_cols)}")
    if not ensure_nucleus_labels_available(args, nucleus_label_cols):
        return

    for nucleus_label_col in nucleus_label_cols:
        log("=" * 80)
        log(f"Starting pipeline for nucleus label column: {nucleus_label_col}")
        paths = output_paths(args, nucleus_label_col)

        if args.verbose:
            log("Output plan:")
            for key, path in paths.items():
                log(f"  {key}: {path}")
        else:
            log(f"Outputs will be written under: {args.output_dir}")

        prepare_h5ad(args, paths["prepared_h5ad"], nucleus_label_col)
        if should_stop(args, "prepare"):
            continue

        build_infer_h5(args, paths, scripts)
        if should_stop(args, "build"):
            continue

        run_model_inference(args, paths, scripts)
        if should_stop(args, "infer"):
            continue

        writeback_predictions(args, paths, scripts)
        if should_stop(args, "writeback"):
            continue

        aggregate_to_cell_h5ad(args, paths, scripts)
        if should_stop(args, "aggregate"):
            continue

        postprocess_cell_h5ad(args, paths, scripts)
        if should_stop(args, "postprocess"):
            log(f"Done for {nucleus_label_col}. Final postprocessed h5ad: {paths['post_h5ad']}")
            continue

        if args.skip_signal_process:
            log(f"Skipping signal process by request. Final postprocessed h5ad: {paths['post_h5ad']}")
            continue

        signal_process_cell_h5ad(args, paths, scripts)
        log(f"Done for {nucleus_label_col}. Final signal-processed h5ad: {paths['signal_h5ad']}")

    log("All requested nucleus-label pipelines finished.")


if __name__ == "__main__":
    main()

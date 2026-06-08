from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import anndata as ad
import numpy as np

from utils import (
    INFER_STEPS,
    INFERENCE_ROOT,
    OLD_CODE_ROOT,
    ROOT,
    TRAIN_FULL_STEPS,
    TRAIN_STEPS,
    _add_train_downstream_args,
    _aggregate_cells_from_binned_adata,
    _align_to_genes,
    _build_instance_chunk_manifest_h5,
    _build_raw_bin_and_cell_h5ad_from_parquet,
    _build_unioned_bin_adata,
    _build_xenium_union_cell_nmf_source,
    _collect_common_genes,
    _default_instance_chunk_manifest_path,
    _detect_train_full_resume_step,
    _ensure_hd_expanded_labels_for_degrade,
    _filter_unioned_bin_adata,
    _fit_cell_nmf,
    _gamma_poisson_degrade_bins,
    _read_gene_list,
    _require_existing_file,
    _run_module_main,
    _step_enabled,
    _validate_step_window,
    _write_h5ad_step,
    _write_json,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified preprocessing pipeline for the current HDpainter mainline. "
            "Use 'train-full' for the full raw->degrade->NMF->train-H5 chain, "
            "'train' for the already-prepared degraded/NMF intermediates, "
            "or 'infer' for reference/real HD inference packaging."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="Build a train-ready H5 from already-prepared degraded HD and cell-level NMF.")
    train.add_argument("--sim-h5ad", type=Path, required=True)
    train.add_argument("--cell-h5ad", type=Path, required=True)
    train.add_argument("--module-csv", type=Path, required=True)
    train.add_argument("--output-h5", type=Path, required=True)
    train.add_argument("--cell-latent-key", type=str, default=None)
    _add_train_downstream_args(train)

    train_full = subparsers.add_parser(
        "train-full",
        help="Run the full raw-data chain: union raw masks, align genes, gamma-poisson degrade, fit cell NMF, then build the final train H5 and all cached targets/features.",
    )
    train_full.add_argument("--raw-bin-h5ad", type=Path, default=None)
    train_full.add_argument("--raw-cell-h5ad", type=Path, default=None)
    train_full.add_argument("--reference-hd-h5ad", type=Path, required=True)
    train_full.add_argument("--raw-transcripts-parquet", type=Path, default=None)
    train_full.add_argument("--raw-nucleus-boundaries-csv", type=Path, default=None)
    train_full.add_argument("--raw-cell-boundaries-csv", type=Path, default=None)
    train_full.add_argument("--output-raw-bin-h5ad", type=Path, default=None)
    train_full.add_argument("--output-raw-cell-h5ad", type=Path, default=None)
    train_full.add_argument("--selected-genes-output-path", type=Path, default=None)
    train_full.add_argument("--raw-bin-size", type=float, default=2.0)
    train_full.add_argument("--out-dir", type=Path, default=None)
    train_full.add_argument("--output-degraded-h5ad", type=Path, default=None)
    train_full.add_argument("--output-cell-nmf-h5ad", type=Path, default=None)
    train_full.add_argument("--output-module-csv", type=Path, default=None)
    train_full.add_argument("--output-h5", type=Path, default=None)
    train_full.add_argument("--output-unioned-bin-h5ad", type=Path, default=None)
    train_full.add_argument("--output-aligned-cell-h5ad", type=Path, default=None)
    train_full.add_argument("--output-aligned-reference-h5ad", type=Path, default=None)
    train_full.add_argument("--output-manifest-json", type=Path, default=None)
    train_full.add_argument("--selected-genes-path", type=Path, default=None)
    train_full.add_argument("--bin-cell-label-key", type=str, default="cell_id")
    train_full.add_argument("--bin-cell-label-int-key", type=str, default="cell_id_int")
    train_full.add_argument("--bin-nucleus-label-key", type=str, default="nucleus_id_int")
    train_full.add_argument("--bin-x-key", type=str, default="bin_x")
    train_full.add_argument("--bin-y-key", type=str, default="bin_y")
    train_full.add_argument("--cell-id-key", type=str, default="cell_id")
    train_full.add_argument("--reference-count-key", type=str, default="n_counts_adjusted")
    train_full.add_argument("--nmf-components", type=int, default=48)
    train_full.add_argument("--nmf-max-iter", type=int, default=300)
    train_full.add_argument("--nmf-random-seed", type=int, default=42)
    train_full.add_argument("--nmf-solver", type=str, choices=("cd", "mu"), default="cd")
    train_full.add_argument(
        "--nmf-beta-loss",
        type=str,
        choices=("frobenius", "kullback-leibler", "itakura-saito"),
        default="frobenius",
    )
    train_full.add_argument("--nmf-tol", type=float, default=1e-4)
    train_full.add_argument("--nmf-alpha-w", type=float, default=0.0)
    train_full.add_argument("--nmf-alpha-h", type=str, default="same")
    train_full.add_argument("--nmf-l1-ratio", type=float, default=0.0)
    train_full.add_argument("--nmf-dense-fit", action=argparse.BooleanOptionalAction, default=True)
    train_full.add_argument("--nmf-verbose", type=int, default=1)
    train_full.add_argument(
        "--nmf-fit-source",
        type=str,
        choices=("xenium-union-cell",),
        default="xenium-union-cell",
        help="NMF fit source. Current workflow intentionally fits only on Xenium union-cell pseudo-cells.",
    )
    train_full.add_argument("--output-nmf-source-cell-h5ad", type=Path, default=None)
    train_full.add_argument("--nmf-source-h5ad", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--degrade-label-key", type=str, default="labels_he")
    train_full.add_argument("--degrade-expanded-label-key", type=str, default="labels_he_expanded")
    train_full.add_argument("--degrade-label-fallback-keys", type=str, default="stardist_id,cellpose_id")
    train_full.add_argument("--nmf-labels-key", dest="degrade_label_key", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-expanded-labels-key", dest="degrade_expanded_label_key", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-label-fallback-keys", dest="degrade_label_fallback_keys", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-qc-min-counts", type=float, default=1.0)
    train_full.add_argument("--nmf-qc-min-genes", type=int, default=3)
    train_full.add_argument("--nmf-qc-min-cells", type=int, default=3)
    train_full.add_argument("--bin2cell-path", type=Path, default=None)
    train_full.add_argument("--reference-image-path", type=Path, default=None)
    train_full.add_argument("--degrade-segmentation-out-dir", type=Path, default=None)
    train_full.add_argument("--no-degrade-auto-stardist", action="store_true")
    train_full.add_argument("--degrade-stardist-label-key", type=str, default="stardist_id")
    train_full.add_argument("--degrade-stardist-library-id", type=str, default="Visium_HD")
    train_full.add_argument("--degrade-stardist-mpp", type=float, default=0.4)
    train_full.add_argument("--degrade-stardist-buffer", type=int, default=150)
    train_full.add_argument("--degrade-stardist-no-crop", action="store_true")
    train_full.add_argument("--degrade-stardist-model", type=str, default="2D_versatile_he")
    train_full.add_argument("--degrade-stardist-block-size", type=int, default=4096)
    train_full.add_argument("--degrade-stardist-min-overlap", type=int, default=128)
    train_full.add_argument("--degrade-stardist-context", type=int, default=128)
    train_full.add_argument("--degrade-stardist-prob-thresh", type=float, default=None)
    train_full.add_argument("--degrade-stardist-nms-thresh", type=float, default=None)
    train_full.add_argument("--nmf-source-image-path", dest="reference_image_path", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-segmentation-out-dir", dest="degrade_segmentation_out_dir", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--no-nmf-auto-stardist", dest="no_degrade_auto_stardist", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-label-key", dest="degrade_stardist_label_key", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-library-id", dest="degrade_stardist_library_id", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-mpp", dest="degrade_stardist_mpp", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-buffer", dest="degrade_stardist_buffer", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-no-crop", dest="degrade_stardist_no_crop", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-model", dest="degrade_stardist_model", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-block-size", dest="degrade_stardist_block_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-min-overlap", dest="degrade_stardist_min_overlap", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-context", dest="degrade_stardist_context", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-prob-thresh", dest="degrade_stardist_prob_thresh", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--nmf-stardist-nms-thresh", dest="degrade_stardist_nms_thresh", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    train_full.add_argument("--degrade-batch-size", type=int, default=2048)
    train_full.add_argument("--degrade-random-seed", type=int, default=42)
    train_full.add_argument("--degrade-reference-label-key", type=str, default=None)
    train_full.add_argument("--degrade-theta", type=float, default=5.0)
    _add_train_downstream_args(train_full)

    infer = subparsers.add_parser("infer", help="Run the current-model real-HD inference pipeline end-to-end.")
    infer.add_argument("--input-h5ad", type=Path, required=True)
    infer.add_argument("--basis-h5ad", type=Path, required=True)
    infer.add_argument("--output-h5", type=Path, required=True, help="Builder output H5 path.")
    infer.add_argument("--output-aligned-h5ad", type=Path, required=True)
    infer.add_argument("--pred-output-h5", type=Path, required=True, help="Prediction H5 path from model inference.")
    infer.add_argument("--output-h5ad", type=Path, required=True, help="Final h5ad with obs cell_id annotations.")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--basis-varm-key", type=str, default="NMF_H_48")
    infer.add_argument("--module-csv", type=Path, default=None)
    infer.add_argument("--sample-type", type=str, default="OV")
    infer.add_argument("--sample-type-id", type=int, default=0)
    infer.add_argument("--start-from", type=str, default=None)
    infer.add_argument("--stop-after", type=str, default=None)
    infer.add_argument("--tile-patch-size", type=int, default=256)
    infer.add_argument("--tile-overlap", type=int, default=32)
    infer.add_argument("--min-nucleus-bins", type=int, default=4)
    infer.add_argument("--mu-iters", type=int, default=50)
    infer.add_argument("--eps", type=float, default=1e-6)
    infer.add_argument("--neighbor-k", type=int, default=4)
    infer.add_argument("--aggregate-radius", type=int, default=5)
    infer.add_argument("--canvas-size", type=int, default=33)
    infer.add_argument("--seed-radius", type=int, default=5)
    infer.add_argument("--boundary-samples", type=int, default=64)
    infer.add_argument("--attention-layers", type=int, default=2)
    infer.add_argument("--attention-heads", type=int, default=4)
    infer.add_argument("--canvas-margin", type=float, default=1.5)
    infer.add_argument("--instance-batch-limit", type=int, default=128)
    infer.add_argument("--assign-score-threshold", type=float, default=0.50)
    infer.add_argument("--device", type=str, default="cuda")
    infer.add_argument("--cell-id-col", type=str, default="cell_id")
    infer.add_argument("--score-col", type=str, default="pred_score")
    infer.add_argument("--tile-col", type=str, default="pred_tile_idx")
    infer.add_argument("--visualize-output-dir", type=Path, default=None)
    infer.add_argument("--visualize-num-tiles", type=int, default=3)
    infer.add_argument("--visualize-min-instances", type=int, default=300)
    infer.add_argument("--visualize-max-instances", type=int, default=600)
    infer.add_argument("--visualize-alpha-cell", type=float, default=0.30)
    return parser


def _run_train_mid_pipeline(args: argparse.Namespace) -> None:
    start_idx, stop_idx = _validate_step_window(
        start_from=getattr(args, "start_from", None),
        stop_after=getattr(args, "stop_after", None),
        ordered_steps=TRAIN_STEPS,
        mode="train",
    )

    if _step_enabled("build_h5", TRAIN_STEPS, start_idx, stop_idx):
        argv = [
            "--sim-h5ad",
            str(args.sim_h5ad),
            "--cell-h5ad",
            str(args.cell_h5ad),
            "--module-csv",
            str(args.module_csv),
            "--output-h5",
            str(args.output_h5),
            "--patch-size",
            str(args.tile_patch_size),
            "--overlap",
            str(args.tile_overlap),
            "--sample-type",
            str(args.sample_type),
            "--sample-type-id",
            str(args.sample_type_id),
            "--min-nucleus-bins",
            str(args.min_nucleus_bins),
            "--min-cell-bins",
            str(args.min_cell_bins),
            "--mu-iters",
            str(args.mu_iters),
            "--eps",
            str(args.eps),
            "--progress-every",
            str(args.progress_every),
        ]
        if getattr(args, "cell_latent_key", None) is not None:
            argv.extend(["--cell-latent-key", str(args.cell_latent_key)])
        _run_module_main("build_train_h5_from_degraded_nmf", OLD_CODE_ROOT / "build_train_h5_from_degraded_nmf.py", argv)

    if _step_enabled("filter", TRAIN_STEPS, start_idx, stop_idx) and not bool(getattr(args, "skip_filter_step", False)):
        _run_module_main(
            "filter_bad_cell",
            OLD_CODE_ROOT / "filter_bad_cell.py",
            [
                "--input-h5",
                str(args.output_h5),
                "--min-nucleus-bins",
                str(args.min_nucleus_bins),
                "--min-cell-bins",
                str(args.min_cell_bins),
                "--neighbor-threshold",
                str(args.filter_neighbor_threshold),
                "--max-fill-iters",
                str(args.filter_max_fill_iters),
                "--p-candidates",
                str(args.superellipse_p_candidates),
                "--binary-search-iters",
                str(args.binary_search_iters),
            ],
        )

    if _step_enabled("regularize", TRAIN_STEPS, start_idx, stop_idx) and bool(args.regularize_masks):
        regularize_argv = [
            "--input-h5",
            str(args.output_h5),
            "--p-candidates",
            str(args.superellipse_p_candidates),
            "--add-cosine-threshold",
            str(args.regularize_add_cosine_threshold),
            "--remove-cosine-threshold",
            str(args.regularize_remove_cosine_threshold),
            "--nmf-coverage-threshold",
            str(args.regularize_nmf_coverage_threshold),
            "--swap-max-distance",
            str(args.regularize_swap_max_distance),
            "--swap-min-gain",
            str(args.regularize_swap_min_gain),
            "--prototype-core-radius",
            str(args.regularize_prototype_core_radius),
            "--area-tolerance-frac",
            str(args.regularize_area_tolerance_frac),
            "--bridge-closing-iters",
            str(args.regularize_bridge_closing_iters),
            "--hole-area-frac",
            str(args.regularize_hole_area_frac),
            "--hole-min-area",
            str(args.regularize_hole_min_area),
            "--size-bin-edges",
            str(args.regularize_size_bin_edges),
            "--num-workers",
            str(args.regularize_num_workers),
        ]
        if args.regularize_report_dir is not None:
            regularize_argv.extend(["--report-dir", str(args.regularize_report_dir)])
        if bool(args.regularize_promote):
            regularize_argv.append("--promote-regularized")
        _run_module_main("regularize_cell_masks", OLD_CODE_ROOT / "regularize_cell_masks.py", regularize_argv)

    if _step_enabled("direction_targets", TRAIN_STEPS, start_idx, stop_idx):
        _run_module_main(
            "precompute_direction_radius_targets",
            OLD_CODE_ROOT / "precompute_direction_radius_targets.py",
            [
                "--input-h5",
                str(args.output_h5),
                "--boundary-samples",
                str(args.boundary_samples),
                "--fourier-order",
                str(args.fourier_order),
                "--p-candidates",
                str(args.superellipse_p_candidates),
                "--binary-search-iters",
                str(args.binary_search_iters),
                "--radius-quantile",
                "0.9",
                "--max-log-scale",
                str(args.max_log_scale),
                "--num-workers",
                str(args.regularize_num_workers),
            ],
        )

    if _step_enabled("microenv", TRAIN_STEPS, start_idx, stop_idx):
        _run_module_main(
            "precompute_microenv_features",
            OLD_CODE_ROOT / "precompute_microenv_features.py",
            [
                "--input-h5",
                str(args.output_h5),
                "--neighbor-k",
                str(args.neighbor_k),
                "--aggregate-radius",
                str(args.aggregate_radius),
                "--canvas-size",
                str(args.canvas_size),
                "--seed-sector-bins",
                str(args.seed_sector_bins),
                "--neighbor-direction-bins",
                str(args.neighbor_direction_bins),
                "--num-workers",
                str(args.regularize_num_workers),
            ],
        )

    if _step_enabled("instance_chunks", TRAIN_STEPS, start_idx, stop_idx):
        instance_chunk_manifest = getattr(args, "instance_chunk_manifest", None)
        if instance_chunk_manifest is None:
            output_h5_path = Path(args.output_h5)
            instance_chunk_manifest = _default_instance_chunk_manifest_path(output_h5_path, args.instance_budget)
        print("\n[preprocess] step=build_instance_chunk_manifest")
        _build_instance_chunk_manifest_h5(
            input_h5=Path(args.output_h5),
            output_h5=Path(instance_chunk_manifest),
            instance_budget=int(args.instance_budget),
            min_nucleus_bins=int(args.min_nucleus_bins),
            val_ratio=float(args.instance_chunk_val_ratio),
            split_mode=str(args.instance_chunk_split_mode),
            seed=int(args.instance_chunk_seed),
            canvas_size=int(args.canvas_size),
            neighbor_k=int(args.neighbor_k),
            aggregate_radius=int(args.aggregate_radius),
            seed_size=int(args.seed_patch_size),
            size_bin_edges=str(args.instance_chunk_size_bin_edges),
        )

    if _step_enabled("mask_refine", TRAIN_STEPS, start_idx, stop_idx):
        refined_output_h5 = getattr(args, "mask_refine_output_h5", None)
        if refined_output_h5 is None:
            source_h5 = Path(getattr(args, "instance_chunk_manifest", args.output_h5))
            refined_output_h5 = source_h5.with_name(source_h5.stem + ".maskrefined.h5")
        source_input_h5 = Path(getattr(args, "instance_chunk_manifest", args.output_h5))
        _run_module_main(
            "mask_refine",
            OLD_CODE_ROOT / "mask_refine.py",
            [
                "--input-h5",
                str(source_input_h5),
                "--output-h5",
                str(refined_output_h5),
                "--hole-min-area",
                str(args.mask_refine_hole_min_area),
                "--hole-area-frac",
                str(args.mask_refine_hole_area_frac),
                "--area-tol-frac",
                str(args.mask_refine_area_tol_frac),
                "--match-radius",
                str(args.mask_refine_match_radius),
                "--nmf-completeness-drop-tol",
                str(args.mask_refine_nmf_completeness_drop_tol),
                "--nmf-sim-drop-tol",
                str(args.mask_refine_nmf_sim_drop_tol),
                "--num-workers",
                str(args.mask_refine_num_workers),
                "--direct-block-size",
                str(args.mask_refine_direct_block_size),
                "--overwrite",
            ],
        )

    if _step_enabled("visualize", TRAIN_STEPS, start_idx, stop_idx) and args.visualize_mask_compare_dir is not None:
        _run_module_main(
            "visualize_mask_vs_ellipse",
            OLD_CODE_ROOT / "visualize_mask_vs_ellipse.py",
            [
                "--input-h5",
                str(args.output_h5),
                "--output-dir",
                str(args.visualize_mask_compare_dir),
                "--num-tiles",
                str(args.visualize_num_tiles),
                "--select-mode",
                str(args.visualize_select_mode),
                "--seed",
                str(args.visualize_seed),
            ],
        )


def run_train_pipeline(args: argparse.Namespace) -> None:
    _run_train_mid_pipeline(args)
    print(f"\n[preprocess] train pipeline done: {args.output_h5}")


def run_train_full_pipeline(args: argparse.Namespace) -> None:
    if args.out_dir is None and args.output_h5 is None:
        raise ValueError("train-full requires --out-dir or --output-h5.")
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_h5).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    output_h5 = Path(args.output_h5) if args.output_h5 is not None else out_dir / "train_ready.h5"
    if args.degrade_reference_label_key is None:
        args.degrade_reference_label_key = args.degrade_expanded_label_key
    output_raw_bin_h5ad = Path(args.output_raw_bin_h5ad) if args.output_raw_bin_h5ad is not None else out_dir / "raw_bin_from_parquet.h5ad"
    output_raw_cell_h5ad = Path(args.output_raw_cell_h5ad) if args.output_raw_cell_h5ad is not None else out_dir / "raw_cell_from_parquet.h5ad"
    selected_genes_output = Path(args.selected_genes_output_path) if args.selected_genes_output_path is not None else out_dir / "selected_genes.txt"
    unioned_path = Path(args.output_unioned_bin_h5ad) if args.output_unioned_bin_h5ad is not None else out_dir / "unioned_aligned.h5ad"
    filter_summary_path = out_dir / "prefilter_summary.json"
    aligned_cell_path = Path(args.output_aligned_cell_h5ad) if args.output_aligned_cell_h5ad is not None else out_dir / "cell_aligned.h5ad"
    aligned_reference_path = Path(args.output_aligned_reference_h5ad) if args.output_aligned_reference_h5ad is not None else out_dir / "reference_aligned.h5ad"
    degraded_h5ad_path = Path(args.output_degraded_h5ad) if args.output_degraded_h5ad is not None else out_dir / "degraded.h5ad"
    cell_nmf_h5ad_path = Path(args.output_cell_nmf_h5ad) if args.output_cell_nmf_h5ad is not None else out_dir / "cell_nmf.h5ad"
    nmf_source_cell_h5ad_path = (
        Path(args.output_nmf_source_cell_h5ad)
        if args.output_nmf_source_cell_h5ad is not None
        else out_dir / "nmf_source_cells.h5ad"
    )
    module_csv_path = (
        Path(args.output_module_csv)
        if args.output_module_csv is not None
        else out_dir / f"module_gene_weights_{int(args.nmf_components)}.csv"
    )
    module_csv_t_path = module_csv_path.with_name(module_csv_path.stem + "_T.csv")
    nmf_summary_csv_path = module_csv_path.with_name(f"nmf_summary_{int(args.nmf_components)}.csv")
    manifest_path = Path(args.output_manifest_json) if args.output_manifest_json is not None else out_dir / "train_full_manifest.json"

    if args.regularize_report_dir is None:
        args.regularize_report_dir = out_dir / "regularize_mask_report"
    if getattr(args, "instance_chunk_manifest", None) is None:
        args.instance_chunk_manifest = _default_instance_chunk_manifest_path(output_h5, args.instance_budget)

    def _remove_stale_output(path: Path | str | None) -> None:
        if path is None:
            return
        stale_path = Path(path)
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()

    latent_key = f"X_nmf_{int(args.nmf_components)}"
    manifest: dict[str, object] = {
        "mode": "train-full",
        "out_dir": str(out_dir),
        "raw_transcripts_parquet": str(args.raw_transcripts_parquet) if args.raw_transcripts_parquet is not None else None,
        "raw_nucleus_boundaries_csv": str(args.raw_nucleus_boundaries_csv) if args.raw_nucleus_boundaries_csv is not None else None,
        "raw_cell_boundaries_csv": str(args.raw_cell_boundaries_csv) if args.raw_cell_boundaries_csv is not None else None,
        "output_raw_bin_h5ad": str(output_raw_bin_h5ad),
        "output_raw_cell_h5ad": str(output_raw_cell_h5ad),
        "selected_genes_output_path": str(selected_genes_output),
        "prefilter_summary_path": str(filter_summary_path),
        "raw_bin_h5ad": str(args.raw_bin_h5ad) if args.raw_bin_h5ad is not None else None,
        "raw_cell_h5ad": str(args.raw_cell_h5ad) if args.raw_cell_h5ad is not None else None,
        "reference_hd_h5ad": str(args.reference_hd_h5ad),
        "output_unioned_bin_h5ad": str(unioned_path),
        "output_aligned_cell_h5ad": str(aligned_cell_path),
        "output_aligned_reference_h5ad": str(aligned_reference_path),
        "output_degraded_h5ad": str(degraded_h5ad_path),
        "output_cell_nmf_h5ad": str(cell_nmf_h5ad_path),
        "output_nmf_source_cell_h5ad": str(nmf_source_cell_h5ad_path),
        "output_module_csv": str(module_csv_path),
        "output_module_csv_t": str(module_csv_t_path),
        "output_nmf_summary_csv": str(nmf_summary_csv_path),
        "output_h5": str(output_h5),
        "cell_latent_key": latent_key,
        "nmf_components": int(args.nmf_components),
        "nmf_max_iter": int(args.nmf_max_iter),
        "nmf_solver": str(args.nmf_solver),
        "nmf_beta_loss": str(args.nmf_beta_loss),
        "nmf_tol": float(args.nmf_tol),
        "nmf_alpha_w": float(args.nmf_alpha_w),
        "nmf_alpha_h": str(args.nmf_alpha_h),
        "nmf_l1_ratio": float(args.nmf_l1_ratio),
        "nmf_dense_fit": bool(args.nmf_dense_fit),
        "nmf_verbose": int(args.nmf_verbose),
        "nmf_fit_source": str(args.nmf_fit_source),
        "degrade_label_key": str(args.degrade_label_key),
        "degrade_expanded_label_key": str(args.degrade_expanded_label_key),
        "degrade_reference_label_key": str(args.degrade_reference_label_key),
        "degrade_label_fallback_keys": str(args.degrade_label_fallback_keys),
        "nmf_qc_min_counts": float(args.nmf_qc_min_counts),
        "nmf_qc_min_genes": int(args.nmf_qc_min_genes),
        "nmf_qc_min_cells": int(args.nmf_qc_min_cells),
        "degrade_auto_stardist": not bool(args.no_degrade_auto_stardist),
        "reference_image_path": str(args.reference_image_path) if args.reference_image_path is not None else None,
        "degrade_segmentation_out_dir": str(args.degrade_segmentation_out_dir) if args.degrade_segmentation_out_dir is not None else None,
        "sample_type": str(args.sample_type),
        "sample_type_id": int(args.sample_type_id),
    }
    if manifest_path.exists():
        try:
            manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[preprocess] warning: failed to read existing manifest {manifest_path}: {exc}")

    have_raw_bundle = (
        args.raw_transcripts_parquet is not None
        and args.raw_nucleus_boundaries_csv is not None
        and args.raw_cell_boundaries_csv is not None
    )
    if (args.raw_bin_h5ad is None or args.raw_cell_h5ad is None) and not have_raw_bundle:
        raise ValueError(
            "train-full requires either both --raw-bin-h5ad and --raw-cell-h5ad, or the raw-ingest trio: "
            "--raw-transcripts-parquet, --raw-nucleus-boundaries-csv, --raw-cell-boundaries-csv."
        )

    raw_bin_source_path = Path(args.raw_bin_h5ad) if args.raw_bin_h5ad is not None else output_raw_bin_h5ad
    raw_cell_source_path = Path(args.raw_cell_h5ad) if args.raw_cell_h5ad is not None else output_raw_cell_h5ad
    if args.nmf_fit_source == "xenium-union-cell" and nmf_source_cell_h5ad_path.exists():
        backed = ad.read_h5ad(nmf_source_cell_h5ad_path, backed="r")
        source_kind = str(backed.uns.get("nmf_source", {}).get("source_kind", ""))
        source_genes = [str(gene) for gene in backed.var_names.tolist()]
        backed.file.close()
        if source_kind != "xenium_union_cell_qc":
            raise ValueError(
                f"Existing NMF source is not a Xenium union-cell QC source: {nmf_source_cell_h5ad_path}. "
                f"source_kind={source_kind!r}. Move or delete this stale file before resuming."
            )
        if not source_genes:
            raise ValueError(f"Existing NMF source h5ad has no genes: {nmf_source_cell_h5ad_path}")
        selected_genes_output.parent.mkdir(parents=True, exist_ok=True)
        selected_genes_output.write_text("\n".join(source_genes), encoding="utf-8")
        args.selected_genes_path = selected_genes_output
        manifest["nmf_source_completed"] = True
        manifest["selected_genes_output_path"] = str(selected_genes_output)
        _write_json(manifest_path, manifest)
        print(
            f"[preprocess] using existing Xenium union-cell NMF source | "
            f"path={nmf_source_cell_h5ad_path} genes={len(source_genes)}",
            flush=True,
        )

    auto_start_from = getattr(args, "start_from", None)
    if auto_start_from is None:
        detected_start, detected_completed = _detect_train_full_resume_step(
            raw_bin_source_path=raw_bin_source_path,
            raw_cell_source_path=raw_cell_source_path,
            unioned_path=unioned_path,
            filter_summary_path=filter_summary_path,
            aligned_cell_path=aligned_cell_path,
            aligned_reference_path=aligned_reference_path,
            degraded_h5ad_path=degraded_h5ad_path,
            cell_nmf_h5ad_path=cell_nmf_h5ad_path,
            module_csv_path=module_csv_path,
            output_h5=output_h5,
            instance_chunk_manifest_path=args.instance_chunk_manifest,
            mask_refine_output_h5=args.mask_refine_output_h5,
            visualize_dir=args.visualize_mask_compare_dir,
            regularize_masks=bool(args.regularize_masks),
        )
        if detected_start is None:
            print(f"[preprocess] all train-full outputs already exist in {out_dir}; nothing to do.")
            return
        auto_start_from = detected_start
        completed_steps = [step for step in TRAIN_FULL_STEPS if detected_completed.get(step, False)]
        manifest["completed_steps"] = completed_steps
        manifest["last_completed_step"] = completed_steps[-1] if completed_steps else None
        _write_json(manifest_path, manifest)
        print(
            f"[preprocess] auto-resume detected | out_dir={out_dir} "
            f"completed_steps={completed_steps} next_step={auto_start_from}"
        )

    requested_stop_after = getattr(args, "stop_after", None)
    if requested_stop_after is not None and auto_start_from is not None:
        if requested_stop_after in TRAIN_FULL_STEPS and auto_start_from in TRAIN_FULL_STEPS:
            if TRAIN_FULL_STEPS.index(auto_start_from) > TRAIN_FULL_STEPS.index(requested_stop_after):
                print(
                    f"[preprocess] requested stop-after '{requested_stop_after}' is already complete; "
                    f"next unfinished step is '{auto_start_from}'."
                )
                return

    start_idx, stop_idx = _validate_step_window(
        start_from=auto_start_from,
        stop_after=getattr(args, "stop_after", None),
        ordered_steps=TRAIN_FULL_STEPS,
        mode="train-full",
    )
    args.start_from = auto_start_from
    args.output_h5 = output_h5
    args.output_raw_bin_h5ad = output_raw_bin_h5ad
    args.output_raw_cell_h5ad = output_raw_cell_h5ad
    args.selected_genes_output_path = selected_genes_output
    args.output_unioned_bin_h5ad = unioned_path
    args.output_aligned_cell_h5ad = aligned_cell_path
    args.output_aligned_reference_h5ad = aligned_reference_path
    args.output_degraded_h5ad = degraded_h5ad_path
    args.output_cell_nmf_h5ad = cell_nmf_h5ad_path
    args.output_nmf_source_cell_h5ad = nmf_source_cell_h5ad_path
    args.output_module_csv = module_csv_path
    args.output_manifest_json = manifest_path
    args.raw_bin_h5ad = raw_bin_source_path
    args.raw_cell_h5ad = raw_cell_source_path
    if args.selected_genes_path is None and selected_genes_output.exists():
        args.selected_genes_path = selected_genes_output

    if _step_enabled("raw_ingest", TRAIN_FULL_STEPS, start_idx, stop_idx):
        if not have_raw_bundle:
            raise ValueError(
                "train-full raw_ingest requires --raw-transcripts-parquet, "
                "--raw-nucleus-boundaries-csv, and --raw-cell-boundaries-csv."
            )
        _build_raw_bin_and_cell_h5ad_from_parquet(
            reference_h5ad_path=Path(args.reference_hd_h5ad),
            transcripts_path=Path(args.raw_transcripts_parquet),
            cell_boundaries_path=Path(args.raw_cell_boundaries_csv),
            nucleus_boundaries_path=Path(args.raw_nucleus_boundaries_csv),
            output_raw_bin_h5ad_path=output_raw_bin_h5ad,
            output_raw_cell_h5ad_path=output_raw_cell_h5ad,
            selected_genes_path=selected_genes_output,
            bin_size=float(args.raw_bin_size),
        )
        raw_bin_source_path = output_raw_bin_h5ad
        raw_cell_source_path = output_raw_cell_h5ad
        manifest["raw_ingest_completed"] = True
        _write_json(manifest_path, manifest)
    elif start_idx > TRAIN_FULL_STEPS.index("raw_ingest"):
        if output_raw_bin_h5ad.exists():
            raw_bin_source_path = output_raw_bin_h5ad
        else:
            if args.raw_bin_h5ad is None:
                raise FileNotFoundError(
                    "raw_ingest: neither --output-raw-bin-h5ad result nor --raw-bin-h5ad is available. "
                    "Please run from --start-from raw_ingest or provide --raw-bin-h5ad."
                )
            _require_existing_file(Path(args.raw_bin_h5ad), "raw_ingest", "raw-bin h5ad or output-raw-bin-h5ad")
            raw_bin_source_path = Path(args.raw_bin_h5ad)
        if output_raw_cell_h5ad.exists():
            raw_cell_source_path = output_raw_cell_h5ad
        else:
            if args.raw_cell_h5ad is None:
                raise FileNotFoundError(
                    "raw_ingest: neither --output-raw-cell-h5ad result nor --raw-cell-h5ad is available. "
                    "Please run from --start-from raw_ingest or provide --raw-cell-h5ad."
                )
            _require_existing_file(Path(args.raw_cell_h5ad), "raw_ingest", "raw-cell h5ad or output-raw-cell-h5ad")
            raw_cell_source_path = Path(args.raw_cell_h5ad)

    if args.nmf_fit_source == "xenium-union-cell":
        rebuild_nmf_source = not nmf_source_cell_h5ad_path.exists()
        invalidated_by_nmf_source_qc = rebuild_nmf_source
        source_genes: list[str] = []
        if nmf_source_cell_h5ad_path.exists():
            _require_existing_file(nmf_source_cell_h5ad_path, "nmf_source", "Xenium union-cell NMF source h5ad")
            backed = ad.read_h5ad(nmf_source_cell_h5ad_path, backed="r")
            source_kind = str(backed.uns.get("nmf_source", {}).get("source_kind", ""))
            source_meta = dict(backed.uns.get("nmf_source", {}))
            source_genes = [str(gene) for gene in backed.var_names.tolist()]
            backed.file.close()
            if source_kind != "xenium_union_cell_qc":
                raise ValueError(
                    f"Existing NMF source is not a Xenium union-cell QC source: {nmf_source_cell_h5ad_path}. "
                    f"source_kind={source_kind!r}. Move or delete this stale file before resuming."
                )
            expected_qc = {
                "min_counts_strict_gt": float(args.nmf_qc_min_counts),
                "min_genes_strict_gt": int(args.nmf_qc_min_genes),
                "min_cells_strict_gt": int(args.nmf_qc_min_cells),
            }
            qc_mismatch = [
                key
                for key, expected in expected_qc.items()
                if source_meta.get(key) != expected
            ]
            if qc_mismatch:
                print(
                    f"[preprocess] rebuilding stale Xenium union-cell NMF source | "
                    f"path={nmf_source_cell_h5ad_path} qc_mismatch={qc_mismatch}",
                    flush=True,
                )
                nmf_source_cell_h5ad_path.unlink(missing_ok=True)
                selected_genes_output.unlink(missing_ok=True)
                source_genes = []
                rebuild_nmf_source = True
                invalidated_by_nmf_source_qc = True
        if rebuild_nmf_source:
            _require_existing_file(raw_cell_source_path, "nmf_source", "raw union-cell h5ad")
            _build_xenium_union_cell_nmf_source(
                input_cell_h5ad=raw_cell_source_path,
                output_cell_h5ad=nmf_source_cell_h5ad_path,
                selected_genes_output_path=selected_genes_output,
                min_counts=float(args.nmf_qc_min_counts),
                min_genes=int(args.nmf_qc_min_genes),
                min_cells=int(args.nmf_qc_min_cells),
            )
        if invalidated_by_nmf_source_qc:
            stale_outputs: list[Path | str | None] = [
                unioned_path,
                filter_summary_path,
                aligned_cell_path,
                aligned_reference_path,
                degraded_h5ad_path,
                cell_nmf_h5ad_path,
                module_csv_path,
                module_csv_t_path,
                nmf_summary_csv_path,
                output_h5,
                getattr(args, "instance_chunk_manifest", None),
                getattr(args, "mask_refine_output_h5", None),
            ]
            for stale_output in stale_outputs:
                _remove_stale_output(stale_output)
            align_idx = TRAIN_FULL_STEPS.index("align")
            if start_idx > align_idx:
                print(
                    "[preprocess] NMF source QC changed; rewinding current run to align so "
                    "unioned bins, filtered cells, degrade, NMF, and train H5 are regenerated.",
                    flush=True,
                )
                start_idx = align_idx
        else:
            if not selected_genes_output.exists():
                selected_genes_output.parent.mkdir(parents=True, exist_ok=True)
                selected_genes_output.write_text("\n".join(source_genes), encoding="utf-8")
        args.selected_genes_path = selected_genes_output
        manifest["nmf_source_completed"] = True
        manifest["selected_genes_output_path"] = str(selected_genes_output)
        manifest["nmf_fit_source"] = "xenium-union-cell"
        _write_json(manifest_path, manifest)

    if args.selected_genes_path is None and selected_genes_output.exists():
        args.selected_genes_path = selected_genes_output

    if _step_enabled("align", TRAIN_FULL_STEPS, start_idx, stop_idx):
        if args.selected_genes_path is None:
            raise FileNotFoundError(
                "align requires a selected gene list. Run raw_ingest first or provide --selected-genes-path."
            )
        print("[preprocess] loading raw inputs for full-chain training pipeline ...")
        raw_bin_adata = ad.read_h5ad(raw_bin_source_path)
        raw_cell_adata = ad.read_h5ad(raw_cell_source_path)
        reference_adata = ad.read_h5ad(args.reference_hd_h5ad)
        print(
            f"[preprocess] loaded raw inputs | "
            f"raw_bin=({raw_bin_adata.n_obs}, {raw_bin_adata.n_vars}) "
            f"raw_cell=({raw_cell_adata.n_obs}, {raw_cell_adata.n_vars}) "
            f"reference=({reference_adata.n_obs}, {reference_adata.n_vars})"
        )

        selected_genes = _read_gene_list(args.selected_genes_path)
        common_genes = _collect_common_genes(raw_bin_adata, raw_cell_adata, reference_adata, selected_genes)
        print(f"[preprocess] aligned full-chain common genes={len(common_genes)}")

        print("[preprocess] align step | substep=align_raw_bin_to_common_genes")
        raw_bin_adata = _align_to_genes(raw_bin_adata, common_genes)
        print(f"[preprocess] align step | substep=align_raw_bin_to_common_genes done shape=({raw_bin_adata.n_obs}, {raw_bin_adata.n_vars})")
        print("[preprocess] align step | substep=align_raw_cell_to_common_genes")
        raw_cell_adata = _align_to_genes(raw_cell_adata, common_genes)
        print(f"[preprocess] align step | substep=align_raw_cell_to_common_genes done shape=({raw_cell_adata.n_obs}, {raw_cell_adata.n_vars})")
        print("[preprocess] align step | substep=align_reference_to_common_genes")
        reference_adata = _align_to_genes(reference_adata, common_genes)
        print(f"[preprocess] align step | substep=align_reference_to_common_genes done shape=({reference_adata.n_obs}, {reference_adata.n_vars})")

        print("[preprocess] align step | substep=build_unioned_bin_adata")
        unioned_bin_adata = _build_unioned_bin_adata(
            raw_bin_adata=raw_bin_adata,
            cell_label_key=args.bin_cell_label_key,
            cell_label_int_key=args.bin_cell_label_int_key,
            nucleus_label_key=args.bin_nucleus_label_key,
            bin_x_key=args.bin_x_key,
            bin_y_key=args.bin_y_key,
        )
        print(
            f"[preprocess] unioned raw bin labels | cells={np.unique(unioned_bin_adata.obs['cell_id_int'].to_numpy())[1:].size} "
            f"nuclei={np.unique(unioned_bin_adata.obs['nucleus_id_int'].to_numpy())[1:].size}"
        )
        del raw_bin_adata
        gc.collect()

        print("[preprocess] align step | substep=write_unioned_bin_h5ad")
        _write_h5ad_step(unioned_bin_adata, unioned_path, "aligned unioned bin h5ad")
        print("[preprocess] align step | substep=write_aligned_cell_h5ad")
        _write_h5ad_step(raw_cell_adata, aligned_cell_path, "aligned raw-cell h5ad")
        print("[preprocess] align step | substep=write_aligned_reference_h5ad")
        _write_h5ad_step(reference_adata, aligned_reference_path, "aligned reference HD h5ad")
        del unioned_bin_adata
        del raw_cell_adata
        del reference_adata
        gc.collect()

        manifest["selected_gene_count"] = len(common_genes)
        manifest["selected_genes"] = common_genes
        manifest["align_completed"] = True
        _write_json(manifest_path, manifest)
    elif stop_idx >= TRAIN_FULL_STEPS.index("align"):
        _require_existing_file(unioned_path, "align", "aligned unioned bin h5ad")
        _require_existing_file(aligned_cell_path, "align", "aligned raw-cell h5ad")
        _require_existing_file(aligned_reference_path, "align", "aligned reference HD h5ad")

    if _step_enabled("filter", TRAIN_FULL_STEPS, start_idx, stop_idx):
        unioned_bin_adata = ad.read_h5ad(unioned_path)
        _require_existing_file(nmf_source_cell_h5ad_path, "filter", "QC-passed Xenium union-cell NMF source h5ad")
        nmf_source_backed = ad.read_h5ad(nmf_source_cell_h5ad_path, backed="r")
        if "cell_id" not in nmf_source_backed.obs.columns:
            nmf_source_backed.file.close()
            raise ValueError(f"NMF source h5ad is missing obs['cell_id']: {nmf_source_cell_h5ad_path}")
        valid_cell_ids = set(nmf_source_backed.obs["cell_id"].astype(str).tolist())
        nmf_source_backed.file.close()
        filtered_unioned_adata, filter_summary = _filter_unioned_bin_adata(
            unioned_bin_adata,
            min_nucleus_bins=args.min_nucleus_bins,
            min_cell_bins=args.min_cell_bins,
            valid_cell_ids=valid_cell_ids,
        )
        filtered_cell_adata = _aggregate_cells_from_binned_adata(
            filtered_unioned_adata,
            cell_id_int_key="cell_id_int",
            cell_id_key="cell_id",
        )
        print(
            f"[preprocess] prefilter unioned cells | cells_before={filter_summary['cells_before']} "
            f"cells_after={filter_summary['cells_after']} removed={filter_summary['removed_cells']} "
            f"removed_by_nmf_source_qc={filter_summary['removed_by_nmf_source_qc']} "
            f"background_bins={filter_summary['bins_reassigned_to_background']}"
        )
        _write_h5ad_step(filtered_unioned_adata, unioned_path, "prefiltered unioned bin h5ad")
        _write_h5ad_step(filtered_cell_adata, aligned_cell_path, "prefiltered cell h5ad")
        _write_json(filter_summary_path, filter_summary)
        manifest["filter_completed"] = True
        manifest["prefilter_summary"] = filter_summary
        _write_json(manifest_path, manifest)
        del unioned_bin_adata
        del filtered_unioned_adata
        del filtered_cell_adata
        gc.collect()
    elif stop_idx >= TRAIN_FULL_STEPS.index("filter"):
        _require_existing_file(filter_summary_path, "filter", "prefilter summary json")

    if _step_enabled("degrade", TRAIN_FULL_STEPS, start_idx, stop_idx):
        unioned_bin_adata = ad.read_h5ad(unioned_path)
        fallback_keys = [
            item.strip()
            for item in str(args.degrade_label_fallback_keys).split(",")
            if item.strip()
        ]
        _ensure_hd_expanded_labels_for_degrade(
            input_h5ad=aligned_reference_path,
            labels_key=str(args.degrade_label_key),
            expanded_labels_key=str(args.degrade_reference_label_key),
            fallback_label_keys=fallback_keys,
            bin2cell_path=args.bin2cell_path,
            auto_stardist=not bool(args.no_degrade_auto_stardist),
            stardist_source_image_path=args.reference_image_path,
            stardist_out_dir=args.degrade_segmentation_out_dir or (out_dir / "degrade_stardist"),
            stardist_label_key=str(args.degrade_stardist_label_key),
            stardist_library_id=str(args.degrade_stardist_library_id),
            stardist_mpp=float(args.degrade_stardist_mpp),
            stardist_buffer=int(args.degrade_stardist_buffer),
            stardist_no_crop=bool(args.degrade_stardist_no_crop),
            stardist_model=str(args.degrade_stardist_model),
            stardist_block_size=int(args.degrade_stardist_block_size),
            stardist_min_overlap=int(args.degrade_stardist_min_overlap),
            stardist_context=int(args.degrade_stardist_context),
            stardist_prob_thresh=args.degrade_stardist_prob_thresh,
            stardist_nms_thresh=args.degrade_stardist_nms_thresh,
        )
        reference_adata = ad.read_h5ad(aligned_reference_path)
        _gamma_poisson_degrade_bins(
            unioned_bin_adata=unioned_bin_adata,
            reference_adata=reference_adata,
            output_degraded_h5ad=degraded_h5ad_path,
            reference_count_key=args.reference_count_key,
            batch_size=args.degrade_batch_size,
            random_seed=args.degrade_random_seed,
            reference_label_key=str(args.degrade_reference_label_key),
            theta=float(args.degrade_theta),
        )
        manifest["degrade_completed"] = True
        _write_json(manifest_path, manifest)
    elif stop_idx >= TRAIN_FULL_STEPS.index("degrade"):
        _require_existing_file(degraded_h5ad_path, "degrade", "degraded synthetic HD h5ad")

    if _step_enabled("nmf", TRAIN_FULL_STEPS, start_idx, stop_idx):
        _require_existing_file(nmf_source_cell_h5ad_path, "nmf", "Xenium union-cell NMF source h5ad")
        nmf_source_cell_adata = ad.read_h5ad(nmf_source_cell_h5ad_path)
        latent_key = _fit_cell_nmf(
            raw_cell_adata=nmf_source_cell_adata,
            output_cell_h5ad=cell_nmf_h5ad_path,
            output_module_csv=module_csv_path,
            n_components=args.nmf_components,
            max_iter=args.nmf_max_iter,
            random_seed=args.nmf_random_seed,
            solver=args.nmf_solver,
            beta_loss=args.nmf_beta_loss,
            tol=args.nmf_tol,
            alpha_w=args.nmf_alpha_w,
            alpha_h=args.nmf_alpha_h,
            l1_ratio=args.nmf_l1_ratio,
            dense_fit=bool(args.nmf_dense_fit),
            verbose=int(args.nmf_verbose),
            cell_id_key=args.cell_id_key,
        )
        manifest["nmf_completed"] = True
        manifest["cell_latent_key"] = latent_key
        _write_json(manifest_path, manifest)
        del nmf_source_cell_adata
        gc.collect()
    elif stop_idx >= TRAIN_FULL_STEPS.index("nmf"):
        _require_existing_file(cell_nmf_h5ad_path, "nmf", "cell-level NMF h5ad")
        _require_existing_file(module_csv_path, "nmf", "module weight csv")
        _require_existing_file(module_csv_t_path, "nmf", "transposed module weight csv")
        _require_existing_file(nmf_summary_csv_path, "nmf", "NMF summary csv")
        if cell_nmf_h5ad_path.exists():
            try:
                cell_h5ad = ad.read_h5ad(cell_nmf_h5ad_path, backed="r")
                if latent_key not in cell_h5ad.obsm_keys():
                    obsm_keys = list(cell_h5ad.obsm_keys())
                    matching = [key for key in obsm_keys if key.startswith("X_nmf_")]
                    if matching:
                        latent_key = matching[0]
                cell_h5ad.file.close()
            except Exception as exc:
                print(f"[preprocess] warning: failed to inspect latent key from {cell_nmf_h5ad_path}: {exc}")

    if stop_idx >= TRAIN_FULL_STEPS.index("build_h5"):
        mid_start = args.start_from if args.start_from in TRAIN_STEPS else TRAIN_STEPS[0]
        if mid_start == "filter":
            mid_start = "regularize"
        mid_stop = args.stop_after if args.stop_after in TRAIN_STEPS else TRAIN_STEPS[-1]
        mid_args = argparse.Namespace(
            start_from=mid_start,
            stop_after=mid_stop,
            sim_h5ad=degraded_h5ad_path,
            cell_h5ad=cell_nmf_h5ad_path,
            module_csv=module_csv_path,
            output_h5=output_h5,
            cell_latent_key=latent_key,
            sample_type=args.sample_type,
            sample_type_id=args.sample_type_id,
            tile_patch_size=args.tile_patch_size,
            tile_overlap=args.tile_overlap,
            min_nucleus_bins=args.min_nucleus_bins,
            min_cell_bins=args.min_cell_bins,
            mu_iters=args.mu_iters,
            eps=args.eps,
            progress_every=args.progress_every,
            filter_neighbor_threshold=args.filter_neighbor_threshold,
            filter_max_fill_iters=args.filter_max_fill_iters,
            superellipse_p_candidates=args.superellipse_p_candidates,
            binary_search_iters=args.binary_search_iters,
            direction_bins=args.direction_bins,
            max_relative_residual=args.max_relative_residual,
            boundary_samples=args.boundary_samples,
            fourier_order=args.fourier_order,
            max_log_scale=args.max_log_scale,
            neighbor_k=args.neighbor_k,
            aggregate_radius=args.aggregate_radius,
            canvas_size=args.canvas_size,
            seed_sector_bins=args.seed_sector_bins,
            neighbor_direction_bins=args.neighbor_direction_bins,
            seed_patch_size=args.seed_patch_size,
            regularize_masks=args.regularize_masks,
            regularize_promote=args.regularize_promote,
            regularize_report_dir=args.regularize_report_dir,
            regularize_add_cosine_threshold=args.regularize_add_cosine_threshold,
            regularize_remove_cosine_threshold=args.regularize_remove_cosine_threshold,
            regularize_nmf_coverage_threshold=args.regularize_nmf_coverage_threshold,
            regularize_swap_max_distance=args.regularize_swap_max_distance,
            regularize_swap_min_gain=args.regularize_swap_min_gain,
            regularize_prototype_core_radius=args.regularize_prototype_core_radius,
            regularize_area_tolerance_frac=args.regularize_area_tolerance_frac,
            regularize_bridge_closing_iters=args.regularize_bridge_closing_iters,
            regularize_hole_area_frac=args.regularize_hole_area_frac,
            regularize_hole_min_area=args.regularize_hole_min_area,
            regularize_size_bin_edges=args.regularize_size_bin_edges,
            regularize_num_workers=args.regularize_num_workers,
            visualize_mask_compare_dir=args.visualize_mask_compare_dir,
            visualize_num_tiles=args.visualize_num_tiles,
            visualize_select_mode=args.visualize_select_mode,
            visualize_seed=args.visualize_seed,
            instance_chunk_manifest=args.instance_chunk_manifest,
            instance_budget=args.instance_budget,
            instance_chunk_val_ratio=args.instance_chunk_val_ratio,
            instance_chunk_split_mode=args.instance_chunk_split_mode,
            instance_chunk_seed=args.instance_chunk_seed,
            instance_chunk_size_bin_edges=args.instance_chunk_size_bin_edges,
            mask_refine_output_h5=args.mask_refine_output_h5,
            mask_refine_hole_min_area=args.mask_refine_hole_min_area,
            mask_refine_hole_area_frac=args.mask_refine_hole_area_frac,
            mask_refine_area_tol_frac=args.mask_refine_area_tol_frac,
            mask_refine_match_radius=args.mask_refine_match_radius,
            mask_refine_nmf_completeness_drop_tol=args.mask_refine_nmf_completeness_drop_tol,
            mask_refine_nmf_sim_drop_tol=args.mask_refine_nmf_sim_drop_tol,
            mask_refine_num_workers=args.mask_refine_num_workers,
            mask_refine_direct_block_size=args.mask_refine_direct_block_size,
            mask_refine_overwrite=args.mask_refine_overwrite,
            skip_filter_step=True,
        )
        _run_train_mid_pipeline(mid_args)

    _, completed_after = _detect_train_full_resume_step(
        raw_bin_source_path=raw_bin_source_path,
        raw_cell_source_path=raw_cell_source_path,
        unioned_path=unioned_path,
        filter_summary_path=filter_summary_path,
        aligned_cell_path=aligned_cell_path,
        aligned_reference_path=aligned_reference_path,
        degraded_h5ad_path=degraded_h5ad_path,
        cell_nmf_h5ad_path=cell_nmf_h5ad_path,
        module_csv_path=module_csv_path,
        output_h5=output_h5,
        instance_chunk_manifest_path=args.instance_chunk_manifest,
        mask_refine_output_h5=args.mask_refine_output_h5,
        visualize_dir=args.visualize_mask_compare_dir,
        regularize_masks=bool(args.regularize_masks),
    )
    completed_steps_after = [step for step in TRAIN_FULL_STEPS if completed_after.get(step, False)]
    manifest["completed_steps"] = completed_steps_after
    manifest["last_completed_step"] = completed_steps_after[-1] if completed_steps_after else None
    _write_json(manifest_path, manifest)

    print(f"\n[preprocess] train-full pipeline done: {output_h5}")


def run_infer_pipeline(args: argparse.Namespace) -> None:
    start_idx, stop_idx = _validate_step_window(
        start_from=getattr(args, "start_from", None),
        stop_after=getattr(args, "stop_after", None),
        ordered_steps=INFER_STEPS,
        mode="infer",
    )

    if _step_enabled("build_h5", INFER_STEPS, start_idx, stop_idx):
        argv = [
            "--input-h5ad",
            str(args.input_h5ad),
            "--basis-h5ad",
            str(args.basis_h5ad),
            "--output-h5",
            str(args.output_h5),
            "--output-aligned-h5ad",
            str(args.output_aligned_h5ad),
            "--patch-size",
            str(args.tile_patch_size),
            "--overlap",
            str(args.tile_overlap),
            "--sample-type",
            str(args.sample_type),
            "--sample-type-id",
            str(args.sample_type_id),
            "--min-nucleus-bins",
            str(args.min_nucleus_bins),
            "--mu-iters",
            str(args.mu_iters),
            "--eps",
            str(args.eps),
        ]
        if args.module_csv is not None:
            argv.extend(["--module-csv", str(args.module_csv)])
        if getattr(args, "basis_varm_key", None) is not None:
            argv.extend(["--basis-varm-key", str(args.basis_varm_key)])
        _run_module_main("build_real_hd_nmf_infer_h5", OLD_CODE_ROOT / "build_real_hd_nmf_infer_h5.py", argv)

    if _step_enabled("infer", INFER_STEPS, start_idx, stop_idx):
        _require_existing_file(Path(args.output_h5), "microenv", "inference H5")
        _run_module_main(
            "infer_real_hd",
            INFERENCE_ROOT / "infer_real_hd.py",
            [
                "--input-h5",
                str(args.output_h5),
                "--checkpoint",
                str(args.checkpoint),
                "--output-h5",
                str(args.pred_output_h5),
                "--device",
                str(args.device),
                "--canvas-size",
                str(args.canvas_size),
                "--seed-radius",
                str(args.seed_radius),
                "--neighbor-k",
                str(args.neighbor_k),
                "--aggregate-radius",
                str(args.aggregate_radius),
                "--boundary-samples",
                str(args.boundary_samples),
                "--attention-layers",
                str(args.attention_layers),
                "--attention-heads",
                str(args.attention_heads),
                "--canvas-margin",
                str(args.canvas_margin),
                "--instance-batch-limit",
                str(args.instance_batch_limit),
                "--assign-score-threshold",
                str(args.assign_score_threshold),
            ],
        )

    if _step_enabled("write_h5ad", INFER_STEPS, start_idx, stop_idx):
        _require_existing_file(Path(args.pred_output_h5), "write_h5ad", "prediction H5")
        _run_module_main(
            "write_real_hd_inference_to_h5ad",
            INFERENCE_ROOT / "write_real_hd_inference_to_h5ad.py",
            [
                "--source-h5ad",
                str(args.input_h5ad),
                "--pred-h5",
                str(args.pred_output_h5),
                "--output-h5ad",
                str(args.output_h5ad),
                "--cell-id-col",
                str(args.cell_id_col),
                "--score-col",
                str(args.score_col),
                "--tile-col",
                str(args.tile_col),
            ],
        )

    if _step_enabled("visualize", INFER_STEPS, start_idx, stop_idx):
        visualize_output_dir = args.visualize_output_dir
        if visualize_output_dir is None:
            visualize_output_dir = args.pred_output_h5.parent / f"{args.pred_output_h5.stem}_vis"
        _require_existing_file(Path(args.pred_output_h5), "visualize", "prediction H5")
        _run_module_main(
            "visualize_real_hd_inference",
            INFERENCE_ROOT / "visualize_real_hd_inference.py",
            [
                "--input-h5",
                str(args.output_h5),
                "--pred-h5",
                str(args.pred_output_h5),
                "--output-dir",
                str(visualize_output_dir),
                "--num-tiles",
                str(args.visualize_num_tiles),
                "--min-instances",
                str(args.visualize_min_instances),
                "--max-instances",
                str(args.visualize_max_instances),
                "--alpha-cell",
                str(args.visualize_alpha_cell),
            ],
        )

    print(f"\n[preprocess] infer pipeline done: pred_h5={args.pred_output_h5} output_h5ad={args.output_h5ad}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "train":
        run_train_pipeline(args)
    elif args.mode == "train-full":
        run_train_full_pipeline(args)
    elif args.mode == "infer":
        run_infer_pipeline(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

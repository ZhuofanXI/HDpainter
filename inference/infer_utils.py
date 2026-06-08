from __future__ import annotations

import os
import csv
import shlex
import subprocess
import sys
import warnings
from argparse import Namespace
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm


def log(message: str) -> None:
    print(f"[infer_hd] {message}", flush=True)


def configure_warnings(show_warnings: bool) -> None:
    if show_warnings:
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Variable names are not unique.*",
        category=UserWarning,
        module=r"anndata\..*",
    )


def script_paths() -> dict[str, Path]:
    inference_dir = Path(__file__).resolve().parent
    project_root = inference_dir.parent
    return {
        "builder": project_root / "preprocess" / "old_code" / "build_real_hd_nmf_infer_h5.py",
        "postprocess": inference_dir / "post_process.py",
        "signal_process": inference_dir / "signal_process.py",
    }


def parse_nucleus_label_cols(args: Namespace) -> list[str]:
    raw = args.nucleus_label_col.strip() or args.nucleus_label_cols.strip()
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one nucleus label column is required.")
    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


def safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def output_paths(args: Namespace, nucleus_label_col: str) -> dict[str, Path]:
    out = args.output_dir
    prefix = f"{args.run_prefix}_{safe_tag(nucleus_label_col)}"
    return {
        "prepared_h5ad": out / f"{prefix}_nucleus_labels_he_for_infer.h5ad",
        "infer_h5": out / f"{prefix}_nmf48_infer.h5",
        "aligned_h5ad": out / f"{prefix}_aligned_to_nmf48.h5ad",
        "pred_h5": out / f"{prefix}_pred.h5",
        "bin_pred_h5ad": out / f"{prefix}_pred_bin_level.h5ad",
        "analysis_dir": out / f"{prefix}_instance_analysis",
        "cell_h5ad": out / f"{prefix}_pred_cell_level.h5ad",
        "post_h5ad": out / f"{prefix}_pred_cell_level_postprocessed.h5ad",
        "signal_h5ad": out / f"{prefix}_pred_cell_level_signal_processed.h5ad",
    }


def maybe_skip(path: Path, force: bool, stage: str) -> bool:
    if path.exists() and not force:
        log(f"Skipping {stage}; output already exists: {path}")
        return True
    return False


def child_env() -> dict[str, str]:
    inference_dir = Path(__file__).resolve().parent
    project_root = inference_dir.parent
    extra_paths = [
        project_root / "model",
        project_root / "preprocess",
        project_root / "preprocess" / "old_code",
        inference_dir,
        project_root,
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    path_items = [str(path) for path in extra_paths if path.exists()]
    if existing:
        path_items.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_items)
    return env


def run_command(
    cmd: list[str],
    dry_run: bool = False,
    *,
    stage: str,
    verbose: bool = False,
) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    if verbose or dry_run:
        log(f"Running {stage}: {printable}")
    else:
        log(f"Running {stage}...")
    if dry_run:
        return
    project_root = Path(__file__).resolve().parent.parent
    subprocess.run(cmd, check=True, cwd=str(project_root), env=child_env())


def script_help(script_path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env(),
    )
    return result.stdout


def write_selected_genes(genes: list[str], save_path: Path, force: bool) -> Path:
    if save_path.exists() and not force:
        log(f"Using existing selected genes file: {save_path}")
        return save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(str(gene) for gene in genes) + "\n", encoding="utf-8")
    log(f"Exported selected genes file: {save_path}")
    return save_path


def export_basis_files_from_basis(args: Namespace, module_csv_path: Path, selected_genes_path: Path) -> tuple[Path, Path]:
    if module_csv_path.exists() and selected_genes_path.exists() and not args.force:
        log(f"Using existing exported module CSV: {module_csv_path}")
        log(f"Using existing exported selected genes file: {selected_genes_path}")
        return module_csv_path, selected_genes_path
    if not args.basis_h5ad.exists():
        raise FileNotFoundError(
            "This builder does not support --basis-h5ad, so infer_hd.py must convert a basis h5ad "
            "into --module-csv/--selected-genes-path first, but the basis h5ad does not exist:\n"
            f"  {args.basis_h5ad}\n"
            "Fix one of these two things:\n"
            "  1. Pass a real basis h5ad path with --basis-h5ad.\n"
            "  2. Pass an existing module CSV directly with --module-csv "
            "and, if your builder requires it, --selected-genes-path."
        )

    log(f"Exporting module CSV from basis h5ad: {args.basis_h5ad} varm['{args.basis_varm_key}']")
    basis = ad.read_h5ad(args.basis_h5ad)
    if args.basis_varm_key not in basis.varm:
        raise ValueError(f"{args.basis_h5ad} is missing varm['{args.basis_varm_key}'].")

    h_gene_module = np.asarray(basis.varm[args.basis_varm_key], dtype=np.float32)
    if h_gene_module.ndim != 2 or h_gene_module.shape[0] != basis.n_vars:
        raise ValueError(
            f"Expected varm['{args.basis_varm_key}'] shape [n_genes, n_modules], "
            f"got {h_gene_module.shape} for n_vars={basis.n_vars}."
        )

    genes = basis.var["gene"].astype(str).tolist() if "gene" in basis.var.columns else basis.var_names.astype(str).tolist()
    module_names = [f"nmf_{idx:02d}" for idx in range(h_gene_module.shape[1])]
    module_df = pd.DataFrame(h_gene_module.T, index=module_names, columns=genes)
    module_csv_path.parent.mkdir(parents=True, exist_ok=True)
    module_df.to_csv(module_csv_path)
    log(f"Exported module CSV: {module_csv_path}")
    write_selected_genes(genes, selected_genes_path, force=True)
    return module_csv_path, selected_genes_path


def selected_genes_from_module_csv(module_csv: Path, save_path: Path, force: bool) -> Path:
    if save_path.exists() and not force:
        log(f"Using existing selected genes file: {save_path}")
        return save_path
    if not module_csv.exists():
        raise FileNotFoundError(f"--module-csv does not exist: {module_csv}")
    df = pd.read_csv(module_csv, index_col=0, nrows=1)
    genes = [str(gene) for gene in df.columns.tolist()]
    if not genes:
        raise ValueError(f"Could not read gene columns from --module-csv: {module_csv}")
    return write_selected_genes(genes, save_path, force=True)


def ensure_builder_module_csv_orientation(module_csv: Path, output_dir: Path, force: bool) -> Path:
    """Ensure module CSV uses module rows and gene columns for the old builder."""
    if not module_csv.exists():
        raise FileNotFoundError(f"--module-csv does not exist: {module_csv}")

    df = pd.read_csv(module_csv, index_col=0)
    n_rows, n_cols = df.shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"--module-csv is empty or malformed: {module_csv}")

    looks_gene_by_module = n_rows > n_cols and n_cols <= 512
    if not looks_gene_by_module:
        log(f"Module CSV orientation looks builder-ready: rows={n_rows}, cols={n_cols}, path={module_csv}")
        return module_csv

    fixed_path = output_dir / f"{module_csv.stem}_builder_rows_modules.csv"
    if fixed_path.exists() and not force:
        log(
            "Module CSV appears to be gene x module; using existing transposed builder-ready CSV: "
            f"{fixed_path}"
        )
        return fixed_path

    output_dir.mkdir(parents=True, exist_ok=True)
    df.T.to_csv(fixed_path)
    log(
        "Module CSV appears to be gene x module; transposed it for the builder: "
        f"{module_csv} -> {fixed_path} (rows={n_cols}, cols={n_rows})"
    )
    return fixed_path


def filter_small_nucleus_labels(labels: np.ndarray, min_bins: int) -> tuple[np.ndarray, int, int]:
    labels = np.asarray(labels, dtype=np.int64).copy()
    positive = labels[labels > 0]
    if positive.size == 0:
        return labels.astype(np.int32), 0, 0
    counts = np.bincount(positive)
    small_label_ids = np.flatnonzero(counts < int(min_bins))
    small_label_ids = small_label_ids[small_label_ids > 0]
    if small_label_ids.size > 0:
        labels[np.isin(labels, small_label_ids)] = 0
    kept = int(np.unique(labels[labels > 0]).size)
    return labels.astype(np.int32, copy=False), int(small_label_ids.size), kept


def prepare_h5ad(args: Namespace, prepared_h5ad: Path, nucleus_label_col: str) -> None:
    if maybe_skip(prepared_h5ad, args.force, "prepare"):
        return
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Input h5ad does not exist: {args.input_h5ad}")

    log(f"Loading nucleus-segmented h5ad: {args.input_h5ad}")
    adata = ad.read_h5ad(args.input_h5ad)
    adata.var_names_make_unique()

    if nucleus_label_col not in adata.obs.columns:
        raise ValueError(
            f"{args.input_h5ad} is missing obs['{nucleus_label_col}']. "
            "Use --nucleus-label-cols stardist_id,cellpose_id or choose an existing obs column."
        )
    labels = np.asarray(adata.obs[nucleus_label_col].fillna(0), dtype=np.int32)
    n_before = int(np.unique(labels[labels > 0]).size)
    n_removed = 0
    n_after = n_before
    if not args.no_filter_small_nuclei:
        labels, n_removed, n_after = filter_small_nucleus_labels(
            labels,
            min_bins=int(args.nucleus_filter_min_bins),
        )
        log(
            f"Filtered small nuclei for {nucleus_label_col}: "
            f"before={n_before}, removed={n_removed}, kept={n_after}, "
            f"min_bins={args.nucleus_filter_min_bins}"
        )
    else:
        log(f"Small-nucleus filtering disabled for {nucleus_label_col}; nuclei={n_before}")
    adata.obs["labels_he"] = labels
    adata.uns["infer_hd_prepare"] = {
        "source_h5ad": str(args.input_h5ad),
        "nucleus_label_col": str(nucleus_label_col),
        "labels_he_source": str(nucleus_label_col),
        "filter_small_nuclei": not bool(args.no_filter_small_nuclei),
        "nucleus_filter_min_bins": int(args.nucleus_filter_min_bins),
        "nucleus_count_before_filter": int(n_before),
        "nucleus_count_removed_by_filter": int(n_removed),
        "nucleus_count_after_filter": int(n_after),
    }

    prepared_h5ad.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing builder-compatible h5ad with obs['labels_he']: {prepared_h5ad}")
    adata.write_h5ad(prepared_h5ad)


def build_infer_h5(args: Namespace, paths: dict[str, Path], scripts: dict[str, Path]) -> None:
    if maybe_skip(paths["infer_h5"], args.force, "build inference H5"):
        return
    builder_help = script_help(scripts["builder"])
    supports_basis_h5ad = "--basis-h5ad" in builder_help
    supports_basis_varm_key = "--basis-varm-key" in builder_help
    supports_module_csv = "--module-csv" in builder_help
    supports_selected_genes = "--selected-genes-path" in builder_help
    cmd = [
        sys.executable,
        str(scripts["builder"]),
        "--input-h5ad",
        str(paths["prepared_h5ad"]),
        "--output-h5",
        str(paths["infer_h5"]),
        "--output-aligned-h5ad",
        str(paths["aligned_h5ad"]),
        "--patch-size",
        str(args.patch_size),
        "--overlap",
        str(args.overlap),
        "--sample-type",
        str(args.sample_type),
        "--sample-type-id",
        str(args.sample_type_id),
        "--min-nucleus-bins",
        str(args.min_nucleus_bins),
        "--mu-iters",
        str(args.mu_iters),
        "--nmf-batch-rows",
        str(args.nmf_batch_rows),
        "--eps",
        str(args.eps),
    ]
    module_csv = args.module_csv
    selected_genes_path = args.selected_genes_path

    if supports_basis_h5ad:
        if not args.basis_h5ad.exists() and module_csv is None:
            raise FileNotFoundError(
                f"--basis-h5ad does not exist: {args.basis_h5ad}\n"
                "Pass a valid --basis-h5ad or pass --module-csv to use a module-gene weight CSV instead."
            )
        cmd.extend(["--basis-h5ad", str(args.basis_h5ad)])
    else:
        log("Builder does not support --basis-h5ad; using --module-csv/--selected-genes-path compatibility mode")
        if module_csv is None:
            module_csv, exported_genes = export_basis_files_from_basis(
                args,
                paths["prepared_h5ad"].parent / "exported_module_gene_weights_from_basis.csv",
                paths["prepared_h5ad"].parent / "exported_selected_genes_from_basis.txt",
            )
            if selected_genes_path is None:
                selected_genes_path = exported_genes
    if supports_basis_varm_key:
        cmd.extend(["--basis-varm-key", str(args.basis_varm_key)])

    if supports_module_csv:
        if module_csv is not None:
            module_csv = ensure_builder_module_csv_orientation(
                module_csv=module_csv,
                output_dir=paths["prepared_h5ad"].parent,
                force=bool(args.force),
            )
            cmd.extend(["--module-csv", str(module_csv)])
    elif module_csv is not None:
        log("Warning: builder does not support --module-csv; ignoring supplied --module-csv")

    if supports_selected_genes:
        if selected_genes_path is None and module_csv is not None:
            selected_genes_path = selected_genes_from_module_csv(
                module_csv,
                paths["prepared_h5ad"].parent / "exported_selected_genes_from_module_csv.txt",
                force=bool(args.force),
            )
        if selected_genes_path is not None:
            if not selected_genes_path.exists():
                raise FileNotFoundError(f"--selected-genes-path does not exist: {selected_genes_path}")
            cmd.extend(["--selected-genes-path", str(selected_genes_path)])
    elif selected_genes_path is not None:
        log("Warning: builder does not support --selected-genes-path; ignoring supplied --selected-genes-path")

    run_command(cmd, dry_run=args.dry_run, stage="build inference H5", verbose=bool(args.verbose))


def _ensure_model_dir_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "model"
    if model_dir.exists() and str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))


def _checkpoint_state_dict(checkpoint_path: Path, device):
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if "model_state" in payload and isinstance(payload["model_state"], dict):
            return payload["model_state"]
        return payload
    return payload


def _prepare_device(device_arg: str):
    import torch

    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_threshold_scan_values(raw: str) -> list[float]:
    if not raw.strip():
        return []
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Threshold-scan value must be in (0, 1), got {value}.")
        values.append(value)
    return sorted(set(values))


def _format_threshold_tag(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):03d}"


def _load_model_state(model, checkpoint_path: Path, device) -> None:
    state_dict = _checkpoint_state_dict(checkpoint_path, device)

    first_error: RuntimeError | None = None
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        first_error = exc
    if isinstance(state_dict, dict) and any(str(key).startswith("module.") for key in state_dict.keys()):
        stripped = {str(key)[len("module."):]: value for key, value in state_dict.items()}
        try:
            model.load_state_dict(stripped, strict=True)
            return
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_path}. "
                f"Original error: {first_error}. Stripped-module error: {exc}"
            ) from exc
    if first_error is not None:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {first_error}") from first_error
    raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: unknown state-dict format.")


def _crop_with_padding_2d(array_2d: np.ndarray, center_y: float, center_x: float, size: int) -> np.ndarray:
    cy = int(round(float(center_y)))
    cx = int(round(float(center_x)))
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size

    height, width = array_2d.shape
    out = np.zeros((size, size), dtype=array_2d.dtype)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(height, y1)
    src_x1 = min(width, x1)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = array_2d[src_y0:src_y1, src_x0:src_x1]
    return out


def _crop_with_padding_torch(tensor, center_y: float, center_x: float, size: int):
    import torch

    cy = int(round(float(center_y)))
    cx = int(round(float(center_x)))
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    _, height, width = tensor.shape
    out = torch.zeros((tensor.shape[0], size, size), dtype=tensor.dtype)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(height, y1)
    src_x1 = min(width, x1)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[:, src_y0:src_y1, src_x0:src_x1]
    return out.squeeze(0) if squeeze else out


def _build_circle_mask(radius: int):
    import torch

    size = 2 * radius + 1
    yy = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    xx = torch.arange(size, dtype=torch.float32).unsqueeze(0)
    center = float(radius)
    return (((yy - center) ** 2 + (xx - center) ** 2) <= float(radius * radius)).float()


def _aggregate_circle_nmf(expr_map, center_y: float, center_x: float, radius: int, circle_mask):
    import torch

    size = 2 * radius + 1
    crop = _crop_with_padding_torch(expr_map, center_y, center_x, size).float()
    ones = torch.ones((1, expr_map.shape[-2], expr_map.shape[-1]), dtype=torch.float32)
    valid_crop = _crop_with_padding_torch(ones, center_y, center_x, size)
    weight = circle_mask.unsqueeze(0) * valid_crop
    denom = weight.sum().clamp_min(1.0)
    return (crop * weight).sum(dim=(-1, -2)) / denom


def _build_neighbor_context_for_tile(
    centers,
    expr_map,
    nucleus_areas,
    neighbor_k: int,
    aggregate_radius: int,
    canvas_size: int,
):
    import torch

    circle_mask = _build_circle_mask(aggregate_radius)
    n_instances = int(centers.shape[0])
    feat_dim = int(expr_map.shape[0])

    seed_features = torch.zeros((n_instances, feat_dim), dtype=torch.float32)
    neighbor_seed_nmfs = torch.zeros((n_instances, neighbor_k, feat_dim), dtype=torch.float32)
    neighbor_nucleus_areas = torch.zeros((n_instances, neighbor_k, 1), dtype=torch.float32)
    neighbor_positions = torch.zeros((n_instances, neighbor_k, 3), dtype=torch.float32)
    neighbor_valid = torch.zeros((n_instances, neighbor_k), dtype=torch.float32)

    half_canvas = max(float(canvas_size // 2), 1.0)
    for inst_idx in range(n_instances):
        center_y = float(centers[inst_idx, 0].item())
        center_x = float(centers[inst_idx, 1].item())
        seed_features[inst_idx] = _aggregate_circle_nmf(expr_map, center_y, center_x, aggregate_radius, circle_mask)

        if n_instances <= 1:
            continue

        others_idx = [idx for idx in range(n_instances) if idx != inst_idx]
        others = centers[others_idx]
        deltas = others - centers[inst_idx].unsqueeze(0)
        dists = torch.sqrt(torch.clamp((deltas ** 2).sum(dim=1), min=1e-8))
        order = torch.argsort(dists)[:neighbor_k]

        for slot, other_slot in enumerate(order.tolist()):
            other_idx = others_idx[other_slot]
            other_center = centers[other_idx]
            dy = float(other_center[0].item() - center_y)
            dx = float(other_center[1].item() - center_x)
            dist = float(np.sqrt(dy * dy + dx * dx))
            neighbor_seed_nmfs[inst_idx, slot] = _aggregate_circle_nmf(
                expr_map=expr_map,
                center_y=float(other_center[0].item()),
                center_x=float(other_center[1].item()),
                radius=aggregate_radius,
                circle_mask=circle_mask,
            )
            neighbor_nucleus_areas[inst_idx, slot, 0] = nucleus_areas[other_idx, 0]
            neighbor_positions[inst_idx, slot] = torch.tensor(
                [dy / half_canvas, dx / half_canvas, dist / half_canvas],
                dtype=torch.float32,
            )
            neighbor_valid[inst_idx, slot] = 1.0

    return seed_features, neighbor_seed_nmfs, neighbor_nucleus_areas, neighbor_positions, neighbor_valid


def _assign_instances_from_scores(
    score_crops: np.ndarray,
    centers: np.ndarray,
    tile_size: int,
    canvas_size: int,
    score_threshold: float,
    global_instance_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(score_crops.shape[0])
    if count == 0:
        empty_map = np.zeros((tile_size, tile_size), dtype=np.int32)
        empty_score = np.full((tile_size, tile_size), -np.inf, dtype=np.float32)
        empty_crops = np.zeros((0, canvas_size, canvas_size), dtype=np.uint8)
        return empty_map, empty_score, empty_crops

    score_stack = np.full((count, tile_size, tile_size), -np.inf, dtype=np.float32)
    half = canvas_size // 2
    for idx in range(count):
        crop = score_crops[idx]
        cy = int(round(float(centers[idx, 0])))
        cx = int(round(float(centers[idx, 1])))
        y0 = cy - half
        x0 = cx - half
        y1 = y0 + canvas_size
        x1 = x0 + canvas_size

        src_y0 = max(0, -y0)
        src_x0 = max(0, -x0)
        src_y1 = canvas_size - max(0, y1 - tile_size)
        src_x1 = canvas_size - max(0, x1 - tile_size)
        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(tile_size, y1)
        dst_x1 = min(tile_size, x1)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            continue
        score_stack[idx, dst_y0:dst_y1, dst_x0:dst_x1] = crop[src_y0:src_y1, src_x0:src_x1]

    max_score = score_stack.max(axis=0)
    winner_local = score_stack.argmax(axis=0)
    assigned_map = np.where(max_score > score_threshold, global_instance_ids[winner_local] + 1, 0).astype(np.int32)

    assigned_crops = np.zeros((count, canvas_size, canvas_size), dtype=np.uint8)
    for idx in range(count):
        full_mask = (assigned_map == (global_instance_ids[idx] + 1)).astype(np.uint8)
        assigned_crops[idx] = _crop_with_padding_2d(full_mask, centers[idx, 0], centers[idx, 1], canvas_size)

    return assigned_map, max_score, assigned_crops


def _copy_static_dataset(fr: h5py.File, fw: h5py.File, name: str) -> None:
    if name in fr:
        fw.create_dataset(name, data=fr[name], compression="gzip" if fr[name].ndim > 0 else None)


def _source_maps_available(fr: h5py.File) -> bool:
    return all(
        name in fr
        for name in ("source_obs_index_map", "source_array_row_map", "source_array_col_map")
    )


def _reconstruct_source_maps_for_tile(
    *,
    tile_center_yx: np.ndarray,
    patch_size: int,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    source_obs_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0 = int(tile_center_yx[0]) - int(patch_size) // 2
    x0 = int(tile_center_yx[1]) - int(patch_size) // 2
    in_patch = (
        (y_coords >= y0)
        & (y_coords < y0 + patch_size)
        & (x_coords >= x0)
        & (x_coords < x0 + patch_size)
    )

    source_obs_index_map = np.full((patch_size, patch_size), -1, dtype=np.int64)
    source_array_row_map = np.full((patch_size, patch_size), -1, dtype=np.int32)
    source_array_col_map = np.full((patch_size, patch_size), -1, dtype=np.int32)
    if not np.any(in_patch):
        return source_obs_index_map, source_array_row_map, source_array_col_map

    local_y = y_coords[in_patch] - y0
    local_x = x_coords[in_patch] - x0
    source_obs_index_map[local_y, local_x] = source_obs_indices[in_patch]
    source_array_row_map[local_y, local_x] = y_coords[in_patch]
    source_array_col_map[local_y, local_x] = x_coords[in_patch]
    return source_obs_index_map, source_array_row_map, source_array_col_map


def _prepare_source_map_reconstruction(
    fr: h5py.File,
    paths: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _source_maps_available(fr):
        return None
    if "tile_center_yx" not in fr:
        raise ValueError(
            f"{paths['infer_h5']} is missing source maps and tile_center_yx, so obs-index maps cannot be reconstructed. "
            "Rebuild the inference H5 with the updated builder by rerunning infer_hd.py with --force --stop-after build."
        )

    source_h5ad = paths["aligned_h5ad"] if paths["aligned_h5ad"].exists() else paths["prepared_h5ad"]
    if not source_h5ad.exists():
        raise FileNotFoundError(
            f"{paths['infer_h5']} is missing source maps and no aligned/prepared h5ad is available for reconstruction."
        )

    log(
        "Inference H5 is missing source maps; reconstructing them from "
        f"{source_h5ad.name} and tile_center_yx."
    )
    adata = ad.read_h5ad(source_h5ad)
    if "array_row" not in adata.obs.columns or "array_col" not in adata.obs.columns:
        raise ValueError(f"{source_h5ad} must contain obs['array_row'] and obs['array_col'] to reconstruct source maps.")

    if "in_tissue" in adata.obs.columns:
        in_tissue = adata.obs["in_tissue"].to_numpy().astype(np.int32) > 0
        adata = adata[in_tissue].copy()

    y_coords = pd.to_numeric(adata.obs["array_row"], errors="coerce").to_numpy(dtype=np.int32)
    x_coords = pd.to_numeric(adata.obs["array_col"], errors="coerce").to_numpy(dtype=np.int32)
    y_coords = y_coords - int(y_coords.min())
    x_coords = x_coords - int(x_coords.min())
    source_obs_indices = np.arange(adata.n_obs, dtype=np.int64)
    return y_coords, x_coords, source_obs_indices


def _valid_prediction_h5(path: Path) -> bool:
    if not path.exists():
        return False
    required = {
        "tile_assigned_instance_map",
        "tile_assigned_score_map",
        "source_obs_index_map",
        "cell_id_pool",
        "pred_mask_prob_crop_pool",
    }
    try:
        with h5py.File(path, "r") as fr:
            return required.issubset(set(fr.keys()))
    except OSError:
        return False


def run_model_inference(args: Namespace, paths: dict[str, Path], scripts: dict[str, Path]) -> None:
    if paths["pred_h5"].exists() and not args.force:
        if _valid_prediction_h5(paths["pred_h5"]):
            log(f"Skipping model inference; output already exists: {paths['pred_h5']}")
            return
        log(f"Existing prediction H5 is incomplete or invalid; rerunning model inference: {paths['pred_h5']}")
    if args.dry_run:
        log(f"Would run embedded model inference: {paths['infer_h5']} -> {paths['pred_h5']}")
        return

    _ensure_model_dir_on_path()
    import torch
    from dataset import (
        build_center_seed_mask,
        build_condition_vector,
        build_coord_maps,
        build_multiscale_crop_sizes,
        build_multiscale_expr_crops,
        build_neighbor_maps,
    )
    from model import SizeLatentClosedRegionModel

    log("Running model inference...")
    device = _prepare_device(args.device)
    threshold_scan_values = _parse_threshold_scan_values(args.threshold_scan_values)

    with h5py.File(paths["infer_h5"], "r") as fr:
        n_tiles = int(fr.attrs["n_samples"])
        patch_size = int(fr.attrs["patch_size"])
        expr_channels = int(fr["x_low"].shape[1])
        latent_dim = int(fr.attrs["latent_dim"])
        instance_offsets = fr["instance_offsets"][:]
        total_instances = int(instance_offsets[-1])

        cond_dim = int(expr_channels * 2 + 4)
        model = SizeLatentClosedRegionModel(
            expr_channels=expr_channels,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            canvas_size=args.canvas_size,
            boundary_samples=args.boundary_samples,
            attention_layers=args.attention_layers,
            attention_heads=args.attention_heads,
            canvas_margin=args.canvas_margin,
        ).to(device)
        _load_model_state(model, args.checkpoint, device)
        model.eval()

        crop_sizes = build_multiscale_crop_sizes()
        seed_template = build_center_seed_mask(args.canvas_size, args.seed_radius)
        coord_y, coord_x = build_coord_maps(args.canvas_size)
        source_map_reconstruction = _prepare_source_map_reconstruction(fr, paths)

        paths["pred_h5"].parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(paths["pred_h5"], "w") as fw:
            fw.attrs["n_samples"] = int(n_tiles)
            fw.attrs["patch_size"] = int(patch_size)
            fw.attrs["canvas_size"] = int(args.canvas_size)
            fw.attrs["latent_dim"] = int(latent_dim)
            fw.attrs["expr_channels"] = int(expr_channels)
            fw.attrs["boundary_samples"] = int(args.boundary_samples)
            fw.attrs["checkpoint"] = str(args.checkpoint)
            fw.attrs["source_input_h5"] = str(paths["infer_h5"])
            fw.attrs["assign_score_threshold"] = float(args.assign_score_threshold)
            fw.attrs["threshold_scan_values"] = ",".join(f"{value:.2f}" for value in threshold_scan_values)

            fw.create_dataset("instance_offsets", data=instance_offsets, compression="gzip")
            _copy_static_dataset(fr, fw, "cell_centers_yx_pool")
            _copy_static_dataset(fr, fw, "nucleus_centers_yx_pool")
            _copy_static_dataset(fr, fw, "cell_id_pool")

            pred_area_ds = fw.create_dataset("pred_area_pool", shape=(total_instances,), dtype=np.float32, compression="gzip")
            pred_canvas_ds = fw.create_dataset("pred_canvas_radius_pref_pool", shape=(total_instances,), dtype=np.float32, compression="gzip")
            pred_scale_ds = fw.create_dataset("pred_scale_weights_pool", shape=(total_instances, len(crop_sizes)), dtype=np.float32, compression="gzip")
            pred_boundary_ds = fw.create_dataset("pred_boundary_radius_pool", shape=(total_instances, args.boundary_samples), dtype=np.float32, compression="gzip")
            pred_mask_prob_ds = fw.create_dataset("pred_mask_prob_crop_pool", shape=(total_instances, args.canvas_size, args.canvas_size), dtype=np.float32, compression="gzip")
            pred_mask_bin_ds = fw.create_dataset("pred_mask_binary_crop_pool", shape=(total_instances, args.canvas_size, args.canvas_size), dtype=np.uint8, compression="gzip")
            pred_mask_area_ds = fw.create_dataset("pred_mask_area_pool", shape=(total_instances,), dtype=np.float32, compression="gzip")
            pred_latent_ds = fw.create_dataset("pred_latent_pool", shape=(total_instances, latent_dim), dtype=np.float32, compression="gzip")
            tile_assign_map_ds = fw.create_dataset("tile_assigned_instance_map", shape=(n_tiles, patch_size, patch_size), dtype=np.int32, compression="gzip")
            tile_assign_score_ds = fw.create_dataset("tile_assigned_score_map", shape=(n_tiles, patch_size, patch_size), dtype=np.float32, compression="gzip")
            source_obs_index_map_ds = fw.create_dataset("source_obs_index_map", shape=(n_tiles, patch_size, patch_size), dtype=np.int64, compression="gzip")
            source_array_row_map_ds = fw.create_dataset("source_array_row_map", shape=(n_tiles, patch_size, patch_size), dtype=np.int32, compression="gzip")
            source_array_col_map_ds = fw.create_dataset("source_array_col_map", shape=(n_tiles, patch_size, patch_size), dtype=np.int32, compression="gzip")

            threshold_groups: dict[float, tuple[h5py.Dataset, h5py.Dataset]] = {}
            if threshold_scan_values:
                scan_root = fw.create_group("threshold_scan")
                for threshold in threshold_scan_values:
                    grp = scan_root.create_group(_format_threshold_tag(threshold))
                    grp.attrs["threshold"] = float(threshold)
                    assign_map_ds = grp.create_dataset("tile_assigned_instance_map", shape=(n_tiles, patch_size, patch_size), dtype=np.int32, compression="gzip")
                    assign_score_ds = grp.create_dataset("tile_assigned_score_map", shape=(n_tiles, patch_size, patch_size), dtype=np.float32, compression="gzip")
                    threshold_groups[threshold] = (assign_map_ds, assign_score_ds)

            for tile_idx in tqdm(range(n_tiles), desc="infer_hd"):
                start = int(instance_offsets[tile_idx])
                end = int(instance_offsets[tile_idx + 1])
                count = end - start

                if source_map_reconstruction is None:
                    source_obs_index_map_ds[tile_idx] = fr["source_obs_index_map"][tile_idx]
                    source_array_row_map_ds[tile_idx] = fr["source_array_row_map"][tile_idx]
                    source_array_col_map_ds[tile_idx] = fr["source_array_col_map"][tile_idx]
                else:
                    y_coords, x_coords, source_obs_indices = source_map_reconstruction
                    obs_map, row_map, col_map = _reconstruct_source_maps_for_tile(
                        tile_center_yx=np.asarray(fr["tile_center_yx"][tile_idx], dtype=np.int32),
                        patch_size=patch_size,
                        y_coords=y_coords,
                        x_coords=x_coords,
                        source_obs_indices=source_obs_indices,
                    )
                    source_obs_index_map_ds[tile_idx] = obs_map
                    source_array_row_map_ds[tile_idx] = row_map
                    source_array_col_map_ds[tile_idx] = col_map

                if count <= 0:
                    empty_map = np.zeros((patch_size, patch_size), dtype=np.int32)
                    empty_score = np.full((patch_size, patch_size), -np.inf, dtype=np.float32)
                    tile_assign_map_ds[tile_idx] = empty_map
                    tile_assign_score_ds[tile_idx] = empty_score
                    for assign_map_ds, assign_score_ds in threshold_groups.values():
                        assign_map_ds[tile_idx] = empty_map
                        assign_score_ds[tile_idx] = empty_score
                    continue

                expr_map = torch.from_numpy(fr["x_low"][tile_idx]).float()
                centers = torch.from_numpy(fr["nucleus_centers_yx_pool"][start:end]).float()
                nucleus_masks = torch.from_numpy(fr["nucleus_mask_pool"][start:end]).float()
                nucleus_areas = nucleus_masks.sum(dim=(-1, -2), keepdim=True).float()

                seed_nmfs, neighbor_seed_nmfs, neighbor_nucleus_areas, neighbor_positions, neighbor_valid = _build_neighbor_context_for_tile(
                    centers=centers,
                    expr_map=expr_map,
                    nucleus_areas=nucleus_areas,
                    neighbor_k=args.neighbor_k,
                    aggregate_radius=args.aggregate_radius,
                    canvas_size=args.canvas_size,
                )

                expr_crops = []
                seed_masks = []
                neighbor_seed_maps = []
                neighbor_distance_maps = []
                cond_vecs = []
                for inst_idx in range(count):
                    center_y = float(centers[inst_idx, 0].item())
                    center_x = float(centers[inst_idx, 1].item())
                    expr_crops.append(build_multiscale_expr_crops(expr_map, center_y, center_x, crop_sizes, args.canvas_size))
                    nseed_map, ndist_map = build_neighbor_maps(
                        neighbor_positions[inst_idx],
                        neighbor_valid[inst_idx],
                        coord_y,
                        coord_x,
                        args.canvas_size,
                        args.seed_radius,
                    )
                    cond_vecs.append(
                        build_condition_vector(
                            seed_nmfs[inst_idx],
                            nucleus_areas[inst_idx],
                            neighbor_seed_nmfs[inst_idx],
                            neighbor_nucleus_areas[inst_idx],
                            neighbor_positions[inst_idx],
                            neighbor_valid[inst_idx],
                        )
                    )
                    seed_masks.append(seed_template)
                    neighbor_seed_maps.append(nseed_map)
                    neighbor_distance_maps.append(ndist_map)

                expr_crops_tensor = torch.stack(expr_crops, dim=0)
                seed_masks_tensor = torch.stack(seed_masks, dim=0)
                neighbor_seed_maps_tensor = torch.stack(neighbor_seed_maps, dim=0)
                neighbor_distance_maps_tensor = torch.stack(neighbor_distance_maps, dim=0)
                cond_vecs_tensor = torch.stack(cond_vecs, dim=0)

                out_pred_area = np.zeros((count,), dtype=np.float32)
                out_pred_canvas = np.zeros((count,), dtype=np.float32)
                out_scale_weights = np.zeros((count, len(crop_sizes)), dtype=np.float32)
                out_boundary = np.zeros((count, args.boundary_samples), dtype=np.float32)
                out_mask_prob = np.zeros((count, args.canvas_size, args.canvas_size), dtype=np.float32)
                out_mask_area = np.zeros((count,), dtype=np.float32)
                out_latent = np.zeros((count, latent_dim), dtype=np.float32)

                for chunk_start in range(0, count, args.instance_batch_limit):
                    chunk_end = min(count, chunk_start + args.instance_batch_limit)
                    sl = slice(chunk_start, chunk_end)
                    with torch.no_grad():
                        outputs = model(
                            expr_crops=expr_crops_tensor[sl].to(device),
                            seed_mask=seed_masks_tensor[sl].to(device),
                            neighbor_seed_map=neighbor_seed_maps_tensor[sl].to(device),
                            neighbor_distance_map=neighbor_distance_maps_tensor[sl].to(device),
                            cond_vec=cond_vecs_tensor[sl].to(device),
                        )
                    out_pred_area[sl] = outputs["pred_area"].detach().cpu().numpy().reshape(-1)
                    out_pred_canvas[sl] = outputs["canvas_radius_pref"].detach().cpu().numpy().reshape(-1)
                    out_scale_weights[sl] = outputs["scale_weights"].detach().cpu().numpy()
                    out_boundary[sl] = outputs["boundary_radius"].detach().cpu().numpy()
                    out_mask_prob[sl] = outputs["mask_prob"].detach().cpu().numpy().squeeze(1)
                    out_mask_area[sl] = outputs["mask_area"].detach().cpu().numpy().reshape(-1)
                    out_latent[sl] = outputs["latent_from_mask"].detach().cpu().numpy()

                global_instance_ids = np.arange(start, end, dtype=np.int32)
                assigned_map, assigned_score, assigned_crops = _assign_instances_from_scores(
                    score_crops=out_mask_prob,
                    centers=centers.numpy(),
                    tile_size=patch_size,
                    canvas_size=args.canvas_size,
                    score_threshold=args.assign_score_threshold,
                    global_instance_ids=global_instance_ids,
                )

                pred_area_ds[start:end] = out_pred_area
                pred_canvas_ds[start:end] = out_pred_canvas
                pred_scale_ds[start:end] = out_scale_weights
                pred_boundary_ds[start:end] = out_boundary
                pred_mask_prob_ds[start:end] = out_mask_prob
                pred_mask_bin_ds[start:end] = assigned_crops
                pred_mask_area_ds[start:end] = out_mask_area
                pred_latent_ds[start:end] = out_latent
                tile_assign_map_ds[tile_idx] = assigned_map
                tile_assign_score_ds[tile_idx] = assigned_score

                for threshold, (assign_map_ds, assign_score_ds) in threshold_groups.items():
                    scan_assigned_map, scan_assigned_score, _ = _assign_instances_from_scores(
                        score_crops=out_mask_prob,
                        centers=centers.numpy(),
                        tile_size=patch_size,
                        canvas_size=args.canvas_size,
                        score_threshold=float(threshold),
                        global_instance_ids=global_instance_ids,
                    )
                    assign_map_ds[tile_idx] = scan_assigned_map
                    assign_score_ds[tile_idx] = scan_assigned_score

    log(f"Saved model inference outputs: {paths['pred_h5']}")


def _decode_cell_ids(raw: np.ndarray) -> np.ndarray:
    decoded = []
    for value in raw:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def _safe_read_vector(fr: h5py.File, name: str, length: int) -> np.ndarray:
    if name not in fr:
        return np.full(length, np.nan, dtype=np.float32)
    return np.asarray(fr[name][:], dtype=np.float32).reshape(-1)


def _compute_instance_quality_stats(fr: h5py.File) -> dict[str, np.ndarray]:
    tile_assigned_instance_map = fr["tile_assigned_instance_map"]
    tile_assigned_score_map = fr["tile_assigned_score_map"]
    cell_id_pool = _decode_cell_ids(fr["cell_id_pool"][:])
    n_instances = int(cell_id_pool.shape[0])

    assigned_bin_count = np.zeros(n_instances, dtype=np.int64)
    score_sum = np.zeros(n_instances, dtype=np.float64)
    score_max = np.full(n_instances, -np.inf, dtype=np.float32)

    for tile_idx in range(tile_assigned_instance_map.shape[0]):
        assigned_map = np.asarray(tile_assigned_instance_map[tile_idx], dtype=np.int64)
        score_map = np.asarray(tile_assigned_score_map[tile_idx], dtype=np.float32)
        valid = (assigned_map > 0) & np.isfinite(score_map)
        if not np.any(valid):
            continue
        instance_idx = assigned_map[valid] - 1
        scores = score_map[valid].astype(np.float64, copy=False)

        binc = np.bincount(instance_idx, minlength=n_instances)
        assigned_bin_count[: binc.shape[0]] += binc[:n_instances]

        weighted = np.bincount(instance_idx, weights=scores, minlength=n_instances)
        score_sum[: weighted.shape[0]] += weighted[:n_instances]
        np.maximum.at(score_max, instance_idx, score_map[valid])

    mean_score = np.divide(
        score_sum,
        assigned_bin_count,
        out=np.full(n_instances, np.nan, dtype=np.float64),
        where=assigned_bin_count > 0,
    ).astype(np.float32)
    score_max[~np.isfinite(score_max)] = np.nan

    stats: dict[str, np.ndarray] = {
        "global_instance_idx": np.arange(n_instances, dtype=np.int64),
        "cell_id": cell_id_pool,
        "assigned_bin_count": assigned_bin_count,
        "mean_assigned_score": mean_score,
        "max_assigned_score": score_max,
        "pred_area": _safe_read_vector(fr, "pred_area_pool", n_instances),
        "pred_mask_area": _safe_read_vector(fr, "pred_mask_area_pool", n_instances),
        "pred_canvas_radius_pref": _safe_read_vector(fr, "pred_canvas_radius_pref_pool", n_instances),
    }

    if "pred_scale_weights_pool" in fr:
        scale_weights = np.asarray(fr["pred_scale_weights_pool"][:], dtype=np.float32)
        for scale_idx in range(scale_weights.shape[1]):
            stats[f"scale_w_{scale_idx}"] = scale_weights[:, scale_idx]
    return stats


def _write_instance_quality_csv(save_path: Path, stats: dict[str, np.ndarray]) -> None:
    fieldnames = list(stats.keys())
    n_rows = int(len(stats["global_instance_idx"]))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(n_rows):
            row = {}
            for key in fieldnames:
                value = stats[key][row_idx]
                if isinstance(value, np.generic):
                    value = value.item()
                row[key] = value
            writer.writerow(row)


def writeback_predictions(args: Namespace, paths: dict[str, Path], scripts: dict[str, Path]) -> None:
    if maybe_skip(paths["bin_pred_h5ad"], args.force, "writeback"):
        return
    if args.dry_run:
        log(f"Would write predictions to h5ad: {paths['pred_h5']} -> {paths['bin_pred_h5ad']}")
        return

    log("Writing predictions to h5ad...")
    adata = ad.read_h5ad(paths["prepared_h5ad"])
    if "in_tissue" not in adata.obs.columns:
        raise ValueError(f"{paths['prepared_h5ad']} is missing obs['in_tissue'].")

    in_tissue_mask = adata.obs["in_tissue"].to_numpy().astype(np.int32) > 0
    filtered_to_full = np.flatnonzero(in_tissue_mask)

    with h5py.File(paths["pred_h5"], "r") as fr:
        required = [
            "tile_assigned_instance_map",
            "tile_assigned_score_map",
            "source_obs_index_map",
            "cell_id_pool",
        ]
        missing = [name for name in required if name not in fr]
        if missing:
            raise ValueError(f"{paths['pred_h5']} is missing required datasets: {missing}")

        basis = str(fr.attrs.get("source_obs_index_basis", "")).strip()
        if basis and basis != "after_in_tissue_filter":
            raise ValueError(f"Unsupported source_obs_index_basis={basis!r}.")

        tile_assigned_instance_map = fr["tile_assigned_instance_map"]
        tile_assigned_score_map = fr["tile_assigned_score_map"]
        source_obs_index_map = fr["source_obs_index_map"]
        cell_id_pool = _decode_cell_ids(fr["cell_id_pool"][:])
        n_tiles = int(tile_assigned_instance_map.shape[0])
        instance_quality_stats = _compute_instance_quality_stats(fr)

        best_score = np.full(adata.n_obs, -np.inf, dtype=np.float32)
        best_cell_id = np.empty(adata.n_obs, dtype=object)
        best_cell_id[:] = None
        best_tile_idx = np.full(adata.n_obs, -1, dtype=np.int32)

        for tile_idx in tqdm(range(n_tiles), desc="write_h5ad"):
            assigned_map = tile_assigned_instance_map[tile_idx]
            score_map = tile_assigned_score_map[tile_idx]
            obs_idx_map = source_obs_index_map[tile_idx]
            valid = (obs_idx_map >= 0) & (assigned_map > 0) & np.isfinite(score_map)
            if not np.any(valid):
                continue

            local_obs_idx = obs_idx_map[valid].astype(np.int64, copy=False)
            local_full_idx = filtered_to_full[local_obs_idx]
            local_global_instance = assigned_map[valid].astype(np.int64, copy=False) - 1
            local_score = score_map[valid].astype(np.float32, copy=False)
            better = local_score > best_score[local_full_idx]
            if not np.any(better):
                continue

            chosen_full_idx = local_full_idx[better]
            chosen_instance_idx = local_global_instance[better]
            best_score[chosen_full_idx] = local_score[better]
            best_tile_idx[chosen_full_idx] = tile_idx
            best_cell_id[chosen_full_idx] = cell_id_pool[chosen_instance_idx]

    out_cell_id = pd.Series(pd.NA, index=adata.obs_names, dtype="object")
    out_score = pd.Series(np.nan, index=adata.obs_names, dtype=np.float32)
    out_tile = pd.Series(-1, index=adata.obs_names, dtype=np.int32)

    assigned_mask = np.array([value is not None for value in best_cell_id], dtype=bool)
    if assigned_mask.any():
        out_cell_id.iloc[np.flatnonzero(assigned_mask)] = best_cell_id[assigned_mask]
        out_score.iloc[np.flatnonzero(np.isfinite(best_score))] = best_score[np.isfinite(best_score)]
        out_tile.iloc[np.flatnonzero(best_tile_idx >= 0)] = best_tile_idx[best_tile_idx >= 0]

    adata.obs[args.cell_id_col] = out_cell_id
    adata.obs[args.score_col] = out_score
    adata.obs[args.tile_col] = out_tile

    paths["bin_pred_h5ad"].parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(paths["bin_pred_h5ad"])
    _write_instance_quality_csv(paths["analysis_dir"] / "instance_quality_summary.csv", instance_quality_stats)

    assigned_count = int(assigned_mask.sum())
    log(f"Assigned bins={assigned_count}/{adata.n_obs} ({assigned_count / max(float(adata.n_obs), 1.0):.4%})")
    log(f"Wrote bin-level prediction h5ad: {paths['bin_pred_h5ad']}")


def _to_csr_float32(x) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(x, dtype=np.float32))


def _valid_cell_id_mask(values: pd.Series, keep_unassigned: bool = False) -> np.ndarray:
    text = values.astype("string").fillna("").to_numpy(dtype=object)
    valid = np.array([(item not in ("", "0", "nan", "None", "<NA>")) for item in text], dtype=bool)
    if keep_unassigned:
        valid[:] = True
    return valid


def _weighted_group_mean(
    values: np.ndarray,
    group_rows: np.ndarray,
    n_groups: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.ones(values.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    numer = np.bincount(group_rows, weights=values * weights, minlength=n_groups)
    denom = np.bincount(group_rows, weights=weights, minlength=n_groups).clip(min=1e-12)
    return (numer / denom).astype(np.float32)


def aggregate_to_cell_h5ad(args: Namespace, paths: dict[str, Path], scripts: dict[str, Path]) -> None:
    if maybe_skip(paths["cell_h5ad"], args.force, "aggregate"):
        return
    if args.dry_run:
        log(f"Would aggregate pseudo-cell h5ad: {paths['bin_pred_h5ad']} -> {paths['cell_h5ad']}")
        return

    log("Aggregating pseudo-cell h5ad...")
    adata = ad.read_h5ad(paths["bin_pred_h5ad"])
    obs = adata.obs

    if args.cell_id_col not in obs.columns:
        raise ValueError(f"{paths['bin_pred_h5ad']} is missing obs['{args.cell_id_col}'].")
    for coord_col in ("array_row", "array_col"):
        if coord_col not in obs.columns:
            raise ValueError(f"{paths['bin_pred_h5ad']} is missing obs['{coord_col}'].")

    cell_id_series = obs[args.cell_id_col].astype("string").fillna("")
    valid = _valid_cell_id_mask(cell_id_series)
    if not np.any(valid):
        raise ValueError(f"No assigned bins found in obs['{args.cell_id_col}'].")

    cell_ids = cell_id_series.to_numpy(dtype=object)
    kept_row_idx = np.flatnonzero(valid).astype(np.int64)
    kept_cell_ids = cell_ids[valid].astype(str)
    unique_cell_ids, inverse = np.unique(kept_cell_ids, return_inverse=True)
    n_cells = int(unique_cell_ids.shape[0])

    projector = sp.csr_matrix(
        (
            np.ones(inverse.shape[0], dtype=np.float32),
            (inverse.astype(np.int64), kept_row_idx),
        ),
        shape=(n_cells, adata.n_obs),
        dtype=np.float32,
    )

    X_cell = (projector @ _to_csr_float32(adata.X)).tocsr().astype(np.float32)
    bin_count = np.asarray(projector.sum(axis=1)).reshape(-1).astype(np.int32)
    total_counts = np.asarray(X_cell.sum(axis=1)).reshape(-1).astype(np.float32)
    n_genes_by_counts = np.asarray((X_cell > 0).sum(axis=1)).reshape(-1).astype(np.int32)

    row_values = pd.to_numeric(obs["array_row"], errors="coerce").to_numpy(dtype=np.float64)[valid]
    col_values = pd.to_numeric(obs["array_col"], errors="coerce").to_numpy(dtype=np.float64)[valid]

    centroid_weights = None
    if args.coord_weight_col:
        if args.coord_weight_col not in obs.columns:
            raise ValueError(f"{paths['bin_pred_h5ad']} is missing obs['{args.coord_weight_col}'].")
        centroid_weights = pd.to_numeric(obs[args.coord_weight_col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)[valid]
        centroid_weights = np.clip(centroid_weights, 0.0, None)
        if float(centroid_weights.sum()) <= 0.0:
            centroid_weights = None

    centroid_row = _weighted_group_mean(row_values, inverse, n_cells, centroid_weights)
    centroid_col = _weighted_group_mean(col_values, inverse, n_cells, centroid_weights)

    out_obs = pd.DataFrame(index=pd.Index(unique_cell_ids.astype(str), name="cell_id"))
    out_obs["cell_id"] = unique_cell_ids.astype(str)
    out_obs["n_bins"] = bin_count
    out_obs["area"] = bin_count
    out_obs["array_row"] = centroid_row
    out_obs["array_col"] = centroid_col
    out_obs["centroid_x"] = centroid_col
    out_obs["centroid_y"] = centroid_row
    out_obs["total_counts"] = total_counts
    out_obs["n_genes_by_counts"] = n_genes_by_counts

    if args.score_col in obs.columns:
        scores = pd.to_numeric(obs[args.score_col], errors="coerce").to_numpy(dtype=np.float64)[valid]
        finite = np.isfinite(scores)
        score_weight = finite.astype(np.float64)
        safe_scores = np.where(finite, scores, 0.0)
        out_obs["mean_pred_score"] = _weighted_group_mean(safe_scores, inverse, n_cells, score_weight)
        max_score = np.full(n_cells, np.nan, dtype=np.float32)
        for group_idx in range(n_cells):
            group_scores = scores[inverse == group_idx]
            group_scores = group_scores[np.isfinite(group_scores)]
            if group_scores.size > 0:
                max_score[group_idx] = float(group_scores.max())
        out_obs["max_pred_score"] = max_score

    if args.tile_col in obs.columns:
        tiles = pd.to_numeric(obs[args.tile_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)[valid]
        first_tile = np.full(n_cells, -1, dtype=np.int32)
        for group_idx in range(n_cells):
            group_tiles = tiles[inverse == group_idx]
            group_tiles = group_tiles[group_tiles >= 0]
            if group_tiles.size > 0:
                vals, counts = np.unique(group_tiles, return_counts=True)
                first_tile[group_idx] = int(vals[np.argmax(counts)])
        out_obs["majority_tile_idx"] = first_tile

    keep_cells = np.asarray(out_obs["n_bins"] >= int(args.aggregate_min_bins), dtype=bool)
    if args.aggregate_min_mean_score is not None and "mean_pred_score" in out_obs.columns:
        keep_cells &= np.asarray(out_obs["mean_pred_score"] >= float(args.aggregate_min_mean_score), dtype=bool)

    out = ad.AnnData(
        X=X_cell[keep_cells],
        obs=out_obs.loc[keep_cells].copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy(),
    )
    out.layers["counts"] = out.X.copy()
    out.obsm["spatial"] = out.obs[["array_col", "array_row"]].to_numpy(dtype=np.float32)
    out.uns["bin_aggregation"] = {
        "source_h5ad": str(paths["bin_pred_h5ad"]),
        "cell_id_col": str(args.cell_id_col),
        "min_bins": int(args.aggregate_min_bins),
        "min_mean_score": None if args.aggregate_min_mean_score is None else float(args.aggregate_min_mean_score),
        "coord_weight_col": str(args.coord_weight_col),
        "n_input_bins": int(adata.n_obs),
        "n_assigned_bins": int(valid.sum()),
        "n_cells_before_filter": int(n_cells),
        "n_cells_after_filter": int(out.n_obs),
    }

    paths["cell_h5ad"].parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(paths["cell_h5ad"])
    log(f"Wrote cell-level h5ad: {paths['cell_h5ad']} shape=({out.n_obs}, {out.n_vars})")


def postprocess_cell_h5ad(args: Namespace, paths: dict[str, Path], scripts: dict[str, Path]) -> None:
    if maybe_skip(paths["post_h5ad"], args.force, "postprocess"):
        return
    cmd = [
        sys.executable,
        str(scripts["postprocess"]),
        "--input-h5ad",
        str(paths["cell_h5ad"]),
        "--output-h5ad",
        str(paths["post_h5ad"]),
        "--target-sum",
        str(args.target_sum),
        "--n-pca",
        str(args.n_pca),
        "--min-counts",
        str(args.post_min_counts),
        "--min-genes",
        str(args.post_min_genes),
        "--min-bins",
        str(args.post_min_bins),
        "--n-top-genes",
        str(args.n_top_genes),
        "--signal-n-top-genes",
        str(args.signal_n_top_genes),
    ]
    if args.use_highly_variable:
        cmd.append("--use-highly-variable")
    run_command(cmd, dry_run=args.dry_run, stage="postprocess cell h5ad", verbose=bool(args.verbose))


def signal_process_cell_h5ad(args: Namespace, paths: dict[str, Path], scripts: dict[str, Path]) -> None:
    if maybe_skip(paths["signal_h5ad"], args.force, "signal process"):
        return
    cmd = [
        sys.executable,
        str(scripts["signal_process"]),
        "--input-h5ad",
        str(paths["post_h5ad"]),
        "--output-h5ad",
        str(paths["signal_h5ad"]),
        "--dim-reduction",
        str(args.signal_dim_reduction),
        "--hidden-dims",
        str(args.signal_hidden_dims[0]),
        str(args.signal_hidden_dims[1]),
        "--graph-input-dim",
        str(args.signal_graph_input_dim),
        "--epochs",
        str(args.signal_epochs),
        "--lr",
        str(args.signal_lr),
        "--att-drop",
        str(args.signal_att_drop),
        "--weight-decay",
        str(args.signal_weight_decay),
        "--gradient-clipping",
        str(args.signal_gradient_clipping),
        "--device-idx",
        str(args.signal_device_idx),
        "--center-msg",
        str(args.signal_center_msg),
        "--key-added",
        str(args.signal_key_added),
        "--random-seed",
        str(args.signal_random_seed),
    ]
    if args.signal_batch_data:
        cmd.append("--batch-data")
        if int(args.signal_num_batch_x) > 0:
            cmd.extend(["--num-batch-x", str(args.signal_num_batch_x)])
        if int(args.signal_num_batch_y) > 0:
            cmd.extend(["--num-batch-y", str(args.signal_num_batch_y)])
        cmd.extend(["--batch-spatial-k", str(args.signal_batch_spatial_k)])
        cmd.extend(["--batch-expression-k", str(args.signal_batch_expression_k)])
    else:
        cmd.append("--no-batch-data")
    if args.signal_no_reconstruction:
        cmd.append("--no-save-reconstruction")
    else:
        cmd.append("--save-reconstruction")
    if args.signal_run_leiden:
        cmd.extend(["--run-leiden", "--leiden-resolution", str(args.signal_leiden_resolution)])
    if args.signal_save_model is not None:
        cmd.extend(["--save-model", str(args.signal_save_model)])
    run_command(cmd, dry_run=args.dry_run, stage="HERGAST-like signal process", verbose=bool(args.verbose))


def should_stop(args: Namespace, stage: str) -> bool:
    order = ["prepare", "build", "infer", "writeback", "aggregate", "postprocess", "signal"]
    return order.index(stage) >= order.index(args.stop_after)

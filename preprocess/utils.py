from __future__ import annotations

import argparse
import gc
import h5py
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cv2

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


ROOT = Path(__file__).resolve().parent
OLD_CODE_ROOT = ROOT / "old_code"
INFERENCE_ROOT = ROOT.parent / "inference"
EPS = 1e-6

TRAIN_STEPS = [
    "build_h5",
    "filter",
    "regularize",
    "direction_targets",
    "microenv",
    "instance_chunks",
    "mask_refine",
    "visualize",
]
TRAIN_FULL_STEPS = [
    "raw_ingest",
    "align",
    "filter",
    "degrade",
    "nmf",
    "build_h5",
    "regularize",
    "direction_targets",
    "microenv",
    "instance_chunks",
    "mask_refine",
    "visualize",
]
INFER_STEPS = [
    "build_h5",
    "infer",
    "write_h5ad",
    "visualize",
]


def _run_module_main(name: str, script_path: Path, argv: list[str]) -> None:
    print(f"\n[preprocess] step={name}")
    print("[preprocess] module:", str(script_path))
    print("[preprocess] argv:", " ".join(argv))
    if not script_path.exists():
        raise FileNotFoundError(f"{name}: script path does not exist: {script_path}")

    module_name = f"_preprocess_runtime_{script_path.stem}_{int(time.time() * 1e6)}"
    old_argv = list(sys.argv)
    old_sys_path = list(sys.path)
    try:
        parent = str(script_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"{name}: failed to create module spec for {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        sys.argv = [str(script_path), *argv]
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            raise AttributeError(f"{name}: module {script_path.name} does not define main()")
        module.main()
    finally:
        sys.argv = old_argv
        sys.path[:] = old_sys_path
        sys.modules.pop(module_name, None)


def _validate_step_window(start_from: str | None, stop_after: str | None, ordered_steps: list[str], mode: str) -> tuple[int, int]:
    start_name = start_from or ordered_steps[0]
    stop_name = stop_after or ordered_steps[-1]
    if start_name not in ordered_steps:
        raise ValueError(f"{mode}: unsupported --start-from '{start_name}'. Valid steps: {ordered_steps}")
    if stop_name not in ordered_steps:
        raise ValueError(f"{mode}: unsupported --stop-after '{stop_name}'. Valid steps: {ordered_steps}")
    start_idx = ordered_steps.index(start_name)
    stop_idx = ordered_steps.index(stop_name)
    if start_idx > stop_idx:
        raise ValueError(f"{mode}: --start-from '{start_name}' must be earlier than or equal to --stop-after '{stop_name}'.")
    return start_idx, stop_idx


def _step_enabled(step_name: str, ordered_steps: list[str], start_idx: int, stop_idx: int) -> bool:
    step_idx = ordered_steps.index(step_name)
    return start_idx <= step_idx <= stop_idx


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _import_bin2cell(bin2cell_path: Path | None) -> Any:
    if bin2cell_path is not None:
        path = bin2cell_path.resolve()
        candidates = []
        if path.is_file() and path.name == "bin2cell.py":
            candidates.append(path.parent)
        elif (path / "bin2cell.py").exists():
            candidates.append(path)
        elif (path / "bin2cell" / "__init__.py").exists():
            candidates.append(path)
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
    try:
        import bin2cell as b2c  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import bin2cell. Install bin2cell in the active environment "
            "or pass --bin2cell-path to the repo/package path."
        ) from exc
    return b2c


def _obs_label_has_positive_values(adata: ad.AnnData, key: str) -> bool:
    if key not in adata.obs.columns:
        return False
    values = pd.to_numeric(adata.obs[key], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return bool(np.any(values > 0))


def _ensure_stardist_label_column(
    input_h5ad: Path,
    *,
    source_image_path: Path,
    out_dir: Path,
    labels_key: str = "stardist_id",
    bin2cell_path: Path | None = None,
    library_id: str = "Visium_HD",
    mpp: float = 0.4,
    buffer: int = 150,
    no_crop: bool = False,
    stardist_model: str = "2D_versatile_he",
    stardist_block_size: int = 4096,
    stardist_min_overlap: int = 128,
    stardist_context: int = 128,
    stardist_prob_thresh: float | None = None,
    stardist_nms_thresh: float | None = None,
) -> None:
    backed = ad.read_h5ad(input_h5ad, backed="r")
    already_labeled = labels_key in backed.obs.columns
    backed.file.close()
    if already_labeled:
        check = ad.read_h5ad(input_h5ad)
        try:
            if _obs_label_has_positive_values(check, labels_key):
                print(
                    f"[preprocess] HD nucleus labels already exist; skipping StarDist | "
                    f"obs['{labels_key}'] in {input_h5ad}",
                    flush=True,
                )
                return
        finally:
            del check
            gc.collect()

    if not source_image_path.exists():
        raise FileNotFoundError(
            f"StarDist source image is required because {input_h5ad} lacks obs['{labels_key}']: "
            f"{source_image_path}"
        )

    inference_dir = str(INFERENCE_ROOT)
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    try:
        from nucleus_segment import (  # type: ignore
            assert_label_npz,
            get_scaled_image_path,
            get_spatial_key,
            get_stardist_npz_path,
            ensure_source_image_metadata,
            import_bin2cell,
            insert_label_column,
            make_scaled_image,
            read_segmentation_image,
            run_stardist,
        )
    except Exception as exc:
        raise ImportError("Could not import inference/nucleus_segment.py StarDist helpers.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    working_h5ad = out_dir / f"{input_h5ad.stem}.stardist_working.h5ad"
    ns = argparse.Namespace(
        input_h5ad=input_h5ad,
        out_dir=out_dir,
        source_image_path=source_image_path,
        library_id=library_id,
        bin2cell_path=bin2cell_path,
        mpp=float(mpp),
        buffer=int(buffer),
        no_crop=bool(no_crop),
        stardist_model=stardist_model,
        stardist_block_size=int(stardist_block_size),
        stardist_min_overlap=int(stardist_min_overlap),
        stardist_context=int(stardist_context),
        stardist_prob_thresh=stardist_prob_thresh,
        stardist_nms_thresh=stardist_nms_thresh,
        cellpose_model="cpsam",
        cellpose_grayscale=False,
        cellpose_chunk_size=3072,
        cellpose_chunk_overlap=256,
    )
    b2c = import_bin2cell(bin2cell_path)
    adata = ad.read_h5ad(input_h5ad)
    ensure_source_image_metadata(adata, source_image_path, library_id)
    scaled_image_path = get_scaled_image_path(ns)
    spatial_key = get_spatial_key(ns)
    if not (scaled_image_path.exists() and spatial_key in adata.obsm):
        print(
            f"[preprocess] StarDist prep | generating scaled image and spatial key "
            f"image={scaled_image_path} spatial_key={spatial_key}",
            flush=True,
        )
        adata, scaled_image_path, spatial_key = make_scaled_image(adata, ns, b2c, working_h5ad)
    else:
        print(
            f"[preprocess] StarDist prep | using cached scaled image {scaled_image_path}",
            flush=True,
        )

    labels_npz = get_stardist_npz_path(ns)
    if labels_npz.exists():
        image_shape = read_segmentation_image(scaled_image_path, grayscale=False).shape[:2]
        assert_label_npz(labels_npz, image_shape)
    else:
        run_stardist(scaled_image_path, labels_npz, ns, b2c)

    insert_label_column(adata, labels_npz, labels_key, spatial_key, float(mpp), b2c)
    adata.uns.setdefault("preprocess_stardist", {})
    adata.uns["preprocess_stardist"].update(
        {
            "source_image_path": str(source_image_path),
            "scaled_image_path": str(scaled_image_path),
            "labels_npz": str(labels_npz),
            "labels_key": str(labels_key),
            "mpp": float(mpp),
            "buffer": int(buffer),
            "crop": not bool(no_crop),
        }
    )
    print(
        f"[preprocess] writing StarDist labels back to reference h5ad | "
        f"path={input_h5ad} obs['{labels_key}']",
        flush=True,
    )
    adata.write_h5ad(input_h5ad)


def _file_nonempty(path: Path | None) -> bool:
    return path is not None and path.exists() and path.stat().st_size > 0


def _dir_nonempty(path: Path | None) -> bool:
    return path is not None and path.exists() and path.is_dir() and any(path.iterdir())


def _h5_has_dataset(path: Path, dataset_key: str) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return dataset_key in handle
    except Exception:
        return False


def _h5_has_attr(path: Path, attr_key: str) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return attr_key in handle.attrs
    except Exception:
        return False


def _h5_has_all_datasets(path: Path, dataset_keys: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return all(key in handle for key in dataset_keys)
    except Exception:
        return False


def _detect_train_full_resume_step(
    *,
    raw_bin_source_path: Path,
    raw_cell_source_path: Path,
    unioned_path: Path,
    filter_summary_path: Path,
    aligned_cell_path: Path,
    aligned_reference_path: Path,
    degraded_h5ad_path: Path,
    cell_nmf_h5ad_path: Path,
    module_csv_path: Path,
    output_h5: Path,
    instance_chunk_manifest_path: Path | None,
    mask_refine_output_h5: Path | None,
    visualize_dir: Path | None,
    regularize_masks: bool,
) -> tuple[str | None, dict[str, bool]]:
    completed: dict[str, bool] = {}
    completed["raw_ingest"] = _file_nonempty(raw_bin_source_path) and _file_nonempty(raw_cell_source_path)
    completed["align"] = _file_nonempty(unioned_path) and _file_nonempty(aligned_cell_path) and _file_nonempty(aligned_reference_path)
    completed["filter"] = _file_nonempty(filter_summary_path)
    completed["degrade"] = _file_nonempty(degraded_h5ad_path)
    completed["nmf"] = _file_nonempty(cell_nmf_h5ad_path) and _file_nonempty(module_csv_path)
    completed["build_h5"] = _file_nonempty(output_h5)
    completed["regularize"] = (not regularize_masks) or (
        completed["build_h5"] and _h5_has_attr(output_h5, "regularized_mask_kind")
    )
    completed["direction_targets"] = completed["build_h5"] and _h5_has_all_datasets(
        output_h5,
        ["boundary_radius_target_pool", "boundary_fourier_target_pool", "ellipse_param_pool"],
    )
    completed["microenv"] = completed["build_h5"] and _h5_has_all_datasets(
        output_h5,
        [
            "seed_feature_pool",
            "seed_sector_feature_pool",
            "neighbor_feature_pool",
            "neighbor_geometry_pool",
            "neighbor_valid_pool",
            "neighbor_direction_feature_pool",
        ],
    ) and _h5_has_attr(output_h5, "microenv_feature_version")
    completed["instance_chunks"] = instance_chunk_manifest_path is None or (
        _file_nonempty(instance_chunk_manifest_path)
        and _h5_has_attr(instance_chunk_manifest_path, "dataset_format")
        and _h5_has_all_datasets(
            instance_chunk_manifest_path,
            [
                "train_chunk_tile_offsets",
                "train_chunk_instance_offsets",
                "train_tile_input_pool",
                "train_instance_mask_targets_pool",
                "val_chunk_tile_offsets",
                "val_chunk_instance_offsets",
                "val_tile_input_pool",
                "val_instance_mask_targets_pool",
            ],
        )
    )
    mask_refine_target = mask_refine_output_h5 or instance_chunk_manifest_path
    completed["mask_refine"] = mask_refine_target is None or (
        _file_nonempty(mask_refine_target)
        and (
            _h5_has_all_datasets(
                mask_refine_target,
                [
                    "train_instance_mask_targets_refined_pool",
                    "train_instance_refined_ellipse_param_pool",
                    "train_instance_boundary_radius_targets_refined_pool",
                    "val_instance_mask_targets_refined_pool",
                    "val_instance_refined_ellipse_param_pool",
                    "val_instance_boundary_radius_targets_refined_pool",
                ],
            )
            or _h5_has_all_datasets(
                mask_refine_target,
                [
                    "xenium_mask_pool_refined",
                    "ellipse_param_pool_refined",
                    "boundary_radius_target_pool_refined",
                ],
            )
        )
    )
    completed["visualize"] = visualize_dir is None or _dir_nonempty(visualize_dir)

    for step_name in TRAIN_FULL_STEPS:
        if not completed.get(step_name, False):
            return step_name, completed
    return None, completed


def _write_h5ad_step(adata: ad.AnnData, path: Path, name: str) -> None:
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    elapsed = time.perf_counter() - t0
    print(f"[preprocess] wrote {name}: {path} | shape=({adata.n_obs}, {adata.n_vars}) elapsed_sec={elapsed:.1f}")


def _require_existing_file(path: Path, step_name: str, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{step_name}: required {description} is missing: {path}. "
            f"Please rerun an earlier step or adjust --start-from."
        )


def _read_gene_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return genes or None


def _to_csr_float32(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(X, dtype=np.float32))


def _row_sum(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    return np.asarray(X, dtype=np.float32).sum(axis=1).astype(np.float32)


def _align_to_genes(adata: ad.AnnData, genes: list[str]) -> ad.AnnData:
    current_genes = [str(gene) for gene in adata.var_names.tolist()]
    if current_genes == genes:
        return adata
    current_gene_set = set(current_genes)
    missing = [gene for gene in genes if gene not in current_gene_set]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{getattr(adata, 'filename', 'input h5ad')} is missing {len(missing)} required genes. "
            f"First missing genes: {preview}"
        )
    return adata[:, genes].copy()


def _rasterize_polygons(df: pd.DataFrame, mask: np.ndarray, id_map: dict[str, int], bin_size: float) -> None:
    work = df[["cell_id", "vertex_x", "vertex_y"]].copy()
    work = work.dropna(subset=["cell_id", "vertex_x", "vertex_y"])
    work["vertex_x"] = pd.to_numeric(work["vertex_x"], errors="coerce")
    work["vertex_y"] = pd.to_numeric(work["vertex_y"], errors="coerce")
    work = work.dropna(subset=["vertex_x", "vertex_y"])
    work["grid_x"] = np.floor(work["vertex_x"].to_numpy(dtype=np.float32) / float(bin_size)).astype(np.int32)
    work["grid_y"] = np.floor(work["vertex_y"].to_numpy(dtype=np.float32) / float(bin_size)).astype(np.int32)
    work = work[(work["grid_x"] >= 0) & (work["grid_y"] >= 0)]
    if work.empty:
        return
    iterator = work.groupby("cell_id", sort=False)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="rasterize_polygons", unit="cell")
    for cell_id, group in iterator:
        cell_id = str(cell_id)
        if cell_id not in id_map:
            continue
        points = group[["grid_x", "grid_y"]].to_numpy(dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], color=int(id_map[cell_id]))


def _build_raw_bin_and_cell_h5ad_from_parquet(
    reference_h5ad_path: Path,
    transcripts_path: Path,
    cell_boundaries_path: Path,
    nucleus_boundaries_path: Path,
    output_raw_bin_h5ad_path: Path,
    output_raw_cell_h5ad_path: Path,
    selected_genes_path: Path | None,
    bin_size: float,
) -> tuple[Path, Path]:
    t0 = time.perf_counter()
    if selected_genes_path is not None and selected_genes_path.exists():
        selected_genes = _read_gene_list(selected_genes_path)
        if not selected_genes:
            raise ValueError(f"Selected gene list is empty: {selected_genes_path}")
        print(
            f"[preprocess] raw_ingest | using existing selected gene list "
            f"genes={len(selected_genes)} path={selected_genes_path}",
            flush=True,
        )
    else:
        adata_reference = ad.read_h5ad(reference_h5ad_path, backed="r")
        selected_genes = [str(g) for g in adata_reference.var_names.tolist()]
        adata_reference.file.close()
    if selected_genes_path is not None:
        selected_genes_path.parent.mkdir(parents=True, exist_ok=True)
        selected_genes_path.write_text("\n".join(selected_genes), encoding="utf-8")

    print("[preprocess] raw_ingest | reading transcript parquet")
    df_trans = pd.read_parquet(
        transcripts_path,
        columns=["feature_name", "x_location", "y_location", "is_gene"],
    )
    df_trans = df_trans[df_trans["is_gene"]].copy()
    df_trans["feature_name"] = df_trans["feature_name"].astype(str)
    df_trans = df_trans[df_trans["feature_name"].isin(selected_genes)].copy()
    df_trans = df_trans[["feature_name", "x_location", "y_location"]]
    if df_trans.empty:
        raise ValueError("No transcript records remain after filtering to the reference gene set.")

    df_trans["bin_x"] = np.floor(df_trans["x_location"].to_numpy(dtype=np.float32) / float(bin_size)).astype(np.int32)
    df_trans["bin_y"] = np.floor(df_trans["y_location"].to_numpy(dtype=np.float32) / float(bin_size)).astype(np.int32)
    df_trans = df_trans[(df_trans["bin_x"] >= 0) & (df_trans["bin_y"] >= 0)].copy()
    if df_trans.empty:
        raise ValueError("No transcript records remain after binning and coordinate filtering.")

    print("[preprocess] raw_ingest | building sparse bin x gene matrix")
    max_x = int(df_trans["bin_x"].max()) + 1
    max_y = int(df_trans["bin_y"].max()) + 1
    bin_linear = (
        df_trans["bin_y"].to_numpy(dtype=np.int64) * np.int64(max_x)
        + df_trans["bin_x"].to_numpy(dtype=np.int64)
    )
    bin_ids, bin_uniques = pd.factorize(bin_linear, sort=False)
    bin_ids = bin_ids.astype(np.int32, copy=False)
    gene_map = {gene: idx for idx, gene in enumerate(selected_genes)}
    gene_ids = df_trans["feature_name"].map(gene_map).to_numpy(dtype=np.int32)
    x_raw = sp.coo_matrix(
        (np.ones(len(df_trans), dtype=np.float32), (bin_ids, gene_ids)),
        shape=(len(bin_uniques), len(selected_genes)),
    ).tocsr()
    df_trans = df_trans[["bin_x", "bin_y"]].copy()

    raw_bin_adata = ad.AnnData(X=x_raw)
    raw_bin_adata.var_names = pd.Index(np.asarray(selected_genes, dtype=str))

    bin_uniques = np.asarray(bin_uniques, dtype=np.int64)
    bin_x_coords = (bin_uniques % np.int64(max_x)).astype(np.int32, copy=False)
    bin_y_coords = (bin_uniques // np.int64(max_x)).astype(np.int32, copy=False)
    raw_bin_adata.obs_names = pd.Index([f"{int(x)}_{int(y)}" for x, y in zip(bin_x_coords, bin_y_coords)])
    coords = np.column_stack([bin_x_coords, bin_y_coords]).astype(np.int32, copy=False)
    raw_bin_adata.obsm["spatial"] = coords.astype(np.float32) * float(bin_size)
    del bin_linear
    del bin_ids
    del bin_uniques
    gc.collect()
    cell_mask = np.zeros((max_y, max_x), dtype=np.int32)
    nucleus_mask = np.zeros((max_y, max_x), dtype=np.int32)

    print("[preprocess] raw_ingest | reading cell boundary csv")
    df_cell = pd.read_csv(cell_boundaries_path, usecols=["cell_id", "vertex_x", "vertex_y"])
    print("[preprocess] raw_ingest | reading nucleus boundary csv")
    df_nucleus = pd.read_csv(nucleus_boundaries_path, usecols=["cell_id", "vertex_x", "vertex_y"])
    unique_cells = pd.unique(
        pd.concat(
            [df_cell["cell_id"].astype(str), df_nucleus["cell_id"].astype(str)],
            ignore_index=True,
        )
    )
    cell_id_to_int = {cell_id: idx + 1 for idx, cell_id in enumerate(unique_cells)}
    int_to_cell_id = {idx + 1: cell_id for idx, cell_id in enumerate(unique_cells)}
    int_to_cell_id[0] = "0"

    print("[preprocess] raw_ingest | rasterizing cell polygons")
    _rasterize_polygons(df_cell, cell_mask, cell_id_to_int, bin_size)
    del df_cell
    gc.collect()
    print("[preprocess] raw_ingest | rasterizing nucleus polygons")
    _rasterize_polygons(df_nucleus, nucleus_mask, cell_id_to_int, bin_size)
    del df_nucleus
    gc.collect()

    union_cell_mask = cell_mask.copy()
    nucleus_only_pixels = (union_cell_mask == 0) & (nucleus_mask > 0)
    union_cell_mask[nucleus_only_pixels] = nucleus_mask[nucleus_only_pixels]
    conflict_pixels = (cell_mask > 0) & (nucleus_mask > 0) & (cell_mask != nucleus_mask)
    if np.any(conflict_pixels):
        print(
            "[preprocess] raw_ingest | warning: cell/nucleus boundary id conflicts "
            f"pixels={int(np.sum(conflict_pixels))}; keeping cell boundary id for conflicts",
            flush=True,
        )
    print(
        "[preprocess] raw_ingest | unioned 10x instance masks "
        f"cell_pixels={int(np.sum(cell_mask > 0))} nucleus_pixels={int(np.sum(nucleus_mask > 0))} "
        f"nucleus_only_pixels={int(np.sum(nucleus_only_pixels))}",
        flush=True,
    )

    raw_bin_adata.obs["cell_id_int"] = union_cell_mask[bin_y_coords, bin_x_coords].astype(np.int32)
    raw_bin_adata.obs["nucleus_id_int"] = nucleus_mask[bin_y_coords, bin_x_coords].astype(np.int32)
    raw_bin_adata.obs["cell_id"] = pd.Categorical([int_to_cell_id.get(int(v), "0") for v in raw_bin_adata.obs["cell_id_int"].tolist()])
    raw_bin_adata.obs["nucleus_id"] = pd.Categorical([int_to_cell_id.get(int(v), "0") for v in raw_bin_adata.obs["nucleus_id_int"].tolist()])
    raw_bin_adata.obs["bin_x"] = bin_x_coords.astype(np.int32)
    raw_bin_adata.obs["bin_y"] = bin_y_coords.astype(np.int32)
    raw_bin_adata.obs["array_col"] = bin_x_coords.astype(np.int32)
    raw_bin_adata.obs["array_row"] = bin_y_coords.astype(np.int32)
    raw_bin_adata.obs["in_tissue"] = np.ones(raw_bin_adata.n_obs, dtype=np.int32)

    print("[preprocess] raw_ingest | building sparse cell x gene matrix")
    transcript_cell_ids = union_cell_mask[
        df_trans["bin_y"].to_numpy(dtype=np.int32),
        df_trans["bin_x"].to_numpy(dtype=np.int32),
    ]
    valid_transcript_mask = transcript_cell_ids > 0
    cell_obs_names = np.asarray(unique_cells, dtype=str)
    if np.any(valid_transcript_mask):
        cell_rows = transcript_cell_ids[valid_transcript_mask].astype(np.int64) - 1
        cell_gene_ids = gene_ids[valid_transcript_mask]
        cell_counts = np.ones(int(valid_transcript_mask.sum()), dtype=np.float32)
        x_cell = sp.coo_matrix(
            (cell_counts, (cell_rows, cell_gene_ids)),
            shape=(len(cell_obs_names), len(selected_genes)),
        ).tocsr()
    else:
        x_cell = sp.csr_matrix((len(cell_obs_names), len(selected_genes)), dtype=np.float32)

    row_sum = np.asarray(x_cell.sum(axis=1)).ravel()
    keep_cells = row_sum > 0
    if not np.any(keep_cells):
        raise ValueError("No valid cell-level expression rows were produced from transcripts and cell boundaries.")
    raw_cell_adata = ad.AnnData(X=x_cell[keep_cells])
    kept_cell_names = cell_obs_names[keep_cells]
    raw_cell_adata.obs_names = pd.Index(kept_cell_names.astype(str))
    raw_cell_adata.var_names = pd.Index(np.asarray(selected_genes, dtype=str))
    raw_cell_adata.obs["cell_id"] = raw_cell_adata.obs_names.astype(str)
    raw_cell_adata.obs["cell_id_int"] = np.arange(1, raw_cell_adata.n_obs + 1, dtype=np.int32)

    output_raw_bin_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    output_raw_cell_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bin_adata.write_h5ad(output_raw_bin_h5ad_path)
    raw_cell_adata.write_h5ad(output_raw_cell_h5ad_path)
    elapsed = time.perf_counter() - t0
    print(
        f"[preprocess] wrote raw bin/cell h5ad from parquet/csv | "
        f"raw_bin={output_raw_bin_h5ad_path} shape=({raw_bin_adata.n_obs}, {raw_bin_adata.n_vars}) "
        f"raw_cell={output_raw_cell_h5ad_path} shape=({raw_cell_adata.n_obs}, {raw_cell_adata.n_vars}) "
        "cell_source=union(cell_boundaries,nucleus_boundaries) "
        f"elapsed_sec={elapsed:.1f}"
    )
    return output_raw_bin_h5ad_path, output_raw_cell_h5ad_path


def _normalize_positive_labels(values: pd.Series) -> tuple[np.ndarray, dict[int, str]]:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy()
        numeric = np.where(numeric > 0, np.rint(numeric), 0).astype(np.int64)
        mapping = {int(v): str(int(v)) for v in np.unique(numeric[numeric > 0]).tolist()}
        return numeric.astype(np.int32), mapping

    string_values = values.astype(str).to_numpy()
    invalid = {"", "0", "0.0", "nan", "None", "<NA>"}
    label_ids = np.zeros(len(string_values), dtype=np.int32)
    mapping: dict[int, str] = {}
    next_id = 1
    for label in pd.unique(string_values).tolist():
        if label in invalid:
            continue
        mapping[next_id] = label
        label_ids[string_values == label] = next_id
        next_id += 1
    return label_ids, mapping


def _collect_common_genes(
    raw_bin_adata: ad.AnnData,
    raw_cell_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    selected_genes: list[str] | None,
) -> list[str]:
    raw_bin_set = set(map(str, raw_bin_adata.var_names.tolist()))
    raw_cell_set = set(map(str, raw_cell_adata.var_names.tolist()))
    reference_set = set(map(str, reference_adata.var_names.tolist()))
    common = raw_bin_set & raw_cell_set & reference_set
    if not common:
        raise ValueError("No common genes exist across raw-bin, raw-cell, and reference HD inputs.")

    if selected_genes is not None:
        ordered = [gene for gene in selected_genes if gene in common]
        missing = [gene for gene in selected_genes if gene not in common]
        if missing:
            preview = ", ".join(missing[:10])
            print(
                f"[preprocess] warning: dropped {len(missing)} selected genes not shared by all inputs. "
                f"First dropped genes: {preview}"
            )
        if not ordered:
            raise ValueError("Selected gene list has no overlap across the three inputs.")
        return ordered

    raw_cell_order = [str(g) for g in raw_cell_adata.var_names.tolist()]
    return [gene for gene in raw_cell_order if gene in common]


def _build_unioned_bin_adata(
    raw_bin_adata: ad.AnnData,
    cell_label_key: str,
    cell_label_int_key: str,
    nucleus_label_key: str,
    bin_x_key: str,
    bin_y_key: str,
) -> ad.AnnData:
    obs = raw_bin_adata.obs.copy()
    if nucleus_label_key not in obs.columns:
        raise ValueError(f"raw-bin h5ad is missing obs['{nucleus_label_key}'].")

    nucleus_ids, _ = _normalize_positive_labels(obs[nucleus_label_key])

    if cell_label_int_key in obs.columns:
        cell_ids_numeric = pd.to_numeric(obs[cell_label_int_key], errors="coerce").fillna(0).to_numpy()
        cell_ids = np.where(cell_ids_numeric > 0, np.rint(cell_ids_numeric), 0).astype(np.int32)
        if cell_label_key in obs.columns:
            raw_cell_labels = obs[cell_label_key]
        else:
            raw_cell_labels = None
        int_to_str: dict[int, str] = {}
        positive_cell_positions = np.nonzero(cell_ids > 0)[0]
        if positive_cell_positions.size:
            unique_cell_ids, first_positions = np.unique(cell_ids[positive_cell_positions], return_index=True)
            first_obs_positions = positive_cell_positions[first_positions]
            if raw_cell_labels is not None:
                first_labels = raw_cell_labels.iloc[first_obs_positions].astype(str).to_numpy()
            else:
                first_labels = unique_cell_ids.astype(str)
            int_to_str = {int(cid): str(label) for cid, label in zip(unique_cell_ids.tolist(), first_labels.tolist())}
    elif cell_label_key in obs.columns:
        cell_ids, int_to_str = _normalize_positive_labels(obs[cell_label_key])
    else:
        raise ValueError(
            f"raw-bin h5ad must contain either obs['{cell_label_int_key}'] or obs['{cell_label_key}']."
        )

    positive_nuclei = np.unique(nucleus_ids[nucleus_ids > 0])
    overlap_mask = (nucleus_ids > 0) & (cell_ids > 0)
    nucleus_to_cell: dict[int, int] = {}
    if np.any(overlap_mask):
        overlap_df = pd.DataFrame(
            {
                "nucleus_id": nucleus_ids[overlap_mask].astype(np.int32, copy=False),
                "cell_id": cell_ids[overlap_mask].astype(np.int32, copy=False),
            }
        )
        pair_counts = overlap_df.groupby(["nucleus_id", "cell_id"], sort=False).size().reset_index(name="count")
        winners = pair_counts.sort_values(["nucleus_id", "count"], ascending=[True, False]).drop_duplicates(
            "nucleus_id", keep="first"
        )
        nucleus_to_cell = {
            int(nucleus_id): int(cell_id)
            for nucleus_id, cell_id in zip(winners["nucleus_id"].to_numpy(), winners["cell_id"].to_numpy())
        }
        del overlap_df
        del pair_counts
        del winners

    next_cell_id = int(cell_ids.max(initial=0)) + 1
    for nucleus_id in positive_nuclei.tolist():
        nucleus_id = int(nucleus_id)
        if nucleus_id in nucleus_to_cell:
            continue
        nucleus_to_cell[nucleus_id] = next_cell_id
        int_to_str[next_cell_id] = f"nucleus_only_{nucleus_id}"
        next_cell_id += 1

    union_cell_ids = cell_ids.copy()
    if nucleus_to_cell:
        max_nucleus_id = int(max(nucleus_to_cell.keys()))
        nucleus_cell_lookup = np.zeros(max_nucleus_id + 1, dtype=np.int32)
        for nucleus_id, cell_id in nucleus_to_cell.items():
            nucleus_cell_lookup[int(nucleus_id)] = int(cell_id)
            int_to_str.setdefault(int(cell_id), f"cell_{int(cell_id)}")
        positive_nucleus_mask = (nucleus_ids > 0) & (nucleus_ids <= max_nucleus_id)
        union_cell_ids[positive_nucleus_mask] = nucleus_cell_lookup[nucleus_ids[positive_nucleus_mask]]

    max_union_cell_id = int(union_cell_ids.max(initial=0))
    cell_label_lookup = np.empty(max_union_cell_id + 1, dtype=object)
    cell_label_lookup[:] = ""
    for cell_id, label in int_to_str.items():
        if 0 < int(cell_id) <= max_union_cell_id:
            cell_label_lookup[int(cell_id)] = str(label)
    union_cell_str = cell_label_lookup[union_cell_ids]

    if bin_x_key not in obs.columns and "array_col" in obs.columns:
        obs[bin_x_key] = obs["array_col"].to_numpy()
    if bin_y_key not in obs.columns and "array_row" in obs.columns:
        obs[bin_y_key] = obs["array_row"].to_numpy()
    if bin_x_key not in obs.columns or bin_y_key not in obs.columns:
        raise ValueError(
            f"raw-bin h5ad must contain bin coordinates via obs['{bin_x_key}'] / obs['{bin_y_key}'] "
            "or array_col / array_row."
        )

    obs["cell_id_int"] = union_cell_ids.astype(np.int32)
    obs["cell_id"] = union_cell_str.astype(str)
    obs["nucleus_id_int"] = nucleus_ids.astype(np.int32)
    obs["bin_x"] = pd.to_numeric(obs[bin_x_key], errors="coerce").to_numpy(dtype=np.int32)
    obs["bin_y"] = pd.to_numeric(obs[bin_y_key], errors="coerce").to_numpy(dtype=np.int32)
    obs["array_col"] = obs["bin_x"].to_numpy(dtype=np.int32)
    obs["array_row"] = obs["bin_y"].to_numpy(dtype=np.int32)
    if "in_tissue" not in obs.columns:
        obs["in_tissue"] = np.ones(raw_bin_adata.n_obs, dtype=np.int32)

    return ad.AnnData(X=raw_bin_adata.X, obs=obs, var=raw_bin_adata.var.copy(), uns=raw_bin_adata.uns.copy())


def _filter_unioned_bin_adata(
    unioned_bin_adata: ad.AnnData,
    *,
    min_nucleus_bins: int,
    min_cell_bins: int,
    valid_cell_ids: set[str] | None = None,
) -> tuple[ad.AnnData, dict[str, int]]:
    obs = unioned_bin_adata.obs.copy()
    cell_ids = pd.to_numeric(obs["cell_id_int"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    nucleus_positive = pd.to_numeric(obs["nucleus_id_int"], errors="coerce").fillna(0).to_numpy(dtype=np.int64) > 0
    positive_cells = cell_ids > 0
    cell_names = (
        obs["cell_id"].astype(str).to_numpy()
        if "cell_id" in obs.columns
        else np.asarray([str(int(v)) for v in cell_ids.tolist()], dtype=str)
    )

    if not np.any(positive_cells):
        summary = {
            "cells_before": 0,
            "cells_after": 0,
            "removed_cells": 0,
            "removed_nucleus_only": 0,
            "removed_cell_only": 0,
            "removed_both": 0,
            "removed_by_nmf_source_qc": 0,
            "bins_reassigned_to_background": 0,
        }
        filtered = unioned_bin_adata.copy()
        filtered.uns["prefilter_summary"] = summary
        return filtered, summary

    max_cell_id = int(cell_ids.max())
    cell_counts = np.bincount(cell_ids[positive_cells], minlength=max_cell_id + 1)
    nucleus_counts = np.bincount(cell_ids[positive_cells & nucleus_positive], minlength=max_cell_id + 1)
    keep_ids = np.flatnonzero((cell_counts >= int(min_cell_bins)) & (nucleus_counts >= int(min_nucleus_bins)))
    keep_ids = keep_ids[keep_ids > 0]
    all_ids = np.flatnonzero(cell_counts > 0)
    all_ids = all_ids[all_ids > 0]
    ids_after_bin_qc = keep_ids.copy()
    removed_by_nmf_qc = 0
    if valid_cell_ids is not None:
        id_to_name: dict[int, str] = {}
        for cid, name in zip(cell_ids.tolist(), cell_names.tolist(), strict=False):
            if cid > 0 and cid not in id_to_name:
                id_to_name[int(cid)] = str(name)
        keep_ids = np.asarray(
            [cid for cid in keep_ids.tolist() if id_to_name.get(int(cid), str(int(cid))) in valid_cell_ids],
            dtype=np.int64,
        )
        removed_by_nmf_qc = int(ids_after_bin_qc.size - keep_ids.size)
    keep_id_set = set(int(v) for v in keep_ids.tolist())

    removed_cells = np.array([cid for cid in all_ids.tolist() if int(cid) not in keep_id_set], dtype=np.int64)
    removed_nucleus_only = 0
    removed_cell_only = 0
    removed_both = 0
    if removed_cells.size > 0:
        removed_nucleus_only = int(np.sum((nucleus_counts[removed_cells] < int(min_nucleus_bins)) & (cell_counts[removed_cells] >= int(min_cell_bins))))
        removed_cell_only = int(np.sum((nucleus_counts[removed_cells] >= int(min_nucleus_bins)) & (cell_counts[removed_cells] < int(min_cell_bins))))
        removed_both = int(np.sum((nucleus_counts[removed_cells] < int(min_nucleus_bins)) & (cell_counts[removed_cells] < int(min_cell_bins))))

    drop_mask = positive_cells & (~np.isin(cell_ids, keep_ids))
    for label_key in ("cell_id", "nucleus_id"):
        if label_key in obs.columns and isinstance(obs[label_key].dtype, pd.CategoricalDtype):
            if "0" not in obs[label_key].cat.categories:
                obs[label_key] = obs[label_key].cat.add_categories(["0"])
    obs.loc[drop_mask, "cell_id_int"] = np.int32(0)
    obs.loc[drop_mask, "cell_id"] = "0"
    obs.loc[drop_mask, "nucleus_id_int"] = np.int32(0)
    if "nucleus_id" in obs.columns:
        obs.loc[drop_mask, "nucleus_id"] = "0"

    filtered = ad.AnnData(X=unioned_bin_adata.X, obs=obs, var=unioned_bin_adata.var.copy(), uns=unioned_bin_adata.uns.copy())
    summary = {
        "cells_before": int(all_ids.size),
        "cells_after": int(keep_ids.size),
        "removed_cells": int(removed_cells.size),
        "removed_nucleus_only": int(removed_nucleus_only),
        "removed_cell_only": int(removed_cell_only),
        "removed_both": int(removed_both),
        "removed_by_nmf_source_qc": int(removed_by_nmf_qc),
        "bins_reassigned_to_background": int(np.sum(drop_mask)),
    }
    filtered.uns["prefilter_summary"] = summary
    return filtered, summary


def _aggregate_cells_from_binned_adata(
    unioned_bin_adata: ad.AnnData,
    *,
    cell_id_int_key: str = "cell_id_int",
    cell_id_key: str = "cell_id",
) -> ad.AnnData:
    obs = unioned_bin_adata.obs
    cell_ids = pd.to_numeric(obs[cell_id_int_key], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    keep_rows = cell_ids > 0
    if not np.any(keep_rows):
        empty = ad.AnnData(X=sp.csr_matrix((0, unioned_bin_adata.n_vars), dtype=np.float32))
        empty.var_names = unioned_bin_adata.var_names.copy()
        empty.obs["cell_id"] = pd.Series(dtype=str)
        empty.obs["cell_id_int"] = pd.Series(dtype=np.int32)
        return empty

    kept_ids = cell_ids[keep_rows]
    unique_ids = pd.unique(kept_ids).astype(np.int64, copy=False)
    row_lookup = {int(cid): idx for idx, cid in enumerate(unique_ids.tolist())}
    group_rows = np.array([row_lookup[int(cid)] for cid in kept_ids.tolist()], dtype=np.int64)
    group_cols = np.flatnonzero(keep_rows).astype(np.int64)
    projector = sp.csr_matrix(
        (np.ones(group_rows.shape[0], dtype=np.float32), (group_rows, group_cols)),
        shape=(unique_ids.shape[0], unioned_bin_adata.n_obs),
        dtype=np.float32,
    )
    X_cell = (projector @ _to_csr_float32(unioned_bin_adata.X)).tocsr().astype(np.float32)

    id_to_name: dict[int, str] = {}
    if cell_id_key in obs.columns:
        cell_names = obs[cell_id_key].astype(str).to_numpy()
        for cid, name in zip(cell_ids.tolist(), cell_names.tolist(), strict=False):
            if cid > 0 and cid not in id_to_name:
                id_to_name[int(cid)] = str(name)
    names = [id_to_name.get(int(cid), str(int(cid))) for cid in unique_ids.tolist()]

    out = ad.AnnData(X=X_cell)
    out.var_names = unioned_bin_adata.var_names.copy()
    out.obs_names = pd.Index(np.asarray(names, dtype=str))
    out.obs["cell_id"] = np.asarray(names, dtype=str)
    out.obs["cell_id_int"] = unique_ids.astype(np.int32)
    return out


def _qc_cell_gene_matrix(
    cell_adata: ad.AnnData,
    *,
    min_counts: float,
    min_genes: int,
    min_cells: int,
) -> tuple[ad.AnnData, dict[str, object]]:
    X = _to_csr_float32(cell_adata.X)
    cell_counts = np.asarray(X.sum(axis=1)).reshape(-1).astype(np.float32)
    cell_genes = np.asarray((X > 0).sum(axis=1)).reshape(-1).astype(np.int32)
    keep_cells = (cell_counts > float(min_counts)) & (cell_genes > int(min_genes))
    if not np.any(keep_cells):
        raise ValueError(
            "No pseudo-cells remain after NMF-source cell QC: "
            f"total_counts > {float(min_counts)} and n_genes_by_counts > {int(min_genes)}."
        )

    cell_adata = cell_adata[keep_cells].copy()
    X = _to_csr_float32(cell_adata.X)
    gene_cells = np.asarray((X > 0).sum(axis=0)).reshape(-1).astype(np.int64)
    keep_genes = gene_cells > int(min_cells)
    if not np.any(keep_genes):
        raise ValueError(
            f"No genes remain after NMF-source gene QC: expressing_cells > {int(min_cells)}."
        )

    out = cell_adata[:, keep_genes].copy()
    X_out = _to_csr_float32(out.X)
    out.obs["total_counts"] = np.asarray(X_out.sum(axis=1)).reshape(-1).astype(np.float32)
    out.obs["n_genes_by_counts"] = np.asarray((X_out > 0).sum(axis=1)).reshape(-1).astype(np.int32)
    summary = {
        "min_counts_strict_gt": float(min_counts),
        "min_genes_strict_gt": int(min_genes),
        "min_cells_strict_gt": int(min_cells),
        "cells_before_qc": int(cell_counts.shape[0]),
        "cells_after_qc": int(out.n_obs),
        "genes_before_qc": int(gene_cells.shape[0]),
        "genes_after_qc": int(out.n_vars),
    }
    return out, summary


def _aggregate_cells_from_label_column(
    adata: ad.AnnData,
    *,
    label_key: str,
) -> ad.AnnData:
    if label_key not in adata.obs.columns:
        raise ValueError(f"NMF source h5ad is missing obs['{label_key}'].")

    labels = pd.to_numeric(adata.obs[label_key], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    keep_rows = labels > 0
    if not np.any(keep_rows):
        raise ValueError(f"NMF source label column obs['{label_key}'] contains no positive labels.")

    kept_row_idx = np.flatnonzero(keep_rows).astype(np.int64)
    unique_labels, inverse = np.unique(labels[keep_rows], return_inverse=True)
    projector = sp.csr_matrix(
        (
            np.ones(inverse.shape[0], dtype=np.float32),
            (inverse.astype(np.int64), kept_row_idx),
        ),
        shape=(unique_labels.shape[0], adata.n_obs),
        dtype=np.float32,
    )
    X_cell = (projector @ _to_csr_float32(adata.X)).tocsr().astype(np.float32)
    n_bins = np.asarray(projector.sum(axis=1)).reshape(-1).astype(np.int32)

    obs = pd.DataFrame(index=pd.Index([str(int(v)) for v in unique_labels.tolist()], name="cell_id"))
    obs["cell_id"] = obs.index.astype(str)
    obs["cell_id_int"] = unique_labels.astype(np.int32)
    obs["source_label"] = unique_labels.astype(np.int32)
    obs["n_bins"] = n_bins
    out = ad.AnnData(X=X_cell, obs=obs, var=adata.var.copy(), uns=adata.uns.copy())
    if "array_row" in adata.obs.columns and "array_col" in adata.obs.columns:
        row_values = pd.to_numeric(adata.obs["array_row"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)[keep_rows]
        col_values = pd.to_numeric(adata.obs["array_col"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)[keep_rows]
        counts = np.bincount(inverse, minlength=unique_labels.shape[0]).astype(np.float64)
        safe_counts = np.maximum(counts, 1.0)
        out.obs["array_row"] = np.bincount(inverse, weights=row_values, minlength=unique_labels.shape[0]) / safe_counts
        out.obs["array_col"] = np.bincount(inverse, weights=col_values, minlength=unique_labels.shape[0]) / safe_counts
        out.obsm["spatial"] = out.obs[["array_col", "array_row"]].to_numpy(dtype=np.float32)
    return out


def _ensure_hd_expanded_labels_for_degrade(
    input_h5ad: Path,
    *,
    labels_key: str,
    expanded_labels_key: str,
    fallback_label_keys: list[str],
    bin2cell_path: Path | None = None,
    auto_stardist: bool = True,
    stardist_source_image_path: Path | None = None,
    stardist_out_dir: Path | None = None,
    stardist_label_key: str = "stardist_id",
    stardist_library_id: str = "Visium_HD",
    stardist_mpp: float = 0.4,
    stardist_buffer: int = 150,
    stardist_no_crop: bool = False,
    stardist_model: str = "2D_versatile_he",
    stardist_block_size: int = 4096,
    stardist_min_overlap: int = 128,
    stardist_context: int = 128,
    stardist_prob_thresh: float | None = None,
    stardist_nms_thresh: float | None = None,
) -> None:
    print(
        f"[preprocess] degrade reference labels | ensuring obs['{expanded_labels_key}'] in {input_h5ad}",
        flush=True,
    )
    adata = ad.read_h5ad(input_h5ad)
    adata.var_names_make_unique()

    if _obs_label_has_positive_values(adata, expanded_labels_key):
        print(
            f"[preprocess] degrade reference labels | using existing obs['{expanded_labels_key}']",
            flush=True,
        )
        return

    if not _obs_label_has_positive_values(adata, labels_key):
        fallback = next((key for key in fallback_label_keys if _obs_label_has_positive_values(adata, key)), None)
        if fallback is None and auto_stardist:
            del adata
            gc.collect()
            if stardist_source_image_path is None:
                raise ValueError(
                    f"{input_h5ad} has no obs['{labels_key}'], obs['{expanded_labels_key}'], "
                    f"or fallback labels {fallback_label_keys}; --reference-image-path is required "
                    "to run StarDist before HD foreground/background degrade."
                )
            _ensure_stardist_label_column(
                input_h5ad,
                source_image_path=stardist_source_image_path,
                out_dir=stardist_out_dir or (input_h5ad.parent / "degrade_stardist"),
                labels_key=stardist_label_key,
                bin2cell_path=bin2cell_path,
                library_id=stardist_library_id,
                mpp=float(stardist_mpp),
                buffer=int(stardist_buffer),
                no_crop=bool(stardist_no_crop),
                stardist_model=stardist_model,
                stardist_block_size=int(stardist_block_size),
                stardist_min_overlap=int(stardist_min_overlap),
                stardist_context=int(stardist_context),
                stardist_prob_thresh=stardist_prob_thresh,
                stardist_nms_thresh=stardist_nms_thresh,
            )
            adata = ad.read_h5ad(input_h5ad)
            adata.var_names_make_unique()
            fallback = next((key for key in fallback_label_keys if _obs_label_has_positive_values(adata, key)), None)
        if fallback is None:
            raise ValueError(
                f"{input_h5ad} has no usable degrade labels. Expected obs['{expanded_labels_key}'], "
                f"obs['{labels_key}'], or one of {fallback_label_keys}."
            )
        print(
            f"[preprocess] degrade reference labels | copying obs['{fallback}'] -> obs['{labels_key}']",
            flush=True,
        )
        adata.obs[labels_key] = pd.to_numeric(adata.obs[fallback], errors="coerce").fillna(0).astype(np.int32)

    b2c = _import_bin2cell(bin2cell_path)
    print(
        f"[preprocess] degrade reference labels | running bin2cell.expand_labels "
        f"{labels_key} -> {expanded_labels_key}",
        flush=True,
    )
    b2c.expand_labels(
        adata,
        labels_key=labels_key,
        expanded_labels_key=expanded_labels_key,
    )
    if not _obs_label_has_positive_values(adata, expanded_labels_key):
        raise ValueError(
            f"bin2cell.expand_labels did not produce positive obs['{expanded_labels_key}'] for {input_h5ad}."
        )
    adata.write_h5ad(input_h5ad)
    print(
        f"[preprocess] degrade reference labels | wrote obs['{expanded_labels_key}'] back to {input_h5ad}",
        flush=True,
    )


def _build_xenium_union_cell_nmf_source(
    input_cell_h5ad: Path,
    output_cell_h5ad: Path,
    selected_genes_output_path: Path,
    *,
    min_counts: float,
    min_genes: int,
    min_cells: int,
) -> Path:
    t0 = time.perf_counter()
    print(
        f"[preprocess] building Xenium union-cell NMF source | input={input_cell_h5ad}",
        flush=True,
    )
    cell_adata = ad.read_h5ad(input_cell_h5ad)
    cell_adata.var_names_make_unique()
    cell_adata, qc_summary = _qc_cell_gene_matrix(
        cell_adata,
        min_counts=float(min_counts),
        min_genes=int(min_genes),
        min_cells=int(min_cells),
    )
    selected_genes = [str(gene) for gene in cell_adata.var_names.tolist()]
    selected_genes_output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_genes_output_path.write_text("\n".join(selected_genes), encoding="utf-8")

    cell_adata.uns["nmf_source"] = {
        "source_kind": "xenium_union_cell_qc",
        "source_h5ad": str(input_cell_h5ad),
        "cell_instance_definition": "union(cell_boundaries,nucleus_boundaries)",
        **qc_summary,
    }
    output_cell_h5ad.parent.mkdir(parents=True, exist_ok=True)
    cell_adata.write_h5ad(output_cell_h5ad)
    elapsed = time.perf_counter() - t0
    print(
        f"[preprocess] wrote Xenium union-cell NMF source | output={output_cell_h5ad} "
        f"shape=({cell_adata.n_obs}, {cell_adata.n_vars}) selected_genes={selected_genes_output_path} "
        f"elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return output_cell_h5ad


def _fit_nmf_mu(
    X: sp.csr_matrix,
    n_components: int,
    max_iter: int,
    random_seed: int,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    n_obs, n_vars = X.shape
    W = (rng.random((n_obs, n_components), dtype=np.float32) + 0.1).astype(np.float32)
    H = (rng.random((n_components, n_vars), dtype=np.float32) + 0.1).astype(np.float32)
    total_iters = max(int(max_iter), 1)
    iterator = range(total_iters)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="nmf_mu", unit="iter")

    for iteration in iterator:
        XHt = (X @ H.T).astype(np.float32)
        HHt = (H @ H.T).astype(np.float32)
        W *= XHt / np.maximum(W @ HHt, eps)
        W = np.maximum(W, eps)

        WtX = (W.T @ X).astype(np.float32)
        WtW = (W.T @ W).astype(np.float32)
        H *= np.asarray(WtX) / np.maximum(WtW @ H, eps)
        H = np.maximum(H, eps)

        if tqdm is None and (iteration == 0 or (iteration + 1) % 50 == 0 or (iteration + 1) == total_iters):
            print(f"[preprocess] nmf_mu iter={iteration + 1}/{total_iters}")

    return W.astype(np.float32, copy=False), H.astype(np.float32, copy=False)


def _fit_cell_nmf(
    raw_cell_adata: ad.AnnData,
    output_cell_h5ad: Path,
    output_module_csv: Path,
    n_components: int,
    max_iter: int,
    random_seed: int,
    cell_id_key: str,
    solver: str = "cd",
    beta_loss: str = "frobenius",
    tol: float = 1e-4,
    alpha_w: float = 0.0,
    alpha_h: str | float = "same",
    l1_ratio: float = 0.0,
    dense_fit: bool = True,
    verbose: int = 1,
) -> str:
    t0 = time.perf_counter()
    raw_cell_adata.var_names_make_unique()
    X_fit_csr = _to_csr_float32(raw_cell_adata.X)
    if X_fit_csr.shape[0] <= 0 or X_fit_csr.shape[1] <= 0:
        raise ValueError("NMF fit source h5ad has empty expression matrix.")
    reconstruction_err: float | None = None

    try:
        from sklearn.decomposition import NMF  # type: ignore

        alpha_h_value: str | float
        if isinstance(alpha_h, str):
            alpha_h_value = alpha_h if alpha_h == "same" else float(alpha_h)
        else:
            alpha_h_value = float(alpha_h)
        use_dense_fit = bool(dense_fit) and str(solver) == "cd" and str(beta_loss) == "frobenius"
        X_fit = X_fit_csr
        if use_dense_fit:
            dense_gb = (float(X_fit_csr.shape[0]) * float(X_fit_csr.shape[1]) * 4.0) / (1024.0**3)
            print(
                f"[preprocess] converting NMF fit matrix to dense float32 for CD/Frobenius speed | "
                f"shape=({X_fit_csr.shape[0]}, {X_fit_csr.shape[1]}) estimated_dense_gb={dense_gb:.2f}",
                flush=True,
            )
            X_fit = X_fit_csr.toarray().astype(np.float32, copy=False)

        print(
            f"[preprocess] fitting cell-level NMF via sklearn.decomposition.NMF ... "
            f"fit_shape=({X_fit_csr.shape[0]}, {X_fit_csr.shape[1]}) "
            f"components={n_components} solver={solver} beta_loss={beta_loss} "
            f"max_iter={max_iter} tol={tol} dense_fit={use_dense_fit}",
            flush=True,
        )
        model = NMF(
            n_components=n_components,
            init="nndsvda",
            solver=solver,
            beta_loss=beta_loss,
            tol=float(tol),
            max_iter=max_iter,
            random_state=random_seed,
            alpha_W=float(alpha_w),
            alpha_H=alpha_h_value,
            l1_ratio=float(l1_ratio),
            verbose=int(verbose),
        )
        W_fit = model.fit_transform(X_fit)
        H = model.components_
        reconstruction_err = float(getattr(model, "reconstruction_err_", np.nan))
    except Exception as exc:
        print(f"[preprocess] sklearn NMF unavailable or failed ({exc}); falling back to internal MU solver.")
        W_fit, H = _fit_nmf_mu(X_fit_csr, n_components=n_components, max_iter=max_iter, random_seed=random_seed)

    module_names = [f"module_{idx:02d}" for idx in range(int(n_components))]
    module_df = pd.DataFrame(H.T, index=raw_cell_adata.var_names.astype(str), columns=module_names)
    output_module_csv.parent.mkdir(parents=True, exist_ok=True)
    module_df.to_csv(output_module_csv)
    output_module_csv_t = output_module_csv.with_name(output_module_csv.stem + "_T.csv")
    module_df.T.to_csv(output_module_csv_t)

    latent_key = f"X_nmf_{int(n_components)}"
    out_adata = raw_cell_adata.copy()
    if cell_id_key not in out_adata.obs.columns:
        out_adata.obs[cell_id_key] = out_adata.obs_names.astype(str)
    out_adata.obsm[latent_key] = np.asarray(W_fit, dtype=np.float32)
    out_adata.varm[f"NMF_H_{int(n_components)}"] = np.asarray(H.T, dtype=np.float32)
    out_adata.obs[f"nmf{int(n_components)}_l1_norm"] = np.asarray(W_fit, dtype=np.float32).sum(axis=1).astype(np.float32)
    out_adata.obs[f"nmf{int(n_components)}_dominant_module"] = np.argmax(W_fit, axis=1).astype(np.int32)
    out_adata.uns[f"nmf_{int(n_components)}"] = {
        "n_components": int(n_components),
        "max_iter": int(max_iter),
        "random_seed": int(random_seed),
        "solver": str(solver),
        "beta_loss": str(beta_loss),
        "tol": float(tol),
        "alpha_W": float(alpha_w),
        "alpha_H": str(alpha_h),
        "l1_ratio": float(l1_ratio),
        "reconstruction_err": reconstruction_err,
        "fit_source": str(out_adata.uns.get("nmf_source", {}).get("source_kind", "cell_h5ad")),
    }
    output_cell_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out_adata.write_h5ad(output_cell_h5ad)

    summary_csv = output_module_csv.with_name(f"nmf_summary_{int(n_components)}.csv")
    summary_df = pd.DataFrame(
        [
            {
                "fit_gene_count": int(raw_cell_adata.n_vars),
                "fit_cell_count": int(raw_cell_adata.n_obs),
                "output_cell_count": int(out_adata.n_obs),
                "nmf_components": int(n_components),
                "nmf_solver": str(solver),
                "nmf_beta_loss": str(beta_loss),
                "nmf_max_iter": int(max_iter),
                "nmf_tol": float(tol),
                "nmf_alpha_w": float(alpha_w),
                "nmf_alpha_h": str(alpha_h),
                "nmf_l1_ratio": float(l1_ratio),
                "nmf_dense_fit": bool(dense_fit),
                "nmf_verbose": int(verbose),
                "cell_nmf_strategy": "fit_transform_output",
                "reconstruction_err": reconstruction_err,
                "cell_latent_key": latent_key,
                "module_csv": str(output_module_csv),
                "module_csv_t": str(output_module_csv_t),
                "cell_h5ad": str(output_cell_h5ad),
                "fit_source": str(out_adata.uns.get("nmf_source", {}).get("source_kind", "cell_h5ad")),
            }
        ]
    )
    summary_df.to_csv(summary_csv, index=False)
    elapsed = time.perf_counter() - t0
    print(
        f"[preprocess] wrote cell NMF outputs | cell_h5ad={output_cell_h5ad} "
        f"module_csv={output_module_csv} module_csv_t={output_module_csv_t} "
        f"summary_csv={summary_csv} latent_key={latent_key} elapsed_sec={elapsed:.1f}"
    )
    return latent_key


def _gamma_poisson_degrade_bins(
    unioned_bin_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    output_degraded_h5ad: Path,
    reference_count_key: str,
    batch_size: int,
    random_seed: int,
    reference_label_key: str = "labels_he_expanded",
    theta: float = 5.0,
) -> None:
    t0 = time.perf_counter()
    _ = reference_count_key, batch_size
    if list(map(str, unioned_bin_adata.var_names.tolist())) != list(map(str, reference_adata.var_names.tolist())):
        raise ValueError(
            "degrade requires synthetic Xenium HD and reference HD to have identical gene order. "
            f"synthetic_genes={unioned_bin_adata.n_vars} reference_genes={reference_adata.n_vars}"
        )
    if reference_label_key not in reference_adata.obs.columns:
        raise ValueError(
            f"reference HD is missing obs['{reference_label_key}']; run StarDist/bin2cell expand_labels before degrade."
        )

    labels = pd.to_numeric(reference_adata.obs[reference_label_key], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    fg_mask = labels > 0
    bg_mask = labels <= 0
    if not np.any(fg_mask):
        raise ValueError(f"reference HD obs['{reference_label_key}'] has no positive foreground/cell labels.")
    if not np.any(bg_mask):
        raise ValueError(f"reference HD obs['{reference_label_key}'] has no background/non-cell bins.")

    X_raw = _to_csr_float32(unioned_bin_adata.X)
    raw_depths = _row_sum(X_raw)
    X_ref = _to_csr_float32(reference_adata.X)
    fg_mean = np.asarray(X_ref[fg_mask].mean(axis=0)).reshape(-1).astype(np.float32)
    bg_mean = np.asarray(X_ref[bg_mask].mean(axis=0)).reshape(-1).astype(np.float32)
    src_mean = np.asarray(X_raw.mean(axis=0)).reshape(-1).astype(np.float32)
    scale_factors = np.divide(
        fg_mean,
        np.maximum(src_mean, EPS),
        out=np.zeros_like(fg_mean, dtype=np.float32),
        where=np.isfinite(fg_mean),
    ).astype(np.float32, copy=False)
    scale_factors = np.nan_to_num(scale_factors, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    theta_val = max(float(theta), EPS)
    print(
        f"[preprocess] HD foreground/background gamma-poisson degrade | "
        f"reference_label_key={reference_label_key} fg_bins={int(np.sum(fg_mask))} bg_bins={int(np.sum(bg_mask))} "
        f"theta={theta_val:g} genes={unioned_bin_adata.n_vars}",
        flush=True,
    )

    rng = np.random.default_rng(random_seed)
    n_obs = X_raw.shape[0]
    degraded_depth = np.zeros(n_obs, dtype=np.float32)
    X_csc = X_raw.tocsc(copy=True)
    del X_raw
    gc.collect()

    iterator = range(X_csc.shape[1])
    if tqdm is not None:
        iterator = tqdm(iterator, desc="degrade_genes", unit="gene")

    new_data: list[np.ndarray] = []
    new_indices: list[np.ndarray] = []
    new_indptr = np.zeros(X_csc.shape[1] + 1, dtype=np.int64)
    nnz_total = 0
    for gene_idx in iterator:
        start = int(X_csc.indptr[gene_idx])
        end = int(X_csc.indptr[gene_idx + 1])
        if end <= start:
            new_indptr[gene_idx + 1] = nnz_total
            continue

        rows = X_csc.indices[start:end].astype(np.int32, copy=False)
        vals = X_csc.data[start:end].astype(np.float32, copy=False)
        mu = vals * float(scale_factors[gene_idx]) + float(bg_mean[gene_idx])
        mu = np.clip(mu, 0.0, None).astype(np.float32, copy=False)
        lam = rng.gamma(shape=theta_val, scale=mu / theta_val).astype(np.float32, copy=False)
        sampled = rng.poisson(lam).astype(np.float32, copy=False)
        keep = sampled > 0
        if np.any(keep):
            kept_rows = rows[keep].astype(np.int32, copy=False)
            kept_vals = sampled[keep].astype(np.float32, copy=False)
            new_indices.append(kept_rows.copy())
            new_data.append(kept_vals.copy())
            np.add.at(degraded_depth, kept_rows, kept_vals)
            nnz_total += int(kept_vals.shape[0])
        new_indptr[gene_idx + 1] = nnz_total
        if tqdm is None and (gene_idx == 0 or (gene_idx + 1) % 1000 == 0 or (gene_idx + 1) == X_csc.shape[1]):
            print(f"[preprocess] degraded genes {gene_idx + 1:,}/{X_csc.shape[1]:,}", flush=True)

    if nnz_total > 0:
        data = np.concatenate(new_data).astype(np.float32, copy=False)
        indices = np.concatenate(new_indices).astype(np.int32, copy=False)
    else:
        data = np.empty(0, dtype=np.float32)
        indices = np.empty(0, dtype=np.int32)
    X_degraded = sp.csc_matrix((data, indices, new_indptr), shape=X_csc.shape).tocsr()
    unioned_bin_adata.X = sp.csr_matrix(X_csc.shape, dtype=np.float32)
    del X_csc
    del new_data
    del new_indices
    del data
    del indices
    gc.collect()

    out_obs = unioned_bin_adata.obs.copy()
    out_obs["n_counts"] = raw_depths.astype(np.float32)
    out_obs["n_counts_adjusted"] = degraded_depth.astype(np.float32)
    out_obs["in_tissue"] = np.ones(unioned_bin_adata.n_obs, dtype=np.int32)

    degraded_adata = ad.AnnData(X=X_degraded, obs=out_obs, var=unioned_bin_adata.var.copy(), uns=unioned_bin_adata.uns.copy())
    degraded_adata.uns["degrade_params"] = {
        "method": "hd_foreground_background_gene_mean_gamma_poisson",
        "reference_label_key": str(reference_label_key),
        "theta": float(theta_val),
        "random_seed": int(random_seed),
        "background_mode": "add_bg_to_existing_nonzero_entries_only",
        "foreground_bins": int(np.sum(fg_mask)),
        "background_bins": int(np.sum(bg_mask)),
    }
    output_degraded_h5ad.parent.mkdir(parents=True, exist_ok=True)
    degraded_adata.write_h5ad(output_degraded_h5ad)
    elapsed = time.perf_counter() - t0
    print(f"[preprocess] wrote degraded synthetic HD h5ad: {output_degraded_h5ad} | shape=({degraded_adata.n_obs}, {degraded_adata.n_vars}) elapsed_sec={elapsed:.1f}")


def _default_instance_chunk_manifest_path(output_h5: Path, instance_budget: int) -> Path:
    return output_h5.with_name(f"{output_h5.stem}.instchunk{int(instance_budget)}_train.h5")


def _parse_size_bin_edges(raw_value: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw_value).split(",") if part.strip()]
    if len(values) < 2:
        raise ValueError("instance chunk size-bin-edges must contain at least two values.")
    return values


def _build_region_split(centers: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    ys = centers[:, 0].astype(np.float32, copy=False)
    xs = centers[:, 1].astype(np.float32, copy=False)
    frac = float(val_ratio) ** 0.5
    y_threshold = ys.max() - frac * (ys.max() - ys.min())
    x_threshold = xs.max() - frac * (xs.max() - xs.min())
    val_mask = (ys >= y_threshold) & (xs >= x_threshold)
    val_indices = np.nonzero(val_mask)[0]
    train_indices = np.nonzero(~val_mask)[0]
    if val_indices.size == 0 or train_indices.size == 0:
        raise ValueError("Region split produced an empty train or val partition.")
    return train_indices.astype(np.int64, copy=False), val_indices.astype(np.int64, copy=False)


def _sort_tiles_spatially(tile_indices: np.ndarray, tile_centers: np.ndarray) -> np.ndarray:
    order = np.lexsort((tile_centers[tile_indices, 1], tile_centers[tile_indices, 0]))
    return tile_indices[order].astype(np.int64, copy=False)


def _build_instance_chunks(
    tile_indices: np.ndarray,
    instance_offsets: np.ndarray,
    instance_budget: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    chunk_tile_indices: list[np.ndarray] = []
    chunk_local_indices: list[np.ndarray] = []
    current_tiles: list[int] = []
    current_locals: list[int] = []

    def flush_current() -> None:
        nonlocal current_tiles, current_locals
        if current_tiles:
            chunk_tile_indices.append(np.asarray(current_tiles, dtype=np.int64))
            chunk_local_indices.append(np.asarray(current_locals, dtype=np.int64))
            current_tiles = []
            current_locals = []

    for tile_idx in tile_indices.tolist():
        tile_idx = int(tile_idx)
        tile_start = int(instance_offsets[tile_idx])
        tile_end = int(instance_offsets[tile_idx + 1])
        tile_count = int(tile_end - tile_start)
        if tile_count <= 0:
            continue
        local_indices = rng.permutation(tile_count).astype(np.int64, copy=False)
        cursor = 0
        while cursor < tile_count:
            remaining = instance_budget - len(current_tiles)
            if remaining <= 0:
                flush_current()
                remaining = instance_budget
            take = min(int(remaining), int(tile_count - cursor))
            selected = local_indices[cursor : cursor + take]
            current_tiles.extend([tile_idx] * int(take))
            current_locals.extend(selected.tolist())
            cursor += int(take)
            if len(current_tiles) >= instance_budget:
                flush_current()
    flush_current()
    return chunk_tile_indices, chunk_local_indices


def _create_resizable_dataset(fw: h5py.File, name: str, sample: np.ndarray, dtype=None) -> h5py.Dataset:
    sample = np.asarray(sample)
    shape = (0,) + sample.shape[1:]
    maxshape = (None,) + sample.shape[1:]
    chunks = (max(1, min(128, sample.shape[0])),) + sample.shape[1:] if sample.shape[0] > 0 else (1,) + sample.shape[1:]
    return fw.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype or sample.dtype)


def _append_array_to_dataset(ds: h5py.Dataset, array: np.ndarray) -> None:
    array = np.asarray(array)
    old = ds.shape[0]
    new = old + array.shape[0]
    ds.resize((new,) + ds.shape[1:])
    ds[old:new] = array


def _write_instance_chunk_split(
    fw: h5py.File,
    prefix: str,
    source_dataset,
    chunk_tile_indices: list[np.ndarray],
    chunk_local_indices: list[np.ndarray],
) -> tuple[int, int]:
    chunk_tile_offsets = [0]
    chunk_instance_offsets = [0]
    datasets: dict[str, h5py.Dataset] = {}
    total_tiles = 0
    total_instances = 0

    for chunk_idx, (tile_ids_arr, local_ids_arr) in enumerate(zip(chunk_tile_indices, chunk_local_indices, strict=False)):
        tile_to_locals: dict[int, list[int]] = {}
        tile_order: list[int] = []
        for tile_idx, local_idx in zip(tile_ids_arr.tolist(), local_ids_arr.tolist(), strict=False):
            tile_idx = int(tile_idx)
            if tile_idx not in tile_to_locals:
                tile_to_locals[tile_idx] = []
                tile_order.append(tile_idx)
            tile_to_locals[tile_idx].append(int(local_idx))

        tile_inputs: list[np.ndarray] = []
        tile_indices_pool: list[np.ndarray] = []
        tile_centers_pool: list[np.ndarray] = []
        instance_tile_ptr_pool: list[np.ndarray] = []
        instance_centers_pool: list[np.ndarray] = []
        instance_seed_nmfs_pool: list[np.ndarray] = []
        instance_nucleus_areas_pool: list[np.ndarray] = []
        instance_neighbor_seed_nmfs_pool: list[np.ndarray] = []
        instance_neighbor_nucleus_areas_pool: list[np.ndarray] = []
        instance_neighbor_positions_pool: list[np.ndarray] = []
        instance_neighbor_valid_pool: list[np.ndarray] = []
        instance_ellipse_targets_pool: list[np.ndarray] = []
        instance_boundary_radius_targets_pool: list[np.ndarray] = []
        instance_boundary_fourier_targets_pool: list[np.ndarray] = []
        instance_mask_targets_pool: list[np.ndarray] = []
        instance_mask_areas_pool: list[np.ndarray] = []
        instance_size_bin_ids_pool: list[np.ndarray] = []
        instance_latent_targets_pool: list[np.ndarray] = []

        for local_tile_ptr, tile_idx in enumerate(tile_order):
            item = source_dataset._get_tile_item(tile_idx, np.asarray(tile_to_locals[tile_idx], dtype=np.int64))
            count = int(item["instance_count"])
            tile_inputs.append(item["tile_input"].numpy()[None, ...])
            tile_indices_pool.append(np.asarray([int(item["tile_index"])], dtype=np.int64))
            tile_centers_pool.append(item["tile_center_yx"].numpy()[None, ...].astype(np.float32, copy=False))
            instance_tile_ptr_pool.append(np.full((count,), local_tile_ptr, dtype=np.int64))
            instance_centers_pool.append(item["instance_centers_yx"].numpy().astype(np.float32, copy=False))
            instance_seed_nmfs_pool.append(item["instance_seed_nmfs"].numpy().astype(np.float32, copy=False))
            instance_nucleus_areas_pool.append(item["instance_nucleus_areas"].numpy().astype(np.float32, copy=False))
            instance_neighbor_seed_nmfs_pool.append(item["instance_neighbor_seed_nmfs"].numpy().astype(np.float32, copy=False))
            instance_neighbor_nucleus_areas_pool.append(item["instance_neighbor_nucleus_areas"].numpy().astype(np.float32, copy=False))
            instance_neighbor_positions_pool.append(item["instance_neighbor_positions"].numpy().astype(np.float32, copy=False))
            instance_neighbor_valid_pool.append(item["instance_neighbor_valid"].numpy().astype(np.float32, copy=False))
            instance_ellipse_targets_pool.append(item["instance_ellipse_targets"].numpy().astype(np.float32, copy=False))
            instance_boundary_radius_targets_pool.append(item["instance_boundary_radius_targets"].numpy().astype(np.float32, copy=False))
            instance_boundary_fourier_targets_pool.append(item["instance_boundary_fourier_targets"].numpy().astype(np.float32, copy=False))
            instance_mask_targets_pool.append(item["instance_mask_targets"].numpy().astype(np.float32, copy=False))
            instance_mask_areas_pool.append(item["instance_mask_areas"].numpy().astype(np.float32, copy=False))
            instance_size_bin_ids_pool.append(item["instance_size_bin_ids"].numpy().astype(np.int64, copy=False))
            instance_latent_targets_pool.append(item["instance_latent_targets"].numpy().astype(np.float32, copy=False))

        arrays = {
            f"{prefix}_tile_input_pool": np.concatenate(tile_inputs, axis=0),
            f"{prefix}_tile_index_pool": np.concatenate(tile_indices_pool, axis=0),
            f"{prefix}_tile_center_yx_pool": np.concatenate(tile_centers_pool, axis=0),
            f"{prefix}_instance_tile_ptr_pool": np.concatenate(instance_tile_ptr_pool, axis=0),
            f"{prefix}_instance_centers_yx_pool": np.concatenate(instance_centers_pool, axis=0),
            f"{prefix}_instance_seed_nmfs_pool": np.concatenate(instance_seed_nmfs_pool, axis=0),
            f"{prefix}_instance_nucleus_areas_pool": np.concatenate(instance_nucleus_areas_pool, axis=0),
            f"{prefix}_instance_neighbor_seed_nmfs_pool": np.concatenate(instance_neighbor_seed_nmfs_pool, axis=0),
            f"{prefix}_instance_neighbor_nucleus_areas_pool": np.concatenate(instance_neighbor_nucleus_areas_pool, axis=0),
            f"{prefix}_instance_neighbor_positions_pool": np.concatenate(instance_neighbor_positions_pool, axis=0),
            f"{prefix}_instance_neighbor_valid_pool": np.concatenate(instance_neighbor_valid_pool, axis=0),
            f"{prefix}_instance_ellipse_targets_pool": np.concatenate(instance_ellipse_targets_pool, axis=0),
            f"{prefix}_instance_boundary_radius_targets_pool": np.concatenate(instance_boundary_radius_targets_pool, axis=0),
            f"{prefix}_instance_boundary_fourier_targets_pool": np.concatenate(instance_boundary_fourier_targets_pool, axis=0),
            f"{prefix}_instance_mask_targets_pool": np.concatenate(instance_mask_targets_pool, axis=0),
            f"{prefix}_instance_mask_areas_pool": np.concatenate(instance_mask_areas_pool, axis=0),
            f"{prefix}_instance_size_bin_ids_pool": np.concatenate(instance_size_bin_ids_pool, axis=0),
            f"{prefix}_instance_latent_targets_pool": np.concatenate(instance_latent_targets_pool, axis=0),
        }

        if not datasets:
            for name, arr in arrays.items():
                datasets[name] = _create_resizable_dataset(fw, name, arr)

        tile_base = total_tiles
        arrays[f"{prefix}_instance_tile_ptr_pool"] = arrays[f"{prefix}_instance_tile_ptr_pool"] + tile_base

        for name, arr in arrays.items():
            _append_array_to_dataset(datasets[name], arr)

        total_tiles += int(arrays[f"{prefix}_tile_input_pool"].shape[0])
        total_instances += int(arrays[f"{prefix}_instance_centers_yx_pool"].shape[0])
        chunk_tile_offsets.append(total_tiles)
        chunk_instance_offsets.append(total_instances)

    fw.create_dataset(f"{prefix}_chunk_tile_offsets", data=np.asarray(chunk_tile_offsets, dtype=np.int64))
    fw.create_dataset(f"{prefix}_chunk_instance_offsets", data=np.asarray(chunk_instance_offsets, dtype=np.int64))
    fw.attrs[f"n_{prefix}_chunks"] = int(len(chunk_tile_indices))
    fw.attrs[f"n_{prefix}_instances"] = int(total_instances)
    return total_tiles, total_instances


def _build_instance_chunk_manifest_h5(
    *,
    input_h5: Path,
    output_h5: Path,
    instance_budget: int,
    min_nucleus_bins: int,
    val_ratio: float,
    split_mode: str,
    seed: int,
    canvas_size: int,
    neighbor_k: int,
    aggregate_radius: int,
    seed_size: int = 5,
    size_bin_edges: str = "0,16,24,32,40,48,64,96,999999",
) -> None:
    model_root = ROOT.parent / "model"
    inserted = False
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
        inserted = True
    try:
        from dataset import SpatialTranscriptomicsDataset
    finally:
        if inserted:
            sys.path.pop(0)

    if not input_h5.exists():
        raise FileNotFoundError(f"Input H5 not found: {input_h5}")
    if instance_budget <= 0:
        raise ValueError("instance_budget must be positive.")
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("instance_chunk_val_ratio must be between 0 and 1.")

    parsed_size_bin_edges = _parse_size_bin_edges(size_bin_edges)
    source_dataset = SpatialTranscriptomicsDataset(
        data_dir=input_h5,
        min_nuc=min_nucleus_bins,
        seed_size=seed_size,
        canvas_size=canvas_size,
        neighbor_k=neighbor_k,
        aggregate_radius=aggregate_radius,
        size_bin_edges=parsed_size_bin_edges,
        chunk_manifest=None,
        chunk_split="full",
    )

    source_h5 = source_dataset._get_h5()
    instance_offsets = source_h5["instance_offsets"][:].astype(np.int64, copy=False)
    tile_centers = source_h5["tile_center_yx"][:].astype(np.int64, copy=False)
    tile_indices = np.asarray(source_dataset.source_tile_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if split_mode == "random":
        shuffled = tile_indices.copy()
        rng.shuffle(shuffled)
        val_size = max(1, int(round(shuffled.shape[0] * float(val_ratio))))
        val_size = min(val_size, shuffled.shape[0] - 1)
        val_tiles = np.sort(shuffled[:val_size])
        train_tiles = np.sort(shuffled[val_size:])
    else:
        train_pos, val_pos = _build_region_split(tile_centers, val_ratio=float(val_ratio))
        train_tiles = np.sort(tile_indices[train_pos])
        val_tiles = np.sort(tile_indices[val_pos])

    train_tiles_sorted = _sort_tiles_spatially(train_tiles, tile_centers)
    val_tiles_sorted = _sort_tiles_spatially(val_tiles, tile_centers)

    train_chunks = _build_instance_chunks(train_tiles_sorted, instance_offsets, int(instance_budget), int(seed))
    val_chunks = _build_instance_chunks(val_tiles_sorted, instance_offsets, int(instance_budget), int(seed) + 1)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as fw:
        fw.attrs["dataset_format"] = "instance_chunk_h5_v1"
        fw.attrs["source_h5"] = str(input_h5)
        fw.attrs["instance_budget"] = int(instance_budget)
        fw.attrs["min_nuc"] = int(min_nucleus_bins)
        fw.attrs["val_ratio"] = float(val_ratio)
        fw.attrs["split_mode"] = str(split_mode)
        fw.attrs["seed"] = int(seed)
        fw.attrs["canvas_size"] = int(canvas_size)
        fw.attrs["aggregate_radius"] = int(aggregate_radius)
        fw.attrs["patch_size"] = int(source_dataset.patch_size)
        fw.attrs["latent_dim"] = int(source_dataset.latent_dim)
        fw.attrs["expr_channels"] = int(source_dataset.expr_channels)
        fw.attrs["seed_nmf_dim"] = int(source_dataset.seed_nmf_dim)
        fw.attrs["neighbor_k"] = int(source_dataset.neighbor_k)
        fw.attrs["n_source_tiles"] = int(source_dataset.kept_samples)
        fw.attrs["n_source_instances"] = int(source_dataset.total_instances)

        train_tiles_written, train_instances_written = _write_instance_chunk_split(
            fw=fw,
            prefix="train",
            source_dataset=source_dataset,
            chunk_tile_indices=train_chunks[0],
            chunk_local_indices=train_chunks[1],
        )
        val_tiles_written, val_instances_written = _write_instance_chunk_split(
            fw=fw,
            prefix="val",
            source_dataset=source_dataset,
            chunk_tile_indices=val_chunks[0],
            chunk_local_indices=val_chunks[1],
        )
        fw.attrs["n_train_tiles_written"] = int(train_tiles_written)
        fw.attrs["n_val_tiles_written"] = int(val_tiles_written)
        fw.attrs["n_train_instances_written"] = int(train_instances_written)
        fw.attrs["n_val_instances_written"] = int(val_instances_written)

    print(
        f"[preprocess] instance chunk manifest built | input={input_h5} output={output_h5} "
        f"train_chunks={len(train_chunks[0])} val_chunks={len(val_chunks[0])} "
        f"train_instances={train_instances_written} val_instances={val_instances_written} "
        f"budget={instance_budget}"
    )


def _add_train_downstream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--stop-after", type=str, default=None)
    parser.add_argument("--sample-type", type=str, default="OV")
    parser.add_argument("--sample-type-id", type=int, default=0)
    parser.add_argument("--tile-patch-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--min-nucleus-bins", type=int, default=4)
    parser.add_argument("--min-cell-bins", type=int, default=9)
    parser.add_argument("--mu-iters", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--filter-neighbor-threshold", type=int, default=7)
    parser.add_argument("--filter-max-fill-iters", type=int, default=4)
    parser.add_argument("--superellipse-p-candidates", type=str, default="1.5,2.0,2.5,3.0")
    parser.add_argument("--binary-search-iters", type=int, default=18)
    parser.add_argument("--direction-bins", type=int, default=8, help="Deprecated compatibility flag for old direct-radii pipeline.")
    parser.add_argument("--max-relative-residual", type=float, default=0.25, help="Deprecated compatibility flag; retained for backward compatibility.")
    parser.add_argument("--boundary-samples", type=int, default=64)
    parser.add_argument("--fourier-order", type=int, default=3)
    parser.add_argument("--max-log-scale", type=float, default=0.45)
    parser.add_argument("--neighbor-k", type=int, default=4)
    parser.add_argument("--aggregate-radius", type=int, default=5)
    parser.add_argument("--canvas-size", type=int, default=24)
    parser.add_argument("--seed-sector-bins", type=int, default=8)
    parser.add_argument("--neighbor-direction-bins", type=int, default=8)
    parser.add_argument("--seed-patch-size", type=int, default=11)
    parser.add_argument("--instance-chunk-manifest", type=Path, default=None)
    parser.add_argument("--instance-budget", type=int, default=512)
    parser.add_argument("--instance-chunk-val-ratio", type=float, default=0.1)
    parser.add_argument("--instance-chunk-split-mode", type=str, default="random", choices=["random", "region"])
    parser.add_argument("--instance-chunk-seed", type=int, default=42)
    parser.add_argument("--instance-chunk-size-bin-edges", type=str, default="0,16,24,32,40,48,64,96,999999")
    parser.add_argument("--mask-refine-output-h5", type=Path, default=None)
    parser.add_argument("--mask-refine-hole-min-area", type=int, default=3)
    parser.add_argument("--mask-refine-hole-area-frac", type=float, default=0.05)
    parser.add_argument("--mask-refine-area-tol-frac", type=float, default=0.10)
    parser.add_argument("--mask-refine-match-radius", type=float, default=1.5)
    parser.add_argument("--mask-refine-nmf-completeness-drop-tol", type=float, default=0.05)
    parser.add_argument("--mask-refine-nmf-sim-drop-tol", type=float, default=0.05)
    parser.add_argument("--mask-refine-num-workers", type=int, default=max(1, min(8, (os.cpu_count() or 1))))
    parser.add_argument("--mask-refine-direct-block-size", type=int, default=2048)
    parser.add_argument("--mask-refine-overwrite", action="store_true")
    parser.add_argument("--regularize-masks", action="store_true", help="Run area-preserving superellipse mask regularization before downstream boundary-target / feature caching.")
    parser.add_argument("--regularize-promote", action="store_true", help="Promote regularized mask / x_low / center datasets back onto xenium_mask_pool / x_low / cell_centers_yx_pool so downstream steps use them automatically.")
    parser.add_argument("--regularize-report-dir", type=Path, default=None)
    parser.add_argument("--regularize-add-cosine-threshold", type=float, default=0.85)
    parser.add_argument("--regularize-remove-cosine-threshold", type=float, default=0.75)
    parser.add_argument("--regularize-nmf-coverage-threshold", type=float, default=0.85)
    parser.add_argument("--regularize-swap-max-distance", type=float, default=2.0)
    parser.add_argument("--regularize-swap-min-gain", type=float, default=-0.02)
    parser.add_argument("--regularize-prototype-core-radius", type=float, default=3.0)
    parser.add_argument("--regularize-area-tolerance-frac", type=float, default=0.05)
    parser.add_argument("--regularize-bridge-closing-iters", type=int, default=1)
    parser.add_argument("--regularize-hole-area-frac", type=float, default=0.02)
    parser.add_argument("--regularize-hole-min-area", type=int, default=2)
    parser.add_argument("--regularize-size-bin-edges", type=str, default="0,16,24,32,40,48,64,96,999999")
    parser.add_argument("--regularize-num-workers", type=int, default=8)
    parser.add_argument("--visualize-mask-compare-dir", type=Path, default=None)
    parser.add_argument("--visualize-num-tiles", type=int, default=10)
    parser.add_argument("--visualize-select-mode", type=str, default="random", choices=["random", "top"])
    parser.add_argument("--visualize-seed", type=int, default=42)



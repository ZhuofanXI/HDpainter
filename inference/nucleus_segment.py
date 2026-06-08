from __future__ import annotations

import argparse
import gc
import gzip
import inspect
import json
import os
import sys
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Visium HD bin h5ad, run bin2cell-style QC/destripe/mpp image "
            "preparation, then add StarDist and Cellpose-SAM nucleus labels."
        )
    )
    parser.add_argument(
        "--bin2cell-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the bin2cell repo root, package directory, or bin2cell.py. "
            "If omitted, the installed bin2cell package is used."
        ),
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=None,
        help="Existing bin-level h5ad. If provided, raw Space Ranger reading is skipped.",
    )
    parser.add_argument(
        "--spaceranger-dir",
        type=Path,
        default=None,
        help="Space Ranger Visium HD output directory, e.g. square_002um.",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=None,
        help="10x matrix directory fallback, e.g. filtered_feature_bc_matrix.",
    )
    parser.add_argument(
        "--count-file",
        type=str,
        default="filtered_feature_bc_matrix.h5",
        help="10x h5 count file name inside --spaceranger-dir when available.",
    )
    parser.add_argument(
        "--spatial-dir",
        type=Path,
        default=None,
        help="Space Ranger spatial directory. Defaults to <spaceranger-dir>/spatial.",
    )
    parser.add_argument(
        "--source-image-path",
        "--image-path",
        dest="source_image_path",
        type=Path,
        required=True,
        help="Full-resolution H&E image used by Space Ranger.",
    )
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument(
        "--working-h5ad-name",
        type=str,
        default="01_read_qc_destriped.h5ad",
        help="Intermediate h5ad written immediately after read/QC/destripe.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=None,
        help="Final h5ad path. Defaults to <out-dir>/nucleus_segmented.h5ad.",
    )
    parser.add_argument("--library-id", type=str, default="Visium_HD")
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--min-counts", type=int, default=1)
    parser.add_argument("--destripe-quantile", type=float, default=0.99)
    parser.add_argument("--mpp", type=float, default=0.4)
    parser.add_argument("--buffer", type=int, default=150)
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop the morphology image before segmentation.",
    )
    parser.add_argument("--stardist-model", type=str, default="2D_versatile_he")
    parser.add_argument("--stardist-block-size", type=int, default=4096)
    parser.add_argument("--stardist-min-overlap", type=int, default=128)
    parser.add_argument("--stardist-context", type=int, default=128)
    parser.add_argument("--stardist-prob-thresh", type=float, default=None)
    parser.add_argument("--stardist-nms-thresh", type=float, default=None)
    parser.add_argument("--cellpose-model", type=str, default="cpsam")
    parser.add_argument("--cellpose-diameter", type=float, default=None)
    parser.add_argument("--cellpose-flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellpose-cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--cellpose-batch-size", type=int, default=8)
    parser.add_argument("--cellpose-tile-overlap", type=float, default=0.1)
    parser.add_argument(
        "--cellpose-chunk-size",
        type=int,
        default=3072,
        help="Run Cellpose-SAM on image chunks of this size. Set <=0 to disable chunked mode.",
    )
    parser.add_argument(
        "--cellpose-chunk-overlap",
        type=int,
        default=256,
        help="Pixel overlap between adjacent Cellpose-SAM chunks.",
    )
    parser.add_argument(
        "--cellpose-grayscale",
        action="store_true",
        help="Convert the mpp-scaled H&E image to grayscale before Cellpose-SAM.",
    )
    parser.add_argument("--cellpose-gpu", action="store_true")
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing out_dir checkpoints and rerun from raw data/QC.",
    )
    parser.add_argument(
        "--rerun-stardist",
        action="store_true",
        help="Rerun StarDist even if the matching StarDist npz or obs['stardist_id'] already exists.",
    )
    parser.add_argument(
        "--rerun-cellpose",
        action="store_true",
        help="Rerun Cellpose-SAM even if the matching Cellpose-SAM npz or obs['cellpose_id'] already exists.",
    )
    return parser.parse_args()


def log_step(message: str) -> None:
    print(f"[nucleus_segment] {message}", flush=True)


STARDIST_OBS_KEY = "stardist_id"
CELLPOSE_OBS_KEY = "cellpose_id"
PREPROCESS_STARDIST_ALIAS_KEYS = (
    "labels_he",
    "labels_he_expanded",
)


@contextmanager
def heartbeat_progress(desc: str, interval: float = 1.0):
    stop_event = threading.Event()
    bar: tqdm | None = None

    def worker() -> None:
        nonlocal bar
        bar = tqdm(total=None, desc=desc, unit="s", dynamic_ncols=True)
        while not stop_event.wait(interval):
            bar.update(1)
        bar.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()


def import_bin2cell(bin2cell_path: Path | None):
    log_step("Importing bin2cell")
    if bin2cell_path is not None:
        path = bin2cell_path.resolve()
        log_step(f"Using bin2cell path: {path}")
        if path.is_file() and path.name == "bin2cell.py":
            sys.path.insert(0, str(path.parent.parent))
        elif (path / "bin2cell.py").exists():
            sys.path.insert(0, str(path.parent))
        elif (path / "bin2cell" / "__init__.py").exists():
            sys.path.insert(0, str(path))
        else:
            sys.path.insert(0, str(path))
    try:
        import bin2cell as b2c
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Could not import bin2cell. Install bin2cell or pass --bin2cell-path "
            "to the repo root/package directory/bin2cell.py."
        ) from exc
    patch_bin2cell_image_loader(b2c)
    log_step("bin2cell imported and image-loader fallback patched")
    return b2c


def patch_bin2cell_image_loader(b2c: Any) -> None:
    original_load_image = b2c.load_image
    original_shape_check = b2c.actual_vs_inferred_image_shape

    def load_image_with_tiff_fallback(image_path: str | Path, gray: bool = False, dtype=np.uint8):
        try:
            img = original_load_image(str(image_path), gray=gray, dtype=dtype)
            if img is not None:
                return img
        except Exception as exc:
            print(f"bin2cell.load_image failed, trying tifffile/imageio fallback: {exc}")

        try:
            import tifffile

            img = tifffile.imread(str(image_path))
        except Exception:
            try:
                import imageio.v3 as iio

                img = iio.imread(str(image_path))
            except Exception as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    f"Could not read image {image_path}. Install tifffile or imageio."
                ) from exc

        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img[:3], 0, -1)
        if img.ndim == 2 and not gray:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if gray and img.ndim == 3:
            try:
                import cv2

                img = cv2.cvtColor(img[..., :3].astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
            except Exception:
                img = img[..., :3].mean(axis=-1)
        if dtype == np.uint8:
            return to_uint8_image(img)
        return img.astype(dtype, copy=False)

    b2c.load_image = load_image_with_tiff_fallback
    for fn_name in ("scaled_he_image", "stardist"):
        fn = getattr(b2c, fn_name, None)
        if fn is not None and hasattr(fn, "__globals__"):
            fn.__globals__["load_image"] = load_image_with_tiff_fallback

    def shape_check_with_missing_hires_guard(adata: ad.AnnData, img: np.ndarray, ratio_threshold: float = 0.99):
        library = list(adata.uns["spatial"].keys())[0]
        images = adata.uns["spatial"][library].get("images", {})
        if "hires" not in images:
            print("Skipping bin2cell image shape check because tissue_hires_image is not available.")
            return
        return original_shape_check(adata, img, ratio_threshold=ratio_threshold)

    b2c.actual_vs_inferred_image_shape = shape_check_with_missing_hires_guard
    fn = getattr(b2c, "scaled_he_image", None)
    if fn is not None and hasattr(fn, "__globals__"):
        fn.__globals__["actual_vs_inferred_image_shape"] = shape_check_with_missing_hires_guard


def _open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exists: " + ", ".join(str(path) for path in paths))


def to_uint8_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    values = img.astype(np.float32, copy=False)
    if values.ndim == 3:
        out = np.empty(values.shape, dtype=np.uint8)
        for channel_idx in range(values.shape[-1]):
            channel = values[..., channel_idx]
            finite = np.isfinite(channel)
            if not finite.any():
                raise ValueError(f"Image channel {channel_idx} contains no finite values.")
            lo, hi = np.percentile(channel[finite], [1, 99])
            if hi <= lo:
                lo = float(channel[finite].min())
                hi = float(channel[finite].max())
            scaled = np.clip((channel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            out[..., channel_idx] = (scaled * 255.0).astype(np.uint8)
        return out

    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Image contains no finite values.")
    lo, hi = np.percentile(values[finite], [1, 99])
    if hi <= lo:
        lo = float(values[finite].min())
        hi = float(values[finite].max())
    values = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (values * 255.0).astype(np.uint8)


def read_10x_matrix_dir(matrix_dir: Path) -> ad.AnnData:
    log_step(f"Reading 10x matrix directory: {matrix_dir}")
    try:
        import scanpy as sc

        with heartbeat_progress("scanpy.read_10x_mtx"):
            adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", make_unique=True)
        log_step(f"Loaded matrix with {adata.n_obs} bins and {adata.n_vars} genes")
        return adata
    except Exception as exc:
        log_step(f"scanpy.read_10x_mtx failed, using manual mtx reader: {exc}")

    matrix_path = _first_existing([matrix_dir / "matrix.mtx.gz", matrix_dir / "matrix.mtx"])
    barcodes_path = _first_existing([matrix_dir / "barcodes.tsv.gz", matrix_dir / "barcodes.tsv"])
    features_path = _first_existing(
        [
            matrix_dir / "features.tsv.gz",
            matrix_dir / "features.tsv",
            matrix_dir / "genes.tsv.gz",
            matrix_dir / "genes.tsv",
        ]
    )

    log_step(f"Reading matrix file: {matrix_path}")
    with heartbeat_progress("manual matrix.mtx read"):
        if matrix_path.suffix == ".gz":
            with gzip.open(matrix_path, "rb") as fr:
                matrix = sio.mmread(fr).tocsr().astype(np.float32).T
        else:
            matrix = sio.mmread(str(matrix_path)).tocsr().astype(np.float32).T

    log_step(f"Reading barcodes: {barcodes_path}")
    with _open_text_auto(barcodes_path) as fr:
        barcodes = [line.strip() for line in fr if line.strip()]

    log_step(f"Reading features: {features_path}")
    features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
    gene_ids = features.iloc[:, 0].astype(str).to_numpy()
    gene_names = features.iloc[:, 1].astype(str).to_numpy() if features.shape[1] >= 2 else gene_ids.copy()
    expected_shape = (len(barcodes), len(gene_names))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match "
            f"{len(barcodes)} barcodes and {len(gene_names)} genes."
        )

    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_name"))
    var["gene_ids"] = gene_ids
    if features.shape[1] >= 3:
        var["feature_types"] = features.iloc[:, 2].astype(str).to_numpy()
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    log_step(f"Loaded matrix with {adata.n_obs} bins and {adata.n_vars} genes")
    return adata


def read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    log_step(f"Resolving tissue positions in: {spatial_dir}")
    path = _first_existing(
        [
            spatial_dir / "tissue_positions.parquet",
            spatial_dir / "tissue_positions.csv",
            spatial_dir / "tissue_positions_list.csv",
        ]
    )
    if path.suffix.lower() == ".parquet":
        positions = pd.read_parquet(path)
    else:
        has_header = path.name == "tissue_positions.csv"
        positions = pd.read_csv(path, header=0 if has_header else None)
        if not has_header:
            positions.columns = [
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ]
    if "barcode" not in positions.columns:
        positions = positions.rename(columns={positions.columns[0]: "barcode"})
    log_step(f"Loaded tissue positions: {path}")
    return positions.set_index("barcode", drop=True)


def add_spatial_metadata(
    adata: ad.AnnData,
    spatial_dir: Path,
    source_image_path: Path,
    library_id: str,
) -> ad.AnnData:
    log_step("Adding Space Ranger spatial metadata to AnnData")
    positions = read_tissue_positions(spatial_dir)
    adata.obs = adata.obs.join(positions, how="left")
    required = ["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    missing = [col for col in required if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing spatial position columns: {missing}")
    adata = adata[adata.obs["array_row"].notna() & adata.obs["array_col"].notna()].copy()

    adata.obs["array_row"] = adata.obs["array_row"].astype(int)
    adata.obs["array_col"] = adata.obs["array_col"].astype(int)
    if "in_tissue" in adata.obs:
        adata.obs["in_tissue"] = adata.obs["in_tissue"].astype(int)
    else:
        adata.obs["in_tissue"] = 1

    # Keep raw pixel columns in obs for debugging coordinate issues, while
    # obsm['spatial'] follows bin2cell's image x/y convention.
    adata.obs["orig_pxl_row_in_fullres"] = adata.obs["pxl_row_in_fullres"].astype(float)
    adata.obs["orig_pxl_col_in_fullres"] = adata.obs["pxl_col_in_fullres"].astype(float)
    adata.obsm["spatial"] = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()

    scalefactors_path = spatial_dir / "scalefactors_json.json"
    if not scalefactors_path.exists():
        raise FileNotFoundError(f"Missing Space Ranger scalefactors file: {scalefactors_path}")
    log_step(f"Reading scalefactors: {scalefactors_path}")
    with scalefactors_path.open("r", encoding="utf-8") as fr:
        scalefactors = json.load(fr)
    images: dict[str, np.ndarray] = {}
    for key, name in (("hires", "tissue_hires_image.png"), ("lowres", "tissue_lowres_image.png")):
        image_path = spatial_dir / name
        if image_path.exists():
            try:
                import imageio.v3 as iio

                images[key] = np.asarray(iio.imread(image_path))
            except Exception:
                pass

    adata.uns["spatial"] = {
        library_id: {
            "images": images,
            "scalefactors": scalefactors,
            "metadata": {"source_image_path": str(source_image_path.resolve())},
        }
    }
    log_step(f"Spatial metadata added; remaining bins with coordinates: {adata.n_obs}")
    return adata


def read_hd_adata(args: argparse.Namespace, b2c: Any) -> ad.AnnData:
    if args.input_h5ad is not None:
        log_step(f"Reading existing input h5ad: {args.input_h5ad}")
        with heartbeat_progress("read input h5ad"):
            adata = ad.read_h5ad(args.input_h5ad)
        ensure_source_image_metadata(adata, args.source_image_path, args.library_id)
        log_step(f"Loaded input h5ad with {adata.n_obs} bins and {adata.n_vars} genes")
        return adata

    if args.spaceranger_dir is None and args.matrix_dir is None:
        raise ValueError("Provide either --input-h5ad or --spaceranger-dir/--matrix-dir.")

    spaceranger_dir = args.spaceranger_dir
    spatial_dir = args.spatial_dir or (spaceranger_dir / "spatial" if spaceranger_dir else None)
    if spatial_dir is None:
        raise ValueError("Provide --spatial-dir when --spaceranger-dir is not provided.")

    count_path = spaceranger_dir / args.count_file if spaceranger_dir is not None else None
    if count_path is not None and count_path.is_file() and count_path.suffix == ".h5":
        try:
            log_step(f"Reading Space Ranger h5 with bin2cell.read_visium: {count_path}")
            with heartbeat_progress("bin2cell.read_visium"):
                adata = b2c.read_visium(
                    spaceranger_dir,
                    count_file=args.count_file,
                    library_id=args.library_id,
                    source_image_path=args.source_image_path,
                    spaceranger_image_path=spatial_dir,
                )
            log_step(f"Loaded Visium HD h5 with {adata.n_obs} bins and {adata.n_vars} genes")
            return adata
        except Exception as exc:
            log_step(f"bin2cell.read_visium failed, trying scanpy.read_10x_h5 fallback: {exc}")
            try:
                import scanpy as sc

                with heartbeat_progress("scanpy.read_10x_h5"):
                    adata = sc.read_10x_h5(count_path)
                adata.var_names_make_unique()
                return add_spatial_metadata(adata, spatial_dir, args.source_image_path, args.library_id)
            except Exception as exc2:
                log_step(f"scanpy.read_10x_h5 fallback failed; falling back to matrix-dir reader: {exc2}")

    matrix_dir = args.matrix_dir
    if matrix_dir is None:
        if count_path is not None and count_path.is_dir():
            matrix_dir = count_path
        elif spaceranger_dir is not None and (spaceranger_dir / "filtered_feature_bc_matrix").exists():
            matrix_dir = spaceranger_dir / "filtered_feature_bc_matrix"
    if matrix_dir is None:
        raise FileNotFoundError("Could not resolve a 10x matrix directory. Pass --matrix-dir explicitly.")

    adata = read_10x_matrix_dir(matrix_dir)
    return add_spatial_metadata(adata, spatial_dir, args.source_image_path, args.library_id)


def ensure_source_image_metadata(adata: ad.AnnData, source_image_path: Path, library_id: str) -> None:
    log_step("Checking existing h5ad spatial metadata")
    if "spatial" not in adata.obsm:
        raise ValueError("--input-h5ad must contain adata.obsm['spatial'].")
    if "spatial" not in adata.uns or not isinstance(adata.uns["spatial"], dict):
        adata.uns["spatial"] = {}
    if not adata.uns["spatial"]:
        adata.uns["spatial"][library_id] = {"images": {}, "scalefactors": {}, "metadata": {}}
    library = library_id if library_id in adata.uns["spatial"] else list(adata.uns["spatial"].keys())[0]
    adata.uns["spatial"][library].setdefault("images", {})
    adata.uns["spatial"][library].setdefault("scalefactors", {})
    adata.uns["spatial"][library].setdefault("metadata", {})
    adata.uns["spatial"][library]["metadata"]["source_image_path"] = str(source_image_path.resolve())
    if "microns_per_pixel" not in adata.uns["spatial"][library]["scalefactors"]:
        raise ValueError(
            "--input-h5ad must contain uns['spatial'][library]['scalefactors']['microns_per_pixel'] "
            "for bin2cell mpp image generation."
        )


def total_counts(x: Any) -> np.ndarray:
    log_step("Computing total counts per bin")
    counts = np.asarray(x.sum(axis=1)).reshape(-1)
    return counts.astype(np.float64, copy=False)


def get_crop_tag(args: argparse.Namespace) -> str:
    return "nocrop" if args.no_crop else f"buffer{args.buffer}"


def get_scaled_image_path(args: argparse.Namespace) -> Path:
    return args.out_dir / f"he_mpp{args.mpp:g}_{get_crop_tag(args)}.png"


def get_spatial_key(args: argparse.Namespace) -> str:
    return "spatial" if args.no_crop else f"spatial_cropped_{args.buffer}_buffer"


def get_stardist_npz_path(args: argparse.Namespace) -> Path:
    return args.out_dir / f"stardist_labels_mpp{args.mpp:g}_{get_crop_tag(args)}.npz"


def get_cellpose_npz_path(args: argparse.Namespace) -> Path:
    gray_tag = "gray" if args.cellpose_grayscale else "rgb"
    chunk_tag = (
        f"chunk{args.cellpose_chunk_size}_ov{args.cellpose_chunk_overlap}"
        if args.cellpose_chunk_size and args.cellpose_chunk_size > 0
        else "full"
    )
    return args.out_dir / f"cellpose_sam_labels_mpp{args.mpp:g}_{get_crop_tag(args)}_{gray_tag}_{chunk_tag}.npz"


def read_h5ad_checkpoint(path: Path, desc: str) -> ad.AnnData:
    log_step(f"Reading h5ad checkpoint: {path}")
    with heartbeat_progress(desc):
        adata = ad.read_h5ad(path)
    log_step(f"Loaded checkpoint with {adata.n_obs} bins and {adata.n_vars} genes")
    return adata


def write_h5ad_checkpoint(adata: ad.AnnData, path: Path, desc: str) -> None:
    log_step(f"Writing h5ad checkpoint: {path}")
    with heartbeat_progress(desc):
        adata.write_h5ad(path)


def has_obs_column(adata: ad.AnnData | None, key: str) -> bool:
    return adata is not None and key in adata.obs.columns


def obs_label_values(adata: ad.AnnData, key: str) -> np.ndarray:
    return pd.to_numeric(adata.obs[key], errors="coerce").fillna(0).to_numpy(dtype=np.int64)


def has_positive_obs_label(adata: ad.AnnData | None, key: str) -> bool:
    return adata is not None and key in adata.obs.columns and bool(np.any(obs_label_values(adata, key) > 0))


def label_column_summary(adata: ad.AnnData | None, key: str) -> str:
    if not has_obs_column(adata, key):
        return "missing"
    values = obs_label_values(adata, key)
    positive = values > 0
    labeled_bins = int(positive.sum())
    labels = int(np.unique(values[positive]).shape[0])
    return f"present, labeled_bins={labeled_bins}, labels={labels}"


def stardist_label_alias_candidates(adata: ad.AnnData) -> list[str]:
    candidates: list[str] = [STARDIST_OBS_KEY]
    preprocess_meta = adata.uns.get("preprocess_stardist")
    if isinstance(preprocess_meta, dict):
        labels_key = preprocess_meta.get("labels_key")
        if labels_key:
            candidates.append(str(labels_key))
    candidates.extend(PREPROCESS_STARDIST_ALIAS_KEYS)

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for key in candidates:
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(key)
    return unique_candidates


def normalize_stardist_label_column(adata: ad.AnnData, *, context: str) -> bool:
    """Normalize preprocess/bin2cell StarDist labels to obs['stardist_id'].

    preprocess.py may have produced bin2cell StarDist labels under labels_he
    before expanding them to labels_he_expanded for degrade. Inference expects
    stardist_id, so we canonicalize any existing positive StarDist-like label
    column before deciding whether to rerun StarDist.
    """
    if has_positive_obs_label(adata, STARDIST_OBS_KEY):
        values = obs_label_values(adata, STARDIST_OBS_KEY)
        adata.obs[STARDIST_OBS_KEY] = values.astype(np.int64)
        log_step(f"{context}: obs['{STARDIST_OBS_KEY}'] already has positive labels")
        return False

    for alias_key in stardist_label_alias_candidates(adata):
        if alias_key == STARDIST_OBS_KEY:
            continue
        if has_positive_obs_label(adata, alias_key):
            values = obs_label_values(adata, alias_key)
            adata.obs[STARDIST_OBS_KEY] = values.astype(np.int64)
            label_aliases = adata.uns.get("nucleus_segment_label_aliases")
            if not isinstance(label_aliases, dict):
                label_aliases = {}
                adata.uns["nucleus_segment_label_aliases"] = label_aliases
            label_aliases[STARDIST_OBS_KEY] = {
                "source_key": alias_key,
                "source_context": context,
            }
            log_step(
                f"{context}: copied existing obs['{alias_key}'] -> "
                f"obs['{STARDIST_OBS_KEY}']; StarDist can be skipped"
            )
            return True

    if STARDIST_OBS_KEY in adata.obs.columns:
        adata.obs[STARDIST_OBS_KEY] = obs_label_values(adata, STARDIST_OBS_KEY).astype(np.int64)
    log_step(f"{context}: no existing positive StarDist label column found")
    return False


def current_resume_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_image_path": str(args.source_image_path.resolve()),
        "mpp": float(args.mpp),
        "buffer": int(args.buffer),
        "crop": not bool(args.no_crop),
        "stardist_model": args.stardist_model,
        "cellpose_model": args.cellpose_model,
        "cellpose_grayscale": bool(args.cellpose_grayscale),
        "cellpose_chunk_size": int(args.cellpose_chunk_size),
        "cellpose_chunk_overlap": int(args.cellpose_chunk_overlap),
    }


def check_checkpoint_params(adata: ad.AnnData, args: argparse.Namespace) -> None:
    previous = adata.uns.get("nucleus_segment")
    if not isinstance(previous, dict):
        previous = adata.uns.get("nucleus_segment_partial")
    if not isinstance(previous, dict):
        log_step("No previous nucleus_segment parameters found in checkpoint; resume parameter check skipped")
        return
    current = current_resume_params(args)
    mismatches: list[str] = []
    has_stardist_result = has_positive_obs_label(adata, STARDIST_OBS_KEY)
    has_cellpose_result = has_positive_obs_label(adata, CELLPOSE_OBS_KEY)
    for key, current_value in current.items():
        if key == "stardist_model" and (args.rerun_stardist or not has_stardist_result):
            continue
        if key in {
            "cellpose_model",
            "cellpose_grayscale",
            "cellpose_chunk_size",
            "cellpose_chunk_overlap",
        } and (args.rerun_cellpose or not has_cellpose_result):
            continue
        if key in {"mpp", "buffer", "crop"} and args.rerun_stardist and args.rerun_cellpose:
            continue
        previous_value = previous.get(key)
        if key == "source_image_path" and previous_value is not None:
            previous_value = str(Path(str(previous_value)).resolve())
        if previous_value != current_value:
            mismatches.append(f"{key}: previous={previous_value!r}, current={current_value!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Existing checkpoint was produced with different segmentation parameters. "
            f"{mismatch_text}. Use --force-restart, --rerun-stardist, or --rerun-cellpose "
            "if you intentionally want to replace cached results."
        )
    log_step("Existing checkpoint parameters match current run")


def inspect_resume_state(
    args: argparse.Namespace,
    working_h5ad: Path,
    output_h5ad: Path,
    scaled_image_path: Path,
    spatial_key: str,
    stardist_npz: Path,
    cellpose_npz: Path,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "adata": None,
        "source_h5ad": None,
        "has_checkpoint": False,
        "scaled_ready": False,
        "stardist_npz": stardist_npz.exists(),
        "cellpose_npz": cellpose_npz.exists(),
        "stardist_obs": False,
        "cellpose_obs": False,
        "final_complete": False,
    }
    log_step("Inspecting out_dir checkpoints for resumable state")
    if args.force_restart:
        log_step("--force-restart is set; existing checkpoints will be ignored")
        return state

    checkpoint_path = None
    if output_h5ad.exists():
        checkpoint_path = output_h5ad
    elif working_h5ad.exists():
        checkpoint_path = working_h5ad

    if checkpoint_path is not None:
        state["adata"] = read_h5ad_checkpoint(checkpoint_path, "resume checkpoint read")
        check_checkpoint_params(state["adata"], args)
        state["source_h5ad"] = checkpoint_path
        state["has_checkpoint"] = True
        state["scaled_ready"] = scaled_image_path.exists() and spatial_key in state["adata"].obsm
        state["stardist_obs"] = has_positive_obs_label(state["adata"], STARDIST_OBS_KEY)
        state["cellpose_obs"] = has_positive_obs_label(state["adata"], CELLPOSE_OBS_KEY)
        state["final_complete"] = (
            output_h5ad.exists() and state["stardist_obs"] and state["cellpose_obs"]
        )

    log_step(f"Resume status: working_h5ad_exists={working_h5ad.exists()}")
    log_step(f"Resume status: output_h5ad_exists={output_h5ad.exists()}")
    log_step(f"Resume status: scaled_image_exists={scaled_image_path.exists()}")
    log_step(f"Resume status: spatial_key='{spatial_key}' ready={state['scaled_ready']}")
    log_step(f"Resume status: stardist_npz_exists={state['stardist_npz']}")
    log_step(f"Resume status: cellpose_npz_exists={state['cellpose_npz']}")
    log_step(f"Resume status: stardist_id={label_column_summary(state['adata'], STARDIST_OBS_KEY)}")
    log_step(f"Resume status: cellpose_id={label_column_summary(state['adata'], CELLPOSE_OBS_KEY)}")
    return state


def qc_and_destripe(adata: ad.AnnData, args: argparse.Namespace, b2c: Any) -> ad.AnnData:
    import scanpy as sc

    log_step("Starting bin-level QC")
    log_step("Making variable names unique before QC")
    adata.var_names_make_unique()
    adata.obs["n_counts"] = total_counts(adata.X)
    log_step(f"Filtering genes with min_cells >= {args.min_cells}")
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    log_step(f"Filtering bins with min_counts >= {args.min_counts}")
    sc.pp.filter_cells(adata, min_counts=args.min_counts)
    adata.obs["n_counts"] = total_counts(adata.X)
    log_step("Saving raw counts layer after QC")
    adata.layers["counts_raw"] = adata.X.copy()
    log_step(f"Running bin2cell destripe with quantile={args.destripe_quantile}")
    with heartbeat_progress("bin2cell.destripe"):
        b2c.destripe(
            adata,
            quantile=args.destripe_quantile,
            counts_key="n_counts",
            adjusted_counts_key="n_counts_adjusted",
            adjust_counts=True,
        )
    log_step("Saving destriped counts layers")
    adata.layers["counts"] = adata.X.copy()
    adata.layers["counts_destriped"] = adata.X.copy()
    log_step(f"QC/destripe complete: {adata.n_obs} bins, {adata.n_vars} genes")
    return adata


def make_scaled_image(
    adata: ad.AnnData,
    args: argparse.Namespace,
    b2c: Any,
    working_h5ad: Path,
) -> tuple[ad.AnnData, Path, str]:
    image_path = get_scaled_image_path(args)
    spatial_cropped_key = get_spatial_key(args)
    log_step(
        f"Generating mpp-scaled H&E image at mpp={args.mpp}, "
        f"crop={not args.no_crop}, buffer={args.buffer}"
    )
    with heartbeat_progress("bin2cell.scaled_he_image"):
        b2c.scaled_he_image(
            adata,
            mpp=args.mpp,
            crop=not args.no_crop,
            buffer=args.buffer,
            spatial_cropped_key=None if args.no_crop else spatial_cropped_key,
            store=True,
            save_path=str(image_path),
        )
    if spatial_cropped_key not in adata.obsm:
        raise KeyError(f"Expected adata.obsm['{spatial_cropped_key}'] after scaled_he_image.")
    coords = adata.obsm[spatial_cropped_key]
    print(
        f"{spatial_cropped_key}: "
        f"x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], "
        f"y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]"
    )
    adata.uns["nucleus_segment_partial"] = {
        **current_resume_params(args),
        "output_type": "partial checkpoint after mpp-scaled image generation",
        "scaled_image_path": str(image_path),
        "spatial_key": spatial_cropped_key,
    }
    log_step(f"Writing working h5ad after mpp image generation: {working_h5ad}")
    with heartbeat_progress("write working h5ad"):
        adata.write_h5ad(working_h5ad)
    return adata, image_path, spatial_cropped_key


def assert_label_npz(labels_path: Path, image_shape: tuple[int, int] | None = None) -> None:
    log_step(f"Checking label npz: {labels_path}")
    try:
        labels = sp.load_npz(str(labels_path))
    except Exception as exc:
        raise ValueError(f"Label file {labels_path} is not a scipy sparse npz.") from exc
    if len(labels.shape) != 2:
        raise ValueError(f"Label matrix must be 2D, got shape {labels.shape}.")
    if image_shape is not None and tuple(labels.shape) != tuple(image_shape):
        raise ValueError(f"Label matrix shape {labels.shape} does not match image shape {image_shape}.")
    max_label = int(labels.data.max()) if labels.nnz else 0
    print(f"{labels_path.name}: shape={labels.shape}, nonzero_pixels={labels.nnz}, max_label={max_label}")
    if max_label == 0:
        raise ValueError(f"Label file {labels_path} contains no labeled objects.")


def run_stardist(image_path: Path, labels_path: Path, args: argparse.Namespace, b2c: Any) -> None:
    log_step(f"Preparing StarDist segmentation on image: {image_path}")
    kwargs: dict[str, float] = {}
    if args.stardist_prob_thresh is not None:
        kwargs["prob_thresh"] = args.stardist_prob_thresh
    if args.stardist_nms_thresh is not None:
        kwargs["nms_thresh"] = args.stardist_nms_thresh
    log_step(
        f"Running StarDist model={args.stardist_model}, "
        f"block_size={args.stardist_block_size}, overlap={args.stardist_min_overlap}"
    )
    with heartbeat_progress("StarDist segmentation"):
        b2c.stardist(
            str(image_path),
            str(labels_path),
            stardist_model=args.stardist_model,
            block_size=args.stardist_block_size,
            min_overlap=args.stardist_min_overlap,
            context=args.stardist_context,
            **kwargs,
        )
    log_step("Reading scaled image to verify StarDist label shape")
    image = read_segmentation_image(image_path, grayscale=False)
    assert_label_npz(labels_path, image.shape[:2])
    log_step(f"StarDist labels saved: {labels_path}")


def read_segmentation_image(image_path: Path, grayscale: bool = False) -> np.ndarray:
    log_step(f"Reading segmentation image: {image_path}")
    try:
        import imageio.v3 as iio

        img = iio.imread(image_path)
    except Exception:
        import cv2

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image for Cellpose-SAM: {image_path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    if img.ndim == 2:
        return to_uint8_image(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if grayscale:
        log_step("Converting segmentation image to grayscale")
        img = img[..., :3].mean(axis=-1)
    return to_uint8_image(img)


def make_cellpose_model(args: argparse.Namespace):
    from cellpose import models

    log_step(f"Initializing Cellpose-SAM model: {args.cellpose_model}")
    try:
        return models.CellposeModel(gpu=args.cellpose_gpu, pretrained_model=args.cellpose_model)
    except TypeError as exc:
        print(f"CellposeModel(pretrained_model=...) failed, trying model_type=...: {exc}")
        return models.CellposeModel(gpu=args.cellpose_gpu, model_type=args.cellpose_model)


def cleanup_accelerator_memory() -> None:
    log_step("Cleaning accelerator memory before Cellpose-SAM")
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def build_cellpose_eval_kwargs(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    signature = inspect.signature(model.eval)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    def accepts(name: str) -> bool:
        return accepts_var_kwargs or name in parameters

    kwargs: dict[str, Any] = {}
    if accepts("channels"):
        kwargs["channels"] = [0, 0]
    if accepts("tile"):
        kwargs["tile"] = True
    if accepts("tile_overlap"):
        kwargs["tile_overlap"] = args.cellpose_tile_overlap
    for key, value in (
        ("diameter", args.cellpose_diameter),
        ("flow_threshold", args.cellpose_flow_threshold),
        ("cellprob_threshold", args.cellpose_cellprob_threshold),
        ("batch_size", args.cellpose_batch_size),
    ):
        if value is not None and accepts(key):
            kwargs[key] = value

    skipped = [
        key
        for key in ("channels", "tile", "tile_overlap", "diameter", "flow_threshold", "cellprob_threshold", "batch_size")
        if key not in kwargs and not accepts(key)
    ]
    if skipped:
        log_step(f"Cellpose eval does not support these kwargs, skipping them: {', '.join(skipped)}")
    log_step(f"Cellpose eval kwargs: {sorted(kwargs.keys())}")
    return kwargs


def eval_cellpose_model(model: Any, image: np.ndarray, eval_kwargs: dict[str, Any]) -> np.ndarray:
    try:
        result = model.eval(image, **eval_kwargs)
    except TypeError as exc:
        print(f"Cellpose eval failed with filtered kwargs, trying bare eval: {exc}")
        result = model.eval(image)
    masks = result[0] if isinstance(result, tuple) else result
    if masks.ndim != 2:
        raise ValueError(f"Expected a 2D Cellpose mask, got shape {masks.shape}.")
    if tuple(masks.shape) != tuple(image.shape[:2]):
        raise ValueError(f"Cellpose mask shape {masks.shape} does not match image shape {image.shape[:2]}.")
    return masks


def chunk_starts(length: int, chunk_size: int, overlap: int) -> list[int]:
    if chunk_size <= 0 or chunk_size >= length:
        return [0]
    step = max(1, chunk_size - 2 * overlap)
    starts = list(range(0, max(length - chunk_size, 0) + 1, step))
    last = max(length - chunk_size, 0)
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_cellpose_sam_chunked(
    image: np.ndarray,
    model: Any,
    args: argparse.Namespace,
) -> np.ndarray:
    eval_kwargs = build_cellpose_eval_kwargs(model, args)
    chunk_size = int(args.cellpose_chunk_size)
    overlap = int(args.cellpose_chunk_overlap)
    if chunk_size <= 0:
        log_step("Cellpose chunk mode disabled; running on the full image")
        with heartbeat_progress("Cellpose-SAM full image"):
            return eval_cellpose_model(model, image, eval_kwargs)
    if overlap < 0:
        raise ValueError("--cellpose-chunk-overlap must be >= 0.")
    if overlap * 2 >= chunk_size:
        raise ValueError("--cellpose-chunk-overlap must be smaller than half of --cellpose-chunk-size.")

    height, width = image.shape[:2]
    y_starts = chunk_starts(height, chunk_size, overlap)
    x_starts = chunk_starts(width, chunk_size, overlap)
    labels = np.zeros((height, width), dtype=np.int32)
    next_label = 1
    total_chunks = len(y_starts) * len(x_starts)
    log_step(
        f"Running Cellpose-SAM in chunks: image={height}x{width}, "
        f"chunk_size={chunk_size}, overlap={overlap}, chunks={total_chunks}"
    )
    with tqdm(total=total_chunks, desc="Cellpose chunks", dynamic_ncols=True) as bar:
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + chunk_size, height)
                x1 = min(x0 + chunk_size, width)
                chunk = image[y0:y1, x0:x1]
                chunk_mask = eval_cellpose_model(model, chunk, eval_kwargs).astype(np.int32, copy=False)

                core_y0 = 0 if y0 == 0 else overlap
                core_x0 = 0 if x0 == 0 else overlap
                core_y1 = chunk_mask.shape[0] if y1 == height else max(core_y0, chunk_mask.shape[0] - overlap)
                core_x1 = chunk_mask.shape[1] if x1 == width else max(core_x0, chunk_mask.shape[1] - overlap)
                core_mask = chunk_mask[core_y0:core_y1, core_x0:core_x1]
                if core_mask.size and core_mask.max() > 0:
                    unique_labels = np.unique(core_mask)
                    unique_labels = unique_labels[unique_labels > 0]
                    relabeled = np.zeros_like(core_mask, dtype=np.int32)
                    for label_id in unique_labels:
                        relabeled[core_mask == label_id] = next_label
                        next_label += 1
                    labels[
                        y0 + core_y0 : y0 + core_y1,
                        x0 + core_x0 : x0 + core_x1,
                    ] = relabeled
                bar.update(1)
    log_step(f"Cellpose chunk merge complete: labels={next_label - 1}")
    return labels


def run_cellpose_sam(image_path: Path, labels_path: Path, args: argparse.Namespace) -> None:
    log_step("Preparing Cellpose-SAM segmentation")
    cleanup_accelerator_memory()
    image = read_segmentation_image(image_path, grayscale=args.cellpose_grayscale)
    print(f"Cellpose input image shape: {image.shape}, dtype={image.dtype}")
    model = make_cellpose_model(args)
    log_step("Running Cellpose-SAM segmentation")
    masks = run_cellpose_sam_chunked(image, model, args)
    log_step("Validating Cellpose-SAM mask")
    log_step(f"Saving Cellpose-SAM labels: {labels_path}")
    with heartbeat_progress("save Cellpose labels"):
        sp.save_npz(str(labels_path), sp.csr_matrix(masks.astype(np.int32, copy=False)))
    assert_label_npz(labels_path, image.shape[:2])
    print(f"Cellpose-SAM found {int(masks.max())} objects")


def insert_label_column(
    adata: ad.AnnData,
    labels_path: Path,
    labels_key: str,
    spatial_key: str,
    mpp: float,
    b2c: Any,
) -> None:
    log_step(f"Inserting labels into adata.obs['{labels_key}'] from {labels_path}")
    with heartbeat_progress(f"insert {labels_key}"):
        b2c.insert_labels(
            adata,
            str(labels_path),
            basis="spatial",
            spatial_key=spatial_key,
            mpp=mpp,
            labels_key=labels_key,
        )
    adata.obs[labels_key] = adata.obs[labels_key].fillna(0).astype(np.int64)
    n_labeled_bins = int((adata.obs[labels_key] > 0).sum())
    n_labels = int(adata.obs.loc[adata.obs[labels_key] > 0, labels_key].nunique())
    print(f"{labels_key}: labeled_bins={n_labeled_bins}, labels={n_labels}")
    if n_labeled_bins == 0 or n_labels == 0:
        raise ValueError(
            f"No bins were assigned to {labels_key}. "
            f"Check mpp={mpp}, spatial_key='{spatial_key}', crop/buffer settings, "
            f"and whether label image coordinates match adata.obsm['{spatial_key}']."
        )


def record_segmentation_params(
    adata: ad.AnnData,
    args: argparse.Namespace,
    scaled_image_path: Path,
    spatial_key: str,
    stardist_npz: Path,
    cellpose_npz: Path,
) -> None:
    log_step("Recording segmentation parameters in adata.uns['nucleus_segment']")
    adata.uns["nucleus_segment"] = {
        "output_type": "bin-level h5ad with nucleus label columns in obs",
        "source_image_path": str(args.source_image_path.resolve()),
        "scaled_image_path": str(scaled_image_path),
        "spatial_key": spatial_key,
        "mpp": float(args.mpp),
        "buffer": int(args.buffer),
        "crop": not bool(args.no_crop),
        "min_cells": int(args.min_cells),
        "min_counts": int(args.min_counts),
        "destripe_quantile": float(args.destripe_quantile),
        "stardist_model": args.stardist_model,
        "stardist_labels_path": str(stardist_npz),
        "cellpose_model": args.cellpose_model,
        "cellpose_grayscale": bool(args.cellpose_grayscale),
        "cellpose_chunk_size": int(args.cellpose_chunk_size),
        "cellpose_chunk_overlap": int(args.cellpose_chunk_overlap),
        "cellpose_gpu": bool(args.cellpose_gpu),
        "cellpose_labels_path": str(cellpose_npz),
    }


def main() -> None:
    args = parse_args()
    log_step("Starting nucleus segmentation pipeline")
    log_step(f"Output directory: {args.out_dir}")
    if not args.source_image_path.exists():
        raise FileNotFoundError(f"Source image does not exist: {args.source_image_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = args.output_h5ad or (args.out_dir / "nucleus_segmented.h5ad")
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    working_h5ad = args.out_dir / args.working_h5ad_name
    scaled_image_path = get_scaled_image_path(args)
    spatial_key = get_spatial_key(args)
    stardist_npz = get_stardist_npz_path(args)
    cellpose_npz = get_cellpose_npz_path(args)

    b2c = import_bin2cell(args.bin2cell_path)
    stardist_ran_this_process = False
    resume_state = inspect_resume_state(
        args,
        working_h5ad,
        output_h5ad,
        scaled_image_path,
        spatial_key,
        stardist_npz,
        cellpose_npz,
    )

    if resume_state["final_complete"] and not args.rerun_stardist and not args.rerun_cellpose:
        log_step("Final output already contains stardist_id and cellpose_id; nothing to resume")
        log_step(f"Final h5ad: {output_h5ad}")
        return

    if resume_state["has_checkpoint"]:
        log_step(f"Resuming from existing h5ad checkpoint: {resume_state['source_h5ad']}")
        adata = resume_state["adata"]
        normalized = normalize_stardist_label_column(adata, context="resume checkpoint")
        if resume_state["source_h5ad"] != working_h5ad:
            write_h5ad_checkpoint(adata, working_h5ad, "materialize working h5ad")
        elif normalized:
            write_h5ad_checkpoint(adata, working_h5ad, "write normalized StarDist labels")
    else:
        log_step("Step 1/9: reading Visium HD data")
        adata = read_hd_adata(args, b2c)
        normalize_stardist_label_column(adata, context="input h5ad")
        log_step("Step 2/9: running QC and destripe")
        adata = qc_and_destripe(adata, args, b2c)
        log_step(f"Step 3/9: writing first read/QC/destriped h5ad: {working_h5ad}")
        write_h5ad_checkpoint(adata, working_h5ad, "write initial h5ad")
        print(f"Wrote read/QC/destriped h5ad: {working_h5ad}")

    # From here on, operate on the h5ad already materialized in out_dir.
    log_step("Step 4/9: reloading working h5ad from out_dir")
    adata = read_h5ad_checkpoint(working_h5ad, "reload working h5ad")
    normalized = normalize_stardist_label_column(adata, context="working h5ad")
    if normalized:
        write_h5ad_checkpoint(adata, working_h5ad, "write normalized StarDist labels")

    scaled_ready = scaled_image_path.exists() and spatial_key in adata.obsm
    needs_scaled_image = not scaled_ready or args.rerun_stardist or args.rerun_cellpose
    needs_scaled_image = needs_scaled_image or (
        not has_positive_obs_label(adata, STARDIST_OBS_KEY) and not stardist_npz.exists()
    )
    needs_scaled_image = needs_scaled_image or (
        not has_positive_obs_label(adata, CELLPOSE_OBS_KEY) and not cellpose_npz.exists()
    )
    if needs_scaled_image:
        log_step("Step 5/9: generating mpp-scaled H&E image")
        adata, scaled_image_path, spatial_key = make_scaled_image(adata, args, b2c, working_h5ad)
        print(f"Wrote mpp-scaled image: {scaled_image_path}")
    else:
        log_step("Step 5/9: mpp-scaled image and spatial key already exist; skipping image generation")

    stardist_obs_ready = has_positive_obs_label(adata, STARDIST_OBS_KEY) and not args.rerun_stardist
    if stardist_obs_ready:
        log_step("Step 6/9: obs['stardist_id'] already exists; skipping StarDist and insertion")
    else:
        if args.rerun_stardist or not stardist_npz.exists():
            log_step("Step 6/9: running StarDist")
            run_stardist(scaled_image_path, stardist_npz, args, b2c)
            stardist_ran_this_process = True
        else:
            log_step("Step 6/9: StarDist label npz already exists; checking shape and inserting labels")
            image_shape = read_segmentation_image(scaled_image_path, grayscale=False).shape[:2]
            assert_label_npz(stardist_npz, image_shape)
        adata = read_h5ad_checkpoint(working_h5ad, "reload h5ad before StarDist insert")
        insert_label_column(adata, stardist_npz, STARDIST_OBS_KEY, spatial_key, args.mpp, b2c)
        write_h5ad_checkpoint(adata, working_h5ad, "write h5ad after StarDist insert")
        print(f"Inserted obs['stardist_id'] and updated {working_h5ad}")

    if stardist_ran_this_process and not has_positive_obs_label(adata, CELLPOSE_OBS_KEY) and not args.rerun_stardist:
        log_step("Restarting Python process before Cellpose-SAM to release StarDist/TensorFlow GPU memory")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    adata = read_h5ad_checkpoint(working_h5ad, "reload h5ad before Cellpose stage")
    cellpose_obs_ready = has_positive_obs_label(adata, CELLPOSE_OBS_KEY) and not args.rerun_cellpose
    if cellpose_obs_ready:
        log_step("Step 7/9: obs['cellpose_id'] already exists; skipping Cellpose-SAM and insertion")
    else:
        if args.rerun_cellpose or not cellpose_npz.exists():
            log_step("Step 7/9: running Cellpose-SAM")
            run_cellpose_sam(scaled_image_path, cellpose_npz, args)
        else:
            log_step("Step 7/9: Cellpose-SAM label npz already exists; checking shape")
            image_shape = read_segmentation_image(
                scaled_image_path,
                grayscale=args.cellpose_grayscale,
            ).shape[:2]
            assert_label_npz(cellpose_npz, image_shape)
        log_step("Step 8/9: inserting cellpose_id")
        adata = read_h5ad_checkpoint(working_h5ad, "reload h5ad before Cellpose insert")
        insert_label_column(adata, cellpose_npz, CELLPOSE_OBS_KEY, spatial_key, args.mpp, b2c)

    record_segmentation_params(adata, args, scaled_image_path, spatial_key, stardist_npz, cellpose_npz)
    log_step("Step 9/9: writing final h5ad outputs")
    write_h5ad_checkpoint(adata, working_h5ad, "write final working h5ad")
    if output_h5ad.resolve() != working_h5ad.resolve():
        write_h5ad_checkpoint(adata, output_h5ad, "write final output h5ad")
    else:
        log_step("Final output h5ad is the same as working h5ad; skipping duplicate write")
    print(f"Wrote final h5ad with obs['stardist_id'] and obs['cellpose_id']: {output_h5ad}")
    log_step("Nucleus segmentation pipeline complete")


if __name__ == "__main__":
    main()

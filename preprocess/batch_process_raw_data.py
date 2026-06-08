from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from argparse import Namespace
from pathlib import Path

import anndata as ad
import numpy as np


PROJECT_ROOT = Path("/root/autodl-tmp/HDpainter1")
RAW_ROOT = PROJECT_ROOT / "raw_data"
HD_ROOT = RAW_ROOT / "HD"
XENIUM_ROOT = RAW_ROOT / "Xenium"
OUT_ROOT = Path("/root/autodl-tmp/OV/batch_preprocess")
LOG_ROOT = Path("/root/autodl-tmp/OV/batch_preprocess_logs")
PYTHON = Path("/root/miniconda3/envs/czf/bin/python")

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
]


SAMPLES: dict[str, dict[str, object]] = {
    "COAD": {
        "sample_id": 0,
        "hd_tar": HD_ROOT / "COAD" / "Visium_HD_6p5mm_Human_Colon_Cancer_binned_outputs.tar.gz",
        "image": HD_ROOT / "COAD" / "Visium_HD_6p5mm_Human_Colon_Cancer_tissue_image.btf",
        "xenium_zips": [
            XENIUM_ROOT / "COAD" / "transcripts.zip",
            XENIUM_ROOT / "COAD" / "segmentation_mask.zip",
        ],
    },
    "NSCLC": {
        "sample_id": 1,
        "hd_tar": HD_ROOT / "NSCLC" / "Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_binned_outputs.tar.gz",
        "image": HD_ROOT / "NSCLC" / "Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_tissue_image.btf",
        "xenium_zips": [
            XENIUM_ROOT / "NSCLC" / "Xenium_V1_Human_Lung_Cancer_FFPE_outs.zip",
        ],
    },
    "PRAD": {
        "sample_id": 2,
        "hd_tar": HD_ROOT / "PRAD" / "Visium_HD_Human_Prostate_Cancer_FFPE_binned_outputs.tar.gz",
        "image": HD_ROOT / "PRAD" / "Visium_HD_Human_Prostate_Cancer_FFPE_tissue_image.tif",
        "xenium_zips": [
            XENIUM_ROOT / "PRAD" / "Xenium_Prime_Human_Prostate_FFPE_outs.zip",
        ],
    },
}


def log(message: str) -> None:
    print(f"[batch_process] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract Xenium/Visium HD raw data and run preprocess.py train-full."
    )
    parser.add_argument("--samples", type=str, default="COAD,NSCLC,PRAD")
    parser.add_argument("--only-extract", action="store_true")
    parser.add_argument("--only-reference", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--stop-after", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=180.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=120.0)
    parser.add_argument("--allow-low-memory", action="store_true")
    parser.add_argument("--degrade-batch-size", type=int, default=2048)
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=24,
        help="Downstream instance/mask target canvas size passed to preprocess.py.",
    )
    parser.add_argument(
        "--nmf-fit-source",
        type=str,
        choices=("xenium-union-cell",),
        default="xenium-union-cell",
        help="NMF fit source. Current workflow intentionally fits only on Xenium union-cell pseudo-cells.",
    )
    parser.add_argument("--nmf-qc-min-counts", type=float, default=1.0)
    parser.add_argument("--nmf-qc-min-genes", type=int, default=3)
    parser.add_argument("--nmf-qc-min-cells", type=int, default=3)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--nmf-solver", type=str, choices=("cd", "mu"), default="cd")
    parser.add_argument(
        "--nmf-beta-loss",
        type=str,
        choices=("frobenius", "kullback-leibler", "itakura-saito"),
        default="frobenius",
    )
    parser.add_argument("--nmf-tol", type=float, default=1e-4)
    parser.add_argument("--nmf-alpha-w", type=float, default=0.0)
    parser.add_argument("--nmf-alpha-h", type=str, default="same")
    parser.add_argument("--nmf-l1-ratio", type=float, default=0.0)
    parser.add_argument("--nmf-dense-fit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nmf-verbose", type=int, default=1)
    parser.add_argument("--regularize-num-workers", type=int, default=8)
    parser.add_argument("--mask-refine-num-workers", type=int, default=8)
    parser.add_argument(
        "--mask-refine-overwrite",
        action="store_true",
        help="Allow mask_refine.py to replace an existing maskrefined output H5.",
    )
    parser.add_argument(
        "--no-split-preprocess-steps",
        action="store_true",
        help=(
            "Run preprocess.py train-full as one long process. By default the batch runner "
            "splits train-full into one subprocess per major step so memory is returned to "
            "the OS between raw_ingest/align/degrade/NMF/downstream target generation."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sample_names(raw: str) -> list[str]:
    names = [item.strip().upper() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in SAMPLES]
    if unknown:
        raise ValueError(f"Unsupported samples: {unknown}. Supported: {sorted(SAMPLES)}")
    return names


def cgroup_memory_limit_gb() -> float | None:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8").strip()
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= 2**60:
            return None
        return value / (1024**3)
    return None


def check_memory_budget(requested_gb: float, *, allow_low_memory: bool) -> None:
    actual = cgroup_memory_limit_gb()
    if actual is None:
        log("Cgroup memory limit: unlimited or unavailable")
        return
    log(f"Cgroup memory limit: {actual:.2f}GB; requested monitor limit: {requested_gb:.2f}GB")
    required = max(16.0, float(requested_gb) * 0.85)
    if actual < required and not allow_low_memory:
        raise MemoryError(
            f"Actual cgroup memory limit is only {actual:.2f}GB; expected at least {required:.2f}GB "
            f"for requested {requested_gb:.2f}GB automation. "
            "Rerun after the server/session is granted the intended memory, "
            "or pass --allow-low-memory only for metadata-only debugging."
        )


def check_disk_budget(required_free_gb: float, *, allow_low_memory: bool) -> None:
    usage = shutil.disk_usage(OUT_ROOT.parent)
    free_gb = usage.free / (1024**3)
    log(f"Disk free under {OUT_ROOT.parent}: {free_gb:.2f}GB; requested minimum: {required_free_gb:.2f}GB")
    if free_gb < float(required_free_gb) and not allow_low_memory:
        raise OSError(
            f"Free disk is only {free_gb:.2f}GB under {OUT_ROOT.parent}; "
            f"expected at least {required_free_gb:.2f}GB before batch preprocessing. "
            "Increase disk space or pass --allow-low-memory for metadata-only debugging."
        )


def run(cmd: list[str], *, log_path: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    if dry_run:
        log(f"DRY-RUN would run {printable}")
        return
    log(f"RUN {printable}")
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path else None
    with (log_path.open("a", encoding="utf-8") if log_path else open(os.devnull, "w", encoding="utf-8")) as lf:
        lf.write(f"\n$ {printable}\n")
        lf.flush()
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)


def extract_zip_members(zip_path: Path, dest: Path, members: list[str] | None, force: bool) -> None:
    marker = dest / f".extract_{zip_path.stem}.done"
    if marker.exists() and not force:
        log(f"Zip already extracted: {zip_path}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        wanted = members or names
        for member in wanted:
            if member not in names:
                raise FileNotFoundError(f"{zip_path} does not contain {member}")
            log(f"Extracting {zip_path.name}:{member}")
            zf.extract(member, dest)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")


def gunzip_keep(path: Path, force: bool) -> Path:
    if path.suffix != ".gz":
        return path
    out = path.with_suffix("")
    if out.exists() and not force:
        return out
    log(f"Decompressing {path} -> {out}")
    with gzip.open(path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)
    return out


def extract_xenium(sample: str, force: bool, dry_run: bool) -> Path:
    cfg = SAMPLES[sample]
    dest = XENIUM_ROOT / sample / "extracted"
    zips = [Path(p) for p in cfg["xenium_zips"]]  # type: ignore[index]
    for zip_path in zips:
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        if dry_run:
            marker = dest / f".extract_{zip_path.stem}.done"
            if marker.exists() and not force:
                log(f"DRY-RUN Xenium zip already extracted: {zip_path}")
            else:
                log(f"DRY-RUN would extract Xenium zip: {zip_path} -> {dest}")
            continue
        if zip_path.name.endswith("_outs.zip"):
            extract_zip_members(
                zip_path,
                dest,
                ["transcripts.parquet", "cell_boundaries.csv.gz", "nucleus_boundaries.csv.gz"],
                force=force,
            )
            gunzip_keep(dest / "cell_boundaries.csv.gz", force=force)
            gunzip_keep(dest / "nucleus_boundaries.csv.gz", force=force)
        else:
            extract_zip_members(zip_path, dest, None, force=force)
    if not dry_run:
        xenium_inputs(dest)
    return dest


def validate_hd_extract(hd_extract_dir: Path) -> Path:
    spaceranger_dir = find_spaceranger_dir(hd_extract_dir)
    required = [
        spaceranger_dir / "filtered_feature_bc_matrix.h5",
        spaceranger_dir / "spatial" / "scalefactors_json.json",
    ]
    if not any((spaceranger_dir / "spatial" / name).exists() for name in (
        "tissue_positions.parquet",
        "tissue_positions.csv",
        "tissue_positions_list.csv",
    )):
        raise FileNotFoundError(f"Missing tissue_positions file under {spaceranger_dir / 'spatial'}")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"HD extract is missing required files: {missing}")
    return spaceranger_dir


def extract_hd(sample: str, force: bool, dry_run: bool) -> Path:
    cfg = SAMPLES[sample]
    tar_path = Path(cfg["hd_tar"])  # type: ignore[index]
    if not tar_path.exists():
        raise FileNotFoundError(tar_path)
    dest = HD_ROOT / sample / "extracted"
    marker = dest / ".extract_square_002um.done"
    if marker.exists() and not force:
        log(f"HD square_002um already extracted: {sample}")
        validate_hd_extract(dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    run(
        [
            "tar",
            "-xzf",
            str(tar_path),
            "-C",
            str(dest),
            "--wildcards",
            "--no-anchored",
            "square_002um/*",
        ],
        log_path=LOG_ROOT / f"{sample}_extract_hd.log",
        dry_run=dry_run,
    )
    if not dry_run:
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
        validate_hd_extract(dest)
    return dest


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {root}")
    return matches[0]


def find_spaceranger_dir(hd_extract_dir: Path) -> Path:
    candidates = sorted(path for path in hd_extract_dir.rglob("square_002um") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"Could not find square_002um under {hd_extract_dir}")
    with_h5 = [path for path in candidates if (path / "filtered_feature_bc_matrix.h5").exists()]
    return (with_h5 or candidates)[0]


def xenium_inputs(xenium_dir: Path) -> tuple[Path, Path, Path]:
    transcripts = find_one(xenium_dir, "transcripts.parquet")
    cell = find_one(xenium_dir, "cell_boundaries.csv")
    nucleus = find_one(xenium_dir, "nucleus_boundaries.csv")
    return transcripts, cell, nucleus


def ensure_reference_counts(h5ad_path: Path) -> None:
    backed = ad.read_h5ad(h5ad_path, backed="r")
    has_n_counts = "n_counts" in backed.obs.columns
    has_adjusted = "n_counts_adjusted" in backed.obs.columns
    backed.file.close()
    if has_n_counts and has_adjusted:
        return

    adata = ad.read_h5ad(h5ad_path)
    if "n_counts" not in adata.obs.columns:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).reshape(-1).astype(np.float64)
    if "n_counts_adjusted" not in adata.obs.columns:
        adata.obs["n_counts_adjusted"] = adata.obs["n_counts"].to_numpy(dtype=np.float64)
    adata.write_h5ad(h5ad_path)


def build_reference_h5ad(sample: str, hd_extract_dir: Path, force: bool) -> Path:
    out = HD_ROOT / sample / "reference_hd_square_002um.h5ad"
    if out.exists() and not force:
        ensure_reference_counts(out)
        log(f"Reference h5ad already exists: {out}")
        return out

    sys.path.insert(0, str(PROJECT_ROOT / "inference"))
    from nucleus_segment import add_spatial_metadata, import_bin2cell, qc_and_destripe

    spaceranger_dir = find_spaceranger_dir(hd_extract_dir)
    count_path = spaceranger_dir / "filtered_feature_bc_matrix.h5"
    image_path = Path(SAMPLES[sample]["image"])  # type: ignore[index]
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not count_path.exists():
        raise FileNotFoundError(count_path)

    ns = Namespace(
        input_h5ad=None,
        spaceranger_dir=spaceranger_dir,
        matrix_dir=None,
        count_file="filtered_feature_bc_matrix.h5",
        spatial_dir=spaceranger_dir / "spatial",
        source_image_path=image_path,
        library_id=f"Visium_HD_{sample}",
        bin2cell_path=None,
        min_cells=3,
        min_counts=1,
        destripe_quantile=0.99,
    )
    b2c = import_bin2cell(None)
    log(
        "Reading HD reference with nucleus_segment.py scanpy fallback logic "
        f"(avoids loading full source image): {count_path}"
    )
    import scanpy as sc

    adata = sc.read_10x_h5(count_path)
    adata.var_names_make_unique()
    adata = add_spatial_metadata(adata, ns.spatial_dir, ns.source_image_path, ns.library_id)
    log("Running QC/destripe for reference h5ad")
    adata = qc_and_destripe(adata, ns, b2c)
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing reference h5ad: {out}")
    adata.write_h5ad(out)
    return out


def process_tree_pids(pid: int) -> list[int]:
    pids = [pid]
    idx = 0
    while idx < len(pids):
        current = pids[idx]
        idx += 1
        children_path = Path(f"/proc/{current}/task/{current}/children")
        try:
            children = [int(x) for x in children_path.read_text().split()]
        except Exception:
            children = []
        for child in children:
            if child not in pids:
                pids.append(child)
    return pids


def rss_gb(pid: int) -> float:
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for item in process_tree_pids(pid):
        try:
            fields = Path(f"/proc/{item}/statm").read_text().split()
            total += int(fields[1]) * page_size
        except Exception:
            pass
    return total / (1024 ** 3)


def run_monitored(cmd: list[str], log_path: Path, memory_limit_gb: float, dry_run: bool) -> None:
    log(f"RUN {' '.join(cmd)}")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak = 0.0
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            preexec_fn=os.setsid,
        )
        try:
            while proc.poll() is None:
                current = rss_gb(proc.pid)
                peak = max(peak, current)
                if current > memory_limit_gb:
                    lf.write(
                        f"\n[batch_process] memory limit exceeded: rss={current:.2f}GB "
                        f"limit={memory_limit_gb:.2f}GB; terminating process group\n"
                    )
                    lf.flush()
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(10)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    raise MemoryError(
                        f"Command exceeded memory limit: peak_rss={current:.2f}GB "
                        f"limit={memory_limit_gb:.2f}GB. Log: {log_path}"
                    )
                time.sleep(15)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        lf.write(f"\n[batch_process] peak_rss_gb={peak:.2f}\n")
    log(f"Finished command. peak_rss_gb={peak:.2f}; log={log_path}")


def selected_train_full_steps(args: argparse.Namespace) -> list[str]:
    start = args.start_from or TRAIN_FULL_STEPS[0]
    stop = args.stop_after or TRAIN_FULL_STEPS[-1]
    if start not in TRAIN_FULL_STEPS:
        raise ValueError(f"Unsupported --start-from for split train-full: {start!r}. Valid: {TRAIN_FULL_STEPS}")
    if stop not in TRAIN_FULL_STEPS:
        raise ValueError(f"Unsupported --stop-after for split train-full: {stop!r}. Valid: {TRAIN_FULL_STEPS}")
    start_idx = TRAIN_FULL_STEPS.index(start)
    stop_idx = TRAIN_FULL_STEPS.index(stop)
    if start_idx > stop_idx:
        raise ValueError(f"--start-from {start!r} must be earlier than or equal to --stop-after {stop!r}.")
    return TRAIN_FULL_STEPS[start_idx : stop_idx + 1]


def run_preprocess_monitored(
    *,
    sample: str,
    base_cmd: list[str],
    final: Path,
    args: argparse.Namespace,
) -> None:
    if args.no_split_preprocess_steps:
        run_monitored(
            base_cmd,
            log_path=LOG_ROOT / f"{sample}_preprocess.log",
            memory_limit_gb=float(args.memory_limit_gb),
            dry_run=args.dry_run,
        )
        return

    steps = selected_train_full_steps(args)
    log(f"Split preprocess enabled for {sample}; steps={steps}")
    for step in steps:
        if final.exists() and args.stop_after is None:
            log(f"Final output exists before step={step}; validating and stopping split run: {final}")
            validate_final(final)
            return
        step_cmd = [*base_cmd, "--start-from", step, "--stop-after", step]
        run_monitored(
            step_cmd,
            log_path=LOG_ROOT / f"{sample}_preprocess_{step}.log",
            memory_limit_gb=float(args.memory_limit_gb),
            dry_run=args.dry_run,
        )


def preprocess_cmd(
    sample: str,
    reference_h5ad: Path,
    xenium_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    transcripts, cell, nucleus = xenium_inputs(xenium_dir)
    out = OUT_ROOT / sample
    base = out / f"regularize_train_tiles_degraded_{sample}_nmf48"
    final = Path(str(base) + ".instchunk512_train.maskrefined.h5")
    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / "preprocess" / "preprocess.py"),
        "train-full",
        "--reference-hd-h5ad",
        str(reference_h5ad),
        "--raw-transcripts-parquet",
        str(transcripts),
        "--raw-cell-boundaries-csv",
        str(cell),
        "--raw-nucleus-boundaries-csv",
        str(nucleus),
        "--output-raw-bin-h5ad",
        str(out / "raw_bin.h5ad"),
        "--output-raw-cell-h5ad",
        str(out / "raw_cell.h5ad"),
        "--output-unioned-bin-h5ad",
        str(out / "unioned_bin.h5ad"),
        "--output-aligned-cell-h5ad",
        str(out / "aligned_cell.h5ad"),
        "--output-aligned-reference-h5ad",
        str(out / "aligned_reference_hd.h5ad"),
        "--output-degraded-h5ad",
        str(out / "degraded.h5ad"),
        "--output-cell-nmf-h5ad",
        str(out / "cell_nmf48.h5ad"),
        "--output-nmf-source-cell-h5ad",
        str(out / "nmf_source_cells.h5ad"),
        "--output-module-csv",
        str(out / "module_gene_weights_nmf48.csv"),
        "--selected-genes-output-path",
        str(out / "selected_genes.txt"),
        "--output-manifest-json",
        str(out / "preprocess_manifest.json"),
        "--out-dir",
        str(out),
        "--output-h5",
        str(base) + ".h5",
        "--instance-chunk-manifest",
        str(base) + ".instchunk512_train.h5",
        "--mask-refine-output-h5",
        str(final),
        "--nmf-components",
        "48",
        "--nmf-fit-source",
        str(args.nmf_fit_source),
        "--nmf-max-iter",
        str(args.nmf_max_iter),
        "--nmf-solver",
        str(args.nmf_solver),
        "--nmf-beta-loss",
        str(args.nmf_beta_loss),
        "--nmf-tol",
        str(args.nmf_tol),
        "--nmf-alpha-w",
        str(args.nmf_alpha_w),
        "--nmf-alpha-h",
        str(args.nmf_alpha_h),
        "--nmf-l1-ratio",
        str(args.nmf_l1_ratio),
        "--nmf-verbose",
        str(args.nmf_verbose),
        "--reference-image-path",
        str(Path(SAMPLES[sample]["image"])),  # type: ignore[index]
        "--degrade-segmentation-out-dir",
        str(out / "degrade_stardist"),
        "--degrade-stardist-library-id",
        f"Visium_HD_{sample}",
        "--nmf-qc-min-counts",
        str(args.nmf_qc_min_counts),
        "--nmf-qc-min-genes",
        str(args.nmf_qc_min_genes),
        "--nmf-qc-min-cells",
        str(args.nmf_qc_min_cells),
        "--instance-budget",
        "512",
        "--canvas-size",
        str(args.canvas_size),
        "--regularize-masks",
        "--regularize-promote",
        "--sample-type",
        sample,
        "--sample-type-id",
        str(SAMPLES[sample]["sample_id"]),
        "--degrade-batch-size",
        str(args.degrade_batch_size),
        "--regularize-num-workers",
        str(args.regularize_num_workers),
        "--mask-refine-num-workers",
        str(args.mask_refine_num_workers),
        "--progress-every",
        "50",
    ]
    cmd.append("--nmf-dense-fit" if bool(args.nmf_dense_fit) else "--no-nmf-dense-fit")
    if bool(args.mask_refine_overwrite):
        cmd.append("--mask-refine-overwrite")
    if args.no_split_preprocess_steps and args.start_from:
        cmd.extend(["--start-from", args.start_from])
    if args.no_split_preprocess_steps and args.stop_after:
        cmd.extend(["--stop-after", args.stop_after])
    return cmd, final


def validate_final(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    import h5py

    required = [
        "train_chunk_tile_offsets",
        "train_chunk_instance_offsets",
        "train_tile_input_pool",
        "train_instance_mask_targets_pool",
        "val_chunk_tile_offsets",
        "val_chunk_instance_offsets",
        "val_tile_input_pool",
        "val_instance_mask_targets_pool",
    ]
    with h5py.File(path, "r") as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise KeyError(f"{path} missing datasets: {missing}")


def write_inventory(sample: str, payload: dict[str, object]) -> None:
    out = OUT_ROOT / sample
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch_inventory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    will_build_reference = not bool(args.skip_reference) and not bool(args.only_extract) and not bool(args.dry_run)
    will_preprocess = (
        not bool(args.skip_preprocess)
        and not bool(args.only_reference)
        and not bool(args.only_extract)
        and not bool(args.dry_run)
    )
    needs_heavy_memory = will_build_reference or will_preprocess
    if not args.dry_run:
        check_disk_budget(float(args.min_free_disk_gb), allow_low_memory=bool(args.allow_low_memory))
    if needs_heavy_memory:
        check_memory_budget(float(args.memory_limit_gb), allow_low_memory=bool(args.allow_low_memory))
    for sample in sample_names(args.samples):
        log("=" * 80)
        log(f"Sample: {sample}")
        hd_extract_dir = HD_ROOT / sample / "extracted"
        xenium_dir = XENIUM_ROOT / sample / "extracted"

        if not args.skip_extract:
            hd_extract_dir = extract_hd(sample, force=args.force_extract, dry_run=args.dry_run)
            xenium_dir = extract_xenium(sample, force=args.force_extract, dry_run=args.dry_run)

        if args.dry_run:
            log(f"DRY-RUN sample paths | hd_extract_dir={hd_extract_dir} xenium_dir={xenium_dir}")
            continue

        if args.only_extract:
            transcripts, cell, nucleus = xenium_inputs(xenium_dir)
            write_inventory(
                sample,
                {
                    "sample": sample,
                    "hd_extract_dir": str(hd_extract_dir),
                    "xenium_dir": str(xenium_dir),
                    "transcripts": str(transcripts),
                    "cell_boundaries": str(cell),
                    "nucleus_boundaries": str(nucleus),
                },
            )
            continue

        reference_h5ad = HD_ROOT / sample / "reference_hd_square_002um.h5ad"
        if not args.skip_reference:
            reference_h5ad = build_reference_h5ad(sample, hd_extract_dir, force=args.force_reference)
            ensure_reference_counts(reference_h5ad)
        elif not reference_h5ad.exists():
            raise FileNotFoundError(
                f"--skip-reference was set, but the expected reference h5ad does not exist: {reference_h5ad}"
            )

        transcripts, cell, nucleus = xenium_inputs(xenium_dir)
        inventory = {
            "sample": sample,
            "hd_extract_dir": str(hd_extract_dir),
            "xenium_dir": str(xenium_dir),
            "reference_h5ad": str(reference_h5ad),
            "transcripts": str(transcripts),
            "cell_boundaries": str(cell),
            "nucleus_boundaries": str(nucleus),
        }
        write_inventory(sample, inventory)
        log(json.dumps(inventory, indent=2))

        if args.only_reference or args.skip_preprocess:
            continue

        cmd, final = preprocess_cmd(sample, reference_h5ad, xenium_dir, args)
        if final.exists() and args.stop_after is None and args.start_from is None:
            log(f"Final output already exists; validating and skipping: {final}")
            validate_final(final)
            continue
        run_preprocess_monitored(sample=sample, base_cmd=cmd, final=final, args=args)
        if args.stop_after is None:
            validate_final(final)
            log(f"Validated final output: {final}")


if __name__ == "__main__":
    main()

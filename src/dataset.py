import glob
import os

import torch
from torch.utils.data import DataLoader, Dataset

# Per-dataset SVD scale factors: normalise features to std ≈ 1.
# Applied offline in build_filtered_dataset (not at runtime).
#   CESC  std ≈ 1.0062 → scale 1/1.0062 ≈ 0.9938
#   OV    std ≈ 1.0455 → scale 1/1.0455 ≈ 0.9565
#   NSCLC std ≈ 0.42   → scale 1/0.42   ≈ 2.38
#   PRAD  std ≈ 0.34   → scale 1/0.34   ≈ 2.94
_DATASET_SCALE: dict[str, float] = {
    "SVD_CESC":  1.0 / 1.0062,  # measured 2026-04-13
    "SVD_OV":    1.0 / 1.0455,  # measured 2026-04-13
    "SVD_NSCLC": 1.0 / 0.42,
    "SVD_PRAD":  1.0 / 0.34,
}
_DEFAULT_SCALE = 1.0


def _infer_scale(name: str) -> float:
    """Return the SVD scale factor for a dataset subdirectory name."""
    return _DATASET_SCALE.get(name, _DEFAULT_SCALE)


def check_and_annotate(src_dir: str, patch_size: int = 128, overlap: int = 16) -> None:
    """Annotate each raw tile in-place with center_nuclei_count.

    Scans all subdirs of src_dir for *.pt tiles. Tiles already containing
    the 'center_nuclei_count' key are skipped (incremental).

    Args:
        src_dir:    Parent directory containing per-dataset subdirs (e.g. SVD_CESC/).
        patch_size: Tile spatial size (default 128).
        overlap:    Overlap between tiles; the centre crop excludes overlap//2 pixels
                    on each edge when counting unique nuclei.
    """
    margin = overlap // 2
    min_bound, max_bound = margin, patch_size - margin

    subdirs = [
        os.path.join(src_dir, d) for d in sorted(os.listdir(src_dir))
        if os.path.isdir(os.path.join(src_dir, d))
    ]

    for subdir in subdirs:
        files = glob.glob(os.path.join(subdir, "*.pt"))
        if not files:
            continue

        # Check if annotation is needed (sample first 10 tiles)
        needs_annotation = any(
            "center_nuclei_count" not in torch.load(f, weights_only=True)
            for f in files[:10]
        )
        if not needs_annotation:
            print(f"[{os.path.basename(subdir)}] Already annotated. Skipping.")
            continue

        print(f"[{os.path.basename(subdir)}] Annotating {len(files)} tiles...")
        for filepath in files:
            tile_data = torch.load(filepath, weights_only=True)
            nuclei_tensor = tile_data["input_nuclei"]

            if nuclei_tensor._nnz() == 0:
                count = 0
            else:
                indices = nuclei_tensor.indices()
                values  = nuclei_tensor.values()
                spatial_mask = (
                    (indices[0] >= min_bound) & (indices[0] < max_bound) &
                    (indices[1] >= min_bound) & (indices[1] < max_bound)
                )
                count = torch.unique(values[spatial_mask]).numel()

            tile_data["center_nuclei_count"] = count
            torch.save(tile_data, filepath)


def build_filtered_dataset(src_dir: str, dst_dir: str, min_nuc: int = 5) -> None:
    """Filter, convert, and scale raw tiles into a training-ready dataset.

    For each subdir in src_dir:
      - Skip tiles with center_nuclei_count < min_nuc (run check_and_annotate first).
      - Convert sparse COO (H, W, C) -> dense (C, H, W).
      - Apply per-dataset SVD scale factor and clamp to [-10, 10].
      - Save processed tiles to dst_dir/<subdir_name>/*.pt.

    Incremental: tiles already present in dst_dir are not reprocessed.

    Args:
        src_dir: Raw annotated tile directory (contains per-dataset subdirs).
        dst_dir: Output directory for processed tiles.
        min_nuc: Minimum unique nuclei count in centre crop to keep a tile.
    """
    subdirs = [
        os.path.join(src_dir, d) for d in sorted(os.listdir(src_dir))
        if os.path.isdir(os.path.join(src_dir, d))
    ]

    total_raw = total_kept = 0

    for subdir in subdirs:
        subdir_name = os.path.basename(subdir)
        file_paths  = glob.glob(os.path.join(subdir, "*.pt"))
        raw_count   = len(file_paths)
        if raw_count == 0:
            continue
        total_raw += raw_count

        scale    = _infer_scale(subdir_name)
        save_dir = os.path.join(dst_dir, subdir_name)
        os.makedirs(save_dir, exist_ok=True)
        kept = 0

        for path in file_paths:
            filename  = os.path.basename(path)
            save_path = os.path.join(save_dir, filename)

            if os.path.exists(save_path):  # incremental
                kept += 1
                continue

            tile_data = torch.load(path, weights_only=True)
            if tile_data.get("center_nuclei_count", 0) < min_nuc:
                continue

            processed = {
                "input_expr":     (tile_data["input_expr"].to_dense().permute(2, 0, 1) * scale).clamp(-10, 10),
                "input_nuclei":   tile_data["input_nuclei"].to_dense().permute(2, 0, 1).float(),
                "target_expr":    (tile_data["target_expr"].to_dense().permute(2, 0, 1) * scale).clamp(-10, 10),
                "target_cell_id": tile_data["target_cell_id"].to_dense().permute(2, 0, 1).float(),
            }
            torch.save(processed, save_path)
            kept += 1

        total_kept += kept
        print(f"{subdir_name}: raw={raw_count}, kept={kept}")

    print("-" * 40)
    print(f"Total: raw={total_raw}, kept={total_kept}")


class SpatialTranscriptomicsDataset(Dataset):
    """Dataset for processed (dense, scaled) spatial transcriptomics tiles.

    Expects tiles produced by build_filtered_dataset():
      - Format: dense tensors in (C, H, W) layout, already scaled and clamped.
      - Directory layout: data_dir/<dataset_name>/*.pt
    """
    def __init__(self, data_dir: str):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.pt")))
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found under {data_dir}/*/")
        sample = torch.load(self.file_paths[0], weights_only=True)
        self.n_genes: int = sample["target_expr"].shape[0]  # (C, H, W) -> C

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return torch.load(self.file_paths[idx], weights_only=True)


def build_dataloader(
    data_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader from a preprocessed dataset directory.

    Expects data_dir produced by build_filtered_dataset(), containing
    per-dataset subdirs with dense, scaled *.pt tiles.
    """
    dataset = SpatialTranscriptomicsDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

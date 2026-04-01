import glob
import os
from typing import List

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Per-dataset global scale factors to normalise SVD features to std ≈ 1.
# Each dataset's SVD was computed independently with its own StandardScaler,
# resulting in different effective scales (measured on non-zero pixels):
#   CESC  std ≈ 1.03  → scale 1/1.03 ≈ 0.97
#   NSCLC std ≈ 0.42  → scale 1/0.42 ≈ 2.38
#   PRAD  std ≈ 0.34  → scale 1/0.34 ≈ 2.94
_DATASET_SCALE: dict[str, float] = {
    "SVD_CESC":  1.0 / 1.03,
    "SVD_NSCLC": 1.0 / 0.42,
    "SVD_PRAD":  1.0 / 0.34,
}
_DEFAULT_SCALE = 1.0


def _infer_scale(data_dir: str) -> float:
    """Return the scale factor for data_dir based on its basename."""
    name = os.path.basename(os.path.normpath(data_dir))
    return _DATASET_SCALE.get(name, _DEFAULT_SCALE)


class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, data_dir: str):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        # Auto-detect gene count from first tile
        sample = torch.load(self.file_paths[0], weights_only=True)
        self.n_genes: int = sample["target_expr"].shape[2]  # [H, W, C] -> C
        # Scale factor to bring SVD features to std ≈ 1 across all datasets
        self.scale: float = _infer_scale(data_dir)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tile = torch.load(self.file_paths[idx], weights_only=True)
        # sparse COO -> dense, then (H, W, C) -> (C, H, W)
        input_expr    = tile["input_expr"].to_dense().permute(2, 0, 1) * self.scale
        input_nuclei  = tile["input_nuclei"].to_dense().permute(2, 0, 1).float()
        target_expr   = tile["target_expr"].to_dense().permute(2, 0, 1) * self.scale
        target_cell_id = tile["target_cell_id"].to_dense().permute(2, 0, 1).float()
        return {
            "input_expr":    input_expr,
            "input_nuclei":  input_nuclei,
            "target_expr":   target_expr,
            "target_cell_id": target_cell_id,
        }


def build_dataloader(
    data_dirs: "str | List[str]",
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Build a DataLoader from one or more SVD tile directories.

    When multiple directories are given, datasets are concatenated and each
    is independently rescaled to std ≈ 1 before mixing.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    datasets = [SpatialTranscriptomicsDataset(d) for d in data_dirs]
    assert len({ds.n_genes for ds in datasets}) == 1, \
        "All datasets must have the same latent dimension (n_genes)"
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    # Expose n_genes on the combined dataset for convenience
    combined.n_genes = datasets[0].n_genes  # type: ignore[attr-defined]
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

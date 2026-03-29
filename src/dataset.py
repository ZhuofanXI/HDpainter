import glob
import os

import torch
from torch.utils.data import DataLoader, Dataset


class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, data_dir: str):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        # Auto-detect gene count from first tile
        sample = torch.load(self.file_paths[0], weights_only=True)
        self.n_genes: int = sample["target_expr"].shape[2]  # [H, W, C] -> C

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tile = torch.load(self.file_paths[idx], weights_only=True)
        # sparse COO -> dense, then (H, W, C) -> (C, H, W)
        input_expr = tile["input_expr"].to_dense().permute(2, 0, 1)
        input_nuclei = tile["input_nuclei"].to_dense().permute(2, 0, 1).float()
        target_expr = tile["target_expr"].to_dense().permute(2, 0, 1)
        target_cell_id = tile["target_cell_id"].to_dense().permute(2, 0, 1).float()
        return {
            "input_expr": input_expr,
            "input_nuclei": input_nuclei,
            "target_expr": target_expr,
            "target_cell_id": target_cell_id,
        }


def build_dataloader(
    data_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = SpatialTranscriptomicsDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        # spawn avoids CUDA re-init errors when workers are created after
        # the main process has already initialised a CUDA context
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

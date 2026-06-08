from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def tile_collate_fn(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    if batch[0]["tile_input"].ndim == 4:
        if len(batch) == 1:
            item = batch[0]
            return {
                "tile_input": item["tile_input"],
                "tile_index": item["tile_index"],
                "tile_center_yx": item["tile_center_yx"],
                "instance_centers_yx": item["instance_centers_yx"],
                "instance_seed_nmfs": item["instance_seed_nmfs"],
                "instance_nucleus_areas": item["instance_nucleus_areas"],
                "instance_neighbor_seed_nmfs": item["instance_neighbor_seed_nmfs"],
                "instance_neighbor_nucleus_areas": item["instance_neighbor_nucleus_areas"],
                "instance_neighbor_positions": item["instance_neighbor_positions"],
                "instance_neighbor_valid": item["instance_neighbor_valid"],
                "instance_ellipse_targets": item["instance_ellipse_targets"],
                "instance_boundary_radius_targets": item["instance_boundary_radius_targets"],
                "instance_boundary_fourier_targets": item["instance_boundary_fourier_targets"],
                "instance_mask_targets": item["instance_mask_targets"],
                "instance_mask_areas": item["instance_mask_areas"],
                "instance_size_bin_ids": item["instance_size_bin_ids"],
                "instance_latent_targets": item["instance_latent_targets"],
                "instance_counts": torch.tensor([item["instance_count"]], dtype=torch.int64),
                "instance_cell_ids": item["instance_cell_ids"],
            }
        return {
            "tile_input": torch.cat([item["tile_input"] for item in batch], dim=0),
            "tile_index": torch.cat([item["tile_index"] for item in batch], dim=0),
            "tile_center_yx": torch.cat([item["tile_center_yx"] for item in batch], dim=0),
            "instance_centers_yx": [tensor for item in batch for tensor in item["instance_centers_yx"]],
            "instance_seed_nmfs": [tensor for item in batch for tensor in item["instance_seed_nmfs"]],
            "instance_nucleus_areas": [tensor for item in batch for tensor in item["instance_nucleus_areas"]],
            "instance_neighbor_seed_nmfs": [tensor for item in batch for tensor in item["instance_neighbor_seed_nmfs"]],
            "instance_neighbor_nucleus_areas": [tensor for item in batch for tensor in item["instance_neighbor_nucleus_areas"]],
            "instance_neighbor_positions": [tensor for item in batch for tensor in item["instance_neighbor_positions"]],
            "instance_neighbor_valid": [tensor for item in batch for tensor in item["instance_neighbor_valid"]],
            "instance_ellipse_targets": [tensor for item in batch for tensor in item["instance_ellipse_targets"]],
            "instance_boundary_radius_targets": [tensor for item in batch for tensor in item["instance_boundary_radius_targets"]],
            "instance_boundary_fourier_targets": [tensor for item in batch for tensor in item["instance_boundary_fourier_targets"]],
            "instance_mask_targets": [tensor for item in batch for tensor in item["instance_mask_targets"]],
            "instance_mask_areas": [tensor for item in batch for tensor in item["instance_mask_areas"]],
            "instance_size_bin_ids": [tensor for item in batch for tensor in item["instance_size_bin_ids"]],
            "instance_latent_targets": [tensor for item in batch for tensor in item["instance_latent_targets"]],
            "instance_counts": torch.tensor([item["instance_count"] for item in batch], dtype=torch.int64),
            "instance_cell_ids": [ids for item in batch for ids in item["instance_cell_ids"]],
        }

    return {
        "tile_input": torch.stack([item["tile_input"] for item in batch], dim=0),
        "tile_index": torch.tensor([item["tile_index"] for item in batch], dtype=torch.int64),
        "tile_center_yx": torch.stack([item["tile_center_yx"] for item in batch], dim=0),
        "instance_centers_yx": [item["instance_centers_yx"] for item in batch],
        "instance_seed_nmfs": [item["instance_seed_nmfs"] for item in batch],
        "instance_nucleus_areas": [item["instance_nucleus_areas"] for item in batch],
        "instance_neighbor_seed_nmfs": [item["instance_neighbor_seed_nmfs"] for item in batch],
        "instance_neighbor_nucleus_areas": [item["instance_neighbor_nucleus_areas"] for item in batch],
        "instance_neighbor_positions": [item["instance_neighbor_positions"] for item in batch],
        "instance_neighbor_valid": [item["instance_neighbor_valid"] for item in batch],
        "instance_ellipse_targets": [item["instance_ellipse_targets"] for item in batch],
        "instance_boundary_radius_targets": [item["instance_boundary_radius_targets"] for item in batch],
        "instance_boundary_fourier_targets": [item["instance_boundary_fourier_targets"] for item in batch],
        "instance_mask_targets": [item["instance_mask_targets"] for item in batch],
        "instance_mask_areas": [item["instance_mask_areas"] for item in batch],
        "instance_size_bin_ids": [item["instance_size_bin_ids"] for item in batch],
        "instance_latent_targets": [item["instance_latent_targets"] for item in batch],
        "instance_counts": torch.tensor([item["instance_count"] for item in batch], dtype=torch.int64),
        "instance_cell_ids": [item["instance_cell_ids"] for item in batch],
    }


class SpatialTranscriptomicsDataset(Dataset):
    neighbor_position_dim = 3

    def __init__(
        self,
        data_dir: str | Path,
        min_nuc: int = 0,
        seed_size: int = 5,
        canvas_size: int = 33,
        neighbor_k: int = 4,
        aggregate_radius: int = 5,
        seed_sector_bins: int = 8,
        neighbor_direction_bins: int = 8,
        size_bin_edges: list[float] | tuple[float, ...] | None = None,
        chunk_manifest: str | Path | None = None,
        chunk_split: str = "full",
    ):
        self.data_path = Path(data_dir)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")
        if not (self.data_path.is_file() and self.data_path.suffix == ".h5"):
            raise ValueError("Expected a dense tile H5 dataset.")

        self.min_nuc = int(min_nuc)
        self.seed_size = int(seed_size)
        self.canvas_size = int(canvas_size)
        self.neighbor_k = int(neighbor_k)
        self.aggregate_radius = int(aggregate_radius)
        self.aggregate_size = 2 * self.aggregate_radius + 1
        self.canvas_radius = float(self.canvas_size) / 2.0
        self.size_bin_edges = (
            np.asarray(size_bin_edges, dtype=np.float32)
            if size_bin_edges is not None
            else np.asarray([0.0, 16.0, 24.0, 32.0, 40.0, 48.0, 64.0, 96.0, 999999.0], dtype=np.float32)
        )
        if self.size_bin_edges.ndim != 1 or self.size_bin_edges.shape[0] < 2:
            raise ValueError("size_bin_edges must contain at least two ordered values.")
        self.size_bin_count = int(self.size_bin_edges.shape[0] - 1)
        self.chunk_manifest_path = Path(chunk_manifest) if chunk_manifest else None
        self.chunk_split = str(chunk_split).strip().lower()
        if self.chunk_split not in {"full", "train", "val"}:
            raise ValueError("chunk_split must be one of {'full', 'train', 'val'}.")
        self.storage_format = "tile_h5"
        self.chunk_mode = False
        self.chunk_offsets: np.ndarray | None = None
        self.chunk_tile_indices: np.ndarray | None = None
        self.chunk_local_instance_indices: np.ndarray | None = None
        self.chunk_instance_counts: np.ndarray | None = None

        self._h5_file: h5py.File | None = None
        self.total_samples = 0
        self.kept_samples = 0
        self.total_instances = 0
        self.patch_size = 0
        self.latent_dim = 0
        self.expr_channels = 0
        self.seed_nmf_dim = 0
        self.tile_centers_all: list[tuple[int, int]] = []
        self.tile_instance_counts_all: list[int] = []
        self.source_tile_indices: list[int] = []
        self.has_cached_seed_features = False
        self.chunk_h5_prefix: str | None = None
        self.use_refined_targets = False
        self.instance_use_for_training: np.ndarray | None = None

        if self._is_instance_chunk_h5():
            self.storage_format = "instance_chunk_h5"
            self._prepare_instance_chunk_metadata()
        else:
            self.indices = self._prepare_indices()
            self._load_chunk_manifest()
        self.circle_mask = self._build_circle_mask(self.aggregate_radius)
        self.canvas_circle_mask = self._build_canvas_circle_mask(self.canvas_size)

    def _is_instance_chunk_h5(self) -> bool:
        with h5py.File(self.data_path, "r") as fr:
            return str(fr.attrs.get("dataset_format", "")).strip() == "instance_chunk_h5_v1"

    def _prepare_instance_chunk_metadata(self) -> None:
        if self.chunk_split not in {"train", "val"}:
            raise ValueError("Instance-chunk H5 requires chunk_split to be 'train' or 'val'.")
        with h5py.File(self.data_path, "r") as fr:
            self.patch_size = int(fr.attrs.get("patch_size", 0))
            self.latent_dim = int(fr.attrs["latent_dim"])
            self.expr_channels = int(fr.attrs["expr_channels"])
            self.seed_nmf_dim = int(fr.attrs.get("seed_nmf_dim", self.expr_channels))
            self.aggregate_radius = int(fr.attrs.get("aggregate_radius", self.aggregate_radius))
            self.aggregate_size = 2 * self.aggregate_radius + 1
            self.canvas_size = int(fr.attrs.get("canvas_size", self.canvas_size))
            self.canvas_radius = float(self.canvas_size) / 2.0
            self.has_cached_seed_features = True
            self.total_samples = int(fr.attrs.get("n_source_tiles", 0))
            self.kept_samples = self.total_samples
            self.total_instances = int(fr.attrs.get("n_source_instances", 0))
            prefix = self.chunk_split
            required = {
                f"{prefix}_chunk_tile_offsets",
                f"{prefix}_chunk_instance_offsets",
                f"{prefix}_tile_input_pool",
                f"{prefix}_tile_index_pool",
                f"{prefix}_tile_center_yx_pool",
                f"{prefix}_instance_tile_ptr_pool",
                f"{prefix}_instance_centers_yx_pool",
                f"{prefix}_instance_seed_nmfs_pool",
                f"{prefix}_instance_nucleus_areas_pool",
                f"{prefix}_instance_neighbor_seed_nmfs_pool",
                f"{prefix}_instance_neighbor_nucleus_areas_pool",
                f"{prefix}_instance_neighbor_positions_pool",
                f"{prefix}_instance_neighbor_valid_pool",
                f"{prefix}_instance_ellipse_targets_pool",
                f"{prefix}_instance_boundary_radius_targets_pool",
                f"{prefix}_instance_boundary_fourier_targets_pool",
                f"{prefix}_instance_mask_targets_pool",
                f"{prefix}_instance_mask_areas_pool",
                f"{prefix}_instance_size_bin_ids_pool",
                f"{prefix}_instance_latent_targets_pool",
            }
            missing = sorted(name for name in required if name not in fr)
            if missing:
                raise ValueError(f"{self.data_path} missing required instance-chunk keys: {missing}")
            self.use_refined_targets = (
                f"{prefix}_instance_mask_targets_refined_pool" in fr
                and f"{prefix}_instance_refined_ellipse_param_pool" in fr
            )
            self.chunk_h5_prefix = prefix
            chunk_offsets = fr[f"{prefix}_chunk_instance_offsets"][:].astype(np.int64, copy=False)
            raw_chunk_counts = np.diff(chunk_offsets).astype(np.int64, copy=False)
            flag_key = f"{prefix}_instance_use_for_training_pool"
            if flag_key in fr:
                self.instance_use_for_training = fr[flag_key][:].astype(np.uint8, copy=False)
                filtered_counts = np.asarray(
                    [
                        int(self.instance_use_for_training[int(chunk_offsets[idx]):int(chunk_offsets[idx + 1])].sum())
                        for idx in range(raw_chunk_counts.shape[0])
                    ],
                    dtype=np.int64,
                )
                active_chunk_ids = np.nonzero(filtered_counts > 0)[0].astype(np.int64, copy=False)
                self.chunk_offsets = chunk_offsets
                self.chunk_instance_counts = filtered_counts[active_chunk_ids]
                self.kept_samples = int(active_chunk_ids.shape[0])
                self.indices = active_chunk_ids.tolist()
            else:
                self.chunk_offsets = chunk_offsets
                self.chunk_instance_counts = raw_chunk_counts
                self.kept_samples = int(fr.attrs.get(f"n_{prefix}_chunks", self.chunk_instance_counts.shape[0]))
                self.indices = list(range(self.chunk_instance_counts.shape[0]))

    def _prepare_indices(self) -> list[int]:
        with h5py.File(self.data_path, "r") as fr:
            required = {
                "x_low",
                "tile_center_yx",
                "core_nucleus_count",
                "instance_offsets",
                "cell_latent_pool",
                "cell_id_pool",
                "nucleus_mask_pool",
                "xenium_mask_pool",
                "ellipse_param_pool",
                "boundary_radius_target_pool",
                "boundary_fourier_target_pool",
            }
            missing = sorted(name for name in required if name not in fr)
            if missing:
                raise ValueError(f"{self.data_path} missing required keys: {missing}")
            if "nucleus_centers_yx_pool" not in fr and "cell_centers_yx_pool" not in fr:
                raise ValueError(f"{self.data_path} missing both nucleus_centers_yx_pool and cell_centers_yx_pool.")

            self.patch_size = int(fr.attrs["patch_size"])
            self.latent_dim = int(fr.attrs["latent_dim"])
            self.expr_channels = int(fr["x_low"].shape[1])
            self.seed_nmf_dim = self.expr_channels
            self.has_cached_seed_features = bool(
                "seed_feature_pool" in fr and int(fr["seed_feature_pool"].shape[1]) >= self.expr_channels
            )
            self.use_refined_targets = ("xenium_mask_pool_refined" in fr) and ("ellipse_param_pool_refined" in fr)

            n_tiles = int(fr.attrs["n_samples"])
            self.total_samples = n_tiles
            core_nucleus_count = fr["core_nucleus_count"][:]
            instance_offsets = fr["instance_offsets"][:]
            tile_centers = fr["tile_center_yx"][:]

            keep_indices: list[int] = []
            for tile_idx in range(n_tiles):
                if self.min_nuc > 0 and int(core_nucleus_count[tile_idx]) < self.min_nuc:
                    continue
                start = int(instance_offsets[tile_idx])
                end = int(instance_offsets[tile_idx + 1])
                if end <= start:
                    continue
                keep_indices.append(tile_idx)
                n_instances = end - start
                self.total_instances += n_instances
                center = tile_centers[tile_idx]
                self.tile_centers_all.append((int(round(float(center[0]))), int(round(float(center[1])))))
                self.tile_instance_counts_all.append(int(n_instances))
                self.source_tile_indices.append(int(tile_idx))

            self.kept_samples = len(keep_indices)

        if not keep_indices:
            raise ValueError(f"No tiles remain in {self.data_path} after applying min_nuc={self.min_nuc}.")
        return keep_indices

    @staticmethod
    def _build_circle_mask(radius: int) -> torch.Tensor:
        size = 2 * radius + 1
        yy = torch.arange(size, dtype=torch.float32).unsqueeze(1)
        xx = torch.arange(size, dtype=torch.float32).unsqueeze(0)
        center = float(radius)
        return (((yy - center) ** 2 + (xx - center) ** 2) <= float(radius * radius)).float()

    @staticmethod
    def _build_canvas_circle_mask(size: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (float(size) - 1.0) / 2.0
        radius = float(size) / 2.0
        return ((((yy - center) ** 2 + (xx - center) ** 2) <= (radius ** 2)).float())

    def _size_bin_id_from_area(self, area: float) -> int:
        idx = int(np.digitize(float(area), bins=self.size_bin_edges[1:-1], right=False))
        return int(min(max(idx, 0), self.size_bin_count - 1))

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data_path, "r")
        return self._h5_file

    def _load_chunk_manifest(self) -> None:
        if self.chunk_manifest_path is None:
            return
        if not self.chunk_manifest_path.exists():
            raise FileNotFoundError(f"Chunk manifest does not exist: {self.chunk_manifest_path}")
        with h5py.File(self.chunk_manifest_path, "r") as fr:
            prefix = f"{self.chunk_split}_chunk"
            required = {
                f"{prefix}_offsets",
                f"{prefix}_tile_indices",
                f"{prefix}_local_instance_indices",
                f"{prefix}_instance_counts",
            }
            missing = sorted(name for name in required if name not in fr)
            if missing:
                raise ValueError(f"{self.chunk_manifest_path} missing required keys: {missing}")
            manifest_data_path = fr.attrs.get("input_h5", "")
            manifest_min_nuc = int(fr.attrs.get("min_nuc", self.min_nuc))
            if manifest_data_path and Path(str(manifest_data_path)) != self.data_path:
                raise ValueError(
                    f"Chunk manifest was built for {manifest_data_path}, but dataset uses {self.data_path}."
                )
            if manifest_min_nuc != self.min_nuc:
                raise ValueError(
                    f"Chunk manifest min_nuc={manifest_min_nuc} does not match dataset min_nuc={self.min_nuc}."
                )
            self.chunk_offsets = fr[f"{prefix}_offsets"][:].astype(np.int64, copy=False)
            self.chunk_tile_indices = fr[f"{prefix}_tile_indices"][:].astype(np.int64, copy=False)
            self.chunk_local_instance_indices = fr[f"{prefix}_local_instance_indices"][:].astype(np.int64, copy=False)
            self.chunk_instance_counts = fr[f"{prefix}_instance_counts"][:].astype(np.int64, copy=False)
        if (
            self.chunk_offsets.ndim != 1
            or self.chunk_tile_indices.ndim != 1
            or self.chunk_local_instance_indices.ndim != 1
            or self.chunk_instance_counts.ndim != 1
        ):
            raise ValueError("Chunk manifest arrays must be one-dimensional.")
        if self.chunk_offsets.shape[0] != self.chunk_instance_counts.shape[0] + 1:
            raise ValueError("chunk_offsets must have length n_chunks + 1.")
        if self.chunk_tile_indices.shape[0] != self.chunk_local_instance_indices.shape[0]:
            raise ValueError("chunk_tile_indices and chunk_local_instance_indices must have the same length.")
        self.chunk_mode = True

    def has_sample_centers(self) -> bool:
        return not self.chunk_mode

    def get_sample_centers(self) -> list[tuple[int, int] | None]:
        if self.chunk_mode or self.storage_format == "instance_chunk_h5":
            return [None for _ in range(len(self))]
        return list(self.tile_centers_all)

    def get_sample_instance_counts(self) -> list[int]:
        if (self.chunk_mode or self.storage_format == "instance_chunk_h5") and self.chunk_instance_counts is not None:
            return [int(value) for value in self.chunk_instance_counts.tolist()]
        return list(self.tile_instance_counts_all)

    def get_source_tile_indices(self) -> list[int]:
        return list(self.source_tile_indices)

    def __len__(self) -> int:
        if self.storage_format == "instance_chunk_h5":
            return len(self.indices)
        if self.chunk_mode and self.chunk_instance_counts is not None:
            return int(self.chunk_instance_counts.shape[0])
        return len(self.indices)

    @staticmethod
    def _crop_with_padding(tensor: torch.Tensor, center_y: float, center_x: float, size: int) -> torch.Tensor:
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

    def _aggregate_circle_nmf(self, expr_map: torch.Tensor, center_y: float, center_x: float) -> torch.Tensor:
        crop = self._crop_with_padding(expr_map, center_y, center_x, self.aggregate_size).float()
        ones = torch.ones((1, expr_map.shape[-2], expr_map.shape[-1]), dtype=torch.float32)
        valid_crop = self._crop_with_padding(ones, center_y, center_x, self.aggregate_size)
        weight = self.circle_mask.unsqueeze(0) * valid_crop
        denom = weight.sum().clamp_min(1.0)
        return (crop * weight).sum(dim=(-1, -2)) / denom

    def _build_neighbor_context(
        self,
        centers: torch.Tensor,
        current_index: int,
        seed_nmfs: torch.Tensor,
        nucleus_areas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        neighbor_seed_nmfs = torch.zeros((self.neighbor_k, self.seed_nmf_dim), dtype=torch.float32)
        neighbor_nucleus_areas = torch.zeros((self.neighbor_k, 1), dtype=torch.float32)
        neighbor_positions = torch.zeros((self.neighbor_k, self.neighbor_position_dim), dtype=torch.float32)
        neighbor_valid = torch.zeros((self.neighbor_k,), dtype=torch.float32)

        if centers.shape[0] <= 1:
            return neighbor_seed_nmfs, neighbor_nucleus_areas, neighbor_positions, neighbor_valid

        current = centers[current_index]
        other_indices = [idx for idx in range(int(centers.shape[0])) if idx != current_index]
        other_centers = centers[other_indices]
        deltas = other_centers - current.unsqueeze(0)
        dists = torch.sqrt(torch.clamp((deltas**2).sum(dim=1), min=1e-8))
        order = torch.argsort(dists)[: self.neighbor_k]
        half_canvas = max(float(self.canvas_size // 2), 1.0)

        for slot, ordered_idx in enumerate(order.tolist()):
            src_idx = other_indices[ordered_idx]
            dy = float(deltas[ordered_idx, 0].item())
            dx = float(deltas[ordered_idx, 1].item())
            dist = float(dists[ordered_idx].item())
            neighbor_seed_nmfs[slot] = seed_nmfs[src_idx]
            neighbor_nucleus_areas[slot, 0] = nucleus_areas[src_idx, 0]
            neighbor_positions[slot] = torch.tensor(
                [dy / half_canvas, dx / half_canvas, dist / half_canvas],
                dtype=torch.float32,
            )
            neighbor_valid[slot] = 1.0

        return neighbor_seed_nmfs, neighbor_nucleus_areas, neighbor_positions, neighbor_valid

    def _get_tile_item(self, tile_idx: int, local_instance_indices: np.ndarray | None = None) -> dict:
        fr = self._get_h5()
        tile_start = int(fr["instance_offsets"][tile_idx])
        tile_end = int(fr["instance_offsets"][tile_idx + 1])
        tile_n_instances = tile_end - tile_start
        if local_instance_indices is None:
            local_indices = np.arange(tile_n_instances, dtype=np.int64)
        else:
            local_indices = np.asarray(local_instance_indices, dtype=np.int64)
            if local_indices.ndim != 1:
                raise ValueError("local_instance_indices must be one-dimensional.")
            if local_indices.size == 0:
                raise ValueError(f"Tile {tile_idx} received an empty local_instance_indices selection.")
            if local_indices.min() < 0 or local_indices.max() >= tile_n_instances:
                raise IndexError(
                    f"Tile {tile_idx} local indices out of range: min={int(local_indices.min())} "
                    f"max={int(local_indices.max())} tile_n_instances={tile_n_instances}"
                )
        pool_indices = tile_start + local_indices
        read_order = np.argsort(pool_indices)
        sorted_pool_indices = pool_indices[read_order]
        restore_order = np.argsort(read_order)
        n_instances = int(local_indices.shape[0])

        x_low = torch.from_numpy(fr["x_low"][tile_idx]).float()
        tile_input = x_low

        center_key = "nucleus_centers_yx_pool" if "nucleus_centers_yx_pool" in fr else "cell_centers_yx_pool"
        centers = torch.from_numpy(fr[center_key][sorted_pool_indices]).float()[restore_order]
        nucleus_masks = torch.from_numpy(fr["nucleus_mask_pool"][sorted_pool_indices]).float()[restore_order]
        nucleus_areas = nucleus_masks.sum(dim=(-1, -2), keepdim=False).unsqueeze(1)
        mask_key = "xenium_mask_pool_refined" if ("xenium_mask_pool_refined" in fr) else "xenium_mask_pool"
        ellipse_key = "ellipse_param_pool_refined" if ("ellipse_param_pool_refined" in fr) else "ellipse_param_pool"
        self.use_refined_targets = self.use_refined_targets or (mask_key != "xenium_mask_pool")
        mask_targets_full = torch.from_numpy(fr[mask_key][sorted_pool_indices]).float()[restore_order]
        ellipse_targets = torch.from_numpy(fr[ellipse_key][sorted_pool_indices]).float()[restore_order]
        boundary_radius_key = (
            "boundary_radius_target_pool_refined"
            if ("boundary_radius_target_pool_refined" in fr)
            else "boundary_radius_target_pool"
        )
        boundary_radius_targets = torch.from_numpy(fr[boundary_radius_key][sorted_pool_indices]).float()[restore_order]
        boundary_fourier_targets = torch.from_numpy(fr["boundary_fourier_target_pool"][sorted_pool_indices]).float()[restore_order]
        latent_targets = torch.from_numpy(fr["cell_latent_pool"][sorted_pool_indices]).float()[restore_order]

        cell_ids_raw = fr["cell_id_pool"][sorted_pool_indices]
        cell_ids = np.asarray(
            [item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item) for item in cell_ids_raw],
            dtype=object,
        )[restore_order]

        if self.has_cached_seed_features:
            seed_features = torch.from_numpy(fr["seed_feature_pool"][sorted_pool_indices]).float()[restore_order]
            seed_nmfs = seed_features[:, : self.seed_nmf_dim]
        else:
            seed_nmfs = torch.zeros((n_instances, self.seed_nmf_dim), dtype=torch.float32)
            for inst_idx in range(n_instances):
                center_y = float(centers[inst_idx, 0].item())
                center_x = float(centers[inst_idx, 1].item())
                seed_nmfs[inst_idx] = self._aggregate_circle_nmf(x_low, center_y, center_x)

        neighbor_seed_nmfs = torch.zeros((n_instances, self.neighbor_k, self.seed_nmf_dim), dtype=torch.float32)
        neighbor_nucleus_areas = torch.zeros((n_instances, self.neighbor_k, 1), dtype=torch.float32)
        neighbor_positions = torch.zeros((n_instances, self.neighbor_k, self.neighbor_position_dim), dtype=torch.float32)
        neighbor_valid = torch.zeros((n_instances, self.neighbor_k), dtype=torch.float32)
        mask_targets = torch.zeros((n_instances, 1, self.canvas_size, self.canvas_size), dtype=torch.float32)
        mask_areas = torch.zeros((n_instances,), dtype=torch.float32)
        size_bin_ids = torch.zeros((n_instances,), dtype=torch.int64)
        canvas_circle = self.canvas_circle_mask.unsqueeze(0)

        for inst_idx in range(n_instances):
            center_y = float(centers[inst_idx, 0].item())
            center_x = float(centers[inst_idx, 1].item())

            n_seed, n_area, n_pos, n_valid = self._build_neighbor_context(
                centers=centers,
                current_index=inst_idx,
                seed_nmfs=seed_nmfs,
                nucleus_areas=nucleus_areas,
            )
            neighbor_seed_nmfs[inst_idx] = n_seed
            neighbor_nucleus_areas[inst_idx] = n_area
            neighbor_positions[inst_idx] = n_pos
            neighbor_valid[inst_idx] = n_valid

            raw_mask_crop = self._crop_with_padding(
                mask_targets_full[inst_idx].unsqueeze(0),
                center_y,
                center_x,
                self.canvas_size,
            )
            mask_targets[inst_idx] = raw_mask_crop * canvas_circle
            area = float(mask_targets_full[inst_idx].sum().item())
            mask_areas[inst_idx] = area
            size_bin_ids[inst_idx] = self._size_bin_id_from_area(area)

        return {
            "tile_input": tile_input,
            "tile_index": int(tile_idx),
            "tile_center_yx": torch.from_numpy(fr["tile_center_yx"][tile_idx]).float(),
            "instance_centers_yx": centers,
            "instance_seed_nmfs": seed_nmfs,
            "instance_nucleus_areas": nucleus_areas,
            "instance_neighbor_seed_nmfs": neighbor_seed_nmfs,
            "instance_neighbor_nucleus_areas": neighbor_nucleus_areas,
            "instance_neighbor_positions": neighbor_positions,
            "instance_neighbor_valid": neighbor_valid,
            "instance_ellipse_targets": ellipse_targets,
            "instance_boundary_radius_targets": boundary_radius_targets,
            "instance_boundary_fourier_targets": boundary_fourier_targets,
            "instance_mask_targets": mask_targets,
            "instance_mask_areas": mask_areas,
            "instance_size_bin_ids": size_bin_ids,
            "instance_latent_targets": latent_targets,
            "instance_count": int(n_instances),
            "instance_cell_ids": cell_ids,
        }

    def _get_instance_chunk_item(self, index: int) -> dict:
        assert self.chunk_h5_prefix is not None
        assert self.chunk_offsets is not None
        prefix = self.chunk_h5_prefix
        chunk_id = int(self.indices[index])
        fr = self._get_h5()
        tile_offsets = fr[f"{prefix}_chunk_tile_offsets"]
        instance_offsets = fr[f"{prefix}_chunk_instance_offsets"]
        t0 = int(tile_offsets[chunk_id])
        t1 = int(tile_offsets[chunk_id + 1])
        i0 = int(instance_offsets[chunk_id])
        i1 = int(instance_offsets[chunk_id + 1])
        tile_input = torch.from_numpy(fr[f"{prefix}_tile_input_pool"][t0:t1]).float()
        tile_index = torch.from_numpy(fr[f"{prefix}_tile_index_pool"][t0:t1].astype(np.int64, copy=False))
        tile_center_yx = torch.from_numpy(fr[f"{prefix}_tile_center_yx_pool"][t0:t1]).float()
        selected: slice | np.ndarray = slice(i0, i1)
        if self.instance_use_for_training is not None:
            local_keep = np.nonzero(self.instance_use_for_training[i0:i1] > 0)[0]
            if local_keep.size == 0:
                raise ValueError(f"Chunk {chunk_id} has no training-enabled instances.")
            selected = i0 + local_keep
        instance_tile_ptr = fr[f"{prefix}_instance_tile_ptr_pool"][selected].astype(np.int64, copy=False) - t0

        instance_centers = torch.from_numpy(fr[f"{prefix}_instance_centers_yx_pool"][selected]).float()
        seed_nmfs = torch.from_numpy(fr[f"{prefix}_instance_seed_nmfs_pool"][selected]).float()
        nucleus_areas = torch.from_numpy(fr[f"{prefix}_instance_nucleus_areas_pool"][selected]).float()
        neighbor_seed_nmfs = torch.from_numpy(fr[f"{prefix}_instance_neighbor_seed_nmfs_pool"][selected]).float()
        neighbor_nucleus_areas = torch.from_numpy(fr[f"{prefix}_instance_neighbor_nucleus_areas_pool"][selected]).float()
        neighbor_positions = torch.from_numpy(fr[f"{prefix}_instance_neighbor_positions_pool"][selected]).float()
        neighbor_valid = torch.from_numpy(fr[f"{prefix}_instance_neighbor_valid_pool"][selected]).float()
        ellipse_key = (
            f"{prefix}_instance_refined_ellipse_param_pool"
            if self.use_refined_targets and f"{prefix}_instance_refined_ellipse_param_pool" in fr
            else f"{prefix}_instance_ellipse_targets_pool"
        )
        mask_key = (
            f"{prefix}_instance_mask_targets_refined_pool"
            if self.use_refined_targets and f"{prefix}_instance_mask_targets_refined_pool" in fr
            else f"{prefix}_instance_mask_targets_pool"
        )
        ellipse_targets = torch.from_numpy(fr[ellipse_key][selected]).float()
        boundary_radius_key = (
            f"{prefix}_instance_boundary_radius_targets_refined_pool"
            if self.use_refined_targets and f"{prefix}_instance_boundary_radius_targets_refined_pool" in fr
            else f"{prefix}_instance_boundary_radius_targets_pool"
        )
        boundary_radius_targets = torch.from_numpy(fr[boundary_radius_key][selected]).float()
        boundary_fourier_targets = torch.from_numpy(fr[f"{prefix}_instance_boundary_fourier_targets_pool"][selected]).float()
        mask_targets = torch.from_numpy(fr[mask_key][selected]).float()
        mask_areas = torch.from_numpy(fr[f"{prefix}_instance_mask_areas_pool"][selected]).float()
        size_bin_ids = torch.from_numpy(fr[f"{prefix}_instance_size_bin_ids_pool"][selected].astype(np.int64, copy=False))
        latent_targets = torch.from_numpy(fr[f"{prefix}_instance_latent_targets_pool"][selected]).float()

        per_tile_centers: list[torch.Tensor] = []
        per_tile_seed_nmfs: list[torch.Tensor] = []
        per_tile_nucleus_areas: list[torch.Tensor] = []
        per_tile_neighbor_seed_nmfs: list[torch.Tensor] = []
        per_tile_neighbor_nucleus_areas: list[torch.Tensor] = []
        per_tile_neighbor_positions: list[torch.Tensor] = []
        per_tile_neighbor_valid: list[torch.Tensor] = []
        per_tile_ellipse_targets: list[torch.Tensor] = []
        per_tile_boundary_radius_targets: list[torch.Tensor] = []
        per_tile_boundary_fourier_targets: list[torch.Tensor] = []
        per_tile_mask_targets: list[torch.Tensor] = []
        per_tile_mask_areas: list[torch.Tensor] = []
        per_tile_size_bin_ids: list[torch.Tensor] = []
        per_tile_latent_targets: list[torch.Tensor] = []
        per_tile_cell_ids: list[np.ndarray] = []

        for local_tile_idx in range(int(tile_input.shape[0])):
            selector = np.nonzero(instance_tile_ptr == local_tile_idx)[0]
            sel_t = torch.from_numpy(selector.astype(np.int64, copy=False))
            per_tile_centers.append(instance_centers.index_select(0, sel_t))
            per_tile_seed_nmfs.append(seed_nmfs.index_select(0, sel_t))
            per_tile_nucleus_areas.append(nucleus_areas.index_select(0, sel_t))
            per_tile_neighbor_seed_nmfs.append(neighbor_seed_nmfs.index_select(0, sel_t))
            per_tile_neighbor_nucleus_areas.append(neighbor_nucleus_areas.index_select(0, sel_t))
            per_tile_neighbor_positions.append(neighbor_positions.index_select(0, sel_t))
            per_tile_neighbor_valid.append(neighbor_valid.index_select(0, sel_t))
            per_tile_ellipse_targets.append(ellipse_targets.index_select(0, sel_t))
            per_tile_boundary_radius_targets.append(boundary_radius_targets.index_select(0, sel_t))
            per_tile_boundary_fourier_targets.append(boundary_fourier_targets.index_select(0, sel_t))
            per_tile_mask_targets.append(mask_targets.index_select(0, sel_t))
            per_tile_mask_areas.append(mask_areas.index_select(0, sel_t))
            per_tile_size_bin_ids.append(size_bin_ids.index_select(0, sel_t))
            per_tile_latent_targets.append(latent_targets.index_select(0, sel_t))
            per_tile_cell_ids.append(np.empty((int(sel_t.shape[0]),), dtype=object))

        return {
            "tile_input": tile_input,
            "tile_index": tile_index,
            "tile_center_yx": tile_center_yx,
            "instance_centers_yx": per_tile_centers,
            "instance_seed_nmfs": per_tile_seed_nmfs,
            "instance_nucleus_areas": per_tile_nucleus_areas,
            "instance_neighbor_seed_nmfs": per_tile_neighbor_seed_nmfs,
            "instance_neighbor_nucleus_areas": per_tile_neighbor_nucleus_areas,
            "instance_neighbor_positions": per_tile_neighbor_positions,
            "instance_neighbor_valid": per_tile_neighbor_valid,
            "instance_ellipse_targets": per_tile_ellipse_targets,
            "instance_boundary_radius_targets": per_tile_boundary_radius_targets,
            "instance_boundary_fourier_targets": per_tile_boundary_fourier_targets,
            "instance_mask_targets": per_tile_mask_targets,
            "instance_mask_areas": per_tile_mask_areas,
            "instance_size_bin_ids": per_tile_size_bin_ids,
            "instance_latent_targets": per_tile_latent_targets,
            "instance_count": int(mask_targets.shape[0]),
            "instance_cell_ids": per_tile_cell_ids,
        }

    def __getitem__(self, index: int) -> dict:
        if self.storage_format == "instance_chunk_h5":
            return self._get_instance_chunk_item(index)
        if not self.chunk_mode:
            tile_idx = self.indices[index]
            return self._get_tile_item(tile_idx)

        assert self.chunk_offsets is not None
        assert self.chunk_tile_indices is not None
        assert self.chunk_local_instance_indices is not None
        start = int(self.chunk_offsets[index])
        end = int(self.chunk_offsets[index + 1])
        raw_tile_indices = self.chunk_tile_indices[start:end]
        raw_local_indices = self.chunk_local_instance_indices[start:end]
        tile_to_locals: dict[int, list[int]] = {}
        tile_order: list[int] = []
        for tile_idx, local_idx in zip(raw_tile_indices.tolist(), raw_local_indices.tolist(), strict=False):
            tile_idx = int(tile_idx)
            if tile_idx not in tile_to_locals:
                tile_to_locals[tile_idx] = []
                tile_order.append(tile_idx)
            tile_to_locals[tile_idx].append(int(local_idx))
        tile_items = [
            self._get_tile_item(tile_idx, np.asarray(tile_to_locals[tile_idx], dtype=np.int64))
            for tile_idx in tile_order
        ]
        if not tile_items:
            raise ValueError(f"Chunk {index} is empty in {self.chunk_manifest_path}.")

        return {
            "tile_input": torch.stack([item["tile_input"] for item in tile_items], dim=0),
            "tile_index": torch.tensor([int(item["tile_index"]) for item in tile_items], dtype=torch.int64),
            "tile_center_yx": torch.stack([item["tile_center_yx"] for item in tile_items], dim=0),
            "instance_centers_yx": [item["instance_centers_yx"] for item in tile_items],
            "instance_seed_nmfs": [item["instance_seed_nmfs"] for item in tile_items],
            "instance_nucleus_areas": [item["instance_nucleus_areas"] for item in tile_items],
            "instance_neighbor_seed_nmfs": [item["instance_neighbor_seed_nmfs"] for item in tile_items],
            "instance_neighbor_nucleus_areas": [item["instance_neighbor_nucleus_areas"] for item in tile_items],
            "instance_neighbor_positions": [item["instance_neighbor_positions"] for item in tile_items],
            "instance_neighbor_valid": [item["instance_neighbor_valid"] for item in tile_items],
            "instance_ellipse_targets": [item["instance_ellipse_targets"] for item in tile_items],
            "instance_boundary_radius_targets": [item["instance_boundary_radius_targets"] for item in tile_items],
            "instance_boundary_fourier_targets": [item["instance_boundary_fourier_targets"] for item in tile_items],
            "instance_mask_targets": [item["instance_mask_targets"] for item in tile_items],
            "instance_mask_areas": [item["instance_mask_areas"] for item in tile_items],
            "instance_size_bin_ids": [item["instance_size_bin_ids"] for item in tile_items],
            "instance_latent_targets": [item["instance_latent_targets"] for item in tile_items],
            "instance_count": int(sum(int(item["instance_count"]) for item in tile_items)),
            "instance_cell_ids": [item["instance_cell_ids"] for item in tile_items],
        }


def build_tile_split_indices(
    dataset: SpatialTranscriptomicsDataset,
    val_ratio: float,
    seed: int,
    split_mode: str = "auto",
) -> tuple[list[int], list[int]]:
    all_indices = list(range(len(dataset)))
    if len(all_indices) < 2:
        raise ValueError("Need at least two samples to build a train/val split.")

    split_mode = str(split_mode).strip().lower()
    if split_mode not in {"auto", "random", "region"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    if split_mode in {"auto", "region"} and dataset.has_sample_centers():
        centers = dataset.get_sample_centers()
        valid = [(idx, center) for idx, center in enumerate(centers) if center is not None]
        if valid:
            ys = np.asarray([center[0] for _, center in valid], dtype=np.float32)
            xs = np.asarray([center[1] for _, center in valid], dtype=np.float32)
            frac = float(val_ratio) ** 0.5
            y_threshold = ys.max() - frac * (ys.max() - ys.min())
            x_threshold = xs.max() - frac * (xs.max() - xs.min())
            val_indices = [
                idx
                for idx, center in valid
                if center[0] >= y_threshold and center[1] >= x_threshold
            ]
            if 0 < len(val_indices) < len(all_indices):
                val_set = set(val_indices)
                train_indices = [idx for idx in all_indices if idx not in val_set]
                return train_indices, val_indices
        if split_mode == "region":
            raise ValueError("Region split requested but no valid regional split could be constructed.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(all_indices, dtype=np.int64))
    val_count = max(1, int(round(len(all_indices) * float(val_ratio))))
    val_indices = sorted(int(idx) for idx in perm[:val_count].tolist())
    train_indices = sorted(int(idx) for idx in perm[val_count:].tolist())
    if not train_indices or not val_indices:
        raise ValueError("Random split produced an empty train or val split.")
    return train_indices, val_indices


def build_multiscale_crop_sizes() -> list[int]:
    return [15, 21, 27, 33]


def build_center_seed_mask(canvas_size: int, seed_radius: int) -> torch.Tensor:
    coords = torch.arange(canvas_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (float(canvas_size) - 1.0) / 2.0
    radius = float(seed_radius)
    mask = (((yy - center) ** 2 + (xx - center) ** 2) <= (radius ** 2)).float()
    return mask.unsqueeze(0)


def build_coord_maps(canvas_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, canvas_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return yy, xx


def build_neighbor_maps(
    neighbor_positions: torch.Tensor,
    neighbor_valid: torch.Tensor,
    coord_y: torch.Tensor,
    coord_x: torch.Tensor,
    canvas_size: int,
    seed_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_map = torch.zeros((1, canvas_size, canvas_size), dtype=torch.float32)
    distance_map = torch.zeros((1, canvas_size, canvas_size), dtype=torch.float32)

    valid_slots = torch.nonzero(neighbor_valid > 0.5, as_tuple=False).flatten()
    if valid_slots.numel() == 0:
        return seed_map, distance_map

    sigma = max(2.0 * float(seed_radius) / float(canvas_size), 0.08)
    nearest_dist = torch.full((canvas_size, canvas_size), float("inf"), dtype=torch.float32)

    for slot in valid_slots.tolist():
        dy = float(neighbor_positions[slot, 0].item())
        dx = float(neighbor_positions[slot, 1].item())
        dist = torch.sqrt((coord_y - dy) ** 2 + (coord_x - dx) ** 2)
        gaussian = torch.exp(-(dist**2) / max(2.0 * sigma * sigma, 1e-6))
        seed_map[0] = torch.maximum(seed_map[0], gaussian)
        nearest_dist = torch.minimum(nearest_dist, dist)

    distance_map[0] = torch.exp(-(nearest_dist**2) / (2.0 * (0.35**2)))
    return seed_map, distance_map


def build_condition_vector(
    seed_nmf: torch.Tensor,
    nucleus_area: torch.Tensor,
    neighbor_seed_nmfs: torch.Tensor,
    neighbor_nucleus_areas: torch.Tensor,
    neighbor_positions: torch.Tensor,
    neighbor_valid: torch.Tensor,
) -> torch.Tensor:
    valid = neighbor_valid > 0.5
    if bool(valid.any()):
        weights = valid.float()
        weights = weights / weights.sum().clamp_min(1.0)
        neighbor_seed_summary = (neighbor_seed_nmfs * weights.unsqueeze(1)).sum(dim=0)
        neighbor_area_summary = (torch.log1p(neighbor_nucleus_areas[:, 0].clamp_min(0.0)) * weights).sum().view(1)
        valid_positions = neighbor_positions[valid]
        neighbor_min_dist = valid_positions[:, 2].min().view(1)
        neighbor_valid_frac = valid.float().mean().view(1)
    else:
        neighbor_seed_summary = torch.zeros_like(seed_nmf)
        neighbor_area_summary = torch.zeros((1,), dtype=torch.float32)
        neighbor_min_dist = torch.ones((1,), dtype=torch.float32)
        neighbor_valid_frac = torch.zeros((1,), dtype=torch.float32)

    nucleus_area_log = torch.log1p(nucleus_area.clamp_min(0.0)).view(1)
    return torch.cat(
        [
            seed_nmf.float(),
            nucleus_area_log.float(),
            neighbor_seed_summary.float(),
            neighbor_area_summary.float(),
            neighbor_min_dist.float(),
            neighbor_valid_frac.float(),
        ],
        dim=0,
    )


def crop_with_padding_and_resize(
    tensor: torch.Tensor,
    center_y: float,
    center_x: float,
    crop_size: int,
    output_size: int,
) -> torch.Tensor:
    crop = SpatialTranscriptomicsDataset._crop_with_padding(tensor, center_y, center_x, crop_size).float()
    if crop.shape[-2] == output_size and crop.shape[-1] == output_size:
        return crop
    crop = crop.unsqueeze(0)
    crop = F.interpolate(
        crop,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    return crop.squeeze(0)


def build_multiscale_expr_crops(
    tile_expr: torch.Tensor,
    center_y: float,
    center_x: float,
    crop_sizes: list[int],
    output_size: int,
) -> torch.Tensor:
    crops = [
        crop_with_padding_and_resize(tile_expr, center_y, center_x, crop_size, output_size)
        for crop_size in crop_sizes
    ]
    return torch.stack(crops, dim=0)

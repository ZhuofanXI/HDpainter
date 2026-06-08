from __future__ import annotations

import argparse
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GNN relational graph autoencoder on segmentation-derived "
            "pseudo-cell AnnData prepared by post_process.py."
        )
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--dim-reduction", choices=("PCA", "HVG", "all"), default="HVG")
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=(100, 32), metavar=("HIDDEN", "OUT"))
    parser.add_argument(
        "--graph-input-dim",
        type=int,
        default=64,
        help="Encode high-dimensional gene features to this dimension before RGAT. 0 disables this encoder.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--att-drop", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clipping", type=float, default=5.0)
    parser.add_argument("--device-idx", type=int, default=0)
    parser.add_argument("--center-msg", choices=("out", "in"), default="out")
    parser.add_argument("--batch-data", dest="batch_data", action="store_true", default=True)
    parser.add_argument("--no-batch-data", dest="batch_data", action="store_false")
    parser.add_argument("--num-batch-x", type=int, default=4)
    parser.add_argument("--num-batch-y", type=int, default=4)
    parser.add_argument("--batch-spatial-k", type=int, default=4)
    parser.add_argument("--batch-expression-k", type=int, default=3)
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=1,
        help="Print training progress every N graph batches. Use 0 to disable batch-level progress logs.",
    )
    parser.add_argument("--key-added", default="GNN")
    parser.add_argument("--save-reconstruction", dest="save_reconstruction", action="store_true", default=True)
    parser.add_argument("--no-save-reconstruction", dest="save_reconstruction", action="store_false")
    parser.add_argument("--run-leiden", action="store_true")
    parser.add_argument("--leiden-resolution", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=2024)
    parser.add_argument("--save-model", type=Path, default=None)
    return parser.parse_args()


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn.conv.rgat_conv import RGATConv
    except ImportError as exc:
        raise ImportError(
            "signal_process.py requires torch and torch_geometric. "
            "Run it in the server conda environment that provides GNN dependencies."
        ) from exc
    return torch, nn, F, Data, DataLoader, RGATConv


def _dense_float32(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def select_features(adata: ad.AnnData, dim_reduction: str) -> np.ndarray:
    if dim_reduction == "PCA":
        if "X_pca" not in adata.obsm:
            raise ValueError("dim_reduction='PCA' requires adata.obsm['X_pca']. Run post_process.py first.")
        return np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    if dim_reduction == "HVG":
        if "highly_variable" not in adata.var:
            raise ValueError("dim_reduction='HVG' requires adata.var['highly_variable'].")
        return _dense_float32(adata[:, adata.var["highly_variable"].to_numpy()].X)
    return _dense_float32(adata.X)


def _network_frame(adata: ad.AnnData, key: str) -> pd.DataFrame:
    if key not in adata.uns:
        raise ValueError(f"Missing adata.uns['{key}']. Run post_process.py to prepare graph inputs.")
    df = pd.DataFrame(adata.uns[key]).copy()
    required = {"Cell1", "Cell2"}
    if not required.issubset(df.columns):
        raise ValueError(f"adata.uns['{key}'] must contain columns Cell1 and Cell2.")
    if "Distance" not in df.columns:
        df["Distance"] = np.nan
    return df


def _edge_arrays(adata: ad.AnnData, net_key: str, center_msg: str) -> tuple[np.ndarray, np.ndarray]:
    df = _network_frame(adata, net_key)
    cell_to_idx = pd.Series(np.arange(adata.n_obs, dtype=np.int64), index=adata.obs_names)
    src = cell_to_idx.reindex(df["Cell1"].astype(str)).to_numpy()
    dst = cell_to_idx.reindex(df["Cell2"].astype(str)).to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64, copy=False)
    dst = dst[keep].astype(np.int64, copy=False)

    self_loop = np.arange(adata.n_obs, dtype=np.int64)
    src = np.concatenate([src, self_loop])
    dst = np.concatenate([dst, self_loop])
    if center_msg == "in":
        src, dst = dst, src
    return src, dst


def build_knn_net(features: np.ndarray, k: int, obs_names: pd.Index) -> pd.DataFrame:
    features = np.asarray(features, dtype=np.float32)
    n_obs = int(features.shape[0])
    if n_obs <= 1:
        return pd.DataFrame(columns=["Cell1", "Cell2", "Distance"])
    k_eff = min(int(k), n_obs - 1)
    if k_eff <= 0:
        return pd.DataFrame(columns=["Cell1", "Cell2", "Distance"])
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    rows = np.repeat(np.arange(n_obs, dtype=np.int64), k_eff)
    cols = indices.reshape(-1).astype(np.int64)
    return pd.DataFrame(
        {
            "Cell1": obs_names.to_numpy()[rows],
            "Cell2": obs_names.to_numpy()[cols],
            "Distance": distances.reshape(-1).astype(np.float32),
        }
    )


def rebuild_batch_networks(
    adata: ad.AnnData,
    dim_reduction: str,
    spatial_k: int,
    expression_k: int,
) -> None:
    if "spatial" not in adata.obsm:
        raise ValueError("Batch graph construction requires adata.obsm['spatial'].")
    adata.uns["Spatial_Net"] = build_knn_net(
        np.asarray(adata.obsm["spatial"], dtype=np.float32),
        k=int(spatial_k),
        obs_names=adata.obs_names,
    )
    adata.uns["Exp_Net"] = build_knn_net(
        select_features(adata, dim_reduction=dim_reduction),
        k=int(expression_k),
        obs_names=adata.obs_names,
    )


def transfer_graph_data(adata: ad.AnnData, dim_reduction: str, center_msg: str):
    torch, _, _, Data, _, _ = _require_torch()
    exp_src, exp_dst = _edge_arrays(adata, "Exp_Net", center_msg=center_msg)
    spa_src, spa_dst = _edge_arrays(adata, "Spatial_Net", center_msg=center_msg)
    edge_index = np.vstack(
        [
            np.concatenate([exp_src, spa_src]),
            np.concatenate([exp_dst, spa_dst]),
        ]
    )
    edge_type = np.zeros(edge_index.shape[1], dtype=np.int64)
    edge_type[exp_src.shape[0] :] = 1
    feat = select_features(adata, dim_reduction=dim_reduction)
    data = Data(
        x=torch.as_tensor(feat.copy(), dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long).contiguous(),
    )
    data.edge_type = torch.as_tensor(edge_type, dtype=torch.long)
    return data


def split_batches(adata: ad.AnnData, num_batch_x: int, num_batch_y: int) -> list[ad.AnnData]:
    if "spatial" not in adata.obsm:
        raise ValueError("Batch training requires adata.obsm['spatial'].")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    x_breaks = np.percentile(coords[:, 0], np.linspace(0, 100, num_batch_x + 1))
    y_breaks = np.percentile(coords[:, 1], np.linspace(0, 100, num_batch_y + 1))
    batches = []
    seen: set[str] = set()
    for ix in range(num_batch_x):
        for iy in range(num_batch_y):
            if ix == num_batch_x - 1:
                x_mask = (coords[:, 0] >= x_breaks[ix]) & (coords[:, 0] <= x_breaks[ix + 1])
            else:
                x_mask = (coords[:, 0] >= x_breaks[ix]) & (coords[:, 0] < x_breaks[ix + 1])
            if iy == num_batch_y - 1:
                y_mask = (coords[:, 1] >= y_breaks[iy]) & (coords[:, 1] <= y_breaks[iy + 1])
            else:
                y_mask = (coords[:, 1] >= y_breaks[iy]) & (coords[:, 1] < y_breaks[iy + 1])
            idx = np.flatnonzero(x_mask & y_mask)
            if idx.shape[0] > 0:
                batches.append(adata[idx].copy())
                seen.update(adata.obs_names[idx].astype(str))
    if not batches:
        raise ValueError("No valid spatial batches were created. Reduce num-batch-x/y.")
    if len(seen) != adata.n_obs:
        raise ValueError(f"Spatial batches cover {len(seen)} cells, expected {adata.n_obs}.")
    return batches


def infer_auto_batches(adata: ad.AnnData) -> tuple[int, int]:
    split = max(round(np.sqrt(adata.n_obs / 10000)), 1)
    return split, split


def make_model(in_dim: int, hidden_dims: tuple[int, int], att_drop: float, dim_reduction: str, graph_input_dim: int):
    _, nn, F, _, _, RGATConv = _require_torch()

    class RelationalGraphAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_dim, out_dim = hidden_dims
            self.input_dim = int(in_dim)
            self.graph_input_dim = int(graph_input_dim) if int(graph_input_dim) > 0 else int(in_dim)
            if self.graph_input_dim < self.input_dim:
                self.feature_encoder = nn.Sequential(
                    nn.Linear(self.input_dim, self.graph_input_dim),
                    nn.ReLU(),
                )
            else:
                self.feature_encoder = nn.Identity()
            self.conv1 = RGATConv(
                self.graph_input_dim,
                hidden_dim,
                num_relations=2,
                heads=1,
                concat=False,
                dropout=att_drop,
                bias=False,
            )
            self.conv2 = RGATConv(
                hidden_dim,
                out_dim,
                num_relations=2,
                heads=1,
                concat=False,
                dropout=att_drop,
                bias=False,
            )
            if dim_reduction == "PCA":
                self.decoder = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.Linear(hidden_dim, self.input_dim))
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(out_dim, hidden_dim),
                    nn.Linear(hidden_dim, self.input_dim),
                    nn.ReLU(),
                )

        def forward(self, features, edge_index, edge_type):
            graph_features = self.feature_encoder(features)
            h = F.elu(self.conv1(graph_features, edge_index, edge_type))
            z = F.elu(self.conv2(h, edge_index, edge_type))
            out = self.decoder(z)
            return z, out

    return RelationalGraphAutoencoder()


def train_signal_model(
    adata: ad.AnnData,
    dim_reduction: str,
    hidden_dims: tuple[int, int],
    epochs: int,
    lr: float,
    att_drop: float,
    weight_decay: float,
    gradient_clipping: float,
    device_idx: int,
    center_msg: str,
    batch_data: bool,
    num_batch_x: int,
    num_batch_y: int,
    batch_spatial_k: int,
    batch_expression_k: int,
    graph_input_dim: int,
    progress_every_batches: int,
    random_seed: int,
):
    torch, _, F, _, DataLoader, _ = _require_torch()
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    if dim_reduction == "PCA":
        in_dim = int(adata.obsm["X_pca"].shape[1])
    elif dim_reduction == "HVG":
        in_dim = int(np.asarray(adata.var["highly_variable"], dtype=bool).sum())
    else:
        in_dim = int(adata.n_vars)
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    model = make_model(
        in_dim=in_dim,
        hidden_dims=(int(hidden_dims[0]), int(hidden_dims[1])),
        att_drop=float(att_drop),
        dim_reduction=str(dim_reduction),
        graph_input_dim=int(graph_input_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    print(
        f"signal_model in_dim={in_dim} hidden_dims={tuple(map(int, hidden_dims))} "
        f"graph_input_dim={int(graph_input_dim)} device={device}",
        flush=True,
    )

    if batch_data:
        if num_batch_x <= 0 or num_batch_y <= 0:
            num_batch_x, num_batch_y = infer_auto_batches(adata)
        batches = split_batches(adata, num_batch_x=num_batch_x, num_batch_y=num_batch_y)
        print(
            f"DIC batches={len(batches)} grid={int(num_batch_x)}x{int(num_batch_y)} "
            f"cell_count_min={min(x.n_obs for x in batches)} "
            f"cell_count_max={max(x.n_obs for x in batches)}",
            flush=True,
        )
        for temp_adata in batches:
            rebuild_batch_networks(
                temp_adata,
                dim_reduction=dim_reduction,
                spatial_k=int(batch_spatial_k),
                expression_k=int(batch_expression_k),
            )
        train_data = [transfer_graph_data(x, dim_reduction=dim_reduction, center_msg=center_msg) for x in batches]
        loader = DataLoader(train_data, batch_size=1, shuffle=True)
    else:
        full_data = transfer_graph_data(adata, dim_reduction=dim_reduction, center_msg=center_msg)
        full_data = full_data.to(device)
        loader = None

    loss_history = []
    for epoch in range(int(epochs)):
        model.train()
        if loader is None:
            optimizer.zero_grad()
            _, out = model(full_data.x, full_data.edge_index, full_data.edge_type)
            loss = F.mse_loss(full_data.x, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clipping))
            optimizer.step()
            loss_history.append(float(loss.item()))
        else:
            epoch_losses = []
            total_batches = len(loader)
            for batch_idx, batch in enumerate(loader, start=1):
                batch = batch.to(device)
                optimizer.zero_grad()
                _, out = model(batch.x, batch.edge_index, batch.edge_type)
                loss = F.mse_loss(batch.x, out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clipping))
                optimizer.step()
                epoch_losses.append(float(loss.item()))
                if progress_every_batches > 0 and (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or batch_idx % int(progress_every_batches) == 0
                ):
                    print(
                        f"epoch={epoch + 1}/{epochs} batch={batch_idx}/{total_batches} "
                        f"cells={int(batch.x.shape[0])} loss={float(loss.item()):.6f}",
                        flush=True,
                    )
            loss_history.append(float(np.mean(epoch_losses)))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"epoch={epoch + 1}/{epochs} loss={loss_history[-1]:.6f}", flush=True)

    model.eval()
    if batch_data:
        embedding = np.zeros((adata.n_obs, int(hidden_dims[1])), dtype=np.float32)
        reconstruction = np.zeros((adata.n_obs, in_dim), dtype=np.float32)
        input_features = np.zeros((adata.n_obs, in_dim), dtype=np.float32)
        obs_to_idx = pd.Series(np.arange(adata.n_obs, dtype=np.int64), index=adata.obs_names.astype(str))
        with torch.no_grad():
            for temp_adata in batches:
                eval_data = transfer_graph_data(temp_adata, dim_reduction=dim_reduction, center_msg=center_msg).to(device)
                z, out = model(eval_data.x, eval_data.edge_index, eval_data.edge_type)
                idx = obs_to_idx.loc[temp_adata.obs_names.astype(str)].to_numpy(dtype=np.int64)
                embedding[idx] = z.detach().cpu().numpy().astype(np.float32)
                reconstruction[idx] = out.detach().cpu().numpy().astype(np.float32)
                input_features[idx] = eval_data.x.detach().cpu().numpy().astype(np.float32)
    else:
        with torch.no_grad():
            eval_data = transfer_graph_data(adata, dim_reduction=dim_reduction, center_msg=center_msg).to(device)
            z, out = model(eval_data.x, eval_data.edge_index, eval_data.edge_type)
            input_features = eval_data.x.detach().cpu().numpy()
        embedding = z.cpu().numpy().astype(np.float32)
        reconstruction = out.cpu().numpy().astype(np.float32)
    if dim_reduction != "PCA":
        reconstruction[input_features == 0] = 0.0
    return model, embedding, reconstruction, loss_history


def run_leiden(adata: ad.AnnData, use_rep: str, resolution: float) -> None:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Leiden clustering requires scanpy.") from exc
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.leiden(adata, random_state=2024, resolution=float(resolution), key_added=use_rep)


def main() -> None:
    args = parse_args()
    print("[1/4] loading prepared h5ad...")
    adata = ad.read_h5ad(args.input_h5ad)

    print("[2/4] training GNN signal model...")
    model, embedding, reconstruction, loss_history = train_signal_model(
        adata,
        dim_reduction=str(args.dim_reduction),
        hidden_dims=(int(args.hidden_dims[0]), int(args.hidden_dims[1])),
        epochs=int(args.epochs),
        lr=float(args.lr),
        att_drop=float(args.att_drop),
        weight_decay=float(args.weight_decay),
        gradient_clipping=float(args.gradient_clipping),
        device_idx=int(args.device_idx),
        center_msg=str(args.center_msg),
        batch_data=bool(args.batch_data),
        num_batch_x=int(args.num_batch_x),
        num_batch_y=int(args.num_batch_y),
        batch_spatial_k=int(args.batch_spatial_k),
        batch_expression_k=int(args.batch_expression_k),
        graph_input_dim=int(args.graph_input_dim),
        progress_every_batches=int(args.progress_every_batches),
        random_seed=int(args.random_seed),
    )
    adata.obsm[args.key_added] = embedding
    adata.uns[f"{args.key_added}_signal_process"] = {
        "source_h5ad": str(args.input_h5ad),
        "dim_reduction": str(args.dim_reduction),
        "hidden_dims": [int(args.hidden_dims[0]), int(args.hidden_dims[1])],
        "graph_input_dim": int(args.graph_input_dim),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "att_drop": float(args.att_drop),
        "weight_decay": float(args.weight_decay),
        "center_msg": str(args.center_msg),
        "batch_data": bool(args.batch_data),
        "loss_history": loss_history,
    }
    if args.save_reconstruction:
        if reconstruction.shape[1] == adata.n_vars:
            adata.layers[f"{args.key_added}_ReX"] = reconstruction
            adata.uns[f"{args.key_added}_signal_process"]["reconstruction_layer"] = f"{args.key_added}_ReX"
        else:
            adata.obsm[f"{args.key_added}_ReX"] = reconstruction
            adata.uns[f"{args.key_added}_signal_process"]["reconstruction_obsm"] = f"{args.key_added}_ReX"
            adata.uns[f"{args.key_added}_signal_process"]["reconstruction_note"] = (
                "Reconstruction is stored in obsm because it covers the selected feature subset, "
                "not all adata.var genes."
            )

    if args.run_leiden:
        print("[3/4] running Leiden on signal embedding...")
        run_leiden(adata, use_rep=args.key_added, resolution=float(args.leiden_resolution))
    else:
        print("[3/4] skipping Leiden.")

    if args.save_model is not None:
        torch, *_ = _require_torch()
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, args.save_model)

    print("[4/4] writing output...")
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_h5ad)
    print(f"output_h5ad={args.output_h5ad}")
    print(f"embedding_key={args.key_added} shape={adata.obsm[args.key_added].shape}")


if __name__ == "__main__":
    main()

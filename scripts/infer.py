"""
HDpainter LDM inference script.

Runs DDIM reverse diffusion (deterministic, fast) on a subset of SVD tiles,
computes latent-space metrics (RMSE, PCC) vs baseline (degraded input),
and saves spatial heatmap comparisons as PNG.

DDIM deterministic step (eta=0):
    x0_hat  = (x_t - sqrt(1-ᾱ_t) * eps_hat) / sqrt(ᾱ_t)
    x_{s}   = sqrt(ᾱ_s) * x0_hat + sqrt(1-ᾱ_s) * eps_hat
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import SpatialTranscriptomicsDataset
from src.models.ldm import ConditionalUNet


# ── Noise schedule ────────────────────────────────────────────────────────────

def make_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


# ── DDIM sampler ──────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model, z_cond, alpha_bars, device, ddim_steps=50, t_start=500):
    """
    Img2img DDIM reverse diffusion (SDEdit style, eta=0).

    Instead of starting from pure Gaussian noise, we add noise to z_cond at
    timestep t_start and denoise from there. This preserves the spatial structure
    of the degraded input as the starting point, letting the model refine rather
    than reconstruct from scratch.

    Args:
        model:      ConditionalUNet
        z_cond:     (1, C, H, W) degraded condition
        alpha_bars: (T,) precomputed ᾱ values
        ddim_steps: number of denoising steps
        t_start:    noise level to start from (1–T); lower = less change, higher = more freedom

    Returns:
        z0_hat: (1, C, H, W) predicted clean latent
    """
    T = len(alpha_bars)
    t_start = min(t_start, T)

    # Add noise to z_cond at t_start to get the starting z_t
    ab_start = alpha_bars[t_start - 1].to(device)
    eps_init = torch.randn_like(z_cond)
    z_t = ab_start.sqrt() * z_cond + (1 - ab_start).sqrt() * eps_init

    # Evenly spaced timestep indices from t_start down to 1 (0-indexed)
    step_indices = torch.linspace(t_start, 1, ddim_steps, dtype=torch.long) - 1
    step_indices = step_indices.clamp(0, T - 1)

    for i, t_idx in enumerate(step_indices):
        t_idx = t_idx.item()
        ab_t = alpha_bars[t_idx].to(device)
        t_tensor = torch.tensor([t_idx + 1], device=device, dtype=torch.long)

        eps_hat, _ = model(z_t, z_cond, t_tensor)

        # Estimate clean z0
        x0_hat = (z_t - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt().clamp(min=1e-8)
        x0_hat = x0_hat.clamp(-10, 10)

        # DDIM step to previous timestep
        if i + 1 < len(step_indices):
            t_prev_idx = step_indices[i + 1].item()
            ab_prev = alpha_bars[t_prev_idx].to(device)
        else:
            ab_prev = torch.tensor(1.0, device=device)  # final step: pure x0

        z_t = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps_hat

    return x0_hat


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, target: np.ndarray, cell_mask: np.ndarray):
    """
    Compute RMSE and mean PCC in the cell region.

    Args:
        pred, target, cell_mask: (C, H, W) numpy arrays, cell_mask boolean
    """
    # Flatten spatial dims, select cell pixels
    C = pred.shape[0]
    mask = cell_mask[0]  # (H, W) bool
    pred_flat   = pred[:, mask]    # (C, N_cells)
    target_flat = target[:, mask]  # (C, N_cells)

    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

    # Per-channel PCC averaged over channels
    pcc_vals = []
    for c in range(C):
        if pred_flat[c].std() > 1e-8 and target_flat[c].std() > 1e-8:
            r, _ = pearsonr(pred_flat[c], target_flat[c])
            pcc_vals.append(r)
    pcc = float(np.mean(pcc_vals)) if pcc_vals else 0.0

    return rmse, pcc


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_comparison(degraded, predicted, target, cell_mask, out_path):
    """Save side-by-side spatial heatmaps (sum over latent dims, in cell region)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping visualisation")
        return

    mask = cell_mask[0].astype(float)

    def to_map(x):
        m = x.sum(axis=0) * mask           # (H, W)
        lo, hi = np.percentile(m[mask > 0], [1, 99])
        return np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Degraded (input)", "Predicted (LDM)", "Ground Truth"]
    maps   = [to_map(degraded), to_map(predicted), to_map(target)]

    for ax, title, m in zip(axes, titles, maps):
        im = ax.imshow(m, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    T = train_args["T"]

    model = ConditionalUNet(
        latent_dim=64,
        base_ch=train_args["base_ch"],
        ch_mult=tuple(train_args["ch_mult"]),
        num_res_blocks=train_args["num_res_blocks"],
        T=T,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}")

    # ── Noise schedule ────────────────────────────────────────────────────────
    _, _, alpha_bars = make_linear_schedule(T)
    alpha_bars = alpha_bars.to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = SpatialTranscriptomicsDataset(args.data_dir)
    n_tiles = min(args.n_tiles, len(dataset))
    print(f"Evaluating {n_tiles} / {len(dataset)} tiles, DDIM steps={args.ddim_steps}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Metric accumulators ───────────────────────────────────────────────────
    results = {"pred_rmse": [], "pred_pcc": [], "base_rmse": [], "base_pcc": []}

    for i in range(n_tiles):
        tile = dataset[i]
        z_cond  = tile["input_expr"].unsqueeze(0).to(device)   # (1, 64, H, W)
        z0      = tile["target_expr"].unsqueeze(0).to(device)  # (1, 64, H, W)
        cell_id = tile["target_cell_id"].unsqueeze(0).to(device)

        print(f"[{i+1}/{n_tiles}] Sampling...")
        z0_hat = ddim_sample(model, z_cond, alpha_bars, device, args.ddim_steps, args.t_start)

        # Move to CPU numpy for metrics
        pred   = z0_hat.squeeze(0).cpu().numpy()   # (64, H, W)
        target = z0.squeeze(0).cpu().numpy()
        degrad = z_cond.squeeze(0).cpu().numpy()
        mask   = (cell_id.squeeze(0).cpu().numpy() != 0)  # (1, H, W) bool

        # Metrics: predicted vs target
        p_rmse, p_pcc = compute_metrics(pred,   target, mask)
        # Baseline: degraded vs target
        b_rmse, b_pcc = compute_metrics(degrad, target, mask)

        results["pred_rmse"].append(p_rmse)
        results["pred_pcc"].append(p_pcc)
        results["base_rmse"].append(b_rmse)
        results["base_pcc"].append(b_pcc)

        print(
            f"  Baseline  — RMSE: {b_rmse:.4f}  PCC: {b_pcc:.4f}\n"
            f"  Predicted — RMSE: {p_rmse:.4f}  PCC: {p_pcc:.4f}"
        )

        # Visualise first few tiles
        if i < args.n_vis:
            tile_name = os.path.basename(dataset.file_paths[i]).replace(".pt", "")
            save_comparison(
                degrad, pred, target, mask,
                out_path=os.path.join(args.out_dir, f"{tile_name}.png"),
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n========== Summary ==========")
    print(f"{'':20s}  {'RMSE':>8}  {'PCC':>8}")
    print(f"{'Baseline (degraded)':20s}  {np.mean(results['base_rmse']):8.4f}  {np.mean(results['base_pcc']):8.4f}")
    print(f"{'Predicted (LDM)':20s}  {np.mean(results['pred_rmse']):8.4f}  {np.mean(results['pred_pcc']):8.4f}")
    print("=============================")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HDpainter LDM inference")
    p.add_argument("--ckpt",        required=True,  help="Path to checkpoint .pt file")
    p.add_argument("--data_dir",    required=True,  help="Path to SVD tile directory")
    p.add_argument("--out_dir",     default="outputs/infer", help="Output directory for PNGs")
    p.add_argument("--n_tiles",     type=int, default=5,   help="Number of tiles to evaluate")
    p.add_argument("--n_vis",       type=int, default=3,   help="Number of tiles to save as PNG")
    p.add_argument("--ddim_steps",  type=int, default=50,  help="DDIM denoising steps")
    p.add_argument("--t_start",     type=int, default=500, help="Noise level to start from (img2img); lower=less change")
    return p.parse_args()


if __name__ == "__main__":
    infer(parse_args())

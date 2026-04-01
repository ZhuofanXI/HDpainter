"""
LDM training script for HDpainter.

Pipeline (zhuofan.md §4):
  z_0    = target_expr  (clean SVD features,    [B, 64, H, W])
  z_cond = input_expr   (degraded SVD features, [B, 64, H, W])
  Forward diffusion: z_t = sqrt(ᾱ_t) * z_0 + sqrt(1 - ᾱ_t) * eps,  eps ~ N(0, I)
  Model predicts eps_hat and boundary logits.

Loss (zhuofan.md §4):
  L_diff  = Huber(eps_hat, eps),           masked to cell regions
  L_bound = BCE(boundary, B_gt) + Dice,    masked to cell regions, weighted by λ₂(t)
  L_total = L_diff + λ₂(t) * L_bound

  λ₂(t) = sigmoid((t_thresh - t) * slope)   — 0 for high t, ~1 for low t
  t_thresh = 200  (boundary loss activates only in the last 200 steps)
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataloader
from src.models.ldm import ConditionalUNet


# ── Noise schedule ────────────────────────────────────────────────────────────

def make_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas = torch.linspace(beta_start, beta_end, T)          # (T,)
    alphas = 1.0 - betas                                      # (T,)
    alpha_bars = torch.cumprod(alphas, dim=0)                 # (T,)
    return betas, alphas, alpha_bars


# ── Loss helpers ──────────────────────────────────────────────────────────────

def cell_id_to_boundary(cell_id: torch.Tensor) -> torch.Tensor:
    """
    Derive inter-cell boundary mask from instance cell-ID map.
    A pixel is a boundary only where two *different* non-zero cells are adjacent
    (cell/background edges are NOT marked). This gives ~26% positive rate
    within cell regions, vs ~91% with the naive "any different neighbour" rule.

    Args:
        cell_id: (B, 1, H, W) int32 instance map
    Returns:
        (B, 1, H, W) float boundary mask, values in {0, 1}
    """
    cid = cell_id.float()
    pad = F.pad(cid, (1, 1, 1, 1), mode="constant", value=0)
    c = pad[:, :, 1:-1, 1:-1]

    def inter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a != b) & (a != 0) & (b != 0))

    boundary = (
        inter(c, pad[:, :, :-2, 1:-1])   # up
        | inter(c, pad[:, :, 2:, 1:-1])  # down
        | inter(c, pad[:, :, 1:-1, :-2]) # left
        | inter(c, pad[:, :, 1:-1, 2:])  # right
    ).float()
    return boundary


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def boundary_loss_fn(logits: torch.Tensor, target: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
    """
    BCE (with pos_weight for class imbalance) + Dice, computed only in cell region.
    Inter-cell boundary positive rate ≈ 26% within cell region → pos_weight ≈ 3.
    """
    # Flatten to cell pixels only
    mask = cell_mask[:, 0].bool()          # (B, H, W)
    logits_c = logits[:, 0][mask]          # (N_cell,)
    target_c = target[:, 0][mask]          # (N_cell,)
    pw = torch.tensor([3.0], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits_c, target_c, pos_weight=pw)

    # Dice on masked region
    pred_m = torch.sigmoid(logits) * cell_mask
    tgt_m  = target * cell_mask
    intersection = (pred_m * tgt_m).sum(dim=(-2, -1))
    union = pred_m.sum(dim=(-2, -1)) + tgt_m.sum(dim=(-2, -1))
    dice = (1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)).mean()

    return bce + dice


def lambda2(t: torch.Tensor, t_thresh: int = 200, slope: float = 0.05) -> torch.Tensor:
    """
    Sigmoid annealing weight for boundary loss (zhuofan.md §4.3).
    λ₂ ≈ 0 when t > t_thresh, λ₂ ≈ 1 when t << t_thresh.
    """
    return torch.sigmoid((t_thresh - t.float()) * slope)  # (B,)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    loader = build_dataloader(
        data_dirs=args.data_dir,   # accepts one or multiple dirs
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    n_genes = loader.dataset.n_genes  # should be 64 for SVD tiles
    print(f"Tiles: {len(loader.dataset)}, latent_dim={n_genes}")

    # ── Noise schedule ────────────────────────────────────────────────────────
    _, _, alpha_bars = make_linear_schedule(args.T)
    alpha_bars = alpha_bars.to(device)  # (T,)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalUNet(
        latent_dim=n_genes,
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        T=args.T,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest = os.path.join(args.ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = diff_loss_sum = bound_loss_sum = 0.0
        n_batches = 0

        for batch in loader:
            z0    = batch["target_expr"].to(device)     # (B, 64, H, W)
            z_cond = batch["input_expr"].to(device)     # (B, 64, H, W)
            cell_id = batch["target_cell_id"].to(device).int()  # (B, 1, H, W)

            B = z0.shape[0]

            # Sample timesteps and noise
            t = torch.randint(1, args.T + 1, (B,), device=device)  # (B,)
            eps = torch.randn_like(z0)

            ab_t = alpha_bars[t - 1]  # (B,)
            sqrt_ab  = ab_t.sqrt()[:, None, None, None]
            sqrt_1ab = (1.0 - ab_t).sqrt()[:, None, None, None]
            z_t = sqrt_ab * z0 + sqrt_1ab * eps

            with autocast():
                eps_hat, boundary_logits = model(z_t, z_cond, t)

                # Cell region mask
                cell_mask = (cell_id != 0).float()  # (B, 1, H, W)

                # ── Diffusion loss (Huber, cell-region only) ──────────────────
                noise_err = F.huber_loss(eps_hat, eps, reduction="none")  # (B, 64, H, W)
                mask_64 = cell_mask.expand_as(noise_err)
                n_valid = mask_64.sum().clamp(min=1)
                diff_loss = (noise_err * mask_64).sum() / n_valid

                # ── Boundary loss (BCE+Dice, cell-region only, time-gated) ────
                b_gt = cell_id_to_boundary(cell_id)         # (B, 1, H, W)
                b_loss = boundary_loss_fn(boundary_logits, b_gt, cell_mask)

                lam2 = lambda2(t, args.t_thresh).mean()     # scalar weight
                loss = diff_loss + lam2 * b_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item()
            diff_loss_sum += diff_loss.item()
            bound_loss_sum += b_loss.item()
            n_batches += 1

        avg = lambda s: s / max(n_batches, 1)
        print(
            f"Epoch {epoch:04d} | "
            f"loss={avg(total_loss):.4f}  "
            f"diff={avg(diff_loss_sum):.4f}  "
            f"bound={avg(bound_loss_sum):.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, latest)
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(args.ckpt_dir, f"epoch_{epoch:04d}.pt"))

    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HDpainter LDM")
    # Data
    p.add_argument("--data_dir",    required=True,  nargs="+", help="One or more SVD tile directories (rescaled to std≈1 before mixing)")
    p.add_argument("--ckpt_dir",    required=True,  help="Checkpoint output directory")
    # Training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--save_every",  type=int,   default=10,  help="Save named checkpoint every N epochs")
    # Diffusion
    p.add_argument("--T",           type=int,   default=1000, help="Total diffusion timesteps")
    p.add_argument("--t_thresh",    type=int,   default=200,  help="Boundary loss activation threshold")
    # Model
    p.add_argument("--base_ch",         type=int,   default=64)
    p.add_argument("--ch_mult",         type=int,   nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num_res_blocks",  type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

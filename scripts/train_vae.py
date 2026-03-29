"""
Train the VisiumVAE on spatial transcriptomics tiles.

Usage:
    uv run python scripts/train_vae.py \\
        --data_dir  ../data/NSCLC \\
        --ckpt_dir  ../checkpoints/vae_nsclc \\
        --epochs    100 \\
        --batch_size 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import SpatialTranscriptomicsDataset
from models.vae import VisiumVAE, vae_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VisiumVAE")
    p.add_argument("--data_dir",   type=str,   required=True,  help="Path to directory with .pt tiles")
    p.add_argument("--ckpt_dir",   type=str,   required=True,  help="Directory to save checkpoints")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int,   default=50)
    p.add_argument("--kl_weight",  type=float, default=1e-5)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--save_every", type=int,   default=10,     help="Save checkpoint every N epochs")
    p.add_argument("--resume",     type=str,   default=None,   help="Path to checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    dataset = SpatialTranscriptomicsDataset(args.data_dir)
    n_genes = dataset.n_genes
    print(f"Tiles: {len(dataset)}, n_genes: {n_genes}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # --- Model ---
    model = VisiumVAE(n_genes=n_genes, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()   # mixed-precision scaler

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = recon_sum = kl_sum = 0.0

        for batch in loader:
            target_expr = batch["target_expr"].to(device)
            mask = (batch["target_cell_id"] > 0).float().to(device)  # [B, 1, H, W]

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                recon, mu, logvar = model(target_expr)
                loss, recon_l, kl_l = vae_loss(
                    recon, target_expr, mu, logvar, mask, args.kl_weight
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            recon_sum  += recon_l.item()
            kl_sum     += kl_l.item()

        n = len(loader)
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"loss={total_loss/n:.4f} | "
            f"recon={recon_sum/n:.4f} | "
            f"kl={kl_sum/n:.6f}"
        )

        if epoch % args.save_every == 0:
            path = ckpt_dir / f"vae_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "n_genes": n_genes,
                    "latent_dim": args.latent_dim,
                },
                path,
            )
            print(f"  -> Saved {path}")

    # Final checkpoint
    final_path = ckpt_dir / "vae_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "n_genes": n_genes,
            "latent_dim": args.latent_dim,
        },
        final_path,
    )
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

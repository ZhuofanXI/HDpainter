from typing import Tuple

import torch
import torch.nn as nn


class VisiumVAE(nn.Module):
    """
    VAE for spatial transcriptomics gene expression.

    Uses exclusively 1x1 convolutions (= per-bin MLP) to preserve spatial
    independence: no information is mixed across neighbouring bins.

    Input/output shape: [B, n_genes, H, W]
    Latent shape:       [B, latent_dim, H, W]
    """

    def __init__(self, n_genes: int, latent_dim: int = 50):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(n_genes, 1024, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.SiLU(),
        )
        self.fc_mu = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.fc_logvar = nn.Conv2d(256, latent_dim, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(1024, n_genes, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    kl_weight: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masked VAE loss. Only penalises pixels where real cells exist.

    Args:
        recon:     [B, C, H, W]  reconstructed expression
        target:    [B, C, H, W]  ground-truth expression
        mu:        [B, latent_dim, H, W]
        logvar:    [B, latent_dim, H, W]
        mask:      [B, 1, H, W]  binary (1 = cell, 0 = background)
        kl_weight: weight for the KL term (small by default to avoid posterior collapse)

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    # --- Reconstruction loss (MSE on cell pixels only) ---
    mask_c = mask.expand_as(recon)                        # [B, C, H, W]
    n_valid = mask_c.sum().clamp(min=1)
    recon_loss = ((recon - target).pow(2) * mask_c).sum() / n_valid

    # --- KL divergence (on cell pixels only) ---
    mask_l = mask.expand_as(mu)                           # [B, latent_dim, H, W]
    n_kl = mask_l.sum().clamp(min=1)
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) * mask_l).sum() / n_kl

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss

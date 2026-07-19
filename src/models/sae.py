"""Sparse autoencoder scaffold for phase-1 plumbing."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class SAEConfig:
    input_dim: int
    expansion: int = 8
    lambda_l1: float = 3.0e-4


if nn is not None:

    class SparseAutoencoder(nn.Module):
        def __init__(self, cfg: SAEConfig) -> None:
            super().__init__()
            self.cfg = cfg
            hidden = cfg.input_dim * cfg.expansion
            self.encoder = nn.Linear(cfg.input_dim, hidden)
            self.decoder = nn.Linear(hidden, cfg.input_dim)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            z = torch.relu(self.encoder(x))
            recon = self.decoder(z)
            return recon, z

        def loss(self, x: "torch.Tensor") -> "torch.Tensor":
            recon, z = self.forward(x)
            recon_loss = ((recon - x) ** 2).mean()
            l1 = z.abs().mean()
            return recon_loss + self.cfg.lambda_l1 * l1

else:

    class SparseAutoencoder:  # type: ignore[no-redef]
        def __init__(self, cfg: SAEConfig) -> None:
            self.cfg = cfg

        def forward(self, x):
            return x, x

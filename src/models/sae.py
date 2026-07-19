"""Sparse autoencoder for post-hoc interpretability of MGN node embeddings.

Follows arXiv:2507.16069 (Hu & Liu, IJCAI 2025 XAI workshop):
  - Trained on the FROZEN MGN node embeddings h_i (pre-decoder latent).
  - encoder: Linear -> ReLU
  - decoder: Linear, with rows (dictionary atoms) constrained to unit L2 norm
    after initialization AND after every optimizer step.
  - loss: ||h_hat - h||^2 + lambda * ||z||_1
This matches the paper's "sparse feature dictionary" semantics, where each
decoder row w_k is a unit-norm atom and activating z_k shifts the embedding
along w_k.
"""

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
            self.decoder = nn.Linear(hidden, cfg.input_dim, bias=False)
            # Dictionary atoms start at unit norm (paper: re-norm after init).
            self.normalize_decoder()

        def normalize_decoder(self) -> None:
            """Constrain decoder rows to unit L2 norm (dictionary atoms)."""
            with torch.no_grad():
                w = self.decoder.weight  # [input_dim, hidden]
                norms = w.norm(dim=0, keepdim=True).clamp_min(1e-8)
                self.decoder.weight.copy_(w / norms)

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(self.encoder(x))

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(z)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            z = self.encode(x)
            recon = self.decode(z)
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

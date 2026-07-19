"""MeshGraphNet model scaffold for phase-1 plumbing."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class MGNConfig:
    hidden_dim: int = 128
    message_passing_steps: int = 9


if nn is not None:

    class MeshGraphNet(nn.Module):
        def __init__(self, cfg: MGNConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.placeholder = nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.placeholder(x)

else:

    class MeshGraphNet:  # type: ignore[no-redef]
        def __init__(self, cfg: MGNConfig) -> None:
            self.cfg = cfg

        def forward(self, x):
            return x

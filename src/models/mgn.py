"""MeshGraphNet model for cylinder-flow next-frame prediction.

Architecture (canonical MeshGraphNets style):
  encode node/edge features -> message passing blocks -> decode velocity + pressure.

Key design point: node count ``N`` varies per example and per split
(train=1876, valid=1896, test=1923). All aggregation uses index-based
scatter (``torch.scatter_add``), never fixed matrix multiplies, so a single
model handles any graph size.

The model consumes a :class:`~src.data.cylinder_flow.GraphSample`-shaped
inputs: ``node_features [N, F]``, ``edge_index [2, E]``, ``edge_attr [E, D]``.
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
class MGNConfig:
    hidden_dim: int = 128
    edge_dim: int = 3
    node_in_dim: int = 8
    message_passing_steps: int = 9
    mlp_hidden: int = 128
    mlp_layers: int = 2


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> "nn.Module":
    assert nn is not None
    if layers <= 1:
        return nn.Linear(in_dim, out_dim)
    parts: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(layers - 2):
        parts += [nn.Linear(hidden, hidden), nn.ReLU()]
    parts.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*parts)


if nn is not None:

    class MeshGraphNet(nn.Module):
        def __init__(self, cfg: MGNConfig) -> None:
            super().__init__()
            self.cfg = cfg
            h = cfg.hidden_dim

            self.node_encoder = nn.Linear(cfg.node_in_dim, h)
            self.edge_encoder = nn.Linear(cfg.edge_dim, h)

            self.edge_mlp = _mlp(h * 3, h, cfg.mlp_hidden, cfg.mlp_layers)
            self.node_mlp = _mlp(h * 2, h, cfg.mlp_hidden, cfg.mlp_layers)

            self.velocity_head = nn.Linear(h, 2)
            self.pressure_head = nn.Linear(h, 1)

        def _scatter_sum(self, src: "torch.Tensor", idx: "torch.Tensor", n: int) -> "torch.Tensor":
            # src: [E, h], idx: [E] destination node -> [N, h]
            return torch.scatter_add(
                torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype),
                0,
                idx.unsqueeze(1).expand(-1, src.shape[1]),
                src,
            )

        def forward(
            self,
            node_features: "torch.Tensor",
            edge_index: "torch.Tensor",
            edge_attr: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            assert torch is not None
            if node_features.dim() != 2 or node_features.shape[1] != self.cfg.node_in_dim:
                raise ValueError(
                    f"node_features must be [N, {self.cfg.node_in_dim}], "
                    f"got {tuple(node_features.shape)}"
                )
            if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                raise ValueError(
                    f"edge_index must be [2, E], got {tuple(edge_index.shape)}"
                )
            if edge_attr.dim() != 2 or edge_attr.shape[1] != self.cfg.edge_dim:
                raise ValueError(
                    f"edge_attr must be [E, {self.cfg.edge_dim}], "
                    f"got {tuple(edge_attr.shape)}"
                )

            n_nodes = node_features.shape[0]
            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            h_node = self.node_encoder(node_features)        # [N, h]
            h_edge = self.edge_encoder(edge_attr)           # [E, h]

            for _ in range(self.cfg.message_passing_steps):
                # Messages from source node features to each edge.
                src_h = h_node[src_idx]                      # [E, h]
                edge_input = torch.cat([src_h, h_edge, h_node[dst_idx]], dim=-1)  # [E, 3h]
                msg = self.edge_mlp(edge_input)             # [E, h]
                aggregated = self._scatter_sum(msg, dst_idx, n_nodes)  # [N, h]
                node_input = torch.cat([h_node, aggregated], dim=-1)   # [N, 2h]
                updated = self.node_mlp(node_input)          # [N, h]
                h_node = h_node + updated                    # residual

            vel = self.velocity_head(h_node)                # [N, 2]
            pres = self.pressure_head(h_node).squeeze(-1)   # [N]
            return vel, pres

        def loss(
            self,
            pred_vel: "torch.Tensor",
            pred_pres: "torch.Tensor",
            target_vel: "torch.Tensor",
            target_pres: "torch.Tensor",
        ) -> "torch.Tensor":
            vel_loss = ((pred_vel - target_vel) ** 2).mean()
            pres_loss = ((pred_pres - target_pres) ** 2).mean()
            return vel_loss + pres_loss

else:

    class MeshGraphNet:  # type: ignore[no-redef]
        def __init__(self, cfg: MGNConfig) -> None:
            self.cfg = cfg

        def forward(self, node_features, edge_index, edge_attr):
            n = node_features.shape[0]
            return node_features.new_zeros(n, 2), node_features.new_zeros(n)

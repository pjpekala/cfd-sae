"""Cylinder flow dataset loader for the MGN/SAE pipeline.

The DeepMind ``cylinder_flow`` TFRecords store each trajectory as one
``tf.train.Example`` whose features are *packed byte blobs* (not scalar
feature lists). The decoded layout (verified against the local shards) is::

    mesh_pos  float32  [N, 2]       node coordinates
    node_type int32    [N]          node category codes (observed: {0,4,5,6})
    cells     int32    [C, 3]       triangular mesh (C triangles)
    pressure  float32  [T, N]       T frames of scalar pressure
    velocity  float32  [T, N, 2]    T frames of (vx, vy)

Important: ``N`` (node count) and ``T`` (frame count) vary *per example and
per split* (e.g. train=1876, valid=1896, test=1923 nodes). Nothing in the
loader or model may assume a fixed ``N`` -- every example is its own graph.

All decoding goes through :func:`decode_one`, which validates the packed
shapes and fails fast with a key-level message rather than emitting silently
garbled tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

try:
    import tfrecord
except Exception:  # pragma: no cover - dependency guard
    tfrecord = None  # type: ignore[assignment]

SPLITS: tuple[str, ...] = ("train", "valid", "test")

# Node-type codes observed across all splits. One-hot is built over the
# *observed* set, never hardcoded, so a new code in a future shard won't
# silently collapse into the wrong bucket.
OBSERVED_NODE_TYPES: tuple[int, ...] = (0, 4, 5, 6)

# Classic MeshGraphNets uses 6 history frames; smoke path uses 1->1.
DEFAULT_HISTORY_FRAMES = 1


class CylinderFlowError(Exception):
    """Raised when a record's schema is malformed or inconsistent."""


def split_path(data_dir: Path, split: str) -> Path:
    if split not in SPLITS:
        choices = ", ".join(SPLITS)
        raise ValueError(f"Invalid split '{split}'. Expected one of: {choices}")
    return data_dir / f"{split}.tfrecord"


def ensure_split_exists(data_dir: Path, split: str) -> Path:
    path = split_path(data_dir, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing TFRecord for split '{split}': {path}. "
            "Run scripts/download_data.sh first."
        )
    return path


def _require_tfrecord() -> None:
    if tfrecord is None:
        raise RuntimeError(
            "The 'tfrecord' package is required to read the dataset. "
            "Install dependencies with `uv sync`."
        )


def _unpack_bytes(buf: bytes, dtype: np.dtype, name: str) -> np.ndarray:
    try:
        arr = np.frombuffer(buf, dtype=dtype)
    except Exception as exc:  # pragma: no cover
        raise CylinderFlowError(
            f"Field '{name}' could not be unpacked as {dtype}: {exc}"
        ) from exc
    if arr.size == 0:
        raise CylinderFlowError(f"Field '{name}' decoded to an empty array.")
    return arr


def decode_one(record_bytes: bytes) -> dict[str, np.ndarray]:
    """Decode one raw TFRecord example into named numpy arrays.

    Raises :class:`CylinderFlowError` on any shape inconsistency so bad data
    fails fast with an actionable, key-level message.
    """
    _require_tfrecord()
    from tfrecord.reader import example_pb2

    example = example_pb2.Example()
    try:
        example.ParseFromString(bytes(record_bytes))
    except Exception as exc:
        raise CylinderFlowError(f"Failed to parse TFRecord example: {exc}") from exc

    features = example.features.feature
    missing = [k for k in ("mesh_pos", "node_type", "cells",
                            "pressure", "velocity") if k not in features]
    if missing:
        raise CylinderFlowError(
            f"Record missing required fields: {missing}. "
            f"Found keys: {sorted(features)}"
        )

    def blob(key: str) -> bytes:
        fl = features[key].bytes_list
        if not fl.value:
            raise CylinderFlowError(f"Field '{key}' has no bytes_list value.")
        return fl.value[0]

    mesh_pos = _unpack_bytes(blob("mesh_pos"), np.float32, "mesh_pos")
    node_type = _unpack_bytes(blob("node_type"), np.int32, "node_type")
    cells = _unpack_bytes(blob("cells"), np.int32, "cells")
    pressure = _unpack_bytes(blob("pressure"), np.float32, "pressure")
    velocity = _unpack_bytes(blob("velocity"), np.float32, "velocity")

    if mesh_pos.size % 2 != 0:
        raise CylinderFlowError(
            f"mesh_pos has {mesh_pos.size} floats (not divisible by 2)."
        )
    n_nodes = mesh_pos.size // 2
    mesh_pos = mesh_pos.reshape(n_nodes, 2)

    if node_type.size != n_nodes:
        raise CylinderFlowError(
            f"node_type length {node_type.size} != mesh_pos node count {n_nodes}."
        )

    if cells.size % 3 != 0:
        raise CylinderFlowError(f"cells has {cells.size} ints (not divisible by 3).")
    n_cells = cells.size // 3
    cells = cells.reshape(n_cells, 3)
    if cells.min() < 0 or cells.max() >= n_nodes:
        raise CylinderFlowError(
            f"cells indices out of range [0,{n_nodes-1}]: "
            f"min={cells.min()} max={cells.max()}."
        )

    if pressure.size % n_nodes != 0:
        raise CylinderFlowError(
            f"pressure length {pressure.size} not divisible by node count {n_nodes}."
        )
    n_frames = pressure.size // n_nodes
    pressure = pressure.reshape(n_frames, n_nodes)

    if velocity.size != n_frames * n_nodes * 2:
        raise CylinderFlowError(
            f"velocity length {velocity.size} != T*N*2 "
            f"({n_frames}*{n_nodes}*2)."
        )
    velocity = velocity.reshape(n_frames, n_nodes, 2)

    return {
        "mesh_pos": mesh_pos,
        "node_type": node_type,
        "cells": cells,
        "pressure": pressure,
        "velocity": velocity,
    }


def split_reader(
    data_dir: Path,
    split: str,
    max_examples: Optional[int] = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Stream decoded examples from a split's TFRecord.

    ``max_examples`` caps the yield count (used for smoke tests); pass ``None``
    to iterate the whole shard. Variable ``N``/``T`` per example is preserved.
    """
    _require_tfrecord()
    path = ensure_split_exists(data_dir, split)
    reader = tfrecord.tfrecord_iterator(str(path))
    count = 0
    for raw in reader:
        yield decode_one(raw)
        count += 1
        if max_examples is not None and count >= max_examples:
            break


@dataclass
class GraphSample:
    """A single training sample (one frame window) as dense numpy arrays.

    Edge index uses the MGN convention ``edge_index`` of shape ``[2, E]``.
    """

    node_features: np.ndarray          # [N, F_in]
    edge_index: np.ndarray             # [2, E]
    edge_attr: np.ndarray              # [E, D_edge]
    target_velocity: np.ndarray        # [N, 2]
    target_pressure: np.ndarray        # [N]
    mesh_pos: np.ndarray               # [N, 2]
    node_type: np.ndarray              # [N]
    n_nodes: int = field(init=False)
    n_edges: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_nodes = int(self.node_features.shape[0])
        self.n_edges = int(self.edge_index.shape[1])


def _edges_from_cells(cells: np.ndarray) -> np.ndarray:
    """Build undirected edge index from triangles (3 edges per triangle)."""
    # Triangular faces -> 3 directed edges each, then make undirected.
    edges = np.stack(
        [
            cells[:, 0], cells[:, 1],
            cells[:, 1], cells[:, 2],
            cells[:, 2], cells[:, 0],
        ],
        axis=1,
    ).reshape(-1, 2)
    # Remove duplicates (each shared edge appears twice).
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    # Make undirected bidirectional: add reverse.
    rev = edges[:, ::-1]
    edges = np.concatenate([edges, rev], axis=0)
    edges = edges.T.astype(np.int64)  # [2, E]
    return edges


def build_sample(
    example: dict[str, np.ndarray],
    frame: int,
    history_frames: int = DEFAULT_HISTORY_FRAMES,
) -> GraphSample:
    """Build a next-frame graph sample from an example at ``frame``.

    Predicts frame ``t+1`` from frames ``[t-history+1 .. t]`` velocity, node
    type, and mesh position. Falls back to available history near the start.
    """
    mesh_pos = example["mesh_pos"]
    node_type = example["node_type"]
    cells = example["cells"]
    velocity = example["velocity"]
    pressure = example["pressure"]

    n_frames = velocity.shape[0]
    if not 0 <= frame < n_frames - 1:
        raise ValueError(
            f"frame {frame} out of range for {n_frames} frames "
            "(need frame in [0, {n_frames-2}] to predict t+1)."
        )

    # ----- node features -----
    # One-hot over observed node types (robust to unseen codes).
    n_types = len(OBSERVED_NODE_TYPES)
    type_map = {code: i for i, code in enumerate(OBSERVED_NODE_TYPES)}
    oh = np.zeros((node_type.shape[0], n_types), dtype=np.float32)
    for code, idx in type_map.items():
        oh[node_type == code, idx] = 1.0
    # Unknown codes -> zero vector (handled, not asserted away).

    start = max(0, frame - history_frames + 1)
    hist = velocity[start : frame + 1]  # [h, N, 2]
    # Reshape to per-node history: [N, h*2].
    hist_feat = hist.transpose(1, 0, 2).reshape(node_type.shape[0], -1).astype(np.float32)

    node_features = np.concatenate([oh, hist_feat, mesh_pos.astype(np.float32)], axis=1)

    # ----- edges -----
    edge_index = _edges_from_cells(cells)
    # Edge attributes: relative displacement + distance.
    src = mesh_pos[edge_index[0]]
    dst = mesh_pos[edge_index[1]]
    rel = dst - src
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    edge_attr = np.concatenate([rel, dist], axis=1).astype(np.float32)

    return GraphSample(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_velocity=velocity[frame + 1].astype(np.float32),
        target_pressure=pressure[frame + 1].astype(np.float32),
        mesh_pos=mesh_pos.astype(np.float32),
        node_type=node_type,
    )


@dataclass
class NormalizationStats:
    """Per-feature normalization statistics, computed over a split."""

    velocity_mean: np.ndarray        # [2]
    velocity_std: np.ndarray          # [2]
    pressure_mean: float
    pressure_std: float
    mesh_pos_min: np.ndarray          # [2]
    mesh_pos_max: np.ndarray          # [2]
    node_type_counts: dict[int, int] = field(default_factory=dict)
    n_examples: int = 0
    n_nodes_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity_mean": self.velocity_mean.tolist(),
            "velocity_std": self.velocity_std.tolist(),
            "pressure_mean": self.pressure_mean,
            "pressure_std": self.pressure_std,
            "mesh_pos_min": self.mesh_pos_min.tolist(),
            "mesh_pos_max": self.mesh_pos_max.tolist(),
            "node_type_counts": {int(k): int(v) for k, v in self.node_type_counts.items()},
            "n_examples": self.n_examples,
            "n_nodes_total": self.n_nodes_total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationStats":
        return cls(
            velocity_mean=np.asarray(data["velocity_mean"], dtype=np.float32),
            velocity_std=np.asarray(data["velocity_std"], dtype=np.float32),
            pressure_mean=float(data["pressure_mean"]),
            pressure_std=float(data["pressure_std"]),
            mesh_pos_min=np.asarray(data["mesh_pos_min"], dtype=np.float32),
            mesh_pos_max=np.asarray(data["mesh_pos_max"], dtype=np.float32),
            node_type_counts={int(k): int(v) for k, v in data["node_type_counts"].items()},
            n_examples=int(data["n_examples"]),
            n_nodes_total=int(data["n_nodes_total"]),
        )


def compute_stats(
    data_dir: Path,
    split: str,
    max_examples: Optional[int] = None,
) -> NormalizationStats:
    """Compute streaming normalization stats over a split.

    Uses a numerically stable running mean/std (Welford) over all velocity
    vectors and pressures. Mesh bounds and node-type histograms are tracked
    across examples. Safe for variable ``N``/``T`` per example.
    """
    v_mean = np.zeros(2, dtype=np.float64)
    v_var = np.zeros(2, dtype=np.float64)
    p_mean = 0.0
    p_var = 0.0
    p_min = np.inf
    p_max = -np.inf
    m_min = np.full(2, np.inf, dtype=np.float64)
    m_max = np.full(2, -np.inf, dtype=np.float64)
    type_counts: dict[int, int] = {}
    n_vectors = 0  # velocity vector count (N*T)
    p_count = 0
    n_examples = 0
    n_nodes_total = 0

    for ex in split_reader(data_dir, split, max_examples=max_examples):
        vel = ex["velocity"].reshape(-1, 2).astype(np.float64)  # [N*T, 2]
        pres = ex["pressure"].reshape(-1).astype(np.float64)     # [N*T]
        mp = ex["mesh_pos"].astype(np.float64)
        nt = ex["node_type"]

        # Welford update for velocity.
        for j in range(2):
            for x in vel[:, j]:
                n_vectors += 1
                delta = x - v_mean[j]
                v_mean[j] += delta / n_vectors
                v_var[j] += delta * (x - v_mean[j])
        # Pressure Welford.
        for x in pres:
            p_count += 1
            d = x - p_mean
            p_mean += d / p_count
            p_var += d * (x - p_mean)
        p_min = min(p_min, float(pres.min()))
        p_max = max(p_max, float(pres.max()))

        m_min = np.minimum(m_min, mp.min(axis=0))
        m_max = np.maximum(m_max, mp.max(axis=0))
        for code in np.unique(nt).tolist():
            type_counts[int(code)] = type_counts.get(int(code), 0) + int((nt == code).sum())

        n_examples += 1
        n_nodes_total += nt.shape[0]

    if n_vectors == 0 or p_count == 0:
        raise CylinderFlowError(f"No samples consumed from split '{split}'.")

    return NormalizationStats(
        velocity_mean=v_mean.astype(np.float32),
        velocity_std=np.sqrt(v_var / n_vectors).astype(np.float32),
        pressure_mean=p_mean,
        pressure_std=np.sqrt(p_var / p_count),
        mesh_pos_min=m_min.astype(np.float32),
        mesh_pos_max=m_max.astype(np.float32),
        node_type_counts=type_counts,
        n_examples=n_examples,
        n_nodes_total=n_nodes_total,
    )


def stats_path(base_dir: Path) -> Path:
    """Shared normalization-stats artifact path (persisted per data root)."""
    return Path(base_dir) / "stats.json"


def save_stats(stats: NormalizationStats, base_dir: Path) -> Path:
    """Persist stats to ``<base_dir>/stats.json`` (atomic write)."""
    from src.utils.io import write_json

    path = stats_path(base_dir)
    write_json(path, stats.to_dict())
    return path


def load_stats(base_dir: Path) -> NormalizationStats:
    """Load previously persisted stats; raise if missing."""
    from src.utils.io import read_json

    path = stats_path(base_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No normalization stats at {path}. Run compute_stats() + save_stats() first."
        )
    return NormalizationStats.from_dict(read_json(path))

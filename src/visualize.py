"""Shared visualization helpers for SAE interpretability analysis.

Every function returns a ``matplotlib.figure.Figure`` so it renders inline in a
notebook (just let the last expression be the figure) and can also be saved to
disk by CLI tools. Plotting logic lives here ONCE; both ``notebooks/`` and any
future CLI reuse it (no duplicated matplotlib code).

Typical flow (see ``notebooks/05_analysis.ipynb``):
    codes = load_codes(...)                 # [M, hidden] from analyze
    scores = salient_scores(codes, bins)     # paper Table-1
    top = np.argsort(-scores['variance'])[:K]
    fig = top_latents_bar(scores, top)
    fig = latent_histogram(codes, latent_idx)
    fig = spatial_scatter(positions, activations)
"""

from __future__ import annotations

from typing import Mapping, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")  # safe default; notebooks switch to inline backend
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:  # pragma: no cover
    plt = None
    np = None


def _fig() -> "plt.Figure":
    assert plt is not None
    return plt.figure(figsize=(7, 4))


def top_latents_bar(
    scores: Mapping[str, "np.ndarray"],
    top_indices: Sequence[int],
    title: str = "Top-K salient latents (Table 1 scores)",
) -> "plt.Figure":
    """Grouped bar chart of the three Table-1 saliency scores for top-K latents."""
    assert plt is not None and np is not None
    score_names = ["variance", "mean_abs", "entropy"]
    present = [s for s in score_names if s in scores]
    k = len(top_indices)
    x = np.arange(k)
    width = 0.8 / max(len(present), 1)

    fig = _fig()
    ax = fig.gca()
    for j, name in enumerate(present):
        vals = [float(scores[name][i]) for i in top_indices]
        ax.bar(x + j * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels([f"L{i}" for i in top_indices], rotation=90, fontsize=7)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def latent_histogram(
    codes: "np.ndarray",
    latent_idx: int,
    bins: int = 50,
    title: str | None = None,
) -> "plt.Figure":
    """Histogram of a single latent's activations across all samples."""
    assert plt is not None and np is not None
    vals = codes[:, latent_idx]
    fig = _fig()
    ax = fig.gca()
    ax.hist(vals, bins=bins)
    ax.set_xlabel(f"activation z_{latent_idx}")
    ax.set_ylabel("count")
    ax.set_title(title or f"Activation distribution: latent {latent_idx}")
    ax.axvline(0.0, color="k", lw=0.8, ls="--")
    fig.tight_layout()
    return fig


def spatial_scatter(
    positions: "np.ndarray",
    values: "np.ndarray",
    title: str | None = None,
    vmax: float | None = None,
) -> "plt.Figure":
    """Scatter node positions colored by a per-node value (e.g. latent activation)."""
    assert plt is not None and np is not None
    fig = _fig()
    ax = fig.gca()
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=values,
        cmap="viridis",
        s=8,
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Spatial activation")
    fig.colorbar(sc, ax=ax, label="activation")
    fig.tight_layout()
    return fig


def recon_error_bar(
    per_feature_mse: "np.ndarray",
    title: str = "Reconstruction MSE per embedding dimension",
) -> "plt.Figure":
    """Per-feature reconstruction error (diagnostic for which dims the SAE misses)."""
    assert plt is not None and np is not None
    fig = _fig()
    ax = fig.gca()
    ax.bar(np.arange(len(per_feature_mse)), per_feature_mse, width=1.0)
    ax.set_xlabel("embedding dim")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    fig.tight_layout()
    return fig

"""Figures for the synthetic claim. Saved under artifacts/."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import Score, align_to, reconstruct_modulator, valid_times
from .process import Trial


def plot_modulator_recovery(
    trial: Trial,
    estimates: dict[str, np.ndarray],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    Tm1 = trial.J.shape[0]
    m = trial.m[:Tm1]
    t = np.arange(Tm1)
    n_panel = 1 + len(estimates)
    fig, axes = plt.subplots(n_panel, 1, figsize=(10, 2.2 * n_panel), sharex=True)
    if n_panel == 1:
        axes = [axes]
    axes[0].plot(t, m, color="0.2", lw=1.2)
    axes[0].set_ylabel("true m (hidden)")
    title = trial.meta.get("scenario", trial.mechanism)
    axes[0].set_title(f"{title}  ρ={trial.rho}")
    for ax, (name, J_hat) in zip(axes[1:], estimates.items()):
        mask = valid_times(J_hat)
        m_hat = align_to(reconstruct_modulator(J_hat, mask), m)
        ax.plot(t[mask], m[mask], color="0.75", lw=1.0, label="true m")
        ax.plot(t[mask], _z(m_hat[mask]), color="C0", lw=1.2, label=name)
        ax.set_ylabel("recovered")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_lag_propagation(score: Score, trial: Trial, path: Path) -> None:
    import matplotlib.pyplot as plt

    corr = np.abs(score.lag_edge_corr)
    max_lag, n, _ = corr.shape
    fig, axes = plt.subplots(1, max_lag, figsize=(3.2 * max_lag, 3.2), squeeze=False)
    vmax = max(0.3, float(np.nanmax(corr)))
    for k in range(max_lag):
        ax = axes[0, k]
        im = ax.imshow(corr[k], vmin=0, vmax=vmax, cmap="magma", origin="upper")
        ax.set_title(f"|corr| of A^{k+1} with m")
        ax.set_xlabel("source")
        if k == 0:
            ax.set_ylabel("target")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle(f"{score.name} on {trial.mechanism}: delayed influence tracking hidden m")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _z(v: np.ndarray) -> np.ndarray:
    s = np.std(v)
    if s < 1e-12:
        return np.zeros_like(v)
    return (v - np.mean(v)) / s

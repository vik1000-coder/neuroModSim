"""Animate one trial: static wired graph, modulator-gated edges changing in time.

The movie shows exactly what the testbed simulates:
  - blue wired synapses: constant (anatomy does not change),
  - orange extrasynaptic edges: width/sign follow the hidden gate α·tanh(m_t)·0.5,
  - node color: the neuron's current activity x_t,
  - bottom panels: hidden m(t) and the observed activity, with a time cursor.

Usage:
    python -m neuromodsim.movie                      # writes artifacts/tour/modulation.gif
    python -m neuromodsim.movie --t0 1000 --t1 1720 --step 3 --fps 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .process import Trial
from .scenarios import make_scenario

WIRED = "#2c5aa0"
EXTRA = "#d35400"
NODE_EDGE = "#1b1b1b"
LW_PER_WEIGHT = 9.0


def render_movie(trial: Trial, path: Path, t0: int = 1000, t1: int = 1720,
                 step: int = 3, fps: int = 20, dpi: int = 85) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import animation, cm, colors
    from matplotlib.patches import FancyArrowPatch

    n = trial.n_neurons
    extras = list(trial.meta.get("extra_edges") or ())
    frames = list(range(t0, min(t1, trial.n_times - 1), step))

    # z-score activity for node colors (per neuron, over the shown window)
    xw = trial.x[t0:t1]
    z = (trial.x - xw.mean(0)) / (xw.std(0) + 1e-9)
    node_norm = colors.Normalize(vmin=-2.5, vmax=2.5)
    node_cmap = cm.coolwarm

    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.9, 0.75, 1.0], hspace=0.42)
    ax_net = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])
    ax_x = fig.add_subplot(gs[2], sharex=ax_m)

    # ---- network panel (drawn once; only widths/styles/colors update) ----
    xs = np.arange(n, dtype=float)
    shrink = 15.0
    for i in range(n - 1):                      # static wired chain
        w = float(trial.A0[i + 1, i])
        ax_net.add_patch(FancyArrowPatch(
            (xs[i], 0.0), (xs[i + 1], 0.0), arrowstyle="-|>", mutation_scale=14,
            lw=w * LW_PER_WEIGHT * 0.7, color=WIRED, alpha=0.85,
            shrinkA=shrink, shrinkB=shrink, zorder=1))

    arc_patches: list[tuple[FancyArrowPatch, int, int]] = []
    arc_labels = []
    max_apex = 0.5
    for k, (s, t) in enumerate(sorted(extras, key=lambda e: abs(e[1] - e[0]))):
        rad = 0.32 + 0.15 * k
        apex = rad * abs(xs[t] - xs[s]) / 2.0
        max_apex = max(max_apex, apex)
        patch = FancyArrowPatch(
            (xs[s], 0.0), (xs[t], 0.0), arrowstyle="-|>", mutation_scale=14,
            lw=0.5, color=EXTRA, connectionstyle=f"arc3,rad={-rad}",
            shrinkA=shrink, shrinkB=shrink, zorder=2)
        ax_net.add_patch(patch)
        arc_patches.append((patch, s, t))
        arc_labels.append(ax_net.text(0.5 * (xs[s] + xs[t]), apex + 0.13, "",
                                      ha="center", va="bottom", fontsize=10, color=EXTRA))

    nodes = ax_net.scatter(xs, np.zeros(n), s=850, facecolor="white",
                           edgecolor=NODE_EDGE, linewidth=1.5, zorder=3)
    for i in range(n):
        ax_net.text(xs[i], 0.0, str(i), ha="center", va="center", fontsize=12, zorder=4)
    readout = ax_net.text(0.01, 0.97, "", transform=ax_net.transAxes,
                          fontsize=12, va="top", family="monospace")
    ax_net.text(0.99, 0.97, "blue = wired synapses (static)\norange = extrasynaptic edges (gated by hidden m)\nnode color = current activity",
                transform=ax_net.transAxes, fontsize=9, va="top", ha="right", color="#555555")
    ax_net.set_xlim(-0.7, n - 0.3)
    ax_net.set_ylim(-0.9, max_apex + 0.8)
    ax_net.axis("off")

    # ---- hidden m panel ----
    tt = np.arange(t0, t1)
    ax_m.plot(tt, trial.m[t0:t1], color="#1b1b1b", lw=1.1)
    ax_m.axhline(0, color="0.85", lw=0.7)
    ax_m.set_ylabel("hidden m(t)")
    ax_m.set_xlim(t0, t1 - 1)
    cursor_m = ax_m.axvline(t0, color=EXTRA, lw=1.6)
    ax_m.set_title("Hidden modulator (never observed by methods)", loc="left", fontsize=10)

    # ---- observed activity panel ----
    ax_x.imshow(z[t0:t1].T, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5,
                origin="upper", extent=[t0, t1, n - 0.5, -0.5])
    ax_x.set_yticks(range(n))
    ax_x.set_ylabel("neuron")
    ax_x.set_xlabel("time")
    cursor_x = ax_x.axvline(t0, color="#1b1b1b", lw=1.6)
    ax_x.set_title("Observed activity x(t) — the only thing a method sees", loc="left", fontsize=10)

    def update(t: int):
        gate = trial.alpha * float(np.tanh(trial.m[t]))
        for patch, s, tgt in arc_patches:
            w = gate * float(trial.A1[tgt, s])
            patch.set_linewidth(max(abs(w) * LW_PER_WEIGHT, 0.3))
            patch.set_alpha(float(min(1.0, 0.12 + 2.2 * abs(w))))
            patch.set_linestyle("-" if w >= 0 else (0, (4, 2)))
        for lbl, (patch, s, tgt) in zip(arc_labels, arc_patches):
            w = gate * float(trial.A1[tgt, s])
            lbl.set_text(f"{s}→{tgt}: {w:+.2f}")
            lbl.set_alpha(float(min(1.0, 0.25 + 2.2 * abs(w))))
        nodes.set_facecolor(node_cmap(node_norm(z[t])))
        readout.set_text(f"t = {t}    m = {trial.m[t]:+.2f}    skip gate = {gate * 0.5:+.2f}")
        cursor_m.set_xdata([t, t])
        cursor_x.set_xdata([t, t])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("artifacts/tour/modulation.gif"))
    p.add_argument("--scenario", default="additive")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=2500)
    p.add_argument("--t0", type=int, default=1000)
    p.add_argument("--t1", type=int, default=1720)
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)
    trial = make_scenario(args.scenario, seed=args.seed, T=args.T)
    out = render_movie(trial, args.out, t0=args.t0, t1=args.t1,
                       step=args.step, fps=args.fps)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

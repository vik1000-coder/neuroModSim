"""Visual tour of one simulated trial: anatomy, modulation, traces, effects.

Everything drawn here is read from a real `Trial` (or its exact operators).
Edge widths use one global scale (LW_PER_WEIGHT) so panels are comparable:
a thicker arrow always means a stronger true weight, across all figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .process import Trial, mix_operator, simulate
from .scenarios import make_scenario

WIRED = "#2c5aa0"      # anatomical / wired edges
EXTRA = "#d35400"      # extrasynaptic, modulator-gated edges
NODE_FILL = "#f7f4ee"
NODE_EDGE = "#1b1b1b"
LW_PER_WEIGHT = 7.0    # global: line width in points per unit |weight|
NODE_SIZE = 780        # scatter marker area (pts^2); keeps circles round


# ---------------------------------------------------------------------------
# network drawing (arc diagram: chain on a line, skips as arcs above)
# ---------------------------------------------------------------------------

def _draw_chain_network(ax, A: np.ndarray, extra_mask: np.ndarray | None,
                        label_weights: bool = True, weight_note: str = "",
                        node_size: float = NODE_SIZE) -> None:
    """Arc diagram of a chain-topology operator A[target, source].

    Wired (non-extra) edges are straight arrows on the baseline; edges in
    extra_mask are arcs above the line. Solid = positive, dashed = negative.
    """
    from matplotlib.patches import FancyArrowPatch

    n = A.shape[0]
    xs = np.arange(n, dtype=float)
    if extra_mask is None:
        extra_mask = np.zeros_like(A, dtype=bool)

    edges = [(s, t) for t in range(n) for s in range(n) if abs(A[t, s]) > 1e-3]
    arc_edges = [(s, t) for (s, t) in edges if extra_mask[t, s]]
    arc_rad = {}
    for k, (s, t) in enumerate(sorted(arc_edges, key=lambda e: abs(e[1] - e[0]))):
        arc_rad[(s, t)] = 0.30 + 0.14 * k          # stagger arcs so they don't overlap

    shrink = float(np.sqrt(node_size) / 2.0 + 2.0)
    max_apex = 0.4
    for (s, t) in edges:
        w = float(A[t, s])
        is_extra = bool(extra_mask[t, s])
        color = EXTRA if is_extra else WIRED
        rad = arc_rad.get((s, t), 0.0)
        ax.add_patch(FancyArrowPatch(
            (xs[s], 0.0), (xs[t], 0.0),
            arrowstyle="-|>", mutation_scale=13,
            lw=max(abs(w) * LW_PER_WEIGHT, 0.8), color=color,
            alpha=min(1.0, 0.30 + 0.85 * abs(w)),
            connectionstyle=f"arc3,rad={-rad}",     # negative rad = arc above the baseline
            linestyle="-" if w > 0 else (0, (4, 2)),
            shrinkA=shrink, shrinkB=shrink, zorder=1,
        ))
        if is_extra:
            dist = abs(xs[t] - xs[s])
            apex = rad * dist / 2.0
            max_apex = max(max_apex, apex)
            if label_weights:
                ax.text(0.5 * (xs[s] + xs[t]), apex + 0.14, f"{s}→{t}: {w:+.2f}",
                        ha="center", va="bottom", fontsize=10, color=color)

    ax.scatter(xs, np.zeros(n), s=node_size, facecolor=NODE_FILL,
               edgecolor=NODE_EDGE, linewidth=1.4, zorder=3)
    node_font = 12 if node_size >= 600 else 10
    for i in range(n):
        ax.text(xs[i], 0.0, str(i), ha="center", va="center", fontsize=node_font, zorder=4)
    if weight_note:
        ax.text(0.5 * (n - 1), -0.62, weight_note, ha="center", va="top",
                fontsize=10, color=WIRED)

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-1.05, max_apex + 0.75)
    ax.axis("off")


def _network_legend(fig, include_extra: bool = True, y: float = 0.0) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=WIRED, lw=3.2, label="wired synapse (A₀)")]
    if include_extra:
        handles += [
            Line2D([0], [0], color=EXTRA, lw=3.2, label="extrasynaptic edge (A₁), excitatory"),
            Line2D([0], [0], color=EXTRA, lw=2.6, linestyle=(0, (4, 2)), label="extrasynaptic, sign flipped (inhibitory)"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, y))


def _matrix_panel(ax, M: np.ndarray, title: str, vmax: float, annotate: bool = True):
    im = ax.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="upper")
    n = M.shape[0]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xlabel("source neuron"); ax.set_ylabel("target neuron")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=0.7)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for t in range(n):
            for s in range(n):
                v = M[t, s]
                if abs(v) > 0.01:
                    ax.text(s, t, f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "−."),
                            ha="center", va="center", fontsize=6.5,
                            color="white" if abs(v) > 0.55 * vmax else "#222222")
    return im


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_anatomy(trial: Trial, path: Path) -> None:
    """The two anatomical layers, drawn at the maximum gate the modulator can reach."""
    import matplotlib.pyplot as plt

    gate = trial.alpha * 1.0  # tanh(m) -> 1 at large m
    A_full = trial.A0 + gate * trial.A1
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.4))
    _draw_chain_network(axes[0], trial.A0, None)
    axes[0].set_title("Wired circuit A₀ — 8 neurons, one-hop synaptic chain (weight 0.70 each)",
                      loc="left", fontsize=13)
    _draw_chain_network(axes[1], A_full, np.abs(trial.A1) > 1e-9)
    axes[1].set_title(
        f"Same circuit with the extrasynaptic layer A₁ fully gated on "
        f"(weight α·tanh(m)·0.5 → {gate * 0.5:+.2f} as m → +∞)",
        loc="left", fontsize=13)
    _network_legend(fig, include_extra=True, y=-0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_modulated_graph(trial: Trial, path: Path,
                         m_values: tuple[float, ...] = (-1.5, 0.0, 1.5)) -> None:
    """Graph + matrix at low / zero / high modulator, on one global width scale."""
    import matplotlib.pyplot as plt

    labels = {-1.5: "LOW modulator", 0.0: "modulator OFF", 1.5: "HIGH modulator"}
    fig, axes = plt.subplots(2, len(m_values), figsize=(16.0, 8.6),
                             gridspec_kw={"height_ratios": [1.0, 1.45]})
    extra_mask = (np.abs(trial.A1) > 1e-9) if trial.mechanism == "additive" else None
    for i, mv in enumerate(m_values):
        A = mix_operator(trial.A0, trial.A1, trial.mechanism, trial.alpha, mv)
        u = float(np.tanh(mv))
        note = f"every wired edge: {A[1, 0]:+.2f}" if trial.mechanism == "gain" else ""
        _draw_chain_network(axes[0, i], A, extra_mask, weight_note=note, node_size=380)
        axes[0, i].set_title(f"{labels.get(mv, f'm={mv:g}')}   (m = {mv:g})",
                             loc="center", fontsize=13, fontweight="bold")
        if trial.mechanism == "additive":
            sub = f"J(m) = A₀ + {trial.alpha:g}·tanh(m)·A₁,  tanh(m) = {u:+.2f}"
        elif trial.mechanism == "gain":
            sub = f"J(m) = (1 + {trial.alpha:g}·tanh(m))·A₀ — every edge × {1 + trial.alpha * u:.2f}"
        else:
            sub = "J(m) = A₀ (modulator disconnected)"
        _matrix_panel(axes[1, i], A, sub, vmax=0.9)
    fig.suptitle(f"{trial.mechanism.capitalize()} neuromodulation: same anatomy, different effective graph J(m)",
                 fontsize=15, y=0.99)
    _network_legend(fig, include_extra=trial.mechanism == "additive", y=-0.015)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_random_network(path: Path, seed: int = 3) -> None:
    """The testbed is not chain-only: a recurrent random graph with a gated layer."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    trial = simulate(n=10, T=400, mechanism="additive", topology="random",
                     density=0.22, radius=0.65, alpha=0.6, seed=seed)
    n = trial.n_neurons
    theta = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = np.column_stack([np.cos(theta), np.sin(theta)])

    def draw(ax, A, extra_mask, title):
        for s in range(n):
            for t in range(n):
                w = float(A[t, s])
                if abs(w) < 0.02 or s == t:
                    continue
                is_extra = bool(extra_mask[t, s]) if extra_mask is not None else False
                ax.add_patch(FancyArrowPatch(
                    pos[s], pos[t], arrowstyle="-|>", mutation_scale=13,
                    lw=max(abs(w) * LW_PER_WEIGHT, 0.7),
                    color=EXTRA if is_extra else WIRED,
                    alpha=0.9 if is_extra else 0.75,
                    connectionstyle="arc3,rad=0.12",
                    linestyle="-" if w > 0 else (0, (4, 2)),
                    shrinkA=15, shrinkB=15, zorder=1))
        ax.scatter(pos[:, 0], pos[:, 1], s=NODE_SIZE, facecolor=NODE_FILL,
                   edgecolor=NODE_EDGE, linewidth=1.4, zorder=3)
        for i in range(n):
            ax.text(pos[i, 0], pos[i, 1], str(i), ha="center", va="center",
                    fontsize=11, zorder=4)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=12)

    extra_mask = np.abs(trial.A1) > 1e-9
    n_wired = int((np.abs(trial.A0) > 1e-9).sum())
    n_extra = int(extra_mask.sum())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    draw(axes[0], trial.A0, None,
         f"Wired recurrent circuit A₀ ({n_wired} signed edges)")
    draw(axes[1], trial.A_of(1.5), extra_mask,
         f"HIGH m: J = A₀ + gated A₁ ({n_extra} extrasynaptic edges on)")
    draw(axes[2], trial.A_of(-1.5), extra_mask,
         "LOW m: same extrasynaptic edges, sign flipped")
    fig.suptitle("Random recurrent topology (n=10): the same instrument on a denser graph",
                 fontsize=14, y=1.0)
    _network_legend(fig, include_extra=True, y=-0.03)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _rolling_lag1_corr(x: np.ndarray, src: int, tgt: int, window: int) -> np.ndarray:
    """corr(x[t, src], x[t+1, tgt]) in a trailing window, at each t."""
    T = x.shape[0]
    out = np.full(T - 1, np.nan)
    a_all, b_all = x[:-1, src], x[1:, tgt]
    for t in range(window, T - 1):
        a = a_all[t - window:t]
        b = b_all[t - window:t]
        sa, sb = a.std(), b.std()
        if sa > 1e-12 and sb > 1e-12:
            out[t] = np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)
    return out


def plot_traces(trial: Trial, path: Path, t0: int = 1000, t1: int = 1700,
                window: int = 100) -> None:
    """Observed traces + the observable signature: rolling coupling tracks hidden m."""
    import matplotlib.pyplot as plt

    t = np.arange(trial.n_times)
    n = trial.n_neurons
    sl = slice(t0, t1)
    extras = trial.meta.get("extra_edges") or ()
    skip_targets = {tgt for _, tgt in extras}

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11.0),
                             gridspec_kw={"height_ratios": [0.8, 1.7, 1.0, 1.1]})

    # (1) hidden modulator, full run
    axes[0].plot(t, trial.m, color="#1b1b1b", lw=1.1)
    axes[0].axhline(0, color="0.8", lw=0.7)
    axes[0].axvspan(t0, t1, color=EXTRA, alpha=0.13, lw=0)
    axes[0].set_ylabel("hidden m(t)")
    axes[0].set_xlim(0, trial.n_times - 1)
    axes[0].set_title("(1) The hidden modulator — never shown to any method. Shaded band = zoom below.",
                      loc="left", fontsize=11)

    # (2) observed traces, zoomed, stacked
    for i in range(n):
        xi = trial.x[sl, i]
        z = (xi - xi.mean()) / (xi.std() + 1e-9)
        color = EXTRA if i in skip_targets else WIRED
        axes[1].plot(t[sl], 0.42 * z + (n - 1 - i), color=color, lw=0.9)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels([f"n{n - 1 - k}" for k in range(n)])
    for lbl in axes[1].get_yticklabels():
        idx = n - 1 - int(lbl.get_text()[1:])
        lbl.set_color(EXTRA if (n - 1 - idx) in skip_targets else WIRED)
    axes[1].set_xlim(t0, t1 - 1)
    axes[1].set_ylabel("observed activity x(t)")
    axes[1].set_title("(2) What a method sees: 8 noisy traces. Orange = neurons receiving extrasynaptic input (4, 6, 7).",
                      loc="left", fontsize=11)

    # (3) true gate on the extrasynaptic edges (all three share it)
    src0, tgt0 = extras[0]
    axes[2].plot(t[:-1][sl], trial.J[sl, tgt0, src0], color=EXTRA, lw=1.6,
                 label=f"true weight of every skip  ({', '.join(f'{s}→{g}' for s, g in extras)})")
    axes[2].axhline(0, color="0.8", lw=0.7)
    axes[2].plot(t[sl], trial.A0[1, 0] * np.ones(t1 - t0), color=WIRED, lw=1.6,
                 label="true weight of every wired edge (constant 0.70)")
    axes[2].set_xlim(t0, t1 - 1)
    axes[2].set_ylabel("true edge weight")
    axes[2].legend(loc="center right", fontsize=9, frameon=False)
    axes[2].set_title("(3) Ground truth (hidden from methods): the modulator gates only the orange edges.",
                      loc="left", fontsize=11)

    # (4) observable signature: rolling lag-1 correlation
    src, tgt = extras[0]
    r_skip = _rolling_lag1_corr(trial.x, src, tgt, window)
    r_wire = _rolling_lag1_corr(trial.x, 0, 1, window)
    axes[3].plot(t[:-1][sl], r_skip[sl], color=EXTRA, lw=1.5,
                 label=f"rolling corr(x{src}[t], x{tgt}[t+1]) — skip edge")
    axes[3].plot(t[:-1][sl], r_wire[sl], color=WIRED, lw=1.5,
                 label="rolling corr(x0[t], x1[t+1]) — wired edge")
    axes[3].plot(t[sl], 0.45 * np.tanh(trial.m[sl]), color="#1b1b1b", lw=1.2,
                 ls=(0, (4, 2)), label="∝ tanh(m)  (hidden)")
    axes[3].axhline(0, color="0.8", lw=0.7)
    axes[3].set_xlim(t0, t1 - 1)
    axes[3].set_ylim(-0.75, 0.95)
    axes[3].set_ylabel(f"lag-1 corr ({window}-step window)")
    axes[3].set_xlabel("time")
    axes[3].legend(loc="lower right", fontsize=9, frameon=False)
    axes[3].set_title("(4) The recoverable signature: the skip's local coupling tracks hidden m; the wired edge does not.",
                      loc="left", fontsize=11)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _frozen_impulse(A: np.ndarray, source: int, horizon: int) -> np.ndarray:
    n = A.shape[0]
    x = np.zeros((horizon + 1, n))
    x[0, source] = 1.0
    for k in range(horizon):
        x[k + 1] = A @ x[k]
    return x


def plot_impulse_effect(trial: Trial, path: Path, source: int = 0, horizon: int = 8) -> None:
    import matplotlib.pyplot as plt

    cases = {
        "LOW m  (skips inhibitory)": trial.A_of(-1.5),
        "m = 0  (skips off: pure chain)": trial.A_of(0.0),
        "HIGH m  (skips excitatory)": trial.A_of(1.5),
    }
    extras = trial.meta.get("extra_edges") or ()
    skip_targets = sorted({tgt for _, tgt in extras})

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    vmax = 1.0
    for ax, (name, A) in zip(axes, cases.items()):
        X = _frozen_impulse(A, source, horizon)
        im = ax.imshow(X.T, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                       origin="upper", aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("time steps after kick")
        ax.set_xticks(range(horizon + 1))
        ax.set_yticks(range(trial.n_neurons))
    axes[0].set_yticklabels([str(i) for i in range(trial.n_neurons)])
    for tgt in skip_targets:
        axes[0].get_yticklabels()[tgt].set_color(EXTRA)
        axes[0].get_yticklabels()[tgt].set_fontweight("bold")
    axes[0].set_ylabel("neuron")
    axes[1].annotate("chain: neuron k lights at lag k", xy=(4.0, 4.0), xytext=(4.6, 1.0),
                     fontsize=9, color=WIRED,
                     arrowprops=dict(arrowstyle="->", color=WIRED, lw=1.0))
    axes[2].annotate("skip 0→4 fires at lag 1\n(then 4→5→6→7 early)", xy=(1.0, 4.0),
                     xytext=(2.6, 6.6), fontsize=9, color=EXTRA,
                     arrowprops=dict(arrowstyle="->", color=EXTRA, lw=1.0))
    axes[0].annotate("same skip, negative:\nneuron 4 suppressed", xy=(1.0, 4.0),
                     xytext=(2.6, 6.6), fontsize=9, color=EXTRA,
                     arrowprops=dict(arrowstyle="->", color=EXTRA, lw=1.0))
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="activity")
    fig.suptitle("Unit kick at neuron 0 with the modulator frozen: "
                 "the modulator reroutes how the kick propagates through time",
                 fontsize=13, y=1.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_modulator_pulse(trial: Trial, path: Path, horizon: int = 8) -> None:
    import matplotlib.pyplot as plt

    energy = np.linalg.norm(trial.x, axis=1)
    t0 = int(np.argmax(energy[200:-horizon - 2]) + 200)
    dx = trial.pulse_response(t0=t0, pulse=1.5, horizon=horizon, rng_seed=1)
    extras = trial.meta.get("extra_edges") or ()
    skip_targets = sorted({tgt for _, tgt in extras})
    affected = set()
    for tgt in skip_targets:
        affected.update(range(tgt, trial.n_neurons))  # skip target + chain downstream

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})
    lim = np.max(np.abs(dx))
    im = axes[0].imshow(dx.T, cmap="coolwarm", vmin=-lim, vmax=lim,
                        origin="upper", aspect="auto",
                        extent=[0.5, horizon + 0.5, trial.n_neurons - 0.5, -0.5])
    axes[0].set_xlabel("time steps after the hidden m pulse")
    axes[0].set_ylabel("neuron")
    axes[0].set_yticks(range(trial.n_neurons))
    axes[0].set_yticklabels([str(i) for i in range(trial.n_neurons)])
    for tgt in skip_targets:
        axes[0].get_yticklabels()[tgt].set_color(EXTRA)
        axes[0].get_yticklabels()[tgt].set_fontweight("bold")
    axes[0].set_title(f"Δ activity caused by pulsing hidden m at t={t0}", fontsize=11)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Δx (treated − control)")

    lags = np.arange(1, horizon + 1)
    for i in range(trial.n_neurons):
        if i in affected:
            axes[1].plot(lags, dx[:, i], marker="o", ms=4, lw=1.4,
                         color=EXTRA if i in skip_targets else "#a06a3d",
                         label=f"n{i}" + ("  (skip target)" if i in skip_targets else ""))
        else:
            axes[1].plot(lags, dx[:, i], color="0.75", lw=1.0)
    axes[1].axhline(0, color="0.8", lw=0.8)
    axes[1].set_xlabel("lag")
    axes[1].set_ylabel("Δx")
    axes[1].set_title("Upstream neurons 0–3 (gray) are untouched:\nthe effect enters only through the gated edges",
                      fontsize=11)
    axes[1].legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_tour(outdir: Path, seed: int = 0, T: int = 2500) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    additive = make_scenario("additive", seed=seed, T=T)
    gain = make_scenario("gain", seed=seed, T=T)
    jobs = [
        ("01_anatomy.png", lambda p: plot_anatomy(additive, p)),
        ("02_additive_graph.png", lambda p: plot_modulated_graph(additive, p)),
        ("03_gain_graph.png", lambda p: plot_modulated_graph(gain, p)),
        ("04_random_network.png", lambda p: plot_random_network(p)),
        ("05_traces.png", lambda p: plot_traces(additive, p)),
        ("06_impulse_effect.png", lambda p: plot_impulse_effect(additive, p)),
        ("07_modulator_pulse.png", lambda p: plot_modulator_pulse(additive, p)),
    ]
    paths = []
    for name, fn in jobs:
        p = outdir / name
        fn(p)
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("artifacts/tour"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=2500)
    args = p.parse_args(argv)
    for path in write_tour(args.outdir, seed=args.seed, T=args.T):
        print(f"  {path}")


if __name__ == "__main__":
    main()

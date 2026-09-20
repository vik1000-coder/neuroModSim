"""Visualization suite for multi-modulator trials.

Four views, all drawn from a real MultiTrial (no cartoon data):

  1. plot_static_graph      the wired circuit A0 + where modulators dock
                            (sources, receptor neurons) — anatomy only.
  2. plot_snapshot          the effective graph J(t) at chosen times, with
                            each modulator's concentration profile above it.
  3. plot_spacetime         the time-unrolled graph: rows = neurons,
                            columns = time steps; node (i, t) is neuron i at
                            time t; an edge (i,t) -> (j,t+1) is J[t][j,i].
  4. render_multi_movie     animation: diffusion fields moving, gated edges
                            breathing, activity flowing over the graph.

Layout assumption: neurons are drawn in the order of their position along
the first axis (the demo uses a line; any 1D-ish arrangement works).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .multi import MultiTrial, _kernels, _softplus, simulate_multi
from .presets import PRESETS, build_circuit

WIRED = "#2c5aa0"
MOD_COLORS = ["#d35400", "#7d3c98", "#1e8449", "#b7950b"]  # per-modulator
NODE_EDGE = "#1b1b1b"
LW_PER_WEIGHT = 6.0


# ---------------------------------------------------------------------------
# shared drawing helpers
# ---------------------------------------------------------------------------

def _mod_color(c: int) -> str:
    return MOD_COLORS[c % len(MOD_COLORS)]


def _arc_rad(span: float) -> float:
    """arc3 rad giving an apex height ≈ 0.35 + 0.11·span (visible at any span)."""
    span = max(span, 1.0)
    apex = 0.35 + 0.11 * span
    return 2.0 * apex / span


def _additive_supports(trial: MultiTrial) -> list[np.ndarray]:
    """Per modulator: boolean (N,N) support of its receptor layer (empty for gain)."""
    out = []
    for mod in trial.modulators:
        if mod.mode == "additive":
            out.append(np.abs(mod.W) > 1e-9)
        else:
            out.append(np.zeros((trial.n_neurons, trial.n_neurons), dtype=bool))
    return out


def _draw_arc_graph(ax, trial: MultiTrial, A: np.ndarray, occ_t: np.ndarray | None,
                    node_color, gain_rings: list[tuple[int, np.ndarray]] | None = None,
                    node_size: float = 240.0) -> None:
    """Arc diagram of operator A: wired arcs below the line, modulator arcs above.

    occ_t (k, N) is only used to scale modulator-edge alpha; pass None for anatomy.
    gain_rings: [(modulator index, per-neuron gain factor)]; each gain modulator
    gets its own concentric ring whose thickness is its current gain.
    """
    from matplotlib.patches import FancyArrowPatch

    n = trial.n_neurons
    order = np.argsort(trial.positions[:, 0])
    col = {node: i for i, node in enumerate(order)}          # node -> x slot
    xs = np.arange(n, dtype=float)
    supports = _additive_supports(trial)
    wired_support = np.abs(trial.A0) > 1e-9
    shrink = float(np.sqrt(node_size) / 2.0 + 1.5)

    for s in range(n):
        for t in range(n):
            w = float(A[t, s])
            if abs(w) < 0.02 or s == t:
                continue
            owner = next((c for c, sup in enumerate(supports) if sup[t, s]), None)
            if owner is None and not wired_support[t, s]:
                continue
            x0, x1 = xs[col[s]], xs[col[t]]
            span = max(abs(x1 - x0), 1.0)
            if owner is None:                                 # wired: arc below
                color, rad = WIRED, _arc_rad(span)
                alpha = min(1.0, 0.25 + 1.1 * abs(w))
            else:                                             # modulated: arc above
                color, rad = _mod_color(owner), -_arc_rad(span)
                alpha = min(1.0, 0.10 + 2.2 * abs(w))
            ax.add_patch(FancyArrowPatch(
                (x0, 0.0), (x1, 0.0), arrowstyle="-|>", mutation_scale=9,
                lw=max(abs(w) * LW_PER_WEIGHT, 0.4), color=color, alpha=alpha,
                connectionstyle=f"arc3,rad={rad}",
                linestyle="-" if w > 0 else (0, (3, 2)),
                shrinkA=shrink, shrinkB=shrink, zorder=1))

    ax.scatter(xs, np.zeros(n), s=node_size, c=node_color, cmap="coolwarm",
               vmin=-2.5, vmax=2.5, edgecolor=NODE_EDGE, linewidth=1.0, zorder=3)
    for j, (c_idx, factor) in enumerate(gain_rings or []):
        ring = np.clip((factor[order] - 1.0) * 6.0, 0.0, 4.0)
        ax.scatter(xs, np.zeros(n), s=node_size * (2.1 + 1.1 * j), facecolor="none",
                   edgecolor=_mod_color(c_idx), zorder=2, linewidths=ring)
    for i, node in enumerate(order):
        ax.text(xs[i], 0.0, str(node), ha="center", va="center", fontsize=6, zorder=4)
    ax.set_xlim(-1.0, n)
    ax.set_ylim(*_arc_ylim(trial))
    ax.axis("off")


def _gain_indices(trial: MultiTrial) -> list[int]:
    return [c for c, mod in enumerate(trial.modulators) if mod.mode == "gain"]


def _gain_rings_at(trial: MultiTrial, t: int) -> list[tuple[int, np.ndarray]]:
    """Per gain modulator: (index, current per-neuron gain factor)."""
    out = []
    for c in _gain_indices(trial):
        mod = trial.modulators[c]
        factor = 1.0 + mod.alpha * trial.occ[t, c] * mod.receptor_mask(trial.n_neurons)
        out.append((c, factor))
    return out


def _anatomical_spans(trial: MultiTrial) -> tuple[float, float]:
    """(wired, modulated) longest edge in layout slots — sizes the arc heights."""
    order = np.argsort(trial.positions[:, 0])
    slot = np.empty(trial.n_neurons, dtype=int)
    slot[order] = np.arange(trial.n_neurons)

    def longest(support: np.ndarray) -> float:
        tgt, src = np.nonzero(support)
        if len(src) == 0:
            return 1.0
        return float(max(np.abs(slot[tgt] - slot[src]).max(), 1))

    mod_support = np.zeros((trial.n_neurons, trial.n_neurons), dtype=bool)
    for sup in _additive_supports(trial):
        mod_support |= sup
    return longest(np.abs(trial.A0) > 1e-9), longest(mod_support)


def _arc_ylim(trial: MultiTrial) -> tuple[float, float]:
    wired_span, mod_span = _anatomical_spans(trial)
    return (-(0.35 + 0.11 * wired_span) - 0.25, (0.35 + 0.11 * mod_span) + 0.25)


def _concentrations(trial: MultiTrial) -> np.ndarray:
    """(T, k, N) concentration fields (recomputed from stored latents)."""
    kern = _kernels(trial.positions, trial.modulators)
    T, k = trial.n_times, trial.n_modulators
    out = np.zeros((T, k, trial.n_neurons))
    for c in range(k):
        out[:, c, :] = _softplus(trial.latents[c]) @ kern[c]
    return out


def _zscored(trial: MultiTrial) -> np.ndarray:
    x = trial.x
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def _legend(fig, trial: MultiTrial, y: float = 0.0) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=WIRED, lw=2.6, label="wired synapse (static)")]
    for c, mod in enumerate(trial.modulators):
        what = "gated edges" if mod.mode == "additive" else "input gain (node ring)"
        handles.append(Line2D([0], [0], color=_mod_color(c), lw=2.6,
                              label=f"{mod.name}: {what}"))
    handles.append(Line2D([0], [0], color="0.4", lw=1.6, linestyle=(0, (3, 2)),
                          label="dashed = inhibitory"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------------------
# 0. model card: the equations + parameters of the chosen model
# ---------------------------------------------------------------------------

def plot_model_card(trial: MultiTrial, path: Path, title: str = "Model card") -> None:
    """One-page rendered summary: the exact dynamics equations (mathtext) and
    every parameter of the chosen circuit and modulators.

    Everything on this card is read from the trial itself, so it always
    matches what was actually simulated.
    """
    import matplotlib.pyplot as plt

    n, k, T = trial.n_neurons, trial.n_modulators, trial.n_times
    n_wired = int((np.abs(trial.A0) > 1e-9).sum())
    radius = float(np.max(np.abs(np.linalg.eigvals(trial.A0))))
    phi_str = r"\tanh" if trial.phi == "tanh" else r"\mathrm{id}"

    fig = plt.figure(figsize=(13.0, 9.0))
    fig.suptitle(title, fontsize=16, x=0.06, ha="left", fontweight="bold")

    def block(x, y, header, lines, size=11, family=None, color="#1b1b1b"):
        fig.text(x, y, header, fontsize=12, fontweight="bold", va="top")
        for i, ln in enumerate(lines):
            fig.text(x + 0.015, y - 0.035 - 0.033 * i, ln, fontsize=size,
                     va="top", family=family, color=color)

    # ---- left column: the dynamics, exactly as simulated -----------------
    eq = [
        (r"$x_{t+1} = \phi\left(A(t)\,x_t\right) + \sigma_x\,\varepsilon_t$"
         + rf",   $\phi = {phi_str}$,   $\sigma_x = {trial.meta.get('sigma_x', float('nan')):g}$",
         "fast activity: one-step update through the current effective operator"),
        (r"$A(t) = \mathrm{diag}(g(t))\,\left[A_0 + \sum_{c\,\in\,\mathrm{additive}} "
         r"\alpha_c\,\mathrm{diag}(o_c(t))\,W_c\right]$",
         "the operator = static wiring + occupancy-gated receptor layers, row-scaled by gain"),
        (r"$g_i(t) = \prod_{c\,\in\,\mathrm{gain}}\left(1+\alpha_c\,o_{c,i}(t)\right)"
         r"^{[i \in R_c]}$",
         "input-gain modulators multiply the total input of receptor-bearing neurons"),
        (r"$o_{c,i}(t) = \frac{u_{c,i}}{u_{c,i}+K_c}$,   "
         r"$u_{c,i}(t) = \sum_{s}\,e^{-d(i,s)/\lambda_c}\;\mathrm{softplus}\,(z_{c,s}(t))$",
         "receptor occupancy: Michaelis–Menten on a distance-decayed concentration field"),
        (r"$z_{c}(t+1) = \rho_c\,z_c(t) + \sqrt{1-\rho_c^2}\;\sigma_c\,\eta_t"
         r" + g^{\mathrm{rel}}_c\,[x_{\mathrm{src}}]_+$",
         "hidden release latents: AR(1) per modulator, optional activity-driven release"),
        (r"$J(t) = \mathrm{diag}\left(1-\tanh^2(u_t)\right)A(t)$,  $u_t = A(t)\,x_t$"
         if trial.phi == "tanh" else r"$J(t) = A(t)$",
         "the estimand: exact Jacobian of the update (what methods must track from x alone)"),
    ]
    fig.text(0.06, 0.90, "Dynamics (as simulated)", fontsize=12, fontweight="bold", va="top")
    y = 0.855
    for math, caption in eq:
        fig.text(0.075, y, math, fontsize=13, va="top")
        fig.text(0.075, y - 0.040, caption, fontsize=8.5, va="top", color="0.35")
        y -= 0.095

    fig.text(0.06, y + 0.02,
             "Methods receive only $x_{0..T}$. Occupancies, latents, and $J(t)$\n"
             "are the hidden answer key used for scoring.",
             fontsize=10, va="top", style="italic")

    # ---- right column: parameters of THIS model ---------------------------
    circuit_lines = [
        f"neurons N = {n}    timesteps T = {T}    seed = {trial.meta.get('seed', '?')}",
        f"wired edges in A0: {n_wired}    spectral radius(A0) = {radius:.3f}",
        f"observation noise = {trial.meta.get('obs_noise', 0):g}",
        f"positions: line layout (1 unit spacing)",
    ]
    block(0.56, 0.90, "Circuit", circuit_lines, size=10, family="monospace")

    y = 0.71
    fig.text(0.56, y, f"Hidden modulators (k = {k})", fontsize=12,
             fontweight="bold", va="top")
    y -= 0.035
    for c, mod in enumerate(trial.modulators):
        tau = -1.0 / np.log(mod.rho) if mod.rho < 1 else float("inf")
        reach = "broadcast (λ=∞)" if not np.isfinite(mod.decay_length) \
            else f"λ = {mod.decay_length:g}"
        if mod.mode == "additive":
            n_edges = int((np.abs(mod.W) > 1e-9).sum())
            mech = f"additive: {n_edges} gated receptor edges (W_c)"
        else:
            mech = f"gain on {len(mod.receptors)} receptor neurons"
        src = f"sources {tuple(mod.sources)}" if mod.sources else "global source"
        lines = [
            mech,
            f"α = {mod.alpha:g}   ρ = {mod.rho:g}  (τ ≈ {tau:,.0f} steps)   σ = {mod.sigma:g}",
            f"{reach}   K½ = {mod.K_half:g}   release_gain = {mod.release_gain:g}   {src}",
        ]
        fig.text(0.56, y, f"● {mod.name}  ({mod.mode})", fontsize=11,
                 fontweight="bold", va="top", color=_mod_color(c))
        for i, ln in enumerate(lines):
            fig.text(0.575, y - 0.028 - 0.026 * i, ln, fontsize=9,
                     va="top", family="monospace")
        y -= 0.125

    fig.text(0.06, 0.045,
             "Taxonomy: additive = extrasynaptic edges gated at the target by receptor "
             "occupancy (H2); gain = input-gain rescaling, Chance–Abbott–Reyes 2002 (H1). "
             "Spatial field = quasi-steady-state exponential kernel (volume transmission).",
             fontsize=8.5, color="0.35")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. static anatomy
# ---------------------------------------------------------------------------

def plot_static_graph(trial: MultiTrial, path: Path) -> None:
    import matplotlib.pyplot as plt

    n = trial.n_neurons
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 6.4),
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    _draw_arc_graph(axes[0], trial, trial.A0, None,
                    node_color=np.zeros(n), node_size=260)
    axes[0].set_title("Wired circuit A₀ (static anatomy; arcs below the line)",
                      loc="left", fontsize=12)

    # docking map: where each modulator enters
    ax = axes[1]
    xs = np.arange(n, dtype=float)
    order = np.argsort(trial.positions[:, 0])
    for c, mod in enumerate(trial.modulators):
        color = _mod_color(c)
        y = -0.5 - 0.55 * c
        mask = mod.receptor_mask(n)[order]
        ax.scatter(xs[mask], np.full(mask.sum(), y), marker="s", s=90,
                   color=color, alpha=0.8,
                   label=f"{mod.name} receptor neurons ({mod.mode})")
        for s in mod.sources:
            slot = float(np.where(order == s)[0][0])
            ax.scatter([slot], [y], marker="*", s=340, color=color,
                       edgecolor="k", zorder=3)
            reach = mod.decay_length
            if np.isfinite(reach):
                ax.plot([slot - reach, slot + reach], [y, y], color=color,
                        lw=1.2, alpha=0.5)
        if not mod.sources:
            ax.plot([0, n - 1], [y, y], color=color, lw=1.2, alpha=0.35)
    ax.scatter(xs, np.zeros(n), s=200, facecolor="#f7f4ee",
               edgecolor=NODE_EDGE, zorder=3)
    for i, node in enumerate(order):
        ax.text(xs[i], 0.0, str(node), ha="center", va="center", fontsize=6, zorder=4)
    ax.set_xlim(-1.0, n)
    ax.set_ylim(-0.6 - 0.55 * trial.n_modulators, 0.6)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Where the modulators dock: ★ release source (line = spatial reach λ), "
                 "■ receptor-bearing neurons", loc="left", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. snapshot at specific times
# ---------------------------------------------------------------------------

def plot_snapshot(trial: MultiTrial, times: tuple[int, ...], path: Path) -> None:
    import matplotlib.pyplot as plt

    conc = _concentrations(trial)
    z = _zscored(trial)
    m = trial.m_summary
    k = trial.n_modulators

    fig, axes = plt.subplots(2, len(times), figsize=(9.5 * len(times), 6.6),
                             gridspec_kw={"height_ratios": [0.7, 1.3]},
                             squeeze=False)
    xs = np.arange(trial.n_neurons)
    order = np.argsort(trial.positions[:, 0])
    for j, t in enumerate(times):
        ax = axes[0, j]
        for c in range(k):
            ax.fill_between(xs, conc[t, c][order], alpha=0.30, color=_mod_color(c))
            ax.plot(xs, conc[t, c][order], color=_mod_color(c), lw=1.6,
                    label=f"{trial.modulators[c].name}  (m̄={m[t, c]:.2f})")
        ax.set_xlim(-1.0, trial.n_neurons)
        ax.set_ylabel("concentration")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.set_title(f"t = {t}: modulator fields over the circuit", loc="left", fontsize=11)
        ax.tick_params(labelbottom=False)

        _draw_arc_graph(axes[1, j], trial, trial.operator_at(t), trial.occ[t],
                        node_color=z[t], gain_rings=_gain_rings_at(trial, t))
        axes[1, j].set_title(f"effective graph J(t={t}); node color = activity",
                             loc="left", fontsize=11)
    _legend(fig, trial, y=-0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2b. diffusion over space AND time (one heatmap per modulator)
# ---------------------------------------------------------------------------

def plot_diffusion(trial: MultiTrial, path: Path) -> None:
    """Per modulator: concentration as a (neuron x time) heatmap.

    This is the clearest view of diffusion: the vertical axis is the circuit
    (neurons in layout order, release sources marked ◀), the horizontal axis
    is time. A local modulator shows a horizontal bright band around its
    source that swells and fades at its own speed; a broadcast modulator
    lights all rows at once. Fast vs slow kinetics read directly as fine vs
    coarse temporal texture.
    """
    import matplotlib.pyplot as plt

    conc = _concentrations(trial)
    n, k, T = trial.n_neurons, trial.n_modulators, trial.n_times
    order = np.argsort(trial.positions[:, 0])
    slot = np.empty(n, dtype=int)
    slot[order] = np.arange(n)

    fig, axes = plt.subplots(k + 1, 1, figsize=(13.0, 1.9 * k + 2.6),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1.0] * k + [0.9]})
    for c, mod in enumerate(trial.modulators):
        ax = axes[c]
        field = conc[:, c][:, order].T                     # (n, T)
        im = ax.imshow(field, aspect="auto", cmap="magma", origin="upper",
                       extent=(0, T, n - 0.5, -0.5), interpolation="nearest")
        for s in mod.sources:
            ax.plot(-T * 0.008, slot[s], marker="<", color=_mod_color(c),
                    markersize=9, clip_on=False)
        reach = "broadcast" if not np.isfinite(mod.decay_length) \
            else f"λ={mod.decay_length:g}"
        ax.set_title(f"{mod.name}: concentration over the circuit — "
                     f"{mod.mode}, ρ={mod.rho:g} "
                     f"({'slow' if mod.rho > 0.99 else 'fast'} kinetics), {reach}",
                     loc="left", fontsize=10, color=_mod_color(c))
        ax.set_ylabel("neuron")
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)

    ax = axes[-1]
    for c, mod in enumerate(trial.modulators):
        ax.plot(trial.m_summary[:, c], color=_mod_color(c), lw=1.0,
                label=f"{mod.name} (ρ={mod.rho:g})")
    ax.set_xlim(0, T)
    ax.set_ylabel("mean receptor\noccupancy")
    ax.set_xlabel("time step")
    ax.legend(fontsize=8, frameon=False, ncol=min(k, 4), loc="upper right")
    ax.set_title("Summary state per modulator (what a method must recover, "
                 "slow curves vs fast curves = the different timescales)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. spacetime (unrolled) graph
# ---------------------------------------------------------------------------

def plot_spacetime(trial: MultiTrial, windows: list[tuple[int, int, str]],
                   path: Path, edge_threshold: float = 0.04) -> None:
    """Rows = neurons, columns = time; edge (i,t)->(j,t+1) drawn iff |J[t][j,i]| > thr."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    z = _zscored(trial)
    supports = _additive_supports(trial)
    n = trial.n_neurons
    order = np.argsort(trial.positions[:, 0])
    row = {node: i for i, node in enumerate(order)}

    fig, axes = plt.subplots(len(windows), 1, figsize=(13.5, 4.4 * len(windows)),
                             squeeze=False)
    for p, (t0, t1, label) in enumerate(windows):
        ax = axes[p, 0]
        cols = np.arange(t0, t1)
        segs, colors, styles, widths = [], [], [], []
        for ci, t in enumerate(cols[:-1]):
            J = trial.J[t] if trial.J is not None else trial.jacobian_at(t)
            src, tgt = np.nonzero(np.abs(J) > edge_threshold)[1], \
                np.nonzero(np.abs(J) > edge_threshold)[0]
            for s, g in zip(src, tgt):
                w = float(J[g, s])
                owner = next((c for c, sup in enumerate(supports) if sup[g, s]), None)
                segs.append([(ci, row[s]), (ci + 1, row[g])])
                if owner is not None:
                    colors.append(_mod_color(owner))
                    widths.append(max(abs(w) * 6.5, 0.4))
                else:
                    colors.append(WIRED)
                    widths.append(max(abs(w) * 3.2, 0.3))
                styles.append("solid" if w > 0 else (0, (3, 2)))
        from matplotlib.colors import to_rgb

        rgba = [(*to_rgb(c), 0.9 if c != WIRED else 0.45) for c in colors]
        lc = LineCollection(segs, colors=rgba, linewidths=widths,
                            linestyles=styles, zorder=1)
        ax.add_collection(lc)
        # nodes: neuron i at time t, colored by activity
        cc, rr = np.meshgrid(np.arange(len(cols)), np.arange(n))
        acts = z[cols][:, order].T                     # (n, len(cols))
        ax.scatter(cc.ravel(), rr.ravel(), c=acts.ravel(), cmap="coolwarm",
                   vmin=-2.5, vmax=2.5, s=42, edgecolor="0.25", linewidth=0.4,
                   zorder=3)
        ax.set_yticks(range(n))
        ax.set_yticklabels([str(node) for node in order], fontsize=6)
        ax.set_ylabel("neuron")
        ax.set_xticks(range(0, len(cols), 2))
        ax.set_xticklabels([str(t) for t in cols[::2]], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{label}   (each column = the circuit at one time step; "
                     "an edge connects neuron i at t to neuron j at t+1 iff J[t][j,i] ≠ 0)",
                     loc="left", fontsize=11)
        ax.set_xlim(-0.6, len(cols) - 0.4)
    axes[-1, 0].set_xlabel("time step")
    _legend(fig, trial, y=-0.01)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. movie: diffusion + gated edges + information flow
# ---------------------------------------------------------------------------

def render_multi_movie(trial: MultiTrial, path: Path, t0: int, t1: int,
                       step: int = 3, fps: int = 16, dpi: int = 80) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import FancyArrowPatch

    n = trial.n_neurons
    k = trial.n_modulators
    conc = _concentrations(trial)
    z = _zscored(trial)
    m = trial.m_summary
    gain_idxs = _gain_indices(trial)
    order = np.argsort(trial.positions[:, 0])
    col = {node: i for i, node in enumerate(order)}
    xs = np.arange(n, dtype=float)
    frames = list(range(t0, min(t1, trial.n_times - 1), step))

    fig = plt.figure(figsize=(13.0, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.7, 0.85, 0.85], hspace=0.45)
    ax_g = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])
    ax_m = fig.add_subplot(gs[2])

    # --- graph panel: static wired arcs + updatable modulator arcs ---
    node_size = 250.0
    shrink = float(np.sqrt(node_size) / 2.0 + 1.5)
    wired_support = np.abs(trial.A0) > 1e-9
    for s in range(n):
        for g in range(n):
            if not wired_support[g, s]:
                continue
            x0, x1 = xs[col[s]], xs[col[g]]
            span = max(abs(x1 - x0), 1.0)
            w = float(trial.A0[g, s])
            ax_g.add_patch(FancyArrowPatch(
                (x0, 0.0), (x1, 0.0), arrowstyle="-|>", mutation_scale=8,
                lw=max(abs(w) * LW_PER_WEIGHT, 0.4), color=WIRED,
                alpha=min(1.0, 0.25 + 1.1 * abs(w)),
                connectionstyle=f"arc3,rad={_arc_rad(span)}",
                linestyle="-" if w > 0 else (0, (3, 2)),
                shrinkA=shrink, shrinkB=shrink, zorder=1))

    mod_patches = []
    for c, mod in enumerate(trial.modulators):
        if mod.mode != "additive":
            continue
        for s, g in zip(*np.nonzero(np.abs(mod.W).T > 1e-9)):
            x0, x1 = xs[col[s]], xs[col[g]]
            span = max(abs(x1 - x0), 1.0)
            patch = FancyArrowPatch(
                (x0, 0.0), (x1, 0.0), arrowstyle="-|>", mutation_scale=9,
                lw=0.4, color=_mod_color(c), alpha=0.0,
                connectionstyle=f"arc3,rad={-_arc_rad(span)}",
                shrinkA=shrink, shrinkB=shrink, zorder=2)
            ax_g.add_patch(patch)
            mod_patches.append((patch, c, int(s), int(g)))

    nodes = ax_g.scatter(xs, np.zeros(n), s=node_size, c=np.zeros(n),
                         cmap="coolwarm", vmin=-2.5, vmax=2.5,
                         edgecolor=NODE_EDGE, linewidth=1.0, zorder=3)
    ring_artists = []
    for j, c_idx in enumerate(gain_idxs):
        ring_artists.append((c_idx, ax_g.scatter(
            xs, np.zeros(n), s=node_size * (2.1 + 1.1 * j), facecolor="none",
            edgecolor=_mod_color(c_idx), zorder=2, linewidths=np.zeros(n))))
    for i, node in enumerate(order):
        ax_g.text(xs[i], 0.0, str(node), ha="center", va="center",
                  fontsize=6, zorder=4)
    readout = ax_g.text(0.01, 0.97, "", transform=ax_g.transAxes, fontsize=11,
                        va="top", family="monospace")
    ax_g.set_xlim(-1.0, n)
    ax_g.set_ylim(*_arc_ylim(trial))
    ax_g.axis("off")
    ax_g.set_title("Effective graph J(t): wired arcs below (static), "
                   "modulator edges above (gated), node color = activity, "
                   "ring = input gain", loc="left", fontsize=11)

    # --- diffusion panel ---
    conc_lines = []
    conc_fills: list = [None] * k
    for c in range(k):
        (ln,) = ax_c.plot(xs, conc[t0, c][order], color=_mod_color(c), lw=1.8,
                          label=trial.modulators[c].name)
        conc_lines.append(ln)
        conc_fills[c] = ax_c.fill_between(xs, conc[t0, c][order],
                                          color=_mod_color(c), alpha=0.25)
    for c, mod in enumerate(trial.modulators):
        for s in mod.sources:
            ax_c.axvline(float(np.where(order == s)[0][0]), color=_mod_color(c),
                         lw=0.8, alpha=0.4)
    ax_c.set_xlim(-1.0, n)
    ax_c.set_ylim(0, float(conc[t0:t1].max()) * 1.15 + 1e-6)
    ax_c.set_ylabel("concentration")
    ax_c.set_xlabel("neuron (by position)")
    ax_c.legend(fontsize=8, frameon=False, loc="upper right")
    ax_c.set_title("Diffusion fields: each modulator's concentration over the circuit "
                   "(vertical line = release source)", loc="left", fontsize=10)

    # --- hidden state panel ---
    tt = np.arange(t0, t1)
    for c in range(k):
        ax_m.plot(tt, m[t0:t1, c], color=_mod_color(c), lw=1.2,
                  label=trial.modulators[c].name)
    ax_m.set_xlim(t0, t1 - 1)
    ax_m.set_ylabel("mean occupancy")
    ax_m.set_xlabel("time")
    ax_m.legend(fontsize=8, frameon=False, loc="upper right")
    cursor = ax_m.axvline(t0, color="#1b1b1b", lw=1.5)
    ax_m.set_title("Hidden modulator states (never observed by methods)",
                   loc="left", fontsize=10)

    def update(t: int):
        for patch, c, s, g in mod_patches:
            mod = trial.modulators[c]
            w = mod.alpha * trial.occ[t, c, g] * float(mod.W[g, s])
            patch.set_linewidth(max(abs(w) * LW_PER_WEIGHT, 0.3))
            patch.set_alpha(float(min(1.0, 0.08 + 2.4 * abs(w))))
            patch.set_linestyle("-" if w >= 0 else (0, (3, 2)))
        nodes.set_array(z[t][order])
        for c_idx, artist in ring_artists:
            mod = trial.modulators[c_idx]
            factor = 1.0 + mod.alpha * trial.occ[t, c_idx] * mod.receptor_mask(n)
            artist.set_linewidths(np.clip((factor[order] - 1.0) * 6.0, 0.0, 4.0))
        for c in range(k):
            conc_lines[c].set_ydata(conc[t, c][order])
            conc_fills[c].remove()
            conc_fills[c] = ax_c.fill_between(xs, conc[t, c][order],
                                              color=_mod_color(c), alpha=0.25)
        vals = "  ".join(f"{trial.modulators[c].name}={m[t, c]:.2f}" for c in range(k))
        readout.set_text(f"t = {t}    {vals}")
        cursor.set_xdata([t, t])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# shared: choosing interesting times to display
# ---------------------------------------------------------------------------

def reference_times(trial: MultiTrial, margin: int = 200) -> tuple[int, int, int]:
    """(reference modulator index, low-m time, high-m time).

    The reference is the slowest ADDITIVE modulator (its gated edges are the
    visually obvious change); falls back to modulator 0. Times avoid the
    first `margin` steps (burn-in) and the last 50 (window edge effects).
    """
    add_idx = [c for c, mod in enumerate(trial.modulators) if mod.mode == "additive"]
    ref = max(add_idx, key=lambda c: trial.modulators[c].rho) if add_idx else 0
    m0 = trial.m_summary[:, ref]
    lo_t = int(np.argmin(m0[margin:-50]) + margin)
    hi_t = int(np.argmax(m0[margin:-50]) + margin)
    return ref, lo_t, hi_t


# ---------------------------------------------------------------------------
# CLI: build the demo model and render everything
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), default="two")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--n", type=int, default=None,
                   help="override the preset's neuron count")
    p.add_argument("--T", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phi", default="tanh", choices=["linear", "tanh"])
    p.add_argument("--skip-movie", action="store_true")
    args = p.parse_args(argv)

    default_n, builder = PRESETS[args.preset]
    n = args.n if args.n is not None else default_n
    outdir = args.outdir if args.outdir is not None \
        else Path("artifacts") / ("multi" if args.preset == "two" else f"multi_{args.preset}")

    trial = simulate_multi(build_circuit(n, args.seed), builder(n, args.seed),
                           T=args.T, phi=args.phi, sigma_x=0.05, seed=args.seed)
    ref, lo_t, hi_t = reference_times(trial)
    m0 = trial.m_summary[:, ref]
    ref_name = trial.modulators[ref].name

    plot_model_card(trial, outdir / "00_model_card.png",
                    title=f"Model card — preset '{args.preset}', phi={args.phi}")
    plot_static_graph(trial, outdir / "10_static_graph.png")
    plot_snapshot(trial, (lo_t, hi_t), outdir / "11_snapshot.png")
    plot_diffusion(trial, outdir / "12_diffusion.png")
    plot_spacetime(trial, [
        (lo_t, lo_t + 16, f"LOW {ref_name} (t={lo_t}, m̄={m0[lo_t]:.2f})"),
        (hi_t, hi_t + 16, f"HIGH {ref_name} (t={hi_t}, m̄={m0[hi_t]:.2f})"),
    ], outdir / "13_spacetime.png")
    print(f"wrote {outdir}/00_model_card.png, 10_static_graph.png, "
          "11_snapshot.png, 12_diffusion.png, 13_spacetime.png")
    if not args.skip_movie:
        span = 500
        t0 = max(200, min(lo_t, hi_t) - 50)
        gif = render_multi_movie(trial, outdir / "14_multi_movie.gif",
                                 t0=t0, t1=min(t0 + span, args.T - 2))
        print(f"wrote {gif}")


if __name__ == "__main__":
    main()

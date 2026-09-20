"""Interactive dashboard for the multi-modulator testbed.

    pip install -e ".[dashboard]"
    streamlit run neuromodsim/dashboard.py

Everything adjustable lives in the sidebar: the wired circuit (or upload
your own A0), the global dynamics (linear/tanh, noise, length), and each
modulator channel (mechanism, coupling, timescale, spatial reach, receptor
connectome — random / hand-typed edge list / CSV upload). The tabs render
the same figures as `python -m neuromodsim.viz_multi`, always from the
model currently configured, plus a live scoring table.

Simulations are cached on the full configuration, so switching tabs and
re-rendering figures is free until you change a parameter.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st

from neuromodsim.methods import SlidingVAR, StaticVAR
from neuromodsim.multi import Modulator, receptor_layer, score_multi, simulate_multi
from neuromodsim.presets import PRESETS, build_circuit
from neuromodsim.viz_multi import (
    plot_diffusion,
    plot_model_card,
    plot_snapshot,
    plot_spacetime,
    plot_static_graph,
    reference_times,
    render_multi_movie,
)

CACHE_DIR = Path("artifacts/dashboard")

# log-spaced timescale choices; tau = -1/ln(rho)  <=>  rho = exp(-1/tau)
TAU_CHOICES = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

st.set_page_config(page_title="neuromodsim", layout="wide")


# ---------------------------------------------------------------------------
# sidebar: configuration widgets
# ---------------------------------------------------------------------------

def _parse_csv_matrix(uploaded, n: int, label: str) -> np.ndarray | None:
    """Read an uploaded CSV as an (n, n) matrix; report problems in the UI."""
    if uploaded is None:
        return None
    try:
        M = np.loadtxt(io.StringIO(uploaded.getvalue().decode()), delimiter=",")
    except Exception as e:  # user data: report, don't crash
        st.sidebar.error(f"{label}: could not parse CSV ({e})")
        return None
    if M.shape != (n, n):
        st.sidebar.error(f"{label}: expected shape ({n},{n}), got {M.shape}")
        return None
    return M


def _parse_edge_lines(text: str, n: int, label: str) -> list[tuple[int, int, float]]:
    """Parse 'source,target,weight' lines; skip blanks; report bad lines."""
    edges = []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            s, t, w = ln.split(",")
            s, t, w = int(s), int(t), float(w)
        except ValueError:
            st.sidebar.error(f"{label}: bad line {ln!r} (want src,tgt,weight)")
            continue
        if not (0 <= s < n and 0 <= t < n):
            st.sidebar.error(f"{label}: neuron index out of range in {ln!r}")
            continue
        edges.append((s, t, w))
    return edges


def _modulator_widgets(c: int, n: int, defaults: Modulator | None,
                       preset: str) -> dict:
    """One modulator's controls. Returns a JSON-serializable spec dict."""
    d = defaults
    key = f"{preset}_{n}_{c}"           # re-keyed on preset/n so defaults reload
    name = st.text_input("name", value=d.name if d else f"mod{c}", key=f"{key}_name")
    mode = st.selectbox("mechanism", ["additive", "gain"],
                        index=0 if (d is None or d.mode == "additive") else 1,
                        key=f"{key}_mode",
                        help="additive = gated extra edges (H2); "
                             "gain = input-gain scaling (H1)")
    alpha = st.slider("alpha (coupling strength)", 0.0, 1.5,
                      float(d.alpha) if d else 0.6, 0.05, key=f"{key}_alpha")
    tau_default = min(TAU_CHOICES,
                      key=lambda t: abs(t - (-1 / np.log(d.rho))) if d and d.rho < 1 else t)
    tau = st.select_slider("timescale tau (steps)", TAU_CHOICES,
                           value=tau_default, key=f"{key}_tau",
                           help="AR(1) persistence rho = exp(-1/tau)")
    st.caption(f"rho = {np.exp(-1.0 / tau):.4f}")
    sigma = st.slider("sigma (release variability)", 0.2, 2.0,
                      float(d.sigma) if d else 1.0, 0.1, key=f"{key}_sigma")
    k_half = st.slider("K half (receptor saturation)", 0.2, 3.0,
                       float(d.K_half) if d else 1.0, 0.1, key=f"{key}_khalf")
    release_gain = st.slider("release gain (activity-driven release)", 0.0, 1.0,
                             float(d.release_gain) if d else 0.0, 0.05,
                             key=f"{key}_rg")

    default_sources = [s for s in (d.sources if d else ())] if d else []
    sources = st.multiselect("release sources (empty = global)", list(range(n)),
                             default=[s for s in default_sources if s < n],
                             key=f"{key}_src")
    broadcast = st.checkbox(
        "broadcast (lambda = inf)",
        value=(d is None or not np.isfinite(d.decay_length)),
        key=f"{key}_bc")
    decay = np.inf if broadcast else st.slider(
        "spatial reach lambda", 1.0, float(n),
        float(d.decay_length) if d and np.isfinite(d.decay_length) else 4.0,
        0.5, key=f"{key}_lam")

    spec: dict = dict(name=name, mode=mode, alpha=alpha,
                      rho=float(np.exp(-1.0 / tau)), sigma=sigma,
                      K_half=k_half, release_gain=release_gain,
                      sources=sources,
                      decay_length=None if broadcast else float(decay))

    if mode == "additive":
        how = st.radio("receptor connectome", ["random edges", "edge list", "upload CSV"],
                       key=f"{key}_how", horizontal=True)
        if how == "random edges":
            n_edges = st.slider("number of edges", 1, 12, 4, key=f"{key}_ne")
            edge_seed = st.number_input("edge seed", 0, 9999,
                                        value=c, key=f"{key}_es")
            spec["edges"] = ("random", int(n_edges), int(edge_seed))
        elif how == "edge list":
            default_text = ""
            if d is not None and d.W is not None:
                tgt, src = np.nonzero(np.abs(d.W) > 1e-9)
                default_text = "\n".join(f"{s},{t},{d.W[t, s]:g}"
                                         for s, t in zip(src, tgt))
            text = st.text_area("edges (source,target,weight per line)",
                                value=default_text, key=f"{key}_txt", height=120)
            spec["edges"] = ("list", _parse_edge_lines(text, n, name))
        else:
            up = st.file_uploader(f"W for {name} (CSV, {n}x{n}, W[target,source])",
                                  type="csv", key=f"{key}_up")
            M = _parse_csv_matrix(up, n, name)
            spec["edges"] = ("matrix", M.tolist() if M is not None else None)
    else:
        default_receptors = [r for r in (d.receptors if d else ()) if r < n]
        receptors = st.multiselect("receptor neurons (gain applies here)",
                                   list(range(n)),
                                   default=default_receptors or list(range(min(3, n))),
                                   key=f"{key}_rec")
        spec["receptors"] = receptors
    return spec


def _build_modulator(spec: dict, n: int) -> Modulator:
    """Turn a widget spec dict back into a Modulator."""
    if spec["mode"] == "additive":
        kind = spec["edges"][0]
        if kind == "random":
            _, n_edges, edge_seed = spec["edges"]
            rng = np.random.default_rng(edge_seed)
            pairs = set()
            while len(pairs) < min(n_edges, n * (n - 1)):
                s, t = rng.integers(0, n, size=2)
                if s != t:
                    pairs.add((int(s), int(t)))
            W = receptor_layer(n, [(s, t, float(rng.choice([-0.35, 0.45])))
                                   for s, t in pairs])
        elif kind == "list":
            W = receptor_layer(n, spec["edges"][1])
        else:
            M = spec["edges"][1]
            W = np.array(M) if M is not None else np.zeros((n, n))
        extra: dict = dict(W=W)
    else:
        extra = dict(receptors=tuple(spec["receptors"]))
    return Modulator(
        name=spec["name"], mode=spec["mode"], alpha=spec["alpha"],
        rho=spec["rho"], sigma=spec["sigma"], K_half=spec["K_half"],
        release_gain=spec["release_gain"], sources=tuple(spec["sources"]),
        decay_length=np.inf if spec["decay_length"] is None else spec["decay_length"],
        **extra,
    )


# ---------------------------------------------------------------------------
# cached simulation + figure rendering
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="simulating...")
def run_simulation(config: str):
    """config is the full JSON spec; returns (trial, cache_key)."""
    cfg = json.loads(config)
    n = cfg["n"]
    if cfg["A0"] is not None:
        A0 = np.array(cfg["A0"])
    else:
        A0 = build_circuit(n, cfg["circuit_seed"],
                           chain_weight=cfg["chain_weight"],
                           recurrent_prob=cfg["recurrent_prob"],
                           recurrent_span=cfg["recurrent_span"])
    mods = [_build_modulator(s, n) for s in cfg["modulators"]]
    trial = simulate_multi(A0, mods, T=cfg["T"], phi=cfg["phi"],
                           sigma_x=cfg["sigma_x"], obs_noise=cfg["obs_noise"],
                           seed=cfg["sim_seed"])
    key = hashlib.md5(config.encode()).hexdigest()[:12]
    return trial, key


def figure(trial, key: str, name: str, render) -> Path:
    """Render figure `name` for this config once; reuse the PNG afterwards."""
    path = CACHE_DIR / key / name
    if not path.exists():
        render(path)
    return path


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Unobserved-modulator testbed")
    st.caption("Configure a circuit and hidden modulator channels; every view "
               "is rendered from the exact model you configured. "
               "See docs/MODEL.md for the dynamics.")

    sb = st.sidebar
    sb.header("Preset")
    preset = sb.selectbox("start from", list(PRESETS), index=0,
                          help="Presets seed the controls below; edit freely.")
    default_n, builder = PRESETS[preset]

    sb.header("Circuit (A0)")
    n = int(sb.number_input("neurons N", 4, 64, default_n))
    up_A0 = sb.file_uploader(f"upload A0 (CSV, {n}x{n}, A0[target,source])",
                             type="csv")
    A0_data = _parse_csv_matrix(up_A0, n, "A0")
    chain_weight = sb.slider("chain weight", 0.0, 0.8, 0.45, 0.05)
    recurrent_prob = sb.slider("recurrent edge probability", 0.0, 0.5, 0.10, 0.02)
    recurrent_span = sb.slider("recurrent span (max |i-j|)", 1, 6, 3)
    circuit_seed = int(sb.number_input("circuit seed", 0, 9999, 0))

    sb.header("Dynamics")
    phi = sb.radio("phi", ["tanh", "linear"], horizontal=True)
    T = int(sb.slider("timesteps T", 500, 8000, 3000, 250))
    sigma_x = sb.slider("process noise sigma_x", 0.0, 0.3, 0.05, 0.01)
    obs_noise = sb.slider("observation noise", 0.0, 0.3, 0.0, 0.01)
    sim_seed = int(sb.number_input("simulation seed", 0, 9999, 0))

    sb.header("Hidden modulators")
    preset_mods = builder(n, circuit_seed) if n >= 12 else builder(max(n, 12), circuit_seed)
    k = int(sb.number_input("number of modulators", 0, 6, len(preset_mods)))
    specs = []
    for c in range(k):
        d = preset_mods[c] if c < len(preset_mods) else None
        with sb.expander(f"modulator {c}: "
                         f"{d.name if d else 'custom'}", expanded=False):
            specs.append(_modulator_widgets(c, n, d, preset))

    config = json.dumps(dict(
        n=n, A0=A0_data.tolist() if A0_data is not None else None,
        chain_weight=chain_weight, recurrent_prob=recurrent_prob,
        recurrent_span=recurrent_span, circuit_seed=circuit_seed,
        phi=phi, T=T, sigma_x=sigma_x, obs_noise=obs_noise,
        sim_seed=sim_seed, modulators=specs,
    ), sort_keys=True)

    try:
        trial, key = run_simulation(config)
    except ValueError as e:            # e.g. linear stability check refused
        st.error(f"Simulation refused: {e}")
        st.stop()
        return

    ref, lo_t, hi_t = reference_times(trial)
    tabs = st.tabs(["Model card", "Anatomy", "Diffusion", "Snapshot",
                    "Spacetime", "Scores", "Movie"])

    with tabs[0]:
        st.image(str(figure(trial, key, "card.png",
                            lambda p: plot_model_card(trial, p,
                                                      title=f"Model card — phi={phi}"))))
    with tabs[1]:
        st.image(str(figure(trial, key, "static.png",
                            lambda p: plot_static_graph(trial, p))))
    with tabs[2]:
        st.image(str(figure(trial, key, "diffusion.png",
                            lambda p: plot_diffusion(trial, p))))
    with tabs[3]:
        st.caption(f"auto-chosen low/high times of '{trial.modulators[ref].name}' "
                   f"(the slowest additive modulator)")
        st.image(str(figure(trial, key, "snapshot.png",
                            lambda p: plot_snapshot(trial, (lo_t, hi_t), p))))
    with tabs[4]:
        m0 = trial.m_summary[:, ref]
        nm = trial.modulators[ref].name
        st.image(str(figure(trial, key, "spacetime.png", lambda p: plot_spacetime(
            trial,
            [(lo_t, lo_t + 16, f"LOW {nm} (t={lo_t}, m={m0[lo_t]:.2f})"),
             (hi_t, hi_t + 16, f"HIGH {nm} (t={hi_t}, m={m0[hi_t]:.2f})")], p))))
    with tabs[5]:
        window = st.slider("sliding VAR window", 50, 1000, 200, 25)
        rows = [("oracle", trial.J.copy()),
                ("static VAR", StaticVAR().fit_jacobians(trial.observed())),
                (f"sliding VAR (w={window})",
                 SlidingVAR(window=window, lam=1e-2).fit_jacobians(trial.observed()))]
        table = []
        for meth, J_hat in rows:
            s = score_multi(trial, J_hat, meth)
            row = {"method": meth}
            for c, mod in enumerate(trial.modulators):
                row[f"R² {mod.name}"] = round(float(s["recovery_r2"][c]), 3)
            aurocs = [f"{v:.2f}" for v in s["layer_auroc"] if v == v]
            row["layer AUROC"] = "/".join(aurocs) if aurocs else "n/a"
            row["cos(J)"] = round(float(s["jacobian_cosine"]), 3)
            row["1-step MSE"] = round(float(s["one_step_mse"]), 4)
            table.append(row)
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("Oracle = upper bound (modulators leave traces in J). "
                   "Static VAR must be ~0. Sliding VAR should recover exactly "
                   "the modulators slower than its window.")
    with tabs[6]:
        st.caption("Rendering takes ~30 s; the GIF is cached per configuration.")
        if st.button("render movie"):
            t0 = max(200, min(lo_t, hi_t) - 50)
            path = figure(trial, key, "movie.gif",
                          lambda p: render_multi_movie(trial, p, t0=t0,
                                                       t1=min(t0 + 500, T - 2)))
            st.image(str(path))
        else:
            existing = CACHE_DIR / key / "movie.gif"
            if existing.exists():
                st.image(str(existing))


main()

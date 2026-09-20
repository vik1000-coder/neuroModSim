# User guide: running, configuring, extending

Everything runnable, every knob, and how to plug in your own circuits,
receptor connectomes, and methods. For the dynamics and the scientific
rationale see [MODEL.md](MODEL.md).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # numpy, matplotlib, pytest
pip install -e ".[dashboard]"  # + streamlit, for the interactive dashboard
pytest                         # 16 tests, all must pass
```

## Entry points

| command | what it does | output |
|---|---|---|
| `python -m neuromodsim.demo` | single-modulator scoring table + recovery figures | stdout, `artifacts/` |
| `python -m neuromodsim.tour` | single-modulator visual tour (anatomy, modulated graphs, traces, impulses) | `artifacts/tour/` |
| `python -m neuromodsim.movie` | single-modulator animated GIF | `artifacts/` |
| `python -m neuromodsim.multi_demo [--preset two\|many]` | multi-modulator scoring table | stdout |
| `python -m neuromodsim.viz_multi [--preset two\|many] [--skip-movie]` | model card + full multi figure suite + movie | `artifacts/multi*/` |
| `streamlit run neuromodsim/dashboard.py` | interactive dashboard (all knobs, live figures, scores) | browser |

`viz_multi` flags: `--preset {two,many}`, `--n` (override neuron count),
`--T`, `--seed`, `--phi {linear,tanh}`, `--outdir`, `--skip-movie`.

## Figure catalog (`viz_multi`)

| file | view |
|---|---|
| `00_model_card.png` | rendered equations + all parameters of the chosen model |
| `10_static_graph.png` | wired anatomy `A0` + docking map (sources ★, receptors ■, reach λ) |
| `11_snapshot.png` | effective graph `J(t)` at auto-chosen low/high modulator times, concentration fields above |
| `12_diffusion.png` | per-modulator (neuron × time) concentration heatmaps + summary states |
| `13_spacetime.png` | time-unrolled graph: rows = neurons, columns = time, edge (i,t)→(j,t+1) iff `J[t][j,i] ≠ 0` |
| `14_multi_movie.gif` | animation: gated edges, activity, diffusion fields, hidden states + cursor |

All are drawn from a real `MultiTrial`; the plotting functions
(`plot_model_card`, `plot_static_graph`, `plot_snapshot`, `plot_diffusion`,
`plot_spacetime`, `render_multi_movie` in `viz_multi.py`) accept any trial.

## The dashboard

```bash
pip install -e ".[dashboard]"
streamlit run neuromodsim/dashboard.py
```

Sidebar controls, top to bottom:

- **Preset** — start from `two` or `many`, then edit anything.
- **Circuit** — neuron count, chain weight, recurrent density/span/strength,
  seed; or upload your own `A0` as CSV (N×N, `A0[target, source]`).
- **Dynamics** — `phi` (linear/tanh), `T`, `sigma_x`, observation noise, seed.
- **Modulators** — add/remove channels; per channel: name, mode
  (additive/gain), α, ρ (with the implied timescale τ shown), σ, K½,
  release_gain, sources, spatial reach λ (or broadcast), and the receptor
  connectome — random edges, hand-typed `source,target,weight` lines, or a
  CSV upload of the full `W_c`.

Tabs: **Model card**, **Anatomy**, **Diffusion**, **Snapshot**, **Spacetime**
(all live-rendered for the current settings), **Scores** (oracle / static /
sliding VAR with an adjustable window), and **Movie** (on demand — slow).
Simulations are cached by configuration, so flipping between tabs is free.

## Parameter reference

### `Modulator` (multi.py)

| field | default | meaning |
|---|---|---|
| `name` | — | label used in figures/tables |
| `mode` | `additive` | `additive` = gated extra edges (H2); `gain` = input-gain scaling (H1) |
| `W` | None | (N×N) signed receptor connectome, `W[target, source]`; additive mode only |
| `receptors` | `()` | neurons whose input gain is scaled; gain mode only |
| `alpha` | 0.6 | coupling strength |
| `rho` | 0.98 | AR(1) persistence; timescale τ ≈ −1/ln ρ steps |
| `sigma` | 1.0 | release latent stationary std |
| `sources` | `()` | releasing neurons; empty = one global source |
| `decay_length` | ∞ | spatial reach λ; ∞ = broadcast |
| `K_half` | 1.0 | Michaelis constant for receptor occupancy |
| `release_gain` | 0.0 | activity-dependent release (0 = open loop) |

### `simulate_multi`

| arg | default | meaning |
|---|---|---|
| `A0` | — | (N×N) wired operator, `A0[target, source]` |
| `modulators` | — | list of `Modulator` |
| `T` | 3000 | timesteps |
| `positions` | line | (N×2) neuron coordinates for diffusion distances |
| `phi` | `tanh` | `linear` or `tanh` |
| `sigma_x` | 0.05 | process noise std |
| `obs_noise` | 0.0 | additive observation noise on returned `x` |
| `store_J` | True | keep exact (T−1, N, N) Jacobians |
| `seed` | 0 | RNG seed |

### `build_circuit` (presets.py)

| arg | default | meaning |
|---|---|---|
| `n` | — | neurons |
| `chain_weight` | 0.45 | feedforward i→i+1 weight |
| `recurrent_prob` | 0.10 | probability of a signed near-diagonal edge |
| `recurrent_span` | 3 | max \|i−j\| for recurrence |
| `recurrent_lo/hi` | 0.1 / 0.25 | magnitude range of recurrent weights |

### Single-modulator scenarios (scenarios.py)

`additive`, `gain`, `none` (false-positive control), `fast_additive`
(timescale control). `make_scenario(name, seed=..., **overrides)`.

## Plug in your own

**A method** — one function, activity in, Jacobians out:

```python
class MyMethod:
    name = "my_method"
    def fit_jacobians(self, x):          # x: (T, N)
        return J_hat                      # (T-1, N, N)

from neuromodsim import simulate_multi, score_multi
trial = simulate_multi(A0, mods, T=3000, phi="tanh")
print(score_multi(trial, MyMethod().fit_jacobians(trial.observed()), "mine"))
```

Never fit on `trial.occ`, `trial.latents`, or `trial.J` — those are the
answer key.

**A real connectome** — `A0` is any (N, N) matrix; positions any (N, 2);
receptor layers from data:

```python
from neuromodsim import Modulator, receptor_layer
W_5ht = receptor_layer(n, [(src, tgt, w), ...])   # e.g. from Bentley 2016
mods = [Modulator(name="5-HT", mode="additive", W=W_5ht,
                  rho=0.995, sources=(nsm_index,), decay_length=...)]
```

## Repository layout

```
neuromodsim/
  process.py     single-modulator generator (Trial, simulate, pulse_response)
  scenarios.py   named single-modulator scenarios + controls
  methods.py     Estimator protocol + StaticVAR / SlidingVAR / OracleJacobian
  metrics.py     single-modulator scoring
  multi.py       multi-modulator generator (Modulator, MultiTrial,
                 simulate_multi, score_multi)
  presets.py     reference circuits + modulator sets ("two", "many")
  demo.py        single-modulator CLI table
  multi_demo.py  multi-modulator CLI table
  plot.py        single-modulator scoring figures
  tour.py        single-modulator visual tour
  movie.py       single-modulator GIF
  viz_multi.py   multi-modulator figure suite + model card + movie
  dashboard.py   streamlit dashboard
tests/           16 tests: generator invariants, method contracts, exact
                 Jacobians (finite-diff), spatial locality, timescales
docs/            MODEL.md (science) + GUIDE.md (this file)
artifacts/       generated figures (gitignored)
```

# The model: dynamics, mechanisms, and what is being tested

This document is the full specification of the synthetic generators, the
biological reasoning behind each design choice, and the scoring metrics.
For how to run things and which knobs exist, see [GUIDE.md](GUIDE.md).
The one-page rendered version of the equations for any concrete model is the
generated `00_model_card.png` figure (`python -m neuromodsim.viz_multi`).

## 1. The problem

Neuromodulators (serotonin, dopamine, neuropeptides, ...) act largely
*extrasynaptically*: they diffuse through neuropil, bind GPCRs, and change
how the wired circuit responds — without changing the anatomy and often
without being measurable in an activity recording. The claim under test:

> A method that sees only neural activity `x(t)` can recover the *effects*
> of an unobserved neuromodulator propagating through time — i.e. track the
> time-varying effective connectivity the modulator induces.

Three graphs must not be conflated (this framing follows the Compass
artifact stored at the repo root):

| graph | meaning | in this testbed |
|---|---|---|
| structural | anatomy, who *can* talk to whom | `A0` and the receptor layers `W_c` (fixed) |
| effective | who *currently* moves whom (Jacobian of dynamics) | `J(t)` — the estimand |
| functional | statistical dependence in recordings | whatever a method computes from `x` |

A testbed for the claim therefore needs: a hidden modulator with slow
dynamics, a known ground-truth `J(t)`, and controls that catch false
positives.

## 2. Single-modulator generator (`process.py`)

The smallest process that makes the claim testable — linear fast dynamics,
one scalar hidden modulator, two timescales:

```
x_{t+1} = A(m_t) x_t + σ_x ε_t        (fast, observed)
m_{t+1} = ρ m_t + σ_m √(1−ρ²) η_t     (slow, hidden, AR(1))
```

Linearity is a feature here, not a limitation: `J(t) = A(m_t)` exactly, so
scoring is unambiguous. The two-timescale structure (`ρ` close to 1 means
`m` drifts over hundreds of steps while `x` mixes in a few) is the master
picture of neuromodulation: fast neural flow `dx/dt = F(x; W, m)` coupled to
slow modulator dynamics `dm/dt = εG(x, m)`.

`A(m)` follows two mechanisms from the neuromodulation-modeling taxonomy:

- **gain (H1)** — `A(m) = (1 + α·tanh(m)) A0`. Multiplicative rescaling of
  the existing circuit; no new edges. This is the Chance–Abbott–Reyes (2002)
  input-gain picture and the "one parameter reconfigures the whole circuit"
  view from the crustacean STG literature (Marder).
- **additive (H2)** — `A(m) = A0 + α·tanh(m)·A1`. A second, modulator-gated
  edge layer `A1` (long-range "skip" edges in the default scenario): the
  multilayer/extrasynaptic picture (Bentley 2016 monoamine connectome;
  Ripoll-Sánchez 2023 neuropeptide connectome), where modulation adds
  couplings that anatomy alone would not predict.

`tanh(m)` keeps the modulated coefficient bounded so the system stays
stable; `A0` and `A0 ± αA1` are rescaled to a safe spectral radius.

Scenarios (`scenarios.py`) add the two controls every claim needs:

| scenario | purpose |
|---|---|
| `additive` | main effect, mechanism H2 |
| `gain` | main effect, mechanism H1 |
| `none` | `m` exists but never enters `A` — **false-positive control**: a method that "finds" modulation here is hallucinating |
| `fast_additive` | `m` faster than a trailing estimation window — **timescale control**: trailing-window methods must degrade |

## 3. Multi-modulator generator (`multi.py`)

The second-generation instrument generalizes along four axes while keeping
the exact-Jacobian property. Full dynamics, exactly as simulated:

```
activity     x_{t+1} = φ(A(t) x_t) + σ_x ε_t,               φ = id | tanh

operator     A(t) = diag(g(t)) · [ A0 + Σ_{c∈additive} α_c diag(o_c(t)) W_c ]

gain         g_i(t) = Π_{c∈gain} (1 + α_c o_{c,i}(t))^{[i ∈ R_c]}

occupancy    o_{c,i}(t) = u_{c,i} / (u_{c,i} + K_c)                (Michaelis–Menten)
field        u_{c,i}(t) = Σ_s exp(−d(i,s)/λ_c) · softplus(z_{c,s}(t))

release      z_c(t+1) = ρ_c z_c(t) + √(1−ρ_c²) σ_c η_t + g_c^rel [x_src]_+

Jacobian     J(t) = A(t)                                 (φ = id)
             J(t) = diag(1 − tanh²(u_t)) A(t), u_t = A(t)x_t   (φ = tanh)
```

Design decisions, one by one:

- **Receptor connectomes are specifiable.** Each modulator `c` carries its
  own signed layer `W_c[target, source]`. In *C. elegans* these layers are
  literally known (Bentley 2016 for monoamines; Ripoll-Sánchez 2023 for
  neuropeptides; Beets 2023 peptide–GPCR pairing), so a user can plug in the
  real thing via `receptor_layer(n, edges)`. Signs encode receptor polarity
  (e.g. inhibitory MOD-1 vs excitatory SER-4 serotonin channels).
- **Additive gating is at the TARGET.** `diag(o_c)·W_c` scales incoming
  modulated edges by the *target* neuron's receptor occupancy — receptors
  sit on the receiving neuron.
- **Space is a static exponential kernel.** Concentration at neuron `i` is
  a distance-decayed sum over release sites. This is a quasi-steady-state
  approximation of a reaction–diffusion field: the spatial profile is fixed
  by `λ_c`, the temporal kinetics come from the release latents. The honest
  caveat (from the Compass artifact): volume transmission is a *field*, not
  an edge; we approximate the field's spatial shape as frozen. `λ = ∞`
  recovers a global broadcast modulator.
- **Michaelis–Menten occupancy** bounds effects at high concentration —
  receptors saturate; effects cannot grow without limit.
- **Per-modulator timescales.** AR(1) persistence `ρ_c` sets each channel's
  autocorrelation time `τ_c ≈ −1/ln ρ_c`. A neuropeptide at ρ=0.9995
  (τ≈2000 steps) and an amine at ρ=0.85 (τ≈6 steps) coexist; a method with
  an estimation window `w` should recover modulators with `τ ≫ w`'s
  resolution and fail on `τ ≪ w` — the timescale control, per channel.
- **Optional closed loop.** `release_gain > 0` makes release depend on the
  source neuron's rectified activity (x→m coupling à la Kringelbach 2020).
  Default is 0 (open loop) because recovering a modulator that is *not*
  driven by activity is the harder, cleaner version of the claim.
- **Nonlinearity that stays scoreable.** `φ = tanh` is the standard bounded
  rate nonlinearity; the Jacobian remains exact in closed form, so ground
  truth never degrades to an approximation. (Verified against finite
  differences in `tests/test_multi.py`.)
- **Stability.** For linear dynamics a conservative worst-case check
  (spectral radius of |A| with all occupancies at 1) refuses configurations
  that could diverge; `tanh` dynamics are self-bounding.

## 4. What methods see, and how they are scored

Methods implement one function: `fit_jacobians(x) -> J_hat` of shape
`(T−1, N, N)`, receiving `trial.observed()` (activity only). Occupancies,
latents, and true `J` are the answer key.

Baselines (`methods.py`) bracket the problem:

- `OracleJacobian` — returns true `J(t)`; upper bound, sanity check that
  the modulators actually leave traces in `J`.
- `StaticVAR` — one ridge-regression operator for all time; lower bound,
  *must* score ≈0 on modulator recovery (a fixed graph cannot carry a
  time-varying signal).
- `SlidingVAR(window)` — ridge VAR in a trailing window; the weakest
  activity-only method that can succeed when the modulator is slow.

Single-modulator metrics (`metrics.py`): `jacobian_cosine` (per-time cosine
to true `J`), `modulator_corr` (correlation between hidden `m` and a blind
1-D PCA readout of `J_hat(t)` — the headline), `lag_modulator_corr` (delayed
influence `A^k` tracking), `support_auroc` (does `ΔJ` select the gated layer
`A1`), `one_step_mse`.

Multi-modulator metrics (`score_multi`): `recovery_r2[c]` — R² for
regressing each hidden `m_c` on the top principal components of
`vec(J_hat(t))` (blind readout; truth enters only at scoring); `layer_auroc[c]`
— for additive modulators, does `|J(high m_c) − J(low m_c)|` rank that
modulator's true receptor edges first; plus `jacobian_cosine`, `one_step_mse`.

Reference numbers (preset `many`, 12 neurons, 4 modulators, T=3000): oracle
recovers all four (R² ≈ 1.0); static VAR recovers none (0.00); sliding VAR
(w=200) recovers the two slow channels (≈0.57, 0.63) and fails the medium
and fast ones (≈0.05, 0.01) — recoverability tracks timescale, per channel.

## 5. Honest limitations

- Space: frozen spatial kernel, no transport dynamics (no delayed arrival
  of a diffusing bolus). Fine for the identification claim; wrong if the
  method under test specifically claims to see diffusion *fronts*.
- Kinetics: AR(1) release, first-order binding. No receptor
  desensitization, no second-messenger cascades.
- One nonlinearity (`tanh`), no spiking, no dale's-law enforcement.
- Line layout in the bundled presets (positions are general in the API).

References: Marder 2012 (STG reconfiguration); Chance, Abbott & Reyes 2002
(gain); Bentley et al. 2016 (monoamine connectome); Ripoll-Sánchez et al.
2023 (neuropeptide connectome); Beets et al. 2023 (peptide–GPCR pairs);
Dag et al. 2023 (serotonin receptor polarity); Kringelbach & Deco 2020
(activity-coupled neuromodulation). See the Compass artifact markdown at the
repo root for the framing document.

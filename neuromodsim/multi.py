"""Second-generation generator: many neurons, many modulators, space, nonlinearity.

Extends the minimal instrument (process.py) along four axes, staying honest
about what each approximation is:

1. MANY MODULATORS. Each `Modulator` c carries its own receptor connectome
   W_c[target, source] (specifiable — in C. elegans these layers are known:
   Bentley 2016 monoamines, Ripoll-Sánchez 2023 neuropeptides, Beets 2023
   peptide–GPCR pairs), its own kinetics, and its own spatial reach.

2. SPACE / DIFFUSION. Neurons have positions. Each modulator is released by
   `sources` and its concentration at neuron i is a distance-decayed sum
   c_i = Σ_s exp(−d(i,s)/λ_c) · release_s(t). Receptor occupancy is
   Michaelis–Menten: occ_i = c_i / (c_i + K_half). This is a quasi-steady-
   state approximation of a reaction–diffusion field (the Compass caveat:
   volume transmission is a field, not an edge) — spatial profile static,
   temporal kinetics from the release dynamics. λ = ∞ gives global broadcast.

3. TIMESCALES. Per-modulator release latents follow AR(1) with persistence
   ρ_c (slow peptide vs faster amine), optionally driven by the source
   neuron's own activity (release_gain > 0 closes the x→m loop à la
   Kringelbach 2020; default 0 = open loop, the harder inference claim).

4. NONLINEARITY. phi="tanh" gives saturating rate dynamics
   x_{t+1} = tanh(A(t) x_t) + noise — the standard biologically-sensible
   bounded rate model. The true Jacobian remains exact:
   J(t) = diag(1 − tanh²(u_t)) · A(t), u_t = A(t) x_t.

How modulators enter A(t), following the Compass taxonomy:
   additive (H2):  A += α_c · diag(occ_c) · W_c        (extrasynaptic edges,
                   gated at the TARGET by local receptor occupancy)
   gain (H1):      row i of A scaled by (1 + α_c·occ_c,i) for receptor-
                   bearing targets (input gain, Chance–Abbott–Reyes 2002)

Methods under test still receive only `trial.observed()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Phi = Literal["linear", "tanh"]
Mode = Literal["additive", "gain"]


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------

@dataclass
class Modulator:
    """One neuromodulator channel.

    additive mode: `W` is the signed receptor connectome layer
        W[target, source] ≠ 0 means "source releases onto target's receptor";
        sign encodes receptor polarity (e.g. inhibitory MOD-1 vs excitatory
        SER-4 style channels, Dag 2023).
    gain mode: `receptors` lists the neurons whose input gain this modulator
        scales; `W` is unused.
    """

    name: str
    mode: Mode = "additive"
    W: np.ndarray | None = None
    receptors: tuple[int, ...] = ()
    alpha: float = 0.6              # coupling strength
    rho: float = 0.98               # release persistence (timescale)
    sigma: float = 1.0              # release latent stationary std
    sources: tuple[int, ...] = ()   # releasing neurons; () = one global source
    decay_length: float = np.inf    # spatial reach λ (∞ = broadcast)
    K_half: float = 1.0             # Michaelis constant for occupancy
    release_gain: float = 0.0       # activity-dependent release (0 = open loop)

    def receptor_mask(self, n: int) -> np.ndarray:
        """Boolean (n,): which target neurons carry this receptor."""
        if self.mode == "additive":
            assert self.W is not None, f"{self.name}: additive mode needs W"
            return np.abs(self.W).sum(axis=1) > 0
        mask = np.zeros(n, dtype=bool)
        mask[list(self.receptors)] = True
        return mask


def receptor_layer(n: int, edges: Sequence[tuple[int, int, float]]) -> np.ndarray:
    """Build a receptor connectome layer from (source, target, signed_weight)."""
    W = np.zeros((n, n))
    for source, target, weight in edges:
        W[target, source] = weight
    return W


def line_positions(n: int, spacing: float = 1.0) -> np.ndarray:
    return np.column_stack([np.arange(n) * spacing, np.zeros(n)])


# ---------------------------------------------------------------------------
# trial
# ---------------------------------------------------------------------------

@dataclass
class MultiTrial:
    x: np.ndarray                      # (T, N) observed activity
    occ: np.ndarray                    # (T, k, N) receptor occupancy per modulator
    latents: list[np.ndarray]          # per modulator: (T, n_sources) release latents
    J: np.ndarray | None               # (T-1, N, N) exact Jacobians (None if not stored)
    A0: np.ndarray
    modulators: list[Modulator]
    positions: np.ndarray
    phi: Phi
    meta: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return self.x.shape[1]

    @property
    def n_times(self) -> int:
        return self.x.shape[0]

    @property
    def n_modulators(self) -> int:
        return len(self.modulators)

    def observed(self) -> np.ndarray:
        """Activity only — the method's data. occ/latents/J are the answer key."""
        return self.x

    @property
    def m_summary(self) -> np.ndarray:
        """(T, k) scalar state per modulator: mean occupancy on its receptor targets."""
        n = self.n_neurons
        out = np.empty((self.n_times, self.n_modulators))
        for c, mod in enumerate(self.modulators):
            mask = mod.receptor_mask(n)
            out[:, c] = self.occ[:, c, mask].mean(axis=1)
        return out

    def operator_at(self, t: int) -> np.ndarray:
        return _build_A(self.A0, self.modulators, self.occ[t])

    def jacobian_at(self, t: int) -> np.ndarray:
        A = self.operator_at(t)
        if self.phi == "linear":
            return A
        u = A @ self.x[t]
        return (1.0 - np.tanh(u) ** 2)[:, None] * A


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def _softplus(v: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, v)


def _build_A(A0: np.ndarray, modulators: list[Modulator], occ_t: np.ndarray) -> np.ndarray:
    """occ_t: (k, N) occupancy at one time step. Returns A(t)."""
    A = A0.copy()
    for c, mod in enumerate(modulators):
        if mod.mode == "additive":
            A += mod.alpha * occ_t[c][:, None] * mod.W
    gain = np.ones(A0.shape[0])
    for c, mod in enumerate(modulators):
        if mod.mode == "gain":
            mask = mod.receptor_mask(A0.shape[0])
            gain = gain * np.where(mask, 1.0 + mod.alpha * occ_t[c], 1.0)
    return gain[:, None] * A


def _kernels(positions: np.ndarray, modulators: list[Modulator]) -> list[np.ndarray]:
    """Per modulator: (n_sources, N) spatial kernel exp(-d/λ) from each source."""
    out = []
    for mod in modulators:
        sources = list(mod.sources) if mod.sources else [None]
        rows = []
        for s in sources:
            if s is None or not np.isfinite(mod.decay_length):
                rows.append(np.ones(positions.shape[0]))
            else:
                d = np.linalg.norm(positions - positions[s], axis=1)
                rows.append(np.exp(-d / mod.decay_length))
        out.append(np.stack(rows))
    return out


def _assert_stable_linear(A0, modulators, limit=0.98) -> None:
    """Conservative: spectral radius of |A| with every occupancy at 1."""
    n = A0.shape[0]
    A_abs = np.abs(A0)
    gain = np.ones(n)
    for mod in modulators:
        if mod.mode == "additive":
            A_abs = A_abs + abs(mod.alpha) * np.abs(mod.W)
        else:
            mask = mod.receptor_mask(n)
            gain = gain * np.where(mask, 1.0 + abs(mod.alpha), 1.0)
    r = float(np.max(np.abs(np.linalg.eigvals(gain[:, None] * A_abs))))
    if r >= limit:
        raise ValueError(f"linear generator may be unstable: worst-case radius {r:.3f}; "
                         "lower alphas/weights or use phi='tanh'")


def simulate_multi(
    A0: np.ndarray,
    modulators: list[Modulator],
    *,
    T: int = 3000,
    positions: np.ndarray | None = None,
    phi: Phi = "tanh",
    sigma_x: float = 0.05,
    obs_noise: float = 0.0,
    store_J: bool = True,
    seed: int = 0,
) -> MultiTrial:
    """Simulate one multi-modulator trial. Methods may use `.observed()` only."""
    rng = np.random.default_rng(seed)
    n = A0.shape[0]
    k = len(modulators)
    if positions is None:
        positions = line_positions(n)
    for mod in modulators:
        if mod.mode == "additive" and (mod.W is None or mod.W.shape != (n, n)):
            raise ValueError(f"{mod.name}: additive modulator needs W of shape ({n},{n})")
    if phi == "linear":
        _assert_stable_linear(A0, modulators)

    kernels = _kernels(positions, modulators)
    latents = [np.zeros((T, kern.shape[0])) for kern in kernels]
    occ = np.zeros((T, k, n))
    x = np.zeros((T, n))
    x[0] = 0.1 * rng.normal(size=n)
    J = np.empty((T - 1, n, n)) if store_J else None

    def occupancy(c: int, lat_row: np.ndarray) -> np.ndarray:
        conc = _softplus(lat_row) @ kernels[c]           # (N,) concentration
        return conc / (conc + modulators[c].K_half)      # Michaelis–Menten

    for c in range(k):
        latents[c][0] = modulators[c].sigma * rng.normal(size=latents[c].shape[1])
        occ[0, c] = occupancy(c, latents[c][0])

    for t in range(T - 1):
        A = _build_A(A0, modulators, occ[t])
        u = A @ x[t]
        drive = u if phi == "linear" else np.tanh(u)
        x[t + 1] = drive + sigma_x * rng.normal(size=n)
        if store_J:
            J[t] = A if phi == "linear" else (1.0 - np.tanh(u) ** 2)[:, None] * A
        for c, mod in enumerate(modulators):
            lat = latents[c][t]
            innov = mod.sigma * np.sqrt(max(1.0 - mod.rho ** 2, 1e-12)) \
                * rng.normal(size=lat.shape)
            drive_m = 0.0
            if mod.release_gain > 0 and mod.sources:
                drive_m = mod.release_gain * np.maximum(x[t][list(mod.sources)], 0.0)
            latents[c][t + 1] = mod.rho * lat + innov + drive_m
            occ[t + 1, c] = occupancy(c, latents[c][t + 1])

    if obs_noise > 0:
        x = x + obs_noise * rng.normal(size=x.shape)

    return MultiTrial(
        x=x, occ=occ, latents=latents, J=J, A0=A0, modulators=modulators,
        positions=positions, phi=phi,
        meta={"sigma_x": sigma_x, "obs_noise": obs_noise, "seed": seed, "T": T},
    )


# ---------------------------------------------------------------------------
# scoring (multi-modulator versions of the metrics)
# ---------------------------------------------------------------------------

def score_multi(trial: MultiTrial, J_hat: np.ndarray, name: str) -> dict:
    """Score an estimated J(t) sequence against the multi-modulator answer key.

    - recovery_r2[c]: how much of modulator c's hidden state is recoverable
      from the top-k principal components of vec(J_hat(t)) (blind readout,
      then regression against the true m_c — truth used for scoring only).
    - layer_auroc[c]: for additive modulators, does |J_hat high-m − low-m|
      light up that modulator's true receptor edges?
    - jacobian_cosine / one_step_mse: sanity metrics as in the simple testbed.
    """
    from .metrics import auroc, mean_cosine, one_step_mse, valid_times

    Tm1 = trial.n_times - 1
    if J_hat.shape != (Tm1, trial.n_neurons, trial.n_neurons):
        raise ValueError(f"J_hat shape {J_hat.shape} != {(Tm1, trial.n_neurons, trial.n_neurons)}")
    mask = valid_times(J_hat)
    m = trial.m_summary[:Tm1]
    k = trial.n_modulators

    # blind low-dim readout of the estimated coupling trajectory
    idx = np.flatnonzero(mask)
    V = J_hat[idx].reshape(idx.size, -1)
    V = V - V.mean(axis=0, keepdims=True)
    r2 = np.zeros(k)
    if idx.size > 10 and np.linalg.norm(V) > 1e-10:
        n_comp = min(max(k, 2), V.shape[0] - 1, V.shape[1])
        _, _, vt = np.linalg.svd(V, full_matrices=False)
        comps = V @ vt[:n_comp].T                         # (n_valid, n_comp)
        X = np.column_stack([comps, np.ones(len(comps))])
        for c in range(k):
            y = m[idx, c]
            if y.std() < 1e-12:
                continue
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            r2[c] = max(0.0, 1.0 - resid.var() / y.var())

    layer_auroc = np.full(k, np.nan)
    off = ~np.eye(trial.n_neurons, dtype=bool)
    for c, mod in enumerate(trial.modulators):
        if mod.mode != "additive" or mask.sum() < 20:
            continue
        mc = m[:, c]
        lo = (mc <= np.quantile(mc[mask], 0.25)) & mask
        hi = (mc >= np.quantile(mc[mask], 0.75)) & mask
        if lo.sum() < 5 or hi.sum() < 5:
            continue
        delta = np.abs(np.nanmean(J_hat[hi], axis=0) - np.nanmean(J_hat[lo], axis=0))
        labels = np.abs(mod.W) > 1e-9
        layer_auroc[c] = auroc(delta[off], labels[off])

    return {
        "name": name,
        "recovery_r2": r2,
        "layer_auroc": layer_auroc,
        "jacobian_cosine": mean_cosine(J_hat, trial.J, mask) if trial.J is not None else float("nan"),
        "one_step_mse": one_step_mse(J_hat, trial.x, mask),
        "n_valid": int(mask.sum()),
    }

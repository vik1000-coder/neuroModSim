"""Minimal two-timescale generator used as a test instrument.

The equations are the Compass toy, kept linear so the Jacobian is exact:

    x_{t+1} = A(m_t) x_t + σ_x ε_t
    m_{t+1} = ρ m_t + σ_m η_t

A(m) is one of three mechanisms, not a new scientific model:

    none      A(m) = A0                         (slow m exists but does nothing)
    gain      A(m) = (1 + α tanh(m)) A0         (H1: diagonal/global gain)
    additive  A(m) = A0 + α tanh(m) A1          (H2: extra extrasynaptic layer)

Methods under test receive only x. m, A(m), and the lagged operators A(m)^k
are evaluation-only ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

Mechanism = Literal["none", "gain", "additive"]
Topology = Literal["chain", "random"]


@dataclass
class Trial:
    """One synthetic rollout.

    `x` is the only object a method is allowed to see. Everything else exists
    so we can score whether the method recovered the hidden modulator's effect.
    """

    x: np.ndarray  # (T, N)
    m: np.ndarray  # (T,)
    J: np.ndarray  # (T-1, N, N)  J[t, target, source] used in x[t] -> x[t+1]
    A0: np.ndarray
    A1: np.ndarray
    mechanism: Mechanism
    alpha: float
    rho: float
    meta: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return self.x.shape[1]

    @property
    def n_times(self) -> int:
        return self.x.shape[0]

    def observed(self) -> np.ndarray:
        """Activity only. This is the method's data."""
        return self.x

    def A_of(self, m_value: float) -> np.ndarray:
        return mix_operator(self.A0, self.A1, self.mechanism, self.alpha, m_value)

    def frozen_lagged(self, m_value: float, max_lag: int) -> np.ndarray:
        """A(m)^k for k=1..max_lag. Slow-m approximation of delayed influence."""
        A = self.A_of(m_value)
        out = np.empty((max_lag, self.n_neurons, self.n_neurons), dtype=float)
        P = np.eye(self.n_neurons)
        for k in range(max_lag):
            P = A @ P
            out[k] = P
        return out

    def pulse_response(self, t0: int, pulse: float, horizon: int, rng_seed: int = 0) -> np.ndarray:
        """Twin-trajectory causal effect of a hidden-modulator pulse.

        Two rollouts share x[t0] and the same process noise after t0. The
        treated rollout adds `pulse` to m at t0 only. Returns dx[1..horizon].
        """
        rng = np.random.default_rng(rng_seed)
        n = self.n_neurons
        noise = rng.normal(size=(horizon, n)) * float(self.meta.get("sigma_x", 0.0))
        x0 = self.x[t0].copy()
        m0 = float(self.m[t0])
        ctrl = _rollout_from(x0, m0, 0.0, self, noise)
        trea = _rollout_from(x0, m0, pulse, self, noise)
        return trea - ctrl


def mix_operator(A0: np.ndarray, A1: np.ndarray, mechanism: Mechanism, alpha: float, m_value: float) -> np.ndarray:
    u = np.tanh(m_value)
    if mechanism == "none":
        return A0.copy()
    if mechanism == "gain":
        return (1.0 + alpha * u) * A0
    if mechanism == "additive":
        return A0 + alpha * u * A1
    raise ValueError(f"unknown mechanism: {mechanism}")


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def scale_radius(A: np.ndarray, radius: float) -> np.ndarray:
    r = spectral_radius(A)
    if r < 1e-12:
        return A.copy()
    return A * (radius / r)


def chain_operator(n: int, hop: float = 0.7) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i + 1, i] = hop
    return A


def random_operator(n: int, density: float, radius: float, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n)) * (rng.random((n, n)) < density)
    np.fill_diagonal(A, 0.0)
    return scale_radius(A, radius)


def extra_layer(n: int, edges: Sequence[tuple[int, int]], weight: float = 0.5) -> np.ndarray:
    A1 = np.zeros((n, n))
    for source, target in edges:
        A1[target, source] = weight
    return A1


def default_skip_edges(n: int) -> tuple[tuple[int, int], ...]:
    """Long-range extrasynaptic shortcuts on a chain, distinct from one-hop A0."""
    if n < 4:
        raise ValueError("chain extrasynaptic layer needs n >= 4")
    edges = [(0, n // 2), (1, n - 1)]
    if n >= 6:
        edges.append((2, n - 2))
    return tuple(edges)


def _rollout_from(x0: np.ndarray, m0: float, pulse: float, trial: Trial, noise: np.ndarray) -> np.ndarray:
    horizon = noise.shape[0]
    x = np.empty((horizon + 1, trial.n_neurons))
    x[0] = x0
    m = m0 + pulse
    # Same process noise; m evolves autonomously after the pulse (no m-noise),
    # so the difference is the causal effect of the hidden pulse on x.
    for t in range(horizon):
        A = trial.A_of(m)
        x[t + 1] = A @ x[t] + noise[t]
        m = trial.rho * m
    return x[1:]


def simulate(
    *,
    n: int = 8,
    T: int = 4000,
    mechanism: Mechanism = "additive",
    topology: Topology = "chain",
    rho: float = 0.98,
    sigma_x: float = 0.05,
    sigma_m: float = 0.20,
    alpha: float = 0.8,
    hop: float = 0.7,
    density: float = 0.25,
    radius: float = 0.7,
    extra_edges: Iterable[tuple[int, int]] | None = None,
    obs_noise: float = 0.0,
    seed: int = 0,
) -> Trial:
    """Simulate one trial. Methods may use `trial.observed()`, nothing else."""
    rng = np.random.default_rng(seed)
    if topology == "chain":
        A0 = chain_operator(n, hop=hop)
        edges = tuple(extra_edges) if extra_edges is not None else default_skip_edges(n)
        A1 = extra_layer(n, edges, weight=0.5)
    elif topology == "random":
        A0 = random_operator(n, density=density, radius=radius, rng=rng)
        A1 = random_operator(n, density=min(0.12, density), radius=0.25, rng=rng)
        overlap = (np.abs(A0) > 1e-12) & (np.abs(A1) > 1e-12)
        A1[overlap] = 0.0
        if spectral_radius(A1) > 1e-12:
            A1 = scale_radius(A1, 0.25)
    else:
        raise ValueError(f"unknown topology: {topology}")

    _assert_stable(A0, A1, mechanism, alpha)

    m = np.zeros(T)
    for t in range(T - 1):
        m[t + 1] = rho * m[t] + sigma_m * rng.normal()

    x = np.zeros((T, n))
    x[0] = rng.normal(size=n) * 0.1
    J = np.empty((T - 1, n, n))
    for t in range(T - 1):
        A = mix_operator(A0, A1, mechanism, alpha, float(m[t]))
        J[t] = A
        x[t + 1] = A @ x[t] + sigma_x * rng.normal(size=n)

    if obs_noise > 0:
        x = x + obs_noise * rng.normal(size=x.shape)

    return Trial(
        x=x,
        m=m,
        J=J,
        A0=A0,
        A1=A1,
        mechanism=mechanism,
        alpha=alpha,
        rho=rho,
        meta={
            "sigma_x": sigma_x,
            "sigma_m": sigma_m,
            "obs_noise": obs_noise,
            "topology": topology,
            "seed": seed,
            "extra_edges": None if topology != "chain" else (tuple(extra_edges) if extra_edges is not None else default_skip_edges(n)),
        },
    )


def _assert_stable(A0: np.ndarray, A1: np.ndarray, mechanism: Mechanism, alpha: float, limit: float = 0.98) -> None:
    for m_value in (-1.5, 0.0, 1.5):
        r = spectral_radius(mix_operator(A0, A1, mechanism, alpha, m_value))
        if r >= limit:
            raise ValueError(
                f"unstable generator: radius {r:.3f} at m={m_value} for {mechanism}; "
                "lower alpha or hop"
            )

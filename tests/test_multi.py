from __future__ import annotations

import numpy as np

from neuromodsim.methods import SlidingVAR, StaticVAR
from neuromodsim.multi import (
    Modulator,
    line_positions,
    receptor_layer,
    score_multi,
    simulate_multi,
)


def _chain(n: int, w: float = 0.5) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i + 1, i] = w
    return A


def _two_modulators(n: int) -> list[Modulator]:
    slow = Modulator(
        name="5HT-like", mode="additive",
        W=receptor_layer(n, [(1, 6, 0.5), (2, 8, 0.5), (0, 9, -0.4),
                             (1, 4, 0.5), (0, 7, 0.5)]),
        alpha=0.9, rho=0.995, sigma=1.2,
        sources=(1,), decay_length=25.0, K_half=1.0,
    )
    fast = Modulator(
        name="amine-like", mode="gain",
        receptors=tuple(range(n // 2)),
        alpha=0.5, rho=0.90, sigma=1.2,
        sources=(), decay_length=np.inf,
    )
    return [slow, fast]


def test_exact_jacobian_linear_and_tanh():
    n = 10
    for phi in ("linear", "tanh"):
        trial = simulate_multi(_chain(n, 0.45), _two_modulators(n), T=300,
                               phi=phi, sigma_x=0.04, seed=1)
        # J must match finite differences of the deterministic step at several t
        for t in (50, 150, 250):
            A = trial.operator_at(t)
            f = (lambda v: A @ v) if phi == "linear" else (lambda v: np.tanh(A @ v))
            eps = 1e-6
            J_fd = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n); e[j] = eps
                J_fd[:, j] = (f(trial.x[t] + e) - f(trial.x[t] - e)) / (2 * eps)
            assert np.allclose(trial.J[t], J_fd, atol=1e-5)
            assert np.allclose(trial.jacobian_at(t), trial.J[t], atol=1e-12)


def test_residual_matches_noise():
    n = 10
    trial = simulate_multi(_chain(n, 0.45), _two_modulators(n), T=1500,
                           phi="tanh", sigma_x=0.05, seed=0)
    pred = np.stack([np.tanh(trial.operator_at(t) @ trial.x[t])
                     for t in range(trial.n_times - 1)])
    resid = trial.x[1:] - pred
    assert abs(resid.std() - 0.05) < 0.005


def test_spatial_locality():
    """A short decay length must confine occupancy variation near the source."""
    n = 30
    W = receptor_layer(n, [(0, 5, 0.5), (0, 29, 0.5)])  # receptors near and far
    mod = Modulator(name="local", mode="additive", W=W, alpha=0.8,
                    rho=0.97, sigma=1.5, sources=(0,), decay_length=3.0)
    trial = simulate_multi(_chain(n, 0.4), [mod], T=1200, phi="tanh", seed=0,
                           positions=line_positions(n))
    var_near = trial.occ[:, 0, 5].var()
    var_far = trial.occ[:, 0, 29].var()
    assert var_near > 20 * var_far


def test_timescale_ordering():
    """Slow modulator occupancy must be more autocorrelated than the fast one."""
    n = 10
    trial = simulate_multi(_chain(n, 0.45), _two_modulators(n), T=3000,
                           phi="tanh", seed=0)
    m = trial.m_summary
    def ac1(v):
        v = v - v.mean()
        return float(np.dot(v[:-1], v[1:]) / (np.dot(v, v) + 1e-12))
    assert ac1(m[:, 0]) > ac1(m[:, 1]) > 0.5


def test_gain_mode_adds_no_edges():
    n = 10
    gain_only = [Modulator(name="g", mode="gain", receptors=(2, 3, 4),
                           alpha=0.5, rho=0.95, sigma=1.0)]
    trial = simulate_multi(_chain(n, 0.5), gain_only, T=200, phi="linear", seed=0)
    support0 = np.abs(trial.A0) > 1e-12
    for t in (10, 100, 190):
        assert (np.abs(trial.jacobian_at(t)) > 1e-12 ).sum() == support0.sum()


def test_oracle_recovers_both_modulators_static_does_not():
    n = 12
    trial = simulate_multi(_chain(n, 0.45), _two_modulators(n), T=2500,
                           phi="tanh", sigma_x=0.04, seed=0)
    oracle = score_multi(trial, trial.J.copy(), "oracle")
    assert oracle["recovery_r2"].min() > 0.7, oracle["recovery_r2"]
    assert oracle["layer_auroc"][0] > 0.9  # additive layer found

    static = StaticVAR().fit_jacobians(trial.observed())
    s = score_multi(trial, static, "static")
    assert s["recovery_r2"].max() < 1e-6


def test_sliding_var_tracks_slow_modulator_in_nonlinear_system():
    n = 12
    trial = simulate_multi(_chain(n, 0.45), _two_modulators(n), T=4000,
                           phi="tanh", sigma_x=0.04, seed=0)
    J_hat = SlidingVAR(window=200, lam=1e-2).fit_jacobians(trial.observed())
    s = score_multi(trial, J_hat, "sliding")
    assert s["recovery_r2"][0] > 0.25   # slow additive modulator is trackable

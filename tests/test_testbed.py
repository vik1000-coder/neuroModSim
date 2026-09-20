from __future__ import annotations

import numpy as np
from neuromodsim.methods import OracleJacobian, SlidingVAR, StaticVAR
from neuromodsim.metrics import evaluate, reconstruct_modulator, valid_times
from neuromodsim.process import simulate, spectral_radius
from neuromodsim.scenarios import make_scenario


def test_generator_stays_stable():
    trial = simulate(n=8, T=200, mechanism="additive", seed=1)
    assert spectral_radius(trial.A0) < 1
    assert np.isfinite(trial.x).all()
    assert trial.J.shape == (199, 8, 8)


def test_methods_receive_only_activity():
    trial = make_scenario("additive", T=400, seed=0)
    x = trial.observed()
    J_hat = SlidingVAR(window=80).fit_jacobians(x)
    assert J_hat.shape == trial.J.shape
    # The estimator API has no slot for m; this is the contract the testbed enforces.
    assert x.shape == trial.x.shape


def test_oracle_tracks_hidden_modulator_when_it_matters():
    trial = make_scenario("additive", T=1500, seed=0, sigma_x=0.02)
    score = evaluate(trial, OracleJacobian(trial))
    assert score.jacobian_cosine > 0.999
    assert score.modulator_corr > 0.95
    assert score.lag_modulator_corr[0] > 0.9
    assert score.support_auroc > 0.95


def test_static_var_cannot_track_modulator():
    trial = make_scenario("additive", T=1500, seed=0, sigma_x=0.02)
    score = evaluate(trial, StaticVAR())
    assert abs(score.modulator_corr) < 1e-8


def test_sliding_var_tracks_slow_additive_better_than_static():
    trial = make_scenario("additive", T=2000, seed=0, sigma_x=0.02, rho=0.985)
    sliding = evaluate(trial, SlidingVAR(window=120, lam=1e-2))
    static = evaluate(trial, StaticVAR())
    assert sliding.modulator_corr > 0.45
    assert sliding.modulator_corr > static.modulator_corr + 0.4
    assert sliding.support_auroc > 0.9
    assert sliding.support_auroc > static.support_auroc + 0.3


def test_null_modulator_is_not_recoverable():
    trial = make_scenario("none", T=2000, seed=0, sigma_x=0.02)
    oracle = evaluate(trial, OracleJacobian(trial))
    sliding = evaluate(trial, SlidingVAR(window=120))
    assert abs(oracle.modulator_corr) < 1e-8
    assert abs(sliding.modulator_corr) < 0.35


def test_gain_oracle_tracks_and_has_no_extra_layer():
    trial = make_scenario("gain", T=1200, seed=0, sigma_x=0.02)
    score = evaluate(trial, OracleJacobian(trial))
    assert score.modulator_corr > 0.95
    assert np.isnan(score.support_auroc)


def test_pulse_response_is_real_for_additive():
    trial = make_scenario("additive", T=800, seed=0)
    t0 = int(np.argmax(np.linalg.norm(trial.x, axis=1)))
    dx = trial.pulse_response(t0=t0, pulse=1.5, horizon=6, rng_seed=0)
    assert dx.shape == (6, trial.n_neurons)
    assert np.linalg.norm(dx) > 1e-6
    targets = [tgt for _, tgt in trial.meta["extra_edges"]]
    assert np.linalg.norm(dx[0, targets]) > 0.0


def test_reconstructed_modulator_uses_only_estimated_J():
    trial = make_scenario("additive", T=600, seed=0)
    J_hat = OracleJacobian(trial).fit_jacobians(trial.observed())
    mask = valid_times(J_hat)
    m_hat = reconstruct_modulator(J_hat, mask)
    assert m_hat.shape[0] == trial.J.shape[0]
    assert np.isfinite(m_hat[mask]).all()

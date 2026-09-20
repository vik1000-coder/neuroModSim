"""Scores for the unobserved-modulator claim.

Primary questions, in order:

1. Does the estimated Jacobian track the true A(m_t)?
2. Can a 1-D readout of that Jacobian reconstruct the hidden m_t?
3. Do modulator-gated *delayed* influences (A(m)^k) track m, not just lag-1?
4. On the null (m present, A fixed), does the method stay quiet?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .methods import Estimator
from .process import Trial


@dataclass
class Score:
    name: str
    jacobian_cosine: float
    modulator_corr: float
    lag_modulator_corr: np.ndarray  # (max_lag,) mean |corr| over changing edges
    lag_edge_corr: np.ndarray  # (max_lag, N, N)
    support_auroc: float
    one_step_mse: float
    n_valid: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "jacobian_cosine": self.jacobian_cosine,
            "modulator_corr": self.modulator_corr,
            "lag1_modulator_corr": float(self.lag_modulator_corr[0]) if len(self.lag_modulator_corr) else float("nan"),
            "lag_modulator_corr": self.lag_modulator_corr.tolist(),
            "support_auroc": self.support_auroc,
            "one_step_mse": self.one_step_mse,
            "n_valid": self.n_valid,
        }


def valid_times(J_hat: np.ndarray) -> np.ndarray:
    return ~np.isnan(J_hat).reshape(J_hat.shape[0], -1).any(axis=1)


def signed_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def mean_cosine(J_hat: np.ndarray, J_true: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for t in np.flatnonzero(mask):
        a = J_hat[t].ravel()
        b = J_true[t].ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            continue
        vals.append(float(np.dot(a, b) / (na * nb)))
    return float(np.mean(vals)) if vals else 0.0


def reconstruct_modulator(J_hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blind 1-D readout: leading PC of vec(J_t - mean). Sign is arbitrary."""
    T = len(mask)
    m_hat = np.full(T, np.nan)
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return m_hat
    V = J_hat[idx].reshape(idx.size, -1)
    V = V - V.mean(axis=0, keepdims=True)
    if np.linalg.norm(V) < 1e-12:
        m_hat[idx] = 0.0
        return m_hat
    _, _, vt = np.linalg.svd(V, full_matrices=False)
    m_hat[idx] = V @ vt[0]
    return m_hat


def align_to(m_hat: np.ndarray, m: np.ndarray) -> np.ndarray:
    mask = np.isfinite(m_hat)
    if signed_pearson(m_hat[mask], m[mask]) < 0:
        return -m_hat
    return m_hat


def lagged_operators(J: np.ndarray, max_lag: int, mask: np.ndarray) -> np.ndarray:
    """Frozen-A delayed influence: P[t, k] = J[t]^{k+1}. Slow-m approximation."""
    T, n, _ = J.shape
    out = np.full((T, max_lag, n, n), np.nan)
    for t in np.flatnonzero(mask):
        P = np.eye(n)
        A = J[t]
        for k in range(max_lag):
            P = A @ P
            out[t, k] = P
    return out


def lag_edge_correlations(J: np.ndarray, m: np.ndarray, max_lag: int, mask: np.ndarray) -> np.ndarray:
    T, n, _ = J.shape
    P = lagged_operators(J, max_lag, mask)
    corr = np.zeros((max_lag, n, n))
    m_use = m[:T][mask]
    for k in range(max_lag):
        for i in range(n):
            for j in range(n):
                series = P[mask, k, j, i]
                corr[k, j, i] = signed_pearson(series, m_use)
    return corr


def changing_edge_mask(trial: Trial, eps: float = 1e-8) -> np.ndarray:
    """Edges whose true operator actually moves with m."""
    A_lo = trial.A_of(-1.0)
    A_hi = trial.A_of(1.0)
    return np.abs(A_hi - A_lo) > eps


def mean_abs_corr_on_changing_edges(lag_corr: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    if not np.any(edge_mask):
        return np.zeros(lag_corr.shape[0])
    return np.array([np.mean(np.abs(lag_corr[k][edge_mask])) for k in range(lag_corr.shape[0])])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def support_auroc(J_hat: np.ndarray, m: np.ndarray, A1: np.ndarray, mask: np.ndarray) -> float:
    """Does |J_high m - J_low m| pick out the extrasynaptic layer?"""
    if mask.sum() < 20:
        return float("nan")
    m_use = m[: len(mask)]
    lo = m_use <= np.quantile(m_use[mask], 0.25)
    hi = m_use >= np.quantile(m_use[mask], 0.75)
    lo &= mask
    hi &= mask
    if lo.sum() < 5 or hi.sum() < 5:
        return float("nan")
    delta = np.abs(np.nanmean(J_hat[hi], axis=0) - np.nanmean(J_hat[lo], axis=0))
    labels = np.abs(A1) > 1e-8
    np.fill_diagonal(labels, False)
    off = ~np.eye(A1.shape[0], dtype=bool)
    return auroc(delta[off], labels[off])


def one_step_mse(J_hat: np.ndarray, x: np.ndarray, mask: np.ndarray) -> float:
    err = []
    for t in np.flatnonzero(mask):
        pred = J_hat[t] @ x[t]
        err.append(np.mean((pred - x[t + 1]) ** 2))
    return float(np.mean(err)) if err else float("nan")


def score_estimator(trial: Trial, J_hat: np.ndarray, name: str, max_lag: int = 4) -> Score:
    Tm1 = trial.J.shape[0]
    if J_hat.shape != trial.J.shape:
        raise ValueError(f"J_hat shape {J_hat.shape} != true J {trial.J.shape}")
    mask = valid_times(J_hat)
    m = trial.m[:Tm1]
    m_hat = align_to(reconstruct_modulator(J_hat, mask), m)
    lag_corr = lag_edge_correlations(J_hat, m, max_lag, mask)
    changing = changing_edge_mask(trial)
    return Score(
        name=name,
        jacobian_cosine=mean_cosine(J_hat, trial.J, mask),
        modulator_corr=signed_pearson(m_hat[mask], m[mask]),
        lag_modulator_corr=mean_abs_corr_on_changing_edges(lag_corr, changing),
        lag_edge_corr=lag_corr,
        support_auroc=support_auroc(J_hat, m, trial.A1, mask) if trial.mechanism == "additive" else float("nan"),
        one_step_mse=one_step_mse(J_hat, trial.x, mask),
        n_valid=int(mask.sum()),
    )


def evaluate(trial: Trial, estimator: Estimator, max_lag: int = 4) -> Score:
    x = trial.observed()
    J_hat = estimator.fit_jacobians(x)
    return score_estimator(trial, J_hat, getattr(estimator, "name", estimator.__class__.__name__), max_lag=max_lag)

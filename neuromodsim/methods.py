"""Estimators that see only activity.

A method under test must implement `fit_jacobians(x) -> (T-1, N, N)` and must
not receive m. The baselines here are not the scientific contribution; they
exist so the testbed can tell a method that tracks the hidden modulator from
one that cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .process import Trial


class Estimator(Protocol):
    name: str

    def fit_jacobians(self, x: np.ndarray) -> np.ndarray:
        """Estimate J[t] for the transition x[t] -> x[t+1]. Shape (T-1, N, N).

        May put NaN in burn-in rows. Must not use the hidden modulator.
        """


def ridge_var(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Y ≈ X A^T with A[target, source], i.e. y = A x.

    X, Y have shape (W, N). Returns A of shape (N, N).
    """
    n = X.shape[1]
    gram = X.T @ X + lam * np.eye(n)
    return np.linalg.solve(gram, X.T @ Y).T


@dataclass
class StaticVAR:
    """One operator for the whole series. Cannot track a moving modulator."""

    lam: float = 1e-3
    name: str = "static_var"

    def fit_jacobians(self, x: np.ndarray) -> np.ndarray:
        X, Y = x[:-1], x[1:]
        A = ridge_var(X, Y, self.lam)
        return np.repeat(A[None, :, :], len(X), axis=0)


@dataclass
class SlidingVAR:
    """Causal local linearization: the weakest method that *can* track slow m."""

    window: int = 120
    lam: float = 1e-2
    name: str = "sliding_var"

    def __post_init__(self) -> None:
        self.name = f"sliding_var(w={self.window})"

    def fit_jacobians(self, x: np.ndarray) -> np.ndarray:
        T, n = x.shape
        J = np.full((T - 1, n, n), np.nan)
        w = self.window
        if w < n + 2:
            raise ValueError(f"window {w} too short for n={n}")
        for t in range(w, T - 1):
            X = x[t - w : t]
            Y = x[t - w + 1 : t + 1]
            J[t] = ridge_var(X, Y, self.lam)
        return J


@dataclass
class OracleJacobian:
    """Cheating ceiling: the true A(m_t). Not a method, a score upper bound."""

    trial: Trial
    name: str = "oracle"

    def fit_jacobians(self, x: np.ndarray) -> np.ndarray:
        if x.shape != self.trial.x.shape or not np.allclose(x, self.trial.x, atol=1e-12):
            raise ValueError("oracle was handed a different activity series")
        return self.trial.J.copy()

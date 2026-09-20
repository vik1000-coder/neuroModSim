"""Synthetic testbed for unobserved neuromodulator effects.

This package is not a proposed model of a nervous system. It is a scoring
instrument: generate activity from a known hidden modulator, hide that
modulator, and ask whether a method that sees only activity recovers the
modulator's delayed effects.
"""

from .metrics import evaluate, score_estimator
from .methods import OracleJacobian, SlidingVAR, StaticVAR
from .multi import Modulator, MultiTrial, receptor_layer, score_multi, simulate_multi
from .presets import PRESETS, build_circuit, build_many_modulators, build_modulators
from .process import Trial, simulate
from .scenarios import SCENARIOS, make_scenario

__all__ = [
    "Modulator",
    "MultiTrial",
    "OracleJacobian",
    "PRESETS",
    "SCENARIOS",
    "SlidingVAR",
    "StaticVAR",
    "Trial",
    "build_circuit",
    "build_many_modulators",
    "build_modulators",
    "evaluate",
    "make_scenario",
    "receptor_layer",
    "score_estimator",
    "score_multi",
    "simulate",
    "simulate_multi",
]

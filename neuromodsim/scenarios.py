"""Named scenarios. These are knobs on the same linear instrument, not models."""

from __future__ import annotations

from typing import Any

from .process import Trial, simulate

SCENARIOS: dict[str, dict[str, Any]] = {
    "additive": {
        "mechanism": "additive",
        "topology": "chain",
        "alpha": 0.8,
        "rho": 0.98,
        "doc": "H2: slow m turns on extra long-range edges. Headline identifiability case.",
    },
    "gain": {
        "mechanism": "gain",
        "topology": "chain",
        "alpha": 0.25,
        "rho": 0.98,
        "doc": "H1: slow m rescales the existing circuit, no new edges.",
    },
    "none": {
        "mechanism": "none",
        "topology": "chain",
        "alpha": 0.0,
        "rho": 0.98,
        "doc": "Specificity control: the same slow m is generated but does not enter A.",
    },
    "fast_additive": {
        "mechanism": "additive",
        "topology": "chain",
        "alpha": 0.8,
        "rho": 0.5,
        "sigma_m": 0.5,
        "doc": "Timescale control: m mixes too fast for a long trailing window to track.",
    },
}


def make_scenario(name: str, **overrides) -> Trial:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    kw = dict(n=8, T=4000, sigma_x=0.05, sigma_m=0.20, seed=0)
    cfg = {k: v for k, v in SCENARIOS[name].items() if k != "doc"}
    kw.update(cfg)
    kw.update(overrides)
    trial = simulate(**kw)
    trial.meta["scenario"] = name
    return trial

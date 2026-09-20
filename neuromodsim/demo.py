"""Run the synthetic claim: can activity-only estimators track hidden m?"""

from __future__ import annotations

import argparse
from pathlib import Path

from .methods import OracleJacobian, SlidingVAR, StaticVAR
from .metrics import score_estimator
from .plot import plot_lag_propagation, plot_modulator_recovery
from .scenarios import SCENARIOS, make_scenario


def _fmt(v: float) -> str:
    if v != v:  # NaN
        return "   n/a"
    return f"{v:7.3f}"


def run(outdir: Path, seed: int = 0, T: int = 4000, window: int = 120) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    estimators_factory = lambda trial: [
        OracleJacobian(trial),
        StaticVAR(),
        SlidingVAR(window=window),
    ]
    print()
    print("Unobserved-modulator testbed")
    print("Methods see only x. m is used for scoring, never as input.")
    print()
    header = f"{'scenario':<16} {'method':<22} {'cos(J,J*)':>10} {'corr(m̂,m)':>10} {'lag1|r|':>10} {'A1 AUROC':>10} {'MSE':>10}"
    print(header)
    print("-" * len(header))
    for name in ("additive", "gain", "none", "fast_additive"):
        trial = make_scenario(name, seed=seed, T=T)
        fitted = {}
        lag_score = None
        for est in estimators_factory(trial):
            J_hat = est.fit_jacobians(trial.observed())
            score = score_estimator(trial, J_hat, est.name)
            fitted[est.name] = J_hat
            print(
                f"{name:<16} {est.name:<22} "
                f"{_fmt(score.jacobian_cosine)} {_fmt(score.modulator_corr)} "
                f"{_fmt(score.lag_modulator_corr[0] if len(score.lag_modulator_corr) else float('nan'))} "
                f"{_fmt(score.support_auroc)} {_fmt(score.one_step_mse)}"
            )
            if est.name.startswith("sliding_var") and name == "additive":
                lag_score = score
        plot_modulator_recovery(trial, fitted, outdir / f"{name}_modulator.png")
        if lag_score is not None:
            plot_lag_propagation(lag_score, trial, outdir / "additive_lag_propagation.png")
    print()
    print("How to read the table")
    print("  Oracle must track m (corr ≈ 1) whenever m actually enters A.")
    print("  Static VAR must fail to track m (corr ≈ 0); it has no time index.")
    print("  Sliding VAR is the weakest activity-only method that can succeed if m is slow.")
    print("  On 'none', everyone must stay quiet: a slow hidden series that does not")
    print("  change A is not recoverable from x, and claiming it would be a false positive.")
    print(f"  Figures: {outdir}")
    print()
    print("Scenario notes:")
    for key, cfg in SCENARIOS.items():
        print(f"  {key:<16} {cfg['doc']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("artifacts"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=4000)
    p.add_argument("--window", type=int, default=120)
    args = p.parse_args(argv)
    run(outdir=args.outdir, seed=args.seed, T=args.T, window=args.window)


if __name__ == "__main__":
    main()

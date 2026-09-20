"""Multi-modulator demo: N neurons, hidden modulators, linear vs tanh.

Circuit: a synaptic chain plus sparse local recurrent synapses (static).

Preset "two" (default, 24 neurons):
  serotonin   slow (rho=0.995), released by one neuron, short spatial reach
              (λ=8), ADDITIVE: gates extra receptor edges with mixed signs.
  dopamine    faster (rho=0.90), broadcast everywhere, GAIN: scales the
              input gain of the anterior half of the circuit (no new edges).

Preset "many" (12 neurons, four modulators, four timescales):
  neuropeptide  very slow (rho=0.9995), broadcast, additive long-range edges
  serotonin     slow (rho=0.997), local release (λ=2.5), additive
  dopamine      medium (rho=0.95), local release (λ=3), gain on back half
  octopamine    fast (rho=0.85), broadcast, gain on front quarter

(Names are evocative of the C. elegans monoamine/peptide systems — Bentley
2016, Ripoll-Sánchez 2023 — not quantitative fits to any specific pathway.
Swap in real receptor connectomes via `receptor_layer`.)

Methods still see only x. Scores: per-modulator recovery R² from a blind
PCA readout of the estimated J(t), per-layer edge AUROC, one-step MSE.
"""

from __future__ import annotations

import argparse

import numpy as np

from .methods import SlidingVAR, StaticVAR
from .multi import score_multi, simulate_multi
from .presets import PRESETS, build_circuit


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), default="two")
    p.add_argument("--n", type=int, default=None,
                   help="override the preset's neuron count")
    p.add_argument("--T", type=int, default=3000)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    default_n, builder = PRESETS[args.preset]
    n = args.n if args.n is not None else default_n
    A0 = build_circuit(n, args.seed)
    mods = builder(n, args.seed)
    print(f"\nMulti-modulator testbed: preset={args.preset}, n={n} neurons, "
          f"T={args.T}, {len(mods)} hidden modulators")
    for mod in mods:
        reach = "broadcast" if not np.isfinite(mod.decay_length) else f"λ={mod.decay_length:g}"
        n_edges = int(np.abs(mod.W).sum() > 0 and (np.abs(mod.W) > 0).sum()) if mod.W is not None else 0
        what = f"{n_edges} receptor edges" if mod.mode == "additive" \
            else f"gain on {len(mod.receptors)} neurons"
        print(f"  {mod.name:<12} {mod.mode:<9} ρ={mod.rho:<6g} {reach:<10} {what}")

    header = (f"\n{'dynamics':<9} {'method':<20} "
              + " ".join(f"{'R²(' + m.name + ')':>15}" for m in mods)
              + f" {'layerAUROC':>11} {'cos(J)':>8} {'MSE':>9}")
    print(header)
    print("-" * len(header))
    for phi in ("linear", "tanh"):
        trial = simulate_multi(A0, mods, T=args.T, phi=phi,
                               sigma_x=0.05, seed=args.seed)
        rows = [
            ("oracle", trial.J.copy()),
            ("static_var", StaticVAR().fit_jacobians(trial.observed())),
            (f"sliding_var(w={args.window})",
             SlidingVAR(window=args.window, lam=1e-2).fit_jacobians(trial.observed())),
        ]
        for name, J_hat in rows:
            s = score_multi(trial, J_hat, name)
            r2s = " ".join(f"{v:15.3f}" for v in s["recovery_r2"])
            las = [v for v in s["layer_auroc"] if v == v]
            la_str = "/".join(f"{v:.2f}" for v in las) if las else "n/a"
            print(f"{phi:<9} {name:<20} {r2s} {la_str:>11} "
                  f"{s['jacobian_cosine']:8.3f} {s['one_step_mse']:9.4f}")

    print("""
Reading the table:
  R²(modulator)  how much of each hidden modulator's state is recoverable
                 from a blind PCA readout of the estimated J(t).
  Oracle high on both -> the modulators leave separable traces in J.
  Static ~0      a single fixed graph cannot carry any modulator signal.
  Sliding VAR    tracks the slow modulator; the fast one is at/near its
                 window limit (the timescale control, now per-modulator).
  layerAUROC     does J(high m)-J(low m) select each additive modulator's
                 receptor edges (one value per additive modulator).
""")


if __name__ == "__main__":
    main()

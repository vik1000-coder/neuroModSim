"""Reference circuits and modulator configurations.

These are the "batteries included" models used by the demo, the figure
suite, and the dashboard. They are deliberately simple and fully inspectable
— swap in real data (e.g. a C. elegans connectome and receptor-expression
maps) by building your own `A0` matrix and `Modulator` list; every consumer
of these presets accepts arbitrary ones.

Naming note: modulator names are evocative of the C. elegans monoamine /
neuropeptide systems (Bentley 2016; Ripoll-Sánchez 2023), not quantitative
fits to any specific pathway.
"""

from __future__ import annotations

import numpy as np

from .multi import Modulator, receptor_layer


def build_circuit(
    n: int,
    seed: int = 0,
    chain_weight: float = 0.45,
    recurrent_prob: float = 0.10,
    recurrent_span: int = 3,
    recurrent_lo: float = 0.1,
    recurrent_hi: float = 0.25,
) -> np.ndarray:
    """Wired anatomy A0: a feedforward chain plus sparse signed local recurrence.

    - chain: neuron i drives neuron i+1 with `chain_weight` (the information
      "spine" that lets pulses propagate — and that modulators can reroute).
    - recurrence: each near-diagonal pair (|i-j| <= recurrent_span) gets a
      signed random weight with probability `recurrent_prob`, so the circuit
      is not a trivially invertible chain.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i + 1, i] = chain_weight
    for t in range(n):
        for s in range(max(0, t - recurrent_span), min(n, t + recurrent_span + 1)):
            if s != t and A[t, s] == 0 and rng.random() < recurrent_prob:
                A[t, s] = rng.choice([-1, 1]) * rng.uniform(recurrent_lo, recurrent_hi)
    return A


def build_modulators(n: int, seed: int = 0) -> list[Modulator]:
    """Preset "two": one slow local additive + one fast broadcast gain modulator.

    serotonin  slow (rho=0.995), released by neuron n//3, spatial reach λ=8,
               ADDITIVE: gates extra receptor edges with mixed signs.
    dopamine   faster (rho=0.90), broadcast everywhere, GAIN: scales the
               input gain of the anterior half of the circuit (no new edges).
    """
    rng = np.random.default_rng(seed + 1)
    source = n // 3
    receptor_neurons = rng.choice(n, size=max(4, n // 4), replace=False)
    edges = [(source, int(t), float(rng.choice([-0.4, 0.5, 0.5])))
             for t in receptor_neurons if t != source]
    slow = Modulator(
        name="serotonin", mode="additive", W=receptor_layer(n, edges),
        alpha=0.9, rho=0.995, sigma=1.2,
        sources=(source,), decay_length=8.0, K_half=1.0,
    )
    fast = Modulator(
        name="dopamine", mode="gain", receptors=tuple(range(n // 2)),
        alpha=0.5, rho=0.90, sigma=1.2, sources=(), decay_length=np.inf,
    )
    return [slow, fast]


def build_many_modulators(n: int, seed: int = 0) -> list[Modulator]:
    """Preset "many": four modulators, four timescales, mixed reaches/mechanisms.

    neuropeptide  very slow (rho=0.9995), broadcast, additive long-range edges
    serotonin     slow (rho=0.997), local release (λ=2.5), additive
    dopamine      medium (rho=0.95), local release (λ=3), gain on back half
    octopamine    fast (rho=0.85), broadcast, gain on front quarter

    Requires n >= 12 for the hand-placed long-range edges to make sense.
    """
    rng = np.random.default_rng(seed + 2)
    ser_src = n // 3
    ser_targets = [t for t in range(max(0, ser_src - 3), min(n, ser_src + 4))
                   if t != ser_src]
    ser_edges = [(ser_src, t, float(rng.choice([-0.35, 0.45])))
                 for t in rng.choice(ser_targets, size=min(4, len(ser_targets)),
                                     replace=False)]
    pep_edges = [(n - 2, 1, 0.45), (0, n - 1, 0.40), (n // 2, 0, -0.35)]
    return [
        Modulator(name="neuropeptide", mode="additive",
                  W=receptor_layer(n, pep_edges), alpha=0.7,
                  rho=0.9995, sigma=0.9, sources=(), decay_length=np.inf),
        Modulator(name="serotonin", mode="additive",
                  W=receptor_layer(n, ser_edges), alpha=0.8,
                  rho=0.997, sigma=1.2, sources=(ser_src,), decay_length=2.5),
        Modulator(name="dopamine", mode="gain",
                  receptors=tuple(range(n // 2, n)), alpha=0.5,
                  rho=0.95, sigma=1.2, sources=(2 * n // 3,), decay_length=3.0),
        Modulator(name="octopamine", mode="gain",
                  receptors=tuple(range(max(3, n // 4))), alpha=0.4,
                  rho=0.85, sigma=1.5, sources=(), decay_length=np.inf),
    ]


#: preset name -> (default neuron count, modulator builder)
PRESETS = {
    "two": (24, build_modulators),
    "many": (12, build_many_modulators),
}

"""Structural accounting of the topology study: ring correlators and reachable rank.

Every decoder of the factorial study reads the same four Z-correlators, ZZII,
IZZI, IIZZ and ZIIZ: the edges of the ring 0-1-2-3-0, which is the union of the
nearest and mirror matchings. By Proposition 3 a correlator joining qubits from
different pairs of the circuit's matching factorises, so

    nearest or mirror circuit   two of the four ring correlators factorise
    crossed circuit             all four do, each into a product of the local
                                Z expectations the decoder already reads

The crossed circuit therefore reaches a smaller set of coefficient vectors
whatever the decoder alignment, before any optimisation. This script

  1. checks the factorisation numerically, <Z_a Z_b> - <Z_a><Z_b>, for every
     circuit topology, phase rule and depth, and
  2. reports the numerical Jacobian rank of the coefficient map (circuit angles
     and decoder log-scales) for every circuit topology x decoder alignment,
     with the estimator and threshold of the sector study's rank check.

Usage, from the repository root:
    python src/topology_rank_check.py --output-dir results/topology_rank
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evalmod_factorial_observable_study as topo  # noqa: E402

RING = (("ZZII", "ZIII", "IZII"), ("IZZI", "IZII", "IIZI"),
        ("IIZZ", "IIZI", "IIIZ"), ("ZIIZ", "ZIII", "IIIZ"))
FACTORISATION_ATOL = 1e-12


def ring_pair(correlator: str) -> tuple:
    return tuple(q for q, s in enumerate(correlator) if s == "Z")


def predicted_to_factorise(topology: str) -> int:
    """Ring correlators that Proposition 3 predicts to factorise."""
    pairs = {frozenset(p) for p in topo.PAIR_TOPOLOGIES[topology]}
    return sum(frozenset(ring_pair(zz)) not in pairs for zz, _, _ in RING)


def factorisation_check(layer_values, n_points: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for L in layer_values:
        for topology in topo.PAIR_TOPOLOGIES:
            for rule in topo.PHASE_RULES:
                worst = {zz: 0.0 for zz, _, _ in RING}
                for _ in range(n_points):
                    theta = rng.uniform(-np.pi, np.pi, topo.n_circuit_params(L))
                    state = topo.run_factorial_ansatz(theta, f"{topology}_{rule}", L, False)
                    ev = lambda o: float(np.real(state.conj() @ topo.pauli_operator(o) @ state))  # noqa: E731
                    for zz, a, b in RING:
                        worst[zz] = max(worst[zz], abs(ev(zz) - ev(a) * ev(b)))
                observed = sum(v <= FACTORISATION_ATOL for v in worst.values())
                rows.append({"n_layers": int(L), "topology": topology, "phase_rule": rule,
                             **{f"connected_{zz}": v for zz, v in worst.items()},
                             "ring_correlators_factorised": int(observed),
                             "predicted_by_proposition_3": predicted_to_factorise(topology),
                             "n_points": int(n_points)})
    return pd.DataFrame(rows)


def rank_study(layer_values, n_points: int, eps: float, rank_rtol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cfg0 = topo.base.Config(n_layers=1)
    rows = []
    for L in layer_values:
        cfg = replace(cfg0, n_layers=int(L))
        n_c, n_s = topo.n_circuit_params(L), topo.n_decoder_scales()
        for topology in topo.PAIR_TOPOLOGIES:
            for rule in topo.PHASE_RULES:
                method = f"{topology}_{rule}"
                for alignment in topo.DECODER_ALIGNMENTS:
                    ranks, ranks_loose = [], []
                    for _ in range(n_points):
                        p = np.concatenate([rng.uniform(-np.pi, np.pi, n_c), rng.normal(-1.0, 0.25, n_s)])
                        f = lambda q: topo.evaluate_model(q, method=method, decoder_alignment=alignment,  # noqa: E731
                                                          cfg=cfg, use_bridge=False)[2]
                        J = np.empty((16, len(p)))
                        for i in range(len(p)):
                            d = np.zeros(len(p))
                            d[i] = eps
                            J[:, i] = (f(p + d) - f(p - d)) / (2.0 * eps)
                        sv = np.linalg.svd(J, compute_uv=False)
                        ranks.append(int(np.sum(sv > rank_rtol * sv[0])))
                        ranks_loose.append(int(np.sum(sv > 1e-6 * sv[0])))
                    rows.append({"n_layers": int(L), "topology": topology, "phase_rule": rule,
                                 "decoder_alignment": alignment,
                                 "decoder_matches_topology": topology == alignment,
                                 "n_parameters": n_c + n_s,
                                 "rank_min": int(np.min(ranks)), "rank_max": int(np.max(ranks)),
                                 "rank_max_rtol_1e6": int(np.max(ranks_loose)),
                                 "n_points": int(n_points), "rank_rtol": rank_rtol})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-points", type=int, default=25)
    ap.add_argument("--rank-rtol", type=float, default=1e-8)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--output-dir", default=os.path.join("results", "topology_rank"))
    args = ap.parse_args()

    fac = factorisation_check(args.layers, args.n_points, args.seed)
    bad = fac[fac.ring_correlators_factorised != fac.predicted_by_proposition_3]
    assert bad.empty, f"factorisation differs from Proposition 3:\n{bad}"
    ranks = rank_study(args.layers, args.n_points, args.eps, args.rank_rtol, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    fac.to_csv(os.path.join(args.output_dir, "ring_factorisation.csv"), index=False)
    ranks.to_csv(os.path.join(args.output_dir, "topology_rank.csv"), index=False)
    print(f"wrote {args.output_dir}\n")
    print("ring correlators factorised (observed = predicted):")
    print(fac.groupby("topology").ring_correlators_factorised.agg(["min", "max"]).to_string())
    print("\nmaximum Jacobian rank, circuit topology by decoder alignment (both phase rules):")
    print(ranks.groupby(["n_layers", "topology", "decoder_alignment"]).rank_max.max().unstack().to_string())
    unstable = ranks[ranks.rank_min != ranks.rank_max]
    print("\ncells whose rank varies over the sampled points:", len(unstable))


if __name__ == "__main__":
    main()

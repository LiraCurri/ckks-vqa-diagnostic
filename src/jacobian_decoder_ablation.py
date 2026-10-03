"""Jacobian rank by parameter block: circuit angles versus decoder scales.

Computes the finite-difference Jacobian of the coefficient map, as in the rank
study of sector_graded_evalmod, and reports its numerical rank over three
column blocks:

    circuit  : the circuit angles
    decoder  : the decoder log-scales
    combined : all parameters

If rank(circuit) equals rank(combined), the decoder scales add no reachable
direction. rank(decoder) is at most the number of decoder families, since each
log-scale moves the output only along its family's projection.

Usage:
    python src/jacobian_decoder_ablation.py --output-dir results/jacobian_ablation
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sector_graded_evalmod import (  # noqa: E402
    ANSATZ_KINDS,
    N_HARMONICS,
    _raw_coefficients,
    effective_circuit_params,
    n_circuit_params,
    n_decoder_scales,
    preconditioner,
    safe_write_csv,
)


def _numeric_rank(sv: np.ndarray, rtol: float) -> int:
    """Number of singular values above rtol * sigma_1."""
    if sv.size == 0:
        return 0
    scale = max(float(sv[0]), 1e-300)
    return int(np.sum(sv > rtol * scale))


def jacobian_block_ablation(
    layer_values,
    weights: np.ndarray,
    n_points: int = 25,
    eps: float = 1e-6,
    rank_rtol: float = 1e-8,
    seed: int = 11,
) -> pd.DataFrame:
    """Rank of the coefficient map over circuit-only, decoder-only, and all columns.

    Uses the same estimator, sampling distribution and rank criterion as the
    rank study in sector_graded_evalmod, so the combined column reproduces it.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for L in layer_values:
        n_circ = n_circuit_params(L)
        n_dec = n_decoder_scales()

        for kind in ANSATZ_KINDS:
            bound = 8 if kind == "equivariant" else 16
            r_circ, r_dec, r_all = [], [], []
            sigma_circ_at_bound = []

            for _ in range(n_points):
                p = np.concatenate([
                    rng.uniform(-np.pi, np.pi, n_circ),
                    rng.normal(-1.0, 0.25, n_dec),
                ])

                J = np.empty((N_HARMONICS, len(p)))
                for i in range(len(p)):
                    dp = np.zeros_like(p)
                    dp[i] = eps
                    J[:, i] = (
                        _raw_coefficients(p + dp, kind, L, weights)
                        - _raw_coefficients(p - dp, kind, L, weights)
                    ) / (2 * eps)

                sv_all = np.linalg.svd(J, compute_uv=False)
                sv_circ = np.linalg.svd(J[:, :n_circ], compute_uv=False)
                sv_dec = np.linalg.svd(J[:, n_circ:], compute_uv=False)

                r_all.append(_numeric_rank(sv_all, rank_rtol))
                r_circ.append(_numeric_rank(sv_circ, rank_rtol))
                r_dec.append(_numeric_rank(sv_dec, rank_rtol))

                # How close the circuit-only block sits to the sector bound.
                if len(sv_circ) >= bound:
                    scale = max(float(sv_circ[0]), 1e-300)
                    sigma_circ_at_bound.append(float(sv_circ[bound - 1]) / scale)
                else:
                    sigma_circ_at_bound.append(0.0)

            rows.append({
                "n_layers": int(L),
                "kind": kind,
                "bound": bound,
                "n_circuit_params": n_circ,
                "effective_circuit_params": effective_circuit_params(kind, L),
                "n_decoder_scales": n_dec,
                "rank_circuit_min": int(np.min(r_circ)),
                "rank_circuit_median": float(np.median(r_circ)),
                "rank_circuit_max": int(np.max(r_circ)),
                "rank_decoder_min": int(np.min(r_dec)),
                "rank_decoder_max": int(np.max(r_dec)),
                "rank_combined_min": int(np.min(r_all)),
                "rank_combined_max": int(np.max(r_all)),
                "scales_add_dimensions": int(np.max(r_all) - np.max(r_circ)),
                "sigma_circuit_at_bound_min": float(np.min(sigma_circ_at_bound)),
                "n_points": int(n_points),
                "rank_rtol": rank_rtol,
            })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/jacobian_ablation")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-points", type=int, default=25)
    ap.add_argument("--decay-p", type=float, default=0.0,
                    help="0.0 = unweighted decoder, 2.0 = decay matched")
    ap.add_argument("--rank-rtol", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    weights = preconditioner(args.decay_p)

    frame = jacobian_block_ablation(
        args.layers,
        weights,
        n_points=args.n_points,
        rank_rtol=args.rank_rtol,
        seed=args.seed,
    )

    path = safe_write_csv(frame, os.path.join(args.output_dir,
                                              "jacobian_block_ablation.csv"))
    print(f"wrote {path}\n")

    show = ["n_layers", "kind", "bound", "rank_circuit_max",
            "rank_decoder_max", "rank_combined_max", "scales_add_dimensions"]
    print(frame[show].to_string(index=False))

    equi = frame[frame.kind == "equivariant"]
    if (equi.scales_add_dimensions == 0).all():
        print("\nequivariant: decoder scales add no rank at any depth")
    else:
        worst = int(equi.scales_add_dimensions.max())
        print(f"\nequivariant: decoder scales add up to {worst} rank")


if __name__ == "__main__":
    main()

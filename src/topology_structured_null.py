"""Structure-matched classical null for the topology study (component 2).

A circuit that entangles only within the pairs of a matching, from a product
state, prepares a product of two two-qubit states (Proposition 3). The null
drops the circuit and keeps exactly that restriction: its parameters are two
arbitrary normalised two-qubit states, one on each pair of the matching, and
the four decoder log-scales, read through the same decoder and trained on the
same objective and grid as the circuits. It is classical (two four-component
vectors) and carries the circuit's structural prior without its gates, depth
or phase rule.

The null is trained with its exact gradient (validated against central
differences before any fit): with forward differences its own restarts stall
about 2e-4 above the optimum in relative terms, enough for a circuit run to
appear to beat it.

The default is 200 restarts per cell: in the hardest cells only about three
restarts in a hundred find the best fit, so fewer can miss it.

Its best fit over many restarts is the reference for what the factorised image
allows in each cell of the design. Comparing cells then separates

    the price of the structure   null optimum of one cell against another
    what the circuits lose       circuit result against the null optimum of its cell

The null's coefficient map has the Jacobian rank of the depth-saturated circuit
in every cell, which is checked against topology_rank.csv when that file exists.

Usage, from the repository root:
    python src/topology_structured_null.py --output-dir results/topology_null --jobs 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evalmod_factorial_observable_study as topo  # noqa: E402

RANK_CSV = os.path.join("results", "topology_rank", "topology_rank.csv")
OPTIONS = {"maxiter": 300_000, "maxfun": 3_000_000, "ftol": 1e-15, "gtol": 1e-12, "maxls": 50}
N_STATE = 16                       # two complex four-vectors, as real and imaginary parts
_LETTERS = "abcd"


def study_config():
    """The factorial study's configuration (run_factorial_study)."""
    return topo.base.Config(q0_values=(8, 16), degree=31, delta=0.18, train_samples_per_period=32,
                            eval_samples_per_period=128, seeds=tuple(range(20)), n_qubits=topo.N_QUBITS,
                            n_layers=1, optimizer="L-BFGS-B", maxiter=3000, target_type="sawtooth",
                            alpha_cheb=1e-8, alpha_monomial=1e-4, alpha_derivative=0.0,
                            success_tolerance=1e-4, reference_method="direct_log_monomial")


def product_state(x: np.ndarray, topology: str) -> np.ndarray:
    """|psi_1> on the first pair of the matching, |psi_2> on the second."""
    v = x[:N_STATE].reshape(2, 2, 4)
    psi = v[:, 0, :] + 1j * v[:, 1, :]
    psi = psi / np.linalg.norm(psi, axis=1, keepdims=True)
    (a, b), (c, d) = topo.PAIR_TOPOLOGIES[topology]
    sub = f"{_LETTERS[a]}{_LETTERS[b]},{_LETTERS[c]}{_LETTERS[d]}->abcd"
    return np.einsum(sub, psi[0].reshape(2, 2), psi[1].reshape(2, 2)).reshape(-1)


def coefficients(x: np.ndarray, topology: str, alignment: str) -> np.ndarray:
    state = product_state(x, topology)
    spec = topo.decoder_spec(alignment, topology)
    mu = np.real(np.einsum("i,kij,j->k", state.conj(), spec.operators, state, optimize=True))
    return np.exp(x[N_STATE:])[spec.family_index] * np.clip(mu, -1.0, 1.0)


def objective_and_gradient(x: np.ndarray, topology: str, alignment: str, cache, cfg):
    """The training objective of the null and its exact gradient.

    The objective depends on the state only through <Psi|H|Psi> to first order,
    with H = sum_k (dL/dmu_k) O_k, and Psi is bilinear in the two pair states;
    the normalisation of each pair state is differentiated through."""
    v = x[:N_STATE].reshape(2, 2, 4)
    phi = v[:, 0, :] + 1j * v[:, 1, :]
    norms = np.linalg.norm(phi, axis=1)
    psi = phi / norms[:, None]
    (a, b), (c, d) = topo.PAIR_TOPOLOGIES[topology]
    p1, p2 = f"{_LETTERS[a]}{_LETTERS[b]}", f"{_LETTERS[c]}{_LETTERS[d]}"
    state = np.einsum(f"{p1},{p2}->abcd", psi[0].reshape(2, 2), psi[1].reshape(2, 2)).reshape(-1)
    spec = topo.decoder_spec(alignment, topology)
    scale = np.exp(x[N_STATE:])[spec.family_index]
    o_state = np.einsum("kij,j->ki", spec.operators, state, optimize=True)
    mu = np.real(o_state @ state.conj())
    coeff = scale * mu
    r = cache.design_masked @ coeff - cache.target_masked
    mono = cache.monomial_map @ coeff
    mono_norm = float(np.linalg.norm(mono))
    value = (float(np.mean(r ** 2)) + cfg.alpha_cheb * float(coeff @ coeff)
             + cfg.alpha_monomial * float(np.log1p(mono_norm)))
    grad_c = 2.0 * cache.design_masked.T @ r / len(r) + 2.0 * cfg.alpha_cheb * coeff
    if mono_norm > 0.0:
        grad_c = grad_c + cfg.alpha_monomial * (cache.monomial_map.T @ mono) / (mono_norm * (1.0 + mono_norm))
    grad = np.empty(len(x))
    for g in range(topo.n_decoder_scales()):
        sel = spec.family_index == g
        grad[N_STATE + g] = float(grad_c[sel] @ coeff[sel])
    h_state = ((grad_c * scale) @ o_state).reshape(2, 2, 2, 2)       # H |Psi>
    partial = (np.einsum(f"abcd,{p2}->{p1}", h_state, psi[1].conj().reshape(2, 2)).reshape(-1),
               np.einsum(f"abcd,{p1}->{p2}", h_state, psi[0].conj().reshape(2, 2)).reshape(-1))
    for j in range(2):
        g_j = (2.0 / norms[j]) * (partial[j] - psi[j] * np.real(np.vdot(psi[j], partial[j])))
        grad[8 * j:8 * j + 4], grad[8 * j + 4:8 * j + 8] = g_j.real, g_j.imag
    return value, grad


def validate_gradient(n_points: int = 3, seed: int = 5, h: float = 1e-6, rtol: float = 1e-6) -> float:
    """The exact gradient against central differences, every cell; returns the worst relative error."""
    cfg, cache = study_config(), bundles()[8].train_cache
    rng, worst = np.random.default_rng(seed), 0.0
    for topology in topo.PAIR_TOPOLOGIES:
        for alignment in topo.DECODER_ALIGNMENTS:
            for _ in range(n_points):
                x = np.concatenate([rng.normal(size=N_STATE), rng.normal(-1.0, 0.25, topo.n_decoder_scales())])
                value, g = objective_and_gradient(x, topology, alignment, cache, cfg)
                assert np.isclose(value, topo.fast_regularized_loss(coefficients(x, topology, alignment), cache, cfg),
                                  rtol=1e-12, atol=0.0)
                num = np.empty(len(x))
                for i in range(len(x)):
                    e = np.zeros(len(x))
                    e[i] = h
                    num[i] = (objective_and_gradient(x + e, topology, alignment, cache, cfg)[0]
                              - objective_and_gradient(x - e, topology, alignment, cache, cfg)[0]) / (2.0 * h)
                worst = max(worst, float(np.max(np.abs(g - num)) / max(np.max(np.abs(g)), 1e-300)))
    assert worst < rtol, f"the null's exact gradient disagrees with central differences by {worst:.2e} (relative)"
    return worst


def null_rank(topology: str, alignment: str, n_points: int = 10, eps: float = 1e-6,
              rank_rtol: float = 1e-8, seed: int = 11) -> int:
    rng = np.random.default_rng(seed)
    ranks = []
    for _ in range(n_points):
        x = np.concatenate([rng.normal(size=N_STATE), rng.normal(-1.0, 0.25, topo.n_decoder_scales())])
        J = np.empty((16, len(x)))
        for i in range(len(x)):
            d = np.zeros(len(x))
            d[i] = eps
            J[:, i] = (coefficients(x + d, topology, alignment) - coefficients(x - d, topology, alignment)) / (2 * eps)
        sv = np.linalg.svd(J, compute_uv=False)
        ranks.append(int(np.sum(sv > rank_rtol * sv[0])))
    assert min(ranks) == max(ranks), (topology, alignment, ranks)
    return ranks[0]


_BUNDLES = None


def bundles():
    global _BUNDLES
    if _BUNDLES is None:
        _BUNDLES = topo.build_dataset_bundles(study_config())
    return _BUNDLES


def run_one(task) -> dict:
    q0, topology, alignment, seed = task
    cfg, bundle = study_config(), bundles()[q0]
    rng = np.random.default_rng([20_261_005, q0, seed])
    x0 = np.concatenate([rng.normal(size=N_STATE), rng.normal(-1.0, 0.25, topo.n_decoder_scales())])
    bounds = [(None, None)] * N_STATE + [cfg.log_scale_bounds] * topo.n_decoder_scales()
    t0 = time.perf_counter()
    res = minimize(objective_and_gradient, x0, args=(topology, alignment, bundle.train_cache, cfg), jac=True,
                   method="L-BFGS-B", bounds=bounds, options=OPTIONS)
    c = coefficients(res.x, topology, alignment)
    return {"q0": int(q0), "topology": topology, "decoder_alignment": alignment,
            "decoder_matches_topology": topology == alignment, "seed": int(seed),
            "objective": float(res.fun),
            "train_masked_rmse": topo.masked_rmse_linf(c, topo.build_grid_cache(
                bundle.train[1], bundle.train[2], bundle.train[3], cfg.degree))[0],
            "nested_masked_rmse": topo.masked_rmse_linf(c, bundle.nested_cache)[0],
            "shifted_masked_rmse": topo.masked_rmse_linf(c, bundle.shifted_cache)[0],
            "nit": int(res.nit), "nfev": int(res.nfev), "message": str(res.message),
            "min_log_scale": float(np.min(res.x[N_STATE:])), "max_log_scale": float(np.max(res.x[N_STATE:])),
            "scale_on_bound": bool(np.any(np.abs(res.x[N_STATE:]) >= max(abs(cfg.log_scale_bounds[0]),
                                                                        abs(cfg.log_scale_bounds[1])) - 0.1)),
            "seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--output-dir", default=os.path.join("results", "topology_null"))
    args = ap.parse_args()

    print(f"exact gradient vs central differences: worst relative error {validate_gradient():.2e}")
    cells = [(t, a) for t in topo.PAIR_TOPOLOGIES for a in topo.DECODER_ALIGNMENTS]
    ranks = pd.DataFrame([{"topology": t, "decoder_alignment": a, "null_rank": null_rank(t, a)} for t, a in cells])
    if os.path.exists(RANK_CSV):
        circ = pd.read_csv(RANK_CSV)
        circ = circ[circ.n_layers == circ.n_layers.max()].groupby(["topology", "decoder_alignment"]).rank_max.max()
        ranks["circuit_rank"] = [int(circ[(t, a)]) for t, a in cells]
        assert (ranks.null_rank == ranks.circuit_rank).all(), f"null and circuit ranks differ:\n{ranks}"
        print("null rank equals the depth-saturated circuit rank in every cell")
    tasks = [(q0, t, a, s) for q0 in study_config().q0_values for t, a in cells for s in range(args.restarts)]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(run_one, tasks, chunksize=4))
    else:
        rows = [run_one(t) for t in tasks]
    frame = pd.DataFrame(rows).merge(ranks, on=["topology", "decoder_alignment"])

    os.makedirs(args.output_dir, exist_ok=True)
    frame.to_csv(os.path.join(args.output_dir, "structured_null_results.csv"), index=False)
    best = frame.loc[frame.groupby(["q0", "topology", "decoder_alignment"]).objective.idxmin()]
    summary = frame.groupby(["q0", "topology", "decoder_alignment"]).agg(
        null_rank=("null_rank", "first"), best_objective=("objective", "min"),
        median_nested=("nested_masked_rmse", "median")).join(
        best.set_index(["q0", "topology", "decoder_alignment"])[["nested_masked_rmse", "shifted_masked_rmse"]]
        .rename(columns={"nested_masked_rmse": "best_nested", "shifted_masked_rmse": "best_shifted"}))
    summary["restarts_within_1pct_of_best"] = frame.assign(
        b=frame.groupby(["q0", "topology", "decoder_alignment"]).objective.transform("min")).assign(
        ok=lambda d: d.objective <= 1.01 * d.b).groupby(["q0", "topology", "decoder_alignment"]).ok.sum()
    summary.reset_index().to_csv(os.path.join(args.output_dir, "structured_null_summary.csv"), index=False)
    print(f"wrote {args.output_dir}\n")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(5).to_string())


if __name__ == "__main__":
    main()

"""Representation-limit case for the executed ablation.

A shortfall whose cause is fixed by construction, before any training: the
equivariant model with its decoder scales frozen at s = FROZEN_SCALE, fitted to
TARGET_FACTOR x triangle. Since |mu_k| <= 1, every reachable coefficient
satisfies |c_k| <= exp(s) w_k, while the target's leading coefficient exceeds
that envelope. A bounded least-squares solve over the coefficient box gives a
proven lower bound on the model's training RMSE; the case is valid only if that
bound already exceeds the class optimum by more than GAP_TOL, which is checked
before any optimisation runs.

The protocol's diagnostics are then run exactly as for any case:

    study        study settings (OPTIONS, maxiter from the saved config)
    normalised   ftol and gtol scaled by the objective at the odd-sector ridge
                 optimum (as in objective_scale_check)
    inverse      direct inverse fit to the odd-sector ridge vector, with the
                 settings of inverse_problem_v7 (DIAGNOSTIC_TIGHT, 40 seeds)
    null         spherical_odd at the study settings

and the frozen rules (attribution_rules) give the verdict, with each component
withheld in turn. The expected verdict is "reachability".

The decoder scales are held fixed outside the optimiser: only the circuit
angles are optimised, and the scales are appended inside the objective.

Usage, from the repository root:
    python src/representation_limit_case.py --output-dir results/representation_limit
    python src/representation_limit_case.py --smoke      # two seeds, short caps
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attribution_rules as rules  # noqa: E402
import sector_graded_evalmod as sge  # noqa: E402
from precision_plateau_diagnostic import DIAGNOSTIC_TIGHT  # noqa: E402
from sector_graded_evalmod import (  # noqa: E402
    SECTOR_COLUMNS,
    build_bundles,
    evaluate_model,
    masked_rmse_linf,
    n_circuit_params,
    n_decoder_scales,
    preconditioner,
    random_params,
    regularized_loss,
    ridge_fit,
    safe_write_csv,
    safe_write_json,
    spherical_fit,
    stopping_scale,
)

STUDY_DIR = os.path.join("results", "sector_results_full", "full_p0")

KIND = "equivariant"
N_LAYERS = 3
FROZEN_SCALE = -1.0          # decoder log-scale, all families
TARGET_FACTOR = 5.0          # target = TARGET_FACTOR x triangle
INVERSE_SEEDS = 40           # as in inverse_problem_v7
INVERSE_MAXITER = 20_000     # as in inverse_problem_v7


def load_study_config() -> dict:
    with open(os.path.join(STUDY_DIR, "sector_study_config.json")) as fh:
        cfg = json.load(fh)
    return {"maxiter": int(cfg["maxiter"]), "alpha": float(cfg["alpha"]),
            "delta": float(cfg["delta"])}


def scaled_bundle(delta: float, factor: float):
    """The triangle bundle with every target multiplied by factor."""
    b = build_bundles(delta)["triangle"]
    scale = lambda g: dataclasses.replace(g, target_masked=factor * g.target_masked)  # noqa: E731
    return dataclasses.replace(b, train=scale(b.train), nested=scale(b.nested),
                               shifted=scale(b.shifted))


def certified_floor(train, weights: np.ndarray, scale: float) -> dict:
    """Lower bound on the model's training RMSE over the coefficient box.

    The reachable coefficients lie in the odd sector with |c_k| <= e^s w_k, so the
    bounded least-squares minimum over that box bounds every reachable fit from
    below. It bounds the training fit term only, hence also the training RMSE.
    """
    cols = SECTOR_COLUMNS["odd"]
    cap = np.exp(scale) * weights[cols]
    sol = lsq_linear(train.design_masked[:, cols], train.target_masked,
                     bounds=(-cap, cap), method="bvls", tol=1e-14)
    c = np.zeros(sge.N_HARMONICS)
    c[cols] = sol.x
    return {"floor_train_rmse": masked_rmse_linf(c, train)[0],
            "envelope": float(np.exp(scale)), "box_active": int(np.sum(sol.active_mask != 0))}


class FrozenScaleModel:
    """Coefficient map over the circuit angles, decoder scales held fixed."""

    def __init__(self, weights: np.ndarray, scale: float):
        self.weights = weights
        self.scales = np.full(n_decoder_scales(), float(scale))
        self.nc = n_circuit_params(N_LAYERS)

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        params = np.concatenate([np.asarray(theta, float), self.scales])
        _, _, _, c, _ = evaluate_model(params, KIND, N_LAYERS, self.weights)
        return c

    def raw_even_max(self, theta: np.ndarray) -> float:
        params = np.concatenate([np.asarray(theta, float), self.scales])
        _, raw, _, _, _ = evaluate_model(params, KIND, N_LAYERS, self.weights)
        return float(np.max(np.abs(raw[sge.EVEN_SLICE])))

    def initial(self, seed: int) -> np.ndarray:
        return random_params(N_LAYERS, seed)[:self.nc]


def fit_runs(arm: str, model: FrozenScaleModel, bundle, alpha: float, options: dict,
             maxiter: int, seeds) -> list:
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        res = minimize(lambda th: regularized_loss(model.coefficients(th), bundle.train, alpha),
                       model.initial(seed), method="L-BFGS-B",
                       bounds=[(None, None)] * model.nc,
                       options={"maxiter": maxiter, **options})
        c = model.coefficients(res.x)
        rows.append({"arm": arm, "seed": int(seed),
                     "train_rmse": masked_rmse_linf(c, bundle.train)[0],
                     "nested_rmse": masked_rmse_linf(c, bundle.nested)[0],
                     "objective": float(res.fun), "success": bool(res.success),
                     "message": str(res.message), "nfev": int(res.nfev), "nit": int(res.nit),
                     "max_abs_coefficient": float(np.max(np.abs(c))),
                     "raw_even_max": model.raw_even_max(res.x),
                     "seconds": time.perf_counter() - t0})
        print(f"{arm:10s} seed {seed:2d}: nested {rows[-1]['nested_rmse']:.3e}, "
              f"{res.nit} it, {res.message}")
    return rows


def inverse_runs(model: FrozenScaleModel, bundle, reference: np.ndarray, maxiter: int,
                 seeds) -> list:
    rows = []
    for seed in seeds:
        res = minimize(lambda th: float(np.sum((model.coefficients(th) - reference) ** 2)),
                       model.initial(seed), method="L-BFGS-B",
                       bounds=[(None, None)] * model.nc,
                       options={"maxiter": maxiter, **DIAGNOSTIC_TIGHT})
        c = model.coefficients(res.x)
        rows.append({"arm": "inverse", "seed": int(seed),
                     "coefficient_l2": float(np.linalg.norm(c - reference)),
                     "train_rmse": masked_rmse_linf(c, bundle.train)[0],
                     "nested_rmse": masked_rmse_linf(c, bundle.nested)[0],
                     "success": bool(res.success), "message": str(res.message),
                     "nfev": int(res.nfev), "nit": int(res.nit)})
        print(f"inverse    seed {seed:2d}: |c - c_ref| {rows[-1]['coefficient_l2']:.3e}, "
              f"nested {rows[-1]['nested_rmse']:.3e}")
    return rows


def null_runs(bundle, alpha: float, maxiter: int, seeds) -> list:
    rows = []
    for seed in seeds:
        c, res = spherical_fit("spherical_odd", bundle.train, alpha, int(seed), maxiter)
        rows.append({"arm": "null", "seed": int(seed),
                     "train_rmse": masked_rmse_linf(c, bundle.train)[0],
                     "nested_rmse": masked_rmse_linf(c, bundle.nested)[0],
                     "success": bool(res.success), "message": str(res.message),
                     "nfev": int(res.nfev)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--inverse-seeds", type=int, default=INVERSE_SEEDS)
    ap.add_argument("--output-dir", default=os.path.join("results", "representation_limit"))
    ap.add_argument("--smoke", action="store_true", help="two seeds and short caps, for a quick check")
    args = ap.parse_args()

    cfg = load_study_config()
    alpha = cfg["alpha"]
    seeds = range(2 if args.smoke else args.seeds)
    inv_seeds = range(2 if args.smoke else args.inverse_seeds)
    maxiter = 50 if args.smoke else cfg["maxiter"]
    inv_maxiter = 100 if args.smoke else INVERSE_MAXITER

    weights = preconditioner(0.0)
    bundle = scaled_bundle(cfg["delta"], TARGET_FACTOR)
    model = FrozenScaleModel(weights, FROZEN_SCALE)

    # Ground truth, fixed before any optimisation.
    ridge_odd = ridge_fit(bundle.train, alpha, "odd")
    class_train = masked_rmse_linf(ridge_odd, bundle.train)[0]
    class_nested = masked_rmse_linf(ridge_odd, bundle.nested)[0]
    full_nested = masked_rmse_linf(ridge_fit(bundle.train, alpha, "full"), bundle.nested)[0]
    floor = certified_floor(bundle.train, weights, FROZEN_SCALE)
    print(f"target {TARGET_FACTOR:g} x triangle; leading coefficient {np.max(np.abs(ridge_odd)):.3f}, "
          f"envelope {floor['envelope']:.3f}")
    print(f"certified training-RMSE floor {floor['floor_train_rmse']:.3e} against the class optimum "
          f"{class_train:.3e} ({floor['floor_train_rmse'] / class_train:.0f}x)")
    if not floor["floor_train_rmse"] > (1.0 + rules.GAP_TOL) * class_train:
        raise AssertionError("the constructed gap is not certified; the case is invalid")

    study_opts = dict(sge.OPTIONS)
    scale = stopping_scale(bundle.train, alpha, study_opts, "odd")
    normalised_opts = {**study_opts, "ftol": study_opts["ftol"] * scale["f_ref"],
                       "gtol": study_opts["gtol"] * scale["f_ref"]}
    scale_check_fails = scale["threshold_over_fit"] > sge.STOPPING_SCALE_LIMIT

    rows = []
    rows += fit_runs("study", model, bundle, alpha, study_opts, maxiter, seeds)
    rows += fit_runs("normalised", model, bundle, alpha, normalised_opts, maxiter, seeds)
    rows += inverse_runs(model, bundle, ridge_odd, inv_maxiter, inv_seeds)
    rows += null_runs(bundle, alpha, maxiter, seeds)
    frame = pd.DataFrame(rows)

    # Sanity: the frozen model must respect its envelope and the selection rule.
    fitted = frame[frame.arm.isin(["study", "normalised"])]
    assert (fitted.max_abs_coefficient <= floor["envelope"] * (1 + 1e-12)).all(), "envelope violated"
    assert (fitted.raw_even_max == 0.0).all(), "selection rule violated"
    assert (fitted.train_rmse >= floor["floor_train_rmse"] * (1 - 1e-9)).all(), "certified floor violated"

    study = frame[frame.arm == "study"]
    evidence = rules.Evidence(
        observed=float(study.nested_rmse.median()),
        reference="class",
        optimizer_success=rules.optimizer_success_rule(study.success),
        restarts_agree=rules.restarts_agree_rule(study.nested_rmse),
        class_optimum=class_nested,
        unrestricted_optimum=full_nested,
        null_result=float(frame[frame.arm == "null"].nested_rmse.median()),
        inverse_best=float(frame[frame.arm == "inverse"].nested_rmse.min()),
        scale_check_fails=bool(scale_check_fails),
        normalised_result=float(frame[frame.arm == "normalised"].nested_rmse.median()),
    )
    verdicts = {"all components": rules.attribute(evidence)}
    for k in (1, 2, 4, 5):
        verdicts[f"without {k}"] = rules.attribute(rules.withhold(evidence, k))
    verdicts["without 4 and 5"] = rules.attribute(rules.withhold(rules.withhold(evidence, 4), 5))

    os.makedirs(args.output_dir, exist_ok=True)
    stem = "representation_limit_smoke" if args.smoke else "representation_limit"
    path = safe_write_csv(frame.assign(frozen_scale=FROZEN_SCALE, target_factor=TARGET_FACTOR),
                          os.path.join(args.output_dir, f"{stem}_runs.csv"))
    summary = {
        "rules_version": rules.RULES_VERSION,
        "construction": {"kind": KIND, "n_layers": N_LAYERS, "frozen_scale": FROZEN_SCALE,
                         "target_factor": TARGET_FACTOR, "alpha": alpha, "delta": cfg["delta"],
                         "smoke": bool(args.smoke)},
        "ground_truth": {**floor, "class_train_rmse": class_train,
                         "floor_over_class": floor["floor_train_rmse"] / class_train},
        "stopping_scale": scale,
        "evidence": dataclasses.asdict(evidence),
        "verdicts": verdicts,
    }
    safe_write_json(summary, os.path.join(args.output_dir, f"{stem}_summary.json"))
    print(f"\nwrote {path}")
    print(json.dumps({"evidence": summary["evidence"], "verdicts": verdicts}, indent=2))
    if verdicts["all components"] != "reachability":
        print("\nNOTE: the verdict differs from the constructed cause; report it as found.")


if __name__ == "__main__":
    main()

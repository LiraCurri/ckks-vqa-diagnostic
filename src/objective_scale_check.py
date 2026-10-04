"""Stopping-test scale check for one cell of the sector study (by default the
depth-three equivariant model on the triangle).

SciPy's L-BFGS-B stops when (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol.
When the objective is far below 1 the denominator is 1, so ftol acts as an
absolute threshold. This script reports the objective's scale and reruns the
principal study twice per seed:

    study       the principal study's settings (OPTIONS, maxiter from the
                saved config)
    normalised  the same settings with ftol and gtol multiplied by the
                objective at the ridge optimum of the sector the model
                occupies (odd for equivariant, full otherwise), f_ref; near
                the optimum this is equivalent to minimising f / f_ref

The study arm must reproduce the published per-seed results; this is checked
before the normalised arm runs.

Usage:
    python src/objective_scale_check.py --output-dir results/objective_scale
    python src/objective_scale_check.py --family generic --target sawtooth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sector_graded_evalmod as sge  # noqa: E402
from sector_graded_evalmod import (  # noqa: E402
    build_bundles,
    masked_rmse_linf,
    preconditioner,
    regularized_loss,
    ridge_fit,
    safe_write_csv,
    train_model,
)

STUDY_DIR = os.path.join("results", "sector_results_full", "full_p0")
SWEEP_CSV = os.path.join("results", "precision_plateau_diagnostic", "ftol_sweep.csv")


def load_study_config() -> dict:
    with open(os.path.join(STUDY_DIR, "sector_study_config.json")) as fh:
        cfg = json.load(fh)
    return {"maxiter": int(cfg["maxiter"]), "alpha": float(cfg["alpha"]),
            "delta": float(cfg["delta"])}


def objective_scale(bundle, alpha: float, sector: str) -> dict:
    """Split the objective at the sector ridge optimum into fit and ridge terms."""
    c = ridge_fit(bundle.train, alpha, sector)
    fit = float(np.mean((bundle.train.design_masked @ c - bundle.train.target_masked) ** 2))
    ridge = alpha * float(c @ c)
    return {"f_ref": regularized_loss(c, bundle.train, alpha), "fit_term": fit,
            "ridge_term": ridge, "ref_nested": masked_rmse_linf(c, bundle.nested)[0]}


def run_arm(arm, options, bundle, cfg, n_layers, seeds, f_ref, weights, family, target):
    sge.OPTIONS.clear()
    sge.OPTIONS.update(options)
    rows = []
    for seed in seeds:
        m = train_model(target, family, n_layers, seed, bundle.train,
                        cfg["alpha"], cfg["maxiter"], weights)
        nested, _ = masked_rmse_linf(m.coefficients, bundle.nested)
        rows.append(dict(arm=arm, family=family, target=target, n_layers=n_layers,
                         seed=seed, nested_rmse=nested,
                         objective=m.fun, objective_gap=m.fun - f_ref,
                         relative_gap=(m.fun - f_ref) / f_ref, nfev=m.nfev, nit=m.nit,
                         message=m.message, seconds=m.seconds, **options))
        print(f"{arm:10s} seed {seed:2d}: nested {nested:.3e}, gap/f_ref "
              f"{(m.fun - f_ref) / f_ref:.1e}, {m.nit:.0f} it, {m.message}")
    return rows


def check_reproduces_study(frame: pd.DataFrame, n_layers: int, family: str, target: str) -> None:
    path = os.path.join(STUDY_DIR, "sector_exact_results.csv")
    if not os.path.exists(path):
        print(f"[reproduction check skipped] {path} not found")
        return
    pub = pd.read_csv(path)
    pub = pub[(pub.target == target) & (pub.method == family)
              & (pub.n_layers == n_layers) & (pub["init"].isin(["random", "n/a"]))]
    ref = pub.drop_duplicates("seed").set_index("seed").nested_rmse
    got = frame.set_index("seed").nested_rmse
    common = got.index.intersection(ref.index)
    if not len(common):
        raise AssertionError("study arm: no published seeds to compare against")
    worst = float(np.max(np.abs(got[common] / ref[common] - 1.0)))
    print(f"  study arm: {len(common)} seeds, worst relative difference {worst:.2e}")
    if worst > 1e-6:
        raise AssertionError("study arm does not reproduce sector_exact_results.csv")


def compare_with_sweep(frame: pd.DataFrame, n_layers: int, family: str, target: str) -> pd.DataFrame:
    """Per-seed comparison of the normalised arm with the ftol = 1e-18 sweep runs
    (the sweep covers the equivariant model on the triangle only)."""
    if (family, target) != ("equivariant", "triangle"):
        return frame.assign(sweep_tight_nested=np.nan)
    if not os.path.exists(SWEEP_CSV):
        print(f"[sweep comparison skipped] {SWEEP_CSV} not found")
        return frame.assign(sweep_tight_nested=np.nan)
    sw = pd.read_csv(SWEEP_CSV)
    sw = sw[(sw.preconditioner == "p0") & (sw.n_layers == n_layers)
            & np.isclose(sw.ftol.astype(float), 1e-18, rtol=1e-6, atol=0.0)]
    ref = sw.set_index("seed").nested_rmse
    norm = frame[frame.arm == "normalised"].set_index("seed").nested_rmse
    common = norm.index.intersection(ref.index)
    rel = (norm[common] / ref[common] - 1.0).abs()
    print(f"  normalised vs ftol=1e-18 sweep: {len(common)} seeds, "
          f"{int((rel < 1e-6).sum())} identical to 1e-6, worst relative difference {rel.max():.2e}")
    return frame.assign(sweep_tight_nested=frame.seed.map(ref))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", default="equivariant", choices=sorted(sge.MODEL_SECTOR))
    ap.add_argument("--target", default="triangle")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--normalised-maxiter", type=int, default=None,
                    help="iteration cap for the normalised arm (default: the study's)")
    ap.add_argument("--output-dir", default=os.path.join("results", "objective_scale"))
    args = ap.parse_args()

    cfg = load_study_config()
    bundle = build_bundles(cfg["delta"])[args.target]
    weights = preconditioner(0.0)
    sector = sge.MODEL_SECTOR[args.family]
    scale = objective_scale(bundle, cfg["alpha"], sector)
    print(f"{args.family} on {args.target}, L={args.n_layers}, {sector} sector")
    study = dict(sge.OPTIONS)
    print(f"f_ref {scale['f_ref']:.3e} = fit {scale['fit_term']:.3e} + ridge "
          f"{scale['ridge_term']:.3e}; ftol {study['ftol']:.0e} is "
          f"{study['ftol'] / scale['f_ref']:.2e} of f_ref and "
          f"{study['ftol'] / scale['fit_term']:.2e} of the fit term")
    for name, opts in (("study", study),
                       ("normalised", {**study, "ftol": study["ftol"] * scale["f_ref"],
                                       "gtol": study["gtol"] * scale["f_ref"]})):
        try:
            sge.validate_stopping_scale(bundle.train, cfg["alpha"], opts, sector)
            print(f"  scale invariant, {name} settings: passes")
        except AssertionError:
            print(f"  scale invariant, {name} settings: fails")

    seeds = range(args.seeds)
    rows = run_arm("study", study, bundle, cfg, args.n_layers, seeds, scale["f_ref"], weights,
                   args.family, args.target)
    check_reproduces_study(pd.DataFrame(rows), args.n_layers, args.family, args.target)
    normalised = {**study, "ftol": study["ftol"] * scale["f_ref"],
                  "gtol": study["gtol"] * scale["f_ref"]}
    cfg_norm = {**cfg, "maxiter": args.normalised_maxiter or cfg["maxiter"]}
    rows += run_arm("normalised", normalised, bundle, cfg_norm, args.n_layers, seeds,
                    scale["f_ref"], weights, args.family, args.target)

    frame = pd.DataFrame(rows).assign(**{k: v for k, v in scale.items()})
    frame = compare_with_sweep(frame, args.n_layers, args.family, args.target)
    os.makedirs(args.output_dir, exist_ok=True)
    name = ("objective_scale_results.csv" if (args.family, args.target) == ("equivariant", "triangle")
            else f"objective_scale_{args.family}_{args.target}_L{args.n_layers}.csv")
    path = safe_write_csv(frame, os.path.join(args.output_dir, name))
    print(f"\nwrote {path}")
    summary = frame.groupby("arm").agg(
        median_nested=("nested_rmse", "median"),
        reached_reference=("nested_rmse", lambda x: float((x < 1.1 * scale["ref_nested"]).mean())),
        median_nit=("nit", "median"),
        stopped_on_limit=("message", lambda m: float(m.str.contains("LIMIT").mean())),
        median_gap_over_fref=("relative_gap", "median"))
    print(summary.to_string())


if __name__ == "__main__":
    main()

"""Noisy-target experiment.

Gaussian noise of standard deviation sigma (relative to the target's RMS) is
added to the training targets; every model is scored on the clean nested and
shifted grids. Six arms are fitted to the same noisy data per seed:

    full_ridge      closed-form ridge fit over all 16 harmonics
    full_ridge_cv   the same, with alpha chosen by 5-fold cross-validation
    cv_selected     full or odd-sector fit, whichever has the lower 5-fold
                    cross-validation error
    odd_ridge       closed-form ridge fit over the odd sector
    spherical_odd   trained dimension-matched classical null
    equivariant     trained quantum model

maxiter, alpha and delta are read from the sector study's saved config. At
sigma = 0 the trained arms must reproduce the published per-seed results;
this is checked before the noisy levels run.

--tight uses the settings of the ftol sweep (DIAGNOSTIC_TIGHT in
precision_plateau_diagnostic, maxiter 50,000) for both trained arms; at
sigma = 0 the quantum runs are then checked against ftol_sweep.csv at 1e-18.

Usage:
    python src/noisy_target_experiment.py --output-dir results/noisy_target
    python src/noisy_target_experiment.py --tight --sigmas 1e-3 1e-2 3e-2 \
        --output-dir results/noisy_target
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sector_graded_evalmod as sge  # noqa: E402
from sector_graded_evalmod import (  # noqa: E402
    build_bundles,
    masked_rmse_linf,
    preconditioner,
    regularized_loss,
    ridge_fit,
    safe_write_csv,
    spherical_fit,
    train_model,
)

STUDY_DIR = os.path.join("results", "sector_results_full", "full_p0")
SWEEP_CSV = os.path.join("results", "precision_plateau_diagnostic", "ftol_sweep.csv")
TIGHT_MAXITER = 50_000          # ftol_sweep_v7 default
DEFAULT_SIGMAS = (0.0, 1e-4, 1e-3, 1e-2, 3e-2)   # multiples of the clean target's RMS


def load_study_config() -> dict:
    path = os.path.join(STUDY_DIR, "sector_study_config.json")
    with open(path) as fh:
        cfg = json.load(fh)
    return {"maxiter": int(cfg["maxiter"]), "alpha": float(cfg["alpha"]),
            "delta": float(cfg["delta"])}


def noisy_cache(cache, sigma_abs: float, rng: np.random.Generator):
    """Training cache with Gaussian noise added to the targets (design unchanged)."""
    if sigma_abs == 0.0:
        return cache
    noise = rng.normal(0.0, sigma_abs, size=cache.target_masked.shape)
    return dataclasses.replace(cache, target_masked=cache.target_masked + noise)


CV_ALPHAS = np.logspace(-10, -1, 46)


def cv_alpha(cache, seed: int, k: int = 5) -> float:
    """Ridge constant for the full fit chosen by k-fold cross-validation."""
    n = cache.target_masked.size
    folds = np.array_split(np.random.default_rng([7, seed]).permutation(n), k)
    errors = []
    for a in CV_ALPHAS:
        e = 0.0
        for f in folds:
            keep = np.ones(n, bool)
            keep[f] = False
            sub = dataclasses.replace(cache, design_masked=cache.design_masked[keep],
                                      target_masked=cache.target_masked[keep])
            c = ridge_fit(sub, a, "full")
            e += float(np.sum((cache.design_masked[f] @ c - cache.target_masked[f]) ** 2))
        errors.append(e)
    return float(CV_ALPHAS[int(np.argmin(errors))])


def cv_error(cache, sector: str, alpha: float, seed: int, k: int = 5) -> float:
    """k-fold cross-validation error of the ridge fit over a sector."""
    n = cache.target_masked.size
    folds = np.array_split(np.random.default_rng([7, seed]).permutation(n), k)
    e = 0.0
    for f in folds:
        keep = np.ones(n, bool)
        keep[f] = False
        sub = dataclasses.replace(cache, design_masked=cache.design_masked[keep],
                                  target_masked=cache.target_masked[keep])
        c = ridge_fit(sub, alpha, sector)
        e += float(np.sum((cache.design_masked[f] @ c - cache.target_masked[f]) ** 2))
    return e


def score(c, bundle, train, alpha) -> dict:
    nested, nested_linf = masked_rmse_linf(c, bundle.nested)
    shifted, _ = masked_rmse_linf(c, bundle.shifted)
    train_rmse, _ = masked_rmse_linf(c, train)
    return {"nested_rmse": nested, "nested_linf": nested_linf,
            "shifted_rmse": shifted, "train_rmse_noisy": train_rmse,
            "objective_noisy": regularized_loss(c, train, alpha)}


def run(target: str, n_layers: int, sigmas, seeds, cfg: dict) -> pd.DataFrame:
    bundle = build_bundles(cfg["delta"])[target]
    weights = preconditioner(0.0)                          # unweighted decoder
    target_rms = float(np.sqrt(np.mean(bundle.train.target_masked ** 2)))
    rows = []
    for rel in sigmas:
        sigma_abs = rel * target_rms
        for seed in seeds:
            rng = np.random.default_rng([20_260_929, seed])   # noise realisation
            train = noisy_cache(bundle.train, sigma_abs, rng)
            base = dict(target=target, n_layers=n_layers, sigma_rel=rel,
                        sigma_abs=sigma_abs, seed=seed,
                        ftol=float(sge.OPTIONS["ftol"]))

            for sector in ("full", "odd"):
                c = ridge_fit(train, cfg["alpha"], sector=sector)
                rows.append({**base, "arm": f"{sector}_ridge", "nfev": 0,
                             "seconds": 0.0, "alpha": cfg["alpha"],
                             **score(c, bundle, train, cfg["alpha"])})

            pick = min(("full", "odd"), key=lambda sec: cv_error(train, sec, cfg["alpha"], seed))
            c = ridge_fit(train, cfg["alpha"], sector=pick)
            rows.append({**base, "arm": "cv_selected", "nfev": 0, "seconds": 0.0,
                         "alpha": cfg["alpha"], "selected": pick,
                         **score(c, bundle, train, cfg["alpha"])})

            a_cv = cv_alpha(train, seed)
            c = ridge_fit(train, a_cv, sector="full")
            rows.append({**base, "arm": "full_ridge_cv", "nfev": 0, "seconds": 0.0,
                         "alpha": a_cv, **score(c, bundle, train, a_cv)})

            t0 = time.perf_counter()
            c, res = spherical_fit("spherical_odd", train, cfg["alpha"], seed, cfg["maxiter"])
            rows.append({**base, "arm": "spherical_odd", "nfev": int(res.nfev),
                         "seconds": time.perf_counter() - t0,
                         **score(c, bundle, train, cfg["alpha"])})

            m = train_model(target, "equivariant", n_layers, seed, train,
                            cfg["alpha"], cfg["maxiter"], weights)
            rows.append({**base, "arm": "equivariant", "nfev": m.nfev,
                         "seconds": m.seconds,
                         **score(m.coefficients, bundle, train, cfg["alpha"])})

            by_arm = {r["arm"]: r["nested_rmse"] for r in rows[-6:]}
            print(f"sigma={rel:g} seed={seed:2d}: full {by_arm['full_ridge']:.3e}  "
                  f"full-cv {by_arm['full_ridge_cv']:.3e}  selected {by_arm['cv_selected']:.3e}  "
                  f"odd {by_arm['odd_ridge']:.3e}  "
                  f"null {by_arm['spherical_odd']:.3e}  quantum {by_arm['equivariant']:.3e}  "
                  f"({m.seconds:.0f}s)")
    return pd.DataFrame(rows)


def check_reproduces_study(frame: pd.DataFrame, target: str, n_layers: int) -> None:
    """At sigma = 0 the trained arms must reproduce the published per-seed rows."""
    path = os.path.join(STUDY_DIR, "sector_exact_results.csv")
    if not os.path.exists(path):
        print(f"[reproduction check skipped] {path} not found")
        return
    pub = pd.read_csv(path)
    pub = pub[(pub.target == target) & (pub["init"].isin(["random", "n/a"]))]
    zero = frame[frame.sigma_rel == 0.0]
    for arm, method, by_layer in (("equivariant", "equivariant", True),
                                  ("spherical_odd", "spherical_odd", False)):
        ref = pub[pub.method == method]
        if by_layer:
            ref = ref[ref.n_layers == n_layers]
        ref = ref.drop_duplicates("seed").set_index("seed").nested_rmse
        got = zero[zero.arm == arm].set_index("seed").nested_rmse
        common = got.index.intersection(ref.index)
        if not len(common):
            raise AssertionError(f"{arm}: no published seeds to compare against")
        worst = float(np.max(np.abs(got[common] / ref[common] - 1.0)))
        print(f"  {arm}: {len(common)} seeds, worst relative difference {worst:.2e}")
        if worst > 1e-6:
            raise AssertionError(
                f"{arm} at sigma=0 does not reproduce the published study "
                f"(worst relative difference {worst:.2e}); stopping before the noisy runs")


def check_reproduces_sweep(frame: pd.DataFrame, n_layers: int) -> None:
    """In tight mode at sigma = 0 the quantum runs must match ftol_sweep.csv at 1e-18."""
    if not os.path.exists(SWEEP_CSV):
        print(f"[reproduction check skipped] {SWEEP_CSV} not found")
        return
    sw = pd.read_csv(SWEEP_CSV)
    sw = sw[(sw.preconditioner == "p0") & (sw.n_layers == n_layers)
            & np.isclose(sw.ftol.astype(float), 1e-18, rtol=1e-6, atol=0.0)]
    ref = sw.set_index("seed").nested_rmse
    got = frame[(frame.sigma_rel == 0.0) & (frame.arm == "equivariant")].set_index("seed").nested_rmse
    common = got.index.intersection(ref.index)
    if not len(common):
        raise AssertionError("no published ftol=1e-18 seeds to compare against")
    worst = float(np.max(np.abs(got[common] / ref[common] - 1.0)))
    print(f"  equivariant (tight): {len(common)} seeds, worst relative difference {worst:.2e}")
    if worst > 1e-6:
        raise AssertionError(
            f"tight mode at sigma=0 does not reproduce ftol_sweep.csv "
            f"(worst relative difference {worst:.2e}); stopping before the noisy runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="triangle")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    ap.add_argument("--output-dir", default=os.path.join("results", "noisy_target"))
    ap.add_argument("--tight", action="store_true",
                    help="use the published tolerance sweep's settings (ftol 1e-18)")
    args = ap.parse_args()

    cfg = load_study_config()
    if args.tight:
        from precision_plateau_diagnostic import DIAGNOSTIC_TIGHT
        # train_model and spherical_fit read this dict, so both trained arms
        # use the sweep's settings.
        sge.OPTIONS.update(DIAGNOSTIC_TIGHT)
        cfg["maxiter"] = TIGHT_MAXITER
        print(f"tight mode: optimiser options {sge.OPTIONS}, maxiter {cfg['maxiter']}")
    seeds = list(range(args.seeds))
    sigmas = sorted(set(args.sigmas) | {0.0})              # sigma = 0 always run first
    print(f"config from study: {cfg}")

    print("\nsigma = 0 (reproduction check):")
    zero = run(args.target, args.n_layers, [0.0], seeds, cfg)
    if args.tight:
        check_reproduces_sweep(zero, args.n_layers)
    else:
        check_reproduces_study(zero, args.target, args.n_layers)

    print("\nnoisy levels:")
    noisy = run(args.target, args.n_layers, [s for s in sigmas if s > 0], seeds, cfg)
    frame = pd.concat([zero, noisy], ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    name = "noisy_target_results_tight.csv" if args.tight else "noisy_target_results.csv"
    path = safe_write_csv(frame, os.path.join(args.output_dir, name))
    print(f"\nwrote {path}")

    med = frame.pivot_table(index="sigma_rel", columns="arm",
                            values="nested_rmse", aggfunc="median")
    med["naive_ratio"] = med.equivariant / med.full_ridge
    med["vs_tuned_full"] = med.equivariant / med.full_ridge_cv
    med["vs_prior"] = med.equivariant / med.odd_ridge
    med["vs_null"] = med.equivariant / med.spherical_odd
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}",
                           "display.width", 160, "display.max_columns", None):
        print("\nmedian clean nested RMSE by noise level:")
        print(med)


if __name__ == "__main__":
    main()

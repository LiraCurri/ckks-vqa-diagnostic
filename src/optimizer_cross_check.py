"""Restart diagnostics for the depth-three equivariant cell.

Idempotence test: run L-BFGS-B at the study settings, then restart it from the
returned point with identical settings and record the further change in the
objective and nested RMSE. A converged run should not improve on restart.

With --restart-ftol, legs after the first use the given ftol instead. Since a
restart discards the L-BFGS curvature history, comparing the result with the
continuous run at the same tolerance separates the effect of the stopping
criterion from that of the history.

With --methods, the same cell is also run under TNC and Powell for reference.
Both are slower on this problem; raise --powell-budget for Powell.

Usage:
    python src/optimizer_cross_check.py --output-dir results/optimizer_cross_check
    python src/optimizer_cross_check.py --restart-ftol 1e-18
    python src/optimizer_cross_check.py --methods --powell-budget 500000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sector_graded_evalmod import (  # noqa: E402
    build_bundles,
    evaluate_model,
    masked_rmse_linf,
    model_bounds,
    preconditioner,
    random_params,
    regularized_loss,
    ridge_fit,
    safe_write_csv,
)
from precision_plateau_diagnostic import (  # noqa: E402
    DIAGNOSTIC_TIGHT,
    _optimizer_diagnostics,
)


# Study settings, taken from precision_plateau_diagnostic so that the first
# leg reproduces the ftol=1e-12 rows of ftol_sweep.csv. Additional options
# (e.g. maxcor) would change the trajectory.
OPT_STUDY = {"maxiter": 50_000, **DIAGNOSTIC_TIGHT, "ftol": 1e-12}


def _setup(target: str, decay_p: float, delta: float, alpha: float):
    bundle = build_bundles(delta)[target]
    weights = preconditioner(decay_p)
    reference = ridge_fit(bundle.train, alpha, sector="odd")
    ref_nested, _ = masked_rmse_linf(reference, bundle.nested)
    ref_objective = regularized_loss(reference, bundle.train, alpha)
    return bundle, weights, reference, ref_nested, ref_objective


def idempotence_test(
    *,
    target: str = "triangle",
    kind: str = "equivariant",
    n_layers: int = 3,
    decay_p: float = 0.0,
    delta: float = 0.06,
    alpha: float = 1e-8,
    seeds=range(20),
    chain: int = 2,
    restart_ftol: float | None = None,
) -> pd.DataFrame:
    """Restart the solver from its own output and measure remaining progress.

    restart_ftol=None repeats the study configuration on every leg (the
    idempotence test proper). A value instead applies that ftol from the
    second leg onward, which asks whether the stopping criterion alone
    accounts for the tighter-tolerance result.
    """
    bundle, weights, reference, ref_nested, ref_objective = _setup(
        target, decay_p, delta, alpha
    )

    def objective(params: np.ndarray) -> float:
        _, _, _, coefficients, _ = evaluate_model(
            params, kind, n_layers, weights
        )
        return regularized_loss(coefficients, bundle.train, alpha)

    opts_first = OPT_STUDY
    opts_later = (OPT_STUDY if restart_ftol is None
                  else {**OPT_STUDY, "ftol": float(restart_ftol)})

    rows = []
    for seed in (int(s) for s in seeds):
        x = random_params(n_layers, seed)
        history = []
        total_nfev = 0
        t0 = time.perf_counter()

        for leg in range(1, chain + 1):
            res = minimize(objective, x, method="L-BFGS-B",
                           bounds=model_bounds(n_layers),
                           options=(opts_first if leg == 1 else opts_later))
            x = res.x
            total_nfev += int(res.nfev)
            _, _, _, c_hat, _ = evaluate_model(x, kind, n_layers, weights)
            nested, _ = masked_rmse_linf(c_hat, bundle.nested)
            history.append({
                "leg": leg,
                "objective": float(res.fun),
                "objective_gap": float(res.fun) - ref_objective,
                "nested_rmse": nested,
                "nfev": int(res.nfev),
                "message": str(res.message),
                "grad": _optimizer_diagnostics(res)["gradient_inf_norm"],
            })

        elapsed = time.perf_counter() - t0
        first, last = history[0], history[-1]
        row = {
            "seed": seed,
            "f_first": first["objective"],
            "f_final": last["objective"],
            "improvement": first["objective"] - last["objective"],
            "relative_improvement": (
                (first["objective"] - last["objective"])
                / max(abs(first["objective"]), 1e-300)
            ),
            "nested_first": first["nested_rmse"],
            "nested_final": last["nested_rmse"],
            "gap_first": first["objective_gap"],
            "gap_final": last["objective_gap"],
            "nfev_first": first["nfev"],
            "nfev_total": total_nfev,
            "legs": chain,
            "restart_ftol": (float(restart_ftol) if restart_ftol is not None
                             else float(OPT_STUDY["ftol"])),
            "message_first": first["message"],
            "grad_first": first["grad"],
            "grad_final": last["grad"],
            "elapsed_seconds": elapsed,
        }
        for h in history:
            row[f"f_leg{h['leg']}"] = h["objective"]
            row[f"nested_leg{h['leg']}"] = h["nested_rmse"]
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.attrs["reference_objective"] = ref_objective
    frame.attrs["reference_nested_rmse"] = ref_nested
    return frame


def method_comparison(
    *,
    target: str = "triangle",
    kind: str = "equivariant",
    n_layers: int = 3,
    decay_p: float = 0.0,
    delta: float = 0.06,
    alpha: float = 1e-8,
    seeds=range(20),
    powell_budget: int = 500_000,
) -> pd.DataFrame:
    """Run the same cell under TNC and Powell."""
    bundle, weights, reference, ref_nested, ref_objective = _setup(
        target, decay_p, delta, alpha
    )

    def objective(params: np.ndarray) -> float:
        _, _, _, coefficients, _ = evaluate_model(
            params, kind, n_layers, weights
        )
        return regularized_loss(coefficients, bundle.train, alpha)

    methods = {
        "L-BFGS-B(study)": dict(method="L-BFGS-B", options=OPT_STUDY),
        "L-BFGS-B(tight)": dict(method="L-BFGS-B",
                                options={**OPT_STUDY, "ftol": 1e-18}),
        "TNC": dict(method="TNC",
                    options=dict(maxfun=powell_budget, ftol=1e-18,
                                 xtol=1e-14, gtol=1e-14)),
        "Powell": dict(method="Powell",
                       options=dict(maxiter=powell_budget,
                                    maxfev=powell_budget,
                                    ftol=1e-14, xtol=1e-14)),
    }

    rows = []
    for label, cfg in methods.items():
        for seed in (int(s) for s in seeds):
            t0 = time.perf_counter()
            res = minimize(objective, random_params(n_layers, seed),
                           method=cfg["method"],
                           bounds=model_bounds(n_layers),
                           options=cfg["options"])
            elapsed = time.perf_counter() - t0
            _, _, _, c_hat, _ = evaluate_model(res.x, kind, n_layers, weights)
            nested, _ = masked_rmse_linf(c_hat, bundle.nested)
            rows.append({
                "optimizer": label,
                "seed": seed,
                "objective": float(res.fun),
                "objective_gap": float(res.fun) - ref_objective,
                "nested_rmse": nested,
                "reached_ceiling": bool(float(res.fun) - ref_objective < 1e-12),
                "elapsed_seconds": elapsed,
                **_optimizer_diagnostics(res),
            })
    return pd.DataFrame(rows)



def _sweep_rows(args, ftol: float) -> pd.DataFrame:
    """Rows of ftol_sweep.csv for this cell at a given tolerance, by seed."""
    path = os.path.join("results", "precision_plateau_diagnostic",
                        "ftol_sweep.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    sweep = pd.read_csv(path)
    sweep["ftol"] = sweep["ftol"].astype(float)
    sel = sweep[
        (sweep.preconditioner == ("p0" if args.decay_p == 0.0 else "matched"))
        & (sweep.n_layers == args.n_layers)
        & (np.isclose(sweep.ftol, ftol, rtol=1e-6, atol=0.0))
    ]
    return sel[["seed", "objective", "nested_rmse"]]


def _check_reproduces_sweep(idem: pd.DataFrame, args) -> None:
    """Verify the first leg matches the published ftol=1e-12 rows.

    The idempotence test is only meaningful if its first leg IS the run the
    paper reports. Any difference in optimiser options changes the
    trajectory, so this compares against ftol_sweep.csv seed by seed.
    """
    ref = _sweep_rows(args, 1e-12)
    if ref.empty:
        print("\n[check skipped] no matching rows in ftol_sweep.csv")
        return

    merged = idem.merge(ref, on="seed", how="inner", suffixes=("", "_sweep"))
    if merged.empty:
        print("\n[check skipped] no overlapping seeds")
        return

    rel = (merged.f_first - merged.objective).abs() / merged.objective.abs()
    worst = float(rel.max())
    print(f"\nfirst leg vs ftol_sweep.csv at ftol=1e-12: "
          f"{len(merged)} seeds, worst relative difference {worst:.2e}")
    if worst > 1e-6:
        raise AssertionError("first leg does not reproduce ftol_sweep.csv; check OPT_STUDY")
    else:
        print("  first leg reproduces ftol_sweep.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/optimizer_cross_check")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--decay-p", type=float, default=0.0)
    ap.add_argument("--chain", type=int, default=2,
                    help="successive restarts; 2 means one restart")
    ap.add_argument("--restart-ftol", type=float, default=None,
                    help="ftol for legs after the first; omit for the plain "
                         "idempotence test, 1e-18 for the criterion probe")
    ap.add_argument("--methods", action="store_true",
                    help="also run the caveated TNC/Powell comparison (slow)")
    ap.add_argument("--powell-budget", type=int, default=500_000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = range(args.seeds)

    idem = idempotence_test(n_layers=args.n_layers, decay_p=args.decay_p,
                            seeds=seeds, chain=args.chain,
                            restart_ftol=args.restart_ftol)
    # Distinct filename per restart tolerance, so the criterion probe never
    # overwrites the plain idempotence result the notebook reads.
    name = ("idempotence_test.csv" if args.restart_ftol is None
            else f"idempotence_test_ftol{args.restart_ftol:.0e}.csv")
    path = safe_write_csv(idem, os.path.join(args.output_dir, name))
    print(f"wrote {path}\n")

    cols = ["seed", "f_first", "f_final", "improvement",
            "nested_first", "nested_final", "nfev_first", "nfev_total"]
    print(idem[cols].to_string(index=False))

    _check_reproduces_sweep(idem, args)

    med_imp = float(idem.improvement.median())
    # The claim at issue is about the reported metric, so the threshold is
    # stated on the nested RMSE, not on the objective. A relative objective
    # change of 1e-6 is floating-point noise at these magnitudes.
    moved = idem.nested_first - idem.nested_final > 0.1 * idem.nested_first
    frac_moved = float(moved.mean())
    print(f"\nmedian objective improvement on restart: {med_imp:.3e}")
    print(f"median nested RMSE   {idem.nested_first.median():.3e}"
          f"  ->  {idem.nested_final.median():.3e}")
    print(f"restarts improving nested RMSE by >10%:  {frac_moved:.0%}")
    print(f"median extra evaluations on restart:     "
          f"{float((idem.nfev_total - idem.nfev_first).median()):.0f}")

    if args.restart_ftol is not None:
        tight = _sweep_rows(args, args.restart_ftol).rename(
            columns={"nested_rmse": "nested_continuous"})
        if tight.empty:
            print(f"\n[comparison skipped] no ftol_sweep rows at "
                  f"{args.restart_ftol:.0e}; reporting the restart medians only")
            print(f"median nested RMSE after restart: "
                  f"{idem.nested_final.median():.3e}")
            return

        # Paired across the same seeds: each restart is compared with the
        # continuous tight run from the same initialisation, which is a
        # stronger statement than median against median.
        m = idem.merge(tight[["seed", "nested_continuous"]], on="seed")
        m["ratio"] = m.nested_final / m.nested_continuous
        ratio = float(m.ratio.median())

        print(f"\npaired over {len(m)} seeds, restart at "
              f"ftol={args.restart_ftol:.0e} versus the continuous run at the "
              f"same tolerance:")
        print(f"  median nested RMSE, restarted : {m.nested_final.median():.3e}")
        print(f"  median nested RMSE, continuous: "
              f"{m.nested_continuous.median():.3e}")
        print(f"  median of the per-seed ratio  : {ratio:.2f}")
        print(f"  restarts within 2x of continuous: "
              f"{float((m.ratio < 2).mean()):.0%}")
        return


    if args.methods:
        print("\nrunning TNC and Powell ...")
        mc = method_comparison(n_layers=args.n_layers, decay_p=args.decay_p,
                               seeds=seeds, powell_budget=args.powell_budget)
        p2 = safe_write_csv(mc, os.path.join(args.output_dir,
                                             "method_comparison.csv"))
        print(f"wrote {p2}\n")
        print(mc.groupby("optimizer").agg(
            median_nested_rmse=("nested_rmse", "median"),
            median_objective_gap=("objective_gap", "median"),
            median_nfev=("nfev", "median"),
        ).to_string())


if __name__ == "__main__":
    main()
"""Constructed calibration cases for the optimiser and reachability verdicts.

Every case here reuses the principal experiment's data (triangle, odd sector),
regularisation (alpha = 1e-8), optimiser settings (sector.OPTIONS, maxiter 3000)
and stopping-scale check, with a classical parameterisation whose cause of
shortfall is fixed by construction:

optimiser_absolute   c_odd = D r, D diagonal with condition number kappa. The
                     problem is convex and D is invertible, so the optimum is
                     reachable (theorem); any shortfall is optimisation. The raw
                     objective is far below one, so the stopping test is absolute
                     and the scale check fails, as in the principal experiment.
optimiser_relative   the same, with the objective divided by its value at the
                     optimum, so the scale check PASSES and the normalised rerun
                     changes nothing; the shortfall is still optimisation, by the
                     same theorem. Only the inverse fit (component 4) can find it.
reachability_near    c_odd = u * tanh(r): every coefficient bounded by u, with u
                     set so that an unregularised bounded least-squares solve
                     certifies a training-RMSE floor of FLOOR_TARGET times the
                     class's least-squares RMSE, just beyond GAP_TOL.

The inverse fit (component 4) is executed exactly as in the principal
experiment (precision_plateau_diagnostic.py): L-BFGS-B on ||c(theta) - c_ref||^2
with numerical gradients, ftol 1e-18, gtol 1e-14, 40 restarts, best nested RMSE. The shortfall is a limit of the reachable set
                     (certified), near the rules' boundary.

Parameters are chosen by rules fixed before any case is attributed, depending
only on each construction's shortfall, never on a verdict:
  kappa  the smallest decade 10^2, 10^3, ... at which the median nested-RMSE
         shortfall at the study settings exceeds GAP_TOL;
  u      bisection to FLOOR_TARGET = 1.25.

Usage: python src/constructed_cases.py --output-dir results/constructed_cases
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attribution_rules as rules  # noqa: E402
import sector_graded_evalmod as S  # noqa: E402

ALPHA, DELTA, MAXITER = 1e-8, 0.06, 3000
SEEDS = range(20)
KAPPA_DECADES = range(2, 13)
FLOOR_TARGET = 1.25                 # fixed before any case is attributed
TARGET = "triangle"


class Problem:
    def __init__(self):
        self.b = S.build_bundles(DELTA)[TARGET]
        self.cols = S.SECTOR_COLUMNS["odd"]
        self.V, self.y = self.b.train.design_masked, self.b.train.target_masked
        self.m = len(self.y)
        self.cstar = S.ridge_fit(self.b.train, ALPHA, "odd")
        self.f_ref = S.regularized_loss(self.cstar, self.b.train, ALPHA)
        self.class_opt = self.nested(self.cstar)
        self.full_opt = self.nested(S.ridge_fit(self.b.train, ALPHA, "full"))

    def coeffs(self, c_odd):
        c = np.zeros(S.N_HARMONICS)
        c[self.cols] = c_odd
        return c

    def nested(self, c):
        return S.masked_rmse_linf(c, self.b.nested)[0]

    def train_rmse(self, c):
        return S.masked_rmse_linf(c, self.b.train)[0]

    def fg_c(self, c):
        """Ridge objective and its gradient in coefficient space."""
        e = self.V @ c - self.y
        return (np.mean(e ** 2) + ALPHA * c @ c,
                (2 * self.V.T @ e / self.m + 2 * ALPHA * c)[self.cols])


def fit(p: Problem, kind: str, param: float, seed: int, options: dict, scale: float = 1.0):
    """Train one restart; returns (coefficients, converged)."""
    rng = np.random.default_rng(seed)
    if kind == "diag":
        d = np.logspace(0, -np.log10(param), len(p.cols))
        x0 = rng.normal(0, 0.3, len(p.cols)) / d
        to_c = lambda r: d * r                                   # noqa: E731
        dc = lambda r: d                                          # noqa: E731
    else:                                                         # bounded envelope
        u = param
        x0 = rng.normal(0, 1.0, len(p.cols))
        to_c = lambda r: u * np.tanh(r)                          # noqa: E731
        dc = lambda r: u / np.cosh(r) ** 2                       # noqa: E731

    def fg(r):
        f, g = p.fg_c(p.coeffs(to_c(r)))
        return f / scale, dc(r) * g / scale

    res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": MAXITER, **options})
    return p.coeffs(to_c(res.x)), bool(str(res.message).startswith("CONVERGENCE"))


INVERSE_OPTIONS = {"ftol": 1e-18, "gtol": 1e-14, "maxls": 100, "maxfun": 1_000_000}
INVERSE_SEEDS, INVERSE_MAXITER = range(40), 20_000


def inverse_fit(p, kind, param):
    """Executed inverse fit to the class optimum, as in the principal experiment;
    returns the best nested RMSE over the restarts."""
    best = np.inf
    for seed in INVERSE_SEEDS:
        rng = np.random.default_rng(seed)
        if kind == "diag":
            d = np.logspace(0, -np.log10(param), len(p.cols))
            x0 = rng.normal(0, 0.3, len(p.cols)) / d
            to_c = lambda r: d * r                               # noqa: E731
        else:
            x0 = rng.normal(0, 1.0, len(p.cols))
            to_c = lambda r: param * np.tanh(r)                  # noqa: E731
        target = p.cstar[p.cols]
        res = minimize(lambda r: float(np.sum((to_c(r) - target) ** 2)), x0, method="L-BFGS-B",
                       options={"maxiter": INVERSE_MAXITER, **INVERSE_OPTIONS})
        best = min(best, p.nested(p.coeffs(to_c(res.x))))
    return best


def arm(p, kind, param, options, scale=1.0):
    out = [fit(p, kind, param, s, options, scale) for s in SEEDS]
    return np.array([p.nested(c) for c, _ in out]), np.array([ok for _, ok in out])


def normalised_options(f_ref_of_objective):
    return {**S.OPTIONS, "ftol": S.OPTIONS["ftol"] * f_ref_of_objective,
            "gtol": S.OPTIONS["gtol"] * f_ref_of_objective}


def choose_kappa(p, scale):
    """Smallest decade at which the median shortfall exceeds GAP_TOL."""
    for k in KAPPA_DECADES:
        vals, _ = arm(p, "diag", 10.0 ** k, S.OPTIONS, scale)
        if np.median(vals) > (1 + rules.GAP_TOL) * p.class_opt:
            return 10.0 ** k
    raise RuntimeError("no decade gives a shortfall")


def certified_floor(p, u):
    """Training-RMSE floor of the bounded class: an unregularised bounded
    least-squares solve, which minimises training RMSE over the box exactly and
    so bounds every model in it, divided by the class's least-squares RMSE."""
    Vo = p.V[:, p.cols] / np.sqrt(p.m)
    sol = lsq_linear(Vo, p.y / np.sqrt(p.m), bounds=(-u, u), tol=1e-14, lsmr_tol="auto")
    ls = p.coeffs(np.linalg.lstsq(Vo, p.y / np.sqrt(p.m), rcond=None)[0])
    return p.train_rmse(p.coeffs(sol.x)) / p.train_rmse(ls), sol.x


def choose_u(p):
    lo, hi = 1e-6, float(np.max(np.abs(p.cstar[p.cols])))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        ratio, _ = certified_floor(p, mid)
        lo, hi = (mid, hi) if ratio > FLOOR_TARGET else (lo, mid)
    return hi


def scale_check_fails(objective_scale):
    """P7 on the objective actually minimised (raw or divided by objective_scale)."""
    c = p_global.cstar
    fit_term = float(np.mean((p_global.V @ c - p_global.y) ** 2)) / objective_scale
    f_ref = p_global.f_ref / objective_scale
    threshold = S.OPTIONS["ftol"] * max(1.0, abs(f_ref))
    return threshold / fit_term > S.STOPPING_SCALE_LIMIT


def evidence(p, kind, param, objective_scale, null_result):
    study_vals, study_ok = arm(p, kind, param, S.OPTIONS, objective_scale)
    norm_vals, _ = arm(p, kind, param,
                       normalised_options(p.f_ref / objective_scale), objective_scale)
    return rules.Evidence(
        observed=float(np.median(study_vals)), reference="class",
        optimizer_success=rules.optimizer_success_rule(study_ok),
        restarts_agree=rules.restarts_agree_rule(study_vals),
        class_optimum=p.class_opt, unrestricted_optimum=p.full_opt,
        null_result=null_result, inverse_best=inverse_fit(p, kind, param),
        scale_check_fails=scale_check_fails(objective_scale),
        normalised_result=float(np.median(norm_vals))), study_vals


WITHHOLD = [(), (1,), (2,), (3,), (4,), (5,), (6,), (4, 5)]


def main():
    global p_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=os.path.join("results", "constructed_cases"))
    args = ap.parse_args()
    p = p_global = Problem()
    null = float(np.median([p.nested(S.spherical_fit("spherical_odd", p.b.train, ALPHA, s, MAXITER)[0])
                            for s in SEEDS]))

    k_abs = choose_kappa(p, 1.0)
    k_rel = choose_kappa(p, p.f_ref)
    u = choose_u(p)
    floor_ratio, floor_x = certified_floor(p, u)

    cases = {
        "optimiser_absolute": (*evidence(p, "diag", k_abs, 1.0, null), "optimiser",
                               {"kappa": k_abs}),
        "optimiser_relative": (*evidence(p, "diag", k_rel, p.f_ref, null), "optimiser",
                               {"kappa": k_rel}),
        "reachability_near": (*evidence(p, "tanh", u, 1.0, null), "reachability",
                              {"u": u, "certified_floor_ratio_train": floor_ratio,
                               "floor_nested_over_class_optimum": p.nested(p.coeffs(floor_x)) / p.class_opt}),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for name, (e, vals, expected, params) in cases.items():
        verdicts = {("all" if not c else "without " + "+".join(map(str, c))):
                    rules.attribute(_withheld(e, c)) for c in WITHHOLD}
        rows.append({"case": name, "expected": expected, "params": params,
                     "shortfall_median": e.observed / e.class_optimum,
                     "shortfall_iqr": [float(np.percentile(vals, 25) / e.class_optimum),
                                       float(np.percentile(vals, 75) / e.class_optimum)],
                     "evidence": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                  for k, v in e.__dict__.items()},
                     "verdicts": verdicts, "correct": verdicts["all"] == expected})
    with open(os.path.join(args.output_dir, "constructed_cases.json"), "w") as fh:
        json.dump({"rules_version": rules.RULES_VERSION, "cases": rows}, fh, indent=1, default=float)
    for r in rows:
        print(f"{r['case']:20s} params {r['params']}  shortfall x{r['shortfall_median']:.3g} "
              f"(IQR {r['shortfall_iqr'][0]:.3g}-{r['shortfall_iqr'][1]:.3g})  expected {r['expected']}")
        print("   " + "  ".join(f"{k}: {v}" for k, v in r["verdicts"].items()))
    print(f"\ncorrect with all components: {sum(r['correct'] for r in rows)} of {len(rows)} "
          f"(rules version {rules.RULES_VERSION})")


def _withheld(e, comps):
    for k in comps:
        e = rules.withhold(e, k)
    return e


if __name__ == "__main__":
    main()
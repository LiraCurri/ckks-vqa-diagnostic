"""The dimension-matched null under the settings of the gradient cross-check.

gradient_cross_check.py reruns the depth-three equivariant cell on the triangle
with the stopping test tightened and the gradient supplied exactly. This script
does the same for the cell's classical null (the spherical odd-sector
parameterisation), so that the two are compared under equal settings:

    fd      SciPy's forward-difference gradient, as in the principal study
    exact   the analytic gradient of c = e^s r / ||r||

at ftol = 1e-12 and 1e-18, with every other setting that of the tolerance sweep
(DIAGNOSTIC_TIGHT, maxiter 50,000) and the starting points of the principal
study's null. The analytic gradient is verified against central differences
before any run.

Usage, from the repository root (a few seconds):
    python src/null_cross_check.py --output-dir results/gradient_cross_check
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sector_graded_evalmod as sge  # noqa: E402
from gradient_cross_check import ALPHA, DELTA, FTOLS, MAXITER, REACHED_GAP, TARGET  # noqa: E402
from precision_plateau_diagnostic import DIAGNOSTIC_TIGHT  # noqa: E402

MODE = "spherical_odd"
N_DIR = 8
BOUNDS = [(-5.0, 5.0)] * N_DIR + [(-8.0, 8.0)]      # as in sge.spherical_fit


class Null:
    def __init__(self):
        self.bundle = sge.build_bundles(DELTA)[TARGET]
        self.V = self.bundle.train.design_masked
        self.y = self.bundle.train.target_masked
        self.m = len(self.y)
        self.cols = np.asarray(sge.SECTOR_COLUMNS["odd"])
        self.reference = sge.ridge_fit(self.bundle.train, ALPHA, "odd")
        self.f_ref = sge.regularized_loss(self.reference, self.bundle.train, ALPHA)

    def objective(self, p: np.ndarray) -> float:
        return sge.regularized_loss(sge.spherical_coefficients(p, MODE), self.bundle.train, ALPHA)

    def objective_and_gradient(self, p: np.ndarray):
        raw, s = p[:N_DIR], p[N_DIR]
        n = np.linalg.norm(raw)
        u = raw / n
        c = np.zeros(sge.N_HARMONICS)
        c[self.cols] = np.exp(s) * u
        r = self.V @ c - self.y
        value = float(np.mean(r ** 2)) + ALPHA * float(c @ c)
        gc = (2.0 * self.V.T @ r / self.m + 2.0 * ALPHA * c)[self.cols]
        grad = np.empty(N_DIR + 1)
        grad[:N_DIR] = np.exp(s) * (gc - u * (u @ gc)) / n
        grad[N_DIR] = gc @ c[self.cols]
        return value, grad


def start(seed: int) -> np.ndarray:
    """The starting point sge.spherical_fit draws for this seed."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(size=N_DIR), rng.normal(0.0, 0.25, size=1)])


def validate_gradient(null: Null, n_points: int = 3, seed: int = 99, h: float = 1e-6,
                      atol: float = 1e-7) -> float:
    worst = 0.0
    for k in range(n_points):
        p = start(seed + k)
        _, g = null.objective_and_gradient(p)
        num = np.array([(null.objective(p + d) - null.objective(p - d)) / (2.0 * h)
                        for d in h * np.eye(len(p))])
        worst = max(worst, float(np.max(np.abs(g - num))))
    assert worst < atol, f"analytic gradient disagrees with central differences by {worst:.2e}"
    return worst


def run_one(null: Null, gradient: str, ftol: float, seed: int) -> dict:
    options = {"maxiter": MAXITER, **DIAGNOSTIC_TIGHT, "ftol": float(ftol)}
    if gradient == "exact":
        res = minimize(null.objective_and_gradient, start(seed), jac=True, method="L-BFGS-B",
                       bounds=BOUNDS, options=options)
    else:
        res = minimize(null.objective, start(seed), method="L-BFGS-B", bounds=BOUNDS, options=options)
    c = sge.spherical_coefficients(res.x, MODE)
    value = sge.regularized_loss(c, null.bundle.train, ALPHA)
    return {"gradient": gradient, "ftol": float(ftol), "seed": int(seed),
            "objective": value, "objective_gap": value - null.f_ref,
            "reached": bool(value - null.f_ref < REACHED_GAP),
            "nested_rmse": sge.masked_rmse_linf(c, null.bundle.nested)[0],
            "nit": int(res.nit), "nfev": int(res.nfev), "message": str(res.message),
            "reference_objective": null.f_ref}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--output-dir", default=os.path.join("results", "gradient_cross_check"))
    args = ap.parse_args()

    null = Null()
    print(f"analytic gradient vs central differences: worst error {validate_gradient(null):.2e}")
    frame = pd.DataFrame([run_one(null, g, f, s)
                          for f in FTOLS for g in ("fd", "exact") for s in range(args.seeds)])
    os.makedirs(args.output_dir, exist_ok=True)
    path = sge.safe_write_csv(frame, os.path.join(args.output_dir, "null_cross_check.csv"))
    print(f"wrote {path}\n")
    print(frame.groupby(["ftol", "gradient"], sort=False).agg(
        runs=("seed", "size"), reached=("reached", "sum"),
        median_nested=("nested_rmse", "median")).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()

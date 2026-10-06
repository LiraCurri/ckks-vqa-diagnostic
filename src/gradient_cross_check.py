"""Gradient cross-check for the depth-three equivariant cell.

The principal study, the tolerance sweep and the inverse fit all call L-BFGS-B
without a gradient, so SciPy estimates it by forward differences. This script
reruns the tolerance sweep's cell (triangle, equivariant, unweighted decoder)
with the gradient supplied exactly, and compares the two seed by seed:

    fd      SciPy's forward-difference gradient, as in ftol_sweep_v7
    exact   the parameter-shift rule for the circuit angles and the analytic
            derivative for the decoder log-scales

at ftol = 1e-12 (the study tolerance) and 1e-18, with every other setting that
of the sweep (DIAGNOSTIC_TIGHT, maxiter 50,000) and the same starting points.

Every circuit parameter enters one gate whose generator has two eigenvalues one
apart (R_y, R_z, controlled phase), so each expectation is a + b cos(t) + c sin(t)
in that parameter and d<O>/dt = [<O>(t + pi/2) - <O>(t - pi/2)] / 2 exactly. The
rule is verified against central differences before any run.

Each row records the error of the forward-difference gradient at the endpoint
against the exact one, the decoder log-scales there, and whether a live
one sits on its bound, since a run held on the box [-8, 8] cannot be read as slow
convergence.

The fd arm must reproduce ftol_sweep.csv; this is checked seed by seed unless
--skip-reproduction-check is given (trajectories depend on the SciPy version).

Usage, from the repository root:
    python src/gradient_cross_check.py --output-dir results/gradient_cross_check
    python src/gradient_cross_check.py --jobs 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sector_graded_evalmod as sge  # noqa: E402
from precision_plateau_diagnostic import DIAGNOSTIC_TIGHT  # noqa: E402

SWEEP_CSV = os.path.join("results", "precision_plateau_diagnostic", "ftol_sweep.csv")
TARGET, KIND, DELTA, ALPHA = "triangle", "equivariant", 0.06, 1e-8
MAXITER = 50_000                    # ftol_sweep_v7 default
FTOLS = (1e-12, 1e-18)
BOUND_MARGIN = 0.1                  # a log-scale within this of +-8 counts as on its bound
SCALE_BOUND = 8.0
REACHED_GAP = 1e-12                 # the sweep's criterion for reaching the class optimum
FD_STEP = 1e-8                      # SciPy's default step for L-BFGS-B without a gradient


class Cell:
    """The fitting problem and both ways of differentiating it."""

    def __init__(self, n_layers: int, decay_p: float = 0.0):
        self.n_layers = int(n_layers)
        self.bundle = sge.build_bundles(DELTA)[TARGET]
        self.weights = sge.preconditioner(decay_p)
        self.V = self.bundle.train.design_masked
        self.y = self.bundle.train.target_masked
        self.m = len(self.y)
        self.nc = sge.n_circuit_params(self.n_layers)
        self.reference = sge.ridge_fit(self.bundle.train, ALPHA, sge.MODEL_SECTOR[KIND])
        self.f_ref = sge.regularized_loss(self.reference, self.bundle.train, ALPHA)
        self.ref_nested = sge.masked_rmse_linf(self.reference, self.bundle.nested)[0]

    def objective(self, params: np.ndarray) -> float:
        c = sge.evaluate_model(params, KIND, self.n_layers, self.weights)[3]
        return sge.regularized_loss(c, self.bundle.train, ALPHA)

    def _expectations(self, theta: np.ndarray) -> np.ndarray:
        e = sge.exact_expectations(sge.run_ansatz(theta, KIND, self.n_layers)).copy()
        if KIND == "equivariant":
            e[sge.EVEN_SLICE] = 0.0
        return e

    def objective_and_gradient(self, params: np.ndarray):
        theta = np.asarray(params[:self.nc], float)
        scales = np.asarray(params[self.nc:], float)
        e = self._expectations(theta)
        c = sge.decode(e, scales, self.weights)
        r = self.V @ c - self.y
        value = float(np.mean(r ** 2)) + ALPHA * float(c @ c)
        grad_c = 2.0 * self.V.T @ r / self.m + 2.0 * ALPHA * c
        grad = np.empty(len(params))
        for i in range(self.nc):                       # parameter-shift rule
            shift = np.zeros(self.nc)
            shift[i] = np.pi / 2.0
            de = 0.5 * (self._expectations(theta + shift) - self._expectations(theta - shift))
            grad[i] = grad_c @ sge.decode(de, scales, self.weights)
        for g in range(sge.n_decoder_scales()):        # dc_k/ds_g = c_k on family g
            eg = np.where(sge.FAMILY_INDEX == g, e, 0.0)
            grad[self.nc + g] = grad_c @ sge.decode(eg, scales, self.weights)
        return value, grad


def validate_gradient(cell: Cell, n_points: int = 3, seed: int = 99, h: float = 1e-5,
                      atol: float = 1e-9) -> float:
    """The exact gradient against central differences; returns the worst error."""
    worst = 0.0
    for k in range(n_points):
        p = sge.random_params(cell.n_layers, seed + k)
        _, g = cell.objective_and_gradient(p)
        num = np.empty(len(p))
        for i in range(len(p)):
            d = np.zeros(len(p))
            d[i] = h
            num[i] = (cell.objective(p + d) - cell.objective(p - d)) / (2.0 * h)
        worst = max(worst, float(np.max(np.abs(g - num))))
    assert worst < atol, f"exact gradient disagrees with central differences by {worst:.2e}"
    return worst


def run_one(task) -> dict:
    gradient, ftol, seed, n_layers, decay_p = task
    cell = Cell(n_layers, decay_p)
    options = {"maxiter": MAXITER, **DIAGNOSTIC_TIGHT, "ftol": float(ftol)}
    x0 = sge.random_params(n_layers, seed)
    t0 = time.perf_counter()
    if gradient == "exact":
        res = minimize(cell.objective_and_gradient, x0, jac=True, method="L-BFGS-B",
                       bounds=sge.model_bounds(n_layers), options=options)
    else:
        res = minimize(cell.objective, x0, method="L-BFGS-B",
                       bounds=sge.model_bounds(n_layers), options=options)
    seconds = time.perf_counter() - t0
    c = sge.evaluate_model(res.x, KIND, n_layers, cell.weights)[3]
    value, grad = cell.objective_and_gradient(res.x)
    scales = np.asarray(res.x[cell.nc:], float)
    # What SciPy's default estimate (forward differences, step 1e-8) returns at
    # the endpoint, against the exact gradient there.
    fd_grad = approx_fprime(res.x, cell.objective, FD_STEP)
    live = sorted({int(sge.FAMILY_INDEX[i]) for i in sge.live_observable_indices(KIND)})
    row = {
        "gradient": gradient, "ftol": float(ftol), "seed": int(seed), "n_layers": int(n_layers),
        "decay_p": float(decay_p),
        "nested_rmse": sge.masked_rmse_linf(c, cell.bundle.nested)[0],
        "train_rmse": sge.masked_rmse_linf(c, cell.bundle.train)[0],
        "objective": value, "objective_gap": value - cell.f_ref,
        "objective_over_optimum": value / cell.f_ref,
        "reached": bool(value - cell.f_ref < REACHED_GAP),
        "exact_gradient_inf_norm": float(np.max(np.abs(grad))),
        "fd_gradient_error_inf": float(np.max(np.abs(fd_grad - grad))),
        "fd_gradient_error_over_gradient": float(np.max(np.abs(fd_grad - grad))
                                                 / max(float(np.max(np.abs(grad))), 1e-300)),
        "nit": int(res.nit), "nfev": int(res.nfev), "message": str(res.message),
        "min_live_log_scale": float(np.min(scales[live])),
        "max_live_log_scale": float(np.max(scales[live])),
        "scale_on_bound": bool(np.any(np.abs(scales[live]) >= SCALE_BOUND - BOUND_MARGIN)),
        "reference_objective": cell.f_ref, "reference_nested_rmse": cell.ref_nested,
        "seconds": seconds,
    }
    for g, name in enumerate(sge.FAMILY_NAMES):
        row[f"log_scale_{name}"] = float(scales[g])
    return row


def check_reproduces_sweep(frame: pd.DataFrame, n_layers: int, decay_p: float) -> float | None:
    """The fd arm must match ftol_sweep.csv seed by seed; returns the worst difference."""
    if not os.path.exists(SWEEP_CSV):
        print(f"[reproduction check skipped] {SWEEP_CSV} not found")
        return None
    sw = pd.read_csv(SWEEP_CSV)
    sw = sw[(sw.preconditioner == ("p0" if decay_p == 0.0 else "matched")) & (sw.n_layers == n_layers)]
    worst = 0.0
    for ftol in FTOLS:
        ref = sw[np.isclose(sw.ftol.astype(float), ftol, rtol=1e-6, atol=0.0)].set_index("seed").nested_rmse
        got = frame[(frame.gradient == "fd") & np.isclose(frame.ftol, ftol, rtol=1e-6, atol=0.0)
                    ].set_index("seed").nested_rmse
        common = got.index.intersection(ref.index)
        if not len(common):
            raise AssertionError(f"no sweep rows at ftol={ftol:g} to compare against")
        worst = max(worst, float(np.max(np.abs(got[common] / ref[common] - 1.0))))
    return worst


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["ftol", "gradient"], sort=False).agg(
        runs=("seed", "size"),
        median_nested=("nested_rmse", "median"),
        max_over_min=("nested_rmse", lambda v: float(v.max() / v.min())),
        reached=("reached", "sum"),
        not_reached_on_bound=("scale_on_bound", lambda v: int((v & ~frame.loc[v.index, "reached"]).sum())),
        median_nit=("nit", "median"),
        median_exact_gradient=("exact_gradient_inf_norm", "median")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--decay-p", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-reproduction-check", action="store_true",
                    help="report, rather than require, agreement of the fd arm with ftol_sweep.csv")
    ap.add_argument("--output-dir", default=os.path.join("results", "gradient_cross_check"))
    args = ap.parse_args()

    cell = Cell(args.n_layers, args.decay_p)
    print(f"exact gradient vs central differences: worst error {validate_gradient(cell):.2e}")
    tasks = [(g, f, s, args.n_layers, args.decay_p)
             for f in FTOLS for g in ("fd", "exact") for s in range(args.seeds)]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(run_one, tasks))
    else:
        rows = [run_one(t) for t in tasks]
    frame = pd.DataFrame(rows)

    worst = check_reproduces_sweep(frame, args.n_layers, args.decay_p)
    if worst is not None:
        print(f"fd arm vs ftol_sweep.csv: worst relative difference {worst:.2e}")
        if worst > 1e-6 and not args.skip_reproduction_check:
            raise AssertionError("the fd arm does not reproduce ftol_sweep.csv; "
                                 "nothing is written (--skip-reproduction-check to override)")
    frame["reproduces_sweep"] = bool(worst is not None and worst <= 1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    path = sge.safe_write_csv(frame, os.path.join(args.output_dir, "gradient_cross_check.csv"))
    print(f"wrote {path}\n")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3e}"):
        print(summarise(frame).to_string(index=False))


if __name__ == "__main__":
    main()

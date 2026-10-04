"""Executed ablation: evidence records for the stored cases, attributed by the frozen rules.

Builds one attribution_rules.Evidence record per case from stored results (no
optimisation runs here), applies attribute() with all components and with each
component withheld, and writes the ablation table.

Cases, and how the cause of each is known
-----------------------------------------
triangle_plateau      equivariant, L=3, triangle. Cause DIAGNOSED by component 5
                      (normalised stopping), so not independent ground truth.
prior_sawtooth        equivariant, L=3, against the full-space optimum. Cause
prior_double_sawtooth fixed by THEOREM: the selection rule confines the model to
                      the odd sector, whose optimum is a linear solve.
readout_<target>_B    equivariant, L=3, sawtooth and double sawtooth, every shot
                      budget B, against the class optimum. Cause fixed by
                      CONSTRUCTION: sampling noise injected on parameters whose
                      exact-expectation fit is already at the class optimum. The
                      expected verdict is set from the variance formula before
                      any sample is read: "readout" if the predicted shot error
                      takes the result beyond GAP_TOL of the optimum, "none"
                      otherwise (the "none" cells are the negative controls).
compound_triangle_B   the triangle shot cells. Two causes compound: the stopping
                      rule leaves the exact fit above the optimum, and shot noise
                      adds a larger error on top. Expected label "compound"; the
                      frozen rules return a single verdict, the first stage in
                      their order whose test passes, not the dominant one.
generic_<target>      generic, L=3, sawtooth and double sawtooth reruns. Cause
                      DIAGNOSED by component 5, as for the plateau.
representation_limit  frozen decoder envelope (representation_limit_case.py).
                      Cause fixed by CONSTRUCTION: a certified gap.

The expected verdict for diagnosed cases is the diagnosis, and is marked as such;
only theorem and construction cases are independent of the protocol, and the
summary reports the two counts separately. Compound cases are reported, not
scored.

Usage, from the repository root:
    python src/ablation_records.py --output-dir results/ablation
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attribution_rules as rules  # noqa: E402
from sector_graded_evalmod import SECTOR_COLUMNS, build_bundles, safe_write_csv  # noqa: E402

SECTOR = os.path.join("results", "sector_results_full", "full_p0")
OBJSCALE = os.path.join("results", "objective_scale")
PLATEAU = os.path.join("results", "precision_plateau_diagnostic")
REPLIMIT = os.path.join("results", "representation_limit", "representation_limit_summary.json")

N_LAYERS = 3
DELTA = 0.06                       # mask half-width of the sector study
PAPER_PLATEAU = 4.794e-5           # paper's median nested RMSE, triangle, L=3 (check)
WITHHOLD = [(), (1,), (2,), (3,), (4,), (5,), (6,), (4, 5)]


def load() -> dict:
    return {
        "exact": pd.read_csv(os.path.join(SECTOR, "sector_exact_results.csv")),
        "shots": pd.read_csv(os.path.join(SECTOR, "sector_shot_results.csv")),
        "checks": pd.read_csv(os.path.join(SECTOR, "prediction_checks.csv")).set_index("prediction"),
        "inverse": pd.read_csv(os.path.join(PLATEAU, "inverse_problem.csv")),
    }


def optimum(exact: pd.DataFrame, target: str, sector: str) -> float:
    """Nested RMSE of the ridge optimum over a sector (the objective-aligned reference)."""
    v = exact[(exact.target == target) & (exact.method == f"ridge_{sector}")].nested_rmse
    if v.nunique() != 1:
        raise AssertionError(f"ridge_{sector} on {target}: rows disagree across depths")
    return float(v.iloc[0])


def cell(exact: pd.DataFrame, target: str, method: str, n_layers: int | None = N_LAYERS):
    sel = (exact.target == target) & (exact.method == method)
    if n_layers is not None:
        sel &= exact.n_layers == n_layers
    g = exact[sel & (exact["init"] != "warm")]   # as in check_predictions
    if method.startswith("spherical"):
        g = g.drop_duplicates("seed")            # nulls are repeated once per depth
    if g.empty:
        raise AssertionError(f"no rows for {method} on {target}")
    return g


def scale_check_fails(checks: pd.DataFrame, target: str, sector: str) -> bool:
    status = checks.loc[f"P7_stopping_threshold_scale_{target}_{sector}", "status"]
    if status not in ("PASS", "FAIL"):
        raise AssertionError(f"P7 for {target}/{sector} has status {status}")
    return status == "FAIL"


def flags(values, converged) -> dict:
    return {"restarts_agree": rules.restarts_agree_rule(values),
            "optimizer_success": rules.optimizer_success_rule(converged)}


def objective_scale_frame(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(OBJSCALE, name))


def triangle_plateau(d: dict):
    ex, t = d["exact"], "triangle"
    g = cell(ex, t, "equivariant")
    os_ = objective_scale_frame("objective_scale_results.csv")
    inv = d["inverse"]
    inv = inv[(inv.target == t) & (inv.preconditioner == "p0") & (inv.n_layers == N_LAYERS)
              & (inv.kind == "equivariant")]
    if inv.empty:
        raise AssertionError("no inverse-problem rows for the depth-three unweighted cell")
    e = rules.Evidence(
        observed=float(g.nested_rmse.median()), reference="class",
        **flags(g.nested_rmse, g.optimizer_success),
        class_optimum=optimum(ex, t, "odd"), unrestricted_optimum=optimum(ex, t, "full"),
        null_result=float(cell(ex, t, "spherical_odd", None).nested_rmse.median()),
        inverse_best=float(inv.nested_rmse.min()),
        scale_check_fails=scale_check_fails(d["checks"], t, "odd"),
        normalised_result=float(os_[os_.arm == "normalised"].nested_rmse.median()))
    if not np.isclose(e.observed, PAPER_PLATEAU, rtol=1e-3, atol=0.0):
        raise AssertionError(f"triangle plateau median {e.observed:.4e} does not reproduce "
                             f"the paper's {PAPER_PLATEAU:.3e}; the records are built wrong")
    return e, "optimiser", "diagnosed (component 5)"


def prior(d: dict, t: str):
    ex = d["exact"]
    g = cell(ex, t, "equivariant")
    e = rules.Evidence(
        observed=float(g.nested_rmse.median()), reference="full",
        **flags(g.nested_rmse, g.optimizer_success),
        class_optimum=optimum(ex, t, "odd"), unrestricted_optimum=optimum(ex, t, "full"),
        null_result=float(cell(ex, t, "spherical_odd", None).nested_rmse.median()),
        scale_check_fails=scale_check_fails(d["checks"], t, "odd"))
    return e, "prior", "theorem (selection rule, closed-form optimum)"


def predicted_shot_rmse(exact_rmse: float, predicted_noise: float) -> float:
    """Nested RMSE expected after adding coefficient noise of the predicted size.

    The variance formula gives E||dc||^2 = predicted_noise^2 over the live (odd)
    coefficients; on the nested grid this adds dc' G dc in expectation, with G the
    Gram matrix of the nested design, approximated by its mean odd diagonal. The
    cross term with the residual has zero mean. Uses no sampled value.
    """
    cols = SECTOR_COLUMNS["odd"]
    v = build_bundles(DELTA)["triangle"].nested.design_masked[:, cols]   # design only
    g_mean = float(np.mean(np.sum(v ** 2, axis=0) / v.shape[0]))
    return float(np.sqrt(exact_rmse ** 2 + g_mean * predicted_noise ** 2))


def shot_evidence(d: dict, t: str, budget: int, base_fields: dict) -> tuple:
    """Evidence for an equivariant L=3 shot cell, and the predicted shot RMSE."""
    ex, sh = d["exact"], d["shots"]
    g = cell(ex, t, "equivariant")
    s = sh[(sh.target == t) & (sh.method == "equivariant") & (sh.n_layers == N_LAYERS)
           & (sh.total_budget == budget) & (sh["init"] != "warm")]
    if s.empty:
        raise AssertionError(f"no shot rows for {t} at budget {budget}")
    per_seed = s.groupby("seed").nested_rmse.median()
    noise = s.groupby("seed").apply(
        lambda h: float(np.sqrt(np.mean(h.coefficient_noise_l2 ** 2))
                        / h.predicted_coefficient_noise.iloc[0]), include_groups=False)
    exact_at = float(g.nested_rmse.median())
    predicted = predicted_shot_rmse(
        exact_at, float(s.groupby("seed").predicted_coefficient_noise.first().median()))
    e = rules.Evidence(
        observed=float(per_seed.median()), reference="class",
        **flags(per_seed, g.optimizer_success),
        exact_result=exact_at, noise_ratio=float(noise.median()), **base_fields)
    return e, predicted


def readout(d: dict, t: str, budget: int):
    ex = d["exact"]
    base = optimum(ex, t, "odd")
    fields_ = dict(class_optimum=base, unrestricted_optimum=optimum(ex, t, "full"),
                   null_result=float(cell(ex, t, "spherical_odd", None).nested_rmse.median()),
                   scale_check_fails=scale_check_fails(d["checks"], t, "odd"))
    e, predicted = shot_evidence(d, t, budget, fields_)
    # Construction requirement: the exact-expectation fit is already at the optimum.
    if not e.exact_result <= (1.0 + rules.GAP_TOL) * base:
        raise AssertionError(f"readout case {t}: exact fit not at the class optimum")
    # Expected verdict from the variance formula, before any sample is read.
    expected = "readout" if predicted > (1.0 + rules.GAP_TOL) * base else "none"
    return e, expected, "construction (sampling noise on converged fit)", predicted


def compound(d: dict, budget: int):
    """Triangle shot cell: stopping-rule shortfall plus readout. Reported, not scored."""
    plateau, _, _ = triangle_plateau(d)
    fields_ = {k: getattr(plateau, k) for k in
               ("class_optimum", "unrestricted_optimum", "null_result", "inverse_best",
                "scale_check_fails", "normalised_result")}
    e, predicted = shot_evidence(d, "triangle", budget, fields_)
    return e, "compound", "compound (stopping rule and readout)", predicted


def generic(d: dict, t: str):
    ex = d["exact"]
    os_ = objective_scale_frame(f"objective_scale_generic_{t}_L{N_LAYERS}.csv")
    study = os_[os_.arm == "study"]
    e = rules.Evidence(
        observed=float(study.nested_rmse.median()), reference="class",
        **flags(study.nested_rmse, study.message.str.startswith("CONVERGENCE")),
        class_optimum=optimum(ex, t, "full"), unrestricted_optimum=optimum(ex, t, "full"),
        null_result=float(cell(ex, t, "spherical_full", None).nested_rmse.median()),
        scale_check_fails=scale_check_fails(d["checks"], t, "full"),
        normalised_result=float(os_[os_.arm == "normalised"].nested_rmse.median()))
    return e, "optimiser", "diagnosed (component 5)"


def representation_limit():
    with open(REPLIMIT) as fh:
        summary = json.load(fh)
    e = rules.Evidence(**summary["evidence"])
    return e, "reachability", "construction (certified envelope)"


def closed_share(e: rules.Evidence) -> float:
    """Share of the remaining log gap closed by the normalised rerun (the margin
    against SHARE in the optimiser rule); NaN where there is no normalised rerun."""
    if e.normalised_result is None or e.class_optimum is None:
        return float("nan")
    remaining = np.log(e.observed / e.class_optimum)
    return float(np.log(e.observed / e.normalised_result) / remaining) if remaining > 0 else float("nan")


def withheld(e: rules.Evidence, components) -> rules.Evidence:
    for k in components:
        e = rules.withhold(e, k)
    return e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", default=os.path.join("results", "ablation"))
    args = ap.parse_args()

    d = load()
    cases = {"triangle_plateau": triangle_plateau(d),
             "prior_sawtooth": prior(d, "sawtooth"),
             "prior_double_sawtooth": prior(d, "double_sawtooth")}
    budgets = sorted(int(b) for b in d["shots"].total_budget.unique())
    for t in ("sawtooth", "double_sawtooth"):
        for b in budgets:
            cases[f"readout_{t}_{b}"] = readout(d, t, b)
    for b in budgets:
        cases[f"compound_triangle_{b}"] = compound(d, b)
    for t in ("sawtooth", "double_sawtooth"):
        cases[f"generic_{t}"] = generic(d, t)
    cases["representation_limit"] = representation_limit()

    evidence_rows, table = [], []
    for name, (e, expected, known_by, *extra) in cases.items():
        evidence_rows.append({"case": name, **dataclasses.asdict(e)})
        row = {"case": name, "cause_known_by": known_by, "expected": expected,
               "predicted_shot_rmse": extra[0] if extra else np.nan,
               "normalised_closes": closed_share(e)}
        for comps in WITHHOLD:
            label = "all" if not comps else "without " + "+".join(map(str, comps))
            row[label] = rules.attribute(withheld(e, comps))
        row["correct_with_all"] = (row["all"] == expected) if expected != "compound" else None
        table.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    table = pd.DataFrame(table)
    safe_write_csv(pd.DataFrame(evidence_rows), os.path.join(args.output_dir, "ablation_evidence.csv"))
    path = safe_write_csv(table.assign(rules_version=rules.RULES_VERSION),
                          os.path.join(args.output_dir, "ablation_table.csv"))
    print(f"wrote {path} (rules version {rules.RULES_VERSION})\n")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(table.drop(columns="cause_known_by").to_string(index=False))
    kind = table.cause_known_by.str.split(" ").str[0]
    indep = table[kind.isin(["theorem", "construction"])]
    diag = table[kind == "diagnosed"]
    comp = table[kind == "compound"]
    print(f"\nindependent cases correct: {int(indep.correct_with_all.sum())} of {len(indep)} "
          f"(theorem {int((kind == 'theorem').sum())}, construction {int((kind == 'construction').sum())})")
    print(f"diagnosed cases consistent with their diagnosis: {int(diag.correct_with_all.sum())} of {len(diag)}")
    print("compound cases (reported, not scored): " +
          ", ".join(f"{c} -> {v}" for c, v in zip(comp.case, comp["all"])))


if __name__ == "__main__":
    main()

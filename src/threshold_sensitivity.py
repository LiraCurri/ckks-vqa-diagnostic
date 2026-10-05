"""Threshold sensitivity of the frozen attribution rules (version 1.3).

Re-applies attribute() to the stored evidence records over a grid of thresholds
fixed before this script was run, and reports, for every scored case, whether
the all-component verdict still matches its expected label. No experiment is
run and the rules' logic is unchanged; only the module-level thresholds are set
for each grid point and restored afterwards.

Grid (fixed in advance; the defaults are the frozen values):
    GAP_TOL      0.05, 0.10, 0.20
    SHARE        0.40, 0.50, 0.60
    READOUT_TOL  0.15, 0.25, 0.35
AGREE_FACTOR is not varied: it enters only the conventional reading used when
components 4 and 5 are both withheld, which stands for conventional practice
rather than for the protocol, and the stored records keep only its flag.

Expected labels that depend on a threshold are recomputed at each grid point:
a readout case is expected "readout" when its predicted shot RMSE exceeds
(1 + GAP_TOL) times the class optimum, "none" otherwise, as in
ablation_records.py; a constructed optimiser case is expected "optimiser" when
its observed shortfall exceeds (1 + GAP_TOL), "none" otherwise. Diagnosed cases are reported against their diagnosis but
counted separately; compound cases are reported, not scored.

Inputs:  results/ablation/ablation_evidence.csv, results/ablation/ablation_table.csv,
         results/constructed_cases/constructed_cases.json
Outputs: results/threshold_sensitivity/{margins.csv, one_at_a_time.csv, full_grid.csv}

Usage: python src/threshold_sensitivity.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import fields

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attribution_rules as rules  # noqa: E402

GRID = {"GAP_TOL": (0.05, 0.10, 0.20), "SHARE": (0.40, 0.50, 0.60),
        "READOUT_TOL": (0.15, 0.25, 0.35)}
DEFAULTS = {k: getattr(rules, k) for k in GRID}
EVIDENCE_FIELDS = [f.name for f in fields(rules.Evidence)]
BOOL_FIELDS = {"optimizer_success", "restarts_agree", "scale_check_fails"}


def _clean(v, name):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if name in BOOL_FIELDS:
        return bool(v) if not isinstance(v, str) else v.strip().lower() == "true"
    if name == "reference":
        return str(v)
    return float(v)


def load_cases(ablation_dir: str, constructed_path: str) -> list:
    ev = pd.read_csv(os.path.join(ablation_dir, "ablation_evidence.csv"))
    tab = pd.read_csv(os.path.join(ablation_dir, "ablation_table.csv")).set_index("case")
    cases = []
    for _, row in ev.iterrows():
        e = rules.Evidence(**{n: _clean(row[n], n) for n in EVIDENCE_FIELDS if n in row})
        t = tab.loc[row["case"]]
        kind = str(t.cause_known_by).split(" ")[0]
        predicted = t.predicted_shot_rmse if "predicted_shot_rmse" in t else np.nan
        cases.append({"case": row["case"], "kind": kind, "evidence": e,
                      "expected": t.expected,
                      "predicted_shot_rmse": None if pd.isna(predicted) else float(predicted)})
    with open(constructed_path) as fh:
        cc = json.load(fh)
    if cc["rules_version"] != rules.RULES_VERSION:
        raise AssertionError("constructed cases were attributed with a different rules version")
    for c in cc["cases"]:
        e = rules.Evidence(**{n: _clean(c["evidence"].get(n), n) for n in EVIDENCE_FIELDS})
        # the near-boundary reachability case proved compound: reported, not scored
        kind = "compound" if c["case"] == "reachability_near" else "construction"
        cases.append({"case": "constructed_" + c["case"], "kind": kind, "evidence": e,
                      "expected": c["expected"], "predicted_shot_rmse": None})
    return cases


def expected_at(case: dict, gap_tol: float) -> str:
    """Expected label at a grid point. The cause of a constructed case is fixed,
    but whether it counts as a shortfall depends on GAP_TOL: readout cases use
    the predicted shot RMSE, as in ablation_records.py (no sampled value), and
    the constructed optimiser cases their observed shortfall."""
    e = case["evidence"]
    if case["predicted_shot_rmse"] is not None and case["case"].startswith("readout_"):
        return "readout" if case["predicted_shot_rmse"] > (1.0 + gap_tol) * e.class_optimum else "none"
    if case["case"].startswith("constructed_") and case["kind"] == "construction":
        return case["expected"] if e.observed > (1.0 + gap_tol) * e.class_optimum else "none"
    return case["expected"]


def attribute_at(e: rules.Evidence, setting: dict) -> str:
    saved = {k: getattr(rules, k) for k in setting}
    try:
        for k, v in setting.items():
            setattr(rules, k, v)
        return rules.attribute(e)
    finally:
        for k, v in saved.items():
            setattr(rules, k, v)


def margins(case: dict) -> dict:
    """Distance of each case to each threshold, at the frozen values."""
    e = case["evidence"]
    base = e.class_optimum if e.reference == "class" else e.unrestricted_optimum
    out = {"case": case["case"], "kind": case["kind"], "expected": case["expected"]}
    if base:
        out["observed_over_reference"] = e.observed / base
    if e.normalised_result is not None and e.class_optimum and e.observed > e.class_optimum:
        out["closed_share"] = (math.log(e.observed / e.normalised_result)
                               / math.log(e.observed / e.class_optimum))
    if e.noise_ratio is not None:
        out["noise_ratio_minus_one"] = abs(e.noise_ratio - 1.0)
    if case["predicted_shot_rmse"] is not None and e.class_optimum:
        out["predicted_over_optimum"] = case["predicted_shot_rmse"] / e.class_optimum
    if e.inverse_best is not None and e.class_optimum:
        out["inverse_over_optimum"] = e.inverse_best / e.class_optimum
    return out


def evaluate(cases: list, setting: dict) -> dict:
    full = {**DEFAULTS, **setting}
    res = {}
    for c in cases:
        if c["kind"] == "compound":
            continue
        got = attribute_at(c["evidence"], full)
        res[c["case"]] = (got, expected_at(c, full["GAP_TOL"]), c["kind"])
    return res


def summarise(res: dict) -> dict:
    indep = [(g, w) for g, w, k in res.values() if k in ("theorem", "construction")]
    diag = [(g, w) for g, w, k in res.values() if k == "diagnosed"]
    wrong = sorted(n for n, (g, w, k) in res.items() if g != w)
    return {"independent_correct": sum(g == w for g, w in indep), "independent_total": len(indep),
            "diagnosed_consistent": sum(g == w for g, w in diag), "diagnosed_total": len(diag),
            "changed_cases": "; ".join(f"{n}: {res[n][0]} (expected {res[n][1]})" for n in wrong)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", default=os.path.join("results", "ablation"))
    ap.add_argument("--constructed", default=os.path.join("results", "constructed_cases",
                                                           "constructed_cases.json"))
    ap.add_argument("--output-dir", default=os.path.join("results", "threshold_sensitivity"))
    args = ap.parse_args()

    cases = load_cases(args.ablation_dir, args.constructed)
    os.makedirs(args.output_dir, exist_ok=True)

    m = pd.DataFrame([margins(c) for c in cases])
    m.to_csv(os.path.join(args.output_dir, "margins.csv"), index=False)

    one = []
    for name, values in GRID.items():
        for v in values:
            one.append({"varied": name, "value": v, **summarise(evaluate(cases, {name: v}))})
    one = pd.DataFrame(one)
    one.to_csv(os.path.join(args.output_dir, "one_at_a_time.csv"), index=False)

    full = []
    for combo in itertools.product(*GRID.values()):
        setting = dict(zip(GRID, combo))
        full.append({**setting, **summarise(evaluate(cases, setting))})
    full = pd.DataFrame(full)
    full.to_csv(os.path.join(args.output_dir, "full_grid.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.max_colwidth", 90):
        print(f"rules version {rules.RULES_VERSION}; defaults {DEFAULTS}\n")
        print("margins at the frozen thresholds:")
        print(m.round(3).to_string(index=False))
        print("\none threshold at a time:")
        print(one.to_string(index=False))
    allok = full[(full.independent_correct == full.independent_total)
                 & (full.diagnosed_consistent == full.diagnosed_total)]
    print(f"\nfull grid: {len(allok)} of {len(full)} settings keep every scored verdict; "
          f"independent cases all correct in "
          f"{int((full.independent_correct == full.independent_total).sum())} of {len(full)}")


if __name__ == "__main__":
    main()

"""Prospective scoring of a one-sided reachability rule (v1.4), against frozen v1.3.

Rules v1.3 read a failed inverse fit as evidence that the base optimum is
unreachable. The inference is one-sided: success shows reachability, failure
shows nothing. The candidate v1.4 encodes that: when the inverse fit's best
point lies outside GAP_TOL of the base optimum, the record is attributed as if
component 4 were withheld; a successful inverse fit is used as before.

This is reported as a prospective change, not applied to the paper's results:
the frozen rules (attribution_rules.py, v1.3) are not modified, and every
verdict the paper scores remains v1.3's. The question answered here is whether
v1.4 would fix the misattributed case without breaking the cases whose
reachability verdict rests on a failed inverse fit (the frozen envelope, and the
near-boundary case's second pass).

Inputs: the same as threshold_sensitivity.py. Output: results/prospective_v14/comparison.csv
Usage:  python src/prospective_rule_change.py
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attribution_rules as rules  # noqa: E402
from threshold_sensitivity import expected_at, load_cases  # noqa: E402


def base_optimum(e: rules.Evidence):
    if e.reference == "full" and e.class_optimum is not None:
        return e.class_optimum
    if e.reference == "class" and e.class_optimum is not None:
        return e.class_optimum
    return e.null_result


def attribute_v14(e: rules.Evidence) -> str:
    """v1.3 with a failed inverse fit treated as no evidence."""
    base = base_optimum(e)
    if e.inverse_best is not None and base is not None and e.inverse_best > (1.0 + rules.GAP_TOL) * base:
        return rules.attribute(replace(e, inverse_best=None))
    return rules.attribute(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", default=os.path.join("results", "ablation"))
    ap.add_argument("--constructed", default=os.path.join("results", "constructed_cases",
                                                           "constructed_cases.json"))
    ap.add_argument("--output-dir", default=os.path.join("results", "prospective_v14"))
    args = ap.parse_args()

    cases = load_cases(args.ablation_dir, args.constructed)
    # the near-boundary case's second pass: the record after the stopping cause is removed
    for c in list(cases):
        if c["case"] == "constructed_reachability_near":
            e = c["evidence"]
            cases.append({"case": "constructed_reachability_near_second_pass", "kind": "construction",
                          "evidence": replace(e, observed=e.normalised_result),
                          "expected": "reachability", "predicted_shot_rmse": None})

    rows = []
    for c in cases:
        e = c["evidence"]
        expected = expected_at(c, rules.GAP_TOL)
        v13, v14 = rules.attribute(e), attribute_v14(e)
        rows.append({"case": c["case"], "kind": c["kind"], "expected": expected,
                     "v1.3": v13, "v1.4": v14, "changed": v13 != v14,
                     "v1.3_correct": None if c["kind"] == "compound" else v13 == expected,
                     "v1.4_correct": None if c["kind"] == "compound" else v14 == expected})
    t = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    t.to_csv(os.path.join(args.output_dir, "comparison.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(t.to_string(index=False))
    scored = t[t.kind.isin(["theorem", "construction"])]
    print(f"\nindependent cases correct: v1.3 {int(scored['v1.3_correct'].sum())} of {len(scored)}, "
          f"v1.4 {int(scored['v1.4_correct'].sum())} of {len(scored)} "
          f"(includes the near-boundary second pass)")
    print("verdicts changed by v1.4: " + (", ".join(f"{r.case}: {r['v1.3']} -> {r['v1.4']}"
          for _, r in t[t.changed].iterrows()) or "none"))


if __name__ == "__main__":
    main()

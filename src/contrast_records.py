"""Structural contrasts of the topology study, attributed by the frozen rules.

Builds one attribution_rules.Contrast per paired contrast of the factorial
study and applies attribute_contrast(), with structural accounting (component 3)
present and withheld. The structural prediction of each class is taken from
the propositions, and for the crossed matching from the factorisation check:

    decoder alignment     "differ"      Proposition 3: a mismatched decoder reads
                                        a factorised image
    crossed vs cycle      "differ"      Proposition 3 on the ring correlators:
                                        the crossed circuit factorises all four,
                                        a cycle circuit two (ring_factorisation.csv)
    relabelling control   "equivalent"  Proposition 1
    phase rule            "none"        no structural prediction

No experiment is run here.

Inputs:  results/factorial_relabelled/factorial_paired_tests_with_bootstrap.csv
         results/factorial_relabelled/relabelling_check_tests.csv
         results/topology_rank/ring_factorisation.csv
Output:  results/ablation/contrast_table.csv

Usage, from the repository root:
    python src/contrast_records.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attribution_rules as rules  # noqa: E402

ALPHA_FWER = 0.05


def crossed_prediction(ring: pd.DataFrame) -> str:
    """'differ' if the crossed circuit factorises a different number of ring
    correlators from the cycle circuits, as Proposition 3 predicts."""
    assert (ring.ring_correlators_factorised == ring.predicted_by_proposition_3).all()
    crossed = set(ring[ring.topology == "crossed"].ring_correlators_factorised)
    cycle = set(ring[ring.topology != "crossed"].ring_correlators_factorised)
    return "differ" if crossed.isdisjoint(cycle) else "none"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--topology-dir", default=os.path.join("results", "factorial_relabelled"))
    ap.add_argument("--rank-dir", default=os.path.join("results", "topology_rank"))
    ap.add_argument("--output-dir", default=os.path.join("results", "ablation"))
    args = ap.parse_args()

    tests = pd.read_csv(os.path.join(args.topology_dir, "factorial_paired_tests_with_bootstrap.csv"),
                        keep_default_na=False)
    relabel = pd.read_csv(os.path.join(args.topology_dir, "relabelling_check_tests.csv"))
    ring = pd.read_csv(os.path.join(args.rank_dir, "ring_factorisation.csv"))

    prediction = {"matched_minus_mean_mismatched": ("decoder alignment", "differ", "theorem (Prop. 3)"),
                  "crossed_minus_nearest_matched": ("crossed vs cycle", crossed_prediction(ring), "theorem (Prop. 3)"),
                  "crossed_minus_mirror_matched": ("crossed vs cycle", crossed_prediction(ring), "theorem (Prop. 3)"),
                  "oriented_minus_uniform": ("phase rule", "none", "no prediction")}
    rows = []
    for r in tests.itertuples():
        cls, pred, known = prediction[r.contrast]
        rows.append({"class": cls, "contrast": r.contrast, "q0": int(r.q0), "n_layers": int(r.n_layers),
                     "topology": r.topology, "decoder_alignment": r.decoder_alignment,
                     "phase_rule": r.phase_rule, "metric": r.metric, "cause_known_by": known,
                     "structural_prediction": pred, "resolved": float(r.p_holm) < ALPHA_FWER})
    for r in relabel.itertuples():
        rows.append({"class": "relabelling control", "contrast": r.contrast, "q0": int(r.q0),
                     "n_layers": int(r.n_layers), "topology": "n/a", "decoder_alignment": "n/a",
                     "phase_rule": r.phase_rule, "metric": r.metric, "cause_known_by": "theorem (Prop. 1)",
                     "structural_prediction": "equivalent", "resolved": float(r.p_holm_within) < ALPHA_FWER})
    table = pd.DataFrame(rows)
    table["all"] = [rules.attribute_contrast(rules.Contrast(bool(a), b))
                    for a, b in zip(table.resolved, table.structural_prediction)]
    table["without 3"] = [rules.attribute_contrast(rules.withhold(rules.Contrast(bool(a), b), 3))
                          for a, b in zip(table.resolved, table.structural_prediction)]

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "contrast_table.csv")
    table.assign(rules_version=rules.RULES_VERSION).to_csv(path, index=False)
    print(f"wrote {path} (rules version {rules.RULES_VERSION})\n")
    summary = table.groupby(["class", "structural_prediction", "resolved"]).agg(
        p_values=("contrast", "size"), verdict=("all", "first"), without_3=("without 3", "first"))
    print(summary.to_string())


if __name__ == "__main__":
    main()

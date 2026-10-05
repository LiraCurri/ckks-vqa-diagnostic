"""Mutation testing of the check layer: which check detects which injected fault.

Each mutant injects one fault into sector_graded_evalmod (by monkeypatching, so
the source is never edited). Each check is an existing invariant or test from
the check layer, run against the mutated module; it "detects" the mutant if it
raises or reports a failure. The baseline (no mutant) must be detected by no
check.

Mutants D1-D4 reproduce the paper's defects; the others were chosen to cover the
rest of the pipeline (sign conventions, basis rotations, harmonic bookkeeping,
symmetry, decoder weights, shot budgets, grids), including two that the check
layer is not expected to catch: training on the evaluation grid (a limitation),
and removing the even-sector projection (an equivalent mutant: for a correct
circuit the even expectations are already exactly zero).

EXPECTED records, for each mutant, the checks predicted to detect it. It was
written before the matrix was first computed; any mismatch is reported, not
edited away.

Usage, from the repository root:
    python src/mutation_matrix.py --output-dir results/mutation
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Callable, Dict, FrozenSet

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sector_graded_evalmod as study  # noqa: E402

ALPHA = 1e-8
DELTA = 0.06

# Decoder table from the manuscript, kept independent of the module's own table.
PAPER_DECODER_TABLE = {
    "ZIII": 1, "IZII": 3, "IIZI": 5, "ZZII": 7,
    "XIII": 9, "IXII": 11, "IIXI": 13, "XXII": 15,
    "IIIX": 2, "XIIX": 4, "IXIX": 6, "IIXX": 8,
    "IIIY": 10, "YIIY": 12, "IYIY": 14, "IIYY": 16,
}


# ---------------------------------------------------------------- checks -----
# Each returns True when it detects a fault. Exceptions count as detection.

def check_design() -> bool:
    """validate_sector_design: ordering, parity, transport identity, bases."""
    study.validate_sector_design()
    return False


def check_paper_table() -> bool:
    """Declared observable-to-harmonic table equals the manuscript's."""
    declared = dict(zip(study.OBSERVABLES, study.OBSERVABLE_HARMONIC.tolist()))
    return declared != PAPER_DECODER_TABLE


def check_selection_rule() -> bool:
    """validate_equivariance: raw even-sector expectations vanish (P1)."""
    study.validate_equivariance(n_trials=3, seed=1234)
    return False


def check_rank() -> bool:
    """Jacobian rank of the raw decode within the sector bound, and exactly 8 (P3)."""
    r = study.jacobian_rank_study((1,), np.ones(study.N_HARMONICS), n_points=2, seed=17)
    eq = r[r.kind == "equivariant"].iloc[0]
    return bool((r.max_rank > r.sector_bound).any() or eq.min_rank != 8 or eq.max_rank != 8)


def check_convergence() -> bool:
    """Sampled expectations converge to exact ones, every basis (the D2 check)."""
    rng = np.random.default_rng(3)
    state = study.run_ansatz(rng.uniform(-np.pi, np.pi, study.n_circuit_params(1)), "generic", 1)
    exact = study.exact_expectations(state)
    budget = 600_000
    cum = study.prepare_cumulatives(state, study.bases_for("generic"))
    raw, _, _, _ = study.sampled_expectations(
        cum, "generic", target="triangle", n_layers=1, train_seed=0,
        total_budget=budget, repetition=0, master_seed=12345)
    tol = 5.0 / np.sqrt(budget / 3)
    return any(np.max(np.abs(raw[list(rows)] - exact[list(rows)])) > tol
               for rows in study.MEASUREMENT_BASES.values())


def check_variance() -> bool:
    """Coefficient noise matches the variance formula (P4a in miniature), generic
    model, under both decoders the study runs: unweighted and decay-matched."""
    return any(_variance_fails(p) for p in (0.0, 2.0))


def _variance_fails(decay_p: float) -> bool:
    rng = np.random.default_rng(5)
    state = study.run_ansatz(rng.uniform(-np.pi, np.pi, study.n_circuit_params(1)), "generic", 1)
    scales = rng.normal(-1.0, 0.25, study.n_decoder_scales())
    weights = study.preconditioner(decay_p)
    exact = study.exact_expectations(state)
    live = study.live_observable_indices("generic")
    exact_live = np.zeros(study.N_HARMONICS)
    exact_live[live] = exact[live]
    c_exact = study.decode(exact_live, scales, weights)
    budget, reps = 3000, 300
    cum = study.prepare_cumulatives(state, study.bases_for("generic"))
    sq = []
    for r in range(reps):
        _, dec, _, _ = study.sampled_expectations(
            cum, "generic", target="triangle", n_layers=1, train_seed=0,
            total_budget=budget, repetition=r, master_seed=777)
        sq.append(float(np.sum((study.decode(dec, scales, weights) - c_exact) ** 2)))
    ratio = np.sqrt(np.mean(sq)) / study.predicted_coefficient_noise(
        exact, scales, weights, "generic", budget)
    return bool(abs(ratio - 1.0) > 0.25)


def check_bounds() -> bool:
    """check_predictions P2a/P2b: no odd-sector fit beats its exact reference."""
    bundle = study.build_bundles(DELTA)["triangle"]
    cache = bundle.train
    rows = []
    for method, init, c in (("ols_odd", "n/a", study.ridge_fit(cache, 0.0, "odd")),
                            ("ridge_odd", "n/a", study.ridge_fit(cache, ALPHA, "odd")),
                            ("equivariant", "random", study.ridge_fit(cache, 0.0, "odd"))):
        rows.append({"target": "triangle", "method": method, "init": init, "n_layers": 1,
                     "train_rmse": study.masked_rmse_linf(c, cache)[0],
                     "train_objective": study.regularized_loss(c, cache, ALPHA),
                     "nested_rmse": study.masked_rmse_linf(c, bundle.nested)[0],
                     "max_raw_even_expectation": 0.0})
    ranks = study.jacobian_rank_study((1,), np.ones(study.N_HARMONICS), n_points=1, seed=17)
    checks = study.check_predictions(pd.DataFrame(rows), ranks).set_index("prediction")
    return bool(any(checks.loc[p, "status"] != "PASS"
                    for p in ("P2a_odd_methods_above_odd_ols_rmse",
                              "P2b_odd_methods_above_odd_ridge_objective")))


def check_stopping_scale() -> bool:
    """validate_stopping_scale (P7) on the configured optimiser options."""
    cache = study.build_bundles(DELTA)["triangle"].train
    study.validate_stopping_scale(cache, ALPHA, dict(study.OPTIONS))
    return False


CHECKS: Dict[str, Callable[[], bool]] = {
    "design": check_design,
    "paper_table": check_paper_table,
    "selection_rule": check_selection_rule,
    "rank": check_rank,
    "convergence": check_convergence,
    "variance": check_variance,
    "bounds": check_bounds,
    "stopping_scale": check_stopping_scale,
}


HARNESS_ERRORS = (TypeError, AttributeError, NameError, ImportError, KeyError, IndexError)


def detects(check: Callable[[], bool]) -> bool:
    """True if the check detects a fault. A failed assertion or a ValueError from
    an invariant is a detection; an error of the harness itself is not, and is
    raised so that it cannot pass for one."""
    try:
        return bool(check())
    except HARNESS_ERRORS:
        raise
    except (AssertionError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return True


# --------------------------------------------------------------- mutants -----

def configure_correct(mp: pytest.MonkeyPatch) -> None:
    """The correct configuration every run starts from: the stopping test scaled
    to the objective (the study's own defaults are the D4 defect)."""
    cache = study.build_bundles(DELTA)["triangle"].train
    f_ref = study.stopping_scale(cache, ALPHA, study.OPTIONS)["f_ref"]
    mp.setattr(study, "OPTIONS", {**study.OPTIONS, "ftol": 1e-12 * f_ref, "gtol": 1e-8 * f_ref})


def m_sign_flip(mp):
    tables = dict(study.BASIS_SIGN_TABLES)
    t = tables["XXXX"].copy()
    t[:, 0] *= -1.0
    tables["XXXX"] = t
    mp.setattr(study, "BASIS_SIGN_TABLES", tables)


def m_y_as_x(mp):
    orig = study.rotate_to_basis
    mp.setattr(study, "rotate_to_basis", lambda s, basis: orig(s, basis.replace("Y", "X")))


def m_no_rotation(mp):
    mp.setattr(study, "rotate_to_basis", lambda s, basis: s)


def m_observable_order(mp):
    orig = study.decode
    mp.setattr(study, "decode",
               lambda e, s, w: orig(e, s, w)[study.OBSERVABLE_HARMONIC - 1])


def m_harmonic_swap(mp):
    h = study.OBSERVABLE_HARMONIC.copy()
    h[[0, 1]] = h[[1, 0]]                 # ZIII <-> IZII: harmonics 1 and 3
    mp.setattr(study, "OBSERVABLE_HARMONIC", h)


def m_symmetry_break(mp):
    mp.setattr(study, "rz", study.ry)     # parity-qubit rotations no longer commute with Z_3


def m_weights_twice(mp):
    orig = study.decode
    mp.setattr(study, "decode", lambda e, s, w: orig(e, s, w) * w)


def m_budget_per_basis(mp):
    orig = study.sampled_expectations

    def per_basis(cum, kind, **kw):
        kw["total_budget"] = kw["total_budget"] * len(study.bases_for(kind))
        return orig(cum, kind, **kw)
    mp.setattr(study, "sampled_expectations", per_basis)


def m_ridge_as_rmse_bound(mp):
    orig = study.p2_lower_bound_check

    def faulty(frame, **kw):
        if kw.get("column") == "train_rmse":
            kw["ceiling_method"] = "ridge_odd"
        return orig(frame, **kw)
    mp.setattr(study, "p2_lower_bound_check", faulty)


def m_coarse_stopping(mp):
    mp.setattr(study, "OPTIONS", {**study.OPTIONS, "ftol": 1e-12, "gtol": 1e-8})


def m_train_on_nested(mp):
    orig = study.build_bundles

    def nested_as_train(delta, n_train=256, n_eval=1024):
        return {k: dataclasses.replace(v, train=v.nested)
                for k, v in orig(delta, n_train, n_eval).items()}
    mp.setattr(study, "build_bundles", nested_as_train)


def m_no_projection(mp):
    def evaluate_model(params, kind, n_layers, weights):
        nc = study.n_circuit_params(n_layers)
        state = study.run_ansatz(np.asarray(params[:nc], float), kind, n_layers)
        raw = study.exact_expectations(state)
        log_scales = np.asarray(params[nc:], float)
        return state, raw, raw.copy(), study.decode(raw, log_scales, weights), log_scales
    mp.setattr(study, "evaluate_model", evaluate_model)


MUTANTS = {
    "sign_flip":           ("sign of one observable flipped in the XXXX sign table", m_sign_flip),
    "y_as_x":              ("Y basis measured with the X rotation", m_y_as_x),
    "no_rotation":         ("basis rotation omitted before measurement (D2)", m_no_rotation),
    "observable_order":    ("decoder returns observable order, not harmonic order (D1)", m_observable_order),
    "harmonic_swap":       ("two same-parity harmonics swapped in the decoder table", m_harmonic_swap),
    "symmetry_break":      ("parity-qubit rotation does not commute with Z_3", m_symmetry_break),
    "weights_twice":       ("decoder weights applied twice", m_weights_twice),
    "budget_per_basis":    ("shot budget applied per basis instead of in total", m_budget_per_basis),
    "ridge_as_rmse_bound": ("ridge optimum used as the RMSE reference (D3)", m_ridge_as_rmse_bound),
    "coarse_stopping":     ("stopping test absolute for an objective below one (D4)", m_coarse_stopping),
    "train_on_nested":     ("training on the nested evaluation grid", m_train_on_nested),
    "no_projection":       ("even-sector projection removed (equivalent mutant)", m_no_projection),
}

# Predicted before the matrix was first computed.
EXPECTED: Dict[str, FrozenSet[str]] = {
    "sign_flip":           frozenset({"convergence", "variance"}),
    "y_as_x":              frozenset({"convergence", "variance"}),
    "no_rotation":         frozenset({"convergence", "variance"}),
    "observable_order":    frozenset({"design"}),
    "harmonic_swap":       frozenset({"paper_table"}),
    "symmetry_break":      frozenset({"selection_rule", "rank"}),
    "weights_twice":       frozenset({"variance"}),
    "budget_per_basis":    frozenset({"variance"}),
    "ridge_as_rmse_bound": frozenset({"bounds"}),
    "coarse_stopping":     frozenset({"stopping_scale"}),
    "train_on_nested":     frozenset(),
    "no_projection":       frozenset(),
}


def detected_by(mutant: str | None) -> FrozenSet[str]:
    """Checks that detect the given mutant (None: the correct code). A harness
    error is reported with the check and mutant that produced it."""
    with pytest.MonkeyPatch.context() as mp:
        configure_correct(mp)
        if mutant is not None:
            MUTANTS[mutant][1](mp)
        found = set()
        for name, check in CHECKS.items():
            try:
                if detects(check):
                    found.add(name)
            except HARNESS_ERRORS as err:
                raise RuntimeError(f"harness error in check '{name}' under mutant "
                                   f"'{mutant}': {err!r}") from err
        return frozenset(found)


def matrix() -> pd.DataFrame:
    rows = []
    for name in [None, *MUTANTS]:
        found = detected_by(name)
        row = {"mutant": name or "baseline (correct code)",
               "fault": MUTANTS[name][0] if name else "",
               **{c: c in found for c in CHECKS}}
        expected = EXPECTED[name] if name else frozenset()
        row["detected"] = bool(found)
        row["as_predicted"] = found == expected
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", default=os.path.join("results", "mutation"))
    args = ap.parse_args()
    m = matrix()
    os.makedirs(args.output_dir, exist_ok=True)
    path = study.safe_write_csv(m, os.path.join(args.output_dir, "mutation_matrix.csv"))
    print(f"wrote {path}\n")
    show = m.drop(columns="fault").replace({True: "x", False: "."})
    print(show.to_string(index=False))
    muts = m.iloc[1:]
    caught = muts[~muts.mutant.isin(["train_on_nested", "no_projection"])]
    print(f"\nbaseline clean: {not m.iloc[0][list(CHECKS)].any()}")
    print(f"mutants detected: {int(caught.detected.sum())} of {len(caught)} "
          f"(plus one limitation and one equivalent mutant, both undetected as predicted: "
          f"{not muts[muts.mutant.isin(['train_on_nested', 'no_projection'])].detected.any()})")
    bad = m[~m.as_predicted]
    print("all as predicted" if bad.empty else f"differ from prediction: {bad.mutant.tolist()}")


if __name__ == "__main__":
    main()

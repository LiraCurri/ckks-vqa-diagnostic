"""Fault-injection tests for the four defects described in the paper (D1-D4).

Each test reproduces a defect and checks that the corresponding check detects
it, and that the same check passes on the correct code.
"""

import numpy as np
import pandas as pd
import pytest

import sector_graded_evalmod as study


def _training_rmse(coefficients, cache) -> float:
    residual = cache.design_masked @ coefficients - cache.target_masked
    return float(np.sqrt(np.mean(residual ** 2)))


# D1: decoder returns coefficients in observable order instead of harmonic order.
def test_d1_observable_order_decoder_is_detected(monkeypatch) -> None:
    study.validate_sector_design()                     # correct decoder passes

    correct_decode = study.decode

    def observable_order_decode(expectations, scales, weights):
        harmonic_order = correct_decode(expectations, scales, weights)
        return harmonic_order[study.OBSERVABLE_HARMONIC - 1]

    monkeypatch.setattr(study, "decode", observable_order_decode)
    with pytest.raises(AssertionError) as err:
        study.validate_sector_design()
    # Index 0 maps to harmonic 1 either way; the first failure is index 1.
    assert err.value.args[0] == (1, study.OBSERVABLES[1])


# D2: sampler omits the basis rotation before computational-basis measurement.
def _sampling_errors(state, budget: int = 600_000) -> dict:
    exact = study.exact_expectations(state)
    cum = study.prepare_cumulatives(state, study.bases_for("generic"))
    raw, _, _, _ = study.sampled_expectations(
        cum, "generic", target="triangle", n_layers=1, train_seed=0,
        total_budget=budget, repetition=0, master_seed=12345)
    return {b: float(np.max(np.abs(raw[list(rows)] - exact[list(rows)])))
            for b, rows in study.MEASUREMENT_BASES.items()}


def test_d2_missing_basis_rotation_is_detected(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    params = rng.uniform(-np.pi, np.pi, size=study.n_circuit_params(1))
    state = study.run_ansatz(params, "generic", 1)
    tol = 5.0 / np.sqrt(600_000 / 3)                   # five standard errors per basis

    correct = _sampling_errors(state)
    assert all(e < tol for e in correct.values())      # all families converge

    monkeypatch.setattr(study, "rotate_to_basis", lambda s, basis: s)
    faulty = _sampling_errors(state)
    assert faulty["ZZZZ"] < tol                        # Z family needs no rotation
    assert faulty["XXXX"] > 10 * tol and faulty["YYYY"] > 10 * tol


# D3: a ridge solution used as a lower bound on training RMSE. The fault is
# injected into the production check: check_predictions's P2a call is made to
# use the ridge optimum, rather than the least-squares one, as the RMSE reference.
def _p2_frames():
    bundle = study.build_bundles(0.06)["triangle"]
    cache = bundle.train
    alpha = 1e-8
    rows = []
    for method, init, c in (("ols_odd", "n/a", study.ridge_fit(cache, 0.0, "odd")),
                            ("ridge_odd", "n/a", study.ridge_fit(cache, alpha, "odd")),
                            ("equivariant", "random", study.ridge_fit(cache, 0.0, "odd"))):
        rows.append({"target": "triangle", "method": method, "init": init, "n_layers": 1,
                     "train_rmse": _training_rmse(c, cache),
                     "train_objective": study.regularized_loss(c, cache, alpha),
                     "nested_rmse": study.masked_rmse_linf(c, bundle.nested)[0],
                     "max_raw_even_expectation": 0.0})
    exact = pd.DataFrame(rows)
    ranks = study.jacobian_rank_study(layer_values=(1,), weights=np.ones(study.N_HARMONICS),
                                      n_points=2, eps=1e-6, rank_rtol=1e-8, seed=17)
    return exact, ranks


def _p2a_status(exact, ranks) -> str:
    checks = study.check_predictions(exact, ranks)
    return checks.set_index("prediction").loc["P2a_odd_methods_above_odd_ols_rmse", "status"]


def test_d3_ridge_as_rmse_reference_is_detected(monkeypatch) -> None:
    exact, ranks = _p2_frames()
    # A valid odd-sector fit (here the least-squares optimum itself) passes.
    assert _p2a_status(exact, ranks) == "PASS"

    production = study.p2_lower_bound_check

    def ridge_as_rmse_reference(frame, **kw):
        if kw.get("column") == "train_rmse":
            kw["ceiling_method"] = "ridge_odd"
        return production(frame, **kw)

    monkeypatch.setattr(study, "p2_lower_bound_check", ridge_as_rmse_reference)
    # The same valid fit now appears to violate the bound.
    assert _p2a_status(exact, ranks) == "FAIL"


# D4: the stopping threshold, absolute for an objective below one, was coarse
# relative to the objective. The scale invariant detects it before any run.
def test_d4_coarse_stopping_threshold_is_detected() -> None:
    cache = study.build_bundles(0.06)["triangle"].train
    study_options = {"ftol": 1e-12, "gtol": 1e-8, "maxls": 50, "maxfun": 200_000}
    with pytest.raises(AssertionError):
        study.validate_stopping_scale(cache, 1e-8, study_options)

    f_ref = study.stopping_scale(cache, 1e-8, study_options)["f_ref"]
    normalised = {**study_options, "ftol": 1e-12 * f_ref, "gtol": 1e-8 * f_ref}
    study.validate_stopping_scale(cache, 1e-8, normalised)


# D4, confirmed by restarts. A converged run is a fixed
# point of a same-settings restart; a stopping-rule plateau is not a fixed
# point once only the tolerance is lowered.
@pytest.mark.slow
def test_d4_stopping_rule_plateau_is_detected() -> None:
    from optimizer_cross_check import idempotence_test

    seeds = range(3)
    same = idempotence_test(n_layers=3, decay_p=0.0, seeds=seeds, chain=2)
    tight = idempotence_test(n_layers=3, decay_p=0.0, seeds=seeds, chain=2,
                             restart_ftol=1e-18)

    def moved(frame):
        return (frame.nested_first - frame.nested_final) > 0.1 * frame.nested_first

    assert not moved(same).any()                       # stable at the study tolerance
    assert moved(tight).sum() >= 2                     # not a floor of the model

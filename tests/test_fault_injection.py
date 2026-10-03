"""Fault-injection tests for the four defects described in the paper (D1-D4).

Each test reproduces a defect and checks that the corresponding check detects
it, and that the same check passes on the correct code.
"""

import numpy as np
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


# D3: a ridge solution used as a lower bound on training RMSE.
def test_d3_ridge_as_rmse_bound_gives_false_violation() -> None:
    cache = study.build_bundles(0.06, n_train=128, n_eval=256)["triangle"].train
    ols = study.ridge_fit(cache, 0.0, "odd")
    ridge = study.ridge_fit(cache, 1e-8, "odd")

    # A valid odd-sector fit: the least-squares optimum itself.
    fit_rmse = _training_rmse(ols, cache)

    faulty_bound = _training_rmse(ridge, cache)        # ridge as RMSE reference
    correct_bound = _training_rmse(ols, cache)         # least-squares reference
    assert fit_rmse < faulty_bound                     # faulty check reports a violation
    assert fit_rmse >= correct_bound - 1e-14           # correct check does not


# D4: a stopping-rule plateau read as a floor. A converged run is a fixed
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
    assert moved(tight).all()                          # not a floor of the model

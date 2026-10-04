"""Fast structural regression tests for the sector-graded experiment.

These tests exercise correctness properties only. They do not rerun the full
optimization, finite-shot study, or precision-plateau diagnostic. Fault
injection tests are in test_fault_injection.py.
"""

import numpy as np
import sector_graded_evalmod as study

def _training_rmse(coefficients, cache) -> float:
    residual = cache.design_masked @ coefficients - cache.target_masked
    return float(np.sqrt(np.mean(residual**2)))

def test_sector_design_validator() -> None:
    study.validate_sector_design()

def test_equivariance_selection_rule_fast() -> None:
    study.validate_equivariance(n_trials=3, seed=1234)

# Decoder table from the paper, kept independent of the module's
# OBSERVABLE_HARMONIC so that a corrupted table cannot certify itself.
PAPER_DECODER_TABLE = {
    "ZIII": 1, "IZII": 3, "IIZI": 5, "ZZII": 7,
    "XIII": 9, "IXII": 11, "IIXI": 13, "XXII": 15,
    "IIIX": 2, "XIIX": 4, "IXIX": 6, "IIXX": 8,
    "IIIY": 10, "YIIY": 12, "IYIY": 14, "IIYY": 16,
}

def test_harmonic_table_matches_manuscript() -> None:
    declared = dict(zip(study.OBSERVABLES, study.OBSERVABLE_HARMONIC.tolist()))
    assert declared == PAPER_DECODER_TABLE

def test_decoder_scatter_matches_harmonic_assignment() -> None:
    weights = np.ones(study.N_HARMONICS)
    scales = np.zeros(len(study.FAMILY_NAMES))
    for observable_index, observable in enumerate(study.OBSERVABLES):
        harmonic = PAPER_DECODER_TABLE[observable]
        expectations = np.zeros(study.N_HARMONICS)
        expectations[observable_index] = 1.0
        coefficients = study.decode(expectations, scales, weights)
        expected = np.zeros(study.N_HARMONICS)
        expected[harmonic - 1] = 1.0
        assert np.array_equal(coefficients, expected)

def test_measurement_basis_rotation_recovers_exact_expectations() -> None:
    rng = np.random.default_rng(9)
    params = rng.uniform(-np.pi, np.pi, size=study.n_circuit_params(1))
    state = study.run_ansatz(params, "generic", 1)
    exact = study.exact_expectations(state)
    for basis, rows_tuple in study.MEASUREMENT_BASES.items():
        rotated = study.rotate_to_basis(state, basis)
        probabilities = np.abs(rotated) ** 2
        probabilities = probabilities / probabilities.sum()
        measured = probabilities @ study.BASIS_SIGN_TABLES[basis]
        assert np.allclose(measured, exact[list(rows_tuple)], atol=1e-12, rtol=1e-12)

def test_jacobian_rank_study_runs_and_saturates() -> None:
    """Smoke test of jacobian_rank_study, which differentiates the raw
    coefficients; test_equivariant_rank_bound_from_circuit_not_projection
    checks the bound directly over several depths and points."""
    ranks = study.jacobian_rank_study(layer_values=(1,), weights=np.ones(study.N_HARMONICS),
        n_points=2, eps=1e-6, rank_rtol=1e-8, seed=17)
    eq = ranks[ranks["kind"] == "equivariant"]
    assert len(eq) == 1
    assert int(eq.iloc[0]["min_rank"]) == 8 and int(eq.iloc[0]["max_rank"]) == 8

def _raw_coefficient_jacobian(p, kind, L, weights, eps=1e-6):
    """Jacobian of decode(raw expectations), without the symmetry projection.

    evaluate_model zeroes the even sector for the equivariant arm before
    decoding, which makes any rank computed from its coefficients <= 8 by
    construction. Differentiating the raw decode tests the circuit instead.
    """
    def f(q):
        _, raw, _, _, s = study.evaluate_model(q, kind, L, weights)
        return study.decode(raw, s, weights)
    J = np.empty((study.N_HARMONICS, len(p)))
    for i in range(len(p)):
        d = np.zeros_like(p); d[i] = eps
        J[:, i] = (f(p + d) - f(p - d)) / (2 * eps)
    return J

def test_equivariant_rank_bound_from_circuit_not_projection() -> None:
    rng = np.random.default_rng(17)
    w = np.ones(study.N_HARMONICS)
    for L in (1, 2):
        for _ in range(2):
            p = np.concatenate([rng.uniform(-np.pi, np.pi, study.n_circuit_params(L)),
                                rng.normal(-1.0, 0.25, study.n_decoder_scales())])
            J = _raw_coefficient_jacobian(p, "equivariant", L, w)
            assert np.abs(J[study.SECTOR_COLUMNS["even"]]).max() < 1e-9
            sv = np.linalg.svd(J, compute_uv=False)
            assert int(np.sum(sv > 1e-8 * sv[0])) == 8

def test_ols_and_ridge_stationarity_on_odd_sector() -> None:
    alpha = 1e-8
    cache = study.build_bundles(0.06)["triangle"].train
    cols = study.SECTOR_COLUMNS["odd"]
    V = cache.design_masked[:, cols]; y = cache.target_masked; m = len(y)
    ols = study.ridge_fit(cache, 0.0, "odd"); ridge = study.ridge_fit(cache, alpha, "odd")
    g_ols = 2.0 * (V.T @ (V @ ols[cols] - y)) / m
    g_ridge = 2.0 * (V.T @ (V @ ridge[cols] - y)) / m + 2.0 * alpha * ridge[cols]
    assert np.linalg.norm(g_ols, ord=np.inf) < 1e-10
    assert np.linalg.norm(g_ridge, ord=np.inf) < 1e-10
    assert _training_rmse(ols, cache) <= _training_rmse(ridge, cache) + 1e-14
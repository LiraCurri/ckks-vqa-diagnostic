"""
Earlier topology comparison, superseded by evalmod_factorial_observable_study.py.

The factorial study imports this module's configuration, datasets and masks,
classical baselines (weighted_ls, direct_spherical, direct_log_monomial),
metrics and gate primitives. The experiment assembled in run_experiment,
run_layer_sweep and main is not used for the reported results; its design
does not match connectivity across arms, reads signed amplitudes rather than
Pauli expectations, has no rotation after the final diagonal entangling block,
and draws its "random_twisted" graph from a single fixed seed.

add_comparative_metrics defines

    recovery_fraction = (RMSE_zero - RMSE_model) / (RMSE_zero - RMSE_reference),

which is normalised by the reference's own gap to the zero polynomial and so
approaches 1 whenever the reference is poor.

The original header follows.

Controlled topology-aware variational synthesis for CKKS EvalMod-style polynomials.

This version adds the experiments needed to determine whether topology affects:
  1. representational quality,
  2. optimization reliability,
  3. coefficient structure, or
  4. optimization cost.

Main controls
-------------
- zero_polynomial
- weighted_ls
- chebyshev_ridge
- monomial_ridge
- direct_log_monomial
- direct_spherical
- flat_matched
- mirror_no_twist
- mirror_twisted
- random_twisted

Circuit decoder
---------------
    c_j = exp(log_scale) * Re(psi_j)

There are no trainable per-coefficient weights or biases.

Important
---------
This is a statevector feasibility study. Signed amplitudes are accessed directly,
which is convenient on a simulator but not ordinary hardware readout.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import cheb2poly, chebder, chebval, chebvander
from scipy.optimize import minimize


# ============================================================
# 1. Configuration
# ============================================================

@dataclass(frozen=True)
class Config:
    q0_values: Tuple[int, ...] = (8, 16)
    degree: int = 31
    delta: float = 0.18

    train_samples_per_period: int = 32
    eval_samples_per_period: int = 128

    seeds: Tuple[int, ...] = tuple(range(20))
    n_qubits: int = 4
    n_layers: int = 2

    optimizer: str = "L-BFGS-B"
    maxiter: int = 3000

    target_type: str = "sawtooth"  # or "arcsin_sine"

    alpha_cheb: float = 1e-8
    alpha_monomial: float = 1e-4
    alpha_derivative: float = 0.0

    cheb_ridge_lambda: float = 1e-6
    monomial_ridge_lambda: float = 1e-8

    success_tolerance: float = 1e-4
    reference_method: str = "direct_log_monomial"

    output_dir: str = "topology_aware_evalmod_v3"
    log_scale_bounds: Tuple[float, float] = (-8.0, 8.0)


# ============================================================
# 2. Targets and datasets
# ============================================================

def centered_sawtooth(x: np.ndarray) -> np.ndarray:
    """Period-1 centered sawtooth with range [-0.5, 0.5)."""
    x = np.asarray(x, dtype=float)
    return np.mod(x + 0.5, 1.0) - 0.5


def arcsin_sine(x: np.ndarray) -> np.ndarray:
    """Period-1 triangular-wave-like arcsin-sine surrogate."""
    x = np.asarray(x, dtype=float)
    z = np.clip(np.sin(2.0 * np.pi * x), -1.0, 1.0)
    return np.arcsin(z) / (2.0 * np.pi)


def make_dataset(
    q0: int,
    samples_per_period: int,
    delta: float,
    target_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if q0 <= 0:
        raise ValueError("q0 must be positive.")
    if samples_per_period < 8:
        raise ValueError("samples_per_period must be at least 8.")
    if not 0.0 <= delta < 0.5:
        raise ValueError("delta must satisfy 0 <= delta < 0.5.")

    n_grid = int(q0 * samples_per_period) + 1
    x = np.linspace(-q0 / 2.0, q0 / 2.0, n_grid)
    t = 2.0 * x / q0

    if target_type == "sawtooth":
        y = centered_sawtooth(x)
        nearest_jump = np.floor(x) + 0.5
        mask = np.abs(x - nearest_jump) >= delta
    elif target_type == "arcsin_sine":
        y = arcsin_sine(x)
        phase = np.mod(x, 1.0)
        distance = np.minimum(np.abs(phase - 0.25), np.abs(phase - 0.75))
        mask = distance >= delta
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return x, t, y, mask


# ============================================================
# 3. Polynomial utilities
# ============================================================

def odd_degrees(degree: int) -> np.ndarray:
    if degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer.")
    return np.arange(1, degree + 1, 2, dtype=int)


def validate_model_size(degree: int, n_qubits: int) -> None:
    n_coeffs = len(odd_degrees(degree))
    if 2**n_qubits != n_coeffs:
        raise ValueError(
            "Amplitude decoding requires 2**n_qubits == number of odd "
            f"coefficients. Got {2**n_qubits} and {n_coeffs}."
        )


def design_matrix(t: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
    degrees = odd_degrees(degree)
    return chebvander(t, degree)[:, degrees], degrees


def evaluate_polynomial(t: np.ndarray, coeffs: np.ndarray, degree: int) -> np.ndarray:
    full = np.zeros(degree + 1)
    full[odd_degrees(degree)] = coeffs
    return chebval(t, full)


def derivative_values(t: np.ndarray, coeffs: np.ndarray, degree: int) -> np.ndarray:
    full = np.zeros(degree + 1)
    full[odd_degrees(degree)] = coeffs
    return chebval(t, chebder(full))


def monomial_coeffs(coeffs: np.ndarray, degree: int) -> np.ndarray:
    full = np.zeros(degree + 1)
    full[odd_degrees(degree)] = coeffs
    converted = cheb2poly(full)
    padded = np.zeros(degree + 1)
    padded[: len(converted)] = converted
    return padded


def cheb_to_monomial_matrix(degree: int) -> np.ndarray:
    n_coeffs = len(odd_degrees(degree))
    matrix = np.zeros((degree + 1, n_coeffs))
    for j in range(n_coeffs):
        basis = np.zeros(n_coeffs)
        basis[j] = 1.0
        matrix[:, j] = monomial_coeffs(basis, degree)
    return matrix


def raw_domain_monomial_coeffs(
    coeffs: np.ndarray,
    q0: int,
    degree: int,
) -> np.ndarray:
    normalized = monomial_coeffs(coeffs, degree)
    powers = np.arange(len(normalized))
    return normalized * (2.0 / q0) ** powers


def coefficient_structure_metrics(coeffs: np.ndarray) -> Dict[str, float]:
    coeffs = np.asarray(coeffs, dtype=float)
    energy = float(np.dot(coeffs, coeffs))
    if energy < 1e-20:
        return {
            "reflection_correlation": 0.0,
            "twisted_reflection_correlation": 0.0,
            "coefficient_roughness": 0.0,
        }

    reflected = coeffs[::-1]
    signs = (-1.0) ** np.arange(len(coeffs))
    return {
        "reflection_correlation": float(np.dot(coeffs, reflected) / energy),
        "twisted_reflection_correlation": float(
            np.dot(coeffs, signs * reflected) / energy
        ),
        "coefficient_roughness": float(np.sum(np.diff(coeffs) ** 2)),
    }


# ============================================================
# 4. Shared loss and classical baselines
# ============================================================

def regularized_loss_from_coeffs(
    coeffs: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
) -> float:
    prediction = evaluate_polynomial(t, coeffs, cfg.degree)
    residual = prediction[mask] - y[mask]

    fit = float(np.mean(residual**2))
    cheb_penalty = float(np.dot(coeffs, coeffs))
    mono_penalty = float(
        np.log1p(np.linalg.norm(monomial_coeffs(coeffs, cfg.degree), ord=2))
    )

    if cfg.alpha_derivative > 0.0:
        derivative = derivative_values(t[mask], coeffs, cfg.degree)
        derivative_penalty = float(np.mean(derivative**2))
    else:
        derivative_penalty = 0.0

    return (
        fit
        + cfg.alpha_cheb * cheb_penalty
        + cfg.alpha_monomial * mono_penalty
        + cfg.alpha_derivative * derivative_penalty
    )


def weighted_ls(t, y, mask, degree) -> np.ndarray:
    matrix, _ = design_matrix(t, degree)
    coeffs, *_ = np.linalg.lstsq(matrix[mask], y[mask], rcond=None)
    return coeffs


def chebyshev_ridge(t, y, mask, degree, lam) -> np.ndarray:
    matrix, _ = design_matrix(t, degree)
    masked_matrix = matrix[mask]
    masked_target = y[mask]
    lhs = masked_matrix.T @ masked_matrix + lam * np.eye(masked_matrix.shape[1])
    rhs = masked_matrix.T @ masked_target
    return np.linalg.solve(lhs, rhs)


def monomial_ridge(t, y, mask, degree, lam) -> np.ndarray:
    matrix, _ = design_matrix(t, degree)
    masked_matrix = matrix[mask]
    masked_target = y[mask]
    transform = cheb_to_monomial_matrix(degree)
    lhs = masked_matrix.T @ masked_matrix + lam * (transform.T @ transform)
    rhs = masked_matrix.T @ masked_target
    return np.linalg.solve(lhs, rhs)


def direct_log_monomial_fit(
    t: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, object, np.ndarray, Dict[str, float]]:
    initial = weighted_ls(t, y, mask, cfg.degree)
    history: List[Tuple[int, float]] = []
    start = time.perf_counter()

    def objective(coeffs: np.ndarray) -> float:
        value = regularized_loss_from_coeffs(coeffs, t, y, mask, cfg)
        history.append((len(history) + 1, value))
        return value

    result = minimize(
        objective,
        initial,
        method=cfg.optimizer,
        options=_optimizer_options(cfg),
    )
    elapsed = time.perf_counter() - start
    diagnostics = {
        "objective_evaluations": float(getattr(result, "nfev", len(history))),
        "optimizer_iterations": float(getattr(result, "nit", np.nan)),
        "elapsed_seconds": float(elapsed),
        "optimized_scale": np.nan,
        "optimized_log_scale": np.nan,
    }
    return result.x, result, np.asarray(history, dtype=float), diagnostics


def spherical_coeffs(params: np.ndarray, degree: int) -> np.ndarray:
    n_coeffs = len(odd_degrees(degree))
    raw = np.asarray(params[:n_coeffs], dtype=float)
    log_scale = float(params[n_coeffs])
    norm = float(np.linalg.norm(raw))
    direction = np.zeros_like(raw) if norm < 1e-12 else raw / norm
    return float(np.exp(log_scale)) * direction


def direct_spherical_fit(
    t: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, object, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_coeffs = len(odd_degrees(cfg.degree))
    initial = np.concatenate([rng.normal(size=n_coeffs), [rng.normal(0.0, 0.25)]])
    bounds = [(-5.0, 5.0)] * n_coeffs + [cfg.log_scale_bounds]
    history: List[Tuple[int, float]] = []
    start = time.perf_counter()

    def objective(params: np.ndarray) -> float:
        coeffs = spherical_coeffs(params, cfg.degree)
        value = regularized_loss_from_coeffs(coeffs, t, y, mask, cfg)
        history.append((len(history) + 1, value))
        return value

    result = minimize(
        objective,
        initial,
        method=cfg.optimizer,
        bounds=bounds,
        options=_optimizer_options(cfg),
    )
    elapsed = time.perf_counter() - start
    coeffs = spherical_coeffs(result.x, cfg.degree)
    optimized_log_scale = float(result.x[n_coeffs])
    diagnostics = {
        "objective_evaluations": float(getattr(result, "nfev", len(history))),
        "optimizer_iterations": float(getattr(result, "nit", np.nan)),
        "elapsed_seconds": float(elapsed),
        "optimized_scale": float(np.exp(optimized_log_scale)),
        "optimized_log_scale": optimized_log_scale,
    }
    return coeffs, result, np.asarray(history, dtype=float), diagnostics


# ============================================================
# 5. Statevector simulator
# ============================================================

def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def apply_1q(state, gate, qubit, n_qubits) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    shape = tensor.shape
    updated = gate @ tensor.reshape(2, -1)
    updated = updated.reshape(shape)
    return np.moveaxis(updated, 0, qubit).reshape(-1)


def apply_cphase(state, q1, q2, phi, n_qubits) -> np.ndarray:
    if q1 == q2:
        raise ValueError("Controlled phase requires distinct qubits.")
    output = state.copy()
    phase = np.exp(1j * phi)
    for index in range(len(output)):
        bit1 = (index >> (n_qubits - 1 - q1)) & 1
        bit2 = (index >> (n_qubits - 1 - q2)) & 1
        if bit1 and bit2:
            output[index] *= phase
    return output


def apply_cz(state, q1, q2, n_qubits) -> np.ndarray:
    return apply_cphase(state, q1, q2, np.pi, n_qubits)


# ============================================================
# 6. Parameter-matched ansatz family
# ============================================================

ANSATZES = (
    "flat_matched",
    "mirror_no_twist",
    "mirror_twisted",
    "random_twisted",
)


def n_pairs(n_qubits: int) -> int:
    return n_qubits // 2


def n_circuit_params(n_qubits: int, n_layers: int) -> int:
    return n_layers * (n_qubits + n_pairs(n_qubits))


def adjacent_pairs(n_qubits: int) -> List[Tuple[int, int]]:
    return [(2 * j, 2 * j + 1) for j in range(n_pairs(n_qubits))]


def mirrored_pairs(n_qubits: int) -> List[Tuple[int, int]]:
    return [(j, n_qubits - 1 - j) for j in range(n_pairs(n_qubits))]


def random_pairs(n_qubits: int, seed: int = 314159) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    candidates = [
        (i, j)
        for i in range(n_qubits)
        for j in range(i + 1, n_qubits)
        if j != i + 1
    ]
    if len(candidates) < n_pairs(n_qubits):
        candidates = [
            (i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
        ]
    selected = rng.choice(len(candidates), size=n_pairs(n_qubits), replace=False)
    return [candidates[int(index)] for index in selected]


def topology_pairs(name: str, n_qubits: int) -> List[Tuple[int, int]]:
    if name == "flat_matched":
        return adjacent_pairs(n_qubits)
    if name in ("mirror_no_twist", "mirror_twisted"):
        return mirrored_pairs(n_qubits)
    if name == "random_twisted":
        return random_pairs(n_qubits)
    raise ValueError(f"Unknown ansatz: {name}")


def run_ansatz(params, name, n_qubits, n_layers) -> np.ndarray:
    expected = n_circuit_params(n_qubits, n_layers)
    if len(params) != expected:
        raise ValueError(f"{name} expected {expected} parameters, got {len(params)}.")

    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    pairs = topology_pairs(name, n_qubits)
    index = 0

    for layer in range(n_layers):
        for qubit in range(n_qubits):
            state = apply_1q(state, ry(params[index]), qubit, n_qubits)
            index += 1

        for qubit in range(n_qubits - 1):
            state = apply_cz(state, qubit, qubit + 1, n_qubits)

        for pair_index, (q1, q2) in enumerate(pairs):
            phi = float(params[index])
            index += 1
            sign = -1.0 if (layer + pair_index) % 2 else 1.0

            if name == "mirror_no_twist":
                # Fixed mirrored entanglement. The matched trainable parameter is
                # consumed only by antisymmetric local phase rotations.
                state = apply_cz(state, q1, q2, n_qubits)
                state = apply_1q(state, rz(phi / 2.0), q1, n_qubits)
                state = apply_1q(state, rz(-phi / 2.0), q2, n_qubits)
            else:
                state = apply_cphase(state, q1, q2, phi, n_qubits)
                state = apply_1q(state, rz(sign * phi / 2.0), q1, n_qubits)
                state = apply_1q(state, rz(-sign * phi / 2.0), q2, n_qubits)

    return state


# ============================================================
# 7. Circuit decoder and optimization
# ============================================================

def params_to_coeffs(params: np.ndarray, name: str, cfg: Config) -> np.ndarray:
    n_circuit = n_circuit_params(cfg.n_qubits, cfg.n_layers)
    if len(params) != n_circuit + 1:
        raise ValueError(f"Expected {n_circuit + 1} parameters, got {len(params)}.")

    state = run_ansatz(params[:n_circuit], name, cfg.n_qubits, cfg.n_layers)
    log_scale = float(params[n_circuit])
    n_coeffs = len(odd_degrees(cfg.degree))
    return float(np.exp(log_scale)) * np.real(state[:n_coeffs])


def initial_circuit_params(cfg: Config, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    circuit = rng.uniform(
        -np.pi,
        np.pi,
        size=n_circuit_params(cfg.n_qubits, cfg.n_layers),
    )
    return np.concatenate([circuit, [rng.normal(0.0, 0.25)]])


def model_bounds(cfg: Config) -> List[Tuple[float, float]]:
    return [(-np.pi, np.pi)] * n_circuit_params(
        cfg.n_qubits, cfg.n_layers
    ) + [cfg.log_scale_bounds]


def _optimizer_options(cfg: Config) -> Dict[str, float | int]:
    if cfg.optimizer.upper() == "L-BFGS-B":
        return {
            "maxiter": cfg.maxiter,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        }
    if cfg.optimizer.upper() == "POWELL":
        return {"maxiter": cfg.maxiter, "xtol": 1e-7, "ftol": 1e-8}
    return {"maxiter": cfg.maxiter}


def train_variational(
    name: str,
    t: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, object, np.ndarray, Dict[str, float]]:
    history: List[Tuple[int, float]] = []
    start = time.perf_counter()

    def objective(params: np.ndarray) -> float:
        coeffs = params_to_coeffs(params, name, cfg)
        value = regularized_loss_from_coeffs(coeffs, t, y, mask, cfg)
        history.append((len(history) + 1, value))
        return value

    result = minimize(
        objective,
        initial_circuit_params(cfg, seed),
        method=cfg.optimizer,
        bounds=model_bounds(cfg),
        options=_optimizer_options(cfg),
    )
    elapsed = time.perf_counter() - start
    coeffs = params_to_coeffs(result.x, name, cfg)
    n_circuit = n_circuit_params(cfg.n_qubits, cfg.n_layers)
    optimized_log_scale = float(result.x[n_circuit])
    diagnostics = {
        "objective_evaluations": float(getattr(result, "nfev", len(history))),
        "optimizer_iterations": float(getattr(result, "nit", np.nan)),
        "elapsed_seconds": float(elapsed),
        "optimized_log_scale": optimized_log_scale,
        "optimized_scale": float(np.exp(optimized_log_scale)),
    }
    return coeffs, result, np.asarray(history, dtype=float), diagnostics


# ============================================================
# 8. Metrics and post-processing
# ============================================================

def metrics(t, y, mask, coeffs, q0, degree) -> Dict[str, float]:
    prediction = evaluate_polynomial(t, coeffs, degree)
    error = prediction - y
    derivative_t = derivative_values(t[mask], coeffs, degree)
    derivative_x = (2.0 / q0) * derivative_t
    normalized_mono = monomial_coeffs(coeffs, degree)
    raw_mono = raw_domain_monomial_coeffs(coeffs, q0, degree)

    result = {
        "masked_rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
        "masked_linf": float(np.max(np.abs(error[mask]))),
        "global_rmse": float(np.sqrt(np.mean(error**2))),
        "global_linf": float(np.max(np.abs(error))),
        "chebyshev_coeff_l2": float(np.linalg.norm(coeffs, ord=2)),
        "normalized_monomial_l2": float(np.linalg.norm(normalized_mono, ord=2)),
        "raw_domain_monomial_l2": float(np.linalg.norm(raw_mono, ord=2)),
        "derivative_rms_t": float(np.sqrt(np.mean(derivative_t**2))),
        "derivative_rms_x": float(np.sqrt(np.mean(derivative_x**2))),
        "included_fraction": float(np.mean(mask)),
    }
    result.update(coefficient_structure_metrics(coeffs))
    return result


def add_comparative_metrics(results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    results = results.copy()
    results["rmse_gap_to_reference"] = np.nan
    results["success"] = False
    results["recovery_fraction"] = np.nan

    group_columns = ["q0", "target_type", "degree", "n_layers"]
    for _, group in results.groupby(group_columns, dropna=False):
        zero = group[group["method"] == "zero_polynomial"]
        reference = group[group["method"] == cfg.reference_method]
        if zero.empty or reference.empty:
            continue

        zero_rmse = float(zero.iloc[0]["masked_rmse"])
        reference_rmse = float(reference.iloc[0]["masked_rmse"])
        denominator = zero_rmse - reference_rmse
        indices = group.index

        results.loc[indices, "rmse_gap_to_reference"] = (
            results.loc[indices, "masked_rmse"] - reference_rmse
        )
        results.loc[indices, "success"] = (
            results.loc[indices, "masked_rmse"]
            <= reference_rmse + cfg.success_tolerance
        )
        if abs(denominator) > 1e-12:
            results.loc[indices, "recovery_fraction"] = (
                zero_rmse - results.loc[indices, "masked_rmse"]
            ) / denominator

    return results


# ============================================================
# 9. Experiment runner
# ============================================================

def _base_row(
    q0: int,
    method: str,
    seed: int | None,
    cfg: Config,
    result: object | None,
    diagnostics: Mapping[str, float] | None,
) -> Dict[str, object]:
    diagnostics = diagnostics or {}
    return {
        "q0": q0,
        "target_type": cfg.target_type,
        "degree": cfg.degree,
        "n_qubits": cfg.n_qubits,
        "n_layers": cfg.n_layers,
        "method": method,
        "seed": np.nan if seed is None else seed,
        "optimizer_success": np.nan if result is None else bool(result.success),
        "final_objective": np.nan if result is None else float(result.fun),
        "optimizer_message": "" if result is None else str(result.message),
        "objective_evaluations": diagnostics.get("objective_evaluations", np.nan),
        "optimizer_iterations": diagnostics.get("optimizer_iterations", np.nan),
        "elapsed_seconds": diagnostics.get("elapsed_seconds", np.nan),
        "optimized_log_scale": diagnostics.get("optimized_log_scale", np.nan),
        "optimized_scale": diagnostics.get("optimized_scale", np.nan),
    }


def run_experiment(
    cfg: Config,
) -> Tuple[
    pd.DataFrame,
    Dict[Tuple[int, str, int], np.ndarray],
    Dict[Tuple[int, str, int | None], np.ndarray],
]:
    validate_model_size(cfg.degree, cfg.n_qubits)
    os.makedirs(cfg.output_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    histories: Dict[Tuple[int, str, int], np.ndarray] = {}
    coefficient_store: Dict[Tuple[int, str, int | None], np.ndarray] = {}

    for q0 in cfg.q0_values:
        print(f"\n=== q0={q0}, degree={cfg.degree}, layers={cfg.n_layers} ===")
        _, t_train, y_train, mask_train = make_dataset(
            q0, cfg.train_samples_per_period, cfg.delta, cfg.target_type
        )
        _, t_eval, y_eval, mask_eval = make_dataset(
            q0, cfg.eval_samples_per_period, cfg.delta, cfg.target_type
        )

        n_coeffs = len(odd_degrees(cfg.degree))
        c_zero = np.zeros(n_coeffs)
        c_ls = weighted_ls(t_train, y_train, mask_train, cfg.degree)
        c_cheb = chebyshev_ridge(
            t_train, y_train, mask_train, cfg.degree, cfg.cheb_ridge_lambda
        )
        c_mono = monomial_ridge(
            t_train, y_train, mask_train, cfg.degree, cfg.monomial_ridge_lambda
        )
        c_log, r_log, h_log, d_log = direct_log_monomial_fit(
            t_train, y_train, mask_train, cfg
        )
        histories[(q0, "direct_log_monomial", -1)] = h_log

        classical = {
            "zero_polynomial": (c_zero, None, {}),
            "weighted_ls": (c_ls, None, {}),
            "chebyshev_ridge": (c_cheb, None, {}),
            "monomial_ridge": (c_mono, None, {}),
            "direct_log_monomial": (c_log, r_log, d_log),
        }
        for method, (coeffs, result, diagnostics) in classical.items():
            coefficient_store[(q0, method, None)] = coeffs
            row = _base_row(q0, method, None, cfg, result, diagnostics)
            row.update(metrics(t_eval, y_eval, mask_eval, coeffs, q0, cfg.degree))
            rows.append(row)

        # Spherical baseline is stochastic and uses the same seed set.
        for seed in cfg.seeds:
            print(f"Training direct_spherical, seed={seed}")
            coeffs, result, history, diagnostics = direct_spherical_fit(
                t_train, y_train, mask_train, cfg, seed
            )
            coefficient_store[(q0, "direct_spherical", seed)] = coeffs
            histories[(q0, "direct_spherical", seed)] = history
            row = _base_row(q0, "direct_spherical", seed, cfg, result, diagnostics)
            row.update(metrics(t_eval, y_eval, mask_eval, coeffs, q0, cfg.degree))
            rows.append(row)

        for name in ANSATZES:
            for seed in cfg.seeds:
                print(f"Training {name}, seed={seed}")
                coeffs, result, history, diagnostics = train_variational(
                    name, t_train, y_train, mask_train, cfg, seed
                )
                coefficient_store[(q0, name, seed)] = coeffs
                histories[(q0, name, seed)] = history
                row = _base_row(q0, name, seed, cfg, result, diagnostics)
                row.update(metrics(t_eval, y_eval, mask_eval, coeffs, q0, cfg.degree))
                rows.append(row)

    results = add_comparative_metrics(pd.DataFrame(rows), cfg)
    results.to_csv(os.path.join(cfg.output_dir, "results.csv"), index=False)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    return results, histories, coefficient_store


# ============================================================
# 10. Manifold sampling
# ============================================================

def sample_ansatz_coefficients(
    name: str,
    cfg: Config,
    n_samples: int = 2000,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    n_circuit = n_circuit_params(cfg.n_qubits, cfg.n_layers)
    for _ in range(n_samples):
        circuit = rng.uniform(-np.pi, np.pi, size=n_circuit)
        params = np.concatenate([circuit, [0.0]])
        samples.append(params_to_coeffs(params, name, cfg))
    return np.asarray(samples)


def effective_rank(samples: np.ndarray) -> float:
    covariance = np.cov(samples, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def analyze_manifolds(cfg: Config, n_samples: int = 2000) -> pd.DataFrame:
    rows = []
    for name in ANSATZES:
        samples = sample_ansatz_coefficients(name, cfg, n_samples=n_samples)
        norms = np.linalg.norm(samples, axis=1)
        structure = [coefficient_structure_metrics(row) for row in samples]
        rows.append(
            {
                "method": name,
                "degree": cfg.degree,
                "n_layers": cfg.n_layers,
                "n_samples": n_samples,
                "effective_rank": effective_rank(samples),
                "mean_coeff_l2": float(np.mean(norms)),
                "mean_abs_reflection_correlation": float(
                    np.mean([abs(item["reflection_correlation"]) for item in structure])
                ),
                "mean_abs_twisted_reflection_correlation": float(
                    np.mean(
                        [abs(item["twisted_reflection_correlation"]) for item in structure]
                    )
                ),
                "mean_coefficient_roughness": float(
                    np.mean([item["coefficient_roughness"] for item in structure])
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(cfg.output_dir, "manifold_summary.csv"), index=False)
    return frame


# ============================================================
# 11. Reporting and plots
# ============================================================

SUMMARY_METRICS = [
    "masked_rmse",
    "masked_linf",
    "global_rmse",
    "normalized_monomial_l2",
    "derivative_rms_x",
    "reflection_correlation",
    "twisted_reflection_correlation",
    "coefficient_roughness",
    "optimized_scale",
    "objective_evaluations",
    "elapsed_seconds",
    "rmse_gap_to_reference",
    "recovery_fraction",
    "success",
]


def summarize(results: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = results.groupby(
        ["q0", "target_type", "degree", "n_layers", "method"]
    )[SUMMARY_METRICS].agg(["mean", "std", "median", "min", "max"])
    grouped.to_csv(os.path.join(cfg.output_dir, "grouped_summary.csv"))

    stochastic = results[results["seed"].notna()]
    success = stochastic.groupby(
        ["q0", "degree", "n_layers", "method"], as_index=False
    ).agg(
        success_rate=("success", "mean"),
        mean_recovery=("recovery_fraction", "mean"),
        median_recovery=("recovery_fraction", "median"),
        mean_rmse_gap=("rmse_gap_to_reference", "mean"),
        median_evaluations=("objective_evaluations", "median"),
        median_seconds=("elapsed_seconds", "median"),
    )
    success.to_csv(os.path.join(cfg.output_dir, "success_summary.csv"), index=False)
    print("\nSuccess summary:")
    print(success)
    return grouped, success


def save_plot(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_recovery(results: pd.DataFrame, cfg: Config) -> None:
    stochastic = results[results["seed"].notna()]
    summary = stochastic.groupby(["q0", "method"], as_index=False).agg(
        mean_recovery=("recovery_fraction", "mean"),
        std_recovery=("recovery_fraction", "std"),
    )
    plt.figure(figsize=(10, 6))
    for method, group in summary.groupby("method"):
        plt.errorbar(
            group["q0"],
            group["mean_recovery"],
            yerr=group["std_recovery"].fillna(0.0),
            marker="o",
            capsize=3,
            label=method,
        )
    plt.axhline(0.0, linewidth=1)
    plt.axhline(1.0, linewidth=1, linestyle="--")
    plt.xlabel("q0")
    plt.ylabel("Recovery fraction over zero")
    plt.title(f"Recovery fraction, degree={cfg.degree}, layers={cfg.n_layers}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_plot(os.path.join(cfg.output_dir, "recovery_fraction.png"))


def plot_success_rate(results: pd.DataFrame, cfg: Config) -> None:
    stochastic = results[results["seed"].notna()]
    summary = stochastic.groupby(["q0", "method"], as_index=False).agg(
        success_rate=("success", "mean")
    )
    q_values = sorted(summary["q0"].unique())
    methods = list(summary["method"].unique())
    x = np.arange(len(q_values))
    width = 0.8 / max(len(methods), 1)
    plt.figure(figsize=(11, 6))
    for index, method in enumerate(methods):
        values = []
        for q0 in q_values:
            subset = summary[(summary["q0"] == q0) & (summary["method"] == method)]
            values.append(float(subset.iloc[0]["success_rate"]) if not subset.empty else np.nan)
        plt.bar(x + (index - (len(methods) - 1) / 2) * width, values, width, label=method)
    plt.xticks(x, [str(q) for q in q_values])
    plt.ylim(0.0, 1.05)
    plt.xlabel("q0")
    plt.ylabel("Success rate")
    plt.title(f"Success within {cfg.success_tolerance:g} RMSE of reference")
    plt.grid(alpha=0.25, axis="y")
    plt.legend(fontsize=8)
    save_plot(os.path.join(cfg.output_dir, "success_rate.png"))


def plot_cost_vs_recovery(results: pd.DataFrame, cfg: Config) -> None:
    stochastic = results[results["seed"].notna()]
    summary = stochastic.groupby(["q0", "method"], as_index=False).agg(
        recovery=("recovery_fraction", "mean"),
        evaluations=("objective_evaluations", "median"),
    )
    plt.figure(figsize=(9, 6))
    for _, row in summary.iterrows():
        plt.scatter(row["evaluations"], row["recovery"], s=80)
        plt.annotate(
            f'{row["method"]}, q0={int(row["q0"])}',
            (row["evaluations"], row["recovery"]),
            fontsize=7,
        )
    plt.xlabel("Median objective evaluations")
    plt.ylabel("Mean recovery fraction")
    plt.title("Optimization cost versus recovered approximation quality")
    plt.grid(alpha=0.25)
    save_plot(os.path.join(cfg.output_dir, "cost_vs_recovery.png"))


# ============================================================
# 12. Layer sweep
# ============================================================

def run_layer_sweep(
    base_cfg: Config,
    layer_values: Sequence[int] = (1, 2, 3, 4),
    manifold_samples: int = 1000,
) -> pd.DataFrame:
    all_results = []
    root = base_cfg.output_dir

    for layers in layer_values:
        cfg = replace(
            base_cfg,
            n_layers=int(layers),
            output_dir=os.path.join(root, f"layers_{layers}"),
        )
        results, histories, store = run_experiment(cfg)
        summarize(results, cfg)
        analyze_manifolds(cfg, n_samples=manifold_samples)
        plot_recovery(results, cfg)
        plot_success_rate(results, cfg)
        plot_cost_vs_recovery(results, cfg)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    os.makedirs(root, exist_ok=True)
    combined.to_csv(os.path.join(root, "layer_sweep_results.csv"), index=False)

    sweep_summary = combined[combined["seed"].notna()].groupby(
        ["q0", "degree", "n_layers", "method"], as_index=False
    ).agg(
        success_rate=("success", "mean"),
        mean_recovery=("recovery_fraction", "mean"),
        median_evaluations=("objective_evaluations", "median"),
        mean_monomial_norm=("normalized_monomial_l2", "mean"),
    )
    sweep_summary.to_csv(os.path.join(root, "layer_sweep_summary.csv"), index=False)
    return combined


# ============================================================
# 13. Main
# ============================================================

def main() -> None:
    cfg = Config()

    # For a quick first run, use:
    # cfg = replace(cfg, seeds=(0, 1, 2), maxiter=800)

    # Single-depth experiment:
    results, histories, store = run_experiment(cfg)
    summarize(results, cfg)
    analyze_manifolds(cfg, n_samples=2000)
    plot_recovery(results, cfg)
    plot_success_rate(results, cfg)
    plot_cost_vs_recovery(results, cfg)

    # Uncomment for the full layer-depth sweep:
    # run_layer_sweep(cfg, layer_values=(1, 2, 3, 4), manifold_samples=1000)

    print(f"Outputs saved to: {cfg.output_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "this module provides shared utilities; run "
        "src/evalmod_factorial_observable_study.py instead"
    )

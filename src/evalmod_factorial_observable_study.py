"""
Factorial observable-decoder study for topology-aware EvalMod polynomial synthesis.

Design: three perfect matchings x two phase rules x three decoder alignments,
at two carry-range conditions, three depths and 20 seeds -> 2,160 circuit
optimisations, plus 42 distinct classical reference optimisations (40
direct_spherical, two direct_log_monomial) reported once per depth.

Statistics: paired Wilcoxon contrasts (Pratt zeros) in three classes --
(a) oriented vs uniform phase rule, (b) matched vs mean-mismatched decoder,
(c) crossed vs nearest and crossed vs mirror under matched decoders -- giving
114 contrasts per metric. Holm correction is applied once, by the caller, across
both metrics (228 p-values); paired_factorial_tests deliberately does not
adjust within a metric. Each contrast carries a percentile bootstrap interval
for the median difference.

Reproduction note:
The main results reported in the manuscript use maxfun=1_000_000 so that
termination is governed primarily by convergence criteria rather than the
SciPy default function-evaluation cap.

An earlier run used the default maxfun=15_000. That run is retained only as
an optimization-budget diagnostic and should not be mixed with the main
reported CSVs.

Requires topology_aware_evalmod_experiments_v3.py for shared utilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon

import topology_aware_evalmod_experiments_v3 as base

N_QUBITS = 4
DIM = 2 ** N_QUBITS

# Circuit matchings. The mirror circuit is the image of the nearest circuit
# under the relabelling sigma: i -> i + 1 (mod 4), with the same pair order and
# orientation, and it is read in the sigma-frame: through the sigma-images of
# the canonical decoders (decoder_spec). Every cell (nearest circuit, decoder a)
# is then the same optimisation problem as (mirror circuit, decoder sigma(a)),
# up to a permutation of the circuit parameters (Proposition 1; checked by
# validate_relabelling_equivalence).
PAIR_TOPOLOGIES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "nearest": ((0, 1), (2, 3)),
    "mirror": ((1, 2), (3, 0)),
    "crossed": ((0, 2), (1, 3)),
}
BRIDGE_EDGES: Dict[str, Tuple[int, int]] = {
    "nearest": (1, 2),
    "mirror": (2, 3),
    "crossed": (0, 1),
}
# Canonical decoder matchings, used by the nearest and crossed circuits.
DECODER_PAIRS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "nearest": ((0, 1), (2, 3)),
    "mirror": ((0, 3), (1, 2)),
    "crossed": ((0, 2), (1, 3)),
}
# sigma maps the nearest and mirror matchings to each other and fixes the crossed one.
SIGMA_ALIGNMENT = {"nearest": "mirror", "mirror": "nearest", "crossed": "crossed"}
CIRCUIT_FRAME = {"nearest": 0, "mirror": 1, "crossed": 0}
USE_SIGMA_FRAME = True
PHASE_RULES = ("uniform", "oriented")
DECODER_ALIGNMENTS = tuple(PAIR_TOPOLOGIES)
CIRCUIT_METHODS = tuple(
    f"{topology}_{phase_rule}" for topology in PAIR_TOPOLOGIES for phase_rule in PHASE_RULES
)
FAMILY_NAMES = ("local", "z_structure", "aligned_same_axis", "aligned_mixed_axis")
FIXED_OBSERVABLE_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "local": ("ZIII", "IZII", "IIZI", "IIIZ"),
    "z_structure": ("ZZII", "IZZI", "IIZZ", "ZIIZ"),
}


def pauli_string(assignments: Mapping[int, str]) -> str:
    symbols = ["I"] * N_QUBITS
    for qubit, symbol in assignments.items():
        symbols[qubit] = symbol
    return "".join(symbols)


def relabel_observable(observable: str, shift: int = 1) -> str:
    """Apply sigma^shift, i -> i + shift (mod 4), to a Pauli string."""
    out = ["I"] * N_QUBITS
    for q, symbol in enumerate(observable):
        out[(q + shift) % N_QUBITS] = symbol
    return "".join(out)


def decoder_observable_families(alignment: str) -> Dict[str, Tuple[str, ...]]:
    """Canonical decoder families for an alignment."""
    if alignment not in DECODER_PAIRS:
        raise ValueError(f"Unknown decoder alignment: {alignment}")
    pairs = DECODER_PAIRS[alignment]
    fixed = dict(FIXED_OBSERVABLE_FAMILIES)
    same_axis: List[str] = []
    mixed_axis: List[str] = []
    for q1, q2 in pairs:
        same_axis += [pauli_string({q1: "X", q2: "X"}), pauli_string({q1: "Y", q2: "Y"})]
        mixed_axis += [pauli_string({q1: "X", q2: "Y"}), pauli_string({q1: "Y", q2: "X"})]
    return {**fixed, "aligned_same_axis": tuple(same_axis),
            "aligned_mixed_axis": tuple(mixed_axis)}


def measurement_groups(alignment: str) -> Dict[str, Tuple[str, ...]]:
    families = decoder_observable_families(alignment)
    pairs = DECODER_PAIRS[alignment]
    mixed_a = ["I"] * N_QUBITS
    mixed_b = ["I"] * N_QUBITS
    for q1, q2 in pairs:
        mixed_a[q1], mixed_a[q2] = "X", "Y"
        mixed_b[q1], mixed_b[q2] = "Y", "X"
    same_axis = families["aligned_same_axis"]
    mixed_axis = families["aligned_mixed_axis"]
    return {
        "ZZZZ": (*families["local"], *families["z_structure"]),
        "XXXX": tuple(o for o in same_axis if "Y" not in o),
        "YYYY": tuple(o for o in same_axis if "X" not in o),
        "".join(mixed_a): tuple(mixed_axis[0::2]),
        "".join(mixed_b): tuple(mixed_axis[1::2]),
    }


def pauli_matrix(symbol: str) -> np.ndarray:
    if symbol == "I":
        return np.eye(2, dtype=complex)
    if symbol == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if symbol == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if symbol == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(f"Unknown Pauli symbol: {symbol}")


def pauli_operator(observable: str) -> np.ndarray:
    operator = pauli_matrix(observable[0])
    for symbol in observable[1:]:
        operator = np.kron(operator, pauli_matrix(symbol))
    return operator


_INDICES = np.arange(DIM)
_BITS = (_INDICES[:, None] >> (N_QUBITS - 1 - np.arange(N_QUBITS)[None, :])) & 1
_EIGENVALUES = 1.0 - 2.0 * _BITS


def _sign_table(group: Tuple[str, ...]) -> np.ndarray:
    columns = []
    for observable in group:
        active = [q for q, s in enumerate(observable) if s != "I"]
        columns.append(np.prod(_EIGENVALUES[:, active], axis=1))
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class DecoderSpec:
    alignment: str
    observables: Tuple[str, ...]
    family_index: np.ndarray
    operators: np.ndarray
    bases: Tuple[str, ...]
    basis_observable_indices: Tuple[np.ndarray, ...]
    basis_sign_tables: Tuple[np.ndarray, ...]


def _build_spec(alignment: str, frame: int = 0) -> DecoderSpec:
    """Decoder for an alignment, in the canonical frame (0) or the sigma-frame (1).

    In the sigma-frame the decoder for alignment a is the sigma-image of the
    canonical decoder for sigma(a): every observable and every measurement basis
    is relabelled, and the slot order is preserved."""
    source = SIGMA_ALIGNMENT[alignment] if frame else alignment
    families = decoder_observable_families(source)
    groups = measurement_groups(source)
    if frame:
        families = {fam: tuple(relabel_observable(o) for o in obs) for fam, obs in families.items()}
        groups = {relabel_observable(b): tuple(relabel_observable(o) for o in g) for b, g in groups.items()}
    observables = tuple(o for fam in FAMILY_NAMES for o in families[fam])
    family_index = np.array([i for i, fam in enumerate(FAMILY_NAMES) for _ in families[fam]], dtype=int)
    operators = np.stack([pauli_operator(o) for o in observables], axis=0)
    position = {o: i for i, o in enumerate(observables)}
    bases, index_arrays, sign_tables = [], [], []
    for basis, group in groups.items():
        bases.append(basis)
        index_arrays.append(np.array([position[o] for o in group], dtype=int))
        sign_tables.append(_sign_table(group))
    return DecoderSpec(alignment=alignment, observables=observables, family_index=family_index,
                       operators=operators, bases=tuple(bases),
                       basis_observable_indices=tuple(index_arrays),
                       basis_sign_tables=tuple(sign_tables))


DECODER_SPECS: Dict[str, DecoderSpec] = {a: _build_spec(a) for a in DECODER_ALIGNMENTS}
DECODER_SPECS_SIGMA: Dict[str, DecoderSpec] = {a: _build_spec(a, frame=1) for a in DECODER_ALIGNMENTS}


def decoder_spec(alignment: str, topology: str) -> DecoderSpec:
    """The decoder a circuit of the given topology reads for an alignment."""
    if USE_SIGMA_FRAME and CIRCUIT_FRAME[topology] == 1:
        return DECODER_SPECS_SIGMA[alignment]
    return DECODER_SPECS[alignment]


def validate_factorial_design(degree: int, n_qubits: int) -> None:
    if n_qubits != N_QUBITS:
        raise ValueError("This study is defined for exactly four qubits.")
    base.validate_model_size(degree, n_qubits)
    expected = len(base.odd_degrees(degree))
    for alignment in DECODER_ALIGNMENTS:
        if not np.array_equal(DECODER_SPECS[alignment].family_index,
                              DECODER_SPECS_SIGMA[alignment].family_index):
            raise ValueError(f"{alignment}: decoder families differ between frames.")
    for alignment, spec in [*DECODER_SPECS.items(), *DECODER_SPECS_SIGMA.items()]:
        if len(spec.observables) != expected:
            raise ValueError(f"{alignment}: {len(spec.observables)} observables, expected {expected}.")
        grouped = [spec.observables[i] for idx in spec.basis_observable_indices for i in idx]
        if sorted(grouped) != sorted(spec.observables):
            raise ValueError(f"{alignment}: bases do not partition the observables.")
        if len(spec.bases) != 5:
            raise ValueError(f"{alignment}: expected 5 bases, got {len(spec.bases)}.")
        for basis, idx in zip(spec.bases, spec.basis_observable_indices):
            for i in idx:
                for b_sym, p_sym in zip(basis, spec.observables[i]):
                    if p_sym != "I" and p_sym != b_sym:
                        raise ValueError(f"{spec.observables[i]} not measurable in basis {basis}.")


def relabel_circuit_params(circuit_params: np.ndarray, n_layers: int) -> np.ndarray:
    """Parameters of the mirror circuit equivalent to nearest-circuit parameters:
    each single-qubit R_y block is permuted by sigma; the pair angles keep their order."""
    out = np.array(circuit_params, dtype=float).copy()
    index = 0
    for _ in range(n_layers):
        block = circuit_params[index:index + N_QUBITS]
        out[index:index + N_QUBITS] = [block[(q - 1) % N_QUBITS] for q in range(N_QUBITS)]
        index += N_QUBITS + len(PAIR_TOPOLOGIES["nearest"])
    block = circuit_params[index:index + N_QUBITS]
    out[index:index + N_QUBITS] = [block[(q - 1) % N_QUBITS] for q in range(N_QUBITS)]
    return out


def validate_relabelling_equivalence(layer_values=(1, 2, 3), trials: int = 3,
                                     seed: int = 2024, atol: float = 1e-10) -> None:
    """Proposition 1 as implemented: for every alignment a, the cells (nearest
    circuit, decoder a) and (mirror circuit, decoder sigma(a)) give the same
    ordered expectation vector, and hence the same coefficients, under the
    relabelling of the circuit parameters."""
    rng = np.random.default_rng(seed)
    for n_layers in layer_values:
        for rule in PHASE_RULES:
            for use_bridge in (False, True):
                for _ in range(trials):
                    theta = rng.uniform(-np.pi, np.pi, n_circuit_params(n_layers))
                    state_n = run_factorial_ansatz(theta, f"nearest_{rule}", n_layers, use_bridge)
                    state_m = run_factorial_ansatz(relabel_circuit_params(theta, n_layers),
                                                   f"mirror_{rule}", n_layers, use_bridge)
                    for alignment in DECODER_ALIGNMENTS:
                        near = exact_expectation_vector(state_n, alignment, "nearest")
                        mirr = exact_expectation_vector(state_m, SIGMA_ALIGNMENT[alignment], "mirror")
                        err = float(np.max(np.abs(near - mirr)))
                        assert err <= atol, ("relabelling equivalence fails", n_layers, rule,
                                             use_bridge, alignment, err)


def exact_expectation_vector(state: np.ndarray, alignment: str, topology: str) -> np.ndarray:
    operators = decoder_spec(alignment, topology).operators
    values = np.einsum("i,kij,j->k", state.conj(), operators, state, optimize=True)
    if np.max(np.abs(values.imag)) > 1e-8:
        raise FloatingPointError("Pauli expectations have imaginary component.")
    return np.clip(values.real, -1.0, 1.0)


def n_pairs() -> int:
    return 2


def n_circuit_params(n_layers: int) -> int:
    return n_layers * (N_QUBITS + n_pairs()) + N_QUBITS


CPHASE_MASKS: Dict[Tuple[int, int], np.ndarray] = {}


def cphase_mask(q1: int, q2: int) -> np.ndarray:
    key = tuple(sorted((q1, q2)))
    if key not in CPHASE_MASKS:
        bit1 = (_INDICES >> (N_QUBITS - 1 - key[0])) & 1
        bit2 = (_INDICES >> (N_QUBITS - 1 - key[1])) & 1
        CPHASE_MASKS[key] = (bit1 == 1) & (bit2 == 1)
    return CPHASE_MASKS[key]


def apply_cphase_fast(state: np.ndarray, q1: int, q2: int, phi: float) -> np.ndarray:
    output = state.copy()
    output[cphase_mask(q1, q2)] *= np.exp(1j * phi)
    return output


def parse_method(method: str) -> Tuple[str, str]:
    topology, phase_rule = method.rsplit("_", maxsplit=1)
    if topology not in PAIR_TOPOLOGIES:
        raise ValueError(f"Unknown topology in method {method}.")
    if phase_rule not in PHASE_RULES:
        raise ValueError(f"Unknown phase rule in method {method}.")
    return topology, phase_rule


def run_factorial_ansatz(circuit_params: np.ndarray, method: str, n_layers: int, use_bridge: bool) -> np.ndarray:
    expected = n_circuit_params(n_layers)
    if len(circuit_params) != expected:
        raise ValueError(f"{method} expected {expected} circuit parameters, got {len(circuit_params)}.")
    topology, phase_rule = parse_method(method)
    pairs = PAIR_TOPOLOGIES[topology]
    bridge = BRIDGE_EDGES[topology]
    state = np.zeros(DIM, dtype=complex)
    state[0] = 1.0
    index = 0
    for layer in range(n_layers):
        for qubit in range(N_QUBITS):
            state = base.apply_1q(state, base.ry(float(circuit_params[index])), qubit, N_QUBITS)
            index += 1
        for pair_index, (q1, q2) in enumerate(pairs):
            phi = float(circuit_params[index])
            index += 1
            if phase_rule == "uniform":
                sign = 1.0
            else:
                sign = 1.0 if (layer + pair_index) % 2 == 0 else -1.0
            state = apply_cphase_fast(state, q1, q2, phi)
            state = base.apply_1q(state, base.rz(sign * phi / 2.0), q1, N_QUBITS)
            state = base.apply_1q(state, base.rz(-sign * phi / 2.0), q2, N_QUBITS)
        if use_bridge:
            state = apply_cphase_fast(state, bridge[0], bridge[1], np.pi)
    for qubit in range(N_QUBITS):
        state = base.apply_1q(state, base.ry(float(circuit_params[index])), qubit, N_QUBITS)
        index += 1
    return state


@dataclass(frozen=True)
class PolynomialCache:
    design_masked: np.ndarray
    target_masked: np.ndarray
    monomial_map: np.ndarray
    derivative_masked: np.ndarray | None


def build_polynomial_cache(t, y, mask, degree, alpha_derivative) -> PolynomialCache:
    design, _ = base.design_matrix(t, degree)
    monomial_map = base.cheb_to_monomial_matrix(degree)
    derivative_masked = None
    if alpha_derivative > 0.0:
        n_coeffs = len(base.odd_degrees(degree))
        derivative_masked = np.empty((int(np.sum(mask)), n_coeffs), dtype=float)
        for j in range(n_coeffs):
            unit = np.zeros(n_coeffs)
            unit[j] = 1.0
            derivative_masked[:, j] = base.derivative_values(t[mask], unit, degree)
    return PolynomialCache(design_masked=design[mask], target_masked=y[mask],
                           monomial_map=monomial_map, derivative_masked=derivative_masked)


def fast_regularized_loss(coefficients, cache: PolynomialCache, cfg) -> float:
    residual = cache.design_masked @ coefficients - cache.target_masked
    fit = float(np.mean(residual ** 2))
    cheb_penalty = float(np.dot(coefficients, coefficients))
    monomial_penalty = float(np.log1p(np.linalg.norm(cache.monomial_map @ coefficients)))
    derivative_penalty = 0.0
    if cache.derivative_masked is not None:
        derivative = cache.derivative_masked @ coefficients
        derivative_penalty = float(np.mean(derivative ** 2))
    return (fit + cfg.alpha_cheb * cheb_penalty + cfg.alpha_monomial * monomial_penalty
            + cfg.alpha_derivative * derivative_penalty)


@dataclass(frozen=True)
class GridCache:
    design_masked: np.ndarray
    target_masked: np.ndarray


def build_grid_cache(t, y, mask, degree) -> GridCache:
    design, _ = base.design_matrix(t, degree)
    return GridCache(design_masked=design[mask], target_masked=y[mask])


def masked_rmse_linf(coefficients, grid: GridCache) -> Tuple[float, float]:
    error = grid.design_masked @ coefficients - grid.target_masked
    return float(np.sqrt(np.mean(error ** 2))), float(np.max(np.abs(error)))


def make_shifted_dataset(q0, samples_per_period, delta, target_type):
    count = int(q0 * samples_per_period)
    x = -q0 / 2.0 + (np.arange(count, dtype=float) + 0.5) * (q0 / count)
    t = 2.0 * x / q0
    if target_type == "sawtooth":
        y = base.centered_sawtooth(x)
        nearest_jump = np.floor(x) + 0.5
        mask = np.abs(x - nearest_jump) >= delta
    elif target_type == "arcsin_sine":
        y = base.arcsin_sine(x)
        phase = np.mod(x, 1.0)
        distance = np.minimum(np.abs(phase - 0.25), np.abs(phase - 0.75))
        mask = distance >= delta
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    return x, t, y, mask


@dataclass(frozen=True)
class DatasetBundle:
    train: tuple
    nested: tuple
    shifted: tuple
    train_cache: PolynomialCache
    nested_cache: GridCache
    shifted_cache: GridCache


def build_dataset_bundles(cfg) -> Dict[int, DatasetBundle]:
    bundles = {}
    for q0 in cfg.q0_values:
        train = base.make_dataset(q0, cfg.train_samples_per_period, cfg.delta, cfg.target_type)
        nested = base.make_dataset(q0, cfg.eval_samples_per_period, cfg.delta, cfg.target_type)
        shifted = make_shifted_dataset(q0, cfg.eval_samples_per_period, cfg.delta, cfg.target_type)
        _, t_train, y_train, mask_train = train
        _, t_nested, y_nested, mask_nested = nested
        _, t_shifted, y_shifted, mask_shifted = shifted
        bundles[q0] = DatasetBundle(
            train=train, nested=nested, shifted=shifted,
            train_cache=build_polynomial_cache(t_train, y_train, mask_train, cfg.degree, cfg.alpha_derivative),
            nested_cache=build_grid_cache(t_nested, y_nested, mask_nested, cfg.degree),
            shifted_cache=build_grid_cache(t_shifted, y_shifted, mask_shifted, cfg.degree),
        )
    return bundles


def n_decoder_scales() -> int:
    return len(FAMILY_NAMES)


def total_model_params(n_layers: int) -> int:
    return n_circuit_params(n_layers) + n_decoder_scales()


def initial_model_params(cfg, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    circuit = rng.uniform(-np.pi, np.pi, size=n_circuit_params(cfg.n_layers))
    log_scales = rng.normal(loc=-1.0, scale=0.25, size=n_decoder_scales())
    return np.concatenate([circuit, log_scales])


def model_bounds(cfg):
    return [(None, None)] * n_circuit_params(cfg.n_layers) + [cfg.log_scale_bounds] * n_decoder_scales()


def unpack_model_params(params, cfg):
    n_circuit = n_circuit_params(cfg.n_layers)
    expected = n_circuit + n_decoder_scales()
    if len(params) != expected:
        raise ValueError(f"Expected {expected} model parameters, got {len(params)}.")
    return np.asarray(params[:n_circuit], dtype=float), np.asarray(params[n_circuit:], dtype=float)


def expectations_to_coefficients(expectations, log_scales, alignment, topology):
    spec = decoder_spec(alignment, topology)
    return np.exp(log_scales)[spec.family_index] * expectations


def evaluate_model(params, *, method, decoder_alignment, cfg, use_bridge):
    circuit_params, log_scales = unpack_model_params(params, cfg)
    state = run_factorial_ansatz(circuit_params, method, cfg.n_layers, use_bridge)
    expectations = exact_expectation_vector(state, decoder_alignment, parse_method(method)[0])
    coefficients = expectations_to_coefficients(expectations, log_scales, decoder_alignment,
                                                parse_method(method)[0])
    return state, expectations, coefficients, log_scales


@dataclass
class TrainedModel:
    q0: int
    n_layers: int
    method: str
    topology: str
    phase_rule: str
    decoder_alignment: str
    seed: int
    params: np.ndarray
    state: np.ndarray
    expectations: np.ndarray
    coefficients: np.ndarray
    log_scales: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    final_objective: float
    objective_evaluations: int
    optimizer_iterations: float
    elapsed_seconds: float


MAXFUN = None   # None -> SciPy default (15,000); set via run_factorial_study(maxfun=...)


def optimizer_options(cfg):
    if cfg.optimizer.upper() != "L-BFGS-B":
        return {"maxiter": cfg.maxiter}
    options = {"maxiter": cfg.maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50}
    if MAXFUN is not None:
        options["maxfun"] = int(MAXFUN)
    return options


def train_model(*, q0, method, decoder_alignment, seed, cache, cfg, use_bridge) -> TrainedModel:
    started = time.perf_counter()
    n_evaluations = 0

    def objective(params):
        nonlocal n_evaluations
        n_evaluations += 1
        _, _, coefficients, _ = evaluate_model(params, method=method, decoder_alignment=decoder_alignment,
                                               cfg=cfg, use_bridge=use_bridge)
        return fast_regularized_loss(coefficients, cache, cfg)

    result = minimize(objective, initial_model_params(cfg, seed), method=cfg.optimizer,
                      bounds=model_bounds(cfg), options=optimizer_options(cfg))
    state, expectations, coefficients, log_scales = evaluate_model(
        result.x, method=method, decoder_alignment=decoder_alignment, cfg=cfg, use_bridge=use_bridge)
    topology, phase_rule = parse_method(method)
    return TrainedModel(q0=q0, n_layers=cfg.n_layers, method=method, topology=topology,
                        phase_rule=phase_rule, decoder_alignment=decoder_alignment, seed=int(seed),
                        params=np.asarray(result.x, dtype=float), state=state, expectations=expectations,
                        coefficients=coefficients, log_scales=log_scales,
                        optimizer_success=bool(result.success), optimizer_message=str(result.message),
                        final_objective=float(result.fun),
                        objective_evaluations=int(getattr(result, "nfev", n_evaluations)),
                        optimizer_iterations=float(getattr(result, "nit", np.nan)),
                        elapsed_seconds=float(time.perf_counter() - started))


def evaluation_metrics(coefficients, *, q0, degree, bundle):
    row = {}
    for prefix, dataset in (("train", bundle.train), ("nested", bundle.nested), ("shifted", bundle.shifted)):
        _, t, y, mask = dataset
        for key, value in base.metrics(t, y, mask, coefficients, q0, degree).items():
            row[f"{prefix}_{key}"] = value
    return row


def model_row(model, *, cfg, bundle, use_bridge):
    row = {
        "q0": model.q0, "target_type": cfg.target_type, "degree": cfg.degree, "n_qubits": cfg.n_qubits,
        "n_layers": model.n_layers, "method": model.method, "topology": model.topology,
        "phase_rule": model.phase_rule, "decoder_alignment": model.decoder_alignment,
        "decoder_matches_topology": model.topology == model.decoder_alignment,
        "bridge_mode": "fixed_cz" if use_bridge else "none", "seed": model.seed,
        "decoder_type": "factorial_pauli_expectation_4_group_scales", "n_observables": 16,
        "n_measurement_bases": 5, "n_circuit_params": n_circuit_params(cfg.n_layers),
        "n_decoder_params": n_decoder_scales(), "n_total_params": total_model_params(cfg.n_layers),
        "optimizer_success": model.optimizer_success, "optimizer_message": model.optimizer_message,
        "final_objective": model.final_objective, "objective_evaluations": model.objective_evaluations,
        "optimizer_iterations": model.optimizer_iterations, "elapsed_seconds": model.elapsed_seconds,
        "optimized_log_scale": float(np.mean(model.log_scales)),
        "optimized_scale": float(np.exp(np.mean(model.log_scales))),
    }
    for index, family in enumerate(FAMILY_NAMES):
        row[f"log_scale_{family}"] = float(model.log_scales[index])
        row[f"scale_{family}"] = float(np.exp(model.log_scales[index]))
    row.update(evaluation_metrics(model.coefficients, q0=model.q0, degree=cfg.degree, bundle=bundle))
    return row


def build_reference_cache(cfg, bundles):
    cache = {}
    n_coeffs = len(base.odd_degrees(cfg.degree))
    for q0, bundle in bundles.items():
        _, t_train, y_train, mask_train = bundle.train
        cache[(q0, "zero_polynomial", None)] = (np.zeros(n_coeffs), None, {})
        cache[(q0, "weighted_ls", None)] = (base.weighted_ls(t_train, y_train, mask_train, cfg.degree), None, {})
        coefficients, result, _, diagnostics = base.direct_log_monomial_fit(t_train, y_train, mask_train, cfg)
        cache[(q0, "direct_log_monomial", None)] = (coefficients, result, diagnostics)
        for seed in cfg.seeds:
            coefficients, result, _, diagnostics = base.direct_spherical_fit(t_train, y_train, mask_train, cfg, int(seed))
            cache[(q0, "direct_spherical", int(seed))] = (coefficients, result, diagnostics)
    return cache


def classical_baseline_rows(*, cfg, q0, bundle, reference_cache):
    rows = []
    for (stored_q0, method, seed), (coefficients, result, diagnostics) in reference_cache.items():
        if stored_q0 != q0:
            continue
        n_params = 0 if method == "zero_polynomial" else len(coefficients)
        row = {
            "q0": q0, "target_type": cfg.target_type, "degree": cfg.degree, "n_qubits": cfg.n_qubits,
            "n_layers": cfg.n_layers, "method": method, "topology": "classical", "phase_rule": "classical",
            "decoder_alignment": "none", "decoder_matches_topology": False, "bridge_mode": "none",
            "seed": np.nan if seed is None else seed, "decoder_type": "classical_baseline",
            "n_observables": np.nan, "n_measurement_bases": np.nan, "n_circuit_params": 0,
            "n_decoder_params": n_params, "n_total_params": n_params,
            "optimizer_success": np.nan if result is None else bool(result.success),
            "optimizer_message": "" if result is None else str(result.message),
            "final_objective": np.nan if result is None else float(result.fun),
            "objective_evaluations": diagnostics.get("objective_evaluations", np.nan),
            "optimizer_iterations": diagnostics.get("optimizer_iterations", np.nan),
            "elapsed_seconds": diagnostics.get("elapsed_seconds", np.nan),
            "optimized_log_scale": diagnostics.get("optimized_log_scale", np.nan),
            "optimized_scale": diagnostics.get("optimized_scale", np.nan),
        }
        for family in FAMILY_NAMES:
            row[f"log_scale_{family}"] = np.nan
            row[f"scale_{family}"] = np.nan
        row.update(evaluation_metrics(coefficients, q0=q0, degree=cfg.degree, bundle=bundle))
        rows.append(row)
    return rows


H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
S_DAGGER_GATE = np.array([[1, 0], [0, -1j]], dtype=complex)


def rotate_state_to_basis(state, basis):
    output = np.asarray(state, dtype=complex)
    for qubit, symbol in enumerate(basis):
        if symbol == "Z":
            continue
        if symbol == "X":
            output = base.apply_1q(output, H_GATE, qubit, N_QUBITS)
        elif symbol == "Y":
            output = base.apply_1q(output, S_DAGGER_GATE, qubit, N_QUBITS)
            output = base.apply_1q(output, H_GATE, qubit, N_QUBITS)
        else:
            raise ValueError(f"Unsupported basis symbol: {symbol}")
    return output


def prepare_measurements(state, spec):
    cumulatives = []
    for basis in spec.bases:
        rotated = rotate_state_to_basis(state, basis)
        probabilities = np.clip((np.abs(rotated) ** 2).real, 0.0, None)
        probabilities /= probabilities.sum()
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        cumulatives.append(cumulative)
    return tuple(cumulatives)


def stable_uint32(text):
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False)


def common_uniforms(*, random_seed, q0, n_layers, training_seed, shots_per_basis, repetition, basis):
    sequence = np.random.SeedSequence([random_seed, q0, n_layers, training_seed, shots_per_basis,
                                       repetition, stable_uint32(basis)])
    return np.random.default_rng(sequence).random(shots_per_basis)


def grouped_shot_expectations(*, cumulatives, spec, q0, n_layers, training_seed, shots_per_basis,
                              repetition, random_seed):
    estimates = np.empty(len(spec.observables), dtype=float)
    for basis, cumulative, indices, signs in zip(spec.bases, cumulatives, spec.basis_observable_indices,
                                                 spec.basis_sign_tables):
        uniforms = common_uniforms(random_seed=random_seed, q0=q0, n_layers=n_layers,
                                   training_seed=training_seed, shots_per_basis=shots_per_basis,
                                   repetition=repetition, basis=basis)
        outcomes = np.searchsorted(cumulative, uniforms, side="right")
        counts = np.bincount(outcomes, minlength=DIM).astype(float)
        estimates[indices] = counts @ signs / shots_per_basis
    return estimates


def add_comparative_columns(frame, reference_frame, cfg, grids=("train", "nested", "shifted")):
    frame = frame.copy()
    for grid in grids:
        column = f"{grid}_masked_rmse"
        if column not in frame.columns:
            continue
        zero = reference_frame.loc[reference_frame["method"] == "zero_polynomial"].groupby("q0")[column].first()
        reference = reference_frame.loc[reference_frame["method"] == cfg.reference_method].groupby("q0")[column].first()
        z = frame["q0"].map(zero)
        r = frame["q0"].map(reference)
        denominator = z - r
        frame[f"{grid}_recovery_fraction"] = np.where(np.abs(denominator) > 1e-12,
                                                      (z - frame[column]) / denominator, np.nan)
        frame[f"{grid}_success"] = frame[column] <= r + cfg.success_tolerance
    return frame


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


N_BOOT = 2000
BOOT_SEED = 20260922


def _bootstrap_median_ci(differences, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap interval for the median paired difference."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    medians = np.median(differences[idx], axis=1)
    lo, hi = np.quantile(medians, [0.025, 0.975])
    return float(lo), float(hi)


def _wilcoxon_row(differences, context):
    differences = np.asarray(differences, dtype=float)
    # Exact zeros, matching the Pratt handling inside scipy.stats.wilcoxon.
    n_zero = int(np.sum(differences == 0.0))
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(differences, zero_method="pratt", alternative="two-sided")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    ci_low, ci_high = _bootstrap_median_ci(differences)
    return {**context, "n_pairs": len(differences), "n_zero_differences": n_zero,
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "ci_low": ci_low, "ci_high": ci_high,
            "wilcoxon_statistic": statistic, "p_value": p_value}


def paired_factorial_tests(circuit_exact, metric="nested_masked_rmse"):
    rows = []
    for (q0, layers, topology, decoder), group in circuit_exact.groupby(
            ["q0", "n_layers", "topology", "decoder_alignment"]):
        pivot = group.pivot(index="seed", columns="phase_rule", values=metric).dropna()
        if {"uniform", "oriented"} <= set(pivot.columns) and len(pivot):
            rows.append(_wilcoxon_row((pivot["oriented"] - pivot["uniform"]).to_numpy(),
                                      {"contrast": "oriented_minus_uniform", "q0": int(q0),
                                       "n_layers": int(layers), "topology": topology,
                                       "decoder_alignment": decoder, "metric": metric,
                                       "phase_rule": "both"}))
    for (q0, layers, method), group in circuit_exact.groupby(["q0", "n_layers", "method"]):
        topology = group["topology"].iloc[0]
        pivot = group.pivot(index="seed", columns="decoder_alignment", values=metric).dropna()
        others = [c for c in pivot.columns if c != topology]
        if topology in pivot.columns and others and len(pivot):
            rows.append(_wilcoxon_row((pivot[topology] - pivot[others].mean(axis=1)).to_numpy(),
                                      {"contrast": "matched_minus_mean_mismatched", "q0": int(q0),
                                       "n_layers": int(layers), "topology": topology,
                                       "decoder_alignment": "n/a", "metric": metric,
                                       "phase_rule": group["phase_rule"].iloc[0]}))
    matched = circuit_exact[circuit_exact["decoder_matches_topology"]]
    for (q0, layers, phase_rule), group in matched.groupby(["q0", "n_layers", "phase_rule"]):
        pivot = group.pivot(index="seed", columns="topology", values=metric).dropna()
        for other in ("nearest", "mirror"):
            if {"crossed", other} <= set(pivot.columns) and len(pivot):
                rows.append(_wilcoxon_row((pivot["crossed"] - pivot[other]).to_numpy(),
                                          {"contrast": f"crossed_minus_{other}_matched", "q0": int(q0),
                                           "n_layers": int(layers), "topology": f"crossed_vs_{other}",
                                           "decoder_alignment": "matched", "metric": metric,
                                           "phase_rule": phase_rule}))
    # No Holm adjustment here: the manuscript adjusts the whole family of
    # 228 p-values (both metrics) jointly, which the caller does.
    return pd.DataFrame(rows)


def _save_frame(frame, output_dir, stem):
    frame.to_csv(os.path.join(output_dir, f"{stem}.csv"), index=False)
    try:
        frame.to_parquet(os.path.join(output_dir, f"{stem}.parquet"), index=False)
    except (ImportError, ValueError, ModuleNotFoundError):
        pass


def run_factorial_study(*, q0_values=(8, 16), layer_values=(1, 2, 3), seeds=tuple(range(20)), maxiter=3000,
                        shots_per_basis_values=(100, 500, 1000, 5000), shot_repetitions=30,
                        shot_random_seed=987654, use_bridge=False, maxfun=None,
                        output_dir=os.path.join("results", "evalmod_factorial_observable_study")):
    """The reported run uses maxfun=1,000,000 (--maxfun 1000000); maxfun=None
    uses SciPy's default cap of 15,000 evaluations."""
    global MAXFUN
    MAXFUN = maxfun
    os.makedirs(output_dir, exist_ok=True)
    template_cfg = base.Config(q0_values=q0_values, degree=31, delta=0.18, train_samples_per_period=32,
                               eval_samples_per_period=128, seeds=seeds, n_qubits=N_QUBITS, n_layers=1,
                               optimizer="L-BFGS-B", maxiter=maxiter, target_type="sawtooth",
                               alpha_cheb=1e-8, alpha_monomial=1e-4, alpha_derivative=0.0,
                               success_tolerance=1e-4, reference_method="direct_log_monomial",
                               output_dir=output_dir)
    validate_factorial_design(template_cfg.degree, template_cfg.n_qubits)
    validate_relabelling_equivalence(layer_values=tuple(layer_values))
    bundles = build_dataset_bundles(template_cfg)
    reference_cache = build_reference_cache(template_cfg, bundles)
    exact_rows, shot_rows, parameter_store = [], [], {}
    for n_layers in layer_values:
        layer_cfg = replace(template_cfg, n_layers=int(n_layers))
        for q0 in q0_values:
            bundle = bundles[q0]
            exact_rows.extend(classical_baseline_rows(cfg=layer_cfg, q0=q0, bundle=bundle,
                                                      reference_cache=reference_cache))
            for method in CIRCUIT_METHODS:
                for decoder_alignment in DECODER_ALIGNMENTS:
                    spec = decoder_spec(decoder_alignment, parse_method(method)[0])
                    for seed in seeds:
                        model = train_model(q0=q0, method=method, decoder_alignment=decoder_alignment,
                                            seed=int(seed), cache=bundle.train_cache, cfg=layer_cfg,
                                            use_bridge=use_bridge)
                        parameter_store[f"q0_{q0}__L_{n_layers}__{method}__dec_{decoder_alignment}__seed_{seed}"] = model.params
                        exact_rows.append(model_row(model, cfg=layer_cfg, bundle=bundle, use_bridge=use_bridge))
                        cumulatives = prepare_measurements(model.state, spec)
                        exact_nested_rmse, _ = masked_rmse_linf(model.coefficients, bundle.nested_cache)
                        exact_shifted_rmse, _ = masked_rmse_linf(model.coefficients, bundle.shifted_cache)
                        for shots_per_basis in shots_per_basis_values:
                            for repetition in range(shot_repetitions):
                                sampled = grouped_shot_expectations(
                                    cumulatives=cumulatives, spec=spec, q0=q0, n_layers=int(n_layers),
                                    training_seed=int(seed), shots_per_basis=int(shots_per_basis),
                                    repetition=repetition, random_seed=shot_random_seed)
                                coefficients = expectations_to_coefficients(sampled, model.log_scales,
                                                                            decoder_alignment, model.topology)
                                nested_rmse, nested_linf = masked_rmse_linf(coefficients, bundle.nested_cache)
                                shifted_rmse, shifted_linf = masked_rmse_linf(coefficients, bundle.shifted_cache)
                                shot_rows.append({
                                    "q0": q0, "degree": layer_cfg.degree, "n_layers": int(n_layers),
                                    "method": method, "topology": model.topology,
                                    "phase_rule": model.phase_rule, "decoder_alignment": decoder_alignment,
                                    "decoder_matches_topology": model.topology == decoder_alignment,
                                    "bridge_mode": "fixed_cz" if use_bridge else "none", "seed": int(seed),
                                    "shots_per_basis": int(shots_per_basis), "n_measurement_bases": 5,
                                    "total_state_preparations": int(5 * shots_per_basis),
                                    "shot_repetition": repetition, "nested_masked_rmse": nested_rmse,
                                    "nested_masked_linf": nested_linf, "shifted_masked_rmse": shifted_rmse,
                                    "shifted_masked_linf": shifted_linf, "nested_exact_rmse": exact_nested_rmse,
                                    "shifted_exact_rmse": exact_shifted_rmse,
                                    "nested_rmse_degradation": nested_rmse - exact_nested_rmse,
                                    "shifted_rmse_degradation": shifted_rmse - exact_shifted_rmse})
    exact_frame = pd.DataFrame(exact_rows)
    exact_frame = add_comparative_columns(exact_frame, exact_frame, template_cfg)
    shot_frame = pd.DataFrame(shot_rows)
    shot_frame = add_comparative_columns(shot_frame, exact_frame, template_cfg, grids=("nested", "shifted"))
    exact_frame.to_csv(os.path.join(output_dir, "factorial_exact_results.csv"), index=False)
    _save_frame(shot_frame, output_dir, "factorial_grouped_shot_results")
    np.savez_compressed(os.path.join(output_dir, "optimized_model_parameters.npz"), **parameter_store)
    circuit_exact = exact_frame[exact_frame["decoder_type"] == "factorial_pauli_expectation_4_group_scales"].copy()
    factor_keys = ["q0", "n_layers", "topology", "phase_rule", "decoder_alignment", "decoder_matches_topology"]
    exact_summary = circuit_exact.groupby(factor_keys, as_index=False).agg(
        median_nested_rmse=("nested_masked_rmse", "median"),
        median_shifted_rmse=("shifted_masked_rmse", "median"),
        median_nested_recovery=("nested_recovery_fraction", "median"),
        median_shifted_recovery=("shifted_recovery_fraction", "median"),
        success_rate_nested=("nested_success", "mean"),
        optimizer_success_rate=("optimizer_success", "mean"),
        median_evaluations=("objective_evaluations", "median"),
        median_seconds=("elapsed_seconds", "median"))
    exact_summary.to_csv(os.path.join(output_dir, "factorial_exact_summary.csv"), index=False)
    if len(shot_frame):
        shot_summary = shot_frame.groupby(factor_keys + ["shots_per_basis", "total_state_preparations"], as_index=False).agg(
            median_nested_rmse=("nested_masked_rmse", "median"),
            q25_nested_rmse=("nested_masked_rmse", lambda v: float(np.quantile(v, 0.25))),
            q75_nested_rmse=("nested_masked_rmse", lambda v: float(np.quantile(v, 0.75))),
            median_shifted_rmse=("shifted_masked_rmse", "median"),
            median_nested_degradation=("nested_rmse_degradation", "median"))
        shot_summary.to_csv(os.path.join(output_dir, "factorial_grouped_shot_summary.csv"), index=False)
    interaction_summary = circuit_exact.groupby(["q0", "n_layers", "decoder_matches_topology"], as_index=False).agg(
        median_nested_rmse=("nested_masked_rmse", "median"),
        median_shifted_rmse=("shifted_masked_rmse", "median"),
        median_nested_recovery=("nested_recovery_fraction", "median"))
    interaction_summary.to_csv(os.path.join(output_dir, "decoder_match_interaction_summary.csv"), index=False)
    tests = pd.concat([paired_factorial_tests(circuit_exact, metric="nested_masked_rmse"),
                       paired_factorial_tests(circuit_exact, metric="shifted_masked_rmse")], ignore_index=True)
    if len(tests):
        tests["p_holm"] = holm_adjust(tests["p_value"].to_numpy())
    tests.to_csv(os.path.join(output_dir, "factorial_paired_tests.csv"), index=False)

    limit = circuit_exact["optimizer_message"].fillna("").str.contains("LIMIT", case=False)
    config = {
        "q0_values": list(q0_values), "degree": template_cfg.degree,
        "delta": template_cfg.delta, "target_type": template_cfg.target_type,
        "layer_values": list(layer_values), "seeds": list(seeds),
        "maxiter": maxiter, "maxfun": "scipy default (15000)" if maxfun is None else int(maxfun),
        "pair_topologies": {k: [list(p) for p in v] for k, v in PAIR_TOPOLOGIES.items()},
        "use_bridge": use_bridge, "phase_rules": list(PHASE_RULES),
        "decoder_alignments": list(DECODER_ALIGNMENTS),
        "shots_per_basis_values": list(shots_per_basis_values),
        "shot_repetitions": shot_repetitions, "shot_random_seed": shot_random_seed,
        "n_circuit_optimisations": int(len(circuit_exact)),
        "n_distinct_classical_optimisations": int(
            exact_frame[exact_frame["decoder_type"] == "classical_baseline"]
            .drop_duplicates(["q0", "method", "seed", "nested_masked_rmse"])["method"]
            .isin(["direct_spherical", "direct_log_monomial"]).sum()),
        "n_contrasts_per_metric": int(len(tests) // 2),
        "holm_family_size": int(len(tests)),
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED},
        "optimizer_limit_fraction": {
            f"q0={int(k[0])},L={int(k[1])}": round(float(v), 3)
            for k, v in limit.groupby([circuit_exact["q0"], circuit_exact["n_layers"]]).mean().items()},
        "training_expectations": "exact",
        "finite_shot_noise": "post hoc grouped measurement sensitivity",
    }
    with open(os.path.join(output_dir, "factorial_study_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"resolved after Holm (p<0.05): {int((tests['p_holm'] < 0.05).sum())} of {len(tests)}")
    print("optimizer-limit fraction by (q0, L):", config["optimizer_limit_fraction"])
    return exact_frame, tests


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Topology-factorial EvalMod study. Nothing expensive runs without --smoke or --full.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny end-to-end run of every output path")
    mode.add_argument("--full", action="store_true", help="the full 2,160-optimisation study")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--maxfun", type=int, default=None,
                        help="function-evaluation cap; omit to reproduce the published run")
    parser.add_argument("--bridge", action="store_true",
                        help="structural diagnostic with a fixed CZ bridge; not used for the reported results")
    args = parser.parse_args()
    if args.smoke:
        run_factorial_study(q0_values=(8,), layer_values=(1,), seeds=(0, 1), maxiter=5,
                            shots_per_basis_values=(100,), shot_repetitions=2,
                            use_bridge=args.bridge, maxfun=args.maxfun,
                            output_dir=args.output_dir or os.path.join("results", "evalmod_factorial_smoke_test"))
    else:
        run_factorial_study(use_bridge=args.bridge, maxfun=args.maxfun,
                            output_dir=args.output_dir or os.path.join("results", "evalmod_factorial_observable_study"))


if __name__ == "__main__":
    _cli()

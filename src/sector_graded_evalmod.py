"""
CKKS-VQA Sector-Graded Fourier Synthesis
========================================

Canonical implementation of the principal sector-graded VQA experiment used
in the CKKS coefficient-synthesis study.

Purpose
-------
The notebook studies a classical-output variational quantum algorithm for
period-one Fourier synthesis,

    p(x) = sum_{k=1}^{16} c_k sin(2*pi*k*x),

using a four-qubit variational circuit whose Pauli expectation values are
decoded into 16 classical Fourier coefficients.

The main goals are to separate:

    * structural restrictions imposed by the ansatz,
    * approximation cost of the imposed sector,
    * reachable coefficient-space dimension,
    * optimizer behaviour,
    * decoder/preconditioning effects, and
    * finite-shot reconstruction cost.

The principal experiments use Pauli-observable decoding with explicit positive
decoder scales. Amplitude encoding and signed-scale profiling are not part of
the main model.

Model and grading
-----------------
Computational-basis index j corresponds to harmonic k = j + 1.
Qubit 3 is the least-significant-bit / harmonic-parity qubit.

Half-period translation induces the coefficient grading

    c_k -> (-1)^(k+1) c_k.

The corresponding register-level transport identity is

    Z3 O_k Z3 = (-1)^(k+1) O_k.

Thus the odd-harmonic sector corresponds to observables commuting with Z3,
while complementary-sector observables anticommute with Z3.

This grading was motivated by the sign structure associated with negacyclic
CKKS arithmetic, but no identification between the Fourier coefficient space
and the CKKS plaintext ring is assumed.

Decoder
-------
Sixteen Pauli expectations are grouped into four decoder families:

    odd harmonics 1,3,5,7
        odd_z  : ZIII IZII IIZI ZZII      measured in ZZZZ

    odd harmonics 9,11,13,15
        odd_x  : XIII IXII IIXI XXII      measured in XXXX

    even harmonics 2,4,6,8
        even_x : IIIX XIIX IXIX IIXX      measured in XXXX

    even harmonics 10,12,14,16
        even_y : IIIY YIIY IYIY IIYY      measured in YYYY

Decoded coefficients are

    c_k = exp(s_g(k)) * w_k * mu_k,

where mu_k is the Pauli expectation value, s_g(k) is a trainable family
log-scale, and

    w_k = k^(-p)

is an optional fixed diagonal preconditioner. Setting p = 0 gives the
unpreconditioned decoder.

Circuit families
----------------
Each model has nominally

    7L + 5 circuit parameters + 4 decoder scales.

generic
    RY acts on qubit 3 throughout; no sector restriction is imposed.

equivariant
    RZ acts on qubit 3 throughout. The circuit commutes with Z3 and the
    even-harmonic decoder sector vanishes exactly.

sector_mixed
    Layer bodies preserve the symmetry, but a final RY on qubit 3 breaks it,
    followed by a trainable CPhase(2,3).

Classical references
--------------------
Both least-squares and ridge references are computed separately.

    OLS references:
        used for RMSE lower-bound checks.

    Ridge references:
        used for the regularized training-objective lower-bound checks.

The two references must not be used interchangeably.

Automated validation checks
---------------------------
P1
    Raw, unprojected complementary-sector expectations of the equivariant
    model vanish to numerical precision.

P2a
    Odd-sector model training RMSE cannot beat the odd-sector OLS optimum.

P2b
    Odd-sector regularized training objective cannot beat the odd-sector
    ridge optimum.

P3a
    Numerical Jacobian rank cannot exceed the structural sector bound.

P3b
    The equivariant model is tested empirically for saturation of the
    eight-dimensional sector bound at sampled parameter points.
    Numerical rank uses relative singular-value threshold rtol = 1e-8.

P4
    Finite-shot coefficient reconstruction is checked against the analytical
    variance model using RMS quantities, with total state preparations as the
    measurement budget.

P5
    The three-target experiment tests the cost of the odd-sector prior:
        triangle       -> negligible cost,
        sawtooth       -> partial cost,
        double sawtooth -> essentially total cost.

Empirical comparisons
---------------------
Preconditioning is compared between

    p = 0        unweighted decoder
    matched p    target-decay-matched decoder.

Differences are evaluated relative to the same classical references.
Preconditioning results are interpreted as optimization/reparameterization
effects, not changes in the reachable output dimension.

Execution
---------
Loading this cell in Jupyter only defines the study functions.

Run a small smoke test with

    result = main([])

Run the full sector experiment with

    result = main([
        "--full",
        "--output-root",
        "sector_results_full"
    ])

The full experiment runs both the unpreconditioned and matched-preconditioner
conditions and stores exact, shot-based, rank, statistical-test, and validation
outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon

# ============================================================
# 0. Robust, atomic output
# ============================================================


def _timestamped_sibling(path: str) -> str:
    stem, ext = os.path.splitext(path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = f"{stem}_{stamp}{ext}"
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{stem}_{stamp}_{counter}{ext}"
        counter += 1
    return candidate


def _atomic_write(path: str, writer) -> str:
    """Write to a temporary sibling and atomically replace the destination."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sector_tmp_", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
        return path
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def safe_write_csv(frame: pd.DataFrame, path: str) -> str:
    try:
        return _atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False))
    except OSError as exc:
        alt = _timestamped_sibling(path)
        _atomic_write(alt, lambda tmp: frame.to_csv(tmp, index=False))
        print(f"  WARNING: could not replace {os.path.basename(path)} "
              f"({exc}); wrote {os.path.basename(alt)} instead")
        return alt


def safe_write_json(payload: dict, path: str) -> str:
    def writer(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    try:
        return _atomic_write(path, writer)
    except OSError as exc:
        alt = _timestamped_sibling(path)
        _atomic_write(alt, writer)
        print(f"  WARNING: could not replace {os.path.basename(path)} "
              f"({exc}); wrote {os.path.basename(alt)} instead")
        return alt


# ============================================================
# 1. Harmonics, observables, sector grading
# ============================================================

N_QUBITS = 4
DIM = 16
N_HARMONICS = 16
HARMONICS = np.arange(1, N_HARMONICS + 1)

# Bit convention: bit(state s, qubit q) = (s >> (3 - q)) & 1.
PARITY_QUBIT = 3

ODD_Z_OBSERVABLES = ("ZIII", "IZII", "IIZI", "ZZII")
ODD_X_OBSERVABLES = ("XIII", "IXII", "IIXI", "XXII")
EVEN_X_OBSERVABLES = ("IIIX", "XIIX", "IXIX", "IIXX")
EVEN_Y_OBSERVABLES = ("IIIY", "YIIY", "IYIY", "IIYY")

OBSERVABLES = (
    ODD_Z_OBSERVABLES + ODD_X_OBSERVABLES + EVEN_X_OBSERVABLES + EVEN_Y_OBSERVABLES
)

OBSERVABLE_HARMONIC = np.array(
    [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16], dtype=int
)

FAMILY_NAMES = ("odd_z", "odd_x", "even_x", "even_y")
FAMILY_INDEX = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4, dtype=int)

ODD_SLICE = slice(0, 8)
EVEN_SLICE = slice(8, 16)

MEASUREMENT_BASES: Dict[str, Tuple[int, ...]] = {
    "ZZZZ": (0, 1, 2, 3),
    "XXXX": (4, 5, 6, 7, 8, 9, 10, 11),
    "YYYY": (12, 13, 14, 15),
}
BASES_ALL = ("ZZZZ", "XXXX", "YYYY")
BASES_EQUIVARIANT = ("ZZZZ", "XXXX")

ANSATZ_KINDS = ("generic", "equivariant", "sector_mixed")


def bases_for(kind: str) -> Tuple[str, ...]:
    return BASES_EQUIVARIANT if kind == "equivariant" else BASES_ALL


def live_observable_indices(kind: str) -> np.ndarray:
    """Observables that actually carry information for this grade."""
    return np.arange(8) if kind == "equivariant" else np.arange(N_HARMONICS)


def live_observable_slice(kind: str) -> slice:
    return ODD_SLICE if kind == "equivariant" else slice(0, N_HARMONICS)


def pauli_matrix(symbol: str) -> np.ndarray:
    return {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }[symbol]


def pauli_operator(observable: str) -> np.ndarray:
    op = pauli_matrix(observable[0])
    for s in observable[1:]:
        op = np.kron(op, pauli_matrix(s))
    return op


PAULI_OPERATORS = np.stack([pauli_operator(o) for o in OBSERVABLES], axis=0)

_INDICES = np.arange(DIM)
_BITS = (_INDICES[:, None] >> (N_QUBITS - 1 - np.arange(N_QUBITS)[None, :])) & 1
_EIGENVALUES = 1.0 - 2.0 * _BITS


def _sign_table(rows: Tuple[int, ...]) -> np.ndarray:
    cols = []
    for r in rows:
        active = [q for q, s in enumerate(OBSERVABLES[r]) if s != "I"]
        cols.append(np.prod(_EIGENVALUES[:, active], axis=1))
    return np.stack(cols, axis=1)


BASIS_SIGN_TABLES = {b: _sign_table(rows) for b, rows in MEASUREMENT_BASES.items()}

# observable index -> its measurement basis
OBSERVABLE_BASIS = {i: b for b, rows in MEASUREMENT_BASES.items() for i in rows}


# ============================================================
# 2. Preconditioned decoder
# ============================================================

DEFAULT_DECAY_P = {"triangle": 2.0, "sawtooth": 1.0, "double_sawtooth": 1.0}


def preconditioner(decay_p: float) -> np.ndarray:
    """w_k = k^{-p}, in HARMONIC order (index k-1). p=0 -> all ones."""
    return HARMONICS.astype(float) ** (-float(decay_p))


def decode(expectations: np.ndarray, log_scales: np.ndarray,
           weights: np.ndarray) -> np.ndarray:
    """c_k = exp(s_{g(k)}) * w_k * mu_k, scattered into harmonic order."""
    raw = np.exp(log_scales)[FAMILY_INDEX] * expectations      # observable order
    coefficients = np.zeros(N_HARMONICS, dtype=float)
    coefficients[OBSERVABLE_HARMONIC - 1] = raw                # harmonic order
    return coefficients * weights


def validate_sector_design() -> None:
    assert sorted(OBSERVABLE_HARMONIC.tolist()) == list(range(1, N_HARMONICS + 1))
    assert len(OBSERVABLES) == N_HARMONICS
    assert len(set(OBSERVABLES)) == N_HARMONICS

    for i, obs in enumerate(OBSERVABLES):
        k = OBSERVABLE_HARMONIC[i]
        sym = obs[PARITY_QUBIT]
        if k % 2 == 1:
            assert sym in "IZ", (obs, k)
            assert i < 8, "odd-sector observables must occupy indices 0-7"
        else:
            assert sym in "XY", (obs, k)
            assert i >= 8, "even-sector observables must occupy indices 8-15"

    assert "IIIZ" not in OBSERVABLES   # identically +1 under equivariance

    grouped = [i for rows in MEASUREMENT_BASES.values() for i in rows]
    assert sorted(grouped) == list(range(N_HARMONICS))
    for basis, rows in MEASUREMENT_BASES.items():
        assert len(basis) == N_QUBITS
        for r in rows:
            for b_sym, p_sym in zip(basis, OBSERVABLES[r]):
                assert p_sym == "I" or p_sym == b_sym, (basis, OBSERVABLES[r])

    # Transport identity: Z3 O Z3 = (-1)^{k+1} O.
    z3 = pauli_operator("IIIZ")
    for i, obs in enumerate(OBSERVABLES):
        sign = 1.0 if OBSERVABLE_HARMONIC[i] % 2 == 1 else -1.0
        assert np.allclose(z3 @ PAULI_OPERATORS[i] @ z3, sign * PAULI_OPERATORS[i]), obs

    # Ordering guard: unit expectation on observable i lands on its harmonic.
    ones = np.ones(N_HARMONICS)
    zero_scales = np.zeros(len(FAMILY_NAMES))
    for i in range(N_HARMONICS):
        unit = np.zeros(N_HARMONICS)
        unit[i] = 1.0
        c = decode(unit, zero_scales, ones)
        assert np.isclose(c[OBSERVABLE_HARMONIC[i] - 1], 1.0), (i, OBSERVABLES[i])
        assert np.isclose(np.sum(np.abs(c)), 1.0), (i, OBSERVABLES[i])

    # Weight guard, after the ordering guard has passed for every observable:
    # decoder weights are applied exactly once, so under non-trivial weights the
    # unit expectation lands on its harmonic with coefficient w_k (with unit
    # weights a doubled weight would be invisible).
    for p in (1.0, 2.0):
        w = preconditioner(p)
        for i in range(N_HARMONICS):
            unit = np.zeros(N_HARMONICS)
            unit[i] = 1.0
            k = OBSERVABLE_HARMONIC[i] - 1
            c = decode(unit, zero_scales, w)
            assert np.isclose(c[k], w[k], rtol=1e-12, atol=0.0), (i, OBSERVABLES[i], p)
            assert np.isclose(np.sum(np.abs(c)), w[k], rtol=1e-12, atol=0.0), (i, OBSERVABLES[i], p)

    for p in (0.0, 1.0, 2.0):
        w = preconditioner(p)
        assert w.shape == (N_HARMONICS,) and np.all(w > 0.0)

    assert set(OBSERVABLE_HARMONIC[EVEN_SLICE].tolist()) == set(range(2, 17, 2))
    assert set(OBSERVABLE_HARMONIC[ODD_SLICE].tolist()) == set(range(1, 17, 2))

    covered = {i for b in BASES_EQUIVARIANT for i in MEASUREMENT_BASES[b]}
    assert set(range(8)).issubset(covered)


def exact_expectations(state: np.ndarray) -> np.ndarray:
    v = np.einsum("i,kij,j->k", state.conj(), PAULI_OPERATORS, state, optimize=True)
    if np.max(np.abs(v.imag)) > 1e-8:
        raise FloatingPointError("Non-real Pauli expectation.")
    return np.clip(v.real, -1.0, 1.0)


# ============================================================
# 3. Simulator and graded ansatz
# ============================================================

def ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def apply_1q(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    t = np.moveaxis(state.reshape([2] * N_QUBITS), qubit, 0)
    t = (gate @ t.reshape(2, -1)).reshape(t.shape)
    return np.moveaxis(t, 0, qubit).reshape(-1)


CPHASE_MASKS: Dict[Tuple[int, int], np.ndarray] = {}


def cphase_mask(q1: int, q2: int) -> np.ndarray:
    key = tuple(sorted((q1, q2)))
    if key not in CPHASE_MASKS:
        CPHASE_MASKS[key] = (_BITS[:, key[0]] == 1) & (_BITS[:, key[1]] == 1)
    return CPHASE_MASKS[key]


def apply_cphase(state: np.ndarray, q1: int, q2: int, phi: float) -> np.ndarray:
    out = state.copy()
    out[cphase_mask(q1, q2)] *= np.exp(1j * phi)
    return out


ENTANGLING_CHAIN = ((0, 1), (1, 2), (2, 3))


def n_circuit_params(n_layers: int) -> int:
    return n_layers * (N_QUBITS + len(ENTANGLING_CHAIN)) + (N_QUBITS + 1)


def effective_circuit_params(kind: str, n_layers: int) -> int:
    """Nominal count minus parameters made vacuous by the symmetry."""
    nominal = n_circuit_params(n_layers)
    if kind == "generic":
        return nominal
    if kind == "sector_mixed":
        return nominal - 2 * n_layers          # body RZ + body CPhase(2,3)
    return nominal - 2 * (n_layers + 1)        # equivariant: also final pair


def run_ansatz(params: np.ndarray, kind: str, n_layers: int) -> np.ndarray:
    if kind not in ANSATZ_KINDS:
        raise ValueError(f"Unknown ansatz kind: {kind}")
    if len(params) != n_circuit_params(n_layers):
        raise ValueError(f"{kind}: expected {n_circuit_params(n_layers)} params.")

    state = np.zeros(DIM, dtype=complex)
    state[0] = 1.0
    i = 0

    body_gate = ry if kind == "generic" else rz
    final_gate = rz if kind == "equivariant" else ry

    for _ in range(n_layers):
        for q in range(3):
            state = apply_1q(state, ry(float(params[i])), q)
            i += 1
        state = apply_1q(state, body_gate(float(params[i])), PARITY_QUBIT)
        i += 1
        for (q1, q2) in ENTANGLING_CHAIN:
            state = apply_cphase(state, q1, q2, float(params[i]))
            i += 1

    # Final readout block: the trailing CPhase(2,3) comes AFTER the parity
    # rotation, so a broken-symmetry qubit 3 can acquire relative phase.
    for q in range(3):
        state = apply_1q(state, ry(float(params[i])), q)
        i += 1
    state = apply_1q(state, final_gate(float(params[i])), PARITY_QUBIT)
    i += 1
    state = apply_cphase(state, 2, 3, float(params[i]))
    i += 1

    assert i == len(params)
    return state


def validate_equivariance(n_trials: int = 25, seed: int = 7) -> None:
    """Selection rule P1: equivariant even-sector expectations vanish."""
    rng = np.random.default_rng(seed)
    for L in (1, 2, 3):
        for _ in range(n_trials):
            p = rng.uniform(-np.pi, np.pi, size=n_circuit_params(L))
            state = run_ansatz(p, "equivariant", L)
            e = exact_expectations(state)
            assert np.max(np.abs(e[EVEN_SLICE])) < 1e-10, "Selection rule violated."
            assert np.allclose(np.abs(state.reshape([2] * N_QUBITS)[:, :, :, 1]), 0.0)


# ============================================================
# 4. Targets, grids, masks, analytic coefficients
# ============================================================

SINGULAR_POINTS = np.array([0.25, 0.5, 0.75])


def sawtooth(x: np.ndarray) -> np.ndarray:
    return x - np.round(x)


def triangle(x: np.ndarray) -> np.ndarray:
    return np.arcsin(np.clip(np.sin(2.0 * np.pi * x), -1.0, 1.0)) / (2.0 * np.pi)


def double_sawtooth(x: np.ndarray) -> np.ndarray:
    return 0.5 * sawtooth(2.0 * x)


TARGETS = {"triangle": triangle, "sawtooth": sawtooth, "double_sawtooth": double_sawtooth}
TARGET_SECTOR = {"triangle": "odd", "sawtooth": "both", "double_sawtooth": "even"}


def analytic_coefficients(target: str) -> np.ndarray:
    c = np.zeros(N_HARMONICS)
    k = HARMONICS.astype(float)
    if target == "sawtooth":
        c[:] = ((-1.0) ** (HARMONICS + 1)) / (np.pi * k)
    elif target == "triangle":
        odd = HARMONICS % 2 == 1
        m = (HARMONICS[odd] - 1) // 2
        c[odd] = (2.0 / np.pi**2) * ((-1.0) ** m) / (k[odd] ** 2)
    elif target == "double_sawtooth":
        even = HARMONICS % 2 == 0
        m = HARMONICS[even] // 2
        c[even] = ((-1.0) ** (m + 1)) / (2.0 * np.pi * m)
    else:
        raise ValueError(f"Unknown target: {target}")
    return c


def make_grid(n: int, offset: float = 0.0) -> np.ndarray:
    return (np.arange(n, dtype=float) + offset) / n


def common_mask(x: np.ndarray, delta: float) -> np.ndarray:
    d = np.min(np.abs(((x[:, None] - SINGULAR_POINTS[None, :]) + 0.5) % 1.0 - 0.5), axis=1)
    return d >= delta


def sine_design(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * np.outer(x, HARMONICS))


@dataclass(frozen=True)
class GridCache:
    design_masked: np.ndarray
    target_masked: np.ndarray


def build_cache(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> GridCache:
    return GridCache(design_masked=sine_design(x)[mask], target_masked=y[mask])


def masked_rmse_linf(c: np.ndarray, g: GridCache) -> Tuple[float, float]:
    e = g.design_masked @ c - g.target_masked
    return float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e)))


@dataclass(frozen=True)
class DatasetBundle:
    train: GridCache
    nested: GridCache
    shifted: GridCache
    included_fraction: float


def build_bundles(delta: float, n_train: int = 256,
                  n_eval: int = 1024) -> Dict[str, DatasetBundle]:
    bundles = {}
    x_train, x_nested = make_grid(n_train), make_grid(n_eval)
    x_shifted = make_grid(n_eval, offset=0.5)   # disjoint from training grid
    m_train, m_nested, m_shifted = (
        common_mask(g, delta) for g in (x_train, x_nested, x_shifted)
    )
    assert not np.intersect1d(x_train, x_shifted).size
    for name, f in TARGETS.items():
        bundles[name] = DatasetBundle(
            train=build_cache(x_train, f(x_train), m_train),
            nested=build_cache(x_nested, f(x_nested), m_nested),
            shifted=build_cache(x_shifted, f(x_shifted), m_shifted),
            included_fraction=float(np.mean(m_train)),
        )
    return bundles


# ============================================================
# 5. Classical ceilings and nulls
# ============================================================

SECTOR_COLUMNS = {
    "full": np.arange(N_HARMONICS),
    "odd": np.where(HARMONICS % 2 == 1)[0],
    "even": np.where(HARMONICS % 2 == 0)[0],
}

OPTIONS = {"ftol": 1e-12, "gtol": 1e-8, "maxls": 50, "maxfun": 200_000}


def ridge_fit(cache: GridCache, alpha: float, sector: str = "full") -> np.ndarray:
    cols = SECTOR_COLUMNS[sector]
    V = cache.design_masked[:, cols]
    m = len(cache.target_masked)
    lhs = V.T @ V / m + alpha * np.eye(len(cols))
    c = np.zeros(N_HARMONICS)
    c[cols] = np.linalg.solve(lhs, V.T @ cache.target_masked / m)
    return c


def regularized_loss(c: np.ndarray, cache: GridCache, alpha: float) -> float:
    r = cache.design_masked @ c - cache.target_masked
    return float(np.mean(r**2)) + alpha * float(np.dot(c, c))


def spherical_coefficients(params: np.ndarray, mode: str) -> np.ndarray:
    c = np.zeros(N_HARMONICS)
    if mode == "spherical_full":
        raw, s = params[:16], params[16]
        n = np.linalg.norm(raw)
        if n > 1e-12:
            c[:] = np.exp(s) * raw / n
    elif mode == "spherical_odd":
        raw, s = params[:8], params[8]
        n = np.linalg.norm(raw)
        if n > 1e-12:
            c[SECTOR_COLUMNS["odd"]] = np.exp(s) * raw / n
    elif mode == "spherical_sector":
        for sector, (a, b, si) in {"odd": (0, 8, 16), "even": (8, 16, 17)}.items():
            raw, s = params[a:b], params[si]
            n = np.linalg.norm(raw)
            if n > 1e-12:
                c[SECTOR_COLUMNS[sector]] = np.exp(s) * raw / n
    else:
        raise ValueError(mode)
    return c


SPHERICAL_N_PARAMS = {"spherical_full": 17, "spherical_odd": 9, "spherical_sector": 18}


def spherical_fit(mode: str, cache: GridCache, alpha: float, seed: int, maxiter: int):
    rng = np.random.default_rng(seed)
    n = SPHERICAL_N_PARAMS[mode]
    n_dir = n - (2 if mode == "spherical_sector" else 1)
    x0 = np.concatenate([rng.normal(size=n_dir), rng.normal(0.0, 0.25, size=n - n_dir)])
    bounds = [(-5.0, 5.0)] * n_dir + [(-8.0, 8.0)] * (n - n_dir)
    res = minimize(
        lambda p: regularized_loss(spherical_coefficients(p, mode), cache, alpha),
        x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, **OPTIONS},
    )
    return spherical_coefficients(res.x, mode), res


# ============================================================
# 6. Model training
# ============================================================

def n_decoder_scales() -> int:
    return len(FAMILY_NAMES)


def n_total_params(n_layers: int) -> int:
    return n_circuit_params(n_layers) + n_decoder_scales()


def model_bounds(n_layers: int):
    return [(None, None)] * n_circuit_params(n_layers) + [(-8.0, 8.0)] * n_decoder_scales()


def random_params(n_layers: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.uniform(-np.pi, np.pi, size=n_circuit_params(n_layers)),
        rng.normal(-1.0, 0.25, size=n_decoder_scales()),
    ])


def analytic_log_scales(reference: np.ndarray, weights: np.ndarray,
                        typical_mu: float = 0.5) -> np.ndarray:
    scales = np.empty(n_decoder_scales())
    for g in range(n_decoder_scales()):
        rows = np.where(FAMILY_INDEX == g)[0]
        harmonics = OBSERVABLE_HARMONIC[rows] - 1
        needed = reference[harmonics] / weights[harmonics]
        rms = float(np.sqrt(np.mean(needed ** 2)))
        scales[g] = np.log(max(rms / typical_mu, 1e-6))
    return np.clip(scales, -8.0, 8.0)


def partial_product_warm_start_params(n_layers: int, seed: int,
                                      reference: np.ndarray,
                                      weights: np.ndarray) -> np.ndarray:
    """Heuristic product-state initialization, not a full inverse decoder.

    With all preceding circuit parameters zero, the final RY rotations on
    qubits 0, 1 and 2 satisfy

        <ZIII> = cos(theta_0),
        <IZII> = cos(theta_1),
        <IIZI> = cos(theta_2).

    The initializer matches only those three odd-Z moments. It does not invert
    the full correlated Pauli decoder and should not be interpreted as an
    analytic initialization of the complete ridge coefficient vector.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros(n_circuit_params(n_layers))
    log_scales = analytic_log_scales(reference, weights)

    rows = np.arange(3)
    harmonics = OBSERVABLE_HARMONIC[rows] - 1
    mu_target = reference[harmonics] / (
        np.exp(log_scales[FAMILY_INDEX[rows]]) * weights[harmonics]
    )
    theta = np.arccos(np.clip(mu_target, -0.999, 0.999))

    final_block_offset = n_layers * (N_QUBITS + len(ENTANGLING_CHAIN))
    p[final_block_offset:final_block_offset + 3] = theta
    p += rng.normal(0.0, 1e-3, size=p.shape)
    return np.concatenate([p, log_scales])


STOPPING_SCALE_LIMIT = 1e-5  # effective threshold / data-fit term


def stopping_scale(cache: GridCache, alpha: float, options: dict | None = None,
                   sector: str = "odd") -> dict:
    """Effective L-BFGS-B stopping threshold against the scale of the objective.

    L-BFGS-B stops when (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol, so
    for an objective below one the threshold is absolute. The threshold is
    compared with the objective and its data-fit term at the sector ridge
    optimum.
    """
    options = OPTIONS if options is None else options
    c = ridge_fit(cache, alpha, sector)
    fit = float(np.mean((cache.design_masked @ c - cache.target_masked) ** 2))
    f_ref = regularized_loss(c, cache, alpha)
    threshold = options["ftol"] * max(1.0, abs(f_ref))
    return {"f_ref": f_ref, "fit_term": fit, "threshold": threshold,
            "threshold_over_objective": threshold / f_ref,
            "threshold_over_fit": threshold / fit}


def validate_stopping_scale(cache: GridCache, alpha: float, options: dict | None = None,
                            sector: str = "odd", limit: float = STOPPING_SCALE_LIMIT) -> None:
    """The stopping threshold must be far below the data-fit term at the optimum."""
    r = stopping_scale(cache, alpha, options, sector)
    assert r["threshold_over_fit"] <= limit, ("stopping threshold too coarse", r)


MODEL_SECTOR = {"equivariant": "odd", "generic": "full", "sector_mixed": "full"}


def mask_shift_checks(bundles: Mapping[str, "DatasetBundle"]) -> pd.DataFrame:
    """Is the excluded set invariant under the half-period shift x -> x + 1/2?

    The grading is a symmetry of the fitting problem only if the mask is. If it
    is not, an anti-periodic target can still be fitted better outside the odd
    sector, and the price of the prior is nonzero; the detail reports that
    price on the training grid, as in the decomposition of the gap.
    """
    shifted = np.sort((SINGULAR_POINTS + 0.5) % 1.0)
    invariant = np.allclose(shifted, np.sort(SINGULAR_POINTS))
    prices = []
    for target, bundle in bundles.items():
        odd, _ = masked_rmse_linf(ridge_fit(bundle.train, 0.0, "odd"), bundle.train)
        full, _ = masked_rmse_linf(ridge_fit(bundle.train, 0.0, "full"), bundle.train)
        prices.append(f"{target}: odd {odd:.3e}, full {full:.3e}, price {odd - full:.3e}")
    return pd.DataFrame([{
        "prediction": "P8_mask_half_period_invariance",
        "status": "PASS" if invariant else "FAIL",
        "detail": (f"excluded points {np.sort(SINGULAR_POINTS).tolist()} shift to "
                   f"{shifted.tolist()}; training-grid least-squares RMSE by sector: "
                   + "; ".join(prices)),
    }])


def stopping_scale_checks(bundles: Mapping[str, "DatasetBundle"], alpha: float,
                          options: dict | None = None) -> pd.DataFrame:
    """Record the stopping-threshold check as PASS or FAIL for every target and
    every coefficient sector a model family occupies: odd for the equivariant
    family, full for the generic and sector-mixed families."""
    rows = []
    sectors = {}
    for kind, sector in MODEL_SECTOR.items():
        sectors.setdefault(sector, []).append(kind)
    for target, bundle in bundles.items():
        for sector, kinds in sectors.items():
            r = stopping_scale(bundle.train, alpha, options, sector)
            rows.append({
                "prediction": f"P7_stopping_threshold_scale_{target}_{sector}",
                "status": "PASS" if r["threshold_over_fit"] <= STOPPING_SCALE_LIMIT else "FAIL",
                "detail": (f"models: {', '.join(kinds)}; effective ftol threshold "
                           f"{r['threshold']:.2e} is {r['threshold_over_fit']:.2e} of the "
                           f"data-fit term and {r['threshold_over_objective']:.2e} of the "
                           f"objective at the {sector}-sector ridge optimum "
                           f"(limit {STOPPING_SCALE_LIMIT:g})"),
            })
    return pd.DataFrame(rows)


def validate_partial_warm_start_mapping() -> None:
    """Guard the intended three-moment product-state mapping."""
    theta = np.array([0.4, 1.1, 2.0])
    for kind in ANSATZ_KINDS:
        p = np.zeros(n_circuit_params(1))
        offset = N_QUBITS + len(ENTANGLING_CHAIN)
        p[offset:offset + 3] = theta
        raw = exact_expectations(run_ansatz(p, kind, 1))
        if not np.allclose(raw[:3], np.cos(theta), atol=1e-12):
            raise RuntimeError(f"Warm-start moment mapping failed for {kind}.")


def evaluate_model(params: np.ndarray, kind: str, n_layers: int,
                   weights: np.ndarray):
    """Return raw and decoder expectations separately.

    Raw expectations are retained for symmetry verification. Decoder
    expectations apply the exact equivariant projection before decoding.
    """
    nc = n_circuit_params(n_layers)
    state = run_ansatz(np.asarray(params[:nc], float), kind, n_layers)
    raw = exact_expectations(state)
    decoder_e = raw.copy()
    if kind == "equivariant":
        decoder_e[EVEN_SLICE] = 0.0
    log_scales = np.asarray(params[nc:], float)
    coefficients = decode(decoder_e, log_scales, weights)
    return state, raw, decoder_e, coefficients, log_scales


@dataclass
class TrainedModel:
    target: str; kind: str; n_layers: int; seed: int; init: str
    params: np.ndarray; state: np.ndarray
    raw_expectations: np.ndarray
    expectations: np.ndarray
    coefficients: np.ndarray
    log_scales: np.ndarray
    success: bool; message: str; fun: float; nfev: int; nit: float; seconds: float


def train_model(target: str, kind: str, n_layers: int, seed: int,
                cache: GridCache, alpha: float, maxiter: int,
                weights: np.ndarray, init: str = "random",
                reference: np.ndarray | None = None) -> TrainedModel:
    t0 = time.perf_counter()

    if init not in {"random", "warm"}:
        raise ValueError(f"Unknown initialization: {init}")
    if init == "warm":
        if reference is None:
            raise ValueError("warm start requires a reference solution")
        x0 = partial_product_warm_start_params(
            n_layers, seed, reference, weights
        )
    else:
        x0 = random_params(n_layers, seed)

    def objective(p):
        _, _, _, c, _ = evaluate_model(p, kind, n_layers, weights)
        return regularized_loss(c, cache, alpha)

    res = minimize(objective, x0, method="L-BFGS-B",
                   bounds=model_bounds(n_layers),
                   options={"maxiter": maxiter, **OPTIONS})
    state, raw_e, decoder_e, c, s = evaluate_model(
        res.x, kind, n_layers, weights
    )
    return TrainedModel(
        target=target, kind=kind, n_layers=n_layers, seed=int(seed), init=init,
        params=np.asarray(res.x, float), state=state,
        raw_expectations=raw_e, expectations=decoder_e,
        coefficients=c, log_scales=s,
        success=bool(res.success), message=str(res.message), fun=float(res.fun),
        nfev=int(getattr(res, "nfev", -1)), nit=float(getattr(res, "nit", np.nan)),
        seconds=float(time.perf_counter() - t0),
    )


def boundary_margin(model: TrainedModel) -> float:
    """
    min_k (1 - |mu_k|) over live observables.

    Used to test whether the precision plateau is limited by the expectation
    bounds. At p=0 the triangle's largest |mu| is 0.83, and the margin falls
    where the error improves, so it is not.
    """
    live = model.expectations[live_observable_slice(model.kind)]
    return float(np.min(1.0 - np.abs(live)))


# ============================================================
# 7. Budget-fair grouped shot sampling, and the variance model
# ============================================================

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
S_DAG = np.array([[1, 0], [0, -1j]], dtype=complex)


def rotate_to_basis(state: np.ndarray, basis: str) -> np.ndarray:
    out = state
    for q, sym in enumerate(basis):
        if sym == "X":
            out = apply_1q(out, H, q)
        elif sym == "Y":
            out = apply_1q(out, S_DAG, q)
            out = apply_1q(out, H, q)
    return out


def prepare_cumulatives(state: np.ndarray, bases: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    cum = {}
    for b in bases:
        p = np.clip(np.abs(rotate_to_basis(state, b)) ** 2, 0.0, None)
        p /= p.sum()
        c = np.cumsum(p)
        c[-1] = 1.0
        cum[b] = c
    return cum


def _u32(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "little")


def sampled_expectations(cumulatives: Mapping[str, np.ndarray], kind: str, *,
                         target: str, n_layers: int, train_seed: int,
                         total_budget: int, repetition: int,
                         master_seed: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return raw measured estimates and symmetry-projected decoder estimates.

    For the equivariant arm, the XXXX basis provides the even-X observables at
    no additional state-preparation cost. They remain in ``raw_est`` as a
    hardware symmetry diagnostic, while ``decoder_est`` sets the full even
    sector to its exact model value zero.
    """
    bases = bases_for(kind)
    per_basis = total_budget // len(bases)
    if per_basis < 1:
        raise ValueError(
            f"Budget {total_budget} gives zero shots per basis for {kind}."
        )

    raw_est = np.full(N_HARMONICS, np.nan)
    for b in bases:
        # Keyed without the ansatz kind, so compared arms share random numbers.
        u = np.random.default_rng(np.random.SeedSequence(
            [master_seed, _u32(target), n_layers, train_seed,
             total_budget, repetition, _u32(b)]
        )).random(per_basis)
        outcomes = np.searchsorted(cumulatives[b], u, side="right")
        counts = np.bincount(outcomes, minlength=DIM).astype(float)
        rows = list(MEASUREMENT_BASES[b])
        raw_est[rows] = counts @ BASIS_SIGN_TABLES[b] / per_basis

    decoder_est = np.zeros(N_HARMONICS)
    live = live_observable_indices(kind)
    decoder_est[live] = raw_est[live]
    return raw_est, decoder_est, per_basis, per_basis * len(bases)


def predicted_coefficient_noise(expectations: np.ndarray, log_scales: np.ndarray,
                                weights: np.ndarray, kind: str,
                                total_budget: int) -> float:
    """
    sqrt(E||dc||^2) from the independent-binomial variance model,

        E||dc||^2 = sum_{k live} exp(2 s_{g(k)}) w_k^2 (1 - mu_k^2) / N_{b(k)}

    with N_b = total_budget / (#bases used). This supersedes the naive
    observable-counting prediction, which assumed W = I and equal scales.
    """
    bases = bases_for(kind)
    per_basis = max(total_budget // len(bases), 1)
    var = 0.0
    for i in live_observable_indices(kind):
        k = OBSERVABLE_HARMONIC[i] - 1
        amp = np.exp(log_scales[FAMILY_INDEX[i]]) * weights[k]
        var += (amp ** 2) * (1.0 - expectations[i] ** 2) / per_basis
    return float(np.sqrt(var))


def naive_counting_variance_ratio() -> float:
    """(#live_generic * #bases_generic) / (#live_equiv * #bases_equiv) = 3."""
    g = len(live_observable_indices("generic")) * len(bases_for("generic"))
    e = len(live_observable_indices("equivariant")) * len(bases_for("equivariant"))
    return g / e


# ============================================================
# 8. Geometry layer: Jacobian rank of the coefficient map (P3)
# ============================================================

def _raw_coefficients(params: np.ndarray, kind: str, n_layers: int,
                      weights: np.ndarray) -> np.ndarray:
    """Decode the raw expectations, without the equivariant projection.

    evaluate_model zeroes the even sector before decoding for the equivariant
    arm. Differentiating that output would force rank <= 8 by construction, so
    the rank bound P3a could never fail. Differentiating the raw decode instead
    tests the circuit. For a correct equivariant circuit the raw even
    expectations vanish (P1), so the measured ranks are unchanged; for a
    symmetry-breaking bug they do not, and the bound is then genuinely tested.
    """
    _, raw, _, _, log_scales = evaluate_model(params, kind, n_layers, weights)
    return decode(raw, log_scales, weights)


def jacobian_rank_study(layer_values, weights: np.ndarray, n_points: int = 25,
                        eps: float = 1e-6, rank_rtol: float = 1e-8,
                        seed: int = 11) -> pd.DataFrame:
    """Study numerical rank and the singular-value gap at the sector bound."""
    rng = np.random.default_rng(seed)
    rows = []
    for L in layer_values:
        for kind in ANSATZ_KINDS:
            ranks = []
            ranks_1e6 = []
            rel_bound = []
            rel_next = []
            bound_gaps = []
            bound = 8 if kind == "equivariant" else 16

            for _ in range(n_points):
                p = np.concatenate([
                    rng.uniform(-np.pi, np.pi, n_circuit_params(L)),
                    rng.normal(-1.0, 0.25, n_decoder_scales()),
                ])
                J = np.empty((N_HARMONICS, len(p)))
                for i in range(len(p)):
                    dp = np.zeros_like(p)
                    dp[i] = eps
                    c_plus = _raw_coefficients(p + dp, kind, L, weights)
                    c_minus = _raw_coefficients(p - dp, kind, L, weights)
                    J[:, i] = (c_plus - c_minus) / (2 * eps)

                sv = np.linalg.svd(J, compute_uv=False)
                scale = max(float(sv[0]), 1e-300)
                ranks.append(int(np.sum(sv > rank_rtol * scale)))
                ranks_1e6.append(int(np.sum(sv > 1e-6 * scale)))

                sigma_bound = float(sv[bound - 1])
                sigma_next = float(sv[bound]) if bound < len(sv) else 0.0
                rel_bound.append(sigma_bound / scale)
                rel_next.append(sigma_next / scale)
                bound_gaps.append(
                    sigma_bound / max(sigma_next, np.finfo(float).tiny)
                )

            rows.append({
                "n_layers": L,
                "kind": kind,
                "nominal_params": n_total_params(L),
                "effective_circuit_params": effective_circuit_params(kind, L),
                "sector_bound": bound,
                "rank_rtol": rank_rtol,
                "median_rank": float(np.median(ranks)),
                "max_rank": int(np.max(ranks)),
                "min_rank": int(np.min(ranks)),
                "median_rank_rtol_1e6": float(np.median(ranks_1e6)),
                "min_relative_sigma_bound": float(np.min(rel_bound)),
                "median_relative_sigma_bound": float(np.median(rel_bound)),
                "median_relative_sigma_next": float(np.median(rel_next)),
                "median_bound_singular_gap": float(np.median(bound_gaps)),
            })
    return pd.DataFrame(rows)


# ============================================================
# 9. Comparative metrics and paired inference
# ============================================================

def add_comparative_columns(frame: pd.DataFrame, ref: pd.DataFrame,
                            tolerance: float = 1e-4) -> pd.DataFrame:
    frame = frame.copy()
    for grid in ("train", "nested", "shifted"):
        col = f"{grid}_rmse"
        if col not in frame.columns:
            continue
        zero = ref.loc[ref["method"] == "zero"].groupby("target")[col].first()
        rref = ref.loc[ref["method"] == "ridge_full"].groupby("target")[col].first()
        z, r = frame["target"].map(zero), frame["target"].map(rref)
        den = z - r
        frame[f"{grid}_recovery"] = np.where(
            np.abs(den) > 1e-12, (z - frame[col]) / den, np.nan)
        frame[f"{grid}_success"] = frame[col] <= r + tolerance
    return frame


def holm_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    out, run = np.empty(m), 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        out[idx] = min(1.0, run)
    return out


CONTRASTS = (
    ("equivariant", "generic"),
    ("sector_mixed", "generic"),
    ("equivariant", "sector_mixed"),
    ("equivariant", "spherical_odd"),
    ("generic", "spherical_full"),
    ("sector_mixed", "spherical_sector"),
)


def paired_tests(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired contrasts on the random-init arm only."""
    rows = []
    seeded = frame[frame["seed"].notna() & (frame["init"] == "random")]
    for (target, L), grp in seeded.groupby(["target", "n_layers"]):
        pivot = grp.pivot_table(index="seed", columns="method", values=metric)
        for a, b in CONTRASTS:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            d = (pivot[a] - pivot[b]).dropna().to_numpy()
            if len(d) == 0:
                continue
            if np.allclose(d, 0.0):
                stat, p = 0.0, 1.0
            else:
                t = wilcoxon(d, zero_method="pratt", alternative="two-sided")
                stat, p = float(t.statistic), float(t.pvalue)
            rows.append({"target": target, "n_layers": int(L),
                         "contrast": f"{a}_minus_{b}", "metric": metric,
                         "n_pairs": len(d), "n_zero": int(np.sum(np.isclose(d, 0.0))),
                         "median_difference": float(np.median(d)),
                         "wilcoxon_statistic": stat, "p_value": p})
    return pd.DataFrame(rows)


# ============================================================
# 10. Automated prediction checks
# ============================================================

def _degradation_exponent(sub: pd.DataFrame, column: str) -> float:
    """Log-log slope of mean degradation vs budget. NaN if unfittable."""
    g = sub.groupby("total_budget")[column].mean()
    if len(g) < 2 or (g <= 0).any():
        return np.nan
    A = np.vstack([np.log(g.index.to_numpy(float)), np.ones(len(g))]).T
    slope, _ = np.linalg.lstsq(A, np.log(g.to_numpy()), rcond=None)[0]
    return float(slope)


# P2 tolerances were chosen after measuring the repeated-computation noise
# floor. The absolute floors are deliberately below rtol*|ceiling| for the
# smallest observed triangle ceilings, so the checks remain relative there.
P2_RMSE_RTOL = 1e-9
P2_RMSE_ATOL = 1e-16
P2_OBJECTIVE_RTOL = 1e-9
P2_OBJECTIVE_ATOL = 1e-20
P2_ODD_METHODS = ("equivariant", "spherical_odd")


def lower_bound_tolerance(
    ceiling: float,
    *,
    rtol: float,
    atol: float,
) -> float:
    """Scale-aware tolerance for a theoretical lower-bound check."""
    return max(float(atol), float(rtol) * abs(float(ceiling)))


def p2_lower_bound_check(
    exact: pd.DataFrame,
    *,
    prediction: str,
    ceiling_method: str,
    column: str,
    tested_methods: Tuple[str, ...] = P2_ODD_METHODS,
    rtol: float,
    atol: float,
) -> dict:
    """Check that odd-sector methods do not beat the exact odd-sector minimum.

    Signed slack is ``model_value - ceiling``:

      * positive slack: safely above the lower bound;
      * negative slack: apparent violation;
      * a negative slack with magnitude <= tolerance is retained in the
        diagnostic text but is not counted as a violation.

    Classical ceiling rows are repeated across layer depths. Their spread is
    checked explicitly so an ordering or construction bug cannot be hidden by
    simply selecting the first row.
    """
    required = {"target", "method", "init", column}
    missing = required - set(exact.columns)
    if missing:
        return {
            "prediction": prediction,
            "status": "ERROR_NO_DATA",
            "detail": f"missing columns: {sorted(missing)}",
        }

    base = exact[exact["init"] != "warm"]
    violations = 0
    checked = 0
    ceiling_inconsistencies = 0
    max_ceiling_spread = 0.0
    min_signed_slack = np.inf
    min_normalized_slack = np.inf
    max_violation = 0.0
    closest = None
    checked_by_method = {method: 0 for method in tested_methods}

    for target, group in base.groupby("target"):
        ceiling_rows = group.loc[
            group["method"] == ceiling_method, column
        ].dropna()
        if ceiling_rows.empty:
            continue

        ceiling_values = ceiling_rows.to_numpy(dtype=float)
        ceiling = float(np.median(ceiling_values))
        tolerance = lower_bound_tolerance(
            ceiling, rtol=rtol, atol=atol
        )
        ceiling_spread = float(np.ptp(ceiling_values))
        max_ceiling_spread = max(max_ceiling_spread, ceiling_spread)
        if ceiling_spread > tolerance:
            ceiling_inconsistencies += 1

        for method in tested_methods:
            method_rows = group.loc[
                group["method"] == method
            ].dropna(subset=[column])
            if method_rows.empty:
                continue

            values = method_rows[column].to_numpy(dtype=float)
            signed_slack = values - ceiling
            normalized_slack = signed_slack / tolerance
            violation_mask = signed_slack < -tolerance

            checked += len(values)
            checked_by_method[method] += len(values)
            violations += int(np.sum(violation_mask))
            min_signed_slack = min(
                min_signed_slack, float(np.min(signed_slack))
            )
            min_normalized_slack = min(
                min_normalized_slack, float(np.min(normalized_slack))
            )
            max_violation = max(
                max_violation,
                float(np.max(np.maximum(-signed_slack, 0.0))),
            )

            local_index = int(np.argmin(signed_slack))
            local_slack = float(signed_slack[local_index])
            if closest is None or local_slack < closest["slack"]:
                row = method_rows.iloc[local_index]
                closest = {
                    "target": target,
                    "method": method,
                    "n_layers": row.get("n_layers", np.nan),
                    "seed": row.get("seed", np.nan),
                    "slack": local_slack,
                    "tolerance": tolerance,
                }

    if checked == 0:
        return {
            "prediction": prediction,
            "status": "ERROR_NO_DATA",
            "detail": (
                f"no tested rows for methods={tested_methods}; "
                f"ceiling={ceiling_method}, column={column}"
            ),
        }

    status = "FAIL" if (violations or ceiling_inconsistencies) else "PASS"
    method_counts = ", ".join(
        f"{method}={count}" for method, count in checked_by_method.items()
    )
    closest_text = "n/a"
    if closest is not None:
        layer = closest["n_layers"]
        seed = closest["seed"]
        layer_text = "n/a" if pd.isna(layer) else str(int(layer))
        seed_text = "n/a" if pd.isna(seed) else str(int(seed))
        closest_text = (
            f"{closest['target']}/{closest['method']}/L={layer_text}/"
            f"seed={seed_text}, slack={closest['slack']:.3e}, "
            f"tol={closest['tolerance']:.3e}"
        )

    return {
        "prediction": prediction,
        "status": status,
        "detail": (
            f"{violations} bound violations over {checked} rows "
            f"({method_counts}); "
            f"ceiling inconsistencies={ceiling_inconsistencies}; "
            f"minimum signed slack={min_signed_slack:.3e}; "
            f"minimum slack/tolerance={min_normalized_slack:.3e}; "
            f"maximum violation={max_violation:.3e}; "
            f"maximum repeated-ceiling spread={max_ceiling_spread:.3e}; "
            f"closest row=({closest_text}); "
            f"rtol={rtol:.1e}, atol={atol:.1e}"
        ),
    }


def check_predictions(
    exact: pd.DataFrame,
    ranks: pd.DataFrame,
    shots: pd.DataFrame | None = None,
    decay_note: str = "",
) -> pd.DataFrame:
    """Check the structural predictions; a failure indicates an implementation error."""
    rows = []
    # Exclude only the warm arm. Classical rows carry init="n/a" and are the
    # ceilings that P2 and P5 are checked against.
    base = exact[exact["init"] != "warm"]

    # P1: use the unprojected trained-model expectations.
    eq_all = exact[exact["method"] == "equivariant"]
    if eq_all.empty or "max_raw_even_expectation" not in eq_all.columns:
        rows.append({
            "prediction": "P1_equivariant_raw_even_zero",
            "status": "ERROR_NO_DATA",
            "detail": "no raw equivariant expectations available",
        })
    else:
        raw_even = eq_all["max_raw_even_expectation"].dropna()
        if raw_even.empty:
            rows.append({
                "prediction": "P1_equivariant_raw_even_zero",
                "status": "ERROR_NO_DATA",
                "detail": "raw equivariant expectation column is empty",
            })
        else:
            worst = float(raw_even.max())
            rows.append({
                "prediction": "P1_equivariant_raw_even_zero",
                "status": "PASS" if worst < 1e-10 else "FAIL",
                "detail": (
                    f"max raw even expectation={worst:.3e} "
                    f"(all inits, n={len(raw_even)})"
                ),
            })

    # P2a: odd-sector RMSE cannot beat the odd-sector OLS minimum.
    rows.append(p2_lower_bound_check(
        exact,
        prediction="P2a_odd_methods_above_odd_ols_rmse",
        ceiling_method="ols_odd",
        column="train_rmse",
        tested_methods=P2_ODD_METHODS,
        rtol=P2_RMSE_RTOL,
        atol=P2_RMSE_ATOL,
    ))

    # P2b: odd-sector regularized objective cannot beat odd ridge.
    rows.append(p2_lower_bound_check(
        exact,
        prediction="P2b_odd_methods_above_odd_ridge_objective",
        ceiling_method="ridge_odd",
        column="train_objective",
        tested_methods=P2_ODD_METHODS,
        rtol=P2_OBJECTIVE_RTOL,
        atol=P2_OBJECTIVE_ATOL,
    ))

    # P3.
    over = ranks[ranks["max_rank"] > ranks["sector_bound"]]
    rows.append({
        "prediction": "P3a_rank_within_sector_bound",
        "status": "PASS" if over.empty else "FAIL",
        "detail": f"{len(over)} cells exceed bound",
    })
    eq_rank = ranks[ranks["kind"] == "equivariant"]
    if eq_rank.empty:
        rows.append({
            "prediction": "P3b_equivariant_rank_exactly_8",
            "status": "ERROR_NO_DATA",
            "detail": "no equivariant rank cells",
        })
    else:
        exact_eight = bool(
            (eq_rank["min_rank"] == 8).all()
            and (eq_rank["max_rank"] == 8).all()
        )
        min_rel = float(eq_rank["min_relative_sigma_bound"].min())
        med_gap = float(eq_rank["median_bound_singular_gap"].median())
        rows.append({
            "prediction": "P3b_equivariant_rank_exactly_8",
            "status": "PASS" if exact_eight else "FAIL",
            "detail": (
                f"rank ranges "
                f"{list(zip(eq_rank['min_rank'], eq_rank['max_rank']))}; "
                f"min sigma8/sigma1={min_rel:.3e}, "
                f"median sigma8/sigma9 gap={med_gap:.3e}"
            ),
        })

    # P5.
    def rmse(target: str, method: str) -> float:
        values = base[
            (base["target"] == target) & (base["method"] == method)
        ]["nested_rmse"]
        return float(values.iloc[0]) if len(values) else np.nan

    vals = {
        (target, method): rmse(target, method)
        for target in ("triangle", "double_sawtooth")
        for method in ("ridge_odd", "ridge_full", "zero")
    }
    if any(np.isnan(value) for value in vals.values()):
        rows.append({
            "prediction": "P5_sector_dose_response",
            "status": "ERROR_NO_DATA",
            "detail": "classical ceilings missing from the frame",
        })
    else:
        tri_free = vals[("triangle", "ridge_odd")] <= (
            vals[("triangle", "ridge_full")] * 1.5 + 1e-9
        )
        ds_total = vals[("double_sawtooth", "ridge_odd")] >= (
            vals[("double_sawtooth", "zero")] * 0.9
        )
        rows.append({
            "prediction": "P5_sector_dose_response",
            "status": "PASS" if (tri_free and ds_total) else "FAIL",
            "detail": (
                f"triangle odd-prior free={tri_free}, "
                f"double_sawtooth odd-prior total={ds_total}"
            ),
        })

    # P4: measured vs the per-model variance model, not a fixed constant.
    if shots is not None and len(shots):
        s = shots[shots["init"] == "random"].copy()
        s = s[s["predicted_coefficient_noise"] > 0]
        if len(s):
            cell_keys = [
                "target", "n_layers", "method", "init", "seed",
                "total_budget",
            ]
            cells = (
                s.groupby(cell_keys, as_index=False)
                .agg(
                    measured_noise_sq=(
                        "coefficient_noise_l2",
                        lambda x: float(np.mean(np.square(x))),
                    ),
                    predicted_noise=(
                        "predicted_coefficient_noise", "first"
                    ),
                )
            )
            cells["measured_rms_noise"] = np.sqrt(
                cells["measured_noise_sq"]
            )
            cells["rms_ratio"] = (
                cells["measured_rms_noise"] / cells["predicted_noise"]
            )
            med = float(cells["rms_ratio"].median())
            q10, q90 = cells["rms_ratio"].quantile([0.1, 0.9]).to_numpy()
            rows.append({
                "prediction": "P4a_rms_vs_variance_model",
                "status": "PASS" if 0.85 <= med <= 1.15 else "CHECK",
                "detail": (
                    f"median cellwise RMS/predicted ||dc||={med:.3f}; "
                    f"10-90% [{q10:.3f}, {q90:.3f}] over {len(cells)} "
                    f"cells (predicted 1.000)"
                ),
            })

        # Naive counting ratio: valid only at W=I with comparable scales.
        naive = []
        for _, cell in s.groupby(["target", "n_layers"]):
            pivot = cell.groupby("method")["coefficient_noise_l2"].median()
            if (
                {"generic", "equivariant"} <= set(pivot.index)
                and pivot["equivariant"] > 0
            ):
                naive.append(pivot["generic"] / pivot["equivariant"])
        if naive:
            med = float(np.median(naive))
            expected = np.sqrt(naive_counting_variance_ratio())
            rows.append({
                "prediction": "P4a_naive_counting_ratio",
                "status": "INFO",
                "detail": (
                    f"median generic/equivariant ||dc||={med:.2f} "
                    f"({expected:.2f} expected ONLY at W=I with "
                    f"comparable scales)"
                ),
            })

        # P4b: mean MSE degradation isolates the quadratic contribution.
        deg = []
        for _, cell in s.groupby(["target", "n_layers"]):
            pivot = cell.groupby("method")["nested_mse_degradation"].mean()
            if (
                {"generic", "equivariant"} <= set(pivot.index)
                and pivot["equivariant"] > 0
            ):
                deg.append(pivot["generic"] / pivot["equivariant"])
        if deg:
            rows.append({
                "prediction": "P4b_mean_mse_degradation_ratio",
                "status": "INFO",
                "detail": (
                    f"median of cellwise mean-degradation ratios="
                    f"{np.median(deg):.2f} over {len(deg)} cells "
                    f"(regime-dependent; see P4c)"
                ),
            })

        # P4c: -1 indicates the second-order regime; -0.5 first-order noise.
        exps = []
        for _, cell in s.groupby(["target", "n_layers"]):
            for kind in ("generic", "equivariant", "sector_mixed"):
                exponent = _degradation_exponent(
                    cell[cell["method"] == kind],
                    "nested_mse_degradation",
                )
                if np.isfinite(exponent):
                    exps.append(exponent)
        rows.append({
            "prediction": "P4c_mse_degradation_exponent",
            "status": "INFO",
            "detail": (
                f"median log-log slope of mean MSE degradation="
                f"{np.median(exps):.2f} over {len(exps)} fits "
                f"(-1 second order, -0.5 first order dominates)"
                if exps else "no fittable cells"
            ),
        })

    if decay_note:
        rows.append({
            "prediction": "P6_preconditioning",
            "status": "INFO",
            "detail": decay_note,
        })
    return pd.DataFrame(rows)


def recompute_prediction_checks(
    result: Mapping[str, pd.DataFrame],
    *,
    alpha: float,
    delta: float,
    decay_note: str = "",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Recompute checks from existing frames without retraining or resampling.

    alpha and delta are those of the run, and are needed for the
    stopping-threshold check.
    """
    required = {"exact", "ranks", "shot"}
    missing = required - set(result)
    if missing:
        raise KeyError(f"result is missing frames: {sorted(missing)}")
    checks = check_predictions(
        result["exact"], result["ranks"], result["shot"],
        decay_note=decay_note,
    )
    bundles = build_bundles(delta)
    checks = pd.concat([checks, stopping_scale_checks(bundles, alpha),
                        mask_shift_checks(bundles)], ignore_index=True)
    if output_path is not None:
        safe_write_csv(checks, output_path)
    return checks

def compare_preconditioning_frames(
        a: pd.DataFrame, b: pd.DataFrame,
        label_a: str = "p0", label_b: str = "matched",
        methods: Tuple[str, ...] = ANSATZ_KINDS,
        ceilings: Mapping[str, str] | None = None,
        verbose: bool = True) -> pd.DataFrame:
    """Compare in-memory summary frames, avoiding stale fallback files."""
    if ceilings is None:
        ceilings = {"equivariant": "ridge_odd", "generic": "ridge_full",
                    "sector_mixed": "ridge_full"}

    a = a[a["init"].isin(["random", "n/a"])].copy()
    b = b[b["init"].isin(["random", "n/a"])].copy()
    rows, skipped = [], []

    def pick(frame, target, L, method, column):
        sel = frame[(frame["target"] == target) & (frame["n_layers"] == L)
                    & (frame["method"] == method)]
        if sel.empty or column not in sel.columns:
            return None
        value = sel[column].iloc[0]
        return None if pd.isna(value) else float(value)

    keys_a = set(zip(a["target"], a["n_layers"].astype(int)))
    keys_b = set(zip(b["target"], b["n_layers"].astype(int)))
    keys = sorted(keys_a | keys_b)

    for target, L in keys:
        for method in methods:
            ceil_name = ceilings.get(method, "ridge_full")
            va = pick(a, target, L, method, "median_nested_rmse")
            vb = pick(b, target, L, method, "median_nested_rmse")
            ca = pick(a, target, L, ceil_name, "median_nested_rmse")
            cb = pick(b, target, L, ceil_name, "median_nested_rmse")
            if None in (va, vb, ca, cb):
                skipped.append((target, L, method,
                                [name for name, value in
                                 zip(("model_a", "model_b", "ceil_a", "ceil_b"),
                                     (va, vb, ca, cb)) if value is None]))
                continue

            ga, gb = va - ca, vb - cb
            rows.append({
                "target": target, "n_layers": L, "method": method,
                "ceiling": ceil_name,
                f"rmse_{label_a}": va, f"rmse_{label_b}": vb,
                f"gap_{label_a}": ga, f"gap_{label_b}": gb,
                "gap_ratio": (ga / gb) if gb > 0 else np.inf,
                f"margin_{label_a}": pick(
                    a, target, L, method, "median_boundary_margin"),
                f"margin_{label_b}": pick(
                    b, target, L, method, "median_boundary_margin"),
                f"nfev_{label_a}": pick(a, target, L, method, "median_nfev"),
                f"nfev_{label_b}": pick(b, target, L, method, "median_nfev"),
                "improved": bool(gb < ga),
            })

    if verbose and skipped:
        print(f"  compare_preconditioning: skipped {len(skipped)} cells")
        for target, L, method, missing in skipped[:10]:
            print(f"    {target} L={L} {method}: missing {', '.join(missing)}")
    return pd.DataFrame(rows)


def compare_preconditioning(dir_a: str, dir_b: str,
                            label_a: str = "p0", label_b: str = "matched",
                            methods: Tuple[str, ...] = ANSATZ_KINDS,
                            ceilings: Mapping[str, str] | None = None,
                            verbose: bool = True) -> pd.DataFrame:
    """File-based compatibility wrapper; in-memory comparison is preferred."""
    def load(directory: str) -> pd.DataFrame:
        path = os.path.join(directory, "sector_exact_summary.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing summary: {path}")
        return pd.read_csv(path)

    return compare_preconditioning_frames(
        load(dir_a), load(dir_b), label_a=label_a, label_b=label_b,
        methods=methods, ceilings=ceilings, verbose=verbose,
    )


# ============================================================
# 11. Full study
# ============================================================

def run_sector_study(*, layer_values=(1, 2, 3), seeds=tuple(range(20)),
                     maxiter: int = 3000, alpha: float = 1e-8, delta: float = 0.06,
                     total_budgets=(300, 1500, 3000, 15000),
                     shot_repetitions: int = 30, shot_master_seed: int = 424242,
                     decay_p: float | None = None,
                     include_warm_start: bool = False,
                     output_dir: str = "sector_graded_fourier_study") -> Dict[str, pd.DataFrame]:
    """
    decay_p:  None  -> per-target defaults (triangle 2, others 1)
              0.0   -> unpreconditioned
              float -> that exponent for every target

    Returns every frame so results survive in memory even if writes fail.
    """
    layer_values = tuple(int(L) for L in layer_values)
    seeds = tuple(int(seed) for seed in seeds)
    total_budgets = tuple(int(B) for B in total_budgets)

    if not layer_values or any(L < 1 for L in layer_values):
        raise ValueError("layer_values must contain positive integers")
    if not seeds:
        raise ValueError("at least one training seed is required")
    if maxiter < 1 or shot_repetitions < 1:
        raise ValueError("maxiter and shot_repetitions must be positive")
    if alpha < 0 or not (0.0 <= delta < 0.5):
        raise ValueError("alpha must be nonnegative and delta must be in [0, 0.5)")
    if any(B < len(BASES_ALL) for B in total_budgets):
        raise ValueError("every shot budget must allow at least one shot per basis")
    if any(B % 6 != 0 for B in total_budgets):
        raise ValueError(
            "budgets must be divisible by 6 for equal 2-basis/3-basis splits"
        )

    os.makedirs(output_dir, exist_ok=True)
    validate_sector_design()
    validate_equivariance()
    validate_partial_warm_start_mapping()

    bundles = build_bundles(delta)
    scale_checks = pd.concat([stopping_scale_checks(bundles, alpha),
                              mask_shift_checks(bundles)], ignore_index=True)
    for _, row in scale_checks[scale_checks.status == "FAIL"].iterrows():
        print(f"warning: {row.prediction}: {row.detail}")

    def weights_for(target: str) -> np.ndarray:
        p = DEFAULT_DECAY_P[target] if decay_p is None else float(decay_p)
        return preconditioner(p)

    def metrics(c: np.ndarray, b: DatasetBundle) -> Dict[str, float]:
        row = {}
        for g, cache in (("train", b.train), ("nested", b.nested), ("shifted", b.shifted)):
            row[f"{g}_rmse"], row[f"{g}_linf"] = masked_rmse_linf(c, cache)
        row["odd_energy"] = float(np.sum(c[SECTOR_COLUMNS["odd"]] ** 2))
        row["even_energy"] = float(np.sum(c[SECTOR_COLUMNS["even"]] ** 2))
        row["train_objective"] = regularized_loss(c, b.train, alpha)
        return row

    exact_rows: List[dict] = []
    shot_rows: List[dict] = []

    classical: Dict[str, Dict[str, np.ndarray]] = {}
    for target, b in bundles.items():
        classical[target] = {
            "zero": np.zeros(N_HARMONICS),
            "analytic_truncation": analytic_coefficients(target),
            "ols_full": ridge_fit(b.train, 0.0, "full"),
            "ols_odd": ridge_fit(b.train, 0.0, "odd"),
            "ols_even": ridge_fit(b.train, 0.0, "even"),
            "ridge_full": ridge_fit(b.train, alpha, "full"),
            "ridge_odd": ridge_fit(b.train, alpha, "odd"),
            "ridge_even": ridge_fit(b.train, alpha, "even"),
        }

    # E4: spherical nulls depend on neither W nor n_layers.
    spherical_cache: Dict[Tuple[str, str, int], Tuple[np.ndarray, int]] = {}
    for target, b in bundles.items():
        for mode in ("spherical_full", "spherical_odd", "spherical_sector"):
            for seed in seeds:
                c, res = spherical_fit(mode, b.train, alpha, int(seed), maxiter)
                spherical_cache[(target, mode, int(seed))] = (
                    c, int(getattr(res, "nfev", -1)))

    inits = ("random", "warm") if include_warm_start else ("random",)

    for L in layer_values:
        for target, b in bundles.items():
            w = weights_for(target)
            reference_full = classical[target]["ridge_full"]

            for method, c in classical[target].items():
                exact_rows.append({
                    "target": target, "sector": TARGET_SECTOR[target], "n_layers": L,
                    "method": method, "kind": "classical", "init": "n/a",
                    "seed": np.nan, "nominal_params": np.nan,
                    "effective_circuit_params": np.nan, "n_bases": np.nan,
                    "boundary_margin": np.nan, "nfev": np.nan, "seconds": np.nan,
                    "included_fraction": b.included_fraction, **metrics(c, b)})

            for mode in ("spherical_full", "spherical_odd", "spherical_sector"):
                for seed in seeds:
                    c, nfev = spherical_cache[(target, mode, int(seed))]
                    exact_rows.append({
                        "target": target, "sector": TARGET_SECTOR[target], "n_layers": L,
                        "method": mode, "kind": "classical_null", "init": "random",
                        "seed": int(seed), "nominal_params": SPHERICAL_N_PARAMS[mode],
                        "effective_circuit_params": np.nan, "n_bases": np.nan,
                        "boundary_margin": np.nan, "nfev": nfev, "seconds": np.nan,
                        "included_fraction": b.included_fraction, **metrics(c, b)})

            for kind in ANSATZ_KINDS:
                for init in inits:
                    for seed in seeds:
                        print(f"target={target}, L={L}, ansatz={kind}, "
                              f"init={init}, seed={seed}")
                        reference = (classical[target]["ridge_odd"]
                                     if kind == "equivariant" else reference_full)
                        m = train_model(target, kind, L, int(seed), b.train, alpha,
                                        maxiter, w, init=init, reference=reference)
                        exact_rows.append({
                            "target": target, "sector": TARGET_SECTOR[target],
                            "n_layers": L, "method": kind, "kind": "variational",
                            "init": init, "seed": int(seed),
                            "nominal_params": n_total_params(L),
                            "effective_circuit_params": effective_circuit_params(kind, L),
                            "n_bases": len(bases_for(kind)),
                            "n_live_observables": len(live_observable_indices(kind)),
                            "boundary_margin": boundary_margin(m),
                            "nfev": m.nfev, "seconds": m.seconds,
                            "optimizer_success": m.success,
                            "optimizer_message": m.message,
                            "optimizer_fun": m.fun,
                            "included_fraction": b.included_fraction,
                            "max_raw_even_expectation": float(
                                np.max(np.abs(m.raw_expectations[EVEN_SLICE]))),
                            "raw_even_expectation_l2": float(
                                np.linalg.norm(m.raw_expectations[EVEN_SLICE])),
                            "max_decoder_even_expectation": float(
                                np.max(np.abs(m.expectations[EVEN_SLICE]))),
                            **{f"log_scale_{f}": float(m.log_scales[i])
                               for i, f in enumerate(FAMILY_NAMES)},
                            **metrics(m.coefficients, b)})

                        cum = prepare_cumulatives(m.state, bases_for(kind))
                        exact_nested, _ = masked_rmse_linf(m.coefficients, b.nested)
                        for budget in total_budgets:
                            pred = predicted_coefficient_noise(
                                m.expectations, m.log_scales, w, kind, int(budget))
                            for rep in range(shot_repetitions):
                                raw_est, decoder_est, per_basis, preps = sampled_expectations(
                                    cum, kind, target=target, n_layers=L,
                                    train_seed=int(seed), total_budget=int(budget),
                                    repetition=rep, master_seed=shot_master_seed)
                                c_noisy = decode(decoder_est, m.log_scales, w)
                                n_rmse, n_linf = masked_rmse_linf(c_noisy, b.nested)
                                s_rmse, _ = masked_rmse_linf(c_noisy, b.shifted)

                                even_x_rows = np.arange(8, 12)
                                raw_even_x = raw_est[even_x_rows]
                                even_x_z = raw_even_x * np.sqrt(per_basis)
                                diagnostic = {
                                    "max_abs_even_x_z": (
                                        float(np.nanmax(np.abs(even_x_z)))
                                        if np.isfinite(raw_even_x).any() else np.nan),
                                }
                                for obs_i, value, z_value in zip(
                                        even_x_rows, raw_even_x, even_x_z):
                                    harmonic = OBSERVABLE_HARMONIC[obs_i]
                                    diagnostic[f"raw_even_x_h{harmonic}"] = float(value)
                                    diagnostic[f"z_even_x_h{harmonic}"] = float(z_value)

                                shot_rows.append({
                                    "target": target, "n_layers": L, "method": kind,
                                    "init": init, "seed": int(seed),
                                    "total_budget": int(budget),
                                    "shots_per_basis": per_basis,
                                    "n_bases": len(bases_for(kind)),
                                    "total_state_preparations": preps,
                                    "repetition": rep,
                                    "nested_rmse": n_rmse, "nested_linf": n_linf,
                                    "shifted_rmse": s_rmse,
                                    "coefficient_noise_l2": float(
                                        np.linalg.norm(c_noisy - m.coefficients)),
                                    "predicted_coefficient_noise": pred,
                                    "nested_degradation": n_rmse - exact_nested,
                                    "nested_mse": n_rmse ** 2,
                                    "nested_mse_degradation": (
                                        n_rmse ** 2 - exact_nested ** 2),
                                    **diagnostic})

    exact_frame = pd.DataFrame(exact_rows)
    exact_frame = add_comparative_columns(exact_frame, exact_frame)
    shot_frame = pd.DataFrame(shot_rows)

    output_paths: Dict[str, str] = {}
    output_paths["exact"] = safe_write_csv(
        exact_frame, os.path.join(output_dir, "sector_exact_results.csv"))
    output_paths["shot"] = safe_write_csv(
        shot_frame, os.path.join(output_dir, "sector_shot_results.csv"))

    # Rank study uses unit weights: W is invertible, so ranks are weight-free.
    ranks = jacobian_rank_study(
        layer_values, np.ones(N_HARMONICS), rank_rtol=1e-8
    )
    output_paths["ranks"] = safe_write_csv(
        ranks, os.path.join(output_dir, "jacobian_ranks.csv"))

    summary = (exact_frame.groupby(["target", "n_layers", "method", "init"],
                                   as_index=False, dropna=False)
               .agg(median_train_rmse=("train_rmse", "median"),
                    median_nested_rmse=("nested_rmse", "median"),
                    median_shifted_rmse=("shifted_rmse", "median"),
                    median_nested_recovery=("nested_recovery", "median"),
                    median_even_energy=("even_energy", "median"),
                    median_boundary_margin=("boundary_margin", "median"),
                    median_train_objective=("train_objective", "median"),
                    median_nfev=("nfev", "median"),
                    median_seconds=("seconds", "median"),
                    optimizer_success_rate=("optimizer_success", "mean")))
    output_paths["summary"] = safe_write_csv(
        summary, os.path.join(output_dir, "sector_exact_summary.csv"))

    shot_summary = (shot_frame.groupby(
        ["target", "n_layers", "method", "init", "total_budget",
         "total_state_preparations"], as_index=False)
        .agg(median_nested_rmse=("nested_rmse", "median"),
             mean_degradation=("nested_degradation", "mean"),
             median_degradation=("nested_degradation", "median"),
             mean_mse_degradation=("nested_mse_degradation", "mean"),
             rms_coefficient_noise=(
                 "coefficient_noise_l2",
                 lambda x: float(np.sqrt(np.mean(np.square(x))))),
             median_coefficient_noise=("coefficient_noise_l2", "median"),
             predicted_coefficient_noise=("predicted_coefficient_noise", "first"),
             median_max_abs_even_x_z=("max_abs_even_x_z", "median")))
    output_paths["shot_summary"] = safe_write_csv(
        shot_summary, os.path.join(output_dir, "sector_shot_summary.csv"))

    tests = pd.concat([paired_tests(exact_frame, "nested_rmse"),
                       paired_tests(exact_frame, "shifted_rmse")], ignore_index=True)
    if len(tests):
        tests["p_holm"] = holm_adjust(tests["p_value"].to_numpy())
    output_paths["tests"] = safe_write_csv(
        tests, os.path.join(output_dir, "sector_paired_tests.csv"))

    decay_note = "per-target defaults" if decay_p is None else f"decay_p={decay_p}"
    checks = check_predictions(exact_frame, ranks, shot_frame,
                               decay_note=f"preconditioner: {decay_note}")
    checks = pd.concat([checks, scale_checks], ignore_index=True)
    output_paths["checks"] = safe_write_csv(
        checks, os.path.join(output_dir, "prediction_checks.csv"))

    output_paths["config"] = safe_write_json(
        {"layer_values": list(layer_values), "seeds": list(seeds),
         "maxiter": maxiter, "maxfun": OPTIONS["maxfun"],
         "alpha": alpha, "delta": delta,
         "decay_p": "per_target" if decay_p is None else decay_p,
         "default_decay_p": DEFAULT_DECAY_P,
         "include_warm_start": include_warm_start,
         "total_budgets": list(total_budgets),
         "shot_repetitions": shot_repetitions,
         "observables": list(OBSERVABLES),
         "observable_harmonics": OBSERVABLE_HARMONIC.tolist(),
         "bases": {k: list(v) for k, v in MEASUREMENT_BASES.items()},
         "entangling_chain": [list(p) for p in ENTANGLING_CHAIN],
         "notes": [
             "sector parity P on functions == Z_3 conjugation on the register",
             "equivariant: raw even expectations are retained for verification; "
             "the decoder projects the even sector to zero",
             "W is diagonal and invertible: ranks, selection rule and ceilings "
             "are all unchanged by preconditioning",
             "P2 uses scale-aware relative tolerances, signed slack, and tests "
             "both equivariant and spherical_odd against exact odd ceilings",
             "P4 compares cellwise RMS coefficient noise with the per-model "
             "variance formula; medians of norms are not RMS quantities",
             "equivariant XXXX shots retain even-X estimates as a zero-symmetry "
             "hardware diagnostic",
             "boundary_margin did NOT behave as predicted: the preconditioning "
             "gain is optimizer conditioning, not relaxed dynamic range",
             "the optional warm arm is a partial three-moment product-state "
             "heuristic, not an inverse of the full decoder",
             "budgets are TOTAL state preparations, split evenly per basis"]},
        os.path.join(output_dir, "sector_study_config.json"))

    print("\nExact summary:\n", summary.to_string())
    print("\nJacobian ranks (P3):\n", ranks.to_string())
    print("\nPrediction checks:\n", checks.to_string())
    print("\nPaired tests (Holm-adjusted):\n", tests.to_string())

    return {"exact": exact_frame, "shot": shot_frame, "ranks": ranks,
            "summary": summary, "shot_summary": shot_summary,
            "tests": tests, "checks": checks, "paths": output_paths}



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sector-graded variational Fourier synthesis study"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "run the full two-preconditioner experiment; "
            "default is a smoke test"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="directory in which run folders and comparison CSV are written",
    )
    return parser


def parse_cli_args(argv=None):
    """Parse terminal arguments while ignoring Jupyter's kernel arguments."""
    parser = build_arg_parser()
    if argv is not None:
        return parser.parse_args(argv)
    if "ipykernel" in sys.modules:
        args, _unknown = parser.parse_known_args()
        return args
    return parser.parse_args()


def main(argv=None):
    """Run the smoke or full two-preconditioner experiment and return frames."""
    args = parse_cli_args(argv)
    os.makedirs(args.output_root, exist_ok=True)

    if args.full:
        config = dict(
            layer_values=(1, 2, 3),
            seeds=tuple(range(20)),
            maxiter=3000,
            total_budgets=(300, 1500, 3000, 15000),
            shot_repetitions=30,
            include_warm_start=False,
        )
        prefix = "full"
    else:
        config = dict(
            layer_values=(1,),
            seeds=(0,),
            maxiter=20,
            total_budgets=(300,),
            shot_repetitions=3,
            include_warm_start=False,
        )
        prefix = "smoke"

    dir_p0 = os.path.join(args.output_root, f"{prefix}_p0")
    dir_matched = os.path.join(args.output_root, f"{prefix}_matched")

    print("=== decay_p = 0 (unpreconditioned) ===")
    res_p0 = run_sector_study(
        **config, decay_p=0.0, output_dir=dir_p0
    )

    print("\n=== matched preconditioner ===")
    res_matched = run_sector_study(
        **config, decay_p=None, output_dir=dir_matched
    )

    print("\n=== P6 COMPARISON ===")
    comparison = compare_preconditioning_frames(
        res_p0["summary"], res_matched["summary"]
    )
    print(comparison.to_string())
    comparison_path = safe_write_csv(
        comparison,
        os.path.join(
            args.output_root,
            f"{prefix}_preconditioning_comparison.csv",
        ),
    )

    return {
        "p0": res_p0,
        "matched": res_matched,
        "comparison": comparison,
        "comparison_path": comparison_path,
    }


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        print(
            "Study functions loaded. No experiment was started automatically.\n"
            "Run main([]) for a smoke test or "
            "main(['--full', '--output-root', 'sector_results_full']) "
            "for the full study."
        )
    else:
        main()
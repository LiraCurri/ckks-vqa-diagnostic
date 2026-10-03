"""
Reachability, conditioning, and optimizer-sensitivity diagnostics for the
sector-graded triangle precision plateau.

Requires ``sector_graded_evalmod.py`` to be importable from the project.
Importing this module only defines the diagnostic functions; no expensive
diagnostic is run automatically.

The diagnostic asks whether the observed ~5e-5 triangle plateau is explained by:
  1. global reachability / expressivity;
  2. local conditioning of the coefficient map; or
  3. the L-BFGS-B stopping tolerance.

It performs:
  * direct coefficient-space inversion against the training-grid ridge reference;
  * a Jacobian singular-spectrum report at the best inverse solution; and
  * an ftol sweep of the original regularized training objective.

The ridge coefficient vector is the optimum of the regularized training
objective for the chosen sector; its nested-grid RMSE is an evaluation
reference, not a lower bound on the nested grid.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Import the canonical sector-study definitions explicitly so this diagnostic
# can be used as a standalone module without hidden notebook state.
from sector_graded_evalmod import (
    N_HARMONICS, DEFAULT_DECAY_P, ANSATZ_KINDS, random_params, model_bounds,
    evaluate_model, live_observable_indices, validate_sector_design,
    validate_equivariance, build_bundles, preconditioner, ridge_fit,
    regularized_loss, masked_rmse_linf, safe_write_csv,
)


DIAGNOSTIC_TIGHT = {
    "ftol": 1e-18,
    "gtol": 1e-14,
    "maxls": 100,
    "maxfun": 1_000_000,
}


def _require_v7_definitions() -> None:
    """Fail early if the imported sector-study module lacks required definitions."""
    required = (
        "N_HARMONICS",
        "DEFAULT_DECAY_P",
        "ANSATZ_KINDS",
        "random_params",
        "model_bounds",
        "evaluate_model",
        "live_observable_indices",
        "validate_sector_design",
        "validate_equivariance",
        "build_bundles",
        "preconditioner",
        "ridge_fit",
        "regularized_loss",
        "masked_rmse_linf",
        "safe_write_csv",
    )
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "The imported sector_graded_evalmod module is missing required "
            f"definitions: {missing}"
        )

    # The canonical evaluate_model interface returns:
    # state, raw_expectations, decoder_expectations, coefficients, log_scales.
    test_weights = preconditioner(0.0)
    test_params = random_params(1, 0)
    test_output = evaluate_model(test_params, "equivariant", 1, test_weights)
    if not isinstance(test_output, tuple) or len(test_output) != 5:
        raise RuntimeError(
            "evaluate_model does not have the expected five-value interface."
        )


def _weights_for_mode(target: str, mode: str) -> tuple[np.ndarray, float]:
    """Return decoder weights and the exponent used by a named mode."""
    if mode == "p0":
        exponent = 0.0
    elif mode == "matched":
        exponent = float(DEFAULT_DECAY_P[target])
    else:
        raise ValueError("preconditioner mode must be 'p0' or 'matched'")
    return preconditioner(exponent), exponent


def _optimizer_diagnostics(res) -> dict:
    jac = getattr(res, "jac", None)
    gradient_inf_norm = (
        float(np.max(np.abs(np.asarray(jac, dtype=float))))
        if jac is not None and np.size(jac)
        else np.nan
    )
    return {
        "success": bool(res.success),
        "status_code": int(getattr(res, "status", -1)),
        "message": str(res.message),
        "nfev": int(getattr(res, "nfev", -1)),
        "nit": int(getattr(res, "nit", -1)),
        "gradient_inf_norm": gradient_inf_norm,
    }


# ============================================================
# 1. Direct inverse problem: can the model reach the ridge reference?
# ============================================================


def inverse_problem_v7(
    *,
    target: str,
    kind: str,
    n_layers: int,
    weights: np.ndarray,
    ceiling: np.ndarray,
    train_cache,
    nested_cache,
    alpha: float,
    seeds: Iterable[int] = range(40),
    maxiter: int = 20_000,
) -> pd.DataFrame:
    """Minimize squared coefficient distance to the classical ridge reference.

    This bypasses the Fourier-grid loss and asks only whether the reference
    coefficient vector can be reached by the circuit/decoder map.

    ``ceiling`` is retained as a legacy parameter/column name for compatibility.
    It denotes the training-grid ridge reference; its nested-grid RMSE is not a
    mathematical lower bound.
    """
    if kind not in ANSATZ_KINDS:
        raise ValueError(f"Unknown ansatz kind: {kind}")

    ceiling = np.asarray(ceiling, dtype=float)
    if ceiling.shape != (N_HARMONICS,):
        raise ValueError(
            f"ceiling must have shape ({N_HARMONICS},), got {ceiling.shape}"
        )

    ceiling_train_rmse, _ = masked_rmse_linf(ceiling, train_cache)
    ceiling_nested_rmse, _ = masked_rmse_linf(ceiling, nested_cache)
    ceiling_objective = regularized_loss(ceiling, train_cache, alpha)

    rows = []
    for seed in (int(s) for s in seeds):
        x0 = random_params(n_layers, seed)

        def objective(params: np.ndarray) -> float:
            _, _, _, coefficients, _ = evaluate_model(
                params, kind, n_layers, weights
            )
            residual = coefficients - ceiling
            return float(np.dot(residual, residual))

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=model_bounds(n_layers),
            options={"maxiter": int(maxiter), **DIAGNOSTIC_TIGHT},
        )

        _, raw_mu, decoder_mu, c_hat, log_scales = evaluate_model(
            res.x, kind, n_layers, weights
        )
        residual = c_hat - ceiling
        train_rmse, _ = masked_rmse_linf(c_hat, train_cache)
        nested_rmse, _ = masked_rmse_linf(c_hat, nested_cache)
        live = live_observable_indices(kind)

        rows.append(
            {
                "target": target,
                "kind": kind,
                "n_layers": int(n_layers),
                "seed": seed,
                "coefficient_l2": float(np.linalg.norm(residual)),
                "coefficient_l2_sq": float(np.dot(residual, residual)),
                "max_abs_residual": float(np.max(np.abs(residual))),
                "train_rmse": train_rmse,
                "nested_rmse": nested_rmse,
                "train_objective": regularized_loss(c_hat, train_cache, alpha),
                "ceiling_train_rmse": ceiling_train_rmse,
                "ceiling_nested_rmse": ceiling_nested_rmse,
                "ceiling_objective": ceiling_objective,
                "nested_rmse_gap": nested_rmse - ceiling_nested_rmse,
                "objective_gap": (
                    regularized_loss(c_hat, train_cache, alpha)
                    - ceiling_objective
                ),
                "max_abs_raw_mu": float(np.max(np.abs(raw_mu[live]))),
                "max_abs_decoder_mu": float(
                    np.max(np.abs(decoder_mu[live]))
                ),
                "min_log_scale": float(np.min(log_scales)),
                "max_log_scale": float(np.max(log_scales)),
                **_optimizer_diagnostics(res),
                # Kept in memory for the conditioning calculation; removed
                # before CSV output.
                "params": np.asarray(res.x, dtype=float),
                "coefficients": np.asarray(c_hat, dtype=float),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 2. Conditioning at the best inverse solution
# ============================================================


def jacobian_at_v7(
    params: np.ndarray,
    kind: str,
    n_layers: int,
    weights: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Central-difference Jacobian of coefficients with respect to parameters."""
    params = np.asarray(params, dtype=float)
    jacobian = np.empty((N_HARMONICS, len(params)), dtype=float)

    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = float(eps)
        c_plus = evaluate_model(
            params + step, kind, n_layers, weights
        )[3]
        c_minus = evaluate_model(
            params - step, kind, n_layers, weights
        )[3]
        jacobian[:, i] = (c_plus - c_minus) / (2.0 * float(eps))

    return jacobian


def conditioning_report_v7(
    *,
    params: np.ndarray,
    residual: np.ndarray,
    kind: str,
    n_layers: int,
    weights: np.ndarray,
    eps: float = 1e-7,
    rank_rtol: float = 1e-8,
) -> dict:
    """Report rank, conditioning, and residual alignment in output space.

    The residual is an output/coefficient-space vector, so its relevant
    singular directions are the LEFT singular vectors of the Jacobian.
    """
    jacobian = jacobian_at_v7(params, kind, n_layers, weights, eps=eps)
    U, singular_values, _ = np.linalg.svd(jacobian, full_matrices=False)

    expected_dim = 8 if kind == "equivariant" else 16
    if len(singular_values) < expected_dim:
        raise RuntimeError(
            f"Only {len(singular_values)} singular values for dimension "
            f"{expected_dim}."
        )

    scale = max(float(singular_values[0]), np.finfo(float).tiny)
    numerical_rank = int(np.sum(singular_values > rank_rtol * scale))
    effective_sv = singular_values[:expected_dim]
    sigma_min = float(effective_sv[-1])

    # A finite full-sector condition number is meaningful only when the
    # expected output dimension is numerically saturated.
    condition_number = (
        float(effective_sv[0] / sigma_min)
        if numerical_rank >= expected_dim and sigma_min > 0.0
        else np.inf
    )

    residual = np.asarray(residual, dtype=float)
    residual_norm = float(np.linalg.norm(residual))

    # Project onto the numerically resolved tangent space rather than blindly
    # using all expected directions. When rank saturates, this is identical to
    # using the full expected sector.
    active_rank = min(numerical_rank, expected_dim)
    U_active = U[:, :active_rank]
    coordinates = U_active.T @ residual
    tangent_component = (
        U_active @ coordinates
        if active_rank > 0
        else np.zeros_like(residual)
    )
    normal_component = residual - tangent_component

    tangent_norm = float(np.linalg.norm(tangent_component))
    normal_norm = float(np.linalg.norm(normal_component))

    if residual_norm > 0.0:
        tangent_fraction = (tangent_norm / residual_norm) ** 2
        normal_fraction = (normal_norm / residual_norm) ** 2
    else:
        tangent_fraction = np.nan
        normal_fraction = np.nan

    coordinate_energy = float(np.dot(coordinates, coordinates))
    if coordinate_energy > 0.0 and active_rank > 0:
        frac_last = float(coordinates[-1] ** 2 / coordinate_energy)
        frac_last_two = float(
            np.sum(coordinates[-min(2, active_rank):] ** 2) / coordinate_energy
        )
    else:
        frac_last = np.nan
        frac_last_two = np.nan

    sigma_next = (
        float(singular_values[expected_dim])
        if expected_dim < len(singular_values)
        else 0.0
    )

    return {
        "numerical_rank": numerical_rank,
        "expected_output_rank": expected_dim,
        "active_tangent_rank": active_rank,
        "rank_rtol": float(rank_rtol),
        "sigma_max": float(effective_sv[0]),
        "sigma_min_effective": sigma_min,
        "sigma_next": sigma_next,
        "condition_number": condition_number,
        "residual_l2": residual_norm,
        "residual_tangent_fraction": tangent_fraction,
        "residual_normal_fraction": normal_fraction,
        "residual_fraction_smallest_left_direction": frac_last,
        "residual_fraction_two_smallest_left_directions": frac_last_two,
        "singular_values": effective_sv.tolist(),
    }


# ============================================================
# 3. Does the L-BFGS-B stopping criterion cause the floor?
# ============================================================


def ftol_sweep_v7(
    *,
    target: str,
    kind: str,
    n_layers: int,
    weights: np.ndarray,
    ceiling: np.ndarray,
    train_cache,
    nested_cache,
    alpha: float,
    seeds: Iterable[int] = range(20),
    ftols: Sequence[float] = (1e-12, 1e-15, 1e-18),
    maxiter: int = 50_000,
) -> pd.DataFrame:
    """Rerun the original objective at successively tighter ftol values.

    ``ceiling`` is retained as a legacy parameter/column name for compatibility.
    It denotes the training-grid ridge reference, not a nested-grid RMSE bound.
    """
    ceiling = np.asarray(ceiling, dtype=float)
    ceiling_train_rmse, _ = masked_rmse_linf(ceiling, train_cache)
    ceiling_nested_rmse, _ = masked_rmse_linf(ceiling, nested_cache)
    ceiling_objective = regularized_loss(ceiling, train_cache, alpha)

    rows = []
    for ftol in (float(value) for value in ftols):
        options = {
            "maxiter": int(maxiter),
            **DIAGNOSTIC_TIGHT,
            "ftol": ftol,
        }

        for seed in (int(s) for s in seeds):
            def objective(params: np.ndarray) -> float:
                _, _, _, coefficients, _ = evaluate_model(
                    params, kind, n_layers, weights
                )
                return regularized_loss(coefficients, train_cache, alpha)

            res = minimize(
                objective,
                random_params(n_layers, seed),
                method="L-BFGS-B",
                bounds=model_bounds(n_layers),
                options=options,
            )

            _, raw_mu, decoder_mu, c_hat, log_scales = evaluate_model(
                res.x, kind, n_layers, weights
            )
            train_rmse, _ = masked_rmse_linf(c_hat, train_cache)
            nested_rmse, _ = masked_rmse_linf(c_hat, nested_cache)
            coefficient_l2 = float(np.linalg.norm(c_hat - ceiling))
            live = live_observable_indices(kind)

            rows.append(
                {
                    "target": target,
                    "kind": kind,
                    "n_layers": int(n_layers),
                    "ftol": ftol,
                    "seed": seed,
                    "objective": float(res.fun),
                    "objective_gap": float(res.fun) - ceiling_objective,
                    "train_rmse": train_rmse,
                    "nested_rmse": nested_rmse,
                    "nested_rmse_gap": nested_rmse - ceiling_nested_rmse,
                    "coefficient_l2_to_ceiling": coefficient_l2,
                    "ceiling_train_rmse": ceiling_train_rmse,
                    "ceiling_nested_rmse": ceiling_nested_rmse,
                    "ceiling_objective": ceiling_objective,
                    "max_abs_raw_mu": float(np.max(np.abs(raw_mu[live]))),
                    "max_abs_decoder_mu": float(
                        np.max(np.abs(decoder_mu[live]))
                    ),
                    "min_log_scale": float(np.min(log_scales)),
                    "max_log_scale": float(np.max(log_scales)),
                    **_optimizer_diagnostics(res),
                    "coefficients": np.asarray(c_hat, dtype=float),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 4. Driver
# ============================================================


def run_precision_floor_diagnostic_v7(
    *,
    delta: float = 0.06,
    alpha: float = 1e-8,
    layer_values: Sequence[int] = (1, 3),
    targets: Sequence[str] = ("triangle",),
    kinds: Sequence[str] = ("equivariant",),
    preconditioner_modes: Sequence[str] = ("p0", "matched"),
    inverse_seeds: int = 40,
    ftol_seeds: int = 20,
    inverse_maxiter: int = 20_000,
    ftol_maxiter: int = 50_000,
    ftols: Sequence[float] = (1e-12, 1e-15, 1e-18),
    jacobian_eps: float = 1e-7,
    rank_rtol: float = 1e-8,
    run_ftol_for_kind: str = "equivariant",
    output_dir: str = "precision_plateau_diagnostic",
) -> dict:
    """Run the complete reachability/conditioning/tolerance diagnostic.

    Defaults focus on the configuration where the precision plateau was observed:
    the triangle target, equivariant ansatz, depths 1 and 3, under both p=0 and
    matched decoder weights. Add ``generic`` to ``kinds`` only as a control.

    The historical ``ceiling_*`` column names are preserved for compatibility,
    but they refer to the training-grid ridge reference.
    """
    _require_v7_definitions()
    validate_sector_design()
    validate_equivariance()

    layer_values = tuple(int(value) for value in layer_values)
    targets = tuple(str(value) for value in targets)
    kinds = tuple(str(value) for value in kinds)
    preconditioner_modes = tuple(str(value) for value in preconditioner_modes)

    if inverse_seeds < 1 or ftol_seeds < 1:
        raise ValueError("inverse_seeds and ftol_seeds must be positive")
    if not layer_values or any(value < 1 for value in layer_values):
        raise ValueError("layer_values must contain positive integers")
    invalid_targets = set(targets) - set(DEFAULT_DECAY_P)
    if invalid_targets:
        raise ValueError(f"Unknown targets: {sorted(invalid_targets)}")
    invalid_kinds = set(kinds) - set(ANSATZ_KINDS)
    if invalid_kinds:
        raise ValueError(f"Unknown ansatz kinds: {sorted(invalid_kinds)}")
    if run_ftol_for_kind not in kinds:
        raise ValueError(
            "run_ftol_for_kind must be included in kinds; "
            f"got {run_ftol_for_kind!r} and kinds={kinds}"
        )

    os.makedirs(output_dir, exist_ok=True)
    bundles = build_bundles(delta)

    inverse_frames = []
    conditioning_rows = []
    ftol_frames = []

    for target in targets:
        bundle = bundles[target]
        ceiling_odd = ridge_fit(bundle.train, alpha, "odd")
        ceiling_full = ridge_fit(bundle.train, alpha, "full")

        for mode in preconditioner_modes:
            weights, decay_p = _weights_for_mode(target, mode)

            for kind in kinds:
                ceiling = ceiling_odd if kind == "equivariant" else ceiling_full

                for n_layers in layer_values:
                    print(
                        "inverse problem: "
                        f"target={target}, mode={mode}, kind={kind}, "
                        f"L={n_layers}"
                    )
                    frame = inverse_problem_v7(
                        target=target,
                        kind=kind,
                        n_layers=n_layers,
                        weights=weights,
                        ceiling=ceiling,
                        train_cache=bundle.train,
                        nested_cache=bundle.nested,
                        alpha=alpha,
                        seeds=range(inverse_seeds),
                        maxiter=inverse_maxiter,
                    )
                    frame.insert(1, "preconditioner", mode)
                    frame.insert(2, "decay_p", decay_p)
                    inverse_frames.append(frame)

                    best_index = frame["coefficient_l2"].idxmin()
                    best = frame.loc[best_index]
                    report = conditioning_report_v7(
                        params=best["params"],
                        residual=best["coefficients"] - ceiling,
                        kind=kind,
                        n_layers=n_layers,
                        weights=weights,
                        eps=jacobian_eps,
                        rank_rtol=rank_rtol,
                    )
                    conditioning_rows.append(
                        {
                            "target": target,
                            "preconditioner": mode,
                            "decay_p": decay_p,
                            "kind": kind,
                            "n_layers": n_layers,
                            "best_seed": int(best["seed"]),
                            "best_coefficient_l2": float(
                                best["coefficient_l2"]
                            ),
                            "best_nested_rmse": float(best["nested_rmse"]),
                            "ceiling_nested_rmse": float(
                                best["ceiling_nested_rmse"]
                            ),
                            "best_nested_rmse_gap": float(
                                best["nested_rmse_gap"]
                            ),
                            **report,
                        }
                    )

            # The stopping-tolerance diagnostic is run only for the chosen
            # plateau-observed ansatz, not for every control ansatz.
            if run_ftol_for_kind in kinds:
                ceiling = (
                    ceiling_odd
                    if run_ftol_for_kind == "equivariant"
                    else ceiling_full
                )
                for n_layers in layer_values:
                    print(
                        "ftol sweep: "
                        f"target={target}, mode={mode}, "
                        f"kind={run_ftol_for_kind}, L={n_layers}"
                    )
                    sweep = ftol_sweep_v7(
                        target=target,
                        kind=run_ftol_for_kind,
                        n_layers=n_layers,
                        weights=weights,
                        ceiling=ceiling,
                        train_cache=bundle.train,
                        nested_cache=bundle.nested,
                        alpha=alpha,
                        seeds=range(ftol_seeds),
                        ftols=ftols,
                        maxiter=ftol_maxiter,
                    )
                    sweep.insert(1, "preconditioner", mode)
                    sweep.insert(2, "decay_p", decay_p)
                    ftol_frames.append(sweep)

    inverse_with_arrays = pd.concat(inverse_frames, ignore_index=True)
    conditioning_with_arrays = pd.DataFrame(conditioning_rows)
    sweep_with_arrays = pd.concat(ftol_frames, ignore_index=True)

    inverse_summary = (
        inverse_with_arrays.groupby(
            ["target", "preconditioner", "decay_p", "kind", "n_layers"],
            as_index=False,
        )
        .agg(
            best_coefficient_l2=("coefficient_l2", "min"),
            median_coefficient_l2=("coefficient_l2", "median"),
            best_nested_rmse=("nested_rmse", "min"),
            median_nested_rmse=("nested_rmse", "median"),
            ceiling_nested_rmse=("ceiling_nested_rmse", "first"),
            best_nested_rmse_gap=("nested_rmse_gap", "min"),
            median_nested_rmse_gap=("nested_rmse_gap", "median"),
            best_objective_gap=("objective_gap", "min"),
            median_nfev=("nfev", "median"),
            success_rate=("success", "mean"),
            max_abs_raw_mu=("max_abs_raw_mu", "max"),
        )
    )

    ftol_summary = (
        sweep_with_arrays.groupby(
            [
                "target",
                "preconditioner",
                "decay_p",
                "kind",
                "n_layers",
                "ftol",
            ],
            as_index=False,
        )
        .agg(
            median_objective=("objective", "median"),
            min_objective=("objective", "min"),
            median_objective_gap=("objective_gap", "median"),
            median_nested_rmse=("nested_rmse", "median"),
            min_nested_rmse=("nested_rmse", "min"),
            median_nested_rmse_gap=("nested_rmse_gap", "median"),
            min_nested_rmse_gap=("nested_rmse_gap", "min"),
            median_coefficient_l2=(
                "coefficient_l2_to_ceiling", "median"
            ),
            min_coefficient_l2=("coefficient_l2_to_ceiling", "min"),
            median_nfev=("nfev", "median"),
            success_rate=("success", "mean"),
        )
    )

    print("\n=== DIRECT INVERSE PROBLEM ===")
    print(inverse_summary.to_string(index=False))

    print("\n=== CONDITIONING AT BEST INVERSE SOLUTION ===")
    display_columns = [
        column
        for column in conditioning_with_arrays.columns
        if column != "singular_values"
    ]
    print(conditioning_with_arrays[display_columns].to_string(index=False))
    for _, row in conditioning_with_arrays.iterrows():
        singular_values = np.asarray(row["singular_values"], dtype=float)
        singular_values_text = np.array2string(
            singular_values,
            precision=3,
            max_line_width=200,
        )
        print(
            f"  {row['target']} {row['preconditioner']} "
            f"{row['kind']} L={int(row['n_layers'])}: "
            f"sv={singular_values_text}"
        )

    print("\n=== FTOL SWEEP ===")
    print(ftol_summary.to_string(index=False))

    inverse_csv = inverse_with_arrays.drop(
        columns=["params", "coefficients"]
    )
    conditioning_csv = conditioning_with_arrays.copy()
    conditioning_csv["singular_values"] = conditioning_csv[
        "singular_values"
    ].map(lambda values: ";".join(f"{value:.17g}" for value in values))
    sweep_csv = sweep_with_arrays.drop(columns=["coefficients"])

    written_paths = {
        "inverse": safe_write_csv(
            inverse_csv,
            os.path.join(output_dir, "inverse_problem.csv"),
        ),
        "inverse_summary": safe_write_csv(
            inverse_summary,
            os.path.join(output_dir, "inverse_problem_summary.csv"),
        ),
        "conditioning": safe_write_csv(
            conditioning_csv,
            os.path.join(output_dir, "conditioning.csv"),
        ),
        "ftol_sweep": safe_write_csv(
            sweep_csv,
            os.path.join(output_dir, "ftol_sweep.csv"),
        ),
        "ftol_summary": safe_write_csv(
            ftol_summary,
            os.path.join(output_dir, "ftol_sweep_summary.csv"),
        ),
    }

    print("\nWritten files:")
    for name, path in written_paths.items():
        print(f"  {name}: {path}")

    return {
        "inverse": inverse_csv,
        "inverse_summary": inverse_summary,
        "conditioning": conditioning_with_arrays,
        "ftol_sweep": sweep_csv,
        "ftol_summary": ftol_summary,
        "written_paths": written_paths,
    }


# Backward-compatible public alias used by the replication package.
run_precision_plateau_diagnostic = run_precision_floor_diagnostic_v7

"""Development-only crossed-REML architecture prototype.

This module is not production code.  Increment 2 uses it to decide whether a
private lme4-aligned implementation is feasible before replacing the current
solver.
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.util
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy import __version__ as scipy_version
from scipy.optimize import Bounds, minimize

_NLOPT_LN_BOBYQA = 34
_NLOPT_OBJECTIVE = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
)


def _system_nlopt_version(library) -> str:
    major = ctypes.c_int()
    minor = ctypes.c_int()
    bugfix = ctypes.c_int()
    library.nlopt_version(
        ctypes.byref(major),
        ctypes.byref(minor),
        ctypes.byref(bugfix),
    )
    return f"{major.value}.{minor.value}.{bugfix.value}"


def _nlopt_bobyqa_2_7(objective, initial: np.ndarray):
    """Call the installed NLopt 2.7 C ABI through a two-parameter binding."""
    library_name = ctypes.util.find_library("nlopt")
    if library_name is None:
        raise RuntimeError("the system NLopt library is unavailable")
    library = ctypes.CDLL(library_name)
    library.nlopt_create.argtypes = [ctypes.c_int, ctypes.c_uint]
    library.nlopt_create.restype = ctypes.c_void_p
    library.nlopt_destroy.argtypes = [ctypes.c_void_p]
    library.nlopt_set_min_objective.argtypes = [
        ctypes.c_void_p,
        _NLOPT_OBJECTIVE,
        ctypes.c_void_p,
    ]
    vector_pointer = ctypes.POINTER(ctypes.c_double)
    library.nlopt_set_lower_bounds.argtypes = [ctypes.c_void_p, vector_pointer]
    library.nlopt_set_upper_bounds.argtypes = [ctypes.c_void_p, vector_pointer]
    library.nlopt_set_xtol_abs1.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.nlopt_set_ftol_abs.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.nlopt_set_maxeval.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.nlopt_optimize.argtypes = [
        ctypes.c_void_p,
        vector_pointer,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nlopt_optimize.restype = ctypes.c_int
    version = _system_nlopt_version(library)
    if version != "2.7.1":
        raise RuntimeError(f"the REML prototype requires system NLopt 2.7.1, found {version}")

    callback_errors: list[Exception] = []

    @_NLOPT_OBJECTIVE
    def callback(count, values, _gradient, _data):
        try:
            parameters = np.ctypeslib.as_array(values, shape=(count,))
            return float(objective(parameters.copy()))
        except Exception as error:  # pragma: no cover - dependency callback safety
            callback_errors.append(error)
            return np.finfo(float).max

    optimizer = library.nlopt_create(_NLOPT_LN_BOBYQA, 2)
    if not optimizer:
        raise RuntimeError("NLopt could not create the BOBYQA optimizer")
    parameters = np.asarray(initial, dtype=float).copy()
    lower = np.zeros(2, dtype=float)
    upper = np.full(2, np.inf, dtype=float)
    optimum = ctypes.c_double()
    try:
        settings = (
            library.nlopt_set_min_objective(optimizer, callback, None),
            library.nlopt_set_lower_bounds(
                optimizer,
                lower.ctypes.data_as(vector_pointer),
            ),
            library.nlopt_set_upper_bounds(
                optimizer,
                upper.ctypes.data_as(vector_pointer),
            ),
            library.nlopt_set_xtol_abs1(optimizer, 1e-8),
            library.nlopt_set_ftol_abs(optimizer, 1e-8),
            library.nlopt_set_maxeval(optimizer, 100_000),
        )
        if any(result < 0 for result in settings):
            raise RuntimeError(f"NLopt rejected prototype settings: {settings}")
        status = library.nlopt_optimize(
            optimizer,
            parameters.ctypes.data_as(vector_pointer),
            ctypes.byref(optimum),
        )
    finally:
        library.nlopt_destroy(optimizer)
    if callback_errors:
        raise RuntimeError("NLopt objective callback failed") from callback_errors[0]
    return parameters, float(optimum.value), int(status), version


def _scipy_cobyqa(objective, initial: np.ndarray):
    """Evaluate SciPy COBYQA without assuming equivalence to NLopt BOBYQA."""

    def finite_objective(parameters):
        try:
            return objective(parameters)
        except Exception:
            return np.finfo(float).max

    result = minimize(
        finite_objective,
        np.asarray(initial, dtype=float),
        method="COBYQA",
        bounds=Bounds(np.zeros(2), np.full(2, np.inf)),
        options={
            "maxfev": 100_000,
            "maxiter": 100_000,
            "final_tr_radius": 1e-8,
        },
    )
    raw_status = 1 if result.success else 5
    return (
        np.asarray(result.x, dtype=float),
        float(result.fun),
        raw_status,
        scipy_version,
    )


def _profiled_state(
    theta: np.ndarray,
    *,
    y: np.ndarray,
    fixed_design: np.ndarray,
    random_incidences: tuple[np.ndarray, np.ndarray],
    normalized_weights: np.ndarray,
) -> dict[str, Any]:
    """Evaluate lme4's fixed-effect/residual-scale-profiled REML criterion."""
    try:
        from sksparse.cholmod import cho_factor
    except ImportError as error:
        raise RuntimeError(
            "the CHOLMOD prototype requires the optional scikit-sparse package"
        ) from error

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,) or not np.all(np.isfinite(theta)):
        raise ValueError("finite two-component theta is required")
    random_design = sparse.csc_array(np.column_stack(random_incidences))
    relative_scales = np.concatenate(
        [
            np.repeat(value, incidence.shape[1])
            for value, incidence in zip(theta, random_incidences, strict=True)
        ]
    )
    square_root_weights = np.sqrt(normalized_weights)
    lambda_transposed_random_design = (
        sparse.diags(relative_scales) @ random_design.T @ sparse.diags(square_root_weights)
    )
    weighted_fixed_design = square_root_weights[:, None] * fixed_design
    weighted_response = square_root_weights * y
    random_information = sparse.csc_matrix(
        lambda_transposed_random_design @ lambda_transposed_random_design.T
    )
    try:
        random_factor = cho_factor(
            random_information,
            beta=1.0,
            lower=True,
            order="default",
        )
    except RuntimeError as error:
        raise ValueError("profiled REML random information is not positive definite") from error
    random_fixed = lambda_transposed_random_design @ weighted_fixed_design
    random_response = lambda_transposed_random_design @ weighted_response
    inverse_random_fixed = random_factor.solve(np.asarray(random_fixed))
    inverse_random_response = random_factor.solve(np.asarray(random_response))
    fixed_information = (
        weighted_fixed_design.T @ weighted_fixed_design - random_fixed.T @ inverse_random_fixed
    )
    try:
        fixed_covariance_unscaled = np.linalg.inv(fixed_information)
    except np.linalg.LinAlgError as error:
        raise ValueError("profiled REML fixed information is singular") from error
    fixed_effect = fixed_covariance_unscaled @ (
        weighted_fixed_design.T @ weighted_response - random_fixed.T @ inverse_random_response
    )
    spherical_random_effects = random_factor.solve(
        np.asarray(random_response - random_fixed @ fixed_effect)
    )
    weighted_residual = (
        weighted_response
        - weighted_fixed_design @ fixed_effect
        - lambda_transposed_random_design.T @ spherical_random_effects
    )
    penalized_residual_sum_squares = float(
        weighted_residual @ weighted_residual + spherical_random_effects @ spherical_random_effects
    )
    degrees_of_freedom = y.size - fixed_design.shape[1]
    residual_variance = penalized_residual_sum_squares / degrees_of_freedom
    random_logdet = float(random_factor.logdet())
    fixed_sign, fixed_logdet = np.linalg.slogdet(fixed_information)
    if fixed_sign <= 0.0 or not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("profiled REML scale is not positive and finite")
    criterion = (
        degrees_of_freedom * (1.0 + np.log(2.0 * np.pi * residual_variance))
        + random_logdet
        + fixed_logdet
        - float(np.log(normalized_weights).sum())
    )
    if not np.isfinite(criterion):
        raise ValueError("profiled REML criterion is not finite")
    random_effects: list[np.ndarray] = []
    start = 0
    for value, incidence in zip(theta, random_incidences, strict=True):
        stop = start + incidence.shape[1]
        random_effects.append(value * spherical_random_effects[start:stop])
        start = stop
    return {
        "criterion": float(criterion),
        "theta": theta.copy(),
        "fixed_effect": fixed_effect,
        "fixed_covariance": residual_variance * fixed_covariance_unscaled,
        "random_effects": random_effects,
        "residual_variance": float(residual_variance),
    }


def _initial_theta(
    y: np.ndarray,
    observation_incidence: np.ndarray,
    bootstrap_incidence: np.ndarray,
) -> np.ndarray:
    """Reproduce ``mkLmerDevfun``'s random-intercept start heuristic."""
    group_variances = []
    for incidence in (observation_incidence, bootstrap_incidence):
        counts = incidence.sum(axis=0)
        means = (incidence.T @ y) / counts
        repeated_means = incidence @ means
        group_variances.append(float(np.var(repeated_means, ddof=1)))
    residual_start = float(np.var(y, ddof=1) - sum(group_variances))
    if residual_start > 0.0:
        relative = np.asarray(group_variances, dtype=float) / residual_start
        if np.all(relative >= 0.0):
            return np.sqrt(relative)
    return np.ones(2, dtype=float)


def _deriv12(function, parameters: np.ndarray, value: float, delta: float = 1e-4):
    """Port lme4 2.0-6's central gradient/Hessian diagnostic stencil."""
    parameters = np.asarray(parameters, dtype=float)
    count = len(parameters)
    gradient = np.empty(count, dtype=float)
    hessian = np.empty((count, count), dtype=float)
    for column in range(count):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[column] += delta
        lower[column] -= delta
        upper_value = function(upper)
        lower_value = function(lower)
        hessian[column, column] = (upper_value - 2.0 * value + lower_value) / delta**2
        gradient[column] = (upper_value - lower_value) / (2.0 * delta)
        for row in range(column):
            upper_upper = parameters.copy()
            upper_lower = parameters.copy()
            lower_upper = parameters.copy()
            lower_lower = parameters.copy()
            upper_upper[[row, column]] += delta
            upper_lower[row] += delta
            upper_lower[column] -= delta
            lower_upper[row] -= delta
            lower_upper[column] += delta
            lower_lower[[row, column]] -= delta
            hessian[row, column] = hessian[column, row] = (
                function(upper_upper)
                - function(upper_lower)
                - function(lower_upper)
                + function(lower_lower)
            ) / (4.0 * delta**2)
    return gradient, hessian


def _convergence_messages(
    theta: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Apply lme4 2.0-6 singular, scaled-gradient, and Hessian checks."""
    singular_tolerance = 1e-4
    derivative_tolerance = 2e-3
    hessian_tolerance = 1e-6
    singular = bool(np.min(theta) < singular_tolerance)
    diagnostics: dict[str, Any] = {
        "gradient": gradient.copy(),
        "hessian": hessian.copy(),
        "hessian_eigenvalues": np.linalg.eigvalsh(hessian),
        "scaled_gradient": None,
        "minimum_gradient": None,
    }
    if singular:
        return singular, ["boundary (singular) fit: see help('isSingular')"], diagnostics

    messages: list[str] = []
    try:
        scaled_gradient = np.linalg.solve(np.linalg.cholesky(hessian).T, gradient)
    except np.linalg.LinAlgError:
        scaled_gradient = None
        messages.append("unable to evaluate scaled gradient")
    if scaled_gradient is not None and np.all(np.isfinite(scaled_gradient)):
        minimum_gradient = np.minimum(np.abs(scaled_gradient), np.abs(gradient))
        diagnostics["scaled_gradient"] = scaled_gradient.copy()
        diagnostics["minimum_gradient"] = minimum_gradient.copy()
        maximum = float(np.max(minimum_gradient))
        if maximum > derivative_tolerance:
            component = int(np.argmax(minimum_gradient)) + 1
            messages.append(
                "Model failed to converge with "
                f"max|grad| = {maximum:g} "
                f"(tol = {derivative_tolerance:g}, component {component})"
            )
    elif scaled_gradient is not None:
        messages.append("unable to evaluate scaled gradient")

    eigenvalues = diagnostics["hessian_eigenvalues"]
    negative = int(np.sum(eigenvalues < -hessian_tolerance))
    if negative:
        messages.append(
            "Model failed to converge: degenerate Hessian with " f"{negative} negative eigenvalues"
        )
        return singular, messages, diagnostics
    zero = int(np.sum(np.abs(eigenvalues) < hessian_tolerance))
    if zero:
        messages.append("Hessian is numerically singular: parameters are not uniquely determined")
        return singular, messages, diagnostics
    condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    diagnostics["hessian_condition"] = condition
    if float(np.max(eigenvalues)) * hessian_tolerance > 1.0:
        messages.append("Model is nearly unidentifiable: very large eigenvalue")
    if float(np.min(eigenvalues) / np.max(eigenvalues)) < hessian_tolerance:
        messages.append("Model is nearly unidentifiable: large eigenvalue ratio")
    return singular, messages, diagnostics


def fit_profiled_reml_prototype(
    frame: pd.DataFrame,
    *,
    optimizer_backend: str = "system_nlopt",
) -> dict[str, Any]:
    """Fit the candidate profiled crossed-intercept REML architecture."""

    def failed(error: Exception) -> dict[str, Any]:
        return {
            "error": f"{type(error).__name__}: {error}",
            "optimizer_code": None,
            "converged": False,
            "singular": False,
        }

    from tests._historical_rarcat_solver import (
        _r_group_incidence,
        _r_value_string,
        _unique_r_strings,
    )

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = ("id", "bootstrap", "ame", "sterror")
    if any(column not in frame for column in required):
        raise ValueError("frame must contain id, bootstrap, ame, and sterror")
    before = copy.deepcopy(frame)
    source = frame.loc[:, required].copy(deep=True)
    source_row_names = _unique_r_strings(
        list(source.index),
        label="frame index labels",
    )
    included = source.loc[~pd.isna(source["ame"])].copy(deep=True)
    if included.empty:
        raise ValueError("crossed pool has no non-missing AME rows")
    if included["id"].isna().any() or included["bootstrap"].isna().any():
        raise ValueError("id and bootstrap must not contain missing values")
    y = included["ame"].to_numpy(dtype=float)
    standard_errors = included["sterror"].to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("included AMEs must contain only finite values")
    if not np.all(np.isfinite(standard_errors)) or np.any(standard_errors == 0.0):
        raise ValueError("included standard errors must be finite and nonzero")
    inverse_variance_weights = 1.0 / np.square(standard_errors)
    normalized_weights = inverse_variance_weights / np.mean(inverse_variance_weights)
    if not np.all(np.isfinite(normalized_weights)) or np.any(normalized_weights <= 0.0):
        raise ValueError("normalized inverse-variance weights must be positive and finite")
    observation_names, observation_incidence = _r_group_incidence(
        included["id"],
        label="id labels",
    )
    bootstrap_names, bootstrap_incidence = _r_group_incidence(
        included["bootstrap"],
        label="bootstrap labels",
    )
    fixed_design = np.ones((included.shape[0], 1), dtype=float)
    semantic_incidences = (observation_incidence, bootstrap_incidence)
    optimizer_order = tuple(
        sorted(
            range(len(semantic_incidences)),
            key=lambda position: (-semantic_incidences[position].shape[1], position),
        )
    )
    optimizer_incidences = tuple(semantic_incidences[position] for position in optimizer_order)

    def objective(theta):
        return _profiled_state(
            np.asarray(theta, dtype=float),
            y=y,
            fixed_design=fixed_design,
            random_incidences=optimizer_incidences,
            normalized_weights=normalized_weights,
        )["criterion"]

    semantic_initial_theta = _initial_theta(
        y,
        observation_incidence,
        bootstrap_incidence,
    )
    initial_theta = semantic_initial_theta[np.asarray(optimizer_order)]
    if optimizer_backend == "system_nlopt":
        optimize = _nlopt_bobyqa_2_7
        optimizer_method = "NLOPT_LN_BOBYQA"
    elif optimizer_backend == "scipy_cobyqa":
        optimize = _scipy_cobyqa
        optimizer_method = "scipy.optimize.COBYQA"
    else:
        raise ValueError(f"unknown REML prototype optimizer {optimizer_backend!r}")
    try:
        theta, objective_value, raw_status, optimizer_version = optimize(
            objective,
            initial_theta,
        )
    except Exception as error:
        return failed(error)

    boundary_positions = np.flatnonzero(theta == 0.0)
    if boundary_positions.size:
        try:
            boundary_value = objective(theta)
            inward_gradients = []
            for position in boundary_positions:
                inward = theta.copy()
                inward[position] = 1e-5
                inward_gradients.append((objective(inward) - boundary_value) / 1e-5)
        except Exception as error:
            return failed(error)
        if any(value < 0.0 for value in inward_gradients):
            try:
                theta, objective_value, raw_status, optimizer_version = optimize(
                    objective,
                    theta,
                )
            except Exception as error:
                return failed(error)

    for position in np.flatnonzero((theta > 0.0) & (theta < 1e-5)):
        boundary = theta.copy()
        boundary[position] = 0.0
        try:
            boundary_value = objective(boundary)
        except Exception as error:
            return failed(error)
        if boundary_value < objective_value:
            theta = boundary
            objective_value = boundary_value

    try:
        state = _profiled_state(
            theta,
            y=y,
            fixed_design=fixed_design,
            random_incidences=optimizer_incidences,
            normalized_weights=normalized_weights,
        )
        gradient, hessian = _deriv12(objective, theta, objective_value)
    except Exception as error:
        return failed(error)
    singular, messages, diagnostics = _convergence_messages(theta, gradient, hessian)
    optimizer_code = raw_status if raw_status < 0 or raw_status == 5 else 0
    converged = optimizer_code == 0 and not messages
    residual_variance = float(state["residual_variance"])
    ordered_variances = residual_variance * np.square(theta)
    semantic_variances = np.empty(2, dtype=float)
    semantic_random_effects: list[np.ndarray | None] = [None, None]
    for ordered_position, semantic_position in enumerate(optimizer_order):
        semantic_variances[semantic_position] = ordered_variances[ordered_position]
        semantic_random_effects[semantic_position] = state["random_effects"][ordered_position]
    observation_random_effects = semantic_random_effects[0]
    bootstrap_random_effects = semantic_random_effects[1]
    assert observation_random_effects is not None
    assert bootstrap_random_effects is not None
    included_row_names = [_r_value_string(value) for value in included.index]
    included_set = set(included.index)
    omitted_row_names = [
        _r_value_string(value) for value in source.index if value not in included_set
    ]
    if not frame.equals(before):
        raise AssertionError("the REML prototype mutated its input")
    return {
        "error": None,
        "warnings": [],
        "formula": "ame ~ (1 | id) + (1 | bootstrap)",
        "source_row_names": source_row_names,
        "included_row_names": included_row_names,
        "omitted_row_names": omitted_row_names,
        "normalized_weights": normalized_weights.copy(),
        "nobs": int(included.shape[0]),
        "fixed_effect_names": ["(Intercept)"],
        "fixed_intercept": float(state["fixed_effect"][0]),
        "fixed_standard_error": float(np.sqrt(state["fixed_covariance"][0, 0])),
        "observation_variance": float(semantic_variances[0]),
        "bootstrap_variance": float(semantic_variances[1]),
        "residual_variance": residual_variance,
        "observation_random_effect_names": observation_names,
        "observation_random_effects": np.asarray(
            observation_random_effects,
            dtype=float,
        ).copy(),
        "bootstrap_random_effect_names": bootstrap_names,
        "bootstrap_random_effects": np.asarray(
            bootstrap_random_effects,
            dtype=float,
        ).copy(),
        "optimizer_code": optimizer_code,
        "convergence_messages": messages,
        "converged": converged,
        "singular": singular,
        "reml": True,
        "log_likelihood": -0.5 * float(state["criterion"]),
        "reml_criterion": float(state["criterion"]),
        "theta": theta.copy(),
        "optimizer": {
            "method": optimizer_method,
            "version": optimizer_version,
            "initial_theta": initial_theta.copy(),
            "raw_status": raw_status,
            "objective": objective_value,
            "xtol_abs": 1e-8,
            "ftol_abs": 1e-8,
            "maxeval": 100_000,
            "boundary_restart": True,
            "boundary_tolerance": 1e-5,
            "diagnostics": diagnostics,
        },
    }

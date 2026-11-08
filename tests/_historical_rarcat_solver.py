from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from weightedcluster._bayesian_rarcat_design import (
    _base_predictors,
    _build_from_design_info,
    _ordered_design,
    _r_character_levels,
    _r_design_column_name,  # noqa: F401
    _r_design_data,
    _r_expression_name,
    _r_sort_strings,
    _r_value_string,
    _split_formula,
    _translate_formula_rhs,
)

_MARGINS_EPS = 1e-7


def _fisher_forward(values: np.ndarray) -> np.ndarray:
    """Apply the source's Fisher forward transform on its finite safe domain."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) >= 1.0):
        raise ValueError("Fisher values must be finite and strictly inside (-1, 1)")
    return np.arctanh(array).copy()


def _fisher_backtransform_pool(
    *,
    coefficient: float,
    standard_error: float,
    observation_variance: float,
    bootstrap_variance: float,
) -> dict[str, float]:
    """Apply WeightedCluster 2.0's installed post-pooling statements exactly."""
    coefficient_value = float(coefficient)
    standard_error_value = float(standard_error)
    variances = np.asarray(
        [observation_variance, bootstrap_variance],
        dtype=float,
    )
    if not np.isfinite(coefficient_value):
        raise ValueError("Fisher coefficient must be finite")
    if not np.isfinite(standard_error_value) or standard_error_value < 0.0:
        raise ValueError("Fisher standard error must be finite and non-negative")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        raise ValueError("Fisher variance components must be finite and non-negative")

    backtransformed_coefficient = float(np.tanh(coefficient_value))
    # This intentionally reapplies tanh to the already backtransformed
    # coefficient because that is the installed WeightedCluster 2.0 order.
    backtransformed_standard_error = standard_error_value * (
        1.0 - float(np.tanh(backtransformed_coefficient)) ** 2
    )
    transformed_variances = np.tanh(variances)
    return {
        "coefficient": backtransformed_coefficient,
        "standard_error": float(backtransformed_standard_error),
        "observation_standard_deviation": float(np.sqrt(transformed_variances[0])),
        "bootstrap_standard_deviation": float(np.sqrt(transformed_variances[1])),
    }


def _unique_r_strings(values: list[Any], *, label: str) -> list[str]:
    rendered = [_r_value_string(value) for value in values]
    if len(set(rendered)) != len(rendered):
        raise ValueError(f"{label} must remain unique after R string normalization")
    return rendered


def _r_group_incidence(
    series: pd.Series,
    *,
    label: str,
) -> tuple[list[str], np.ndarray]:
    """Return lme4-compatible group names and an aligned incidence matrix."""
    observed_values = list(pd.unique(series))
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(observed_values)
        levels = [level for level in series.cat.categories if level in observed]
    elif pd.api.types.is_bool_dtype(series.dtype):
        observed = set(bool(value) for value in observed_values)
        levels = [value for value in (False, True) if value in observed]
    elif pd.api.types.is_numeric_dtype(series.dtype):
        levels = sorted(observed_values, key=float)
    else:
        levels = _r_character_levels(observed_values, label=label)

    names = _unique_r_strings(levels, label=label)
    if len(levels) < 2:
        raise ValueError(f"{label} require at least two groups")
    row_level_names = np.asarray([_r_value_string(value) for value in series], dtype=object)
    incidence = np.equal.outer(row_level_names, names).astype(float)
    if not np.all(np.sum(incidence, axis=1) == 1.0):
        raise ValueError(f"{label} could not be aligned to R grouping levels")
    return names, incidence


def _r_column_class(series: pd.Series) -> list[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return ["factor"]
    if pd.api.types.is_bool_dtype(series.dtype):
        return ["logical"]
    if pd.api.types.is_numeric_dtype(series.dtype):
        return ["numeric"]
    return ["character"]


def _effect_specs(
    rhs: str,
    retained_data: pd.DataFrame,
) -> list[tuple[str, str, Any]]:
    specs: list[tuple[str, str, Any]] = []
    for variable in _base_predictors(rhs, retained_data):
        series = retained_data[variable]
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = list(series.cat.categories)
            if not categories:
                raise ValueError(f"factor {variable} has no declared levels")
            base = categories[0]
            observed = set(series.dropna().tolist())
            unused = [level for level in categories if level not in observed]
            if unused:
                raise ValueError(f"factor {variable} has new level {_r_value_string(unused[0])}")
            for level in categories[1:]:
                specs.append(
                    (
                        variable + _r_value_string(level),
                        "factor",
                        (variable, base, level),
                    )
                )
        elif pd.api.types.is_bool_dtype(series.dtype):
            specs.append((variable, "logical", variable))
        elif pd.api.types.is_numeric_dtype(series.dtype):
            specs.append((variable, "numeric", variable))
        else:
            observed = list(pd.unique(series))
            if not observed:
                raise ValueError(f"factor {variable} has no observed levels")
            levels = _r_character_levels(
                observed,
                label=f"factor {variable} levels",
            )
            base = levels[0]
            for level in levels[1:]:
                specs.append(
                    (
                        variable + _r_value_string(level),
                        "factor",
                        (variable, base, level),
                    )
                )
    ordered_names = _r_sort_strings([item[0] for item in specs])
    specs_by_name = {item[0]: item for item in specs}
    return [specs_by_name[name] for name in ordered_names]


def _average_marginal_effects(
    beta: np.ndarray,
    *,
    retained_data: pd.DataFrame,
    design_info: patsy.DesignInfo,
    ordered_positions: list[int],
    effect_specs: list[tuple[str, str, Any]],
) -> np.ndarray:
    effects: list[float] = []
    for _, effect_type, specification in effect_specs:
        d0 = retained_data.copy(deep=True)
        d1 = retained_data.copy(deep=True)
        if effect_type == "factor":
            variable, base, level = specification
            d0[variable] = base
            d1[variable] = level
            denominator: np.ndarray | None = None
        elif effect_type == "logical":
            variable = str(specification)
            d0[variable] = False
            d1[variable] = True
            denominator = None
        else:
            variable = str(specification)
            values = retained_data[variable].to_numpy(dtype=float)
            step = max(float(np.max(np.abs(values))), 1.0) * np.sqrt(_MARGINS_EPS)
            d0[variable] = values - step
            d1[variable] = values + step
            denominator = d1[variable].to_numpy(dtype=float) - d0[variable].to_numpy(dtype=float)
        X0 = _build_from_design_info(design_info, ordered_positions, d0)
        X1 = _build_from_design_info(design_info, ordered_positions, d1)
        difference = expit(X1 @ beta) - expit(X0 @ beta)
        if denominator is not None:
            difference = difference / denominator
        effects.append(float(np.mean(difference)))
    return np.asarray(effects, dtype=float)


def _margins_with_delta_se(
    beta: np.ndarray,
    covariance: np.ndarray,
    *,
    retained_data: pd.DataFrame,
    design_info: patsy.DesignInfo,
    ordered_positions: list[int],
    effect_specs: list[tuple[str, str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    ames = _average_marginal_effects(
        beta,
        retained_data=retained_data,
        design_info=design_info,
        ordered_positions=ordered_positions,
        effect_specs=effect_specs,
    )
    jacobian = np.empty((len(effect_specs), beta.size), dtype=float)
    for position in range(beta.size):
        shifted = beta.copy()
        shifted[position] += _MARGINS_EPS
        jacobian[:, position] = (
            _average_marginal_effects(
                shifted,
                retained_data=retained_data,
                design_info=design_info,
                ordered_positions=ordered_positions,
                effect_specs=effect_specs,
            )
            - ames
        ) / _MARGINS_EPS
    variances = np.diag(jacobian @ covariance @ jacobian.T)
    standard_errors = np.sqrt(np.maximum(variances, 0.0))
    return ames, standard_errors


def _model_frame_contract(
    *,
    lhs: str,
    source_data: pd.DataFrame,
    retained_data: pd.DataFrame,
    rhs: str,
    design_info: patsy.DesignInfo,
    design_column_names: list[str],
    term_labels: list[str],
    intercept: int,
) -> dict[str, Any]:
    factor_columns: list[str] = []
    transformed_values: dict[str, pd.Series] = {}
    for factor_info in design_info.factor_infos.values():
        expression = factor_info.factor.name()
        rendered = _r_expression_name(expression)
        if expression in source_data.columns:
            column_name = expression
        else:
            power_match = re.fullmatch(
                r"I\(\s*([A-Za-z_]\w*)\s*\*\*\s*(\d+)\s*\)",
                expression,
            )
            if power_match is None:
                raise ValueError(f"unsupported formula factor: {rendered}")
            source_name, power = power_match.groups()
            column_name = rendered
            transformed_values[column_name] = retained_data[source_name].astype(float) ** int(power)
        if column_name not in factor_columns:
            factor_columns.append(column_name)

    column_names = [lhs, *factor_columns]
    column_classes: dict[str, list[str]] = {lhs: _r_column_class(retained_data[lhs])}
    factor_levels: dict[str, list[str]] = {}
    for column_name in factor_columns:
        if column_name in retained_data:
            series = retained_data[column_name]
            column_classes[column_name] = _r_column_class(series)
        else:
            series = transformed_values[column_name]
            column_classes[column_name] = ["AsIs"]
    for column_name in column_names:
        series = (
            retained_data[column_name]
            if column_name in retained_data
            else transformed_values[column_name]
        )
        factor_levels[column_name] = (
            [_r_value_string(value) for value in series.cat.categories]
            if isinstance(series.dtype, pd.CategoricalDtype)
            else []
        )

    contrasts: dict[str, str] = {}
    generated_factor_levels: dict[str, list[str]] = {}
    contrast_matrices: dict[str, dict[str, Any]] = {}
    for factor_info in design_info.factor_infos.values():
        expression = factor_info.factor.name()
        if factor_info.type == "categorical" and expression in source_data.columns:
            contrasts[expression] = "contr.treatment"
            levels = [_r_value_string(value) for value in factor_info.categories]
            generated_factor_levels[expression] = levels
            contrast_matrices[expression] = {
                "row_names": levels,
                "column_names": levels[1:],
                "values": np.vstack(
                    [
                        np.zeros((1, max(len(levels) - 1, 0)), dtype=float),
                        np.eye(max(len(levels) - 1, 0), dtype=float),
                    ]
                ).tolist(),
            }

    retained_names = [_r_value_string(value) for value in retained_data.index]
    retained_set = set(retained_data.index)
    return {
        "column_names": column_names,
        "column_classes": column_classes,
        "row_names": retained_names,
        "omitted_row_names": [
            _r_value_string(value) for value in source_data.index if value not in retained_set
        ],
        "response": [_r_value_string(value) for value in retained_data[lhs]],
        "term_labels": term_labels,
        "intercept": intercept,
        "factor_levels": factor_levels,
        "design_column_names": design_column_names,
        "contrasts": contrasts,
        "generated_factor_levels": generated_factor_levels,
        "contrast_matrices": contrast_matrices,
    }


def _fit_naive_rarcat_models(
    formula: str,
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Fit the private Phase 3A one-vs-rest GLM and AME solver contract.

    This is intentionally not a public ``rarcat`` implementation. It covers
    only the formula/model-frame and naive GLM/margins behavior proven against
    the pinned WeightedCluster 2.0 oracle.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one row")
    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")
    if not data.index.is_unique:
        raise ValueError("data index must contain unique row labels")
    _unique_r_strings(list(data.index), label="data index labels")
    source_data = data.copy(deep=True)
    lhs, rhs = _split_formula(formula)
    if lhs not in source_data:
        raise ValueError(f"formula response column {lhs!r} is absent from data")
    translated_rhs = _translate_formula_rhs(rhs)
    design_data = _r_design_data(translated_rhs, source_data)
    # R's `glm` drops incomplete rows under the default `na.omit`, and this
    # solver exists to reproduce that behaviour, so it asks for "drop"
    # explicitly.  The Bayesian path takes the "raise" default instead
    # (finding F-16 of the 2026-08-25 audit).
    (
        design_all,
        design_info,
        ordered_positions,
        design_column_names,
        term_labels,
        intercept,
    ) = _ordered_design(translated_rhs, design_data, na_action="drop")
    design_index = patsy.dmatrix(
        translated_rhs,
        design_data,
        NA_action="drop",
        return_type="dataframe",
        eval_env=-1,
    ).index
    retained_index = design_index[source_data.loc[design_index, lhs].notna()]
    retained_data = source_data.loc[retained_index].copy(deep=True)
    if retained_data.empty:
        raise ValueError("formula model frame has no complete rows")
    design_row_positions = [design_index.get_loc(value) for value in retained_index]
    X = design_all[design_row_positions, :]
    if not np.all(np.isfinite(X)):
        raise ValueError("formula design matrix must contain only finite values")

    cluster_values = list(pd.unique(retained_data[lhs]))
    if len(cluster_values) < 2:
        raise ValueError("formula response must contain at least two observed clusters")
    cluster_names = _unique_r_strings(cluster_values, label="cluster labels")
    effect_specs = _effect_specs(translated_rhs, retained_data)
    design_rank = int(np.linalg.matrix_rank(X))
    if design_rank < X.shape[1]:
        raise ValueError("formula design matrix is rank deficient")
    factor_names = [name for name, _, _ in effect_specs]
    model_frame = _model_frame_contract(
        lhs=lhs,
        source_data=source_data,
        retained_data=retained_data,
        rhs=rhs,
        design_info=design_info,
        design_column_names=design_column_names,
        term_labels=term_labels,
        intercept=intercept,
    )

    clusters: dict[str, Mapping[str, Any]] = {}
    for cluster_value, cluster_name in zip(cluster_values, cluster_names, strict=True):
        membership = (retained_data[lhs].to_numpy() == cluster_value).astype(float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(
                membership,
                X,
                family=sm.families.Binomial(),
            ).fit(
                method="IRLS",
                maxiter=25,
                tol=1e-8,
                atol=1e-9,
                rtol=1e-8,
                tol_criterion="deviance",
            )
        separation = any(isinstance(item.message, PerfectSeparationWarning) for item in caught)
        captured_warnings = (
            [
                "glm.fit: algorithm did not converge",
                "glm.fit: fitted probabilities numerically 0 or 1 occurred",
            ]
            if separation
            else [str(item.message) for item in caught]
        )
        beta = np.asarray(fit.params, dtype=float).reshape(-1)
        covariance = np.asarray(fit.cov_params(), dtype=float)
        covariance_unscaled = np.asarray(
            fit.normalized_cov_params,
            dtype=float,
        )
        fit_history_deviance = [float(value) for value in fit.fit_history["deviance"]]
        fit_history_parameters = [
            (
                None
                if np.isscalar(value) and not np.isfinite(value)
                else np.asarray(value, dtype=float).reshape(-1).copy()
            )
            for value in fit.fit_history["params"]
        ]
        ames, standard_errors = _margins_with_delta_se(
            beta,
            covariance,
            retained_data=retained_data,
            design_info=design_info,
            ordered_positions=ordered_positions,
            effect_specs=effect_specs,
        )
        clusters[cluster_name] = {
            "error": None,
            "warnings": captured_warnings,
            "converged": bool(fit.converged and not separation),
            "nobs": int(fit.nobs),
            "rank": design_rank,
            "df_residual": int(fit.df_resid),
            "iterations": int(fit.fit_history["iteration"]),
            "deviance": float(fit.deviance),
            "null_deviance": float(fit.null_deviance),
            "dispersion": float(fit.scale),
            "convergence_control": {
                "maxiter": 25,
                "atol": 1e-9,
                "rtol": 1e-8,
                "criterion": "deviance",
            },
            "fit_history": {
                "iterations": int(fit.fit_history["iteration"]),
                "deviance": fit_history_deviance,
                "parameters": fit_history_parameters,
            },
            "coefficient_names": list(design_column_names),
            "coefficients": beta.copy(),
            "covariance": covariance.copy(),
            "covariance_unscaled": covariance_unscaled.copy(),
            "covariance_scaled": covariance.copy(),
            "design_column_names": list(design_column_names),
            "ame_factor_names": list(factor_names),
            "ames": ames.copy(),
            "standard_errors": standard_errors.copy(),
        }

    return {
        "formula": formula,
        "lhs": lhs,
        "rhs": rhs,
        "model_frame": model_frame,
        "cluster_names": cluster_names,
        "factor_names": factor_names,
        "clusters": clusters,
    }


def _crossed_reml_state(
    variances: np.ndarray,
    *,
    y: np.ndarray,
    X: np.ndarray,
    covariance_derivatives: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Evaluate the Gaussian crossed-intercept REML criterion and score state.

    For ``V = sum(v_j G_j)``, the negative restricted log likelihood is

    ``0.5 * ((n-p)log(2π) + log|V| + log|X'V^-1X| + y'Py)``.

    Its log-variance score is

    ``0.5 * v_j * (tr(PG_j) - y'PG_jPy)``.

    The three covariance derivatives are the observation incidence product,
    bootstrap incidence product, and source-normalized inverse-weight
    residual diagonal. This deliberately implements no general mixed-model
    surface.
    """
    variances = np.asarray(variances, dtype=float)
    if variances.shape != (3,):
        raise ValueError("REML variances must contain exactly three components")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        raise ValueError("REML variances must be finite and non-negative")
    if not np.any(variances > 0.0):
        raise ValueError("REML variances must define a positive covariance")
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("REML y must be a finite one-dimensional non-empty vector")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or not np.all(np.isfinite(X)):
        raise ValueError("REML X must be a finite two-dimensional matrix with fixed effects")
    if X.shape[0] != y.size:
        raise ValueError("REML X rows must match y")
    if y.size <= X.shape[1]:
        raise ValueError("REML requires more observations than fixed effects")
    if len(covariance_derivatives) != 3:
        raise ValueError("REML requires exactly three covariance derivatives")
    resolved_derivatives = tuple(
        np.asarray(derivative, dtype=float) for derivative in covariance_derivatives
    )
    if any(derivative.shape != (y.size, y.size) for derivative in resolved_derivatives):
        raise ValueError("REML covariance derivatives must each have shape (n, n)")
    if any(not np.all(np.isfinite(derivative)) for derivative in resolved_derivatives):
        raise ValueError("REML covariance derivatives must contain only finite values")
    covariance_derivatives = resolved_derivatives
    covariance = sum(
        variance * derivative
        for variance, derivative in zip(
            variances,
            covariance_derivatives,
            strict=True,
        )
    )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("REML covariance must contain only finite values")
    try:
        factor = cho_factor(covariance, lower=True, check_finite=False)
        inverse = cho_solve(
            factor,
            np.eye(covariance.shape[0]),
            check_finite=False,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("REML covariance must be numerically positive definite") from error
    inverse_X = inverse @ X
    fixed_information = X.T @ inverse_X
    try:
        fixed_covariance = np.linalg.inv(fixed_information)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "REML fixed-effect information must be numerically positive definite"
        ) from error
    fixed_effect = fixed_covariance @ X.T @ inverse @ y
    residual = y - X @ fixed_effect
    projected_y = inverse @ residual
    projection = inverse - inverse_X @ fixed_covariance @ inverse_X.T
    logdet_covariance = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    fixed_sign, logdet_fixed = np.linalg.slogdet(fixed_information)
    if fixed_sign <= 0.0:
        raise ValueError("REML fixed-effect information must be numerically positive definite")
    quadratic = float(residual @ projected_y)
    degrees_of_freedom = y.size - X.shape[1]
    objective = 0.5 * (
        degrees_of_freedom * np.log(2.0 * np.pi) + logdet_covariance + logdet_fixed + quadratic
    )
    gradient = 0.5 * np.asarray(
        [
            variance
            * (
                float(np.sum(projection * derivative.T))
                - float(projected_y @ derivative @ projected_y)
            )
            for variance, derivative in zip(
                variances,
                covariance_derivatives,
                strict=True,
            )
        ],
        dtype=float,
    )
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise ValueError("REML state must have a finite objective and analytic score")
    return {
        "objective": float(objective),
        "gradient": gradient,
        "variances": variances,
        "covariance": covariance,
        "fixed_information": fixed_information,
        "fixed_covariance": fixed_covariance,
        "fixed_effect": fixed_effect,
        "projected_y": projected_y,
    }


def _crossed_reml_objective(
    log_variances: np.ndarray,
    *,
    y: np.ndarray,
    X: np.ndarray,
    covariance_derivatives: tuple[np.ndarray, np.ndarray, np.ndarray],
    return_state: bool = False,
) -> Any:
    if not isinstance(return_state, (bool, np.bool_)):
        raise TypeError("return_state must be a boolean")
    log_variances = np.asarray(log_variances, dtype=float)
    if log_variances.shape != (3,):
        raise ValueError("REML log-variances must contain exactly three components")
    if not np.all(np.isfinite(log_variances)):
        raise ValueError("finite REML log-variances are required")
    representable_min = np.log(np.nextafter(0.0, 1.0))
    representable_max = np.nextafter(
        np.log(np.finfo(float).max),
        -np.inf,
    )
    if np.any(log_variances < representable_min) or np.any(log_variances > representable_max):
        raise ValueError("log-variances exceed the representable REML variance domain")
    variances = np.exp(log_variances)
    state = _crossed_reml_state(
        variances,
        y=y,
        X=X,
        covariance_derivatives=covariance_derivatives,
    )
    if return_state:
        return state
    return state["objective"], state["gradient"]


def _fit_weighted_crossed_pool(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit RARCAT's private weighted crossed random-intercept REML model.

    Rows with missing AMEs are removed exactly where the installed R source
    removes them. Standard errors become normalized inverse-variance prior
    weights. The solver is intentionally restricted to one fixed intercept and
    crossed random intercepts for the ``id`` and ``bootstrap`` columns.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = ("id", "bootstrap", "ame", "sterror")
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        raise ValueError("frame must contain id, bootstrap, ame, and sterror")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    if not frame.columns.is_unique:
        raise ValueError("frame columns must be unique")
    if not frame.index.is_unique:
        raise ValueError("frame index must contain unique row labels")

    source = frame.loc[:, required].copy(deep=True)
    source_row_names = _unique_r_strings(
        list(source.index),
        label="frame index labels",
    )
    included = source.loc[~pd.isna(source["ame"])].copy(deep=True)
    included_row_names = [_r_value_string(value) for value in included.index]
    included_set = set(included.index)
    omitted_row_names = [
        _r_value_string(value) for value in source.index if value not in included_set
    ]
    if included.empty:
        raise ValueError("crossed pool has no non-missing AME rows")
    if included["id"].isna().any() or included["bootstrap"].isna().any():
        raise ValueError("id and bootstrap must not contain missing values")

    y = included["ame"].to_numpy(dtype=float)
    standard_errors = included["sterror"].to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("included AMEs must contain only finite values")
    if not np.all(np.isfinite(standard_errors)):
        raise ValueError("included standard errors must contain only finite values")
    if np.any(standard_errors == 0.0):
        raise ValueError("included standard errors must be nonzero")
    absolute_standard_errors = np.abs(standard_errors)
    safe_minimum = np.sqrt(
        included.shape[0] / np.finfo(float).max,
    )
    safe_maximum = np.sqrt(np.finfo(float).max)
    if np.any(absolute_standard_errors < safe_minimum) or np.any(
        absolute_standard_errors > safe_maximum
    ):
        raise ValueError(
            "included standard errors exceed the safe inverse-variance magnitude domain"
        )
    inverse_variance_weights = 1.0 / np.square(standard_errors)
    normalized_weights = inverse_variance_weights / np.mean(inverse_variance_weights)
    if not np.all(np.isfinite(normalized_weights)) or np.any(normalized_weights <= 0.0):
        raise ValueError("normalized inverse-variance weights must be positive and finite")

    id_names, observation_incidence = _r_group_incidence(
        included["id"],
        label="id labels",
    )
    bootstrap_names, bootstrap_incidence = _r_group_incidence(
        included["bootstrap"],
        label="bootstrap labels",
    )
    if len(id_names) < 2 or len(bootstrap_names) < 2:
        raise ValueError("crossed pool requires at least two observation and bootstrap groups")
    X = np.ones((included.shape[0], 1), dtype=float)
    covariance_derivatives = (
        observation_incidence @ observation_incidence.T,
        bootstrap_incidence @ bootstrap_incidence.T,
        np.diag(1.0 / normalized_weights),
    )

    total_variance = max(float(np.var(y, ddof=1)), np.finfo(float).eps)
    initial_variances = np.full(3, total_variance / 3.0, dtype=float)
    # Backward-stable Cholesky requires perturbations to remain well above
    # n*eps. A symmetric sqrt(n*eps) component floor therefore bounds the
    # largest-to-smallest candidate variance ratio by 1/(n*eps), while still
    # allowing a lower-face component to represent a structural zero during
    # the final singular-boundary refactorization.
    relative_variance_floor = np.sqrt(included.shape[0] * np.finfo(float).eps)
    log_variance_radius = -float(np.log(relative_variance_floor))
    log_variance_center = float(np.log(total_variance))
    log_variance_bounds = np.asarray(
        [
            log_variance_center - log_variance_radius,
            log_variance_center + log_variance_radius,
        ],
        dtype=float,
    )
    initial_log_variances = np.log(initial_variances)
    initial_parameters = np.arctanh(
        (initial_log_variances - log_variance_center) / log_variance_radius
    )

    def bounded_log_variances(parameters: np.ndarray) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
            raise ValueError("finite bounded REML parameters are required")
        return log_variance_center + log_variance_radius * np.tanh(parameters)

    def objective(parameters: np.ndarray):
        transformed = np.tanh(parameters)
        objective_value, log_gradient = _crossed_reml_objective(
            log_variance_center + log_variance_radius * transformed,
            y=y,
            X=X,
            covariance_derivatives=covariance_derivatives,
        )
        parameter_gradient = log_gradient * log_variance_radius * (1.0 - np.square(transformed))
        return objective_value, parameter_gradient

    optimization = minimize(
        objective,
        initial_parameters,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": 1_000},
    )
    final_log_variances = bounded_log_variances(optimization.x)
    state = _crossed_reml_objective(
        final_log_variances,
        y=y,
        X=X,
        covariance_derivatives=covariance_derivatives,
        return_state=True,
    )
    gradient_norm = float(np.max(np.abs(state["gradient"])))
    normalized_log_positions = (final_log_variances - log_variance_center) / log_variance_radius
    active_lower_bounds = np.isclose(
        normalized_log_positions,
        -1.0,
        rtol=0.0,
        atol=1e-8,
    )
    active_upper_bounds = np.isclose(
        normalized_log_positions,
        1.0,
        rtol=0.0,
        atol=1e-8,
    )
    active_bounds = active_lower_bounds | active_upper_bounds
    optimizer_converged = bool(gradient_norm <= 1e-7 and not np.any(active_bounds))
    variances = np.asarray(state["variances"], dtype=float)
    residual_variance = float(variances[2])
    theta = np.sqrt(variances[:2] / residual_variance)
    singular_components = (theta < 1e-4) | active_lower_bounds[:2]
    singular = bool(np.any(singular_components))
    if singular:
        variances = variances.copy()
        variances[:2][singular_components] = 0.0
        state = _crossed_reml_state(
            variances,
            y=y,
            X=X,
            covariance_derivatives=covariance_derivatives,
        )
        theta = theta.copy()
        theta[singular_components] = 0.0
    if singular:
        optimizer_code = 0
        convergence_messages = ["boundary (singular) fit: see help('isSingular')"]
        converged = False
    elif optimizer_converged:
        optimizer_code = 0
        convergence_messages = []
        converged = True
    else:
        optimizer_code = int(optimization.status or 1)
        convergence_messages = [
            "REML analytic score did not converge " f"(max|gradient| = {gradient_norm:.6g})"
        ]
        if np.any(active_bounds):
            convergence_messages.append(
                "REML optimum reached the machine-precision variance boundary"
            )
        converged = False
    fixed_intercept = float(state["fixed_effect"][0])
    fixed_standard_error = float(np.sqrt(state["fixed_covariance"][0, 0]))
    projected_y = np.asarray(state["projected_y"], dtype=float)
    observation_random_effects = float(variances[0]) * observation_incidence.T @ projected_y
    bootstrap_random_effects = float(variances[1]) * bootstrap_incidence.T @ projected_y

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
        "fixed_intercept": fixed_intercept,
        "fixed_standard_error": fixed_standard_error,
        "observation_variance": float(variances[0]),
        "bootstrap_variance": float(variances[1]),
        "residual_variance": residual_variance,
        "observation_random_effect_names": id_names,
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
        "convergence_messages": convergence_messages,
        "converged": converged,
        "singular": singular,
        "reml": True,
        "log_likelihood": -float(state["objective"]),
        "reml_criterion": 2.0 * float(state["objective"]),
        "theta": theta.copy(),
        "optimizer": {
            "method": "BFGS",
            "iterations": int(optimization.nit),
            "gradient_max_abs": gradient_norm,
            "bounded_domain": True,
            "log_variance_bounds": log_variance_bounds.copy(),
            "active_bound_components": [
                name
                for name, active in zip(
                    ("observation", "bootstrap", "residual"),
                    active_bounds,
                    strict=True,
                )
                if active
            ],
            "raw_status": int(optimization.status),
            "raw_success": bool(optimization.success),
            "raw_message": str(optimization.message),
        },
    }

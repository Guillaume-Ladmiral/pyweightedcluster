"""Private conditional categorical posteriors for Bayesian RARCAT B3.

This module implements only ``p(beta_u | z_u, X)`` for the exact distinct
aligned partitions supplied by B2.  Partition masses are retained as
provenance and are never used in a conditional likelihood or effect average.
Mixture integration and the public result/API belong to B4 and B5.
"""

from __future__ import annotations

import logging
import operator
import re
import time
import warnings as warnings_module
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import patsy

from ._bayesian_rarcat_design import (
    _base_predictors,
    _build_from_design_info,
    _ordered_design,
    _r_design_data,
    _split_formula,
    _translate_formula_rhs,
)
from ._bayesian_rarcat_partition import PartitionPosteriorResult

_FLOAT_ATOL = 1e-12
_FLOAT_RTOL = 1e-12
_POWER_PATTERN = re.compile(r"I\(\s*([A-Za-z_]\w*)\s*(?:\^|\*\*)\s*([^\s()]+)\s*\)")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_]\w)(?:\d+(?:\.\d*)?|\.\d+)(?!\w)")


@dataclass(frozen=True)
class ScalingTransform:
    """Observed-data population scaling for one raw continuous predictor."""

    predictor: str
    raw_values: np.ndarray
    mean: float
    population_sd: float


@dataclass(frozen=True)
class PreparedConditionalDesign:
    """Validated common B3 model frame and design provenance.

    ``design_matrix`` is read-only float64 ``(n, P)``. Object identities and
    labels are retained without stringification. ``design_info`` and
    ``ordered_positions`` rebuild exact counterfactual matrices later.
    """

    formula: str
    response_name: str
    translated_rhs: str
    row_ids: tuple[object, ...]
    cluster_order: tuple[object, ...]
    reference_category: object
    base_predictors: tuple[str, ...]
    continuous_predictors: tuple[str, ...]
    categorical_predictors: tuple[str, ...]
    nonlinear_continuous_predictors: tuple[str, ...]
    counterfactual_steps: tuple[tuple[str, float], ...]
    factor_levels: tuple[tuple[str, tuple[object, ...]], ...]
    factor_references: tuple[tuple[str, object], ...]
    term_labels: tuple[str, ...]
    engine_design_labels: tuple[str, ...]
    coefficient_coordinates: tuple[str, ...]
    engine_to_coefficient: tuple[tuple[str, str], ...]
    design_rank: int
    design_matrix: np.ndarray
    response_codes: np.ndarray
    scaling_transforms: tuple[ScalingTransform, ...]
    observed_model_frame: pd.DataFrame
    scaled_design_frame: pd.DataFrame
    design_info: Any
    ordered_positions: tuple[int, ...]


@dataclass(frozen=True)
class PriorSpecification:
    """Realized complete prior tensor in exact model-coordinate order."""

    source: str
    coefficient_coordinates: tuple[str, ...]
    nonreference_cluster_coordinates: tuple[object, ...]
    mean: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class EffectMetadata:
    """Complete estimand and coordinate provenance for one draw-wise AME."""

    effect_identifier: str
    source_variable: str
    estimand_kind: str
    method: str
    original_scale_units: str
    factor_reference: object | None
    factor_comparison: object | None
    half_step: float | None
    evaluated_offsets: tuple[float, float] | None
    denominator: float | None
    averaging_population: str
    formula: str
    term_labels: tuple[str, ...]
    coefficient_coordinates: tuple[str, ...]
    row_ids: tuple[object, ...]
    component_identifier: str
    chain_coordinates: tuple[int, ...]
    draw_coordinates: tuple[int, ...]
    cluster_order: tuple[object, ...]
    zero_sum_atol: float
    zero_sum_rtol: float


@dataclass(frozen=True)
class DrawwiseEffect:
    """Observation-level and equally averaged posterior-draw effects."""

    metadata: EffectMetadata
    observation_effects: np.ndarray
    ame_draws: np.ndarray


@dataclass(frozen=True)
class SamplerConfiguration:
    chains: int
    tune: int
    draws: int
    cores: int
    blas_cores: int
    target_accept: float
    nuts_sampler: str


SAMPLER_CONFIGURATION = SamplerConfiguration(
    chains=4,
    tune=500,
    draws=500,
    cores=1,
    blas_cores=1,
    target_accept=0.90,
    nuts_sampler="pymc",
)


@dataclass(frozen=True)
class B3SeedIdentifier:
    name: str
    entropy: int
    spawn_key: tuple[int, ...]
    pool_size: int
    generated_seed: int

    def seed_sequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            entropy=self.entropy,
            spawn_key=self.spawn_key,
            pool_size=self.pool_size,
        )


@dataclass(frozen=True)
class ComponentSeedManifest:
    component_index: int
    name: str
    chain_streams: tuple[B3SeedIdentifier, ...]
    posterior_predictive: B3SeedIdentifier

    @property
    def chain_seeds(self) -> tuple[int, ...]:
        return tuple(stream.generated_seed for stream in self.chain_streams)


@dataclass(frozen=True)
class B3SeedManifest:
    root_seed: int
    prior_predictive: B3SeedIdentifier
    components: tuple[ComponentSeedManifest, ...]


@dataclass(frozen=True)
class DiagnosticThresholds:
    rhat_exclusive_max: float = 1.01
    bulk_ess_minimum: int = 400
    tail_ess_minimum: int = 400
    divergences_maximum: int = 0
    bfmi_each_chain_minimum: float = 0.30
    mean_change_in_sd_maximum: float = 0.10
    mcse_fraction_of_sd_maximum: float = 0.05
    predictive_count_fraction_error_maximum: float = 0.20


DIAGNOSTIC_THRESHOLDS = DiagnosticThresholds()

# The logger PyMC reports sampler diagnostics through.  Its descendants
# propagate to it, so one handler here observes every convergence message.
_SAMPLER_LOGGER_NAME = "pymc"

# Explicit allowlist of material warning vocabulary, as
# ``SCIENTIFIC_GOVERNANCE.md`` "Diagnostic policy" requires: each term names a
# diagnostic that bears on whether the fitted posterior can support the
# declared inference, and nothing here is a blanket "any warning fails" check.
# The sampler terms are taken from the texts PyMC 6.0.1 itself constructs in
# ``pymc/stats/convergence.py``; matching is case-folded substring containment.
MATERIAL_WARNING_TERMS: tuple[str, ...] = (
    # Divergences: "There were 10 divergences after tuning."  "divergent" is
    # retained because it is the adjectival form used in older texts.
    "divergence",
    "divergent",
    # Tree depth: "Chain 0 reached the maximum tree depth."
    "maximum tree depth",
    "max_treedepth",
    # Rank-normalized R-hat: "The rhat statistic is larger than 1.01 ..."
    "rhat statistic",
    "r-hat statistic",
    # Effective sample size: "The effective sample size per chain is smaller..."
    "effective sample size",
    # Arithmetic failures that make individual draws uninterpretable.  No
    # separate numeric gate observes these, so the warning channel is the only
    # place they can be caught.
    "nonfinite",
    "non-finite",
    "invalid value encountered",
    "overflow encountered",
    "divide by zero encountered",
)


@dataclass(frozen=True)
class SamplerDiagnosticRecord:
    status: str
    all_finite: bool
    max_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    divergences: int
    bfmi_by_chain: tuple[float, ...]
    minimum_bfmi: float
    reference_zero: bool
    probability_simplex: bool
    failure_reasons: tuple[str, ...]
    thresholds: DiagnosticThresholds


@dataclass(frozen=True)
class GrowthDiagnostic:
    status: str
    posterior_sd: float
    half_mean: float
    full_mean: float
    mean_change_in_sd: float
    mcse_mean: float
    mcse_fraction_of_sd: float
    constant_nonstructural: bool
    failure_reasons: tuple[str, ...]
    thresholds: DiagnosticThresholds


@dataclass(frozen=True)
class WarningRecord:
    message: str
    classification: str
    material: bool


@dataclass(frozen=True)
class PredictiveCoordinates:
    chain: tuple[object, ...]
    draw: tuple[object, ...]
    observation: tuple[object, ...]
    cluster: tuple[object, ...]


@dataclass(frozen=True)
class PosteriorPredictiveRecord:
    status: str
    seed: int
    observed_counts: tuple[tuple[object, int], ...] = ()
    mean_replicated_counts: tuple[tuple[object, float], ...] = ()
    maximum_count_fraction_error: float = float("nan")
    cluster_order: tuple[object, ...] = ()
    coordinates: PredictiveCoordinates | None = None
    warnings: tuple[WarningRecord, ...] = ()
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorPredictiveRecord:
    status: str
    seed: int
    predictive: Any = None
    finite_probability: bool = False
    probability_simplex: bool = False
    mean_cluster_counts: tuple[tuple[object, float], ...] = ()
    cluster_order: tuple[object, ...] = ()
    coordinates: PredictiveCoordinates | None = None
    warnings: tuple[WarningRecord, ...] = ()
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetSummary:
    target_identifier: str
    posterior_mean: float
    posterior_sd: float
    interval_94: tuple[float, float]
    probability_above_zero: float
    probability_below_zero: float
    mcse_mean: float
    bulk_ess: float
    tail_ess: float
    rhat: float
    growth: GrowthDiagnostic
    constant_nonstructural: bool


@dataclass(frozen=True)
class ComponentFitRecord:
    component_index: int
    component_identifier: str
    partition: np.ndarray
    count: int
    mass: float
    mass_mcse: float
    seeds: ComponentSeedManifest
    status: str
    posterior: Any = None
    posterior_predictive: Any = None
    posterior_predictive_diagnostics: PosteriorPredictiveRecord | None = None
    sampler_diagnostics: SamplerDiagnosticRecord | None = None
    coefficient_summaries: tuple[TargetSummary, ...] = ()
    effects: tuple[DrawwiseEffect, ...] = ()
    effect_summaries: tuple[TargetSummary, ...] = ()
    timing_seconds: float = 0.0
    warnings: tuple[WarningRecord, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    exception_type: str | None = None
    exception_message: str | None = None


@dataclass(frozen=True)
class ConditionalPosteriorResult:
    status: str
    n: int
    B: int
    K: int
    U: int
    observation_ids: np.ndarray
    reference_labels: np.ndarray
    cluster_order: tuple[object, ...]
    distinct_partitions: np.ndarray
    counts: np.ndarray
    masses: np.ndarray
    mass_mcse: np.ndarray
    root_seed: int
    seed_manifest: B3SeedManifest
    formula: str
    design: PreparedConditionalDesign
    prior: PriorSpecification
    sampler_configuration: SamplerConfiguration
    software_versions: tuple[tuple[str, str], ...]
    prior_predictive: PriorPredictiveRecord
    components: tuple[ComponentFitRecord, ...]
    failures: tuple[ComponentFitRecord, ...]
    mean_component_time_seconds: float
    projected_serial_time_u16_seconds: float


def _readonly_array(values: object, *, dtype: object | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _exact_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer and cannot be boolean")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an exact integer") from error
    integer = int(integer)
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


_BAYES_RARCAT_ENVELOPE = {
    "n": (2, 12),
    "K": (2, 3),
    "B": (1, 128),
    "U": (1, 16),
    "design_width": (1, 4),
}
_BAYES_RARCAT_MAX_U_OVER_B = 0.25


def validate_private_envelope(
    *, n: object, K: object, B: object, U: object, design_width: object
) -> tuple[int, int, int, int, int]:
    """Validate the B1-measured private B3 operating envelope."""
    supplied = {"n": n, "K": K, "B": B, "U": U, "design_width": design_width}
    validated = {
        name: _exact_integer(value, name=name, minimum=_BAYES_RARCAT_ENVELOPE[name][0])
        for name, value in supplied.items()
    }
    n_int = validated["n"]
    k_int = validated["K"]
    b_int = validated["B"]
    u_int = validated["U"]
    width_int = validated["design_width"]
    for name, actual in validated.items():
        maximum = _BAYES_RARCAT_ENVELOPE[name][1]
        if actual > maximum:
            raise ValueError(f"{name}={actual} exceeds the private maximum {maximum}")
    if k_int > n_int:
        raise ValueError("K cannot exceed n")
    if u_int > b_int:
        raise ValueError("U cannot exceed B")
    if u_int / b_int > _BAYES_RARCAT_MAX_U_OVER_B:
        raise ValueError(f"U / B exceeds the private maximum {_BAYES_RARCAT_MAX_U_OVER_B}")
    return n_int, k_int, b_int, u_int, width_int


def _values_equal(left: Sequence[object], right: Sequence[object]) -> bool:
    if len(left) != len(right):
        return False
    return all(a == b for a, b in zip(left, right, strict=True))


def _stable_distinct_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: dict[tuple[object, ...], int] = {}
    rows: list[list[object]] = []
    counts: list[int] = []
    inverse: list[int] = []
    for row in np.asarray(matrix, dtype=object):
        key = tuple(row.tolist())
        position = positions.get(key)
        if position is None:
            position = len(rows)
            positions[key] = position
            rows.append(row.tolist())
            counts.append(0)
        counts[position] += 1
        inverse.append(position)
    return (
        np.asarray(rows, dtype=object),
        np.asarray(counts, dtype=np.int64),
        np.asarray(inverse, dtype=np.int64),
    )


def validate_partition_posterior(result: object) -> tuple[int, int, int, int]:
    """Validate the complete B2 identity/compression boundary without sampling."""
    if not isinstance(result, PartitionPosteriorResult):
        raise TypeError("partition_posterior must be a PartitionPosteriorResult")
    n = _exact_integer(result.n, name="n", minimum=2)
    B = _exact_integer(result.B, name="B", minimum=1)
    K = _exact_integer(result.K, name="K", minimum=2)
    order = tuple(result.reference_cluster_order)
    if len(order) != K or len(set(order)) != K:
        raise ValueError("reference cluster order must contain exactly K unique labels")
    shapes = {
        "observation_ids": (result.observation_ids, (n,)),
        "reference_labels": (result.reference_labels, (n,)),
        "weights": (result.weights, (B, n)),
        "raw_partitions": (result.raw_partitions, (B, n)),
        "aligned_partitions": (result.aligned_partitions, (B, n)),
        "membership_probabilities": (result.membership_probabilities, (n, K)),
        "entropy": (result.entropy, (n,)),
        "co_clustering": (result.co_clustering, (n, n)),
    }
    for name, (values, expected) in shapes.items():
        if np.shape(values) != expected:
            raise ValueError(f"partition_posterior {name} must have shape {expected}")
    if not pd.Index(result.observation_ids).is_unique:
        raise ValueError("partition_posterior observation identifiers must be unique")
    if not _values_equal(pd.unique(result.reference_labels).tolist(), list(order)):
        raise ValueError("reference labels do not preserve the reference cluster order")
    aligned = np.asarray(result.aligned_partitions, dtype=object)
    order_set = set(order)
    for index, row in enumerate(aligned):
        if set(row.tolist()) != order_set:
            raise ValueError(
                f"aligned partition {index} must contain every cluster in cluster order"
            )
    distinct, counts, inverse = _stable_distinct_rows(aligned)
    U = len(distinct)
    if np.shape(result.distinct_partitions) != (U, n) or not np.array_equal(
        result.distinct_partitions, distinct
    ):
        raise ValueError("distinct partitions are not the exact stable B2 compression")
    if np.shape(result.counts) != (U,) or not np.array_equal(result.counts, counts):
        raise ValueError("counts do not match the exact distinct partitions")
    if np.shape(result.inverse_index) != (B,) or not np.array_equal(result.inverse_index, inverse):
        raise ValueError("inverse index does not reconstruct the aligned partitions")
    expected_masses = counts.astype(float) / float(B)
    if np.shape(result.masses) != (U,) or not np.allclose(
        result.masses, expected_masses, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL
    ):
        raise ValueError("masses do not equal B2 counts divided by B")
    expected_mcse = np.sqrt(expected_masses * (1.0 - expected_masses) / float(B))
    if np.shape(result.mass_mcse) != (U,) or not np.allclose(
        result.mass_mcse, expected_mcse, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL
    ):
        raise ValueError("mass MCSE values do not match the B2 compression")
    return n, B, K, U


def _validated_formula(formula: object) -> tuple[str, str, str]:
    response, rhs = _split_formula(formula)  # type: ignore[arg-type]
    if "|" in rhs:
        raise ValueError("unsupported group-specific | formula term")
    if re.search(r"(?<![A-Za-z0-9_])\.(?![A-Za-z0-9_])", rhs):
        raise ValueError("unsupported formula . expansion")
    calls = re.findall(r"I\([^)]*\)", rhs)
    for call in calls:
        match = _POWER_PATTERN.fullmatch(call)
        if match is None:
            raise ValueError("formula I() supports only positive-integer raw-predictor powers")
        exponent_text = match.group(2)
        if not exponent_text.isdigit() or int(exponent_text) < 1:
            raise ValueError("formula power exponent must be a positive integer")
    without_powers = _POWER_PATTERN.sub("", rhs)
    constants = _NUMBER_PATTERN.findall(without_powers)
    if any(float(value) not in (0.0, 1.0) for value in constants):
        raise ValueError("unsupported numeric constant in formula")
    return response, rhs, _translate_formula_rhs(rhs)


def _categorical_levels(series: pd.Series, *, name: str) -> tuple[object, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        levels = tuple(series.cat.categories.tolist())
        observed = list(pd.unique(series.dropna()))
        if len(observed) != len(levels) or set(observed) != set(levels):
            raise ValueError(f"categorical predictor {name!r} has unused declared levels")
        return levels
    if pd.api.types.is_bool_dtype(series.dtype):
        observed = set(bool(value) for value in pd.unique(series.dropna()))
        return tuple(value for value in (False, True) if value in observed)
    design_data = _r_design_data(name, pd.DataFrame({name: series}))
    converted = design_data[name]
    if isinstance(converted.dtype, pd.CategoricalDtype):
        return tuple(converted.cat.categories.tolist())
    return tuple(pd.unique(series.dropna()).tolist())


def _validate_model_frame_values(frame: pd.DataFrame) -> None:
    if frame.isna().any(axis=None):
        raise ValueError("referenced model-frame values must be complete")
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series.dtype):
            if not np.isfinite(np.asarray(series, dtype=float)).all():
                raise ValueError(f"numeric model-frame predictor {name!r} must be finite")
            continue
        for value in series.astype(object).tolist():
            if isinstance(value, (int, float, complex, np.number)) and not isinstance(
                value, (bool, np.bool_)
            ):
                if not np.isfinite(value):
                    raise ValueError(f"numeric model-frame value in {name!r} must be finite")


def _nonlinear_continuous_predictors(
    translated_rhs: str, continuous: tuple[str, ...]
) -> tuple[str, ...]:
    description = patsy.ModelDesc.from_formula("~ " + translated_rhs)
    nonlinear: set[str] = set()
    for term in description.rhs_termlist:
        factors = tuple(factor.name() for factor in term.factors)
        for name in continuous:
            direct = name in factors
            transformed = any(
                factor != name and re.search(rf"\b{re.escape(name)}\b", factor)
                for factor in factors
            )
            if transformed or (direct and len(factors) > 1):
                nonlinear.add(name)
    return tuple(name for name in continuous if name in nonlinear)


def _validated_steps(
    steps: Mapping[str, object], required: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    if not isinstance(steps, Mapping):
        raise TypeError("counterfactual_steps must be a mapping")
    keys = tuple(steps)
    if set(keys) != set(required) or len(keys) != len(required):
        missing = [name for name in required if name not in steps]
        extra = [name for name in keys if name not in required]
        raise ValueError(
            "counterfactual_steps must contain exactly nonlinear continuous predictors; "
            f"missing={missing}, extra={extra}"
        )
    result: list[tuple[str, float]] = []
    for name in required:
        value = steps[name]
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"counterfactual step for {name!r} cannot be boolean")
        try:
            step = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"counterfactual step for {name!r} must be positive finite") from error
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"counterfactual step for {name!r} must be positive finite")
        result.append((name, step))
    return tuple(result)


def prepare_conditional_design(
    *,
    partition_posterior: PartitionPosteriorResult,
    formula: str,
    data: pd.DataFrame,
    continuous_predictors: Sequence[str],
    counterfactual_steps: Mapping[str, object],
) -> PreparedConditionalDesign:
    """Prepare the common B3 design after validating the B2/data boundary."""
    n, B, K, U = validate_partition_posterior(partition_posterior)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not data.index.is_unique:
        raise ValueError("data index must be unique")
    if not _values_equal(data.index.tolist(), partition_posterior.observation_ids.tolist()):
        raise ValueError(
            "data index must be elementwise identical to partition_posterior observation IDs"
        )
    response, _, translated_rhs = _validated_formula(formula)
    if response not in data.columns:
        raise ValueError(f"formula response {response!r} is absent from data")
    response_series = data[response]
    if (
        not isinstance(response_series.dtype, pd.CategoricalDtype)
        or not response_series.cat.ordered
    ):
        raise ValueError("formula response must be an ordered categorical")
    response_categories = tuple(response_series.cat.categories.tolist())
    if not _values_equal(response_categories, partition_posterior.reference_cluster_order):
        raise ValueError("response categories must equal the B2 reference cluster order")
    if not _values_equal(
        response_series.astype(object).tolist(), partition_posterior.reference_labels.tolist()
    ):
        raise ValueError("response values must be elementwise identical to B2 reference labels")

    base_predictors = tuple(_base_predictors(translated_rhs, data))
    if not base_predictors and translated_rhs.strip() not in {"1", "0", "-1"}:
        raise ValueError("formula must reference supported predictor columns")
    missing_identifiers = [
        name
        for name in re.findall(r"\b[A-Za-z_]\w*\b", translated_rhs)
        if name != "I" and name not in data.columns
    ]
    if missing_identifiers:
        raise ValueError(f"formula references absent predictor {missing_identifiers[0]!r}")
    observed_frame = data[[response, *base_predictors]].copy(deep=True)
    _validate_model_frame_values(observed_frame)

    if isinstance(continuous_predictors, (str, bytes)):
        raise TypeError("continuous_predictors must be an ordered sequence of names")
    continuous = tuple(continuous_predictors)
    if any(not isinstance(name, str) for name in continuous):
        raise TypeError("continuous_predictors must contain column names")
    if len(set(continuous)) != len(continuous):
        raise ValueError("continuous predictors must be listed exactly once")
    unused = [name for name in continuous if name not in base_predictors]
    if unused:
        raise ValueError(f"continuous_predictors contains unused name {unused[0]!r}")
    numeric = tuple(
        name
        for name in base_predictors
        if pd.api.types.is_numeric_dtype(observed_frame[name].dtype)
        and not pd.api.types.is_bool_dtype(observed_frame[name].dtype)
        and not isinstance(observed_frame[name].dtype, pd.CategoricalDtype)
    )
    categorical = tuple(name for name in base_predictors if name not in numeric)
    categorical_in_continuous = [name for name in continuous if name in categorical]
    if categorical_in_continuous:
        raise TypeError(
            f"categorical predictor {categorical_in_continuous[0]!r} cannot be continuous"
        )
    if continuous != numeric:
        raise ValueError(
            "continuous_predictors must name every referenced raw numeric predictor in formula order"
        )

    factor_levels = tuple(
        (name, _categorical_levels(observed_frame[name], name=name)) for name in categorical
    )
    if any(len(levels) < 2 for _, levels in factor_levels):
        raise ValueError("constant categorical predictors are unsupported")
    factor_references = tuple((name, levels[0]) for name, levels in factor_levels)
    nonlinear = _nonlinear_continuous_predictors(translated_rhs, continuous)
    steps = _validated_steps(counterfactual_steps, nonlinear)

    scaled_frame = observed_frame.copy(deep=True)
    transforms: list[ScalingTransform] = []
    for name in continuous:
        raw = np.asarray(observed_frame[name], dtype=float)
        mean = float(np.mean(raw))
        population_sd = float(np.std(raw, ddof=0))
        constant_tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(raw))))
        if not np.isfinite(population_sd) or population_sd <= constant_tolerance:
            raise ValueError(f"constant continuous predictor is unsupported: {name}")
        scaled_frame[name] = (raw - mean) / population_sd
        transforms.append(
            ScalingTransform(
                predictor=name,
                raw_values=_readonly_array(raw, dtype=np.float64),
                mean=mean,
                population_sd=population_sd,
            )
        )
    design_frame = _r_design_data(translated_rhs, scaled_frame)
    design, design_info, ordered_positions, coordinates, term_labels, _ = _ordered_design(
        translated_rhs, design_frame
    )
    engine_labels = tuple(design_info.column_names[position] for position in ordered_positions)
    coefficient_coordinates = tuple(coordinates)
    for index, coordinate in enumerate(coefficient_coordinates):
        if coordinate != "(Intercept)" and np.ptp(design[:, index]) <= np.finfo(float).eps:
            raise ValueError(f"constant predictor/design column is unsupported: {coordinate}")
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError("aliased or rank-deficient design columns are unsupported")
    validate_private_envelope(n=n, K=K, B=B, U=U, design_width=design.shape[1])
    cluster_positions = {label: index for index, label in enumerate(response_categories)}
    response_codes = np.asarray(
        [cluster_positions[value] for value in response_series.astype(object)], dtype=np.int64
    )
    return PreparedConditionalDesign(
        formula=formula,
        response_name=response,
        translated_rhs=translated_rhs,
        row_ids=tuple(data.index.tolist()),
        cluster_order=response_categories,
        reference_category=response_categories[0],
        base_predictors=base_predictors,
        continuous_predictors=continuous,
        categorical_predictors=categorical,
        nonlinear_continuous_predictors=nonlinear,
        counterfactual_steps=steps,
        factor_levels=factor_levels,
        factor_references=factor_references,
        term_labels=tuple(term_labels),
        engine_design_labels=engine_labels,
        coefficient_coordinates=coefficient_coordinates,
        engine_to_coefficient=tuple(zip(engine_labels, coefficient_coordinates, strict=True)),
        design_rank=rank,
        design_matrix=_readonly_array(design, dtype=np.float64),
        response_codes=_readonly_array(response_codes, dtype=np.int64),
        scaling_transforms=tuple(transforms),
        observed_model_frame=observed_frame,
        scaled_design_frame=design_frame,
        design_info=design_info,
        ordered_positions=tuple(ordered_positions),
    )


def reference_softmax_probabilities(
    design_matrix: np.ndarray, beta_nonreference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate stable categorical probabilities with a structural reference zero.

    Not reached from ``bayes_rarcat``: this is intentional private API, called
    by the verification suite and by callers outside the public path, and it
    has no reference in ``src/`` other than this definition. It is retained
    deliberately as part of the declared verification surface of a completed
    subsystem, and it is not dead code left behind by a refactor (finding F-09
    of the 2026-08-25 audit re-adjudicated the earlier reference counts, which
    had counted test and documentation references).
    """
    design = np.asarray(design_matrix, dtype=float)
    beta = np.asarray(beta_nonreference, dtype=float)
    if design.ndim != 2 or beta.ndim != 2:
        raise ValueError("design and beta_nonreference must be two-dimensional")
    if design.shape[1] != beta.shape[0] or beta.shape[1] < 1:
        raise ValueError("design and coefficient dimensions are incompatible")
    if not np.isfinite(design).all() or not np.isfinite(beta).all():
        raise ValueError("design and coefficient values must be finite")
    beta_full = np.concatenate([np.zeros((beta.shape[0], 1)), beta], axis=1)
    logits = design @ beta_full
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return (
        _readonly_array(probabilities, dtype=np.float64),
        _readonly_array(beta_full, dtype=np.float64),
    )


def prepare_prior(
    prepared: PreparedConditionalDesign,
    *,
    user_prior: Mapping[str, object] | None = None,
) -> PriorSpecification:
    """Realize either the complete frozen default or one complete user prior."""
    if not isinstance(prepared, PreparedConditionalDesign):
        raise TypeError("prepared must be a PreparedConditionalDesign")
    coefficients = prepared.coefficient_coordinates
    nonreference = prepared.cluster_order[1:]
    expected_shape = (len(coefficients), len(nonreference))
    if user_prior is None:
        mean = np.zeros(expected_shape, dtype=np.float64)
        row_sigma = np.asarray(
            [1.5 if coordinate == "(Intercept)" else 1.0 for coordinate in coefficients],
            dtype=np.float64,
        )
        sigma = np.broadcast_to(row_sigma[:, None], expected_shape)
        source = "default"
    else:
        if not isinstance(user_prior, Mapping):
            raise TypeError("user prior must be a complete mapping")
        required = {
            "mean",
            "sigma",
            "coefficient_coordinates",
            "nonreference_cluster_coordinates",
        }
        if set(user_prior) != required:
            raise ValueError("user prior must contain exactly the complete prior fields")
        mean = np.asarray(user_prior["mean"], dtype=float)
        sigma = np.asarray(user_prior["sigma"], dtype=float)
        supplied_coefficients = tuple(user_prior["coefficient_coordinates"])  # type: ignore[arg-type]
        supplied_clusters = tuple(user_prior["nonreference_cluster_coordinates"])  # type: ignore[arg-type]
        if mean.shape != expected_shape or sigma.shape != expected_shape:
            raise ValueError(f"user prior shape must be exactly {expected_shape}")
        if supplied_coefficients != coefficients or not _values_equal(
            supplied_clusters, nonreference
        ):
            raise ValueError("user prior coordinates must match the model exactly and in order")
        if not np.isfinite(mean).all() or not np.isfinite(sigma).all():
            raise ValueError("user prior mean and sigma must be finite")
        if np.any(sigma <= 0.0):
            raise ValueError("user prior sigma must be strictly positive")
        source = "user"
    return PriorSpecification(
        source=source,
        coefficient_coordinates=coefficients,
        nonreference_cluster_coordinates=nonreference,
        mean=_readonly_array(mean, dtype=np.float64),
        sigma=_readonly_array(sigma, dtype=np.float64),
    )


def _validated_response_codes(response_codes: object, *, n: int, K: int) -> np.ndarray:
    codes = np.asarray(response_codes)
    if codes.dtype == np.dtype(bool) or not np.issubdtype(codes.dtype, np.integer):
        raise TypeError("response_codes must contain exact non-boolean integers")
    if codes.ndim != 1 or codes.shape != (n,):
        raise ValueError(f"response_codes must have shape ({n},)")
    integer_codes = np.asarray(codes, dtype=np.int64)
    if set(integer_codes.tolist()) != set(range(K)):
        raise ValueError("response_codes must contain every cluster code 0..K-1")
    return integer_codes


def _import_pymc():
    try:
        import pymc as pm
    except ImportError as error:
        raise ImportError("PyMC is required for private Bayesian RARCAT B3 fitting") from error
    return pm


def build_component_model(
    *,
    prepared: PreparedConditionalDesign,
    response_codes: object,
    prior: PriorSpecification,
):
    """Build one normalized direct-PyMC conditional categorical model."""
    if not isinstance(prepared, PreparedConditionalDesign):
        raise TypeError("prepared must be a PreparedConditionalDesign")
    expected_prior = (
        prepared.coefficient_coordinates,
        prepared.cluster_order[1:],
        (len(prepared.coefficient_coordinates), len(prepared.cluster_order) - 1),
    )
    if not isinstance(prior, PriorSpecification) or (
        prior.coefficient_coordinates != expected_prior[0]
        or not _values_equal(prior.nonreference_cluster_coordinates, expected_prior[1])
        or prior.mean.shape != expected_prior[2]
        or prior.sigma.shape != expected_prior[2]
    ):
        raise ValueError("prior does not match the prepared model coordinates and shape")
    codes = _validated_response_codes(
        response_codes, n=len(prepared.row_ids), K=len(prepared.cluster_order)
    )
    pm = _import_pymc()
    coords = {
        "observation": prepared.row_ids,
        "cluster": prepared.cluster_order,
        "nonreference_cluster": prepared.cluster_order[1:],
        "coefficient": prepared.coefficient_coordinates,
    }
    P = len(prepared.coefficient_coordinates)
    with pm.Model(coords=coords) as model:
        design = pm.Data(
            "design_matrix",
            prepared.design_matrix,
            dims=("observation", "coefficient"),
        )
        beta_nonreference = pm.Normal(
            "beta_nonreference",
            mu=prior.mean,
            sigma=prior.sigma,
            dims=("coefficient", "nonreference_cluster"),
        )
        beta_full = pm.Deterministic(
            "beta_full",
            pm.math.concatenate([pm.math.zeros((P, 1)), beta_nonreference], axis=1),
            dims=("coefficient", "cluster"),
        )
        logits = pm.Deterministic(
            "logits",
            pm.math.dot(design, beta_full),
            dims=("observation", "cluster"),
        )
        probability = pm.Deterministic(
            "probability",
            pm.math.softmax(logits, axis=1),
            dims=("observation", "cluster"),
        )
        pm.Categorical("cluster_id", p=probability, observed=codes, dims="observation")
    return model


def _validated_beta_full(prepared: PreparedConditionalDesign, beta_full: object) -> np.ndarray:
    beta = np.asarray(beta_full, dtype=float)
    expected_tail = (len(prepared.coefficient_coordinates), len(prepared.cluster_order))
    if beta.ndim != 4 or beta.shape[2:] != expected_tail:
        raise ValueError(
            "beta_full must have shape (chain, draw, coefficient, cluster) with tail "
            f"{expected_tail}"
        )
    if beta.shape[0] < 1 or beta.shape[1] < 1:
        raise ValueError("beta_full must contain positive chain and draw dimensions")
    if not np.isfinite(beta).all():
        raise ValueError("beta_full must contain only finite values")
    if not np.array_equal(beta[..., 0], np.zeros_like(beta[..., 0])):
        raise ValueError("beta_full reference coefficient column must be exactly zero")
    return beta


def _probabilities_for_beta(design: np.ndarray, beta: np.ndarray) -> np.ndarray:
    logits = np.einsum("np,cdpk->cdnk", np.asarray(design, dtype=float), beta)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    probability = exponentiated / exponentiated.sum(axis=-1, keepdims=True)
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("calculated posterior probabilities are nonfinite or outside [0, 1]")
    if not np.allclose(probability.sum(axis=-1), 1.0, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL):
        raise ValueError("calculated posterior probabilities do not form simplexes")
    return probability


def _scaling_transform(prepared: PreparedConditionalDesign, name: str) -> ScalingTransform:
    return next(item for item in prepared.scaling_transforms if item.predictor == name)


def _counterfactual_design(
    prepared: PreparedConditionalDesign, raw_frame: pd.DataFrame
) -> np.ndarray:
    scaled = raw_frame.copy(deep=True)
    for transform in prepared.scaling_transforms:
        values = np.asarray(raw_frame[transform.predictor], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("counterfactual continuous values must be finite")
        scaled[transform.predictor] = (values - transform.mean) / transform.population_sd
    design_data = _r_design_data(prepared.translated_rhs, scaled)
    rebuilt = _build_from_design_info(
        prepared.design_info,
        list(prepared.ordered_positions),
        design_data,
    )
    if rebuilt.shape != prepared.design_matrix.shape or not np.isfinite(rebuilt).all():
        raise ValueError("counterfactual design changed shape or produced nonfinite values")
    return rebuilt


def _frame_at_factor_level(
    prepared: PreparedConditionalDesign,
    *,
    predictor: str,
    level: object,
    levels: tuple[object, ...],
) -> pd.DataFrame:
    frame = prepared.observed_model_frame.copy(deep=True)
    original = frame[predictor]
    if level not in levels:
        raise ValueError(f"unknown counterfactual level for {predictor!r}")
    if isinstance(original.dtype, pd.CategoricalDtype):
        frame[predictor] = pd.Categorical(
            [level] * len(frame),
            categories=list(levels),
            ordered=original.cat.ordered,
        )
    elif pd.api.types.is_bool_dtype(original.dtype):
        if not isinstance(level, (bool, np.bool_)):
            raise ValueError(f"counterfactual boolean level for {predictor!r} was reordered")
        frame[predictor] = np.full(len(frame), bool(level), dtype=bool)
    else:
        frame[predictor] = np.asarray([level] * len(frame), dtype=object)
    return frame


def _effect_metadata(
    prepared: PreparedConditionalDesign,
    beta: np.ndarray,
    *,
    component_identifier: str,
    effect_identifier: str,
    source_variable: str,
    estimand_kind: str,
    method: str,
    original_scale_units: str,
    factor_reference: object | None = None,
    factor_comparison: object | None = None,
    half_step: float | None = None,
) -> EffectMetadata:
    offsets = None if half_step is None else (-half_step, half_step)
    denominator = None if half_step is None else 2.0 * half_step
    return EffectMetadata(
        effect_identifier=effect_identifier,
        source_variable=source_variable,
        estimand_kind=estimand_kind,
        method=method,
        original_scale_units=original_scale_units,
        factor_reference=factor_reference,
        factor_comparison=factor_comparison,
        half_step=half_step,
        evaluated_offsets=offsets,
        denominator=denominator,
        averaging_population=(f"all {len(prepared.row_ids)} observed model-frame rows, equal 1/n"),
        formula=prepared.formula,
        term_labels=prepared.term_labels,
        coefficient_coordinates=prepared.coefficient_coordinates,
        row_ids=prepared.row_ids,
        component_identifier=component_identifier,
        chain_coordinates=tuple(range(beta.shape[0])),
        draw_coordinates=tuple(range(beta.shape[1])),
        cluster_order=prepared.cluster_order,
        zero_sum_atol=_FLOAT_ATOL,
        zero_sum_rtol=_FLOAT_RTOL,
    )


def _checked_effect(
    metadata: EffectMetadata,
    observation_effects: np.ndarray,
    *,
    roundoff_scale: float,
) -> DrawwiseEffect:
    """Project the bounded round-off residual and reject a material one.

    ``roundoff_scale`` is the factor by which the construction of this effect
    multiplies the simplex round-off it inherits: ``1 / (2h)`` for a central
    counterfactual difference, ``1 / population_sd`` for an analytic derivative
    returned on the predictor's original scale, and ``1`` for a direct
    probability contrast.  Every tolerance below is stated on the standardized
    scale — that is what ``metadata.zero_sum_atol`` records — and multiplied by
    that factor here.  Without it an effect whose predictor happens to be
    measured in small units fails a guard meant to catch material errors, on a
    perfectly correct fit (finding F-04 of the 2026-08-25 audit).
    """
    values = np.array(observation_effects, dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"effect {metadata.effect_identifier!r} contains nonfinite draws")
    if not np.isfinite(roundoff_scale) or roundoff_scale <= 0.0:
        raise ValueError("roundoff_scale must be finite and strictly positive")
    residual = values.sum(axis=-1, keepdims=True)
    roundoff_bound = 100.0 * np.finfo(float).eps * roundoff_scale
    if np.max(np.abs(residual)) > roundoff_bound:
        raise ValueError(
            f"effect {metadata.effect_identifier!r} has a non-roundoff zero-sum residual"
        )
    # Project only that bounded floating-point residual symmetrically; any
    # larger, scientifically material residual failed immediately above.
    values -= residual / float(values.shape[-1])
    zero_sum_atol = metadata.zero_sum_atol * roundoff_scale
    if not np.allclose(values.sum(axis=-1), 0.0, atol=zero_sum_atol, rtol=metadata.zero_sum_rtol):
        raise ValueError(
            f"effect {metadata.effect_identifier!r} violates the observation-level zero-sum identity"
        )
    ame = np.sum(values, axis=2) / float(values.shape[2])
    if not np.allclose(ame.sum(axis=-1), 0.0, atol=zero_sum_atol, rtol=metadata.zero_sum_rtol):
        raise ValueError(
            f"effect {metadata.effect_identifier!r} violates the AME zero-sum identity"
        )
    return DrawwiseEffect(
        metadata=metadata,
        observation_effects=_readonly_array(values, dtype=np.float64),
        ame_draws=_readonly_array(ame, dtype=np.float64),
    )


def compute_drawwise_effects(
    *,
    prepared: PreparedConditionalDesign,
    beta_full: object,
    component_identifier: str,
) -> tuple[DrawwiseEffect, ...]:
    """Compute every original-scale continuous derivative and factor contrast."""
    if not isinstance(prepared, PreparedConditionalDesign):
        raise TypeError("prepared must be a PreparedConditionalDesign")
    if not isinstance(component_identifier, str) or not component_identifier:
        raise ValueError("component_identifier must be a nonempty string")
    beta = _validated_beta_full(prepared, beta_full)
    observed_probability = _probabilities_for_beta(prepared.design_matrix, beta)
    effects: list[DrawwiseEffect] = []
    step_map = dict(prepared.counterfactual_steps)

    for predictor in prepared.continuous_predictors:
        transform = _scaling_transform(prepared, predictor)
        if predictor not in prepared.nonlinear_continuous_predictors:
            position = prepared.coefficient_coordinates.index(predictor)
            coefficient = beta[:, :, position, :]
            weighted = np.sum(
                observed_probability * coefficient[:, :, None, :],
                axis=-1,
                keepdims=True,
            )
            observation_effects = (
                observed_probability
                * (coefficient[:, :, None, :] - weighted)
                / transform.population_sd
            )
            metadata = _effect_metadata(
                prepared,
                beta,
                component_identifier=component_identifier,
                effect_identifier=f"derivative:{predictor}",
                source_variable=predictor,
                estimand_kind="derivative",
                method="analytic_softmax",
                original_scale_units=f"probability per original {predictor} unit",
            )
            # The standardized derivative is divided by the population standard
            # deviation above, so its zero-sum round-off is amplified by the
            # same factor.
            roundoff_scale = 1.0 / transform.population_sd
        else:
            half_step = step_map[predictor]
            lower = prepared.observed_model_frame.copy(deep=True)
            upper = prepared.observed_model_frame.copy(deep=True)
            raw = np.asarray(prepared.observed_model_frame[predictor], dtype=float)
            lower[predictor] = raw - half_step
            upper[predictor] = raw + half_step
            lower_probability = _probabilities_for_beta(
                _counterfactual_design(prepared, lower), beta
            )
            upper_probability = _probabilities_for_beta(
                _counterfactual_design(prepared, upper), beta
            )
            observation_effects = (upper_probability - lower_probability) / (2.0 * half_step)
            metadata = _effect_metadata(
                prepared,
                beta,
                component_identifier=component_identifier,
                effect_identifier=f"derivative:{predictor}",
                source_variable=predictor,
                estimand_kind="derivative",
                method="central_counterfactual",
                original_scale_units=f"probability per original {predictor} unit",
                half_step=half_step,
            )
            # A central difference amplifies the two simplex round-off
            # residuals by 1/(2h).
            roundoff_scale = 1.0 / metadata.denominator
        effects.append(
            _checked_effect(metadata, observation_effects, roundoff_scale=roundoff_scale)
        )

    level_map = dict(prepared.factor_levels)
    for predictor in prepared.categorical_predictors:
        levels = level_map[predictor]
        reference = levels[0]
        reference_frame = _frame_at_factor_level(
            prepared, predictor=predictor, level=reference, levels=levels
        )
        reference_probability = _probabilities_for_beta(
            _counterfactual_design(prepared, reference_frame), beta
        )
        for level_index, comparison in enumerate(levels[1:], start=1):
            comparison_frame = _frame_at_factor_level(
                prepared, predictor=predictor, level=comparison, levels=levels
            )
            comparison_probability = _probabilities_for_beta(
                _counterfactual_design(prepared, comparison_frame), beta
            )
            metadata = _effect_metadata(
                prepared,
                beta,
                component_identifier=component_identifier,
                effect_identifier=f"contrast:{predictor}:level-{level_index}-vs-0",
                source_variable=predictor,
                estimand_kind="contrast",
                method="discrete_counterfactual",
                original_scale_units="probability difference",
                factor_reference=reference,
                factor_comparison=comparison,
            )
            effects.append(
                # A direct probability contrast inherits the simplex round-off
                # unamplified.
                _checked_effect(
                    metadata,
                    comparison_probability - reference_probability,
                    roundoff_scale=1.0,
                )
            )
    return tuple(effects)


def _seed_identifier(name: str, seed: np.random.SeedSequence) -> B3SeedIdentifier:
    return B3SeedIdentifier(
        name=name,
        entropy=int(seed.entropy),
        spawn_key=tuple(int(value) for value in seed.spawn_key),
        pool_size=int(seed.pool_size),
        generated_seed=int(seed.generate_state(1, dtype=np.uint32)[0]),
    )


def derive_b3_seed_manifest(*, root_seed: object, U: object) -> B3SeedManifest:
    """Derive independent shared, per-chain, and per-component predictive streams."""
    root = _exact_integer(root_seed, name="root_seed", minimum=0)
    component_count = _exact_integer(U, name="U", minimum=1)
    root_sequence = np.random.SeedSequence(root)
    prior_sequence, component_root = root_sequence.spawn(2)
    components: list[ComponentSeedManifest] = []
    for component_index in range(component_count):
        component_sequence = np.random.SeedSequence(
            entropy=component_root.entropy,
            spawn_key=(*component_root.spawn_key, component_index),
            pool_size=component_root.pool_size,
        )
        chain_root, predictive_sequence = component_sequence.spawn(2)
        chain_sequences = chain_root.spawn(SAMPLER_CONFIGURATION.chains)
        component_name = f"component-{component_index:03d}"
        components.append(
            ComponentSeedManifest(
                component_index=component_index,
                name=component_name,
                chain_streams=tuple(
                    _seed_identifier(f"{component_name}:chain-{chain_index}", sequence)
                    for chain_index, sequence in enumerate(chain_sequences)
                ),
                posterior_predictive=_seed_identifier(
                    f"{component_name}:posterior_predictive", predictive_sequence
                ),
            )
        )
    manifest = B3SeedManifest(
        root_seed=root,
        prior_predictive=_seed_identifier("prior_predictive", prior_sequence),
        components=tuple(components),
    )
    realized = [manifest.prior_predictive.generated_seed]
    for component in manifest.components:
        realized.extend(component.chain_seeds)
        realized.append(component.posterior_predictive.generated_seed)
    if len(realized) != len(set(realized)):
        raise RuntimeError("B3 seed derivation produced a duplicate realized seed")
    return manifest


def _reduced_max(values: object) -> float:
    """Largest value, or NaN when the collection is empty or has a nonfinite.

    Python's ``max``/``min`` propagate a NaN only when it is the first
    argument, so reducing the per-target diagnostic summaries with the
    builtins made a component's verdict depend on where a constant target sat
    in the tuple (finding F-08 of the 2026-08-25 audit).  Checking finiteness
    across the whole collection first removes that dependence.
    """
    collected = np.asarray([float(value) for value in values], dtype=float)
    if collected.size == 0 or not np.isfinite(collected).all():
        return float("nan")
    return float(np.max(collected))


def _reduced_min(values: object) -> float:
    """Smallest value, under the same rule as :func:`_reduced_max`."""
    collected = np.asarray([float(value) for value in values], dtype=float)
    if collected.size == 0 or not np.isfinite(collected).all():
        return float("nan")
    return float(np.min(collected))


def evaluate_sampler_diagnostics(
    *,
    all_finite: bool,
    max_rhat: float,
    min_bulk_ess: float,
    min_tail_ess: float,
    divergences: int,
    bfmi_by_chain: Sequence[float],
    reference_zero: bool,
    probability_simplex: bool,
) -> SamplerDiagnosticRecord:
    """Apply the frozen B1 material sampler and posterior diagnostic policy."""
    bfmi = tuple(float(value) for value in bfmi_by_chain)
    failures: list[str] = []
    if not all_finite:
        failures.append("nonfinite")
    if not np.isfinite(max_rhat) or max_rhat >= DIAGNOSTIC_THRESHOLDS.rhat_exclusive_max:
        failures.append("rhat")
    if not np.isfinite(min_bulk_ess) or min_bulk_ess < DIAGNOSTIC_THRESHOLDS.bulk_ess_minimum:
        failures.append("bulk_ess")
    if not np.isfinite(min_tail_ess) or min_tail_ess < DIAGNOSTIC_THRESHOLDS.tail_ess_minimum:
        failures.append("tail_ess")
    if divergences > DIAGNOSTIC_THRESHOLDS.divergences_maximum:
        failures.append("divergences")
    minimum_bfmi = float(np.min(bfmi)) if bfmi else float("nan")
    if (
        len(bfmi) != SAMPLER_CONFIGURATION.chains
        or not np.isfinite(bfmi).all()
        or minimum_bfmi < DIAGNOSTIC_THRESHOLDS.bfmi_each_chain_minimum
    ):
        failures.append("bfmi")
    if not reference_zero:
        failures.append("reference_zero")
    if not probability_simplex:
        failures.append("probability_simplex")
    return SamplerDiagnosticRecord(
        status="pass" if not failures else "fail",
        all_finite=bool(all_finite),
        max_rhat=float(max_rhat),
        min_bulk_ess=float(min_bulk_ess),
        min_tail_ess=float(min_tail_ess),
        divergences=int(divergences),
        bfmi_by_chain=bfmi,
        minimum_bfmi=minimum_bfmi,
        reference_zero=bool(reference_zero),
        probability_simplex=bool(probability_simplex),
        failure_reasons=tuple(failures),
        thresholds=DIAGNOSTIC_THRESHOLDS,
    )


def evaluate_monte_carlo_growth(values: object, *, mcse_mean: object) -> GrowthDiagnostic:
    """Compare first-half and complete retained draws for one scalar target."""
    draws = np.asarray(values, dtype=float)
    mcse = float(mcse_mean)
    if draws.ndim != 2 or draws.shape[0] < 1 or draws.shape[1] < 2:
        raise ValueError("growth target must have shape (chain, at least two draws)")
    if not np.isfinite(draws).all() or not np.isfinite(mcse) or mcse < 0.0:
        raise ValueError("growth target and MCSE must be finite with nonnegative MCSE")
    flattened = draws.reshape(-1)
    half = draws[:, : draws.shape[1] // 2].reshape(-1)
    posterior_sd = float(np.std(flattened, ddof=1))
    half_mean = float(np.mean(half))
    full_mean = float(np.mean(flattened))
    constant = posterior_sd <= np.finfo(float).eps
    if constant:
        change = float("inf")
        mcse_fraction = float("inf")
        failures = ["constant_nonstructural"]
    else:
        change = abs(half_mean - full_mean) / posterior_sd
        mcse_fraction = mcse / posterior_sd
        failures = []
        if change > DIAGNOSTIC_THRESHOLDS.mean_change_in_sd_maximum:
            failures.append("mean_change")
        if mcse_fraction > DIAGNOSTIC_THRESHOLDS.mcse_fraction_of_sd_maximum:
            failures.append("mcse")
    return GrowthDiagnostic(
        status="pass" if not failures else "fail",
        posterior_sd=posterior_sd,
        half_mean=half_mean,
        full_mean=full_mean,
        mean_change_in_sd=float(change),
        mcse_mean=mcse,
        mcse_fraction_of_sd=float(mcse_fraction),
        constant_nonstructural=constant,
        failure_reasons=tuple(failures),
        thresholds=DIAGNOSTIC_THRESHOLDS,
    )


def classify_warning(message: object) -> WarningRecord:
    """Classify one warning against the material-vocabulary allowlist."""
    text = str(message)
    folded = text.casefold()
    material = any(term in folded for term in MATERIAL_WARNING_TERMS)
    return WarningRecord(
        message=text,
        classification="invalid" if material else "informational",
        material=material,
    )


class _SamplerLogCollector(logging.Handler):
    """Collect the sampler's own ``logging`` output at warning level or above."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def captured_sampler_warnings() -> Iterator[list[WarningRecord]]:
    """Capture Python warnings *and* PyMC's logger output around a sampling call.

    PyMC 6.0.1 reports divergences, tree-depth exhaustion, R-hat and effective
    sample size through the ``pymc`` logger rather than through ``warnings``
    (``pymc/stats/convergence.py``), so a ``catch_warnings`` block on its own
    observes none of them.  Records are appended to the yielded list when the
    block exits normally; a raising block propagates its exception unchanged,
    exactly as the plain ``catch_warnings`` blocks did.

    Like ``warnings.catch_warnings``, which it wraps, this is process-global
    and not thread-safe.  B3 fits components serially with ``cores=1``, so
    concurrent use does not arise on the supported path.
    """
    records: list[WarningRecord] = []
    handler = _SamplerLogCollector()
    logger = logging.getLogger(_SAMPLER_LOGGER_NAME)
    logger.addHandler(handler)
    try:
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            yield records
        records.extend(
            classify_warning(f"{item.category.__name__}: {item.message}") for item in caught
        )
        records.extend(classify_warning(message) for message in handler.messages)
    finally:
        logger.removeHandler(handler)


def _predictive_coordinates(
    values: Any, *, cluster_order: Sequence[object]
) -> PredictiveCoordinates:
    def coordinate(name: str) -> tuple[object, ...]:
        if name in values.coords:
            return tuple(np.asarray(values.coords[name].values, dtype=object).tolist())
        return tuple(range(int(values.sizes[name])))

    return PredictiveCoordinates(
        chain=coordinate("chain"),
        draw=coordinate("draw"),
        observation=coordinate("observation"),
        cluster=tuple(cluster_order),
    )


def evaluate_posterior_predictive(
    *,
    observed_codes: object,
    replicated_codes: object,
    cluster_order: Sequence[object],
    seed: int,
    warnings: Sequence[WarningRecord],
    coordinates: PredictiveCoordinates | None = None,
) -> PosteriorPredictiveRecord:
    observed = np.asarray(observed_codes)
    replicated = np.asarray(replicated_codes)
    order = tuple(cluster_order)
    if observed.ndim != 1 or replicated.ndim != 3 or replicated.shape[-1] != len(observed):
        raise ValueError("posterior predictive codes have incompatible dimensions")
    if not np.issubdtype(observed.dtype, np.integer) or not np.issubdtype(
        replicated.dtype, np.integer
    ):
        raise TypeError("posterior predictive codes must be integers")
    observed_counts = tuple(
        (label, int(np.sum(observed == index))) for index, label in enumerate(order)
    )
    mean_counts = tuple(
        (label, float(np.mean(np.sum(replicated == index, axis=-1))))
        for index, label in enumerate(order)
    )
    maximum_error = max(
        abs(mean_count - observed_count) / float(len(observed))
        for (_, observed_count), (_, mean_count) in zip(observed_counts, mean_counts, strict=True)
    )
    failures = []
    if maximum_error > DIAGNOSTIC_THRESHOLDS.predictive_count_fraction_error_maximum:
        failures.append("posterior_predictive")
    if any(record.material for record in warnings):
        failures.append("material_warning")
    return PosteriorPredictiveRecord(
        status="pass" if not failures else "fail",
        seed=int(seed),
        observed_counts=observed_counts,
        mean_replicated_counts=mean_counts,
        maximum_count_fraction_error=float(maximum_error),
        cluster_order=order,
        coordinates=coordinates,
        warnings=tuple(warnings),
        failure_reasons=tuple(failures),
    )


def sample_component_posterior(*, model: Any, chain_seeds: Sequence[int]):
    """Run the fixed serial direct-PyMC NUTS profile exactly once."""
    seeds = tuple(int(value) for value in chain_seeds)
    if len(seeds) != SAMPLER_CONFIGURATION.chains or len(set(seeds)) != len(seeds):
        raise ValueError("chain_seeds must contain four independent seeds")
    pm = _import_pymc()
    with model:
        return pm.sample(
            draws=SAMPLER_CONFIGURATION.draws,
            tune=SAMPLER_CONFIGURATION.tune,
            chains=SAMPLER_CONFIGURATION.chains,
            cores=SAMPLER_CONFIGURATION.cores,
            blas_cores=SAMPLER_CONFIGURATION.blas_cores,
            random_seed=list(seeds),
            target_accept=SAMPLER_CONFIGURATION.target_accept,
            nuts_sampler=SAMPLER_CONFIGURATION.nuts_sampler,
            progressbar=False,
            return_inferencedata=True,
        )


def sample_component_posterior_predictive(*, model: Any, posterior: Any, seed: int):
    """Sample labels from every retained conditional posterior draw once."""
    pm = _import_pymc()
    with model:
        return pm.sample_posterior_predictive(
            posterior,
            var_names=["cluster_id"],
            random_seed=int(seed),
            progressbar=False,
            return_inferencedata=True,
        )


def _import_arviz():
    try:
        import arviz as az
    except ImportError as error:
        raise ImportError("ArviZ is required for private Bayesian RARCAT B3 diagnostics") from error
    return az


def summarize_target_draws(values: object, *, target_identifier: str) -> TargetSummary:
    """Summarize one unmixed scalar coefficient/effect target across chains."""
    draws = np.asarray(values, dtype=float)
    expected_shape = (SAMPLER_CONFIGURATION.chains, SAMPLER_CONFIGURATION.draws)
    if draws.ndim != 2 or draws.shape != expected_shape:
        # Derived from the frozen configuration so the message cannot go
        # stale beside the check (2026-09-02 audit, S-09 informational).
        raise ValueError(f"summary target must have shape {expected_shape}")
    if not np.isfinite(draws).all():
        raise ValueError("summary target draws must be finite")
    flattened = draws.reshape(-1)
    posterior_sd = float(np.std(flattened, ddof=1))
    constant = posterior_sd <= np.finfo(float).eps
    if constant:
        mcse = bulk_ess = tail_ess = rhat = float("nan")
    else:
        summary = (
            _import_arviz()
            .summary(
                {"target": draws},
                var_names=["target"],
                kind="diagnostics",
                round_to="none",
            )
            .iloc[0]
        )
        mcse = float(summary["mcse_mean"])
        bulk_ess = float(summary["ess_bulk"])
        tail_ess = float(summary["ess_tail"])
        rhat = float(summary["r_hat"])
    growth = evaluate_monte_carlo_growth(draws, mcse_mean=0.0 if constant else mcse)
    interval = np.quantile(flattened, [0.03, 0.97])
    return TargetSummary(
        target_identifier=target_identifier,
        posterior_mean=float(np.mean(flattened)),
        posterior_sd=posterior_sd,
        interval_94=(float(interval[0]), float(interval[1])),
        probability_above_zero=float(np.mean(flattened > 0.0)),
        probability_below_zero=float(np.mean(flattened < 0.0)),
        mcse_mean=mcse,
        bulk_ess=bulk_ess,
        tail_ess=tail_ess,
        rhat=rhat,
        growth=growth,
        constant_nonstructural=constant,
    )


def _bfmi_values(posterior: Any) -> tuple[float, ...]:
    try:
        from arviz_stats import bfmi
    except ImportError as error:
        raise ImportError("ArviZ stats is required for B3 BFMI diagnostics") from error
    values = bfmi(posterior).dataset["energy"].values
    return tuple(float(value) for value in np.asarray(values).ravel())


def _component_target_summaries(
    *,
    prepared: PreparedConditionalDesign,
    beta_full: np.ndarray,
    effects: tuple[DrawwiseEffect, ...],
) -> tuple[tuple[TargetSummary, ...], tuple[TargetSummary, ...]]:
    coefficient_summaries = tuple(
        summarize_target_draws(
            beta_full[:, :, coefficient_index, cluster_index],
            target_identifier=(
                f"coefficient:{prepared.coefficient_coordinates[coefficient_index]}:"
                f"cluster-{cluster_index}"
            ),
        )
        for coefficient_index in range(len(prepared.coefficient_coordinates))
        for cluster_index in range(1, len(prepared.cluster_order))
    )
    effect_summaries = tuple(
        summarize_target_draws(
            effect.ame_draws[:, :, cluster_index],
            target_identifier=(
                f"effect:{effect.metadata.effect_identifier}:cluster-{cluster_index}"
            ),
        )
        for effect in effects
        for cluster_index in range(len(prepared.cluster_order))
    )
    return coefficient_summaries, effect_summaries


def run_prior_predictive(
    *,
    prepared: PreparedConditionalDesign,
    prior: PriorSpecification,
    seed: B3SeedIdentifier,
) -> PriorPredictiveRecord:
    """Run and retain the one shared 500-draw prior predictive check."""
    model = build_component_model(
        prepared=prepared,
        response_codes=prepared.response_codes,
        prior=prior,
    )
    pm = _import_pymc()
    with captured_sampler_warnings() as captured:
        with model:
            predictive = pm.sample_prior_predictive(
                draws=500,
                var_names=["probability", "cluster_id"],
                random_seed=seed.generated_seed,
                return_inferencedata=True,
            )
    probability_values = predictive.prior["probability"]
    probability = np.asarray(probability_values.values, dtype=float)
    replicated = np.asarray(predictive.prior_predictive["cluster_id"].values)
    finite = bool(np.isfinite(probability).all())
    simplex = bool(
        finite
        and np.all((probability >= 0.0) & (probability <= 1.0))
        and np.allclose(probability.sum(axis=-1), 1.0, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL)
    )
    mean_counts = tuple(
        (label, float(np.mean(np.sum(replicated == index, axis=-1))))
        for index, label in enumerate(prepared.cluster_order)
    )
    failures: list[str] = []
    if not finite:
        failures.append("prior_predictive_nonfinite")
    if not simplex:
        failures.append("prior_predictive_simplex")
    if any(record.material for record in captured):
        failures.append("material_warning")
    return PriorPredictiveRecord(
        status="pass" if not failures else "fail",
        seed=seed.generated_seed,
        predictive=predictive,
        finite_probability=finite,
        probability_simplex=simplex,
        mean_cluster_counts=mean_counts,
        cluster_order=prepared.cluster_order,
        coordinates=_predictive_coordinates(
            probability_values,
            cluster_order=prepared.cluster_order,
        ),
        warnings=tuple(captured),
        failure_reasons=tuple(failures),
    )


def fit_partition_component(
    *,
    prepared: PreparedConditionalDesign,
    prior: PriorSpecification,
    response_codes: object,
    component_index: int,
    component_identifier: str,
    partition: np.ndarray,
    count: int,
    mass: float,
    mass_mcse: float,
    seeds: ComponentSeedManifest,
) -> ComponentFitRecord:
    """Fit, diagnose, predict, and summarize one exact conditional component."""
    started = time.perf_counter()
    model = build_component_model(
        prepared=prepared,
        response_codes=response_codes,
        prior=prior,
    )
    with captured_sampler_warnings() as captured:
        posterior = sample_component_posterior(
            model=model,
            chain_seeds=seeds.chain_seeds,
        )
        beta_full = np.asarray(posterior.posterior["beta_full"].values, dtype=float)
        probability = np.asarray(posterior.posterior["probability"].values, dtype=float)
        effects = compute_drawwise_effects(
            prepared=prepared,
            beta_full=beta_full,
            component_identifier=component_identifier,
        )
        coefficient_summaries, effect_summaries = _component_target_summaries(
            prepared=prepared,
            beta_full=beta_full,
            effects=effects,
        )
        posterior_predictive = sample_component_posterior_predictive(
            model=model,
            posterior=posterior,
            seed=seeds.posterior_predictive.generated_seed,
        )
    reference_zero = bool(
        beta_full.ndim == 4 and np.array_equal(beta_full[..., 0], np.zeros_like(beta_full[..., 0]))
    )
    probability_simplex = bool(
        probability.ndim == 4
        and np.isfinite(probability).all()
        and np.all((probability >= 0.0) & (probability <= 1.0))
        and np.allclose(probability.sum(axis=-1), 1.0, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL)
    )
    summaries = (*coefficient_summaries, *effect_summaries)
    sampler_diagnostics = evaluate_sampler_diagnostics(
        all_finite=bool(
            np.isfinite(beta_full).all()
            and np.isfinite(probability).all()
            and all(np.isfinite(effect.ame_draws).all() for effect in effects)
        ),
        max_rhat=_reduced_max(summary.rhat for summary in summaries),
        min_bulk_ess=_reduced_min(summary.bulk_ess for summary in summaries),
        min_tail_ess=_reduced_min(summary.tail_ess for summary in summaries),
        divergences=int(posterior.sample_stats["diverging"].sum().to_numpy()),
        bfmi_by_chain=_bfmi_values(posterior),
        reference_zero=reference_zero,
        probability_simplex=probability_simplex,
    )
    replicated_values = posterior_predictive.posterior_predictive["cluster_id"]
    replicated = np.asarray(replicated_values.values)
    predictive_diagnostics = evaluate_posterior_predictive(
        observed_codes=response_codes,
        replicated_codes=replicated,
        cluster_order=prepared.cluster_order,
        seed=seeds.posterior_predictive.generated_seed,
        warnings=tuple(captured),
        coordinates=_predictive_coordinates(
            replicated_values,
            cluster_order=prepared.cluster_order,
        ),
    )
    failures = list(sampler_diagnostics.failure_reasons)
    for summary in summaries:
        failures.extend(
            f"{summary.target_identifier}:{reason}" for reason in summary.growth.failure_reasons
        )
    failures.extend(predictive_diagnostics.failure_reasons)
    if any(record.material for record in captured) and "material_warning" not in failures:
        failures.append("material_warning")
    return ComponentFitRecord(
        component_index=int(component_index),
        component_identifier=component_identifier,
        partition=_readonly_array(partition, dtype=object),
        count=int(count),
        mass=float(mass),
        mass_mcse=float(mass_mcse),
        seeds=seeds,
        status="pass" if not failures else "fail",
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        posterior_predictive_diagnostics=predictive_diagnostics,
        sampler_diagnostics=sampler_diagnostics,
        coefficient_summaries=coefficient_summaries,
        effects=effects,
        effect_summaries=effect_summaries,
        timing_seconds=time.perf_counter() - started,
        warnings=tuple(captured),
        failure_reasons=tuple(failures),
    )


def _software_versions() -> tuple[tuple[str, str], ...]:
    """Record every package whose version can change a recorded quantity.

    This must remain a superset of ``weightedcluster.bayesian``'s declared
    runtime dependencies; ``test_recorded_software_versions_cover_every_
    declared_runtime_dependency`` is the contract that keeps the two from
    drifting apart, as they had by finding F-15 of the 2026-08-25 audit.
    ``patsy`` builds the design matrix and every counterfactual, and
    ``arviz-stats`` computes BFMI, a material diagnostic.
    """
    names = (
        # samplers and posterior machinery
        "pymc",
        "arviz",
        "arviz-stats",
        "pytensor",
        # numerics
        "numpy",
        "scipy",
        "pandas",
        # design construction
        "patsy",
        # storage of the private bundle
        "xarray",
        "h5netcdf",
        "h5py",
    )
    return tuple((name, importlib_metadata.version(name)) for name in names)


def fit_conditional_posteriors(
    *,
    partition_posterior: PartitionPosteriorResult,
    formula: str,
    data: pd.DataFrame,
    continuous_predictors: Sequence[str],
    counterfactual_steps: Mapping[str, object],
    root_seed: object,
    user_prior: Mapping[str, object] | None = None,
) -> ConditionalPosteriorResult:
    """Fit every exact distinct B2 partition once without mixing components."""
    validated_root_seed = _exact_integer(root_seed, name="root_seed", minimum=0)
    prepared = prepare_conditional_design(
        partition_posterior=partition_posterior,
        formula=formula,
        data=data,
        continuous_predictors=continuous_predictors,
        counterfactual_steps=counterfactual_steps,
    )
    n, B, K, U = validate_partition_posterior(partition_posterior)
    prior = prepare_prior(prepared, user_prior=user_prior)
    seed_manifest = derive_b3_seed_manifest(root_seed=validated_root_seed, U=U)
    prior_predictive = run_prior_predictive(
        prepared=prepared,
        prior=prior,
        seed=seed_manifest.prior_predictive,
    )
    cluster_positions = {
        label: index for index, label in enumerate(partition_posterior.reference_cluster_order)
    }
    components: list[ComponentFitRecord] = []
    for component_index, partition in enumerate(partition_posterior.distinct_partitions):
        component_identifier = f"component-{component_index:03d}"
        seeds = seed_manifest.components[component_index]
        codes = np.asarray([cluster_positions[label] for label in partition], dtype=np.int64)
        started = time.perf_counter()
        try:
            component = fit_partition_component(
                prepared=prepared,
                prior=prior,
                response_codes=codes,
                component_index=component_index,
                component_identifier=component_identifier,
                partition=np.array(partition, dtype=object, copy=True),
                count=int(partition_posterior.counts[component_index]),
                mass=float(partition_posterior.masses[component_index]),
                mass_mcse=float(partition_posterior.mass_mcse[component_index]),
                seeds=seeds,
            )
        except Exception as error:
            component = ComponentFitRecord(
                component_index=component_index,
                component_identifier=component_identifier,
                partition=_readonly_array(partition, dtype=object),
                count=int(partition_posterior.counts[component_index]),
                mass=float(partition_posterior.masses[component_index]),
                mass_mcse=float(partition_posterior.mass_mcse[component_index]),
                seeds=seeds,
                status="fail",
                timing_seconds=time.perf_counter() - started,
                failure_reasons=("sampling_exception",),
                exception_type=type(error).__name__,
                exception_message=str(error),
            )
        components.append(component)
    failures = tuple(component for component in components if component.status != "pass")
    status = "pass" if prior_predictive.status == "pass" and not failures else "fail"
    timings = [component.timing_seconds for component in components]
    mean_time = float(np.mean(timings))
    return ConditionalPosteriorResult(
        status=status,
        n=n,
        B=B,
        K=K,
        U=U,
        observation_ids=_readonly_array(partition_posterior.observation_ids, dtype=object),
        reference_labels=_readonly_array(partition_posterior.reference_labels, dtype=object),
        cluster_order=partition_posterior.reference_cluster_order,
        distinct_partitions=_readonly_array(partition_posterior.distinct_partitions, dtype=object),
        counts=_readonly_array(partition_posterior.counts, dtype=np.int64),
        masses=_readonly_array(partition_posterior.masses, dtype=np.float64),
        mass_mcse=_readonly_array(partition_posterior.mass_mcse, dtype=np.float64),
        root_seed=validated_root_seed,
        seed_manifest=seed_manifest,
        formula=formula,
        design=prepared,
        prior=prior,
        sampler_configuration=SAMPLER_CONFIGURATION,
        software_versions=_software_versions(),
        prior_predictive=prior_predictive,
        components=tuple(components),
        failures=failures,
        mean_component_time_seconds=mean_time,
        projected_serial_time_u16_seconds=float(_BAYES_RARCAT_ENVELOPE["U"][1]) * mean_time,
    )

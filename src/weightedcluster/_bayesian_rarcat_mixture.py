"""Private deterministic mixture integration for Bayesian RARCAT B4.

This module combines completed B2 partition masses with completed B3
conditional posterior draws.  It never reconstructs partitions, fits a model,
or changes the diagnostic disposition of a B3 component.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from . import _bayesian_rarcat_conditional as _b3
from . import _bayesian_rarcat_partition as _b2
from ._bayesian_rarcat_conditional import ConditionalPosteriorResult
from ._bayesian_rarcat_partition import PartitionPosteriorResult
from ._parallel import run_task_batch, validate_parallel_controls
from .bayesian import BayesRARCATResult, _same_identity_sequence
from .bayesian import _same_identity_value as _same_identity

_MASS_ATOL = 1e-15
_SUMMARY_ATOL = 1e-12
_SUMMARY_RTOL = 1e-12
_ALLOCATION_TOTAL = 10_000
PRIVATE_SCHEMA_NAME = "weightedcluster.bayes_rarcat.private"
PRIVATE_SCHEMA_VERSION = 1


def _readonly_array(values: object, *, dtype: object | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ScalarMixtureSummary:
    """Exact weighted support and decomposition for one scalar target.

    Support is concatenated in stable component order.  Draws within a
    component retain their input C-order.  Component variances are population
    variances (``ddof=0``), as required by the B4 mixture identity.
    """

    values: np.ndarray
    weights: np.ndarray
    component_indices: np.ndarray
    component_draw_counts: np.ndarray
    component_weight_sums: np.ndarray
    component_means: np.ndarray
    component_variances: np.ndarray
    masses: np.ndarray
    support_count: int
    component_count: int
    posterior_mean: float
    posterior_sd: float
    interval_94: tuple[float, float]
    probability_above_zero: float
    probability_below_zero: float
    within_component_variance: float
    between_component_variance: float
    total_variance: float
    second_moment_variance: float
    variance_residual: float
    projected_roundoff: bool


@dataclass(frozen=True)
class EffectTargetMetadata:
    """Component-independent effect coordinates retained for one B4 target."""

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
    cluster_order: tuple[object, ...]
    zero_sum_atol: float
    zero_sum_rtol: float


@dataclass(frozen=True)
class TargetDescriptor:
    """Stable semantic identity for one coefficient/effect by cluster."""

    target_index: int
    target_identifier: str
    target_kind: str
    coefficient_coordinate: str | None
    effect_metadata: EffectTargetMetadata | None
    cluster_coordinate: object
    cluster_index: int


@dataclass(frozen=True)
class WeightedTargetSupport:
    """Exact per-draw support in component, chain, then draw order."""

    descriptor: TargetDescriptor
    values: np.ndarray
    weights: np.ndarray
    component_indices: np.ndarray
    component_identifiers: tuple[str, ...]
    chain_coordinates: np.ndarray
    draw_coordinates: np.ndarray
    component_offsets: np.ndarray


@dataclass(frozen=True)
class IntegratedTargetSummary:
    """Mass-weighted scalar posterior and total-variance decomposition."""

    descriptor: TargetDescriptor
    support_count: int
    component_count: int
    posterior_mean: float
    posterior_sd: float
    interval_94: tuple[float, float]
    probability_above_zero: float
    probability_below_zero: float
    within_component_variance: float
    between_component_variance: float
    total_variance: float
    second_moment_variance: float
    variance_residual: float
    component_means: np.ndarray
    component_variances: np.ndarray
    component_masses: np.ndarray
    component_draw_counts: np.ndarray
    projected_roundoff: bool


@dataclass(frozen=True)
class AllocationRecord:
    """Deterministic largest-remainder presentation allocation metadata."""

    requested_total: int
    allocations: np.ndarray
    effective_masses: np.ndarray
    signed_mass_errors: np.ndarray
    absolute_mass_errors: np.ndarray
    max_absolute_mass_error: float


@dataclass(frozen=True)
class _ExtractedComponent:
    component_index: int
    component_identifier: str
    descriptors: tuple[TargetDescriptor, ...]
    draws: tuple[np.ndarray, ...]
    chain_coordinates: np.ndarray
    draw_coordinates: np.ndarray


def _validated_support_vectors(values: object, weights: object) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(values, dtype=np.float64)
    support_weights = np.asarray(weights, dtype=np.float64)
    if support.ndim != 1 or support_weights.ndim != 1 or support.shape != support_weights.shape:
        raise ValueError("values and weights must be one-dimensional arrays with equal shape")
    if support.size < 1:
        raise ValueError("values and weights must contain positive support")
    if not np.isfinite(support).all() or not np.isfinite(support_weights).all():
        raise ValueError("values and weights must be finite")
    if np.any(support_weights <= 0.0):
        raise ValueError("weights must be strictly positive")
    return support, support_weights


def weighted_empirical_quantile(values: object, weights: object, q: object) -> float:
    """Return the inverse weighted empirical CDF without interpolation."""
    if isinstance(q, (bool, np.bool_)):
        raise TypeError("q must be a finite probability")
    try:
        probability = float(q)
    except (TypeError, ValueError) as error:
        raise TypeError("q must be a finite probability") from error
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("q must be between zero and one")
    support, support_weights = _validated_support_vectors(values, weights)
    order = np.argsort(support, kind="stable")
    ordered_values = support[order]
    ordered_weights = support_weights[order]
    cumulative = np.cumsum(ordered_weights, dtype=np.float64)
    threshold = probability * math.fsum(ordered_weights)
    eligible = np.flatnonzero(cumulative >= threshold)
    position = int(eligible[0]) if eligible.size else ordered_values.size - 1
    return float(ordered_values[min(position, ordered_values.size - 1)])


def _project_roundoff_variance(value: float, *, name: str) -> tuple[float, bool]:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} variance must be finite")
    if numeric >= 0.0:
        return numeric, False
    if numeric >= -_SUMMARY_ATOL:
        return 0.0, True
    raise ValueError(f"{name} variance is materially negative")


def summarize_scalar_mixture(
    component_draws: Sequence[object], masses: object
) -> ScalarMixtureSummary:
    """Build exact weighted support and total-variance summaries.

    Each component draw array may have any nonempty shape; it is flattened in
    NumPy C-order.  The supplied masses are the inferential weights and must
    already sum to one.  They are never replaced by component draw counts.
    """
    components = tuple(component_draws)
    if not components:
        raise ValueError("component_draws must contain at least one component")
    mass_array = np.asarray(masses, dtype=np.float64)
    if mass_array.ndim != 1 or mass_array.shape != (len(components),):
        raise ValueError("masses must contain one mass per component")
    if not np.isfinite(mass_array).all() or np.any(mass_array <= 0.0):
        raise ValueError("component masses must be finite and strictly positive")
    if not np.isclose(float(np.sum(mass_array)), 1.0, atol=_MASS_ATOL, rtol=0.0):
        raise ValueError("component masses must sum to one")

    flattened: list[np.ndarray] = []
    counts: list[int] = []
    for values in components:
        draws = np.asarray(values, dtype=np.float64)
        if draws.size < 1:
            raise ValueError("each component must have a positive retained draw count")
        if not np.isfinite(draws).all():
            raise ValueError("component draws must be finite")
        flattened.append(np.ravel(draws, order="C"))
        counts.append(int(draws.size))

    count_array = np.asarray(counts, dtype=np.int64)
    values = np.concatenate(flattened)
    component_indices = np.repeat(np.arange(len(components), dtype=np.int64), count_array)
    weights = np.concatenate(
        [
            np.full(count, mass_array[index] / float(count), dtype=np.float64)
            for index, count in enumerate(counts)
        ]
    )
    weight_sums = np.asarray(
        [float(np.sum(weights[component_indices == index])) for index in range(len(components))],
        dtype=np.float64,
    )
    if not np.allclose(weight_sums, mass_array, atol=_MASS_ATOL, rtol=0.0):
        raise RuntimeError("component support weights do not reproduce component masses")
    if not np.isclose(float(np.sum(weights)), 1.0, atol=_MASS_ATOL, rtol=0.0):
        raise RuntimeError("total support weight does not equal one")

    component_means = np.asarray([float(np.mean(draws)) for draws in flattened])
    component_variances = np.asarray(
        [float(np.var(draws, ddof=0)) for draws in flattened], dtype=np.float64
    )
    posterior_mean = float(np.sum(mass_array * component_means))
    direct_mean = float(np.sum(weights * values))
    if not np.isclose(direct_mean, posterior_mean, atol=_SUMMARY_ATOL, rtol=_SUMMARY_RTOL):
        raise RuntimeError("weighted support mean does not match component-mass mean")

    within = float(np.sum(mass_array * component_variances))
    between = float(np.sum(mass_array * np.square(component_means - posterior_mean)))
    within, within_projected = _project_roundoff_variance(within, name="within-component")
    between, between_projected = _project_roundoff_variance(between, name="between-component")
    total, total_projected = _project_roundoff_variance(within + between, name="total")
    # Take the second moment about the mixture mean.  The algebraically
    # identical uncentred form, ``sum(m*(v + mu**2)) - M**2``, cancels
    # catastrophically once the component means are large relative to their
    # standard deviations, so it failed this identity on correct mixtures
    # (finding F-11 of the 2026-08-25 audit).
    second = float(
        np.sum(mass_array * (component_variances + np.square(component_means - posterior_mean)))
    )
    second, second_projected = _project_roundoff_variance(second, name="second-moment")
    residual = second - total
    if not np.isclose(second, total, atol=_SUMMARY_ATOL, rtol=_SUMMARY_RTOL):
        raise RuntimeError("law-of-total-variance second-moment identity failed")

    probability_above = float(np.sum(weights[values > 0.0]))
    probability_below = float(np.sum(weights[values < 0.0]))
    return ScalarMixtureSummary(
        values=_readonly_array(values, dtype=np.float64),
        weights=_readonly_array(weights, dtype=np.float64),
        component_indices=_readonly_array(component_indices, dtype=np.int64),
        component_draw_counts=_readonly_array(count_array, dtype=np.int64),
        component_weight_sums=_readonly_array(weight_sums, dtype=np.float64),
        component_means=_readonly_array(component_means, dtype=np.float64),
        component_variances=_readonly_array(component_variances, dtype=np.float64),
        masses=_readonly_array(mass_array, dtype=np.float64),
        support_count=int(values.size),
        component_count=len(components),
        posterior_mean=posterior_mean,
        posterior_sd=float(np.sqrt(total)),
        interval_94=(
            weighted_empirical_quantile(values, weights, 0.03),
            weighted_empirical_quantile(values, weights, 0.97),
        ),
        probability_above_zero=probability_above,
        probability_below_zero=probability_below,
        within_component_variance=within,
        between_component_variance=between,
        total_variance=total,
        second_moment_variance=second,
        variance_residual=residual,
        projected_roundoff=bool(
            within_projected or between_projected or total_projected or second_projected
        ),
    )


def largest_remainder_allocation(masses: object, *, total: object) -> AllocationRecord:
    """Allocate a deterministic presentation count without sampling draws."""
    if isinstance(total, (bool, np.bool_)):
        raise TypeError("total must be a positive exact integer")
    try:
        total_integer = operator.index(total)
    except TypeError as error:
        raise TypeError("total must be a positive exact integer") from error
    total_integer = int(total_integer)
    if total_integer < 1:
        raise ValueError("total must be a positive exact integer")
    mass_array = np.asarray(masses, dtype=np.float64)
    if mass_array.ndim != 1 or mass_array.size < 1:
        raise ValueError("masses must be a nonempty one-dimensional array")
    if not np.isfinite(mass_array).all() or np.any(mass_array <= 0.0):
        raise ValueError("masses must be finite and strictly positive")
    if not np.isclose(float(np.sum(mass_array)), 1.0, atol=_MASS_ATOL, rtol=0.0):
        raise ValueError("masses must sum to one")
    quotas = mass_array * float(total_integer)
    allocations = np.floor(quotas).astype(np.int64)
    remainder = total_integer - int(np.sum(allocations))
    if remainder < 0 or remainder > mass_array.size:
        raise RuntimeError("largest-remainder floor total is inconsistent")
    fractional = quotas - allocations
    order = np.argsort(-fractional, kind="stable")
    allocations[order[:remainder]] += 1
    if int(np.sum(allocations)) != total_integer:
        raise RuntimeError("largest-remainder allocations do not sum to total")
    effective = allocations.astype(np.float64) / float(total_integer)
    signed = effective - mass_array
    absolute = np.abs(signed)
    maximum = float(np.max(absolute))
    if maximum > 1.0 / float(total_integer) + np.finfo(np.float64).eps:
        raise RuntimeError("largest-remainder mass error exceeds one allocation unit")
    return AllocationRecord(
        requested_total=total_integer,
        allocations=_readonly_array(allocations, dtype=np.int64),
        effective_masses=_readonly_array(effective, dtype=np.float64),
        signed_mass_errors=_readonly_array(signed, dtype=np.float64),
        absolute_mass_errors=_readonly_array(absolute, dtype=np.float64),
        max_absolute_mass_error=maximum,
    )


def _require_array(
    value: object, *, name: str, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact NumPy array")
    array = value
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return array


def _require_identity_array_equal(left: np.ndarray, right: np.ndarray, *, name: str) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"{name} shape or dtype differs across B2 and B3")
    if not _same_identity_sequence(left.ravel().tolist(), right.ravel().tolist()):
        raise ValueError(f"{name} differ across B2 and B3")


def _validate_b2(result: PartitionPosteriorResult) -> tuple[int, int, int, int]:
    # npass >= 1 matches the B2 gate exactly (2026-09-02 audit, S-09):
    # validate_partition_inputs never produces npass == 0, so revalidating a
    # weaker minimum here only masked hand-built inconsistent results.
    for name, minimum in (("n", 2), ("B", 1), ("K", 2), ("npass", 1)):
        value = getattr(result, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"B2 {name} must be an exact integer at least {minimum}")
    n, B, K = result.n, result.B, result.K
    observation_ids = _require_array(
        result.observation_ids, name="B2 observation IDs", shape=(n,), dtype=np.dtype(object)
    )
    reference_labels = _require_array(
        result.reference_labels,
        name="B2 reference labels",
        shape=(n,),
        dtype=np.dtype(object),
    )
    if not pd.Index(observation_ids).is_unique:
        raise ValueError("B2 observation IDs must be unique")
    order = tuple(result.reference_cluster_order)
    if len(order) != K or len({_identity_token(value) for value in order}) != K:
        raise ValueError("B2 cluster order must contain exactly K typed identities")
    stable_levels: list[object] = []
    seen: set[tuple[type[object], object]] = set()
    for value in reference_labels.tolist():
        token = _identity_token(value)
        if token not in seen:
            seen.add(token)
            stable_levels.append(value)
    if not _same_identity_sequence(stable_levels, order):
        raise ValueError("B2 reference labels do not preserve cluster order")

    _require_array(result.weights, name="B2 weights", shape=(B, n), dtype=np.dtype("float64"))
    _require_array(
        result.replay_starts,
        name="B2 replay starts",
        shape=(result.npass, K),
        dtype=np.dtype("int64"),
    )
    _require_array(
        result.raw_medoid_levels,
        name="B2 raw medoid levels",
        shape=(B, K),
        dtype=np.dtype("int64"),
    )
    _require_array(
        result.raw_partitions,
        name="B2 raw partitions",
        shape=(B, n),
        dtype=np.dtype("int64"),
    )
    aligned = _require_array(
        result.aligned_partitions,
        name="B2 aligned partitions",
        shape=(B, n),
        dtype=np.dtype(object),
    )
    if len(result.alignment_diagnostics) != B or any(
        type(item) is not _b2.AlignmentResult for item in result.alignment_diagnostics
    ):
        raise ValueError("B2 alignment diagnostics must contain one exact record per draw")
    for draw_index, (row, diagnostic) in enumerate(
        zip(aligned, result.alignment_diagnostics, strict=True)
    ):
        if not _same_identity_sequence(row.tolist(), diagnostic.aligned.tolist()):
            raise ValueError(f"B2 alignment diagnostic {draw_index} does not retain its row")
        row_tokens = {_identity_token(value) for value in row.tolist()}
        if row_tokens != {_identity_token(value) for value in order}:
            raise ValueError(f"B2 aligned partition {draw_index} has an empty or extra cluster")

    rows: list[list[object]] = []
    row_positions: dict[tuple[tuple[type[object], object], ...], int] = {}
    counts: list[int] = []
    inverse: list[int] = []
    for row in aligned:
        key = tuple(_identity_token(value) for value in row.tolist())
        position = row_positions.get(key)
        if position is None:
            position = len(rows)
            row_positions[key] = position
            rows.append(row.tolist())
            counts.append(0)
        counts[position] += 1
        inverse.append(position)
    U = len(rows)
    distinct = _require_array(
        result.distinct_partitions,
        name="B2 distinct partitions",
        shape=(U, n),
        dtype=np.dtype(object),
    )
    if not _same_identity_sequence(
        distinct.ravel().tolist(), np.asarray(rows, object).ravel().tolist()
    ):
        raise ValueError("B2 distinct partitions are not the stable exact compression")
    count_array = _require_array(
        result.counts, name="B2 counts", shape=(U,), dtype=np.dtype("int64")
    )
    if (
        np.any(count_array <= 0)
        or int(np.sum(count_array)) != B
        or not np.array_equal(count_array, np.asarray(counts, dtype=np.int64))
    ):
        raise ValueError("B2 counts must be positive, sum to B, and match compression")
    inverse_array = _require_array(
        result.inverse_index, name="B2 inverse index", shape=(B,), dtype=np.dtype("int64")
    )
    if not np.array_equal(inverse_array, np.asarray(inverse, dtype=np.int64)):
        raise ValueError("B2 inverse index does not reconstruct aligned partitions")
    reconstructed = distinct[inverse_array]
    if not _same_identity_sequence(reconstructed.ravel().tolist(), aligned.ravel().tolist()):
        raise ValueError("B2 inverse index reconstruction failed")
    masses = _require_array(result.masses, name="B2 masses", shape=(U,), dtype=np.dtype("float64"))
    expected_masses = count_array.astype(np.float64) / float(B)
    if not np.array_equal(masses, expected_masses):
        raise ValueError("B2 masses must equal counts divided by B exactly")
    mass_mcse = _require_array(
        result.mass_mcse, name="B2 mass MCSEs", shape=(U,), dtype=np.dtype("float64")
    )
    expected_mcse = np.sqrt(expected_masses * (1.0 - expected_masses) / float(B))
    if not np.array_equal(mass_mcse, expected_mcse):
        raise ValueError("B2 mass MCSEs do not match the empirical masses")
    numeric_shapes = {
        "membership probabilities": (result.membership_probabilities, (n, K)),
        "entropy": (result.entropy, (n,)),
        "co-clustering": (result.co_clustering, (n, n)),
    }
    for name, (values, shape) in numeric_shapes.items():
        array = _require_array(values, name=f"B2 {name}", shape=shape, dtype=np.dtype("float64"))
        if not np.isfinite(array).all():
            raise ValueError(f"B2 {name} must be finite")
    _require_array(
        result.ambiguity_flags,
        name="B2 ambiguity flags",
        shape=(B,),
        dtype=np.dtype("bool"),
    )
    _require_array(
        result.weighted_unweighted_difference_flags,
        name="B2 weighted/unweighted flags",
        shape=(B,),
        dtype=np.dtype("bool"),
    )
    return n, B, K, U


def _identity_token(value: object) -> tuple[type[object], object]:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError("identity values must be scalar and hashable") from error
    return type(value), value


def _coordinate_values(data_array: object, name: str, size: int) -> np.ndarray:
    if name in data_array.coords:
        values = np.asarray(data_array.coords[name].values)
    else:
        values = np.arange(size, dtype=np.int64)
    if values.ndim != 1 or values.shape != (size,):
        raise ValueError(f"posterior {name} coordinate has the wrong shape")
    return values


def _effect_target_metadata(metadata: _b3.EffectMetadata) -> EffectTargetMetadata:
    return EffectTargetMetadata(
        effect_identifier=metadata.effect_identifier,
        source_variable=metadata.source_variable,
        estimand_kind=metadata.estimand_kind,
        method=metadata.method,
        original_scale_units=metadata.original_scale_units,
        factor_reference=metadata.factor_reference,
        factor_comparison=metadata.factor_comparison,
        half_step=metadata.half_step,
        evaluated_offsets=metadata.evaluated_offsets,
        denominator=metadata.denominator,
        averaging_population=metadata.averaging_population,
        formula=metadata.formula,
        term_labels=metadata.term_labels,
        coefficient_coordinates=metadata.coefficient_coordinates,
        row_ids=metadata.row_ids,
        cluster_order=metadata.cluster_order,
        zero_sum_atol=metadata.zero_sum_atol,
        zero_sum_rtol=metadata.zero_sum_rtol,
    )


def _semantic_metadata_key(metadata: EffectTargetMetadata | None) -> object:
    if metadata is None:
        return None

    def identity(value: object | None) -> object:
        return None if value is None else _identity_token(value)

    return (
        metadata.effect_identifier,
        metadata.source_variable,
        metadata.estimand_kind,
        metadata.method,
        metadata.original_scale_units,
        identity(metadata.factor_reference),
        identity(metadata.factor_comparison),
        metadata.half_step,
        metadata.evaluated_offsets,
        metadata.denominator,
        metadata.averaging_population,
        metadata.formula,
        metadata.term_labels,
        metadata.coefficient_coordinates,
        tuple(_identity_token(value) for value in metadata.row_ids),
        tuple(_identity_token(value) for value in metadata.cluster_order),
        metadata.zero_sum_atol,
        metadata.zero_sum_rtol,
    )


def _descriptor_key(descriptor: TargetDescriptor) -> object:
    return (
        descriptor.target_index,
        descriptor.target_identifier,
        descriptor.target_kind,
        descriptor.coefficient_coordinate,
        _semantic_metadata_key(descriptor.effect_metadata),
        _identity_token(descriptor.cluster_coordinate),
        descriptor.cluster_index,
    )


def _inspect_component(
    component: _b3.ComponentFitRecord,
    conditional: ConditionalPosteriorResult,
    *,
    copy_arrays: bool,
) -> _ExtractedComponent:
    design = conditional.design
    sampler = conditional.sampler_configuration
    try:
        beta_data = component.posterior.posterior["beta_full"]
        probability_data = component.posterior.posterior["probability"]
    except (AttributeError, KeyError) as error:
        raise ValueError("passing B3 component posterior variables are incomplete") from error
    expected_beta_dims = ("chain", "draw", "coefficient", "cluster")
    expected_probability_dims = ("chain", "draw", "observation", "cluster")
    if tuple(beta_data.dims) != expected_beta_dims:
        raise ValueError("B3 beta_full dimensions are inconsistent")
    if tuple(probability_data.dims) != expected_probability_dims:
        raise ValueError("B3 probability dimensions are inconsistent")
    expected_beta_shape = (
        sampler.chains,
        sampler.draws,
        len(design.coefficient_coordinates),
        conditional.K,
    )
    beta = np.asarray(beta_data.values)
    probability = np.asarray(probability_data.values)
    if beta.shape != expected_beta_shape or not np.issubdtype(beta.dtype, np.floating):
        raise ValueError("B3 beta_full shape or dtype disagrees with sampler/design")
    if probability.shape != (
        sampler.chains,
        sampler.draws,
        conditional.n,
        conditional.K,
    ):
        raise ValueError("B3 probability shape disagrees with sampler/design")
    if not np.isfinite(beta).all() or not np.isfinite(probability).all():
        raise ValueError("B3 posterior arrays must be finite")
    if not np.array_equal(beta[..., 0], np.zeros_like(beta[..., 0])):
        raise ValueError("B3 structural-zero reference coefficient is not exactly zero")
    if (
        np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or not np.allclose(
            np.sum(probability, axis=-1), 1.0, atol=_SUMMARY_ATOL, rtol=_SUMMARY_RTOL
        )
    ):
        raise ValueError("B3 posterior probabilities do not satisfy the simplex")

    chain_coordinates = _coordinate_values(beta_data, "chain", sampler.chains)
    draw_coordinates = _coordinate_values(beta_data, "draw", sampler.draws)
    coefficient_coordinates = _coordinate_values(
        beta_data, "coefficient", len(design.coefficient_coordinates)
    )
    cluster_coordinates = _coordinate_values(beta_data, "cluster", conditional.K)
    if not _same_identity_sequence(
        coefficient_coordinates.tolist(), design.coefficient_coordinates
    ):
        raise ValueError("B3 coefficient targets are missing, extra, or reordered")
    if not _same_identity_sequence(cluster_coordinates.tolist(), design.cluster_order):
        raise ValueError("B3 coefficient cluster targets are missing, extra, or reordered")
    expected_coefficient_ids = tuple(
        f"coefficient:{coordinate}:cluster-{cluster_index}"
        for coordinate in design.coefficient_coordinates
        for cluster_index in range(1, conditional.K)
    )
    if tuple(item.target_identifier for item in component.coefficient_summaries) != (
        expected_coefficient_ids
    ):
        raise ValueError("B3 coefficient summaries are missing, extra, or reordered")

    descriptors: list[TargetDescriptor] = []
    target_draws: list[np.ndarray] = []
    for coefficient_index, coordinate in enumerate(design.coefficient_coordinates):
        for cluster_index in range(1, conditional.K):
            target_index = len(descriptors)
            descriptors.append(
                TargetDescriptor(
                    target_index=target_index,
                    target_identifier=(f"coefficient:{coordinate}:cluster-{cluster_index}"),
                    target_kind="coefficient",
                    coefficient_coordinate=coordinate,
                    effect_metadata=None,
                    cluster_coordinate=design.cluster_order[cluster_index],
                    cluster_index=cluster_index,
                )
            )
            target_draws.append(beta[:, :, coefficient_index, cluster_index])

    expected_effect_summary_ids: list[str] = []
    for effect in component.effects:
        if (
            type(effect) is not _b3.DrawwiseEffect
            or type(effect.metadata) is not _b3.EffectMetadata
        ):
            raise TypeError("B3 effect records have the wrong type")
        metadata = effect.metadata
        if metadata.component_identifier != component.component_identifier:
            raise ValueError("B3 effect component identity is inconsistent")
        if not _same_identity_sequence(metadata.chain_coordinates, chain_coordinates.tolist()):
            raise ValueError("B3 effect chain coordinates are inconsistent")
        if not _same_identity_sequence(metadata.draw_coordinates, draw_coordinates.tolist()):
            raise ValueError("B3 effect draw coordinates are inconsistent")
        if not _same_identity_sequence(metadata.cluster_order, design.cluster_order):
            raise ValueError("B3 effect cluster coordinates are inconsistent")
        if metadata.formula != conditional.formula:
            raise ValueError("B3 effect formula provenance is inconsistent")
        observation_effects = np.asarray(effect.observation_effects)
        ame_draws = np.asarray(effect.ame_draws)
        if observation_effects.shape != (
            sampler.chains,
            sampler.draws,
            conditional.n,
            conditional.K,
        ) or ame_draws.shape != (
            sampler.chains,
            sampler.draws,
            conditional.K,
        ):
            raise ValueError("B3 effect arrays disagree with sampler/design dimensions")
        if not np.isfinite(observation_effects).all() or not np.isfinite(ame_draws).all():
            raise ValueError("B3 effect arrays must be finite")
        if not np.allclose(
            np.sum(observation_effects, axis=-1),
            0.0,
            atol=metadata.zero_sum_atol,
            rtol=metadata.zero_sum_rtol,
        ) or not np.allclose(
            np.sum(ame_draws, axis=-1),
            0.0,
            atol=metadata.zero_sum_atol,
            rtol=metadata.zero_sum_rtol,
        ):
            raise ValueError("B3 effect arrays violate their zero-sum contract")
        semantic_metadata = _effect_target_metadata(metadata)
        for cluster_index in range(conditional.K):
            target_index = len(descriptors)
            identifier = f"effect:{metadata.effect_identifier}:cluster-{cluster_index}"
            expected_effect_summary_ids.append(identifier)
            descriptors.append(
                TargetDescriptor(
                    target_index=target_index,
                    target_identifier=identifier,
                    target_kind="effect",
                    coefficient_coordinate=None,
                    effect_metadata=semantic_metadata,
                    cluster_coordinate=design.cluster_order[cluster_index],
                    cluster_index=cluster_index,
                )
            )
            target_draws.append(ame_draws[:, :, cluster_index])
    if tuple(item.target_identifier for item in component.effect_summaries) != tuple(
        expected_effect_summary_ids
    ):
        raise ValueError("B3 effect targets are missing, extra, or reordered")

    try:
        replicated = component.posterior_predictive.posterior_predictive["cluster_id"]
    except (AttributeError, KeyError) as error:
        raise ValueError("B3 posterior-predictive data are incomplete") from error
    if tuple(replicated.dims) != ("chain", "draw", "observation") or replicated.shape != (
        sampler.chains,
        sampler.draws,
        conditional.n,
    ):
        raise ValueError("B3 posterior-predictive dimensions are inconsistent")
    if not np.issubdtype(np.asarray(replicated.values).dtype, np.integer):
        raise TypeError("B3 posterior-predictive labels must be integer codes")

    arrays = (
        tuple(_readonly_array(values, dtype=np.float64) for values in target_draws)
        if copy_arrays
        else tuple(np.asarray(values) for values in target_draws)
    )
    return _ExtractedComponent(
        component_index=component.component_index,
        component_identifier=component.component_identifier,
        descriptors=tuple(descriptors),
        draws=arrays,
        chain_coordinates=_readonly_array(chain_coordinates),
        draw_coordinates=_readonly_array(draw_coordinates),
    )


def _validate_b3(
    partition: PartitionPosteriorResult,
    result: ConditionalPosteriorResult,
    dimensions: tuple[int, int, int, int],
) -> None:
    n, B, K, U = dimensions
    for name, expected in zip(("n", "B", "K", "U"), dimensions, strict=True):
        if type(getattr(result, name)) is not int or getattr(result, name) != expected:
            raise ValueError(f"B3 {name} does not match B2 {name}")
    b3_observation_ids = _require_array(
        result.observation_ids,
        name="B3 observation IDs",
        shape=(n,),
        dtype=np.dtype(object),
    )
    b3_reference_labels = _require_array(
        result.reference_labels,
        name="B3 reference labels",
        shape=(n,),
        dtype=np.dtype(object),
    )
    _require_identity_array_equal(
        partition.observation_ids, b3_observation_ids, name="observation IDs"
    )
    _require_identity_array_equal(
        partition.reference_labels, b3_reference_labels, name="reference labels"
    )
    if not _same_identity_sequence(result.cluster_order, partition.reference_cluster_order):
        raise ValueError("B3 cluster order does not match B2")
    b3_distinct = _require_array(
        result.distinct_partitions,
        name="B3 distinct partitions",
        shape=(U, n),
        dtype=np.dtype(object),
    )
    _require_identity_array_equal(
        partition.distinct_partitions, b3_distinct, name="distinct partitions"
    )
    for name, dtype in (("counts", "int64"), ("masses", "float64"), ("mass_mcse", "float64")):
        b3_array = _require_array(
            getattr(result, name), name=f"B3 {name}", shape=(U,), dtype=np.dtype(dtype)
        )
        if not np.array_equal(getattr(partition, name), b3_array):
            label = "mass MCSEs" if name == "mass_mcse" else name
            raise ValueError(f"B3 {label} do not match B2")

    if type(result.design) is not _b3.PreparedConditionalDesign:
        raise TypeError("B3 design must be an exact PreparedConditionalDesign")
    design = result.design
    if result.formula != design.formula:
        raise ValueError("B3 formula does not match its design")
    if not _same_identity_sequence(design.row_ids, partition.observation_ids.tolist()):
        raise ValueError("B3 design row IDs do not match B2 observation IDs")
    if not _same_identity_sequence(design.cluster_order, partition.reference_cluster_order):
        raise ValueError("B3 design cluster order does not match B2")
    if design.reference_category is not design.cluster_order[0] and not _same_identity(
        design.reference_category, design.cluster_order[0]
    ):
        raise ValueError("B3 design reference category is inconsistent")
    width = len(design.coefficient_coordinates)
    matrix = _require_array(
        design.design_matrix,
        name="B3 design matrix",
        shape=(n, width),
        dtype=np.dtype("float64"),
    )
    if not np.isfinite(matrix).all() or design.design_rank != width:
        raise ValueError("B3 design matrix must be finite and full rank")
    response_codes = _require_array(
        design.response_codes,
        name="B3 response codes",
        shape=(n,),
        dtype=np.dtype("int64"),
    )
    expected_codes = np.asarray(
        [
            next(
                index
                for index, label in enumerate(design.cluster_order)
                if _same_identity(value, label)
            )
            for value in partition.reference_labels
        ],
        dtype=np.int64,
    )
    if not np.array_equal(response_codes, expected_codes):
        raise ValueError("B3 design response codes do not match B2 reference labels")
    _b3.validate_private_envelope(n=n, K=K, B=B, U=U, design_width=width)

    if type(result.prior) is not _b3.PriorSpecification:
        raise TypeError("B3 prior must be an exact PriorSpecification")
    if result.prior.coefficient_coordinates != design.coefficient_coordinates:
        raise ValueError("B3 prior coefficient coordinates do not match the design")
    if not _same_identity_sequence(
        result.prior.nonreference_cluster_coordinates, design.cluster_order[1:]
    ):
        raise ValueError("B3 prior cluster coordinates do not match the design")
    for name in ("mean", "sigma"):
        values = _require_array(
            getattr(result.prior, name),
            name=f"B3 prior {name}",
            shape=(width, K - 1),
            dtype=np.dtype("float64"),
        )
        if not np.isfinite(values).all() or (name == "sigma" and np.any(values <= 0.0)):
            raise ValueError(f"B3 prior {name} is invalid")

    sampler = result.sampler_configuration
    if type(sampler) is not _b3.SamplerConfiguration:
        raise TypeError("B3 sampler configuration has the wrong type")
    if type(sampler.chains) is not int or sampler.chains < 1:
        raise ValueError("B3 sampler chains must be positive")
    if type(sampler.draws) is not int or sampler.draws < 1:
        raise ValueError("B3 sampler draws must be positive")
    if type(result.seed_manifest) is not _b3.B3SeedManifest:
        raise TypeError("B3 seed manifest has the wrong type")
    if (
        result.seed_manifest.root_seed != result.root_seed
        or len(result.seed_manifest.components) != U
    ):
        raise ValueError("B3 seed manifest does not match the result")
    if type(result.prior_predictive) is not _b3.PriorPredictiveRecord:
        raise TypeError("B3 prior predictive record has the wrong type")
    if result.prior_predictive.status == "pass" and result.prior_predictive.predictive is None:
        raise ValueError("passing B3 prior predictive data are missing")
    if len(result.components) != U:
        raise ValueError("B3 components must contain exactly U records")
    for index, component in enumerate(result.components):
        if type(component) is not _b3.ComponentFitRecord:
            raise TypeError("B3 component has the wrong type")
        if component.component_index != index:
            raise ValueError("B3 component index/order is inconsistent")
        if (
            component.seeds is not result.seed_manifest.components[index]
            and component.seeds != result.seed_manifest.components[index]
        ):
            raise ValueError("B3 component seed manifest is inconsistent")
        if component.component_identifier != result.seed_manifest.components[index].name:
            raise ValueError("B3 component identifier is inconsistent")
        if not _same_identity_sequence(
            component.partition.tolist(), partition.distinct_partitions[index].tolist()
        ):
            raise ValueError("B3 component partition does not match B2")
        if component.count != int(partition.counts[index]):
            raise ValueError("B3 component count does not match B2")
        if component.mass != float(partition.masses[index]):
            raise ValueError("B3 component mass does not match B2")
        if component.mass_mcse != float(partition.mass_mcse[index]):
            raise ValueError("B3 component mass MCSE does not match B2")
        if component.status == "pass":
            if (
                component.failure_reasons
                or component.posterior is None
                or component.posterior_predictive is None
            ):
                raise ValueError("passing B3 component is incomplete")
            if (
                type(component.sampler_diagnostics) is not _b3.SamplerDiagnosticRecord
                or component.sampler_diagnostics.status != "pass"
            ):
                raise ValueError("passing B3 component lacks passing sampler diagnostics")
            if (
                type(component.posterior_predictive_diagnostics)
                is not _b3.PosteriorPredictiveRecord
                or component.posterior_predictive_diagnostics.status != "pass"
            ):
                raise ValueError("passing B3 component lacks passing predictive diagnostics")
            if any(warning.material for warning in component.warnings):
                raise ValueError("passing B3 component retains a material warning")
        elif component.status != "fail":
            raise ValueError("B3 component status must be pass or fail")
    expected_failures = tuple(
        component for component in result.components if component.status != "pass"
    )
    if len(result.failures) != len(expected_failures) or any(
        actual is not expected
        for actual, expected in zip(result.failures, expected_failures, strict=True)
    ):
        raise ValueError("B3 failure records do not retain the failed component identities")
    expected_status = (
        "pass" if result.prior_predictive.status == "pass" and not expected_failures else "fail"
    )
    if result.status != expected_status:
        raise ValueError("B3 result status is inconsistent with its diagnostics and failures")


def validate_phase_inputs(
    partition_posterior: object, conditional_posterior: object
) -> tuple[int, int, int, int]:
    """Validate the exact private phase types before B4 integration."""
    if type(partition_posterior) is not PartitionPosteriorResult:
        raise TypeError("partition_posterior must have exact type PartitionPosteriorResult")
    if type(conditional_posterior) is not ConditionalPosteriorResult:
        raise TypeError("conditional_posterior must have exact type ConditionalPosteriorResult")
    dimensions = _validate_b2(partition_posterior)
    _validate_b3(partition_posterior, conditional_posterior, dimensions)
    return dimensions


def _combined_warnings(
    partition: PartitionPosteriorResult, conditional: ConditionalPosteriorResult
) -> tuple[object, ...]:
    records: list[object] = list(partition.warnings)
    records.extend(conditional.prior_predictive.warnings)
    for component in conditional.components:
        records.extend(component.warnings)
    return tuple(records)


def _result_common(
    partition: PartitionPosteriorResult,
    conditional: ConditionalPosteriorResult,
) -> dict[str, object]:
    return {
        "partition_posterior": partition,
        "conditional_posterior": conditional,
        "n": partition.n,
        "B": partition.B,
        "K": partition.K,
        "U": len(partition.distinct_partitions),
        "formula": conditional.formula,
        "observation_ids": _readonly_array(partition.observation_ids, dtype=object),
        "reference_labels": _readonly_array(partition.reference_labels, dtype=object),
        "cluster_order": partition.reference_cluster_order,
        "masses": _readonly_array(partition.masses, dtype=np.float64),
        "design": conditional.design,
        "prior": conditional.prior,
        "software_versions": conditional.software_versions,
        "b2_root_seed": partition.root_seed,
        "b3_root_seed": conditional.root_seed,
        "schema_name": PRIVATE_SCHEMA_NAME,
        "schema_version": PRIVATE_SCHEMA_VERSION,
    }


def _integrated_summary(
    descriptor: TargetDescriptor, scalar: ScalarMixtureSummary
) -> IntegratedTargetSummary:
    return IntegratedTargetSummary(
        descriptor=descriptor,
        support_count=scalar.support_count,
        component_count=scalar.component_count,
        posterior_mean=scalar.posterior_mean,
        posterior_sd=scalar.posterior_sd,
        interval_94=scalar.interval_94,
        probability_above_zero=scalar.probability_above_zero,
        probability_below_zero=scalar.probability_below_zero,
        within_component_variance=scalar.within_component_variance,
        between_component_variance=scalar.between_component_variance,
        total_variance=scalar.total_variance,
        second_moment_variance=scalar.second_moment_variance,
        variance_residual=scalar.variance_residual,
        component_means=_readonly_array(scalar.component_means, dtype=np.float64),
        component_variances=_readonly_array(scalar.component_variances, dtype=np.float64),
        component_masses=_readonly_array(scalar.masses, dtype=np.float64),
        component_draw_counts=_readonly_array(scalar.component_draw_counts, dtype=np.int64),
        projected_roundoff=scalar.projected_roundoff,
    )


def integrate_conditional_posteriors(
    partition_posterior: PartitionPosteriorResult,
    conditional_posterior: ConditionalPosteriorResult,
    *,
    parallel: bool = False,
    n_jobs: int | None = None,
) -> BayesRARCATResult:
    """Integrate a completed coherent B2/B3 pair without fitting or mutation.

    Direct callers receive the same strict ``parallel`` and ``n_jobs``
    validation as the public facade.
    """
    dimensions = validate_phase_inputs(partition_posterior, conditional_posterior)
    resolved_workers = validate_parallel_controls(
        parallel,
        n_jobs,
        n_tasks=dimensions[3],
    )
    execution_mode = (
        "parallel" if parallel and resolved_workers > 1 and dimensions[3] > 1 else "serial"
    )
    common = _result_common(partition_posterior, conditional_posterior)
    warnings = _combined_warnings(partition_posterior, conditional_posterior)
    if conditional_posterior.status != "pass":
        return BayesRARCATResult(
            status="blocked",
            failures=conditional_posterior.failures,
            warnings=warnings,
            supports=(),
            coefficient_summaries=(),
            effect_summaries=(),
            allocation=None,
            execution_mode=execution_mode,
            resolved_workers=resolved_workers,
            **common,
        )

    inspected = tuple(
        _inspect_component(component, conditional_posterior, copy_arrays=False)
        for component in conditional_posterior.components
    )
    reference_keys = tuple(_descriptor_key(item) for item in inspected[0].descriptors)
    for component in inspected[1:]:
        if tuple(_descriptor_key(item) for item in component.descriptors) != reference_keys:
            if len(component.descriptors) != len(inspected[0].descriptors):
                raise ValueError("B3 component effect targets are missing or extra")
            raise ValueError("B3 component targets or metadata are semantically inconsistent")

    def extract(index: int) -> _ExtractedComponent:
        return _inspect_component(
            conditional_posterior.components[index],
            conditional_posterior,
            copy_arrays=True,
        )

    extracted = run_task_batch(
        extract,
        list(range(dimensions[3])),
        parallel=parallel,
        n_jobs=n_jobs,
        progressbar=False,
        desc="Bayesian RARCAT B4 extraction",
        executor_kind="thread",
    )
    if [item.component_index for item in extracted] != list(range(dimensions[3])):
        raise RuntimeError("B4 component extraction order changed")

    supports: list[WeightedTargetSupport] = []
    summaries: list[IntegratedTargetSummary] = []
    component_identifiers = tuple(item.component_identifier for item in extracted)
    masses = partition_posterior.masses
    for target_index, descriptor in enumerate(extracted[0].descriptors):
        component_draws = tuple(item.draws[target_index] for item in extracted)
        scalar = summarize_scalar_mixture(component_draws, masses)
        chain_coordinates = np.concatenate(
            [np.repeat(item.chain_coordinates, item.draw_coordinates.size) for item in extracted]
        )
        draw_coordinates = np.concatenate(
            [np.tile(item.draw_coordinates, item.chain_coordinates.size) for item in extracted]
        )
        offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(scalar.component_draw_counts)]
        )
        supports.append(
            WeightedTargetSupport(
                descriptor=descriptor,
                values=_readonly_array(scalar.values, dtype=np.float64),
                weights=_readonly_array(scalar.weights, dtype=np.float64),
                component_indices=_readonly_array(scalar.component_indices, dtype=np.int64),
                component_identifiers=component_identifiers,
                chain_coordinates=_readonly_array(chain_coordinates, dtype=object),
                draw_coordinates=_readonly_array(draw_coordinates, dtype=object),
                component_offsets=_readonly_array(offsets, dtype=np.int64),
            )
        )
        summaries.append(_integrated_summary(descriptor, scalar))

    coefficient_summaries = tuple(
        summary for summary in summaries if summary.descriptor.target_kind == "coefficient"
    )
    effect_summaries = tuple(
        summary for summary in summaries if summary.descriptor.target_kind == "effect"
    )
    effect_ids = tuple(
        dict.fromkeys(
            summary.descriptor.effect_metadata.effect_identifier
            for summary in effect_summaries
            if summary.descriptor.effect_metadata is not None
        )
    )
    for effect_identifier in effect_ids:
        effect_group = [
            summary
            for summary in effect_summaries
            if summary.descriptor.effect_metadata.effect_identifier == effect_identifier
        ]
        if len(effect_group) != dimensions[2]:
            raise ValueError("integrated effect does not retain every cluster")
        metadata = effect_group[0].descriptor.effect_metadata
        if not np.isclose(
            sum(summary.posterior_mean for summary in effect_group),
            0.0,
            atol=metadata.zero_sum_atol,
            rtol=metadata.zero_sum_rtol,
        ):
            raise ValueError("integrated posterior-mean effect violates zero-sum identity")

    allocation = largest_remainder_allocation(masses, total=_ALLOCATION_TOTAL)
    if np.any(allocation.allocations <= 0):
        raise RuntimeError("B4 allocation omitted a positive supported component")
    return BayesRARCATResult(
        status="pass",
        failures=(),
        warnings=warnings,
        supports=tuple(supports),
        coefficient_summaries=coefficient_summaries,
        effect_summaries=effect_summaries,
        allocation=allocation,
        execution_mode=execution_mode,
        resolved_workers=resolved_workers,
        **common,
    )


def _require_exact_array_equal(left: np.ndarray, right: np.ndarray, *, name: str) -> None:
    if left.dtype == object or right.dtype == object:
        if left.shape != right.shape or not _same_identity_sequence(
            left.ravel().tolist(), right.ravel().tolist()
        ):
            raise ValueError(f"B4 {name} are not coherent")
    elif not np.array_equal(left, right):
        raise ValueError(f"B4 {name} are not coherent")
    if left.flags.writeable:
        raise ValueError(f"B4 {name} must be read-only")


def validate_bayes_rarcat_result(result: object) -> BayesRARCATResult:
    """Rerun phase and mixture invariants for an in-memory or loaded result."""
    if type(result) is not BayesRARCATResult:
        raise TypeError("result must have exact type BayesRARCATResult")
    dimensions = validate_phase_inputs(result.partition_posterior, result.conditional_posterior)
    if (result.n, result.B, result.K, result.U) != dimensions:
        raise ValueError("B4 dimensions are not coherent with B2/B3")
    if result.formula != result.conditional_posterior.formula:
        raise ValueError("B4 formula is not coherent with B3")
    if not _same_identity_sequence(
        result.cluster_order, result.partition_posterior.reference_cluster_order
    ):
        raise ValueError("B4 cluster order is not coherent with B2")
    for attribute, name, nested in (
        ("observation_ids", "observation IDs", result.partition_posterior.observation_ids),
        ("reference_labels", "reference labels", result.partition_posterior.reference_labels),
        ("masses", "masses", result.partition_posterior.masses),
    ):
        _require_exact_array_equal(getattr(result, attribute), nested, name=name)
    if result.design is not result.conditional_posterior.design:
        raise ValueError("B4 design must be the authoritative nested B3 design")
    if result.prior is not result.conditional_posterior.prior:
        raise ValueError("B4 prior must be the authoritative nested B3 prior")
    if result.software_versions != result.conditional_posterior.software_versions:
        raise ValueError("B4 software versions are not coherent with B3")
    if result.b2_root_seed != result.partition_posterior.root_seed:
        raise ValueError("B4 B2 root seed is not coherent")
    if result.b3_root_seed != result.conditional_posterior.root_seed:
        raise ValueError("B4 B3 root seed is not coherent")
    if result.schema_name != PRIVATE_SCHEMA_NAME or result.schema_version != PRIVATE_SCHEMA_VERSION:
        raise ValueError("B4 private schema name/version is invalid")
    if result.execution_mode not in {"serial", "parallel"}:
        raise ValueError("B4 execution mode is invalid")
    if type(result.resolved_workers) is not int or result.resolved_workers < 1:
        raise ValueError("B4 resolved worker count is invalid")
    if result.status == "blocked":
        if result.conditional_posterior.status == "pass":
            raise ValueError("blocked B4 result has a passing B3 input")
        if (
            result.supports
            or result.coefficient_summaries
            or result.effect_summaries
            or result.allocation is not None
        ):
            raise ValueError("blocked B4 result contains integrated output")
        if result.failures != result.conditional_posterior.failures:
            raise ValueError("blocked B4 failure records are not coherent")
        return result
    if result.status != "pass" or result.conditional_posterior.status != "pass":
        raise ValueError("B4 status is not coherent with B3")
    if result.failures:
        raise ValueError("passing B4 result cannot contain failures")

    expected = integrate_conditional_posteriors(
        result.partition_posterior,
        result.conditional_posterior,
        parallel=result.execution_mode == "parallel",
        n_jobs=result.resolved_workers,
    )
    if len(result.supports) != len(expected.supports):
        raise ValueError("B4 support target count is not coherent")
    for actual, reference in zip(result.supports, expected.supports, strict=True):
        if _descriptor_key(actual.descriptor) != _descriptor_key(reference.descriptor):
            raise ValueError("B4 support descriptor is not coherent")
        if actual.component_identifiers != reference.component_identifiers:
            raise ValueError("B4 support component identities are not coherent")
        for name in (
            "values",
            "weights",
            "component_indices",
            "chain_coordinates",
            "draw_coordinates",
            "component_offsets",
        ):
            _require_exact_array_equal(
                getattr(actual, name), getattr(reference, name), name=f"support {name}"
            )
    actual_summaries = (*result.coefficient_summaries, *result.effect_summaries)
    expected_summaries = (*expected.coefficient_summaries, *expected.effect_summaries)
    if len(actual_summaries) != len(expected_summaries):
        raise ValueError("B4 summary target count is not coherent")
    scalar_fields = (
        "support_count",
        "component_count",
        "posterior_mean",
        "posterior_sd",
        "interval_94",
        "probability_above_zero",
        "probability_below_zero",
        "within_component_variance",
        "between_component_variance",
        "total_variance",
        "second_moment_variance",
        "variance_residual",
        "projected_roundoff",
    )
    for actual, reference in zip(actual_summaries, expected_summaries, strict=True):
        if _descriptor_key(actual.descriptor) != _descriptor_key(reference.descriptor) or any(
            getattr(actual, name) != getattr(reference, name) for name in scalar_fields
        ):
            raise ValueError("B4 integrated summary is not coherent")
        for name in (
            "component_means",
            "component_variances",
            "component_masses",
            "component_draw_counts",
        ):
            _require_exact_array_equal(
                getattr(actual, name), getattr(reference, name), name=f"summary {name}"
            )
    if result.allocation is None:
        raise ValueError("passing B4 allocation is missing")
    for name in (
        "allocations",
        "effective_masses",
        "signed_mass_errors",
        "absolute_mass_errors",
    ):
        _require_exact_array_equal(
            getattr(result.allocation, name),
            getattr(expected.allocation, name),
            name=f"allocation {name}",
        )
    if (
        result.allocation.requested_total != expected.allocation.requested_total
        or result.allocation.max_absolute_mass_error != expected.allocation.max_absolute_mass_error
    ):
        raise ValueError("B4 allocation metadata are not coherent")
    return result


def _require_passing_result(result: object) -> BayesRARCATResult:
    if type(result) is not BayesRARCATResult:
        raise TypeError("result must have exact type BayesRARCATResult")
    if result.status != "pass":
        raise ValueError("semantic plot data require a passing B4 result")
    return result


def component_mass_data(result: BayesRARCATResult) -> pd.DataFrame:
    """Return fresh deterministic component-mass semantic plot data.

    Not reached from ``bayes_rarcat``: like the report presenter that calls
    it, this is intentional private API for the verification suite and
    callers outside the public path (2026-09-02 audit, S-07).
    """
    result = _require_passing_result(result)
    support = result.supports[0]
    draw_counts = np.diff(support.component_offsets)
    weight_sums = np.asarray(
        [np.sum(support.weights[support.component_indices == index]) for index in range(result.U)]
    )
    columns = [
        "component_index",
        "component_identifier",
        "count",
        "mass",
        "mass_mcse",
        "conditional_draw_count",
        "support_weight_sum",
        "allocation",
        "effective_allocation_mass",
        "allocation_mass_error",
    ]
    records = [
        {
            "component_index": index,
            "component_identifier": support.component_identifiers[index],
            "count": int(result.partition_posterior.counts[index]),
            "mass": float(result.masses[index]),
            "mass_mcse": float(result.partition_posterior.mass_mcse[index]),
            "conditional_draw_count": int(draw_counts[index]),
            "support_weight_sum": float(weight_sums[index]),
            "allocation": int(result.allocation.allocations[index]),
            "effective_allocation_mass": float(result.allocation.effective_masses[index]),
            "allocation_mass_error": float(result.allocation.signed_mass_errors[index]),
        }
        for index in range(result.U)
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _summary_record(summary: IntegratedTargetSummary) -> dict[str, object]:
    descriptor = summary.descriptor
    metadata = descriptor.effect_metadata
    return {
        "target_order": descriptor.target_index,
        "target_kind": descriptor.target_kind,
        "target_identifier": descriptor.target_identifier,
        "coefficient_coordinate": descriptor.coefficient_coordinate,
        "effect_identifier": None if metadata is None else metadata.effect_identifier,
        "cluster_coordinate": descriptor.cluster_coordinate,
        "cluster_index": descriptor.cluster_index,
        "posterior_mean": summary.posterior_mean,
        "posterior_sd": summary.posterior_sd,
        "interval_lower": summary.interval_94[0],
        "interval_upper": summary.interval_94[1],
        "probability_above_zero": summary.probability_above_zero,
        "probability_below_zero": summary.probability_below_zero,
        "within_component_variance": summary.within_component_variance,
        "between_component_variance": summary.between_component_variance,
        "total_variance": summary.total_variance,
        "second_moment_variance": summary.second_moment_variance,
        "variance_residual": summary.variance_residual,
        "support_count": summary.support_count,
        "component_count": summary.component_count,
    }


def integrated_target_data(result: BayesRARCATResult) -> pd.DataFrame:
    """Return fresh integrated coefficient/effect summary data in stored order.

    Not reached from ``bayes_rarcat``: like the report presenter that calls
    it, this is intentional private API for the verification suite and
    callers outside the public path (2026-09-02 audit, S-07).
    """
    result = _require_passing_result(result)
    summaries = (*result.coefficient_summaries, *result.effect_summaries)
    columns = list(_summary_record(summaries[0]))
    frame = pd.DataFrame.from_records(
        [_summary_record(summary) for summary in summaries], columns=columns
    )
    frame["cluster_coordinate"] = pd.Series(
        [summary.descriptor.cluster_coordinate for summary in summaries], dtype=object
    )
    return frame


def component_target_data(result: BayesRARCATResult) -> pd.DataFrame:
    """Return one stable row per target and component decomposition term.

    Not reached from ``bayes_rarcat``: this is intentional private API, called
    by the verification suite and by callers outside the public path, and it
    has no reference in ``src/`` other than this definition. See finding F-09
    of the 2026-08-25 audit, which re-adjudicated the earlier reference counts.
    """
    result = _require_passing_result(result)
    summaries = (*result.coefficient_summaries, *result.effect_summaries)
    records: list[dict[str, object]] = []
    for summary in summaries:
        descriptor = summary.descriptor
        for component_index in range(result.U):
            mass = float(summary.component_masses[component_index])
            mean = float(summary.component_means[component_index])
            variance = float(summary.component_variances[component_index])
            records.append(
                {
                    "target_order": descriptor.target_index,
                    "target_kind": descriptor.target_kind,
                    "target_identifier": descriptor.target_identifier,
                    "cluster_coordinate": descriptor.cluster_coordinate,
                    "cluster_index": descriptor.cluster_index,
                    "component_index": component_index,
                    "component_identifier": result.supports[
                        descriptor.target_index
                    ].component_identifiers[component_index],
                    "mass": mass,
                    "draw_count": int(summary.component_draw_counts[component_index]),
                    "component_mean": mean,
                    "component_population_variance": variance,
                    "weighted_mean_contribution": mass * mean,
                    "weighted_within_variance_contribution": mass * variance,
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["cluster_coordinate"] = pd.Series(
        [record["cluster_coordinate"] for record in records], dtype=object
    )
    return frame


def effect_data(result: BayesRARCATResult) -> pd.DataFrame:
    """Return fresh effect metadata plus every integrated decomposition field.

    Not reached from ``bayes_rarcat``: this is intentional private API, called
    by the verification suite and by callers outside the public path, and it
    has no reference in ``src/`` other than this definition. See finding F-09
    of the 2026-08-25 audit, which re-adjudicated the earlier reference counts.
    """
    result = _require_passing_result(result)
    records: list[dict[str, object]] = []
    for summary in result.effect_summaries:
        metadata = summary.descriptor.effect_metadata
        record = {
            "effect_source": metadata.source_variable,
            "effect_identifier": metadata.effect_identifier,
            "estimand_kind": metadata.estimand_kind,
            "method": metadata.method,
            "factor_reference": metadata.factor_reference,
            "factor_comparison": metadata.factor_comparison,
            "half_step": metadata.half_step,
            "original_scale_units": metadata.original_scale_units,
            "averaging_population": metadata.averaging_population,
            **_summary_record(summary),
        }
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    for column in ("factor_reference", "factor_comparison", "cluster_coordinate"):
        frame[column] = pd.Series([record[column] for record in records], dtype=object)
    return frame


def format_bayes_rarcat_result(result: BayesRARCATResult) -> str:
    """Format a deterministic headless private summary without global options.

    Not reached from ``bayes_rarcat``: this is intentional private API, called
    by the verification suite and by callers outside the public path, and it
    has no reference in ``src/`` other than this definition. See finding F-09
    of the 2026-08-25 audit, which re-adjudicated the earlier reference counts.
    """
    if type(result) is not BayesRARCATResult:
        raise TypeError("result must have exact type BayesRARCATResult")
    lines = [
        f"Bayesian RARCAT B4 status: {result.status}",
        f"dimensions: n={result.n} B={result.B} K={result.K} U={result.U}",
        f"formula: {result.formula}",
        f"failures: {len(result.failures)}",
        f"targets: {len(result.supports)}",
    ]
    if result.status != "pass":
        lines.append("integration: blocked; component masses were retained unchanged")
        return "\n".join(lines)
    mass_rows = [
        (f"{row.component_identifier}: mass={row.mass:.12g}, " f"allocation={row.allocation}")
        for row in component_mass_data(result).itertuples(index=False)
    ]

    def format_float(value: float) -> str:
        return f"{value:.12g}"

    table = integrated_target_data(result)[
        [
            "target_identifier",
            "cluster_coordinate",
            "posterior_mean",
            "posterior_sd",
            "interval_lower",
            "interval_upper",
            "within_component_variance",
            "between_component_variance",
            "total_variance",
        ]
    ].copy(deep=True)
    table["cluster_coordinate"] = table["cluster_coordinate"].map(repr)
    lines.extend(("components:", *mass_rows, "integrated targets:"))
    lines.append(table.to_string(index=False, float_format=format_float))
    return "\n".join(lines)

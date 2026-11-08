from __future__ import annotations

import math
import sys
import warnings
from decimal import ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from traminex.diss import weighted_pairwise_sum

from ._quality_contract import has_cluster_weight_below_one
from ._validation import coerce_public_distance_matrix, validate_distance_matrix

_R_CLUSTER_QUALITY_NAMES = (
    "PBC",
    "HG",
    "HGSD",
    "ASW",
    "ASWw",
    "CH",
    "R2",
    "CHsq",
    "R2sq",
    "HC",
)


def _factorize_quality_labels(labels: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
    """Mirror R ``factor`` ordering while retaining public cluster labels."""
    if isinstance(labels, pd.Series) and isinstance(labels.dtype, pd.CategoricalDtype):
        categorical = labels.array
    elif isinstance(labels, pd.Categorical):
        categorical = labels
    else:
        categorical = None

    if categorical is not None:
        values = np.asarray(categorical, dtype=object)
        if values.ndim != 1:
            raise ValueError("clustering must be a 1D sequence")
        if np.any(pd.isna(values)):
            raise ValueError("clustering must not contain missing values")
        used = set(values.tolist())
        levels = [value for value in categorical.categories.tolist() if value in used]
        mapping = {value: index for index, value in enumerate(levels)}
        return np.asarray([mapping[value] for value in values], dtype=int), levels

    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("clustering must be a 1D sequence")
    if arr.size == 0:
        return np.asarray([], dtype=int), []
    if np.any(pd.isna(arr)):
        raise ValueError("clustering must not contain missing values")

    if arr.dtype.kind in "biufc" and arr.dtype.kind != "c":
        levels_array = np.unique(arr)
        values = arr
    else:
        # R atomic vectors have one type. Mixed/object inputs therefore follow
        # R's character coercion and lexical factor-level ordering.
        values = np.asarray([str(value) for value in arr], dtype=object)
        levels_array = np.unique(values)
    levels = [
        value.item() if isinstance(value, np.generic) else value for value in levels_array.tolist()
    ]
    mapping = {value: index for index, value in enumerate(levels)}
    return np.asarray([mapping[value] for value in values], dtype=int), levels


def _normalize_quality_labels(labels: Sequence[Any]) -> np.ndarray:
    """Normalize arbitrary cluster labels to contiguous 0-based integers."""
    normalized, _levels = _factorize_quality_labels(labels)
    return normalized


def _coerce_quality_distance(diss: np.ndarray, *, expected_n: int | None = None) -> np.ndarray:
    """Apply the shared public distance contract, then check the shape agrees.

    The contract — only the lower triangle is data, as R's ``as.dist`` reads a
    matrix — lives in
    :func:`weightedcluster._validation.coerce_public_distance_matrix`
    so that ``wcClusterQuality``, ``wcSilhouetteObs``, ``as_clustrange`` and
    ``wcKMedoids`` cannot drift apart (findings F-06 and F-07).
    """
    D = coerce_public_distance_matrix(diss, name="diss")
    if expected_n is not None and D.shape[0] != int(expected_n):
        raise ValueError("different number of elements in diss, clustering and/or weights")
    return D


def _validate_quality_partition(labels: np.ndarray, cluster_levels: Sequence[Any]) -> None:
    if len(cluster_levels) < 2:
        raise ValueError("The clustering should have at least two different values")
    if labels.ndim != 1:
        raise ValueError("clustering must be a 1D sequence")


def _coerce_quality_weights(
    weights: Optional[Sequence[float]], *, expected_n: int
) -> np.ndarray | None:
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape[0] != expected_n:
        raise ValueError("different number of elements in diss, clustering and/or weights")
    if np.any(np.isnan(values)):
        # The R source reaches the same rejection through its weighted-size
        # conditional ("missing value where TRUE/FALSE needed").
        raise ValueError("weights must not contain missing values")
    return values


_PositiveTuple = tuple[float, int]
_ZERO_POSITIVE_TUPLE: _PositiveTuple = (0.0, 0)


def _positive_float_tuple(value: float) -> _PositiveTuple:
    """Represent a finite non-negative binary64 as ``mantissa * 2**exponent``."""
    if value == 0.0:
        return _ZERO_POSITIVE_TUPLE
    return math.frexp(float(value))


def _positive_sum_tuples(values: Sequence[_PositiveTuple]) -> _PositiveTuple:
    """Sum non-negative exponent tuples without premature IEEE range loss."""
    positive = [value for value in values if value[0] > 0.0]
    if not positive:
        return _ZERO_POSITIVE_TUPLE
    maximum_exponent = max(exponent for _mantissa, exponent in positive)
    scaled_total = math.fsum(
        math.ldexp(mantissa, exponent - maximum_exponent) for mantissa, exponent in positive
    )
    mantissa, adjustment = math.frexp(scaled_total)
    return mantissa, maximum_exponent + adjustment


def _positive_sum(values: Sequence[float] | np.ndarray) -> _PositiveTuple:
    return _positive_sum_tuples(
        [_positive_float_tuple(float(value)) for value in np.asarray(values).reshape(-1)]
    )


def _positive_terms_need_exact_sum(values: Sequence[_PositiveTuple]) -> bool:
    """Whether a positive addend can fall below the accumulator's ulp window."""
    positive = [value for value in values if value[0] > 0.0]
    if len(positive) < 2:
        return False
    exponents = [value[1] for value in positive]
    rounding_window = 52 - (len(positive) - 1).bit_length()
    return max(exponents) - min(exponents) >= rounding_window


def _positive_product_tuple(*values: float) -> _PositiveTuple:
    """Multiply exact-positive finite values without materializing the product."""
    if any(value == 0.0 for value in values):
        return _ZERO_POSITIVE_TUPLE
    mantissa = 1.0
    exponent = 0
    for value in values:
        value_mantissa, value_exponent = math.frexp(float(value))
        mantissa *= value_mantissa
        mantissa, adjustment = math.frexp(mantissa)
        exponent += value_exponent + adjustment
    return mantissa, exponent


def _positive_power_product_tuple(base: float, power: float, multiplier: float) -> _PositiveTuple:
    """Form ``multiplier * base**power`` before binary64 underflow can erase it."""
    return _positive_tuple_power_product_tuple(
        _positive_float_tuple(base),
        power,
        _positive_float_tuple(multiplier),
    )


def _positive_tuple_power_product_tuple(
    base: _PositiveTuple,
    power: float,
    multiplier: _PositiveTuple,
) -> _PositiveTuple:
    """Form ``multiplier * base**power`` for a conceptual positive dyadic."""
    if base[0] == 0.0 or multiplier[0] == 0.0:
        return _ZERO_POSITIVE_TUPLE
    if base == _positive_float_tuple(1.0):
        return multiplier
    with localcontext() as context:
        context.prec = 80
        base_log2 = Decimal(base[1]) + Decimal.from_float(math.log2(base[0]))
        powered_log2 = base_log2 * Decimal.from_float(power)
        integer_decimal = powered_log2.to_integral_value(rounding=ROUND_FLOOR)
        integer_exponent = int(integer_decimal)
        fractional_exponent = float(powered_log2 - integer_decimal)
    mantissa = multiplier[0] * math.exp2(fractional_exponent)
    mantissa, adjustment = math.frexp(mantissa)
    return mantissa, multiplier[1] + integer_exponent + adjustment


def _positive_tuple_product(
    left: _PositiveTuple,
    right: _PositiveTuple,
) -> _PositiveTuple:
    if left[0] == 0.0 or right[0] == 0.0:
        return _ZERO_POSITIVE_TUPLE
    mantissa, adjustment = math.frexp(left[0] * right[0])
    return mantissa, left[1] + right[1] + adjustment


def _positive_tuple_divide(
    numerator: _PositiveTuple,
    denominator: _PositiveTuple,
) -> _PositiveTuple:
    if numerator[0] == 0.0:
        return _ZERO_POSITIVE_TUPLE
    mantissa, adjustment = math.frexp(numerator[0] / denominator[0])
    return mantissa, numerator[1] - denominator[1] + adjustment


def _positive_tuple_to_float(value: _PositiveTuple) -> float:
    if value[0] == 0.0:
        return 0.0
    try:
        return math.ldexp(value[0], value[1])
    except OverflowError:
        return float("inf")


def _positive_tuple_less(left: _PositiveTuple, right: _PositiveTuple) -> bool:
    if left[0] == 0.0:
        return right[0] > 0.0
    if right[0] == 0.0:
        return False
    return (left[1], left[0]) < (right[1], right[0])


def _positive_tuple_ratio(numerator: _PositiveTuple, denominator: _PositiveTuple) -> float:
    return _positive_tuple_to_float(_positive_tuple_divide(numerator, denominator))


_DyadicComponent = tuple[int, int]
_PositiveDyadicRatio = tuple[_DyadicComponent, _DyadicComponent]
_PositivePowerBase = _PositiveTuple | _PositiveDyadicRatio


def _tuple_to_dyadic_component(value: _PositiveTuple, sign: int = 1) -> _DyadicComponent:
    numerator, denominator = value[0].as_integer_ratio()
    denominator_exponent = denominator.bit_length() - 1
    return sign * numerator, value[1] - denominator_exponent


def _signed_expansion_component(
    positive: Sequence[_PositiveTuple],
    negative: Sequence[_PositiveTuple] = (),
) -> _DyadicComponent:
    """Return the exact sign-bearing dyadic sum without exponent-sized shifts."""
    components = [_tuple_to_dyadic_component(value) for value in positive if value[0] > 0.0]
    components.extend(_tuple_to_dyadic_component(value, -1) for value in negative if value[0] > 0.0)
    while components:
        components = [(integer, exponent) for integer, exponent in components if integer]
        if not components:
            return 0, 0
        if len(components) == 1:
            return components[0]
        top_exponent = max(exponent + abs(integer).bit_length() for integer, exponent in components)
        # Retain enough lower bits to round a binary64 result correctly. A
        # much smaller sign-only window would prove the sign, but would erase
        # material same-sign addends when this component is later converted.
        margin = (len(components) - 1).bit_length() + 55
        cutoff = top_exponent - margin
        high: list[_DyadicComponent] = []
        low: list[_DyadicComponent] = []
        for component in components:
            target = high if component[1] + abs(component[0]).bit_length() >= cutoff else low
            target.append(component)
        if len(high) == 1:
            return high[0]
        common_exponent = min(exponent for _integer, exponent in high)
        combined = sum(integer << (exponent - common_exponent) for integer, exponent in high)
        if combined:
            low.append((combined, common_exponent))
        components = low
    return 0, 0


def _positive_tuple_from_component(component: _DyadicComponent) -> _PositiveTuple:
    integer, exponent = component
    if integer <= 0:
        return _ZERO_POSITIVE_TUPLE
    bit_count = integer.bit_length()
    if bit_count > 53:
        shift = bit_count - 53
        retained, remainder = divmod(integer, 1 << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and retained & 1):
            retained += 1
        integer = retained
        exponent += shift
        bit_count = integer.bit_length()
    mantissa = math.ldexp(float(integer), -bit_count)
    return mantissa, exponent + bit_count


def _exact_expansion_component(
    positive: Sequence[_PositiveTuple],
    negative: Sequence[_PositiveTuple] = (),
) -> _DyadicComponent:
    """Compress bounded binary64-derived terms into one exact dyadic integer."""
    components = [_tuple_to_dyadic_component(value) for value in positive if value[0] > 0.0]
    components.extend(_tuple_to_dyadic_component(value, -1) for value in negative if value[0] > 0.0)
    if not components:
        return 0, 0
    common_exponent = min(exponent for _integer, exponent in components)
    integer = sum(value << (exponent - common_exponent) for value, exponent in components)
    return integer, common_exponent


def _sum_dyadic_components(
    positive: Sequence[_DyadicComponent],
    negative: Sequence[_DyadicComponent] = (),
) -> _DyadicComponent:
    components = [component for component in positive if component[0] != 0]
    components.extend((-integer, exponent) for integer, exponent in negative if integer != 0)
    if not components:
        return 0, 0
    common_exponent = min(exponent for _integer, exponent in components)
    return (
        sum(integer << (exponent - common_exponent) for integer, exponent in components),
        common_exponent,
    )


def _exact_positive_float_product_component(*values: float) -> _DyadicComponent:
    component = (1, 0)
    for value in values:
        if value == 0.0:
            return 0, 0
        component = _dyadic_component_product(
            component,
            _tuple_to_dyadic_component(_positive_float_tuple(float(value))),
        )
    return component


def _power_base_is_positive(base: _PositivePowerBase) -> bool:
    if isinstance(base[0], tuple):
        return base[0][0] > 0
    return base[0] > 0.0


def _power_base_denominator_inexact(base: _PositivePowerBase) -> bool:
    if not isinstance(base[0], tuple):
        return False
    denominator_tuple = _positive_tuple_from_component(base[1])
    return (
        _dyadic_component_difference(_tuple_to_dyadic_component(denominator_tuple), base[1])[0] != 0
    )


def _power_base_requires_exact_ratio(base: _PositivePowerBase, power: float) -> bool:
    if not isinstance(base[0], tuple):
        return False
    if abs(power) <= 1024.0:
        return False
    denominator_tuple = _positive_tuple_from_component(base[1])
    difference = _dyadic_component_difference(
        _tuple_to_dyadic_component(denominator_tuple), base[1]
    )
    if difference[0] == 0:
        return False
    relative_error = _positive_tuple_ratio(
        _positive_tuple_from_component((abs(difference[0]), difference[1])),
        denominator_tuple,
    )
    return relative_error * abs(power) >= sys.float_info.epsilon / 2.0


def _positive_power_base_product_tuple(
    base: _PositivePowerBase,
    power: float,
    multiplier: _PositiveTuple,
) -> _PositiveTuple:
    if not isinstance(base[0], tuple):
        return _positive_tuple_power_product_tuple(base, power, multiplier)
    numerator, denominator = base
    if numerator[0] == 0 or multiplier[0] == 0.0:
        return _ZERO_POSITIVE_TUPLE
    numerator_tuple = _positive_tuple_from_component(numerator)
    denominator_tuple = _positive_tuple_from_component(denominator)
    represented_denominator = _tuple_to_dyadic_component(denominator_tuple)
    if _dyadic_component_difference(represented_denominator, denominator)[0] == 0:
        return _positive_tuple_power_product_tuple(
            _positive_tuple_divide(numerator_tuple, denominator_tuple),
            power,
            multiplier,
        )
    with localcontext() as context:
        context.prec = 800
        log_two = Decimal(2).ln()
        numerator_log2 = Decimal(numerator[1]) + Decimal(abs(numerator[0])).ln() / log_two
        denominator_log2 = Decimal(denominator[1]) + Decimal(abs(denominator[0])).ln() / log_two
        powered_log2 = (numerator_log2 - denominator_log2) * Decimal.from_float(power)
        integer_decimal = powered_log2.to_integral_value(rounding=ROUND_FLOOR)
        integer_exponent = int(integer_decimal)
        fractional_exponent = float(powered_log2 - integer_decimal)
    mantissa = multiplier[0] * math.exp2(fractional_exponent)
    mantissa, adjustment = math.frexp(mantissa)
    return mantissa, multiplier[1] + integer_exponent + adjustment


def _dyadic_log_decimal(component: _DyadicComponent, log_two: Decimal) -> Decimal:
    return Decimal(component[1]) * log_two + Decimal(abs(component[0])).ln()


def _dyadic_decimal(component: _DyadicComponent) -> Decimal:
    return Decimal(component[0]) * (Decimal(2) ** component[1])


def _dyadic_ratio_decimal_float(
    numerator: _DyadicComponent,
    denominator: _DyadicComponent,
) -> float:
    exponent = numerator[1] - denominator[1]
    if exponent >= 0:
        ratio = Fraction(numerator[0] << exponent, denominator[0])
    else:
        ratio = Fraction(numerator[0], denominator[0] << -exponent)
    try:
        return float(ratio)
    except OverflowError:
        signs_differ = (ratio.numerator < 0) != (ratio.denominator < 0)
        return float("-inf") if signs_differ else float("inf")


def _power_base_log_decimal(base: _PositivePowerBase, log_two: Decimal) -> Decimal:
    if isinstance(base[0], tuple):
        return _dyadic_log_decimal(base[0], log_two) - _dyadic_log_decimal(base[1], log_two)
    return Decimal(base[1]) * log_two + Decimal.from_float(base[0]).ln()


def _decimal_powered_weighted_squared_distance_ratio(
    distances: np.ndarray,
    membership_bases: Sequence[_PositivePowerBase],
    weights: np.ndarray,
    *,
    membership_power: float,
    distance_denominator: float,
    coefficient_weights: np.ndarray,
) -> float:
    """Evaluate fuzzy XB in one guard-precision log ratio to avoid double rounding."""
    with localcontext() as context:
        context.prec = 800
        log_two = Decimal(2).ln()
        term_logs: list[Decimal] = []
        membership_log_cache: dict[_PositivePowerBase, Decimal] = {}
        scalar_log_cache: dict[float, Decimal] = {}
        for distance, membership, weight in zip(
            np.asarray(distances, dtype=float).reshape(-1),
            membership_bases,
            np.asarray(weights, dtype=float).reshape(-1),
            strict=True,
        ):
            if distance <= 0.0 or weight <= 0.0 or not _power_base_is_positive(membership):
                continue
            if membership not in membership_log_cache:
                membership_log_cache[membership] = _power_base_log_decimal(membership, log_two)
            membership_log = membership_log_cache[membership]
            if float(weight) not in scalar_log_cache:
                scalar_log_cache[float(weight)] = Decimal.from_float(float(weight)).ln()
            weight_log = scalar_log_cache[float(weight)]
            if float(distance) not in scalar_log_cache:
                scalar_log_cache[float(distance)] = Decimal.from_float(float(distance)).ln()
            distance_log = scalar_log_cache[float(distance)]
            term_logs.append(
                membership_log * Decimal.from_float(membership_power)
                + weight_log
                + 2 * distance_log
            )
        if not term_logs:
            return 0.0
        maximum_log = max(term_logs)
        scaled_sum = sum(
            (value - maximum_log).exp()
            for value in term_logs
            if value - maximum_log > Decimal(-2000)
        )
        denominator = (
            sum(
                Decimal.from_float(float(weight))
                for weight in np.asarray(coefficient_weights, dtype=float).reshape(-1)
            )
            * Decimal.from_float(distance_denominator) ** 2
        )
        result_log = maximum_log + scaled_sum.ln() - denominator.ln()
        if result_log > Decimal(1000):
            return float("inf")
        if result_log < Decimal(-1000):
            return 0.0
        return float(result_log.exp())


def _power_base_decimal(base: _PositivePowerBase) -> Decimal:
    if isinstance(base[0], tuple):
        return _dyadic_decimal(base[0]) / _dyadic_decimal(base[1])
    return Decimal.from_float(_positive_tuple_to_float(base))


def _decimal_membership_entropy(row: Sequence[_PositivePowerBase]) -> float:
    with localcontext() as context:
        context.prec = 800
        entropy = Decimal(0)
        for base in row:
            if not _power_base_is_positive(base):
                continue
            probability = _power_base_decimal(base)
            entropy -= probability * probability.ln()
        return float(entropy)


def _decimal_membership_fpc(row: Sequence[_PositivePowerBase]) -> float:
    with localcontext() as context:
        context.prec = 100
        return float(
            sum(_power_base_decimal(base) ** 2 for base in row if _power_base_is_positive(base))
        )


def _decimal_ambiguity_weighted_silhouette(
    silhouettes: np.ndarray,
    weights: np.ndarray,
    ambiguity_bases: Sequence[_PositivePowerBase],
) -> float:
    with localcontext() as context:
        context.prec = 800
        numerator = Decimal(0)
        denominator = Decimal(0)
        for silhouette, weight, ambiguity in zip(
            silhouettes, weights, ambiguity_bases, strict=True
        ):
            if (
                not np.isfinite(silhouette)
                or weight <= 0.0
                or not _power_base_is_positive(ambiguity)
            ):
                continue
            effective_weight = Decimal.from_float(float(weight)) * _power_base_decimal(ambiguity)
            denominator += effective_weight
            numerator += effective_weight * Decimal.from_float(float(silhouette))
        if denominator == 0:
            return float("nan")
        return float(numerator / denominator)


def _dyadic_component_product(
    left: _DyadicComponent,
    right: _DyadicComponent,
) -> _DyadicComponent:
    return left[0] * right[0], left[1] + right[1]


def _dyadic_component_difference(
    left: _DyadicComponent,
    right: _DyadicComponent,
) -> _DyadicComponent:
    if left[0] == 0:
        return -right[0], right[1]
    if right[0] == 0:
        return left
    common_exponent = min(left[1], right[1])
    return (
        (left[0] << (left[1] - common_exponent)) - (right[0] << (right[1] - common_exponent)),
        common_exponent,
    )


def _dyadic_ratio_less(
    left_numerator: _DyadicComponent,
    left_denominator: _DyadicComponent,
    right_numerator: _DyadicComponent,
    right_denominator: _DyadicComponent,
) -> bool:
    difference = _dyadic_component_difference(
        _dyadic_component_product(left_numerator, right_denominator),
        _dyadic_component_product(right_numerator, left_denominator),
    )
    return difference[0] < 0


def _silhouette_from_dyadic_ratios(
    within_numerator: _DyadicComponent,
    within_denominator: _DyadicComponent,
    between_numerator: _DyadicComponent,
    between_denominator: _DyadicComponent,
) -> float:
    if within_numerator[0] == 0 and between_numerator[0] == 0:
        return float("nan")
    within_cross = _dyadic_component_product(within_numerator, between_denominator)
    between_cross = _dyadic_component_product(between_numerator, within_denominator)
    difference = _dyadic_component_difference(between_cross, within_cross)
    if difference[0] == 0:
        return 0.0
    if difference[0] > 0:
        return _dyadic_ratio_decimal_float(difference, between_cross)
    return -_dyadic_ratio_decimal_float(
        (-difference[0], difference[1]),
        within_cross,
    )


def _signed_expansion_tuple(
    positive: Sequence[_PositiveTuple],
    negative: Sequence[_PositiveTuple] = (),
) -> _PositiveTuple:
    return _positive_tuple_from_component(_signed_expansion_component(positive, negative))


def _positive_expansion_less(
    left: Sequence[_PositiveTuple],
    right: Sequence[_PositiveTuple],
) -> bool:
    return _signed_expansion_component(left, right)[0] < 0


def _positive_weighted_distance_terms(
    distances: np.ndarray,
    weights: np.ndarray,
) -> list[_PositiveTuple]:
    return [
        _positive_product_tuple(float(distance), float(weight))
        for distance, weight in zip(distances, weights, strict=True)
        if distance > 0.0 and weight > 0.0
    ]


def _exact_weighted_distance_component(
    distances: np.ndarray,
    weights: np.ndarray,
) -> _DyadicComponent:
    return _sum_dyadic_components(
        [
            _exact_positive_float_product_component(float(distance), float(weight))
            for distance, weight in zip(distances, weights, strict=True)
            if distance > 0.0 and weight > 0.0
        ]
    )


def _positive_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """Average non-negative finite values without changing safe ordinary bits."""
    values_array = np.asarray(values, dtype=float).reshape(-1)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        products = values_array * weights_array
        denominator = float(np.sum(weights_array))
        numerator = float(np.sum(products))
    active = (values_array > 0.0) & (weights_array > 0.0)
    positive_weights = weights_array[weights_array > 0.0]
    denominator_terms = [_positive_float_tuple(float(weight)) for weight in positive_weights]
    numerator_terms = [
        _positive_product_tuple(float(value), float(weight))
        for value, weight in zip(values_array, weights_array, strict=True)
        if value > 0.0 and weight > 0.0
    ]
    dynamic_weight_ratio = _positive_terms_need_exact_sum(
        denominator_terms
    ) or _positive_terms_need_exact_sum(numerator_terms)
    ordinary_safe = bool(
        np.isfinite(denominator)
        and denominator > 0.0
        and denominator >= sys.float_info.min
        and np.isfinite(numerator)
        and not np.any(active & ((products == 0.0) | ~np.isfinite(products)))
        and not np.any(active & (products < sys.float_info.min))
    )
    if ordinary_safe and not dynamic_weight_ratio:
        return numerator / denominator
    numerator_component = _sum_dyadic_components(
        [
            _exact_positive_float_product_component(float(value), float(weight))
            for value, weight in zip(values_array, weights_array, strict=True)
            if value > 0.0 and weight > 0.0
        ]
    )
    denominator_component = _sum_dyadic_components(
        [
            _tuple_to_dyadic_component(_positive_float_tuple(float(weight)))
            for weight in weights_array
            if weight > 0.0
        ]
    )
    return _dyadic_ratio_decimal_float(numerator_component, denominator_component)


def _signed_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """Average signed finite values with one final exact dyadic ratio."""
    values_array = np.asarray(values, dtype=float).reshape(-1)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        products = values_array * weights_array
        numerator = float(np.sum(products))
        denominator = float(np.sum(weights_array))
    active = (values_array != 0.0) & (weights_array > 0.0)
    try:
        positive_total = math.fsum(float(product) for product in products[products > 0.0])
        negative_total = math.fsum(float(-product) for product in products[products < 0.0])
    except OverflowError:
        positive_total = float("inf")
        negative_total = float("inf")
    cancellation_scale = max(positive_total, negative_total, sys.float_info.min)
    cancellation_bound = (
        8.0 * max(1, int(np.count_nonzero(active))) * np.finfo(float).eps * cancellation_scale
    )
    cancellation_risk = bool(
        positive_total > 0.0
        and negative_total > 0.0
        and np.isfinite(positive_total)
        and np.isfinite(negative_total)
        and abs(positive_total - negative_total) <= cancellation_bound
    )
    ordinary_safe = bool(
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > 0.0
        and not np.any(active & ((products == 0.0) | ~np.isfinite(products)))
        and not np.any(active & (np.abs(products) < sys.float_info.min))
    )
    if ordinary_safe and not cancellation_risk:
        return numerator / denominator

    positive_components: list[_DyadicComponent] = []
    negative_components: list[_DyadicComponent] = []
    for value, weight in zip(values_array, weights_array, strict=True):
        if value == 0.0 or weight <= 0.0:
            continue
        component = _exact_positive_float_product_component(abs(float(value)), float(weight))
        (positive_components if value > 0.0 else negative_components).append(component)
    numerator_component = _sum_dyadic_components(positive_components, negative_components)
    denominator_component = _sum_dyadic_components(
        [
            _tuple_to_dyadic_component(_positive_float_tuple(float(weight)))
            for weight in weights_array
            if weight > 0.0
        ]
    )
    if denominator_component[0] == 0:
        return float("nan")
    sign = -1.0 if numerator_component[0] < 0 else 1.0
    return sign * _dyadic_ratio_decimal_float(
        (abs(numerator_component[0]), numerator_component[1]),
        denominator_component,
    )


def _signed_weighted_average_factors(
    values: np.ndarray,
    *weight_factors: np.ndarray,
) -> float:
    """Average signed values without first rounding a product of weight factors."""
    values_array = np.asarray(values, dtype=float).reshape(-1)
    factors = [np.asarray(factor, dtype=float).reshape(-1) for factor in weight_factors]
    if not factors:
        raise ValueError("at least one weight factor is required")
    if any(factor.shape != values_array.shape for factor in factors):
        raise ValueError("weight factors must match the values shape")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        materialized_weights = np.prod(np.stack(factors, axis=0), axis=0)
    conceptually_positive = np.logical_and.reduce([factor > 0.0 for factor in factors])
    product_range_loss = conceptually_positive & (
        (materialized_weights == 0.0)
        | ~np.isfinite(materialized_weights)
        | (materialized_weights < sys.float_info.min)
    )
    if not np.any(product_range_loss):
        return _signed_weighted_average(values_array, materialized_weights)

    denominator_components: list[_DyadicComponent] = []
    positive_components: list[_DyadicComponent] = []
    negative_components: list[_DyadicComponent] = []
    for index, value in enumerate(values_array):
        factor_values = [float(factor[index]) for factor in factors]
        if any(factor <= 0.0 for factor in factor_values):
            continue
        denominator_component = _exact_positive_float_product_component(*factor_values)
        denominator_components.append(denominator_component)
        if not np.isfinite(value) or value == 0.0:
            continue
        numerator_component = _exact_positive_float_product_component(
            abs(float(value)),
            *factor_values,
        )
        (positive_components if value > 0.0 else negative_components).append(numerator_component)
    denominator_component = _sum_dyadic_components(denominator_components)
    if denominator_component[0] == 0:
        return float("nan")
    numerator_component = _sum_dyadic_components(positive_components, negative_components)
    if numerator_component[0] == 0:
        return 0.0
    sign = -1.0 if numerator_component[0] < 0 else 1.0
    return sign * _dyadic_ratio_decimal_float(
        (abs(numerator_component[0]), numerator_component[1]),
        denominator_component,
    )


def _silhouette_needs_stable_accumulation(D: np.ndarray, w: np.ndarray) -> bool:
    """Detect finite positive inputs whose raw accumulation can lose a ratio."""
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(w))):
        return False
    if np.any(D < 0.0) or np.any(w < 0.0):
        return False
    positive_distances = D[D > 0.0]
    positive_weights = w[w > 0.0]
    if positive_distances.size == 0 or positive_weights.size == 0:
        return False
    n = max(1, D.shape[0])
    maximum_distance = float(np.max(positive_distances))
    minimum_distance = float(np.min(positive_distances))
    maximum_weight = float(np.max(positive_weights))
    minimum_weight = float(np.min(positive_weights))
    maximum_product = _positive_product_tuple(maximum_distance, maximum_weight)
    minimum_product = _positive_product_tuple(minimum_distance, minimum_weight)
    positive_mask = (D > 0.0) & (w[None, :] > 0.0)
    product_count = int(np.count_nonzero(positive_mask))
    product_range_loss = False
    if product_count > 1:
        _distance_mantissas, distance_exponents = np.frexp(D)
        _weight_mantissas, weight_exponents = np.frexp(w)
        product_exponents = (distance_exponents + weight_exponents[None, :])[positive_mask]
        rounding_window = 52 - (product_count - 1).bit_length()
        product_range_loss = bool(
            int(np.max(product_exponents)) - int(np.min(product_exponents)) >= rounding_window
        )
    return bool(
        maximum_weight > sys.float_info.max / n
        or minimum_weight < sys.float_info.min
        or minimum_distance < sys.float_info.min * n
        or not _positive_tuple_less(
            maximum_product,
            _positive_float_tuple(sys.float_info.max / n),
        )
        or _positive_tuple_less(
            minimum_product,
            _positive_float_tuple(sys.float_info.min * n),
        )
        or product_range_loss
    )


def _silhouette_has_ordinary_near_tie(
    D: np.ndarray,
    w: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
    cluster_count: int,
    measures: Sequence[str],
) -> bool:
    """Detect mean gaps within the forward-error bound of ordinary sums."""
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(w))):
        return False
    operation_count = max(1, 2 * D.shape[0])
    gamma = (operation_count * sys.float_info.epsilon) / (
        1.0 - operation_count * sys.float_info.epsilon
    )
    for index in range(D.shape[0]):
        cluster = int(labels[index])
        if sizes[cluster] <= 0.0:
            continue
        within_terms = w[labels == cluster] * D[index, labels == cluster]
        within_sum = float(np.sum(within_terms))
        within_error = gamma * float(np.sum(np.abs(within_terms)))
        between_mean = float("inf")
        between_error = 0.0
        for other in range(cluster_count):
            if other == cluster or sizes[other] <= 0.0:
                continue
            between_terms = w[labels == other] * D[index, labels == other]
            candidate = float(np.sum(between_terms) / sizes[other])
            candidate_error = gamma * float(np.sum(np.abs(between_terms))) / sizes[other]
            if candidate < between_mean:
                between_mean = candidate
                between_error = candidate_error
        if not np.isfinite(between_mean):
            continue
        within_weighted = within_sum / sizes[cluster]
        weighted_error = within_error / sizes[cluster]
        if abs(between_mean - within_weighted) <= between_error + weighted_error:
            return True
        if "ASW" in measures and sizes[cluster] > 1.0:
            within_individual = within_sum / (sizes[cluster] - 1.0)
            individual_error = within_error / (sizes[cluster] - 1.0)
            if abs(between_mean - within_individual) <= between_error + individual_error:
                return True
    return False


def _cluster_mass_expansions(
    labels: np.ndarray, w: np.ndarray, k: int
) -> tuple[list[list[_PositiveTuple]], list[_DyadicComponent]]:
    """Exact positive cluster masses as tuple expansions and dyadic components."""
    size_expansions = [
        [_positive_float_tuple(float(weight)) for weight in w[labels == c] if weight > 0.0]
        for c in range(k)
    ]
    size_components = [_exact_expansion_component(expansion) for expansion in size_expansions]
    return size_expansions, size_components


def _stable_observation_silhouettes(
    D: np.ndarray,
    labels: np.ndarray,
    w: np.ndarray,
    k: int,
    size_expansions: Sequence[Sequence[_PositiveTuple]],
    size_components: Sequence[_DyadicComponent],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation ASW and ASWw with exact dyadic accumulation.

    This is the stable branch of :func:`wcSilhouetteObs`, shared with the
    permissive kernel's range-risk route (finding WC-090304) so that every
    tier forms the same silhouettes where binary64 products would lose range.
    A cluster of weighted size at most one contributes zero own-cluster mean
    to ASW, as R's kernel does; a genuine 0/0 observation is NaN. When every
    other cluster has zero mass, R leaves ``b_ik`` at ``DBL_MAX`` (the
    2026-08-24 F-05 sentinel), which this branch dropped into NaN until the
    2026-09-07 finding WC-090307; the ordinary float branch already kept it.
    """
    n = D.shape[0]
    asw_i = np.zeros(n, dtype=float)
    asw_w = np.zeros(n, dtype=float)
    unit_component = _tuple_to_dyadic_component(_positive_float_tuple(1.0))
    sentinel_between = (
        _tuple_to_dyadic_component(_positive_float_tuple(sys.float_info.max)),
        unit_component,
    )
    for i in range(n):
        ci = int(labels[i])
        within = _exact_weighted_distance_component(D[i], w * (labels == ci))
        between: tuple[_DyadicComponent, _DyadicComponent] | None = None
        for c in range(k):
            if c == ci or not size_expansions[c]:
                continue
            group_mean = (
                _exact_weighted_distance_component(D[i], w * (labels == c)),
                size_components[c],
            )
            if between is None or _dyadic_ratio_less(
                group_mean[0], group_mean[1], between[0], between[1]
            ):
                between = group_mean
        if not size_expansions[ci]:
            asw_i[i] = float("nan")
            asw_w[i] = float("nan")
            continue
        if between is None:
            between = sentinel_between
        individual_denominator = _exact_expansion_component(
            size_expansions[ci],
            [_positive_float_tuple(1.0)],
        )
        individual_within = within
        if individual_denominator[0] <= 0:
            individual_within = (0, 0)
            individual_denominator = unit_component
        asw_i[i] = _silhouette_from_dyadic_ratios(
            individual_within,
            individual_denominator,
            between[0],
            between[1],
        )
        asw_w[i] = _silhouette_from_dyadic_ratios(
            within,
            size_components[ci],
            between[0],
            between[1],
        )
    return asw_i, asw_w


def wcSilhouetteObs(
    diss: np.ndarray,
    clustering: Sequence[Any],
    weights: Optional[Sequence[float]] = None,
    measure: str | Sequence[str] = "ASW",
) -> np.ndarray:
    """Return observation-level ASW/ASWw with stable finite-distance accumulation.

    The inherited formulas are unchanged, but finite common distance and
    weight scales are removed while accumulating. The absolute weighted-size
    correction in ASW is retained explicitly; ASWw remains common-weight-scale
    invariant. This is a deliberate numerical improvement over installed
    WeightedCluster 2.0.
    """
    labels, levels = _factorize_quality_labels(clustering)
    _validate_quality_partition(labels, levels)
    D = _coerce_quality_distance(diss, expected_n=labels.size)
    measures = [measure] if isinstance(measure, str) else [str(value) for value in measure]
    if not all(value in ("ASW", "ASWw") for value in measures):
        raise ValueError("measure must be 'ASW' or 'ASWw'")
    weights_array = _coerce_quality_weights(weights, expected_n=D.shape[0])
    w = np.ones(D.shape[0], dtype=float) if weights_array is None else weights_array

    stable_accumulation = _silhouette_needs_stable_accumulation(D, w)
    w_accumulation = w
    inverse_weight_scale = 1.0

    k = int(labels.max()) + 1 if labels.size > 0 else 0
    sizes = np.bincount(labels, weights=w_accumulation, minlength=k).astype(float)
    size_expansions, size_components = _cluster_mass_expansions(labels, w, k)
    unit_component = _tuple_to_dyadic_component(_positive_float_tuple(1.0))
    mass_rounding = any(
        not np.isfinite(sizes[index])
        or (sizes[index] == 1.0 and _dyadic_component_difference(component, unit_component)[0] != 0)
        for index, component in enumerate(size_components)
    )
    if "ASW" in measures and mass_rounding:
        stable_accumulation = True
    if not stable_accumulation and _silhouette_has_ordinary_near_tie(
        D, w, labels, sizes, k, measures
    ):
        stable_accumulation = True

    # Match R behaviour: ASW is undefined when a cluster has weighted size < 1.
    cluster_below_one = any(
        _positive_expansion_less(expansion, [_positive_float_tuple(1.0)])
        for expansion in size_expansions
    )
    if "ASW" in measures and cluster_below_one:
        if len(measures) == 1:
            raise ValueError(
                "ASW can not be computed because at least one cluster has less than one "
                "observation. Use measure='ASWw'."
            )
        warnings.warn(
            "ASW can not be computed because at least one cluster has less than one "
            "observation. Only 'ASWw' is computed.",
            UserWarning,
            stacklevel=2,
        )
        measures = ["ASWw"]

    n = D.shape[0]
    if stable_accumulation:
        asw_i, asw_w = _stable_observation_silhouettes(
            D, labels, w, k, size_expansions, size_components
        )
        values = {"ASW": asw_i, "ASWw": asw_w}
        if len(measures) == 1:
            return values[measures[0]]
        return np.column_stack([values[value] for value in measures])

    asw_i = np.zeros(n, dtype=float)
    asw_w = np.zeros(n, dtype=float)

    for i in range(n):
        ci = int(labels[i])
        aik = 0.0
        othergroups = np.zeros(k, dtype=float)

        for j in range(n):
            if i == j:
                continue
            contrib = float(w_accumulation[j] * D[i, j])
            cj = int(labels[j])
            if cj == ci:
                aik += contrib
            else:
                othergroups[cj] += contrib

        bik = sys.float_info.max
        with np.errstate(divide="ignore", invalid="ignore"):
            for c in range(k):
                if c == ci:
                    continue
                val = float(np.divide(othergroups[c], sizes[c]))
                if val <= bik:
                    bik = val

        aik_w = aik / sizes[ci] if sizes[ci] > 0.0 else np.nan
        aik_i = (
            0.0 if sizes[ci] <= inverse_weight_scale else (aik / (sizes[ci] - inverse_weight_scale))
        )

        denom_i = np.fmax(aik_i, bik)
        denom_w = np.fmax(aik_w, bik)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            asw_i[i] = np.divide(bik - aik_i, denom_i)
            asw_w[i] = np.divide(bik - aik_w, denom_w)

    values = {"ASW": asw_i, "ASWw": asw_w}
    if len(measures) == 1:
        return values[measures[0]]
    return np.column_stack([values[value] for value in measures])


def _coerce_internal_index_inputs(
    diss: np.ndarray, labels: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the Python-only index shape contract and factorize labels."""
    D = np.asarray(diss, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("diss must be a square matrix")
    if D.shape[0] > 0:
        validate_distance_matrix(D, name="diss")
    normalized = _normalize_quality_labels(labels)
    if normalized.shape[0] != D.shape[0]:
        raise ValueError("Dimensions mismatch between diss and labels")
    return D, normalized


def _stable_weighted_squared_distance_ratio(
    distances: np.ndarray,
    coefficients: np.ndarray,
    *,
    distance_denominator: float,
    coefficient_denominator: float,
) -> float:
    """Return ``sum(c * (d / s)^2) / sum(c)`` without raw squares."""
    components: list[tuple[float, int]] = []
    coefficient_denominator_mantissa, coefficient_denominator_exponent = math.frexp(
        coefficient_denominator
    )
    distance_denominator_mantissa, distance_denominator_exponent = math.frexp(distance_denominator)
    for distance, coefficient in zip(
        np.asarray(distances, dtype=float).reshape(-1),
        np.asarray(coefficients, dtype=float).reshape(-1),
        strict=True,
    ):
        if distance == 0.0 or coefficient == 0.0:
            continue
        coefficient_mantissa, coefficient_exponent = math.frexp(float(coefficient))
        distance_mantissa, distance_exponent = math.frexp(float(distance))
        mantissa = (coefficient_mantissa / coefficient_denominator_mantissa) * (
            distance_mantissa / distance_denominator_mantissa
        ) ** 2
        mantissa, adjustment = math.frexp(mantissa)
        exponent = (
            coefficient_exponent
            - coefficient_denominator_exponent
            + 2 * (distance_exponent - distance_denominator_exponent)
            + adjustment
        )
        components.append((mantissa, exponent))
    if not components:
        return 0.0
    maximum_exponent = max(exponent for _, exponent in components)
    scaled_total = math.fsum(
        math.ldexp(mantissa, exponent - maximum_exponent) for mantissa, exponent in components
    )
    total_mantissa, adjustment = math.frexp(scaled_total)
    try:
        return math.ldexp(total_mantissa, maximum_exponent + adjustment)
    except OverflowError:
        return float("inf")


def _stable_squared_tuple_product_ratio(
    numerators: Sequence[_PositiveTuple],
    denominators: Sequence[_PositiveTuple],
) -> float:
    """Square a product/ratio of positive tuples at the final IEEE boundary."""
    numerator_component: _DyadicComponent = (1, 0)
    for numerator in numerators:
        numerator_component = _dyadic_component_product(
            numerator_component,
            _tuple_to_dyadic_component(numerator),
        )
    denominator_component: _DyadicComponent = (1, 0)
    for denominator in denominators:
        denominator_component = _dyadic_component_product(
            denominator_component,
            _tuple_to_dyadic_component(denominator),
        )
    return _dyadic_ratio_decimal_float(
        _dyadic_component_product(numerator_component, numerator_component),
        _dyadic_component_product(denominator_component, denominator_component),
    )


def _stable_powered_weighted_squared_distance_ratio(
    distances: np.ndarray,
    memberships: np.ndarray,
    weights: np.ndarray,
    *,
    membership_power: float,
    distance_denominator: float,
    coefficient_denominator: _PositiveTuple,
    membership_tuples: Sequence[_PositivePowerBase] | None = None,
    weight_tuples: Sequence[_PositiveTuple] | None = None,
) -> float:
    """Form fuzzy XB before membership powers or distance squares can vanish."""
    terms: list[_PositiveTuple] = []
    distance_values = np.asarray(distances, dtype=float).reshape(-1)
    membership_values = np.asarray(memberships, dtype=float).reshape(-1)
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    conceptual_memberships = (
        [_positive_float_tuple(float(value)) for value in membership_values]
        if membership_tuples is None
        else list(membership_tuples)
    )
    conceptual_weights = (
        [_positive_float_tuple(float(value)) for value in weight_values]
        if weight_tuples is None
        else list(weight_tuples)
    )
    for distance, membership, weight in zip(
        distance_values,
        conceptual_memberships,
        conceptual_weights,
        strict=True,
    ):
        if distance == 0.0 or not _power_base_is_positive(membership) or weight[0] == 0.0:
            continue
        coefficient = _positive_power_base_product_tuple(
            membership,
            membership_power,
            weight,
        )
        distance_tuple = _positive_float_tuple(float(distance))
        terms.append(
            _positive_tuple_product(
                coefficient,
                _positive_tuple_product(distance_tuple, distance_tuple),
            )
        )
    numerator = _positive_sum_tuples(terms)
    denominator_distance = _positive_float_tuple(distance_denominator)
    denominator = _positive_tuple_product(
        coefficient_denominator,
        _positive_tuple_product(denominator_distance, denominator_distance),
    )
    return _positive_tuple_ratio(numerator, denominator)


def _powered_coefficients_for_objective(
    memberships: np.ndarray,
    weights: np.ndarray,
    power: float,
    membership_tuples: Sequence[_PositivePowerBase] | None = None,
    weight_tuples: Sequence[_PositiveTuple] | None = None,
) -> tuple[np.ndarray, list[_PositiveTuple] | None]:
    """Return ordinary powered weights unless a positive coefficient vanishes."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ordinary = (memberships**power) * weights
    conceptual_memberships = (
        [_positive_float_tuple(float(value)) for value in memberships]
        if membership_tuples is None
        else list(membership_tuples)
    )
    conceptual_weights = (
        [_positive_float_tuple(float(value)) for value in weights]
        if weight_tuples is None
        else list(weight_tuples)
    )
    mathematical_positive = np.asarray(
        [
            _power_base_is_positive(membership) and weight[0] > 0.0
            for membership, weight in zip(conceptual_memberships, conceptual_weights, strict=True)
        ]
    )
    if not np.any(
        mathematical_positive
        & (((memberships == 0.0) | (weights == 0.0)) | (ordinary == 0.0) | ~np.isfinite(ordinary))
    ) and not any(_power_base_requires_exact_ratio(base, power) for base in conceptual_memberships):
        return ordinary, None
    tuples = [
        (
            _positive_power_base_product_tuple(membership, power, weight)
            if _power_base_is_positive(membership) and weight[0] > 0.0
            else _ZERO_POSITIVE_TUPLE
        )
        for membership, weight in zip(conceptual_memberships, conceptual_weights, strict=True)
    ]
    return ordinary, tuples


def _guard_precision_medoid_candidate(
    subD: np.ndarray,
    candidates: Sequence[int],
    power_bases: Sequence[_PositivePowerBase],
    raw_weight_tuples: Sequence[_PositiveTuple],
    membership_power: float,
) -> int:
    def signed_log_sum_sign(terms: list[tuple[int, Decimal]]) -> int:
        remaining = terms
        while remaining:
            maximum_log = max(log_value for _sign, log_value in remaining)
            group = [term for term in remaining if maximum_log - term[1] <= Decimal(1000)]
            rest = [term for term in remaining if maximum_log - term[1] > Decimal(1000)]
            scaled = sum(
                Decimal(sign) * (log_value - maximum_log).exp() for sign, log_value in group
            )
            if scaled != 0:
                combined = (1 if scaled > 0 else -1, maximum_log + abs(scaled).ln())
                if not rest or combined[1] - max(value for _sign, value in rest) > Decimal(1000):
                    return combined[0]
                rest.append(combined)
            remaining = rest
        return 0

    with localcontext() as context:
        context.prec = 800
        log_two = Decimal(2).ln()
        coefficient_logs: list[Decimal | None] = []
        for base, weight in zip(power_bases, raw_weight_tuples, strict=True):
            if not _power_base_is_positive(base) or weight[0] == 0.0:
                coefficient_logs.append(None)
                continue
            coefficient_logs.append(
                _power_base_log_decimal(base, log_two) * Decimal.from_float(membership_power)
                + Decimal(weight[1]) * log_two
                + Decimal.from_float(weight[0]).ln()
            )
        best = int(candidates[0])
        for candidate_raw in candidates[1:]:
            candidate = int(candidate_raw)
            difference_terms: list[tuple[int, Decimal]] = []
            for coefficient_log, candidate_distance, best_distance in zip(
                coefficient_logs,
                subD[candidate],
                subD[best],
                strict=True,
            ):
                distance_difference = float(candidate_distance - best_distance)
                if coefficient_log is None or distance_difference == 0.0:
                    continue
                difference_terms.append(
                    (
                        1 if distance_difference > 0.0 else -1,
                        coefficient_log + Decimal.from_float(abs(distance_difference)).ln(),
                    )
                )
            if signed_log_sum_sign(difference_terms) < 0:
                best = candidate
        return best


def _find_medoid_without_range_loss(
    D: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    weight_tuples: Sequence[_PositiveTuple] | None = None,
    power_bases: Sequence[_PositivePowerBase] | None = None,
    raw_weight_tuples: Sequence[_PositiveTuple] | None = None,
    membership_power: float | None = None,
) -> int:
    """Preserve the ordinary medoid path unless its weighted sums lose range."""
    if weight_tuples is None and not _silhouette_needs_stable_accumulation(
        D[np.ix_(indices, indices)],
        weights[indices],
    ):
        subD = D[np.ix_(indices, indices)]
        ordinary_products = weights[indices].reshape(1, -1) * subD
        ordinary_costs = ordinary_products.sum(axis=1)
        best_cost = float(np.min(ordinary_costs))
        best_index = int(np.argmin(ordinary_costs))
        operation_count = max(1, 2 * indices.size)
        gamma = (operation_count * sys.float_info.epsilon) / (
            1.0 - operation_count * sys.float_info.epsilon
        )
        error_bounds = gamma * np.sum(np.abs(ordinary_products), axis=1)
        nearby = np.flatnonzero(
            ordinary_costs - best_cost <= error_bounds + error_bounds[best_index]
        )
        if (
            nearby.size > 1
            and power_bases is not None
            and raw_weight_tuples is not None
            and membership_power is not None
        ):
            selected = _guard_precision_medoid_candidate(
                subD,
                nearby.tolist(),
                [power_bases[index] for index in indices],
                [raw_weight_tuples[index] for index in indices],
                membership_power,
            )
            return int(indices[selected])
        return int(indices[best_index])
    subD = D[np.ix_(indices, indices)]
    coefficient_tuples = (
        [_positive_float_tuple(float(weights[index])) for index in indices]
        if weight_tuples is None
        else [weight_tuples[index] for index in indices]
    )
    costs = [
        [
            _positive_tuple_product(
                coefficient,
                _positive_float_tuple(float(distance)),
            )
            for coefficient, distance in zip(coefficient_tuples, row, strict=True)
            if coefficient[0] > 0.0 and distance > 0.0
        ]
        for row in subD
    ]
    best = 0
    for candidate in range(1, len(costs)):
        if _positive_expansion_less(costs[candidate], costs[best]):
            best = candidate
    if (
        power_bases is not None
        and raw_weight_tuples is not None
        and membership_power is not None
        and len(costs) > 1
    ):
        best = _guard_precision_medoid_candidate(
            subD,
            list(range(len(costs))),
            [power_bases[index] for index in indices],
            [raw_weight_tuples[index] for index in indices],
            membership_power,
        )
    return int(indices[best])


def _squares_leave_binary64_range(values: np.ndarray) -> bool:
    """Whether any exact-positive value cannot be squared as binary64."""
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return False
    # A representable subnormal square can still lose most of its significand;
    # route before the square enters the subnormal range, not only when it
    # becomes exact zero.
    square_min = math.sqrt(sys.float_info.min)
    square_max = math.sqrt(float(np.finfo(float).max))
    return bool(np.min(positive) < square_min or np.max(positive) > square_max)


def compute_internal_indices(diss: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    """Compute Davies–Bouldin (DB), Xie–Beni (XB), and PBM indices on a full distance matrix.

    Parameters
    ----------
    diss : np.ndarray
        (N, N) distance matrix
    labels : array-like
        Cluster labels (accepts 0-based or 1-based)

    Returns
    -------
    dict
        Dict with keys 'DB', 'XB', 'PBM' (NaN if cannot be computed)

    """
    D_full, labs = _coerce_internal_index_inputs(diss, labels)
    N = D_full.shape[0]
    if N == 0 or labs.size == 0:
        return {"DB": float("nan"), "XB": float("nan"), "PBM": float("nan")}
    k = int(labs.max() + 1) if labs.size > 0 else 0
    if k < 2:
        return {"DB": float("nan"), "XB": float("nan"), "PBM": float("nan")}

    # Keep the original finite distances until a ratio helper sees every local
    # exact-positive term. Dividing the whole matrix by its maximum can erase a
    # valid local separation when the matrix spans the full binary64 range.
    med_full = np.empty(k, dtype=int)
    for c in range(k):
        idx = np.where(labs == c)[0]
        subD = D_full[np.ix_(idx, idx)]
        row_terms = [
            [_positive_float_tuple(float(value)) for value in row if value > 0.0] for row in subD
        ]
        best_row = 0
        for candidate in range(1, len(row_terms)):
            if _positive_expansion_less(row_terms[candidate], row_terms[best_row]):
                best_row = candidate
        med_full[c] = int(idx[best_row])

    scatter: list[_PositiveTuple] = []
    within_distances: list[np.ndarray] = []
    for c in range(k):
        idx = np.where(labs == c)[0]
        d_to_med = D_full[idx, med_full[c]]
        within_distances.append(d_to_med)
        scatter.append(
            _positive_tuple_divide(
                _positive_sum(d_to_med),
                _positive_float_tuple(float(idx.size)),
            )
        )
    all_within_distances = np.concatenate(within_distances)
    EK_tuple = _positive_sum(all_within_distances)

    M = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            if med_full[i] >= 0 and med_full[j] >= 0:
                M[i, j] = M[j, i] = float(D_full[med_full[i], med_full[j]])
            else:
                M[i, j] = M[j, i] = np.nan

    DB_vals: list[_PositiveTuple] = []
    for i in range(k):
        if med_full[i] < 0:
            continue
        max_ratio = _ZERO_POSITIVE_TUPLE
        for j in range(k):
            if i == j or med_full[j] < 0:
                continue
            denom = M[i, j]
            if denom <= 0 or np.isnan(denom):
                continue
            ratio = _positive_tuple_divide(
                _positive_sum_tuples((scatter[i], scatter[j])),
                _positive_float_tuple(float(denom)),
            )
            if _positive_tuple_less(max_ratio, ratio):
                max_ratio = ratio
        DB_vals.append(max_ratio)
    if not DB_vals:
        DB = float("nan")
    else:
        DB = _positive_tuple_ratio(
            _positive_sum_tuples(DB_vals),
            _positive_float_tuple(float(len(DB_vals))),
        )

    min_cdist = np.inf
    for i in range(k):
        for j in range(i + 1, k):
            if med_full[i] >= 0 and med_full[j] >= 0:
                d = M[i, j]
                if d > 0 and d < min_cdist:
                    min_cdist = d
    if np.isfinite(min_cdist) and min_cdist > 0:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            ordinary_ss = float(np.sum(all_within_distances**2))
        positive_within = bool(np.any(all_within_distances > 0.0))
        aggregate_square_range_loss = positive_within and (
            ordinary_ss == 0.0 or not np.isfinite(ordinary_ss)
        )
        if (
            _squares_leave_binary64_range(np.append(all_within_distances, min_cdist))
            or min_cdist > math.sqrt(sys.float_info.max / float(N))
            or aggregate_square_range_loss
        ):
            XB = _stable_weighted_squared_distance_ratio(
                all_within_distances,
                np.ones(all_within_distances.size, dtype=float),
                distance_denominator=min_cdist,
                coefficient_denominator=float(N),
            )
        else:
            XB = float(ordinary_ss / (N * (min_cdist**2)))
    else:
        XB = float("nan")

    # PBM (Pakhira, Bandyopadhyay, and Maulik):
    # ((E1 / EK) * (Dk / k)) ** 2.
    row_terms = [
        [_positive_float_tuple(float(value)) for value in row if value > 0.0] for row in D_full
    ]
    global_med = 0
    for candidate in range(1, len(row_terms)):
        if _positive_expansion_less(row_terms[candidate], row_terms[global_med]):
            global_med = candidate
    E1_tuple = _signed_expansion_tuple(row_terms[global_med])
    if k >= 2 and np.isfinite(M).any():
        Dk = float(np.nanmax(M))
    else:
        Dk = float("nan")
    # The exact-positive branch is scale-sensitive by definition; replacing a
    # positive EK with an absolute floor changes the published PBM formula.
    if EK_tuple[0] == 0.0 or not np.isfinite(Dk):
        PBM = float("nan")
    else:
        E1 = _positive_tuple_to_float(E1_tuple)
        EK = _positive_tuple_to_float(EK_tuple)
        try:
            pbm_ratio = float(E1 / EK)
            pbm_base = float(pbm_ratio * Dk / max(1, k))
            PBM = float(pbm_base**2)
        except OverflowError:
            pbm_base = float("inf")
            PBM = float("inf")
        if (
            not np.isfinite(pbm_base)
            or pbm_base == 0.0
            or PBM == 0.0
            or (0.0 < abs(pbm_ratio) < sys.float_info.min)
        ):
            PBM = _stable_squared_tuple_product_ratio(
                (E1_tuple, _positive_float_tuple(Dk)),
                (EK_tuple, _positive_float_tuple(float(max(1, k)))),
            )
    return {"DB": DB, "XB": XB, "PBM": PBM}


def db_index(diss: np.ndarray, labels: Sequence[int]) -> float:
    """Compute the medoid-based Davies–Bouldin index.

    Parameters
    ----------
    diss : numpy.ndarray
        Square pairwise distance matrix.
    labels : sequence of int
        Cluster labels; arbitrary discrete labels are factorized.

    Returns
    -------
    float
        The average, over every cluster, of its largest scatter-to-separation
        ratio. Valid zero contributions are retained; degeneracies with no
        usable separation return ``nan``.

    Notes
    -----
    This is the medoid/dissimilarity adaptation of Davies and Bouldin (1979).
    """
    return float(compute_internal_indices(diss, labels)["DB"])


def xb_index(diss: np.ndarray, labels: Sequence[int]) -> float:
    """Compute the medoid-based Xie–Beni index.

    Parameters
    ----------
    diss : numpy.ndarray
        Square pairwise distance matrix.
    labels : sequence of int
        Cluster labels; arbitrary discrete labels are factorized.

    Returns
    -------
    float
        Sum of squared distances to cluster medoids divided by sample size and
        squared minimum positive medoid separation. Degenerate separation
        returns ``nan``.

    Notes
    -----
    The formula follows Xie and Beni (1991), adapted to medoids.
    """
    return float(compute_internal_indices(diss, labels)["XB"])


def pbm_index(diss: np.ndarray, labels: Sequence[int]) -> float:
    """Compute the medoid-based PBM validity index.

    Parameters
    ----------
    diss : numpy.ndarray
        Square pairwise distance matrix.
    labels : sequence of int
        Cluster labels; arbitrary discrete labels are factorized.

    Returns
    -------
    float
        ``((E1 / EK) * (Dk / k)) ** 2`` using global and cluster medoids.
        Undefined within-cluster dispersion returns ``nan``.

    Notes
    -----
    The squared formula is from Pakhira, Bandyopadhyay, and Maulik (2004),
    adapted to a dissimilarity/medoid representation. Every exact positive
    within-cluster dispersion is used directly; only zero or nonfinite
    dispersion is undefined.
    """
    return float(compute_internal_indices(diss, labels)["PBM"])


def dunn_index(diss: np.ndarray, labels: Sequence[int]) -> float:
    """Compute Dunn index: min(inter-cluster distance) / max(intra-cluster diameter).

    Parameters
    ----------
    diss : np.ndarray
        (N, N) distance matrix
    labels : array-like
        Cluster labels (1-based or 0-based)

    Returns
    -------
    float
        Dunn index (higher is better)

    Notes
    -----
    Dunn index = min_{i≠j} δ(C_i, C_j) / max_k Δ(C_k)
    where δ is minimum inter-cluster distance and Δ is cluster diameter.
    """
    D, labs = _coerce_internal_index_inputs(diss, labels)

    k = int(labs.max() + 1) if labs.size > 0 else 0
    if k < 2:
        return float("nan")

    # Compute cluster diameters (max intra-cluster distance)
    max_intra = 0.0
    for c in range(k):
        idx = np.where(labs == c)[0]
        if idx.size <= 1:
            continue
        sub = D[np.ix_(idx, idx)]
        diam = float(np.max(sub))
        if diam > max_intra:
            max_intra = diam

    if max_intra <= 0:
        return float("nan")

    # Compute minimum inter-cluster distance
    min_inter = np.inf
    for i in range(k):
        for j in range(i + 1, k):
            idx_i = np.where(labs == i)[0]
            idx_j = np.where(labs == j)[0]
            if idx_i.size == 0 or idx_j.size == 0:
                continue
            # Min distance between any pair from different clusters
            sub = D[np.ix_(idx_i, idx_j)]
            min_dist = float(np.min(sub))
            if min_dist < min_inter:
                min_inter = min_dist

    if not np.isfinite(min_inter):
        return float("nan")

    return float(min_inter / max_intra)


def _compute_hg_hgsd_weighted(
    D: np.ndarray, labels0: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Compatibility wrapper around the directly testable pure quality kernel."""
    from ._fallback import _compute_hg_hgsd_weighted as _fallback_hg_hgsd

    return _fallback_hg_hgsd(D, labels0, weights)


def _apply_nonfinite_silhouette_contract(
    result: Dict[str, Any],
    *,
    D: np.ndarray,
    labels0: np.ndarray,
    cluster_levels: Sequence[Any],
    weights: np.ndarray | None,
) -> Dict[str, Any]:
    """Propagate non-finite observation widths like the R C quality kernel."""
    if np.all(np.isfinite(D)):
        return result
    w = np.ones(D.shape[0], dtype=float) if weights is None else weights
    sizes = np.bincount(labels0, weights=w, minlength=len(cluster_levels)).astype(float)
    asww_obs = wcSilhouetteObs(D, labels0, weights=w, measure="ASWw")
    if np.any(sizes < 1.0):
        asw_obs = np.full(D.shape[0], np.nan, dtype=float)
    else:
        asw_obs = wcSilhouetteObs(D, labels0, weights=w, measure="ASW")

    active = w > 0.0

    def _summarize(values: np.ndarray) -> tuple[float, Dict[Any, float]]:
        numerators = np.zeros(len(cluster_levels), dtype=float)
        for index in range(len(cluster_levels)):
            selected = (labels0 == index) & active
            numerators[index] = float(np.sum(w[selected] * values[selected]))
        total = float(np.sum(w))
        overall = float(np.sum(numerators) / total) if total != 0.0 else float("nan")
        per_cluster = {
            cluster_levels[index]: (
                float(numerators[index] / sizes[index]) if sizes[index] != 0.0 else float("nan")
            )
            for index in range(len(cluster_levels))
        }
        return overall, per_cluster

    stats_asw, per_asw = _summarize(asw_obs)
    stats_asww, per_asww = _summarize(asww_obs)
    result["stats"]["ASW"] = stats_asw
    result["stats"]["ASWw"] = stats_asww
    result["ASW"] = per_asw
    result["ASWw"] = per_asww
    return result


def _pure_public_quality_result(
    *,
    D: np.ndarray,
    labels0: np.ndarray,
    cluster_levels: Sequence[Any],
    weights: np.ndarray | None,
) -> Dict[str, Any]:
    from ._fallback import cluster_quality_r_permissive as _cq_pure

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        stats, asw_raw, asww_raw = _cq_pure(
            D, labels0, weights=weights, pairwise_sum=weighted_pairwise_sum
        )
    asw = {cluster_levels[int(key) - 1]: float(value) for key, value in asw_raw.items()}
    asww = {cluster_levels[int(key) - 1]: float(value) for key, value in asww_raw.items()}
    effective_weights = np.ones(D.shape[0], dtype=float) if weights is None else weights
    if has_cluster_weight_below_one(labels0, effective_weights):
        for key in asw:
            asw[key] = float("nan")
        stats["ASW"] = float("nan")
        warnings.warn(
            "ASW can not be computed because at least one cluster has less than one observation.",
            UserWarning,
            stacklevel=3,
        )
    stats = {name: float(stats[name]) for name in _R_CLUSTER_QUALITY_NAMES}
    if (
        np.all(np.isfinite(D))
        and np.all(D >= 0.0)
        and np.allclose(D, D.T)
        and np.allclose(np.diag(D), 0.0)
    ):
        internal_idx = compute_internal_indices(D, labels0)
        stats["DB"] = internal_idx.get("DB", float("nan"))
        stats["XB"] = internal_idx.get("XB", float("nan"))
        stats["PBM"] = internal_idx.get("PBM", float("nan"))
        stats["Dunn"] = dunn_index(D, labels0)
    else:
        # The ten inherited R statistics intentionally retain upstream edge
        # propagation. Python-only indices have a stricter distance contract
        # and are undefined rather than weakening their direct public APIs.
        stats.update({name: float("nan") for name in ("DB", "XB", "PBM", "Dunn")})
    return _apply_nonfinite_silhouette_contract(
        {"stats": stats, "ASW": asw, "ASWw": asww},
        D=D,
        labels0=labels0,
        cluster_levels=cluster_levels,
        weights=weights,
    )


def wcFuzzyQuality(
    diss: np.ndarray,
    membership: np.ndarray,
    *,
    medoids: Optional[Sequence[int]] = None,
    m: float = 1.5,
    weights: Optional[Sequence[float]] = None,
) -> Dict:
    """Compute FPC, partition entropy, XBf, and AMS for fuzzy membership.

    Partition entropy uses every exact positive normalized membership. Exact
    zero contributes zero by continuity and is never passed to ``log``.
    Membership rows, observation weights, and finite distances are each
    carried on a common relative scale through the scale-invariant indices.
    """
    D = np.asarray(diss, dtype=float)
    validate_distance_matrix(D, name="diss")
    U = np.asarray(membership, dtype=float)
    if U.ndim != 2 or U.shape[0] != D.shape[0]:
        raise ValueError("membership must be (N,K) with same N as diss")
    N, K = U.shape
    if np.any(~np.isfinite(U)) or np.any(U < 0.0):
        raise ValueError("membership must contain finite non-negative values")
    if K == 0:
        raise ValueError("membership rows must have positive membership")
    raw_membership = U.copy()
    row_scale = np.max(U, axis=1, keepdims=True)
    if np.any(row_scale <= 0.0):
        raise ValueError("membership rows must have positive membership")
    m_value = float(m)
    if not np.isfinite(m_value) or m_value <= 1.0:
        raise ValueError("m must be a finite number greater than one")

    if weights is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (N,):
            raise ValueError("weights must be length N")
        if np.any(~np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must contain finite non-negative values")

    if not np.any(w > 0.0):
        raise ValueError("weights must sum to a positive value")
    weight_tuples = [_positive_float_tuple(float(weight)) for weight in w]
    weight_total_tuple = _positive_sum(w)
    exact_weight_total_required = _positive_terms_need_exact_sum(weight_tuples)

    membership_tuples: list[list[_PositivePowerBase]] = []
    membership_total_components: list[_DyadicComponent] = []
    for row in raw_membership:
        row_components = [
            (
                _tuple_to_dyadic_component(_positive_float_tuple(float(value)))
                if value > 0.0
                else (0, 0)
            )
            for value in row
        ]
        row_total_component = _exact_expansion_component(
            [_positive_float_tuple(float(value)) for value in row if value > 0.0]
        )
        membership_total_components.append(row_total_component)
        membership_tuples.append(
            [
                (component, row_total_component) if component[0] > 0 else _ZERO_POSITIVE_TUPLE
                for component in row_components
            ]
        )

    # Normalize each row only after dividing by its largest finite entry, so
    # the reduction cannot overflow even when the finite raw row total cannot
    # itself be represented. Observation weights are likewise carried on one
    # common relative scale through every downstream fuzzy statistic.
    with np.errstate(over="ignore"):
        ordinary_row_total = U.sum(axis=1, keepdims=True)
    ordinary_rows = np.isfinite(ordinary_row_total[:, 0])
    normalized_membership = np.empty_like(U)
    normalized_membership[ordinary_rows] = U[ordinary_rows] / ordinary_row_total[ordinary_rows]
    scaled_membership = U[~ordinary_rows] / row_scale[~ordinary_rows]
    normalized_membership[~ordinary_rows] = scaled_membership / scaled_membership.sum(
        axis=1,
        keepdims=True,
    )
    U = normalized_membership
    # FPC and FPE
    row_fpc = np.sum(U * U, axis=1)
    entropy_terms = np.zeros_like(U)
    positive = U > 0.0
    entropy_terms[positive] = U[positive] * np.log(U[positive])
    row_fpe = -np.sum(entropy_terms, axis=1)
    for index, (row, _total_component) in enumerate(
        zip(membership_tuples, membership_total_components, strict=True)
    ):
        if any(_power_base_denominator_inexact(base) for base in row):
            raw_terms = [
                _positive_float_tuple(float(value))
                for value in raw_membership[index]
                if value > 0.0
            ]
            represented_total = _positive_tuple_from_component(_total_component)
            residual = _dyadic_component_difference(
                _total_component,
                _tuple_to_dyadic_component(represented_total),
            )
            relative_residual = (
                _positive_tuple_ratio(
                    _positive_tuple_from_component((abs(residual[0]), residual[1])),
                    represented_total,
                )
                if residual[0] != 0
                else 0.0
            )
            range_loss_signature = _positive_terms_need_exact_sum(raw_terms) or (
                relative_residual >= math.ulp(float(row_fpe[index])) / 2.0
            )
            if range_loss_signature:
                row_fpc[index] = _decimal_membership_fpc(row)
                row_fpe[index] = _decimal_membership_entropy(row)
    FPC = _positive_weighted_average(row_fpc, w)
    FPE = _positive_weighted_average(row_fpe, w)
    # medoids
    if medoids is None:
        med = np.empty(K, dtype=int)
        all_idx = np.arange(N, dtype=int)
        for k in range(K):
            ordinary_coefficients, coefficient_tuples = _powered_coefficients_for_objective(
                U[:, k],
                w,
                m_value,
                [row[k] for row in membership_tuples],
                weight_tuples,
            )
            med[k] = _find_medoid_without_range_loss(
                D,
                all_idx,
                ordinary_coefficients,
                coefficient_tuples,
                [row[k] for row in membership_tuples],
                weight_tuples,
                m_value,
            )
        med_arr = med
    else:
        med_raw = np.asarray(medoids)
        try:
            med_numeric = np.asarray(med_raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("medoids must contain integer observation indices") from exc
        if (
            med_raw.ndim != 1
            or np.issubdtype(med_raw.dtype, np.bool_)
            or not np.all(np.isfinite(med_numeric))
            or not np.all(med_numeric == np.floor(med_numeric))
        ):
            raise ValueError("medoids must contain integer observation indices")
        med_arr = med_numeric.astype(int)
        if med_arr.shape != (K,):
            raise ValueError("medoids must have length equal to membership.shape[1]")
        if np.any((med_arr < 0) | (med_arr >= N)):
            raise ValueError("medoids must contain valid observation indices")
    # XBf
    min_cdist = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            dij = float(D[med_arr[i], med_arr[j]])
            if dij > 0 and dij < min_cdist:
                min_cdist = dij
    if np.isfinite(min_cdist) and min_cdist > 0:
        fuzzy_distances: list[np.ndarray] = []
        fuzzy_memberships: list[np.ndarray] = []
        fuzzy_weights: list[np.ndarray] = []
        fuzzy_coefficients: list[np.ndarray] = []
        fuzzy_membership_tuples: list[_PositivePowerBase] = []
        fuzzy_weight_tuples: list[_PositiveTuple] = []
        for k in range(K):
            d_to_med = D[:, med_arr[k]].astype(float)
            fuzzy_distances.append(d_to_med)
            fuzzy_memberships.append(U[:, k])
            fuzzy_weights.append(w)
            fuzzy_membership_tuples.extend(row[k] for row in membership_tuples)
            fuzzy_weight_tuples.extend(weight_tuples)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                fuzzy_coefficients.append((U[:, k] ** m_value) * w)
        all_fuzzy_distances = np.concatenate(fuzzy_distances)
        all_fuzzy_memberships = np.concatenate(fuzzy_memberships)
        all_fuzzy_weights = np.concatenate(fuzzy_weights)
        all_fuzzy_coefficients = np.concatenate(fuzzy_coefficients)
        mathematical_positive = np.asarray(
            [
                _power_base_is_positive(membership) and weight[0] > 0.0
                for membership, weight in zip(
                    fuzzy_membership_tuples, fuzzy_weight_tuples, strict=True
                )
            ]
        )
        active = mathematical_positive & (all_fuzzy_distances > 0.0)
        active_distances = all_fuzzy_distances[active]
        exact_membership_ratio = any(
            _power_base_requires_exact_ratio(base, m_value) for base in fuzzy_membership_tuples
        )
        coefficient_range_loss = (
            np.any(
                mathematical_positive
                & ((all_fuzzy_coefficients == 0.0) | ~np.isfinite(all_fuzzy_coefficients))
            )
            or exact_membership_ratio
        )
        weight_total = _positive_tuple_to_float(weight_total_tuple)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            ordinary_denominator = weight_total * float(np.square(min_cdist))
            ordinary_active_terms = all_fuzzy_coefficients[active] * (
                all_fuzzy_distances[active] ** 2
            )
            ordinary_numerator = float(np.sum(ordinary_active_terms))
        joint_product_range_loss = bool(
            ordinary_denominator == 0.0
            or not np.isfinite(ordinary_denominator)
            or (0.0 < ordinary_denominator < sys.float_info.min)
            or (
                np.any(ordinary_active_terms > 0.0)
                and (
                    ordinary_numerator == 0.0
                    or not np.isfinite(ordinary_numerator)
                    or ordinary_numerator < sys.float_info.min
                )
            )
            or np.any((ordinary_active_terms > 0.0) & (ordinary_active_terms < sys.float_info.min))
            or np.any(
                (all_fuzzy_coefficients[active] > 0.0)
                & (all_fuzzy_distances[active] > 0.0)
                & ((ordinary_active_terms == 0.0) | ~np.isfinite(ordinary_active_terms))
            )
        )
        if exact_membership_ratio or exact_weight_total_required:
            XBf = _decimal_powered_weighted_squared_distance_ratio(
                all_fuzzy_distances,
                fuzzy_membership_tuples,
                all_fuzzy_weights,
                membership_power=m_value,
                distance_denominator=min_cdist,
                coefficient_weights=w,
            )
        elif (
            coefficient_range_loss
            or joint_product_range_loss
            or not np.isfinite(weight_total)
            or weight_total < sys.float_info.min
            or _squares_leave_binary64_range(np.append(active_distances, min_cdist))
            or (weight_total > 0.0 and min_cdist > math.sqrt(sys.float_info.max / weight_total))
        ):
            if (
                not coefficient_range_loss
                and not exact_membership_ratio
                and not exact_weight_total_required
                and np.isfinite(weight_total)
                and weight_total > 0.0
            ):
                # The normalized binary64 coefficients are already the public
                # ordinary-path values. Preserve those bits when only the
                # distance square/reduction needs a wider exponent range.
                XBf = _stable_weighted_squared_distance_ratio(
                    all_fuzzy_distances,
                    all_fuzzy_coefficients,
                    distance_denominator=min_cdist,
                    coefficient_denominator=weight_total,
                )
            else:
                XBf = _stable_powered_weighted_squared_distance_ratio(
                    all_fuzzy_distances,
                    all_fuzzy_memberships,
                    all_fuzzy_weights,
                    membership_power=m_value,
                    distance_denominator=min_cdist,
                    coefficient_denominator=weight_total_tuple,
                    membership_tuples=fuzzy_membership_tuples,
                    weight_tuples=fuzzy_weight_tuples,
                )
        else:
            XBf = float(ordinary_numerator / ordinary_denominator)
    else:
        XBf = float("nan")
    # AMS
    labels1b = (np.argmax(U, axis=1) + 1).astype(int)
    sil_measure = "ASWw" if weights is not None else "ASW"
    sil = wcSilhouetteObs(D, labels1b, weights=w, measure=sil_measure)
    diffs = []
    ambiguity_bases: list[_PositivePowerBase] = []
    for i in range(N):
        row = np.sort(U[i])[::-1]
        diffs.append((row[0] - row[1]) if row.size >= 2 else (row[0] if row.size == 1 else 0.0))
        order = np.argsort(raw_membership[i])[::-1]
        top = membership_tuples[i][int(order[0])]
        second = membership_tuples[i][int(order[1])] if order.size >= 2 else _ZERO_POSITIVE_TUPLE
        if isinstance(top[0], tuple):
            top_numerator, row_denominator = top
        else:
            top_numerator = _tuple_to_dyadic_component(top)
            row_denominator = _tuple_to_dyadic_component(_positive_float_tuple(1.0))
        second_numerator = (
            second[0] if isinstance(second[0], tuple) else _tuple_to_dyadic_component(second)
        )
        ambiguity_bases.append(
            (_dyadic_component_difference(top_numerator, second_numerator), row_denominator)
        )
    diffs = np.asarray(diffs, dtype=float)
    effective_weights = [
        _positive_product_tuple(float(weight), float(diff))
        for weight, diff in zip(w, diffs, strict=True)
        if weight > 0.0 and diff > 0.0
    ]
    if effective_weights:
        positive_components: list[_DyadicComponent] = []
        negative_components: list[_DyadicComponent] = []
        denominator_components: list[_DyadicComponent] = []
        for weight, diff, silhouette in zip(w, diffs, sil, strict=True):
            if weight <= 0.0 or diff <= 0.0 or not np.isfinite(silhouette):
                continue
            effective = _exact_positive_float_product_component(float(weight), float(diff))
            denominator_components.append(effective)
            if silhouette > 0.0:
                positive_components.append(
                    _dyadic_component_product(
                        effective,
                        _tuple_to_dyadic_component(_positive_float_tuple(float(silhouette))),
                    )
                )
            elif silhouette < 0.0:
                negative_components.append(
                    _dyadic_component_product(
                        effective,
                        _tuple_to_dyadic_component(_positive_float_tuple(float(-silhouette))),
                    )
                )
        numerator_component = _sum_dyadic_components(positive_components, negative_components)
        denominator_component = _sum_dyadic_components(denominator_components)
        numerator_sign = 1.0 if numerator_component[0] >= 0 else -1.0
        AMS = (
            0.0
            if denominator_component[0] == 0
            else numerator_sign
            * _dyadic_ratio_decimal_float(
                (abs(numerator_component[0]), numerator_component[1]),
                denominator_component,
            )
        )
    else:
        finite = np.isfinite(sil)
        positive_components = []
        negative_components = []
        for weight, silhouette in zip(w[finite], sil[finite], strict=True):
            term = _exact_positive_float_product_component(float(weight), float(abs(silhouette)))
            (positive_components if silhouette >= 0.0 else negative_components).append(term)
        numerator_component = _sum_dyadic_components(positive_components, negative_components)
        denominator_component = _sum_dyadic_components(
            [
                _tuple_to_dyadic_component(_positive_float_tuple(float(weight)))
                for weight in w[finite]
                if weight > 0.0
            ]
        )
        AMS = (
            0.0
            if denominator_component[0] == 0
            else (1.0 if numerator_component[0] >= 0 else -1.0)
            * _dyadic_ratio_decimal_float(
                (abs(numerator_component[0]), numerator_component[1]),
                denominator_component,
            )
        )
    if any(_power_base_denominator_inexact(base) for base in ambiguity_bases):
        AMS = _decimal_ambiguity_weighted_silhouette(sil, w, ambiguity_bases)
    return {"stats": {"FPC": FPC, "FPE": FPE, "XBf": XBf, "AMS": AMS}}


def _wc_cluster_quality_internal(
    diss: np.ndarray,
    clustering: Sequence[Any],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Evaluate the **unguarded** quality kernel that R's bootstrap paths use.

    The name is the inverse of R's, and deliberately so. In WeightedCluster,
    ``wcClusterQualityInternal`` is where the "at least one cluster has less
    than one observation -> ASW = NA" guard lives, and ``clustrangeboot``
    bypasses it by calling the native routine directly. Here the guard lives in
    the public :func:`wcClusterQuality` and *this* function is the permissive
    kernel the bootstrap paths reach, so a reader porting from R will look for
    the guard under this name and not find it.

    The behaviour is correct and must not change: the bootstrap path agrees with
    R's real ``clustrangeboot`` to 1e-12. Finding F-19 of the 2026-08-24
    whole-codebase audit records the naming hazard only.
    """
    labels0, cluster_levels = _factorize_quality_labels(clustering)
    _validate_quality_partition(labels0, cluster_levels)
    D = _coerce_quality_distance(diss, expected_n=labels0.size)
    w = _coerce_quality_weights(weights, expected_n=D.shape[0])

    from ._fallback import cluster_quality_r_permissive

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        stats, asw_raw, asww_raw = cluster_quality_r_permissive(
            D,
            labels0,
            weights=w,
            pairwise_sum=weighted_pairwise_sum,
        )
    return {
        "stats": {name: float(stats[name]) for name in _R_CLUSTER_QUALITY_NAMES},
        "ASW": {cluster_levels[int(key) - 1]: float(value) for key, value in asw_raw.items()},
        "ASWw": {cluster_levels[int(key) - 1]: float(value) for key, value in asww_raw.items()},
    }


def wcClusterQuality(
    diss: np.ndarray, clustering: Sequence[int], weights: Optional[Sequence[float]] = None
) -> Dict:
    """Compute cluster quality statistics with optimized memory usage.

    Parameters
    ----------
    diss : np.ndarray
        (N, N) distance matrix
    clustering : array-like
        Cluster labels (1-based or 0-based, will be normalized)
    weights : array-like, optional
        Sample weights

    Returns
    -------
    dict
        Dictionary with keys 'stats', 'ASW', 'ASWw' containing quality metrics

    Notes
    -----
    This function attempts to use compiled implementations first, then falls back
    to a memory-optimized Python implementation that computes pairwise statistics
    incrementally rather than materializing full upper triangle arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from weightedcluster import wcClusterQuality
    >>> # Create a simple distance matrix
    >>> D = np.array([[0.0, 1.0, 4.0, 5.0],
    ...               [1.0, 0.0, 3.0, 4.0],
    ...               [4.0, 3.0, 0.0, 1.0],
    ...               [5.0, 4.0, 1.0, 0.0]])
    >>> # 1-based cluster labels
    >>> labels = np.array([1, 1, 2, 2])
    >>> result = wcClusterQuality(D, labels)
    >>> # Access overall quality statistics
    >>> result['stats']['ASW']  # Average Silhouette Width
    0.746031746031746
    >>> result['stats']['CH']   # Calinski-Harabasz index
    7.0
    >>> # Per-cluster ASW (unweighted)
    >>> result['ASW']
    {1: 0.746031746031746, 2: 0.746031746031746}

    >>> # With custom weights
    >>> weights = np.array([1.0, 2.0, 1.5, 1.0])
    >>> result_w = wcClusterQuality(D, labels, weights=weights)
    >>> result_w['stats']['ASWw']  # Weighted Average Silhouette Width
    0.8788990688455929
    >>> # Per-cluster weighted ASW
    >>> result_w['ASWw']
    {1: 0.8841354723707665, 2: 0.8726153846153846}
    """
    labels0, cluster_levels = _factorize_quality_labels(clustering)
    _validate_quality_partition(labels0, cluster_levels)
    D = _coerce_quality_distance(diss, expected_n=labels0.size)
    N = D.shape[0]
    w = _coerce_quality_weights(weights, expected_n=N)
    effective_weights = np.ones(N, dtype=float) if w is None else w
    cluster_weights = np.bincount(labels0, weights=effective_weights, minlength=len(cluster_levels))
    if (
        np.any(effective_weights < 0.0)
        or np.any(cluster_weights == 0.0)
        or not np.all(np.isfinite(effective_weights))
        or not np.all(np.isfinite(D))
        or np.any(D < 0.0)
    ):
        # Native R ignores non-positive observations in ordered-pair terms but
        # retains their raw values in cluster-size denominators. The optimized
        # kernels intentionally target ordinary non-negative case weights, so
        # this source-supported edge path uses the faithful scalar fallback.
        return _pure_public_quality_result(
            D=D,
            labels0=labels0,
            cluster_levels=cluster_levels,
            weights=w,
        )
    # Prefer compiled fast path
    try:
        from ._wc_core import cluster_quality as _cluster_quality  # type: ignore
    except ImportError:
        _cluster_quality = None
    if _cluster_quality is not None:
        stats, asw, asw_w = _cluster_quality(D, labels0, weights=w)
        # Convert py::dict to Python dict of primitives
        stats_py: Dict[str, float] = {str(k): float(v) for k, v in stats.items()}
        ASW: Dict[Any, float] = {cluster_levels[int(k) - 1]: float(v) for k, v in asw.items()}
        ASWw: Dict[Any, float] = {cluster_levels[int(k) - 1]: float(v) for k, v in asw_w.items()}

        # Mirror R behavior: when any cluster has weighted size < 1, ASW is undefined.
        if labels0.size > 0:
            w_eff = np.ones(N, dtype=float) if w is None else w
            if has_cluster_weight_below_one(labels0, w_eff):
                for key in list(ASW.keys()):
                    ASW[key] = float("nan")
                stats_py["ASW"] = float("nan")
                warnings.warn(
                    "ASW can not be computed because at least one cluster has less than one observation.",
                    UserWarning,
                    stacklevel=2,
                )

        # Augment with additional indices not in core
        internal_idx = compute_internal_indices(D, labels0)
        stats_py["DB"] = internal_idx.get("DB", float("nan"))
        stats_py["XB"] = internal_idx.get("XB", float("nan"))
        stats_py["PBM"] = internal_idx.get("PBM", float("nan"))
        stats_py["Dunn"] = dunn_index(D, labels0)

        # No non-finite contract call here: wcClusterQuality's guard above
        # routes every non-finite matrix to _pure_public_quality_result, so this
        # branch only ever sees a wholly finite D and the contract would return
        # immediately. Finding F-18 removed the two no-op calls so the routing
        # guard is the single place the non-finite contract is decided.
        return {"stats": stats_py, "ASW": ASW, "ASWw": ASWw}
    # Try the Numba fallback. Import/availability failure is the only event
    # that may select a different backend; runtime ImportError must propagate.
    try:
        from ._numba import cluster_quality_numba as _cq_nb  # type: ignore
    except ImportError:
        _cq_nb = None
    if _cq_nb is not None:
        stats_nb, asw_nb, asw_w_nb = _cq_nb(D, labels0, weights=w)
        if labels0.size > 0:
            w_eff = np.ones(N, dtype=float) if w is None else w
            if has_cluster_weight_below_one(labels0, w_eff):
                for key in list(asw_nb.keys()):
                    asw_nb[key] = float("nan")
                stats_nb["ASW"] = float("nan")
                warnings.warn(
                    "ASW can not be computed because at least one cluster has less than one observation.",
                    UserWarning,
                    stacklevel=2,
                )

        # Augment with additional indices not in numba
        asw_nb = {cluster_levels[int(key) - 1]: float(value) for key, value in asw_nb.items()}
        asw_w_nb = {cluster_levels[int(key) - 1]: float(value) for key, value in asw_w_nb.items()}
        internal_idx = compute_internal_indices(D, labels0)
        stats_nb["DB"] = internal_idx.get("DB", float("nan"))
        stats_nb["XB"] = internal_idx.get("XB", float("nan"))
        stats_nb["PBM"] = internal_idx.get("PBM", float("nan"))
        stats_nb["Dunn"] = dunn_index(D, labels0)

        # As for the compiled branch above: unreachable with a non-finite D.
        return {"stats": stats_nb, "ASW": asw_nb, "ASWw": asw_w_nb}
    # Direct pure kernel: the ten R statistics stay separate from the four
    # Python-only indices added by this public workflow.
    return _pure_public_quality_result(
        D=D,
        labels0=labels0,
        cluster_levels=cluster_levels,
        weights=w,
    )

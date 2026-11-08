from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def validate_kernel_distance(diss: np.ndarray) -> np.ndarray:
    """Coerce the shared private-kernel distance boundary safely."""
    try:
        matrix = np.asarray(diss, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("diss must contain numeric distances") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("diss must be square")
    if matrix.shape[0] == 0:
        raise ValueError("diss must be non-empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("diss must contain only finite distances")
    if np.any(matrix < 0.0):
        raise ValueError("diss distances must be non-negative")
    return matrix


def relative_finite_distance_matrix(diss: np.ndarray) -> np.ndarray:
    """Scale only when raw finite distance squares leave binary64 range."""
    matrix = np.asarray(diss, dtype=float)
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return matrix
    distance_scale = float(np.max(matrix))
    positive = matrix[matrix > 0.0]
    if positive.size == 0:
        return matrix
    if not finite_distance_squares_need_scaling(matrix):
        return matrix
    return matrix / distance_scale


def finite_distance_squares_need_scaling(diss: np.ndarray) -> bool:
    """Return whether squared finite-distance reductions can lose range/precision."""
    matrix = np.asarray(diss, dtype=float)
    positive = matrix[matrix > 0.0]
    if positive.size == 0 or not np.all(np.isfinite(positive)):
        return False
    square_min = math.sqrt(float(np.finfo(float).tiny))
    # A reduction can overflow even when every individual square is finite.
    square_max = math.sqrt(float(np.finfo(float).max) / int(positive.size))
    _mantissas, exponents = np.frexp(positive)
    rounding_window = 52 - (int(positive.size) - 1).bit_length()
    exponent_spread_loses_moments = bool(
        2 * (int(np.max(exponents)) - int(np.min(exponents))) >= rounding_window
    )
    return bool(
        float(np.min(positive)) < square_min
        or float(np.max(positive)) > square_max
        or exponent_spread_loses_moments
    )


def finite_pair_masses_need_scaling(weights: np.ndarray) -> bool:
    """Return whether positive pair-mass products or their sum lose binary64 range."""
    values = np.asarray(weights, dtype=float)
    positive = values[values > 0.0]
    if positive.size == 0:
        return False
    scale = float(np.max(positive))
    relative = positive / scale
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        squares = positive * positive
    pair_count = max(1, int(positive.size) ** 2)
    aggregate_limit = math.sqrt(float(np.finfo(float).max) / (2.0 * pair_count))
    second_order_minimum = math.sqrt(math.sqrt(float(np.finfo(float).tiny)))
    second_order_maximum = math.sqrt(
        math.sqrt(float(np.finfo(float).max) / (4.0 * pair_count * pair_count))
    )
    _mantissas, exponents = np.frexp(positive)
    pair_rounding_window = 52 - (pair_count - 1).bit_length()
    exponent_spread_loses_pairs = bool(
        2 * (int(np.max(exponents)) - int(np.min(exponents))) >= pair_rounding_window
    )
    return bool(
        np.any(relative == 0.0)
        or np.any(~np.isfinite(squares))
        or np.any(squares < np.finfo(float).tiny)
        or scale > aggregate_limit
        or float(np.min(positive)) < second_order_minimum
        or scale > second_order_maximum
        or exponent_spread_loses_pairs
    )


def validate_kernel_labels(labels: np.ndarray, *, expected_n: int) -> np.ndarray:
    """Require the zero-based contiguous labels used by owned kernels."""
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.shape[0] != expected_n:
        raise ValueError(f"labels must be a vector of length {expected_n}")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("labels must contain integers")
    labels0 = raw.astype(int, copy=False)
    unique = np.unique(labels0)
    if unique.size == 0 or not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("labels must be contiguous zero-based integers")
    return labels0


def validate_kernel_weights(
    weights: np.ndarray | None,
    *,
    expected_n: int,
) -> np.ndarray:
    """Coerce finite direct-kernel weights using the tiers' clipped contract."""
    if weights is None:
        return np.ones(expected_n, dtype=float)
    try:
        values = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must contain numeric values") from exc
    if values.ndim != 1 or values.shape[0] != expected_n:
        raise ValueError(f"weights must be a vector of length {expected_n}")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must contain only finite values")
    return np.maximum(values, 0.0)


def weighted_cluster_sizes(labels0: np.ndarray, weights: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels0, dtype=int)
    w = np.asarray(weights, dtype=float)
    if labels.size == 0:
        return np.zeros(0, dtype=float)
    k = int(labels.max()) + 1
    return np.bincount(labels, weights=w, minlength=k).astype(float)


def has_cluster_weight_below_one(labels0: np.ndarray, weights: np.ndarray) -> bool:
    labels = np.asarray(labels0, dtype=int)
    values = np.asarray(weights, dtype=float)
    if labels.size == 0:
        return False
    if not np.all(np.isfinite(values)):
        sizes = weighted_cluster_sizes(labels, values)
        return bool(sizes.size > 0 and np.any(sizes < 1.0))
    for cluster in range(int(labels.max()) + 1):
        components: list[tuple[int, int]] = []
        for weight in values[labels == cluster]:
            if weight == 0.0:
                continue
            numerator, denominator = float(weight).as_integer_ratio()
            components.append((numerator, -(denominator.bit_length() - 1)))
        if not components:
            return True
        common_exponent = min(exponent for _integer, exponent in components)
        total = sum(integer << (exponent - common_exponent) for integer, exponent in components)
        if common_exponent >= 0:
            below_one = (total << common_exponent) < 1
        else:
            below_one = total < (1 << -common_exponent)
        if below_one:
            return True
    return False


def cluster_asw_core_compatible(
    diss: np.ndarray,
    labels0: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
    """Match the core ASW formulas with exact-positive finite accumulation.

    Tier contract: the compiled kernel's inline aggregation, the permissive
    kernel's loop, this helper (the Numba and validated pure tiers), and R's
    kernel all propagate a non-finite positive-weight observation silhouette
    into the overall and per-cluster aggregates. The masking that once lived
    here was finding WC-090303.
    """
    D = np.asarray(diss, dtype=float)
    labels = np.asarray(labels0, dtype=int)
    w = np.asarray(weights, dtype=float)
    k = int(labels.max()) + 1 if labels.size else 0
    if k == 0:
        return float("nan"), float("nan"), {}, {}

    # Import lazily: quality.py imports this contract module during startup.
    from .quality import wcSilhouetteObs

    asww_observations = wcSilhouetteObs(D, labels, weights=w, measure="ASWw")
    try:
        asw_observations = wcSilhouetteObs(D, labels, weights=w, measure="ASW")
        asw_undefined = False
    except ValueError:
        asw_observations = np.full(labels.shape, np.nan, dtype=float)
        asw_undefined = True
    return aggregate_observation_silhouettes(
        asw_observations,
        asww_observations,
        labels,
        w,
        k,
        asw_undefined=asw_undefined,
    )


def aggregate_observation_silhouettes(
    asw_observations: np.ndarray,
    asww_observations: np.ndarray,
    labels0: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    asw_undefined: bool = False,
) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
    """Aggregate observation silhouettes into the kernel's ASW/ASWw outputs.

    R's kernel accumulates every positive-weight observation silhouette, so one
    genuine 0/0 = NaN observation makes the aggregate NaN. This helper used to
    mask non-finite values instead, which made the Numba and validated pure
    tiers disagree with live R, the compiled kernel, and the permissive kernel
    on such inputs, and made the public answer flip under a positive rescaling
    into the range-risk path (2026-09-03 audit, finding WC-090303). The
    permissive kernel's range-risk route shares it since finding WC-090304.
    """
    from .quality import _signed_weighted_average

    labels = np.asarray(labels0, dtype=int)
    w = np.asarray(weights, dtype=float)

    def weighted_average(values: np.ndarray, mask: np.ndarray) -> float:
        selected = mask & (w > 0.0)
        if not np.any(selected):
            return float("nan")
        selected_values = values[selected]
        if not np.all(np.isfinite(selected_values)):
            return float("nan")
        return _signed_weighted_average(selected_values, w[selected])

    asw = {
        cluster
        + 1: (
            float("nan") if asw_undefined else weighted_average(asw_observations, labels == cluster)
        )
        for cluster in range(k)
    }
    asww = {
        cluster + 1: weighted_average(asww_observations, labels == cluster) for cluster in range(k)
    }
    stats_asw = (
        float("nan")
        if asw_undefined
        else weighted_average(asw_observations, np.ones(labels.shape, dtype=bool))
    )
    stats_asww = weighted_average(asww_observations, np.ones(labels.shape, dtype=bool))
    return stats_asw, stats_asww, asw, asww

"""Pure-Python clustering fallbacks for medoids and cluster quality.

Sequence *distances* used to live here too, but Phase 1 / A1 of MIGRATION_PLAN.md
moved all distance computation to traminex; ``weightedcluster._seqdist`` now
delegates there. Only clustering and quality kernels remain — this project's
own (WeightedCluster) territory, kept as NumPy fallbacks for the compiled core.
"""

from __future__ import annotations

import math
import sys
from decimal import Decimal, localcontext
from typing import Optional, Sequence, Tuple

import numpy as np
from traminex.diss import weighted_pairwise_sum as _tx_weighted_pairwise_sum

from ._constants import PAMONCE_SWAP_TOLERANCE
from ._quality_contract import (
    aggregate_observation_silhouettes,
    cluster_asw_core_compatible,
    finite_distance_squares_need_scaling,
    finite_pair_masses_need_scaling,
    relative_finite_distance_matrix,
    validate_kernel_distance,
    validate_kernel_labels,
    validate_kernel_weights,
)

_DyadicComponent = tuple[int, int]


def _float_component(value: float, *, sign: int = 1) -> _DyadicComponent:
    """Return one finite binary64 as an exact signed dyadic component."""
    if value == 0.0:
        return 0, 0
    numerator, denominator = float(value).as_integer_ratio()
    return sign * numerator, -(denominator.bit_length() - 1)


def _product_component(
    left: float,
    right: float,
    *,
    sign: int = 1,
) -> _DyadicComponent:
    left_integer, left_exponent = _float_component(left)
    right_integer, right_exponent = _float_component(right)
    return sign * left_integer * right_integer, left_exponent + right_exponent


def _sum_components(components: Sequence[_DyadicComponent]) -> _DyadicComponent:
    nonzero = [component for component in components if component[0] != 0]
    if not nonzero:
        return 0, 0
    common_exponent = min(exponent for _integer, exponent in nonzero)
    return (
        sum(integer << (exponent - common_exponent) for integer, exponent in nonzero),
        common_exponent,
    )


def _component_less(left: _DyadicComponent, right: _DyadicComponent) -> bool:
    common_exponent = min(left[1], right[1])
    return (left[0] << (left[1] - common_exponent)) < (right[0] << (right[1] - common_exponent))


def _component_to_float(component: _DyadicComponent) -> float:
    integer, exponent = component
    if integer == 0:
        return 0.0
    sign = -1.0 if integer < 0 else 1.0
    integer = abs(integer)
    bit_count = integer.bit_length()
    mantissa = integer / (1 << bit_count)
    try:
        return sign * math.ldexp(mantissa, exponent + bit_count)
    except OverflowError:
        return math.copysign(float("inf"), sign)


def _pam_requires_exact_objectives(D: np.ndarray, weights: np.ndarray) -> bool:
    positive_weights = weights[weights > 0.0]
    positive_distances = D[D > 0.0]
    if positive_weights.size == 0 or positive_distances.size == 0:
        return False
    n = max(1, int(weights.size))
    if (
        float(np.max(positive_weights)) > sys.float_info.max / n
        or float(np.min(positive_weights)) < sys.float_info.min
    ):
        return True
    positive_mask = (weights[:, None] > 0.0) & (D > 0.0)
    if not np.any(positive_mask):
        return False
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ordinary_products = weights[:, None] * D
    if np.any(
        positive_mask
        & (
            (ordinary_products == 0.0)
            | ~np.isfinite(ordinary_products)
            | (ordinary_products < sys.float_info.min)
        )
    ):
        return True
    _weight_mantissas, weight_exponents = np.frexp(weights)
    _distance_mantissas, distance_exponents = np.frexp(D)
    product_exponents = (weight_exponents[:, None] + distance_exponents)[positive_mask]
    if int(np.max(product_exponents)) >= math.frexp(sys.float_info.max / n)[1]:
        return True
    return bool(
        int(np.max(product_exponents)) - int(np.min(product_exponents)) >= 52 - (n - 1).bit_length()
    )


def _weighted_medoids_objective_component(
    D: np.ndarray,
    medoids: np.ndarray,
    weights: np.ndarray,
) -> _DyadicComponent:
    nearest = np.min(np.asarray(D, dtype=float)[:, np.asarray(medoids, dtype=int)], axis=1)
    return _sum_components(
        [
            _product_component(float(weight), float(distance))
            for weight, distance in zip(weights, nearest, strict=True)
        ]
    )


def _weighted_medoids_exact(
    D: np.ndarray,
    k: int,
    w: np.ndarray,
    *,
    max_iter: int,
    n_init: int,
    random_state: int,
    init_medoids: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """PAM swap loop with exact dyadic objective comparisons."""
    n = int(D.shape[0])
    rng = np.random.default_rng(int(random_state))

    def nearest_two(medoids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.empty(n, dtype=int)
        nearest = np.full(n, np.inf, dtype=float)
        second = np.full(n, np.inf, dtype=float)
        for observation in range(n):
            for slot, medoid in enumerate(medoids):
                distance = float(D[observation, medoid])
                if nearest[observation] > distance:
                    second[observation] = nearest[observation]
                    nearest[observation] = distance
                    labels[observation] = slot
                elif second[observation] > distance:
                    second[observation] = distance
        return labels, nearest, second

    best_objective: _DyadicComponent | None = None
    best_labels: np.ndarray | None = None
    best_medoids: np.ndarray | None = None
    for run in range(max(1, int(n_init))):
        if init_medoids is not None and run == 0:
            medoids = np.asarray(init_medoids, dtype=int).copy()
            if medoids.ndim != 1 or medoids.size != int(k):
                raise ValueError("init_medoids must be shape (k,)")
            if medoids.min() < 0 or medoids.max() >= n:
                raise ValueError("init_medoids out of range")
            if np.unique(medoids).size != medoids.size:
                raise ValueError("init_medoids must be unique")
        else:
            medoids = np.asarray(rng.choice(n, size=int(k), replace=False), dtype=int)

        for _iteration in range(max(1, int(max_iter))):
            labels, nearest, second = nearest_two(medoids)
            best_delta = (0, 0)
            best_slot = -1
            best_candidate = -1
            for slot in range(k):
                current = int(medoids[slot])
                fallback = np.where(labels == slot, second, nearest)
                for candidate in range(n):
                    if D[current, candidate] <= 0.0:
                        continue
                    delta_terms: list[_DyadicComponent] = []
                    for observation in range(n):
                        candidate_distance = float(D[candidate, observation])
                        fallback_distance = float(fallback[observation])
                        replacement_distance = min(candidate_distance, fallback_distance)
                        weight = float(w[observation])
                        delta_terms.append(_product_component(weight, replacement_distance))
                        delta_terms.append(
                            _product_component(
                                weight,
                                float(nearest[observation]),
                                sign=-1,
                            )
                        )
                    delta = _sum_components(delta_terms)
                    if _component_less(delta, best_delta):
                        best_delta = delta
                        best_slot = slot
                        best_candidate = candidate
            if best_slot < 0:
                break
            medoids[best_slot] = best_candidate

        labels, nearest, _second = nearest_two(medoids)
        run_objective = _weighted_medoids_objective_component(D, medoids, w)
        if best_objective is None or _component_less(run_objective, best_objective):
            best_objective = run_objective
            best_labels = labels.copy()
            best_medoids = medoids.copy()

    assert best_objective is not None and best_labels is not None and best_medoids is not None
    return best_labels, best_medoids, _component_to_float(best_objective)


def weighted_medoids(
    D: np.ndarray,
    k: int,
    weights: Optional[Sequence[float]] = None,
    max_iter: int = 100,
    n_init: int = 1,
    random_state: int = 0,
    init_medoids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pure-Python fallback weighted k-medoids (PAM-like swap).
    Returns (labels_0based, medoids_idx_0based, objective).
    """
    D = validate_kernel_distance(D)
    N = D.shape[0]
    if not (2 <= int(k) <= N):
        raise ValueError("k out of range")
    w = validate_kernel_weights(weights, expected_n=N)
    if _pam_requires_exact_objectives(D, w):
        return _weighted_medoids_exact(
            D,
            int(k),
            w,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            init_medoids=init_medoids,
        )
    rng = np.random.default_rng(int(random_state))

    max_distance = 1.1 * float(np.max(D[np.triu_indices(N, 1)])) + 1.0

    def nearest_two(meds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.empty(N, dtype=int)
        nearest = np.full(N, max_distance, dtype=float)
        second = np.full(N, max_distance, dtype=float)
        for observation in range(N):
            for slot, medoid in enumerate(meds):
                distance = float(D[observation, medoid])
                if nearest[observation] > distance:
                    second[observation] = nearest[observation]
                    nearest[observation] = distance
                    labels[observation] = slot
                elif second[observation] > distance:
                    second[observation] = distance
        return labels, nearest, second

    best_obj = np.inf
    best_labels = None
    best_meds = None
    for run in range(max(1, int(n_init))):
        if init_medoids is not None and run == 0:
            meds = np.asarray(init_medoids, dtype=int).copy()
            if meds.ndim != 1 or meds.size != int(k):
                raise ValueError("init_medoids must be shape (k,)")
            if meds.min() < 0 or meds.max() >= N:
                raise ValueError("init_medoids out of range")
            if np.unique(meds).size != meds.size:
                raise ValueError("init_medoids must be unique")
        else:
            meds = np.array(rng.choice(N, size=int(k), replace=False), dtype=int)
        obj = -1.0
        for _ in range(max(1, int(max_iter))):
            labels, nearest, second = nearest_two(meds)
            if obj < 0.0:
                obj = 0.0
                for observation in range(N):
                    obj += float(w[observation]) * float(nearest[observation])
            best_delta = 1.0
            best_slot = -1
            best_candidate = -1
            for mi in range(k):
                current = int(meds[mi])
                removal_cost = 0.0
                fallback = np.empty(N, dtype=float)
                for observation in range(N):
                    if labels[observation] == mi:
                        removal_cost += float(w[observation]) * float(
                            second[observation] - nearest[observation]
                        )
                        fallback[observation] = second[observation]
                    else:
                        fallback[observation] = nearest[observation]
                for h in range(N):
                    # WeightedCluster's PAMonce skips objects at zero distance
                    # from the medoid being replaced, including pseudometric
                    # duplicates. It does not use a generic is-medoid guard.
                    if D[current, h] <= 0.0:
                        continue
                    addition_gain = removal_cost
                    for observation in range(N):
                        candidate_distance = float(D[h, observation])
                        if candidate_distance < fallback[observation]:
                            addition_gain += float(w[observation]) * float(
                                candidate_distance - fallback[observation]
                            )
                    if best_delta > addition_gain:
                        best_delta = addition_gain
                        best_slot = mi
                        best_candidate = h
            if best_delta >= PAMONCE_SWAP_TOLERANCE:
                break
            meds[best_slot] = best_candidate
            obj += best_delta
        labels, _, _ = nearest_two(meds)
        if obj < best_obj:
            best_obj = obj
            best_meds = meds.copy()
            best_labels = labels.copy()
    return best_labels.astype(int), best_meds.astype(int), float(best_obj)


def _ordered_pair_quality_terms(
    D: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    pairwise_sum=_tx_weighted_pairwise_sum,
) -> dict[str, float]:
    """Port the ordered-pair calculations in WeightedCluster's clusterqualitybody.cpp."""
    ranking_D = np.asarray(D, dtype=float)
    active_indices = np.flatnonzero(np.asarray(weights, dtype=float) > 0.0)
    D = np.zeros_like(ranking_D)
    if active_indices.size:
        active_distances = ranking_D[np.ix_(active_indices, active_indices)]
        D[np.ix_(active_indices, active_indices)] = relative_finite_distance_matrix(
            active_distances
        )
    n = int(D.shape[0])
    k = int(labels.max()) + 1 if labels.size else 0
    sizes = np.bincount(labels, weights=weights, minlength=k).astype(float)
    positive_weights = np.where(weights > 0.0, weights, 0.0)
    errors = np.zeros(k, dtype=float)
    errors2 = np.zeros(k, dtype=float)
    buckets: dict[float, list[float]] = {0.0: [0.0, 0.0]}
    scaled_bucket_distances: dict[float, float] = {0.0: 0.0}
    total_pair_weight = wxy = wy = 0.0

    for i in range(n):
        wi = float(weights[i])
        if wi <= 0.0:
            continue
        diagonal_mass = wi * wi
        total_pair_weight += diagonal_mass
        wy += diagonal_mass
        buckets[0.0][0] += diagonal_mass
        ci = int(labels[i])
        for j in range(i + 1, n):
            wj = float(weights[j])
            if wj <= 0.0:
                continue
            mass = 2.0 * wi * wj
            distance = float(D[i, j])
            ranking_distance = float(ranking_D[i, j])
            weighted_distance = mass * distance
            total_pair_weight += mass
            bucket = buckets.setdefault(ranking_distance, [0.0, 0.0])
            scaled_bucket_distances.setdefault(ranking_distance, distance)
            if ci == int(labels[j]):
                wxy += weighted_distance
                wy += mass
                bucket[0] += mass
            else:
                bucket[1] += mass

    # Keep the migration boundary load-bearing: these residual sums are owned
    # by traminex.diss, while the surrounding quality formulas are
    # WeightedCluster's own ordered-pair definitions.
    wx = 2.0 * pairwise_sum(D, None, positive_weights)
    wx2 = 2.0 * pairwise_sum(D * D, None, positive_weights)
    for cluster in range(k):
        indices = np.where(labels == cluster)[0]
        errors[cluster] = 2.0 * pairwise_sum(D, indices, positive_weights)
        errors2[cluster] = 2.0 * pairwise_sum(D * D, indices, positive_weights)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_distance = float(np.divide(wx, total_pair_weight))
        mean_within = float(np.divide(wy, total_pair_weight))
        var_distance = float(np.divide(wx2, total_pair_weight) - mean_distance * mean_distance)
        var_within = float(np.divide(wy, total_pair_weight) - mean_within * mean_within)
        cov_distance_within = float(np.divide(wxy, total_pair_weight) - mean_distance * mean_within)
    pbc = (
        -cov_distance_within / np.sqrt(var_distance * var_within)
        if var_distance > 0.0 and var_within > 0.0
        else np.nan
    )

    total_within = wy
    total_between = max(0.0, total_pair_weight - wy)
    current_within = current_between = nc = nd = tied = 0.0
    for _distance, (within, between) in sorted(buckets.items()):
        tied += between * within
        nc += between * current_within
        nd += within * current_between
        current_within += within
        current_between += between
        nc += within * max(0.0, total_between - current_between)
        nd += between * max(0.0, total_within - current_within)
    hg_denominator = nc + nd
    hg = (nc - nd) / hg_denominator if hg_denominator > 0.0 else np.nan
    hgsd_denominator = hg_denominator + tied
    hgsd = (nc - nd) / hgsd_denominator if hgsd_denominator > 0.0 else np.nan

    smin = 0.0
    smax = 0.0
    current_mass = 0.0
    for ranking_distance, (within, between) in sorted(buckets.items()):
        distance = scaled_bucket_distances[ranking_distance]
        mass = within + between
        if mass <= 0.0:
            continue
        if current_mass <= total_within:
            if current_mass + mass > total_within:
                smin += (total_within - current_mass) * distance
            else:
                smin += mass * distance
        current_mass += mass
        if current_mass > total_between:
            if current_mass - mass < total_between:
                smax += (current_mass - total_between) * distance
            else:
                smax += mass * distance
    hc = (wxy - smin) / (smax - smin) if smax > smin else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        # WeightedCluster divides every cluster residual by its total case
        # weight. An exactly zero-mass cluster therefore contributes 0/0 and
        # makes all four residual-derived statistics undefined.
        observation_weight = float(sizes.sum())
        ss_res = float(np.sum(np.divide(errors, sizes)))
        ss2_res = float(np.sum(np.divide(errors2, sizes)))
        ss_expl = float(np.divide(wx, observation_weight) - ss_res)
        ss2_expl = float(np.divide(wx2, observation_weight) - ss2_res)
        ch = float(
            np.divide(
                np.divide(ss_expl, k - 1.0),
                np.divide(ss_res, observation_weight - k),
            )
        )
        chsq = float(
            np.divide(
                np.divide(ss2_expl, k - 1.0),
                np.divide(ss2_res, observation_weight - k),
            )
        )
        r2_denominator = ss_res + ss_expl
        r2sq_denominator = ss2_res + ss2_expl
        r2 = float(np.divide(ss_expl, r2_denominator))
        r2sq = float(np.divide(ss2_expl, r2sq_denominator))
    return {
        "PBC": float(pbc),
        "HG": float(hg),
        "HGSD": float(hgsd),
        "CH": float(ch),
        "R2": float(r2),
        "CHsq": float(chsq),
        "R2sq": float(r2sq),
        "HC": float(hc),
    }


def _exact_pair_rank_statistics(
    D: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Compute dimensionless quality statistics without range-losing products."""
    with localcontext() as context:
        context.prec = 2500
        zero = Decimal(0)
        cluster_count = int(labels.max()) + 1 if labels.size else 0
        sizes = [zero for _ in range(cluster_count)]
        errors = [zero for _ in range(cluster_count)]
        errors2 = [zero for _ in range(cluster_count)]
        total_pair_weight = zero
        wx = zero
        wx2 = zero
        wy = zero
        wxy = zero
        buckets: dict[float, list[Decimal]] = {0.0: [zero, zero]}
        for i in range(D.shape[0]):
            if weights[i] <= 0.0:
                continue
            wi = Decimal.from_float(float(weights[i]))
            sizes[int(labels[i])] += wi
            diagonal_mass = wi * wi
            total_pair_weight += diagonal_mass
            wy += diagonal_mass
            buckets[0.0][0] += diagonal_mass
            for j in range(i + 1, D.shape[0]):
                if weights[j] <= 0.0:
                    continue
                wj = Decimal.from_float(float(weights[j]))
                mass = Decimal(2) * wi * wj
                distance_value = float(D[i, j])
                distance = Decimal.from_float(distance_value)
                total_pair_weight += mass
                wx += mass * distance
                wx2 += mass * distance * distance
                bucket = buckets.setdefault(distance_value, [zero, zero])
                if labels[i] == labels[j]:
                    wy += mass
                    wxy += mass * distance
                    errors[int(labels[i])] += mass * distance
                    errors2[int(labels[i])] += mass * distance * distance
                    bucket[0] += mass
                else:
                    bucket[1] += mass

        mean_distance = wx / total_pair_weight
        mean_within = wy / total_pair_weight
        variance_distance = wx2 / total_pair_weight - mean_distance * mean_distance
        variance_within = wy / total_pair_weight - mean_within * mean_within
        covariance = wxy / total_pair_weight - mean_distance * mean_within
        pbc = (
            -(covariance / (variance_distance * variance_within).sqrt())
            if variance_distance > 0 and variance_within > 0
            else Decimal("NaN")
        )

        total_within = wy
        total_between = total_pair_weight - wy
        current_within = zero
        current_between = zero
        current_mass = zero
        concordant = zero
        discordant = zero
        tied = zero
        smin = zero
        smax = zero
        for distance_value, (within, between) in sorted(buckets.items()):
            distance = Decimal.from_float(float(distance_value))
            mass = within + between
            if current_mass <= total_within:
                if current_mass + mass > total_within:
                    smin += (total_within - current_mass) * distance
                else:
                    smin += mass * distance
            current_mass += mass
            if current_mass > total_between:
                if current_mass - mass < total_between:
                    smax += (current_mass - total_between) * distance
                else:
                    smax += mass * distance
            tied += between * within
            concordant += between * current_within
            discordant += within * current_between
            current_within += within
            current_between += between
            concordant += within * (total_between - current_between)
            discordant += between * (total_within - current_within)
        denominator = concordant + discordant
        hg = (concordant - discordant) / denominator if denominator > 0 else Decimal("NaN")
        denominator_ties = denominator + tied
        hgsd = (
            (concordant - discordant) / denominator_ties if denominator_ties > 0 else Decimal("NaN")
        )
        hc = (wxy - smin) / (smax - smin) if smax > smin else Decimal("NaN")

        observation_weight = sum(sizes, zero)
        if observation_weight > 0 and all(size > 0 for size in sizes):
            ss_res = sum(
                (errors[cluster] / sizes[cluster] for cluster in range(cluster_count)),
                zero,
            )
            ss2_res = sum(
                (errors2[cluster] / sizes[cluster] for cluster in range(cluster_count)),
                zero,
            )
            ss_expl = wx / observation_weight - ss_res
            ss2_expl = wx2 / observation_weight - ss2_res
            # R's kernel forms these ratios in double arithmetic, so a zero
            # denominator yields +/-Inf (or NaN for 0/0) rather than being
            # guarded away: perfect clusters give CH = Inf in R and on the
            # ordinary float paths. The former ``ss_res != 0`` guard returned
            # NaN here instead (2026-09-06 finding WC-090306).
            r2 = _ieee_ratio(ss_expl, ss_res + ss_expl)
            r2sq = _ieee_ratio(ss2_expl, ss2_res + ss2_expl)
            residual_dof = observation_weight - Decimal(cluster_count)
            cluster_dof = Decimal(cluster_count - 1)
            ch = _ieee_ratio(
                _ieee_decimal_ratio(ss_expl, cluster_dof),
                _ieee_decimal_ratio(ss_res, residual_dof),
            )
            chsq = _ieee_ratio(
                _ieee_decimal_ratio(ss2_expl, cluster_dof),
                _ieee_decimal_ratio(ss2_res, residual_dof),
            )
        else:
            r2 = float("nan")
            r2sq = float("nan")
            ch = float("nan")
            chsq = float("nan")
        return {
            "PBC": float(pbc),
            "HG": float(hg),
            "HGSD": float(hgsd),
            "HC": float(hc),
            "R2": float(r2),
            "R2sq": float(r2sq),
            "CH": float(ch),
            "CHsq": float(chsq),
        }


def _ieee_decimal_ratio(numerator: Decimal, denominator: Decimal) -> Decimal | float:
    """Divide two exact quantities, taking the C kernel's semantics at zero.

    The quotient stays exact (a ``Decimal``) whenever the denominator is not
    zero; ``x / 0`` becomes ``+/-Inf`` and ``0 / 0`` becomes NaN, as binary64
    division would produce them in WeightedCluster's kernel.
    """
    if denominator == 0:
        if numerator == 0:
            return float("nan")
        return math.copysign(float("inf"), float(numerator))
    return numerator / denominator


def _ieee_ratio(numerator: Decimal | float, denominator: Decimal | float) -> float:
    """Final ratio of two exact-or-IEEE operands with binary64 division semantics."""
    if isinstance(numerator, Decimal) and isinstance(denominator, Decimal):
        return float(_ieee_decimal_ratio(numerator, denominator))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(float(numerator), float(denominator)))


def _compute_hg_hgsd_weighted(
    D: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    terms = _ordered_pair_quality_terms(
        np.asarray(D, dtype=float),
        np.asarray(labels, dtype=int),
        np.maximum(np.asarray(weights, dtype=float), 0.0),
    )
    return terms["HG"], terms["HGSD"]


def _cluster_quality_from_arrays(
    matrix: np.ndarray,
    labels0: np.ndarray,
    w: np.ndarray,
    *,
    pairwise_sum,
) -> tuple[dict[str, float], dict[int, float], dict[int, float]]:
    with np.errstate(all="ignore"):
        stats = _ordered_pair_quality_terms(matrix, labels0, w, pairwise_sum=pairwise_sum)
    positive = w > 0.0
    if np.any(positive):
        pair_range_loss = finite_pair_masses_need_scaling(w)
        active_indices = np.flatnonzero(positive)
        active_distances = matrix[np.ix_(active_indices, active_indices)]
        distance_range_loss = finite_distance_squares_need_scaling(active_distances)
        if pair_range_loss or distance_range_loss:
            stats.update(_exact_pair_rank_statistics(matrix, labels0, w))
    asw, asww, asw_by_cluster, asww_by_cluster = cluster_asw_core_compatible(matrix, labels0, w)
    stats["ASW"] = float(asw)
    stats["ASWw"] = float(asww)
    return stats, asw_by_cluster, asww_by_cluster


def cluster_quality(
    D: np.ndarray,
    labels: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    *,
    pairwise_sum=_tx_weighted_pairwise_sum,
) -> tuple[dict[str, float], dict[int, float], dict[int, float]]:
    """Validated direct ten-statistic kernel matching WeightedCluster's core."""
    matrix = validate_kernel_distance(D)
    labels0 = validate_kernel_labels(labels, expected_n=matrix.shape[0])
    w = validate_kernel_weights(weights, expected_n=matrix.shape[0])
    return _cluster_quality_from_arrays(
        matrix,
        labels0,
        w,
        pairwise_sum=pairwise_sum,
    )


def cluster_quality_r_permissive(
    D: np.ndarray,
    labels: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    *,
    pairwise_sum=_tx_weighted_pairwise_sum,
) -> tuple[dict[str, float], dict[int, float], dict[int, float]]:
    """R-permissive internal path after public shape/label validation."""
    ranking_matrix = np.asarray(D, dtype=float)
    labels0 = np.asarray(labels, dtype=int)
    w = (
        np.ones(ranking_matrix.shape[0], dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    # Ordered-pair moments scale their active distances internally, while rank
    # buckets must retain the caller's exact finite keys. Scaling here can map
    # the minimum positive binary64 value to zero when a far distance is also
    # present, corrupting HG and HGSD before the ordered-pair pass begins. The
    # float pass runs R's silent arithmetic, as in the validated kernel; on
    # the range-risk inputs routed below, its overflowed ratios are replaced.
    with np.errstate(all="ignore"):
        stats = _ordered_pair_quality_terms(ranking_matrix, labels0, w, pairwise_sum=pairwise_sum)
    k = int(labels0.max()) + 1 if labels0.size else 0
    # On the validated domain (finite non-negative distances and weights) this
    # kernel takes the same exact range-risk route as the compiled, Numba, and
    # validated kernels, so the publicly selected pure tier and the bootstrap
    # replicate path return the deliberately corrected extreme-range results
    # rather than R's range-lost float values (2026-09-06 finding WC-090304,
    # the demonstrated 2026-09-02 Suspected lead (a)). Ordinary inputs never
    # enter it, so the R-mirroring float arithmetic below stays bit-identical
    # there, and the R-permissive domain (negative or non-finite weights,
    # non-finite distances) keeps R's arithmetic throughout.
    validated_domain = bool(
        np.all(np.isfinite(ranking_matrix))
        and np.all(ranking_matrix >= 0.0)
        and np.all(np.isfinite(w))
        and np.all(w >= 0.0)
    )
    if validated_domain and np.any(w > 0.0):
        active_indices = np.flatnonzero(w > 0.0)
        active_distances = ranking_matrix[np.ix_(active_indices, active_indices)]
        if finite_pair_masses_need_scaling(w) or finite_distance_squares_need_scaling(
            active_distances
        ):
            stats.update(_exact_pair_rank_statistics(ranking_matrix, labels0, w))
        from .quality import (
            _cluster_mass_expansions,
            _silhouette_needs_stable_accumulation,
            _stable_observation_silhouettes,
        )

        if _silhouette_needs_stable_accumulation(ranking_matrix, w):
            size_expansions, size_components = _cluster_mass_expansions(labels0, w, k)
            asw_i, asw_w = _stable_observation_silhouettes(
                ranking_matrix, labels0, w, k, size_expansions, size_components
            )
            stats_asw, stats_asww, asw, asww = aggregate_observation_silhouettes(
                asw_i, asw_w, labels0, w, k
            )
            stats["ASW"] = float(stats_asw)
            stats["ASWw"] = float(stats_asww)
            return stats, asw, asww
    # The silhouette loop below reads a distance only where the outer row has
    # positive weight and the inner column has nonzero weight, so the scale
    # decision must cover exactly that block. Deciding and applying the scale
    # on the whole matrix let a zero-weight row's far distance underflow every
    # active contribution to zero, turning ASW/ASWw into NaN where the
    # compiled, Numba, and validated pure kernels return the R value —
    # 2026-09-03 audit, finding WC-090301, the same scaling-domain family as
    # WC-090202. Cells outside the contributing block are zeroed when scaling
    # fires; they only ever meet a zero weight factor.
    #
    # A non-finite value anywhere disables scaling deliberately: after the
    # public as.dist coercion every non-finite cell is symmetric, so the loop
    # reads it (possibly as 0 * NaN = NaN) and R's float kernel contaminates
    # the aggregates the same way — the raw matrix is the R-faithful input.
    # For a fully finite matrix, a contributing block can need scaling only if
    # the whole matrix does (its positives are a subset and the size-dependent
    # thresholds only relax), so the whole-matrix check is a cheap exact
    # pre-filter that skips the block gather on ordinary inputs.
    matrix = ranking_matrix
    silhouette_rows = np.flatnonzero(w > 0.0)
    silhouette_columns = np.flatnonzero(w != 0.0)
    if (
        silhouette_rows.size
        and bool(np.all(np.isfinite(ranking_matrix)))
        and finite_distance_squares_need_scaling(ranking_matrix)
    ):
        contributing = ranking_matrix[np.ix_(silhouette_rows, silhouette_columns)]
        if finite_distance_squares_need_scaling(contributing):
            matrix = np.zeros_like(ranking_matrix)
            matrix[np.ix_(silhouette_rows, silhouette_columns)] = contributing / float(
                np.max(contributing)
            )
    sizes = np.bincount(labels0, weights=w, minlength=k).astype(float)
    asw_cluster_num = np.zeros(k, dtype=float)
    asww_cluster_num = np.zeros(k, dtype=float)
    with np.errstate(all="ignore"):
        for i in range(matrix.shape[0]):
            wi = float(w[i])
            if wi <= 0.0:
                continue
            cluster = int(labels0[i])
            within = 0.0
            other = np.zeros(k, dtype=float)
            for j in range(matrix.shape[0]):
                if i == j:
                    continue
                contribution = float(w[j]) * float(matrix[i, j])
                if int(labels0[j]) == cluster:
                    within += contribution
                else:
                    other[int(labels0[j])] += contribution

            between = sys.float_info.max
            for candidate_cluster in range(k):
                if candidate_cluster == cluster:
                    continue
                candidate = other[candidate_cluster] / sizes[candidate_cluster]
                if between >= candidate:
                    between = float(candidate)

            within_weighted = within / sizes[cluster]
            within_unweighted = 0.0 if sizes[cluster] <= 1.0 else within / (sizes[cluster] - 1.0)
            silhouette = wi * ((between - within_unweighted) / np.fmax(within_unweighted, between))
            silhouette_weighted = wi * (
                (between - within_weighted) / np.fmax(within_weighted, between)
            )
            asw_cluster_num[cluster] += silhouette
            asww_cluster_num[cluster] += silhouette_weighted

        total_weight = float(sizes.sum())
        stats["ASW"] = float(asw_cluster_num.sum() / total_weight)
        stats["ASWw"] = float(asww_cluster_num.sum() / total_weight)
        asw = {
            cluster + 1: float(asw_cluster_num[cluster] / sizes[cluster]) for cluster in range(k)
        }
        asww = {
            cluster + 1: float(asww_cluster_num[cluster] / sizes[cluster]) for cluster in range(k)
        }
    return stats, asw, asww

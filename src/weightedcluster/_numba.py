from __future__ import annotations

from numbers import Integral

import numpy as np

from ._quality_contract import (
    cluster_asw_core_compatible,
    finite_distance_squares_need_scaling,
    finite_pair_masses_need_scaling,
    validate_kernel_distance,
    validate_kernel_labels,
    validate_kernel_weights,
)

try:
    from numba import njit, prange
except ImportError as e:  # pragma: no cover
    raise ImportError("Numba is required for the numba fallback module") from e


@njit(cache=True)
def _cluster_quality_numba(  # pragma: no cover - JIT body, untraceable by coverage
    D: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple:
    N = D.shape[0]
    k = 0
    for i in range(N):
        if labels[i] + 1 > k:
            k = labels[i] + 1
    # Ordered-pair terms from WeightedCluster/src/clusterqualitybody.cpp.
    sizes = np.zeros(k, np.float64)
    errors = np.zeros(k, np.float64)
    errors2 = np.zeros(k, np.float64)
    total_pair_weight = 0.0
    wx = 0.0
    wx2 = 0.0
    wy = 0.0
    wxy = 0.0
    max_events = N * (N + 1) // 2
    distances = np.empty(max_events, np.float64)
    pair_mass = np.empty(max_events, np.float64)
    pair_within = np.empty(max_events, np.uint8)
    pair_count = 0
    for i in range(N):
        wi = weights[i]
        sizes[labels[i]] += wi
        if wi <= 0.0:
            continue
        diagonal_mass = wi * wi
        distances[pair_count] = 0.0
        pair_mass[pair_count] = diagonal_mass
        pair_within[pair_count] = 1
        pair_count += 1
        total_pair_weight += diagonal_mass
        wy += diagonal_mass
        for j in range(i + 1, N):
            wj = weights[j]
            if wj <= 0.0:
                continue
            mass = 2.0 * wi * wj
            distance = D[i, j]
            weighted_distance = mass * distance
            total_pair_weight += mass
            wx += weighted_distance
            wx2 += weighted_distance * distance
            distances[pair_count] = distance
            pair_mass[pair_count] = mass
            if labels[i] == labels[j]:
                pair_within[pair_count] = 1
                errors[labels[i]] += weighted_distance
                errors2[labels[i]] += weighted_distance * distance
                wy += mass
                wxy += weighted_distance
            else:
                pair_within[pair_count] = 0
            pair_count += 1

    if total_pair_weight > 0.0:
        mean_dist = wx / total_pair_weight
        mean_within = wy / total_pair_weight
        var_dist = (wx2 / total_pair_weight) - (mean_dist * mean_dist)
        var_within = (wy / total_pair_weight) - (mean_within * mean_within)
        cov_dist_within = (wxy / total_pair_weight) - (mean_dist * mean_within)
        if var_dist > 0.0 and var_within > 0.0:
            PBC = -(cov_dist_within / np.sqrt(var_dist * var_within))
        else:
            PBC = np.nan
    else:
        PBC = np.nan

    # Stable ordering is material for equal-distance buckets: the canonical
    # C++ std::map path accumulates pair mass in the original i/j loop order.
    order = np.argsort(distances[:pair_count], kind="mergesort")
    total_within = wy
    total_between = max(0.0, total_pair_weight - wy)
    current_within = 0.0
    current_between = 0.0
    current_mass = 0.0
    nc = 0.0
    nd = 0.0
    tied = 0.0
    smin = 0.0
    smax = 0.0
    pos = 0
    while pos < pair_count:
        distance = distances[order[pos]]
        within = 0.0
        between = 0.0
        while pos < pair_count and distances[order[pos]] == distance:
            original = order[pos]
            if pair_within[original] == 1:
                within += pair_mass[original]
            else:
                between += pair_mass[original]
            pos += 1
        bucket_mass = within + between
        if current_mass <= total_within:
            if current_mass + bucket_mass > total_within:
                smin += (total_within - current_mass) * distance
            else:
                smin += bucket_mass * distance
        current_mass += bucket_mass
        if current_mass > total_between:
            if current_mass - bucket_mass < total_between:
                smax += (current_mass - total_between) * distance
            else:
                smax += bucket_mass * distance

        tied += between * within
        nc += between * current_within
        nd += within * current_between
        current_within += within
        current_between += between
        nc += within * max(0.0, total_between - current_between)
        nd += between * max(0.0, total_within - current_within)

    hg_denominator = nc + nd
    HG = (nc - nd) / hg_denominator if hg_denominator > 0.0 else np.nan
    hgsd_denominator = hg_denominator + tied
    HGSD = (nc - nd) / hgsd_denominator if hgsd_denominator > 0.0 else np.nan
    HC = (wxy - smin) / (smax - smin) if smax > smin else np.nan

    observation_weight = 0.0
    ss_res = 0.0
    ss2_res = 0.0
    for c in range(k):
        observation_weight += sizes[c]
        ss_res += np.divide(errors[c], sizes[c])
        ss2_res += np.divide(errors2[c], sizes[c])
    ss_expl = np.divide(wx, observation_weight) - ss_res
    ss2_expl = np.divide(wx2, observation_weight) - ss2_res
    CH = np.divide(np.divide(ss_expl, k - 1.0), np.divide(ss_res, observation_weight - k))
    CHsq = np.divide(
        np.divide(ss2_expl, k - 1.0),
        np.divide(ss2_res, observation_weight - k),
    )
    R2 = np.divide(ss_expl, ss_res + ss_expl)
    R2sq = np.divide(ss2_expl, ss2_res + ss2_expl)

    return PBC, HG, HGSD, CH, R2, CHsq, R2sq, HC


def cluster_quality_numba(D: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None):
    original_D = validate_kernel_distance(D).astype(np.float64, copy=False)
    labels = validate_kernel_labels(labels, expected_n=original_D.shape[0]).astype(
        np.int32, copy=False
    )
    w = validate_kernel_weights(weights, expected_n=original_D.shape[0]).astype(
        np.float64, copy=False
    )
    positive = w > 0.0
    if np.any(positive):
        pair_range_loss = finite_pair_masses_need_scaling(w)
    else:
        pair_range_loss = False
    active = w > 0.0
    active_pairs = active[:, None] & active[None, :]
    active_distances = original_D[active_pairs]
    distance_range_loss = finite_distance_squares_need_scaling(active_distances)
    if pair_range_loss or distance_range_loss:
        from ._fallback import cluster_quality as stable_cluster_quality

        return stable_cluster_quality(original_D, labels, w)
    # The escape hatch above guarantees the positive-weight submatrix carries
    # no squared-reduction range risk, and the kernel never reads a distance
    # touching a non-positive weight. Rescaling by the *whole* matrix maximum
    # here let a zero-weight row's far distance underflow the active rank keys
    # to a shared bucket, corrupting HGSD/PBC/CH/R2 relative to the compiled
    # and pure kernels (2026-09-02 audit, finding WC-090202). The raw matrix
    # is the correct kernel input.
    PBC, HG, HGSD, CH, R2, CHsq, R2sq, HC = _cluster_quality_numba(original_D, labels, w)
    stats_asw, stats_asww, ASW, ASWw = cluster_asw_core_compatible(original_D, labels, w)
    stats = {
        "PBC": float(PBC),
        "HG": float(HG),
        "HGSD": float(HGSD),
        "ASW": float(stats_asw),
        "ASWw": float(stats_asww),
        "CH": float(CH),
        "R2": float(R2),
        "CHsq": float(CHsq),
        "R2sq": float(R2sq),
        "HC": float(HC),
    }
    return stats, ASW, ASWw


# ---------------------
# seqnull generators (selected models)
# ---------------------
@njit(cache=True)
def _discrete_pick(prob: np.ndarray, u: float) -> int:  # pragma: no cover - JIT body
    c = 0.0
    for i in range(prob.size):
        c += prob[i]
        if u <= c:
            return i
    return prob.size - 1


@njit(parallel=True, cache=True)
def _seqnull_random_from_probs_numba(  # pragma: no cover - JIT body, untraceable
    freqs_t: np.ndarray, p_glob: np.ndarray, time_varying: bool, N: int, T: int, seed: int
) -> np.ndarray:
    Y = np.empty((N, T), np.int32)
    if time_varying:
        for i in prange(N):
            np.random.seed(seed + i)
            for t in range(T):
                u = np.random.random()
                Y[i, t] = _discrete_pick(freqs_t[t], u)
    else:
        for i in prange(N):
            np.random.seed(seed + i)
            for t in range(T):
                u = np.random.random()
                Y[i, t] = _discrete_pick(p_glob, u)
    return Y


@njit(parallel=True, cache=True)
def _seqnull_markov_numba(  # pragma: no cover - JIT body, untraceable
    trans: np.ndarray, p0: np.ndarray, N: int, T: int, seed: int
) -> np.ndarray:
    Y = np.empty((N, T), np.int32)
    for i in prange(N):
        np.random.seed(seed + i)
        u0 = np.random.random()
        Y[i, 0] = _discrete_pick(p0, u0)
        for t in range(1, T):
            s = Y[i, t - 1]
            u = np.random.random()
            Y[i, t] = _discrete_pick(trans[s], u)
    return Y


def _validate_numba_dimension(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer")
    return resolved


def _validate_numba_seed(value: int, *, n_rows: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("seed must be a non-negative integer")
    seed = int(value)
    max_seed = int(np.iinfo(np.uint32).max) - (n_rows - 1)
    if seed < 0 or seed > max_seed:
        raise ValueError(f"seed must be between 0 and {max_seed} for N={n_rows}")
    return seed


def _validate_probability_rows(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric probabilities") from exc
    if probabilities.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any(probabilities < 0.0):
        raise ValueError(f"{name} probabilities must be non-negative")
    row_sums = probabilities.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{name} probability rows must sum to one")
    return probabilities


def seqnull_random_from_probs_numba(
    freqs_t: np.ndarray,
    p_glob: np.ndarray,
    time_varying: bool,
    N: int,
    T: int,
    seed: int,
) -> np.ndarray:
    """Validated direct wrapper for the Numba independent-state generator."""
    n_rows = _validate_numba_dimension(N, name="N")
    n_time = _validate_numba_dimension(T, name="T")
    global_probs = np.asarray(p_glob)
    if global_probs.ndim != 1 or global_probs.size == 0:
        raise ValueError("p_glob must be a non-empty probability vector")
    n_states = int(global_probs.size)
    validated_global = _validate_probability_rows(
        global_probs,
        name="p_glob",
        expected_shape=(n_states,),
    )
    validated_time = _validate_probability_rows(
        freqs_t,
        name="freqs_t",
        expected_shape=(n_time, n_states),
    )
    validated_seed = _validate_numba_seed(seed, n_rows=n_rows)
    return _seqnull_random_from_probs_numba(
        validated_time,
        validated_global,
        bool(time_varying),
        n_rows,
        n_time,
        validated_seed,
    )


def seqnull_markov_numba(
    trans: np.ndarray,
    p0: np.ndarray,
    N: int,
    T: int,
    seed: int,
) -> np.ndarray:
    """Validated direct wrapper for the Numba Markov-state generator."""
    n_rows = _validate_numba_dimension(N, name="N")
    n_time = _validate_numba_dimension(T, name="T")
    initial = np.asarray(p0)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("p0 must be a non-empty probability vector")
    n_states = int(initial.size)
    validated_initial = _validate_probability_rows(
        initial,
        name="p0",
        expected_shape=(n_states,),
    )
    validated_transition = _validate_probability_rows(
        trans,
        name="trans",
        expected_shape=(n_states, n_states),
    )
    validated_seed = _validate_numba_seed(seed, n_rows=n_rows)
    return _seqnull_markov_numba(
        validated_transition,
        validated_initial,
        n_rows,
        n_time,
        validated_seed,
    )

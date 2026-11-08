from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:
    from ._wc_core import weighted_medoids as _weighted_medoids
except ImportError:
    try:
        from ._fallback import weighted_medoids as _weighted_medoids
    except ImportError:
        warnings.warn(
            "Neither compiled core nor fallback available for weighted_medoids. "
            "This should not happen in a properly installed package.",
            UserWarning,
            stacklevel=2,
        )
        raise

# The exact-dyadic objective helpers are owned by weightedcluster._fallback;
# kmedoids used to carry a verbatim copy, which drifted (the copy lost the
# empty-active-mask guard) and crashed wcKMedoids on valid degenerate input
# (2026-09-02 audit, finding WC-090203). One implementation, one owner.
from ._constants import (
    DEFAULT_KMEDOIDS_MAX_ITER,
    EXTERNAL_LABEL_BASE,
    MIN_CLUSTERS,
    PAMONCE_SWAP_TOLERANCE,
)
from ._fallback import (
    _component_less,
    _component_to_float,
    _pam_requires_exact_objectives,
    _product_component,
    _sum_components,
    _weighted_medoids_objective_component,
)
from ._utils import find_medoid, generate_random_seed
from ._validation import (
    coerce_public_distance_matrix,
    cut_linkage_by_merge_count,
    require_finite_non_negative_distances,
    validate_integer_control,
    validate_linkage_matrix,
)
from .quality import (
    _factorize_quality_labels,
    _wc_cluster_quality_internal,
    wcClusterQuality,
)
from .schemas import ClusteringTable, KMedoidsResult, KMedRangeResult, rows_to_stats_table

_KMEDOIDS_METHODS = ("kmedoids", "pam", "pamonce")
#: Backwards-compatible alias; the value is owned by ``_constants``.
_PAMONCE_SWAP_TOLERANCE = PAMONCE_SWAP_TOLERANCE
_R_KMEDOIDS_INFO_NAMES = ("Sum of Distances", "N. Found", "Method")
_R_QUALITY_NAMES = (
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


def _coerce_kmedoids_distance(diss: np.ndarray) -> np.ndarray:
    """Apply the shared public distance contract, then the one narrowing.

    Symmetry and the diagonal follow R's ``as.dist`` reading: only the lower
    triangle is data (findings F-06, F-07). R's *bare matrix* path differs on a
    non-symmetric input; ``docs/R_MIGRATION_GUIDE.md`` records why Python does
    not follow it. The value domain does not: the compiled medoid
    search cannot run on negative or non-finite dissimilarities, where R's PAM
    still returns a partition. That refusal is the single documented narrowing
    of the public contract; see `docs/R_MIGRATION_GUIDE.md`.
    """
    D = coerce_public_distance_matrix(diss, name="diss")
    require_finite_non_negative_distances(D, name="diss")
    return D


def _normalize_kmedoids_method(method: str | int) -> int:
    """Translate R's exact case-insensitive partial matching to method code 1:3."""
    if isinstance(method, str):
        lowered = method.lower()
        if lowered in _KMEDOIDS_METHODS:
            return _KMEDOIDS_METHODS.index(lowered) + 1
        matches = [
            index + 1 for index, name in enumerate(_KMEDOIDS_METHODS) if name.startswith(lowered)
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"unknown clustering method {method}")
    if isinstance(method, (int, np.integer)) and not isinstance(method, (bool, np.bool_)):
        method_code = int(method)
        if method_code in (1, 2, 3):
            return method_code
    raise ValueError(f"unknown clustering method {method}")


def _coerce_kmedoids_weights(weights: Optional[Sequence[float]], *, expected_n: int) -> np.ndarray:
    if weights is None:
        return np.ones(expected_n, dtype=float)
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape[0] != expected_n:
        raise ValueError(f"weights should be a vector of length {expected_n}")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must contain finite values")
    return values


def _assign_to_medoids(
    D: np.ndarray,
    medoids: np.ndarray,
    weights: np.ndarray,
    *,
    medoids_own_cluster: bool = False,
) -> tuple[np.ndarray, float]:
    """Assign with R's scalar first-minimum order.

    Iterative ``KMedoids`` additionally stops at an observation's own medoid,
    which keeps every valid medoid slot nonempty on pseudometric duplicates.
    PAM/PAMonce deliberately do not use that special case.
    """
    labels = np.empty(D.shape[0], dtype=int)
    objective = 0.0
    for observation in range(D.shape[0]):
        nearest = float("inf")
        assigned = 0
        for slot, medoid in enumerate(medoids):
            if medoids_own_cluster and observation == medoid:
                nearest = 0.0
                assigned = slot
                break
            distance = float(D[observation, medoid])
            if distance < nearest:
                nearest = distance
                assigned = slot
        labels[observation] = assigned
        objective += float(weights[observation]) * nearest
    return labels, objective


def _nearest_two_medoids(
    D: np.ndarray, medoids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R PAM/PAMonce's first/second distances and first-min labels."""
    n = D.shape[0]
    max_distance = 1.1 * float(np.max(D[np.triu_indices(n, 1)])) + 1.0
    labels = np.empty(n, dtype=int)
    nearest = np.full(n, max_distance, dtype=float)
    second = np.full(n, max_distance, dtype=float)
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


def _pamonce_from_start(
    D: np.ndarray, medoids: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the R 2.0 PAMonce global-swap loop from explicit zero-based medoids."""
    selected = np.asarray(medoids, dtype=int).copy()
    objective = -1.0
    while True:
        labels, nearest, second = _nearest_two_medoids(D, selected)
        if objective < 0.0:
            objective = 0.0
            for observation in range(D.shape[0]):
                objective += float(weights[observation]) * float(nearest[observation])
        best_delta = 1.0
        best_slot = -1
        best_candidate = -1
        for slot, current in enumerate(selected):
            removal_cost = 0.0
            fallback = np.empty(D.shape[0], dtype=float)
            for observation in range(D.shape[0]):
                if labels[observation] == slot:
                    removal_cost += float(weights[observation]) * float(
                        second[observation] - nearest[observation]
                    )
                    fallback[observation] = second[observation]
                else:
                    fallback[observation] = nearest[observation]
            for candidate in range(D.shape[0]):
                if D[current, candidate] <= 0.0:
                    continue
                addition_gain = removal_cost
                for observation in range(D.shape[0]):
                    candidate_distance = float(D[candidate, observation])
                    if candidate_distance < fallback[observation]:
                        addition_gain += float(weights[observation]) * float(
                            candidate_distance - fallback[observation]
                        )
                if best_delta > addition_gain:
                    best_delta = addition_gain
                    best_slot = slot
                    best_candidate = candidate
        if best_delta >= _PAMONCE_SWAP_TOLERANCE:
            return labels, selected, float(objective)
        selected[best_slot] = best_candidate
        objective += best_delta


def _pam_from_start(
    D: np.ndarray, medoids: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run WeightedCluster 2.0's square-matrix PAM loop from fixed medoids."""
    selected = np.asarray(medoids, dtype=int).copy()
    objective = -1.0
    while True:
        labels, nearest, second = _nearest_two_medoids(D, selected)
        if objective < 0.0:
            objective = 0.0
            for observation in range(D.shape[0]):
                objective += float(weights[observation]) * float(nearest[observation])
        best_delta = 1.0
        best_slot = -1
        best_candidate = -1
        for candidate in range(D.shape[0]):
            if any(D[candidate, medoid] == 0.0 for medoid in selected):
                continue
            for slot, current in enumerate(selected):
                delta = 0.0
                for observation in range(D.shape[0]):
                    candidate_distance = float(D[candidate, observation])
                    if D[current, observation] == nearest[observation]:
                        replacement = min(second[observation], candidate_distance)
                        delta += float(weights[observation]) * float(
                            replacement - nearest[observation]
                        )
                    elif candidate_distance < nearest[observation]:
                        delta += float(weights[observation]) * float(
                            candidate_distance - nearest[observation]
                        )
                if best_delta > delta:
                    best_delta = delta
                    best_slot = slot
                    best_candidate = candidate
        if best_delta >= 0.0:
            return labels, selected, float(objective)
        selected[best_slot] = best_candidate
        objective += best_delta


def _kmedoids_from_start(
    D: np.ndarray, medoids: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run R's iterative KMedoids update rule from explicit zero-based medoids."""
    selected = np.asarray(medoids, dtype=int).copy()
    previous = float("inf")
    labels = np.zeros(D.shape[0], dtype=int)
    for iteration in range(DEFAULT_KMEDOIDS_MAX_ITER):
        if iteration:
            for cluster in range(selected.size):
                members = np.where(labels == cluster)[0]
                if members.size:
                    selected[cluster] = find_medoid(D, members, weights=weights)
        labels, objective = _assign_to_medoids(
            D,
            selected,
            weights,
            medoids_own_cluster=True,
        )
        if objective >= previous:
            return labels, selected, float(previous)
        previous = objective
    return labels, selected, float(previous)


def _exact_medoid_for_members(
    D: np.ndarray,
    members: np.ndarray,
    weights: np.ndarray,
) -> int:
    best = int(members[0])
    best_cost: tuple[int, int] | None = None
    for candidate in members:
        cost = _sum_components(
            [
                _product_component(float(weights[observation]), float(D[candidate, observation]))
                for observation in members
            ]
        )
        if best_cost is None or _component_less(cost, best_cost):
            best = int(candidate)
            best_cost = cost
    return best


def _kmedoids_from_start_exact(
    D: np.ndarray,
    medoids: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the iterative KMedoids rule with exact objective comparisons."""
    selected = np.asarray(medoids, dtype=int).copy()
    previous: tuple[int, int] | None = None
    labels = np.zeros(D.shape[0], dtype=int)
    for iteration in range(DEFAULT_KMEDOIDS_MAX_ITER):
        if iteration:
            for cluster in range(selected.size):
                members = np.where(labels == cluster)[0]
                if members.size:
                    selected[cluster] = _exact_medoid_for_members(D, members, weights)
        labels, _objective = _assign_to_medoids(
            D,
            selected,
            weights,
            medoids_own_cluster=True,
        )
        objective = _weighted_medoids_objective_component(D, selected, weights)
        if previous is not None and not _component_less(objective, previous):
            return labels, selected, _component_to_float(previous)
        previous = objective
    assert previous is not None
    return labels, selected, _component_to_float(previous)


def _pam_from_start_exact(
    D: np.ndarray,
    medoids: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the PAM loop while retaining exact signed swap deltas."""
    selected = np.asarray(medoids, dtype=int).copy()
    while True:
        labels, nearest, second = _nearest_two_medoids(D, selected)
        best_delta = (0, 0)
        best_slot = -1
        best_candidate = -1
        for candidate in range(D.shape[0]):
            if any(D[candidate, medoid] == 0.0 for medoid in selected):
                continue
            for slot, current in enumerate(selected):
                terms: list[tuple[int, int]] = []
                for observation in range(D.shape[0]):
                    candidate_distance = float(D[candidate, observation])
                    if D[current, observation] == nearest[observation]:
                        replacement = min(second[observation], candidate_distance)
                    else:
                        replacement = min(nearest[observation], candidate_distance)
                    weight = float(weights[observation])
                    terms.append(_product_component(weight, replacement))
                    terms.append(_product_component(weight, float(nearest[observation]), sign=-1))
                delta = _sum_components(terms)
                if _component_less(delta, best_delta):
                    best_delta = delta
                    best_slot = slot
                    best_candidate = candidate
        if best_slot < 0:
            objective = _weighted_medoids_objective_component(D, selected, weights)
            return labels, selected, _component_to_float(objective)
        selected[best_slot] = best_candidate


def _pam_build_initial_medoids(D: np.ndarray, k: int, weights: np.ndarray) -> np.ndarray:
    """Port the tie-sensitive BUILD initialization in R's kmedoidbasebody.cpp."""
    n = D.shape[0]
    exact_objectives = _pam_requires_exact_objectives(D, weights)
    observed_maximum = float(np.max(D[np.triu_indices(n, 1)]))
    max_distance = observed_maximum if exact_objectives else 1.1 * observed_maximum + 1.0
    nearest = np.full(n, max_distance, dtype=float)
    selected = np.zeros(n, dtype=bool)
    medoids: list[int] = []
    for _ in range(k):
        best_gain = 0.0
        best_gain_component = (0, 0)
        best_index = -1
        for candidate in range(n):
            if selected[candidate]:
                continue
            # This deliberately mirrors R's scalar loop. A vector reduction can
            # reassociate the additions and change the last-wins BUILD tie rule.
            gain = 0.0
            gain_terms: list[tuple[int, int]] = []
            for observation in range(n):
                candidate_distance = float(D[candidate, observation])
                if nearest[observation] > candidate_distance:
                    if exact_objectives:
                        gain_terms.append(
                            _product_component(
                                float(weights[observation]),
                                float(nearest[observation]),
                            )
                        )
                        gain_terms.append(
                            _product_component(
                                float(weights[observation]),
                                candidate_distance,
                                sign=-1,
                            )
                        )
                    else:
                        improvement = float(nearest[observation] - candidate_distance)
                        gain += float(weights[observation]) * improvement
            gain_component = _sum_components(gain_terms) if exact_objectives else (0, 0)
            improves_or_ties = (
                not _component_less(gain_component, best_gain_component)
                if exact_objectives
                else best_gain <= gain
            )
            if improves_or_ties:
                best_gain = gain
                best_gain_component = gain_component
                best_index = candidate
        selected[best_index] = True
        medoids.append(best_index)
        for observation in range(n):
            if nearest[observation] > D[best_index, observation]:
                nearest[observation] = D[best_index, observation]
    return np.asarray(medoids, dtype=int)


def _initial_membership_medoids(
    D: np.ndarray,
    initialclust: Sequence[Any] | np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
) -> np.ndarray:
    labels0, levels = _factorize_quality_labels(initialclust)
    if len(levels) != k:
        raise ValueError(f"initialclust membership must use exactly k={k} groups")
    exact_objectives = _pam_requires_exact_objectives(D, weights)
    return np.asarray(
        [
            (
                _exact_medoid_for_members(
                    D,
                    np.where(labels0 == cluster)[0],
                    weights,
                )
                if exact_objectives
                else find_medoid(
                    D,
                    np.where(labels0 == cluster)[0],
                    weights=weights,
                )
            )
            for cluster in range(k)
        ],
        dtype=int,
    )


def _coerce_initial_medoids(
    D: np.ndarray,
    initialclust: Sequence[Any] | np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
) -> np.ndarray:
    n = D.shape[0]
    init = np.asarray(initialclust)
    if init.shape == (n,):
        return _initial_membership_medoids(D, initialclust, k=k, weights=weights)
    if init.ndim == 2 and init.shape[1] == 4:
        validate_linkage_matrix(init, n, name="initialclust")
        try:
            membership = cut_linkage_by_merge_count(init.astype(float), n_clusters=k)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid linkage matrix for initialclust") from exc
        return _initial_membership_medoids(D, membership, k=k, weights=weights)
    flattened = np.asarray(initialclust)
    if flattened.ndim != 1 or flattened.size != k:
        raise ValueError(
            f"initialclust must be linkage (n-1x4), length {k} medoid indices, "
            f"or length {n} membership"
        )
    try:
        numeric = np.asarray(flattened, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("initialclust medoid indices must be integers") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("initialclust medoid indices must be integers")
    one_based = numeric.astype(int)
    if np.any(one_based < 1) or np.any(one_based > n):
        raise ValueError(f"starting medoids should be in 1:{n}")
    medoids = one_based - 1
    if np.unique(medoids).size != medoids.size:
        raise ValueError("starting medoids must be unique")
    return medoids


def _coerce_replay_starts(
    replay_starts: Sequence[Sequence[int]] | np.ndarray | None,
    *,
    k: int,
    n: int,
) -> np.ndarray | None:
    if replay_starts is None:
        return None
    starts = np.asarray(replay_starts)
    if starts.ndim != 2 or starts.shape[1] != k:
        raise ValueError(f"replay_starts must have shape (n_starts, {k})")
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("replay_starts must contain zero-based integer indices")
    starts = starts.astype(int)
    if np.any(starts < 0) or np.any(starts >= n):
        raise ValueError("replay_starts contains an out-of-range zero-based index")
    if any(np.unique(row).size != k for row in starts):
        raise ValueError("replay_starts rows must contain unique medoids")
    return starts


def wcKMedoids(
    diss: np.ndarray,
    k: int,
    weights: Optional[Sequence[float]] = None,
    npass: int = 1,
    initialclust: Optional[Sequence[Any] | np.ndarray] = None,
    method: str | int = "PAMonce",
    cluster_only: bool = False,
    debuglevel: int = 0,
    *,
    random_state: Optional[int] = None,
    replay_starts: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> KMedoidsResult | np.ndarray:
    """Weighted k-medoids (PAM) clustering on a distance matrix.

    Parameters
    ----------
    diss : np.ndarray
        Square (N, N) distance matrix or condensed distance vector.
    k : int
        Number of clusters (must be >= 2)
    weights : array-like, optional
        Case weights (length N), by default None (uniform weights)
    npass : int, optional
        Number of random initializations, by default 1
    initialclust : array-like, optional
        Initial clustering: membership vector, one-based medoid positions, an
        or a SciPy linkage matrix. Medoid positions are always one-based, as in
        WeightedCluster; use ``replay_starts`` for explicit zero-based starts.
    method : str or int, optional
        R-compatible KMedoids/PAM/PAMonce name, partial name, or numeric code.
        PAM is supported for square matrices, matching WeightedCluster 2.0;
        its condensed-distance branch remains unimplemented upstream.
    cluster_only : bool, optional
        If True, return only cluster labels; if False, return full statistics
    debuglevel : int, optional
        R-compatible debug level. The installed 2.0 build emits no debug text.
    random_state : int, optional
        Random seed for reproducibility
    replay_starts : array-like, optional
        Explicit zero-based start plan. For positive ``npass``, row zero
        represents R's wrapper-level unused draw and later rows replay the
        native multipass starts.

    Returns
    -------
    dict or np.ndarray
        If cluster_only=True: array of one-based medoid observation positions
        If cluster_only=False: dict with keys 'clustering', 'stats', 'ASW', 'ASWw', 'info'

    Raises
    ------
    ValueError
        If k < 2, k > N, or distance matrix is invalid

    Examples
    --------
    >>> import numpy as np
    >>> from weightedcluster import wcKMedoids
    >>> D = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 3.0], [4.0, 3.0, 0.0]])
    >>> result = wcKMedoids(D, k=2, random_state=42)
    >>> print(result['clustering'])
    [2 2 3]
    """
    input_is_condensed = np.asarray(diss).ndim == 1
    # Validate inputs
    D = _coerce_kmedoids_distance(diss)

    N = D.shape[0]
    k_int = validate_integer_control(k, name="k")
    npass_int = validate_integer_control(npass, name="npass")
    if npass_int < 0:
        raise ValueError("npass should be greater than 0")
    method_code = _normalize_kmedoids_method(method)
    if method_code == 2 and input_is_condensed:
        raise NotImplementedError(
            "PAM method is not (yet) implemented for condensed distance input"
        )
    validate_integer_control(debuglevel, name="debuglevel")

    if k_int < MIN_CLUSTERS:
        raise ValueError(
            f"k must be >= {MIN_CLUSTERS} (single cluster is meaningless), got {k_int}"
        )
    if k_int > N:
        raise ValueError(f"k ({k_int}) cannot exceed number of observations ({N})")

    w = _coerce_kmedoids_weights(weights, expected_n=N)
    exact_objectives = bool(np.all(w >= 0.0)) and _pam_requires_exact_objectives(D, w)

    init_m = None
    if initialclust is not None:
        init_m = _coerce_initial_medoids(D, initialclust, k=k_int, weights=w)
        npass_int = 0

    replay = _coerce_replay_starts(
        replay_starts,
        k=k_int,
        n=N,
    )
    rng = np.random.default_rng(generate_random_seed(random_state, 0))
    run_plan: list[tuple[np.ndarray, float]] = []
    if init_m is not None:
        if replay is not None:
            raise ValueError("replay_starts cannot be combined with initialclust")
        run_plan.append((init_m, 0.0))
    elif npass_int == 0:
        if replay is not None:
            if replay.shape[0] != 1:
                raise ValueError("npass=0 requires exactly one replay start")
            start = replay[0]
        else:
            start = np.asarray(rng.choice(N, size=k_int, replace=False), dtype=int)
        run_plan.append((start, 0.0))
    else:
        run_plan.append((_pam_build_initial_medoids(D, k_int, w), 1.0))
        if replay is not None:
            if replay.shape[0] != npass_int:
                raise ValueError(
                    "positive npass replay_starts must include the R wrapper's "
                    "first unused start and one row for each later pass"
                )
            random_starts = replay[1:]
        else:
            random_starts = np.asarray(
                [rng.choice(N, size=k_int, replace=False) for _ in range(npass_int - 1)],
                dtype=int,
            ).reshape((-1, k_int))
        run_plan.extend((row, 3.0) for row in random_starts)

    best_labels: np.ndarray | None = None
    best_obj = float("inf")
    best_objective_component: tuple[int, int] | None = None
    best_method = 0.0
    found = 0
    for start, start_method in run_plan:
        if exact_objectives and method_code == 1:
            labels, medoids_idx, objective = _kmedoids_from_start_exact(D, start, w)
        elif method_code == 1:
            labels, medoids_idx, objective = _kmedoids_from_start(D, start, w)
        elif exact_objectives and method_code == 2:
            labels, medoids_idx, objective = _pam_from_start_exact(D, start, w)
        elif method_code == 2:
            labels, medoids_idx, objective = _pam_from_start(D, start, w)
        elif exact_objectives:
            from ._fallback import weighted_medoids as stable_weighted_medoids

            labels, medoids_idx, objective = stable_weighted_medoids(
                D,
                k_int,
                weights=w,
                max_iter=DEFAULT_KMEDOIDS_MAX_ITER,
                n_init=1,
                random_state=0,
                init_medoids=np.asarray(start, dtype=int),
            )
        elif np.all(w >= 0.0):
            labels, medoids_idx, objective = _weighted_medoids(
                D,
                k_int,
                weights=w,
                max_iter=DEFAULT_KMEDOIDS_MAX_ITER,
                n_init=1,
                random_state=0,
                init_medoids=np.asarray(start, dtype=int),
            )
        else:
            labels, medoids_idx, objective = _pamonce_from_start(D, start, w)
        medoid_membership = medoids_idx[labels]
        objective_component = (
            _weighted_medoids_objective_component(D, medoids_idx, w) if exact_objectives else None
        )
        if best_labels is None:
            best_labels = medoid_membership.copy()
            best_obj = float(objective)
            best_method = start_method
            found = 1
            best_objective_component = objective_component
        elif np.array_equal(best_labels, medoid_membership):
            found += 1
        elif (
            exact_objectives
            and objective_component is not None
            and best_objective_component is not None
            and _component_less(objective_component, best_objective_component)
        ) or (not exact_objectives and objective < best_obj):
            best_labels = medoid_membership.copy()
            best_obj = float(objective)
            best_method = start_method
            found = 1
            best_objective_component = objective_component

    clustering_1b = (best_labels + EXTERNAL_LABEL_BASE).astype(int)
    if cluster_only:
        return clustering_1b

    # Compute full cluster quality statistics to match R WeightedCluster.
    # Unlike the upstream R object, expose unweighted and weighted per-cluster
    # silhouettes under distinct keys to avoid semantic ambiguity.
    cq = wcClusterQuality(D, clustering_1b, weights=w)
    asw_per_cluster: Dict[int, float] = cq.get("ASW", {})
    asww_per_cluster: Dict[int, float] = cq.get("ASWw", {})

    ret = {
        "clustering": clustering_1b,
        "info": dict(
            zip(
                _R_KMEDOIDS_INFO_NAMES,
                (float(best_obj), float(found), float(best_method)),
                strict=True,
            )
        ),
        "stats": cq.get("stats", {}),
        "ASW": asw_per_cluster,
        "ASWw": asww_per_cluster,
    }
    return ret


def print_kmedoids(result: KMedoidsResult, *, digits: int = 7) -> str:
    """Return the semantic text of R's ``print.kmedoids`` presentation."""
    digits_value = validate_integer_control(digits, name="digits")
    if not 1 <= digits_value <= 22:
        raise ValueError("digits must be between 1 and 22")

    def _number(value: object) -> str:
        numeric = float(value)
        if np.isnan(numeric):
            return "NaN"
        return format(numeric, f".{digits_value}g")

    def _row(values: Sequence[object]) -> str:
        numeric = np.asarray([float(value) for value in values], dtype=float)
        finite = np.abs(numeric[np.isfinite(numeric)])
        if finite.size == 0 or float(np.max(finite)) == 0.0:
            return " ".join(_number(value) for value in numeric)
        decimals = max(
            0,
            digits_value - 1 - int(np.floor(np.log10(float(np.max(finite))))),
        )
        return " ".join("NaN" if np.isnan(value) else f"{value:.{decimals}f}" for value in numeric)

    stats = result["stats"]
    cluster_names = list(result["ASW"])
    lines = [
        "Cluster quality:",
        " ".join(_R_QUALITY_NAMES),
        " ".join(_number(stats[name]) for name in _R_QUALITY_NAMES),
        "",
        "Per cluster Weighted Average Silhouette Width:",
        " ".join(str(name) for name in cluster_names),
        _row([result["ASW"][name] for name in cluster_names]),
        _row([result["ASWw"][name] for name in cluster_names]),
    ]
    return "\n".join(lines)


def wcKMedRange(
    diss: np.ndarray,
    kvals: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    R: int = 1,
    samplesize: Optional[int] = None,
    npass: int = 1,
    initialclust: Optional[Sequence[Any] | np.ndarray] = None,
    method: str | int = "PAMonce",
    debuglevel: int = 0,
    *,
    random_state: Optional[int] = None,
    replay_starts: (
        Mapping[int, Sequence[Sequence[int]] | np.ndarray]
        | Sequence[Sequence[int]]
        | np.ndarray
        | None
    ) = None,
    replay_samples: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> KMedRangeResult:
    """Compute k-medoids clustering across a range of K values.

    Parameters
    ----------
    diss : np.ndarray
        Square (N, N) distance matrix or condensed distance vector. PAM is
        supported only for square matrices, matching WeightedCluster 2.0.
    kvals : sequence of int
        Values of k to evaluate (each must be >= 2)
    weights : array-like, optional
        Case weights, by default None
    R : int, optional
        Number of bootstrap samples for confidence intervals, by default 1
    samplesize : int, optional
        Bootstrap sample size, by default sum of weights or N
    npass : int, optional
        Number of random initializations per k-medoids fit, by default 1
    initialclust, method, debuglevel
        R-compatible options forwarded to every ``wcKMedoids`` fit.
    random_state : int, optional
        Random seed for reproducibility (full sample and bootstrap), by default None
    replay_starts : mapping or array-like, optional
        Explicit zero-based k-medoids start plans.
    replay_samples : array-like, optional
        Explicit zero-based replacement samples with shape ``(R, samplesize)``.

    Returns
    -------
    KMedRangeResult
        Typed mapping with R fields ``R``, ``kvals``, ``clustering``,
        ``stats``, and (when ``R > 1``) ``boot``.

    Examples
    --------
    >>> import numpy as np
    >>> from weightedcluster import wcKMedRange
    >>> # Create a simple distance matrix
    >>> D = np.array([[0, 1, 4, 5],
    ...               [1, 0, 3, 4],
    ...               [4, 3, 0, 1],
    ...               [5, 4, 1, 0]], dtype=float)
    >>> # Evaluate clustering for k=2 and k=3
    >>> result = wcKMedRange(D, kvals=[2, 3], random_state=42)
    >>> result['kvals']
    [2, 3]
    >>> # Access cluster assignments for k=2
    >>> result['clustering']['cluster2'].shape
    (4,)
    >>> # Access quality statistics
    >>> len(result['stats']['ASW'])
    2

    >>> # With bootstrap resampling for confidence intervals
    >>> result_boot = wcKMedRange(D, kvals=[2, 3], R=10, random_state=42)
    >>> len(result_boot['boot'][2]['ASW'])
    11

    >>> # With custom weights
    >>> weights = np.array([1.0, 2.0, 1.5, 1.0])
    >>> result_weighted = wcKMedRange(D, kvals=[2], weights=weights, random_state=42)
    >>> len(result_weighted['stats']['ASWw'])
    1
    """
    input_is_condensed = np.asarray(diss).ndim == 1
    D = _coerce_kmedoids_distance(diss)
    N = D.shape[0]
    w = _coerce_kmedoids_weights(weights, expected_n=N)
    from ._validation import validate_kvals

    kvals_list = validate_kvals(kvals, max_k=N)
    npass_int = validate_integer_control(npass, name="npass")
    if npass_int < 0:
        raise ValueError("npass should be greater than 0")
    if _normalize_kmedoids_method(method) == 2 and input_is_condensed:
        raise NotImplementedError(
            "PAM method is not (yet) implemented for condensed distance input"
        )
    validate_integer_control(debuglevel, name="debuglevel")
    Rb = validate_integer_control(R, name="R")
    if Rb < 1:
        raise ValueError("R must be at least one")

    def _starts_for_k(k: int):
        if replay_starts is None:
            return None
        if isinstance(replay_starts, Mapping):
            return replay_starts.get(int(k))
        if len(kvals_list) != 1:
            raise ValueError("array replay_starts is only valid for a single k")
        return replay_starts

    clustering: Dict[str, np.ndarray] = {}
    stats_rows: List[Dict[str, float]] = []
    for k in kvals_list:
        labels1 = wcKMedoids(
            D,
            k,
            weights=w,
            npass=npass_int,
            initialclust=initialclust,
            method=method,
            cluster_only=True,
            debuglevel=debuglevel,
            random_state=random_state,
            replay_starts=_starts_for_k(k),
        )
        clustering[f"cluster{k}"] = labels1
        cq = wcClusterQuality(D, labels1, weights=w)
        stats_rows.append({name: float(cq["stats"][name]) for name in _R_QUALITY_NAMES})

    row_names = [f"cluster{k}" for k in kvals_list]
    stats = rows_to_stats_table(
        stats_rows,
        keys=_R_QUALITY_NAMES,
        row_names=row_names,
    )
    out = KMedRangeResult(
        {
            "R": Rb,
            "kvals": kvals_list,
            "clustering": ClusteringTable(clustering, column_names=row_names),
            "stats": stats,
        }
    )

    if Rb > 1:
        total_weight = float(np.sum(w))
        ss = (
            validate_integer_control(samplesize, name="samplesize")
            if samplesize is not None
            else int(np.floor(total_weight))
        )
        if ss < 1:
            raise ValueError("samplesize must be at least one")
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            raise ValueError("bootstrap sampling requires a positive finite total weight")
        probs = w / total_weight
        if np.any(probs < 0.0):
            raise ValueError("bootstrap sampling probabilities must be non-negative")

        supplied_samples = None
        if replay_samples is not None:
            supplied_samples = np.asarray(replay_samples)
            if supplied_samples.shape != (Rb, ss):
                raise ValueError(f"replay_samples must have shape ({Rb}, {ss})")
            if not np.issubdtype(supplied_samples.dtype, np.integer):
                raise ValueError("replay_samples must contain zero-based integer indices")
            supplied_samples = supplied_samples.astype(int)
            if np.any(supplied_samples < 0) or np.any(supplied_samples >= N):
                raise ValueError("replay_samples contains an out-of-range zero-based index")

        seed_boot = generate_random_seed(random_state, 1)
        rng = np.random.default_rng(seed_boot)
        sample_plan = (
            supplied_samples
            if supplied_samples is not None
            else np.asarray(
                [rng.choice(N, size=ss, replace=True, p=probs) for _ in range(Rb)],
                dtype=int,
            )
        )
        boot: Dict[int, Dict[str, List[float]]] = {}
        bootstrap_stats_rows: List[Dict[str, float]] = []
        for k in kvals_list:
            labels = clustering[f"cluster{k}"]
            t0_quality = _wc_cluster_quality_internal(D, labels, weights=w)
            t0_row = {name: float(t0_quality["stats"][name]) for name in _R_QUALITY_NAMES}
            bootstrap_stats_rows.append(t0_row)
            block = {name: [t0_row[name]] for name in _R_QUALITY_NAMES}
            for sample in sample_plan:
                counts = np.bincount(sample, minlength=N).astype(float)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="ASW can not be computed because at least one cluster",
                        category=UserWarning,
                    )
                    cq = _wc_cluster_quality_internal(D, labels, weights=counts)
                for name in _R_QUALITY_NAMES:
                    block[name].append(float(cq["stats"][name]))
            boot[int(k)] = block
        out["stats"] = rows_to_stats_table(
            bootstrap_stats_rows,
            keys=_R_QUALITY_NAMES,
            row_names=row_names,
        )
        out["boot"] = boot
    return out

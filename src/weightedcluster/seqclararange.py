from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._constants import (
    CLARA_BASE_SAMPLE_SIZE,
    CLARA_K_MULTIPLIER,
    FUZZY_CONVERGENCE_TOL,
    FUZZY_MAX_ITER,
)
from ._parallel import run_task_batch
from ._seqclara_runtime import (
    assemble_seqclararange_family_from_cache as _runtime_assemble_seqclararange_family_from_cache,
)
from ._seqclara_runtime import (
    assemble_seqclararange_result_from_cache as _runtime_assemble_seqclararange_result_from_cache,
)
from ._seqclara_runtime import (
    build_seqclararange_context as _runtime_build_seqclararange_context,
)
from ._seqclara_runtime import (
    run_seqclara_cache as _runtime_run_seqclara_cache,
)
from ._seqclara_runtime import (
    run_seqclara_iterations_for_k as _runtime_run_seqclara_iterations_for_k,
)
from ._seqclara_runtime import (
    select_best_run_index as _runtime_select_best_run_index,
)
from ._seqclara_runtime import (
    stability_metrics as _runtime_stability_metrics,
)
from ._seqdist import (
    sequences_distance,
    sequences_distance_between,
)
from ._utils import generate_random_seed
from ._validation import validate_seqdist_args
from .aggregate import wcAggregateCases
from .quality import (
    _positive_weighted_average,
    _signed_weighted_average,
    _signed_weighted_average_factors,
    wcClusterQuality,
    wcFuzzyQuality,
)
from .schemas import SeqClaraRangeResult, rows_to_stats_table, rows_to_table

_WCQ_CRITERIA_TO_STAT = {
    "pbc": "PBC",
    "hg": "HG",
    "hgsd": "HGSD",
    "asw": "ASW",
    "asww": "ASWw",
    "ch": "CH",
    "r2": "R2",
    "chsq": "CHsq",
    "r2sq": "R2sq",
    "hc": "HC",
}

_ALLOWED_SEQCLARA_CRITERIA = (
    "distance",
    "db",
    "xb",
    "pbm",
    "ams",
    *tuple(_WCQ_CRITERIA_TO_STAT.keys()),
)
_SEQCLARA_MEDOID_CRITERIA = ("distance", "xb", "pbm", "ams")
_SEQCLARA_WCQ_CRITERIA = frozenset(("db", *tuple(_WCQ_CRITERIA_TO_STAT.keys())))
_SEQCLARA_STATS_CORE = (
    "Avg dist",
    "PBM",
    "DB",
    "XB",
    "AMS",
    "ARI>0.8",
    "JC>0.8",
    "Best iter",
)
_SEQCLARA_FUZZY_METHODS = ("fuzzy", "noise", "hncdd", "pcmdd")
_R_SEQCLARA_METHODS = ("crisp", "fuzzy", "representativeness", "noise")
_R_SEQCLARA_CRITERIA = ("distance", "db", "xb", "pbm", "ams")


def _partial_match(value: object, choices: Sequence[str], *, name: str) -> str:
    """Apply R ``pmatch`` semantics to one case-sensitive workflow option."""
    candidate = str(value).lower()
    exact = [choice for choice in choices if choice.lower() == candidate]
    if exact:
        return exact[0]
    matches = [choice for choice in choices if choice.lower().startswith(candidate)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Unknown {name} {value!r}. Allowed values are: " + ", ".join(str(v) for v in choices)
    )


def _normalize_criteria(
    criteria: Sequence[str] | str | None,
    *,
    default: Sequence[str],
    allowed: Optional[Sequence[str]] = None,
) -> List[str]:
    """Normalize criteria input to a non-empty lowercase list."""
    if criteria is None:
        raw = list(default)
    elif isinstance(criteria, str):
        raw = [criteria]
    else:
        raw = list(criteria)
    if not raw:
        raw = list(default)
    norm = [str(c).lower() for c in raw]
    if allowed is not None:
        choices = tuple(str(a).lower() for a in allowed)
        resolved: list[str] = []
        for criterion in norm:
            resolved.append(_partial_match(criterion, choices, name="criteria"))
        norm = resolved
    return norm


def _medoid_silhouette_from_distances(D_to_m: np.ndarray) -> np.ndarray:
    """Compute medoid-based silhouette proxy from two nearest medoid distances."""
    D_to_m = np.asarray(D_to_m, dtype=float)
    N = D_to_m.shape[0]
    if D_to_m.size == 0:
        return np.zeros(N, dtype=float)
    srt = np.sort(D_to_m, axis=1)
    alpha = srt[:, 0]
    if srt.shape[1] >= 2:
        beta = srt[:, 1]
    else:
        beta = np.zeros_like(alpha)
    denom = np.maximum(alpha, beta)
    return np.divide(beta - alpha, denom, out=np.zeros_like(alpha), where=denom > 0)


def _membership_confidence(U: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 membership per row (or top-1 when K=1)."""
    U = np.asarray(U, dtype=float)
    N = U.shape[0]
    K = U.shape[1] if U.ndim == 2 else 0
    if K <= 0:
        return np.zeros(N, dtype=float)
    srt = np.sort(U, axis=1)
    if K == 1:
        return srt[:, -1].copy()
    return srt[:, -1] - srt[:, -2]


def _medoid_separation(D_full: np.ndarray, medoids_global_idx: np.ndarray) -> tuple[float, float]:
    """Return (min_sep, max_sep) among medoids in full distance matrix."""
    meds = np.asarray(medoids_global_idx, dtype=int).ravel()
    meds = np.unique(meds)
    if meds.size < 2:
        return float("nan"), float("nan")
    sub = np.asarray(D_full[np.ix_(meds, meds)], dtype=float)
    ti, tj = np.triu_indices(sub.shape[0], k=1)
    vals = sub[ti, tj]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.min(vals)), float(np.max(vals))


def _positive_term_reduction_needs_decimal(*factors: np.ndarray) -> bool:
    """Return whether a positive product or its reduction leaves normal range."""
    arrays = np.broadcast_arrays(*[np.asarray(factor, dtype=float) for factor in factors])
    positive = np.logical_and.reduce([array > 0.0 for array in arrays])
    if not np.any(positive):
        return False
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        products = np.prod(np.stack(arrays, axis=0), axis=0)
        total = float(np.sum(products))
    return bool(
        not np.isfinite(total)
        or np.any(positive & ((products == 0.0) | ~np.isfinite(products)))
        or np.any(positive & (products < np.finfo(float).tiny))
    )


def _decimal_fuzzy_functional(
    distances: np.ndarray,
    memberships: np.ndarray,
    exponent: float,
    weights: np.ndarray,
) -> Decimal:
    """Evaluate a positive fuzzy functional without a binary64 range boundary."""
    with localcontext() as context:
        context.prec = 800
        exponent_decimal = Decimal.from_float(float(exponent))
        total = Decimal(0)
        for row in range(distances.shape[0]):
            weight = float(weights[row])
            if weight <= 0.0:
                continue
            weight_decimal = Decimal.from_float(weight)
            for column in range(distances.shape[1]):
                distance = float(distances[row, column])
                membership = float(memberships[row, column])
                if distance <= 0.0 or membership <= 0.0:
                    continue
                total += (
                    Decimal.from_float(distance)
                    * context.power(Decimal.from_float(membership), exponent_decimal)
                    * weight_decimal
                )
        return +total


def _decimal_crisp_functional(distances: np.ndarray, weights: np.ndarray) -> Decimal:
    """Evaluate a crisp positive functional without rounding its term products."""
    with localcontext() as context:
        context.prec = 800
        return +sum(
            (
                Decimal.from_float(float(distance)) * Decimal.from_float(float(weight))
                for distance, weight in zip(distances, weights, strict=True)
                if distance > 0.0 and weight > 0.0
            ),
            Decimal(0),
        )


def _decimal_fuzzy_db(
    D_to_m: np.ndarray,
    memberships: np.ndarray,
    weights: np.ndarray,
    medoids_global_idx: np.ndarray,
) -> float:
    """Compute fuzzy DB while retaining membership-by-weight factors."""
    k = memberships.shape[1]
    with localcontext() as context:
        context.prec = 800
        diameters: list[Decimal | None] = []
        for cluster in range(k):
            numerator = Decimal(0)
            denominator = Decimal(0)
            for row in range(D_to_m.shape[0]):
                membership = float(memberships[row, cluster])
                weight = float(weights[row])
                if membership <= 0.0 or weight <= 0.0:
                    continue
                factor = Decimal.from_float(membership) * Decimal.from_float(weight)
                denominator += factor
                distance = float(D_to_m[row, cluster])
                if distance > 0.0:
                    numerator += Decimal.from_float(distance) * factor
            diameters.append(numerator / denominator if denominator > 0 else None)

        separation = D_to_m[np.asarray(medoids_global_idx, dtype=int), :]
        maxima: list[Decimal] = []
        for cluster in range(k):
            if diameters[cluster] is None:
                continue
            ratios: list[Decimal] = []
            for other in range(k):
                if diameters[other] is None:
                    continue
                distance = float(separation[cluster, other])
                if distance <= 0.0:
                    continue
                ratios.append(
                    (diameters[cluster] + diameters[other]) / Decimal.from_float(distance)
                )
            if ratios:
                maxima.append(max(ratios))
        if not maxima:
            return float("nan")
        return float(sum(maxima, Decimal(0)) / Decimal(len(maxima)))


def _compute_clara_criteria(
    *,
    method_l: str,
    D_full: np.ndarray,
    D_to_m: np.ndarray,
    labels_full: np.ndarray,
    medoids_global_idx: np.ndarray,
    memb_full_all: Optional[np.ndarray],
    fuzzifier_m: float,
    dnoise_value: Optional[float],
    case_weights: Optional[np.ndarray] = None,
    requested_criteria: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute CLARA run criteria with inherited and Python-specific directions."""
    N = D_to_m.shape[0]
    k = D_to_m.shape[1]
    weights = (
        np.ones(N, dtype=float)
        if case_weights is None
        else np.asarray(case_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (N,):
        raise ValueError("case_weights must have one value per aggregated sequence")
    # Sampling probabilities and analysis weights are different quantities.
    # The caller supplies aggregate case weights here; only the unweighted
    # default needs conversion to the mean-compatible 1 / N vector.
    probs = weights if case_weights is not None else weights / float(max(1, N))
    requested = (
        {str(c).lower() for c in requested_criteria}
        if requested_criteria is not None
        else {str(c).lower() for c in _ALLOWED_SEQCLARA_CRITERIA}
    )
    need_distance = "distance" in requested
    need_ams = "ams" in requested
    need_separation = bool(requested & {"xb", "pbm"})
    need_mean_diss = bool(requested & {"distance", "xb", "pbm"})
    need_quality = bool(requested & _SEQCLARA_WCQ_CRITERIA)

    mean_diss = float("nan")
    pbm_denom = float("nan")
    mean_diss_decimal: Decimal | None = None
    pbm_denom_decimal: Decimal | None = None
    ams = float("nan")

    if method_l in ("fuzzy", "noise", "hncdd", "pcmdd") and memb_full_all is not None:
        from .fuzzy import _stable_fuzzy_functional

        U_all = np.asarray(memb_full_all, dtype=float)
        U_core = U_all[:, :k] if U_all.shape[1] >= k else U_all
        functional_distances = D_to_m
        functional_memberships = U_core
        if method_l in ("noise", "hncdd") and need_mean_diss:
            dnoise_eff = float(dnoise_value) if dnoise_value is not None else float("nan")
            if not np.isfinite(dnoise_eff):
                dnoise_eff = float(np.nanmax(D_to_m)) if D_to_m.size else 1.0
            D3 = np.concatenate([D_to_m, np.full((N, 1), dnoise_eff, dtype=float)], axis=1)
            functional_distances = D3
            functional_memberships = U_all
            mean_diss = _stable_fuzzy_functional(
                D3,
                U_all,
                float(fuzzifier_m),
                probs,
            )
            pbm_denom = _stable_fuzzy_functional(D3, U_all, 1.0, probs)
        elif need_mean_diss:
            mean_diss = _stable_fuzzy_functional(
                D_to_m,
                U_core,
                float(fuzzifier_m),
                probs,
            )
            pbm_denom = _stable_fuzzy_functional(D_to_m, U_core, 1.0, probs)
        if need_mean_diss:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                powered_memberships = functional_memberships ** float(fuzzifier_m)
            conceptually_positive = (
                (functional_distances > 0.0)
                & (functional_memberships > 0.0)
                & (probs[:, None] > 0.0)
            )
            powered_membership_range_loss = conceptually_positive & (
                (powered_memberships == 0.0)
                | ~np.isfinite(powered_memberships)
                | (powered_memberships < np.finfo(float).tiny)
            )
            if np.any(powered_membership_range_loss) or _positive_term_reduction_needs_decimal(
                functional_distances,
                powered_memberships,
                probs[:, None],
            ):
                mean_diss_decimal = _decimal_fuzzy_functional(
                    functional_distances,
                    functional_memberships,
                    float(fuzzifier_m),
                    probs,
                )
                mean_diss = float(mean_diss_decimal)
            if _positive_term_reduction_needs_decimal(
                functional_distances,
                functional_memberships,
                probs[:, None],
            ):
                pbm_denom_decimal = _decimal_fuzzy_functional(
                    functional_distances,
                    functional_memberships,
                    1.0,
                    probs,
                )
                pbm_denom = float(pbm_denom_decimal)
        if need_ams:
            sil = _medoid_silhouette_from_distances(D_to_m)
            diffs = _membership_confidence(U_core)
            ams = (
                _signed_weighted_average_factors(sil, diffs, probs)
                if np.any((diffs > 0.0) & (probs > 0.0))
                else _signed_weighted_average(sil, probs)
            )
    else:
        if need_mean_diss:
            alpha = np.min(D_to_m, axis=1) if D_to_m.size else np.zeros(N, dtype=float)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                mean_diss = float(np.nansum(alpha * probs))
            pbm_denom = mean_diss
            if _positive_term_reduction_needs_decimal(alpha, probs):
                mean_diss_decimal = _decimal_crisp_functional(alpha, probs)
                pbm_denom_decimal = mean_diss_decimal
                mean_diss = float(mean_diss_decimal)
                pbm_denom = mean_diss
        if need_ams:
            sil = _medoid_silhouette_from_distances(D_to_m)
            ams = float(np.nansum(sil * probs))

    crits: Dict[str, float] = {}
    if need_distance:
        crits["distance"] = float(mean_diss)
    if need_ams:
        crits["ams"] = float(ams)
    if need_separation:
        min_sep, max_sep = _medoid_separation(D_full, medoids_global_idx)
        if "xb" in requested:
            if np.isfinite(min_sep) and min_sep > 0:
                crits["xb"] = (
                    float(mean_diss_decimal / Decimal.from_float(min_sep))
                    if mean_diss_decimal is not None
                    else float(mean_diss / min_sep)
                )
            else:
                crits["xb"] = float("nan")
        if "pbm" in requested:
            if (
                np.isfinite(max_sep)
                and k > 0
                and ((pbm_denom_decimal is not None and pbm_denom_decimal > 0) or pbm_denom > 0)
            ):
                if pbm_denom_decimal is not None:
                    ratio = Decimal.from_float(max_sep) / Decimal(k) / pbm_denom_decimal
                    crits["pbm"] = float(ratio * ratio)
                else:
                    crits["pbm"] = float(((max_sep / float(k)) / pbm_denom) ** 2)
            else:
                crits["pbm"] = float("nan")
    if need_quality:
        if "db" in requested:
            if memb_full_all is not None and _positive_term_reduction_needs_decimal(
                np.asarray(memb_full_all, dtype=float)[:, :k],
                weights[:, None],
            ):
                crits["db"] = _decimal_fuzzy_db(
                    D_to_m,
                    np.asarray(memb_full_all, dtype=float)[:, :k],
                    weights,
                    medoids_global_idx,
                )
            elif memb_full_all is not None:
                memberships = np.asarray(memb_full_all, dtype=float)[:, :k]
                diameters = np.asarray(
                    [
                        _positive_weighted_average(
                            D_to_m[:, cluster],
                            memberships[:, cluster] * weights,
                        )
                        for cluster in range(k)
                    ],
                    dtype=float,
                )
            else:
                labels0 = np.asarray(labels_full, dtype=int)
                diameters = np.asarray(
                    [
                        (
                            np.average(
                                D_to_m[labels0 == cluster, cluster],
                                weights=weights[labels0 == cluster],
                            )
                            if np.any(labels0 == cluster)
                            else np.nan
                        )
                        for cluster in range(k)
                    ],
                    dtype=float,
                )
            if "db" not in crits:
                separation = D_to_m[np.asarray(medoids_global_idx, dtype=int), :]
                per_cluster: list[float] = []
                for cluster in range(k):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratios = (diameters[cluster] + diameters) / separation[cluster]
                    finite = ratios[np.isfinite(ratios)]
                    per_cluster.append(float(np.max(finite)) if finite.size else float("nan"))
                crits["db"] = float(np.mean(per_cluster))
        requested_quality = requested & set(_WCQ_CRITERIA_TO_STAT)
        if requested_quality:
            labels1 = (np.asarray(labels_full, dtype=int) + 1).astype(int)
            cq_stats = wcClusterQuality(D_full, labels1, weights=weights)["stats"]
            for key in requested_quality:
                stat_name = _WCQ_CRITERIA_TO_STAT[key]
                crits[key] = float(cq_stats.get(stat_name, np.nan))
    return crits


def _objective_value(primary: str, crits: Dict[str, float]) -> float:
    """Map a criterion value to minimization objective."""
    p = str(primary).lower()
    if p in ("distance", "db", "xb", "hc"):
        val = float(crits.get(p, np.nan))
    elif p in ("pbm", "ams", *tuple(_WCQ_CRITERIA_TO_STAT.keys())):
        val = -float(crits.get(p, np.nan))
    else:
        val = float("nan")
    return float(val) if np.isfinite(val) else float("inf")


def _ward_initial_hierarchy(Ds: np.ndarray, *, weights: np.ndarray) -> np.ndarray:
    """Build the Ward hierarchy R seeds each CLARA iteration from.

    ``R/clara_clustrange.R:101`` is
    ``fastcluster::hclust(as.dist(diss), method="ward.D", members=ac2$aggWeights)``
    — an explicitly qualified fastcluster call, so this uses the same port
    ``wcCmpCluster`` does.
    """
    from ._fastcluster_linkage import linkage_fastcluster

    return linkage_fastcluster(Ds, method="ward.d", members=weights)


def _cluster_crisp(
    Ds: np.ndarray,
    k: int,
    sample_weights: np.ndarray,
    pam_n_init: int,
    seed_run: int,
    X: np.ndarray,
    Xs: np.ndarray,
    idx_global: np.ndarray,
    metric: str,
    indel: float,
    substitution_costs: Optional[np.ndarray],
    timecost: Optional[float],
    init_membership_sub: Optional[np.ndarray] = None,
    normalize_distances: bool = True,
    distance_scale: float = 1.0,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Execute crisp (k-medoids) clustering on subsample and assign full data to nearest medoids.

    This function implements the CLARA algorithm's core clustering step for crisp (hard) clustering.
    It performs k-medoids on a subsample Xs (using precomputed distances Ds), then assigns all
    observations in the full dataset X to their nearest medoid by computing distances from X to
    the medoid sequences.

    Parameters
    ----------
    Ds : np.ndarray
        Precomputed pairwise distance matrix for subsample Xs, shape (n_sample, n_sample)
    k : int
        Number of clusters/medoids
    pam_n_init : int
        Number of random initializations for PAM algorithm
    seed_run : int
        Random seed for this iteration
    X : np.ndarray
        Full sequence dataset, shape (N, T)
    Xs : np.ndarray
        Subsample sequence dataset, shape (n_sample, T)
    metric : str
        Distance metric ('hamming', 'om', etc.)
    indel : float
        Indel cost for OM distance
    substitution_costs : Optional[np.ndarray]
        Substitution cost matrix for OM distance, shape (n_states, n_states)

    Returns
    -------
    labels_full : np.ndarray
        Cluster assignments (0-indexed) for full dataset X, shape (N,)
    memb_full : None
        Always None for crisp clustering (placeholder for consistency with fuzzy interface)
    medoids_global_idx : np.ndarray
        Global medoid indices in full dataset X, shape (k,)
    D_full_to_m : np.ndarray
        Distances from each full-data sequence to each medoid, shape (N, k)

    Notes
    -----
    This function performs two phases:
    1. K-medoids clustering on the subsample using precomputed distances Ds
    2. Assignment of full dataset by computing distances from X to identified medoid sequences

    The medoid indices from phase 1 are local to the subsample Xs, but the returned labels_full
    are aligned to the full dataset X indices.
    """
    from .kmedoids import wcKMedoids

    # R seeds every CLARA iteration deterministically rather than searching:
    # `hc <- fastcluster::hclust(as.dist(diss), method="ward.D",
    # members=ac2$aggWeights)` then `wcKMedoids(diss, k, cluster.only=TRUE,
    # initialclust=hc, weights=ac2$aggWeights)` (R/clara_clustrange.R:101,:125).
    # Until finding F-05 of the 2026-08-24 whole-codebase audit this branch ran
    # random-restart PAM, so for a given subsample R's clustering was
    # deterministic while Python's was a search, and the selected iteration, its
    # medoids and its reported criteria could all differ from R.
    initial = (
        np.asarray(init_membership_sub).reshape(-1)
        if init_membership_sub is not None
        else _ward_initial_hierarchy(Ds, weights=sample_weights)
    )
    medoid_membership_1b = np.asarray(
        wcKMedoids(
            Ds,
            k,
            weights=sample_weights,
            npass=0,
            initialclust=initial,
            method="PAMonce",
            cluster_only=True,
        ),
        dtype=int,
    )
    # R: `medoids <- mysample[ac2$aggIndex[unique(clustering)]]`
    # (R/clara_clustrange.R:127). R's `unique()` keeps first-appearance order,
    # whereas `np.unique` sorts. The column order matters: the assignment below
    # takes the first minimum, as R's `which.min` does (:193), so a case
    # equidistant from two medoids joins whichever comes first. Sorting put the
    # tie-breaking medoid in a different place from R and changed the returned
    # partition and DB on tie-heavy data (2026-09-21 targeted audit, M-1).
    _, first_positions = np.unique(medoid_membership_1b, return_index=True)
    medoids_idx_sub = medoid_membership_1b[np.sort(first_positions)] - 1

    medoid_seqs = Xs[medoids_idx_sub]
    D_full_to_m = float(distance_scale) * sequences_distance_between(
        X,
        medoid_seqs,
        metric=metric,
        normalize=bool(normalize_distances),
        indel=indel,
        substitution_costs=substitution_costs,
        timecost=timecost,
    )
    labels_full = D_full_to_m.argmin(axis=1)
    medoids_global_idx = np.asarray(idx_global, dtype=int)[np.asarray(medoids_idx_sub, dtype=int)]
    return labels_full, None, medoids_global_idx, D_full_to_m


def _cluster_fuzzy(
    Ds: np.ndarray,
    k: int,
    sample_weights: np.ndarray,
    method: str,
    pam_n_init: int,
    seed_run: int,
    X: np.ndarray,
    Xs: np.ndarray,
    idx_global: np.ndarray,
    metric: str,
    indel: float,
    substitution_costs: Optional[np.ndarray],
    timecost: Optional[float],
    m: float,
    dnoise: Optional[float],
    dlambda: Optional[float],
    eta: Optional[np.ndarray],
    init_memberships_sub: Optional[Sequence[np.ndarray]] = None,
    normalize_distances: bool = True,
    distance_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """Execute fuzzy clustering (FCMdd, NCdd, HNCdd, PCMdd) on subsample and assign full data.

    This function implements CLARA with fuzzy clustering variants. It performs two-phase clustering:
    1. Initialize with crisp k-medoids on subsample, then refine with fuzzy algorithm
    2. Compute membership matrix for full dataset relative to identified medoids

    Parameters
    ----------
    Ds : np.ndarray
        Precomputed pairwise distance matrix for subsample Xs, shape (n_sample, n_sample)
    k : int
        Number of clusters/medoids (excluding noise cluster if applicable)
    method : str
        Fuzzy clustering variant: 'fuzzy' (FCMdd), 'noise' (NCdd), 'hncdd', or 'pcmdd'
    pam_n_init : int
        Number of random initializations for initial crisp clustering
    seed_run : int
        Random seed for this iteration
    X : np.ndarray
        Full sequence dataset, shape (N, T)
    Xs : np.ndarray
        Subsample sequence dataset, shape (n_sample, T)
    metric : str
        Distance metric ('hamming', 'om', etc.)
    indel : float
        Indel cost for OM distance
    substitution_costs : Optional[np.ndarray]
        Substitution cost matrix for OM distance
    m : float
        Fuzziness exponent (typically 1.5-2.5)
    dnoise : Optional[float]
        Noise distance parameter for NCdd/HNCdd methods
    dlambda : Optional[float]
        Lambda parameter controlling noise cluster weight for NCdd/HNCdd
    eta : Optional[np.ndarray]
        Scale parameters for PCMdd method, shape (k,). Auto-estimated if None.

    Returns
    -------
    labels_full : np.ndarray
        Hard cluster assignments (0-indexed) for full dataset X, shape (N,)
        Determined by maximum membership value
    memb_full : np.ndarray
        Fuzzy membership matrix for full dataset, shape (N, k) for FCMdd/PCMdd
        or (N, k+1) for NCdd/HNCdd (last column is noise cluster)
    medoids_global_idx : np.ndarray
        Global medoid indices in full dataset X, shape (k,)
    D_full_to_m : np.ndarray
        Distances from each full-data sequence to each medoid, shape (N, k)
    dnoise_value : Optional[float]
        Effective noise-distance value used by noise methods.

    Notes
    -----
    The fuzzy clustering process:
    1. Initializes with crisp k-medoids to get starting medoid positions
    2. Converts crisp labels to initial membership matrix (one-hot encoding)
    3. Runs fuzzy c-medoids algorithm (fcmdd) to refine memberships on subsample
    4. Computes membership for full dataset based on distances to final medoids

    For PCMdd, if eta is not provided, it is estimated as the mean distance from each
    cluster member to its medoid in the initial crisp clustering.

    For noise clustering (NCdd/HNCdd), the membership matrix includes an additional
    column for the noise cluster, which represents outliers with uniform distance dnoise.
    """
    from ._utils import find_medoid
    from ._validation import cut_linkage_by_merge_count

    # R starts the fuzzy branch from the same Ward hierarchy as the crisp one:
    # `memb <- as.memb(cutree(hc, k))` (R/clara_clustrange.R:112). Until finding
    # F-05 of the 2026-08-24 whole-codebase audit this was a random-restart
    # crisp fit instead, so the starting membership was a search result rather
    # than R's deterministic cut.
    # `cut_linkage_by_merge_count` returns R's one-based `cutree` labels; the
    # membership matrix below is indexed from zero.
    labels_init = (
        cut_linkage_by_merge_count(
            _ward_initial_hierarchy(Ds, weights=sample_weights), n_clusters=k
        )
        - 1
    )
    medoids_idx_sub_init = np.asarray(
        [
            find_medoid(Ds, np.where(labels_init == cluster)[0], weights=sample_weights)
            for cluster in range(k)
        ],
        dtype=int,
    )
    default_u0 = np.zeros((Xs.shape[0], k), dtype=float)
    for c in range(k):
        default_u0[labels_init == c, c] = 1.0
    candidate_memberships = (
        [np.asarray(value, dtype=float) for value in init_memberships_sub]
        if init_memberships_sub is not None
        else [default_u0]
    )
    for membership in candidate_memberships:
        if membership.shape != (Xs.shape[0], k):
            raise ValueError(f"replayed initial membership must have shape {(Xs.shape[0], k)}")

    from .fuzzy import _normalise_inverse_distances, fcmdd

    # Determine method and parameters
    method_l = method.lower()
    if method_l == "fuzzy":
        methname = "FCMdd"
        dnoise_arg = None
    elif method_l == "noise":
        methname = "NCdd"
        dnoise_arg = float(dnoise) if dnoise is not None else None
    elif method_l == "hncdd":
        methname = "HNCdd"
        dnoise_arg = float(dnoise) if dnoise is not None else None
    else:  # pcmdd
        methname = "PCMdd"
        dnoise_arg = None

    # Estimate eta for PCMdd if not provided
    eta_arg = None
    if methname == "PCMdd":
        if eta is not None:
            eta_arg = eta
        else:
            eta_est = np.zeros(k, dtype=float)
            for c in range(k):
                idxc = np.where(labels_init == c)[0]
                if idxc.size > 0:
                    eta_est[c] = float(
                        np.average(
                            Ds[idxc, int(medoids_idx_sub_init[c])],
                            weights=sample_weights[idxc],
                        )
                    )
                else:
                    eta_est[c] = float(np.mean(Ds))
            eta_arg = eta_est

    # R compares the Ward initialization with fanny's initialization and keeps
    # fanny on an exact functional tie. Replay plans therefore order candidates
    # as Ward first and fanny second.
    candidates = [
        fcmdd(
            Ds,
            membership,
            weights=sample_weights,
            method=methname,
            m=float(m),
            dnoise=dnoise_arg,
            dlambda=(
                float(dlambda) if (methname in ("NCdd", "HNCdd") and dlambda is not None) else None
            ),
            eta=eta_arg,
            alpha=FUZZY_CONVERGENCE_TOL,
            max_iter=FUZZY_MAX_ITER,
        )
        for membership in candidate_memberships
    ]
    fz = candidates[-1]
    if len(candidates) > 1 and float(candidates[0]["functional"]) < float(fz["functional"]):
        fz = candidates[0]

    medoids_idx_sub = np.array(fz["mobileCenters"], dtype=int)
    medoid_seqs = Xs[medoids_idx_sub]
    D_full_to_m = float(distance_scale) * sequences_distance_between(
        X,
        medoid_seqs,
        metric=metric,
        normalize=bool(normalize_distances),
        indel=indel,
        substitution_costs=substitution_costs,
        timecost=timecost,
    )

    # Compute membership for full data
    N = X.shape[0]
    D3 = (
        D_full_to_m
        if method_l == "fuzzy"
        else np.concatenate(
            [
                D_full_to_m,
                np.full((N, 1), float(fz.get("dnoise", np.nan)), dtype=float),
            ],
            axis=1,
        )
    )
    memb_full = _normalise_inverse_distances(D3, 1.0 / (float(m) - 1.0))
    labels_full = memb_full.argmax(axis=1)

    medoids_global_idx = np.asarray(idx_global, dtype=int)[medoids_idx_sub]
    dnoise_out = float(fz.get("dnoise", np.nan)) if method_l in ("noise", "hncdd") else None
    return labels_full, memb_full, medoids_global_idx, D_full_to_m, dnoise_out


def _stability_metrics(
    best_labels: np.ndarray,
    all_labels: List[np.ndarray],
    *,
    weights: Optional[np.ndarray] = None,
    fuzzy: bool = False,
) -> Dict[str, object]:
    """Compute stability counts and means against the selected solution."""
    return _runtime_stability_metrics(
        best_labels,
        all_labels,
        weights=weights,
        fuzzy=fuzzy,
    )


def _reaggregate_sampled_cases(
    *,
    rng_local: np.random.Generator,
    Xagg: np.ndarray,
    agg_index: np.ndarray,
    probs: np.ndarray,
    sample_size: int,
    min_unique: int,
    sampled_idx: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample aggregated cases with replacement and compress duplicates into weights."""
    if sampled_idx is None:
        sampled_idx = np.asarray(
            rng_local.choice(
                Xagg.shape[0],
                size=int(sample_size),
                replace=True,
                p=np.asarray(probs, dtype=float) / float(np.sum(probs)),
            ),
            dtype=int,
        )
    else:
        sampled_idx = np.asarray(sampled_idx, dtype=int).reshape(-1)
    sample_ac = wcAggregateCases(sampled_idx.reshape(-1, 1))
    sampled_unique_idx = sampled_idx[np.asarray(sample_ac["aggIndex"], dtype=int)]
    if sampled_unique_idx.size >= int(min_unique):
        return (
            Xagg[sampled_unique_idx],
            np.asarray(agg_index, dtype=int)[sampled_unique_idx],
            np.asarray(sample_ac["aggWeights"], dtype=float),
        )
    # Re-aggregation can collapse the subsample below k; fall back to the
    # uncompressed draw so clustering remains feasible.
    return (
        Xagg[sampled_idx],
        np.asarray(agg_index, dtype=int)[sampled_idx],
        np.ones(sampled_idx.shape[0], dtype=float),
    )


def _build_seqclararange_context(
    seqdata: np.ndarray,
    *,
    R: int,
    sample_size: Optional[int],
    kvals: Sequence[int],
    seqdist_args: Optional[Dict],
    method: str,
    criteria_norm: Sequence[str],
    stability: bool,
    dnoise: Optional[float],
    m: float,
    dlambda: Optional[float],
    eta: Optional[np.ndarray],
    pam_n_init: int,
    parallel: bool,
    n_jobs: Optional[int],
    progressbar: bool,
    keep_diss: bool,
    max_dist: Optional[float],
    random_state: Optional[int],
    export_membership: bool,
    replay_samples: Optional[np.ndarray],
    replay_initial_memberships: Optional[Mapping[int, Sequence[object]]],
) -> Dict[str, object]:
    """Prepare immutable inputs shared by single-criterion and family runs."""
    return _runtime_build_seqclararange_context(
        seqdata,
        R=R,
        sample_size=sample_size,
        kvals=kvals,
        seqdist_args=seqdist_args,
        method=method,
        criteria_norm=criteria_norm,
        stability=stability,
        dnoise=dnoise,
        m=m,
        dlambda=dlambda,
        eta=eta,
        pam_n_init=pam_n_init,
        parallel=parallel,
        n_jobs=n_jobs,
        progressbar=progressbar,
        keep_diss=keep_diss,
        max_dist=max_dist,
        random_state=random_state,
        export_membership=export_membership,
        replay_samples=replay_samples,
        replay_initial_memberships=replay_initial_memberships,
        clara_base_sample_size=CLARA_BASE_SAMPLE_SIZE,
        clara_k_multiplier=CLARA_K_MULTIPLIER,
        aggregate_cases_fn=wcAggregateCases,
        validate_seqdist_args_fn=validate_seqdist_args,
        sequences_distance_fn=sequences_distance,
        generate_random_seed_fn=generate_random_seed,
    )


def _run_seqclara_iterations_for_k(
    *,
    k: int,
    context: Dict[str, object],
) -> List[Dict[str, object]]:
    """Generate and evaluate all CLARA iterations for one k."""
    return _runtime_run_seqclara_iterations_for_k(
        k=k,
        context=context,
        reaggregate_sampled_cases_fn=_reaggregate_sampled_cases,
        cluster_crisp_fn=_cluster_crisp,
        cluster_fuzzy_fn=_cluster_fuzzy,
        compute_clara_criteria_fn=_compute_clara_criteria,
        sequences_distance_fn=sequences_distance,
        run_task_batch_fn=run_task_batch,
        generate_random_seed_fn=generate_random_seed,
        fuzzy_methods=_SEQCLARA_FUZZY_METHODS,
    )


def _select_best_run_index(runs: Sequence[Dict[str, object]], primary: str) -> Optional[int]:
    """Return the index of the best run for a given criterion."""
    return _runtime_select_best_run_index(runs, primary, _objective_value)


def _run_seqclara_cache(context: Dict[str, object]) -> Dict[int, List[Dict[str, object]]]:
    """Run CLARA once and keep every evaluated iteration for later selection."""
    return _runtime_run_seqclara_cache(
        context,
        run_seqclara_iterations_for_k_fn=_run_seqclara_iterations_for_k,
    )


def _assemble_seqclararange_result_from_cache(
    context: Dict[str, object],
    runs_by_k: Dict[int, List[Dict[str, object]]],
    *,
    primary: str,
) -> SeqClaraRangeResult:
    """Select the best run per k from a cached CLARA iteration matrix."""
    return _runtime_assemble_seqclararange_result_from_cache(
        context,
        runs_by_k,
        primary=primary,
        select_best_run_index_fn=_select_best_run_index,
        stability_metrics_fn=_stability_metrics,
        compute_clara_criteria_fn=_compute_clara_criteria,
        sequences_distance_between_fn=sequences_distance_between,
        wc_cluster_quality_fn=wcClusterQuality,
        wc_fuzzy_quality_fn=wcFuzzyQuality,
        rows_to_stats_table_fn=rows_to_stats_table,
        seqclara_stats_core=_SEQCLARA_STATS_CORE,
        seqclara_medoid_criteria=_SEQCLARA_MEDOID_CRITERIA,
        seqclara_fuzzy_methods=_SEQCLARA_FUZZY_METHODS,
    )  # type: ignore[return-value]


def _assemble_seqclararange_family_from_cache(
    context: Dict[str, object],
    runs_by_k: Dict[int, List[Dict[str, object]]],
) -> Dict[str, object]:
    """Build the family view by re-selecting across one cached iteration matrix."""
    return _runtime_assemble_seqclararange_family_from_cache(
        context,
        runs_by_k,
        assemble_seqclararange_result_from_cache_fn=_assemble_seqclararange_result_from_cache,
        rows_to_table_fn=rows_to_table,
    )


def seqclararange_family(
    seqdata: Any,
    *,
    R: int = 100,
    sample_size: Optional[int] = None,
    kvals: Sequence[int] = tuple(range(2, 11)),
    seqdist_args: Optional[Dict] = None,
    method: str = "crisp",
    criteria: Sequence[str] | str | None = ("distance", "db", "xb", "pbm", "ams"),
    stability: bool = False,
    dnoise: Optional[float] = None,
    m: float = 1.5,
    dlambda: Optional[float] = None,
    eta: Optional[np.ndarray] = None,
    pam_n_init: int = 2,
    parallel: bool = False,
    n_jobs: Optional[int] = None,
    progressbar: bool = False,
    keep_diss: bool = False,
    max_dist: Optional[float] = None,
    random_state: Optional[int] = None,
    export_membership: bool = False,
    replay_samples: Optional[np.ndarray] = None,
    replay_initial_memberships: Optional[Mapping[int, Sequence[object]]] = None,
) -> Dict[str, object]:
    """Run seqclararange for each criterion and assemble a family object.

    The five WeightedCluster 2.0 criteria mirror R's seqclararange family
    behavior, and the additional wcClusterQuality criteria are Python-only
    extensions.
    Selection minimizes distance, DB, XB, and HC and maximizes PBM, AMS, PBC,
    HG, HGSD, ASW, ASWw, CH, R2, CHsq, and R2sq, keeping HC aligned with
    clustrange and clustrangefamily.
    """
    crits = _normalize_criteria(
        criteria,
        default=("distance", "db", "xb", "pbm", "ams"),
        allowed=_ALLOWED_SEQCLARA_CRITERIA,
    )
    method_resolved = _partial_match(
        method,
        (*_R_SEQCLARA_METHODS, "hncdd", "pcmdd"),
        name="method",
    )
    context = _build_seqclararange_context(
        seqdata,
        R=R,
        sample_size=sample_size,
        kvals=kvals,
        seqdist_args=seqdist_args,
        method=method_resolved,
        criteria_norm=crits,
        stability=stability,
        dnoise=dnoise,
        m=m,
        dlambda=dlambda,
        eta=eta,
        pam_n_init=pam_n_init,
        parallel=parallel,
        n_jobs=n_jobs,
        progressbar=progressbar,
        keep_diss=keep_diss,
        max_dist=max_dist,
        random_state=random_state,
        export_membership=export_membership,
        replay_samples=replay_samples,
        replay_initial_memberships=replay_initial_memberships,
    )
    runs_by_k = _run_seqclara_cache(context)
    return _assemble_seqclararange_family_from_cache(context, runs_by_k)


def seqclararange(
    seqdata: Any,
    *,
    R: int = 100,
    sample_size: Optional[int] = None,
    kvals: Sequence[int] = tuple(range(2, 11)),
    seqdist_args: Optional[Dict] = None,
    method: str = "crisp",
    criteria: Sequence[str] | str | None = ("distance",),
    stability: bool = False,
    dnoise: Optional[float] = None,
    m: float = 1.5,
    dlambda: Optional[float] = None,
    eta: Optional[np.ndarray] = None,
    pam_n_init: int = 2,
    parallel: bool = False,
    n_jobs: Optional[int] = None,
    progressbar: bool = False,
    keep_diss: bool = False,
    max_dist: Optional[float] = None,
    random_state: Optional[int] = None,
    export_membership: bool = False,
    replay_samples: Optional[np.ndarray] = None,
    replay_initial_memberships: Optional[Mapping[int, Sequence[object]]] = None,
) -> SeqClaraRangeResult | Dict[str, object]:
    """CLARA-like clustering for sequences across K (crisp, fuzzy, noise, PCMdd).

    Parameters
    - seqdata: (N, T) integer-coded sequences
    - R: number of subsamples (iterations)
    - sample_size: subsample size per iteration; R default 40 + 2*max(K), sampled with replacement
    - kvals: iterable of K values to evaluate
    - seqdist_args: distance config passed to sequence-distance backend (R default: {'method': 'LCS'})
    - method: 'crisp' | 'representativeness' | 'fuzzy' | 'noise' | 'hncdd' | 'pcmdd'
    - criteria: primary selection criterion. The five WeightedCluster 2.0
      criteria are distance, DB, XB, PBM, and AMS. The Python-only choices are
      PBC, HG, HGSD, ASW, ASWw, CH, R2, CHsq, R2sq, and HC. Selection minimizes
      distance, DB, XB, and HC and maximizes PBM, AMS, PBC, HG, HGSD, ASW, ASWw,
      CH, R2, CHsq, and R2sq, keeping HC aligned with clustrange and
      clustrangefamily.
    - parallel: evaluate CLARA iterations concurrently when True
    - n_jobs: optional worker count for parallel execution; shared semantics with
      bootclustrange and seqnullcqi
    - random_state: base seed; per-iteration seeds are derived as base + iteration

    Weighted subsampling uses normalized aggregate weights as probabilities,
    while selection criteria use the unnormalized aggregate analysis weights.
    This intentionally corrects WeightedCluster 2.0's reuse of probabilities
    for weighted criteria, which changes scale-dependent objectives.
    Fuzzy full-data membership uses the exact positive ``m - 1`` exponent;
    every representable finite ``m > 1`` accepted by validation retains that
    value rather than being replaced by an absolute floor.

    Returns a dict {'kvals','clustering','stats','quality_stats','clara'} where:
    - clustering contains best 1-based labels for each K
    - stats follows R seqclararange core columns (Avg dist, PBM, DB, XB, AMS, ARI>0.8, JC>0.8, Best iter)
    - quality_stats stores extended Python diagnostics from wcClusterQuality and,
      for fuzzy/PCMdd runs, fuzzy-only metrics such as FPC/FPE/XBf

    Example
    >>> import numpy as np
    >>> from weightedcluster import seqclararange
    >>> X = np.array([[0,0,0,0],[0,0,0,1],[1,1,1,1],[1,1,1,0]], dtype=int)
    >>> out = seqclararange(X, R=5, sample_size=3, kvals=[2], seqdist_args={"method":"HAM"}, random_state=7)
    >>> out['kvals']
    [2]
    """
    criteria_norm = _normalize_criteria(
        criteria, default=("distance",), allowed=_ALLOWED_SEQCLARA_CRITERIA
    )
    method_resolved = _partial_match(
        method,
        (*_R_SEQCLARA_METHODS, "hncdd", "pcmdd"),
        name="method",
    )
    context = _build_seqclararange_context(
        seqdata,
        R=R,
        sample_size=sample_size,
        kvals=kvals,
        seqdist_args=seqdist_args,
        method=method_resolved,
        criteria_norm=criteria_norm,
        stability=stability,
        dnoise=dnoise,
        m=m,
        dlambda=dlambda,
        eta=eta,
        pam_n_init=pam_n_init,
        parallel=parallel,
        n_jobs=n_jobs,
        progressbar=progressbar,
        keep_diss=keep_diss,
        max_dist=max_dist,
        random_state=random_state,
        export_membership=export_membership,
        replay_samples=replay_samples,
        replay_initial_memberships=replay_initial_memberships,
    )
    runs_by_k = _run_seqclara_cache(context)
    if len(criteria_norm) > 1:
        return _assemble_seqclararange_family_from_cache(context, runs_by_k)
    return _assemble_seqclararange_result_from_cache(context, runs_by_k, primary=criteria_norm[0])

from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as _fast_linkage  # type: ignore
from scipy.spatial.distance import squareform  # type: ignore

from ._cmpcluster_algorithms import _cluster_diana, _embed_cmds
from ._cmpcluster_controls import (
    _canonical_cmpcluster_method,
    _normalize_cmpcluster_methods,
    _validate_cmpcluster_replay_starts,
)
from ._cmpcluster_result import _R_QUALITY_NAMES, _build_clustrange_family_result
from ._fastcluster_linkage import linkage_fastcluster
from ._validation import validate_integer_control, validate_kvals
from .kmedoids import wcKMedoids, wcKMedRange
from .range import _coerce_range_distance, _coerce_range_weights, as_clustrange
from .schemas import (
    ClusteringTable,
    ClustrangeFamilyResult,
    KMedRangeResult,
    rows_to_stats_table,
)

# `_fast_linkage` is SciPy's linkage and serves only the optional
# coordinate-embedding route below, which clusters Torgerson/Gower coordinates
# rather than a precomputed dissimilarity matrix.
#
# It used to be selected by `_select_linkage_backend`, which preferred an
# optional Python `fastcluster` when one was importable. Finding F-11 of the
# 2026-08-24 whole-codebase audit removed that branch: `fastcluster` appears in
# no `pyproject.toml` dependency or extra, so no documented installation reached
# it; its numerics were never exercised, only its selector logic against an
# injected fake; and a user who happened to have the package installed silently
# got a different code path. Every dissimilarity-matrix route now goes through
# `_linkage_hclust`, which reproduces `fastcluster::hclust` exactly in Python and
# needs no optional package to do so.


def _linkage_hclust(
    D: np.ndarray,
    *,
    method: str,
    members: np.ndarray | None = None,
) -> np.ndarray:
    """Build the hierarchy R's ``wcCmpCluster`` builds, for one hclust method.

    ``orig_R_pkg/WeightedCluster/NAMESPACE`` binds ``hclust`` to
    **fastcluster's** implementation, and ``R/clustrangefamily.R:27`` calls that
    binding, so this delegates to the fastcluster port in
    :mod:`weightedcluster._fastcluster_linkage`.

    Until the 2026-08-24 whole-codebase audit (finding F-01) this function was a
    port of ``stats::hclust`` instead. The two implementations break
    equal-distance merges differently, and for ``centroid`` they disagree on the
    merge heights as well, so every route through here returned a different
    partition from R whenever the dissimilarities contained ties — which, for
    integer-valued sequence dissimilarities, is the ordinary case.
    """

    return linkage_fastcluster(D, method=method, members=members)


def _pam_range_from_memberships(
    D: np.ndarray,
    source_range: Mapping[str, object],
    *,
    weights: np.ndarray | None,
) -> KMedRangeResult:
    """Run the hierarchy-initialized PAM branch for per-k memberships."""

    kvals = [int(k) for k in source_range["kvals"]]
    clustering: dict[str, np.ndarray] = {}
    rows: list[dict[str, float]] = []
    for k in kvals:
        result = wcKMedoids(
            D,
            k,
            weights=weights,
            initialclust=source_range["clustering"][f"cluster{k}"],
        )
        clustering[f"cluster{k}"] = np.asarray(
            result["clustering"],
            dtype=int,
        )
        rows.append(
            {stat_name: float(result["stats"][stat_name]) for stat_name in _R_QUALITY_NAMES}
        )
    row_names = [f"cluster{k}" for k in kvals]
    return KMedRangeResult(
        {
            "R": 1,
            "kvals": kvals,
            "clustering": ClusteringTable(
                clustering,
                column_names=row_names,
            ),
            "stats": rows_to_stats_table(
                rows,
                keys=_R_QUALITY_NAMES,
                row_names=row_names,
            ),
        }
    )


def wcCmpCluster(
    diss: np.ndarray,
    *,
    weights: Optional[Sequence[float]] = None,
    maxcluster: int,
    method: Sequence[str] | str = "all",
    pam_combine: bool = True,
    embed_then_ward: bool = False,
    random_state: Optional[int] = None,
    replay_starts: Mapping[int, Sequence[int] | np.ndarray] | None = None,
) -> ClustrangeFamilyResult:
    """Compare R-compatible clustering families and quality trajectories.

    - diss: square or condensed distance data
    - weights: optional (N,) weights
    - maxcluster: maximum K (solutions from 2..maxcluster), which cannot exceed N
    - method: R's eight hclust names, PAM, DIANA, flexible-beta, or ``all``
    - pam_combine: initialize a PAM range from every non-PAM hierarchy
    - embed_then_ward: if True and method includes 'ward', performs classical MDS (Torgerson/Gower)
      after normalizing distances before squaring, retains a scale-relative
      positive spectrum, normalizes coordinates for Ward, and restores linkage
      heights (approximation mode). If False, uses exact ward.D2 Lance-Williams
      updates directly on the precomputed distance matrix.
    - random_state: optional base seed used by stochastic PAM branches.
    - replay_starts: explicit zero-based standalone-PAM medoids by K.

    Returns a typed, dict-compatible ``clustrangefamily`` mapping.

    Flexible-beta chooses the exact smallest dissimilarity. Only exactly equal
    candidates use deterministic pair ordering; no absolute near-tie tolerance
    changes the hierarchy under a common positive distance scale.
    """
    D = _coerce_range_distance(diss)
    w = _coerce_range_weights(weights, D.shape[0])
    maxcluster_int = validate_integer_control(maxcluster, name="maxcluster")
    if maxcluster_int < 2:
        raise ValueError("maxcluster should be greater than 2")
    validate_kvals(
        (maxcluster_int,),
        max_k=D.shape[0],
    )
    kvals = list(range(2, maxcluster_int + 1))
    if not isinstance(pam_combine, (bool, np.bool_)):
        raise TypeError("pam_combine must be a boolean")
    if not isinstance(embed_then_ward, (bool, np.bool_)):
        raise TypeError("embed_then_ward must be a boolean")
    if random_state is not None:
        random_state = validate_integer_control(random_state, name="random_state")
        if random_state < 0:
            raise ValueError("random_state must be non-negative")
    methods = _normalize_cmpcluster_methods(method, has_weights=w is not None)

    out: dict[str, object] = {}
    pam_replay = _validate_cmpcluster_replay_starts(
        replay_starts,
        maxcluster=maxcluster_int,
        n_obs=D.shape[0],
    )

    for method_name in methods:
        algorithm_name = _canonical_cmpcluster_method(method_name)
        if algorithm_name == "pam":
            out[method_name] = wcKMedRange(
                D,
                kvals,
                weights=w,
                npass=0 if pam_replay is not None else 1,
                random_state=random_state,
                replay_starts=pam_replay,
            )
            continue

        if algorithm_name.startswith("ward."):
            if embed_then_ward:
                coordinates = _embed_cmds(D)
                coordinate_scale = float(np.max(np.abs(coordinates)))
                linkage_coordinates = (
                    coordinates / coordinate_scale if coordinate_scale > 0.0 else coordinates
                )
                hierarchy = _fast_linkage(
                    linkage_coordinates,
                    method="ward",
                    metric="euclidean",
                )
                hierarchy[:, 2] *= coordinate_scale
            else:
                hierarchy = _linkage_hclust(
                    D,
                    method=algorithm_name,
                    members=w,
                )
        elif algorithm_name == "diana":
            labels_by_k = _cluster_diana(D, maxcluster_int)
            hierarchy_columns = {f"cluster{k}": labels_by_k[k] for k in kvals if k in labels_by_k}
            diana_range = as_clustrange(
                hierarchy_columns,
                D,
                weights=w,
            )
            out[method_name] = diana_range
            if pam_combine:
                out[f"pam.{method_name}"] = _pam_range_from_memberships(
                    D,
                    diana_range,
                    weights=w,
                )
            continue
        elif algorithm_name == "beta.flexible":
            if D.shape[0] > 500:
                warnings.warn(
                    f"Flexible-beta linkage has O(N³) complexity. "
                    f"With N={D.shape[0]}, this may be very slow. "
                    f"Consider using fastcluster-compatible methods or reducing data size.",
                    UserWarning,
                    stacklevel=2,
                )
            from ._constants import FLEXIBLE_ALPHA_DEFAULT

            hierarchy = _linkage_flexible_beta(
                squareform(D, checks=False),
                alpha=FLEXIBLE_ALPHA_DEFAULT,
            )
        else:
            # Every hclust method takes the same route, weighted or not. R's
            # `wcCmpCluster` calls one `hclust` for all eight, and splitting
            # them across two implementations is what let finding F-01 hide:
            # the methods this branch used to send to SciPy happened to agree
            # with R, while the rest did not.
            hierarchy = _linkage_hclust(
                D,
                method=algorithm_name,
                members=w,
            )

        out[method_name] = as_clustrange(
            hierarchy,
            D,
            weights=w,
            ncluster=maxcluster_int,
        )
        if pam_combine:
            out[f"pam.{method_name}"] = wcKMedRange(
                D,
                kvals,
                weights=w,
                initialclust=hierarchy,
                random_state=random_state,
            )

    return _build_clustrange_family_result(
        out,
        methods=methods,
        pam_combine=pam_combine,
        kvals=kvals,
        embed_then_ward=embed_then_ward,
        random_state=random_state,
    )


def summary_clustrangefamily(
    object: Mapping[str, object],
    *,
    max_rank: int = 1,
) -> pd.DataFrame:
    """Return R's method-and-cluster ranking table for a family."""

    allstats = object["allstats"]
    rank = validate_integer_control(max_rank, name="max_rank")
    row_count = len(allstats["method"])
    if rank < 1 or rank > row_count:
        raise ValueError("max_rank must be between one and the number of family rows")
    columns: list[str] = []
    for index in range(1, rank + 1):
        columns.extend([f"{index}. Cluster", f"{index}.  stat"])
    rows: list[list[str]] = []
    for stat_name in _R_QUALITY_NAMES:
        values = np.asarray(allstats[stat_name], dtype=float)
        order = (
            np.argsort(values, kind="stable")
            if stat_name == "HC"
            else np.argsort(-values, kind="stable")
        )
        row: list[str] = []
        for position in order[:rank]:
            row.extend(
                [
                    f"{allstats['method'][position]} {int(allstats['ngroup'][position])}",
                    format(float(values[position]), ".15g"),
                ]
            )
        rows.append(row)
    return pd.DataFrame(rows, index=list(_R_QUALITY_NAMES), columns=columns)


def print_clustrangefamily(
    object: Mapping[str, object],
    *,
    max_rank: int = 1,
) -> str:
    """Return the table text printed by R's ``print.clustrangefamily``."""

    table = summary_clustrangefamily(object, max_rank=max_rank)
    index_values = [str(value) for value in table.index]
    index_width = max(len(value) for value in index_values)
    column_values = [
        [str(value) for value in table.iloc[:, index].tolist()] for index in range(table.shape[1])
    ]
    widths = [
        max(len(str(table.columns[index])), *(len(value) for value in values))
        for index, values in enumerate(column_values)
    ]
    lines = [
        " " * (index_width + 1)
        + " ".join(
            str(name).rjust(width) for name, width in zip(table.columns, widths, strict=True)
        )
    ]
    for row_index, index_value in enumerate(index_values):
        lines.append(
            index_value.ljust(index_width)
            + " "
            + " ".join(
                column_values[column][row_index].rjust(widths[column])
                for column in range(table.shape[1])
            )
        )
    return "\n".join(lines)


def _linkage_flexible_beta(y: np.ndarray, alpha: float = 0.625) -> np.ndarray:
    """Build a SciPy-like linkage matrix using flexible-beta Lance-Williams update.

    y: condensed distance vector (len = n*(n-1)/2)
    Returns (n-1,4) Z matrix.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    m = int(y.size)
    n = int((1 + np.sqrt(1 + 8 * m)) / 2)
    if n * (n - 1) // 2 != m:
        raise ValueError("Invalid condensed distance vector length")
    if n < 2:
        return np.zeros((0, 4), dtype=float)

    D0 = squareform(y, force="tomatrix", checks=False)
    max_nodes = 2 * n - 1
    dist = np.full((max_nodes, max_nodes), np.nan, dtype=float)
    dist[:n, :n] = D0
    np.fill_diagonal(dist[:n, :n], np.nan)

    sizes = np.zeros(max_nodes, dtype=float)
    sizes[:n] = 1.0
    active = list(range(n))
    Z = np.zeros((n - 1, 4), dtype=float)

    # ``cluster::agnes(method="flexible", par.method=a)`` expands a scalar
    # parameter to (a, a, 1 - 2a, 0).
    alpha = float(alpha)
    beta_coefficient = 1.0 - 2.0 * alpha
    gamma = 0.0

    for step in range(n - 1):
        best_pair = None
        best_d = np.inf
        for ai, i in enumerate(active[:-1]):
            for j in active[ai + 1 :]:
                dij = float(dist[i, j])
                if not np.isfinite(dij):
                    continue
                pair = (i, j) if i < j else (j, i)
                if dij < best_d or (dij == best_d and best_pair is not None and pair < best_pair):
                    best_d = dij
                    best_pair = pair

        if best_pair is None:
            raise ValueError("flexible-beta linkage failed: no active pair found")

        i, j = best_pair
        new_id = n + step
        si = float(sizes[i])
        sj = float(sizes[j])
        sij = si + sj
        Z[step, 0] = float(i)
        Z[step, 1] = float(j)
        Z[step, 2] = float(best_d)
        Z[step, 3] = float(sij)

        for c in active:
            if c == i or c == j:
                continue
            dic = float(dist[i, c])
            djc = float(dist[j, c])
            dnew = alpha * dic + alpha * djc + beta_coefficient * best_d + gamma * abs(dic - djc)
            dist[new_id, c] = dist[c, new_id] = dnew

        sizes[new_id] = sij
        active = [a for a in active if a not in (i, j)]
        active.append(new_id)

    return Z

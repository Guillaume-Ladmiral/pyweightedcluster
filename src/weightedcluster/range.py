from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._validation import (
    coerce_public_distance_matrix,
    cut_linkage_by_merge_count,
    validate_integer_control,
    validate_linkage_matrix,
)
from .quality import _wc_cluster_quality_internal, wcClusterQuality
from .schemas import (
    ClusteringTable,
    ClustrangeResult,
    LabeledStatsTable,
    rows_to_stats_table,
)

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


def _coerce_range_distance(diss: np.ndarray) -> np.ndarray:
    """Apply the shared public distance contract (findings F-06, F-07).

    This used to reject an asymmetric matrix, a non-zero diagonal and negative
    entries, all of which R computes from — and `wcClusterQuality` already
    accepted the negative case and agreed with R exactly, so the two public
    entry points disagreed with each other.
    """
    return coerce_public_distance_matrix(diss, name="diss")


def _coerce_range_weights(
    weights: Optional[Sequence[float] | np.ndarray],
    n: int,
) -> np.ndarray | None:
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size != n:
        raise ValueError(f"weights should be a vector of length {n}")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must contain finite values")
    return values.copy()


def _predictor_columns(
    object: object,
    *,
    n_obs: int,
) -> list[np.ndarray] | None:
    if isinstance(object, pd.DataFrame):
        columns = [
            (
                column.array.copy()
                if isinstance(column.dtype, pd.CategoricalDtype)
                else column.to_numpy(copy=True)
            )
            for _, column in object.items()
        ]
    elif isinstance(object, Mapping) and "class" not in object:
        columns = [np.asarray(column).reshape(-1).copy() for column in object.values()]
    else:
        try:
            values = np.asarray(object)
        except (TypeError, ValueError):
            return None
        if values.ndim != 2 or values.shape[1] == 4 and values.shape[0] == n_obs - 1:
            return None
        columns = [values[:, index].copy() for index in range(values.shape[1])]
    if not columns:
        raise ValueError("object must contain at least one clustering column")
    for column in columns:
        if len(column) != n_obs:
            raise ValueError(
                "different number of elements in diss, clustering and/or weights arguments"
            )
    return columns


def _unique_count(column: Sequence[object] | np.ndarray) -> int:
    if isinstance(column, pd.Categorical):
        return int(pd.unique(np.asarray(column)).size)
    return int(pd.unique(np.asarray(column, dtype=object)).size)


def _bootstrap_plan(
    *,
    n: int,
    weights: np.ndarray | None,
    R: int,
    samplesize: int | None,
    random_state: int | None,
    replay_samples: Sequence[Sequence[int]] | np.ndarray | None,
) -> np.ndarray:
    sampling_weights = np.ones(n, dtype=float) if weights is None else weights
    total = float(np.sum(sampling_weights))
    size = (
        int(np.floor(total))
        if samplesize is None
        else validate_integer_control(samplesize, name="samplesize")
    )
    if size < 1:
        raise ValueError("samplesize must be at least one")
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("bootstrap sampling requires a positive finite total weight")
    probabilities = sampling_weights / total
    if np.any(probabilities < 0.0):
        raise ValueError("bootstrap sampling probabilities must be non-negative")
    if replay_samples is not None:
        plan = np.asarray(replay_samples)
        if plan.shape != (R, size):
            raise ValueError(f"replay_samples must have shape ({R}, {size})")
        if not np.issubdtype(plan.dtype, np.integer):
            raise ValueError("replay_samples must contain zero-based integer indices")
        plan = plan.astype(int)
        if np.any(plan < 0) or np.any(plan >= n):
            raise ValueError("replay_samples contains an out-of-range zero-based index")
        return plan
    rng = np.random.default_rng(random_state)
    return np.asarray(
        [rng.choice(n, size=size, replace=True, p=probabilities) for _ in range(R)],
        dtype=int,
    )


def _bootstrap_clustrange(
    D: np.ndarray,
    columns: Sequence[Sequence[object] | np.ndarray],
    stats: LabeledStatsTable,
    *,
    weights: np.ndarray | None,
    R: int,
    samplesize: int | None,
    random_state: int | None,
    replay_samples: Sequence[Sequence[int]] | np.ndarray | None,
) -> tuple[
    dict[int, dict[str, list[float]]],
    LabeledStatsTable,
    LabeledStatsTable,
    LabeledStatsTable,
]:
    plan = _bootstrap_plan(
        n=D.shape[0],
        weights=weights,
        R=R,
        samplesize=samplesize,
        random_state=random_state,
        replay_samples=replay_samples,
    )
    boot: dict[int, dict[str, list[float]]] = {}
    mean_rows: list[dict[str, float]] = []
    stderr_rows: list[dict[str, float]] = []
    t0_rows: list[dict[str, float]] = []
    row_names = list(stats.row_names)
    for row_name, labels in zip(row_names, columns, strict=True):
        t0_quality = _wc_cluster_quality_internal(D, labels, weights=weights)
        t0_row = {name: float(t0_quality["stats"][name]) for name in _R_QUALITY_NAMES}
        t0_rows.append(t0_row)
        block = {name: [t0_row[name]] for name in _R_QUALITY_NAMES}
        for sample in plan:
            counts = np.bincount(sample, minlength=D.shape[0]).astype(float)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="ASW can not be computed because at least one cluster",
                    category=UserWarning,
                )
                quality = _wc_cluster_quality_internal(D, labels, weights=counts)
            for name in _R_QUALITY_NAMES:
                block[name].append(float(quality["stats"][name]))
        key = int(row_name.removeprefix("cluster"))
        boot[key] = block
        mean_rows.append(
            {
                name: float(np.mean(np.asarray(block[name], dtype=float)))
                for name in _R_QUALITY_NAMES
            }
        )
        stderr_rows.append(
            {
                name: float(np.std(np.asarray(block[name], dtype=float), ddof=1))
                for name in _R_QUALITY_NAMES
            }
        )
    return (
        boot,
        rows_to_stats_table(t0_rows, keys=_R_QUALITY_NAMES, row_names=row_names),
        rows_to_stats_table(mean_rows, keys=_R_QUALITY_NAMES, row_names=row_names),
        rows_to_stats_table(stderr_rows, keys=_R_QUALITY_NAMES, row_names=row_names),
    )


def as_clustrange(
    object,
    diss: np.ndarray,
    weights: Optional[np.ndarray] = None,
    R: int = 1,
    samplesize: Optional[int] = None,
    ncluster: int = 20,
    *,
    labels: bool = True,
    random_state: int | None = None,
    replay_samples: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> ClustrangeResult:
    """Translate predictor, hierarchy, or property-tree clusterings to ``clustrange``.

    The returned :class:`~weightedcluster.schemas.ClustrangeResult` is a typed
    ``dict`` subclass. Existing key access is unchanged, while ``r_class`` and
    table ``row_names``/``column_names`` retain R's class and label metadata.
    """
    D = _coerce_range_distance(diss)
    w = _coerce_range_weights(weights, D.shape[0])
    columns = _predictor_columns(object, n_obs=D.shape[0])

    if columns is None and isinstance(object, Mapping) and "dtclust" in object.get("class", []):
        from .proptree import dtcut

        ncluster_int = validate_integer_control(ncluster, name="ncluster")
        if ncluster_int < 3:
            raise ValueError("ncluster should be greater than 2")
        fitted = np.asarray(object["fitted"])
        max_k = int(np.unique(fitted[:, 0] if fitted.ndim == 2 else fitted).size)
        if ncluster_int > max_k:
            raise ValueError(f"ncluster should be less than {max_k + 1}")
        columns = [
            np.asarray(dtcut(object, k, labels=bool(labels))).reshape(-1).copy()
            for k in range(2, ncluster_int + 1)
        ]
    elif columns is None:
        try:
            Z = np.asarray(object, dtype=float)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError(
                "as.clustrange supports predictor frames, hierarchy linkages, or dtclust trees"
            ) from exc
        if Z.ndim != 2 or Z.shape[1] != 4:
            raise NotImplementedError(
                "as.clustrange supports predictor frames, hierarchy linkages, or dtclust trees"
            )
        validate_linkage_matrix(Z, D.shape[0], name="linkage matrix")
        ncluster_int = validate_integer_control(ncluster, name="ncluster")
        if ncluster_int < 3:
            raise ValueError("ncluster should be greater than 2")
        if ncluster_int > D.shape[0]:
            raise ValueError(f"ncluster should be less than {D.shape[0]}")
        columns = [cut_linkage_by_merge_count(Z, n_clusters=k) for k in range(2, ncluster_int + 1)]

    kvals = [_unique_count(column) for column in columns]
    if len(set(kvals)) != len(kvals):
        raise ValueError("duplicate 'row.names' are not allowed")
    column_names = [f"cluster{k}" for k in kvals]
    factor_levels = {
        name: list(column.categories)
        for name, column in zip(column_names, columns, strict=True)
        if isinstance(column, pd.Categorical)
    }
    clustering = ClusteringTable(
        dict(zip(column_names, columns, strict=True)),
        column_names=column_names,
        factor_levels=factor_levels,
    )
    stats_rows = []
    for column in columns:
        quality = wcClusterQuality(D, column, weights=w)
        stats_rows.append({name: float(quality["stats"][name]) for name in _R_QUALITY_NAMES})
    stats = rows_to_stats_table(
        stats_rows,
        keys=_R_QUALITY_NAMES,
        row_names=column_names,
    )
    result = ClustrangeResult(
        {
            "clustering": clustering,
            "kvals": kvals,
            "stats": stats,
        }
    )
    R_int = validate_integer_control(R, name="R")
    if R_int < 1:
        raise ValueError("R must be at least one")
    if R_int > 1:
        boot, stats, meant, stderr = _bootstrap_clustrange(
            D,
            columns,
            stats,
            weights=w,
            R=R_int,
            samplesize=samplesize,
            random_state=random_state,
            replay_samples=replay_samples,
        )
        result["stats"] = stats
        result["boot"] = boot
        result["meant"] = meant
        result["stderr"] = stderr
    return result


def summary_clustrange(
    object: Mapping[str, Any],
    *,
    max_rank: int = 1,
) -> pd.DataFrame:
    """Return R's best-cluster ranking table (``HC`` is minimized)."""
    kvals = np.asarray(object["kvals"], dtype=int)
    stats = object["stats"]
    rank = validate_integer_control(max_rank, name="max_rank")
    if rank < 1 or rank > len(kvals):
        raise ValueError("max_rank must be between one and the number of clusterings")
    columns: list[str] = []
    for index in range(1, rank + 1):
        columns.extend([f"{index}. N groups", f"{index}.  stat"])
    rows: list[list[float]] = []
    for name, values in stats.items():
        numeric = np.asarray(values, dtype=float)
        order = (
            np.argsort(numeric, kind="stable")
            if str(name) == "HC"
            else np.argsort(-numeric, kind="stable")
        )
        selected = order[:rank]
        row: list[float] = []
        for position in selected:
            row.extend([float(kvals[position]), float(numeric[position])])
        rows.append(row)
    return pd.DataFrame(rows, index=list(stats), columns=columns)


def print_clustrange(
    object: Mapping[str, Any],
    *,
    digits: int = 2,
    bootstat: str | Sequence[str] = ("t0", "mean", "stderr"),
) -> str:
    """Return the semantic table printed by WeightedCluster ``print.clustrange``."""
    if isinstance(bootstat, str):
        selection = bootstat
    else:
        choices = list(bootstat)
        if "boot" in object and len(choices) != 1:
            raise ValueError("'length = 3' in coercion to 'logical(1)'")
        selection = choices[0] if choices else "t0"
    if selection not in {"t0", "mean", "stderr"}:
        raise ValueError("bootstat must be one of 't0', 'mean', or 'stderr'")
    if "boot" in object and selection != "t0":
        field = "meant" if selection == "mean" else "stderr"
        if field not in object:
            raise ValueError(f"clustrange has no {field} table")
        table = object[field]
    else:
        table = object["stats"]
    frame = pd.DataFrame(table, index=getattr(table, "row_names", None))
    return frame.round(validate_integer_control(digits, name="digits")).to_string()

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict, cast

import numpy as np

from ._validation import validate_integer_control, validate_kvals


def _unique_string_names(values: Sequence[object], *, name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a string")
    names = [str(value) for value in values]
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must not contain duplicates")
    return names


def _validate_labeled_columns(
    columns: Mapping[str, Sequence[object]],
    *,
    row_names: Sequence[str],
    name: str,
) -> None:
    lengths = {len(column) for column in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"{name} columns must all have the same length")
    if row_names and lengths and len(row_names) != next(iter(lengths)):
        raise ValueError(f"{name} row_names length must match the column length")


# Canonical table aliases used by range-like return objects.
class LabeledStatsTable(dict[str, List[float]]):
    """Column-oriented numeric table carrying R-compatible row/column labels."""

    def __init__(
        self,
        values: Mapping[str, Sequence[float]],
        *,
        row_names: Sequence[str] = (),
    ) -> None:
        column_names = _unique_string_names(list(values), name="column names")
        columns = {
            name: [float(value) for value in column]
            for name, column in zip(column_names, values.values(), strict=True)
        }
        resolved_rows = _unique_string_names(row_names, name="row_names")
        _validate_labeled_columns(
            columns,
            row_names=resolved_rows,
            name="LabeledStatsTable",
        )
        super().__init__(columns)
        self.row_names = resolved_rows
        self.column_names = list(self)


class LabeledObjectTable(dict[str, List[object]]):
    """Column-oriented mixed table carrying R-compatible labels."""

    def __init__(
        self,
        values: Mapping[str, Sequence[object]],
        *,
        row_names: Sequence[str] = (),
    ) -> None:
        column_names = _unique_string_names(list(values), name="column names")
        columns = {
            name: [value for value in column]
            for name, column in zip(column_names, values.values(), strict=True)
        }
        resolved_rows = _unique_string_names(row_names, name="row_names")
        _validate_labeled_columns(
            columns,
            row_names=resolved_rows,
            name="LabeledObjectTable",
        )
        super().__init__(columns)
        self.row_names = resolved_rows
        self.column_names = list(self)


class ClusteringTable(dict[str, np.ndarray]):
    """Named clustering columns with explicit positional column metadata."""

    def __init__(
        self,
        values: Mapping[str, Sequence[object] | np.ndarray],
        *,
        column_names: Sequence[str] | None = None,
        factor_levels: Mapping[str, Sequence[object]] | None = None,
    ) -> None:
        value_names = _unique_string_names(list(values), name="clustering column names")
        columns: dict[str, np.ndarray] = {}
        for name, column in zip(value_names, values.values(), strict=True):
            array = np.asarray(column)
            if array.ndim != 1:
                raise ValueError(f"clustering[{name!r}] must be one-dimensional")
            columns[name] = array.copy()
        lengths = {int(column.size) for column in columns.values()}
        if len(lengths) > 1:
            raise ValueError("clustering columns must all have the same length")

        resolved_columns = (
            _unique_string_names(column_names, name="column_names")
            if column_names is not None
            else list(columns)
        )
        if len(resolved_columns) != len(columns) or set(resolved_columns) != set(columns):
            raise ValueError("column_names must name every clustering column exactly once")

        raw_factor_levels = factor_levels or {}
        factor_names = _unique_string_names(list(raw_factor_levels), name="factor_levels keys")
        if not set(factor_names).issubset(columns):
            raise ValueError("factor_levels keys must refer to clustering columns")
        resolved_factor_levels = {
            name: [value for value in levels]
            for name, levels in zip(factor_names, raw_factor_levels.values(), strict=True)
        }

        super().__init__(columns)
        self.column_names = resolved_columns
        self.factor_levels = resolved_factor_levels


StatsTable = Dict[str, List[float]]
ObjectTable = Dict[str, List[object]]
ClusteringMap = Dict[str, np.ndarray]
BootByKStats = Dict[int, Dict[str, List[float]]]


class ClustrangeResult(dict[str, Any]):
    """R-compatible typed mapping while retaining historical ``result[key]`` access."""

    def __init__(
        self,
        values: Mapping[str, object],
        *,
        r_class: Sequence[str] = ("clustrange", "list"),
    ) -> None:
        super().__init__(values)
        self.r_class = [str(value) for value in r_class]


class KMedoidsResult(TypedDict):
    """Return payload for wcKMedoids(cluster_only=False)."""

    clustering: np.ndarray
    info: Dict[str, float]
    stats: Dict[str, float]
    ASW: Dict[int, float]
    ASWw: Dict[int, float]


class KMedRangeResult(ClustrangeResult):
    """Typed ``clustrange`` mapping returned by :func:`wcKMedRange`."""


class ClustrangeFamilyResult(dict[str, Any]):
    """Typed ``clustrangefamily`` mapping retaining historical key access."""

    def __init__(
        self,
        values: Mapping[str, object],
        *,
        r_class: Sequence[str] = ("clustrangefamily",),
    ) -> None:
        super().__init__(values)
        self.r_class = [str(value) for value in r_class]


class ClustrangeFamilyParam(dict[str, object]):
    """Exact R parameters with non-R compatibility metadata off-table."""

    def __init__(
        self,
        values: Mapping[str, object],
        *,
        additions: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(values)
        self.additions = dict(additions or {})

    def copy(self) -> "ClustrangeFamilyParam":
        """Copy the exact R mapping and its separate Python metadata."""
        return type(self)(self, additions=self.additions)


class BootClustrangeResult(dict[str, Any]):
    """Exact R ``bootclustrange`` fields with Python diagnostics off-table."""

    def __init__(
        self,
        values: Mapping[str, object],
        *,
        additions: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(values)
        self.r_class = ["bootclustrange", "clustrange", "list"]
        self.additions = dict(additions or {})

    def copy(self) -> "BootClustrangeResult":
        """Copy the exact R mapping and its separate Python metadata."""
        return type(self)(self, additions=self.additions)


class ClustassocResult(dict[str, Any]):
    """Exact R ``clustassoc`` table with Python diagnostics off-table."""

    def __init__(
        self,
        values: Mapping[str, object],
        *,
        row_names: Sequence[str],
        additions: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(values)
        self.r_class = ["clustassoc", "data.frame"]
        self.row_names = [str(value) for value in row_names]
        self.column_names = list(self)
        self.additions = dict(additions or {})

    def copy(self) -> "ClustassocResult":
        """Copy the exact R table and its separate Python metadata."""
        return type(self)(self, row_names=self.row_names, additions=self.additions)


class SeqClaraResult(dict[str, Any]):
    """R-compatible ``seqclara`` best-iteration payload."""

    def __init__(self, values: Mapping[str, object]) -> None:
        super().__init__(values)
        self.r_class = ["seqclara"]


class SeqClaraRangeResult(ClustrangeResult):
    """R-compatible ``seqclararange``/``clustrange`` result mapping."""

    def __init__(self, values: Mapping[str, object]) -> None:
        super().__init__(values, r_class=("seqclararange", "clustrange"))


class SeqClaraRangeFamilyResult(ClustrangeFamilyResult):
    """R-compatible multiple-criterion CLARA family mapping."""

    def __init__(self, values: Mapping[str, object]) -> None:
        super().__init__(values, r_class=("seqclarafamily", "clustrangefamily"))


class SeqNullCQIResult(dict[str, Any]):
    """Exact R ``seqnullcqi`` fields with Python conveniences off-table."""

    def __init__(
        self,
        values: Mapping[str, object],
        *,
        additions: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(values)
        self.r_class = ["seqnullcqi"]
        self.additions = dict(additions or {})

    def copy(self) -> "SeqNullCQIResult":
        """Copy the exact R mapping and its separate Python metadata."""
        return type(self)(self, additions=self.additions)


def rows_to_table(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: Optional[Sequence[str]] = None,
) -> ObjectTable:
    """Convert a sequence of row mappings to a dict-of-lists table."""
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        names = [str(key) for key in row]
        if len(set(names)) != len(names):
            raise ValueError("row column names must remain unique after string conversion")
        normalized_rows.append(
            {name: value for name, value in zip(names, row.values(), strict=True)}
        )
    if keys is None:
        ordered_keys: List[str] = []
        seen: set[str] = set()
        for row in normalized_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered_keys.append(key)
    else:
        ordered_keys = _unique_string_names(keys, name="keys")

    out: ObjectTable = {key: [] for key in ordered_keys}
    for row in normalized_rows:
        for key in ordered_keys:
            out[key].append(row.get(key))
    return out


def coerce_stats_table(
    stats: Mapping[str, Sequence[object]],
    *,
    expected_len: Optional[int] = None,
    name: str = "stats",
) -> StatsTable:
    """Coerce a stats dict-of-sequences to float lists and validate row lengths."""
    out: StatsTable = {}
    row_len = None
    if expected_len is not None:
        row_len = validate_integer_control(expected_len, name="expected_len")
        if row_len < 0:
            raise ValueError("expected_len must be a non-negative integer")
    column_names = _unique_string_names(list(stats), name=f"{name} column names")
    for key, values in zip(column_names, stats.values(), strict=True):
        arr = np.asarray(list(values), dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name}[{key!r}] must be one-dimensional")
        if row_len is None:
            row_len = int(arr.size)
        elif int(arr.size) != int(row_len):
            raise ValueError(f"{name}[{key!r}] has length {arr.size}, expected {row_len}")
        out[key] = [float(v) for v in arr.tolist()]
    if row_len is not None:
        for key in list(out.keys()):
            if len(out[key]) != int(row_len):
                raise ValueError(f"{name}[{key!r}] has length {len(out[key])}, expected {row_len}")
    return out


def rows_to_stats_table(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: Optional[Sequence[str]] = None,
    row_names: Optional[Sequence[str]] = None,
) -> LabeledStatsTable:
    """Build a float-valued stats table from row mappings."""
    table = rows_to_table(rows, keys=keys)
    return LabeledStatsTable(
        coerce_stats_table(table, expected_len=len(rows)),
        row_names=() if row_names is None else row_names,
    )


def validate_clustrange_result(
    obj: Mapping[str, object],
    *,
    n_obs: Optional[int] = None,
) -> ClustrangeResult:
    """Validate/coerce a clustrange-like dict into the canonical schema."""
    kvals_raw = obj.get("kvals")
    clustering_raw = obj.get("clustering")
    stats_raw = obj.get("stats")
    if kvals_raw is None:
        raise ValueError("clustrange result missing sequence field 'kvals'")
    if not isinstance(clustering_raw, Mapping):
        raise ValueError("clustrange result missing mapping field 'clustering'")
    if not isinstance(stats_raw, Mapping):
        raise ValueError("clustrange result missing mapping field 'stats'")

    kvals = validate_kvals(cast(Sequence[int], kvals_raw))
    n_obs_int = None
    if n_obs is not None:
        n_obs_int = validate_integer_control(n_obs, name="n_obs")
        if n_obs_int < 0:
            raise ValueError("n_obs must be a non-negative integer")
    expected_columns = [f"cluster{k}" for k in kvals]
    actual_columns = _unique_string_names(list(clustering_raw), name="clustering column names")
    clustering_values = {
        name: value for name, value in zip(actual_columns, clustering_raw.values(), strict=True)
    }
    if len(actual_columns) != len(expected_columns) or set(actual_columns) != set(expected_columns):
        missing = [name for name in expected_columns if name not in actual_columns]
        extra = [name for name in actual_columns if name not in expected_columns]
        raise ValueError(
            "clustering columns must match kvals; " f"missing={missing or []}, extra={extra or []}"
        )
    clustering: ClusteringMap = {}
    for key in expected_columns:
        labels = clustering_values[key]
        arr = np.asarray(labels)
        if arr.ndim != 1:
            raise ValueError(f"clustering[{key!r}] must be one-dimensional")
        if n_obs_int is not None and int(arr.size) != n_obs_int:
            raise ValueError(f"clustering[{key!r}] has length {arr.size}, expected {n_obs_int}")
        clustering[key] = arr.copy()
    clustering_table = ClusteringTable(clustering, column_names=expected_columns)
    stats = LabeledStatsTable(
        coerce_stats_table(
            cast(Mapping[str, Sequence[object]], stats_raw), expected_len=len(kvals)
        ),
        row_names=[f"cluster{k}" for k in kvals],
    )
    values = dict(obj)
    values.update(
        {
            "kvals": kvals,
            "clustering": clustering_table,
            "stats": stats,
        }
    )
    return ClustrangeResult(values)

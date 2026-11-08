from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

_RESULT_FIELDS = ("aggIndex", "aggWeights", "disaggIndex", "disaggWeights")


def _r_display_number(value: float) -> str:
    """Translate R's default scalar ``format`` to a whitespace-free Python string."""
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Inf"
    if np.isneginf(value):
        return "-Inf"
    return format(float(value), ".7g")


@dataclass(frozen=True)
class WCAggregateCasesResult(Mapping[str, np.ndarray]):
    """Typed, mapping-compatible counterpart of R's ``wcAggregateCases`` list.

    ``str(result)`` preserves R's four summary lines while omitting the trailing
    separator spaces produced by ``cat(..., "\\n")``.
    """

    aggIndex: np.ndarray
    aggWeights: np.ndarray
    disaggIndex: np.ndarray
    disaggWeights: np.ndarray

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in _RESULT_FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_RESULT_FIELDS)

    def __len__(self) -> int:
        return len(_RESULT_FIELDS)

    def __str__(self) -> str:
        n_disaggregated = len(self.disaggWeights)
        n_aggregated = len(self.aggWeights)
        average_cases = n_disaggregated / n_aggregated if n_aggregated else float("nan")
        average_weight = float(np.mean(self.aggWeights)) if n_aggregated else float("nan")
        return "\n".join(
            (
                f"Number of disaggregated cases:  {n_disaggregated}",
                f"Number of aggregated cases:  {n_aggregated}",
                f"Average aggregated cases:  {_r_display_number(average_cases)}",
                f"Average (weighted) aggregation:  {_r_display_number(average_weight)}",
            )
        )


def _factor_codes_like_r(series: pd.Series) -> np.ndarray:
    """Return the numeric labels created by R's ``factor`` relabeling loop."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return np.asarray(series.cat.codes, dtype=int) + 1

    missing = np.asarray(series.isna(), dtype=bool)
    values = series[~missing]
    if pd.api.types.is_numeric_dtype(values.dtype):
        levels = sorted(pd.unique(values).tolist())
    else:
        levels = sorted(pd.unique(values).tolist(), key=str)
    codes = pd.Categorical(series, categories=levels).codes.astype(int) + 1
    return np.asarray(codes, dtype=int)


def _as_case_frame(x: Any) -> tuple[pd.DataFrame, Any | None]:
    # Imported unguarded, like the four sibling sites in _boot_inputs, fuzzy,
    # seqclustname and seqnull. traminex is a hard runtime dependency
    # (pyproject.toml), so the handler this replaces was unreachable in any
    # supported installation — and had it ever run, its `STSList = ()` sentinel
    # would have made `isinstance` silently False, quietly failing to recognise
    # sequence objects instead of failing loudly. Finding F-30.
    from traminex import STSList

    if isinstance(x, STSList):
        return pd.DataFrame(np.asarray(x.data).copy()), x
    if isinstance(x, pd.DataFrame):
        return x.copy(deep=True), None

    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("x must be a vector or 2D case table")
    return pd.DataFrame(arr.copy()), None


def wcAggregateCases(
    x,
    weights: Optional[np.ndarray] = None,
    *,
    weighted: bool = True,
) -> WCAggregateCasesResult:
    """Aggregate identical cases (rows), returning indices and aggregated weights.

    This function identifies duplicate rows in the input data and aggregates their weights.
    It's useful for reducing computational complexity when working with datasets that contain
    many identical observations.

    Parameters
    ----------
    x : array-like, pd.DataFrame, or traminex.STSList
        Vector or 2D input data where each row is a case.
    weights : array-like, optional
        1D array of weights (length N). If None, uniform weights of 1.0 are used.
    weighted : bool, default True
        For an STSList only, use its attached weights when ``weights`` is absent.

    Returns
    -------
    WCAggregateCasesResult
        Typed mapping with keys:
        - 'aggIndex': first original indices in R aggregate-key order
        - 'aggWeights': aggregated weights per unique case
        - 'disaggIndex': for each original row, the index in the unique cases
        - 'disaggWeights': the original input weights

    Examples
    --------
    >>> import numpy as np
    >>> from weightedcluster import wcAggregateCases
    >>> # Data with duplicate rows
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2]])
    >>> result = wcAggregateCases(X)
    >>> result['aggIndex']  # Indices of unique rows
    array([0, 1])
    >>> result['aggWeights']  # Counts of each unique row
    array([3., 2.])

    >>> # With custom weights
    >>> weights = np.array([1.0, 2.0, 1.5, 0.5, 2.0])
    >>> result = wcAggregateCases(X, weights=weights)
    >>> result['aggWeights']  # Sum of weights for each unique row
    array([4.5, 2.5])

    >>> # With pandas DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 1], 'b': [3, 4, 3]})
    >>> result = wcAggregateCases(df)
    >>> result['aggWeights']
    array([2., 1.])
    """
    if not isinstance(weighted, (bool, np.bool_)):
        raise TypeError("weighted must be a boolean")
    df, sequence_object = _as_case_frame(x)
    N = len(df)
    if N == 0:
        raise ValueError("arguments must have same length")
    if weights is None and sequence_object is not None and weighted:
        weights = sequence_object.weights
    if weights is None:
        w = np.ones(N, dtype=float)
    else:
        raw_weights = np.asarray(weights)
        if raw_weights.dtype.kind not in "buifc":
            raise TypeError("invalid 'type' (character) of argument")
        w = np.asarray(raw_weights, dtype=float).reshape(-1)

    factor_columns = [_factor_codes_like_r(df[column]) for column in df.columns]
    row_keys = [
        "@@@WC_SEP@@".join(
            "NA" if column[row] == 0 else str(int(column[row])) for column in factor_columns
        )
        for row in range(N)
    ]
    sorted_keys = sorted(set(row_keys))
    key_to_group = {key: group for group, key in enumerate(sorted_keys)}
    inverse = np.asarray([key_to_group[key] for key in row_keys], dtype=int)
    first_idx = np.asarray([row_keys.index(key) for key in sorted_keys], dtype=int)

    effective_weights = np.full(N, np.nan, dtype=float)
    effective_weights[: min(N, len(w))] = w[:N]
    agg_w = np.asarray(
        [np.sum(effective_weights[inverse == group]) for group in range(len(sorted_keys))],
        dtype=float,
    )

    return WCAggregateCasesResult(
        aggIndex=first_idx,
        aggWeights=agg_w,
        disaggIndex=inverse,
        disaggWeights=w,
    )

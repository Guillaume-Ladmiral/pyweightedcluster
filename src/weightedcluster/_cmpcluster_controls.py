from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

_HCLUST_METHODS = (
    "ward.D",
    "ward.D2",
    "single",
    "complete",
    "average",
    "mcquitty",
    "median",
    "centroid",
)
_ALL_METHODS = (*_HCLUST_METHODS, "pam", "diana", "beta.flexible")
_METHOD_ALIASES = MappingProxyType(
    {
        "ward": "ward.d2",
        "weighted": "mcquitty",
        "flexible": "beta.flexible",
    }
)
_NO_WEIGHTS_METHODS = frozenset(("diana", "beta.flexible"))
_METHOD_PRESENTATION = MappingProxyType({value.lower(): value for value in _ALL_METHODS})


def _canonical_cmpcluster_method(method_name: str) -> str:
    """Return the lower-case canonical dispatch name."""
    return _METHOD_ALIASES.get(method_name.lower(), method_name.lower())


def _present_cmpcluster_method(method_name: str) -> str:
    """Return the public method name used by ``method='all'``."""
    canonical = _canonical_cmpcluster_method(method_name)
    return _METHOD_PRESENTATION[canonical]


def _normalize_cmpcluster_methods(
    method: Sequence[str] | str,
    *,
    has_weights: bool,
) -> list[str]:
    """Return canonical public method names in stable, unique order."""
    methods = [method] if isinstance(method, str) else list(method)
    methods = [str(value) for value in methods]
    lowered = [value.lower() for value in methods]
    if "all" in lowered:
        return list(_ALL_METHODS if not has_weights else (*_HCLUST_METHODS, "pam"))
    if not methods:
        raise ValueError("method must contain at least one clustering method")
    supported = {value.lower() for value in _ALL_METHODS}
    unknown = [
        value
        for value in lowered
        if value.lower() not in supported and value.lower() not in _METHOD_ALIASES
    ]
    if unknown:
        raise ValueError(f"unknown clustering method {unknown[0]}")
    canonical = [_canonical_cmpcluster_method(value) for value in methods]
    if has_weights and any(value in _NO_WEIGHTS_METHODS for value in canonical):
        raise ValueError("methods diana, beta.flexible cannot be used with weights.")
    presented: list[str] = []
    seen: set[str] = set()
    for value in canonical:
        public_name = _present_cmpcluster_method(value)
        if public_name not in seen:
            presented.append(public_name)
            seen.add(public_name)
    return presented


def _validate_cmpcluster_replay_starts(
    replay_starts: Mapping[int, Sequence[int] | np.ndarray] | None,
    *,
    maxcluster: int,
    n_obs: int,
) -> dict[int, np.ndarray] | None:
    """Return the existing one-row, zero-based integer plans by K."""
    if replay_starts is None:
        return None
    pam_replay: dict[int, np.ndarray] = {}
    for k in range(2, maxcluster + 1):
        if k not in replay_starts:
            raise ValueError(f"replay_starts is missing the plan for k={k}")
        starts = np.asarray(replay_starts[k])
        if starts.ndim != 1 or starts.size != k:
            raise ValueError(f"replay_starts[{k}] must be a zero-based vector of length {k}")
        if not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("replay_starts must contain zero-based integer indices")
        starts = starts.astype(int)
        if np.any(starts < 0) or np.any(starts >= n_obs):
            raise ValueError("replay_starts contains an out-of-range zero-based index")
        if np.unique(starts).size != k:
            raise ValueError("replay_starts vectors must contain unique medoids")
        pam_replay[k] = starts.reshape(1, k)
    return pam_replay

"""Sequence distances — delegated to traminex (MIGRATION_PLAN.md Phase 1 / A1).

This project no longer computes HAM / OM / LCS / DHD distances itself: those are
owned by traminex, the family's R-validated (TraMineR 2.2-14) base, and reached
here through ``traminex.distances.pairwise`` /
``traminex.distances.pairwise_between_sets``. The thin layer below only maps this
project's ``(metric, normalize)`` API onto traminex's method/``norm`` vocabulary;
it does no distance arithmetic of its own.

Low-level normalization semantics are pinned against R in
``tests/test_phase1_a1_distances_traminex_parity.py``. The helper's
``normalize=True`` and ``normalize=False`` forms map as follows:

============  ==================================  ==============================
metric        ``normalize=True``                  ``normalize=False``
============  ==================================  ==============================
HAM           mismatch sum / L  (R maxlength)      mismatch sum  (R norm="none")
OM            raw / L           (R norm="length")  raw OM        (R norm="none")
LCS           1 - lcs/L = raw/2L (R norm="length") |a|+|b|-2lcs  (R norm="none")
DHD           cost sum / L      (R maxlength)       cost sum     (R norm="none")
============  ==================================  ==============================

HAM is Hamming with unit mismatch cost; it is computed as DHD with a unit
off-diagonal substitution matrix so the raw form is the exact integer mismatch
sum (R ``norm="none"``), and ``/L`` recovers R ``norm="maxlength"`` — equal here
because the sequences are fixed-length.

Normalization is workflow/configuration dependent. In the
``seqclararange`` runtime, ``norm="none"`` passes ``normalize=False``;
``norm="length"`` passes ``normalize=True``; and ``norm="maxlength"`` passes
``normalize=False`` and applies an explicit scale of ``1 / sequence_length``.
The default ``seqdist_args=None`` path therefore uses LCS with ``norm="none"``.
Other callers may deliberately select either low-level normalization form.

DHD here is *position-cost-weighted Hamming* (this project's historical meaning),
served by traminex's explicit-``subm`` DHD path:
``substitution_costs=None`` -> unit costs (plain Hamming); a ``(S, S)`` matrix ->
R ``seqdist(method="HAM", sm=...)``; a ``(T, S, S)`` array ->
R ``seqdist(method="DHD", sm=...)``.

OMSPELL is the R-faithful TraMineR OMspell (traminex ``method="OMSPELL"``):
spell durations enter as ``dur**tpow - 1`` (``tpow=1``) and ``timecost`` is R's
``expcost``. ``normalize=True`` == R ``norm="maxlength"``, ``normalize=False`` ==
``norm="none"``. (The pre-migration local OMspell omitted the ``-1`` duration
shift and was not R-faithful; no test pinned its values.)
"""

from __future__ import annotations

import numpy as np

# traminex is a declared runtime dependency (see pyproject.toml). It is the
# single, R-golden-tested home for sequence distances in this family.
from traminex.distances import pairwise as _tx_pairwise
from traminex.distances import pairwise_between_sets as _tx_pairwise_between

from ._metrics import normalize_metric

_SEQDIST_CONFIG_KEYS = frozenset(
    {
        "method",
        "metric",
        "norm",
        "normalize",
        "indel",
        "substitution_costs",
        "sm",
        "timecost",
        "time_cost",
    }
)
_SEQDIST_ALIAS_GROUPS = (
    ("method", "metric"),
    ("substitution_costs", "sm"),
    ("timecost", "time_cost"),
    ("norm", "normalize"),
)


def _alias_values_equal(group: tuple[str, str], left: object, right: object) -> bool:
    if group == ("method", "metric"):
        try:
            return normalize_metric(str(left))[0] == normalize_metric(str(right))[0]
        except ValueError:
            return False
    if group == ("norm", "normalize"):

        def canonical_norm(value: object, *, boolean: bool) -> str:
            if boolean:
                if not isinstance(value, (bool, np.bool_)):
                    return "<invalid>"
                return "maxlength" if bool(value) else "none"
            return str(value).lower()

        return canonical_norm(left, boolean=False) == canonical_norm(right, boolean=True)
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
    except TypeError:
        return left == right


def normalize_seqdist_config(
    seqdist_args: dict | None, *, default_method: str = "HAM"
) -> dict[str, object]:
    """Normalize seqdist-style configuration to backend distance arguments.

    Parameters
    ----------
    seqdist_args : dict or None
        User-provided distance arguments. Supports R-style aliases such as `sm`
        for substitution costs.
    default_method : str, optional
        Default distance method when `seqdist_args` is None or does not include
        a method.

    Returns
    -------
    dict
        Normalized config with keys: `method`, `metric`, `indel`,
        `substitution_costs`, `timecost`.
    """
    if seqdist_args is None:
        raw: dict[str, object] = {}
    elif isinstance(seqdist_args, dict):
        raw = dict(seqdist_args)
    else:
        raise TypeError("seqdist_args must be a dict or None")

    unknown = sorted(set(raw) - _SEQDIST_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"seqdist_args contains unsupported keys: {unknown}")
    for aliases in _SEQDIST_ALIAS_GROUPS:
        supplied = [name for name in aliases if name in raw]
        if len(supplied) > 1 and not _alias_values_equal(
            aliases, raw[supplied[0]], raw[supplied[1]]
        ):
            raise ValueError(f"seqdist_args contains conflicting aliases: {supplied}")

    method_in = raw.get("method", raw.get("metric", default_method))
    method, metric = normalize_metric(str(method_in))

    substitution_costs = raw.get("substitution_costs", raw.get("sm", None))

    if "indel" in raw and method not in ("OM", "OMSPELL"):
        raise ValueError(f"seqdist_args['indel'] is not supported for method {method}")
    if substitution_costs is not None and method not in ("HAM", "DHD", "OM", "OMSPELL"):
        raise ValueError(f"seqdist_args['substitution_costs'] is not supported for method {method}")
    supplied_timecost = raw.get("timecost", raw.get("time_cost", None))
    if supplied_timecost is not None and method != "OMSPELL":
        raise ValueError(f"seqdist_args['timecost'] is not supported for method {method}")

    indel = 1.0
    if method in ("OM", "OMSPELL"):
        try:
            indel = float(raw.get("indel", 1.0))
        except (TypeError, ValueError) as e:
            raise ValueError("seqdist_args['indel'] must be numeric") from e
        if not np.isfinite(indel):
            raise ValueError("seqdist_args['indel'] must be finite")
        if indel < 0:
            raise ValueError("seqdist_args['indel'] must be non-negative")

    timecost = None
    if method == "OMSPELL":
        tc_raw = supplied_timecost
        if tc_raw is not None:
            try:
                timecost = float(tc_raw)
            except (TypeError, ValueError) as e:
                raise ValueError("seqdist_args['timecost'] must be numeric") from e
            if not np.isfinite(timecost):
                raise ValueError("seqdist_args['timecost'] must be finite")

    if "norm" in raw:
        norm = str(raw["norm"]).lower()
    elif "normalize" in raw:
        if not isinstance(raw["normalize"], (bool, np.bool_)):
            raise ValueError("seqdist_args['normalize'] must be a boolean")
        norm = "maxlength" if bool(raw["normalize"]) else "none"
    else:
        norm = "none"
    if norm not in {"none", "length", "maxlength"}:
        raise ValueError("seqdist_args['norm'] must be one of none, length, or maxlength")

    return {
        "method": method,
        "metric": metric,
        "indel": float(indel),
        "substitution_costs": substitution_costs,
        "timecost": timecost,
        "norm": norm,
        "normalize": norm != "none",
    }


def _seq_length(X: np.ndarray) -> int:
    return int(X.shape[1]) if X.ndim == 2 and X.shape[1] > 0 else 0


def _validate_observed_substitution_costs(
    seqdata: np.ndarray,
    substitution_costs: object,
    *,
    method: str,
) -> None:
    """Reject non-finite costs that an observed state pair can reach.

    Fixed costs are reachable for every state observed anywhere in the input.
    Position-dependent costs are reachable only for states observed at the
    corresponding position.  Shape/domain validation remains owned by the
    traminex provider; this guard only improves non-finite failure timing while
    preserving intentional infinities for unobserved states.
    """

    method_name = str(method).upper()
    if method_name not in {"HAM", "DHD"}:
        return
    costs = np.asarray(substitution_costs, dtype=float)
    if costs.ndim == 0:
        finite = bool(np.isfinite(costs.item()))
    else:
        sequences = np.asarray(seqdata)
        if sequences.ndim != 2 or costs.ndim not in {2, 3}:
            return
        try:
            integer_sequences = np.asarray(sequences, dtype=int)
        except (TypeError, ValueError):
            return
        if costs.ndim == 2:
            if costs.shape[0] != costs.shape[1]:
                return
            states = np.unique(integer_sequences)
            states = states[(states >= 0) & (states < costs.shape[0])]
            finite = bool(states.size == 0 or np.all(np.isfinite(costs[np.ix_(states, states)])))
        else:
            if costs.shape[0] != integer_sequences.shape[1] or costs.shape[1] != costs.shape[2]:
                return
            finite = True
            for position in range(integer_sequences.shape[1]):
                states = np.unique(integer_sequences[:, position])
                states = states[(states >= 0) & (states < costs.shape[1])]
                if states.size and not np.all(np.isfinite(costs[position][np.ix_(states, states)])):
                    finite = False
                    break
    if not finite:
        raise ValueError(f"{method_name} substitution costs for observed states must be finite")


def _dhd_subm(X: np.ndarray, substitution_costs, *, method: str) -> np.ndarray:
    """Resolve this project's DHD substitution costs to an explicit traminex subm.

    ``None`` -> unit off-diagonal matrix (plain Hamming, cost 1 per mismatch).
    A 2D ``(S, S)`` or 3D ``(T, S, S)`` array is passed straight through.
    """
    if substitution_costs is None:
        K = int(np.asarray(X).max()) + 1 if np.asarray(X).size else 1
        M = np.ones((max(K, 1), max(K, 1)), dtype=float)
        np.fill_diagonal(M, 0.0)
        return M
    resolved = np.asarray(substitution_costs, dtype=float)
    _validate_observed_substitution_costs(X, resolved, method=method)
    return _neutralize_unreachable_costs(X, resolved)


def _neutralize_unreachable_costs(X: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Give the provider 0 where a non-finite cost can be reached by no observed pair.

    This project keeps infinite costs for unobserved states permitted
    (:func:`_validate_observed_substitution_costs`). traminex's explicit-cost DHD,
    as R's DHD, refuses any non-finite cost. A distance reads ``costs[a, b]``
    (or ``costs[t, a, b]``) only for states ``a`` and ``b`` observed (at position
    ``t``), so an entry outside those never enters a distance, and replacing it
    leaves every value unchanged. Inconsistent shapes are passed on unchanged for
    the provider to refuse.
    """
    if costs.ndim not in (2, 3) or np.all(np.isfinite(costs)):
        return costs
    sequences = np.asarray(X)
    if sequences.ndim != 2 or costs.shape[-1] != costs.shape[-2]:
        return costs
    try:
        codes = np.asarray(sequences, dtype=int)
    except (TypeError, ValueError):
        return costs
    size = costs.shape[-1]
    reachable = np.zeros(costs.shape, dtype=bool)
    if costs.ndim == 2:
        states = np.unique(codes)
        states = states[(states >= 0) & (states < size)]
        reachable[np.ix_(states, states)] = True
    else:
        if costs.shape[0] != codes.shape[1]:
            return costs
        for position in range(codes.shape[1]):
            states = np.unique(codes[:, position])
            states = states[(states >= 0) & (states < size)]
            reachable[position][np.ix_(states, states)] = True
    neutral = costs.copy()
    neutral[~reachable & ~np.isfinite(neutral)] = 0.0
    return neutral


def sequences_distance(
    X: np.ndarray,
    *,
    metric: str = "HAM",
    normalize: bool = True,
    indel: float = 1.0,
    substitution_costs=None,
    timecost: float | None = None,
) -> np.ndarray:
    X = np.asarray(X)
    method, _alias = normalize_metric(str(metric))
    T = _seq_length(X)
    Xi = np.asarray(X, dtype=np.int32)

    if method == "HAM":
        # Hamming == DHD with unit mismatch costs: exact integer mismatch sum,
        # normalized by length (R maxlength == norm="none"/L for fixed length).
        D = _tx_pairwise(
            Xi,
            method="DHD",
            subm=_dhd_subm(Xi, substitution_costs, method=method),
        )
        return D / float(T) if (normalize and T > 0) else D
    if method == "OM":
        subm = 2.0 if substitution_costs is None else np.asarray(substitution_costs, dtype=float)
        return _tx_pairwise(
            Xi,
            method="OM",
            indel=float(indel),
            subm=subm,
            norm="length" if normalize else None,
        )
    if method == "LCS":
        return _tx_pairwise(Xi, method="LCS", norm="length" if normalize else None)
    if method == "DHD":
        D = _tx_pairwise(
            Xi,
            method="DHD",
            subm=_dhd_subm(Xi, substitution_costs, method=method),
        )
        return D / float(T) if (normalize and T > 0) else D
    if method == "OMSPELL":
        subm = 2.0 if substitution_costs is None else np.asarray(substitution_costs, dtype=float)
        return _tx_pairwise(
            Xi,
            method="OMSPELL",
            indel=float(indel),
            subm=subm,
            expcost=0.0 if timecost is None else float(timecost),
            tpow=1.0,
            norm="maxlength" if normalize else None,
        )
    raise NotImplementedError("Unsupported metric: %s" % metric)


def sequences_distance_between(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric: str = "HAM",
    normalize: bool = True,
    indel: float = 1.0,
    substitution_costs=None,
    timecost: float | None = None,
) -> np.ndarray:
    X = np.asarray(X)
    Y = np.asarray(Y)
    method, _alias = normalize_metric(str(metric))
    T = _seq_length(X)
    Xi = np.asarray(X, dtype=np.int32)
    Yi = np.asarray(Y, dtype=np.int32)

    if method == "HAM":
        D = _tx_pairwise_between(
            Xi,
            Yi,
            method="DHD",
            subm=_dhd_subm(
                np.vstack([Xi, Yi]),
                substitution_costs,
                method=method,
            ),
        )
        return D / float(T) if (normalize and T > 0) else D
    if method == "OM":
        subm = 2.0 if substitution_costs is None else np.asarray(substitution_costs, dtype=float)
        return _tx_pairwise_between(
            Xi,
            Yi,
            method="OM",
            indel=float(indel),
            subm=subm,
            norm="length" if normalize else None,
        )
    if method == "LCS":
        return _tx_pairwise_between(Xi, Yi, method="LCS", norm="length" if normalize else None)
    if method == "DHD":
        subm = _dhd_subm(
            np.vstack([Xi, Yi]),
            substitution_costs,
            method=method,
        )
        D = _tx_pairwise_between(Xi, Yi, method="DHD", subm=subm)
        return D / float(T) if (normalize and T > 0) else D
    if method == "OMSPELL":
        subm = 2.0 if substitution_costs is None else np.asarray(substitution_costs, dtype=float)
        return _tx_pairwise_between(
            Xi,
            Yi,
            method="OMSPELL",
            indel=float(indel),
            subm=subm,
            expcost=0.0 if timecost is None else float(timecost),
            tpow=1.0,
            norm="maxlength" if normalize else None,
        )
    raise NotImplementedError("Unsupported metric: %s" % metric)

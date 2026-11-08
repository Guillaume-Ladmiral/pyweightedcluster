"""
Property-based clustering for sequence analysis.

The faithful public tree builders are structural adapters over traminex's
TraMineR-compatible ``disstree`` and ``seqtree`` implementations. Sequence
property extraction remains local because it is the clustering layer's domain.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Split selection, discrepancy, permutation, and tree growth are owned by
# traminex. This module only converts the public traminex objects to the legacy
# mapping interface and applies WeightedCluster's structural split schedule.
from traminex import (
    seqdss as _traminex_seqdss,
)
from traminex import (
    seqdur as _traminex_seqdur,
)
from traminex import (
    seqeapplysub as _traminex_seqeapplysub,
)
from traminex import (
    seqecreate as _traminex_seqecreate,
)
from traminex import (
    seqefsub as _traminex_seqefsub,
)
from traminex import (
    seqici as _traminex_seqici,
)
from traminex import (
    seqient as _traminex_seqient,
)
from traminex import (
    seqistatd as _traminex_seqistatd,
)
from traminex import (
    seqST as _traminex_seqST,
)
from traminex import (
    seqtransn as _traminex_seqtransn,
)
from traminex.diss import dissassoc as _traminex_dissassoc
from traminex.disstree import disstree as _traminex_disstree
from traminex.format import STSList
from traminex.format import seqdef as _traminex_seqdef
from traminex.seqtree import seqtree as _traminex_seqtree

from ._validation import validate_integer_control


def _validate_tree_execution_controls(
    *,
    perm_n_jobs: int | None,
    n_jobs: int | None,
    progressbar: bool,
    owner: str,
) -> None:
    """Normalize serial defaults and reject unavailable tree execution modes."""
    if perm_n_jobs is not None and n_jobs is not None:
        raise TypeError(f"{owner} received both perm_n_jobs and n_jobs")
    if perm_n_jobs is not None:
        raw_jobs = perm_n_jobs
        jobs_name = "perm_n_jobs"
    elif n_jobs is not None:
        raw_jobs = n_jobs
        jobs_name = "n_jobs"
    else:
        raw_jobs = 1
        jobs_name = "perm_n_jobs"
    jobs = validate_integer_control(raw_jobs, name=jobs_name)
    if jobs != 1:
        raise NotImplementedError(f"{owner} supports serial permutation execution only")
    if not isinstance(progressbar, (bool, np.bool_)):
        raise TypeError("progressbar must be a boolean")
    if progressbar:
        raise NotImplementedError(f"{owner} progress reporting is not available")


def _pop_exclusive_alias(options: dict[str, Any], primary: str, alias: str, *, default: Any) -> Any:
    if primary in options and alias in options:
        raise TypeError(f"{primary} and {alias} cannot both be specified")
    if primary in options:
        return options.pop(primary)
    if alias in options:
        return options.pop(alias)
    return default


def _set_weightedcluster_split_schedule(root: Any) -> None:
    """Apply R WeightedCluster's frontier-based ``clusterSplitSchedule``."""

    def reset(node: Any) -> None:
        node.info["splitschedule"] = 0
        if node.kids is not None:
            reset(node.kids[0])
            reset(node.kids[1])

    def candidates(node: Any, out: list[Any]) -> None:
        if node.kids is None:
            return
        if int(node.kids[0].info["splitschedule"]) == 0:
            out.append(node)
            return
        candidates(node.kids[0], out)
        candidates(node.kids[1], out)

    reset(root)
    root.info["splitschedule"] = 1
    schedule = 2
    while True:
        frontier: list[Any] = []
        candidates(root, frontier)
        if not frontier:
            break

        def explained(node: Any) -> float:
            # R's order of operations: SCtot <- vardis * n, then SCexpl <- R2 * SCtot
            # (propclustering.R:195-196). Multiplied left to right, an exact tie
            # between splits can round apart and select the other one.
            return float(node.split.info["R2"] * (node.info["vardis"] * node.info["n"]))

        selected = max(frontier, key=explained)
        selected.kids[0].info["splitschedule"] = schedule
        selected.kids[1].info["splitschedule"] = schedule
        schedule += 1


def _split_to_mapping(split: Any, columns: Sequence[str]) -> dict[str, Any]:
    info = dict(split.info)
    categorical = split.breaks is None
    labels = [] if split.labels is None else list(split.labels)
    index = None if split.index is None else np.asarray(split.index, dtype=int).copy()
    left_levels = (
        [level for level, side in zip(labels, index, strict=True) if int(side) == 1]
        if categorical and index is not None
        else []
    )
    return {
        "var": str(columns[int(split.varindex)]),
        "varindex": int(split.varindex),
        "type": "cat" if categorical else "num",
        "threshold": None if categorical else float(split.breaks),
        "levels": left_levels,
        "gain": float(info["R2"]),
        "pvalue": info.get("pval"),
        "prob": np.asarray(split.prob, dtype=float).copy(),
        "index": index,
        "breaks": None if split.breaks is None else float(split.breaks),
        "na_group": split.na_group,
        "labels": labels,
        "info": info,
    }


def _node_to_mapping(node: Any, columns: Sequence[str]) -> dict[str, Any]:
    return {
        "id": int(node.id),
        "info": {
            key: (np.asarray(value, dtype=int).copy() if key == "ind" else value)
            for key, value in node.info.items()
        },
        "split": None if node.split is None else _split_to_mapping(node.split, columns),
        "kids": (
            None
            if node.kids is None
            else [
                _node_to_mapping(node.kids[0], columns),
                _node_to_mapping(node.kids[1], columns),
            ]
        ),
    }


def _tree_adapter(
    tree: Any,
    diss: np.ndarray,
    *,
    classes: Sequence[str],
    apply_schedule: bool = True,
) -> dict[str, Any]:
    if apply_schedule:
        _set_weightedcluster_split_schedule(tree.root)
    parameters = dict(tree.info.get("parameters", {}))
    info = dict(tree.info)
    info.update(
        {
            "R": int(parameters.get("R", 1)),
            "pval": float(parameters.get("pval", 1.0)),
            "min_size": float(parameters.get("min_size", parameters.get("minSize", 0.0))),
            "max_depth": int(parameters.get("max_depth", parameters.get("maxdepth", 0))),
            "weight_permutation": str(tree.info.get("weight_permutation", "none")),
            "weights": np.asarray(tree.weights, dtype=float).copy(),
        }
    )
    return {
        "class": list(classes),
        "root": _node_to_mapping(tree.root, list(tree.data.columns)),
        "data": tree.data.copy(),
        "info": info,
        "fitted": np.asarray(tree.fitted, dtype=int).copy(),
        "weights": np.asarray(tree.weights, dtype=float).copy(),
        "adjustment": tree.adjustment,
        "_diss": np.asarray(diss, dtype=float).copy(),
    }


def _complete_random_sampling_tree(
    tree: Any,
    diss: np.ndarray,
    *,
    R: int,
    pval: float,
    random_state: Any,
) -> None:
    """Complete WeightedCluster's documented random-sampling tree mode.

    R's tree code applies replicate-style split tests (with integer-truncated
    weights) for this legacy mode, then uses genuine random sampling for the
    final adjustment. Public traminex owns both discrepancy calculations:
    an unconditional public tree supplies each best split, and public
    ``dissassoc`` tests and adjusts it. This adapter only prunes rejected nodes
    and restores R's preorder node identifiers.
    """

    requested_R = int(R)
    effective_pval = 1.0 if requested_R <= 1 else float(pval)
    if requested_R > 1 and effective_pval < 1.0 / requested_R:
        effective_pval = 1.0 / requested_R
        warnings.warn(
            f"minimum possible p-value with R={requested_R} permutations is "
            f"{effective_pval}; pval changed to {effective_pval}",
            stacklevel=3,
        )
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    def test_splits(node: Any) -> None:
        if node.kids is None or requested_R <= 1:
            return
        members = np.asarray(node.info["ind"], dtype=int)
        left_members = np.asarray(node.kids[0].info["ind"], dtype=int)
        group = np.isin(members, left_members)
        replicate_weights = np.trunc(np.asarray(tree.weights, dtype=float)[members]).astype(float)
        association = _traminex_dissassoc(
            np.asarray(diss, dtype=float)[np.ix_(members, members)],
            group,
            weights=replicate_weights,
            R=requested_R,
            weight_permutation="replicate",
            random_state=rng,
        )
        split_pval = float(association.stat.loc["Pseudo F", "p_value"])
        if split_pval > effective_pval:
            node.split = None
            node.kids = None
            return
        node.split.info["pval"] = split_pval
        test_splits(node.kids[0])
        test_splits(node.kids[1])

    test_splits(tree.root)

    fitted = np.zeros(np.asarray(tree.fitted).shape[0], dtype=int)
    next_id = 1

    def renumber(node: Any) -> None:
        nonlocal next_id
        node.id = next_id
        next_id += 1
        if node.kids is None:
            fitted[np.asarray(node.info["ind"], dtype=int)] = int(node.id)
            return
        renumber(node.kids[0])
        renumber(node.kids[1])

    renumber(tree.root)
    tree.fitted = fitted
    tree.info["parameters"]["R"] = requested_R
    tree.info["parameters"]["pval"] = effective_pval
    tree.info["weight_permutation"] = "random-sampling"
    tree.adjustment = _traminex_dissassoc(
        np.asarray(diss, dtype=float),
        np.asarray(tree.fitted, dtype=int),
        weights=np.asarray(tree.weights, dtype=float),
        R=requested_R,
        weight_permutation="random-sampling",
        random_state=rng,
    )


def _leaf_membership(root: dict[str, Any], n_obs: int) -> np.ndarray:
    fitted = np.zeros(int(n_obs), dtype=int)

    def assign(node: dict[str, Any]) -> None:
        if node.get("kids") is None:
            fitted[np.asarray(node["info"]["ind"], dtype=int)] = int(node["id"])
            return
        assign(node["kids"][0])
        assign(node["kids"][1])

    assign(root)
    return fitted


def wcPropertyClustering(
    diss: np.ndarray,
    properties,
    maxcluster: Optional[int] = None,
    *,
    perm_n_jobs: int | None = None,
    n_jobs: int | None = None,
    progressbar: bool = False,
    **kwargs,
) -> Dict:
    """Build an R-faithful property dissimilarity tree through traminex.

    The returned mapping preserves pyweightedcluster's historical dictionary
    interface. Its nodes and statistics are structural views of
    :func:`traminex.disstree`; no split or discrepancy mathematics is computed
    in this package.

    ``perm_n_jobs`` and ``n_jobs`` are aliases for the permutation worker
    count. ``None`` resolves to one serial worker; both aliases cannot have
    non-``None`` values in one call, which raises ``TypeError``. Only ``1`` is
    currently available, and any other integer raises ``NotImplementedError``.
    Job values must be integers or ``None``; boolean, nonnumeric, and
    nonintegral values fail eagerly. ``progressbar`` must be a boolean and
    defaults to ``False``; requesting ``True`` raises ``NotImplementedError``
    because property-tree progress reporting is not available.
    """
    _validate_tree_execution_controls(
        perm_n_jobs=perm_n_jobs,
        n_jobs=n_jobs,
        progressbar=progressbar,
        owner="wcPropertyClustering",
    )
    maxcluster_int = (
        None if maxcluster is None else validate_integer_control(maxcluster, name="maxcluster")
    )
    D = np.asarray(diss, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("diss must be a square (N,N) matrix")
    if isinstance(properties, pd.DataFrame):
        props_df = properties.reset_index(drop=True)
    else:
        props_df = pd.DataFrame(properties)
    props_df = props_df.copy()
    for column in props_df.columns:
        if isinstance(props_df[column].dtype, pd.CategoricalDtype):
            props_df[column] = props_df[column].cat.remove_unused_categories()
    if props_df.shape[0] != D.shape[0]:
        raise ValueError("properties and diss must have the same number of rows")

    options = dict(kwargs)
    weights = options.pop("weights", None)
    min_size = float(_pop_exclusive_alias(options, "min_size", "minSize", default=0.05))
    max_depth = validate_integer_control(
        _pop_exclusive_alias(options, "max_depth", "maxdepth", default=5),
        name="max_depth",
    )
    R = validate_integer_control(
        _pop_exclusive_alias(options, "perm_R", "R", default=1),
        name="R",
    )
    pval = float(options.pop("pval", 0.01))
    if not np.isfinite(min_size) or min_size <= 0.0:
        raise ValueError("min_size must be a positive finite number")
    if R < 0:
        raise ValueError("R must be non-negative")
    if not np.isfinite(pval) or not 0.0 <= pval <= 1.0:
        raise ValueError("pval must be between zero and one")
    weight_permutation = str(options.pop("weight_permutation", "replicate"))
    squared = options.pop("squared", False)
    if not isinstance(squared, (bool, np.bool_)):
        raise TypeError("squared must be a boolean")
    first = options.pop("first", None)
    random_state = options.pop("random_state", None)
    if random_state is not None:
        random_state = validate_integer_control(random_state, name="random_state")
        if random_state < 0:
            raise ValueError("random_state must be non-negative")
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"unexpected wcPropertyClustering keyword(s): {names}")

    complete_random_sampling = weight_permutation == "random-sampling" and weights is not None
    delegate_permutation = "diss" if weight_permutation == "random-sampling" else weight_permutation
    tree = _traminex_disstree(
        D,
        props_df,
        weights=weights,
        min_size=min_size,
        max_depth=max_depth,
        R=1 if complete_random_sampling else R,
        pval=1.0 if complete_random_sampling else pval,
        weight_permutation=delegate_permutation,
        squared=squared,
        first=first,
        random_state=random_state,
    )
    if complete_random_sampling:
        _complete_random_sampling_tree(
            tree,
            D**2 if squared else D,
            R=R,
            pval=pval,
            random_state=random_state,
        )
    out = _tree_adapter(tree, D, classes=("dtclust", "disstree", "list"))
    out["info"]["random_state"] = random_state
    if maxcluster_int is not None:
        out = dtprune(out, maxcluster_int)
    return out


_ALLOWED_SEQPROP_PROPERTIES = {
    "state",
    "duration",
    "spell.age",
    "spell.dur",
    "transition",
    "pattern",
    "aftransition",
    "afpattern",
    "complexity",
}


def _normalize_seqprop_properties(properties: Sequence[str]) -> list[str]:
    out = [str(p).strip().lower() for p in properties]
    invalid = sorted({p for p in out if p not in _ALLOWED_SEQPROP_PROPERTIES})
    if invalid:
        raise ValueError(
            "Unknown properties: "
            + ", ".join(invalid)
            + ". Allowed values are: "
            + ", ".join(sorted(_ALLOWED_SEQPROP_PROPERTIES))
        )
    return out


def _r_make_names(names: Sequence[str]) -> list[str]:
    """Translate property names like R ``make.names(..., unique=TRUE)``."""
    import re

    reserved = {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "in",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_complex_",
        "NA_character_",
    }
    out: list[str] = []
    counts: dict[str, int] = {}
    for value in names:
        original = str(value)
        name = re.sub(r"[^0-9A-Za-z._]", ".", original)
        if not re.match(r"^[A-Za-z]|^\.(?![0-9])", original):
            name = "X" + name
        if name in reserved:
            name += "."
        base = name
        suffix = counts.get(base, 0)
        while name in out:
            suffix += 1
            name = f"{base}.{suffix}"
        counts[base] = suffix
        out.append(name)
    return out


def _sequence_labels(sequence: STSList) -> tuple[list[str], dict[int, str]]:
    labels = list(sequence.labels or sequence.alphabet)
    alphabet = list(sequence.alphabet)
    if len(labels) != len(alphabet):
        labels = alphabet
    label_by_code = {index: str(label) for index, label in enumerate(labels)}
    return [str(label) for label in labels], label_by_code


def _extract_seq_properties(
    seqdata: STSList,
    *,
    properties: Sequence[str],
    pmin_support: float = 0.05,
    max_k: int = -1,
    with_missing: bool = True,
    force_group_prefix: bool = False,
) -> "pd.DataFrame":
    """Extract the nine WeightedCluster property families through traminex."""

    import pandas as _pd  # local import

    if not isinstance(seqdata, STSList):
        raise TypeError("seqdata must be a traminex STSList")
    X = np.asarray(seqdata.data, dtype=int)
    N, T = X.shape
    if N == 0:
        return _pd.DataFrame(index=np.arange(0))

    prop_set = set(_normalize_seqprop_properties(properties))
    row_index: Any = (
        _pd.Index(list(seqdata.ids)) if seqdata.ids is not None else _pd.RangeIndex(start=0, stop=N)
    )
    state_labels, label_by_code = _sequence_labels(seqdata)
    full_factor_levels = list(state_labels)
    for marker in (seqdata.nr, seqdata.void):
        if marker is not None and str(marker) not in full_factor_levels:
            full_factor_levels.append(str(marker))

    groups: list[tuple[str, _pd.DataFrame]] = []

    if "state" in prop_set:
        state_columns: dict[str, Any] = {}
        for t in range(T):
            values = [label_by_code.get(int(code), str(code)) for code in X[:, t]]
            state_columns[f"T{t + 1}"] = _pd.Categorical(
                values,
                categories=full_factor_levels,
            )
        groups.append(("state", _pd.DataFrame(state_columns, index=row_index)))

    include_spell_dur = "spell.dur" in prop_set
    include_spell_age = "spell.age" in prop_set
    if include_spell_dur or include_spell_age:
        dss = _traminex_seqdss(seqdata, include_missing=bool(with_missing))
        durations = np.asarray(
            _traminex_seqdur(seqdata, include_missing=bool(with_missing)),
            dtype=float,
        )
        spell_states = [
            str(label)
            for index, label in enumerate(dss.labels or dss.alphabet)
            if index not in {dss.void_code} and (bool(with_missing) or index not in {dss.nr_code})
        ]
        max_spells: dict[str, int] = {}
        for state in spell_states:
            code = next(
                (
                    index
                    for index, label in enumerate(dss.labels or dss.alphabet)
                    if str(label) == state
                ),
                -1,
            )
            max_spells[state] = max(
                (int(np.sum(row == code)) for row in np.asarray(dss.data, dtype=int)),
                default=0,
            )

        raw_names: list[str] = []
        kinds: list[str] = []
        for state in spell_states:
            for number in range(1, max_spells[state] + 1):
                if include_spell_dur:
                    raw_names.append(f"{state}_dur_{number}")
                    kinds.append("duration")
                if include_spell_age:
                    raw_names.append(f"{state}_age_{number}")
                    kinds.append("age")
        safe_names = _r_make_names(raw_names)
        spell_frame = _pd.DataFrame(index=row_index)
        for safe_name, kind in zip(safe_names, kinds, strict=True):
            if kind == "duration":
                spell_frame[safe_name] = np.zeros(N, dtype=float)
            else:
                spell_frame[safe_name] = _pd.Series(
                    _pd.array([_pd.NA] * N, dtype="boolean"),
                    index=row_index,
                )

        dss_labels = list(dss.labels or dss.alphabet)
        void_code = dss.void_code
        for i, row in enumerate(np.asarray(dss.data, dtype=int)):
            age = 0
            counts: dict[str, int] = {}
            for j, code in enumerate(row):
                if void_code is not None and int(code) == int(void_code):
                    break
                state = str(dss_labels[int(code)])
                counts[state] = counts.get(state, 0) + 1
                occurrence = counts[state]
                duration = float(durations[i, j])
                if include_spell_dur:
                    name = f"{state}_dur_{occurrence}"
                    if name not in spell_frame.columns:
                        spell_frame[name] = np.full(N, np.nan, dtype=float)
                    spell_frame.iloc[i, spell_frame.columns.get_loc(name)] = duration
                if include_spell_age:
                    name = f"{state}_age_{occurrence}"
                    if name not in spell_frame.columns:
                        spell_frame[name] = np.full(N, np.nan, dtype=float)
                    elif pd.api.types.is_bool_dtype(spell_frame[name].dtype):
                        spell_frame[name] = np.full(N, np.nan, dtype=float)
                    spell_frame.iloc[i, spell_frame.columns.get_loc(name)] = float(age)
                age += duration
        groups.append(("spell", spell_frame))

    if "duration" in prop_set:
        counts = np.asarray(
            _traminex_seqistatd(seqdata, include_missing=bool(with_missing)),
            dtype=int,
        )
        duration_columns: dict[str, np.ndarray] = {}
        for code, label in enumerate(seqdata.labels or seqdata.alphabet):
            if code == seqdata.void_code:
                continue
            if not with_missing and code == seqdata.nr_code:
                continue
            duration_columns[str(label)] = counts[:, code]
        if with_missing and str(seqdata.nr) not in duration_columns:
            duration_columns[str(seqdata.nr)] = np.zeros(N, dtype=int)
        groups.append(("duration", _pd.DataFrame(duration_columns, index=row_index, dtype=int)))

    def event_properties(tevent: str, method: str) -> _pd.DataFrame:
        events = _traminex_seqecreate(seqdata, tevent=tevent)
        frequent = _traminex_seqefsub(
            events,
            pmin_support=float(pmin_support),
            max_k=int(max_k),
        )
        frame = _traminex_seqeapplysub(frequent, method=method).reset_index(drop=True)
        if frame.shape[1] == 0:
            raise ValueError("no frequent subsequences satisfy pmin_support and max_k")
        if method == "age":
            frame = frame.replace(-1, np.nan)
        return frame.astype(float)

    if "transition" in prop_set:
        groups.append(("transition", event_properties("transition", "count")))
    if "pattern" in prop_set:
        groups.append(("pattern", event_properties("state", "count")))
    if "aftransition" in prop_set:
        groups.append(("AFtransition", event_properties("transition", "age")))
    if "afpattern" in prop_set:
        groups.append(("AFpattern", event_properties("state", "age")))

    if "complexity" in prop_set:
        groups.append(
            (
                "Complexity",
                _pd.DataFrame(
                    {
                        "C": np.asarray(_traminex_seqici(seqdata), dtype=float),
                        "Entropy": np.asarray(_traminex_seqient(seqdata), dtype=float),
                        "Turbulence": np.asarray(_traminex_seqST(seqdata), dtype=float),
                        "Trans.": np.asarray(_traminex_seqtransn(seqdata), dtype=float),
                    },
                    index=row_index,
                ),
            )
        )

    if not groups:
        return _pd.DataFrame(index=row_index)
    if len(groups) == 1 and not force_group_prefix:
        return groups[0][1]

    first_index = groups[0][1].index
    prefixed: list[_pd.DataFrame] = []
    for group_name, frame in groups:
        copied = frame.reset_index(drop=True).copy()
        copied.columns = [f"{group_name}.{column}" for column in copied.columns]
        prefixed.append(copied)
    result = _pd.concat(prefixed, axis=1)
    result.index = first_index
    return result


def seqpropclust(
    seqdata: np.ndarray,
    diss: np.ndarray,
    properties: Sequence[str] = (
        "state",
        "duration",
        "spell.age",
        "spell.dur",
        "transition",
        "pattern",
        "AFtransition",
        "AFpattern",
        "Complexity",
    ),
    other_prop=None,
    prop_only: bool = False,
    pmin_support: float = 0.05,
    max_k: int = -1,
    with_missing: bool = True,
    R: int = 1,
    weight_permutation: str = "diss",
    min_size: float = 0.01,
    max_depth: int = 5,
    maxcluster: Optional[int] = None,
    *,
    random_state: Optional[int] = None,
    perm_n_jobs: int | None = None,
    n_jobs: int | None = None,
    progressbar: bool = False,
    **kwargs,
) -> Dict:
    """Build a property-based clustering tree from sequence-derived features.

    Sequence-property extraction is governed by ``properties``,
    ``pmin_support``, ``max_k``, and ``with_missing``. Former Python-only
    feature controls that never affected runtime behavior have been removed;
    unknown keywords are rejected.

    ``perm_n_jobs`` and ``n_jobs`` are aliases for the permutation worker
    count. ``None`` resolves to one serial worker; both aliases cannot have
    non-``None`` values in one call, which raises ``TypeError``. Only ``1`` is
    currently available, and any other integer raises ``NotImplementedError``.
    Job values must be integers or ``None``; boolean, nonnumeric, and
    nonintegral values fail eagerly. ``progressbar`` must be a boolean and
    defaults to ``False``; requesting ``True`` raises ``NotImplementedError``
    because property-tree progress reporting is not available.
    """
    _validate_tree_execution_controls(
        perm_n_jobs=perm_n_jobs,
        n_jobs=n_jobs,
        progressbar=progressbar,
        owner="seqpropclust",
    )
    if not isinstance(prop_only, (bool, np.bool_)):
        raise TypeError("prop_only must be a boolean")
    if not isinstance(with_missing, (bool, np.bool_)):
        raise TypeError("with_missing must be a boolean")
    max_k_int = validate_integer_control(max_k, name="max_k")
    max_depth_int = validate_integer_control(max_depth, name="max_depth")
    R_int = validate_integer_control(R, name="R")
    pmin_support_value = float(pmin_support)
    min_size_value = float(min_size)
    if not np.isfinite(pmin_support_value) or not 0.0 <= pmin_support_value <= 1.0:
        raise ValueError("pmin_support must be between zero and one")
    if max_k_int != -1 and max_k_int < 1:
        raise ValueError("max_k must be -1 or a positive integer")
    if not np.isfinite(min_size_value) or min_size_value <= 0.0:
        raise ValueError("min_size must be a positive finite number")
    if R_int < 0:
        raise ValueError("R must be non-negative")
    maxcluster_int = (
        None if maxcluster is None else validate_integer_control(maxcluster, name="maxcluster")
    )
    if random_state is not None:
        random_state = validate_integer_control(random_state, name="random_state")
        if random_state < 0:
            raise ValueError("random_state must be non-negative")
    supported_kwargs = {
        "weights",
        "minSize",
        "maxdepth",
        "perm_R",
        "pval",
        "weight_permutation",
        "squared",
        "first",
        "weighted",
    }
    unknown_kwargs = sorted(set(kwargs) - supported_kwargs)
    if unknown_kwargs:
        raise TypeError(f"unexpected seqpropclust keyword(s): {', '.join(unknown_kwargs)}")
    for flag_name in ("squared", "weighted"):
        if flag_name in kwargs and not isinstance(kwargs[flag_name], (bool, np.bool_)):
            raise TypeError(f"{flag_name} must be a boolean")
    prop_list = _normalize_seqprop_properties(tuple(properties))
    weights_for_sequence = kwargs.get("weights")
    if isinstance(seqdata, STSList):
        sequence_object = seqdata
    else:
        sequence_object = _traminex_seqdef(
            np.asarray(seqdata),
            weights=weights_for_sequence,
        )

    built_props = _extract_seq_properties(
        sequence_object,
        properties=prop_list,
        pmin_support=pmin_support_value,
        max_k=max_k_int,
        with_missing=with_missing,
        force_group_prefix=other_prop is not None,
    )

    other_df = None if other_prop is None else pd.DataFrame(other_prop)
    if other_df is not None and other_df.shape[0] != sequence_object.nseq:
        raise ValueError("other_prop must have same number of rows as seqdata")

    if other_df is None:
        props_df = built_props
    elif built_props.shape[1] == 0:
        props_df = other_df
    else:
        other_df = other_df.reset_index(drop=True).copy()
        other_df.columns = [f"other.prop.{column}" for column in other_df.columns]
        props_df = pd.concat(
            [other_df, built_props.reset_index(drop=True)],
            axis=1,
        )

    if props_df.shape[1] == 0 and not prop_list and other_df is None:
        raise ValueError("no sequence or user-defined properties were selected")
    if prop_only:
        return props_df

    props_df = props_df.copy()
    for column in props_df.columns:
        if isinstance(props_df[column].dtype, pd.CategoricalDtype):
            props_df[column] = props_df[column].cat.remove_unused_categories()
        elif isinstance(props_df[column].dtype, pd.BooleanDtype):
            props_df[column] = props_df[column].astype(object)

    options = dict(kwargs)
    options.pop("weights", None)
    min_size = float(options.pop("minSize", min_size_value))
    if not np.isfinite(min_size) or min_size <= 0.0:
        raise ValueError("min_size must be a positive finite number")
    max_depth = validate_integer_control(options.pop("maxdepth", max_depth_int), name="max_depth")
    perm_R = validate_integer_control(options.pop("perm_R", R_int), name="R")
    pval = float(options.pop("pval", 0.01))
    if perm_R < 0:
        raise ValueError("R must be non-negative")
    if not np.isfinite(pval) or not 0.0 <= pval <= 1.0:
        raise ValueError("pval must be between zero and one")
    weight_permutation = str(options.pop("weight_permutation", weight_permutation))
    squared = options.pop("squared", False)
    first = options.pop("first", None)
    weighted = options.pop("weighted", True)
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"unexpected seqpropclust keyword(s): {names}")

    complete_random_sampling = (
        weight_permutation == "random-sampling" and weighted and sequence_object.weights is not None
    )
    delegate_permutation = "diss" if weight_permutation == "random-sampling" else weight_permutation
    tree = _traminex_seqtree(
        sequence_object,
        props_df,
        diss=np.asarray(diss, dtype=float),
        weighted=weighted,
        min_size=min_size,
        max_depth=max_depth,
        R=1 if complete_random_sampling else perm_R,
        pval=1.0 if complete_random_sampling else pval,
        weight_permutation=delegate_permutation,
        squared=squared,
        first=first,
        random_state=random_state,
    )
    if complete_random_sampling:
        _complete_random_sampling_tree(
            tree,
            np.asarray(diss, dtype=float) ** 2 if squared else np.asarray(diss, dtype=float),
            R=perm_R,
            pval=pval,
            random_state=random_state,
        )
    out = _tree_adapter(
        tree,
        np.asarray(diss, dtype=float),
        classes=("seqtreeclust", "dtclust", "seqtree", "disstree", "list"),
    )
    out["info"]["random_state"] = random_state
    if maxcluster_int is not None:
        out = dtprune(out, maxcluster_int)
    return out


def _count_leaves(root: Dict) -> int:
    if root.get("kids") is None:
        return 1
    return _count_leaves(root["kids"][0]) + _count_leaves(root["kids"][1])


def _pruned_root_for_k(root: Dict, k: int, *, min_gain: float | None = None) -> Dict:
    def prune(node: Dict) -> Dict:
        node_copy = {
            "id": int(node["id"]),
            "info": {
                key: (np.asarray(value).copy() if isinstance(value, np.ndarray) else value)
                for key, value in node["info"].items()
            },
            "split": None,
            "kids": None,
        }
        if node.get("kids") is None:
            return node_copy
        child_schedule = int(node["kids"][0]["info"].get("splitschedule", 0))
        gain = float(node.get("split", {}).get("gain", 0.0))
        if child_schedule > int(k) or (min_gain is not None and gain < float(min_gain)):
            return node_copy
        node_copy["split"] = {
            key: (
                dict(value)
                if isinstance(value, dict)
                else (
                    np.asarray(value).copy()
                    if isinstance(value, np.ndarray)
                    else list(value) if isinstance(value, list) else value
                )
            )
            for key, value in node["split"].items()
        }
        node_copy["kids"] = [prune(node["kids"][0]), prune(node["kids"][1])]
        return node_copy

    return prune(root)


def dtcut(st: Dict, k: int, labels: bool = True):
    """Cut a property tree at WeightedCluster split-schedule step ``k``."""
    if (
        not isinstance(st, dict)
        or "root" not in st
        or "data" not in st
        or "fitted" not in st
        or not isinstance(st.get("root"), dict)
    ):
        raise TypeError("st must be a property-tree mapping with root, data, and fitted")
    if not isinstance(labels, (bool, np.bool_)):
        raise TypeError("labels must be a boolean")
    if not isinstance(k, (int, float, np.integer, np.floating)) or not np.isfinite(k):
        raise ValueError("k must be a finite number")
    k_value = float(k)
    max_k = _count_leaves(st["root"])
    if k_value > max_k:
        raise ValueError(f"The maximum number of groups is {max_k}")
    n_obs = int(st["data"].shape[0])
    leaf_ids = np.full(n_obs, int(st["root"]["id"]), dtype=int)
    if k_value == 1:
        return leaf_ids
    if k_value < 1:
        if not labels:
            return leaf_ids
        return np.full(n_obs, None, dtype=object)

    def assign(node: Dict) -> None:
        if float(node["info"].get("splitschedule", 0)) <= k_value:
            leaf_ids[np.asarray(node["info"]["ind"], dtype=int)] = int(node["id"])
        if node.get("kids") is not None:
            assign(node["kids"][0])
            assign(node["kids"][1])

    assign(st["root"])
    if not labels:
        return leaf_ids
    label_map = dtlabels(st)
    return np.asarray([label_map[str(int(leaf_id))] for leaf_id in leaf_ids], dtype=object)


def _r_format_number(value: float, digits: int = 5) -> str:
    """Reproduce R's ``format(x, digits=digits)`` for one real number.

    R's ``dtlabels`` renders a numeric split with
    ``format(x, digits = getOption("digits") - 2)`` (R/propclustering.R:254-255),
    which is 5 digits under R's default ``digits`` option of 7. R rounds to that
    many significant digits, drops trailing zeros, and then prints fixed
    notation unless it is wider than scientific notation (``scipen`` 0). The
    ``".5g"`` format this replaces switches to exponents on its own rule, so
    ``1200000`` became ``1.2e+06`` where R prints ``1200000``, and ``1e-04``
    became ``0.0001`` (2026-09-21 targeted audit, L-8). Checked equal to live
    R's ``format`` on 5,474 values spanning 1e-100 to 1e100, both signs, and
    the five-digit rounding boundaries.
    """
    x = float(value)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0.0:
        return "0"
    negative = 1 if x < 0 else 0
    mantissa, exponent = format(abs(x), f".{digits - 1}e").split("e")
    power = int(exponent)
    significant = max(1, len(mantissa.replace(".", "").rstrip("0")))
    decimals = max(0, significant - power - 1)
    fixed_width = (
        negative + (power + 1 if power >= 0 else 1) + decimals + (1 if decimals > 0 else 0)
    )
    scientific_width = (
        negative
        + (significant + 1 if significant > 1 else significant)
        + (5 if abs(power) >= 100 else 4)
    )
    if fixed_width <= scientific_width:
        return f"{x:.{decimals}f}"
    return f"{x:.{significant - 1}e}"


def dtlabels(tree: Dict):
    """Return R-compatible root-to-node labels for all non-root tree nodes."""
    labels: dict[str, str] = {}

    def split_labels(split: Dict) -> tuple[str, str]:
        variable = str(split["var"])
        if split.get("breaks") is not None:
            threshold = _r_format_number(split["breaks"])
            sides = [f"{variable} <= {threshold}", f"{variable} > {threshold}"]
        else:
            values = list(split.get("labels", []))
            index = split.get("index")
            if index is None:
                left = list(split.get("levels", []))
                all_values = list(pd.unique(tree["data"][variable]))
                right = [value for value in all_values if value not in left]
            else:
                left = [value for value, side in zip(values, index, strict=True) if int(side) == 1]
                right = [value for value, side in zip(values, index, strict=True) if int(side) == 2]
            sides = [
                f"{variable} [{', '.join(map(str, left))}]",
                f"{variable} [{', '.join(map(str, right))}]",
            ]
        na_group = split.get("na_group")
        if na_group is not None:
            side = int(na_group) - 1
            sides[side] = f"{sides[side]} with NA"
        return sides[0], sides[1]

    def visit(node: Dict, path: Sequence[str]) -> None:
        if node.get("kids") is None:
            return
        left_text, right_text = split_labels(node["split"])
        left_path = list(path) + [left_text]
        right_path = list(path) + [right_text]
        labels[str(node["kids"][0]["id"])] = " & ".join(left_path)
        labels[str(node["kids"][1]["id"])] = " & ".join(right_path)
        visit(node["kids"][0], left_path)
        visit(node["kids"][1], right_path)

    visit(tree["root"], [])
    return labels


def dtprune(st: Dict, k: int, *, min_gain: float | None = None):
    """Prune a property tree structurally to ``k`` scheduled leaves."""
    k_int = validate_integer_control(k, name="k")
    if min_gain is not None and not np.isfinite(float(min_gain)):
        raise ValueError("min_gain must be finite")
    if k_int < 1:
        raise ValueError("k must be >= 1")
    max_k = _count_leaves(st["root"])
    if k_int > max_k:
        raise ValueError(f"The maximum number of groups is {max_k}")
    tree = dict(st)
    if "info" in tree and isinstance(tree["info"], dict):
        tree["info"] = dict(tree["info"])
    new_root = _pruned_root_for_k(st["root"], k_int, min_gain=min_gain)
    tree["root"] = new_root
    tree["fitted"] = _leaf_membership(new_root, int(st["data"].shape[0]))
    tree.setdefault("info", {})
    tree["info"]["prune"] = k_int
    if "_diss" in tree:
        weight_permutation = str(tree["info"].get("weight_permutation", "none"))
        # WeightedCluster 2.0's dtprune reads ``st$info$weights``, but disstree
        # stores weights at top level. The resulting pruned adjustment is
        # therefore unweighted for every permutation mode.
        weights = None
        adjustment = _traminex_dissassoc(
            np.asarray(tree["_diss"], dtype=float),
            tree["fitted"],
            weights=weights,
            R=int(tree["info"].get("R", 1)),
            weight_permutation=(
                "replicate" if weight_permutation == "none" else weight_permutation
            ),
            random_state=tree["info"].get("random_state"),
        )
        tree["adjustment"] = adjustment
        tree["info"]["adjustment"] = adjustment
    return tree

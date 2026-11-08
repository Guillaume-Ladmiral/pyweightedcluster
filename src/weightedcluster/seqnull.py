from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Callable, Dict, List, Optional

import numpy as np
from traminex import (
    STSList,
    seqdss,
    seqdur,
    seqlength,
    seqstatd,
    seqstatf,
    seqtrate,
)
from traminex import (
    seqdist as traminex_seqdist,
)

from ._parallel import run_task_batch
from ._seqdist import _validate_observed_substitution_costs
from ._utils import generate_random_seed
from ._validation import cut_linkage_by_merge_count, validate_integer_control, validate_kvals
from .kmedoids import wcKMedRange
from .quality import wcClusterQuality
from .schemas import SeqNullCQIResult, coerce_stats_table, validate_clustrange_result

_SEQNULLCQI_R_STATS = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")
_SEQNULL_MODELS = frozenset(
    {"combined", "duration", "sequencing", "stateindep", "markov", "userpos"}
)
_SEQNULL_CONTROL_KEYS = frozenset(
    {
        "imp_trans",
        "imp_trans_limit",
        "trate",
        "begin",
        "time_varying",
        "weighted",
        "replay_plan",
    }
)


class _SeqNullReplay:
    def __init__(self, plan: Sequence[Mapping[str, object]]) -> None:
        if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
            raise TypeError("replay_plan must be a sequence of mappings")
        if any(not isinstance(draw, Mapping) for draw in plan):
            raise TypeError("replay_plan must be a sequence of mappings")
        self._draws = [
            {
                "kind": str(draw.get("kind", "")),
                "values": np.asarray(draw.get("values", []), dtype=object).reshape(-1).copy(),
            }
            for draw in plan
        ]
        self._index = 0

    def take(self, kind: str, size: int) -> np.ndarray:
        if self._index >= len(self._draws):
            raise ValueError("seqnull replay plan exhausted")
        draw = self._draws[self._index]
        self._index += 1
        if draw["kind"] != kind:
            raise ValueError(
                f"seqnull replay draw {self._index} has kind {draw['kind']!r}; "
                f"expected {kind!r}"
            )
        values = np.asarray(draw["values"], dtype=object).reshape(-1)
        if values.size != int(size):
            raise ValueError(
                f"seqnull replay draw {self._index} has {values.size} values; "
                f"expected {int(size)}"
            )
        return values.copy()

    def finish(self) -> None:
        if self._index != len(self._draws):
            raise ValueError(
                f"seqnull replay plan has {len(self._draws) - self._index} unused draw(s)"
            )


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float).reshape(-1)
    if p.size == 0:
        return p
    p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)
    s = float(p.sum())
    if s <= 0.0:
        return np.full(p.size, 1.0 / float(p.size), dtype=float)
    return p / s


def _sampling_probs(probs: np.ndarray, *, context: str) -> np.ndarray:
    p = np.asarray(probs, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError(f"{context} contains no probabilities")
    if np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError(f"{context} contains invalid probabilities")
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError(f"{context} has too few positive probabilities")
    return p / total


def _all_sampling_rows_valid(probs: np.ndarray) -> bool:
    rows = np.asarray(probs, dtype=float)
    return bool(
        rows.ndim == 2
        and np.all(np.isfinite(rows))
        and np.all(rows >= 0.0)
        and np.all(np.sum(rows, axis=1) > 0.0)
    )


def _encode_states(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states, inv = np.unique(X, return_inverse=True)
    return inv.reshape(X.shape).astype(np.int32), states


def _resolve_begin_distribution(
    *,
    begin: str | np.ndarray,
    freqs_t: np.ndarray,
    p_glob: np.ndarray,
    n_states: int,
) -> np.ndarray:
    if isinstance(begin, str):
        b = str(begin).lower()
        if b == "freq":
            if freqs_t.shape[0] == 0:
                return np.full(n_states, 1.0 / float(n_states), dtype=float)
            return _sampling_probs(freqs_t[0], context="begin")
        if b == "ofreq":
            return _sampling_probs(p_glob, context="begin")
        raise ValueError("unknown begin string")
    p0 = np.asarray(begin, dtype=float).reshape(-1)
    if p0.size != n_states:
        raise ValueError(f"begin must have length {n_states}")
    return _sampling_probs(p0, context="begin")


def _is_default_trate_arg(trate: str | np.ndarray) -> bool:
    return isinstance(trate, str) and str(trate).lower() == "trate"


def _is_default_begin_arg(begin: str | np.ndarray) -> bool:
    return isinstance(begin, str) and str(begin).lower() == "freq"


def _reject_seqnull_arg(model: str, arg_name: str, reason: str) -> None:
    raise ValueError(f"{arg_name} {reason} for seqnull(model='{model}')")


def _normalize_seqnull_controls(
    *,
    model: object,
    imp_trans: object,
    imp_trans_limit: object,
    trate: str | np.ndarray,
    begin: str | np.ndarray,
    time_varying: object,
    weighted: object,
    validate_applicability: bool = True,
) -> tuple[str, Mapping[object, object] | None, float, bool | None, bool]:
    """Validate public model controls before any empty/replay shortcut."""
    model_l = str(model).lower()
    if model_l not in _SEQNULL_MODELS:
        raise ValueError("unknown model")
    if imp_trans is not None and not isinstance(imp_trans, Mapping):
        raise TypeError("imp_trans must be a mapping or None")
    if isinstance(imp_trans_limit, (bool, np.bool_)) or not isinstance(imp_trans_limit, Real):
        raise TypeError("imp_trans_limit must be a finite number")
    imp_trans_limit_value = float(imp_trans_limit)
    if not np.isfinite(imp_trans_limit_value):
        raise ValueError("imp_trans_limit must be a finite number")
    if time_varying is not None and not isinstance(time_varying, (bool, np.bool_)):
        raise TypeError("time_varying must be a boolean or None")
    if not isinstance(weighted, (bool, np.bool_)):
        raise TypeError("weighted must be a boolean")
    time_varying_value = None if time_varying is None else bool(time_varying)
    weighted_value = bool(weighted)
    if validate_applicability:
        _validate_seqnull_args(
            model=model_l,
            imp_trans=None if imp_trans is None else dict(imp_trans),
            imp_trans_limit=imp_trans_limit_value,
            trate=trate,
            begin=begin,
            time_varying=time_varying_value,
        )
    return (
        model_l,
        None if imp_trans is None else dict(imp_trans),
        imp_trans_limit_value,
        time_varying_value,
        weighted_value,
    )


def _validate_seqnull_args(
    *,
    model: str,
    imp_trans: Optional[Dict[int, int]],
    imp_trans_limit: float,
    trate: str | np.ndarray,
    begin: str | np.ndarray,
    time_varying: Optional[bool],
) -> None:
    spell_models = {"combined", "duration", "sequencing"}
    if model in spell_models:
        if not _is_default_trate_arg(trate):
            _reject_seqnull_arg(model, "trate", "is ignored")
        if not _is_default_begin_arg(begin):
            _reject_seqnull_arg(model, "begin", "is ignored")
        if time_varying is not None:
            _reject_seqnull_arg(model, "time_varying", "is ignored")
        return

    if model == "stateindep":
        if imp_trans is not None:
            _reject_seqnull_arg(model, "imp_trans", "is only supported by spell-based models")
        if float(imp_trans_limit) != -1.0:
            _reject_seqnull_arg(model, "imp_trans_limit", "is only supported by spell-based models")
        if not _is_default_trate_arg(trate):
            _reject_seqnull_arg(model, "trate", "is ignored")
        if not _is_default_begin_arg(begin):
            _reject_seqnull_arg(model, "begin", "is ignored")
        if time_varying is False:
            _reject_seqnull_arg(model, "time_varying", "must be True or omitted")
        return

    if model == "markov":
        if imp_trans is not None:
            _reject_seqnull_arg(model, "imp_trans", "is only supported by spell-based models")
        if float(imp_trans_limit) != -1.0:
            _reject_seqnull_arg(model, "imp_trans_limit", "is only supported by spell-based models")
        if not _is_default_trate_arg(trate):
            _reject_seqnull_arg(model, "trate", "is ignored")
        if not _is_default_begin_arg(begin):
            _reject_seqnull_arg(model, "begin", "is ignored")
        if time_varying is True:
            _reject_seqnull_arg(model, "time_varying", "must be False or omitted")
        return

    if model == "userpos":
        if imp_trans is not None:
            _reject_seqnull_arg(model, "imp_trans", "is only supported by spell-based models")
        if float(imp_trans_limit) != -1.0:
            _reject_seqnull_arg(model, "imp_trans_limit", "is only supported by spell-based models")
        return


def _state_lookup(states: np.ndarray) -> dict[object, int]:
    lk: dict[object, int] = {}
    for i, st in enumerate(states.tolist()):
        lk[st] = int(i)
        lk[str(st)] = int(i)
    return lk


def _state_code(value: object, lookup: dict[object, int]) -> Optional[int]:
    if value in lookup:
        return int(lookup[value])
    try:
        iv = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return int(lookup[iv]) if iv in lookup else None


def _seqnull_spell(
    X: np.ndarray,
    *,
    randomize: str,
    imp_trans: Optional[Dict[int, int]],
    imp_trans_limit: float,
    states_values: np.ndarray,
    rng: np.random.Generator,
    seq_obj: STSList | None = None,
    replay: _SeqNullReplay | None = None,
) -> np.ndarray:
    N, T = X.shape
    n_states = int(states_values.size)
    if N == 0 or T == 0 or n_states == 0:
        return np.zeros_like(X)

    if seq_obj is None:
        seq_obj = STSList(
            data=np.asarray(X, dtype=int).copy(),
            alphabet=[str(value) for value in states_values.tolist()],
            labels=[str(value) for value in states_values.tolist()],
        )
    dss_obj = seqdss(seq_obj)
    dur_matrix = np.asarray(seqdur(seq_obj), dtype=float)
    max_spells = int(dur_matrix.shape[1]) if dur_matrix.ndim == 2 else 0
    dss_matrix = np.asarray(dss_obj.data[:, :max_spells], dtype=np.int32)
    valid_spell = np.isfinite(dur_matrix)
    spell_states_arr = dss_matrix.reshape(-1, order="F")[valid_spell.reshape(-1, order="F")]
    spell_durs_arr = (
        dur_matrix.reshape(-1, order="F")[valid_spell.reshape(-1, order="F")]
        .astype(np.int32)
        .copy()
    )
    if spell_states_arr.size == 0:
        return np.zeros_like(X)
    spells_by_seq = [
        (
            dss_matrix[i, valid_spell[i]].astype(np.int32, copy=True),
            dur_matrix[i, valid_spell[i]].astype(np.int32, copy=True),
        )
        for i in range(N)
    ]
    stateprop = _normalize_probs(np.bincount(spell_states_arr, minlength=n_states).astype(float))

    if randomize == "sequencing":
        if replay is None:
            picks = rng.choice(spell_states_arr.size, size=N * T, replace=True)
        else:
            picks = np.asarray(replay.take("sample", N * T), dtype=int) - 1
            if np.any((picks < 0) | (picks >= spell_states_arr.size)):
                raise ValueError("seqnull replay spell indexes are out of bounds")
        dss2 = spell_states_arr[picks].reshape(N, T, order="F")
        dur2 = spell_durs_arr[picks].reshape(N, T, order="F")
        valid = np.ones((N, T), dtype=bool)
    else:
        if randomize == "combined":
            tr_rates = np.asarray(
                seqtrate(dss_obj, weighted=True, time_varying=False, as_dataframe=False),
                dtype=float,
            )
            tr = np.tile(stateprop.reshape(1, -1), (n_states, 1))
            np.fill_diagonal(tr, 0.0)
            if float(imp_trans_limit) >= 0.0:
                tr[tr_rates <= float(imp_trans_limit)] = 0.0
            if imp_trans:
                lookup = _state_lookup(states_values)
                for k, vals in imp_trans.items():
                    from_code = _state_code(k, lookup)
                    if from_code is None:
                        continue
                    if isinstance(vals, (list, tuple, np.ndarray, set)):
                        to_list = list(vals)
                    else:
                        to_list = [vals]
                    for v in to_list:
                        to_code = _state_code(v, lookup)
                        if to_code is not None:
                            tr[from_code, to_code] = 0.0

            replay_lookup = _state_lookup(states_values)

            def sample_states(size: int, probs: np.ndarray) -> np.ndarray:
                if replay is None:
                    return rng.choice(
                        n_states,
                        size=size,
                        replace=True,
                        p=_sampling_probs(probs, context="transition"),
                    ).astype(np.int32)
                raw = replay.take("sample", size)
                values: list[int] = []
                for item in raw.tolist():
                    code = _state_code(item, replay_lookup)
                    if code is None:
                        raise ValueError(f"seqnull replay contains unknown state {item!r}")
                    values.append(code)
                return np.asarray(values, dtype=np.int32)

            dss2 = np.zeros((N, T), dtype=np.int32)
            dss2[:, 0] = sample_states(N, stateprop)
            for t in range(1, T):
                prev = dss2[:, t - 1]
                for s in range(n_states):
                    mask = prev == s
                    n_case = int(np.sum(mask))
                    if n_case > 0:
                        dss2[mask, t] = sample_states(n_case, tr[s, :])
            valid = np.ones((N, T), dtype=bool)
        elif randomize == "duration":
            dss2 = np.full((N, max_spells), -1, dtype=np.int32)
            valid = np.zeros((N, max_spells), dtype=bool)
            for i, (st, _dr) in enumerate(spells_by_seq):
                nsp = int(st.size)
                if nsp > 0:
                    dss2[i, :nsp] = st[:nsp]
                    valid[i, :nsp] = True
        else:
            raise ValueError("unknown randomize mode")

        if randomize == "combined":
            if replay is None:
                sampled_durations = rng.choice(spell_durs_arr, size=N * T, replace=True).astype(
                    np.int32
                )
            else:
                sampled_durations = np.asarray(replay.take("sample", N * T), dtype=np.int32)
            dur2 = sampled_durations.reshape(N, T, order="F")
        else:
            nspell = valid.sum(axis=1).astype(int)
            draw_count = int(dss2.size)
            if replay is None:
                durprop = rng.random(draw_count, dtype=float).reshape(dss2.shape, order="F")
            else:
                durprop = np.asarray(replay.take("runif", draw_count), dtype=float).reshape(
                    dss2.shape, order="F"
                )
            durprop[~valid] = np.nan
            rowsum = np.nansum(durprop, axis=1, keepdims=True)
            rowsum = np.where(rowsum > 0.0, rowsum, 1.0)
            durprop = 1.0 + ((float(T) - nspell.reshape(-1, 1)) * durprop / rowsum)
            dur2f = np.floor(durprop)
            dur2f[~valid] = np.nan
            durrest = durprop - dur2f
            durtot = np.nansum(dur2f, axis=1).astype(int)
            for i in range(N):
                rem = int(T - durtot[i])
                if rem <= 0:
                    continue
                idx = np.where(valid[i])[0]
                if idx.size == 0:
                    continue
                row_for_order = np.where(valid[i], durrest[i], -np.inf)
                order_1b = np.argsort(-row_for_order, kind="stable") + 1
                cond = order_1b <= rem
                dur2f[i, cond] += 1.0
            dur2 = np.nan_to_num(dur2f, nan=0.0).astype(np.int32)

    Y = np.zeros_like(X, dtype=np.int32)
    for i in range(N):
        mask = (dss2[i, :] >= 0) & (dur2[i, :] > 0)
        if not np.any(mask):
            raise ValueError("seqnull generated no spells")
        seq = np.repeat(dss2[i, mask], dur2[i, mask].astype(int))
        if seq.size < T:
            raise ValueError("seqnull generated a sequence shorter than the input width")
        Y[i, :] = seq[:T]
    return Y


def _seqnull_pos(
    X: np.ndarray,
    *,
    prob: str | np.ndarray,
    begin: str | np.ndarray,
    time_varying: bool,
    rng: np.random.Generator,
    weighted: bool = True,
    seq_obj: STSList | None = None,
    replay: _SeqNullReplay | None = None,
    backend_seed: int | None = None,
) -> np.ndarray:
    N, T = X.shape
    if seq_obj is None:
        n_states = int(X.max()) + 1 if X.size else 0
        seq_obj = STSList(
            data=np.asarray(X, dtype=int).copy(),
            alphabet=[str(index) for index in range(n_states)],
            labels=[str(index) for index in range(n_states)],
        )
    else:
        n_states = len(seq_obj.alphabet)
    if N == 0 or T == 0 or n_states == 0:
        return np.zeros_like(X)

    lengths = np.asarray(seqlength(seq_obj), dtype=int)
    if lengths.size and np.any(lengths != T):
        raise ValueError("positional seqnull models require equal full sequence lengths")
    maxage = int(np.max(lengths)) if lengths.size else T
    n_trans = max(0, maxage - 1)
    freqs_t = np.asarray(seqstatd(seq_obj, weighted=bool(weighted))["Frequencies"], dtype=float)
    frequency_table = seqstatf(seq_obj, weighted=bool(weighted))
    p_glob = np.asarray(frequency_table["Percent"], dtype=float) / 100.0
    p0 = _resolve_begin_distribution(begin=begin, freqs_t=freqs_t, p_glob=p_glob, n_states=n_states)

    if isinstance(prob, str):
        pm = prob.lower()
        if pm == "trate":
            if bool(time_varying):
                agedtr = np.asarray(
                    seqtrate(
                        seq_obj,
                        time_varying=True,
                        weighted=bool(weighted),
                        as_dataframe=False,
                    ),
                    dtype=float,
                )
            else:
                tr = np.asarray(
                    seqtrate(
                        seq_obj,
                        time_varying=False,
                        weighted=bool(weighted),
                        as_dataframe=False,
                    ),
                    dtype=float,
                )
                agedtr = np.repeat(tr[:, :, None], n_trans, axis=2)
        elif pm == "freq":
            agedtr = np.zeros((n_states, n_states, n_trans), dtype=float)
            if bool(time_varying):
                for t in range(n_trans):
                    agedtr[:, :, t] = np.tile(freqs_t[t + 1], (n_states, 1))
            else:
                agedtr[:, :, :] = np.tile(p_glob, (n_states, 1))[:, :, None]
        else:
            raise ValueError("unknown method to compute transition rate")
    else:
        arr = np.asarray(prob, dtype=float)
        if arr.ndim == 3:
            if arr.shape != (n_states, n_states, n_trans):
                raise ValueError(
                    "Transition rate array must have shape " f"({n_states}, {n_states}, {n_trans})"
                )
            agedtr = arr.copy()
        elif arr.ndim == 2:
            if arr.shape != (n_states, n_states):
                raise ValueError(f"Transition rate matrix must have shape ({n_states}, {n_states})")
            agedtr = np.repeat(arr[:, :, None], n_trans, axis=2)
        else:
            raise ValueError("Transition rate should be 2D or 3D array")

    if replay is None and backend_seed is not None:
        Y = np.zeros((N, T), dtype=np.int32)

        def pick(probs: np.ndarray, uniform: float) -> int:
            normalized = _sampling_probs(probs, context="transition")
            index = int(np.searchsorted(np.cumsum(normalized), uniform, side="left"))
            return min(index, normalized.size - 1)

        for i in range(N):
            random = np.random.RandomState(int(backend_seed) + i)
            Y[i, 0] = pick(p0, float(random.random_sample()))
            for t in range(1, T):
                Y[i, t] = pick(
                    agedtr[Y[i, t - 1], :, t - 1],
                    float(random.random_sample()),
                )
        return Y

    def random_state(size: int, probs: np.ndarray) -> np.ndarray:
        normalized = _sampling_probs(probs, context="transition")
        if replay is None:
            return rng.choice(n_states, size=size, replace=True, p=normalized).astype(np.int32)
        values = np.asarray(replay.take("sample.int", size), dtype=int)
        if np.any((values < 1) | (values > n_states)):
            raise ValueError("seqnull replay positional states must be one-based state indexes")
        return values.astype(np.int32) - 1

    Y = np.zeros((N, T), dtype=np.int32)
    Y[:, 0] = random_state(N, p0)
    for t in range(1, T):
        trans_t = agedtr[:, :, t - 1]
        prev = Y[:, t - 1]
        for s in range(n_states):
            mask = prev == s
            n_case = int(np.sum(mask))
            if n_case > 0:
                Y[mask, t] = random_state(n_case, trans_t[s, :])
    return Y


def _contains_missing_values(values: np.ndarray) -> bool:
    if np.issubdtype(values.dtype, np.floating):
        return bool(np.any(~np.isfinite(values)))
    if values.dtype.kind not in {"O", "U", "S"}:
        return False
    for value in values.reshape(-1).tolist():
        if value is None:
            return True
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return True
    return False


def _prepare_seqnull_input(
    seqdata: np.ndarray | STSList,
) -> tuple[np.ndarray, np.ndarray, STSList, bool, np.dtype | None]:
    if isinstance(seqdata, STSList):
        X = np.asarray(seqdata.data, dtype=np.int32)
        if X.ndim != 2:
            raise ValueError("seqdata must be a 2D sequence object")
        if seqdata.nr_code is not None and np.any(X == int(seqdata.nr_code)):
            raise ValueError("missing data are not supported in seqnull")
        excluded_codes = {
            int(code) for code in (seqdata.nr_code, seqdata.void_code) if code is not None
        }
        real_codes = [
            index for index in range(len(seqdata.alphabet)) if index not in excluded_codes
        ]
        if real_codes != list(range(len(real_codes))):
            raise ValueError("seqnull requires real-state codes before missing and void codes")
        states_values = np.asarray([seqdata.alphabet[index] for index in real_codes], dtype=object)
        return X.copy(), states_values, seqdata, True, None

    raw = np.asarray(seqdata)
    if raw.ndim != 2:
        raise ValueError("seqdata must be a 2D array (N, T)")
    if _contains_missing_values(raw):
        raise ValueError("missing data are not supported in seqnull")
    Xc, states_values = _encode_states(raw)
    labels = [str(value) for value in states_values.tolist()]
    seq_obj = STSList(
        data=Xc.copy(),
        alphabet=labels.copy(),
        labels=labels,
    )
    return Xc, states_values, seq_obj, False, raw.dtype


def _restore_seqnull_output(
    Yc: np.ndarray,
    *,
    states_values: np.ndarray,
    seq_obj: STSList,
    return_sts: bool,
    dtype: np.dtype | None,
    preserve_weights: bool,
) -> np.ndarray | STSList:
    codes = np.asarray(Yc, dtype=np.int32)
    if not return_sts:
        values = states_values[codes]
        return np.asarray(values, dtype=dtype)
    result = seq_obj.copy()
    result.data = codes.copy()
    if not preserve_weights:
        result.weights = None
    return result


def seqnull(
    seqdata: np.ndarray | STSList,
    *,
    model: str = "combined",
    imp_trans: Optional[Dict[int, int]] = None,
    imp_trans_limit: float = -1.0,
    trate: str | np.ndarray = "trate",
    begin: str | np.ndarray = "freq",
    time_varying: Optional[bool] = None,
    weighted: bool = True,
    random_state: Optional[int] = None,
    replay_plan: Sequence[Mapping[str, object]] | None = None,
) -> np.ndarray | STSList:
    """Generate nonclustered sequence data according to WeightedCluster null models."""
    X, states_values, seq_obj, return_sts, source_dtype = _prepare_seqnull_input(seqdata)
    (
        model_l,
        imp_trans_value,
        imp_trans_limit_value,
        time_varying_value,
        weighted_value,
    ) = _normalize_seqnull_controls(
        model=model,
        imp_trans=imp_trans,
        imp_trans_limit=imp_trans_limit,
        trate=trate,
        begin=begin,
        time_varying=time_varying,
        weighted=weighted,
    )
    random_state_value = (
        None
        if random_state is None
        else validate_integer_control(random_state, name="random_state")
    )
    if random_state_value is not None and random_state_value < 0:
        raise ValueError("random_state must be non-negative")
    seed = generate_random_seed(random_state_value, 0)
    rng = np.random.default_rng(seed)
    replay = None if replay_plan is None else _SeqNullReplay(replay_plan)
    N, T = X.shape
    if N == 0 or T == 0:
        if replay is not None:
            replay.finish()
        if return_sts:
            return seq_obj.copy()
        return np.asarray(seqdata).copy()
    if T < 2:
        raise ValueError("seqnull requires at least two sequence positions")
    if model_l in {"stateindep", "markov", "userpos"}:
        lengths = np.asarray(seqlength(seq_obj), dtype=int)
        if lengths.size and np.any(lengths != T):
            raise ValueError("positional seqnull models require equal full sequence lengths")
    n_states = int(states_values.size)
    hardcoded_stats = np.asarray(seqstatd(seq_obj, weighted=True)["Frequencies"], dtype=float)
    hardcoded_global = np.asarray(seqstatf(seq_obj, weighted=True)["Percent"], dtype=float) / 100.0

    if model_l in ("combined", "duration", "sequencing"):
        Yc = _seqnull_spell(
            X,
            randomize=model_l,
            imp_trans=None if imp_trans_value is None else dict(imp_trans_value),
            imp_trans_limit=imp_trans_limit_value,
            states_values=states_values,
            rng=rng,
            seq_obj=seq_obj,
            replay=replay,
        )
    elif model_l == "stateindep" and replay is None:
        try:
            from ._numba import seqnull_random_from_probs_numba as _seqnull_probs_nb  # type: ignore

            Yc = _seqnull_probs_nb(
                hardcoded_stats.astype(np.float64),
                hardcoded_global.astype(np.float64),
                True,
                int(N),
                int(T),
                int(seed),
            )
        except ImportError:
            Yc = _seqnull_pos(
                X,
                prob="freq",
                begin="freq",
                time_varying=True,
                rng=rng,
                weighted=True,
                seq_obj=seq_obj,
                replay=replay,
                backend_seed=seed,
            )
    elif model_l == "stateindep":
        Yc = _seqnull_pos(
            X,
            prob="freq",
            begin="freq",
            time_varying=True,
            rng=rng,
            weighted=True,
            seq_obj=seq_obj,
            replay=replay,
        )
    elif model_l == "markov" and replay is None:
        tr = np.asarray(
            seqtrate(
                seq_obj,
                weighted=True,
                time_varying=False,
                as_dataframe=False,
            ),
            dtype=float,
        )
        p0 = _resolve_begin_distribution(
            begin="freq",
            freqs_t=hardcoded_stats,
            p_glob=hardcoded_global,
            n_states=n_states,
        )
        if _all_sampling_rows_valid(tr):
            try:
                from ._numba import seqnull_markov_numba as _seqnull_markov_nb  # type: ignore

                Yc = _seqnull_markov_nb(
                    tr.astype(np.float64),
                    p0.astype(np.float64),
                    int(N),
                    int(T),
                    int(seed),
                )
            except ImportError:
                Yc = _seqnull_pos(
                    X,
                    prob="trate",
                    begin="freq",
                    time_varying=False,
                    rng=rng,
                    weighted=True,
                    seq_obj=seq_obj,
                    replay=replay,
                    backend_seed=seed,
                )
        else:
            Yc = _seqnull_pos(
                X,
                prob="trate",
                begin="freq",
                time_varying=False,
                rng=rng,
                weighted=True,
                seq_obj=seq_obj,
                replay=replay,
                backend_seed=seed,
            )
    elif model_l == "markov":
        Yc = _seqnull_pos(
            X,
            prob="trate",
            begin="freq",
            time_varying=False,
            rng=rng,
            weighted=True,
            seq_obj=seq_obj,
            replay=replay,
        )
    elif model_l == "userpos":
        prob_arg: str | np.ndarray
        time_varying_eff = True if time_varying_value is None else time_varying_value
        if isinstance(trate, str):
            if str(trate).lower() not in {"trate", "freq"}:
                raise ValueError("unknown trate string")
            prob_arg = str(trate).lower()
        else:
            prob_arg = np.asarray(trate, dtype=float)

        used_numba = False
        python_backend_seed: int | None = None
        if replay is None and isinstance(prob_arg, str):
            if prob_arg == "trate" and not time_varying_eff:
                python_backend_seed = seed
                tr = np.asarray(
                    seqtrate(
                        seq_obj,
                        weighted=weighted_value,
                        time_varying=False,
                        as_dataframe=False,
                    ),
                    dtype=float,
                )
                weighted_stats = np.asarray(
                    seqstatd(seq_obj, weighted=weighted_value)["Frequencies"],
                    dtype=float,
                )
                weighted_global = (
                    np.asarray(seqstatf(seq_obj, weighted=weighted_value)["Percent"], dtype=float)
                    / 100.0
                )
                p0 = _resolve_begin_distribution(
                    begin=begin,
                    freqs_t=weighted_stats,
                    p_glob=weighted_global,
                    n_states=n_states,
                )
                if not _all_sampling_rows_valid(tr):
                    used_numba = False
                else:
                    try:
                        from ._numba import (
                            seqnull_markov_numba as _seqnull_markov_nb,  # type: ignore
                        )

                        Yc = _seqnull_markov_nb(
                            tr.astype(np.float64),
                            p0.astype(np.float64),
                            int(N),
                            int(T),
                            int(seed),
                        )
                        used_numba = True
                    except ImportError:
                        used_numba = False
        elif replay is None:
            arr = np.asarray(prob_arg, dtype=float)
            if arr.ndim == 2 and arr.shape == (n_states, n_states):
                python_backend_seed = seed
                weighted_stats = np.asarray(
                    seqstatd(seq_obj, weighted=weighted_value)["Frequencies"],
                    dtype=float,
                )
                weighted_global = (
                    np.asarray(seqstatf(seq_obj, weighted=weighted_value)["Percent"], dtype=float)
                    / 100.0
                )
                p0 = _resolve_begin_distribution(
                    begin=begin,
                    freqs_t=weighted_stats,
                    p_glob=weighted_global,
                    n_states=n_states,
                )
                if _all_sampling_rows_valid(arr):
                    tr = np.zeros_like(arr, dtype=float)
                    for s in range(n_states):
                        tr[s, :] = _sampling_probs(arr[s, :], context="transition")
                    try:
                        from ._numba import (
                            seqnull_markov_numba as _seqnull_markov_nb,  # type: ignore
                        )

                        Yc = _seqnull_markov_nb(
                            tr.astype(np.float64),
                            p0.astype(np.float64),
                            int(N),
                            int(T),
                            int(seed),
                        )
                        used_numba = True
                    except ImportError:
                        used_numba = False
        if not used_numba:
            Yc = _seqnull_pos(
                X,
                prob=prob_arg,
                begin=begin,
                time_varying=time_varying_eff,
                rng=rng,
                weighted=weighted_value,
                seq_obj=seq_obj,
                replay=replay,
                backend_seed=python_backend_seed,
            )
    else:
        raise ValueError("unknown model")

    if replay is not None:
        replay.finish()
    return _restore_seqnull_output(
        Yc,
        states_values=states_values,
        seq_obj=seq_obj,
        return_sts=return_sts,
        dtype=source_dtype,
        preserve_weights=model_l in {"combined", "duration", "sequencing"},
    )


def _seqnull_by_k_to_by_stat(
    by_k: Dict[int, Dict[str, float]], kvals: List[int]
) -> Dict[str, List[float]]:
    keys: List[str] = []
    for k in kvals:
        for stat in by_k.get(int(k), {}).keys():
            if stat not in keys:
                keys.append(stat)
    return {
        stat: [float(by_k.get(int(k), {}).get(stat, float("nan"))) for k in kvals] for stat in keys
    }


def _coerce_seqnull_clustrange_stats(
    stats_obj: object,
    kvals: List[int],
    observed_by_k: Dict[int, Dict[str, float]],
) -> Dict[str, List[float]]:
    observed_by_stat = _seqnull_by_k_to_by_stat(observed_by_k, kvals)
    if not isinstance(stats_obj, dict) or not stats_obj:
        return observed_by_stat
    if all(isinstance(v, dict) for v in stats_obj.values()):
        by_k: Dict[int, Dict[str, float]] = {}
        for k in kvals:
            vals = stats_obj.get(int(k))
            if vals is None:
                vals = stats_obj.get(str(int(k)))
            if isinstance(vals, dict):
                by_k[int(k)] = {str(sk): float(sv) for sk, sv in vals.items()}
        coerced = _seqnull_by_k_to_by_stat(by_k, kvals)
    else:
        coerced = {}
        for stat, vals in stats_obj.items():
            if isinstance(vals, (list, tuple, np.ndarray)):
                arr = np.asarray(vals, dtype=float).reshape(-1)
                coerced[str(stat)] = [
                    float(arr[i]) if i < arr.size else float("nan") for i in range(len(kvals))
                ]
    for stat, vals in observed_by_stat.items():
        coerced.setdefault(stat, vals)
    return coerced


def _normalize_seqnull_distance_config(
    seqdist_args: Mapping[str, object],
) -> Dict[str, object]:
    """Normalize only compatibility aliases; traminex owns distance semantics."""
    forwarded = dict(seqdist_args)
    for aliases in (
        ("method", "metric"),
        ("norm", "normalize"),
        ("sm", "subm", "substitution_costs"),
    ):
        supplied = [name for name in aliases if name in forwarded]
        if len(supplied) > 1:
            raise ValueError(f"seqdist_args contains conflicting aliases: {supplied}")
    method = forwarded.pop("method", forwarded.pop("metric", "OM"))
    if "sm" in forwarded:
        forwarded["subm"] = forwarded.pop("sm")
    if "substitution_costs" in forwarded:
        forwarded["subm"] = forwarded.pop("substitution_costs")
    if "normalize" in forwarded:
        normalize = forwarded.pop("normalize")
        if not isinstance(normalize, (bool, np.bool_)):
            raise TypeError("normalize must be a boolean")
        forwarded["norm"] = "maxlength" if bool(normalize) else "none"
    forwarded["method"] = str(method)
    return forwarded


def _seqnull_distance_matrix(
    seqdata: np.ndarray,
    *,
    seqdist_args: Mapping[str, object],
) -> np.ndarray:
    forwarded = _normalize_seqnull_distance_config(seqdist_args)
    method = forwarded.pop("method")
    if "subm" in forwarded:
        _validate_observed_substitution_costs(
            seqdata,
            forwarded["subm"],
            method=str(method),
        )
    return np.asarray(
        traminex_seqdist(seqdata, method=method, **forwarded),
        dtype=float,
    )


def _compute_seqnull_observed_stats(
    seqdata: np.ndarray,
    clustering: Dict[str, object],
    kvals: List[int],
    *,
    seqdist_args: Mapping[str, object],
) -> tuple[np.ndarray, Dict[int, Dict[str, float]], List[str]]:
    D_full = _seqnull_distance_matrix(
        seqdata,
        seqdist_args=seqdist_args,
    )
    observed_by_k: Dict[int, Dict[str, float]] = {}
    for k in kvals:
        labels = clustering.get(f"cluster{k}")
        if labels is None:
            continue
        cq = wcClusterQuality(D_full, labels)
        observed_by_k[int(k)] = {str(sk): float(sv) for sk, sv in cq["stats"].items()}
    observed_by_stat = _seqnull_by_k_to_by_stat(observed_by_k, kvals)
    stat_keys = list(observed_by_stat.keys()) or ["ASW", "ASWw"]
    return D_full, observed_by_k, stat_keys


def _prepare_seqnull_clustrange(
    clustrange: object,
    *,
    clustering: Dict[str, object],
    kvals: List[int],
    observed_by_k: Dict[int, Dict[str, float]],
    n_obs: int,
) -> Dict[str, object]:
    clustrange_out_raw = dict(clustrange) if isinstance(clustrange, dict) else {}
    clustrange_out_raw["kvals"] = list(kvals)
    clustrange_out_raw["clustering"] = clustering
    clustrange_out_raw["stats"] = _coerce_seqnull_clustrange_stats(
        clustrange_out_raw.get("stats"), kvals, observed_by_k
    )
    clustrange_out_raw["stats"] = coerce_stats_table(
        clustrange_out_raw["stats"],
        expected_len=len(kvals),
        name="clustrange.stats",
    )
    return validate_clustrange_result(clustrange_out_raw, n_obs=n_obs)


def _recluster_seqnull_distance(
    D: np.ndarray,
    kvals: List[int],
    *,
    kmedoid: bool,
    hclust_method: str,
    random_state: int,
    replay_kmedoid_starts: Mapping[int, Sequence[Sequence[int]] | np.ndarray] | None = None,
    partitions_out: Dict[int, np.ndarray] | None = None,
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    if kmedoid:
        kr = wcKMedRange(
            D,
            kvals,
            weights=None,
            R=1,
            random_state=int(random_state),
            replay_starts=replay_kmedoid_starts,
        )
        for k in kr.get("kvals", []):
            labels = kr.get("clustering", {}).get(f"cluster{k}")
            if labels is None:
                continue
            if partitions_out is not None:
                partitions_out[int(k)] = np.asarray(labels, dtype=int).reshape(-1).copy()
            if np.unique(np.asarray(labels)).size < 2:
                continue
            cq = wcClusterQuality(D, labels)
            out[int(k)] = {name: float(cq["stats"][name]) for name in _SEQNULLCQI_R_STATS}
        return out

    from ._fastcluster_linkage import linkage_fastcluster

    # WeightedCluster 2.0 stores hclust.method in the result but the installed
    # implementation always reclusters null distances with Ward.D. Its call at
    # R/seqnullcqi.R:239 is an unqualified `hclust`, which resolves inside the
    # package namespace to fastcluster's implementation (finding F-01).
    Z = linkage_fastcluster(D, method="ward.d")
    for k in kvals:
        labels = cut_linkage_by_merge_count(Z, n_clusters=int(k))
        if partitions_out is not None:
            partitions_out[int(k)] = labels.copy()
        if np.unique(labels).size < 2:
            continue
        cq = wcClusterQuality(D, labels)
        out[int(k)] = {name: float(cq["stats"][name]) for name in _SEQNULLCQI_R_STATS}
    return out


def _run_seqnull_bootstrap_once(
    i_seed: int,
    *,
    seqdata: np.ndarray,
    kvals: List[int],
    model: str,
    seqnull_kwargs: Dict[str, object],
    seqdist_args: Mapping[str, object],
    kmedoid: bool,
    hclust_method: str,
    replay_null_sequence: np.ndarray | None = None,
    replay_kmedoid_starts: Mapping[int, Sequence[Sequence[int]] | np.ndarray] | None = None,
) -> tuple[np.ndarray, Dict[int, Dict[str, float]], Dict[int, np.ndarray]]:
    Y = (
        np.asarray(replay_null_sequence).copy()
        if replay_null_sequence is not None
        else seqnull(seqdata, model=model, random_state=i_seed, **seqnull_kwargs)
    )
    D_Y = _seqnull_distance_matrix(
        Y,
        seqdist_args=seqdist_args,
    )
    partitions: Dict[int, np.ndarray] = {}
    out = _recluster_seqnull_distance(
        D_Y,
        kvals,
        kmedoid=kmedoid,
        hclust_method=hclust_method,
        random_state=int(i_seed),
        replay_kmedoid_starts=replay_kmedoid_starts,
        partitions_out=partitions,
    )
    return Y, out, partitions


def _execute_seqnull_bootstraps(
    seeds: List[int],
    worker: Callable[[int], tuple[np.ndarray, Dict[int, Dict[str, float]], Dict[int, np.ndarray]]],
    *,
    parallel: bool,
    n_jobs: Optional[int] = None,
    progressbar: bool,
) -> List[tuple[np.ndarray, Dict[int, Dict[str, float]], Dict[int, np.ndarray]]]:
    return run_task_batch(
        worker,
        seeds,
        parallel=parallel,
        n_jobs=n_jobs,
        progressbar=progressbar,
        desc="seqnull boot",
        executor_kind="thread",
    )


def _assemble_seqnull_boot_outputs(
    results: List[tuple[np.ndarray, Dict[int, Dict[str, float]], Dict[int, np.ndarray]]],
    kvals: List[int],
    stat_keys: List[str],
    *,
    n_timepoints: int,
    dtype: np.dtype,
) -> tuple[
    Dict[str, np.ndarray],
    Dict[int, Dict[str, List[float]]],
    np.ndarray,
    List[Dict[int, np.ndarray]],
]:
    n_boot = len(results)
    n_k = len(kvals)
    stats_matrices: Dict[str, np.ndarray] = {
        stat: np.full((n_boot, n_k), np.nan, dtype=float) for stat in stat_keys
    }
    allseq: List[np.ndarray] = []
    reclusterings: List[Dict[int, np.ndarray]] = []
    for i, (Y, out_stats, partitions) in enumerate(results):
        allseq.append(np.asarray(Y))
        reclusterings.append(
            {
                int(k): np.asarray(labels, dtype=int).reshape(-1).copy()
                for k, labels in partitions.items()
            }
        )
        for j, k in enumerate(kvals):
            statdict = out_stats.get(int(k), {})
            for stat, val in statdict.items():
                if stat not in stats_matrices:
                    stats_matrices[stat] = np.full((n_boot, n_k), np.nan, dtype=float)
                stats_matrices[stat][i, j] = float(val)

    boot: Dict[int, Dict[str, List[float]]] = {int(k): {} for k in kvals}
    for j, k in enumerate(kvals):
        for stat, mat in stats_matrices.items():
            boot[int(k)][stat] = [float(v) for v in np.asarray(mat[:, j], dtype=float).tolist()]

    seqdata_null = np.vstack(allseq) if allseq else np.empty((0, int(n_timepoints)), dtype=dtype)
    return stats_matrices, boot, seqdata_null, reclusterings


def seqnullcqi(
    seqdata: np.ndarray,
    clustrange: Dict,
    *,
    R: int,
    model: str = "combined",
    seqdist_args: Optional[Dict] = None,
    kmedoid: bool = False,
    hclust_method: str = "ward.D",
    parallel: bool = False,
    n_jobs: Optional[int] = None,
    progressbar: bool = False,
    random_state: Optional[int] = None,
    replay_null_sequences: Sequence[np.ndarray] | np.ndarray | None = None,
    replay_kmedoid_starts: (
        Sequence[Mapping[int, Sequence[Sequence[int]] | np.ndarray]] | None
    ) = None,
    **seqnull_kwargs: object,
) -> SeqNullCQIResult:
    """Parametric bootstrap validation of sequence typologies (recluster null data).

    - seqdata: (N,T) sequences
    - clustrange: {'kvals','clustering',...} baseline solutions
    - R: number of bootstraps
    - model: null model for seqnull
    - hclust_method: retained in result metadata; when kmedoid=False,
      reclustering is fixed to Ward.D for WeightedCluster 2.0 compatibility.
      It runs through the quadratic-time nearest-neighbour-chain port of
      ``fastcluster::hclust`` for every replicate; the Python-loop overhead
      still makes large workloads slow, and they emit a runtime warning.
      kmedoid=True selects a scientifically different medoid path.
    - parallel / n_jobs: shared bootstrap worker controls used across sequence workflows

    Returns a dict with per-K null distributions of cluster-quality indices from reclustered null datasets.
    """

    X = np.asarray(seqdata)
    if X.ndim != 2:
        raise ValueError("seqdata must be a 2D array (N, T)")
    N, T = X.shape

    if (
        not isinstance(clustrange, dict)
        or not hasattr(clustrange, "r_class")
        or "clustrange" not in clustrange.r_class
    ):
        raise TypeError("clustrange must be a clustrange object")
    R_int = validate_integer_control(R, name="R")
    if R_int < 1:
        raise ValueError("R must be at least one")
    if not isinstance(kmedoid, (bool, np.bool_)):
        raise TypeError("kmedoid must be a boolean")
    kmedoid_value = bool(kmedoid)
    _warn_for_linkage_workload(n_obs=N, replicates=R_int, kmedoid=kmedoid_value)
    unknown_seqnull_args = sorted(set(seqnull_kwargs) - _SEQNULL_CONTROL_KEYS)
    if unknown_seqnull_args:
        raise TypeError(f"unsupported seqnull arguments: {unknown_seqnull_args}")
    if replay_null_sequences is not None and "replay_plan" in seqnull_kwargs:
        raise ValueError("replay_plan conflicts with replay_null_sequences")
    (
        model_value,
        imp_trans_value,
        imp_trans_limit_value,
        time_varying_value,
        weighted_value,
    ) = _normalize_seqnull_controls(
        model=model,
        imp_trans=seqnull_kwargs.get("imp_trans"),
        imp_trans_limit=seqnull_kwargs.get("imp_trans_limit", -1.0),
        trate=seqnull_kwargs.get("trate", "trate"),
        begin=seqnull_kwargs.get("begin", "freq"),
        time_varying=seqnull_kwargs.get("time_varying"),
        weighted=seqnull_kwargs.get("weighted", True),
        validate_applicability=replay_null_sequences is None,
    )
    validated_seqnull_kwargs = dict(seqnull_kwargs)
    if "imp_trans" in validated_seqnull_kwargs:
        validated_seqnull_kwargs["imp_trans"] = imp_trans_value
    if "imp_trans_limit" in validated_seqnull_kwargs:
        validated_seqnull_kwargs["imp_trans_limit"] = imp_trans_limit_value
    if "time_varying" in validated_seqnull_kwargs:
        validated_seqnull_kwargs["time_varying"] = time_varying_value
    if "weighted" in validated_seqnull_kwargs:
        validated_seqnull_kwargs["weighted"] = weighted_value
    if replay_kmedoid_starts is not None and not kmedoid_value:
        raise ValueError("replay_kmedoid_starts requires kmedoid=True")
    random_state_value = (
        None
        if random_state is None
        else validate_integer_control(random_state, name="random_state")
    )
    if random_state_value is not None and random_state_value < 0:
        raise ValueError("random_state must be non-negative")
    clustering = clustrange.get("clustering", {})
    if not isinstance(clustering, dict) or not clustering:
        raise ValueError("clustrange must contain clustering columns")
    clustrange_dict = dict(clustrange)
    if "kvals" not in clustrange_dict:
        raise ValueError("clustrange must contain kvals")
    try:
        kvals = validate_kvals(clustrange_dict["kvals"])
    except ValueError as exc:
        if "non-empty" in str(exc):
            raise ValueError("clustrange must contain kvals") from exc
        raise
    expected_kvals = list(range(2, max(kvals) + 1))
    if kvals != expected_kvals:
        raise ValueError("clustrange kvals must be consecutive from 2 to ncluster")
    for k in kvals:
        labels = clustering.get(f"cluster{k}")
        if labels is None:
            raise ValueError(f"clustrange is missing cluster{k}")
        if np.asarray(labels).reshape(-1).size != N:
            raise ValueError(f"cluster{k} must have length {N}")

    if seqdist_args is None:
        seqdist_config: Dict[str, object] = {}
    elif isinstance(seqdist_args, dict):
        seqdist_config = dict(seqdist_args)
    else:
        raise TypeError("seqdist_args must be a dict or None")
    _normalize_seqnull_distance_config(seqdist_config)

    _D_full, observed_stats_by_k, stat_keys = _compute_seqnull_observed_stats(
        X,
        clustering,
        kvals,
        seqdist_args=seqdist_config,
    )
    clustrange_out = _prepare_seqnull_clustrange(
        clustrange_dict,
        clustering=clustering,
        kvals=kvals,
        observed_by_k=observed_stats_by_k,
        n_obs=N,
    )

    base_seed = generate_random_seed(random_state_value, 0)
    seeds = [generate_random_seed(base_seed, 2, i) for i in range(R_int)]
    replay_by_seed: Dict[int, np.ndarray] = {}
    if replay_null_sequences is not None:
        replay_array = np.asarray(replay_null_sequences)
        if replay_array.shape != (R_int, N, T):
            raise ValueError(f"replay_null_sequences must have shape ({R_int}, {N}, {T})")
        replay_by_seed = {
            int(seed): np.asarray(replay_array[index]).copy() for index, seed in enumerate(seeds)
        }
    starts_by_seed: Dict[int, Mapping[int, Sequence[Sequence[int]] | np.ndarray]] = {}
    if replay_kmedoid_starts is not None:
        if len(replay_kmedoid_starts) != R_int:
            raise ValueError(f"replay_kmedoid_starts must have length {R_int}")
        starts_by_seed = {
            int(seed): replay_kmedoid_starts[index] for index, seed in enumerate(seeds)
        }
    results = _execute_seqnull_bootstraps(
        seeds,
        lambda i_seed: _run_seqnull_bootstrap_once(
            i_seed,
            seqdata=X,
            kvals=kvals,
            model=model_value,
            seqnull_kwargs=dict(validated_seqnull_kwargs),
            seqdist_args=seqdist_config,
            kmedoid=kmedoid_value,
            hclust_method=hclust_method,
            replay_null_sequence=replay_by_seed.get(int(i_seed)),
            replay_kmedoid_starts=starts_by_seed.get(int(i_seed)),
        ),
        parallel=parallel,
        n_jobs=n_jobs,
        progressbar=progressbar,
    )
    stats_matrices, boot, seqdata_null, reclusterings = _assemble_seqnull_boot_outputs(
        results,
        kvals,
        stat_keys,
        n_timepoints=T,
        dtype=X.dtype,
    )
    seqdist_out = dict(seqdist_args) if isinstance(seqdist_args, dict) else {}
    nullmodel = {"model": model}
    nullmodel.update(
        {
            {
                "imp_trans": "imp.trans",
                "imp_trans_limit": "imp.trans.limit",
                "time_varying": "time.varying",
            }.get(str(name), str(name)): value
            for name, value in seqnull_kwargs.items()
        }
    )

    return SeqNullCQIResult(
        {
            "seqdata": seqdata_null,
            "stats": {name: stats_matrices[name] for name in _SEQNULLCQI_R_STATS},
            "clustrange": clustrange_out,
            "R": int(len(results)),
            "kmedoid": kmedoid_value,
            "hclust.method": str(hclust_method),
            "seqdist.args": seqdist_out,
            "nullmodel": nullmodel,
        },
        additions={"boot": boot, "reclusterings": reclusterings},
    )


# The Ward.D reclustering has been the quadratic-time nearest-neighbour-chain
# port of fastcluster::hclust since finding F-01 of the 2026-08-24 audit; the
# heuristic and warning text claimed the pre-F-01 implementation's cubic cost
# until the 2026-09-03 audit-and-fix cycle (finding WC-090302). The threshold
# is anchored so N=100 with R=100 still warns; away from that point the
# boundary follows the corrected quadratic cost model, so mid-N few-replicate
# workloads the cubic model over-flagged fall silent and small-N
# many-replicate workloads of equal quadratic cost now warn.
_LINKAGE_WARNING_WORK = 1_000_000


def _warn_for_linkage_workload(*, n_obs: int, replicates: int, kmedoid: bool) -> None:
    """Warn when the R-faithful quadratic linkage path is likely material."""
    if not kmedoid and int(replicates) * int(n_obs) ** 2 >= _LINKAGE_WARNING_WORK:
        warnings.warn(
            "seqnullcqi(kmedoid=False) reclusters every replicate with the "
            "R-compatible quadratic nearest-neighbour-chain linkage "
            f"(N={int(n_obs)}, R={int(replicates)}); this may be slow. "
            "Use kmedoid=True only when that different clustering method is scientifically "
            "appropriate.",
            RuntimeWarning,
            stacklevel=3,
        )

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._boot_inputs import _BootInputs
from ._boot_runtime import sample_has_all_clusters as _sample_has_all_clusters


@dataclass(slots=True)
class _BootstrapControls:
    sampling: str
    strata_idx: dict[object, np.ndarray] | None
    expected_unique: dict[int, np.ndarray]
    replay_samples: np.ndarray | None


def _strata_indices(strata: np.ndarray) -> dict[object, np.ndarray]:
    keys = list(dict.fromkeys(np.asarray(strata, dtype=object).reshape(-1).tolist()))
    return {
        key: np.flatnonzero(np.asarray(strata, dtype=object).reshape(-1) == key) for key in keys
    }


def _strata_allocation(
    strata_idx: Mapping[object, np.ndarray],
    sample_size: int,
) -> tuple[np.ndarray, int]:
    counts = np.asarray([len(indices) for indices in strata_idx.values()], dtype=float)
    proportions = counts / float(np.sum(counts))
    rounded = np.rint(proportions * int(sample_size)).astype(int)
    return rounded, int(sample_size - int(np.sum(rounded)))


def _validate_replay_samples(
    replay_samples: Sequence[Sequence[int]] | np.ndarray,
    *,
    R: int,
    sample_size: int,
    n_obs: int,
    sampling: str,
    strata_idx: Mapping[object, np.ndarray] | None,
    medoids: np.ndarray | None,
    clustering: Mapping[str, np.ndarray],
    kvals: Sequence[int],
) -> np.ndarray:
    raw = np.asarray(replay_samples)
    if raw.ndim != 2 or raw.shape != (R, sample_size):
        raise ValueError(f"replay_samples must have shape ({R}, {sample_size})")
    replay = np.asarray(raw, dtype=int)
    if not np.array_equal(raw, replay):
        raise ValueError("replay_samples must contain integer indices")
    if np.any(replay < 0) or np.any(replay >= n_obs):
        raise ValueError("replay_samples contain out-of-bounds indices")

    medoid_route = sampling == "medoids" and medoids is not None
    for row in replay:
        if medoid_route:
            resolved_medoids = medoids
            if resolved_medoids is None:  # defensive narrowing for alternate callers
                raise RuntimeError("medoid replay route requires medoid indices")
            if row.size < resolved_medoids.size or not np.array_equal(
                row[: resolved_medoids.size], resolved_medoids
            ):
                raise ValueError("each medoid replay sample must begin with the medoids")
            tail = row[resolved_medoids.size :]
            if np.unique(tail).size != tail.size:
                raise ValueError("the random medoid-sample tail must not contain duplicates")
        elif np.unique(row).size != row.size:
            raise ValueError("replay_samples must not contain duplicate indices")

        if sampling == "strata" and strata_idx:
            base, correction = _strata_allocation(strata_idx, sample_size)
            actual = np.asarray(
                [np.count_nonzero(np.isin(row, indices)) for indices in strata_idx.values()],
                dtype=int,
            )
            delta = actual - base
            direction = int(np.sign(correction))
            if correction == 0:
                valid = np.all(delta == 0)
            else:
                valid = np.all(np.isin(delta, (0, direction))) and np.count_nonzero(
                    delta == direction
                ) == abs(correction)
            if not valid:
                raise ValueError("replay sample does not match stratified rounding allocation")

        if not _sample_has_all_clusters(
            row,
            clustering=dict(clustering),
            expected_unique={
                int(k): np.unique(np.asarray(clustering[f"cluster{int(k)}"])) for k in kvals
            },
            kvals=kvals,
        ):
            raise ValueError("replay sample must preserve every cluster count")
    return replay.copy()


def _prepare_bootstrap_controls(
    inputs: _BootInputs,
    *,
    sampling: str,
    strata: Sequence[int] | None,
    replay_samples: Sequence[Sequence[int]] | np.ndarray | None,
) -> _BootstrapControls:
    X = inputs.sequence_matrix
    kvals = inputs.kvals
    clustering = inputs.clustering
    medoids = inputs.medoids
    R_int = inputs.R
    sample_size_int = inputs.sample_size
    n_obs = int(X.shape[0])
    sampling_requested = str(sampling)
    strata_array = None if strata is None else np.asarray(strata).reshape(-1).copy()
    if strata_array is not None and strata_array.size != n_obs:
        raise ValueError("strata must have the same length as seqdata")

    resolved_sampling = sampling_requested
    strata_idx: dict[object, np.ndarray] | None = None
    if strata_array is None and sampling_requested == "clustering":
        strata_array = np.asarray(clustering[f"cluster{kvals[-1]}"]).reshape(-1).copy()
        strata_idx = _strata_indices(strata_array)
        allocation, _correction = _strata_allocation(strata_idx, sample_size_int)
        if np.any(allocation < 2):
            minimum_proportion = min(len(indices) / float(n_obs) for indices in strata_idx.values())
            considered = 2.0 / minimum_proportion
            raise ValueError(
                "[!] sample size is to small for the stratified sampling of "
                f"clustering. Consider a minimum value of {considered:g}"
            )
        resolved_sampling = "strata"
    elif sampling_requested == "strata":
        if strata_array is None or strata_array.size == 0:
            raise ValueError("sampling='strata' requires non-empty strata")
        strata_idx = _strata_indices(strata_array)

    medoid_route = resolved_sampling == "medoids" and medoids is not None
    random_draw_size = (
        sample_size_int - int(medoids.size)
        if medoid_route and medoids is not None
        else sample_size_int
    )
    if random_draw_size < 0:
        raise ValueError("negative 'size' argument")
    if resolved_sampling != "strata" and random_draw_size > n_obs:
        raise ValueError("cannot take a sample larger than the population when 'replace = False'")

    expected_unique = {int(k): np.unique(np.asarray(clustering[f"cluster{int(k)}"])) for k in kvals}
    max_clusters = max(len(values) for values in expected_unique.values())
    if sample_size_int < max_clusters:
        raise ValueError(
            "sample_size is too small to preserve every cluster count; "
            f"need at least {max_clusters}"
        )

    replay = (
        None
        if replay_samples is None
        else _validate_replay_samples(
            replay_samples,
            R=R_int,
            sample_size=sample_size_int,
            n_obs=n_obs,
            sampling=resolved_sampling,
            strata_idx=strata_idx,
            medoids=medoids,
            clustering=clustering,
            kvals=kvals,
        )
    )
    return _BootstrapControls(
        sampling=resolved_sampling,
        strata_idx=strata_idx,
        expected_unique=expected_unique,
        replay_samples=replay,
    )


def _build_bootstrap_worker_context(
    inputs: _BootInputs,
    controls: _BootstrapControls,
    *,
    seed: int,
) -> dict[str, object]:
    return {
        "seed": seed,
        "X": inputs.sequence_matrix,
        "N": int(inputs.sequence_matrix.shape[0]),
        "sample_size": inputs.sample_size,
        "sampling": controls.sampling,
        "strata_idx": controls.strata_idx,
        "medoids_idx": inputs.medoids,
        "seqdist_args": inputs.seqdist_args,
        "kvals": list(inputs.kvals),
        "clustering": dict(inputs.clustering),
        "expected_unique": controls.expected_unique,
        "replay_samples": controls.replay_samples,
    }

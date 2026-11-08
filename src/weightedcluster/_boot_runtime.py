from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

_BOOTSTRAP_WORKER_CONTEXT: Optional[Dict[str, object]] = None


def init_bootstrap_worker_context(context: Dict[str, object]) -> None:
    """Initialize shared bootstrap worker context."""
    global _BOOTSTRAP_WORKER_CONTEXT
    _BOOTSTRAP_WORKER_CONTEXT = context


def clear_bootstrap_worker_context() -> None:
    """Clear shared bootstrap worker context in the local process."""
    global _BOOTSTRAP_WORKER_CONTEXT
    _BOOTSTRAP_WORKER_CONTEXT = None


def draw_stratified_sample(
    *,
    rng: np.random.Generator,
    strata_idx: Mapping[object, np.ndarray],
    sample_size: int,
) -> np.ndarray:
    """Mirror WeightedCluster's proportional, without-replacement strata draw."""
    total = int(sum(len(v) for v in strata_idx.values()))
    if total <= 0:
        raise ValueError("strata_idx must contain at least one observation")
    keys = list(strata_idx.keys())
    counts = np.array([len(strata_idx[k]) for k in keys], dtype=float)
    props = counts / float(total)
    tosample = np.rint(props * int(sample_size)).astype(int)
    correction = int(sample_size - int(tosample.sum()))
    if correction != 0 and tosample.size > 0:
        picks = rng.choice(tosample.size, size=abs(correction), replace=False)
        tosample[picks] += int(np.sign(correction))
    idx_list = []
    for i, key in enumerate(keys):
        idxs = np.asarray(strata_idx[key], dtype=int)
        n_i = int(tosample[i])
        if n_i < 0:
            raise ValueError("negative 'size' argument")
        if n_i == 0:
            continue
        if n_i > idxs.size:
            raise ValueError(
                "cannot take a sample larger than the population when 'replace = False'"
            )
        idx_list.append(rng.choice(idxs, size=n_i, replace=False))
    return np.concatenate(idx_list) if idx_list else np.array([], dtype=int)


def sample_has_all_clusters(
    idx: np.ndarray,
    *,
    clustering: Dict[str, np.ndarray],
    expected_unique: Dict[int, np.ndarray],
    kvals: Sequence[int],
) -> bool:
    """Check whether sample contains all clusters for every k solution."""
    for k in kvals:
        labels = clustering.get(f"cluster{k}")
        if labels is None:
            continue
        got = np.unique(np.asarray(labels, dtype=int)[idx])
        exp = expected_unique.get(int(k), np.unique(np.asarray(labels, dtype=int)))
        if got.size < exp.size:
            return False
    return True


def draw_bootstrap_sample(
    *,
    rng: np.random.Generator,
    sampling: str,
    N: int,
    sample_size: int,
    strata_idx: Optional[Mapping[object, np.ndarray]],
    medoids_idx: Optional[np.ndarray],
) -> np.ndarray:
    """Draw one bootstrap subsample for the requested sampling mode."""
    if sampling == "medoids" and medoids_idx is not None and medoids_idx.size > 0:
        base = np.asarray(medoids_idx, dtype=int).reshape(-1)
        remain = int(sample_size - base.size)
        if remain < 0:
            raise ValueError("negative 'size' argument")
        if remain > N:
            raise ValueError(
                "cannot take a sample larger than the population when 'replace = False'"
            )
        # The installed source samples from the full population, so a forced
        # medoid may legally occur again in the random tail.
        add = rng.choice(N, size=remain, replace=False)
        return np.asarray(np.concatenate([base, add]), dtype=int)
    if sampling == "strata" and strata_idx:
        return draw_stratified_sample(rng=rng, strata_idx=strata_idx, sample_size=int(sample_size))
    if sample_size > N:
        raise ValueError("cannot take a sample larger than the population when 'replace = False'")
    return np.asarray(rng.choice(N, size=sample_size, replace=False), dtype=int)


def bootstrap_worker_once(
    iter_idx: int,
    context: Dict[str, object],
    *,
    generate_random_seed_fn: Callable[..., int],
    sequence_distance_fn: Callable[..., np.ndarray],
    as_clustrange_fn: Callable[..., Dict[str, object]],
    draw_bootstrap_sample_fn: Callable[..., np.ndarray],
    sample_has_all_clusters_fn: Callable[..., bool],
) -> Dict[str, object]:
    """Perform one bootstrap iteration from shared static context."""
    seed = int(context["seed"])
    X = np.asarray(context["X"])
    N = int(context["N"])
    sample_size = int(context["sample_size"])
    sampling = str(context["sampling"])
    strata_idx = context["strata_idx"]
    medoids_idx = context["medoids_idx"]
    seqdist_args = dict(context["seqdist_args"])
    kvals = list(context["kvals"])
    clustering = dict(context["clustering"])
    expected_unique = dict(context["expected_unique"])
    replay_samples = context.get("replay_samples")
    max_attempts = 1 if replay_samples is not None else max(10, 5 * max(1, len(kvals)))

    if replay_samples is not None:
        idx = np.asarray(replay_samples, dtype=int)[int(iter_idx)].reshape(-1).copy()
        if sample_has_all_clusters_fn(
            idx,
            clustering=clustering,
            expected_unique=expected_unique,
            kvals=kvals,
        ):
            sample_ok = True
        else:
            sample_ok = False
    else:
        iter_seed = generate_random_seed_fn(seed, 3, iter_idx)
        iter_rng = np.random.default_rng(iter_seed)
        idx = np.array([], dtype=int)
        sample_ok = False
        for _ in range(max_attempts):
            idx = draw_bootstrap_sample_fn(
                rng=iter_rng,
                sampling=sampling,
                N=N,
                sample_size=sample_size,
                strata_idx=strata_idx,
                medoids_idx=medoids_idx,
            )
            if sample_has_all_clusters_fn(
                idx,
                clustering=clustering,
                expected_unique=expected_unique,
                kvals=kvals,
            ):
                sample_ok = True
                break
    if not sample_ok:
        attempt_label = "attempt" if max_attempts == 1 else "attempts"
        raise RuntimeError(
            "Unable to draw a bootstrap subsample containing all clusters "
            f"after {max_attempts} {attempt_label} "
            f"(sampling={sampling!r}, sample_size={sample_size})."
        )

    Xs = X[idx]
    Dsub = sequence_distance_fn(
        Xs,
        seqdist_args=seqdist_args,
    )
    subset_clustering = {
        f"cluster{int(k)}": np.asarray(clustering[f"cluster{int(k)}"])[idx] for k in kvals
    }
    cluster_range = as_clustrange_fn(subset_clustering, diss=Dsub)
    return {
        "sample": idx,
        "distance": Dsub,
        "clustering": np.column_stack([subset_clustering[f"cluster{int(k)}"] for k in kvals]),
        "stats": cluster_range["stats"],
    }


def bootstrap_worker_fn(
    iter_idx: int,
    *,
    generate_random_seed_fn: Callable[..., int],
    sequence_distance_fn: Callable[..., np.ndarray],
    as_clustrange_fn: Callable[..., Dict[str, object]],
    draw_bootstrap_sample_fn: Callable[..., np.ndarray],
    sample_has_all_clusters_fn: Callable[..., bool],
) -> Dict[str, object]:
    """Perform one bootstrap iteration from process-local shared context."""
    if _BOOTSTRAP_WORKER_CONTEXT is None:
        raise RuntimeError("bootstrap worker context has not been initialized")
    return bootstrap_worker_once(
        iter_idx,
        _BOOTSTRAP_WORKER_CONTEXT,
        generate_random_seed_fn=generate_random_seed_fn,
        sequence_distance_fn=sequence_distance_fn,
        as_clustrange_fn=as_clustrange_fn,
        draw_bootstrap_sample_fn=draw_bootstrap_sample_fn,
        sample_has_all_clusters_fn=sample_has_all_clusters_fn,
    )

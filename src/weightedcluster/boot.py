from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, Optional

import numpy as np
from traminex import seqdist as traminex_seqdist

from ._boot_controls import (
    _build_bootstrap_worker_context,
    _prepare_bootstrap_controls,
)
from ._boot_inputs import _prepare_bootstrap_inputs
from ._boot_result import _build_bootclustrange_result
from ._boot_runtime import (
    bootstrap_worker_fn as _runtime_bootstrap_worker_fn,
)
from ._boot_runtime import (
    clear_bootstrap_worker_context as _runtime_clear_bootstrap_worker_context,
)
from ._boot_runtime import (
    draw_bootstrap_sample as _runtime_draw_bootstrap_sample,
)
from ._boot_runtime import (
    draw_stratified_sample as _runtime_draw_stratified_sample,
)
from ._boot_runtime import (
    init_bootstrap_worker_context as _runtime_init_bootstrap_worker_context,
)
from ._boot_runtime import (
    sample_has_all_clusters as _runtime_sample_has_all_clusters,
)
from ._parallel import run_task_batch
from ._seqdist import _validate_observed_substitution_costs, normalize_seqdist_config
from ._utils import generate_random_seed
from .range import as_clustrange
from .schemas import BootClustrangeResult

_init_bootstrap_worker_context = _runtime_init_bootstrap_worker_context
_clear_bootstrap_worker_context = _runtime_clear_bootstrap_worker_context
_draw_stratified_sample = _runtime_draw_stratified_sample
_sample_has_all_clusters = _runtime_sample_has_all_clusters
_draw_bootstrap_sample = _runtime_draw_bootstrap_sample


def _boot_distance_matrix(
    seqdata: np.ndarray,
    *,
    seqdist_args: Mapping[str, object],
) -> np.ndarray:
    """Delegate one installed-style sequence-distance call to public traminex."""
    config = normalize_seqdist_config(dict(seqdist_args), default_method="LCS")
    method = str(config["method"])
    forwarded: dict[str, object] = {}
    if config["substitution_costs"] is not None:
        forwarded["subm"] = config["substitution_costs"]
    if method in {"OM", "OMSPELL"}:
        forwarded["indel"] = float(config["indel"])
    if method == "OMSPELL" and config["timecost"] is not None:
        forwarded["expcost"] = float(config["timecost"])
    if config["norm"] != "none":
        forwarded["norm"] = str(config["norm"])
    if "subm" in forwarded:
        _validate_observed_substitution_costs(
            seqdata,
            forwarded["subm"],
            method=method,
        )
    return np.asarray(
        traminex_seqdist(seqdata, method=method, **forwarded),
        dtype=float,
    )


def _bootstrap_worker_fn(iter_idx: int) -> Dict[str, object]:
    """Perform one bootstrap iteration using the shared runtime helper."""
    return _runtime_bootstrap_worker_fn(
        iter_idx,
        generate_random_seed_fn=generate_random_seed,
        sequence_distance_fn=_boot_distance_matrix,
        as_clustrange_fn=as_clustrange,
        draw_bootstrap_sample_fn=_draw_bootstrap_sample,
        sample_has_all_clusters_fn=_sample_has_all_clusters,
    )


def bootclustrange(
    object: object,
    seqdata: np.ndarray,
    *,
    seqdist_args: Optional[Dict] = None,
    R: int = 100,
    sample_size: int = 1000,
    progressbar: bool = False,
    sampling: str = "clustering",
    strata: Optional[Sequence[int]] = None,
    random_state: Optional[int] = None,
    parallel: bool = False,
    n_jobs: Optional[int] = None,
    replay_samples: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> BootClustrangeResult:
    """Estimate cluster-quality sampling distributions like WeightedCluster 2.0.

    ``replay_samples`` is a zero-based deterministic seam. It replaces only
    the stochastic subsample draw; public traminex distance computation and
    the package's ``as_clustrange`` quality path still run for every replicate.

    In the WeightedCluster 2.0 contract, weights attached to a traminex
    ``STSList`` are preserved as result metadata but are not passed to the
    baseline or replicate cluster-quality calculations. Pass a clustering
    whose construction already reflects the desired weighting; this function
    deliberately mirrors R rather than silently applying attached weights.
    """
    inputs = _prepare_bootstrap_inputs(
        object,
        seqdata,
        seqdist_args=seqdist_args,
        R=R,
        sample_size=sample_size,
    )
    baseline_distance = _boot_distance_matrix(
        inputs.sequence_matrix,
        seqdist_args=inputs.seqdist_args,
    )
    baseline_stats = as_clustrange(
        dict(inputs.clustering),
        diss=baseline_distance,
    )["stats"]

    controls = _prepare_bootstrap_controls(
        inputs,
        sampling=sampling,
        strata=strata,
        replay_samples=replay_samples,
    )
    seed = generate_random_seed(random_state, 0)
    worker_context = _build_bootstrap_worker_context(inputs, controls, seed=seed)

    try:
        results = run_task_batch(
            _bootstrap_worker_fn,
            list(range(inputs.R)),
            parallel=parallel,
            n_jobs=n_jobs,
            progressbar=progressbar,
            desc="Bootstrap",
            executor_kind="process",
            initializer=_init_bootstrap_worker_context,
            initargs=(worker_context,),
        )
    finally:
        _clear_bootstrap_worker_context()

    return _build_bootclustrange_result(
        inputs,
        controls,
        baseline_stats=baseline_stats,
        results=results,
    )

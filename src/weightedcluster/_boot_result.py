from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ._boot_controls import _BootstrapControls
from ._boot_inputs import _BootInputs
from .schemas import BootClustrangeResult, ClusteringTable, LabeledStatsTable

_BOOT_R_STATS = (
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


def _build_bootclustrange_result(
    inputs: _BootInputs,
    controls: _BootstrapControls,
    *,
    baseline_stats: object,
    results: Sequence[Mapping[str, object]],
) -> BootClustrangeResult:
    attached_weights = inputs.attached_weights
    kvals = inputs.kvals
    clustering = inputs.clustering
    R_int = inputs.R
    seqdist_config = inputs.seqdist_args
    row_names = [f"cluster{k}" for k in kvals]
    boot: dict[int, LabeledStatsTable] = {}
    for k_position, kval in enumerate(kvals):
        boot[int(kval)] = LabeledStatsTable(
            {
                stat: [
                    float(np.asarray(result["stats"][stat], dtype=float)[k_position])
                    for result in results
                ]
                for stat in _BOOT_R_STATS
            },
            row_names=[str(index + 1) for index in range(R_int)],
        )

    meant = LabeledStatsTable(
        {
            stat: [float(np.mean(np.asarray(boot[int(k)][stat], dtype=float))) for k in kvals]
            for stat in _BOOT_R_STATS
        },
        row_names=row_names,
    )
    stderr = LabeledStatsTable(
        {
            stat: [
                (
                    float(np.std(np.asarray(boot[int(k)][stat], dtype=float), ddof=1))
                    if R_int > 1
                    else float("nan")
                )
                for k in kvals
            ]
            for stat in _BOOT_R_STATS
        },
        row_names=row_names,
    )
    output_clustering = ClusteringTable(
        {f"cluster{k}": clustering[f"cluster{k}"].copy() for k in kvals},
        column_names=row_names,
    )
    return BootClustrangeResult(
        {
            "clustering": output_clustering,
            "kvals": list(kvals),
            "meant": meant,
            "stderr": stderr,
            "boot": boot,
            "stats": meant,
        },
        additions={
            "sampling": controls.sampling,
            "R": R_int,
            "seqdist.args": dict(seqdist_config),
            "baseline_stats": baseline_stats,
            "samples": [
                np.asarray(result["sample"], dtype=int).reshape(-1).copy() for result in results
            ],
            "distances": [np.asarray(result["distance"], dtype=float).copy() for result in results],
            "subset_clusterings": [np.asarray(result["clustering"]).copy() for result in results],
            "weights": attached_weights,
        },
    )

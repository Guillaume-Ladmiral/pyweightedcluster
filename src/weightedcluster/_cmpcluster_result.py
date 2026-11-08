"""Typed result assembly for :func:`weightedcluster.cmpcluster.wcCmpCluster`."""

from __future__ import annotations

from typing import Sequence

from .schemas import (
    ClustrangeFamilyParam,
    ClustrangeFamilyResult,
    LabeledObjectTable,
)

_R_QUALITY_NAMES = (
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


def _build_clustrange_family_result(
    method_results: dict[str, object],
    *,
    methods: Sequence[str],
    pam_combine: bool,
    kvals: Sequence[int],
    embed_then_ward: bool,
    random_state: int | None,
) -> ClustrangeFamilyResult:
    """Build the exact typed, dict-compatible family result."""

    method_names = list(method_results)
    allstats_values: dict[str, list[object]] = {
        name: [] for name in (*_R_QUALITY_NAMES, "method", "ngroup")
    }
    allstats_row_names: list[str] = []
    for method_name in method_names:
        result = method_results[method_name]
        for index, k in enumerate(result["kvals"]):
            for stat_name in _R_QUALITY_NAMES:
                allstats_values[stat_name].append(float(result["stats"][stat_name][index]))
            allstats_values["method"].append(method_name)
            allstats_values["ngroup"].append(int(k))
            allstats_row_names.append(f"{method_name}.cluster{k}")

    method_results["param"] = ClustrangeFamilyParam(
        {
            "method": list(methods),
            "pam.combine": bool(pam_combine),
            "all.methods": method_names,
            "kvals": kvals,
        },
        additions={
            "embed_then_ward": bool(embed_then_ward),
            "random_state": None if random_state is None else int(random_state),
        },
    )
    method_results["allstats"] = LabeledObjectTable(
        allstats_values,
        row_names=allstats_row_names,
    )
    return ClustrangeFamilyResult(method_results)

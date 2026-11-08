from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from traminex import STSList, disscenter, seqdef
from traminex.seqformat import seqformat


def _as_stslist(seqdata: Any) -> STSList:
    if isinstance(seqdata, STSList):
        return seqdata
    return seqdef(seqdata)


def _subset_stslist(seqdata: STSList, indices: np.ndarray) -> STSList:
    subset = seqdata.copy()
    subset.data = np.asarray(seqdata.data)[indices].copy()
    subset.weights = (
        None
        if seqdata.weights is None
        else np.asarray(seqdata.weights, dtype=float)[indices].copy()
    )
    subset.ids = None if seqdata.ids is None else [seqdata.ids[int(index)] for index in indices]
    subset.issues = None
    return subset


def seqclustname(
    seqdata: STSList | np.ndarray | pd.DataFrame,
    group: Sequence[Any] | pd.Categorical,
    diss: np.ndarray,
    weighted: bool = True,
    perc: bool = False,
) -> pd.Categorical:
    """Label cluster observations with their compressed medoid trajectories.

    This is the Python counterpart of WeightedCluster 2.0 ``seqclustname``.
    Dissimilarity medoids and STS-to-SPS formatting delegate to traminex. The
    returned categorical preserves factor-equivalent levels, codes, and missing
    observations.
    """
    if not isinstance(weighted, (bool, np.bool_)):
        raise TypeError("weighted must be a boolean")
    if not isinstance(perc, (bool, np.bool_)):
        raise TypeError("perc must be a boolean")
    sequences = _as_stslist(seqdata)
    groups = np.asarray(group, dtype=object).reshape(-1)
    matrix = np.asarray(diss, dtype=float)
    n = sequences.nseq
    if matrix.ndim != 2 or matrix.shape != (n, n):
        raise ValueError("diss must be a square matrix with one row per sequence")
    if groups.shape != (n,):
        raise ValueError("group must have one value per sequence")
    if bool(pd.isna(groups).any()):
        raise ValueError("'x' and 'w' must have the same length")

    if isinstance(group, pd.Categorical):
        observed_group_levels = [
            level for level in group.categories if bool(np.any(groups == level))
        ]
        level_to_code = {level: code for code, level in enumerate(observed_group_levels)}
        medoid_groups = np.asarray([level_to_code[value] for value in groups], dtype=int)
    else:
        observed_group_levels = np.unique(groups).tolist()
        medoid_groups = groups

    weights = sequences.weights if weighted else None
    medoids = disscenter(
        matrix,
        group=medoid_groups,
        medoids_index="first",
        allcenter=False,
        weights=weights,
    )
    if isinstance(medoids, dict):
        group_levels = observed_group_levels
        medoid_indices = np.asarray(list(medoids.values()), dtype=int)
    else:
        group_levels = observed_group_levels
        medoid_indices = np.asarray([medoids], dtype=int)

    medoid_sequences = _subset_stslist(sequences, medoid_indices)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"'missing' set as .* the 'nr' and 'void' code from the 'data' state sequence object",
            category=UserWarning,
        )
        labels = seqformat(
            medoid_sequences,
            from_="STS",
            to="SPS",
            compress=True,
            sps_out={"xfix": "", "sdsep": "/"},
        )
    if not isinstance(labels, list):
        raise TypeError("traminex.seqformat(..., to='SPS') must return a list")

    if len(group_levels) == 1:
        # R 2.0's single-group disscenter result is unnamed; factor(levels=NULL,
        # labels=ll) retains the label level but makes every observation missing.
        return pd.Categorical.from_codes(np.full(n, -1, dtype=int), categories=[str(labels[0])])

    base = pd.Categorical(groups, categories=group_levels)
    result_labels = [str(label) for label in labels]
    if perc:
        if weights is None:
            counts = np.asarray([(groups == level).sum() for level in group_levels], dtype=float)
        else:
            resolved_weights = np.asarray(weights, dtype=float)
            counts = np.asarray(
                [resolved_weights[groups == level].sum() for level in group_levels],
                dtype=float,
            )
        percentages = np.round(counts / counts.sum() * 100.0, 1)
        result_labels = [
            f"{label} ({format(float(value), 'g')}%)"
            for label, value in zip(result_labels, percentages, strict=True)
        ]
    return base.rename_categories(result_labels)

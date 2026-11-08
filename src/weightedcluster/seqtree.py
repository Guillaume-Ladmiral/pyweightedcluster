from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from traminex import dissassoc, disscenter, disstreeleaf, dissvar, weighted_inertia
from traminex.disstree import DissTree, DissTreeNode, DissTreeSplit

from ._validation import (
    cut_linkage_by_merge_count,
    validate_distance_matrix,
    validate_integer_control,
    validate_linkage_matrix,
)
from .proptree import _tree_adapter


def _extract_weights(seqdata: Any, *, weighted: bool, n_obs: int) -> np.ndarray:
    if not bool(weighted):
        return np.ones(n_obs, dtype=float)

    candidates: list[Any] = []
    if isinstance(seqdata, Mapping) and "weights" in seqdata:
        candidates.append(seqdata.get("weights"))
    attrs = getattr(seqdata, "attrs", None)
    if isinstance(attrs, Mapping) and "weights" in attrs:
        candidates.append(attrs.get("weights"))
    if hasattr(seqdata, "weights"):
        candidates.append(seqdata.weights)
    for candidate in candidates:
        if candidate is None:
            continue
        weights = np.asarray(candidate, dtype=float).reshape(-1)
        if weights.shape != (n_obs,):
            raise ValueError("sequence weights must have one value per observation")
        return weights
    return np.ones(n_obs, dtype=float)


def _hierarchy_predictors(linkage_matrix: np.ndarray, ncluster: int) -> pd.DataFrame:
    requested = [2, 3] if int(ncluster) == 2 else list(range(2, int(ncluster) + 1))
    columns: dict[str, pd.Categorical] = {}
    for cluster_count in requested:
        labels = cut_linkage_by_merge_count(linkage_matrix, n_clusters=cluster_count)
        columns[f"Split{cluster_count}"] = pd.Categorical(
            labels, categories=sorted(np.unique(labels).tolist())
        )
    return pd.DataFrame(columns)


def _predictor_frame(object: Any, n_obs: int) -> pd.DataFrame:
    if isinstance(object, pd.DataFrame):
        predictors = object.copy(deep=True).reset_index(drop=True)
    elif isinstance(object, Mapping):
        predictors = pd.DataFrame(object).reset_index(drop=True)
    else:
        raise TypeError("default-equivalent object must be a pandas predictor frame")
    if predictors.shape[0] != n_obs:
        raise ValueError("predictors and diss must have the same number of rows")
    if predictors.shape[1] == 0:
        raise ValueError(" [!] ncluster should be bigger than 2")
    if bool(predictors.isna().to_numpy().any()):
        raise ValueError("predictors must not contain missing split values")
    return predictors


def _node_medoid(diss: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> int:
    local = diss[np.ix_(indices, indices)]
    medoid = disscenter(
        local,
        medoids_index="first",
        allcenter=False,
        weights=weights[indices],
    )
    return int(indices[int(medoid)])


def _scheduled_tree(
    diss: np.ndarray,
    predictors: pd.DataFrame,
    weights: np.ndarray,
    *,
    ncluster: int,
    seqdata: Any,
    weighted: bool,
) -> DissTree:
    counter = 1
    predictor_values = [predictors.iloc[:, index] for index in range(predictors.shape[1])]

    def build(
        indices: np.ndarray,
        vardis: float,
        depth: int,
        current: int,
    ) -> DissTreeNode:
        nonlocal counter
        node = DissTreeNode(
            id=counter,
            info={
                "depth": int(depth),
                "vardis": float(vardis),
                "n": float(weights[indices].sum()),
                "medoid": _node_medoid(diss, indices, weights),
                "ind": indices.copy(),
                "splitschedule": int(depth),
            },
        )
        counter += 1
        if current > ncluster or indices.size == 1:
            return node

        sc_total = float(vardis * node.info["n"])
        for cluster_count in range(current, ncluster + 1):
            column_index = cluster_count - 2
            values = predictor_values[column_index].iloc[indices]
            labels = list(pd.unique(values))
            if len(labels) != 2:
                continue
            left_mask = np.asarray(values == labels[0], dtype=bool)
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
            left_sc = weighted_inertia(diss, left_indices, weights)
            right_sc = weighted_inertia(diss, right_indices, weights)
            left_pop = float(weights[left_indices].sum())
            right_pop = float(weights[right_indices].sum())
            residual = float(left_sc + right_sc)
            left_var = float(left_sc / left_pop)
            right_var = float(right_sc / right_pop)
            with np.errstate(divide="ignore", invalid="ignore"):
                r_squared = float(1.0 - np.divide(residual, sc_total))
            split_info = {
                "lpop": left_pop,
                "rpop": right_pop,
                "SCres": residual,
                "lvar": left_var,
                "rvar": right_var,
                "R2": r_squared,
            }
            node.split = DissTreeSplit(
                varindex=column_index,
                prob=np.asarray([left_pop, right_pop], dtype=float) / float(node.info["n"]),
                info=split_info,
                index=np.asarray([1, 2], dtype=int),
                labels=labels,
            )
            node.kids = [
                build(
                    left_indices,
                    left_var,
                    depth=cluster_count,
                    current=cluster_count + 1,
                ),
                build(
                    right_indices,
                    right_var,
                    depth=cluster_count,
                    current=cluster_count + 1,
                ),
            ]
            return node
        return node

    population = float(weights.sum())
    root_variance = dissvar(diss, weights=weights)
    root = build(np.arange(diss.shape[0], dtype=int), root_variance, depth=1, current=2)
    fitted = disstreeleaf(root)
    adjustment = dissassoc(
        diss,
        fitted,
        weights=weights if weighted else None,
        R=0,
        weight_permutation="diss" if weighted else "replicate",
    )
    return DissTree(
        root=root,
        fitted=fitted,
        info={
            "method": "disstree",
            "n": population,
            "parameters": {
                "minSize": 1,
                "maxdepth": int(ncluster),
                "R": 0,
                "pval": 1,
            },
            "object": seqdata,
            "weight_permutation": "diss",
        },
        data=predictors,
        weights=weights,
        adjustment=adjustment,
    )


def as_seqtree(
    object: Any,
    seqdata: Any,
    diss: np.ndarray,
    weighted: bool = True,
    ncluster: int | None = None,
) -> dict[str, Any]:
    """Convert a hierarchy or ordered split frame to a sequence tree.

    A SciPy linkage matrix is the Python representation for both R ``hclust``
    and ``twins`` hierarchies. A pandas frame containing ordered ``Split2`` …
    ``SplitK`` partitions is the default-equivalent path.
    """
    if not isinstance(weighted, (bool, np.bool_)):
        raise TypeError("weighted must be a boolean")
    requested_ncluster = (
        None if ncluster is None else validate_integer_control(ncluster, name="ncluster")
    )
    distance = np.asarray(diss, dtype=float)
    validate_distance_matrix(distance, name="diss")
    n_obs = distance.shape[0]

    if isinstance(object, (pd.DataFrame, Mapping)):
        predictors = _predictor_frame(object, n_obs)
        effective_ncluster = int(predictors.shape[1] + 1)
    else:
        if requested_ncluster is None:
            raise TypeError("ncluster is required for hierarchy input")
        if requested_ncluster < 2 or requested_ncluster > n_obs:
            raise ValueError(f"ncluster must be between 2 and {n_obs}")
        effective_ncluster = 3 if requested_ncluster == 2 else requested_ncluster
        if effective_ncluster > n_obs:
            raise ValueError("ncluster=2 requires at least three hierarchy observations")
        linkage_matrix = np.asarray(object, dtype=float)
        validate_linkage_matrix(linkage_matrix, n_obs, name="linkage matrix")
        predictors = _hierarchy_predictors(linkage_matrix, requested_ncluster)

    weights = _extract_weights(seqdata, weighted=weighted, n_obs=n_obs)
    tree = _scheduled_tree(
        distance,
        predictors,
        weights,
        ncluster=effective_ncluster,
        seqdata=seqdata,
        weighted=weighted,
    )
    result = _tree_adapter(
        tree,
        distance,
        classes=("seqtree", "disstree", "list"),
        apply_schedule=False,
    )
    result["info"]["adjustment"] = tree.adjustment
    return result

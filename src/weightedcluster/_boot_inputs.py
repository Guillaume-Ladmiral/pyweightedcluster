from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from traminex import STSList

from ._seqdist import normalize_seqdist_config
from ._validation import validate_integer_control
from .schemas import ClusteringTable


@dataclass(slots=True)
class _BootInputs:
    sequence_matrix: np.ndarray
    attached_weights: np.ndarray | None
    kvals: list[int]
    clustering: ClusteringTable
    medoids: np.ndarray | None
    R: int
    sample_size: int
    seqdist_args: dict[str, object]


def _coerce_sequence_matrix(seqdata: object) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(seqdata, STSList):
        matrix = np.asarray(seqdata.data)
        weights = (
            None
            if getattr(seqdata, "weights", None) is None
            else np.asarray(seqdata.weights, dtype=float).reshape(-1).copy()
        )
    else:
        matrix = np.asarray(seqdata)
        weights = None
    if matrix.ndim != 2:
        raise ValueError("seqdata must be a two-dimensional sequence matrix")
    if matrix.shape[0] < 1:
        raise ValueError("seqdata must contain at least one sequence")
    return matrix.copy(), weights


def _mapping_columns(values: Mapping[str, object]) -> list[tuple[str, np.ndarray]]:
    column_names = getattr(values, "column_names", None)
    names = list(values) if column_names is None else [str(name) for name in column_names]
    return [(str(name), np.asarray(values[str(name)]).reshape(-1).copy()) for name in names]


def _coerce_clustering_columns(
    object: object,
    *,
    n_obs: int,
) -> tuple[list[int], ClusteringTable, np.ndarray | None]:
    is_seqclararange = (
        isinstance(object, Mapping)
        and hasattr(object, "r_class")
        and "seqclararange" in object.r_class
    )
    medoids: np.ndarray | None = None

    if is_seqclararange:
        raw = object.get("clustering")
        if not isinstance(raw, Mapping):
            raise ValueError("seqclararange object must contain clustering columns")
        columns = _mapping_columns(raw)
        clara = object.get("clara")
        if isinstance(clara, Mapping):
            entries = list(clara.values())
        elif isinstance(clara, Sequence) and not isinstance(clara, (str, bytes)):
            entries = list(clara)
        else:
            entries = []
        if not entries or not isinstance(entries[-1], Mapping):
            raise ValueError("seqclararange object must contain a final clara result")
        if "medoids" not in entries[-1]:
            raise ValueError("final seqclararange clara result must contain medoids")
        medoids = np.asarray(entries[-1]["medoids"], dtype=int).reshape(-1).copy()
    elif isinstance(object, Mapping) and isinstance(object.get("clustering"), Mapping):
        # Historical Python clustrange mappings remain valid inputs.
        columns = _mapping_columns(object["clustering"])
    elif isinstance(object, Mapping):
        columns = _mapping_columns(object)
    elif hasattr(object, "columns") and hasattr(object, "__getitem__"):
        columns = [
            (str(name), np.asarray(object[name]).reshape(-1).copy())
            for name in list(object.columns)
        ]
    else:
        array = np.asarray(object)
        if array.ndim != 2:
            raise ValueError("object must be a clustering frame or seqclararange result")
        columns = [
            (f"column{index + 1}", array[:, index].reshape(-1).copy())
            for index in range(array.shape[1])
        ]

    if not columns:
        raise ValueError("object must contain at least one clustering column")
    kvals: list[int] = []
    renamed: dict[str, np.ndarray] = {}
    for _name, labels in columns:
        if labels.size != n_obs:
            raise ValueError(f"clustering column has length {labels.size}, expected {n_obs}")
        if np.any(np.asarray([value is None for value in labels], dtype=bool)):
            raise ValueError("clustering columns must not contain missing values")
        kval = int(np.unique(labels).size)
        if kval < 1:
            raise ValueError("clustering columns must contain at least one cluster")
        output_name = f"cluster{kval}"
        if output_name in renamed:
            raise ValueError(f"multiple clustering columns contain {kval} clusters")
        kvals.append(kval)
        renamed[output_name] = labels.copy()

    if medoids is not None:
        if medoids.size < 1:
            raise ValueError("seqclararange medoids must not be empty")
        if np.any(medoids < 0) or np.any(medoids >= n_obs):
            raise ValueError("seqclararange medoids are out of bounds")
    return kvals, ClusteringTable(renamed, column_names=list(renamed)), medoids


def _prepare_bootstrap_inputs(
    object: object,
    seqdata: object,
    *,
    seqdist_args: Mapping[str, object] | None,
    R: object,
    sample_size: object,
) -> _BootInputs:
    sequence_matrix, attached_weights = _coerce_sequence_matrix(seqdata)
    kvals, clustering, medoids = _coerce_clustering_columns(
        object,
        n_obs=int(sequence_matrix.shape[0]),
    )

    R_int = validate_integer_control(R, name="R")
    if R_int < 1:
        raise ValueError("R must be a positive integer")
    sample_size_int = validate_integer_control(sample_size, name="sample_size")
    if sample_size_int < 1:
        raise ValueError("sample_size must be a positive integer")

    if seqdist_args is None:
        seqdist_config: dict[str, object] = {"method": "LCS"}
    elif isinstance(seqdist_args, Mapping):
        seqdist_config = dict(seqdist_args)
    else:
        raise TypeError("seqdist_args must be a mapping or None")
    # Validate through the same alias/method contract as every other sequence
    # distance entry point while preserving bootclustrange's LCS default and
    # the caller spellings stored in the replay context.
    normalize_seqdist_config(seqdist_config, default_method="LCS")
    return _BootInputs(
        sequence_matrix=sequence_matrix,
        attached_weights=attached_weights,
        kvals=kvals,
        clustering=clustering,
        medoids=medoids,
        R=R_int,
        sample_size=sample_size_int,
        seqdist_args=seqdist_config,
    )

"""Input validation utilities for weightedcluster package.

This module provides validation functions for distance matrices, sequence-distance
arguments, integer controls, k-value lists, and linkage matrices, with clear error
messages. Kernel label and weight validation is owned by
``weightedcluster._quality_contract`` (``validate_kernel_labels``,
``validate_kernel_weights``); the copies that used to live here were removed in
the 2026-08-24 remediation (2026-09-02 audit, finding WC-090207).
"""

from __future__ import annotations

import warnings
from numbers import Real
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.distance import squareform  # type: ignore

from ._constants import DISTANCE_DIAGONAL_TOL, DISTANCE_SYMMETRY_ATOL, DISTANCE_SYMMETRY_RTOL
from ._seqdist import normalize_seqdist_config


def validate_distance_matrix(
    diss: np.ndarray,
    name: str = "diss",
    *,
    strict: bool = True,
) -> None:
    """Validate that input is a proper distance matrix.

    Parameters
    ----------
    diss : np.ndarray
        Distance matrix to validate
    name : str
        Parameter name for error messages
    strict : bool, default=True
        If True, raise ValueError when matrix is non-symmetric or has non-zero
        diagonal. If False, emit warnings for those conditions.

    Raises
    ------
    ValueError
        If matrix is not 2D or square, and (when strict=True) if it is not
        symmetric or has non-zero diagonal
    """
    if not isinstance(diss, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(diss)}")

    if diss.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {diss.ndim}D array")

    if diss.shape[0] != diss.shape[1]:
        raise ValueError(f"{name} must be square, got shape {diss.shape}")

    if diss.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")

    try:
        finite = np.isfinite(diss)
    except TypeError as exc:
        raise TypeError(f"{name} must contain numeric distances") from exc
    if not np.all(finite):
        raise ValueError(f"{name} must contain only finite distances")
    if np.any(diss < 0):
        raise ValueError(f"{name} distances must be non-negative")

    # Check symmetry (with tolerance for floating point)
    if not np.allclose(diss, diss.T, rtol=DISTANCE_SYMMETRY_RTOL, atol=DISTANCE_SYMMETRY_ATOL):
        max_diff = np.abs(diss - diss.T).max()
        msg = (
            f"{name} is not symmetric (max difference: {max_diff:.2e}). "
            "Distance matrices must be symmetric."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + " Continuing because strict=False.", UserWarning, stacklevel=2)

    # Check diagonal is close to zero
    diag_max = np.abs(diss.diagonal()).max()
    if diag_max > DISTANCE_DIAGONAL_TOL:
        msg = (
            f"{name} diagonal is not zero (max: {diag_max:.2e}). "
            "Distance from object to itself should be zero."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + " Continuing because strict=False.", UserWarning, stacklevel=2)


def coerce_public_distance_matrix(diss: object, *, name: str = "diss") -> np.ndarray:
    """Apply the one public distance-matrix contract, as R's ``as.dist`` reads it.

    ``wcClusterQuality``, ``wcSilhouetteObs``, ``as_clustrange`` and
    ``wcKMedoids`` all route their distance argument through here, so the four
    agree with each other and with R.

    R's ``as.dist`` treats **only the lower triangle** of a square matrix as
    data; the upper triangle and the diagonal are ignored. A square input is
    therefore reduced to its lower triangle, mirrored back into a symmetric
    matrix with a zero diagonal. A condensed vector is expanded first.

    R's own behaviour when handed a *bare matrix* is not the same, and this
    deliberately does not follow it. Measured on R 4.6.1 / WeightedCluster 2.0:
    for a non-symmetric matrix the silhouette loop reads both triangles while
    the ordered-pair statistics read one, so R's answer differs between ``D``
    and ``as.dist(D)`` — in ``ASW``/``ASWw`` for ``wcClusterQuality`` and
    ``as.clustrange``, in the whole vector for ``wcSilhouetteObs``, and in the
    partition for ``wcKMedoids``. Reducing here is the only reading under which
    the four public entry points agree with each other; the divergence is
    recorded in ``docs/R_MIGRATION_GUIDE.md`` and pinned by
    ``tests/test_r_parity_distance_matrix_contract.py``.

    Python adds one thing R does not: a ``UserWarning`` when the discarded part
    disagrees with the lower triangle beyond tolerance. Silently computing on
    half of a caller's matrix is the failure mode this contract exists to make
    visible, and R gives no signal at all.

    Findings F-06 and F-07 of the 2026-08-24 whole-codebase audit: before this
    existed, three of the four entry points raised on input R computes from,
    ``wcSilhouetteObs`` accepted it and returned different numbers from R
    without any signal, and the four did not agree with each other.
    """
    D = np.asarray(diss, dtype=float)
    if D.ndim == 1:
        try:
            D = squareform(D)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a square matrix or condensed distance vector"
            ) from exc
    elif D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{name} must be a square matrix or condensed distance vector")
    if D.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")

    diagonal = np.diagonal(D)
    already_reduced = bool(np.array_equal(D, D.T, equal_nan=True) and np.all(diagonal == 0.0))
    if already_reduced:
        return D

    if not np.allclose(
        D, D.T, rtol=DISTANCE_SYMMETRY_RTOL, atol=DISTANCE_SYMMETRY_ATOL, equal_nan=True
    ):
        warnings.warn(
            f"{name} is not symmetric (max difference: "
            f"{np.nanmax(np.abs(D - D.T)):.2e}); only its lower triangle is used, "
            "as R's as.dist does. The upper triangle is discarded.",
            UserWarning,
            stacklevel=3,
        )
    if np.nanmax(np.abs(diagonal)) > DISTANCE_DIAGONAL_TOL:
        warnings.warn(
            f"{name} diagonal is not zero (max: {np.nanmax(np.abs(diagonal)):.2e}); "
            "only its lower triangle is used, as R's as.dist does. The diagonal is "
            "discarded.",
            UserWarning,
            stacklevel=3,
        )

    lower = np.tril(np.ones_like(D, dtype=bool), k=-1)
    reduced = np.zeros_like(D)
    reduced[lower] = D[lower]
    reduced = reduced + reduced.T
    return reduced


def require_finite_non_negative_distances(D: np.ndarray, *, name: str = "diss") -> None:
    """Refuse the input the compiled medoid search cannot run on.

    R's PAM returns a partition for negative and non-finite dissimilarities.
    This project's medoid kernels do not, so ``wcKMedoids`` is the single
    documented narrowing of the public distance contract; see
    ``docs/R_MIGRATION_GUIDE.md``.
    """
    if not np.all(np.isfinite(D)):
        raise ValueError(f"{name} must contain only finite distances")
    if np.any(D < 0):
        raise ValueError(f"{name} distances must be non-negative")


def validate_seqdist_args(
    seqdist_args: Optional[dict],
    *,
    default_method: str = "HAM",
) -> dict[str, object]:
    """Validate and normalize sequence-distance arguments.

    Parameters
    ----------
    seqdist_args : dict or None
        Distance computation arguments

    Returns
    -------
    dict
        Normalized arguments with keys such as ``method``, ``metric``, ``indel``,
        ``substitution_costs``, and ``timecost``.

    Raises
    ------
    ValueError
        If arguments are invalid
    """
    return normalize_seqdist_config(seqdist_args, default_method=default_method)


def validate_kvals(kvals: Sequence[int], max_k: Optional[int] = None) -> list[int]:
    """Validate k values for clustering range.

    Parameters
    ----------
    kvals : sequence of int
        Number of clusters to evaluate
    max_k : int or None
        Maximum allowed k (optional)

    Returns
    -------
    list of int
        Validated k values

    Raises
    ------
    ValueError
        If k values are invalid
    """
    if isinstance(kvals, (str, bytes)):
        raise ValueError("kvals must be a non-empty sequence of integers")
    try:
        raw_kvals = list(kvals)
    except TypeError as exc:
        raise ValueError("kvals must be a non-empty sequence of integers") from exc
    if not raw_kvals:
        raise ValueError("kvals must be non-empty")

    k_list: list[int] = []
    for k in raw_kvals:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Real):
            raise ValueError(f"All kvals must be integers, got {k}")
        try:
            k_float = float(k)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"All kvals must be integers, got {k}") from exc
        if not np.isfinite(k_float) or k_float != np.floor(k_float):
            raise ValueError(f"All kvals must be integers, got {k}")
        k_int = int(k)

        if k_int < 2:
            raise ValueError(f"All kvals must be >= 2 (single cluster is meaningless), got {k_int}")

        if max_k is not None and k_int > max_k:
            raise ValueError(f"kvals cannot exceed max_k={max_k}, got {k_int}")

        k_list.append(k_int)

    if len(set(k_list)) != len(k_list):
        raise ValueError("kvals must not contain duplicates")
    return k_list


def validate_integer_control(value: object, *, name: str) -> int:
    """Return a finite, losslessly integral numeric public control value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an integer")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def validate_linkage_matrix(
    linkage_matrix: np.ndarray, N: int, name: str = "linkage_matrix"
) -> None:
    """Validate that input is a proper SciPy linkage matrix matching data size.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix to validate
    N : int
        Number of observations the linkage should correspond to
    name : str
        Parameter name for error messages

    Raises
    ------
    ValueError
        If matrix has wrong shape or doesn't match data size
    """
    Z = np.asarray(linkage_matrix)

    if Z.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {Z.ndim}D array")

    if Z.shape[1] != 4:
        raise ValueError(f"{name} must have 4 columns (SciPy linkage format), got {Z.shape[1]}")

    expected_rows = N - 1
    if Z.shape[0] != expected_rows:
        raise ValueError(
            f"{name} has {Z.shape[0]} rows but should have {expected_rows} "
            f"for {N} observations (n-1 merges in hierarchical clustering)"
        )


def cut_linkage_by_merge_count(
    linkage_matrix: np.ndarray,
    *,
    n_clusters: int,
) -> np.ndarray:
    """Cut a SciPy linkage by merge order, including inverted hierarchies.

    R's ``cutree(h, k=...)`` applies the first ``n-k`` merges and does not use
    height monotonicity. SciPy's ``cut_tree`` can return fewer than ``k``
    groups for valid centroid and median inversions, so those hierarchies need
    this merge-count formulation.
    """
    hierarchy = np.asarray(linkage_matrix, dtype=float)
    if hierarchy.ndim != 2 or hierarchy.shape[1] != 4:
        raise ValueError("linkage matrix must have shape (n-1, 4)")
    n_obs = int(hierarchy.shape[0] + 1)
    cluster_count = validate_integer_control(n_clusters, name="n_clusters")
    if not 1 <= cluster_count <= n_obs:
        raise ValueError(f"n_clusters must be between 1 and {n_obs}")
    if not np.all(np.isfinite(hierarchy)) or np.any(hierarchy[:, 2:] < 0.0):
        raise ValueError("linkage matrix must contain finite non-negative heights and counts")

    active: dict[int, np.ndarray] = {
        observation: np.asarray([observation], dtype=int) for observation in range(n_obs)
    }
    cut_members = (
        [members.copy() for members in active.values()] if cluster_count == n_obs else None
    )
    for step, row in enumerate(hierarchy):
        left_value, right_value = float(row[0]), float(row[1])
        if left_value != np.floor(left_value) or right_value != np.floor(right_value):
            raise ValueError("linkage matrix child identifiers must be integers")
        left = int(left_value)
        right = int(right_value)
        if left == right or left not in active or right not in active:
            raise ValueError("linkage matrix must describe a proper binary merge tree")
        merged = np.concatenate((active.pop(left), active.pop(right)))
        expected_count = float(merged.size)
        if not np.isclose(float(row[3]), expected_count, rtol=0.0, atol=1e-12):
            raise ValueError("linkage matrix cluster counts are inconsistent with its merges")
        active[n_obs + step] = merged
        if len(active) == cluster_count:
            cut_members = [members.copy() for members in active.values()]

    if cut_members is None:
        raise RuntimeError("linkage cut invariant failed to capture the requested cluster count")
    labels = np.empty(n_obs, dtype=int)
    for label, members in enumerate(
        sorted(cut_members, key=lambda values: int(values.min())), start=1
    ):
        labels[members] = label
    return labels

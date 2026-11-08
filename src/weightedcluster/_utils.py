"""Shared utility functions for clustering operations.

This module provides reusable functions to avoid code duplication across the package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from traminex.diss import weighted_pairwise_sum as _tx_weighted_pairwise_sum

from ._validation import validate_integer_control


def find_medoid(D: np.ndarray, idx: np.ndarray, weights: Optional[np.ndarray] = None) -> int:
    """Find the medoid (observation with minimum average distance) within a subset.

    The medoid is the observation that minimizes the sum of weighted distances
    to all other observations in the subset.

    Parameters
    ----------
    D : np.ndarray
        (N, N) distance matrix
    idx : np.ndarray
        Indices of observations in the subset
    weights : np.ndarray, optional
        Per-observation weights (length N). If None, uniform weights are used.

    Returns
    -------
    int
        Global index of the medoid (within [0, N))

    Raises
    ------
    ValueError
        If idx is empty

    Examples
    --------
    >>> D = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]], dtype=float)
    >>> idx = np.array([0, 1])
    >>> find_medoid(D, idx)
    0
    """
    if idx.size == 0:
        raise ValueError("Cannot find medoid of empty subset")

    if idx.size == 1:
        return int(idx[0])

    # Extract submatrix
    subD = D[np.ix_(idx, idx)]

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if weights is None:
            # Unweighted: sum of distances per row
            w_sub = np.ones(idx.size, dtype=float)
            cost = subD.sum(axis=1)
        else:
            # Weighted: broadcast weights across columns
            w_sub = weights[idx]
            cost = (w_sub.reshape(1, -1) * subD).sum(axis=1)

    # Preserve exact-positive objective ordering when finite products or their
    # reduction leave the ordinary binary64 precision window. Keep the risk
    # predicate local: backend-dispatch tests deliberately replace the fallback
    # module with a narrow test double.
    positive_mask = (w_sub[:, None] > 0.0) & (subD > 0.0)
    exact_required = False
    if np.any(positive_mask):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            products = w_sub[:, None] * subD
        exact_required = bool(
            np.any(
                positive_mask
                & ((products == 0.0) | ~np.isfinite(products) | (products < np.finfo(float).tiny))
            )
        )
        if not exact_required:
            _, weight_exponents = np.frexp(w_sub)
            _, distance_exponents = np.frexp(subD)
            product_exponents = (weight_exponents[:, None] + distance_exponents)[positive_mask]
            count = max(1, int(idx.size))
            rounding_window = 52 - (count - 1).bit_length()
            overflow_exponent = np.frexp(np.finfo(float).max / count)[1]
            exact_required = bool(
                int(np.max(product_exponents)) >= overflow_exponent
                or int(np.max(product_exponents)) - int(np.min(product_exponents))
                >= rounding_window
            )

    if exact_required:
        from .quality import (
            _exact_positive_float_product_component,
            _sum_dyadic_components,
        )

        exact_costs = [
            _sum_dyadic_components(
                [
                    _exact_positive_float_product_component(float(weight), float(distance))
                    for weight, distance in zip(w_sub, row, strict=True)
                    if weight > 0.0 and distance > 0.0
                ]
            )
            for row in subD
        ]
        medoid_local = 0
        for candidate in range(1, len(exact_costs)):
            left_integer, left_exponent = exact_costs[candidate]
            right_integer, right_exponent = exact_costs[medoid_local]
            common_exponent = min(left_exponent, right_exponent)
            if left_integer << (left_exponent - common_exponent) < right_integer << (
                right_exponent - common_exponent
            ):
                medoid_local = candidate
        return int(idx[medoid_local])

    medoid_local = int(np.argmin(cost))
    return int(idx[medoid_local])


def pairwise_sum(D: np.ndarray, idx: np.ndarray) -> float:
    """Compute sum of pairwise distances within a subset.

    Delegates to traminex's ``weighted_pairwise_sum`` (unit weights) — the
    R-anchored weighted-inertia primitive that owns this maths family-wide
    (migration plan Phase 2.5 / DUPLICATES A5–A8).

    Parameters
    ----------
    D : np.ndarray
        (N, N) distance matrix
    idx : np.ndarray
        Indices of observations in the subset

    Returns
    -------
    float
        Sum of all pairwise distances (upper triangle only, no diagonal)

    Examples
    --------
    >>> D = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]], dtype=float)
    >>> pairwise_sum(D, np.array([0, 1, 2]))
    8.0
    """
    return _tx_weighted_pairwise_sum(D, idx)


def generate_random_seed(base_seed: Optional[int], offset: int = 0, *path: int) -> int:
    """Generate a structured child seed for reproducible parallel operations.

    Parameters
    ----------
    base_seed : int or None
        Base random seed. If None, generates a truly random seed.
    offset : int
        First non-negative stream-path component.
    *path : int
        Additional non-negative components for nested stream identities.

    Returns
    -------
    int
        Derived seed value

    Examples
    --------
    >>> generate_random_seed(42, 5) == generate_random_seed(42, 5)
    True
    >>> generate_random_seed(42, 5) != generate_random_seed(43, 4)
    True
    >>> seed = generate_random_seed(None, 0)
    >>> isinstance(seed, int)
    True
    """
    components = [validate_integer_control(offset, name="offset")]
    components.extend(
        validate_integer_control(value, name=f"path[{index}]") for index, value in enumerate(path)
    )
    if any(component < 0 for component in components):
        raise ValueError("offset and path components must be non-negative")
    if base_seed is None:
        root = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    else:
        root = validate_integer_control(base_seed, name="base_seed")
        if root < 0:
            raise ValueError("base_seed must be non-negative")
    sequence = np.random.SeedSequence([root, *components])
    # Several compiled/R-facing consumers require a signed 32-bit seed.
    return int(sequence.generate_state(1, dtype=np.uint32)[0] & np.uint32(0x7FFFFFFF))

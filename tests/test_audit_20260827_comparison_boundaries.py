"""Regression contracts for comparison-method scale and tie boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from tests.r_parity_utils import canonical_partition
from weightedcluster import wcCmpCluster
from weightedcluster._cmpcluster_algorithms import _embed_cmds
from weightedcluster.cmpcluster import _linkage_flexible_beta


def _flexible_tie_distances(delta: float, scale: float = 1.0) -> np.ndarray:
    distances = np.array(
        [
            [0.0, 1.0 + delta, 1.0, 10.0],
            [1.0 + delta, 0.0, 2.0, 9.0],
            [1.0, 2.0, 0.0, 8.0],
            [10.0, 9.0, 8.0, 0.0],
        ],
        dtype=float,
    )
    return distances * scale


def _expected_flexible_heights(delta: float) -> np.ndarray:
    if delta > 0.0:
        return np.array(
            [
                1.0,
                1.625 + 0.625 * delta,
                12.09375 - 0.15625 * delta,
            ],
            dtype=float,
        )
    return np.array(
        [
            1.0 + delta,
            1.625 - 0.25 * delta,
            11.859375 - 0.09375 * delta,
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    ("delta", "scale", "first_pair"),
    [
        pytest.param(5e-13, 1.0, (0, 2), id="near-tie-right"),
        pytest.param(5e-13, 1e-9, (0, 2), id="near-tie-right-small-scale"),
        pytest.param(5e-13, 1e9, (0, 2), id="near-tie-right-large-scale"),
        pytest.param(0.0, 1.0, (0, 1), id="exact-tie-deterministic"),
        pytest.param(-5e-13, 1.0, (0, 1), id="near-tie-left"),
    ],
)
def test_flexible_beta_uses_exact_distance_ordering(
    delta: float,
    scale: float,
    first_pair: tuple[int, int],
) -> None:
    """R-06: a fixed absolute tolerance cannot turn near values into ties."""
    hierarchy = _linkage_flexible_beta(
        squareform(_flexible_tie_distances(delta, scale), checks=False),
        alpha=0.625,
    )

    np.testing.assert_array_equal(hierarchy[0, :2], first_pair)
    np.testing.assert_allclose(
        hierarchy[:, 2],
        _expected_flexible_heights(delta) * scale,
        rtol=3e-15,
        atol=0.0,
    )


def test_public_flexible_beta_near_tie_matches_exact_expected_partition() -> None:
    result = wcCmpCluster(
        _flexible_tie_distances(5e-13),
        maxcluster=3,
        method=["beta.flexible"],
        pam_combine=False,
    )

    np.testing.assert_array_equal(
        result["beta.flexible"]["clustering"]["cluster3"],
        [1, 2, 1, 3],
    )


def test_public_flexible_beta_near_tie_matches_live_r(r_parity_session) -> None:
    distances = _flexible_tie_distances(5e-13)
    python_result = wcCmpCluster(
        distances,
        maxcluster=3,
        method=["beta.flexible"],
        pam_combine=False,
    )
    r_result = r_parity_session.wc_cmp_cluster(
        distances,
        maxcluster=3,
        method="beta.flexible",
        pam_combine=False,
    )

    python_labels = python_result["beta.flexible"]["clustering"]["cluster3"]
    r_labels = r_result["clustering"]["cluster3"]
    assert canonical_partition(python_labels) == canonical_partition(r_labels)
    np.testing.assert_array_equal(r_labels, [1, 2, 1, 3])


WARD_POSITIONS = np.array([0.0, 10.0, 1.0, 20.0], dtype=float)
WARD_DISTANCES = np.abs(WARD_POSITIONS[:, None] - WARD_POSITIONS[None, :])
WARD_HEIGHTS = np.array([1.0, 10.0, 20.506096654409877], dtype=float)


@pytest.mark.parametrize("scale", [1.0, 1e-7, 1e-8, 1e-170, 1e160])
def test_classical_mds_embedding_retains_scale_relative_positive_spectrum(
    scale: float,
) -> None:
    """R-07: finite common scales cannot collapse or overflow the embedding."""
    coordinates = _embed_cmds(WARD_DISTANCES * scale)
    scaled_coordinates = coordinates / scale
    expected_centered = WARD_POSITIONS - WARD_POSITIONS.mean()

    np.testing.assert_allclose(
        scaled_coordinates @ scaled_coordinates.T,
        np.outer(expected_centered, expected_centered),
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        linkage(scaled_coordinates, method="ward", metric="euclidean")[:, 2],
        WARD_HEIGHTS,
        rtol=2e-14,
        atol=0.0,
    )


@pytest.mark.parametrize("scale", [1.0, 1e-7, 1e-8])
def test_public_embed_then_ward_partition_is_invariant_to_positive_scale(scale: float) -> None:
    result = wcCmpCluster(
        WARD_DISTANCES * scale,
        maxcluster=3,
        method=["ward"],
        pam_combine=False,
        embed_then_ward=True,
    )["ward.D2"]["clustering"]

    np.testing.assert_array_equal(result["cluster2"], [1, 2, 1, 2])
    np.testing.assert_array_equal(result["cluster3"], [1, 2, 1, 3])


@pytest.mark.parametrize("scale", [1e-170, 1e160])
def test_public_embed_then_ward_partition_survives_extreme_finite_scale(
    scale: float,
) -> None:
    result = wcCmpCluster(
        WARD_DISTANCES * scale,
        maxcluster=3,
        method=["ward"],
        pam_combine=False,
        embed_then_ward=True,
    )["ward.D2"]["clustering"]

    np.testing.assert_array_equal(result["cluster2"], [1, 2, 1, 2])
    np.testing.assert_array_equal(result["cluster3"], [1, 2, 1, 3])


def test_classical_mds_embedding_preserves_degenerate_zero_contract() -> None:
    coordinates = _embed_cmds(np.zeros((4, 4), dtype=float))

    np.testing.assert_array_equal(coordinates, np.zeros((4, 1), dtype=float))


@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e-170, 1e160])
def test_classical_mds_embedding_drops_only_nonpositive_non_euclidean_axes(
    scale: float,
) -> None:
    base_distances = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    distances = base_distances * scale
    centering = np.eye(3) - np.ones((3, 3), dtype=float) / 3.0
    gram = -0.5 * centering @ (base_distances * base_distances) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    positive = eigenvalues > 0.0
    expected_positive_gram = (
        eigenvectors[:, positive] @ np.diag(eigenvalues[positive]) @ eigenvectors[:, positive].T
    )

    scaled_coordinates = _embed_cmds(distances) / scale

    np.testing.assert_allclose(
        scaled_coordinates @ scaled_coordinates.T,
        expected_positive_gram,
        rtol=2e-14,
        atol=5e-16,
    )

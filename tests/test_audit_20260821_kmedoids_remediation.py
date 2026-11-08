from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from weightedcluster import wcKMedoids, wcKMedRange

pytest_plugins = ["tests.r_parity_utils"]


_QUALITY_NAMES = (
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

_D6 = np.array(
    [
        [0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
        [1.0, 0.0, 1.0, 4.0, 5.0, 6.0],
        [2.0, 1.0, 0.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 0.0, 1.0, 2.0],
        [6.0, 5.0, 4.0, 1.0, 0.0, 1.0],
        [7.0, 6.0, 5.0, 2.0, 1.0, 0.0],
    ],
    dtype=float,
)


def _assert_kmedoids_matches_r(
    actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    assert expected["error"] is None
    np.testing.assert_array_equal(actual["clustering"], expected["clustering"])
    np.testing.assert_allclose(
        [actual["info"][name] for name in expected["info_names"]],
        expected["info"],
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        [actual["stats"][name] for name in _QUALITY_NAMES],
        expected["stats"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "weights",
    [
        np.ones(6, dtype=float),
        np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2]),
        np.array([1.0, 2.0, 0.0, 1.0, 0.0, 1.2]),
    ],
    ids=["uniform", "nonuniform", "zero_containing"],
)
def test_square_matrix_pam_matches_live_r_for_explicit_start(r_parity_session, weights):
    start = np.array([1, 5], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        _D6,
        k=2,
        weights=weights,
        npass=0,
        initialclust=start,
        method="PAM",
        distance_kind="matrix",
    )

    actual = wcKMedoids(
        _D6,
        2,
        weights=weights,
        npass=0,
        initialclust=start,
        method="PAM",
    )

    _assert_kmedoids_matches_r(actual, expected)


def test_square_matrix_pam_multipass_replay_matches_live_r(r_parity_session):
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2])
    starts_1b = np.array([[1, 5], [1, 4], [2, 6]], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        _D6,
        k=2,
        weights=weights,
        npass=3,
        method="PAM",
        distance_kind="matrix",
        replay_starts_1b=starts_1b,
    )

    actual = wcKMedoids(
        _D6,
        2,
        weights=weights,
        npass=3,
        method="PAM",
        replay_starts=starts_1b - 1,
        random_state=999,
    )

    assert expected["replay_consumed"] == starts_1b.shape[0]
    _assert_kmedoids_matches_r(actual, expected)


def test_condensed_pam_retains_live_r_unsupported_branch(r_parity_session):
    expected = r_parity_session.wc_kmedoids_complete_contract(
        _D6,
        k=2,
        npass=0,
        initialclust=[1, 5],
        method="PAM",
        distance_kind="dist",
    )

    assert "not (yet) implemented" in expected["error"].lower()
    with pytest.raises(NotImplementedError, match="condensed distance input"):
        wcKMedoids(
            squareform(_D6),
            2,
            npass=0,
            initialclust=[1, 5],
            method="PAM",
        )

    with pytest.raises(NotImplementedError, match="condensed distance input"):
        wcKMedRange(squareform(_D6), [2], method="PAM")


def test_square_matrix_pam_propagates_through_kmedrange(r_parity_session):
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2])
    hierarchy = linkage(squareform(_D6), method="single")
    expected = r_parity_session.wc_kmedrange_complete_contract(
        _D6,
        kvals=[2, 3],
        weights=weights,
        R=1,
        distance_kind="matrix",
        initial_kind="hclust",
        method="PAM",
        npass=0,
    )

    actual = wcKMedRange(
        _D6,
        [2, 3],
        weights=weights,
        R=1,
        initialclust=hierarchy,
        method="PAM",
        npass=0,
    )

    assert expected["error"] is None
    np.testing.assert_array_equal(
        np.column_stack([actual["clustering"][name] for name in actual["clustering"].column_names]),
        expected["clustering_numeric"],
    )
    np.testing.assert_allclose(
        np.column_stack([actual["stats"][name] for name in _QUALITY_NAMES]),
        expected["stats"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_pamonce_exact_scalar_tie_names_same_medoid_as_live_r(r_parity_session):
    points = np.array(
        [
            [-0.6516269232149341, 0.18852947059760727],
            [1.5435190148377664, -0.3658268296797904],
            [0.4903106954189639, 0.7910641335558763],
            [0.475304359593854, 0.5245871836284709],
            [0.9682686612451152, -0.5072094508862384],
            [-0.8268456280177282, 1.4516735228114135],
            [-0.8299934347321918, -0.3430041573765951],
            [0.611341568736739, 1.0523969446961727],
            [0.21438270030702525, 1.1982709004818117],
        ],
        dtype=float,
    )
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distance, 0.0)
    start = np.array([2, 5, 9], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        distance,
        k=3,
        npass=0,
        initialclust=start,
        method="PAMonce",
        distance_kind="matrix",
    )

    actual = wcKMedoids(
        distance,
        3,
        npass=0,
        initialclust=start,
        method="PAMonce",
    )

    assert expected["clustering"].tolist() == [1, 2, 3, 3, 2, 1, 1, 3, 3]
    _assert_kmedoids_matches_r(actual, expected)


@pytest.mark.parametrize(
    ("start", "expected_labels"),
    [
        ([1, 2, 5], [1, 1, 3, 3, 5, 5]),
        ([1, 2, 3], [1, 2, 3, 3, 3, 3]),
    ],
)
def test_kmedoids_pseudometric_starts_keep_every_medoid_cluster_nonempty(
    r_parity_session,
    start,
    expected_labels,
):
    coordinates = np.array([0.0, 0.0, 1.0, 1.0, 5.0, 5.0])
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    expected = r_parity_session.wc_kmedoids_complete_contract(
        distance,
        k=3,
        npass=0,
        initialclust=start,
        method="KMedoids",
        distance_kind="matrix",
    )

    actual = wcKMedoids(
        distance,
        3,
        npass=0,
        initialclust=start,
        method="KMedoids",
    )

    assert expected["clustering"].tolist() == expected_labels
    assert len(np.unique(actual["clustering"])) == 3
    _assert_kmedoids_matches_r(actual, expected)

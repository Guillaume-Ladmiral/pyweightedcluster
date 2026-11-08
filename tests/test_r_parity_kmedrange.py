from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from tests.r_parity_utils import canonical_partition
from weightedcluster import wcKMedoids, wcKMedRange

pytest_plugins = ["tests.r_parity_utils"]

R_QUALITY_STATS = (
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

D6 = np.array(
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
W6 = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
INITIAL_MEDOIDS_1B = np.array([1, 5], dtype=int)


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence) and not isinstance(right, (str, bytes))
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
        return
    assert left == right


def test_live_oracle_is_weightedcluster_2_0(r_parity_session):
    versions = r_parity_session.oracle_versions()
    assert versions["WeightedCluster"] == "2.0"
    assert versions["TraMineR"] == "2.2.14"


@pytest.mark.parametrize("method", ["KMedoids", "PAMonce"])
def test_fixed_init_wckmedoids_matches_live_r_partition_and_stats(r_parity_session, method):
    py = wcKMedoids(
        D6,
        2,
        weights=W6,
        npass=1,
        initialclust=INITIAL_MEDOIDS_1B,
        random_state=42,
    )
    r = r_parity_session.wc_kmedoids_method(
        D6,
        k=2,
        weights=W6,
        initialclust=INITIAL_MEDOIDS_1B,
        method=method,
        npass=0,
    )

    assert "error" not in r
    assert canonical_partition(py["clustering"]) == ((0, 1, 2), (3, 4, 5))
    assert canonical_partition(r["clustering"]) == canonical_partition(py["clustering"])
    np.testing.assert_allclose(
        [py["stats"][name] for name in R_QUALITY_STATS],
        [r["stats"][name] for name in R_QUALITY_STATS],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_live_r_condensed_pam_method_is_not_implemented(r_parity_session):
    r = r_parity_session.wc_kmedoids_method(
        D6,
        k=2,
        weights=W6,
        initialclust=INITIAL_MEDOIDS_1B,
        method="PAM",
        npass=0,
    )

    assert "not (yet) implemented" in r["error"].lower()
    assert inspect.signature(wcKMedoids).parameters["method"].default == "PAMonce"
    with pytest.raises(NotImplementedError, match="not \\(yet\\) implemented"):
        wcKMedoids(
            squareform(D6),
            2,
            weights=W6,
            initialclust=INITIAL_MEDOIDS_1B,
            method="PAM",
        )


def test_wckmedrange_structure_matches_live_r_contract(r_parity_session):
    r = r_parity_session.wc_kmedrange_contract(
        D6,
        kvals=[2, 3],
        weights=W6,
        seed=7,
    )
    assert r["class"] == ["clustrange", "list"]
    assert r["names"] == ["R", "kvals", "clustering", "stats"]
    assert r["kvals"] == [2, 3]
    assert r["clustering_names"] == ["cluster2", "cluster3"]
    assert r["stats_names"] == list(R_QUALITY_STATS)
    assert r["stats_shape"] == [2, 10]

    py = wcKMedRange(D6, kvals=[2, 3], weights=W6, random_state=7)
    assert py["kvals"] == [2, 3]
    assert list(py["clustering"]) == ["cluster2", "cluster3"]
    assert set(R_QUALITY_STATS).issubset(py["stats"])
    assert all(len(py["stats"][name]) == 2 for name in R_QUALITY_STATS)


def test_wckmedrange_stats_match_r_for_the_same_python_labels(r_parity_session):
    py = wcKMedRange(D6, kvals=[2, 3], weights=W6, random_state=7)
    r = r_parity_session.as_clustrange_default(D6, py["clustering"], weights=W6)

    assert r["stats_names"] == list(R_QUALITY_STATS)
    py_stats = np.column_stack([py["stats"][name] for name in R_QUALITY_STATS])
    np.testing.assert_allclose(
        py_stats,
        r["stats_matrix"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_wckmedrange_full_stochastic_run_is_seed_reproducible():
    kwargs = {
        "kvals": [2, 3],
        "weights": W6,
        "R": 5,
        "npass": 3,
        "random_state": 13,
    }
    first = wcKMedRange(D6, **kwargs)
    second = wcKMedRange(D6, **kwargs)

    assert list(first) == ["R", "kvals", "clustering", "stats", "boot"]
    _assert_nested_equal(first, second)


def test_stochastic_r_and_python_labels_are_not_a_parity_anchor():
    """Document why cross-runtime seeded labels are intentionally not compared."""
    assert "random_state" in inspect.signature(wcKMedRange).parameters
    assert "seed" not in inspect.signature(wcKMedRange).parameters

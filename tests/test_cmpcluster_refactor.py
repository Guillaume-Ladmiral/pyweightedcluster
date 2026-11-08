from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform

import weightedcluster.cmpcluster as cmpcluster_module
from weightedcluster._cmpcluster_algorithms import _cluster_diana, _embed_cmds
from weightedcluster._cmpcluster_controls import (
    _canonical_cmpcluster_method,
    _normalize_cmpcluster_methods,
    _validate_cmpcluster_replay_starts,
)
from weightedcluster._cmpcluster_result import (
    _R_QUALITY_NAMES,
    _build_clustrange_family_result,
)
from weightedcluster.schemas import (
    ClusteringTable,
    ClustrangeFamilyResult,
    KMedRangeResult,
    rows_to_stats_table,
)

ALL_UNWEIGHTED = [
    "ward.D",
    "ward.D2",
    "single",
    "complete",
    "average",
    "mcquitty",
    "median",
    "centroid",
    "pam",
    "diana",
    "beta.flexible",
]


DIANA_DISTANCE = np.array(
    [
        [0.0, 0.2, 1.0, 1.1],
        [0.2, 0.0, 1.1, 1.1],
        [1.0, 1.1, 0.0, 0.2],
        [1.1, 1.1, 0.2, 0.0],
    ],
    dtype=float,
)


def test_no_optional_linkage_backend_is_selected_at_import() -> None:
    """The optional `fastcluster` selection is gone (finding F-11).

    `_select_linkage_backend` preferred an optional Python `fastcluster` that no
    documented installation provided, was covered only by injecting a fake
    module, and silently changed results for anyone who happened to have the
    package installed. Every dissimilarity-matrix route now goes through
    `_linkage_hclust`, which reproduces `fastcluster::hclust` in Python; SciPy
    serves only the coordinate-embedding route.
    """
    assert not hasattr(cmpcluster_module, "_select_linkage_backend")
    assert cmpcluster_module._fast_linkage is scipy_linkage
    source = Path(cmpcluster_module.__file__).read_text(encoding="utf-8")
    assert "import fastcluster" not in source
    assert "from fastcluster" not in source


def test_embed_cmds_centers_and_reconstructs_euclidean_distances() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]],
        dtype=float,
    )
    distances = squareform(pdist(points))

    embedded = _embed_cmds(distances)

    np.testing.assert_allclose(embedded.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        squareform(pdist(embedded)),
        distances,
        rtol=0.0,
        atol=1e-12,
    )


def test_embed_cmds_returns_one_zero_axis_for_degenerate_distances() -> None:
    np.testing.assert_array_equal(_embed_cmds(np.zeros((3, 3))), np.zeros((3, 1)))


def test_cluster_diana_has_literal_deterministic_partitions() -> None:
    labels = _cluster_diana(DIANA_DISTANCE, 4)

    assert list(labels) == [2, 3, 4]
    np.testing.assert_array_equal(labels[2], [1, 1, 2, 2])
    np.testing.assert_array_equal(labels[3], [1, 2, 3, 3])
    np.testing.assert_array_equal(labels[4], [1, 2, 3, 4])


def test_cluster_diana_preserves_r_merge_order_for_equal_split_levels() -> None:
    rng = np.random.default_rng(44_713)
    points = rng.uniform(-3.0, 3.0, size=(13, 6))
    distances = squareform(pdist(points))

    labels = _cluster_diana(distances, 5)

    np.testing.assert_array_equal(labels[2], [1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1])
    np.testing.assert_array_equal(labels[3], [1, 2, 2, 2, 3, 2, 3, 3, 2, 3, 2, 2, 2])
    np.testing.assert_array_equal(labels[4], [1, 2, 3, 3, 4, 3, 4, 4, 3, 4, 2, 3, 2])
    np.testing.assert_array_equal(labels[5], [1, 2, 3, 3, 4, 3, 4, 5, 3, 4, 2, 3, 2])


def test_cluster_diana_handles_complete_splinter_shift_like_r() -> None:
    distances = np.array(
        [
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 8.0, 5.0],
            [1.0, 8.0, 0.0, 4.0],
            [2.0, 5.0, 4.0, 0.0],
        ]
    )

    labels = _cluster_diana(distances, 4)

    np.testing.assert_array_equal(labels[2], [1, 1, 2, 1])
    np.testing.assert_array_equal(labels[3], [1, 1, 2, 3])
    np.testing.assert_array_equal(labels[4], [1, 2, 3, 4])


def test_normalize_cmpcluster_methods_preserves_order_and_deduplicates() -> None:
    assert _normalize_cmpcluster_methods("all", has_weights=False) == ALL_UNWEIGHTED
    assert _normalize_cmpcluster_methods("ALL", has_weights=True) == [
        *ALL_UNWEIGHTED[:8],
        "pam",
    ]
    assert _normalize_cmpcluster_methods(["Single", "SINGLE"], has_weights=False) == ["single"]
    assert _normalize_cmpcluster_methods(["single", "ALL"], has_weights=False) == ALL_UNWEIGHTED


def test_cmpcluster_method_aliases_and_errors_are_exact() -> None:
    assert _canonical_cmpcluster_method("ward") == "ward.d2"
    assert _canonical_cmpcluster_method("weighted") == "mcquitty"
    assert _canonical_cmpcluster_method("flexible") == "beta.flexible"
    assert _canonical_cmpcluster_method("ward.D") == "ward.d"
    with pytest.raises(ValueError, match="method must contain at least one"):
        _normalize_cmpcluster_methods([], has_weights=False)
    with pytest.raises(ValueError, match="unknown clustering method unknown"):
        _normalize_cmpcluster_methods("unknown", has_weights=False)
    with pytest.raises(
        ValueError,
        match=r"methods diana, beta\.flexible cannot be used with weights\.",
    ):
        _normalize_cmpcluster_methods("flexible", has_weights=True)
    with pytest.raises(ValueError, match="unknown clustering method unknown"):
        _normalize_cmpcluster_methods(
            ["unknown", "diana"],
            has_weights=True,
        )


def test_validate_cmpcluster_replay_starts_accepts_zero_based_plans() -> None:
    raw = {
        2: [0, 3],
        3: np.array([0, 2, 3], dtype=np.int64),
        4: np.array([0, 1, 2, 3], dtype=np.int64),
    }

    actual = _validate_cmpcluster_replay_starts(
        raw,
        maxcluster=3,
        n_obs=4,
    )

    assert actual is not None
    assert list(actual) == [2, 3]
    np.testing.assert_array_equal(actual[2], [[0, 3]])
    np.testing.assert_array_equal(actual[3], [[0, 2, 3]])
    assert actual[2].dtype == np.dtype(int)
    assert actual[3].dtype == np.dtype(int)
    np.testing.assert_array_equal(raw[3], [0, 2, 3])
    assert not np.shares_memory(actual[3], raw[3])
    assert (
        _validate_cmpcluster_replay_starts(
            None,
            maxcluster=3,
            n_obs=4,
        )
        is None
    )


@pytest.mark.parametrize(
    ("plans", "message"),
    [
        ({2: [0, 1]}, "missing the plan for k=3"),
        ({2: [[0, 1]], 3: [0, 1, 2]}, "zero-based vector of length 2"),
        ({2: [0], 3: [0, 1, 2]}, "zero-based vector of length 2"),
        ({2: [0.0, 1.0], 3: [0, 1, 2]}, "zero-based integer indices"),
        ({2: [-1, 1], 3: [0, 1, 2]}, "out-of-range zero-based index"),
        ({2: [0, 4], 3: [0, 1, 2]}, "out-of-range zero-based index"),
        ({2: [1, 1], 3: [0, 1, 2]}, "unique medoids"),
    ],
)
def test_validate_cmpcluster_replay_starts_rejects_invalid_plans(
    plans: dict[int, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_cmpcluster_replay_starts(
            plans,
            maxcluster=3,
            n_obs=4,
        )


def test_build_clustrange_family_result_preserves_schema_and_order() -> None:
    stats_row = {name: float(index + 1) for index, name in enumerate(_R_QUALITY_NAMES)}
    method_result = KMedRangeResult(
        {
            "R": 1,
            "kvals": [2],
            "clustering": ClusteringTable(
                {"cluster2": np.array([1, 1, 2], dtype=int)},
                column_names=["cluster2"],
            ),
            "stats": rows_to_stats_table(
                [stats_row],
                keys=_R_QUALITY_NAMES,
                row_names=["cluster2"],
            ),
        }
    )

    actual = _build_clustrange_family_result(
        {"single": method_result},
        methods=["single"],
        pam_combine=False,
        kvals=[2],
        embed_then_ward=True,
        random_state=17,
    )

    assert isinstance(actual, ClustrangeFamilyResult)
    assert actual.r_class == ["clustrangefamily"]
    assert list(actual) == ["single", "param", "allstats"]
    assert actual["single"] is method_result
    assert list(actual["param"]) == ["method", "pam.combine", "all.methods", "kvals"]
    assert actual["param"]["method"] == ["single"]
    assert actual["param"]["pam.combine"] is False
    assert actual["param"]["all.methods"] == ["single"]
    assert actual["param"]["kvals"] == [2]
    assert actual["param"].additions["embed_then_ward"] is True
    assert actual["param"].additions["random_state"] == 17
    assert list(actual["allstats"]) == [*_R_QUALITY_NAMES, "method", "ngroup"]
    assert actual["allstats"].row_names == ["single.cluster2"]
    assert actual["allstats"]["method"] == ["single"]
    assert actual["allstats"]["ngroup"] == [2]
    assert [actual["allstats"][name][0] for name in _R_QUALITY_NAMES] == [
        float(index + 1) for index in range(len(_R_QUALITY_NAMES))
    ]

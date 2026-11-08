from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform

import weightedcluster.cmpcluster as cmpcluster_module
from weightedcluster import wcCmpCluster
from weightedcluster._cmpcluster_controls import _normalize_cmpcluster_methods
from weightedcluster._validation import cut_linkage_by_merge_count
from weightedcluster.plot import _seqclararange_plot_data
from weightedcluster.schemas import SeqClaraRangeResult, SeqClaraResult

pytest_plugins = ["tests.r_parity_utils"]


def _canonical_partition(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    return labels[:, None] == labels[None, :]


def test_cmpcluster_explicit_methods_use_all_presentation_names_and_stable_deduplication() -> None:
    assert _normalize_cmpcluster_methods(
        ["WARD.D2", "ward", "Single", "SINGLE", "weighted", "mcquitty"],
        has_weights=False,
    ) == ["ward.D2", "single", "mcquitty"]

    points = np.array([[0.0], [0.7], [2.1], [5.0]], dtype=float)
    distances = squareform(pdist(points))
    explicit = wcCmpCluster(
        distances,
        maxcluster=3,
        method=["WARD.D2", "single", "ward"],
        pam_combine=False,
    )
    expanded = wcCmpCluster(distances, maxcluster=3, method="all", pam_combine=False)

    assert list(explicit) == ["ward.D2", "single", "param", "allstats"]
    assert explicit["param"]["method"] == ["ward.D2", "single"]
    assert explicit["param"]["all.methods"] == ["ward.D2", "single"]
    for method_name in ("ward.D2", "single"):
        for k in (2, 3):
            np.testing.assert_array_equal(
                _canonical_partition(explicit[method_name]["clustering"][f"cluster{k}"]),
                _canonical_partition(expanded[method_name]["clustering"][f"cluster{k}"]),
            )


def test_precomputed_unique_ward_d2_uses_r_slot_backend(monkeypatch) -> None:
    points = np.array([[0.0, 0.0], [0.2, 0.7], [1.9, 0.3], [4.2, 2.6], [8.0, 0.1]])
    distances = squareform(pdist(points))
    calls: list[str] = []
    real_local_linkage = cmpcluster_module._linkage_hclust

    def _spy_local_linkage(*args, **kwargs):
        calls.append(str(kwargs["method"]))
        return real_local_linkage(*args, **kwargs)

    def _unexpected_fast_linkage(*args, **kwargs):
        raise AssertionError("precomputed ward.D2 must preserve R's update tie order")

    monkeypatch.setattr(cmpcluster_module, "_linkage_hclust", _spy_local_linkage)
    monkeypatch.setattr(cmpcluster_module, "_fast_linkage", _unexpected_fast_linkage)

    result = wcCmpCluster(
        distances,
        maxcluster=3,
        method="ward.D2",
        pam_combine=False,
    )

    assert calls == ["ward.d2"]
    assert "ward.D2" in result


def test_tied_ward_d2_retains_r_tie_order_in_local_backend(monkeypatch) -> None:
    distances = np.ones((7, 7), dtype=float)
    np.fill_diagonal(distances, 0.0)
    calls: list[str] = []
    real_local_linkage = cmpcluster_module._linkage_hclust

    def _spy_local_linkage(*args, **kwargs):
        calls.append(str(kwargs["method"]))
        return real_local_linkage(*args, **kwargs)

    def _unexpected_fast_linkage(*args, **kwargs):
        raise AssertionError("tied ward.D2 must preserve the R slot-order backend")

    monkeypatch.setattr(cmpcluster_module, "_linkage_hclust", _spy_local_linkage)
    monkeypatch.setattr(cmpcluster_module, "_fast_linkage", _unexpected_fast_linkage)

    result = wcCmpCluster(
        distances,
        maxcluster=5,
        method="ward.D2",
        pam_combine=False,
    )

    assert calls == ["ward.d2"]
    assert "ward.D2" in result


def test_unique_input_with_generated_ward_tie_retains_r_slot_order(
    monkeypatch,
    r_parity_session,
) -> None:
    distances = np.array(
        [
            [0.0, 12.0, 5.0, 7.0, 13.0],
            [12.0, 0.0, 1.0, 2.0, 9.0],
            [5.0, 1.0, 0.0, 3.0, 11.0],
            [7.0, 2.0, 3.0, 0.0, 4.0],
            [13.0, 9.0, 11.0, 4.0, 0.0],
        ]
    )
    assert np.unique(distances[np.triu_indices(5, k=1)]).size == 10
    expected = r_parity_session.fastcluster_hclust_contract(
        distances,
        method="ward.D2",
        maxcluster=3,
    )
    calls: list[str] = []
    real_local_linkage = cmpcluster_module._linkage_hclust

    def _spy_local_linkage(*args, **kwargs):
        calls.append(str(kwargs["method"]))
        return real_local_linkage(*args, **kwargs)

    def _unexpected_fast_linkage(*args, **kwargs):
        raise AssertionError("input uniqueness cannot certify R tie order after Ward updates")

    monkeypatch.setattr(cmpcluster_module, "_linkage_hclust", _spy_local_linkage)
    monkeypatch.setattr(cmpcluster_module, "_fast_linkage", _unexpected_fast_linkage)

    actual = wcCmpCluster(
        distances,
        maxcluster=3,
        method="ward.D2",
        pam_combine=False,
    )["ward.D2"]

    assert calls == ["ward.d2"]
    for column, k in enumerate((2, 3)):
        np.testing.assert_array_equal(
            _canonical_partition(actual["clustering"][f"cluster{k}"]),
            _canonical_partition(expected["cuts"][:, column]),
        )


def test_fast_ward_d2_matches_local_heights_and_partitions_on_unique_adversarial_inputs() -> None:
    fixtures = [
        np.array(
            [
                [0.0, 1.0, 4.0, 9.0, 15.0],
                [1.0, 0.0, 7.0, 13.0, 22.0],
                [4.0, 7.0, 0.0, 18.0, 31.0],
                [9.0, 13.0, 18.0, 0.0, 40.0],
                [15.0, 22.0, 31.0, 40.0, 0.0],
            ]
        ),
        squareform(
            pdist(
                np.random.default_rng(20260821).normal(size=(17, 5)),
                metric="euclidean",
            )
        ),
    ]

    for distances in fixtures:
        local = cmpcluster_module._linkage_hclust(distances, method="ward.d2")
        fast = scipy_linkage(squareform(distances, checks=False), method="ward")
        np.testing.assert_allclose(
            np.sort(fast[:, 2]),
            np.sort(local[:, 2]),
            rtol=0.0,
            atol=1e-12,
        )
        for k in range(2, distances.shape[0] + 1):
            np.testing.assert_array_equal(
                _canonical_partition(cut_linkage_by_merge_count(fast, n_clusters=k)),
                _canonical_partition(cut_linkage_by_merge_count(local, n_clusters=k)),
            )


def test_adjudicated_ward_d2_routes_match_direct_r_heights_and_partitions(
    r_parity_session,
) -> None:
    unique_pseudometric = np.array(
        [
            [0.0, 1.0, 4.0, 9.0, 15.0],
            [1.0, 0.0, 7.0, 13.0, 22.0],
            [4.0, 7.0, 0.0, 18.0, 31.0],
            [9.0, 13.0, 18.0, 0.0, 40.0],
            [15.0, 22.0, 31.0, 40.0, 0.0],
        ]
    )
    tied = np.ones((7, 7), dtype=float)
    np.fill_diagonal(tied, 0.0)

    for distances in (unique_pseudometric, tied):
        maxcluster = distances.shape[0] - 1
        expected = r_parity_session.fastcluster_hclust_contract(
            distances,
            method="ward.D2",
            maxcluster=maxcluster,
        )
        hierarchy = cmpcluster_module._linkage_hclust(distances, method="ward.d2")
        np.testing.assert_allclose(
            np.sort(hierarchy[:, 2]),
            np.sort(expected["height"]),
            rtol=0.0,
            atol=1e-12,
        )
        actual = wcCmpCluster(
            distances,
            maxcluster=maxcluster,
            method="ward.D2",
            pam_combine=False,
        )["ward.D2"]
        for column, k in enumerate(range(2, maxcluster + 1)):
            np.testing.assert_array_equal(
                _canonical_partition(actual["clustering"][f"cluster{k}"]),
                _canonical_partition(expected["cuts"][:, column]),
            )


def test_ari_boxplot_direct_r_capture_and_deliberate_per_k_python_correction(
    r_parity_session,
) -> None:
    kvals = [4, 3]
    arimatrices = [
        np.column_stack(([0.10, 0.11, 0.12, 0.13], [0.90, 0.91, 0.92, 0.93])),
        np.column_stack(([0.20, 0.21, 0.22, 0.23], [0.80, 0.81, 0.82, 0.83])),
    ]
    iterations = [1, 2]

    installed_r = r_parity_session.seqclararange_ari_boxplot_data(
        kvals=kvals,
        arimatrices=arimatrices,
        iterations=iterations,
    )

    np.testing.assert_array_equal(installed_r[4], arimatrices[0][:, 0])
    np.testing.assert_array_equal(installed_r[3], arimatrices[1][:, 0])

    value = SeqClaraRangeResult(
        {
            "kvals": kvals,
            "clara": {
                k: SeqClaraResult(
                    {
                        "arimatrix": matrix,
                        "iteration": iteration,
                        "iter.objective": np.arange(matrix.shape[0], dtype=float),
                    }
                )
                for k, matrix, iteration in zip(
                    kvals,
                    arimatrices,
                    iterations,
                    strict=True,
                )
            },
            "clustering": {},
            "stats": {"PBM": [1.0, 2.0], "DB": [1.0, 2.0], "XB": [1.0, 2.0], "AMS": [1.0, 2.0]},
            "quality_stats": {},
        }
    )
    corrected = _seqclararange_plot_data(value, stat="ari.boxplot")

    np.testing.assert_array_equal(corrected["values"][0], arimatrices[0][:, 0])
    np.testing.assert_array_equal(corrected["values"][1], arimatrices[1][:, 1])

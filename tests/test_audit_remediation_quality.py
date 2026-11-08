from __future__ import annotations

import numpy as np
import pytest

from weightedcluster._fallback import cluster_quality
from weightedcluster.quality import compute_internal_indices, db_index, pbm_index, xb_index


def test_pure_quality_zero_mass_cluster_propagates_undefined_residual_terms():
    D = np.array(
        [[0.0, 1.0, 3.0, 4.0], [1.0, 0.0, 2.0, 3.0], [3.0, 2.0, 0.0, 1.0], [4.0, 3.0, 1.0, 0.0]]
    )
    stats, _asw, _asww = cluster_quality(D, [0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    for key in ("CH", "R2", "CHsq", "R2sq"):
        assert np.isnan(stats[key]), (key, stats[key])


def test_zero_mass_quality_contract_aligns_across_all_backends():
    core = pytest.importorskip("weightedcluster._wc_core")
    numba_backend = pytest.importorskip("weightedcluster._numba")
    D = np.array(
        [
            [0.0, 1.0, 3.0, 4.0],
            [1.0, 0.0, 2.0, 3.0],
            [3.0, 2.0, 0.0, 1.0],
            [4.0, 3.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    weights = np.array([0.0, 0.0, 1.0, 1.0])
    results = [
        cluster_quality(D, labels, weights),
        core.cluster_quality(D, labels, weights),
        numba_backend.cluster_quality_numba(D, labels, weights),
    ]
    expected_stats = {
        "PBC": np.nan,
        "HG": np.nan,
        "HGSD": np.nan,
        "ASW": np.nan,
        "ASWw": 1.0,
        "CH": np.nan,
        "R2": np.nan,
        "CHsq": np.nan,
        "R2sq": np.nan,
        "HC": np.nan,
    }
    expected_asw = {1: np.nan, 2: np.nan}
    expected_asww = {1: np.nan, 2: 1.0}
    for stats, asw, asww in results:
        for key, expected in expected_stats.items():
            if np.isnan(expected):
                assert np.isnan(stats[key]), (key, stats[key])
            else:
                assert stats[key] == pytest.approx(expected, abs=1e-12)
        for actual, expected in ((asw, expected_asw), (asww, expected_asww)):
            assert actual.keys() == expected.keys()
            np.testing.assert_allclose(
                list(actual.values()),
                list(expected.values()),
                atol=1e-12,
                rtol=0.0,
                equal_nan=True,
            )


def _published_internal_indices(D: np.ndarray, labels: np.ndarray):
    levels = np.unique(labels)
    medoids = []
    scatters = []
    ek = 0.0
    for level in levels:
        members = np.flatnonzero(labels == level)
        medoid = members[np.argmin(D[np.ix_(members, members)].sum(axis=1))]
        medoids.append(int(medoid))
        distances = D[members, medoid]
        scatters.append(float(distances.mean()))
        ek += float(distances.sum())
    medoids = np.asarray(medoids)
    between = D[np.ix_(medoids, medoids)]
    db_terms = []
    for i in range(len(levels)):
        ratios = [
            (scatters[i] + scatters[j]) / between[i, j]
            for j in range(len(levels))
            if i != j and between[i, j] > 0.0
        ]
        db_terms.append(max(ratios, default=0.0))
    db = float(np.mean(db_terms))
    positive = between[np.triu_indices(len(levels), 1)]
    positive = positive[positive > 0.0]
    xb = sum(
        float((D[np.flatnonzero(labels == level), medoids[i]] ** 2).sum())
        for i, level in enumerate(levels)
    ) / (D.shape[0] * float(positive.min()) ** 2)
    global_medoid = int(np.argmin(D.sum(axis=1)))
    e1 = float(D[global_medoid].sum())
    dk = float(between.max())
    pbm = ((e1 * dk) / (len(levels) * ek)) ** 2
    return db, xb, pbm


def test_internal_indices_match_independent_published_formulas():
    points = np.array([[0.0], [1.0], [4.0], [6.0], [10.0], [13.0]])
    D = np.abs(points - points.T)
    labels = np.array([0, 0, 1, 1, 2, 2])
    expected = _published_internal_indices(D, labels)
    actual = compute_internal_indices(D, labels)
    assert actual["DB"] == pytest.approx(expected[0])
    assert actual["XB"] == pytest.approx(expected[1])
    assert actual["PBM"] == pytest.approx(expected[2])
    assert db_index(D, labels) == actual["DB"]
    assert xb_index(D, labels) == actual["XB"]
    assert pbm_index(D, labels) == actual["PBM"]


def test_davies_bouldin_keeps_valid_zero_cluster_contributions():
    D = np.array(
        [[0.0, 0.0, 3.0, 3.0], [0.0, 0.0, 3.0, 3.0], [3.0, 3.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]]
    )
    assert db_index(D, [0, 0, 1, 1]) == 0.0


def test_internal_indices_are_undefined_for_one_cluster():
    D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

    values = compute_internal_indices(D, [1, 1, 1])

    assert all(np.isnan(values[name]) for name in ("DB", "XB", "PBM"))
    assert np.isnan(db_index(D, [1, 1, 1]))


@pytest.mark.parametrize("function", [db_index, xb_index, pbm_index])
def test_public_internal_index_wrappers_have_complete_docstrings(function):
    assert function.__doc__
    for word in ("Parameters", "Returns", "labels"):
        assert word in function.__doc__


def test_importerror_from_selected_quality_backend_propagates(monkeypatch):
    core = pytest.importorskip("weightedcluster._wc_core")
    from weightedcluster import wcClusterQuality

    def raises_after_import(*args, **kwargs):
        raise ImportError("backend computation failed")

    monkeypatch.setattr(core, "cluster_quality", raises_after_import)
    D = np.array(
        [[0.0, 1.0, 4.0, 5.0], [1.0, 0.0, 3.0, 4.0], [4.0, 3.0, 0.0, 1.0], [5.0, 4.0, 1.0, 0.0]]
    )
    with pytest.raises(ImportError, match="backend computation failed"):
        wcClusterQuality(D, [0, 0, 1, 1])


def test_missing_compiled_quality_symbol_selects_an_available_fallback(monkeypatch):
    core = pytest.importorskip("weightedcluster._wc_core")
    from weightedcluster import wcClusterQuality

    D = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 3.0, 4.0],
            [4.0, 3.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )
    expected = wcClusterQuality(D, [0, 0, 1, 1])
    monkeypatch.delattr(core, "cluster_quality")

    actual = wcClusterQuality(D, [0, 0, 1, 1])

    for key in ("PBC", "HG", "ASW", "R2", "R2sq"):
        assert actual["stats"][key] == pytest.approx(expected["stats"][key])

from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import seqclararange


def _toy_sequences() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )


def test_seqclararange_stats_follow_r_core_columns():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=3,
        sample_size=4,
        kvals=[2, 3],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=17,
    )
    expected = [
        "Avg dist",
        "PBM",
        "DB",
        "XB",
        "AMS",
        "ARI>0.8",
        "JC>0.8",
        "Best iter",
    ]
    assert list(out["stats"].keys()) == expected
    for k in expected:
        assert len(out["stats"][k]) == 2
    assert all(1 <= int(v) <= 3 for v in out["stats"]["Best iter"])


def test_seqclararange_exposes_extended_quality_stats_separately():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=18,
    )
    assert "quality_stats" in out
    q = out["quality_stats"]
    assert "ASW" in q and "CH" in q and "PBC" in q
    assert len(q["ASW"]) == 1


def test_seqclararange_fuzzy_quality_stats_include_fuzzy_metrics():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        method="fuzzy",
        m=1.6,
        criteria="distance",
        random_state=22,
    )
    q = out["quality_stats"]
    assert "FPC" in q and "FPE" in q and "XBf" in q
    assert len(q["FPC"]) == 1
    assert np.isfinite(q["FPC"][0])
    assert np.isfinite(q["FPE"][0])


def test_seqclararange_clara_contains_r_style_iteration_fields():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=4,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=19,
    )
    c = out["clara"][2]
    required = [
        "iter.objective",
        "evol.diss",
        "objective",
        "iteration",
        "criteria",
        "method",
        "avg_dist",
        "pbm",
        "db",
        "xb",
        "ams",
        "ari08",
        "jc08",
        "k",
    ]
    for key in required:
        assert key in c
    assert len(c["iter.objective"]) == 4
    assert len(c["evol.diss"]) == 4
    it = int(c["iteration"])
    assert 1 <= it <= 4
    assert c["objective"] == pytest.approx(float(c["iter.objective"][it - 1]))
    # WeightedCluster 2.0 stores the position in kvals here (not the value).
    assert c["k"] == 1
    assert c["ngroup"] == 2


def test_seqclararange_keep_diss_exports_distances_to_medoids():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        keep_diss=True,
        random_state=20,
    )
    c = out["clara"][2]
    assert "diss" in c
    Dm = np.asarray(c["diss"], dtype=float)
    assert Dm.shape[0] == X.shape[0]
    assert Dm.shape[1] == 2


def test_seqclararange_stability_false_returns_missing_stability_values():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=3,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        stability=False,
        random_state=21,
    )
    assert np.isnan(out["stats"]["ARI>0.8"][0])
    assert np.isnan(out["stats"]["JC>0.8"][0])
    assert "ARImean" not in out["quality_stats"]
    c = out["clara"][2]
    assert np.isnan(c["ari08"])
    assert np.isnan(c["jc08"])
    assert c["stability"] is None

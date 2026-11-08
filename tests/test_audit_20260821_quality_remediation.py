from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from weightedcluster import (
    _fallback,
    _numba,
    _wc_core,
    as_clustrange,
    seqclararange,
    wcClusterQuality,
    wcSilhouetteObs,
)
from weightedcluster._seqdist import sequences_distance
from weightedcluster.aggregate import wcAggregateCases


def _assert_quality_silhouettes_match_live_r(actual: dict, expected: dict) -> None:
    stat_names = expected["stats_names"]
    expected_stats = dict(zip(stat_names, expected["stats"], strict=True))
    for name in ("ASW", "ASWw"):
        np.testing.assert_allclose(
            actual["stats"][name],
            expected_stats[name],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    np.testing.assert_allclose(
        np.column_stack(
            [
                [actual["ASW"][key] for key in actual["ASW"]],
                [actual["ASWw"][key] for key in actual["ASWw"]],
            ]
        ),
        expected["asw"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_zero_denominator_silhouettes_match_live_r_and_every_quality_tier(
    r_parity_session,
):
    distance = np.zeros((4, 4), dtype=float)
    labels1 = np.array([1, 1, 2, 2], dtype=int)
    labels0 = labels1 - 1
    weights = np.ones(4, dtype=float)

    expected = r_parity_session.wc_cluster_quality_complete_contract(
        distance,
        labels1,
        weights=weights,
        distance_kind="matrix",
    )
    actual = wcClusterQuality(distance, labels1, weights=weights)

    assert expected["error"] is None
    _assert_quality_silhouettes_match_live_r(actual, expected)
    assert np.isnan(wcSilhouetteObs(distance, labels1, weights=weights, measure="ASW")).all()
    assert np.isnan(wcSilhouetteObs(distance, labels1, weights=weights, measure="ASWw")).all()

    for kernel in (
        _fallback.cluster_quality,
        _numba.cluster_quality_numba,
        _wc_core.cluster_quality,
    ):
        stats, asw, asww = kernel(distance, labels0, weights=weights)
        assert np.isnan(stats["ASW"])
        assert np.isnan(stats["ASWw"])
        assert all(np.isnan(value) for value in asw.values())
        assert all(np.isnan(value) for value in asww.values())


def test_zero_weight_other_cluster_observation_silhouette_matches_live_r(
    r_parity_session,
):
    points = np.array([0.0, 0.2, 1.0, 1.3, 2.4, 2.6], dtype=float)
    distance = np.abs(points[:, None] - points[None, :])
    labels = np.array([1, 1, 1, 2, 2, 2], dtype=int)
    weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

    expected = r_parity_session.wc_silhouette_obs_complete_contract(
        distance,
        labels,
        weights=weights,
        measure="ASWw",
        distance_kind="matrix",
    )
    actual = wcSilhouetteObs(distance, labels, weights=weights, measure="ASWw")

    assert expected["error"] is None
    np.testing.assert_allclose(
        actual,
        expected["values"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual[:3], np.ones(3, dtype=float))
    assert np.isnan(actual[3:]).all()


def test_numba_hg_accumulates_equal_distance_buckets_in_r_pair_order():
    distance = np.zeros((9, 9), dtype=float)
    labels0 = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    weights = np.array(
        [
            float.fromhex("0x1.8f5bee3413debp+0"),
            float.fromhex("0x1.0dd56305c8bf0p+0"),
            float.fromhex("0x1.12617de95cd0dp+1"),
            float.fromhex("0x1.42765dcd377a9p-1"),
            float.fromhex("0x1.1301e008a6286p-2"),
            float.fromhex("0x1.3ae4804b3de05p+0"),
            float.fromhex("0x1.6806ec98548a7p+1"),
            float.fromhex("0x1.662cd23719628p-1"),
            float.fromhex("0x1.303e2794a8fecp-1"),
        ],
        dtype=float,
    )

    compiled = _wc_core.cluster_quality(distance, labels0, weights=weights)[0]
    pure = _fallback.cluster_quality(distance, labels0, weights=weights)[0]
    numba = _numba.cluster_quality_numba(distance, labels0, weights=weights)[0]

    assert compiled["HG"] == pure["HG"] == 1.0
    assert compiled["HGSD"] == pure["HGSD"]
    assert np.isclose(compiled["HGSD"], 2.5365531733104917e-16, rtol=1e-14, atol=0.0)
    assert numba["HG"] == compiled["HG"]
    assert numba["HGSD"] == compiled["HGSD"]


def test_clustrange_bootstrap_replay_uses_r_internal_asw_for_t0_and_replicates():
    points = np.array([0.0, 0.2, 1.0, 1.3, 2.4, 2.6], dtype=float)
    distance = np.abs(points[:, None] - points[None, :])
    clustering = pd.DataFrame(
        {
            "c2": [1, 1, 1, 2, 2, 2],
            "c3": [1, 1, 2, 2, 3, 3],
        }
    )
    replay = np.array(
        [
            [0, 3, 1, 1, 2, 5],
            [3, 0, 1, 3, 2, 2],
            [5, 1, 3, 1, 4, 1],
        ],
        dtype=int,
    )
    expected_asw = {
        2: np.array(
            [
                0.452839485076327,
                0.436740355161408,
                0.315073815073815,
                0.718118686868687,
            ]
        ),
        3: np.array(
            [
                0.789049378819362,
                0.835377322905927,
                0.806297143825748,
                0.944055944055944,
            ]
        ),
    }

    actual = as_clustrange(
        clustering,
        distance,
        R=3,
        samplesize=6,
        replay_samples=replay,
    )

    for k, expected in expected_asw.items():
        np.testing.assert_allclose(
            actual["boot"][k]["ASW"],
            expected,
            rtol=0.0,
            atol=1e-13,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual["stats"]["ASW"][k - 2],
            expected[0],
            rtol=0.0,
            atol=1e-13,
        )


def test_weighted_seqclararange_optimizes_and_reports_analysis_weight_criteria():
    rng = np.random.default_rng(7)
    sequences = rng.integers(0, 3, size=(40, 8))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = seqclararange(
            sequences,
            R=8,
            sample_size=22,
            kvals=[3],
            seqdist_args={"method": "HAM"},
            criteria="ch",
            random_state=3,
        )

    aggregated = wcAggregateCases(sequences)
    aggregate_sequences = sequences[np.asarray(aggregated["aggIndex"], dtype=int)]
    analysis_weights = np.asarray(aggregated["aggWeights"], dtype=float)
    distance = sequences_distance(aggregate_sequences, metric="HAM", normalize=False)
    info = actual["clara"][3]
    expected_objectives = []
    for iteration in info["iterations"]:
        labels1 = np.asarray(iteration["clustering"], dtype=int) + 1
        expected_objectives.append(
            float(wcClusterQuality(distance, labels1, weights=analysis_weights)["stats"]["CH"])
        )

    np.testing.assert_allclose(
        info["iter.objective"],
        expected_objectives,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    expected_best = int(np.argmax(expected_objectives)) + 1
    assert info["iteration"] == expected_best
    assert info["objective"] == expected_objectives[expected_best - 1]
    assert info["objective"] == actual["quality_stats"]["CH"][0]


def test_seqclararange_normalize_alias_consumes_the_canonical_configuration():
    sequences = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=int,
    )
    replay_samples = np.array([[0, 1, 2, 3, 4, 5]], dtype=int)
    common = {
        "R": 1,
        "sample_size": 6,
        "kvals": [2],
        "criteria": "distance",
        "random_state": 17,
        "replay_samples": replay_samples,
    }

    canonical = seqclararange(
        sequences,
        seqdist_args={"method": "HAM", "norm": "maxlength"},
        **common,
    )
    alias = seqclararange(
        sequences,
        seqdist_args={"method": "HAM", "normalize": True},
        **common,
    )

    assert alias["clara"][2]["objective"] == canonical["clara"][2]["objective"]
    assert list(alias["quality_stats"]) == list(canonical["quality_stats"])
    for name in alias["quality_stats"]:
        np.testing.assert_allclose(
            alias["quality_stats"][name],
            canonical["quality_stats"][name],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

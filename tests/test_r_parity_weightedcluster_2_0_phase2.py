from __future__ import annotations

import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from traminex import seqdef

from tests.r_parity_utils import canonical_partition
from weightedcluster import (
    as_clustrange,
    bootclustrange,
    clustassoc,
    clustrange_plot_data,
    clustrangefamily_plot_data,
    dtcut,
    dtlabels,
    plot_bootclustrange,
    plot_clustrange,
    plot_clustrangefamily,
    plot_proptree_labels,
    plot_proptree_panel,
    plot_proptree_tree,
    plot_seqclararange_stability,
    print_clustrange,
    print_clustrangefamily,
    print_kmedoids,
    proptree_plot_data,
    seqclararange,
    seqnull,
    seqnullcqi,
    seqpropclust,
    summary_clustrange,
    summary_clustrangefamily,
    wcClusterQuality,
    wcCmpCluster,
    wcKMedoids,
    wcKMedRange,
    wcPropertyClustering,
    wcSilhouetteObs,
)
from weightedcluster._seqdist import sequences_distance
from weightedcluster.cmpcluster import _linkage_hclust
from weightedcluster.plot import (
    _seqclararange_plot_data,
    _seqnullcqi_plot_data,
    plot_seqnullcqi,
    print_seqnullcqi,
)
from weightedcluster.schemas import ClustrangeResult, SeqClaraRangeResult
from weightedcluster.seqnull import _recluster_seqnull_distance

pytest_plugins = ["tests.r_parity_utils"]

R_QUALITY_NAMES = (
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

SEQNULL_X = np.array(
    [
        [1, 1, 2, 2],
        [1, 2, 2, 3],
        [2, 2, 3, 3],
        [3, 3, 1, 1],
    ],
    dtype=int,
)


def test_seqnull_signature_exposes_r_options_and_deterministic_replay_seam():
    parameters = inspect.signature(seqnull).parameters
    assert {
        "model",
        "imp_trans",
        "imp_trans_limit",
        "trate",
        "begin",
        "time_varying",
        "weighted",
        "random_state",
        "replay_plan",
    } <= set(parameters)


def test_seqnull_positional_replay_controls_every_generated_state():
    source = np.array([[0, 0, 1], [1, 1, 0]], dtype=int)
    before = source.copy()
    replay = [
        {"kind": "sample.int", "values": [1, 2]},
        {"kind": "sample.int", "values": [2]},
        {"kind": "sample.int", "values": [1]},
        {"kind": "sample.int", "values": [2]},
        {"kind": "sample.int", "values": [1]},
    ]
    actual = seqnull(
        source,
        model="stateindep",
        random_state=999,
        replay_plan=replay,
    )
    np.testing.assert_array_equal(actual, [[0, 1, 0], [1, 0, 1]])
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize(
    ("python_model", "r_model"),
    [
        ("combined", "combined"),
        ("duration", "duration"),
        ("sequencing", "sequencing"),
        ("stateindep", "stateindep"),
        ("markov", "Markov"),
        ("userpos", "userpos"),
    ],
)
def test_seqnull_all_six_models_replay_actual_sequences_from_live_r(
    r_parity_session, python_model, r_model
):
    expected = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model=r_model,
        seed=20260718,
    )
    assert expected["error"] is None
    assert expected["replay_plan"]
    actual = seqnull(
        SEQNULL_X,
        model=python_model,
        random_state=1,
        replay_plan=expected["replay_plan"],
    )
    np.testing.assert_array_equal(actual.astype(str), expected["sequences"])


def _assert_seqnull_replay_matches_live_r(expected, **kwargs):
    assert expected["error"] is None
    actual = seqnull(
        SEQNULL_X,
        model="userpos",
        random_state=314159,
        replay_plan=expected["replay_plan"],
        **kwargs,
    )
    np.testing.assert_array_equal(actual.astype(str), expected["sequences"])


def test_seqnull_dispatch_defaults_are_the_exact_hard_coded_live_r_models(
    r_parity_session,
):
    stateindep = r_parity_session.seqnull_complete_contract(
        SEQNULL_X, model="stateindep", weighted=False, seed=17
    )
    stateindep_explicit = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        trate="freq",
        begin="freq",
        time_varying=True,
        weighted=True,
        seed=17,
    )
    markov = r_parity_session.seqnull_complete_contract(
        SEQNULL_X, model="Markov", weighted=False, seed=19
    )
    markov_explicit = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        trate="trate",
        begin="freq",
        time_varying=False,
        weighted=True,
        seed=19,
    )
    np.testing.assert_array_equal(stateindep["sequences"], stateindep_explicit["sequences"])
    np.testing.assert_array_equal(markov["sequences"], markov_explicit["sequences"])
    assert [draw["kind"] for draw in stateindep["replay_plan"]] == [
        draw["kind"] for draw in stateindep_explicit["replay_plan"]
    ]
    assert [draw["kind"] for draw in markov["replay_plan"]] == [
        draw["kind"] for draw in markov_explicit["replay_plan"]
    ]

    actual_stateindep = seqnull(
        SEQNULL_X,
        model="stateindep",
        weighted=False,
        replay_plan=stateindep["replay_plan"],
    )
    actual_markov = seqnull(
        SEQNULL_X,
        model="markov",
        weighted=False,
        replay_plan=markov["replay_plan"],
    )
    np.testing.assert_array_equal(actual_stateindep.astype(str), stateindep["sequences"])
    np.testing.assert_array_equal(actual_markov.astype(str), markov["sequences"])


@pytest.mark.parametrize(
    ("trate", "begin", "time_varying"),
    [
        ("trate", "freq", True),
        ("trate", "ofreq", False),
        ("freq", "freq", True),
        ("freq", "ofreq", False),
        (
            np.array(
                [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]],
                dtype=float,
            ),
            np.array([0.2, 0.3, 0.5], dtype=float),
            False,
        ),
        (
            np.repeat(
                np.array(
                    [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]],
                    dtype=float,
                )[:, :, None],
                SEQNULL_X.shape[1] - 1,
                axis=2,
            ),
            np.array([0.5, 0.3, 0.2], dtype=float),
            True,
        ),
    ],
    ids=[
        "trate-time-varying-freq-begin",
        "trate-global-ofreq-begin",
        "freq-time-varying-freq-begin",
        "freq-global-ofreq-begin",
        "explicit-global",
        "explicit-time-varying",
    ],
)
def test_seqnull_userpos_probability_and_begin_options_replay_live_r(
    r_parity_session, trate, begin, time_varying
):
    expected = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        trate=trate,
        begin=begin,
        time_varying=time_varying,
        seed=23,
    )
    _assert_seqnull_replay_matches_live_r(
        expected,
        trate=trate,
        begin=begin,
        time_varying=time_varying,
    )


def _sts_strings(value) -> np.ndarray:
    return np.asarray(value.alphabet, dtype=object)[np.asarray(value.data, dtype=int)]


def test_seqnull_attached_weights_and_sequence_metadata_match_live_r(
    r_parity_session,
):
    weights = np.array([20.0, 1.0, 1.0, 1.0])
    metadata = {
        "alphabet": [1, 2, 3],
        "states": ["A", "B", "C"],
        "labels": ["Alpha", "Beta", "Gamma"],
    }
    source = seqdef(SEQNULL_X, weights=weights, **metadata)
    expected_weighted = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        trate="freq",
        begin="ofreq",
        time_varying=False,
        weighted=True,
        weights=weights,
        seed=29,
        **metadata,
    )
    expected_unweighted = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        trate="freq",
        begin="ofreq",
        time_varying=False,
        weighted=False,
        weights=weights,
        seed=29,
        **metadata,
    )
    assert not np.allclose(
        expected_weighted["replay_plan"][0]["prob"],
        expected_unweighted["replay_plan"][0]["prob"],
    )

    for expected, use_weights in (
        (expected_weighted, True),
        (expected_unweighted, False),
    ):
        actual = seqnull(
            source,
            model="userpos",
            trate="freq",
            begin="ofreq",
            time_varying=False,
            weighted=use_weights,
            replay_plan=expected["replay_plan"],
        )
        np.testing.assert_array_equal(_sts_strings(actual), expected["sequences"])
        assert actual.alphabet == expected["alphabet"] == ["A", "B", "C"]
        assert actual.labels == expected["labels"] == ["Alpha", "Beta", "Gamma"]
        assert actual.weights is expected["weights"] is None
        np.testing.assert_array_equal(
            np.full(SEQNULL_X.shape[0], SEQNULL_X.shape[1]),
            expected["lengths"],
        )

    expected_spell = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="combined",
        weights=weights,
        seed=31,
        **metadata,
    )
    actual_spell = seqnull(
        source,
        model="combined",
        replay_plan=expected_spell["replay_plan"],
    )
    np.testing.assert_array_equal(_sts_strings(actual_spell), expected_spell["sequences"])
    np.testing.assert_array_equal(actual_spell.weights, expected_spell["weights"])


@pytest.mark.parametrize(
    "options",
    [
        {"imp_trans": {1: [2]}},
        {"imp_trans_limit": 0.0},
    ],
    ids=["explicit-impossible-transition", "inclusive-zero-rate-limit"],
)
def test_seqnull_combined_transition_restrictions_replay_live_r(r_parity_session, options):
    expected = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="combined",
        seed=37,
        **options,
    )
    assert expected["error"] is None
    actual = seqnull(
        SEQNULL_X,
        model="combined",
        replay_plan=expected["replay_plan"],
        **options,
    )
    np.testing.assert_array_equal(actual.astype(str), expected["sequences"])


def test_seqnull_zero_probability_transition_boundaries_are_stable(
    r_parity_session,
):
    combined_error = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="combined",
        imp_trans_limit=1.0,
        seed=37,
    )
    assert "probability" in combined_error["error"]
    with pytest.raises(ValueError, match="positive probabilities"):
        seqnull(
            SEQNULL_X,
            model="combined",
            imp_trans_limit=1.0,
            random_state=37,
        )

    zero_row = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    positional_error = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        trate=zero_row,
        begin=np.array([0.0, 1.0, 0.0]),
        time_varying=False,
        seed=41,
    )
    assert "positive probabilities" in positional_error["error"]
    with pytest.raises(ValueError, match="positive probabilities"):
        seqnull(
            SEQNULL_X,
            model="userpos",
            trate=zero_row,
            begin=np.array([0.0, 1.0, 0.0]),
            time_varying=False,
            random_state=41,
        )


def test_seqnull_malformed_and_short_input_boundaries_are_live_r_pinned(
    r_parity_session,
):
    one_column = r_parity_session.seqnull_complete_contract(
        np.array([[1], [2]], dtype=int),
        model="combined",
        seed=43,
    )
    assert "subscript out of bounds" in one_column["error"]
    with pytest.raises(ValueError, match="at least two"):
        seqnull(np.array([[1], [2]], dtype=int), model="combined")

    wrong_begin = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        begin=np.array([1.0, 0.0]),
    )
    assert "length 3" in wrong_begin["error"]
    with pytest.raises(ValueError, match="length 3"):
        seqnull(
            SEQNULL_X,
            model="userpos",
            begin=np.array([1.0, 0.0]),
        )

    wrong_matrix = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="userpos",
        trate=np.ones((2, 2), dtype=float),
    )
    assert "3x3" in wrong_matrix["error"]
    with pytest.raises(ValueError, match="shape"):
        seqnull(
            SEQNULL_X,
            model="userpos",
            trate=np.ones((2, 2), dtype=float),
        )

    unknown = r_parity_session.seqnull_complete_contract(
        SEQNULL_X,
        model="unknown",
    )
    assert "Unknow seqnull model" in unknown["error"]
    with pytest.raises(ValueError, match="unknown model"):
        seqnull(SEQNULL_X, model="unknown")


def test_seqnullcqi_signature_exposes_deterministic_downstream_replay_seams():
    parameters = inspect.signature(seqnullcqi).parameters
    assert {"replay_null_sequences", "replay_kmedoid_starts"} <= set(parameters)


def _seqnullcqi_replay_fixture():
    rng = np.random.default_rng(20260718)
    X = rng.integers(0, 4, size=(18, 6))
    nulls = rng.integers(0, 4, size=(4, 18, 6))
    clusters = {
        "cluster2": np.repeat([1, 2], 9),
        "cluster3": np.repeat([1, 2, 3], 6),
    }
    return X, clusters, nulls


def _seqnullcqi_costs():
    rng = np.random.default_rng(91)
    costs = rng.uniform(0.6, 1.7, size=(4, 4))
    costs = (costs + costs.T) / 2.0
    np.fill_diagonal(costs, 0.0)
    return costs


def _seqnullcqi_clustrange_from_contract(expected):
    return as_clustrange(
        {f"cluster{k}": expected["clustrange_clustering"][int(k)] for k in expected["kvals"]},
        diss=expected["semantic"]["observed_distance"],
        R=1,
    )


def test_seqnullcqi_replays_exact_ward_d_partitions_cqis_and_result(r_parity_session):
    X, clusters, nulls = _seqnullcqi_replay_fixture()
    source_before = X.copy()
    clusters_before = {name: values.copy() for name, values in clusters.items()}
    nulls_before = nulls.copy()
    costs = _seqnullcqi_costs()
    expected = r_parity_session.seqnullcqi_complete_contract(
        X,
        clusters,
        nulls,
        model="combined",
        seqdist_method="OM",
        seqdist_norm="none",
        seqdist_indel=0.83,
        seqdist_sm=costs,
        hclust_method="single",
        trate="trate",
        begin="freq",
        time_varying=False,
        weighted=False,
        seed=13,
    )
    assert expected["error"] is None
    clustrange = _seqnullcqi_clustrange_from_contract(expected)
    actual = seqnullcqi(
        X,
        clustrange,
        R=4,
        model="combined",
        seqdist_args={
            "method": "OM",
            "norm": "none",
            "sm": costs,
            "indel": 0.83,
        },
        hclust_method="single",
        trate="trate",
        begin="freq",
        time_varying=False,
        weighted=False,
        replay_null_sequences=nulls,
        random_state=999,
    )

    assert actual.r_class == expected["class"]
    assert list(actual) == expected["names"]
    assert actual["R"] == expected["R"]
    assert actual["kmedoid"] is expected["kmedoid"]
    assert actual["hclust.method"] == expected["hclust_method"] == "single"
    assert list(actual["seqdist.args"]) == expected["seqdist_names"]
    assert actual["seqdist.args"]["method"] == expected["seqdist_method"]
    assert actual["seqdist.args"]["norm"] == expected["seqdist_norm"]
    assert actual["seqdist.args"]["indel"] == expected["seqdist_indel"]
    np.testing.assert_array_equal(actual["seqdist.args"]["sm"], expected["seqdist_sm"])
    assert list(actual["nullmodel"]) == expected["nullmodel_names"]
    assert actual["nullmodel"] == {
        "model": "combined",
        "trate": "trate",
        "begin": "freq",
        "time.varying": False,
        "weighted": False,
    }
    np.testing.assert_array_equal(
        np.asarray(actual["seqdata"], dtype=int),
        np.asarray(expected["sequences"], dtype=int),
    )
    assert list(actual["stats"]) == expected["stats_names"] == list(R_QUALITY_NAMES)
    for stat in R_QUALITY_NAMES:
        np.testing.assert_allclose(
            actual["stats"][stat],
            expected["stats"][stat],
            rtol=0.0,
            atol=1e-10,
        )
    assert actual["clustrange"].r_class == ["clustrange", "list"]
    np.testing.assert_array_equal(actual["clustrange"]["kvals"], expected["kvals"])
    for column, stat in enumerate(expected["clustrange_stats_names"]):
        np.testing.assert_allclose(
            actual["clustrange"]["stats"][stat],
            expected["clustrange_stats"][:, column],
            rtol=0.0,
            atol=1e-10,
        )
    for kval in expected["kvals"]:
        assert canonical_partition(
            actual["clustrange"]["clustering"][f"cluster{kval}"]
        ) == canonical_partition(expected["clustrange_clustering"][int(kval)])
    for actual_iteration, expected_iteration in zip(
        actual.additions["reclusterings"], expected["partitions"], strict=True
    ):
        for kval in expected["kvals"]:
            assert canonical_partition(actual_iteration[int(kval)]) == canonical_partition(
                expected_iteration[int(kval)]
            )
    np.testing.assert_array_equal(X, source_before)
    np.testing.assert_array_equal(nulls, nulls_before)
    for name, values in clusters.items():
        np.testing.assert_array_equal(values, clusters_before[name])


def test_seqnullcqi_requires_typed_complete_consecutive_clustrange():
    X, clusters, nulls = _seqnullcqi_replay_fixture()
    with pytest.raises(TypeError, match="required keyword-only argument: 'R'"):
        seqnullcqi(X, ClustrangeResult({}))
    with pytest.raises(TypeError, match="clustrange object"):
        seqnullcqi(X, {"kvals": [2, 3], "clustering": clusters}, R=2)
    typed = as_clustrange(
        clusters,
        diss=np.zeros((X.shape[0], X.shape[0]), dtype=float),
        R=1,
    )
    malformed = type(typed)(
        {
            **typed,
            "kvals": [2, 4],
            "clustering": {
                "cluster2": clusters["cluster2"],
                "cluster4": np.repeat([1, 2, 3], 6),
            },
        }
    )
    with pytest.raises(ValueError, match="consecutive"):
        seqnullcqi(X, malformed, R=2, replay_null_sequences=nulls[:2])
    with pytest.raises(ValueError, match="R must be at least one"):
        seqnullcqi(X, typed, R=0)


def test_seqnullcqi_kmedoid_replay_matches_exact_partitions_and_cqis(
    r_parity_session,
):
    X, clusters, nulls = _seqnullcqi_replay_fixture()
    costs = _seqnullcqi_costs()
    expected = r_parity_session.seqnullcqi_complete_contract(
        X,
        clusters,
        nulls,
        kmedoid=True,
        seqdist_method="OM",
        seqdist_norm="none",
        seqdist_indel=0.83,
        seqdist_sm=costs,
        seed=13,
    )
    assert expected["error"] is None
    actual = seqnullcqi(
        X,
        _seqnullcqi_clustrange_from_contract(expected),
        R=4,
        kmedoid=True,
        seqdist_args={
            "method": "OM",
            "norm": "none",
            "sm": costs,
            "indel": 0.83,
        },
        replay_null_sequences=nulls,
        replay_kmedoid_starts=expected["kmedoid_starts_0b"],
        random_state=999,
    )
    assert actual["kmedoid"] is True
    for stat in R_QUALITY_NAMES:
        np.testing.assert_allclose(
            actual["stats"][stat],
            expected["stats"][stat],
            rtol=0.0,
            atol=1e-10,
        )
    for actual_iteration, expected_iteration in zip(
        actual.additions["reclusterings"], expected["partitions"], strict=True
    ):
        for kval in expected["kvals"]:
            assert canonical_partition(actual_iteration[int(kval)]) == canonical_partition(
                expected_iteration[int(kval)]
            )


@pytest.mark.parametrize(
    "model",
    ["combined", "duration", "sequencing", "stateindep", "Markov", "userpos"],
)
def test_seqnullcqi_all_models_and_forwarded_seqnull_options_match_live_r(
    r_parity_session,
    model,
):
    X, clusters, nulls = _seqnullcqi_replay_fixture()
    costs = _seqnullcqi_costs()
    expected = r_parity_session.seqnullcqi_complete_contract(
        X,
        clusters,
        nulls[:2],
        model=model,
        seqdist_method="OM",
        seqdist_norm="none",
        seqdist_indel=0.83,
        seqdist_sm=costs,
        trate="trate",
        begin="freq",
        time_varying=False,
        weighted=False,
        seed=31,
    )
    actual = seqnullcqi(
        X,
        _seqnullcqi_clustrange_from_contract(expected),
        R=2,
        model=model,
        seqdist_args={
            "method": "OM",
            "norm": "none",
            "sm": costs,
            "indel": 0.83,
        },
        trate="trate",
        begin="freq",
        time_varying=False,
        weighted=False,
        replay_null_sequences=nulls[:2],
        random_state=999,
    )
    assert expected["nullmodel_model"] == actual["nullmodel"]["model"] == model
    assert list(actual["nullmodel"]) == expected["nullmodel_names"]
    assert actual["nullmodel"]["time.varying"] is expected["nullmodel_time_varying"]
    assert actual["nullmodel"]["weighted"] is expected["nullmodel_weighted"]
    np.testing.assert_array_equal(
        np.asarray(actual["seqdata"], dtype=int),
        np.asarray(expected["sequences"], dtype=int),
    )
    for stat in R_QUALITY_NAMES:
        np.testing.assert_allclose(
            actual["stats"][stat],
            expected["stats"][stat],
            rtol=0.0,
            atol=1e-10,
        )


def test_seqnullcqi_forwards_seqdist_arguments_through_public_traminex(monkeypatch):
    import importlib

    from traminex import seqdist as public_seqdist

    seqnull_module = importlib.import_module("weightedcluster.seqnull")
    X, clusters, nulls = _seqnullcqi_replay_fixture()
    distance = public_seqdist(X, method="LCS", norm="maxlength")
    clustrange = as_clustrange(clusters, diss=distance, R=1)
    calls = []

    def spy(seqdata, *, method, **kwargs):
        calls.append((np.asarray(seqdata).copy(), method, dict(kwargs)))
        return public_seqdist(seqdata, method=method, **kwargs)

    monkeypatch.setattr(seqnull_module, "traminex_seqdist", spy)
    result = seqnullcqi(
        X,
        clustrange,
        R=2,
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        replay_null_sequences=nulls[:2],
        random_state=50,
    )
    assert len(calls) == 3
    assert [method for _data, method, _kwargs in calls] == ["LCS"] * 3
    assert [kwargs for _data, _method, kwargs in calls] == [{"norm": "maxlength"}] * 3
    np.testing.assert_array_equal(calls[0][0], X)
    np.testing.assert_array_equal(calls[1][0], nulls[0])
    np.testing.assert_array_equal(calls[2][0], nulls[1])
    assert result["seqdist.args"] == {"method": "LCS", "norm": "maxlength"}


def test_seqnullcqi_serial_parallel_progress_and_seed_are_invariant():
    from traminex import seqdist as public_seqdist

    X, clusters, nulls = _seqnullcqi_replay_fixture()
    distance = public_seqdist(X, method="LCS", norm="maxlength")
    clustrange = as_clustrange(clusters, diss=distance, R=1)
    before = {
        "X": X.copy(),
        "nulls": nulls.copy(),
        **{name: values.copy() for name, values in clusters.items()},
    }
    kwargs = {
        "R": 4,
        "seqdist_args": {"method": "LCS", "norm": "maxlength"},
        "replay_null_sequences": nulls,
    }
    serial = seqnullcqi(
        X,
        clustrange,
        parallel=False,
        progressbar=False,
        random_state=1,
        **kwargs,
    )
    parallel = seqnullcqi(
        X,
        clustrange,
        parallel=True,
        n_jobs=2,
        progressbar=True,
        random_state=999,
        **kwargs,
    )
    for stat in R_QUALITY_NAMES:
        np.testing.assert_array_equal(serial["stats"][stat], parallel["stats"][stat])
    np.testing.assert_array_equal(serial["seqdata"], parallel["seqdata"])
    for serial_iteration, parallel_iteration in zip(
        serial.additions["reclusterings"],
        parallel.additions["reclusterings"],
        strict=True,
    ):
        for kval in clustrange["kvals"]:
            assert canonical_partition(serial_iteration[kval]) == canonical_partition(
                parallel_iteration[kval]
            )
    np.testing.assert_array_equal(X, before["X"])
    np.testing.assert_array_equal(nulls, before["nulls"])
    for name, values in clusters.items():
        np.testing.assert_array_equal(values, before[name])


def test_seqnullcqi_print_and_plot_semantics_match_live_r(r_parity_session):
    X, clusters, nulls = _seqnullcqi_replay_fixture()
    costs = _seqnullcqi_costs()
    expected = r_parity_session.seqnullcqi_complete_contract(
        X,
        clusters,
        nulls,
        seqdist_method="OM",
        seqdist_norm="none",
        seqdist_indel=0.83,
        seqdist_sm=costs,
        plot_stat="ASW",
        print_norm=True,
        print_quant=0.95,
        print_digits=2,
        seed=13,
    )
    actual = seqnullcqi(
        X,
        _seqnullcqi_clustrange_from_contract(expected),
        R=4,
        seqdist_args={
            "method": "OM",
            "norm": "none",
            "sm": costs,
            "indel": 0.83,
        },
        replay_null_sequences=nulls,
        random_state=13,
    )

    printed = print_seqnullcqi(actual, norm=True, quant=0.95, digits=2)
    assert printed.visible is False
    assert printed.headers[0] == expected["print"]["output"][0]
    assert printed.headers[2:] == [
        "Number of bootstraps: 4",
        "Clustering method: hclust with ward.D",
        "Seqdist arguments: method=OM, norm=none, sm=<4x4>, indel=0.83",
    ]
    assert list(printed.index) == expected["print"]["row_names"]
    assert list(printed.columns) == expected["print"]["column_names"]
    np.testing.assert_array_equal(printed.to_numpy(), expected["print"]["table"])

    line = _seqnullcqi_plot_data(actual, stat="ASW", type="line", quant=0.95, norm=True)
    np.testing.assert_allclose(line["raw_observed"], expected["semantic"]["orig_raw"])
    np.testing.assert_allclose(line["raw_null"], expected["semantic"]["null_raw"])
    np.testing.assert_allclose(
        line["standardized_observed"], expected["semantic"]["orig_normalized"]
    )
    np.testing.assert_allclose(line["standardized_null"], expected["semantic"]["null_normalized"])
    np.testing.assert_allclose(line["raw_quantile_band"], expected["semantic"]["per_k_quantiles"])
    assert line["raw_reference"] == pytest.approx(expected["semantic"]["plot_threshold"])
    assert line["panel_count"] == 2
    assert line["labels"] == ["Raw ASW", "Standardized ASW"]

    density = _seqnullcqi_plot_data(actual, stat="ASW", type="density", quant=0.95, norm=True)
    np.testing.assert_allclose(
        density["max_t"], np.max(expected["semantic"]["null_normalized"], axis=1)
    )
    assert density["panel_count"] == 1
    box = _seqnullcqi_plot_data(actual, stat="ASW", type="boxplot")
    assert box["panel_count"] == 2
    unknown = _seqnullcqi_plot_data(actual, stat="ASW", type="unknown")
    assert unknown["type"] == "line"
    assert expected["plot_errors"]["unknown"] is None
    all_stats = _seqnullcqi_plot_data(actual, stat="all", type="line")
    for stat in R_QUALITY_NAMES:
        np.testing.assert_allclose(
            all_stats["all_standardized"][stat],
            expected["semantic"]["all_standardized"][stat],
        )
    assert all_stats["panel_count"] == 1
    seqdplot = _seqnullcqi_plot_data(actual, stat="ASW", type="seqdplot")
    assert seqdplot["panel_count"] == 1
    assert expected["plot_errors"]["invalid_stat"] == "undefined columns selected"
    with pytest.raises(ValueError, match="undefined columns selected"):
        _seqnullcqi_plot_data(actual, stat="not_a_stat", type="line")

    import matplotlib.pyplot as plt

    for plot_type, panel_count in (
        ("line", 2),
        ("density", 1),
        ("boxplot", 2),
        ("seqdplot", 1),
        ("unknown", 2),
    ):
        figure, _axes = plot_seqnullcqi(actual, stat="ASW", type=plot_type, quant=0.95, norm=True)
        assert len(figure.axes) == panel_count
        plt.close(figure)
    figure, _axes = plot_seqnullcqi(actual, stat="all", type="line")
    assert len(figure.axes) == 1
    plt.close(figure)


BOOT_X = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 1],
        [2, 2, 1, 1],
        [2, 1, 1, 0],
    ],
    dtype=int,
)
BOOT_CLUSTER2 = np.array([1] * 8 + [2] * 4, dtype=int)
BOOT_CLUSTER3 = np.array([1] * 4 + [2] * 4 + [3] * 4, dtype=int)
BOOT_SAMPLES_1B = np.array(
    [
        [1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        [1, 2, 4, 5, 6, 7, 9, 10, 11, 12],
        [1, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    ],
    dtype=int,
)


def _boot_plain_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster2": BOOT_CLUSTER2.copy(),
            "cluster3": BOOT_CLUSTER3.copy(),
        }
    )


def _boot_column_matrix(table: dict, names: tuple[str, ...] = R_QUALITY_NAMES) -> np.ndarray:
    return np.column_stack([np.asarray(table[name], dtype=float) for name in names])


def _assert_boot_execution_results_equal(serial, parallel):
    for serial_sample, parallel_sample in zip(
        serial.additions["samples"], parallel.additions["samples"], strict=True
    ):
        np.testing.assert_array_equal(serial_sample, parallel_sample)
    for serial_distance, parallel_distance in zip(
        serial.additions["distances"], parallel.additions["distances"], strict=True
    ):
        np.testing.assert_allclose(
            serial_distance,
            parallel_distance,
            rtol=0.0,
            atol=0.0,
        )
    assert serial["kvals"] == parallel["kvals"]
    for kval in serial["kvals"]:
        np.testing.assert_allclose(
            _boot_column_matrix(serial["boot"][kval]),
            _boot_column_matrix(parallel["boot"][kval]),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    np.testing.assert_allclose(
        _boot_column_matrix(serial["meant"]),
        _boot_column_matrix(parallel["meant"]),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_bootclustrange_signature_exposes_r_options_and_exact_replay_seam():
    parameters = inspect.signature(bootclustrange).parameters
    assert {
        "object",
        "seqdata",
        "seqdist_args",
        "R",
        "sample_size",
        "parallel",
        "progressbar",
        "sampling",
        "strata",
        "random_state",
        "replay_samples",
    } <= set(parameters)
    plot_parameters = inspect.signature(plot_bootclustrange).parameters
    assert {
        "stat",
        "legendpos",
        "norm",
        "withlegend",
        "lwd",
        "col",
        "ylab",
        "xlab",
        "conf_int",
        "ci_method",
        "ci_alpha",
        "line",
    } <= set(plot_parameters)


def test_bootclustrange_plain_frame_replays_exact_numeric_contract(
    r_parity_session,
):
    frame = _boot_plain_frame()
    frame_before = frame.copy(deep=True)
    sequence_before = BOOT_X.copy()
    expected = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {
            "cluster2": BOOT_CLUSTER2,
            "cluster3": BOOT_CLUSTER3,
        },
        BOOT_SAMPLES_1B,
        sampling="random",
        seqdist_method="LCS",
        seqdist_norm="maxlength",
        seed=91,
    )
    assert expected["error"] is None

    actual = bootclustrange(
        frame,
        BOOT_X,
        R=3,
        sample_size=10,
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        sampling="random",
        random_state=999,
        replay_samples=BOOT_SAMPLES_1B - 1,
    )

    assert list(actual) == expected["names"]
    assert actual.r_class == expected["class"]
    assert actual["kvals"] == expected["kvals"].tolist() == [2, 3]
    assert actual["stats"] is actual["meant"]
    assert list(actual["meant"]) == expected["meant_names"] == list(R_QUALITY_NAMES)
    assert actual["meant"].row_names == expected["meant_rows"]
    np.testing.assert_array_equal(
        np.column_stack([actual["clustering"][name] for name in actual["clustering"].column_names]),
        expected["clustering"],
    )
    np.testing.assert_allclose(
        _boot_column_matrix(actual["meant"]),
        expected["meant"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        _boot_column_matrix(actual["stderr"]),
        expected["stderr"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    for index, kval in enumerate(actual["kvals"]):
        assert list(actual["boot"][kval]) == expected["boot_names"][index]
        np.testing.assert_allclose(
            _boot_column_matrix(actual["boot"][kval]),
            expected["boot"][index],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    np.testing.assert_array_equal(
        np.vstack(actual.additions["samples"]) + 1,
        BOOT_SAMPLES_1B,
    )
    pd.testing.assert_frame_equal(frame, frame_before)
    np.testing.assert_array_equal(BOOT_X, sequence_before)


def test_bootclustrange_seqclararange_extracts_medoids_and_permits_source_duplicates(
    r_parity_session,
):
    medoids_0b = np.array([0, 4, 8], dtype=int)
    samples_1b = np.array(
        [
            [1, 5, 9, 1, 2, 3, 6, 7, 10, 11],
            [1, 5, 9, 2, 4, 5, 6, 8, 10, 12],
        ],
        dtype=int,
    )
    seqclara = SeqClaraRangeResult(
        {
            "kvals": [2, 3],
            "clustering": {
                "cluster2": BOOT_CLUSTER2.copy(),
                "cluster3": BOOT_CLUSTER3.copy(),
            },
            "stats": {},
            "clara": {
                2: {"medoids": [0, 8]},
                3: {"medoids": medoids_0b.tolist()},
            },
        }
    )
    expected = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {
            "cluster2": BOOT_CLUSTER2,
            "cluster3": BOOT_CLUSTER3,
        },
        samples_1b,
        sampling="medoids",
        medoids_1b=medoids_0b + 1,
        object_kind="seqclararange",
        seqdist_method="LCS",
        seqdist_norm="maxlength",
        seed=93,
    )
    assert expected["error"] is None

    actual = bootclustrange(
        seqclara,
        BOOT_X,
        R=2,
        sample_size=10,
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        sampling="medoids",
        replay_samples=samples_1b - 1,
    )
    assert actual.additions["sampling"] == "medoids"
    assert actual.additions["samples"][0].tolist().count(0) == 2
    np.testing.assert_allclose(
        _boot_column_matrix(actual["meant"]),
        expected["meant"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("sampling", "strata", "samples_1b", "expected_mode"),
    [
        (
            "clustering",
            None,
            np.array(
                [
                    [1, 2, 3, 5, 6, 7, 9, 10, 11],
                    [2, 3, 4, 6, 7, 8, 10, 11, 12],
                ],
                dtype=int,
            ),
            "strata",
        ),
        (
            "strata",
            np.array([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3),
            np.array(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
                    [1, 2, 3, 4, 5, 6, 8, 9, 11, 12],
                ],
                dtype=int,
            ),
            "strata",
        ),
        (
            "ordinary-source-fallback",
            None,
            BOOT_SAMPLES_1B[:2],
            "ordinary-source-fallback",
        ),
    ],
    ids=["default-clustering-to-strata", "explicit-strata-rounding", "unknown-is-ordinary"],
)
def test_bootclustrange_sampling_routes_replay_exact_source_subsamples(
    r_parity_session,
    sampling,
    strata,
    samples_1b,
    expected_mode,
):
    expected = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {
            "cluster2": BOOT_CLUSTER2,
            "cluster3": BOOT_CLUSTER3,
        },
        samples_1b,
        sampling=sampling,
        strata=strata,
        seqdist_method="LCS",
        seqdist_norm="maxlength",
        seed=97,
    )
    assert expected["error"] is None
    actual = bootclustrange(
        _boot_plain_frame(),
        BOOT_X,
        R=int(samples_1b.shape[0]),
        sample_size=int(samples_1b.shape[1]),
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        sampling=sampling,
        strata=strata,
        replay_samples=samples_1b - 1,
    )
    assert actual.additions["sampling"] == expected_mode
    for iteration, expected_call in enumerate(expected["seqdist_calls"]):
        np.testing.assert_array_equal(
            BOOT_X[np.asarray(actual.additions["samples"][iteration])].astype(str),
            expected_call["seqdata"],
        )
        np.testing.assert_allclose(
            actual.additions["distances"][iteration],
            expected_call["distance"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            actual.additions["subset_clusterings"][iteration],
            expected["as_clustrange_calls"][iteration]["clustering"],
        )


def test_bootclustrange_forwards_distance_arguments_through_public_traminex(
    monkeypatch,
):
    import weightedcluster.boot as bootmod

    calls = []
    original = bootmod.traminex_seqdist

    def spy(seqdata, *args, **kwargs):
        calls.append((np.asarray(seqdata).copy(), dict(kwargs)))
        return original(seqdata, *args, **kwargs)

    monkeypatch.setattr(bootmod, "traminex_seqdist", spy)
    substitution = np.array(
        [[0.0, 1.2, 1.7], [1.2, 0.0, 0.9], [1.7, 0.9, 0.0]],
        dtype=float,
    )
    bootclustrange(
        _boot_plain_frame(),
        BOOT_X,
        R=1,
        sample_size=10,
        seqdist_args={
            "method": "OM",
            "norm": "maxlength",
            "indel": 0.7,
            "sm": substitution,
        },
        sampling="random",
        replay_samples=BOOT_SAMPLES_1B[:1] - 1,
    )
    assert len(calls) == 2
    for _sequences, options in calls:
        assert options["method"] == "OM"
        assert options["norm"] == "maxlength"
        assert options["indel"] == 0.7
        np.testing.assert_array_equal(options["subm"], substitution)


def test_bootclustrange_attached_sequence_weights_are_preserved_and_source_inert(
    r_parity_session,
):
    weights = np.arange(1, BOOT_X.shape[0] + 1, dtype=float)
    source = seqdef(BOOT_X, weights=weights)
    source_data = np.asarray(source.data).copy()
    source_weights = np.asarray(source.weights).copy()
    expected = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {"cluster2": BOOT_CLUSTER2, "cluster3": BOOT_CLUSTER3},
        BOOT_SAMPLES_1B,
        sampling="random",
        seqdist_method="LCS",
        seqdist_norm="maxlength",
        weights=weights,
    )
    assert expected["error"] is None
    actual = bootclustrange(
        _boot_plain_frame(),
        source,
        R=3,
        sample_size=10,
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        sampling="random",
        replay_samples=BOOT_SAMPLES_1B - 1,
    )
    np.testing.assert_allclose(
        _boot_column_matrix(actual["meant"]),
        expected["meant"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual.additions["weights"], weights)
    np.testing.assert_array_equal(source.data, source_data)
    np.testing.assert_array_equal(source.weights, source_weights)


def test_bootclustrange_boundaries_and_field_validation_are_source_pinned(
    r_parity_session,
):
    too_small = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {"cluster2": BOOT_CLUSTER2, "cluster3": BOOT_CLUSTER3},
        np.array([[1, 5, 9]], dtype=int),
        sampling="clustering",
    )
    assert "sample size is to small" in too_small["error"]
    with pytest.raises(ValueError, match="sample size is to small"):
        bootclustrange(
            _boot_plain_frame(),
            BOOT_X,
            R=1,
            sample_size=3,
            sampling="clustering",
            replay_samples=np.array([[0, 4, 8]], dtype=int),
        )

    too_large = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {"cluster2": BOOT_CLUSTER2, "cluster3": BOOT_CLUSTER3},
        np.array([[*range(1, 13), 1]], dtype=int),
        sampling="random",
    )
    assert "cannot take a sample larger than the population" in too_large["error"]
    with pytest.raises(ValueError, match="cannot take a sample larger"):
        bootclustrange(
            _boot_plain_frame(),
            BOOT_X,
            R=1,
            sample_size=13,
            sampling="random",
            replay_samples=np.arange(13, dtype=int)[None, :],
        )
    with pytest.raises(ValueError, match="duplicate"):
        bootclustrange(
            _boot_plain_frame(),
            BOOT_X,
            R=1,
            sample_size=10,
            sampling="random",
            replay_samples=np.array([[0, 0, 1, 2, 3, 4, 5, 8, 9, 10]], dtype=int),
        )
    with pytest.raises(ValueError, match="clustering"):
        bootclustrange(
            {"not_clustering": []},
            BOOT_X,
            R=1,
            sample_size=4,
            replay_samples=np.array([[0, 1, 2, 3]], dtype=int),
        )


@pytest.mark.parametrize(
    ("kwargs", "name"), [({"R": True}, "R"), ({"sample_size": True}, "sample_size")]
)
def test_bootclustrange_rejects_boolean_integer_controls(kwargs, name):
    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        bootclustrange(_boot_plain_frame(), BOOT_X, sampling="random", **kwargs)


def test_bootclustrange_serial_parallel_progress_seed_and_inputs_are_invariant():
    kwargs = {
        "R": 3,
        "sample_size": 10,
        "seqdist_args": {"method": "LCS", "norm": "maxlength"},
        "sampling": "random",
        "replay_samples": BOOT_SAMPLES_1B - 1,
    }
    source = BOOT_X.copy()
    frame = _boot_plain_frame()
    frame_before = frame.copy(deep=True)
    serial = bootclustrange(
        frame,
        source,
        random_state=101,
        parallel=False,
        progressbar=False,
        **kwargs,
    )
    parallel = bootclustrange(
        frame,
        source,
        random_state=202,
        parallel=True,
        n_jobs=2,
        progressbar=True,
        **kwargs,
    )
    _assert_boot_execution_results_equal(serial, parallel)
    pd.testing.assert_frame_equal(frame, frame_before)
    np.testing.assert_array_equal(source, BOOT_X)


def test_bootclustrange_nonreplay_same_seed_matches_serial_and_process_results():
    kwargs = {
        "R": 2,
        "sample_size": 9,
        "seqdist_args": {"method": "HAM"},
        "sampling": "random",
        "random_state": 303,
    }
    serial = bootclustrange(
        _boot_plain_frame(),
        BOOT_X,
        parallel=False,
        **kwargs,
    )
    parallel = bootclustrange(
        _boot_plain_frame(),
        BOOT_X,
        parallel=True,
        n_jobs=2,
        **kwargs,
    )

    _assert_boot_execution_results_equal(serial, parallel)


@pytest.mark.parametrize(
    ("stat", "norm", "ci_method", "line"),
    [
        ("noCH", "none", "perc", "median"),
        (("ASW", "PBC"), "range", "norm", "mean"),
        ("all", "none", "none", "t0"),
    ],
)
def test_bootclustrange_print_and_plot_semantics_match_live_r(
    r_parity_session,
    stat,
    norm,
    ci_method,
    line,
):
    from weightedcluster.plot import print_bootclustrange

    expected = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {"cluster2": BOOT_CLUSTER2, "cluster3": BOOT_CLUSTER3},
        BOOT_SAMPLES_1B,
        sampling="random",
        seqdist_method="LCS",
        seqdist_norm="maxlength",
        print_digits=4,
        print_bootstat="mean",
        plot_stat=stat,
        plot_norm=norm,
        plot_ci_method=ci_method,
        plot_line=line,
        seed=103,
    )
    assert expected["error"] is None
    assert expected["plot"]["error"] is None
    actual = bootclustrange(
        _boot_plain_frame(),
        BOOT_X,
        R=3,
        sample_size=10,
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        sampling="random",
        replay_samples=BOOT_SAMPLES_1B - 1,
    )

    printed = print_bootclustrange(actual, digits=4)
    assert printed.visible is False
    assert printed.index.tolist() == expected["print"]["row_names"]
    assert printed.columns.tolist() == expected["print"]["column_names"]
    np.testing.assert_allclose(
        printed.to_numpy(dtype=float),
        expected["print"]["table"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    stderr_printed = print_bootclustrange(
        actual,
        digits=4,
        bootstat="anything-else-selects-stderr",
    )
    np.testing.assert_allclose(
        stderr_printed.to_numpy(dtype=float),
        np.round(_boot_column_matrix(actual["stderr"]), 4),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    semantic = clustrange_plot_data(
        actual,
        stat=stat,
        norm=norm,
        conf_int=0.95,
        ci_method=ci_method,
        ci_alpha=0.3,
        line=line,
    )
    assert semantic["stat_names"] == expected["plot"]["stat_names"]
    assert semantic["norm"] == expected["plot"]["norm"]
    assert semantic["legend_labels"] == expected["plot"]["labels"]
    np.testing.assert_allclose(
        semantic["values"],
        expected["plot"]["values"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    for key in ("lower", "upper"):
        if expected["plot"][key] is None:
            assert semantic[key] is None
        else:
            np.testing.assert_allclose(
                semantic[key],
                expected["plot"][key],
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    figure, axis = plot_bootclustrange(
        actual,
        stat=stat,
        norm=norm,
        conf_int=0.95,
        ci_method=ci_method,
        line=line,
    )
    assert figure is axis.figure
    assert axis.get_xlabel() == expected["plot"]["xlab"]
    assert axis.get_ylabel() == expected["plot"]["ylab"]
    assert len(axis.lines) == len(expected["plot"]["stat_names"])
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_bootclustrange_single_stat_plot_preserves_installed_drop_error(
    r_parity_session,
):
    expected = r_parity_session.bootclustrange_complete_contract(
        BOOT_X,
        {"cluster2": BOOT_CLUSTER2, "cluster3": BOOT_CLUSTER3},
        BOOT_SAMPLES_1B,
        sampling="random",
        seqdist_method="LCS",
        seqdist_norm="maxlength",
        plot_stat="ASW",
    )
    assert expected["error"] is None
    assert expected["plot"]["error"] == "dim(X) must have a positive length"
    actual = bootclustrange(
        _boot_plain_frame(),
        BOOT_X,
        R=3,
        sample_size=10,
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        sampling="random",
        replay_samples=BOOT_SAMPLES_1B - 1,
    )
    with pytest.raises(ValueError, match=r"dim\(X\) must have a positive length"):
        plot_bootclustrange(actual, stat="ASW")


def _quality_distance() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.5, 1.0, 4.0, 4.5],
            [0.5, 0.0, 0.7, 3.7, 4.2],
            [1.0, 0.7, 0.0, 3.2, 3.7],
            [4.0, 3.7, 3.2, 0.0, 0.6],
            [4.5, 4.2, 3.7, 0.6, 0.0],
        ],
        dtype=float,
    )


def _kmedoids_distance() -> np.ndarray:
    return np.array(
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


def _cmpcluster_distance() -> np.ndarray:
    points = np.array(
        [
            [0.0, 0.0],
            [0.31, 0.08],
            [1.05, 0.23],
            [3.02, 3.11],
            [3.27, 2.86],
            [3.83, 3.57],
            [7.01, 0.19],
            [7.43, 0.41],
        ],
        dtype=float,
    )
    return squareform(
        np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))[
            np.triu_indices(points.shape[0], k=1)
        ]
    )


def _inverting_hclust_distance() -> np.ndarray:
    rng = np.random.default_rng(44_712)
    points = rng.uniform(-3.0, 3.0, size=(13, 6))
    return np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))


def _assert_quality_result_matches_r(actual: dict, expected: dict) -> None:
    assert expected["error"] is None
    assert expected["names"] == ["stats", "ASW"]
    assert expected["stats_names"] == list(R_QUALITY_NAMES)
    assert list(actual["stats"])[: len(R_QUALITY_NAMES)] == list(R_QUALITY_NAMES)
    np.testing.assert_allclose(
        [actual["stats"][name] for name in R_QUALITY_NAMES],
        expected["stats"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    actual_levels = list(actual["ASW"])
    assert [str(level) for level in actual_levels] == expected["asw_row_names"]
    assert list(actual["ASWw"]) == actual_levels
    np.testing.assert_allclose(
        np.column_stack(
            [
                [actual["ASW"][name] for name in actual_levels],
                [actual["ASWw"][name] for name in actual_levels],
            ]
        ),
        expected["asw"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def _assert_kmedoids_result_matches_r(actual: dict, expected: dict) -> None:
    assert expected["error"] is None
    assert expected["class"] == ["kmedoids", "list"]
    assert expected["names"] == ["clustering", "info", "stats", "ASW"]
    assert list(actual) == ["clustering", "info", "stats", "ASW", "ASWw"]
    np.testing.assert_array_equal(actual["clustering"], expected["clustering"])
    assert list(actual["info"]) == expected["info_names"]
    np.testing.assert_allclose(
        list(actual["info"].values()),
        expected["info"],
        rtol=0.0,
        atol=1e-12,
    )
    assert list(actual["stats"])[: len(R_QUALITY_NAMES)] == list(R_QUALITY_NAMES)
    assert expected["stats_names"] == list(R_QUALITY_NAMES)
    np.testing.assert_allclose(
        [actual["stats"][name] for name in R_QUALITY_NAMES],
        expected["stats"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    actual_levels = list(actual["ASW"])
    assert [str(level) for level in actual_levels] == expected["asw_row_names"]
    assert list(actual["ASWw"]) == actual_levels
    assert expected["asw_column_names"] == ["ASW", "ASWw"]
    np.testing.assert_allclose(
        np.column_stack(
            [
                [actual["ASW"][level] for level in actual_levels],
                [actual["ASWw"][level] for level in actual_levels],
            ]
        ),
        expected["asw"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def _assert_family_method_matches_r(actual: dict, expected: dict) -> None:
    assert list(actual) == expected["names"]
    assert actual.r_class == expected["class"]
    assert actual["kvals"] == expected["kvals"]
    np.testing.assert_array_equal(
        np.column_stack([actual["clustering"][name] for name in actual["clustering"].column_names]),
        expected["clustering"],
    )
    np.testing.assert_allclose(
        np.column_stack([actual["stats"][name] for name in R_QUALITY_NAMES]),
        expected["stats"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_cluster_quality_condensed_distance_matches_live_r_complete_contract(
    r_parity_session,
):
    square = _quality_distance()
    condensed = squareform(square)
    labels = np.array([10, 10, 10, 30, 30], dtype=int)
    weights = np.array([1.0, 2.0, 0.5, 1.5, 1.0], dtype=float)

    expected = r_parity_session.wc_cluster_quality_complete_contract(
        square,
        labels,
        weights=weights,
        distance_kind="dist",
    )
    actual = wcClusterQuality(condensed, labels, weights=weights)

    assert expected["error"] is None
    assert expected["warnings"] == []
    assert expected["inputs_unchanged"] is True
    _assert_quality_result_matches_r(actual, expected)


def test_cluster_quality_string_labels_preserve_r_factor_level_order(
    r_parity_session,
):
    square = _quality_distance()
    labels = np.array(["zeta", "zeta", "zeta", "alpha", "alpha"], dtype=object)

    expected = r_parity_session.wc_cluster_quality_complete_contract(
        square,
        labels,
        distance_kind="matrix",
    )
    actual = wcClusterQuality(square, labels)

    assert expected["error"] is None
    assert expected["asw_row_names"] == ["alpha", "zeta"]
    assert list(actual["ASW"]) == expected["asw_row_names"]
    assert list(actual["ASWw"]) == expected["asw_row_names"]
    _assert_quality_result_matches_r(actual, expected)


def test_cluster_quality_categorical_level_order_and_singletons_match_live_r(
    r_parity_session,
):
    square = _quality_distance()
    labels = pd.Categorical(
        ["zeta", "zeta", "middle", "alpha", "alpha"],
        categories=["zeta", "unused", "middle", "alpha"],
        ordered=True,
    )

    expected = r_parity_session.wc_cluster_quality_complete_contract(
        square,
        labels,
        distance_kind="dist",
        factor_levels=labels.categories,
    )
    actual = wcClusterQuality(square, labels)

    assert expected["asw_row_names"] == ["zeta", "middle", "alpha"]
    assert list(actual["ASW"]) == expected["asw_row_names"]
    _assert_quality_result_matches_r(actual, expected)


@pytest.mark.parametrize(
    "weights",
    [
        None,
        np.ones(5, dtype=float),
        np.array([1.0, 2.0, 0.5, 1.5, 1.0], dtype=float),
        np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 1.0, 1.0, 1.0, -0.5], dtype=float),
        np.array([1.0, np.inf, 1.0, 1.0, 1.0], dtype=float),
        np.array([1.0, -np.inf, 1.0, 1.0, 1.0], dtype=float),
        np.zeros(5, dtype=float),
    ],
    ids=[
        "absent",
        "unit",
        "nonuniform",
        "zero_observation",
        "negative_edge",
        "positive_infinity",
        "negative_infinity",
        "all_zero",
    ],
)
def test_cluster_quality_weight_boundaries_match_live_r(r_parity_session, weights):
    square = _quality_distance()
    labels = np.array([1, 1, 1, 2, 2], dtype=int)
    expected = r_parity_session.wc_cluster_quality_complete_contract(
        square,
        labels,
        weights=weights,
        distance_kind="matrix",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = wcClusterQuality(square, labels, weights=weights)

    assert bool(caught) == bool(expected["warnings"])
    if caught:
        assert "less than one observation" in str(caught[0].message)
        assert "less than one observation" in expected["warnings"][0]
    _assert_quality_result_matches_r(actual, expected)


def test_cluster_quality_zero_total_cluster_weight_matches_live_r(r_parity_session):
    square = np.array(
        [
            [0.0, 1.0, 3.0, 4.0],
            [1.0, 0.0, 2.0, 3.0],
            [3.0, 2.0, 0.0, 1.0],
            [4.0, 3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([1, 1, 2, 2], dtype=int)
    weights = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    expected = r_parity_session.wc_cluster_quality_complete_contract(
        square,
        labels,
        weights=weights,
        distance_kind="matrix",
    )

    with pytest.warns(UserWarning, match="less than one observation"):
        actual = wcClusterQuality(square, labels, weights=weights)

    assert expected["error"] is None
    assert len(expected["warnings"]) == 1
    assert "less than one observation" in expected["warnings"][0]
    _assert_quality_result_matches_r(actual, expected)


@pytest.mark.parametrize(
    ("distance", "labels", "weights", "distance_kind"),
    [
        (_quality_distance(), [1, 1, 1, 1, 1], None, "matrix"),
        (_quality_distance(), [1, 1, 2, 2], None, "matrix"),
        (_quality_distance(), [1, 1, 1, 2, 2], np.ones(4), "dist"),
        (np.ones((5, 4), dtype=float), [1, 1, 1, 2, 2], None, "matrix"),
        (
            _quality_distance(),
            [1, 1, 1, 2, 2],
            np.array([1.0, 1.0, np.nan, 1.0, 1.0]),
            "dist",
        ),
    ],
    ids=["one_group", "label_length", "weight_length", "nonsquare", "missing_weight"],
)
def test_cluster_quality_validation_boundaries_follow_live_r(
    r_parity_session,
    distance,
    labels,
    weights,
    distance_kind,
):
    expected = r_parity_session.wc_cluster_quality_complete_contract(
        distance,
        labels,
        weights=weights,
        distance_kind=distance_kind,
    )

    assert expected["error"] is not None
    with pytest.raises(ValueError):
        wcClusterQuality(distance, labels, weights=weights)


def test_silhouette_vector_measure_and_low_weight_warning_match_live_r(
    r_parity_session,
):
    square = _quality_distance()
    labels = np.array(["left", "left", "left", "right", "right"], dtype=object)
    weights = np.array([1.0, 1.0, 1.0, 0.2, 0.2], dtype=float)

    expected = r_parity_session.wc_silhouette_obs_complete_contract(
        square,
        labels,
        weights=weights,
        measure=["ASW", "ASWw"],
        distance_kind="matrix",
    )
    with pytest.warns(UserWarning, match="Only 'ASWw' is computed"):
        actual = wcSilhouetteObs(
            square,
            labels,
            weights=weights,
            measure=["ASW", "ASWw"],
        )

    assert expected["error"] is None
    assert len(expected["warnings"]) == 1
    assert "Only 'ASWw' is computed" in expected["warnings"][0]
    assert expected["shape"] == [square.shape[0]]
    np.testing.assert_allclose(
        actual,
        expected["values"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf], ids=["nan", "infinity"])
def test_cluster_quality_nonfinite_distance_propagation_matches_live_r(
    r_parity_session,
    nonfinite,
):
    square = _quality_distance()
    square[0, 1] = square[1, 0] = nonfinite
    labels = np.array([1, 1, 2, 2, 2], dtype=int)
    before = square.copy()

    expected = r_parity_session.wc_cluster_quality_complete_contract(
        square,
        labels,
        distance_kind="matrix",
    )
    actual = wcClusterQuality(square, labels)

    assert expected["error"] is None
    assert np.array_equal(square, before, equal_nan=True)
    _assert_quality_result_matches_r(actual, expected)


@pytest.mark.parametrize("measure", ["ASW", "ASWw", ["ASW", "ASWw"]])
def test_silhouette_square_condensed_and_measure_forms_match_live_r(
    r_parity_session,
    measure,
):
    square = _quality_distance()
    condensed = squareform(square)
    labels = np.array([10, 10, 10, 30, 30], dtype=int)
    weights = np.array([1.0, 2.0, 0.5, 1.5, 1.0], dtype=float)
    square_before = square.copy()
    condensed_before = condensed.copy()
    labels_before = labels.copy()
    weights_before = weights.copy()

    expected = r_parity_session.wc_silhouette_obs_complete_contract(
        square,
        labels,
        weights=weights,
        measure=measure,
        distance_kind="dist",
    )
    actual = wcSilhouetteObs(condensed, labels, weights=weights, measure=measure)

    assert expected["error"] is None
    assert expected["warnings"] == []
    assert expected["inputs_unchanged"] is True
    expected_values = expected["values"]
    if expected["shape"] == [square.shape[0], 2]:
        expected_values = expected_values.reshape((square.shape[0], 2), order="F")
        assert expected["column_names"] == ["ASW", "ASWw"]
    np.testing.assert_allclose(
        actual,
        expected_values,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(square, square_before)
    np.testing.assert_array_equal(condensed, condensed_before)
    np.testing.assert_array_equal(labels, labels_before)
    np.testing.assert_array_equal(weights, weights_before)


def test_silhouette_stop_warning_and_invalid_measure_boundaries_match_live_r(
    r_parity_session,
):
    square = _quality_distance()
    labels = np.array([1, 1, 1, 2, 2], dtype=int)
    low_weights = np.array([1.0, 1.0, 1.0, 0.2, 0.2], dtype=float)

    stopped = r_parity_session.wc_silhouette_obs_complete_contract(
        square,
        labels,
        weights=low_weights,
        measure="ASW",
    )
    invalid = r_parity_session.wc_silhouette_obs_complete_contract(
        square,
        labels,
        measure="unknown",
    )

    assert "Use measure='ASWw'" in stopped["error"]
    with pytest.raises(ValueError, match="Use measure='ASWw'"):
        wcSilhouetteObs(square, labels, weights=low_weights, measure="ASW")
    assert "Unknow silhouette measure" in invalid["error"]
    with pytest.raises(ValueError, match="measure must be"):
        wcSilhouetteObs(square, labels, measure="unknown")


def test_seqnullcqi_hierarchical_reclustering_preserves_tied_height_cuts_from_live_r(
    r_parity_session,
):
    distance = np.ones((4, 4), dtype=float)
    np.fill_diagonal(distance, 0.0)
    expected_hierarchy = r_parity_session.fastcluster_hclust_contract(
        distance,
        method="ward.D",
        maxcluster=3,
    )
    partitions: dict[int, np.ndarray] = {}

    actual = _recluster_seqnull_distance(
        distance,
        [2, 3],
        kmedoid=False,
        hclust_method="ward.D",
        random_state=7,
        partitions_out=partitions,
    )

    for column, k in enumerate((2, 3)):
        expected_labels = expected_hierarchy["cuts"][:, column].astype(int)
        assert canonical_partition(partitions[k]) == canonical_partition(expected_labels)
        expected_quality = r_parity_session.wc_cluster_quality(
            distance,
            expected_labels,
        )["stats"]
        assert set(actual[k]) == set(R_QUALITY_NAMES)
        np.testing.assert_allclose(
            [actual[k][name] for name in R_QUALITY_NAMES],
            [expected_quality[name] for name in R_QUALITY_NAMES],
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("digits", [4, 7])
def test_kmedoids_fixed_start_condensed_contract_matches_live_r(
    r_parity_session,
    digits,
):
    square = _kmedoids_distance()
    condensed = squareform(square)
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    initial_medoids_1b = np.array([1, 5], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        weights=weights,
        npass=0,
        initialclust=initial_medoids_1b,
        method="PAMonce",
        distance_kind="dist",
        digits=digits,
    )

    actual = wcKMedoids(
        condensed,
        2,
        weights=weights,
        npass=0,
        initialclust=initial_medoids_1b,
        method="PAMonce",
        random_state=7,
    )

    assert expected["error"] is None
    assert expected["inputs_unchanged"] is True
    _assert_kmedoids_result_matches_r(actual, expected)
    expected_tokens = [line.split() for line in expected["print_lines"] if line]
    actual_tokens = [
        line.split() for line in print_kmedoids(actual, digits=digits).splitlines() if line
    ]
    assert actual_tokens == expected_tokens


@pytest.mark.parametrize(
    "method",
    ["PAMonce", "pamonce", "PAMO", "K", "kmed", 1, 3],
    ids=[
        "default_name",
        "case_insensitive",
        "partial_pamonce",
        "partial_kmedoids",
        "longer_partial_kmedoids",
        "numeric_kmedoids",
        "numeric_pamonce",
    ],
)
def test_kmedoids_method_names_codes_and_partial_matching_follow_live_r(
    r_parity_session,
    method,
):
    square = _kmedoids_distance()
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    initial = np.array([1, 5], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        weights=weights,
        npass=0,
        initialclust=initial,
        method=method,
        distance_kind="matrix",
    )
    actual = wcKMedoids(
        square,
        2,
        weights=weights,
        npass=0,
        initialclust=initial,
        method=method,
        random_state=11,
    )

    _assert_kmedoids_result_matches_r(actual, expected)


@pytest.mark.parametrize(
    "method", ["P", "unknown", 0, 4], ids=["ambiguous", "name", "zero", "four"]
)
def test_kmedoids_unknown_methods_match_live_r_errors(r_parity_session, method):
    square = _kmedoids_distance()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        initialclust=[1, 5],
        npass=0,
        method=method,
    )

    assert "unknow clustering method" in expected["error"]
    with pytest.raises(ValueError, match="unknown clustering method"):
        wcKMedoids(square, 2, initialclust=[1, 5], npass=0, method=method)


@pytest.mark.parametrize("method", ["PAM", 2], ids=["name", "numeric"])
def test_kmedoids_installed_r_condensed_pam_branch_remains_not_implemented(
    r_parity_session,
    method,
):
    square = _kmedoids_distance()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        initialclust=[1, 5],
        npass=0,
        method=method,
    )

    assert "not (yet) implemented" in expected["error"].lower()
    with pytest.raises(NotImplementedError, match="not \\(yet\\) implemented"):
        wcKMedoids(squareform(square), 2, initialclust=[1, 5], npass=0, method=method)


@pytest.mark.parametrize("distance_kind", ["matrix", "dist"])
def test_kmedoids_default_pamonce_build_matches_live_r(
    r_parity_session,
    distance_kind,
):
    square = _kmedoids_distance()
    python_distance = square if distance_kind == "matrix" else squareform(square)
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    distance_before = python_distance.copy()
    weights_before = weights.copy()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        weights=weights,
        npass=1,
        method="PAMonce",
        distance_kind=distance_kind,
        seed=19,
    )
    actual = wcKMedoids(
        python_distance,
        2,
        weights=weights,
        random_state=19,
    )

    _assert_kmedoids_result_matches_r(actual, expected)
    np.testing.assert_array_equal(python_distance, distance_before)
    np.testing.assert_array_equal(weights, weights_before)


@pytest.mark.parametrize(
    ("npass", "replay_starts_1b"),
    [
        (0, np.array([[1, 5]], dtype=int)),
        (3, np.array([[1, 5], [1, 4], [2, 6]], dtype=int)),
    ],
    ids=["single_random_start", "build_plus_two_replayed_starts"],
)
def test_kmedoids_replayed_multipass_selection_matches_live_r(
    r_parity_session,
    npass,
    replay_starts_1b,
):
    square = _kmedoids_distance()
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        weights=weights,
        npass=npass,
        method="PAMonce",
        replay_starts_1b=replay_starts_1b,
        distance_kind="matrix",
    )
    replay_starts_0b = replay_starts_1b - 1
    np.testing.assert_array_equal(replay_starts_0b, replay_starts_1b - 1)
    actual = wcKMedoids(
        square,
        2,
        weights=weights,
        npass=npass,
        method="PAMonce",
        replay_starts=replay_starts_0b,
        random_state=999,
    )

    assert expected["replay_consumed"] == replay_starts_1b.shape[0]
    _assert_kmedoids_result_matches_r(actual, expected)


def test_kmedoids_random_state_repeats_python_multipass_selection():
    square = _kmedoids_distance()
    kwargs = {
        "weights": np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float),
        "npass": 5,
        "random_state": 23,
    }

    first = wcKMedoids(square, 2, **kwargs)
    second = wcKMedoids(square, 2, **kwargs)

    np.testing.assert_array_equal(first["clustering"], second["clustering"])
    assert first["info"] == second["info"]
    for name in R_QUALITY_NAMES:
        np.testing.assert_allclose(
            first["stats"][name],
            second["stats"][name],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("initial_kind", "initialclust"),
    [
        ("membership", np.array([9, 9, 9, 2, 2, 2], dtype=int)),
        (
            "factor",
            pd.Categorical(
                ["zeta", "zeta", "zeta", "alpha", "alpha", "alpha"],
                categories=["zeta", "unused", "alpha"],
                ordered=True,
            ),
        ),
    ],
    ids=["arbitrary_numeric_membership", "factor_membership"],
)
def test_kmedoids_membership_initializations_match_live_r(
    r_parity_session,
    initial_kind,
    initialclust,
):
    square = _kmedoids_distance()
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    before = np.asarray(initialclust, dtype=object).copy()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        weights=weights,
        npass=7,
        initialclust=initialclust,
        initial_kind=initial_kind,
        method="PAMonce",
    )
    actual = wcKMedoids(
        square,
        2,
        weights=weights,
        npass=7,
        initialclust=initialclust,
        method="PAMonce",
    )

    _assert_kmedoids_result_matches_r(actual, expected)
    np.testing.assert_array_equal(np.asarray(initialclust, dtype=object), before)
    assert actual["info"]["Method"] == 0.0


@pytest.mark.parametrize(
    ("initial_kind", "linkage_method"),
    [("hclust", "single"), ("twins", "average")],
)
def test_kmedoids_hierarchy_initializations_match_live_r(
    r_parity_session,
    initial_kind,
    linkage_method,
):
    square = _kmedoids_distance()
    python_linkage = linkage(squareform(square), method=linkage_method)
    linkage_before = python_linkage.copy()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        initial_kind=initial_kind,
        method="PAMonce",
    )
    actual = wcKMedoids(
        square,
        2,
        initialclust=python_linkage,
        method="PAMonce",
    )

    _assert_kmedoids_result_matches_r(actual, expected)
    np.testing.assert_array_equal(python_linkage, linkage_before)


def test_kmedoids_first_medoid_ties_and_index_base_match_live_r(r_parity_session):
    square = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 3.0, 4.0],
            [4.0, 3.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    membership = np.array([20, 20, 10, 10], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        initialclust=membership,
        initial_kind="membership",
        npass=0,
    )
    actual = wcKMedoids(square, 2, initialclust=membership, npass=0)

    _assert_kmedoids_result_matches_r(actual, expected)
    np.testing.assert_array_equal(np.unique(actual["clustering"]), [1, 3])
    np.testing.assert_array_equal(
        np.unique(actual["clustering"]) - 1,
        np.unique(expected["clustering"]) - 1,
    )

    with pytest.raises(ValueError, match="starting medoids should be in 1:4"):
        wcKMedoids(square, 2, initialclust=[0, 2], npass=0)
    zero_based = wcKMedoids(square, 2, npass=0, replay_starts=[[0, 2]])
    np.testing.assert_array_equal(zero_based["clustering"], expected["clustering"])


def test_kmedoids_discriminating_best_swap_and_zero_guard_match_live_r(r_parity_session):
    square = np.array(
        [
            [0, 2.5, 0.5, 1, 1.5, 0.5, 1, 0, 0.5],
            [2.5, 0, 1, 1.5, 2, 0, 1, 1.5, 1],
            [0.5, 1, 0, 0, 1, 1.5, 1, 1, 0],
            [1, 1.5, 0, 0, 1, 2, 2, 1.5, 0.5],
            [1.5, 2, 1, 1, 0, 0.5, 0, 3, 3],
            [0.5, 0, 1.5, 2, 0.5, 0, 0.5, 1, 1],
            [1, 1, 1, 2, 0, 0.5, 0, 1.5, 2.5],
            [0, 1.5, 1, 1.5, 3, 1, 1.5, 0, 0.5],
            [0.5, 1, 0, 0.5, 3, 1, 2.5, 0.5, 0],
        ],
        dtype=float,
    )
    weights = np.array([2, 1, 3, 0, 1, 4, 2, 1, 4], dtype=float)
    start = np.array([1, 6], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        weights=weights,
        npass=0,
        initialclust=start,
        method="PAMonce",
    )
    actual = wcKMedoids(
        square,
        2,
        weights=weights,
        npass=0,
        initialclust=start,
        method="PAMonce",
    )
    _assert_kmedoids_result_matches_r(actual, expected)
    np.testing.assert_array_equal(actual["clustering"], [9, 6, 9, 9, 6, 6, 6, 9, 9])


@pytest.mark.parametrize("metric", ["HAM", "LCS", "OM"])
@pytest.mark.parametrize(
    "weights",
    [
        np.ones(8, dtype=float),
        np.array([1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 0.75, 1.25]),
        np.array([1.0, 2.0, 0.0, 3.0, 1.5, 2.5, 0.0, 1.25]),
    ],
    ids=["uniform", "positive_nonuniform", "zero_containing"],
)
def test_kmedoids_social_sequence_distances_match_live_r_from_identical_starts(
    r_parity_session,
    metric,
    weights,
):
    trajectories = np.array(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 2],
            [0, 1, 1, 1, 2, 2],
            [2, 2, 2, 1, 1, 1],
            [2, 2, 1, 1, 0, 0],
            [2, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [2, 2, 2, 1, 1, 1],
        ],
        dtype=int,
    )
    distance = sequences_distance(trajectories, metric=metric, normalize=False)
    start = np.array([2, 5], dtype=int)
    expected = r_parity_session.wc_kmedoids_complete_contract(
        distance,
        k=2,
        weights=weights,
        npass=0,
        initialclust=start,
        method="PAMonce",
        distance_kind="matrix",
    )
    actual = wcKMedoids(
        distance,
        2,
        weights=weights,
        npass=0,
        initialclust=start,
        method="PAMonce",
    )

    assert expected["error"] is None
    assert canonical_partition(actual["clustering"]) == canonical_partition(expected["clustering"])
    expected_info = dict(zip(expected["info_names"], expected["info"], strict=True))
    assert actual["info"]["Sum of Distances"] == pytest.approx(
        expected_info["Sum of Distances"], abs=1e-12
    )


@pytest.mark.parametrize(
    "weights",
    [
        None,
        np.ones(6, dtype=float),
        np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, -0.5], dtype=float),
        np.zeros(6, dtype=float),
    ],
    ids=["absent", "unit", "nonuniform", "zero_observation", "negative_edge", "all_zero"],
)
def test_kmedoids_weight_boundaries_match_live_r(r_parity_session, weights):
    square = _kmedoids_distance()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        weights=weights,
        initialclust=[1, 5],
        npass=0,
        method="PAMonce",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = wcKMedoids(
            square,
            2,
            weights=weights,
            initialclust=[1, 5],
            npass=0,
            method="PAMonce",
        )

    assert bool(caught) == bool(expected["warnings"])
    _assert_kmedoids_result_matches_r(actual, expected)


@pytest.mark.parametrize(
    ("initialclust", "initial_kind"),
    [
        ([1, 7], "medoids"),
        ([1, 3, 5], "medoids"),
        ([1, 1, 1, 1, 1, 1], "membership"),
        (["left", "right"], "medoids"),
    ],
    ids=[
        "out_of_range",
        "wrong_medoid_length",
        "wrong_membership_groups",
        "non_numeric_medoids",
    ],
)
def test_kmedoids_invalid_initializations_follow_live_r_error_boundaries(
    r_parity_session,
    initialclust,
    initial_kind,
):
    square = _kmedoids_distance()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        initialclust=initialclust,
        initial_kind=initial_kind,
        npass=0,
    )

    assert expected["error"] is not None
    with pytest.raises(ValueError):
        wcKMedoids(square, 2, initialclust=initialclust, npass=0)


@pytest.mark.parametrize(
    ("distance", "k", "weights", "npass", "distance_kind"),
    [
        (_kmedoids_distance(), 1, None, 1, "dist"),
        (_kmedoids_distance(), 7, None, 1, "dist"),
        (np.ones((6, 5), dtype=float), 2, None, 1, "matrix"),
        (_kmedoids_distance(), 2, np.ones(5), 1, "dist"),
        (_kmedoids_distance(), 2, None, -1, "dist"),
    ],
    ids=["k_too_small", "k_too_large", "nonsquare", "weight_length", "negative_npass"],
)
def test_kmedoids_validation_boundaries_follow_live_r(
    r_parity_session,
    distance,
    k,
    weights,
    npass,
    distance_kind,
):
    expected = r_parity_session.wc_kmedoids_complete_contract(
        distance,
        k=k,
        weights=weights,
        npass=npass,
        distance_kind=distance_kind,
    )

    assert expected["error"] is not None
    with pytest.raises(ValueError):
        wcKMedoids(distance, k, weights=weights, npass=npass)


def test_kmedoids_replay_plan_rejects_duplicate_and_wrong_base_rows():
    square = _kmedoids_distance()

    # WeightedCluster 2.0 does not validate duplicate explicit medoids before
    # entering native code and can fail to terminate. The Python translation
    # rejects that unsafe documented-invalid boundary synchronously.
    with pytest.raises(ValueError, match="unique"):
        wcKMedoids(square, 2, initialclust=[1, 1], npass=0)
    with pytest.raises(ValueError, match="unique medoids"):
        wcKMedoids(square, 2, npass=2, replay_starts=[[0, 4], [1, 1]])
    with pytest.raises(ValueError, match="zero-based"):
        wcKMedoids(square, 2, npass=2, replay_starts=[[1, 5], [2, 6]])
    with pytest.raises(ValueError, match="shape"):
        wcKMedoids(square, 2, npass=2, replay_starts=[0, 4])


def test_kmedoids_cluster_only_debuglevel_and_return_type_match_live_r(
    r_parity_session,
    capsys,
):
    square = _kmedoids_distance()
    expected = r_parity_session.wc_kmedoids_complete_contract(
        square,
        k=2,
        initialclust=[1, 5],
        npass=0,
        method="PAMonce",
        cluster_only=True,
        debuglevel=2,
    )
    actual = wcKMedoids(
        square,
        2,
        initialclust=[1, 5],
        npass=0,
        method="PAMonce",
        cluster_only=True,
        debuglevel=2,
    )

    assert expected["error"] is None
    assert expected["class"] == ["numeric"]
    assert expected["names"] == []
    assert expected["debug_lines"] == []
    np.testing.assert_array_equal(actual, expected["clustering"])
    assert isinstance(actual, np.ndarray)
    assert np.issubdtype(actual.dtype, np.integer)
    assert capsys.readouterr().out == ""


def test_kmedoids_signature_exposes_r_options_and_python_replay_seams():
    signature = inspect.signature(wcKMedoids)

    assert list(signature.parameters) == [
        "diss",
        "k",
        "weights",
        "npass",
        "initialclust",
        "method",
        "cluster_only",
        "debuglevel",
        "random_state",
        "replay_starts",
    ]
    assert signature.parameters["method"].default == "PAMonce"
    assert signature.parameters["cluster_only"].default is False
    assert signature.parameters["debuglevel"].default == 0
    assert signature.parameters["random_state"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["replay_starts"].kind is inspect.Parameter.KEYWORD_ONLY


def test_kmedoids_rejects_malformed_condensed_distance():
    with pytest.raises(ValueError, match="square matrix or condensed"):
        wcKMedoids(np.array([1.0, 2.0], dtype=float), 2)


def test_kmedrange_ordered_nonconsecutive_dist_and_forwarded_hierarchy_match_live_r(
    r_parity_session,
):
    square = _kmedoids_distance()
    condensed = squareform(square)
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    hierarchy = linkage(condensed, method="single")
    condensed_before = condensed.copy()
    weights_before = weights.copy()
    hierarchy_before = hierarchy.copy()
    expected = r_parity_session.wc_kmedrange_complete_contract(
        square,
        kvals=[3, 2],
        weights=weights,
        distance_kind="dist",
        initial_kind="hclust",
    )

    actual = wcKMedRange(
        condensed,
        [3, 2],
        weights=weights,
        initialclust=hierarchy,
        method="PAMonce",
        npass=7,
    )

    assert expected["error"] is None
    assert expected["inputs_unchanged"] is True
    assert expected["names"] == ["R", "kvals", "clustering", "stats"]
    assert actual["R"] == expected["R"] == 1
    assert actual["kvals"] == expected["kvals"] == [3, 2]
    assert actual.r_class == expected["class"] == ["clustrange", "list"]
    assert list(actual) == expected["names"]
    assert actual["clustering"].column_names == expected["clustering_column_names"]
    assert actual["stats"].column_names == expected["stats_column_names"]
    assert actual["stats"].row_names == expected["stats_row_names"]
    np.testing.assert_array_equal(
        np.column_stack([actual["clustering"][name] for name in actual["clustering"].column_names]),
        expected["clustering_numeric"],
    )
    np.testing.assert_allclose(
        np.column_stack([actual["stats"][name] for name in R_QUALITY_NAMES]),
        expected["stats"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    assert np.min(expected["clustering_numeric"]) >= 1
    np.testing.assert_array_equal(
        np.column_stack([actual["clustering"][name] for name in actual["clustering"].column_names])
        - 1,
        expected["clustering_numeric"] - 1,
    )
    np.testing.assert_allclose(
        summary_clustrange(actual).to_numpy(dtype=float),
        expected["summary"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(condensed, condensed_before)
    np.testing.assert_array_equal(weights, weights_before)
    np.testing.assert_array_equal(hierarchy, hierarchy_before)


@pytest.mark.parametrize("distance_kind", ["matrix", "dist"])
@pytest.mark.parametrize("sample_size", [None, 8], ids=["default", "explicit"])
def test_kmedrange_replayed_bootstrap_counts_match_live_r(
    r_parity_session,
    distance_kind,
    sample_size,
):
    globals_before = set(
        str(name) for name in r_parity_session.r("ls(envir=.GlobalEnv, all.names=TRUE)")
    )
    square = _kmedoids_distance()
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    width = int(np.floor(np.sum(weights))) if sample_size is None else sample_size
    samples_1b = np.asarray(
        [
            ([1, 2, 3, 4, 5, 6] + [1] * width)[:width],
            ([1, 2, 3, 4, 5, 6] + [6] * width)[:width],
        ],
        dtype=int,
    )
    expected = r_parity_session.wc_kmedrange_complete_contract(
        square,
        kvals=[3, 2],
        weights=weights,
        R=2,
        samplesize=sample_size,
        distance_kind=distance_kind,
        initial_kind="hclust",
        replay_samples_1b=samples_1b,
    )
    python_distance = square if distance_kind == "matrix" else squareform(square)
    hierarchy = linkage(squareform(square), method="single")
    actual = wcKMedRange(
        python_distance,
        [3, 2],
        weights=weights,
        R=2,
        samplesize=sample_size,
        initialclust=hierarchy,
        replay_samples=samples_1b - 1,
    )

    assert expected["error"] is None
    assert (
        list(actual)
        == expected["names"]
        == [
            "R",
            "kvals",
            "clustering",
            "stats",
            "boot",
        ]
    )
    assert "meant" not in actual and "stderr" not in actual
    for index, k in enumerate(actual["kvals"]):
        np.testing.assert_allclose(
            np.column_stack([actual["boot"][k][name] for name in R_QUALITY_NAMES]),
            expected["boot"][index],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    assert (
        set(str(name) for name in r_parity_session.r("ls(envir=.GlobalEnv, all.names=TRUE)"))
        == globals_before
    )


def test_kmedrange_duplicate_and_invalid_k_boundaries_follow_live_r(r_parity_session):
    square = _kmedoids_distance()
    duplicate = r_parity_session.wc_kmedrange_complete_contract(
        square,
        kvals=[2, 2],
        initial_kind="hclust",
    )
    too_small = r_parity_session.wc_kmedrange_complete_contract(
        square,
        kvals=[1],
        initial_kind="hclust",
    )

    assert "duplicate 'row.names'" in duplicate["error"]
    with pytest.raises(ValueError, match="duplicates"):
        wcKMedRange(square, [2, 2])
    assert too_small["error"] is not None
    with pytest.raises(ValueError, match=">= 2"):
        wcKMedRange(square, [1])


def test_as_clustrange_predictor_factor_order_and_labels_match_live_r(r_parity_session):
    square = _kmedoids_distance()
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    frame = pd.DataFrame(
        {
            "factor_first": pd.Categorical(
                ["z", "z", "z", "a", "a", "a"],
                categories=["z", "unused", "a"],
                ordered=True,
            ),
            "numeric_second": [3, 3, 2, 2, 1, 1],
        }
    )
    square_before = square.copy()
    weights_before = weights.copy()
    frame_before = frame.copy(deep=True)
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        {
            "factor_first": frame["factor_first"].array,
            "numeric_second": frame["numeric_second"].to_numpy(),
        },
        weights=weights,
        factor_levels={"factor_first": ["z", "unused", "a"]},
    )
    actual = as_clustrange(frame, squareform(square), weights=weights)

    assert expected["error"] is None
    assert expected["names"] == ["clustering", "kvals", "stats"]
    assert list(actual) == expected["names"]
    assert actual.r_class == expected["class"] == ["clustrange", "list"]
    assert actual["kvals"] == expected["kvals"] == [2, 3]
    assert actual["clustering"].column_names == ["cluster2", "cluster3"]
    assert actual["clustering"].factor_levels["cluster2"] == ["z", "unused", "a"]
    assert actual["stats"].row_names == expected["stats_row_names"]
    np.testing.assert_allclose(
        np.column_stack([actual["stats"][name] for name in R_QUALITY_NAMES]),
        expected["stats"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(square, square_before)
    np.testing.assert_array_equal(weights, weights_before)
    pd.testing.assert_frame_equal(frame, frame_before)


def test_as_clustrange_replayed_bootstrap_tables_match_live_r(r_parity_session):
    globals_before = set(
        str(name) for name in r_parity_session.r("ls(envir=.GlobalEnv, all.names=TRUE)")
    )
    square = _kmedoids_distance()
    columns = {
        "first": np.array([1, 1, 1, 2, 2, 2], dtype=int),
        "second": np.array([1, 1, 2, 2, 3, 3], dtype=int),
    }
    samples_1b = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 1, 2, 2, 3, 3],
        ],
        dtype=int,
    )
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        columns,
        R=2,
        samplesize=6,
        replay_samples_1b=samples_1b,
        bootstat="mean",
    )
    actual = as_clustrange(
        pd.DataFrame(columns),
        square,
        R=2,
        samplesize=6,
        replay_samples=samples_1b - 1,
    )

    assert expected["error"] is None
    assert (
        list(actual)
        == expected["names"]
        == [
            "clustering",
            "kvals",
            "stats",
            "boot",
            "meant",
            "stderr",
        ]
    )
    assert (
        set(str(name) for name in r_parity_session.r("ls(envir=.GlobalEnv, all.names=TRUE)"))
        == globals_before
    )
    for index, k in enumerate(actual["kvals"]):
        actual_boot = np.column_stack([actual["boot"][k][name] for name in R_QUALITY_NAMES])
        np.testing.assert_allclose(
            actual_boot,
            expected["boot"][index],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        residual_columns = [R_QUALITY_NAMES.index(name) for name in ("CH", "R2", "CHsq", "R2sq")]
        assert np.all(np.isnan(actual_boot[-1, residual_columns]))
    for field in ("meant", "stderr"):
        np.testing.assert_allclose(
            np.column_stack([actual[field][name] for name in R_QUALITY_NAMES]),
            expected[field],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("input_kind", "linkage_method"),
    [("hclust", "single"), ("twins", "average")],
)
def test_as_clustrange_hierarchy_dispatch_cuts_match_live_r(
    r_parity_session,
    input_kind,
    linkage_method,
):
    points = np.array([[0.0, 0.0], [0.4, 0.1], [1.5, 0.3], [3.2, 1.1], [6.0, 2.4], [10.0, 4.7]])
    square = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        input_kind=input_kind,
        linkage_method=linkage_method,
        ncluster=3,
    )
    hierarchy = linkage(squareform(square), method=linkage_method)
    actual = as_clustrange(hierarchy, square, ncluster=3)

    assert expected["error"] is None
    assert actual["kvals"] == expected["kvals"] == [2, 3]
    np.testing.assert_array_equal(
        np.column_stack([actual["clustering"][name] for name in actual["clustering"].column_names]),
        expected["clustering"],
    )
    np.testing.assert_allclose(
        np.column_stack([actual["stats"][name] for name in R_QUALITY_NAMES]),
        expected["stats"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("ncluster", [2, 7])
def test_as_clustrange_hierarchy_ncluster_boundaries_follow_live_r(
    r_parity_session,
    ncluster,
):
    square = _kmedoids_distance()
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        input_kind="hclust",
        ncluster=ncluster,
    )
    hierarchy = linkage(squareform(square), method="single")

    assert expected["error"] is not None
    with pytest.raises(ValueError, match="ncluster should"):
        as_clustrange(hierarchy, square, ncluster=ncluster)


def test_as_clustrange_dtclust_dispatch_matches_live_r(r_parity_session):
    y = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 12.0, 12.1, 12.2])
    square = np.abs(y[:, None] - y[None, :])
    group = np.repeat(["A", "B"], 6)
    phase = np.tile(np.repeat([0, 1], 3), 2)
    properties = pd.DataFrame({"group": pd.Categorical(group), "phase": phase})
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        input_kind="dtclust",
        ncluster=3,
        dt_properties={"group": group, "phase": phase},
        dt_categorical=["group"],
    )
    tree = wcPropertyClustering(
        square,
        properties,
        R=1,
        min_size=2,
        max_depth=3,
        pval=1,
        weight_permutation="none",
    )
    actual = as_clustrange(tree, square, ncluster=3)

    assert expected["error"] is None
    assert actual["kvals"] == expected["kvals"] == [2, 3]
    for index, name in enumerate(actual["clustering"].column_names):
        labels = np.asarray(actual["clustering"][name], dtype=object)
        expected_labels = expected["clustering"][:, index]
        np.testing.assert_array_equal(
            labels[:, None] == labels[None, :],
            expected_labels[:, None] == expected_labels[None, :],
        )
    np.testing.assert_allclose(
        np.column_stack([actual["stats"][name] for name in R_QUALITY_NAMES]),
        expected["stats"],
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    )


def test_clustrange_summary_and_print_tables_match_live_r(r_parity_session):
    square = _kmedoids_distance()
    columns = {
        "first": np.array([1, 1, 1, 2, 2, 2], dtype=int),
        "second": np.array([1, 1, 2, 2, 3, 3], dtype=int),
    }
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        columns,
        digits=3,
        max_rank=2,
    )
    actual = as_clustrange(pd.DataFrame(columns), square)
    summary = summary_clustrange(actual, max_rank=2)

    assert list(summary.index) == expected["summary_row_names"]
    assert list(summary.columns) == expected["summary_column_names"]
    np.testing.assert_allclose(
        summary.to_numpy(dtype=float),
        expected["summary"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    assert (
        summary.loc["HC", "1. N groups"] == actual["kvals"][int(np.argmin(actual["stats"]["HC"]))]
    )
    actual_tokens = [line.split() for line in print_clustrange(actual, digits=3).splitlines()]
    expected_tokens = [line.split() for line in expected["print_lines"]]
    assert actual_tokens == expected_tokens


@pytest.mark.parametrize("bootstat", ["t0", "mean", "stderr"])
def test_clustrange_bootstrap_print_selection_matches_live_r(
    r_parity_session,
    bootstat,
):
    square = _kmedoids_distance()
    columns = {
        "first": np.array([1, 1, 1, 2, 2, 2], dtype=int),
        "second": np.array([1, 1, 2, 2, 3, 3], dtype=int),
    }
    samples_1b = np.array(
        [[1, 2, 3, 4, 5, 6], [1, 1, 2, 4, 5, 6]],
        dtype=int,
    )
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        columns,
        R=2,
        samplesize=6,
        replay_samples_1b=samples_1b,
        digits=3,
        bootstat=bootstat,
    )
    actual = as_clustrange(
        pd.DataFrame(columns),
        square,
        R=2,
        samplesize=6,
        replay_samples=samples_1b - 1,
    )

    assert expected["print_error"] is None
    actual_tokens = [
        line.split() for line in print_clustrange(actual, digits=3, bootstat=bootstat).splitlines()
    ]
    assert actual_tokens == [line.split() for line in expected["print_lines"]]

    if bootstat == "t0":
        default_error = r_parity_session.as_clustrange_complete_contract(
            square,
            columns,
            R=2,
            samplesize=6,
            replay_samples_1b=samples_1b,
            bootstat=["t0", "mean", "stderr"],
        )
        assert "length = 3" in default_error["print_error"]
        with pytest.raises(ValueError, match="length = 3"):
            print_clustrange(actual)


@pytest.mark.parametrize("norm", ["none", "range", "zscore", "zscoremed"])
def test_clustrange_plot_data_selection_normalization_and_intervals(norm):
    square = _kmedoids_distance()
    columns = {
        "first": np.array([1, 1, 1, 2, 2, 2], dtype=int),
        "second": np.array([1, 1, 2, 2, 3, 3], dtype=int),
    }
    samples = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 0, 1, 3, 4, 5],
        ],
        dtype=int,
    )
    result = as_clustrange(
        pd.DataFrame(columns),
        square,
        R=2,
        samplesize=6,
        replay_samples=samples,
    )

    all_data = clustrange_plot_data(result, stat="all", norm=norm)
    assert all_data["stat_names"] == list(R_QUALITY_NAMES)
    assert all_data["norm"] == ("range" if norm == "none" else norm)
    noch = clustrange_plot_data(result, stat="noCH", norm=norm)
    assert "CH" not in noch["stat_names"] and "CHsq" not in noch["stat_names"]
    explicit = clustrange_plot_data(
        result,
        stat=["ASW", "HC", "RHC"],
        norm=norm,
        ci_method="perc",
        conf_int=0.8,
        line="median",
    )
    assert explicit["stat_names"] == ["ASW", "HC", "RHC"]
    assert explicit["with_intervals"] is True
    hc_transform = (
        1.0 - np.asarray(explicit["values"])[:, 1]
        if norm in {"none", "range"}
        else -np.asarray(explicit["values"])[:, 1]
    )
    np.testing.assert_allclose(
        np.asarray(explicit["values"])[:, 2],
        hc_transform,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    normal = clustrange_plot_data(
        result,
        stat=["ASW", "R2"],
        norm=norm,
        ci_method="norm",
        line="mean",
    )
    assert np.asarray(normal["lower"]).shape == (2, 2)
    assert np.asarray(normal["upper"]).shape == (2, 2)


def test_clustrange_plot_driving_arrays_match_live_r(r_parity_session):
    square = _kmedoids_distance()
    columns = {
        "first": np.array([1, 1, 1, 2, 2, 2], dtype=int),
        "second": np.array([1, 1, 2, 2, 3, 3], dtype=int),
    }
    samples_1b = np.array(
        [[1, 2, 3, 4, 5, 6], [1, 1, 2, 4, 5, 6]],
        dtype=int,
    )
    options = {
        "stat": ["ASW", "HC", "RHC"],
        "norm": "zscore",
        "conf_int": 0.8,
        "ci_method": "perc",
        "ci_alpha": 0.2,
        "line": "median",
    }
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        columns,
        R=2,
        samplesize=6,
        replay_samples_1b=samples_1b,
        plot_options=options,
    )
    result = as_clustrange(
        pd.DataFrame(columns),
        square,
        R=2,
        samplesize=6,
        replay_samples=samples_1b - 1,
    )
    actual = clustrange_plot_data(result, **options)

    assert "subscript out of bounds" in expected["plot"]["render_error"]
    assert actual["stat_names"] == expected["plot"]["stat_names"]
    assert actual["norm"] == expected["plot"]["norm"]
    assert actual["with_intervals"] is expected["plot"]["with_intervals"]
    assert actual["ci_alpha"] == expected["plot"]["ci_alpha"]
    assert actual["colors"] == expected["plot"]["colors"]
    assert actual["legend_labels"] == expected["plot"]["legend_labels"]
    for field in ("values", "lower", "upper"):
        np.testing.assert_allclose(
            actual[field],
            expected["plot"][field],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    all_expected = r_parity_session.as_clustrange_complete_contract(
        square,
        columns,
        plot_options={"stat": "all", "norm": "none"},
    )
    all_actual = clustrange_plot_data(
        as_clustrange(pd.DataFrame(columns), square),
        stat="all",
        norm="none",
    )
    assert all_expected["plot"]["render_error"] is None
    assert all_actual["norm"] == all_expected["plot"]["norm"] == "range"
    np.testing.assert_allclose(
        all_actual["values"],
        all_expected["plot"]["values"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_clustrange_scalar_plot_keeps_safe_python_translation_of_live_r_drop_bug(
    r_parity_session,
):
    square = _kmedoids_distance()
    columns = {
        "first": np.array([1, 1, 1, 2, 2, 2], dtype=int),
        "second": np.array([1, 1, 2, 2, 3, 3], dtype=int),
    }
    expected = r_parity_session.as_clustrange_complete_contract(
        square,
        columns,
        plot_options={"stat": "ASW"},
    )
    actual = clustrange_plot_data(
        as_clustrange(pd.DataFrame(columns), square),
        stat="ASW",
    )

    assert "argument of length 0" in expected["plot"]["render_error"]
    assert actual["stat_names"] == ["ASW"]
    np.testing.assert_allclose(
        actual["values"],
        expected["plot"]["values"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_clustrange_plot_validation_and_headless_rendering():
    square = _kmedoids_distance()
    result = as_clustrange(
        pd.DataFrame(
            {
                "first": [1, 1, 1, 2, 2, 2],
                "second": [1, 1, 2, 2, 3, 3],
            }
        ),
        square,
        R=2,
        samplesize=6,
        replay_samples=np.array(
            [[0, 1, 2, 3, 4, 5], [0, 0, 1, 3, 4, 5]],
            dtype=int,
        ),
    )

    with pytest.raises(ValueError, match="one color"):
        clustrange_plot_data(result, stat=["ASW", "R2"], col=["black"])
    with pytest.raises(ValueError, match="unknown statistic"):
        clustrange_plot_data(result, stat="unknown")
    with pytest.raises(ValueError, match="ci_method"):
        clustrange_plot_data(result, ci_method="bad")
    with pytest.raises(ValueError, match="line must"):
        clustrange_plot_data(result, ci_method="perc", line="bad")
    fig_plain, ax_plain = plot_clustrange(result, stat="noCH", withlegend=False)
    fig_ci, ax_ci = plot_clustrange(
        result,
        stat=["ASW", "R2"],
        ci_method="perc",
        line="median",
        ci_alpha=0.2,
    )
    assert len(ax_plain.lines) == 8
    assert len(ax_ci.lines) == 2
    assert len(ax_ci.collections) == 2
    fig_plain.canvas.draw()
    fig_ci.canvas.draw()


def test_cmp_cluster_default_pam_combine_family_contract_matches_live_r(
    r_parity_session,
):
    square = _kmedoids_distance()
    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method=["single", "complete"],
        pam_combine=True,
        max_rank=2,
    )
    actual = wcCmpCluster(
        square,
        maxcluster=3,
        method=["single", "complete"],
    )

    assert expected["error"] is None
    assert (
        list(actual)
        == expected["names"]
        == [
            "single",
            "pam.single",
            "complete",
            "pam.complete",
            "param",
            "allstats",
        ]
    )
    assert actual.r_class == expected["class"] == ["clustrangefamily"]
    assert list(actual["param"]) == expected["param_names"]
    assert actual["param"]["method"] == expected["param_method"]
    assert actual["param"]["pam.combine"] is expected["param_pam_combine"] is True
    assert actual["param"]["all.methods"] == expected["param_all_methods"]
    assert actual["param"]["kvals"] == expected["param_kvals"] == [2, 3]
    for method_name in expected["param_all_methods"]:
        method_actual = actual[method_name]
        method_expected = expected["methods"][method_name]
        assert list(method_actual) == method_expected["names"]
        assert method_actual.r_class == method_expected["class"]
        np.testing.assert_array_equal(
            np.column_stack(
                [
                    method_actual["clustering"][name]
                    for name in method_actual["clustering"].column_names
                ]
            ),
            method_expected["clustering"],
        )
        np.testing.assert_allclose(
            np.column_stack([method_actual["stats"][name] for name in R_QUALITY_NAMES]),
            method_expected["stats"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )


def test_cmp_cluster_signature_and_maxcluster_boundaries_follow_live_r(
    r_parity_session,
):
    signature = inspect.signature(wcCmpCluster)
    assert signature.parameters["maxcluster"].default is inspect.Parameter.empty
    assert signature.parameters["pam_combine"].default is True
    square = _kmedoids_distance()

    below = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=1,
        method="pam",
        pam_combine=False,
    )
    pam_two = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=2,
        method="pam",
        pam_combine=False,
        replay_starts_1b={2: [1, 4]},
    )
    hierarchy_two = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=2,
        method="single",
        pam_combine=False,
    )
    hierarchy_n = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=square.shape[0],
        method="single",
        pam_combine=False,
    )
    above_n = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=square.shape[0] + 1,
        method="single",
        pam_combine=False,
    )

    assert "greater than 2" in below["error"]
    with pytest.raises(ValueError, match="greater than 2"):
        wcCmpCluster(square, maxcluster=1, method="pam", pam_combine=False)
    assert pam_two["error"] is None
    assert wcCmpCluster(
        square,
        maxcluster=2,
        method="pam",
        pam_combine=False,
        random_state=7,
    )[
        "param"
    ]["kvals"] == [2]
    assert "ncluster should be greater than 2" in hierarchy_two["error"]
    with pytest.raises(ValueError, match="ncluster should be greater than 2"):
        wcCmpCluster(square, maxcluster=2, method="single", pam_combine=False)
    assert hierarchy_n["error"] is None
    assert wcCmpCluster(
        square,
        maxcluster=square.shape[0],
        method="single",
        pam_combine=False,
    )[
        "param"
    ]["kvals"] == list(range(2, square.shape[0] + 1))
    assert "ncluster should be less than" in above_n["error"]
    with pytest.raises(
        ValueError,
        match=rf"kvals cannot exceed max_k={square.shape[0]}, got {square.shape[0] + 1}",
    ):
        wcCmpCluster(
            square,
            maxcluster=square.shape[0] + 1,
            method="single",
            pam_combine=False,
        )


@pytest.mark.parametrize(
    "method_name",
    [
        "single",
        "complete",
        "average",
        "mcquitty",
        "median",
        "centroid",
        "diana",
        "beta.flexible",
    ],
)
def test_cmp_cluster_nonward_methods_match_live_r(
    r_parity_session,
    method_name,
):
    square = _cmpcluster_distance()
    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=4,
        method=method_name,
        pam_combine=False,
    )
    actual = wcCmpCluster(
        squareform(square),
        maxcluster=4,
        method=method_name.upper(),
        pam_combine=False,
    )

    assert expected["error"] is None
    assert expected["inputs_unchanged"] is True
    assert list(actual) == [method_name, "param", "allstats"]
    _assert_family_method_matches_r(actual[method_name], expected["methods"][method_name])


@pytest.mark.parametrize("method_name", ["centroid", "median"])
def test_cmp_cluster_inverting_hierarchies_match_live_r_and_bypass_optional_linkage(
    r_parity_session,
    monkeypatch,
    method_name,
):
    import weightedcluster.cmpcluster as cmpcluster_module

    square = _inverting_hclust_distance()
    expected = r_parity_session.fastcluster_hclust_contract(
        square,
        method=method_name,
        maxcluster=5,
    )
    own_hierarchy = _linkage_hclust(square, method=method_name)
    scipy_hierarchy = linkage(squareform(square), method=method_name)

    assert np.any(np.diff(expected["height"]) < 0.0)
    np.testing.assert_allclose(
        own_hierarchy[:, 2],
        expected["height"],
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(
        scipy_hierarchy[:, 2],
        expected["height"],
        rtol=1e-9,
        atol=1e-9,
    )

    def unexpected_optional_linkage(*args, **kwargs):
        raise AssertionError("centroid/median must use the R-compatible linkage")

    monkeypatch.setattr(cmpcluster_module, "_fast_linkage", unexpected_optional_linkage)
    actual = wcCmpCluster(
        square,
        maxcluster=5,
        method=method_name,
        pam_combine=False,
    )[method_name]
    all_ones = wcCmpCluster(
        square,
        weights=np.ones(square.shape[0]),
        maxcluster=5,
        method=method_name,
        pam_combine=False,
    )[method_name]
    actual_cuts = np.column_stack([actual["clustering"][f"cluster{k}"] for k in actual["kvals"]])
    ones_cuts = np.column_stack([all_ones["clustering"][f"cluster{k}"] for k in all_ones["kvals"]])
    np.testing.assert_array_equal(actual_cuts, expected["cuts"])
    np.testing.assert_array_equal(ones_cuts, expected["cuts"])


def test_cmp_cluster_all_completes_inverting_centroid_and_median_branches(
    r_parity_session,
):
    square = _inverting_hclust_distance()
    expected = {
        method: r_parity_session.fastcluster_hclust_contract(
            square,
            method=method,
            maxcluster=5,
        )["cuts"]
        for method in ("centroid", "median")
    }

    actual = wcCmpCluster(
        square,
        maxcluster=5,
        method="all",
        random_state=17,
    )

    for method in ("centroid", "median"):
        assert f"pam.{method}" in actual
        actual_cuts = np.column_stack(
            [actual[method]["clustering"][f"cluster{k}"] for k in actual[method]["kvals"]]
        )
        np.testing.assert_array_equal(actual_cuts, expected[method])


def test_cmp_cluster_weights_affect_hierarchy_and_preserve_inputs(r_parity_session):
    square = _cmpcluster_distance()
    condensed = squareform(square)
    weights = np.array([8.0, 1.0, 0.5, 4.0, 1.5, 0.75, 3.0, 1.0])
    condensed_before = condensed.copy()
    weights_before = weights.copy()
    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=4,
        method="average",
        weights=weights,
        pam_combine=False,
        distance_kind="dist",
    )
    with pytest.warns(UserWarning, match="ASW can not be computed"):
        actual = wcCmpCluster(
            condensed,
            weights=weights,
            maxcluster=4,
            method="average",
            pam_combine=False,
        )

    assert expected["error"] is None
    _assert_family_method_matches_r(actual["average"], expected["methods"]["average"])
    np.testing.assert_array_equal(condensed, condensed_before)
    np.testing.assert_array_equal(weights, weights_before)


@pytest.mark.parametrize("method_name", ["ward.D", "ward.D2"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unit", "weighted"])
def test_cmp_cluster_ward_safe_translation_matches_live_hclust(
    r_parity_session,
    method_name,
    weighted,
):
    square = _cmpcluster_distance()
    weights = np.array([4.0, 1.0, 0.5, 2.0, 1.5, 0.75, 3.0, 1.0]) if weighted else None
    wrapper = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=4,
        method=method_name,
        weights=weights,
        pam_combine=False,
    )
    expected = r_parity_session.wc_cmp_cluster(
        square,
        maxcluster=4,
        method=method_name,
        weights=weights,
        pam_combine=False,
    )
    method_key = method_name
    if weighted:
        with pytest.warns(UserWarning, match="ASW can not be computed"):
            actual = wcCmpCluster(
                square,
                maxcluster=4,
                method=method_name,
                weights=weights,
                pam_combine=False,
            )[method_key]
    else:
        actual = wcCmpCluster(
            square,
            maxcluster=4,
            method=method_name,
            weights=weights,
            pam_combine=False,
        )[method_key]

    assert wrapper["error"] == "object 'hc' not found"
    assert actual["kvals"] == expected["kvals"].tolist()
    np.testing.assert_array_equal(
        np.column_stack([actual["clustering"][name] for name in actual["clustering"].column_names]),
        np.column_stack(
            [expected["clustering"][name] for name in actual["clustering"].column_names]
        ),
    )
    np.testing.assert_allclose(
        np.column_stack([actual["stats"][name] for name in R_QUALITY_NAMES]),
        expected["stats_matrix"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_cmp_cluster_method_expansion_order_deduplication_and_validation(
    r_parity_session,
):
    square = _cmpcluster_distance()
    duplicate = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method=["complete", "single", "complete"],
        pam_combine=False,
    )
    actual_duplicate = wcCmpCluster(
        square,
        maxcluster=3,
        method=["COMPLETE", "single", "complete"],
        pam_combine=False,
    )
    raw_all = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method="all",
        pam_combine=False,
    )
    all_unweighted = wcCmpCluster(
        square,
        maxcluster=3,
        method=["single", "ALL"],
        pam_combine=False,
        random_state=11,
    )
    all_weighted = wcCmpCluster(
        square,
        weights=np.ones(square.shape[0]),
        maxcluster=3,
        method="all",
        pam_combine=False,
        random_state=11,
    )
    empty = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method=[],
        pam_combine=False,
    )
    unknown = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method="unknown",
        pam_combine=False,
    )

    assert duplicate["error"] is None
    assert duplicate["param_method"] == ["complete", "single", "complete"]
    assert actual_duplicate["param"]["method"] == ["complete", "single"]
    assert actual_duplicate["param"]["all.methods"] == ["complete", "single"]
    assert list(actual_duplicate) == ["complete", "single", "param", "allstats"]
    assert raw_all["error"] is None
    assert (
        all_unweighted["param"]["method"]
        == raw_all["param_method"]
        == [
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
    )
    assert all_weighted["param"]["method"] == [
        "ward.D",
        "ward.D2",
        "single",
        "complete",
        "average",
        "mcquitty",
        "median",
        "centroid",
        "pam",
    ]
    assert empty["error"] is None
    assert empty["names"] == ["param"]
    with pytest.raises(ValueError, match="at least one"):
        wcCmpCluster(square, maxcluster=3, method=[], pam_combine=False)
    assert unknown["error"] == "object 'hc' not found"
    with pytest.raises(ValueError, match="unknown clustering method"):
        wcCmpCluster(square, maxcluster=3, method="unknown", pam_combine=False)
    for method_name in ("diana", "beta.flexible"):
        rejected = r_parity_session.wc_cmp_cluster_complete_contract(
            square,
            weights=np.ones(square.shape[0]),
            maxcluster=3,
            method=method_name,
            pam_combine=False,
        )
        assert "cannot be used with weights" in rejected["error"]
        with pytest.raises(ValueError, match="cannot be used with weights"):
            wcCmpCluster(
                square,
                weights=np.ones(square.shape[0]),
                maxcluster=3,
                method=method_name,
                pam_combine=False,
            )


def test_cmp_cluster_plain_pam_replay_uses_explicit_zero_based_starts(
    r_parity_session,
):
    square = _cmpcluster_distance()
    starts_1b = {
        2: np.array([1, 7], dtype=int),
        3: np.array([1, 4, 7], dtype=int),
        4: np.array([1, 3, 4, 7], dtype=int),
    }
    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=4,
        method="pam",
        pam_combine=False,
        replay_starts_1b=starts_1b,
    )
    starts_0b = {k: starts - 1 for k, starts in starts_1b.items()}
    for k in starts_1b:
        np.testing.assert_array_equal(starts_0b[k], starts_1b[k] - 1)
    actual = wcCmpCluster(
        square,
        maxcluster=4,
        method="pam",
        pam_combine=False,
        replay_starts=starts_0b,
    )

    assert expected["error"] is None
    _assert_family_method_matches_r(actual["pam"], expected["methods"]["pam"])
    assert (
        set(
            str(name)
            for name in r_parity_session.r("ls(envir=.GlobalEnv, all.names=TRUE)")
            if str(name).startswith("wc_phase2_cmpcluster_")
        )
        == set()
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maxcluster": 3.5}, "maxcluster must be an integer"),
        ({"maxcluster": 3, "pam_combine": 1}, "pam_combine must be a boolean"),
        ({"maxcluster": 3, "embed_then_ward": "no"}, "embed_then_ward must be a boolean"),
        ({"maxcluster": 3, "random_state": 1.5}, "random_state must be an integer"),
    ],
)
def test_cmp_cluster_rejects_lossy_or_truthy_controls(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        wcCmpCluster(
            _cmpcluster_distance(),
            method="single",
            **kwargs,
        )


def test_clustrangefamily_summary_rejects_lossy_rank():
    family = wcCmpCluster(
        _cmpcluster_distance(),
        maxcluster=3,
        method="single",
        pam_combine=False,
    )
    with pytest.raises(ValueError, match="max_rank must be an integer"):
        summary_clustrangefamily(family, max_rank=1.5)


def test_clustrangefamily_summary_print_allstats_and_stable_ties_match_live_r(
    r_parity_session,
):
    square = _kmedoids_distance()
    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method=["single", "complete"],
        pam_combine=True,
        max_rank=2,
    )
    actual = wcCmpCluster(
        square,
        maxcluster=3,
        method=["single", "complete"],
    )
    summary = summary_clustrangefamily(actual, max_rank=2)
    rendered = print_clustrangefamily(actual, max_rank=2)

    assert actual["allstats"].column_names == expected["allstats_column_names"]
    assert actual["allstats"].row_names == expected["allstats_row_names"]
    np.testing.assert_allclose(
        np.column_stack([actual["allstats"][name] for name in R_QUALITY_NAMES]).astype(float),
        expected["allstats_values"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    assert actual["allstats"]["method"] == expected["allstats_method"]
    assert actual["allstats"]["ngroup"] == expected["allstats_ngroup"]
    assert summary.index.tolist() == expected["summary_row_names"]
    assert summary.columns.tolist() == expected["summary_column_names"]
    np.testing.assert_array_equal(summary.to_numpy(dtype=str), expected["summary"])
    assert rendered.splitlines() == expected["print_lines"]


@pytest.mark.parametrize(
    ("group", "method", "pam_combine", "stat", "norm", "colors"),
    [
        (
            "stat",
            "all",
            False,
            ["ASW", "RHC"],
            "range",
            None,
        ),
        (
            "method",
            "pam.combine",
            True,
            ["PBC", "RHC"],
            "zscoremed",
            ["black", "orange"],
        ),
    ],
)
def test_clustrangefamily_plot_driving_panels_match_live_r(
    r_parity_session,
    group,
    method,
    pam_combine,
    stat,
    norm,
    colors,
):
    square = _kmedoids_distance()
    plot_options = {
        "group": group,
        "method": method,
        "pam_combine": pam_combine,
        "stat": stat,
        "norm": norm,
        "col": colors,
        "rows": 1,
        "cols": 2,
        "main": "Comparison",
        "xlab": "groups",
        "ylab": "quality",
    }
    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method=["single", "complete"],
        pam_combine=True,
        plot_options=plot_options,
    )
    family = wcCmpCluster(
        square,
        maxcluster=3,
        method=["single", "complete"],
    )
    actual = clustrangefamily_plot_data(family, **plot_options)
    figure, axes = plot_clustrangefamily(family, **plot_options)

    assert expected["plot"]["data_error"] is None
    if group == "method":
        assert "incorrect number of subscripts" in expected["plot"]["render_error"]
    else:
        assert expected["plot"]["render_error"] is None
    assert actual["group"] == expected["plot"]["group"]
    assert actual["methods"] == expected["plot"]["methods"]
    assert actual["stats"] == expected["plot"]["stats"]
    assert actual["titles"] == expected["plot"]["titles"]
    assert actual["colors"] == expected["plot"]["colors"]
    assert actual["xlab"] == expected["plot"]["xlab"]
    assert actual["ylab"] == expected["plot"]["ylab"]
    np.testing.assert_allclose(
        actual["values"],
        expected["plot"]["values"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual["laymat"], expected["plot"]["laymat"])
    np.testing.assert_allclose(actual["heights"], expected["plot"]["heights"])
    np.testing.assert_allclose(actual["widths"], expected["plot"]["widths"])
    assert actual["legpos"] == expected["plot"]["legpos"]
    assert len(axes) == 2
    figure.canvas.draw()


def test_clustrangefamily_plot_selection_and_validation_paths():
    square = _kmedoids_distance()
    family = wcCmpCluster(
        square,
        maxcluster=3,
        method=["single", "complete"],
        pam_combine=False,
    )

    all_data = clustrangefamily_plot_data(
        family,
        group="method",
        stat="all",
        withlegend=False,
    )
    noch_data = clustrangefamily_plot_data(family, stat="noCH")
    assert all_data["stats"] == list(R_QUALITY_NAMES)
    assert noch_data["stats"] == [name for name in R_QUALITY_NAMES if "CH" not in name]
    with pytest.raises(ValueError, match="used pam.combine"):
        clustrangefamily_plot_data(family, method="pam.combine")
    with pytest.raises(ValueError, match="unknown clustering method"):
        clustrangefamily_plot_data(family, method="unknown")
    with pytest.raises(ValueError, match="unknown statistic"):
        clustrangefamily_plot_data(family, stat="unknown")
    with pytest.raises(ValueError, match="one title"):
        clustrangefamily_plot_data(
            family,
            stat=["ASW", "R2"],
            main=["only one", "two", "extra"],
        )
    with pytest.raises(ValueError, match="one color"):
        clustrangefamily_plot_data(
            family,
            method=["single", "complete"],
            stat="ASW",
            col=["black"],
        )


def test_cmp_cluster_all_pam_combination_branches_match_live_r(
    r_parity_session,
):
    square = _cmpcluster_distance()
    starts_1b = {
        2: np.array([1, 7], dtype=int),
        3: np.array([1, 4, 7], dtype=int),
    }
    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        square,
        maxcluster=3,
        method="all",
        pam_combine=True,
        replay_starts_1b=starts_1b,
    )
    actual = wcCmpCluster(
        square,
        maxcluster=3,
        method="all",
        pam_combine=True,
        replay_starts={k: starts - 1 for k, starts in starts_1b.items()},
    )

    assert expected["error"] is None
    assert actual["param"]["method"] == expected["param_method"]
    assert actual["param"]["all.methods"] == expected["param_all_methods"]
    assert list(actual) == expected["names"]
    for method_name in expected["param_all_methods"]:
        _assert_family_method_matches_r(
            actual[method_name],
            expected["methods"][method_name],
        )


SEQPROP_EXTRACTION_ROWS = np.array(
    [
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ],
    dtype=int,
)


def _assert_seqprop_frame_matches_live_r(
    actual: pd.DataFrame,
    expected: dict,
) -> None:
    assert expected["ok"]
    assert expected["inputs_unchanged"]
    assert list(actual.columns) == expected["names"]
    assert [str(int(index) + 1) for index in actual.index] == expected["row_names"]

    for position, contract in enumerate(expected["columns"]):
        series = actual.iloc[:, position]
        r_class = contract["class"]
        if r_class == ["factor"]:
            assert isinstance(series.dtype, pd.CategoricalDtype)
            assert [str(value) for value in series.cat.categories] == contract["levels"]
            values = ["<NA>" if pd.isna(value) else str(value) for value in series]
            assert values == contract["values"]
        elif r_class == ["integer"]:
            assert pd.api.types.is_integer_dtype(series.dtype)
            np.testing.assert_allclose(
                series.to_numpy(dtype=float),
                contract["numeric"],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        elif r_class == ["logical"]:
            assert pd.api.types.is_bool_dtype(series.dtype)
            np.testing.assert_allclose(
                series.astype("Float64").to_numpy(dtype=float, na_value=np.nan),
                contract["numeric"],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            assert r_class == ["numeric"]
            assert pd.api.types.is_float_dtype(series.dtype)
            np.testing.assert_allclose(
                series.to_numpy(dtype=float),
                contract["numeric"],
                rtol=1e-13,
                atol=1e-13,
                equal_nan=True,
            )


@pytest.mark.parametrize(
    "family",
    [
        "state",
        "duration",
        "spell.age",
        "spell.dur",
        "transition",
        "pattern",
        "AFtransition",
        "AFpattern",
        "Complexity",
    ],
)
def test_seqpropclust_all_property_families_match_live_r(
    r_parity_session,
    family,
):
    expected = r_parity_session.seqpropclust_complete_contract(
        SEQPROP_EXTRACTION_ROWS,
        properties=[family],
        pmin_support=0.3,
        max_k=3,
    )
    actual = seqpropclust(
        SEQPROP_EXTRACTION_ROWS,
        np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
        properties=[family],
        prop_only=True,
        pmin_support=0.3,
        max_k=3,
    )

    _assert_seqprop_frame_matches_live_r(actual, expected)


def test_seqpropclust_combined_property_order_types_and_values_match_live_r(
    r_parity_session,
):
    families = [
        "state",
        "duration",
        "spell.age",
        "spell.dur",
        "transition",
        "pattern",
        "AFtransition",
        "AFpattern",
        "Complexity",
    ]
    expected = r_parity_session.seqpropclust_complete_contract(
        SEQPROP_EXTRACTION_ROWS,
        properties=families,
        pmin_support=0.5,
        max_k=2,
    )
    actual = seqpropclust(
        SEQPROP_EXTRACTION_ROWS,
        np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
        properties=families,
        prop_only=True,
        pmin_support=0.5,
        max_k=2,
    )

    _assert_seqprop_frame_matches_live_r(actual, expected)


@pytest.mark.parametrize("combined", [False, True])
def test_seqpropclust_other_properties_alone_and_combined_match_live_r(
    r_parity_session,
    combined,
):
    other = pd.DataFrame(
        {
            "score": [1.5, np.nan, 2.5, 3.0, 4.5, 5.0],
            "group": pd.Categorical(
                ["low", "low", "mid", "mid", "high", None],
                categories=["low", "mid", "high", "unused"],
            ),
        }
    )
    families = ["state"] if combined else []
    expected = r_parity_session.seqpropclust_complete_contract(
        SEQPROP_EXTRACTION_ROWS,
        properties=families,
        other_properties={
            "score": other["score"].to_numpy(),
            "group": other["group"].astype(object).to_numpy(),
        },
        categorical=["group"],
        factor_levels={"group": list(other["group"].cat.categories)},
    )
    before_sequences = SEQPROP_EXTRACTION_ROWS.copy()
    before_other = other.copy(deep=True)
    actual = seqpropclust(
        SEQPROP_EXTRACTION_ROWS,
        np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
        properties=families,
        other_prop=other,
        prop_only=True,
    )

    _assert_seqprop_frame_matches_live_r(actual, expected)
    np.testing.assert_array_equal(SEQPROP_EXTRACTION_ROWS, before_sequences)
    pd.testing.assert_frame_equal(other, before_other)


@pytest.mark.parametrize("with_missing", [False, True])
def test_seqpropclust_missing_states_and_row_identity_match_live_r(
    r_parity_session,
    with_missing,
):
    rows = np.array(
        [
            ["A", None, "B", "B"],
            ["A", "A", "B", None],
            ["B", None, "A", "A"],
            ["B", "B", None, "A"],
        ],
        dtype=object,
    )
    row_names = ["case-a", "case-b", "case-c", "case-d"]
    expected = r_parity_session.seqpropclust_complete_contract(
        rows,
        properties=["state", "duration", "spell.age", "spell.dur"],
        with_missing=with_missing,
        row_names=row_names,
    )
    sequences = seqdef(rows, id=row_names)
    actual = seqpropclust(
        sequences,
        np.zeros((len(rows), len(rows))),
        properties=["state", "duration", "spell.age", "spell.dur"],
        with_missing=with_missing,
        prop_only=True,
    )

    assert list(actual.index) == row_names
    assert expected["row_names"] == row_names
    expected_with_python_index = dict(expected)
    expected_with_python_index["row_names"] = row_names
    actual_for_contract = actual.copy()
    actual_for_contract.index = np.arange(len(actual))
    expected_with_python_index["row_names"] = [str(index + 1) for index in range(len(actual))]
    _assert_seqprop_frame_matches_live_r(actual_for_contract, expected_with_python_index)


@pytest.mark.parametrize(
    ("family", "pmin_support", "max_k"),
    [
        ("transition", 0.0, 1),
        ("transition", 0.5, 2),
        ("transition", 0.5000001, -1),
        ("pattern", 0.0, 1),
        ("pattern", 0.5, 2),
        ("pattern", 1.0, -1),
        ("AFtransition", 0.5, 3),
        ("AFpattern", 0.5, 3),
    ],
)
def test_seqpropclust_support_and_max_k_boundaries_match_live_r(
    r_parity_session,
    family,
    pmin_support,
    max_k,
):
    expected = r_parity_session.seqpropclust_complete_contract(
        SEQPROP_EXTRACTION_ROWS,
        properties=[family],
        pmin_support=pmin_support,
        max_k=max_k,
    )
    if not expected["ok"]:
        assert "VECTOR_ELT" in expected["error"]
        with pytest.raises(ValueError, match="no frequent subsequences"):
            seqpropclust(
                SEQPROP_EXTRACTION_ROWS,
                np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
                properties=[family],
                prop_only=True,
                pmin_support=pmin_support,
                max_k=max_k,
            )
        return
    actual = seqpropclust(
        SEQPROP_EXTRACTION_ROWS,
        np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
        properties=[family],
        prop_only=True,
        pmin_support=pmin_support,
        max_k=max_k,
    )

    _assert_seqprop_frame_matches_live_r(actual, expected)


def test_seqpropclust_property_selection_and_row_mismatch_errors_are_pinned(
    r_parity_session,
):
    empty = r_parity_session.seqpropclust_complete_contract(
        SEQPROP_EXTRACTION_ROWS,
        properties=[],
    )
    unknown = r_parity_session.seqpropclust_complete_contract(
        SEQPROP_EXTRACTION_ROWS,
        properties=["not-a-property"],
    )
    mismatch = r_parity_session.seqpropclust_complete_contract(
        SEQPROP_EXTRACTION_ROWS,
        properties=[],
        other_properties={"x": [1.0, 2.0]},
    )
    assert not empty["ok"] and "subscript out of bounds" in empty["error"]
    assert not unknown["ok"] and "subscript out of bounds" in unknown["error"]
    assert not mismatch["ok"] and "number of row" in mismatch["error"]
    assert empty["inputs_unchanged"]
    assert unknown["inputs_unchanged"]
    assert mismatch["inputs_unchanged"]

    with pytest.raises(ValueError, match="no sequence or user-defined properties"):
        seqpropclust(
            SEQPROP_EXTRACTION_ROWS,
            np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
            properties=[],
            prop_only=True,
        )
    with pytest.raises(ValueError, match="Unknown properties"):
        seqpropclust(
            SEQPROP_EXTRACTION_ROWS,
            np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
            properties=["not-a-property"],
            prop_only=True,
        )
    with pytest.raises(ValueError, match="same number of rows"):
        seqpropclust(
            SEQPROP_EXTRACTION_ROWS,
            np.zeros((len(SEQPROP_EXTRACTION_ROWS), len(SEQPROP_EXTRACTION_ROWS))),
            properties=[],
            other_prop={"x": [1.0, 2.0]},
            prop_only=True,
        )


PROPERTY_TREE_Y = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 12.0, 12.1, 12.2])
PROPERTY_TREE_D = np.abs(PROPERTY_TREE_Y[:, None] - PROPERTY_TREE_Y[None, :])
PROPERTY_TREE_GROUP = np.repeat(["A", "B"], 6)
PROPERTY_TREE_PHASE = np.tile(np.repeat([0, 1], 3), 2)


def _flatten_property_tree(tree: dict) -> tuple[np.ndarray, list[dict]]:
    rows: list[list[float]] = []
    nodes: list[dict] = []

    def walk(node: dict, parent: int | None = None, side: int | None = None) -> None:
        nodes.append(node)
        split = node.get("split")
        info = node["info"]
        split_info = {} if split is None else split["info"]
        rows.append(
            [
                float(node["id"]),
                np.nan if parent is None else float(parent),
                np.nan if side is None else float(side),
                float(info["depth"]),
                float(info["splitschedule"]),
                float(info["n"]),
                float(info["vardis"]),
                float(info["medoid"]) + 1.0,
                float(split is not None),
                np.nan if split is None else float(split["varindex"]) + 1.0,
                np.nan if split is None or split.get("breaks") is None else float(split["breaks"]),
                np.nan if split is None else float(split_info["SCres"]),
                np.nan if split is None else float(split_info["lpop"]),
                np.nan if split is None else float(split_info["rpop"]),
                np.nan if split is None else float(split_info["lvar"]),
                np.nan if split is None else float(split_info["rvar"]),
                np.nan if split is None else float(split_info["R2"]),
                (
                    np.nan
                    if split is None or split_info.get("pval") is None
                    else float(split_info["pval"])
                ),
            ]
        )
        if node.get("kids") is not None:
            walk(node["kids"][0], int(node["id"]), 1)
            walk(node["kids"][1], int(node["id"]), 2)

    walk(tree["root"])
    return np.asarray(rows, dtype=float), nodes


def _assert_complete_property_tree_matches_live_r(actual: dict, expected: dict) -> None:
    assert expected["ok"]
    assert expected["inputs_unchanged"]
    assert actual["class"] == expected["class"]
    actual_nodes, node_objects = _flatten_property_tree(actual)
    np.testing.assert_allclose(
        actual_nodes,
        expected["nodes"],
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    )
    assert len(node_objects) == len(expected["members_1b"])
    for node, members_1b, probability, split_index, split_labels in zip(
        node_objects,
        expected["members_1b"],
        expected["split_prob"],
        expected["split_index"],
        expected["split_labels"],
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(node["info"]["ind"], dtype=int) + 1,
            members_1b,
        )
        if node["split"] is None:
            assert probability.size == 0
            assert split_index.size == 0
            assert split_labels == []
        else:
            np.testing.assert_allclose(
                node["split"]["prob"],
                probability,
                rtol=0.0,
                atol=1e-12,
            )
            actual_index = node["split"]["index"]
            if actual_index is None:
                assert split_index.size == 0
            else:
                np.testing.assert_array_equal(actual_index, split_index)
            assert node["split"]["labels"] == split_labels

    np.testing.assert_array_equal(actual["fitted"], expected["fitted"])
    np.testing.assert_allclose(actual["weights"], expected["weights"], rtol=0.0, atol=0.0)
    assert dtlabels(actual) == dict(
        zip(expected["label_names"], expected["label_values"], strict=True)
    )
    if len(np.unique(actual["fitted"])) >= 2:
        np.testing.assert_array_equal(
            dtcut(actual, 2, labels=False),
            expected["cut2_ids"],
        )
        assert list(dtcut(actual, 2, labels=True)) == expected["cut2_labels"]
    else:
        np.testing.assert_array_equal(
            dtcut(actual, 1, labels=False),
            expected["cut2_ids"],
        )

    adjustment = actual["adjustment"].stat["t0"].to_dict()
    np.testing.assert_allclose(
        [adjustment[name] for name in expected["adjustment"]],
        list(expected["adjustment"].values()),
        rtol=0.0,
        atol=1e-10,
    )
    assert list(actual["data"].columns) == expected["data_names"]
    for position, r_class in enumerate(expected["data_classes"]):
        series = actual["data"].iloc[:, position]
        if r_class == ["factor"]:
            assert isinstance(series.dtype, pd.CategoricalDtype)
            assert [str(value) for value in series.cat.categories] == expected["data_levels"][
                position
            ]
        else:
            assert r_class == ["numeric"]
            assert pd.api.types.is_numeric_dtype(series.dtype)

    parameters = actual["info"]["parameters"]
    np.testing.assert_allclose(
        [
            parameters["min_size"],
            parameters["max_depth"],
            parameters["R"],
            parameters["pval"],
        ],
        expected["parameter_values"],
        rtol=0.0,
        atol=0.0,
    )
    assert actual["info"].get("prune") == expected["prune"]


@pytest.mark.parametrize("variant", ["mixed", "missing", "first_squared"])
def test_property_clustering_complete_tree_contract_matches_live_r(
    r_parity_session,
    variant,
):
    if variant == "missing":
        group = PROPERTY_TREE_GROUP.astype(object)
        group[[2, 8]] = None
        phase = PROPERTY_TREE_PHASE.astype(float)
        phase[[4, 10]] = np.nan
        r_properties = {"group": group, "phase": phase}
        py_properties = pd.DataFrame(
            {
                "group": pd.Categorical(group, categories=["A", "B", "unused"]),
                "phase": phase,
            }
        )
        levels = {"group": ["A", "B", "unused"]}
        first = None
        squared = False
    else:
        r_properties = {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        }
        py_properties = pd.DataFrame(
            {
                "group": pd.Categorical(PROPERTY_TREE_GROUP),
                "phase": PROPERTY_TREE_PHASE.astype(float),
            }
        )
        levels = None
        first = "phase" if variant == "first_squared" else None
        squared = variant == "first_squared"

    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        r_properties,
        categorical=["group"],
        factor_levels=levels,
        R=1,
        pval=1.0,
        min_size=2,
        max_depth=3,
        squared=squared,
        first=first,
    )
    actual = wcPropertyClustering(
        PROPERTY_TREE_D,
        py_properties,
        R=1,
        pval=1.0,
        min_size=2,
        max_depth=3,
        squared=squared,
        first=first,
    )

    _assert_complete_property_tree_matches_live_r(actual, expected)


@pytest.mark.parametrize("maxcluster", [1, 2, 3, 4])
def test_property_clustering_pruning_and_adjustment_match_live_r(
    r_parity_session,
    maxcluster,
):
    r_properties = {
        "group": PROPERTY_TREE_GROUP,
        "phase": PROPERTY_TREE_PHASE,
    }
    py_properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        r_properties,
        categorical=["group"],
        R=1,
        min_size=2,
        max_depth=3,
        maxcluster=maxcluster,
    )
    actual = wcPropertyClustering(
        PROPERTY_TREE_D,
        py_properties,
        R=1,
        min_size=2,
        max_depth=3,
        maxcluster=maxcluster,
    )

    _assert_complete_property_tree_matches_live_r(actual, expected)


def test_dtcut_every_schedule_boundary_and_labels_mode_match_live_r(
    r_parity_session,
):
    r_properties = {
        "group": PROPERTY_TREE_GROUP,
        "phase": PROPERTY_TREE_PHASE,
    }
    py_properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.dtcut_complete_contract(
        PROPERTY_TREE_D,
        r_properties,
        categorical=["group"],
        min_size=2,
        max_depth=3,
        k_values=[-1, 0, 1, 2, 3, 4, 5],
    )
    tree = wcPropertyClustering(
        PROPERTY_TREE_D,
        py_properties,
        R=1,
        min_size=2,
        max_depth=3,
    )

    assert expected["max_k"] == 4
    assert expected["inputs_unchanged"]
    for k in [-1, 0, 1, 2, 3, 4]:
        ids = dtcut(tree, k, labels=False)
        labels = dtcut(tree, k, labels=True)
        np.testing.assert_allclose(
            np.asarray(ids, dtype=float),
            expected["cuts"][f"{k}:false"]["numeric"],
            rtol=0.0,
            atol=0.0,
        )
        normalized_labels = [
            "<NA>" if value is None or pd.isna(value) else str(value) for value in labels
        ]
        assert normalized_labels == expected["cuts"][f"{k}:true"]["values"]

    for labels in (False, True):
        with pytest.raises(ValueError, match="maximum number of groups is 4"):
            dtcut(tree, 5, labels=labels)


def test_dtcut_invalid_tree_inputs_are_rejected_without_mutation(
    r_parity_session,
):
    expected = r_parity_session.dtcut_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        min_size=2,
        max_depth=3,
        k_values=[2],
    )
    assert all(expected["invalid"].values())
    for invalid in (None, {}, {"fitted": np.array([1, 2])}):
        with pytest.raises((TypeError, ValueError), match="property-tree mapping"):
            dtcut(invalid, 2)

    tree = wcPropertyClustering(
        PROPERTY_TREE_D,
        pd.DataFrame(
            {
                "group": pd.Categorical(PROPERTY_TREE_GROUP),
                "phase": PROPERTY_TREE_PHASE.astype(float),
            }
        ),
        R=1,
        min_size=2,
        max_depth=3,
    )
    before_ids = dtcut(tree, 4, labels=False).copy()
    before_root = tree["root"]
    dtcut(tree, 2, labels=False)
    dtcut(tree, 2, labels=True)
    assert tree["root"] is before_root
    np.testing.assert_array_equal(dtcut(tree, 4, labels=False), before_ids)


def test_seqpropclust_complete_tree_arguments_and_attached_weights_match_live_r(
    r_parity_session,
):
    sequences = np.tile(
        np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 0, 0],
                [1, 0, 0, 1],
            ],
            dtype=int,
        ),
        (3, 1),
    )
    weights = np.array([1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1], dtype=float)
    other = {
        "group": PROPERTY_TREE_GROUP,
        "phase": PROPERTY_TREE_PHASE,
    }
    expected = r_parity_session.seqpropclust_complete_contract(
        sequences,
        D=PROPERTY_TREE_D,
        properties=[],
        other_properties=other,
        categorical=["group"],
        weights=weights,
        prop_only=False,
        R=1,
        pval=1.0,
        weight_permutation="diss",
        min_size=2,
        max_depth=3,
        maxcluster=3,
        first="phase",
    )
    actual = seqpropclust(
        sequences,
        PROPERTY_TREE_D,
        properties=[],
        other_prop=pd.DataFrame(
            {
                "group": pd.Categorical(PROPERTY_TREE_GROUP),
                "phase": PROPERTY_TREE_PHASE.astype(float),
            }
        ),
        weights=weights,
        R=1,
        pval=1.0,
        weight_permutation="diss",
        min_size=2,
        max_depth=3,
        maxcluster=3,
        first="phase",
    )

    _assert_complete_property_tree_matches_live_r(actual, expected)
    signature = inspect.signature(seqpropclust)
    assert signature.parameters["with_missing"].default is True
    assert signature.parameters["weight_permutation"].default == "diss"


def test_seqpropclust_random_sampling_tree_mode_matches_live_r_at_r1(
    r_parity_session,
):
    sequences = np.tile(
        np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 0, 0],
                [1, 0, 0, 1],
            ],
            dtype=int,
        ),
        (3, 1),
    )
    weights = np.array(
        [1, 2, 1.5, 1, 0.5, 1.2, 1, 1.5, 0.75, 2, 1, 1.25],
        dtype=float,
    )
    expected = r_parity_session.seqpropclust_complete_contract(
        sequences,
        D=PROPERTY_TREE_D,
        properties=[],
        other_properties={
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        weights=weights,
        prop_only=False,
        R=1,
        pval=1.0,
        weight_permutation="random-sampling",
        min_size=2,
        max_depth=3,
    )
    actual = seqpropclust(
        sequences,
        PROPERTY_TREE_D,
        properties=[],
        other_prop=pd.DataFrame(
            {
                "group": pd.Categorical(PROPERTY_TREE_GROUP),
                "phase": PROPERTY_TREE_PHASE.astype(float),
            }
        ),
        weights=weights,
        R=1,
        pval=1.0,
        weight_permutation="random-sampling",
        min_size=2,
        max_depth=3,
    )

    _assert_complete_property_tree_matches_live_r(actual, expected)


@pytest.mark.parametrize(
    ("weight_permutation", "weights"),
    [
        ("none", np.array([1, 2, 1, 3, 2, 1, 1, 1, 2, 3, 1, 2], dtype=float)),
        (
            "diss",
            np.array([1, 2, 1.5, 1, 0.5, 1.2, 1, 1.5, 0.75, 2, 1, 1.25]),
        ),
        (
            "group",
            np.array([1, 2, 1.5, 1, 0.5, 1.2, 1, 1.5, 0.75, 2, 1, 1.25]),
        ),
        (
            "replicate",
            np.array([1, 2, 1, 3, 2, 1, 1, 1, 2, 3, 1, 2], dtype=float),
        ),
        (
            "rounded-replicate",
            np.array([1, 2, 1.5, 1, 0.5, 1.2, 1, 1.5, 0.75, 2, 1, 1.25]),
        ),
        (
            "random-sampling",
            np.array([1, 2, 1.5, 1, 0.5, 1.2, 1, 1.5, 0.75, 2, 1, 1.25]),
        ),
    ],
)
def test_property_clustering_weight_permutation_modes_match_live_r_at_r1(
    r_parity_session,
    weight_permutation,
    weights,
):
    r_properties = {
        "group": PROPERTY_TREE_GROUP,
        "phase": PROPERTY_TREE_PHASE,
    }
    py_properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        r_properties,
        categorical=["group"],
        weights=weights,
        R=1,
        pval=1.0,
        min_size=2,
        max_depth=3,
        weight_permutation=weight_permutation,
    )
    actual = wcPropertyClustering(
        PROPERTY_TREE_D,
        py_properties,
        weights=weights,
        R=1,
        pval=1.0,
        min_size=2,
        max_depth=3,
        weight_permutation=weight_permutation,
    )

    _assert_complete_property_tree_matches_live_r(actual, expected)


def test_property_clustering_permutation_count_and_seeded_pvalues_are_pinned(
    r_parity_session,
):
    properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        R=7,
        pval=1.0,
        min_size=2,
        max_depth=3,
        seed=23,
    )
    first = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        R=7,
        pval=1.0,
        min_size=2,
        max_depth=3,
        random_state=23,
    )
    second = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        R=7,
        pval=1.0,
        min_size=2,
        max_depth=3,
        random_state=23,
    )

    assert expected["parameter_values"][2] == 7
    assert first["info"]["parameters"]["R"] == 7
    first_nodes, _ = _flatten_property_tree(first)
    second_nodes, _ = _flatten_property_tree(second)
    np.testing.assert_allclose(
        first_nodes,
        second_nodes,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    pvalues = first_nodes[np.isfinite(first_nodes[:, -1]), -1]
    assert np.all(pvalues >= 1 / 7)
    np.testing.assert_allclose(pvalues * 7, np.round(pvalues * 7), atol=1e-12)


@pytest.mark.parametrize(("min_size", "max_depth"), [(0.2, 2), (3.0, 2)])
def test_property_clustering_size_and_depth_boundaries_match_live_r(
    r_parity_session,
    min_size,
    max_depth,
):
    properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        R=1,
        pval=1.0,
        min_size=min_size,
        max_depth=max_depth,
    )
    actual = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        R=1,
        pval=1.0,
        min_size=min_size,
        max_depth=max_depth,
    )

    _assert_complete_property_tree_matches_live_r(actual, expected)


def test_property_clustering_one_node_r_failure_has_safe_python_translation(
    r_parity_session,
):
    properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        R=1,
        pval=1.0,
        min_size=0.2,
        max_depth=1,
    )
    actual = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        R=1,
        pval=1.0,
        min_size=0.2,
        max_depth=1,
    )

    assert not expected["ok"]
    assert "select less than one element" in expected["error"]
    assert actual["root"]["kids"] is None
    np.testing.assert_array_equal(actual["fitted"], np.ones(len(properties), dtype=int))


def test_property_clustering_deprecated_size_depth_aliases_match_canonical():
    properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    canonical = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        R=1,
        min_size=2,
        max_depth=2,
    )
    aliased = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        R=1,
        minSize=2,
        maxdepth=2,
    )

    np.testing.assert_allclose(
        _flatten_property_tree(aliased)[0],
        _flatten_property_tree(canonical)[0],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_property_clustering_random_sampling_split_path_is_source_pinned(
    r_parity_session,
):
    weights = np.array(
        [1, 2, 1.5, 1, 0.5, 1.2, 1, 1.5, 0.75, 2, 1, 1.25],
        dtype=float,
    )
    properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        weights=weights,
        R=7,
        pval=1.0,
        min_size=2,
        max_depth=3,
        weight_permutation="random-sampling",
        seed=23,
    )
    first = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        weights=weights,
        R=7,
        pval=1.0,
        min_size=2,
        max_depth=3,
        weight_permutation="random-sampling",
        random_state=23,
    )
    second = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        weights=weights,
        R=7,
        pval=1.0,
        min_size=2,
        max_depth=3,
        weight_permutation="random-sampling",
        random_state=23,
    )

    assert expected["ok"]
    first_nodes, _ = _flatten_property_tree(first)
    second_nodes, _ = _flatten_property_tree(second)
    np.testing.assert_allclose(
        first_nodes[:, :-1],
        expected["nodes"][:, :-1],
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        first_nodes,
        second_nodes,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    pvalues = first_nodes[np.isfinite(first_nodes[:, -1]), -1]
    assert np.all(pvalues >= 1 / 7)
    np.testing.assert_allclose(pvalues * 7, np.round(pvalues * 7), atol=1e-12)
    assert first["info"]["weight_permutation"] == "random-sampling"
    assert first["adjustment"].weight_permutation == "random-sampling"
    np.testing.assert_allclose(
        first["adjustment"].stat["t0"].to_numpy(),
        np.asarray(list(expected["adjustment"].values()), dtype=float),
        rtol=0.0,
        atol=1e-10,
    )


def test_property_clustering_validation_errors_are_source_pinned(
    r_parity_session,
):
    unknown_first = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        first="absent",
    )
    bad_weights = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        weights=np.ones(3),
    )
    assert not unknown_first["ok"] and unknown_first["error"]
    assert not bad_weights["ok"]

    properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    with pytest.raises(ValueError, match="square"):
        wcPropertyClustering(np.zeros((3, 2)), properties.iloc[:3])
    with pytest.raises(ValueError, match="same number of rows"):
        wcPropertyClustering(PROPERTY_TREE_D, properties.iloc[:-1])
    with pytest.raises(ValueError, match="unknown 'first' variable"):
        wcPropertyClustering(PROPERTY_TREE_D, properties, first="absent")
    with pytest.raises(ValueError, match="weights"):
        wcPropertyClustering(PROPERTY_TREE_D, properties, weights=np.ones(3))
    with pytest.raises(TypeError, match="unexpected"):
        wcPropertyClustering(PROPERTY_TREE_D, properties, not_an_option=True)


def test_property_tree_semantic_plot_data_and_headless_rendering_match_contract(
    r_parity_session,
):
    properties = pd.DataFrame(
        {
            "group": pd.Categorical(PROPERTY_TREE_GROUP),
            "phase": PROPERTY_TREE_PHASE.astype(float),
        }
    )
    expected = r_parity_session.wc_property_clustering_complete_contract(
        PROPERTY_TREE_D,
        {
            "group": PROPERTY_TREE_GROUP,
            "phase": PROPERTY_TREE_PHASE,
        },
        categorical=["group"],
        R=1,
        min_size=2,
        max_depth=3,
    )
    tree = wcPropertyClustering(
        PROPERTY_TREE_D,
        properties,
        R=1,
        min_size=2,
        max_depth=3,
    )
    semantic = proptree_plot_data(tree)

    assert list(semantic) == ["nodes", "edges", "leaves"]
    np.testing.assert_array_equal(semantic["nodes"]["id"], expected["nodes"][:, 0])
    np.testing.assert_array_equal(semantic["nodes"]["depth"], expected["nodes"][:, 3])
    np.testing.assert_array_equal(
        semantic["nodes"]["schedule"],
        expected["nodes"][:, 4],
    )
    np.testing.assert_allclose(
        semantic["nodes"]["n"],
        expected["nodes"][:, 5],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        semantic["nodes"]["vardis"],
        expected["nodes"][:, 6],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        semantic["nodes"]["medoid"] + 1,
        expected["nodes"][:, 7],
    )
    assert len(semantic["edges"]) == len(semantic["nodes"]) - 1
    assert int(semantic["leaves"]["count"].sum()) == len(PROPERTY_TREE_Y)
    assert semantic["leaves"]["label"].tolist() == [
        dtlabels(tree)[str(value)] for value in semantic["leaves"]["id"]
    ]

    sequence_tree = seqpropclust(
        np.tile(
            np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1]]),
            (3, 1),
        ),
        PROPERTY_TREE_D,
        properties=[],
        other_prop=properties,
        R=1,
        min_size=2,
        max_depth=3,
    )
    import matplotlib.pyplot as plt

    figures = []
    for candidate in (tree, sequence_tree):
        figures.extend(
            [
                plot_proptree_labels(candidate)[0],
                plot_proptree_tree(candidate)[0],
                plot_proptree_panel(candidate)[0],
            ]
        )
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


SEQCLARA_X = np.array(
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


def test_seqclararange_signature_exposes_deterministic_replay_seams():
    signature = inspect.signature(seqclararange)
    assert "replay_samples" in signature.parameters
    assert "replay_initial_memberships" in signature.parameters
    assert signature.parameters["replay_samples"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["replay_initial_memberships"].kind is inspect.Parameter.KEYWORD_ONLY


def test_seqclararange_default_sample_formula_and_r_partial_matching():
    out = seqclararange(
        SEQCLARA_X,
        R=1,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        method="cri",
        random_state=17,
    )
    assert out["clara"][2]["subsample_size"] == 44
    assert out["clara"][2]["method"] == "crisp"

    with pytest.raises(ValueError, match="theoretical maximum distance"):
        seqclararange(
            SEQCLARA_X,
            R=1,
            sample_size=4,
            kvals=[2],
            seqdist_args={"method": "HAM"},
            method="rep",
            random_state=17,
        )


def test_seqclararange_r_classes_and_stability_matrix_shape():
    out = seqclararange(
        SEQCLARA_X,
        R=3,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        method="crisp",
        criteria="distance",
        stability=True,
        random_state=29,
    )
    assert out.r_class == ["seqclararange", "clustrange"]
    assert out["clara"][2].r_class == ["seqclara"]
    assert np.asarray(out["clara"][2]["arimatrix"]).shape == (3, 2)
    fig, axes = plot_seqclararange_stability(out, stat="stability")
    assert len(axes) == 1
    assert axes[0].get_xlabel() == "Number of clusters"
    assert axes[0].get_legend() is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


SEQCLARA_REPLAY_1B = np.array(
    [
        [1, 2, 3, 4, 5, 6],
        [1, 1, 2, 4, 5, 6],
        [1, 2, 3, 3, 5, 6],
    ],
    dtype=int,
)

# Eight distinct trajectories, so that a CLARA subsample can retain enough
# distinct cases for R's `cluster::fanny` to accept k = 2.
SEQCLARA_FUZZY_X = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ],
    dtype=int,
)

SEQCLARA_FUZZY_REPLAY_1B = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 7],
        [1, 2, 3, 4, 5, 6, 6, 8],
    ],
    dtype=int,
)


def _seqclara_python_replay_plan(expected: dict, *, R: int) -> dict[int, list[dict]]:
    return {
        int(k): [
            {
                "ward": np.asarray(plan["ward"], dtype=float),
                "fanny": np.asarray(plan["fanny"], dtype=float),
            }
            for plan in plans[:R]
        ]
        for k, plans in expected["initial_memberships"].items()
    }


def _seqclara_analysis_weight_scale(expected: dict) -> float:
    aggregate_weights = np.asarray(expected["agg_weights"], dtype=float)
    probabilities = np.asarray(expected["probs"], dtype=float)
    positive = probabilities > 0.0
    ratios = aggregate_weights[positive] / probabilities[positive]
    assert ratios.size > 0
    np.testing.assert_allclose(ratios, ratios[0], rtol=0.0, atol=1e-12)
    return float(ratios[0])


def _analysis_weight_adjusted_seqclara_values(
    name: str,
    values: np.ndarray | list[float] | float,
    *,
    scale: float,
    method: str = "crisp",
):
    array = np.asarray(values, dtype=float)
    scale_linear = {"avg dist", "distance", "xb"}
    if method not in {"fuzzy", "noise", "hncdd", "pcmdd"}:
        scale_linear.add("ams")
    if name.lower() in scale_linear:
        array = array * scale
    elif name.lower() == "pbm":
        array = array / (scale * scale)
    return float(array) if array.ndim == 0 else array


def _assert_seqclara_range_matches_live_r(actual: dict, expected: dict) -> None:
    """Match R structure while enforcing the audited analysis-weight correction."""
    assert expected["error"] is None
    assert actual.r_class == expected["class"] == ["seqclararange", "clustrange"]
    assert list(actual)[:4] == expected["names"]
    assert actual["kvals"] == expected["kvals"].tolist()
    assert actual["stats"].row_names == expected["stats_row_names"]
    assert list(actual["stats"]) == expected["stats_column_names"]
    scale = _seqclara_analysis_weight_scale(expected)
    method = str(next(iter(expected["clara"].values()))["method"])
    expected_stats = np.asarray(expected["stats"], dtype=float).copy()
    for column, name in enumerate(expected["stats_column_names"]):
        expected_stats[:, column] = _analysis_weight_adjusted_seqclara_values(
            name,
            expected_stats[:, column],
            scale=scale,
            method=method,
        )
    np.testing.assert_allclose(
        np.column_stack(list(actual["stats"].values())),
        expected_stats,
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    )
    for k in actual["kvals"]:
        py_clara = actual["clara"][int(k)]
        r_clara = expected["clara"][int(k)]
        assert py_clara.r_class == r_clara["class"] == ["seqclara"]
        assert py_clara["criteria"] == r_clara["criteria"]
        assert py_clara["method"] == r_clara["method"]
        assert py_clara["iteration"] == r_clara["iteration"]
        np.testing.assert_array_equal(py_clara["medoids"], r_clara["medoids_0b"])
        np.testing.assert_allclose(
            np.asarray(py_clara["clustering"], dtype=float),
            r_clara["clustering"],
            rtol=0.0,
            atol=1e-10,
        )
        for field in ("iter.objective", "evol.diss"):
            np.testing.assert_allclose(
                py_clara[field],
                _analysis_weight_adjusted_seqclara_values(
                    py_clara["criteria"],
                    r_clara[field],
                    scale=scale,
                    method=py_clara["method"],
                ),
                rtol=0.0,
                atol=1e-10,
                equal_nan=True,
            )
        assert py_clara["objective"] == pytest.approx(
            _analysis_weight_adjusted_seqclara_values(
                py_clara["criteria"],
                r_clara["objective"],
                scale=scale,
                method=py_clara["method"],
            ),
            rel=0.0,
            abs=1e-10,
        )
        expected_summary = np.asarray(r_clara["summary"], dtype=float).copy()
        for index, name in enumerate(("distance", "pbm", "db", "xb", "ams")):
            expected_summary[index] = _analysis_weight_adjusted_seqclara_values(
                name,
                expected_summary[index],
                scale=scale,
                method=py_clara["method"],
            )
        np.testing.assert_allclose(
            [
                py_clara["avg_dist"],
                py_clara["pbm"],
                py_clara["db"],
                py_clara["xb"],
                py_clara["ams"],
                py_clara["ari08"],
                py_clara["jc08"],
                py_clara["R"],
                py_clara["k"],
            ],
            expected_summary,
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        )


def test_seqclararange_replayed_crisp_complete_contract_matches_live_r(
    r_parity_session,
):
    X = SEQCLARA_X + 1
    replay = SEQCLARA_REPLAY_1B[:3]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=3,
        sample_size=6,
        method="crisp",
        criteria="distance",
        stability=True,
        keep_diss=True,
        max_dist=1.0,
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    actual = seqclararange(
        X,
        R=3,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        method="crisp",
        criteria="distance",
        stability=True,
        keep_diss=True,
        max_dist=1.0,
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=3),
        random_state=999,
    )
    assert expected["replay_consumed"] == 3
    np.testing.assert_array_equal(
        actual["clara"][2]["iterations"][0]["sample_draw"],
        replay[0] - 1,
    )
    np.testing.assert_array_equal(
        actual["clara"][2]["iterations"][1]["sample_unique_indices"],
        expected["initial_memberships"][2][1]["sample_unique_indices_0b"],
    )
    np.testing.assert_allclose(
        actual["clara"][2]["iterations"][1]["sample_weights"],
        expected["initial_memberships"][2][1]["sample_weights"],
        rtol=0.0,
        atol=0.0,
    )
    _assert_seqclara_range_matches_live_r(actual, expected)
    np.testing.assert_allclose(
        actual["clara"][2]["diss"],
        expected["clara"][2]["diss"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual["clara"][2]["arimatrix"],
        expected["clara"][2]["arimatrix"],
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("kvals", ([2], [2, 3]))
def test_seqclararange_default_initialization_matches_live_r(r_parity_session, kvals):
    """Drive the **default** initialization path against R (finding F-05).

    Every other `seqclararange` parity test injects R's own per-iteration
    memberships through `replay_initial_memberships`, so a wrong Python
    initializer was invisible to the suite. This test replays only the CLARA
    *sample draw* — which is an RNG stream that cannot be shared across
    languages — and leaves the initialization to Python.

    R seeds each iteration deterministically from a Ward hierarchy on the
    subsample: `fastcluster::hclust(as.dist(diss), method="ward.D",
    members=ac2$aggWeights)`, then `wcKMedoids(diss, k, cluster.only=TRUE,
    initialclust=hc, weights=ac2$aggWeights)` (R/clara_clustrange.R:101,:125).
    Python used random-restart PAM instead, so for a given subsample R's
    clustering was deterministic while Python's was a search.
    """
    X = SEQCLARA_X + 1
    replay = SEQCLARA_REPLAY_1B[:3]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=kvals,
        R=3,
        sample_size=6,
        method="crisp",
        criteria="distance",
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    actual = seqclararange(
        X,
        R=3,
        sample_size=6,
        kvals=kvals,
        seqdist_args={"method": "LCS"},
        method="crisp",
        criteria="distance",
        replay_samples=replay - 1,
        # Deliberately no replay_initial_memberships: this is the point.
        random_state=999,
    )
    _assert_seqclara_range_matches_live_r(actual, expected)


def test_seqclararange_fuzzy_default_start_is_r_ward_cut(r_parity_session, monkeypatch):
    """The fuzzy default path starts from R's Ward cut (finding F-05).

    R's fuzzy branch builds `memb <- as.memb(cutree(hc, k))` from the same Ward
    hierarchy the crisp branch uses, runs FCMdd from it, then runs a *second*
    FCMdd from `fanny(diss, k, diss=TRUE, memb.exp=m, iniMem.p=memb)` and keeps
    whichever reaches the lower functional (R/clara_clustrange.R:112-118).

    Python implements the Ward candidate and does not implement `fanny`, which
    belongs to the R `cluster` package. That residual is a deliberate,
    documented divergence — see `docs/R_MIGRATION_GUIDE.md` — so this test
    asserts what the fix does implement: on the **default** path, with no
    membership replay, every iteration's starting membership is R's own Ward
    cut. Before the fix it was a random-restart crisp fit, which is what made a
    wrong initializer invisible to the suite.
    """
    import weightedcluster.fuzzy as fuzzy_module

    # R's `cluster::fanny` refuses k outside {1, ..., n/2 - 1}, so the fuzzy
    # branch needs a subsample with enough distinct cases to reach it; every
    # replay row below draws at least six of the eight aggregated cases.
    X = SEQCLARA_FUZZY_X + 1
    replay = SEQCLARA_FUZZY_REPLAY_1B
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=3,
        sample_size=8,
        method="fuzzy",
        criteria="distance",
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    assert expected["error"] is None

    starts: list[np.ndarray] = []
    real_fcmdd = fuzzy_module.fcmdd

    def spy_fcmdd(D_in, u_init, **kwargs):
        starts.append(np.asarray(u_init, dtype=float).copy())
        return real_fcmdd(D_in, u_init, **kwargs)

    monkeypatch.setattr(fuzzy_module, "fcmdd", spy_fcmdd)

    seqclararange(
        X,
        R=3,
        sample_size=8,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        method="fuzzy",
        criteria="distance",
        replay_samples=replay - 1,
        # Deliberately no replay_initial_memberships: this is the point.
        random_state=999,
    )

    r_plans = expected["initial_memberships"][2][:3]
    assert len(starts) == len(r_plans) == 3
    for iteration, (python_start, plan) in enumerate(zip(starts, r_plans, strict=True)):
        np.testing.assert_allclose(
            python_start,
            np.asarray(plan["ward"], dtype=float),
            rtol=0.0,
            atol=0.0,
            err_msg=f"iteration {iteration}",
        )


@pytest.mark.parametrize("criterion", ["distance", "db", "xb", "pbm", "ams"])
def test_seqclararange_all_r_criteria_replay_match_live_r(
    r_parity_session,
    criterion,
):
    X = SEQCLARA_X + 1
    replay = SEQCLARA_REPLAY_1B[:2]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=2,
        sample_size=6,
        method="crisp",
        criteria=criterion,
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    actual = seqclararange(
        X,
        R=2,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        method="crisp",
        criteria=criterion,
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=2),
    )
    _assert_seqclara_range_matches_live_r(actual, expected)


@pytest.mark.parametrize(
    ("method", "dnoise", "max_dist"),
    [
        ("fuzzy", None, None),
        ("noise", 0.75, None),
        ("representativeness", None, 1.0),
    ],
)
def test_seqclararange_all_r_methods_replay_match_live_r(
    r_parity_session,
    method,
    dnoise,
    max_dist,
):
    X = SEQCLARA_X + 1
    replay = (
        np.vstack([SEQCLARA_REPLAY_1B[0], SEQCLARA_REPLAY_1B[0][::-1]])
        if method in {"fuzzy", "noise"}
        else SEQCLARA_REPLAY_1B[:2]
    )
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=2,
        sample_size=6,
        method=method,
        criteria="distance",
        dnoise=dnoise,
        max_dist=max_dist,
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    actual = seqclararange(
        X,
        R=2,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        method=method,
        criteria="distance",
        dnoise=dnoise,
        max_dist=max_dist,
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=2),
    )
    _assert_seqclara_range_matches_live_r(actual, expected)
    np.testing.assert_allclose(
        actual["clustering"]["cluster2"],
        expected["clustering"]["cluster2"],
        rtol=0.0,
        atol=1e-10,
    )
    if method == "noise":
        assert np.asarray(actual["clara"][2]["clustering"]).shape == (len(X), 3)
    if method == "representativeness":
        np.testing.assert_allclose(
            actual["clara"][2]["representativeness"],
            expected["clara"][2]["representativeness"],
            rtol=0.0,
            atol=1e-12,
        )


def test_seqclararange_sts_weights_aggregation_and_no_mutation_match_live_r(
    r_parity_session,
):
    X = np.vstack([SEQCLARA_X + 1, SEQCLARA_X[[0, 3]] + 1])
    weights = np.array([1.0, 0.5, 1.5, 2.0, 0.75, 1.25, 0.5, 1.5])
    sequences = seqdef(X, weights=weights)
    before_data = np.asarray(sequences.data).copy()
    before_weights = np.asarray(sequences.weights).copy()
    replay = np.array([[1, 2, 3, 4, 5, 6]], dtype=int)
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=1,
        sample_size=6,
        method="crisp",
        criteria="distance",
        weights=weights,
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    actual = seqclararange(
        sequences,
        R=1,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=1),
    )
    np.testing.assert_array_equal(np.asarray(sequences.data), before_data)
    np.testing.assert_array_equal(np.asarray(sequences.weights), before_weights)
    np.testing.assert_array_equal(
        expected["agg_index_0b"],
        np.asarray([0, 1, 2, 5, 4, 3]),
    )
    np.testing.assert_allclose(
        actual["clara"][2]["iterations"][0]["sample_weights"],
        expected["initial_memberships"][2][0]["sample_weights"],
        rtol=0.0,
        atol=0.0,
    )
    _assert_seqclara_range_matches_live_r(actual, expected)


def test_seqclararange_family_schema_and_criteria_order_match_live_r(
    r_parity_session,
):
    X = SEQCLARA_X + 1
    criteria = ["distance", "db", "xb", "pbm", "ams"]
    replay = SEQCLARA_REPLAY_1B[:2]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=2,
        sample_size=6,
        criteria=criteria,
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    actual = seqclararange(
        X,
        R=2,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        criteria=criteria,
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=2),
    )
    # WeightedCluster 2.0 overwrites `criteria` inside its aggregation loop, so
    # the installed multi-criteria call incorrectly falls through to a single
    # seqclararange result. Python exposes the source-intended family schema.
    assert expected["family_requested"] is True
    assert expected["family_source_bug"] is True
    assert expected["class"] == ["seqclararange", "clustrange"]
    assert actual.r_class == ["seqclarafamily", "clustrangefamily"]
    assert list(actual) == ["param", *criteria, "allstats"]
    assert actual["param"]["criteria"] == criteria
    assert actual["param"]["all.criterias"] == criteria
    assert actual["param"]["kvals"] == [2]
    assert list(actual["allstats"]) == [
        "Avg dist",
        "PBM",
        "DB",
        "XB",
        "AMS",
        "ARI>0.8",
        "JC>0.8",
        "Best iter",
        "ngroup",
        "criteria",
    ]
    np.testing.assert_array_equal(actual["allstats"]["ngroup"], [2] * 5)
    assert actual["allstats"]["criteria"] == criteria


def test_seqclararange_replay_parallel_progress_and_seed_do_not_change_results(
    r_parity_session,
):
    X = SEQCLARA_X + 1
    replay = SEQCLARA_REPLAY_1B[:3]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=3,
        sample_size=6,
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    kwargs = dict(
        R=3,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=3),
    )
    serial = seqclararange(X, parallel=False, progressbar=False, random_state=1, **kwargs)
    parallel = seqclararange(
        X,
        parallel=True,
        n_jobs=2,
        progressbar=True,
        random_state=999,
        **kwargs,
    )
    np.testing.assert_allclose(
        np.column_stack(list(serial["stats"].values())),
        np.column_stack(list(parallel["stats"].values())),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        serial["clustering"]["cluster2"],
        parallel["clustering"]["cluster2"],
    )


def test_seqclararange_plot_modes_and_live_r_source_errors(
    r_parity_session,
):
    X = SEQCLARA_X + 1
    replay = SEQCLARA_REPLAY_1B[:2]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=2,
        sample_size=6,
        stability=True,
        replay_samples_1b=replay,
        seqdist_method="LCS",
    )
    actual = seqclararange(
        X,
        R=2,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        stability=True,
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=2),
    )
    cqi = _seqclararange_plot_data(actual, stat="CQI", norm="none")
    scale = _seqclara_analysis_weight_scale(expected)
    expected_pbm = _analysis_weight_adjusted_seqclara_values(
        "pbm", expected["stats"][0, 1], scale=scale
    )
    expected_db = _analysis_weight_adjusted_seqclara_values(
        "db", expected["stats"][0, 2], scale=scale
    )
    expected_xb = _analysis_weight_adjusted_seqclara_values(
        "xb", expected["stats"][0, 3], scale=scale
    )
    expected_ams = _analysis_weight_adjusted_seqclara_values(
        "ams", expected["stats"][0, 4], scale=scale
    )
    np.testing.assert_allclose(
        cqi["values"][0],
        [
            expected_pbm,
            expected_db,
            expected_xb,
            expected_ams,
            expected_pbm * np.sqrt(2),
        ],
        rtol=0.0,
        atol=1e-10,
    )
    for norm in ("range", "zscore", "zscoremed"):
        normalized = _seqclararange_plot_data(actual, stat="CQI", norm=norm)
        assert normalized["series"] == ["PBM", "DB", "XB", "AMS", "PBMK"]
        assert np.asarray(normalized["values"]).shape == (1, 5)
    stability_data = _seqclararange_plot_data(actual, stat="stability")
    matrix = expected["clara"][2]["arimatrix"]
    np.testing.assert_array_equal(
        stability_data["values"][0],
        [
            np.sum(matrix[:, 0] >= 0.9),
            np.sum(matrix[:, 0] >= 0.8),
            np.sum(matrix[:, 0] >= 0.7),
            np.sum(matrix[:, 1] >= 0.5),
            np.sum(matrix[:, 1] >= 2 / 3),
            np.sum(matrix[:, 1] >= 0.8),
        ],
    )
    assert expected["plot_errors"]["CQImv"] is not None
    with pytest.raises(ValueError, match="SASW"):
        _seqclararange_plot_data(actual, stat="CQImv")

    unavailable = seqclararange(
        X,
        R=2,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS"},
        stability=False,
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=2),
    )
    with pytest.raises(ValueError, match="stability evaluation is not available"):
        _seqclararange_plot_data(unavailable, stat="stability")

    import matplotlib.pyplot as plt

    for mode in ("CQI", "ari.boxplot", "stability", "stabmean"):
        if expected["plot_errors"][mode] is None:
            figure, axes = plot_seqclararange_stability(actual, stat=mode)
            assert axes[0].figure is figure
            plt.close(figure)


def test_seqclararange_distance_normalization_replay_matches_live_r(
    r_parity_session,
):
    X = SEQCLARA_X + 1
    replay = SEQCLARA_REPLAY_1B[:1]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=1,
        sample_size=6,
        replay_samples_1b=replay,
        seqdist_method="LCS",
        seqdist_norm="maxlength",
    )
    actual = seqclararange(
        X,
        R=1,
        sample_size=6,
        kvals=[2],
        seqdist_args={"method": "LCS", "norm": "maxlength"},
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=1),
    )
    _assert_seqclara_range_matches_live_r(actual, expected)


def test_seqclararange_r_partial_method_and_criterion_matching_live_r(
    r_parity_session,
):
    X = SEQCLARA_X + 1
    replay = SEQCLARA_REPLAY_1B[:1]
    expected = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=1,
        sample_size=6,
        method="cri",
        criteria="dis",
        replay_samples_1b=replay,
    )
    actual = seqclararange(
        X,
        R=1,
        sample_size=6,
        kvals=[2],
        method="cri",
        criteria="dis",
        replay_samples=replay - 1,
        replay_initial_memberships=_seqclara_python_replay_plan(expected, R=1),
    )
    assert expected["clara"][2]["method"] == actual["clara"][2]["method"] == "crisp"
    assert expected["clara"][2]["criteria"] == actual["clara"][2]["criteria"] == "distance"
    _assert_seqclara_range_matches_live_r(actual, expected)


def test_seqclararange_exact_boundary_errors_match_live_r(r_parity_session):
    X = SEQCLARA_X + 1
    too_many = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[4],
        R=1,
        sample_size=3,
    )
    assert "More clusters than the size of the sample" in too_many["error"]
    with pytest.raises(ValueError, match="More clusters"):
        seqclararange(X, R=1, sample_size=3, kvals=[4])

    represent = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=1,
        sample_size=6,
        method="representativeness",
    )
    assert "theoretical maximum distance" in represent["error"]
    with pytest.raises(ValueError, match="theoretical maximum distance"):
        seqclararange(
            X,
            R=1,
            sample_size=6,
            kvals=[2],
            method="representativeness",
        )

    with pytest.raises(ValueError, match="replay_samples"):
        seqclararange(
            X,
            R=2,
            sample_size=6,
            kvals=[2],
            replay_samples=np.zeros((1, 6), dtype=int),
        )

    unknown_method = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=1,
        sample_size=6,
        method="unknown",
    )
    assert "Unknow method" in unknown_method["error"]
    with pytest.raises(ValueError, match="Unknown method"):
        seqclararange(X, R=1, sample_size=6, kvals=[2], method="unknown")

    unknown_criterion = r_parity_session.seqclararange_complete_contract(
        X,
        kvals=[2],
        R=1,
        sample_size=6,
        criteria="unknown",
    )
    assert "unkown criteria" in unknown_criterion["error"]
    with pytest.raises(ValueError, match="Unknown criteria"):
        seqclararange(X, R=1, sample_size=6, kvals=[2], criteria="unknown")

    with pytest.raises(ValueError, match="norm"):
        seqclararange(
            X,
            R=1,
            sample_size=6,
            kvals=[2],
            seqdist_args={"method": "LCS", "norm": "not-a-normalization"},
        )


CLUSTASSOC_POSITION = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 1.8, 2.3, 2.9, 3.4, 4.0, 4.5, 5.1])
CLUSTASSOC_D = np.abs(CLUSTASSOC_POSITION[:, None] - CLUSTASSOC_POSITION[None, :])
CLUSTASSOC_CLUSTER2 = np.repeat([1, 2], 6)
CLUSTASSOC_CLUSTER3 = np.repeat([1, 2, 3], 4)
CLUSTASSOC_NUMERIC = np.array([0.0, 0.2, 0.1, 0.4, 1.1, 1.0, 1.3, 1.4, 2.2, 2.0, 2.3, 2.4])
CLUSTASSOC_CATEGORICAL = np.array(["a", "a", "b", "a", "b", "b", "a", "b", "a", "b", "b", "b"])
CLUSTASSOC_WEIGHTS = np.array([0.5, 1.0, 1.5, 2.0, 1.0, 2.0, 1.0, 1.5, 0.75, 1.25, 2.25, 0.8])


def _clustassoc_range():
    return {
        "kvals": [2, 3],
        "clustering": {
            "cluster2": CLUSTASSOC_CLUSTER2.copy(),
            "cluster3": CLUSTASSOC_CLUSTER3.copy(),
        },
        "stats": {
            "PBC": [0.0, 0.0],
            "HG": [0.0, 0.0],
            "HGSD": [0.0, 0.0],
        },
    }


def test_clustassoc_numeric_result_is_exact_typed_r_table(r_parity_session):
    expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
    )
    assert expected["error"] is None
    actual = clustassoc(
        _clustassoc_range(),
        CLUSTASSOC_D,
        CLUSTASSOC_NUMERIC,
    )

    assert list(actual) == expected["column_names"]
    assert getattr(actual, "r_class", None) == expected["class"]
    assert getattr(actual, "row_names", None) == expected["row_names"]
    np.testing.assert_allclose(
        np.column_stack([actual[name] for name in actual]),
        expected["table"],
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("categorical", [False, True], ids=["numeric", "factor"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unit", "nonuniform"])
def test_clustassoc_all_models_and_discrepancy_terms_match_live_r(
    r_parity_session,
    categorical,
    weighted,
):
    covar = CLUSTASSOC_CATEGORICAL if categorical else CLUSTASSOC_NUMERIC
    weights = CLUSTASSOC_WEIGHTS if weighted else None
    expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        covar,
        kvals=[2, 3],
        categorical=categorical,
        weights=weights,
    )
    assert expected["error"] is None
    actual = clustassoc(_clustassoc_range(), CLUSTASSOC_D, covar, weights=weights)
    actual_table = np.column_stack([actual[name] for name in actual])

    np.testing.assert_allclose(
        actual_table,
        expected["table"],
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual_table[:, :2],
        expected["direct_association"],
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual["BIC"],
        expected["model_diagnostics"][:, 0],
        rtol=0.0,
        atol=1e-7,
    )
    assert actual.additions["BICConverged"] == (expected["model_diagnostics"][:, 4] == 0).tolist()


def test_clustassoc_delegates_every_association_to_public_traminex_without_mutation(
    monkeypatch,
):
    import importlib

    import traminex

    clustassoc_module = importlib.import_module("weightedcluster.clustassoc")

    distance = CLUSTASSOC_D.copy()
    covar = CLUSTASSOC_NUMERIC.copy()
    weights = CLUSTASSOC_WEIGHTS.copy()
    cluster_range = _clustassoc_range()
    before_range = {
        "kvals": list(cluster_range["kvals"]),
        "clustering": {name: values.copy() for name, values in cluster_range["clustering"].items()},
        "stats": {name: list(values) for name, values in cluster_range["stats"].items()},
    }
    calls: list[tuple[str, ...]] = []
    delegate = traminex.dissmfacw

    def spy(*args, **kwargs):
        calls.append(tuple(kwargs["factors"]))
        return delegate(*args, **kwargs)

    monkeypatch.setattr(clustassoc_module, "dissmfacw", spy)
    clustassoc(cluster_range, distance, covar, weights=weights)

    assert calls == [("covar",), ("clustf", "covar"), ("clustf", "covar")]
    np.testing.assert_array_equal(distance, CLUSTASSOC_D)
    np.testing.assert_array_equal(covar, CLUSTASSOC_NUMERIC)
    np.testing.assert_array_equal(weights, CLUSTASSOC_WEIGHTS)
    assert cluster_range["kvals"] == before_range["kvals"]
    for name in cluster_range["clustering"]:
        np.testing.assert_array_equal(
            cluster_range["clustering"][name],
            before_range["clustering"][name],
        )
    assert cluster_range["stats"] == before_range["stats"]


def test_clustassoc_missing_values_and_unused_factor_levels_match_live_r(
    r_parity_session,
):
    numeric = CLUSTASSOC_NUMERIC.copy()
    numeric[1] = np.nan
    numeric_expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        numeric,
        kvals=[2, 3],
    )
    numeric_actual = clustassoc(_clustassoc_range(), CLUSTASSOC_D, numeric)
    np.testing.assert_allclose(
        np.column_stack([numeric_actual[name] for name in numeric_actual]),
        numeric_expected["table"],
        rtol=0.0,
        atol=1e-10,
    )

    values = CLUSTASSOC_CATEGORICAL.astype(object)
    values[1] = None
    categorical = pd.Categorical(values, categories=["a", "b", "unused"])
    factor_expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        values,
        kvals=[2, 3],
        categorical=True,
        covar_levels=["a", "b", "unused"],
    )
    assert factor_expected["warnings"] == ["group ‘unused’ is empty"] * 3
    with pytest.warns(RuntimeWarning) as caught:
        factor_actual = clustassoc(
            _clustassoc_range(),
            CLUSTASSOC_D,
            categorical,
        )
    assert [str(warning.message) for warning in caught] == factor_expected["warnings"]
    np.testing.assert_allclose(
        np.column_stack([factor_actual[name] for name in factor_actual]),
        factor_expected["table"],
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("malformed", "mutation"),
    [
        ("missing_stats", lambda value: value.pop("stats")),
        (
            "stats_two_columns",
            lambda value: value.__setitem__(
                "stats",
                {"PBC": [0.0, 0.0], "HG": [0.0, 0.0]},
            ),
        ),
        ("missing_clustering", lambda value: value.pop("clustering")),
        (
            "clustering_row_mismatch",
            lambda value: value.__setitem__(
                "clustering",
                {name: column[:-1] for name, column in value["clustering"].items()},
            ),
        ),
        (
            "stats_row_mismatch",
            lambda value: value.__setitem__(
                "stats",
                {"PBC": [0.0], "HG": [0.0], "HGSD": [0.0]},
            ),
        ),
        (
            "kvals_mismatch",
            lambda value: value.__setitem__("kvals", [2]),
        ),
    ],
)
def test_clustassoc_malformed_clustrange_errors_are_live_r_pinned(
    r_parity_session,
    malformed,
    mutation,
):
    expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        malformed=malformed,
    )
    assert expected["error"]
    value = _clustassoc_range()
    mutation(value)
    with pytest.raises(ValueError, match=re.escape(expected["error"])):
        clustassoc(value, CLUSTASSOC_D, CLUSTASSOC_NUMERIC)


def test_clustassoc_distance_covar_and_model_boundaries_match_live_r(
    r_parity_session,
):
    columns = {
        "cluster2": CLUSTASSOC_CLUSTER2,
        "cluster3": CLUSTASSOC_CLUSTER3,
    }
    nonsquare = r_parity_session.clustassoc_complete_contract(
        np.ones((11, 12)),
        {
            "cluster2": CLUSTASSOC_CLUSTER2[:11],
            "cluster3": CLUSTASSOC_CLUSTER3[:11],
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
    )
    short_covar = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        columns,
        CLUSTASSOC_NUMERIC[:-1],
        kvals=[2, 3],
    )
    assert nonsquare["error"] == short_covar["error"]
    for distance, covar in (
        (np.ones((11, 12)), CLUSTASSOC_NUMERIC),
        (CLUSTASSOC_D, CLUSTASSOC_NUMERIC[:-1]),
    ):
        with pytest.raises(ValueError, match=re.escape(nonsquare["error"])):
            clustassoc(_clustassoc_range(), distance, covar)

    all_missing = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        columns,
        np.full(CLUSTASSOC_D.shape[0], np.nan),
        kvals=[2, 3],
    )
    assert all_missing["error"] == "subscript out of bounds"
    with pytest.raises(ValueError, match="subscript out of bounds"):
        clustassoc(
            _clustassoc_range(),
            CLUSTASSOC_D,
            np.full(CLUSTASSOC_D.shape[0], np.nan),
        )

    one_level = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        columns,
        np.repeat("a", CLUSTASSOC_D.shape[0]),
        kvals=[2, 3],
        categorical=True,
    )
    assert one_level["error"] == ("contrasts can be applied only to factors with 2 or more levels")
    with pytest.raises(ValueError, match="contrasts can be applied"):
        clustassoc(
            _clustassoc_range(),
            CLUSTASSOC_D,
            np.repeat("a", CLUSTASSOC_D.shape[0]),
        )


def test_clustassoc_weight_boundaries_match_live_r(r_parity_session):
    columns = {
        "cluster2": CLUSTASSOC_CLUSTER2,
        "cluster3": CLUSTASSOC_CLUSTER3,
    }
    for weights in (
        np.repeat(1.0, CLUSTASSOC_D.shape[0] - 1),
        np.r_[np.nan, np.ones(CLUSTASSOC_D.shape[0] - 1)],
        np.r_[np.inf, np.ones(CLUSTASSOC_D.shape[0] - 1)],
    ):
        expected = r_parity_session.clustassoc_complete_contract(
            CLUSTASSOC_D,
            columns,
            CLUSTASSOC_NUMERIC,
            kvals=[2, 3],
            weights=weights,
        )
        assert expected["error"]
        with pytest.raises(ValueError, match=re.escape(expected["error"])):
            clustassoc(
                _clustassoc_range(),
                CLUSTASSOC_D,
                CLUSTASSOC_NUMERIC,
                weights=weights,
            )

    negative = np.r_[-1.0, np.ones(CLUSTASSOC_D.shape[0] - 1)]
    negative_expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        columns,
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        weights=negative,
    )
    assert negative_expected["warnings"] == ["NaNs produced"]
    with pytest.warns(RuntimeWarning, match="NaNs produced"):
        with pytest.raises(ValueError, match=re.escape(negative_expected["error"])):
            clustassoc(
                _clustassoc_range(),
                CLUSTASSOC_D,
                CLUSTASSOC_NUMERIC,
                weights=negative,
            )

    zero = np.zeros(CLUSTASSOC_D.shape[0])
    zero_expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        columns,
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        weights=zero,
    )
    assert zero_expected["error"] is None
    zero_actual = clustassoc(
        _clustassoc_range(),
        CLUSTASSOC_D,
        CLUSTASSOC_NUMERIC,
        weights=zero,
    )
    np.testing.assert_allclose(
        np.column_stack([zero_actual[name] for name in zero_actual]),
        zero_expected["table"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    categorical_zero_expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        columns,
        CLUSTASSOC_CATEGORICAL,
        kvals=[2, 3],
        categorical=True,
        weights=zero,
    )
    categorical_zero_actual = clustassoc(
        _clustassoc_range(),
        CLUSTASSOC_D,
        CLUSTASSOC_CATEGORICAL,
        weights=zero,
    )
    np.testing.assert_allclose(
        np.column_stack([categorical_zero_actual[name] for name in categorical_zero_actual]),
        categorical_zero_expected["table"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_clustassoc_separation_and_convergence_match_live_r(r_parity_session):
    distance = np.ones((6, 6), dtype=float)
    np.fill_diagonal(distance, 0.0)
    distance[:3, :3] = 0.1
    distance[3:, 3:] = 0.1
    np.fill_diagonal(distance, 0.0)
    columns = {
        "cluster2": np.array([1, 1, 1, 2, 2, 2]),
        "cluster3": np.array([1, 2, 3, 1, 2, 3]),
    }
    covar = np.array(["a", "a", "a", "b", "b", "b"])
    expected = r_parity_session.clustassoc_complete_contract(
        distance,
        columns,
        covar,
        kvals=[2, 3],
        categorical=True,
    )
    actual = clustassoc(
        {
            "kvals": [2, 3],
            "clustering": columns,
            "stats": {"PBC": [0.0, 0.0], "HG": [0.0, 0.0], "HGSD": [0.0, 0.0]},
        },
        distance,
        covar,
    )
    # cluster2 is collinear with covar: its covariate term adds no rank to the design.
    # R's dissmfacw then gives a finite value of an arbitrary QR direction, which R's
    # clustassoc reports; traminex gives NaN for such a term (3a3a5149, a DIVERGENCE of its
    # ledger), and so does this port, a deliberate departure from WeightedCluster
    # (docs/R_MIGRATION_GUIDE.md; owner's request of 2026-11-07, recorded option (a)).
    observed = np.column_stack([actual[name] for name in actual])[:, :2]
    assert np.all(np.isfinite(expected["table"][1, :2]))
    assert np.all(np.isnan(observed[1]))
    np.testing.assert_allclose(
        observed[[0, 2]],
        expected["table"][[0, 2], :2],
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual["BIC"],
        expected["table"][:, 2],
        rtol=0.0,
        atol=2e-4,
    )
    assert actual.additions["BICConverged"] == [True, True, True]


@pytest.mark.parametrize(
    ("stat", "plot_type"),
    [
        ("Unaccounted", "b"),
        ("Remaining", "l"),
        ("BIC", "p"),
    ],
)
def test_plot_clustassoc_semantics_options_axes_and_agg_match_live_r(
    r_parity_session,
    stat,
    plot_type,
):
    from weightedcluster import plot_clustassoc
    from weightedcluster.clustassoc import _clustassoc_plot_data

    expected = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        plot_stat=[stat, "BIC"],
        plot_type=plot_type,
        plot_options={
            "main": "Pinned association",
            "col": "red",
            "lwd": 2,
            "pch": 3,
            "xlim": [0.5, 3.5],
        },
    )
    assert expected["plot"]["error"] is None
    actual = clustassoc(_clustassoc_range(), CLUSTASSOC_D, CLUSTASSOC_NUMERIC)
    semantic = _clustassoc_plot_data(
        actual,
        stat=[stat, "BIC"],
        type=plot_type,
    )
    assert semantic["stat"] == expected["plot"]["stat"]
    assert semantic["xlab"] == expected["plot"]["xlab"]
    if expected["plot"]["ylab"] is None:
        # F-09 deliberately repairs WeightedCluster 2.0's empty label caused
        # by indexing the title table with obsolete internal statistic names.
        expected_ylabel = {
            "Unaccounted": "Unreproduced proportion of the association",
            "Remaining": "Unexplained association",
        }[stat]
    else:
        expected_ylabel = expected["plot"]["ylab"]
    assert semantic["ylab"] == expected_ylabel
    np.testing.assert_array_equal(semantic["x"], expected["plot"]["x"])
    np.testing.assert_allclose(
        semantic["y"],
        expected["plot"]["y"],
        rtol=0.0,
        atol=1e-10,
    )

    import matplotlib.pyplot as plt

    figure, supplied_axis = plt.subplots()
    returned_figure, returned_axis = plot_clustassoc(
        actual,
        stat=[stat, "BIC"],
        type=plot_type,
        ax=supplied_axis,
        tooltip=False,
        main="Pinned association",
        col="red",
        lwd=2,
        pch=3,
        xlim=[0.5, 3.5],
    )
    assert returned_figure is figure
    assert returned_axis is supplied_axis
    assert returned_axis.get_xlabel() == expected["plot"]["xlab"]
    assert returned_axis.get_ylabel() == expected_ylabel
    assert returned_axis.get_title() == "Pinned association"
    np.testing.assert_allclose(returned_axis.get_xlim(), [0.5, 3.5])
    figure.canvas.draw()
    plt.close(figure)


def test_plot_clustassoc_every_type_and_source_error_boundary(r_parity_session):
    from weightedcluster import plot_clustassoc

    actual = clustassoc(_clustassoc_range(), CLUSTASSOC_D, CLUSTASSOC_NUMERIC)
    import matplotlib.pyplot as plt

    for plot_type in ("b", "l", "p", "o", "h", "s", "S", "n"):
        figure, axis = plot_clustassoc(
            actual,
            type=plot_type,
            tooltip=False,
        )
        assert axis.figure is figure
        figure.canvas.draw()
        plt.close(figure)

    invalid_stat = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        plot_stat="not-a-stat",
    )
    assert invalid_stat["plot"]["error"] == "undefined columns selected"
    with pytest.raises(ValueError, match="undefined columns selected"):
        plot_clustassoc(actual, stat="not-a-stat", tooltip=False)

    invalid_type = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        plot_stat="BIC",
        plot_type="z",
    )
    assert invalid_type["plot"]["error"]
    with pytest.raises(ValueError, match=re.escape(invalid_type["plot"]["error"])):
        plot_clustassoc(actual, stat="BIC", type="z", tooltip=False)

    truncated = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        plot_stat="BIC",
        plot_type="bogus",
    )
    with pytest.warns(RuntimeWarning, match="will be truncated"):
        figure, _ = plot_clustassoc(
            actual,
            stat="BIC",
            type="bogus",
            tooltip=False,
        )
    assert truncated["plot"]["warnings"] == [
        "plot type 'bogus' will be truncated to first character"
    ]
    plt.close(figure)

    duplicate_label = r_parity_session.clustassoc_complete_contract(
        CLUSTASSOC_D,
        {
            "cluster2": CLUSTASSOC_CLUSTER2,
            "cluster3": CLUSTASSOC_CLUSTER3,
        },
        CLUSTASSOC_NUMERIC,
        kvals=[2, 3],
        plot_stat="BIC",
        plot_options={"xlab": "duplicate"},
    )
    assert duplicate_label["plot"]["error"]
    with pytest.raises(TypeError, match=re.escape(duplicate_label["plot"]["error"])):
        plot_clustassoc(
            actual,
            stat="BIC",
            tooltip=False,
            xlab="duplicate",
        )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        plot_clustassoc(
            actual,
            stat="BIC",
            tooltip=False,
            not_a_graphics_parameter=True,
        )

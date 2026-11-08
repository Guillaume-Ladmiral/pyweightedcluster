from __future__ import annotations

import numpy as np
import pytest

from weightedcluster._boot_inputs import _prepare_bootstrap_inputs
from weightedcluster._seqdist import (
    normalize_seqdist_config,
    sequences_distance,
    sequences_distance_between,
)
from weightedcluster.boot import _boot_distance_matrix

X = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]], dtype=int)
SM = np.array([[0.0, 4.0], [4.0, 0.0]])


def test_ham_honors_substitution_costs_within_and_between_sets():
    ordinary = sequences_distance(X, metric="HAM", normalize=False)
    weighted = sequences_distance(X, metric="HAM", normalize=False, substitution_costs=SM)
    np.testing.assert_allclose(weighted, 4.0 * ordinary)
    between = sequences_distance_between(
        X[:2], X[2:], metric="HAM", normalize=False, substitution_costs=SM
    )
    np.testing.assert_allclose(between, weighted[:2, 2:])


@pytest.mark.parametrize(
    ("config", "expected_method"),
    [
        ({"method": "HAM", "metric": "hamming"}, "HAM"),
        ({"method": "OM", "sm": SM, "substitution_costs": SM.copy()}, "OM"),
        ({"method": "OMSPELL", "timecost": 0.5, "time_cost": 0.5}, "OMSPELL"),
        ({"method": "LCS", "norm": "maxlength"}, "LCS"),
        ({"method": "HAM", "normalize": True}, "HAM"),
    ],
)
def test_normalizer_accepts_matching_aliases(config, expected_method):
    assert normalize_seqdist_config(config)["method"] == expected_method


@pytest.mark.parametrize(
    "config",
    [
        {"method": "HAM", "metric": "OM"},
        {"method": "OM", "sm": 2.0, "substitution_costs": 3.0},
        {"method": "OMSPELL", "timecost": 0.5, "time_cost": 0.7},
        {"method": "LCS", "norm": "none", "normalize": True},
        {"method": "HAM", "unknown": 1},
    ],
)
def test_normalizer_rejects_conflicting_aliases_and_unknown_keys(config):
    with pytest.raises(ValueError, match="seqdist_args"):
        normalize_seqdist_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"method": "HAM", "indel": 1.0},
        {"method": "LCS", "sm": SM},
        {"method": "OM", "timecost": 0.5},
        {"method": "DHD", "time_cost": 0.5},
    ],
)
def test_normalizer_rejects_method_specific_parameters_that_would_be_ignored(config):
    with pytest.raises(ValueError, match="not supported"):
        normalize_seqdist_config(config)


def test_bootstrap_uses_shared_alias_contract_and_keeps_lcs_default():
    clustering = {"cluster2": np.array([1, 1, 2])}
    configured = _prepare_bootstrap_inputs(clustering, X, seqdist_args=None, R=1, sample_size=3)
    assert configured.seqdist_args["method"] == "LCS"
    with pytest.raises(ValueError, match="conflicting aliases"):
        _prepare_bootstrap_inputs(
            clustering,
            X,
            seqdist_args={"method": "HAM", "metric": "OM"},
            R=1,
            sample_size=3,
        )
    with pytest.raises(ValueError, match="unsupported keys"):
        _prepare_bootstrap_inputs(
            clustering,
            X,
            seqdist_args={"method": "LCS", "metod": "HAM"},
            R=1,
            sample_size=3,
        )


def test_bootstrap_distance_forwards_canonical_ham_substitution_matrix():
    actual = _boot_distance_matrix(
        X,
        seqdist_args={"method": "HAM", "substitution_costs": SM, "norm": "none"},
    )
    expected = sequences_distance(X, metric="HAM", substitution_costs=SM, normalize=False)
    np.testing.assert_allclose(actual, expected)


def test_om_family_delegations_require_a_substitution_cost_like_r_seqdist():
    """traminex ``6e76440b`` (re-pinned 2026-09-06) makes ``seqdist`` stop on a
    missing substitution cost for the OM family, as R's ``seqdist`` does
    (``msg.stop.miss("sm")``); only HAM, DHD, and LCS auto-generate costs. The
    two delegations that forward ``seqdist_args`` to public ``traminex.seqdist``
    therefore inherit R's refusal, and an R-style ``sm`` unblocks them. The
    low-level ``sequences_distance`` adapter keeps this project's documented
    ``subm=2`` OM default and is not affected.
    """
    from weightedcluster.seqnull import _seqnull_distance_matrix

    for delegate in (_boot_distance_matrix, _seqnull_distance_matrix):
        with pytest.raises(ValueError, match="subm must be specified for method OM"):
            delegate(X, seqdist_args={"method": "OM", "indel": 1.0})
        with_cost = delegate(X, seqdist_args={"method": "OM", "indel": 1.0, "sm": 2})
        expected = sequences_distance(X, metric="OM", indel=1.0, normalize=False)
        np.testing.assert_allclose(with_cost, expected)


def test_live_r_bootclustrange_deliberately_ignores_attached_stslist_weights(
    r_parity_session,
):
    sequences = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    clusters = {
        "cluster2": np.array([1, 1, 1, 2, 2, 2]),
        "cluster3": np.array([1, 1, 2, 2, 3, 3]),
    }
    samples = np.array([[1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]])
    common = dict(
        X=sequences,
        cluster_columns=clusters,
        replay_samples_1b=samples,
        seqdist_method="LCS",
        seqdist_norm="none",
        seed=20260812,
    )
    ordinary = r_parity_session.bootclustrange_complete_contract(**common)
    attached = r_parity_session.bootclustrange_complete_contract(
        **common,
        weights=np.array([1000.0, 0.01, 50.0, 0.2, 700.0, 0.03]),
    )
    assert ordinary["error"] is None
    assert attached["error"] is None
    np.testing.assert_allclose(attached["stats"], ordinary["stats"], equal_nan=True)
    for weighted_block, ordinary_block in zip(attached["boot"], ordinary["boot"], strict=True):
        np.testing.assert_allclose(weighted_block, ordinary_block, equal_nan=True)

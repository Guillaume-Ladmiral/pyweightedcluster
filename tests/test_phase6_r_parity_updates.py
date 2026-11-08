from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import bootclustrange, seqclararange, seqnull, seqnullcqi, wcKMedoids
from weightedcluster._seqdist import sequences_distance
from weightedcluster._utils import generate_random_seed
from weightedcluster.schemas import ClustrangeResult


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


def test_seqclararange_default_seqdist_method_is_lcs(monkeypatch):
    import importlib

    scmod = importlib.import_module("weightedcluster.seqclararange")

    seen_metrics: list[str] = []
    real = sequences_distance

    def _spy(*args, **kwargs):
        seen_metrics.append(str(kwargs.get("metric")))
        return real(*args, **kwargs)

    monkeypatch.setattr(scmod, "sequences_distance", _spy)
    X = _toy_sequences()
    seqclararange(X, R=1, sample_size=4, kvals=[2], random_state=17)
    assert seen_metrics
    assert str(seen_metrics[0]).lower() == "lcs"


def test_seqclararange_avoids_direct_internal_index_recompute(monkeypatch):
    import importlib

    scmod = importlib.import_module("weightedcluster.seqclararange")
    calls = {"n": 0}

    def _forbidden(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("seqclararange should not directly call compute_internal_indices")

    monkeypatch.setattr(scmod, "compute_internal_indices", _forbidden, raising=False)

    X = _toy_sequences()
    out = seqclararange(
        X,
        R=1,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=77,
    )
    assert out["kvals"] == [2]
    assert calls["n"] == 0


def test_bootclustrange_default_seqdist_method_is_lcs(monkeypatch):
    import weightedcluster.boot as bmod

    seen_metrics: list[str] = []
    real = bmod.traminex_seqdist

    def _spy(*args, **kwargs):
        seen_metrics.append(str(kwargs.get("method")))
        return real(*args, **kwargs)

    monkeypatch.setattr(bmod, "traminex_seqdist", _spy)
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    cr = {"kvals": [2], "clustering": {"cluster2": labels}}
    bootclustrange(cr, X, R=1, sample_size=4, sampling="random", random_state=18)
    assert seen_metrics
    assert str(seen_metrics[0]).lower() == "lcs"


def test_seqnull_combined_respects_impossible_transition_map():
    X = np.array(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 2, 2, 1, 1, 0],
            [2, 2, 0, 0, 1, 1],
            [1, 1, 2, 2, 0, 0],
        ],
        dtype=int,
    )
    Y = seqnull(X, model="combined", imp_trans={0: [1]}, random_state=19)

    spell_01 = 0
    for row in np.asarray(Y, dtype=int):
        spells = [int(row[0])]
        for v in row[1:]:
            if int(v) != spells[-1]:
                spells.append(int(v))
        for a, b in zip(spells[:-1], spells[1:], strict=False):
            if a == 0 and b == 1:
                spell_01 += 1
    assert spell_01 == 0


def test_seqnull_userpos_accepts_time_varying_transition_array():
    X = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )
    prob = np.zeros((2, 2, 3), dtype=float)
    prob[:, :, 0] = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    prob[:, :, 1] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    prob[:, :, 2] = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)

    Y = seqnull(
        X,
        model="userpos",
        trate=prob,
        begin=np.array([1.0, 0.0]),
        time_varying=True,
        random_state=20,
    )
    assert np.all(np.asarray(Y)[:, 0] == 0)
    assert np.all(np.asarray(Y)[:, 1] == 1)
    assert np.all(np.asarray(Y)[:, 2] == 0)
    assert np.all(np.asarray(Y)[:, 3] == 1)


def test_seqnullcqi_installed_method_always_uses_ward_d(monkeypatch):
    import weightedcluster._fastcluster_linkage as linkage_module

    seen_methods: list[str] = []
    real_linkage = linkage_module.linkage_fastcluster

    def _spy_linkage(*args, **kwargs):
        seen_methods.append(str(kwargs["method"]))
        return real_linkage(*args, **kwargs)

    monkeypatch.setattr(linkage_module, "linkage_fastcluster", _spy_linkage)

    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    clustrange = ClustrangeResult({"kvals": [2], "clustering": {"cluster2": labels}})

    seqnullcqi(
        X,
        clustrange,
        R=1,
        model="combined",
        seqdist_args={"method": "HAM"},
        hclust_method="single",
        random_state=21,
    )
    assert seen_methods
    assert seen_methods[0] == "ward.d"


def test_seqnullcqi_kmedoid_propagates_bootstrap_seed_to_kmedrange(monkeypatch):
    import importlib

    snmod = importlib.import_module("weightedcluster.seqnull")

    seen_random_states: list[int] = []
    real_kmedrange = snmod.wcKMedRange

    def _spy_kmedrange(*args, **kwargs):
        seen_random_states.append(int(kwargs.get("random_state")))
        return real_kmedrange(*args, **kwargs)

    monkeypatch.setattr(snmod, "wcKMedRange", _spy_kmedrange)

    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    clustrange = ClustrangeResult({"kvals": [2], "clustering": {"cluster2": labels}})

    seqnullcqi(
        X,
        clustrange,
        R=3,
        model="combined",
        seqdist_args={"method": "HAM"},
        kmedoid=True,
        random_state=50,
    )

    base_seed = generate_random_seed(50, 0)
    assert seen_random_states == [generate_random_seed(base_seed, 2, index) for index in range(3)]


def test_seqnullcqi_hierarchical_failure_does_not_fall_back_to_kmedrange(monkeypatch):
    import importlib

    import weightedcluster._fastcluster_linkage as linkage_module

    snmod = importlib.import_module("weightedcluster.seqnull")

    def _boom(*args, **kwargs):
        raise RuntimeError("linkage exploded")

    def _unexpected_kmedrange(*args, **kwargs):
        raise AssertionError("wcKMedRange fallback should not be used")

    monkeypatch.setattr(linkage_module, "linkage_fastcluster", _boom)
    monkeypatch.setattr(snmod, "wcKMedRange", _unexpected_kmedrange)

    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    clustrange = ClustrangeResult({"kvals": [2], "clustering": {"cluster2": labels}})

    with pytest.raises(RuntimeError, match="linkage exploded"):
        seqnullcqi(
            X,
            clustrange,
            R=1,
            model="combined",
            seqdist_args={"method": "HAM"},
            hclust_method="ward.D",
            random_state=53,
        )


def test_execute_seqnull_bootstraps_does_not_retry_parallel_failures():
    import importlib

    snmod = importlib.import_module("weightedcluster.seqnull")
    calls: list[int] = []

    def _boom(seed: int):
        calls.append(int(seed))
        raise RuntimeError("worker exploded")

    with pytest.raises(RuntimeError, match="worker exploded"):
        snmod._execute_seqnull_bootstraps(
            [7],
            _boom,
            parallel=True,
            progressbar=False,
        )

    assert calls == [7]

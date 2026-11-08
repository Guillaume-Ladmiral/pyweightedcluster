from __future__ import annotations

import sys
import types

import numpy as np

from weightedcluster import seqclararange, seqclararange_family, seqnull, wcCmpCluster
from weightedcluster._utils import generate_random_seed


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


def test_seqclararange_accepts_string_criteria_without_family_routing():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=11,
    )
    assert "clustering" in out and "stats" in out and "clara" in out
    assert "param" not in out
    assert "cluster2" in out["clustering"]


def test_seqclararange_parallel_runs_without_nameerror():
    X = _toy_sequences()
    out = seqclararange(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        parallel=True,
        random_state=12,
    )
    assert out["kvals"] == [2]
    assert "cluster2" in out["clustering"]


def test_seqclararange_family_accepts_single_string_criterion():
    X = _toy_sequences()
    fam = seqclararange_family(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=13,
    )
    assert "distance" in fam
    assert "d" not in fam
    assert fam["distance"]["kvals"] == [2]


def test_seqnull_numba_paths_receive_deterministic_seed(monkeypatch):
    calls: dict[str, list[int] | int] = {}

    fake_numba = types.ModuleType("weightedcluster._numba")

    def _fake_probs(
        freqs_t: np.ndarray,
        p_glob: np.ndarray,
        time_varying: bool,
        N: int,
        T: int,
        seed: int,
    ) -> np.ndarray:
        calls["stateindep"] = int(seed)
        return np.zeros((N, T), dtype=np.int32)

    def _fake_markov(
        trans: np.ndarray,
        p0: np.ndarray,
        N: int,
        T: int,
        seed: int,
    ) -> np.ndarray:
        calls.setdefault("markov", [])
        assert isinstance(calls["markov"], list)
        calls["markov"].append(int(seed))
        return np.zeros((N, T), dtype=np.int32)

    fake_numba.seqnull_random_from_probs_numba = _fake_probs
    fake_numba.seqnull_markov_numba = _fake_markov
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", fake_numba)

    X = _toy_sequences()
    seqnull(
        X, model="combined", random_state=11
    )  # spell-based models no longer use this numba path
    seqnull(X, model="stateindep", random_state=101)
    seqnull(X, model="markov", random_state=202)
    seqnull(X, model="userpos", trate="trate", time_varying=False, random_state=303)

    assert calls["stateindep"] == generate_random_seed(101, 0)
    assert calls["markov"] == [
        generate_random_seed(202, 0),
        generate_random_seed(303, 0),
    ]


def test_wccmpcluster_allstats_include_hierarchical_rows():
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1],
            [0.2, 0.0, 1.0, 1.1],
            [1.0, 1.0, 0.0, 0.2],
            [1.1, 1.1, 0.2, 0.0],
        ],
        dtype=float,
    )
    out = wcCmpCluster(D, maxcluster=3, method=["single", "pam"])
    methods = list(out["allstats"]["method"])
    assert methods.count("single") == 2
    assert methods.count("pam") == 2

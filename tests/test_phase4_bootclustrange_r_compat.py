from __future__ import annotations

import numpy as np
import pytest

import weightedcluster._boot_runtime as boot_runtime
from weightedcluster import bootclustrange, seqclararange
from weightedcluster._boot_runtime import bootstrap_worker_once, draw_stratified_sample
from weightedcluster.schemas import SeqClaraRangeResult


def _toy_sequences() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [2, 2, 2, 2],
            [2, 2, 2, 1],
            [2, 2, 1, 1],
            [2, 1, 1, 1],
        ],
        dtype=int,
    )


def _toy_clustrange_dict() -> dict:
    # 3 clear blocks for K=3, merged first two blocks for K=2
    cluster3 = np.array([1] * 4 + [2] * 4 + [3] * 4, dtype=int)
    cluster2 = np.array([1] * 8 + [2] * 4, dtype=int)
    return {"kvals": [2, 3], "clustering": {"cluster2": cluster2, "cluster3": cluster3}}


def test_bootclustrange_exposes_r_style_meant_stderr():
    X = _toy_sequences()
    cr = _toy_clustrange_dict()
    out = bootclustrange(
        cr,
        X,
        R=4,
        sample_size=9,
        seqdist_args={"method": "HAM"},
        sampling="random",
        random_state=31,
    )
    assert "meant" in out and "stderr" in out and "stats" in out
    assert "baseline_stats" not in out
    assert "baseline_stats" in out.additions
    assert out["stats"] == out["meant"]
    for key in out["meant"].keys():
        assert len(out["meant"][key]) == len(cr["kvals"])
        assert len(out["stderr"][key]) == len(cr["kvals"])


def test_bootclustrange_supports_explicit_strata_sampling():
    X = _toy_sequences()
    cr = _toy_clustrange_dict()
    strata = np.array([1] * 4 + [2] * 4 + [3] * 4, dtype=int)
    out = bootclustrange(
        cr,
        X,
        R=3,
        sample_size=9,
        seqdist_args={"method": "HAM"},
        sampling="strata",
        strata=strata,
        random_state=32,
    )
    assert out.additions["sampling"] == "strata"
    assert set(out["boot"].keys()) == {2, 3}


def test_bootclustrange_ignores_off_table_clustrange_distance_metadata():
    X = _toy_sequences()
    cr = _toy_clustrange_dict()
    cr["diss"] = np.zeros((X.shape[0], X.shape[0]), dtype=float)

    out = bootclustrange(
        cr,
        X,
        R=1,
        sample_size=9,
        seqdist_args={"method": "HAM"},
        sampling="random",
        random_state=320,
    )
    assert list(out) == ["clustering", "kvals", "meant", "stderr", "boot", "stats"]


def test_bootclustrange_clustering_sampling_checks_minimum_size():
    X = _toy_sequences()
    cr = _toy_clustrange_dict()
    with pytest.raises(ValueError, match="sample size is to small"):
        bootclustrange(
            cr,
            X,
            R=2,
            sample_size=3,  # too small for 3 balanced strata with R-style >=2 constraint
            seqdist_args={"method": "HAM"},
            sampling="clustering",
            random_state=33,
        )


def test_bootclustrange_medoids_sampling_uses_seqclararange_structure():
    X = _toy_sequences()
    sc = seqclararange(
        X,
        R=2,
        sample_size=8,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=34,
    )
    out = bootclustrange(
        sc,
        X,
        R=3,
        sample_size=8,
        seqdist_args={"method": "HAM"},
        sampling="medoids",
        random_state=35,
    )
    assert out.additions["sampling"] == "medoids"
    assert 2 in out["boot"]


def test_bootclustrange_medoids_sampling_rejects_malformed_clara_entry():
    X = _toy_sequences()
    cr = SeqClaraRangeResult(
        {
            **_toy_clustrange_dict(),
            "stats": {},
            "clara": {2: []},
        }
    )

    with pytest.raises(ValueError, match="final clara result"):
        bootclustrange(
            cr,
            X,
            R=1,
            sample_size=9,
            seqdist_args={"method": "HAM"},
            sampling="medoids",
            random_state=351,
        )


def test_bootclustrange_rejects_impossible_cluster_coverage():
    X = _toy_sequences()
    cr = _toy_clustrange_dict()
    with pytest.raises(ValueError, match="preserve every cluster count"):
        bootclustrange(
            cr,
            X,
            R=1,
            sample_size=2,
            seqdist_args={"method": "HAM"},
            sampling="random",
            random_state=36,
        )


def test_bootclustrange_raises_when_retry_budget_cannot_find_valid_sample(monkeypatch):
    import weightedcluster.boot as bootmod

    X = _toy_sequences()
    cr = _toy_clustrange_dict()

    monkeypatch.setattr(bootmod, "_sample_has_all_clusters", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="Unable to draw a bootstrap subsample"):
        bootclustrange(
            cr,
            X,
            R=1,
            sample_size=9,
            seqdist_args={"method": "HAM"},
            sampling="random",
            random_state=37,
        )


def test_bootstrap_worker_reports_invalid_replay_without_unbound_local_error():
    context = {
        "seed": 1,
        "X": np.zeros((2, 1), dtype=int),
        "N": 2,
        "sample_size": 1,
        "sampling": "random",
        "strata_idx": None,
        "medoids_idx": None,
        "seqdist_args": {},
        "kvals": [2],
        "clustering": {"cluster2": np.array([1, 2])},
        "expected_unique": {2: np.array([1, 2])},
        "replay_samples": np.array([[0]], dtype=int),
    }

    with pytest.raises(RuntimeError, match="after 1 attempt"):
        bootstrap_worker_once(
            0,
            context,
            generate_random_seed_fn=lambda *_args: 1,
            sequence_distance_fn=lambda *_args, **_kwargs: pytest.fail(
                "invalid replay reached distance computation"
            ),
            as_clustrange_fn=lambda *_args, **_kwargs: pytest.fail(
                "invalid replay reached range computation"
            ),
            draw_bootstrap_sample_fn=lambda *_args, **_kwargs: pytest.fail(
                "replay path drew a random sample"
            ),
            sample_has_all_clusters_fn=lambda *_args, **_kwargs: False,
        )


def test_bootclustrange_clears_local_worker_context_when_runner_raises(monkeypatch):
    import weightedcluster.boot as bootmod

    initialized_context = None

    def failing_runner(_worker, _tasks, *, initializer, initargs, **_kwargs):
        nonlocal initialized_context
        initializer(*initargs)
        initialized_context = initargs[0]
        assert boot_runtime._BOOTSTRAP_WORKER_CONTEXT is initialized_context
        raise RuntimeError("runner failed after initialization")

    boot_runtime.clear_bootstrap_worker_context()
    context_after_failure = None
    try:
        monkeypatch.setattr(bootmod, "run_task_batch", failing_runner)
        with pytest.raises(RuntimeError, match="runner failed after initialization"):
            bootclustrange(
                _toy_clustrange_dict(),
                _toy_sequences(),
                R=1,
                sample_size=9,
                seqdist_args={"method": "HAM"},
                sampling="random",
                random_state=38,
                parallel=True,
                n_jobs=2,
            )
        context_after_failure = boot_runtime._BOOTSTRAP_WORKER_CONTEXT
    finally:
        boot_runtime.clear_bootstrap_worker_context()

    assert initialized_context is not None
    assert context_after_failure is None


def test_stratified_draw_rejects_negative_r_rounding_correction():
    strata = {
        0: np.array([0, 1, 2]),
        1: np.array([3, 4, 5]),
        2: np.array([6, 7, 8]),
        3: np.array([9]),
    }

    with pytest.raises(ValueError, match="negative 'size' argument"):
        draw_stratified_sample(
            rng=np.random.default_rng(0),
            strata_idx=strata,
            sample_size=2,
        )

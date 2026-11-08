from __future__ import annotations

import importlib

import numpy as np
import pytest

from weightedcluster import (
    bootclustrange,
    seqclararange,
    seqclararange_family,
    seqnullcqi,
    wcKMedoids,
)
from weightedcluster._parallel import resolve_n_jobs, run_task_batch
from weightedcluster._seqdist import sequences_distance
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


def _toy_clustrange(X: np.ndarray) -> dict:
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    return ClustrangeResult({"kvals": [2], "clustering": {"cluster2": labels}})


def _serial_runner_spy_factory(seen: dict[str, object]):
    def _spy(
        worker,
        tasks,
        *,
        parallel,
        n_jobs,
        progressbar,
        desc,
        executor_kind,
        initializer=None,
        initargs=(),
    ):
        items = list(tasks)
        seen["parallel"] = bool(parallel)
        seen["n_jobs"] = n_jobs
        seen["progressbar"] = bool(progressbar)
        seen["desc"] = str(desc)
        seen["executor_kind"] = str(executor_kind)
        seen["task_count"] = len(items)
        if initializer is not None:
            initializer(*tuple(initargs))
        return [worker(item) for item in items]

    return _spy


@pytest.mark.parametrize("invalid", (1.5, "2", True))
def test_resolve_n_jobs_rejects_lossy_or_non_integer_values(invalid):
    with pytest.raises(ValueError, match="n_jobs must be a positive integer"):
        resolve_n_jobs(invalid, n_tasks=3)


def test_shared_runner_validates_controls_before_serial_shortcuts():
    with pytest.raises(TypeError, match="parallel must be a boolean"):
        run_task_batch(
            lambda value: value,
            [1],
            parallel=1,
            n_jobs=1,
            progressbar=False,
            desc="invalid",
            executor_kind="thread",
        )
    with pytest.raises(TypeError, match="progressbar must be a boolean"):
        run_task_batch(
            lambda value: value,
            [1],
            parallel=False,
            n_jobs=1,
            progressbar="no",
            desc="invalid",
            executor_kind="thread",
        )
    with pytest.raises(ValueError, match="Unknown executor_kind"):
        run_task_batch(
            lambda value: value,
            [1],
            parallel=False,
            n_jobs=1,
            progressbar=False,
            desc="invalid",
            executor_kind="bogus",
        )


def test_resolve_n_jobs_rejects_invalid_task_count():
    with pytest.raises(ValueError, match="n_tasks must be a non-negative integer"):
        resolve_n_jobs(2, n_tasks=1.5)


def test_process_runner_uses_spawn_context(monkeypatch):
    parallel_module = importlib.import_module("weightedcluster._parallel")

    class ExecutorReached(Exception):
        pass

    def capture_executor(**kwargs):
        assert kwargs["mp_context"].get_start_method() == "spawn"
        raise ExecutorReached

    monkeypatch.setattr(parallel_module, "ProcessPoolExecutor", capture_executor)

    with pytest.raises(ExecutorReached):
        run_task_batch(
            abs,
            [-1, -2],
            parallel=True,
            n_jobs=2,
            progressbar=False,
            desc="spawn contract",
            executor_kind="process",
        )


def test_seqclararange_family_forwards_parallel_n_jobs_to_shared_runner(monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    seen: dict[str, object] = {}

    monkeypatch.setattr(scmod, "run_task_batch", _serial_runner_spy_factory(seen))

    X = _toy_sequences()
    out = seqclararange_family(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria=["distance", "db"],
        parallel=True,
        n_jobs=2,
        random_state=41,
    )

    assert "distance" in out and "db" in out
    assert seen["parallel"] is True
    assert seen["n_jobs"] == 2
    assert seen["executor_kind"] == "thread"
    assert seen["task_count"] == 2


def test_seqnullcqi_forwards_parallel_n_jobs_to_shared_runner(monkeypatch):
    snmod = importlib.import_module("weightedcluster.seqnull")
    seen: dict[str, object] = {}

    monkeypatch.setattr(snmod, "run_task_batch", _serial_runner_spy_factory(seen))

    X = _toy_sequences()
    clustrange = _toy_clustrange(X)
    out = seqnullcqi(
        X,
        clustrange,
        R=2,
        model="combined",
        seqdist_args={"method": "HAM"},
        parallel=True,
        n_jobs=2,
        random_state=42,
    )

    assert out["R"] == 2
    assert seen["parallel"] is True
    assert seen["n_jobs"] == 2
    assert seen["executor_kind"] == "thread"
    assert seen["task_count"] == 2


def test_bootclustrange_forwards_parallel_n_jobs_to_shared_runner(monkeypatch):
    bmod = importlib.import_module("weightedcluster.boot")
    seen: dict[str, object] = {}

    monkeypatch.setattr(bmod, "run_task_batch", _serial_runner_spy_factory(seen))

    X = _toy_sequences()
    clustrange = _toy_clustrange(X)
    out = bootclustrange(
        clustrange,
        X,
        R=2,
        sample_size=4,
        seqdist_args={"method": "HAM"},
        sampling="random",
        parallel=True,
        n_jobs=2,
        random_state=43,
    )

    assert out["kvals"] == [2]
    assert seen["parallel"] is True
    assert seen["n_jobs"] == 2
    assert seen["executor_kind"] == "process"
    assert seen["task_count"] == 2


@pytest.mark.parametrize(
    "runner",
    (
        lambda X, cr: seqclararange(
            X,
            R=1,
            sample_size=4,
            kvals=[2],
            seqdist_args={"method": "HAM"},
            criteria="distance",
            parallel=True,
            n_jobs=0,
            random_state=44,
        ),
        lambda X, cr: seqclararange_family(
            X,
            R=1,
            sample_size=4,
            kvals=[2],
            seqdist_args={"method": "HAM"},
            criteria="distance",
            parallel=True,
            n_jobs=0,
            random_state=45,
        ),
        lambda X, cr: seqnullcqi(
            X,
            cr,
            R=1,
            model="combined",
            seqdist_args={"method": "HAM"},
            parallel=True,
            n_jobs=0,
            random_state=46,
        ),
        lambda X, cr: bootclustrange(
            cr,
            X,
            R=1,
            sample_size=4,
            seqdist_args={"method": "HAM"},
            sampling="random",
            parallel=True,
            n_jobs=0,
            random_state=47,
        ),
    ),
)
def test_sequence_parallel_workflows_reject_invalid_n_jobs(runner):
    X = _toy_sequences()
    clustrange = _toy_clustrange(X)

    with pytest.raises(ValueError, match="n_jobs must be a positive integer"):
        runner(X, clustrange)

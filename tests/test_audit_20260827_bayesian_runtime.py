from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from weightedcluster import _bayesian_rarcat_conditional as b3
from weightedcluster import _bayesian_rarcat_mixture as b4
from weightedcluster import _bayesian_rarcat_partition as b2
from weightedcluster import _parallel


def _public_inputs() -> tuple[np.ndarray, dict[str, object]]:
    frame = pd.DataFrame(
        {
            "cluster": pd.Categorical(
                ["early", "early", "late", "late"],
                categories=["early", "late"],
                ordered=True,
            ),
            "x": [-1.0, -0.5, 0.5, 1.0],
        },
        index=pd.Index(["case-0", "case-1", "case-2", "case-3"], name="case_id"),
    )
    return np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float), {
        "formula": "cluster ~ 1 + x",
        "data": frame,
        "B": 4,
        "npass": 2,
        "partition_root_seed": 2026082701,
        "conditional_root_seed": 2026082702,
        "continuous_predictors": ("x",),
        "counterfactual_steps": {},
    }


@pytest.mark.parametrize(
    ("controls", "exception", "message"),
    [
        ({"parallel": 1}, TypeError, "parallel"),
        ({"parallel": "yes"}, TypeError, "parallel"),
        ({"n_jobs": True}, ValueError, "n_jobs"),
        ({"n_jobs": 1.5}, ValueError, "n_jobs"),
        ({"n_jobs": "2"}, ValueError, "n_jobs"),
        ({"n_jobs": 0}, ValueError, "n_jobs"),
        ({"n_jobs": -2}, ValueError, "n_jobs"),
    ],
)
def test_public_parallel_controls_fail_before_any_bayesian_stage(
    monkeypatch, controls, exception, message
):
    public = importlib.import_module("weightedcluster.bayesian")
    dissimilarity, arguments = _public_inputs()
    calls: list[str] = []

    def forbidden_b2(**kwargs):
        calls.append("B2")
        raise AssertionError("B2 ran before public control validation")

    def forbidden_b3(**kwargs):
        calls.append("B3")
        raise AssertionError("B3 ran before public control validation")

    def forbidden_b4(*args, **kwargs):
        calls.append("B4")
        raise AssertionError("B4 ran before public control validation")

    monkeypatch.setattr(b2, "build_partition_posterior", forbidden_b2)
    monkeypatch.setattr(b3, "fit_conditional_posteriors", forbidden_b3)
    monkeypatch.setattr(b4, "integrate_conditional_posteriors", forbidden_b4)

    with pytest.raises(exception, match=message):
        public.bayes_rarcat(dissimilarity, **arguments, **controls)

    assert calls == []


@pytest.mark.parametrize(
    ("parallel", "n_jobs", "n_tasks", "expected"),
    [
        (False, None, 1, 1),
        (False, 3, 2, 2),
        (True, 1, 4, 1),
        (True, 3, 2, 2),
    ],
)
def test_shared_parallel_validator_accepts_valid_serial_and_parallel_controls(
    parallel, n_jobs, n_tasks, expected
):
    assert _parallel.validate_parallel_controls(parallel, n_jobs, n_tasks=n_tasks) == expected


def test_public_valid_controls_preserve_stage_order_seed_identity_and_arguments(monkeypatch):
    public = importlib.import_module("weightedcluster.bayesian")
    dissimilarity, arguments = _public_inputs()
    partition = object()
    conditional = object()
    integrated = object()
    calls: list[tuple[str, object]] = []

    def fake_b2(**kwargs):
        calls.append(("B2", kwargs))
        return partition

    def fake_b3(**kwargs):
        calls.append(("B3", kwargs))
        return conditional

    def fake_b4(*args, **kwargs):
        calls.append(("B4", (args, kwargs)))
        return integrated

    monkeypatch.setattr(b2, "build_partition_posterior", fake_b2)
    monkeypatch.setattr(b3, "fit_conditional_posteriors", fake_b3)
    monkeypatch.setattr(b4, "integrate_conditional_posteriors", fake_b4)

    result = public.bayes_rarcat(
        dissimilarity,
        **arguments,
        parallel=True,
        n_jobs=2,
    )

    assert result is integrated
    assert [name for name, _ in calls] == ["B2", "B3", "B4"]
    assert calls[0][1]["root_seed"] == arguments["partition_root_seed"]
    assert calls[1][1]["root_seed"] == arguments["conditional_root_seed"]
    assert calls[2][1] == ((partition, conditional), {"parallel": True, "n_jobs": 2})


@pytest.mark.parametrize(
    ("parallel", "n_jobs", "exception"),
    [
        (1, 1, TypeError),
        (False, True, ValueError),
        (False, 0, ValueError),
        (False, "2", ValueError),
    ],
)
def test_empty_task_batches_still_validate_parallel_controls(parallel, n_jobs, exception):
    with pytest.raises(exception, match="parallel|n_jobs"):
        _parallel.run_task_batch(
            lambda value: value,
            [],
            parallel=parallel,
            n_jobs=n_jobs,
            progressbar=False,
            desc="empty validation",
            executor_kind="thread",
        )

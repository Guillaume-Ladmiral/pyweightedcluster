from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from weightedcluster._threads import limit_threads
from weightedcluster.schemas import (
    BootClustrangeResult,
    ClustassocResult,
    ClustrangeFamilyParam,
    SeqNullCQIResult,
)


def _result_cases():
    return (
        ClustrangeFamilyParam({"r_field": 1}, additions={"diagnostic": "family"}),
        BootClustrangeResult({"r_field": 2}, additions={"diagnostic": "boot"}),
        ClustassocResult(
            {"r_field": 3},
            row_names=["row"],
            additions={"diagnostic": "association"},
        ),
        SeqNullCQIResult({"r_field": 4}, additions={"diagnostic": "null"}),
    )


@pytest.mark.parametrize("result", _result_cases())
def test_python_additions_are_not_virtual_mapping_keys(result):
    assert list(result) == ["r_field"]
    assert dict(result) == {"r_field": result["r_field"]}
    assert "diagnostic" not in result
    assert result.get("diagnostic") is None
    with pytest.raises(KeyError, match="diagnostic"):
        result["diagnostic"]
    assert result.additions == {"diagnostic": result.additions["diagnostic"]}


@pytest.mark.parametrize("result", _result_cases())
def test_result_copy_and_json_round_trip_keep_mapping_and_metadata_contracts_separate(result):
    copied = result.copy()

    assert type(copied) is type(result)
    assert dict(copied) == dict(result)
    assert copied.additions == result.additions
    assert copied.additions is not result.additions
    copied.additions["copy_only"] = True
    assert "copy_only" not in result.additions
    assert json.loads(json.dumps(result)) == dict(result)
    assert json.loads(json.dumps(result.additions)) == result.additions
    if isinstance(result, ClustassocResult):
        assert copied.row_names == result.row_names
        assert copied.row_names is not result.row_names


def _pool_threads() -> dict[tuple[str | None, str | None], int | None]:
    return {
        (pool.get("user_api"), pool.get("prefix")): pool.get("num_threads")
        for pool in threadpool_info()
    }


def test_thread_limits_reconfigure_loaded_pools_and_restore_nested_exceptional_scopes():
    np.ones((4, 4)) @ np.ones((4, 4))
    before = _pool_threads()
    if not before:
        pytest.skip("no native threadpool is loaded in this environment")

    with limit_threads(blas=2, omp=2):
        outer = _pool_threads()
        assert all(value == 2 for value in outer.values())
        with pytest.raises(RuntimeError, match="sentinel"):
            with limit_threads(blas=1, omp=1):
                inner = _pool_threads()
                assert all(value == 1 for value in inner.values())
                raise RuntimeError("sentinel")
        assert _pool_threads() == outer

    assert _pool_threads() == before


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_thread_limits_reject_non_positive_integer_controls(value):
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        with limit_threads(blas=value):
            pass


def test_thread_limit_environment_is_restored_after_controller_scope(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("BLAS_NUM_THREADS", raising=False)

    with limit_threads(omp=1, blas=1):
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["BLAS_NUM_THREADS"] == "1"

    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "BLAS_NUM_THREADS" not in os.environ


def test_dead_private_backends_are_absent_from_modules_and_registry():
    from weightedcluster import _fallback, _numba, _wc_core
    from weightedcluster._backend_contract import CAPABILITY_REGISTRY

    assert "weighted_silhouette_kernel" not in CAPABILITY_REGISTRY
    assert not hasattr(_fallback, "silhouette_weighted")
    assert not hasattr(_numba, "silhouette_weighted_numba")
    assert not hasattr(_numba, "weighted_medoids_numba")
    assert not hasattr(_wc_core, "silhouette_weighted")
    assert [
        item["backend"] for item in CAPABILITY_REGISTRY["weighted_medoids"]["implementations"]
    ] == ["compiled", "pure_python"]


@pytest.mark.parametrize("optimized", [False, True], ids=["ordinary", "python-O"])
def test_linkage_cut_invariant_is_explicit_under_optimized_python(optimized):
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend(
        [
            "-c",
            """
import inspect
import numpy as np
from weightedcluster._validation import cut_linkage_by_merge_count

source = inspect.getsource(cut_linkage_by_merge_count)
assert 'assert cut_members is not None' not in source
assert 'if cut_members is None:' in source
labels = cut_linkage_by_merge_count(
    np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]]),
    n_clusters=2,
)
assert labels.tolist() == [1, 1, 2]
""",
        ]
    )

    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_validation_module_contains_no_production_assert_statements():
    import weightedcluster._validation as validation

    assert not any(
        line.lstrip().startswith("assert ") for line in inspect.getsource(validation).splitlines()
    )

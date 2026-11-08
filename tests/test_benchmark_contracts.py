from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from weightedcluster._threads import limit_threads

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "benchmarks" / "bench_backend.py"
RECORDED_REPORT = REPO_ROOT / "benchmarks" / "results" / "phase4-reference-ubuntu-x86_64.json"


def _load_harness():
    spec = importlib.util.spec_from_file_location("phase4_benchmark_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load benchmark harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_module_import_is_scientific_stack_safe_in_parent_process():
    script = f"""
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location('bench_backend_import_probe', {str(HARNESS_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({{
    'numpy': 'numpy' in sys.modules,
    'numba': 'numba' in sys.modules,
    'weightedcluster': 'weightedcluster' in sys.modules,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert json.loads(completed.stdout) == {
        "numpy": False,
        "numba": False,
        "weightedcluster": False,
    }


def test_direct_kernel_registry_is_explicit_and_excludes_foreign_work():
    harness = _load_harness()

    assert set(harness.DIRECT_KERNELS) == {"weighted_medoids", "cluster_quality"}
    assert harness.DIRECT_KERNELS["weighted_medoids"] == {
        "compiled": "weightedcluster._wc_core.weighted_medoids",
        "pure_python": "weightedcluster._fallback.weighted_medoids",
    }
    assert harness.DIRECT_KERNELS["cluster_quality"] == {
        "compiled": "weightedcluster._wc_core.cluster_quality",
        "numba": "weightedcluster._numba.cluster_quality_numba",
        "pure_python": "weightedcluster._fallback.cluster_quality",
    }
    registry_text = json.dumps(harness.DIRECT_KERNELS, sort_keys=True).lower()
    assert "distance" not in registry_text
    assert "wcsilhouetteobs" not in registry_text


def test_direct_kernel_resolution_never_substitutes_a_different_backend():
    harness = _load_harness()
    called_modules: list[str] = []

    def _compiled(*_args: object, **_kwargs: object) -> None:
        return None

    fake_core = types.SimpleNamespace(weighted_medoids=_compiled)

    def _import_module(name: str) -> object:
        called_modules.append(name)
        if name == "weightedcluster._wc_core":
            return fake_core
        raise AssertionError(f"unexpected fallback import: {name}")

    resolved = harness.resolve_direct_kernel(
        "weighted_medoids", "compiled", import_module=_import_module
    )
    assert resolved["status"] == "AVAILABLE"
    assert resolved["symbol"] == "weightedcluster._wc_core.weighted_medoids"
    assert resolved["kernel"] is _compiled
    assert called_modules == ["weightedcluster._wc_core"]

    def _missing(_name: str) -> object:
        raise ImportError("compiled core deliberately absent")

    unavailable = harness.resolve_direct_kernel(
        "weighted_medoids", "compiled", import_module=_missing
    )
    assert unavailable["status"] == "SKIP"
    assert unavailable["kernel"] is None
    assert unavailable["symbol"] == "weightedcluster._wc_core.weighted_medoids"


def test_workload_generation_is_deterministic_and_valid():
    harness = _load_harness()
    first = harness.generate_workload(n=18, k=3, seed=71, weighted=True)
    second = harness.generate_workload(n=18, k=3, seed=71, weighted=True)

    np.testing.assert_array_equal(first["D"], second["D"])
    np.testing.assert_array_equal(first["labels"], second["labels"])
    np.testing.assert_array_equal(first["weights"], second["weights"])
    harness.validate_workload(first, n=18, k=3)
    assert first["D"].shape == (18, 18)
    assert np.allclose(first["D"], first["D"].T)
    assert np.allclose(np.diag(first["D"]), 0.0)
    assert np.all(first["weights"] > 0.0)
    assert not np.allclose(first["weights"], 1.0)

    unit = harness.generate_workload(n=18, k=3, seed=71, weighted=False)
    assert np.array_equal(unit["weights"], np.ones(18))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 2, "k": 3, "seed": 1, "weighted": False}, "n must be"),
        ({"n": 8, "k": 1, "seed": 1, "weighted": False}, "k must be"),
    ],
)
def test_workload_generation_rejects_invalid_arguments(kwargs: dict, message: str):
    harness = _load_harness()
    with pytest.raises(ValueError, match=message):
        harness.generate_workload(**kwargs)


def test_default_timing_contract_and_pilot_selection_are_explicit():
    harness = _load_harness()
    config = harness.benchmark_config("standard")

    assert config["warmups"] == 3
    assert config["calibration_target_ns"] == 200_000_000
    assert config["samples"] == 7
    assert config["pilot_target_ns"] == {
        "small": 10_000_000,
        "medium": 100_000_000,
        "large": 1_000_000_000,
    }
    observations = [
        {"n": 16, "slowest_ns": 4_000_000},
        {"n": 32, "slowest_ns": 30_000_000},
        {"n": 64, "slowest_ns": 300_000_000},
        {"n": 96, "slowest_ns": 1_200_000_000},
    ]
    selected = harness.select_pilot_sizes(observations, config["pilot_target_ns"])
    assert selected == {"small": 16, "medium": 32, "large": 64}


def test_timing_summary_uses_median_iqr_min_without_threshold_assertions():
    harness = _load_harness()
    summary = harness.summarize_samples([7, 3, 5, 11, 9], calls_per_sample=4)

    assert summary == {
        "median_ns_per_call": 7.0,
        "q1_ns_per_call": 4.0,
        "q3_ns_per_call": 10.0,
        "iqr_ns_per_call": 6.0,
        "min_ns_per_call": 3.0,
        "calls_per_sample": 4,
        "sample_count": 5,
    }


def test_worker_environment_isolated_without_mutating_parent(monkeypatch: pytest.MonkeyPatch):
    harness = _load_harness()
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    before = {name: os.environ.get(name) for name in names}

    child = harness.build_worker_environment(os.environ)

    assert {name: child[name] for name in names} == {name: "1" for name in names}
    assert {name: os.environ.get(name) for name in names} == before


def test_existing_limit_threads_context_reconfigures_initialized_native_pools_and_restores():
    before = {
        (pool.get("user_api"), pool.get("prefix")): pool.get("num_threads")
        for pool in threadpool_info()
    }
    with limit_threads(omp=1, mkl=1, openblas=1, blas=1):
        inside = {
            (pool.get("user_api"), pool.get("prefix")): pool.get("num_threads")
            for pool in threadpool_info()
        }

    assert inside
    assert all(value == 1 for value in inside.values())
    assert {
        (pool.get("user_api"), pool.get("prefix")): pool.get("num_threads")
        for pool in threadpool_info()
    } == before


def test_threadpoolctl_is_an_explicit_dev_measurement_dependency():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "threadpoolctl>=3.6" in pyproject["project"]["optional-dependencies"]["dev"]


def test_cli_rejects_invalid_candidates():
    completed = subprocess.run(
        [sys.executable, str(HARNESS_PATH), "--candidates", "12,nope,24"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert completed.returncode != 0
    assert "candidates" in completed.stderr.lower()


def test_cold_worker_rejects_structurally_valid_result_that_differs_from_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = _load_harness()
    workload = {
        "D": np.zeros((4, 4), dtype=float),
        "labels": np.array([0, 0, 1, 1], dtype=np.int32),
        "weights": np.ones(4, dtype=float),
    }

    def _numba_result():
        return ({key: float(key == "PBC") for key in harness.QUALITY_KEYS}, {1: 0.0}, {1: 0.0})

    def _compiled_oracle():
        return ({key: 0.0 for key in harness.QUALITY_KEYS}, {1: 0.0}, {1: 0.0})

    def _resolve(operation: str, backend: str) -> dict[str, object]:
        assert operation == "cluster_quality"
        if backend == "numba":
            return {
                "status": "AVAILABLE",
                "symbol": "weightedcluster._numba.cluster_quality_numba",
                "kernel": _numba_result,
                "reason": None,
            }
        if backend == "compiled":
            return {
                "status": "AVAILABLE",
                "symbol": "weightedcluster._wc_core.cluster_quality",
                "kernel": _compiled_oracle,
                "reason": None,
            }
        raise AssertionError(f"unexpected backend {backend}")

    monkeypatch.setattr(harness, "generate_workload", lambda **_kwargs: workload)
    monkeypatch.setattr(harness, "resolve_direct_kernel", _resolve)
    monkeypatch.setattr(
        harness,
        "_invoke_kernel",
        lambda _operation, kernel, _workload, **_kwargs: kernel(),
    )
    monkeypatch.setattr(harness, "_normalize_threadpools", lambda _pools: [])
    monkeypatch.setattr(harness, "_over_limit_pools", lambda _pools: [])

    row = harness._cold_worker_report(
        {"cold_n": 4, "k": 2, "seed": 7},
        "cluster_quality",
    )

    assert row["status"] == "ERROR"
    assert row["correctness"] == "ERROR"
    assert row["reason"] == "quality statistic PBC differs from the oracle"
    assert row["oracle_backend"] == "compiled"
    assert row["oracle_symbol"] == "weightedcluster._wc_core.cluster_quality"


def test_rss_orchestration_uses_one_process_per_available_backend_workload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    harness = _load_harness()
    warm_rows = [
        {
            "operation": "weighted_medoids",
            "backend": "compiled",
            "symbol": "weightedcluster._wc_core.weighted_medoids",
            "phase": "warm",
            "scale": "small",
            "n": 8,
            "k": 2,
            "weights": "unit",
            "status": "PASS",
            "correctness": "PASS",
            "reason": None,
            "timing": {},
        },
        {
            "operation": "cluster_quality",
            "backend": "pure_python",
            "symbol": "weightedcluster._fallback.cluster_quality",
            "phase": "warm",
            "scale": "large",
            "n": 12,
            "k": 2,
            "weights": "nonuniform",
            "status": "PASS",
            "correctness": "PASS",
            "reason": None,
            "timing": {},
        },
        {
            "operation": "cluster_quality",
            "backend": "numba",
            "symbol": "weightedcluster._numba.cluster_quality_numba",
            "phase": "warm",
            "scale": "large",
            "n": 12,
            "k": 2,
            "weights": "nonuniform",
            "status": "SKIP",
            "correctness": "NA",
            "reason": "Numba deliberately unavailable",
            "timing": None,
        },
    ]
    rss_measurements: list[dict[str, object]] = []

    def _worker(
        _script_path: Path,
        config: dict[str, object],
        *,
        mode: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        if mode == "--_cold-worker":
            return {"status": "SKIP"}
        if mode == "--_worker":
            return {"status": "PASS", "errors": [], "results": warm_rows}
        assert mode == "--_rss-worker"
        measurement = dict(config["rss_measurement"])
        rss_measurements.append(measurement)
        return {
            **measurement,
            "status": "PASS",
            "max_rss_kib": 12345,
            "reason": None,
        }

    monkeypatch.setattr(harness, "_run_json_worker", _worker)

    report = harness.run_benchmark_report(
        profile="tiny",
        candidates=[8, 12],
        seed=17,
        k=2,
        include_rss=True,
        repo_root=tmp_path,
    )

    assert len(rss_measurements) == 2
    assert {(row["operation"], row["backend"], row["scale"]) for row in rss_measurements} == {
        ("weighted_medoids", "compiled", "small"),
        ("cluster_quality", "pure_python", "large"),
    }
    assert report["rss"]["status"] == "PASS"
    assert report["rss"]["measurement_process"] == "one_subprocess_per_backend_workload"
    assert len(report["rss"]["rows"]) == 3
    skipped = next(row for row in report["rss"]["rows"] if row["backend"] == "numba")
    assert skipped["status"] == "SKIP"
    assert skipped["max_rss_kib"] is None
    assert skipped["reason"] == "Numba deliberately unavailable"


def test_tiny_cli_smoke_emits_complete_report_without_timing_thresholds(tmp_path: Path):
    output = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(HARNESS_PATH),
            "--profile",
            "tiny",
            "--candidates",
            "8,12",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.1"
    assert report["status"] == "PASS"
    assert report["build"]["WC_FAST_SWAP"] == "OFF"
    assert report["benchmark_config"]["profile"] == "tiny"
    assert report["pilot"]["candidates"] == [8, 12]
    assert set(report["pilot"]["selected_sizes"]) == {"small", "medium", "large"}
    assert set(report["symbols"]) == {"weighted_medoids", "cluster_quality"}
    assert report["provenance"]["python"]["executable"]
    assert report["provenance"]["packages"]["numpy"]["version"]
    assert report["provenance"]["repositories"]["pyweightedcluster"]["commit"]
    assert report["provenance"]["repositories"]["traminex"]["commit"]

    requested = report["threads"]["requested_env"]
    assert requested == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "BLAS_NUM_THREADS": "1",
    }
    assert report["threads"]["over_limit"] == []
    assert report["threads"]["inside_context"]
    assert any(
        pool["user_api"] in {"blas", "openmp"} for pool in report["threads"]["inside_context"]
    )
    assert all(pool["num_threads"] <= 1 for pool in report["threads"]["inside_context"])

    assert report["numba_cold"]
    assert all(row["phase"] == "cold_import_first_call" for row in report["numba_cold"])
    assert all(row["threads"]["over_limit"] == [] for row in report["numba_cold"])
    assert all(row["correctness"] in {"PASS", "NA"} for row in report["numba_cold"])
    assert all(row["oracle_backend"] for row in report["numba_cold"] if row["status"] == "PASS")
    assert all(row["oracle_symbol"] for row in report["numba_cold"] if row["status"] == "PASS")
    assert report["results"]
    assert not any("distance" in row["operation"] for row in report["results"])
    for row in report["results"]:
        assert row["status"] in {"PASS", "SKIP"}
        if row["status"] == "PASS":
            assert row["correctness"] == "PASS"
            assert row["phase"] == "warm"
            assert row["timing"]["sample_count"] >= 1
            assert row["timing"]["calls_per_sample"] >= 1
    assert "weighted_medoids" in completed.stdout
    assert "cluster_quality" in completed.stdout


@pytest.mark.skipif(
    not RECORDED_REPORT.is_file(),
    reason="needs benchmarks/results/, machine-specific evidence the published package leaves out",
)
def test_recorded_report_has_complete_correctness_thread_and_provenance_evidence():
    assert RECORDED_REPORT.is_file()
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))

    assert report["schema_version"] == "1.1"
    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["build"]["WC_FAST_SWAP"] == "OFF"
    assert report["benchmark_config"]["profile"] == "standard"
    assert report["benchmark_config"]["warmups"] == 3
    assert report["benchmark_config"]["samples"] == 7
    assert report["benchmark_config"]["calibration_target_ns"] == 200_000_000
    assert report["pilot"]["candidates"]
    assert set(report["pilot"]["selected_sizes"]) == {"small", "medium", "large"}
    assert report["pilot"]["observations"]

    requested = report["threads"]["requested_env"]
    assert requested == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "BLAS_NUM_THREADS": "1",
    }
    assert report["threads"]["over_limit"] == []
    assert report["threads"]["inside_context"]
    assert all(pool["num_threads"] <= 1 for pool in report["threads"]["inside_context"])

    symbol_rows = [
        symbol
        for operation in ("weighted_medoids", "cluster_quality")
        for symbol in report["symbols"][operation]
    ]
    assert len(symbol_rows) == 6
    assert {row["backend"] for row in symbol_rows} == {"compiled", "numba", "pure_python"}
    assert all(row["status"] in {"AVAILABLE", "SKIP"} for row in symbol_rows)
    assert all(row["reason"] for row in symbol_rows if row["status"] == "SKIP")

    assert len(report["numba_cold"]) == 2
    assert all(row["phase"] == "cold_import_first_call" for row in report["numba_cold"])
    assert all(row["status"] in {"PASS", "SKIP"} for row in report["numba_cold"])
    assert all(row["correctness"] in {"PASS", "NA"} for row in report["numba_cold"])
    assert all(row["oracle_backend"] for row in report["numba_cold"] if row["status"] == "PASS")
    assert all(row["oracle_symbol"] for row in report["numba_cold"] if row["status"] == "PASS")
    assert report["rss"] == {
        "status": "NOT_REQUESTED",
        "measurement_process": "one_subprocess_per_backend_workload",
        "rows": [],
    }
    assert report["results"]
    assert {row["operation"] for row in report["results"]} == {
        "weighted_medoids",
        "cluster_quality",
    }
    assert {row["weights"] for row in report["results"]} == {"unit", "nonuniform"}
    for row in report["results"]:
        assert row["status"] in {"PASS", "SKIP"}
        if row["status"] == "PASS":
            assert row["correctness"] == "PASS"
            assert row["timing"]["sample_count"] == 7
            assert row["timing"]["calls_per_sample"] >= 1

    provenance = report["provenance"]
    assert provenance["system"]["platform"] == "Linux"
    assert provenance["system"]["machine"] == "x86_64"
    assert provenance["system"]["cpu_model"]
    assert provenance["python"]["executable"]
    assert provenance["toolchain"]["compiler"]
    assert provenance["toolchain"]["cmake"]
    for package in ("numpy", "numba", "traminex", "pyweightedcluster"):
        assert provenance["packages"][package]["version"]
        assert provenance["packages"][package]["path"]
    for repository in ("pyweightedcluster", "traminex"):
        assert provenance["repositories"][repository]["commit"]
        assert provenance["repositories"][repository]["dirty"] is False

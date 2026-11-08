"""Reproducible correctness-checked backend benchmark report.

The importable parent/orchestrator intentionally uses only the Python standard
library.  Scientific and native modules are imported only in fresh worker
processes after thread-limit environment variables have been established.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "1.1"
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLAS_NUM_THREADS",
)
QUALITY_KEYS = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")

DIRECT_KERNELS = {
    "weighted_medoids": {
        "compiled": "weightedcluster._wc_core.weighted_medoids",
        "pure_python": "weightedcluster._fallback.weighted_medoids",
    },
    "cluster_quality": {
        "compiled": "weightedcluster._wc_core.cluster_quality",
        "numba": "weightedcluster._numba.cluster_quality_numba",
        "pure_python": "weightedcluster._fallback.cluster_quality",
    },
}


def benchmark_config(profile: str) -> dict[str, object]:
    """Return the explicit timing and pilot contract for a named profile."""
    if profile == "standard":
        return {
            "profile": "standard",
            "warmups": 3,
            "calibration_target_ns": 200_000_000,
            "samples": 7,
            "pilot_target_ns": {
                "small": 10_000_000,
                "medium": 100_000_000,
                "large": 1_000_000_000,
            },
            "default_candidates": [12, 18, 27, 40, 60, 90, 135, 200],
            "max_calls_per_sample": 10_000_000,
        }
    if profile == "tiny":
        return {
            "profile": "tiny",
            "warmups": 1,
            "calibration_target_ns": 1_000_000,
            "samples": 2,
            "pilot_target_ns": {
                "small": 100_000,
                "medium": 1_000_000,
                "large": 10_000_000,
            },
            "default_candidates": [8, 12],
            "max_calls_per_sample": 100,
        }
    raise ValueError("profile must be 'standard' or 'tiny'")


def build_worker_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """Copy the parent environment and request one native thread in the child."""
    child = dict(parent)
    for name in THREAD_ENV_NAMES:
        child[name] = "1"
    child["PYTHONHASHSEED"] = "0"
    return child


def resolve_direct_kernel(
    operation: str,
    backend: str,
    *,
    import_module: Callable[[str], object] = importlib.import_module,
) -> dict[str, object]:
    """Resolve one exact private symbol without substituting another backend."""
    if operation not in DIRECT_KERNELS:
        raise ValueError(f"unknown benchmark operation {operation!r}")
    if backend not in DIRECT_KERNELS[operation]:
        raise ValueError(f"unknown backend {backend!r} for {operation}")
    symbol = DIRECT_KERNELS[operation][backend]
    module_name, _, attribute = symbol.rpartition(".")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        return {
            "status": "SKIP",
            "symbol": symbol,
            "kernel": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    kernel = getattr(module, attribute, None)
    if not callable(kernel):
        return {
            "status": "SKIP",
            "symbol": symbol,
            "kernel": None,
            "reason": f"symbol {symbol} is missing or not callable",
        }
    return {"status": "AVAILABLE", "symbol": symbol, "kernel": kernel, "reason": None}


def generate_workload(*, n: int, k: int, seed: int, weighted: bool) -> dict[str, object]:
    """Generate a deterministic separated Euclidean dissimilarity workload."""
    if int(n) < max(4, int(k)):
        raise ValueError("n must be at least max(4, k)")
    if int(k) < 2:
        raise ValueError("k must be at least 2")
    if int(k) > int(n):
        raise ValueError("k cannot exceed n")

    import numpy as np

    n_int = int(n)
    k_int = int(k)
    rng = np.random.default_rng(int(seed))
    dimensions = max(4, k_int)
    labels = (np.arange(n_int, dtype=np.int32) % k_int).astype(np.int32)
    centers = np.zeros((k_int, dimensions), dtype=float)
    for cluster in range(k_int):
        centers[cluster, cluster % dimensions] = 12.0 * float(cluster + 1)
    points = centers[labels] + rng.normal(0.0, 0.12, size=(n_int, dimensions))
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2, dtype=float))
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)
    weights = (
        0.5 + 1.5 * rng.random(n_int, dtype=float) if weighted else np.ones(n_int, dtype=float)
    )
    workload = {
        "D": distances,
        "labels": labels,
        "weights": weights,
        "points": points,
        "weighted": bool(weighted),
        "seed": int(seed),
    }
    validate_workload(workload, n=n_int, k=k_int)
    return workload


def validate_workload(workload: Mapping[str, object], *, n: int, k: int) -> None:
    """Validate all matrix, label, and weight invariants before timing."""
    import numpy as np

    distances = np.asarray(workload["D"], dtype=float)
    labels = np.asarray(workload["labels"], dtype=np.int32)
    weights = np.asarray(workload["weights"], dtype=float)
    if distances.shape != (int(n), int(n)):
        raise ValueError("workload dissimilarity matrix has the wrong shape")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("workload dissimilarities must be finite and non-negative")
    if not np.allclose(distances, distances.T, atol=1e-12, rtol=0.0):
        raise ValueError("workload dissimilarity matrix must be symmetric")
    if not np.allclose(np.diag(distances), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("workload dissimilarity diagonal must be zero")
    if labels.shape != (int(n),) or set(np.unique(labels)) != set(range(int(k))):
        raise ValueError("workload labels must contain every contiguous zero-based cluster")
    if weights.shape != (int(n),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("workload weights must be finite and positive")


def select_pilot_sizes(
    observations: Sequence[Mapping[str, object]], targets: Mapping[str, int]
) -> dict[str, int]:
    """Choose the largest candidate whose slowest warmed kernel fits each target."""
    if not observations:
        raise ValueError("pilot observations cannot be empty")
    ordered = sorted(observations, key=lambda item: int(item["n"]))
    selected: dict[str, int] = {}
    for scale, target in targets.items():
        eligible = [int(item["n"]) for item in ordered if int(item["slowest_ns"]) <= int(target)]
        selected[str(scale)] = max(eligible) if eligible else int(ordered[0]["n"])
    return selected


def summarize_samples(samples_ns_per_call: Sequence[float], *, calls_per_sample: int) -> dict:
    """Summarize per-call samples with Tukey hinges for a stable IQR."""
    values = sorted(float(value) for value in samples_ns_per_call)
    if not values:
        raise ValueError("at least one timing sample is required")
    midpoint = len(values) // 2
    lower = values[:midpoint] or values
    upper = values[(midpoint + 1 if len(values) % 2 else midpoint) :] or values
    q1 = float(statistics.median(lower))
    q3 = float(statistics.median(upper))
    return {
        "median_ns_per_call": float(statistics.median(values)),
        "q1_ns_per_call": q1,
        "q3_ns_per_call": q3,
        "iqr_ns_per_call": float(q3 - q1),
        "min_ns_per_call": float(min(values)),
        "calls_per_sample": int(calls_per_sample),
        "sample_count": len(values),
    }


def _invoke_kernel(
    operation: str,
    kernel: Callable[..., object],
    workload: Mapping[str, object],
    *,
    k: int,
    seed: int,
) -> object:
    if operation == "weighted_medoids":
        return kernel(
            workload["D"],
            int(k),
            weights=workload["weights"],
            max_iter=40,
            n_init=3,
            random_state=int(seed),
        )
    if operation == "cluster_quality":
        return kernel(
            workload["D"],
            workload["labels"],
            weights=workload["weights"],
        )
    raise ValueError(f"unknown operation {operation!r}")


def _canonical_partition(labels: object) -> tuple[tuple[int, ...], ...]:
    import numpy as np

    array = np.asarray(labels, dtype=int).reshape(-1)
    groups = [
        tuple(int(index) for index in np.where(array == label)[0]) for label in np.unique(array)
    ]
    return tuple(sorted(groups))


def _close_or_matching_nan(left: float, right: float, *, atol: float, rtol: float) -> bool:
    import numpy as np

    if np.isnan(left) and np.isnan(right):
        return True
    return bool(np.isclose(left, right, atol=atol, rtol=rtol))


def _compare_results(operation: str, result: object, oracle: object) -> tuple[bool, str | None]:
    import numpy as np

    if operation == "weighted_medoids":
        labels, medoids, objective = result
        ref_labels, ref_medoids, ref_objective = oracle
        if _canonical_partition(labels) != _canonical_partition(ref_labels):
            return False, "partition differs from the direct-kernel oracle"
        if np.asarray(medoids).shape != np.asarray(ref_medoids).shape:
            return False, "medoid result shape differs from the oracle"
        if not np.isclose(float(objective), float(ref_objective), atol=1e-8, rtol=1e-6):
            return False, "objective differs from the direct-kernel oracle"
        return True, None

    if operation == "cluster_quality":
        stats, asw, asww = result
        ref_stats, ref_asw, ref_asww = oracle
        for key in QUALITY_KEYS:
            if not _close_or_matching_nan(
                float(stats[key]), float(ref_stats[key]), atol=1e-10, rtol=1e-9
            ):
                return False, f"quality statistic {key} differs from the oracle"
        for name, values, reference in (("ASW", asw, ref_asw), ("ASWw", asww, ref_asww)):
            if set(values) != set(reference):
                return False, f"{name} cluster keys differ from the oracle"
            for key in values:
                if not _close_or_matching_nan(
                    float(values[key]), float(reference[key]), atol=1e-10, rtol=1e-9
                ):
                    return False, f"{name} cluster {key} differs from the oracle"
        return True, None
    return False, f"unknown operation {operation!r}"


def _measure_kernel(
    function: Callable[[], object], config: Mapping[str, object]
) -> dict[str, object]:
    for _ in range(int(config["warmups"])):
        function()

    start = time.perf_counter_ns()
    function()
    elapsed_once = max(1, time.perf_counter_ns() - start)
    target = int(config["calibration_target_ns"])
    calls = max(1, int(math.ceil(target / elapsed_once)))
    calls = min(calls, int(config["max_calls_per_sample"]))
    while calls < int(config["max_calls_per_sample"]):
        start = time.perf_counter_ns()
        for _call in range(calls):
            function()
        calibration_elapsed = max(1, time.perf_counter_ns() - start)
        if calibration_elapsed >= target:
            break
        adjusted = int(math.ceil(calls * target / calibration_elapsed))
        calls = min(
            int(config["max_calls_per_sample"]),
            max(calls + 1, adjusted),
        )

    samples: list[float] = []
    for _ in range(int(config["samples"])):
        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            start = time.perf_counter_ns()
            for _call in range(calls):
                function()
            elapsed = time.perf_counter_ns() - start
        finally:
            if was_enabled:
                gc.enable()
        samples.append(float(elapsed) / float(calls))

    summary = summarize_samples(samples, calls_per_sample=calls)
    summary["samples_ns_per_call"] = samples
    summary["timer"] = "time.perf_counter_ns"
    summary["gc_disabled_inside_batch"] = True
    return summary


def _normalize_threadpools(pools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = (
        "user_api",
        "internal_api",
        "num_threads",
        "prefix",
        "version",
        "threading_layer",
        "filepath",
    )
    return [{key: pool.get(key) for key in keys} for pool in pools]


def _over_limit_pools(pools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    relevant = []
    for pool in pools:
        count = pool.get("num_threads")
        if pool.get("user_api") in {"blas", "openmp"} and isinstance(count, int) and count > 1:
            relevant.append(dict(pool))
    return relevant


def _git_record(path: Path) -> dict[str, object]:
    def _git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "path": str(path.resolve()),
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"path": str(path.resolve()), "commit": "unknown", "error": str(exc)}


def _command_version(command: str, *args: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        return "unavailable"
    try:
        completed = subprocess.run([executable, *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"error: {exc}"
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else "unknown"


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _package_record(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "available",
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(Path(getattr(module, "__file__", "unknown")).resolve()),
    }


def _provenance(repo_root: Path) -> dict[str, object]:
    traminex_root = repo_root.parent / "PROJECT_TRAMINEX"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "system": {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "toolchain": {
            "compiler": _command_version("g++", "--version"),
            "cmake": _command_version("cmake", "--version"),
        },
        "packages": {
            "numpy": _package_record("numpy"),
            "numba": _package_record("numba"),
            "traminex": _package_record("traminex"),
            "pyweightedcluster": _package_record("weightedcluster"),
        },
        "repositories": {
            "pyweightedcluster": _git_record(repo_root),
            "traminex": _git_record(traminex_root),
        },
    }


def _resolve_all_kernels() -> tuple[dict[str, dict[str, dict[str, object]]], dict]:
    resolved: dict[str, dict[str, dict[str, object]]] = {}
    serializable: dict[str, list[dict[str, object]]] = {}
    for operation, backends in DIRECT_KERNELS.items():
        resolved[operation] = {}
        serializable[operation] = []
        for backend in backends:
            entry = resolve_direct_kernel(operation, backend)
            resolved[operation][backend] = entry
            serializable[operation].append(
                {
                    "backend": backend,
                    "symbol": entry["symbol"],
                    "status": entry["status"],
                    "reason": entry["reason"],
                }
            )
    return resolved, serializable


def _pilot(
    resolved: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    candidates: Sequence[int],
    k: int,
    seed: int,
    targets: Mapping[str, int],
) -> dict[str, object]:
    observations = []
    for n in candidates:
        workload = generate_workload(n=int(n), k=k, seed=seed + int(n), weighted=True)
        measurements = []
        for operation, backend_entries in resolved.items():
            for backend, entry in backend_entries.items():
                if entry["status"] != "AVAILABLE":
                    continue
                kernel = entry["kernel"]
                assert callable(kernel)
                _invoke_kernel(operation, kernel, workload, k=k, seed=seed)
                start = time.perf_counter_ns()
                _invoke_kernel(operation, kernel, workload, k=k, seed=seed)
                elapsed = max(1, time.perf_counter_ns() - start)
                measurements.append(
                    {"operation": operation, "backend": backend, "elapsed_ns": int(elapsed)}
                )
        if not measurements:
            raise RuntimeError("no direct kernels were available for pilot calibration")
        observations.append(
            {
                "n": int(n),
                "slowest_ns": max(int(item["elapsed_ns"]) for item in measurements),
                "measurements": measurements,
            }
        )
    return {
        "candidates": [int(value) for value in candidates],
        "target_ns": {str(key): int(value) for key, value in targets.items()},
        "observations": observations,
        "selected_sizes": select_pilot_sizes(observations, targets),
    }


def _benchmark_rows(
    resolved: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    selected_sizes: Mapping[str, int],
    k: int,
    seed: int,
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    oracle_order = ("compiled", "pure_python", "numba")
    for scale, n in selected_sizes.items():
        for weighted in (False, True):
            workload = generate_workload(
                n=int(n),
                k=k,
                seed=seed + int(n) + (10_000 if weighted else 0),
                weighted=weighted,
            )
            for operation, backend_entries in resolved.items():
                oracle_backend = next(
                    backend
                    for backend in oracle_order
                    if backend_entries[backend]["status"] == "AVAILABLE"
                )
                oracle_kernel = backend_entries[oracle_backend]["kernel"]
                assert callable(oracle_kernel)
                oracle = _invoke_kernel(operation, oracle_kernel, workload, k=k, seed=seed)
                for backend, entry in backend_entries.items():
                    base = {
                        "operation": operation,
                        "backend": backend,
                        "symbol": entry["symbol"],
                        "phase": "warm",
                        "scale": scale,
                        "n": int(n),
                        "k": int(k),
                        "weights": "nonuniform" if weighted else "unit",
                    }
                    if entry["status"] != "AVAILABLE":
                        rows.append(
                            {
                                **base,
                                "status": "SKIP",
                                "correctness": "NA",
                                "reason": entry["reason"],
                                "timing": None,
                            }
                        )
                        continue
                    kernel = entry["kernel"]
                    assert callable(kernel)
                    result = _invoke_kernel(operation, kernel, workload, k=k, seed=seed)
                    correct, reason = _compare_results(operation, result, oracle)
                    if not correct:
                        rows.append(
                            {
                                **base,
                                "status": "ERROR",
                                "correctness": "ERROR",
                                "reason": reason,
                                "timing": None,
                            }
                        )
                        continue
                    timing = _measure_kernel(
                        lambda op=operation, fn=kernel, data=workload: _invoke_kernel(
                            op, fn, data, k=k, seed=seed
                        ),
                        config,
                    )
                    rows.append(
                        {
                            **base,
                            "status": "PASS",
                            "correctness": "PASS",
                            "reason": None,
                            "timing": timing,
                        }
                    )
    return rows


def _worker_report(config: Mapping[str, object]) -> dict[str, object]:
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    # Load native pools only after this worker inherited its one-thread
    # environment, then enforce and inspect them throughout measurement.
    del np
    repo_root = Path(str(config["repo_root"])).resolve()
    before_context = _normalize_threadpools(threadpool_info())
    with threadpool_limits(limits=1):
        resolved, symbols = _resolve_all_kernels()
        inside_context = _normalize_threadpools(threadpool_info())
        over_limit = _over_limit_pools(inside_context)
        errors = []
        if over_limit:
            errors.append("relevant native thread pool remained above one thread")
            pilot = {
                "candidates": list(config["candidates"]),
                "target_ns": dict(config["timing"]["pilot_target_ns"]),
                "observations": [],
                "selected_sizes": {},
            }
            rows = []
        else:
            pilot = _pilot(
                resolved,
                candidates=config["candidates"],
                k=int(config["k"]),
                seed=int(config["seed"]),
                targets=config["timing"]["pilot_target_ns"],
            )
            rows = _benchmark_rows(
                resolved,
                selected_sizes=pilot["selected_sizes"],
                k=int(config["k"]),
                seed=int(config["seed"]),
                config=config["timing"],
            )
            errors.extend(str(row["reason"]) for row in rows if row["status"] == "ERROR")
        inside_context = _normalize_threadpools(threadpool_info())
        final_over_limit = _over_limit_pools(inside_context)
        if final_over_limit:
            over_limit = final_over_limit
            errors.append("relevant native thread pool exceeded one thread during measurement")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ERROR" if errors else "PASS",
        "errors": errors,
        "build": {"WC_FAST_SWAP": "OFF"},
        "benchmark_config": dict(config["timing"]),
        "workload": {
            "generator": "separated_euclidean",
            "seed": int(config["seed"]),
            "k": int(config["k"]),
            "weight_variants": ["unit", "nonuniform"],
        },
        "threads": {
            "requested_env": {name: os.environ.get(name) for name in THREAD_ENV_NAMES},
            "before_context": before_context,
            "inside_context": inside_context,
            "over_limit": over_limit,
            "controller": "threadpoolctl.threadpool_limits(limits=1)",
        },
        "symbols": symbols,
        "pilot": pilot,
        "results": rows,
        "provenance": _provenance(repo_root),
    }


def _cold_worker_report(config: Mapping[str, object], operation: str) -> dict[str, object]:
    from threadpoolctl import threadpool_info, threadpool_limits

    workload = generate_workload(
        n=int(config["cold_n"]),
        k=int(config["k"]),
        seed=int(config["seed"]),
        weighted=True,
    )
    before_context = _normalize_threadpools(threadpool_info())
    oracle_backend = None
    oracle_symbol = None
    correctness = "NA"
    with threadpool_limits(limits=1):
        inside_context = _normalize_threadpools(threadpool_info())
        over_limit = _over_limit_pools(inside_context)
        start = time.perf_counter_ns()
        entry = resolve_direct_kernel(operation, "numba")
        if entry["status"] == "AVAILABLE" and not over_limit:
            kernel = entry["kernel"]
            assert callable(kernel)
            result = _invoke_kernel(
                operation,
                kernel,
                workload,
                k=int(config["k"]),
                seed=int(config["seed"]),
            )
            elapsed = time.perf_counter_ns() - start
            if operation == "weighted_medoids":
                structurally_valid = len(result) == 3
            else:
                structurally_valid = len(result) == 3 and set(QUALITY_KEYS) <= set(result[0])
            if not structurally_valid:
                status = "ERROR"
                correctness = "ERROR"
                reason = "Numba cold result failed structural validation"
            else:
                oracle_entry = None
                for candidate in ("compiled", "pure_python"):
                    candidate_entry = resolve_direct_kernel(operation, candidate)
                    if candidate_entry["status"] == "AVAILABLE":
                        oracle_backend = candidate
                        oracle_entry = candidate_entry
                        break
                if oracle_entry is None:
                    status = "ERROR"
                    correctness = "ERROR"
                    reason = "no independent direct-kernel oracle is available"
                else:
                    oracle_symbol = oracle_entry["symbol"]
                    oracle_kernel = oracle_entry["kernel"]
                    assert callable(oracle_kernel)
                    oracle = _invoke_kernel(
                        operation,
                        oracle_kernel,
                        workload,
                        k=int(config["k"]),
                        seed=int(config["seed"]),
                    )
                    correct, reason = _compare_results(operation, result, oracle)
                    status = "PASS" if correct else "ERROR"
                    correctness = "PASS" if correct else "ERROR"
        else:
            elapsed = None
            status = "ERROR" if over_limit else "SKIP"
            reason = (
                "native thread pool remained above one thread" if over_limit else entry["reason"]
            )
        inside_context = _normalize_threadpools(threadpool_info())
        final_over_limit = _over_limit_pools(inside_context)
        if final_over_limit:
            over_limit = final_over_limit
            status = "ERROR"
            reason = "native thread pool exceeded one thread during cold measurement"
    return {
        "operation": operation,
        "backend": "numba",
        "symbol": entry["symbol"],
        "oracle_backend": oracle_backend,
        "oracle_symbol": oracle_symbol,
        "phase": "cold_import_first_call",
        "status": status,
        "correctness": correctness,
        "elapsed_ns": elapsed,
        "reason": reason,
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "threads": {
            "requested_env": {name: os.environ.get(name) for name in THREAD_ENV_NAMES},
            "before_context": before_context,
            "inside_context": inside_context,
            "over_limit": over_limit,
        },
    }


def _rss_worker_report(config: Mapping[str, object]) -> dict[str, object]:
    from threadpoolctl import threadpool_limits

    measurement = dict(config["rss_measurement"])
    operation = str(measurement["operation"])
    backend = str(measurement["backend"])
    n = int(measurement["n"])
    weighted = measurement["weights"] == "nonuniform"
    base = {
        "operation": operation,
        "backend": backend,
        "symbol": str(measurement["symbol"]),
        "scale": str(measurement["scale"]),
        "n": n,
        "k": int(config["k"]),
        "weights": str(measurement["weights"]),
        "metric": "ru_maxrss",
        "unit": "KiB",
    }
    if platform.system() not in {"Linux", "Darwin"}:
        return {
            **base,
            "status": "NA",
            "max_rss_kib": None,
            "reason": f"ru_maxrss units are not normalized for {platform.system()}",
        }

    try:
        import resource
    except ImportError as exc:  # pragma: no cover - resource exists on supported Unix
        return {
            **base,
            "status": "NA",
            "max_rss_kib": None,
            "reason": f"resource module unavailable: {exc}",
        }

    workload = generate_workload(
        n=n,
        k=int(config["k"]),
        seed=int(config["seed"]) + n + (10_000 if weighted else 0),
        weighted=weighted,
    )
    with threadpool_limits(limits=1):
        entry = resolve_direct_kernel(operation, backend)
        if entry["status"] != "AVAILABLE":
            return {
                **base,
                "status": "SKIP",
                "max_rss_kib": None,
                "reason": entry["reason"],
            }
        kernel = entry["kernel"]
        assert callable(kernel)
        _invoke_kernel(
            operation,
            kernel,
            workload,
            k=int(config["k"]),
            seed=int(config["seed"]),
        )
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        maximum = int(math.ceil(maximum / 1024.0))
    return {
        **base,
        "status": "PASS",
        "max_rss_kib": maximum,
        "reason": None,
    }


def _run_json_worker(
    script_path: Path,
    config: Mapping[str, object],
    *,
    mode: str,
    operation: str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    environment = build_worker_environment(os.environ)
    if extra_env:
        environment.update(extra_env)
    command = [
        sys.executable,
        str(script_path),
        mode,
        "--_config-json",
        json.dumps(config, sort_keys=True),
    ]
    if operation is not None:
        command.extend(["--_cold-operation", operation])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(str(config["repo_root"])),
        env=environment,
    )
    return json.loads(completed.stdout)


def run_benchmark_report(
    *,
    profile: str,
    candidates: Sequence[int],
    seed: int,
    k: int,
    include_rss: bool,
    repo_root: Path,
) -> dict[str, object]:
    """Orchestrate fresh cold, warm, and optional RSS worker processes."""
    timing = benchmark_config(profile)
    config = {
        "repo_root": str(repo_root.resolve()),
        "profile": profile,
        "candidates": [int(value) for value in candidates],
        "cold_n": int(min(candidates)),
        "seed": int(seed),
        "k": int(k),
        "timing": timing,
    }
    script_path = Path(__file__).resolve()
    cold_rows = []
    for operation, implementations in DIRECT_KERNELS.items():
        if "numba" not in implementations:
            continue
        with tempfile.TemporaryDirectory(prefix="wc-numba-cache-") as cache_dir:
            cold_rows.append(
                _run_json_worker(
                    script_path,
                    config,
                    mode="--_cold-worker",
                    operation=operation,
                    extra_env={"NUMBA_CACHE_DIR": cache_dir},
                )
            )
    report = _run_json_worker(script_path, config, mode="--_worker")
    report["numba_cold"] = cold_rows
    if include_rss:
        rss_rows = []
        for result in report["results"]:
            measurement = {
                key: result[key]
                for key in ("operation", "backend", "symbol", "scale", "n", "weights")
            }
            if result["status"] != "PASS":
                rss_rows.append(
                    {
                        **measurement,
                        "k": int(result["k"]),
                        "metric": "ru_maxrss",
                        "unit": "KiB",
                        "status": "SKIP",
                        "max_rss_kib": None,
                        "reason": result["reason"],
                    }
                )
                continue
            rss_config = dict(config)
            rss_config["rss_measurement"] = measurement
            rss_rows.append(_run_json_worker(script_path, rss_config, mode="--_rss-worker"))
        rss_status = "ERROR" if any(row["status"] == "ERROR" for row in rss_rows) else "PASS"
        report["rss"] = {
            "status": rss_status,
            "measurement_process": "one_subprocess_per_backend_workload",
            "rows": rss_rows,
        }
    else:
        report["rss"] = {
            "status": "NOT_REQUESTED",
            "measurement_process": "one_subprocess_per_backend_workload",
            "rows": [],
        }
    if any(row["status"] == "ERROR" for row in cold_rows):
        report["status"] = "ERROR"
        report["errors"].append("Numba cold worker reported an error")
    if report["rss"]["status"] == "ERROR":
        report["status"] = "ERROR"
        report["errors"].append("RSS worker reported an error")
    return report


def render_human_table(report: Mapping[str, object]) -> str:
    """Render a concise warm-timing table plus the separate Numba cold rows."""
    lines = [
        "operation          backend      scale   n   weights      status   median_ms   iqr_ms",
        "-----------------  -----------  ------  ---  -----------  -------  ----------  -------",
    ]
    for row in report.get("results", []):
        timing = row.get("timing")
        if timing:
            median_ms = float(timing["median_ns_per_call"]) / 1_000_000.0
            iqr_ms = float(timing["iqr_ns_per_call"]) / 1_000_000.0
            median_text = f"{median_ms:10.3f}"
            iqr_text = f"{iqr_ms:7.3f}"
        else:
            median_text = "        NA"
            iqr_text = "     NA"
        lines.append(
            f"{row['operation']:<17}  {row['backend']:<11}  {row['scale']:<6}  "
            f"{row['n']:>3}  {row['weights']:<11}  {row['status']:<7}  "
            f"{median_text}  {iqr_text}"
        )
    lines.append("")
    lines.append("Numba cold import + first call (fresh process/cache):")
    for row in report.get("numba_cold", []):
        elapsed = row.get("elapsed_ns")
        elapsed_text = "NA" if elapsed is None else f"{float(elapsed) / 1_000_000.0:.3f} ms"
        lines.append(f"  {row['operation']}: {row['status']} ({elapsed_text})")
    lines.append(f"overall status: {report.get('status', 'unknown')}")
    return "\n".join(lines)


def _parse_candidates(value: str) -> list[int]:
    try:
        candidates = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidates must be comma-separated integers") from exc
    if not candidates or any(value < 4 for value in candidates):
        raise argparse.ArgumentTypeError("candidates must contain integers >= 4")
    if candidates != sorted(set(candidates)):
        raise argparse.ArgumentTypeError("candidates must be unique and increasing")
    return candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("standard", "tiny"), default="standard")
    parser.add_argument("--candidates", help="comma-separated geometric pilot candidates")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, help="write the complete JSON report here")
    parser.add_argument(
        "--rss",
        action="store_true",
        help="measure RSS in one fresh worker per backend/workload",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_cold-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_rss-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_cold-operation", choices=tuple(DIRECT_KERNELS), help=argparse.SUPPRESS)
    parser.add_argument("--_config-json", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args._worker or args._cold_worker or args._rss_worker:
        if not args._config_json:
            parser.error("internal workers require --_config-json")
        config = json.loads(args._config_json)
        if args._worker:
            payload = _worker_report(config)
        elif args._cold_worker:
            if args._cold_operation is None:
                parser.error("cold worker requires --_cold-operation")
            payload = _cold_worker_report(config, args._cold_operation)
        else:
            payload = _rss_worker_report(config)
        print(json.dumps(payload, sort_keys=True))
        return 0

    config = benchmark_config(args.profile)
    if args.candidates is None:
        candidates = list(config["default_candidates"])
    else:
        try:
            candidates = _parse_candidates(args.candidates)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    if args.k < 2:
        parser.error("k must be at least 2")
    if min(candidates) < args.k:
        parser.error("every candidate must be at least k")

    repo_root = Path(__file__).resolve().parents[1]
    report = run_benchmark_report(
        profile=args.profile,
        candidates=candidates,
        seed=args.seed,
        k=args.k,
        include_rss=bool(args.rss),
        repo_root=repo_root,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(render_human_table(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

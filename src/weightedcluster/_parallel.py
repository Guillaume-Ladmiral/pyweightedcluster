from __future__ import annotations

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from numbers import Integral
from typing import Callable, Iterable, Literal, Optional, Sequence, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")
ExecutorKind = Literal["thread", "process"]


def resolve_n_jobs(n_jobs: Optional[int], *, n_tasks: Optional[int] = None) -> int:
    """Resolve a validated worker count shared by all parallel workflows."""
    if n_jobs is None:
        workers = os.cpu_count() or 1
    else:
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
            raise ValueError("n_jobs must be a positive integer")
        workers = int(n_jobs)
        if workers < 1:
            raise ValueError("n_jobs must be a positive integer")
    if n_tasks is not None:
        if isinstance(n_tasks, bool) or not isinstance(n_tasks, Integral) or n_tasks < 0:
            raise ValueError("n_tasks must be a non-negative integer")
        if n_tasks > 0:
            workers = min(workers, int(n_tasks))
    return max(1, workers)


def validate_parallel_controls(
    parallel: bool,
    n_jobs: int | None,
    *,
    n_tasks: int | None = None,
) -> int:
    """Validate shared execution controls and return the resolved worker count."""
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    return resolve_n_jobs(n_jobs, n_tasks=n_tasks)


def _wrap_progress(
    iterable: Iterable[T],
    *,
    enabled: bool,
    total: int,
    desc: str,
) -> Iterable[T]:
    """Wrap an iterator in tqdm when available and requested."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
    except ImportError:
        # Degrade the way the package's other optional dependencies do:
        # clustassoc warns when mplcursors is missing and _threads warns when
        # threadpoolctl is. This branch used to return silently, so an explicit
        # progressbar=True did nothing and said nothing (finding F-28).
        warnings.warn(
            "tqdm is not installed; the requested progress bar is disabled",
            RuntimeWarning,
            stacklevel=2,
        )
        return iterable
    return _tqdm(iterable, total=total, desc=desc)


def run_task_batch(
    worker: Callable[[T], R],
    tasks: Sequence[T],
    *,
    parallel: bool,
    n_jobs: Optional[int],
    progressbar: bool,
    desc: str,
    executor_kind: ExecutorKind,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Sequence[object] = (),
) -> list[R]:
    """Run a task batch with shared `parallel`/`n_jobs` semantics."""
    if not isinstance(progressbar, bool):
        raise TypeError("progressbar must be a boolean")
    if executor_kind not in ("thread", "process"):
        raise ValueError(f"Unknown executor_kind: {executor_kind!r}")
    items = list(tasks)
    max_workers = validate_parallel_controls(parallel, n_jobs, n_tasks=len(items))
    if not items:
        return []

    use_parallel = parallel and max_workers > 1 and len(items) > 1

    if not use_parallel:
        if initializer is not None:
            initializer(*tuple(initargs))
        return [
            worker(item)
            for item in _wrap_progress(items, enabled=progressbar, total=len(items), desc=desc)
        ]

    if executor_kind == "thread":
        executor_cls = ThreadPoolExecutor
    elif executor_kind == "process":
        executor_cls = ProcessPoolExecutor

    executor_kwargs: dict[str, object] = {"max_workers": max_workers}
    if executor_kind == "process":
        executor_kwargs["mp_context"] = multiprocessing.get_context("spawn")
    if initializer is not None:
        executor_kwargs["initializer"] = initializer
        executor_kwargs["initargs"] = tuple(initargs)

    with executor_cls(**executor_kwargs) as executor:
        futures = {executor.submit(worker, item): idx for idx, item in enumerate(items)}
        ordered: list[Optional[R]] = [None] * len(items)
        for future in _wrap_progress(
            as_completed(futures),
            enabled=progressbar,
            total=len(items),
            desc=desc,
        ):
            ordered[futures[future]] = future.result()
    return [cast(R, result) for result in ordered]

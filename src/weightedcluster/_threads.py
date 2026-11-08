import contextlib
import os
import warnings
from contextlib import ExitStack
from numbers import Integral
from typing import Iterator


def _validate_thread_limit(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer or None")
    validated = int(value)
    if validated <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return validated


@contextlib.contextmanager
def limit_threads(
    omp: int | None = None,
    mkl: int | None = None,
    openblas: int | None = None,
    blas: int | None = None,
) -> Iterator[None]:
    """Temporarily limit native threadpools and their environment controls.

    Already loaded BLAS and OpenMP runtimes are controlled through
    :mod:`threadpoolctl` when it is installed. The environment variables also
    apply to libraries loaded inside the scope. Without ``threadpoolctl``, the
    context emits a warning because only not-yet-loaded runtimes can be limited.

    Example:
    with limit_threads(omp=1, mkl=1, openblas=1):
        ... heavy numerical work ...
    """
    omp = _validate_thread_limit(omp, name="omp")
    mkl = _validate_thread_limit(mkl, name="mkl")
    openblas = _validate_thread_limit(openblas, name="openblas")
    blas = _validate_thread_limit(blas, name="blas")
    keys = {
        "OMP_NUM_THREADS": omp,
        "MKL_NUM_THREADS": mkl,
        "OPENBLAS_NUM_THREADS": openblas,
        "BLAS_NUM_THREADS": blas,
    }
    old = {}
    try:
        for k, v in keys.items():
            old[k] = os.environ.get(k)
            if v is not None:
                os.environ[k] = str(v)

        requested = any(value is not None for value in (omp, mkl, openblas, blas))
        try:
            from threadpoolctl import ThreadpoolController
        except ImportError:
            if requested:
                warnings.warn(
                    "threadpoolctl is unavailable; only environment controls were set, "
                    "so already loaded native threadpools were not reconfigured",
                    RuntimeWarning,
                    stacklevel=2,
                )
            yield
        else:
            controller = ThreadpoolController()
            with ExitStack() as stack:
                if blas is not None:
                    stack.enter_context(controller.select(user_api="blas").limit(limits=blas))
                if mkl is not None:
                    stack.enter_context(controller.select(internal_api="mkl").limit(limits=mkl))
                if openblas is not None:
                    stack.enter_context(
                        controller.select(internal_api="openblas").limit(limits=openblas)
                    )
                if omp is not None:
                    stack.enter_context(controller.select(user_api="openmp").limit(limits=omp))
                yield
    finally:
        for k, v in old.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v

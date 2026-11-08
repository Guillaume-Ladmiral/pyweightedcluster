"""Cross-project parity helpers: pyweightedcluster vs traminex.

This project delegates sequence distance and cost behavior to traminex. The
comparison tests route identical inputs through pyweightedcluster's public
delegators, traminex, and, where available, the R reference (TraMineR via
rpy2). The acceptance test is agreement with the R source, not merely agreement
between the two Python entry points.

This module provides the traminex side of that harness:

- :func:`import_traminex_or_skip` / the ``traminex_mod`` fixture — gate tests
  on traminex being importable. traminex is a *local* project (not on PyPI),
  installed editable into this project's environment (AGENTS.md, "Standard
  Commands"); when it is absent the tests skip, or fail when
  ``WC_PARITY_STRICT=1`` (same strict switch as ``r_parity_utils``).
- :func:`to_traminex_sts` — convert this project's plain integer/string state
  matrices (plus optional per-sequence weights) into a traminex sequence
  object ready for ``traminex.seqdist`` / cost functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from .r_parity_utils import is_strict_parity_mode

# The sibling checkout sits beside this repository, whether the family lives side by
# side in the projects folder or inside its parent folder, so the path is computed.
TRAMINEX_ROOT = Path(__file__).resolve().parents[2] / "PROJECT_TRAMINEX"

_TRAMINEX_MISSING = (
    "traminex is not importable. It is a local dependency: install it editable "
    "into this project's environment first: "
    f".venv/bin/python -m pip install --no-deps --no-build-isolation -e {TRAMINEX_ROOT}"
)


def import_traminex_or_skip() -> Any:
    """Return the traminex module, or skip (fail in strict mode) if absent."""
    try:
        import traminex
    except ImportError:
        if is_strict_parity_mode():
            pytest.fail(_TRAMINEX_MISSING)
        pytest.skip(_TRAMINEX_MISSING)
    return traminex


@pytest.fixture(scope="module")
def traminex_mod() -> Any:
    return import_traminex_or_skip()


def to_traminex_sts(
    traminex: Any,
    sequences: np.ndarray,
    *,
    alphabet: list | None = None,
    weights: np.ndarray | None = None,
) -> Any:
    """Build a traminex STSList from a (N, T) state matrix.

    States may be ints or strings; they are normalized to strings the same way
    ``traminex.seqdef`` encodes them. ``alphabet`` defaults to the sorted
    unique states of the input, which matches TraMineR's ``seqdef`` default.
    """
    mat = np.asarray(sequences)
    if mat.ndim != 2:
        raise ValueError(f"sequences must be a 2D (N, T) matrix, got shape {mat.shape}")
    mat_str = mat.astype(str)
    if alphabet is None:
        alphabet_str = sorted(np.unique(mat_str).tolist())
    else:
        alphabet_str = [str(a) for a in alphabet]
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    return traminex.seqdef(mat_str, alphabet=alphabet_str, weights=w)

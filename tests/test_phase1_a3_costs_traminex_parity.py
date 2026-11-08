"""Phase 1 / A3 — substitution & indel costs converge on traminex (R-anchored).

MIGRATION_PLAN.md Phase 1 replaces this project's private cost code
(``weightedcluster.costs``) with traminex's R-validated, weight-aware cost
functions. Per the ground rules the acceptance test is **agreement with the
current live R source (TraMineR 2.2-14)**, not with either Python copy:

- The load-bearing assertions compare **traminex** ``seqsubm(method=...)`` to R
  ``TraMineR::seqcost(...)`` for CONSTANT and TRATE costs, including the
  **weighted** TRATE path (``seqdef(weights=)`` / ``seqtrate(weighted=TRUE)``),
  which is exactly where two independent copies are most likely to diverge.
At migration time the pre-migration local ``weightedcluster.costs`` TRATE
matrix was verified to reproduce R's costs by an independent route (weighted and
unweighted), confirming the convergence does not change the numbers for these
inputs; the local module was then removed in the same commit.

R TRATE (non-time-varying), from ``SOURCES/TraMineR/R/seqcost.R``:
``cost[i,j] = cval - tr[i,j] - tr[j,i]`` with ``cval=2`` and
``indel = 0.5 * max(cost)``; ``tr`` is the (optionally weighted) transition-rate
matrix. traminex's ``seqsubm(method="TRATE")`` (via
``trate_substitution_matrix(mode="symmetric")``) computes the same thing with
**no smoothing** (``smoothing=0``) and zeroes any absent-source row exactly as
R's ``seqtrate`` (``PA == 0``) — bit-faithful to R since traminex 0.0.8, which
dropped the earlier non-R default ``smoothing=1e-9`` (that default both
perturbed every rate by ~1e-9 and turned an absent-source row into a uniform
``1/K`` row, biasing those costs). The project's documented indel convention is
also ``0.5 * max(sm)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from .r_parity_utils import _HAS_RPY2, is_strict_parity_mode
from .traminex_parity_utils import import_traminex_or_skip

# Small 0-based state matrices (alphabet {0,1,2}) plus per-sequence weights.
_SEQS = np.array(
    [
        [0, 0, 0, 1, 1, 2, 2, 2],
        [0, 1, 0, 1, 2, 1, 2, 0],
        [2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [2, 0, 2, 0, 1, 2, 1, 0],
    ],
    dtype=int,
)
_WEIGHTS = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.0], dtype=float)

_ATOL = 1e-8


def _letters(seqs: np.ndarray) -> np.ndarray:
    """Map 0-based codes to letters A, B, C... so R's seqdef alphabet order matches."""
    return np.vectorize(lambda c: chr(ord("A") + int(c)))(seqs)


def _r_seqcost(method: str, *, weighted: bool, cval: float | None = None):
    """Return (sm, indel) from TraMineR::seqcost for the shared sequences/weights.

    Skips (or fails in strict mode) when rpy2 or R/TraMineR is unavailable.
    """
    if not _HAS_RPY2:
        msg = "rpy2 not available for R-anchored A3 cost parity"
        pytest.fail(msg) if is_strict_parity_mode() else pytest.skip(msg)
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    try:
        importr("TraMineR")
    except Exception as e:  # pragma: no cover - environment dependent
        msg = f"TraMineR not importable in R: {e}"
        pytest.fail(msg) if is_strict_parity_mode() else pytest.skip(msg)

    letters = _letters(_SEQS)
    nrow, ncol = letters.shape
    ro.r.assign(
        "seqs",
        ro.r.matrix(
            ro.StrVector([str(x) for x in letters.ravel(order="C")]),
            nrow=nrow,
            ncol=ncol,
            byrow=True,
        ),
    )
    ro.r.assign("w", ro.FloatVector([float(x) for x in _WEIGHTS]))
    ro.r("suppressMessages(library(TraMineR))")
    ro.r('sts <- suppressMessages(seqdef(seqs, alphabet=c("A","B","C"), weights=w))')
    cval_expr = "NULL" if cval is None else f"{float(cval)}"
    ro.r(
        f"sc <- suppressMessages(seqcost(sts, method='{method}', "
        f"weighted={'TRUE' if weighted else 'FALSE'}, cval={cval_expr}))"
    )
    K = int(ro.r("nrow(sc$sm)")[0])
    sm = np.asarray(ro.r("as.numeric(sc$sm)"), dtype=float).reshape((K, K), order="F")
    indel = float(np.asarray(ro.r("as.numeric(sc$indel)"), dtype=float).reshape(-1)[0])
    return sm, indel


@pytest.mark.parametrize("weighted", [False, True])
def test_traminex_trate_matches_r(weighted):
    """traminex TRATE substitution costs == R seqcost(method='TRATE'); weighted path included."""
    traminex = import_traminex_or_skip()
    w = _WEIGHTS if weighted else None

    sm_tx = np.asarray(traminex.seqsubm(_SEQS, method="TRATE", weights=w), dtype=float)
    indel_tx = 0.5 * float(sm_tx.max())

    sm_r, indel_r = _r_seqcost("TRATE", weighted=weighted)

    assert sm_tx.shape == sm_r.shape
    np.testing.assert_allclose(sm_tx, sm_r, atol=_ATOL)
    assert indel_tx == pytest.approx(indel_r, abs=_ATOL)


def test_traminex_constant_matches_r():
    """traminex CONSTANT substitution costs == R seqcost(method='CONSTANT', cval=2)."""
    traminex = import_traminex_or_skip()

    sm_tx = np.asarray(traminex.seqsubm(_SEQS, method="CONSTANT", cval=2.0), dtype=float)
    sm_r, _indel_r = _r_seqcost("CONSTANT", weighted=False, cval=2.0)

    np.testing.assert_allclose(sm_tx, sm_r, atol=_ATOL)

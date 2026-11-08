"""Phase 1 / A2 — OMspell converges on traminex (R-anchored).

MIGRATION_PLAN.md Phase 1 / A2 removes this project's private ``_omspell`` kernel
and routes ``sequences_distance(metric="OMSPELL")`` through traminex's R-faithful
OMspell (``traminex.distances.pairwise(method="OMSPELL")``). The acceptance test
is agreement with TraMineR 2.2-14, not the old Python copy — which in fact was
**not** R-faithful (it omitted R's ``-1`` spell-duration shift), so converging it
is a correctness fix as well as a de-duplication.

Mapping pinned here: this project's ``timecost`` is R/traminex ``expcost``,
durations use ``tpow=1``, and ``normalize=True`` == R ``norm="maxlength"`` while
``normalize=False`` == R ``norm="none"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from weightedcluster._seqdist import sequences_distance, sequences_distance_between

from .r_parity_utils import _HAS_RPY2, is_strict_parity_mode

_SEQS = np.array(
    [
        [0, 0, 0, 1, 1, 2, 2, 2],
        [0, 1, 0, 1, 2, 1, 2, 0],
        [2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [2, 0, 2, 0, 1, 2, 1, 0],
    ],
    dtype=np.int32,
)
_ATOL = 1e-9


def _r_session():
    if not _HAS_RPY2:
        msg = "rpy2 not available for R-anchored A2 OMspell parity"
        pytest.fail(msg) if is_strict_parity_mode() else pytest.skip(msg)
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    try:
        importr("TraMineR")
    except Exception as e:  # pragma: no cover - environment dependent
        msg = f"TraMineR not importable in R: {e}"
        pytest.fail(msg) if is_strict_parity_mode() else pytest.skip(msg)

    letters = np.vectorize(lambda c: chr(ord("A") + int(c)))(_SEQS)
    nrow, ncol = letters.shape
    ro.r.assign(
        "seqs",
        ro.r.matrix(ro.StrVector(list(letters.ravel())), nrow=nrow, ncol=ncol, byrow=True),
    )
    ro.r("suppressMessages(library(TraMineR))")
    ro.r('sts <- suppressMessages(seqdef(seqs, alphabet=c("A","B","C")))')
    ro.r("cmat <- matrix(2,3,3); diag(cmat) <- 0")
    return ro


def _r_dist(ro, expr: str) -> np.ndarray:
    ro.r(f"D <- suppressMessages(as.matrix({expr}))")
    K = int(ro.r("nrow(D)")[0])
    return np.asarray(ro.r("as.numeric(D)"), dtype=float).reshape((K, K), order="F")


@pytest.mark.parametrize("expcost", [0.0, 0.5])
@pytest.mark.parametrize("normalize, r_norm", [(True, "maxlength"), (False, "none")])
def test_omspell_matches_r(expcost, normalize, r_norm):
    ro = _r_session()
    wc = sequences_distance(
        _SEQS, metric="OMSPELL", normalize=normalize, indel=1.0, timecost=expcost
    )
    R = _r_dist(
        ro,
        f"seqdist(sts, method='OMspell', sm=cmat, indel=1, expcost={expcost}, norm='{r_norm}')",
    )
    assert wc.shape == R.shape
    np.testing.assert_allclose(np.asarray(wc, dtype=float), R, atol=_ATOL)


def test_omspell_between_matches_full_block():
    full = sequences_distance(_SEQS, metric="OMSPELL", normalize=True, indel=1.0, timecost=0.5)
    between = sequences_distance_between(
        _SEQS[:2], _SEQS[2:], metric="OMSPELL", normalize=True, indel=1.0, timecost=0.5
    )
    np.testing.assert_allclose(between, full[:2, 2:], atol=_ATOL)

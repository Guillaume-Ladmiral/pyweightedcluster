"""Phase 1 / A1 — sequence distances converge on traminex (R-anchored).

MIGRATION_PLAN.md Phase 1 routes this project's distances (``_seqdist``) through
traminex instead of its private kernels. The acceptance test is agreement with
the current live R source (TraMineR 2.2-14), not with the old Python copy.

These tests pin the public ``sequences_distance`` path (now traminex-backed) to
``TraMineR::seqdist`` and — crucially — record, per metric, **which R
normalization** ``normalize=True`` corresponds to. That is the low-level helper
mapping; clustering workflows may select normalized or raw forms from their
configuration. In particular, ``seqclararange`` maps ``norm="none"`` to
``normalize=False``, ``norm="length"`` to ``normalize=True``, and
``norm="maxlength"`` to a raw distance plus an explicit length scale. The
low-level mapping is:

- ``HAM``  : Hamming with unit mismatch cost.
  ``normalize=True``  == R ``seqdist(method="HAM", sm=unit, norm="maxlength")``;
  ``normalize=False`` == ``norm="none"`` (raw mismatch count).
- ``OM``   : indel=1, constant substitution cost 2 by default.
  ``normalize=True``  == R ``seqdist(method="OM", sm=2, indel=1, norm="maxlength")``;
  ``normalize=False`` == ``norm="none"``.
- ``LCS``  : ``normalize=True`` == R ``seqdist(method="LCS", norm="gmean")``
  (== ``"auto"`` for fixed-length sequences); ``normalize=False`` ==
  ``norm="none"`` (raw ``|a|+|b|-2*lcs``).
- ``DHD``  : position-cost-weighted Hamming. With a ``(T,S,S)`` cost array it
  equals R ``seqdist(method="DHD", sm=...)``; with a ``(S,S)`` matrix it equals
  R ``seqdist(method="HAM", sm=...)``. (traminex's explicit-subm DHD path is
  itself R-golden tested in traminex; here we re-pin end-to-end.)

Note: the pre-migration local copy length-normalized HAM the same way (the
Phase 0 finding), so HAM agreeing proved nothing on its own — the load-bearing
checks here are against R, and they also lock OM/LCS/DHD, where the old copy's
``normalize=False`` LCS was a non-R-faithful half of R's raw value.
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
        msg = "rpy2 not available for R-anchored A1 distance parity"
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
    ro.r("umat <- matrix(1,3,3); diag(umat) <- 0")
    return ro


def _r_dist(ro, expr: str) -> np.ndarray:
    ro.r(f"D <- suppressMessages(as.matrix({expr}))")
    K = int(ro.r("nrow(D)")[0])
    return np.asarray(ro.r("as.numeric(D)"), dtype=float).reshape((K, K), order="F")


def _r_asymmetric_distance(rows: np.ndarray, *, method: str) -> np.ndarray:
    """Return TraMineR's warned asymmetric-cost result for a two-state corpus."""
    ro = _r_session()
    letters = np.vectorize(lambda state: chr(ord("A") + int(state)))(rows)
    nrow, ncol = letters.shape
    ro.r.assign(
        "asymmetric_rows",
        ro.r.matrix(
            ro.StrVector(list(letters.ravel())),
            nrow=nrow,
            ncol=ncol,
            byrow=True,
        ),
    )
    ro.r("asymmetric_sts <- suppressMessages(" "seqdef(asymmetric_rows, alphabet=c('A', 'B')))")
    ro.r("asymmetric_sm <- matrix(c(0, 1, 2, 0), nrow=2, byrow=TRUE)")
    if method == "OM":
        expression = (
            "seqdist(asymmetric_sts, method='OM', sm=asymmetric_sm, " "indel=2, norm='none')"
        )
    else:
        expression = "seqdist(asymmetric_sts, method='HAM', sm=asymmetric_sm, norm='none')"
    ro.r(f"asymmetric_D <- suppressWarnings(as.matrix({expression}))")
    return np.asarray(ro.r("as.numeric(asymmetric_D)"), dtype=float).reshape(
        (nrow, nrow), order="F"
    )


@pytest.mark.parametrize(
    "metric, normalize, r_expr",
    [
        ("HAM", True, "seqdist(sts, method='HAM', sm=umat, norm='maxlength')"),
        ("HAM", False, "seqdist(sts, method='HAM', sm=umat, norm='none')"),
        ("OM", True, "seqdist(sts, method='OM', sm=cmat, indel=1, norm='maxlength')"),
        ("OM", False, "seqdist(sts, method='OM', sm=cmat, indel=1, norm='none')"),
        ("LCS", True, "seqdist(sts, method='LCS', norm='gmean')"),
        ("LCS", False, "seqdist(sts, method='LCS', norm='none')"),
    ],
)
def test_distance_matches_r(metric, normalize, r_expr):
    ro = _r_session()
    kwargs = {"indel": 1.0} if metric == "OM" else {}
    wc = sequences_distance(_SEQS, metric=metric, normalize=normalize, **kwargs)
    R = _r_dist(ro, r_expr)
    assert wc.shape == R.shape
    np.testing.assert_allclose(np.asarray(wc, dtype=float), R, atol=_ATOL)


def test_dhd_static_subm_matches_r_ham_with_sm():
    """DHD with a (S,S) cost matrix == R seqdist(method='HAM', sm=...) (raw)."""
    ro = _r_session()
    sm = np.full((3, 3), 2.0)
    np.fill_diagonal(sm, 0.0)
    wc = sequences_distance(_SEQS, metric="DHD", substitution_costs=sm, normalize=False)
    R = _r_dist(ro, "seqdist(sts, method='HAM', sm=cmat, norm='none')")
    np.testing.assert_allclose(np.asarray(wc, dtype=float), R, atol=_ATOL)


def test_between_matches_full_block():
    """sequences_distance_between reproduces the cross block of the full matrix."""
    full = sequences_distance(_SEQS, metric="OM", normalize=True, indel=1.0)
    between = sequences_distance_between(
        _SEQS[:2], _SEQS[2:], metric="OM", normalize=True, indel=1.0
    )
    np.testing.assert_allclose(between, full[:2, 2:], atol=_ATOL)


@pytest.mark.parametrize("metric,r_method", [("OM", "OM"), ("DHD", "HAM")])
@pytest.mark.parametrize(
    "rows,directional_cost",
    [
        (np.array([[0], [1], [0]], dtype=np.int32), 1.0),
        (np.array([[1], [0], [1]], dtype=np.int32), 2.0),
    ],
)
def test_asymmetric_costs_match_live_r_stable_first_occurrence_orientation(
    metric, r_method, rows, directional_cost
):
    """Duplicate rows must inherit TraMineR's first-distinct-pair orientation."""
    substitution = np.array([[0.0, 1.0], [2.0, 0.0]])
    kwargs = {
        "metric": metric,
        "substitution_costs": substitution,
        "normalize": False,
    }
    if metric == "OM":
        kwargs["indel"] = 2.0

    with pytest.warns(UserWarning, match="Non-symmetric substitution cost matrix!"):
        observed = sequences_distance(rows, **kwargs)

    expected = _r_asymmetric_distance(rows, method=r_method)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(observed[0], observed[2])
    assert observed[0, 1] == directional_cost


def _r_asymmetric_refseq_list(left: np.ndarray, right: np.ndarray, *, method: str) -> np.ndarray:
    """Return TraMineR's ``seqdist(refseq=list(a, b))`` for the warned asymmetric costs."""
    ro = _r_session()
    rows = np.vstack([left, right])
    letters = np.vectorize(lambda state: chr(ord("A") + int(state)))(rows)
    nrow, ncol = letters.shape
    ro.r.assign(
        "refseq_rows",
        ro.r.matrix(ro.StrVector(list(letters.ravel())), nrow=nrow, ncol=ncol, byrow=True),
    )
    ro.r("refseq_sts <- suppressMessages(seqdef(refseq_rows, alphabet=c('A', 'B')))")
    ro.r("refseq_sm <- matrix(c(0, 1, 2, 0), nrow=2, byrow=TRUE)")
    sets = f"refseq=list(1:{left.shape[0]}, {left.shape[0] + 1}:{nrow})"
    indel = ", indel=2" if method == "OM" else ""
    ro.r(
        "refseq_D <- suppressWarnings(suppressMessages(as.matrix("
        f"seqdist(refseq_sts, method='{method}', sm=refseq_sm{indel}, norm='none', {sets}))))"
    )
    return np.asarray(ro.r("as.numeric(refseq_D)"), dtype=float).reshape(
        (left.shape[0], right.shape[0]), order="F"
    )


@pytest.mark.parametrize("metric, r_method", [("OM", "OM"), ("DHD", "HAM")])
def test_asymmetric_between_set_distance_follows_r_refseq_list(metric, r_method):
    """Distances between two sets are R's ``refseq=list(a, b)``: d(first-set row, second-set row).

    WeightedCluster's CLARA computes the distances to the medoids this way
    (``clara_clustrange.R:133, 339, 458``), and the medoids are rows of the first set.
    With asymmetric costs and a second set that repeats rows of the first, R's full
    matrix keeps the direction of each pair's first occurrence instead, so the cross
    block of the full matrix is not the between-set result. traminex follows R since
    TX-AUD-20261019-009; the earlier provider gave the cross block. Two positions,
    because R's ``refseq=list`` stops on one-position sequences.
    """
    left = np.array([[1, 1], [0, 0]], dtype=np.int32)
    right = np.array([[1, 1], [0, 0]], dtype=np.int32)
    substitution = np.array([[0.0, 1.0], [2.0, 0.0]])
    kwargs = {
        "metric": metric,
        "substitution_costs": substitution,
        "normalize": False,
    }
    if metric == "OM":
        kwargs["indel"] = 2.0

    with pytest.warns(UserWarning, match="Non-symmetric substitution cost matrix!"):
        between = sequences_distance_between(left, right, **kwargs)

    np.testing.assert_array_equal(between, _r_asymmetric_refseq_list(left, right, method=r_method))
    assert between[1, 0] == 2.0  # A-A to B-B: two substitutions A -> B at cost 1

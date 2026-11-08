"""Phase 0.1 smoke tests: the traminex-side parity harness works.

These tests validate harness mechanics only (import gate, state-matrix
conversion, a traminex distance call round-trip). The actual old-local vs
traminex vs R comparison tests are Phase 1 work (MIGRATION_PLAN.md) and are
deliberately NOT started here.
"""

from __future__ import annotations

import numpy as np

from .traminex_parity_utils import to_traminex_sts

pytest_plugins = ["tests.traminex_parity_utils"]

SEQS = np.array(
    [
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 2, 2, 2],
        [2, 2, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)


def test_to_traminex_sts_builds_sequence_object(traminex_mod):
    sts = to_traminex_sts(traminex_mod, SEQS)
    assert sts.data.shape == SEQS.shape
    assert sts.alphabet == ["0", "1", "2"]


def test_to_traminex_sts_accepts_weights(traminex_mod):
    weights = np.array([1.0, 2.0, 0.5, 1.5])
    sts = to_traminex_sts(traminex_mod, SEQS, weights=weights)
    assert sts.data.shape == SEQS.shape
    np.testing.assert_allclose(np.asarray(sts.weights, dtype=float), weights)


def test_traminex_distance_call_round_trip(traminex_mod):
    sts = to_traminex_sts(traminex_mod, SEQS)
    D = np.asarray(traminex_mod.seqdist(sts, method="HAM"), dtype=float)
    assert D.shape == (4, 4)
    np.testing.assert_array_equal(np.diag(D), np.zeros(4))
    np.testing.assert_array_equal(D, D.T)
    # Invariants only (no normalization assumption): rows 0 and 1 differ at 2
    # positions, rows 0 and 3 at 4, so D[0,1] < D[0,3] under any scaling.
    # NOTE for Phase 1: traminex HAM is currently length-normalized with no
    # norm kwarg, whereas R seqdist(method="HAM") defaults to raw counts —
    # the Phase 1 comparison tests must pin this against the R source.
    assert 0.0 < D[0, 1] < D[0, 3]

"""Phase 2.5 (DUPLICATES A5-A8): inline weighted maths -> traminex primitives.

The faithful weighted pairwise-sum / inertia maths formerly lived *inline* in
weightedcluster (seqtree._weighted_pairwise_sum, _utils.pairwise_sum, and
quality's T/W moment loops). It now
lives in traminex (`traminex.diss.weighted_pairwise_sum` /
`weighted_inertia`), where it is anchored to R TraMineR 2.2-14
(tmrWeightedInertiaDist; see traminex tests/test_r_goldens_dissvar.py etc.).

These tests keep verbatim copies of the removed inline maths as *reference
implementations* and prove the traminex primitives (and the refactored
weightedcluster call sites) reproduce them exactly — the cross-check required
by MIGRATION_PLAN.md Phase 2 before the local maths was deleted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.r_parity_utils import import_optional_or_skip

traminex_diss = import_optional_or_skip("traminex.diss", allow_module_level=True)


def _random_case(seed, n=23, k=4):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 2)) * 3.0
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(D, 0.0)
    w = rng.uniform(0.2, 3.0, size=n)
    labels1b = rng.integers(1, k + 1, size=n)
    return D, w, labels1b


# --- verbatim copies of the removed inline maths (reference implementations) –


def _ref_seqtree_weighted_pairwise_sum(D, idx, weights):
    # formerly weightedcluster.seqtree._weighted_pairwise_sum
    if idx.size <= 1:
        return 0.0
    sub = D[np.ix_(idx, idx)]
    ws = weights[idx].reshape(-1)
    wprod = ws[:, None] * ws[None, :]
    return float(0.5 * np.sum(wprod * sub))


def _ref_utils_pairwise_sum(D, idx):
    # formerly weightedcluster._utils.pairwise_sum
    if idx.size <= 1:
        return 0.0
    sub = D[np.ix_(idx, idx)]
    iu = np.triu_indices(idx.size, k=1)
    return float(sub[iu].sum())


def _ref_r_quality_terms(D, labels0, weights):
    """Port the five affected formulas from WeightedCluster clusterqualitybody.cpp."""
    k = int(labels0.max()) + 1
    sizes = np.zeros(k, dtype=float)
    errors = np.zeros(k, dtype=float)
    errors2 = np.zeros(k, dtype=float)
    buckets: dict[float, float] = {0.0: 0.0}
    wx = wx2 = wy = wxy = 0.0

    for i in range(D.shape[0]):
        ci = int(labels0[i])
        sizes[ci] += weights[i]
        if weights[i] <= 0.0:
            continue
        diagonal_mass = float(weights[i] * weights[i])
        wy += diagonal_mass
        buckets[0.0] += diagonal_mass
        for j in range(i + 1, D.shape[0]):
            if weights[j] <= 0.0:
                continue
            mass = float(2.0 * weights[i] * weights[j])
            distance = float(D[i, j])
            weighted_distance = mass * distance
            wx += weighted_distance
            wx2 += weighted_distance * distance
            buckets[distance] = buckets.get(distance, 0.0) + mass
            if labels0[i] == labels0[j]:
                errors[ci] += weighted_distance
                errors2[ci] += weighted_distance * distance
                wxy += weighted_distance
                wy += mass

    total_weight = float(sizes.sum())
    ss_res = float(np.sum(errors / sizes))
    ss2_res = float(np.sum(errors2 / sizes))
    ss_expl = wx / total_weight - ss_res
    ss2_expl = wx2 / total_weight - ss2_res
    ch = (ss_expl / (k - 1.0)) / (ss_res / (total_weight - k))
    r2 = ss_expl / (ss_res + ss_expl)
    chsq = (ss2_expl / (k - 1.0)) / (ss2_res / (total_weight - k))
    r2sq = ss2_expl / (ss2_res + ss2_expl)

    def _extreme_sum(*, reverse: bool) -> float:
        remaining = wy
        total = 0.0
        for distance, mass in sorted(buckets.items(), reverse=reverse):
            assigned = min(remaining, mass)
            total += assigned * distance
            remaining -= assigned
            if remaining <= 0.0:
                break
        return total

    smin = _extreme_sum(reverse=False)
    smax = _extreme_sum(reverse=True)
    hc = (wxy - smin) / (smax - smin)
    return {"CH": ch, "R2": r2, "CHsq": chsq, "R2sq": r2sq, "HC": hc}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_traminex_primitive_matches_old_seqtree_inline_maths(seed):
    D, w, _ = _random_case(seed)
    rng = np.random.default_rng(seed + 100)
    for size in (2, 5, 11, D.shape[0]):
        idx = np.sort(rng.choice(D.shape[0], size=size, replace=False))
        ref = _ref_seqtree_weighted_pairwise_sum(D, idx, w)
        new = traminex_diss.weighted_pairwise_sum(D, idx, w)
        np.testing.assert_allclose(new, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [3, 4])
def test_traminex_primitive_matches_old_utils_pairwise_sum(seed):
    from weightedcluster._utils import pairwise_sum

    D, _, _ = _random_case(seed)
    rng = np.random.default_rng(seed + 200)
    for size in (0, 1, 2, 7, D.shape[0]):
        idx = np.sort(rng.choice(D.shape[0], size=size, replace=False))
        ref = _ref_utils_pairwise_sum(D, idx)
        np.testing.assert_allclose(
            traminex_diss.weighted_pairwise_sum(D, idx), ref, rtol=1e-12, atol=1e-12
        )
        # The refactored weightedcluster._utils.pairwise_sum delegates there.
        np.testing.assert_allclose(pairwise_sum(D, idx), ref, rtol=1e-12, atol=1e-12)


def test_quality_fallback_matches_r_ordered_pair_formulas(monkeypatch):
    # Force the pure-Python fallback and compare the five formerly incorrect
    # statistics with WeightedCluster's ordered-pair formulas.
    import sys
    import types

    import weightedcluster.quality as qmod

    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", types.ModuleType("_wc_core"))
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", types.ModuleType("_numba"))

    D, w, labels1b = _random_case(7, n=18, k=3)
    out = qmod.wcClusterQuality(D, labels1b, weights=w)

    _, labels0 = np.unique(labels1b, return_inverse=True)
    expected = _ref_r_quality_terms(D, labels0, w)

    stats = out["stats"]
    for key, value in expected.items():
        np.testing.assert_allclose(stats[key], value, rtol=1e-9, atol=1e-10)

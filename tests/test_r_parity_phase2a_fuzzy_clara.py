from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import seqclararange
from weightedcluster._seqdist import sequences_distance
from weightedcluster.fuzzy import crispness, fcmdd
from weightedcluster.quality import wcFuzzyQuality

pytest_plugins = ["tests.r_parity_utils"]


# Small fixed sequence dataset -> distance matrix shared by the deterministic tests.
_X = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
    ],
    dtype=int,
)


def _diss():
    return sequences_distance(_X, metric="LCS", normalize=True)


def test_r_oracle_exposes_fuzzy_internals(r_parity_session):
    # Verify the internal R primitives this phase depends on are reachable.
    assert r_parity_session.r_internal_exists("crispness")
    assert r_parity_session.r_internal_exists("wfcmdd")


def _example_membership():
    # A deterministic, non-degenerate (N=6, K=3) membership matrix.
    U = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.2, 0.6],
        ],
        dtype=float,
    )
    return U / U.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("norm", [True, False])
def test_crispness_matches_r(r_parity_session, norm):
    U = _example_membership()
    py = np.asarray(crispness(U, norm=norm), dtype=float)
    r = np.asarray(r_parity_session.r_crispness(U, norm=norm), dtype=float)
    assert np.allclose(py, r, atol=1e-10, rtol=1e-8)


def test_fpc_equals_weighted_mean_of_unnormalized_r_crispness(r_parity_session):
    # wcFuzzyQuality FPC is the (weight-normalized) mean of sum_j u_ij^2, which is
    # exactly R's unnormalized crispness averaged over observations.
    U = _example_membership()
    w = np.array([1.0, 2.0, 1.0, 0.5, 1.5, 1.0])
    q = wcFuzzyQuality(_diss(), U, m=1.5, weights=w)
    r_cr = np.asarray(r_parity_session.r_crispness(U, norm=False), dtype=float)
    fpc_from_r = float(np.sum(w * r_cr) / w.sum())
    assert np.isclose(float(q["stats"]["FPC"]), fpc_from_r, atol=1e-10, rtol=1e-8)


def _crisp_init_membership(D, k, seed):
    # Build a crisp (0/1) initial membership from this package's weighted medoids,
    # so Python fcmdd and R wfcmdd start from the *same* memb matrix.
    try:
        from weightedcluster._wc_core import weighted_medoids as wm
    except ImportError:
        from weightedcluster._fallback import weighted_medoids as wm  # type: ignore
    labels, _med, _ = wm(D, int(k), weights=None, max_iter=50, n_init=2, random_state=int(seed))
    U0 = np.zeros((D.shape[0], int(k)), dtype=float)
    for c in range(int(k)):
        U0[labels == c, c] = 1.0
    return U0


@pytest.mark.parametrize("weights_kind", ["unit", "nonuniform"])
@pytest.mark.parametrize("k", [2, 3])
def test_fcmdd_functional_matches_r_wfcmdd(r_parity_session, k, weights_kind):
    D = _diss()
    N = D.shape[0]
    U0 = _crisp_init_membership(D, k, seed=42)
    # Weights are passed explicitly to *both* sides. Python fcmdd treats
    # weights=None as ones, and the installed oracle (WeightedCluster 2.0) does
    # the same -- its wfcmdd opens with `if (is.null(weights)) weights <- rep(1,
    # n)`. (The vendored 1.8-1 source had no such guard and would error on NULL;
    # passing weights explicitly keeps the comparison symmetric and
    # version-independent.) The non-uniform case is load-bearing for the
    # raw-weights convention: R computes `pweights <- weights/sum(weights)` but
    # the medoid search and the functional both use the *raw* `weights`, so an
    # exact functional match (not off by sum(weights)) pins Python to raw weights.
    if weights_kind == "unit":
        w = np.ones(N, dtype=float)
    else:
        w = np.array([1.0, 2.0, 1.0, 0.5, 1.5, 1.0], dtype=float)
        assert w.shape == (N,)

    py = fcmdd(D, U0, weights=w, method="FCMdd", m=1.5)
    r = r_parity_session.r_wfcmdd(D, U0, weights=w, method="FCMdd", m=1.5)

    # The objective functional is permutation-invariant -> the load-bearing check.
    assert np.isclose(float(py["functional"]), float(r["functional"]), atol=1e-8, rtol=1e-6)

    # Medoid sets must coincide (Python 0-based, R 1-based).
    py_med = set(int(x) for x in py["mobileCenters"])
    r_med = set(int(x) - 1 for x in r["mobile_centers_1b"])
    assert py_med == r_med


# A fixed 10-sequence dataset and a user-supplied *arbitrary* crisp init (not a
# k-medoids solution). R's wfcmdd medoid search only considers points that are
# not fully committed to another cluster (`apply(u[, -k], 1, max) < 1`, wfcmdd
# line 63 in both 1.8-1 and 2.0). On this init that rule changes the medoid
# chosen on the first sweep, so the fuzzy run converges to a different,
# R-faithful optimum: the functional gap is ~0.19 (well clear of any FP tie),
# making this a robust load-bearing check of fcmdd's candidate set.
_X_FCMDD_ARB = np.array(
    [
        [0, 0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 2, 2, 1],
        [2, 2, 1, 0, 0, 0, 1],
        [2, 1, 2, 0, 1, 0, 0],
        [1, 0, 2, 1, 0, 1, 2],
        [1, 2, 0, 2, 1, 0, 2],
        [0, 2, 1, 1, 2, 0, 1],
        [2, 0, 0, 2, 2, 1, 0],
        [1, 1, 2, 0, 0, 2, 1],
        [0, 2, 2, 1, 0, 1, 0],
    ],
    dtype=int,
)
_FCMDD_ARBITRARY_INIT_LABELS = (0, 1, 0, 2, 2, 0, 1, 2, 1, 1)


def test_fcmdd_matches_r_on_arbitrary_user_init(r_parity_session):
    D = sequences_distance(_X_FCMDD_ARB, metric="LCS", normalize=True)
    labels = np.asarray(_FCMDD_ARBITRARY_INIT_LABELS, dtype=int)
    k = int(labels.max()) + 1
    U0 = np.zeros((D.shape[0], k), dtype=float)
    U0[np.arange(D.shape[0]), labels] = 1.0

    py = fcmdd(D, U0, method="FCMdd", m=1.5)
    r = r_parity_session.r_wfcmdd(D, U0, method="FCMdd", m=1.5)

    assert np.isclose(float(py["functional"]), float(r["functional"]), atol=1e-8, rtol=1e-6)
    py_med = set(int(x) for x in py["mobileCenters"])
    r_med = set(int(x) - 1 for x in r["mobile_centers_1b"])
    assert py_med == r_med


R_CLARA_STAT_COLUMNS = ("Avg dist", "PBM", "DB", "XB", "AMS", "ARI>0.8", "JC>0.8", "Best iter")

# A larger, all-distinct dataset (16 unique sequences) for the R structural
# contract. With sample_size >= 10 every random subsample keeps enough distinct
# sequences that R's internal cutree(., k=max(kvals)) is always feasible, so the
# check cannot fail on a degenerate draw regardless of the shared R RNG state.
_X_CLARA = np.array(
    [
        [1, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 2, 1, 1],
        [1, 1, 0, 1, 0, 1],
        [0, 2, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 1],
        [0, 1, 0, 2, 0, 1],
        [0, 2, 1, 1, 0, 0],
        [2, 0, 1, 1, 0, 0],
        [2, 2, 2, 1, 0, 0],
        [2, 2, 1, 2, 0, 0],
        [2, 1, 2, 2, 1, 0],
        [1, 2, 2, 2, 1, 0],
        [1, 1, 0, 2, 1, 0],
        [1, 1, 2, 0, 1, 0],
    ],
    dtype=int,
)


def test_seqclararange_is_seed_reproducible():
    kw = dict(
        R=5,
        sample_size=6,
        kvals=[2, 3],
        seqdist_args={"method": "LCS"},
        criteria=("distance",),
        random_state=7,
    )
    a = seqclararange(_X, **kw)
    b = seqclararange(_X, **kw)
    assert list(a["kvals"]) == list(b["kvals"]) == [2, 3]
    # Same seed -> identical chosen stats table.
    for stat, col in a["stats"].items():
        assert np.allclose(
            np.asarray(col, dtype=float),
            np.asarray(b["stats"][stat], dtype=float),
            equal_nan=True,
        )


def test_seqclararange_stats_contract_matches_r(r_parity_session):
    # Structural (not value) parity: R's crisp-CLARA stats matrix exposes the
    # documented 8 columns over the requested kvals. seqclararange is stochastic,
    # so we pin the *contract*, not the numbers.
    cols, kvals = r_parity_session.r_seqclararange_stats_contract(
        _X_CLARA, kvals=[2, 3], r_iter=5, sample_size=10, seed=20260615
    )
    assert tuple(cols) == R_CLARA_STAT_COLUMNS
    assert list(kvals) == [2, 3]


# 2026-09-21 targeted audit, M-3: R's wfcmdd sets m <- 1 in its HNCdd branch, so the
# caller's m never reaches HNCdd's medoid search or functional. From a fuzzy initial
# membership the first medoid search depends on it: with the caller's m the audit found
# 22 of 25 runs choosing other medoids. Ten Dirichlet initialisations of a 10-object
# corpus, with m = 2.0 passed to both sides.
@pytest.mark.parametrize("seed", range(10))
def test_hncdd_ignores_m_as_r_wfcmdd_does(r_parity_session, seed):
    D = sequences_distance(_X_FCMDD_ARB, metric="HAM", normalize=False)
    rng = np.random.default_rng(20261107 + seed)
    U0 = rng.dirichlet(np.ones(3), size=D.shape[0])
    dnoise = float(np.quantile(D[np.triu_indices_from(D, 1)], 0.6))

    py = fcmdd(D, U0, method="HNCdd", m=2.0, dnoise=dnoise)
    r = r_parity_session.r_wfcmdd(D, U0, method="HNCdd", m=2.0, dnoise=dnoise)

    assert [int(x) for x in py["mobileCenters"]] == [int(x) - 1 for x in r["mobile_centers_1b"]]
    np.testing.assert_array_equal(py["memb"], r["memb"])
    assert np.isclose(float(py["functional"]), float(r["functional"]), atol=1e-10, rtol=0)


def test_hncdd_accepts_any_finite_m_and_gives_the_same_clustering():
    D = sequences_distance(_X_FCMDD_ARB, metric="HAM", normalize=False)
    U0 = np.random.default_rng(20261107).dirichlet(np.ones(3), size=D.shape[0])
    runs = [fcmdd(D, U0, method="HNCdd", m=m, dnoise=4.0) for m in (0.5, 1.0, 1.5, 2.0, 7.0)]
    for run in runs[1:]:
        assert run["mobileCenters"] == runs[0]["mobileCenters"]
        np.testing.assert_array_equal(run["memb"], runs[0]["memb"])
        assert run["functional"] == runs[0]["functional"]
    with pytest.raises(ValueError, match="finite number"):
        fcmdd(D, U0, method="HNCdd", m=float("nan"), dnoise=4.0)
    with pytest.raises(ValueError, match="greater than one"):
        fcmdd(D, U0, method="FCMdd", m=1.0)

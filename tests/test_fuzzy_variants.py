import numpy as np
import pytest

from weightedcluster._seqdist import sequences_distance
from weightedcluster.fuzzy import fcmdd


def test_fuzzy_noise_mode_smoke():
    # Two obvious groups
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [2, 2, 2, 2],  # potential "noise" far from both clusters
        ],
        dtype=int,
    )
    D = sequences_distance(X, metric="HAM", normalize=True)
    N, K = X.shape[0], 2
    u0 = np.zeros((N, K), dtype=float)
    u0[:2, 0] = 1.0
    u0[2:4, 1] = 1.0
    # Noise distance set to large value; use NCdd
    out = fcmdd(D, u0, method="NCdd", m=1.6, dnoise=0.9, max_iter=30)
    memb = out["memb"]
    # NCdd adds a noise column, so memb shape is (N, K+1)
    assert memb.shape[1] == K + 1
    # Rows should sum to ~1
    assert np.allclose(memb.sum(axis=1), 1.0, atol=1e-6)


def test_fuzzy_pcmdd_mode_smoke():
    # Two clusters, same D as above sans noise
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )
    D = sequences_distance(X, metric="HAM", normalize=True)
    N, K = X.shape[0], 2
    u0 = np.zeros((N, K), dtype=float)
    u0[:2, 0] = 1.0
    u0[2:, 1] = 1.0
    # Set eta roughly to within-cluster average distances to medoids estimate
    eta = np.array([0.1, 0.1], dtype=float)
    out = fcmdd(D, u0, method="PCMdd", m=1.6, eta=eta, max_iter=30)
    memb = out["memb"]
    assert memb.shape == (N, K)
    # Possibilistic memberships are not strictly row-stochastic, but should be bounded in [0,1]
    assert np.all(memb >= 0.0) and np.all(memb <= 1.0)


def test_fuzzy_hncdd_zero_distance_handling():
    # Two identical points and one distinct; HNCdd should hard-assign including possible noise
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=int,
    )
    D = sequences_distance(X, metric="HAM", normalize=True)
    N, K = X.shape[0], 2
    u0 = np.zeros((N, K), dtype=float)
    u0[0, 0] = 1.0
    u0[1, 0] = 1.0
    u0[2, 1] = 1.0
    out = fcmdd(D, u0, method="HNCdd", m=1.6, dnoise=0.9, max_iter=30)
    memb = out["memb"]
    assert memb.shape[1] in (K, K + 1)


def test_fuzzy_hncdd_requires_dnoise():
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )
    D = sequences_distance(X, metric="HAM", normalize=True)
    N, K = X.shape[0], 2
    u0 = np.zeros((N, K), dtype=float)
    u0[:2, 0] = 1.0
    u0[2:, 1] = 1.0

    with pytest.raises(ValueError, match="requires dnoise"):
        fcmdd(D, u0, method="HNCdd", m=1.6, dnoise=None, max_iter=30)

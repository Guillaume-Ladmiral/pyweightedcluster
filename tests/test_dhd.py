import numpy as np
import pytest

from weightedcluster._seqdist import sequences_distance, sequences_distance_between


@pytest.mark.parametrize("normalize", [True, False])
def test_dhd_equals_hamming_when_unit_cost(normalize):
    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )
    # substitution costs 2D with ones off-diagonal -> mismatch cost 1 like Hamming
    S = int(X.max()) + 1
    sm = np.ones((S, S), dtype=float)
    np.fill_diagonal(sm, 0.0)

    D_ham = sequences_distance(X, metric="HAM", normalize=normalize)
    D_dhd = sequences_distance(X, metric="DHD", substitution_costs=sm, normalize=normalize)
    assert np.allclose(D_ham, D_dhd)


def test_dhd_time_varying_changes_values():
    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=int,
    )
    S = 2
    T = X.shape[1]
    sm_t = np.zeros((T, S, S), dtype=float)
    # make mismatches at t=1 much more expensive
    for t in range(T):
        sm_t[t, :, :] = 1.0
        np.fill_diagonal(sm_t[t], 0.0)
    sm_t[1, 0, 1] = sm_t[1, 1, 0] = 5.0

    D = sequences_distance(X, metric="DHD", substitution_costs=sm_t, normalize=False)
    # Only mismatch at t=1 is weighted to 5.0; others equal
    assert D.shape == (2, 2)
    assert D[0, 1] == 5.0


def test_dhd_between_smoke():
    X = np.array([[0, 0, 1], [1, 1, 0]], dtype=int)
    Y = np.array([[0, 1, 1]], dtype=int)
    D = sequences_distance_between(X, Y, metric="DHD", normalize=True)
    assert D.shape == (2, 1)
    assert np.isfinite(D).all()

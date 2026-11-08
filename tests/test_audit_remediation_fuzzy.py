from __future__ import annotations

import numpy as np
import pytest

from weightedcluster.fuzzy import fcmdd

D = np.array(
    [
        [0.0, 1.0, 4.0, 5.0, 7.0],
        [1.0, 0.0, 3.0, 4.0, 6.0],
        [4.0, 3.0, 0.0, 1.0, 3.0],
        [5.0, 4.0, 1.0, 0.0, 2.0],
        [7.0, 6.0, 3.0, 2.0, 0.0],
    ]
)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("FCMdd", {}),
        ("NCdd", {"dnoise": 3.0}),
        ("HNCdd", {"dnoise": 3.0}),
        ("PCMdd", {"eta": np.array([1.5, 2.5])}),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fcmdd_never_mutates_caller_membership(method, kwargs, dtype):
    backing = np.zeros((5, 4), dtype=dtype)
    membership = backing[:, ::2]
    membership[:] = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.0, 1.0]],
        dtype=dtype,
    )
    before = backing.copy()
    fcmdd(D, membership, method=method, max_iter=2, **kwargs)
    np.testing.assert_array_equal(backing, before)


def test_fcmdd_does_not_mutate_input_when_validation_fails_after_coercion():
    membership = np.array([[1.0, 0.0], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.0, 1.0]])
    before = membership.copy()
    with pytest.raises(ValueError, match="eta"):
        fcmdd(D, membership, method="PCMdd", eta=[1.0], max_iter=2)
    np.testing.assert_array_equal(membership, before)


def test_fcmdd_max_iter_is_an_actual_bound(monkeypatch):
    membership = np.array([[1.0, 0.0], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.0, 1.0]])
    original = np.delete
    calls = 0

    def counted_delete(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "delete", counted_delete)
    fcmdd(D, membership, max_iter=1, alpha=0.0)
    assert calls == membership.shape[1]

    calls = 0
    fcmdd(D, membership, max_iter=2, alpha=0.0)
    assert calls == 2 * membership.shape[1]

    calls = 0
    fcmdd(D, membership, max_iter=5, alpha=0.0)
    assert calls == 3 * membership.shape[1]


@pytest.mark.parametrize("max_iter", [0, -1])
def test_fcmdd_rejects_nonpositive_max_iter(max_iter):
    membership = np.array([[1.0, 0.0], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.0, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        fcmdd(D, membership, max_iter=max_iter)

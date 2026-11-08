import numpy as np
import pytest

from weightedcluster import print_kmedoids, wcClusterQuality, wcKMedoids, wcKMedRange


def test_wckmedoids_basic_labels_and_stats():
    # Simple distance matrix for 4 points in two clusters
    # Points 0-1 close; 2-3 close; far across clusters
    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ]
    )

    res = wcKMedoids(D, 2)
    cl = res["clustering"]
    assert cl.shape == (4,)
    # Expect exactly k distinct positive labels
    uniq = np.unique(cl)
    assert len(uniq) == 2
    assert np.all(uniq >= 1)

    # Quality call should succeed with positive-integer labels
    q = wcClusterQuality(D, cl)
    assert "ASW" in q["stats"]
    assert "ASWw" in q["stats"]


def test_wckmedoids_returns_explicit_asw_and_asww_payloads():
    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 1.5, 1.0], dtype=float)

    res = wcKMedoids(D, 2, weights=weights, random_state=42)
    q = wcClusterQuality(D, res["clustering"], weights=weights)

    assert set(res.keys()) >= {"clustering", "info", "stats", "ASW", "ASWw"}
    assert res["ASW"] == q["ASW"]
    assert res["ASWw"] == q["ASWw"]


def test_wckmedrange_forwards_r_method_kwarg():
    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ]
    )
    out = wcKMedRange(D, kvals=[2], method="PAMonce")
    assert out["kvals"] == [2]


@pytest.mark.parametrize(
    ("function", "kwargs", "parameter"),
    [
        (wcKMedoids, {"k": 2.5}, "k"),
        (wcKMedoids, {"k": 2, "npass": 1.5}, "npass"),
        (wcKMedoids, {"k": 2, "debuglevel": True}, "debuglevel"),
        (wcKMedRange, {"kvals": [2], "R": 1.5}, "R"),
        (wcKMedRange, {"kvals": [2], "npass": "1"}, "npass"),
        (wcKMedRange, {"kvals": [2], "R": 2, "samplesize": 2.5}, "samplesize"),
    ],
)
def test_kmedoid_integer_controls_reject_lossy_or_boolean_values(function, kwargs, parameter):
    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ]
    )

    with pytest.raises(ValueError, match=rf"{parameter} must be an integer"):
        function(D, **kwargs)


@pytest.mark.parametrize("digits", [1.5, True, "2"])
def test_print_kmedoids_rejects_lossy_digits(digits):
    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ]
    )
    result = wcKMedoids(D, 2, random_state=7)

    with pytest.raises(ValueError, match="digits must be an integer"):
        print_kmedoids(result, digits=digits)


@pytest.mark.parametrize("digits", [0, 23])
def test_print_kmedoids_rejects_digits_outside_r_print_range(digits):
    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ]
    )
    result = wcKMedoids(D, 2, random_state=7)

    with pytest.raises(ValueError, match="digits must be between 1 and 22"):
        print_kmedoids(result, digits=digits)


def test_wckmedrange_bootstrap_respects_random_state():
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1, 2.1, 2.2],
            [0.2, 0.0, 1.1, 1.0, 2.2, 2.1],
            [1.0, 1.1, 0.0, 0.2, 1.1, 1.2],
            [1.1, 1.0, 0.2, 0.0, 1.2, 1.1],
            [2.1, 2.2, 1.1, 1.2, 0.0, 0.2],
            [2.2, 2.1, 1.2, 1.1, 0.2, 0.0],
        ]
    )

    out1 = wcKMedRange(D, kvals=[2, 3], R=6, random_state=123)
    out2 = wcKMedRange(D, kvals=[2, 3], R=6, random_state=123)

    assert out1["kvals"] == out2["kvals"]
    for k in out1["kvals"]:
        np.testing.assert_array_equal(
            out1["clustering"][f"cluster{k}"], out2["clustering"][f"cluster{k}"]
        )
        for stat in out1["boot"][k].keys():
            np.testing.assert_allclose(out1["boot"][k][stat], out2["boot"][k][stat], equal_nan=True)


def test_wckmedoids_initial_linkage_runtime_errors_propagate(monkeypatch):
    import weightedcluster.kmedoids as kmedoids_module

    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ],
        dtype=float,
    )
    Z = np.array(
        [
            [0.0, 1.0, 0.1, 2.0],
            [2.0, 3.0, 0.2, 2.0],
            [4.0, 5.0, 2.0, 4.0],
        ],
        dtype=float,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("cut_tree exploded")

    monkeypatch.setattr(kmedoids_module, "cut_linkage_by_merge_count", _boom)

    with pytest.raises(RuntimeError, match="cut_tree exploded"):
        wcKMedoids(D, 2, initialclust=Z)

import numpy as np
import pytest

from weightedcluster._seqdist import sequences_distance
from weightedcluster._utils import generate_random_seed
from weightedcluster.fuzzy import crispness, fcmdd, wcFuzzyRange
from weightedcluster.proptree import dtcut, dtlabels, dtprune, wcPropertyClustering
from weightedcluster.quality import wcFuzzyQuality


def test_fuzzy_fcmdd_smoke_hardening():
    # Build a toy distance matrix with two tight groups
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

    # Initialize memberships for K=2 roughly by halves
    N = X.shape[0]
    K = 2
    u0 = np.zeros((N, K), dtype=float)
    u0[:2, 0] = 1.0
    u0[2:, 1] = 1.0

    out = fcmdd(D, u0, method="FCMdd", m=1.6, max_iter=50)
    assert set(out.keys()) >= {"memb", "mobileCenters", "functional"}
    memb = out["memb"]
    assert memb.shape == (N, K) or memb.shape == (N, K + 1)  # in case noise columns appear
    # Membership rows should sum close to 1 for FCMdd
    s = memb.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-6)


def test_proptree_basic_build_and_cut():
    # Simple distances on four items
    D = np.array(
        [
            [0.0, 0.1, 1.2, 1.1],
            [0.1, 0.0, 1.1, 1.2],
            [1.2, 1.1, 0.0, 0.2],
            [1.1, 1.2, 0.2, 0.0],
        ]
    )
    # Two properties to split on; numeric and categorical
    props = {
        "x": np.array([0.1, 0.2, 1.0, 1.1]),
        "grp": np.array(["A", "A", "B", "B"], dtype=object),
    }
    tree = wcPropertyClustering(D, props, maxcluster=None, min_size=0.25, max_depth=3)
    assert "root" in tree and "data" in tree
    leaf_ids = dtcut(tree, k=2, labels=False)
    labels = dtcut(tree, k=2, labels=True)
    assert len(labels) == D.shape[0]
    assert np.asarray(leaf_ids).dtype.kind in ("i", "u")
    assert np.asarray(labels).dtype.kind in ("U", "O")
    pruned = dtprune(tree, k=2)
    lbls = dtlabels(pruned)
    assert len(lbls) == 2
    assert set(np.asarray(labels, dtype=object)).issubset(set(lbls.values()))


def test_wcfuzzyrange_crisp_labels_output():
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
    out = wcFuzzyRange(D, [2], method="FCMdd", m=1.6, crisp_labels=True)
    assert "labels" in out and "cluster2" in out["labels"]


def _noise_fixture():
    X = np.random.default_rng(424242).integers(0, 3, size=(20, 6))
    return (X[:, None] != X[None]).sum(-1).astype(float)


def test_wcfuzzyrange_refuses_noise_variants_up_front():
    """NCdd and HNCdd return a K+1-column membership that wcFuzzyQuality cannot
    take, and no R rule defines FPC/FPE/XBf on a noise column. They used to
    fail deep inside the quality step with an unrelated shape message while
    the docstring advertised them (2026-09-21 targeted audit, M-2). They are
    now refused before any work, under every spelling fcmdd accepts.
    """
    D = _noise_fixture()
    for method, extra in (
        ("NCdd", {"dnoise": 4.0}),
        ("NCdd", {"dlambda": 0.5}),
        ("HNCdd", {"dnoise": 4.0}),
        ("ncdd", {"dnoise": 4.0}),
        ("HNCDD", {"dnoise": 4.0}),
    ):
        with pytest.raises(NotImplementedError, match="fcmdd"):
            wcFuzzyRange(D, [2, 3], method=method, m=1.5, **extra)


def test_wcfuzzyrange_supported_methods_run_and_fcmdd_serves_the_noise_variants():
    """The refusal must not reach the methods that work, and the error's
    advice -- call fcmdd for the noise variants -- must actually hold."""
    D = _noise_fixture()
    for method, extra in (("FCMdd", {}), ("PCMdd", {"eta": np.full(2, 2.0)})):
        stats = wcFuzzyRange(D, [2], method=method, m=1.5, **extra)["stats"]
        assert {"AMS", "FPC", "FPE", "XBf"} <= set(stats)
    u0 = np.random.default_rng(7).dirichlet(np.ones(3), size=D.shape[0])
    for method in ("NCdd", "HNCdd"):
        result = fcmdd(D, u0.copy(), method=method, m=1.5, dnoise=4.0)
        # K real clusters plus the noise column.
        assert np.asarray(result["memb"]).shape == (D.shape[0], 4)


def test_dtlabels_formats_numeric_thresholds_like_r():
    """Numeric split labels must be rendered as R's `format(x, digits = 5)`.

    R's dtlabels uses `format(x, digits = getOption("digits") - 2)`
    (R/propclustering.R:254-255); Python used `format(x, ".5g")`, which prints
    `1.2e+06` where R prints `1200000` and `0.0001` where R prints `1e-04`
    (2026-09-21 targeted audit, L-8). Every expected string below was captured
    from live R `format(x, digits = 5)`, not derived by hand. The live-R
    dtlabels parity test never exercised a break of 1e5 or more, which is why
    it stayed green.
    """
    from weightedcluster.proptree import _r_format_number

    captured_from_r = (
        (1200000, "1200000"),
        (14814804, "14814804"),
        (123456, "123456"),
        (100000, "1e+05"),
        (-1200000, "-1200000"),
        (1234.5, "1234.5"),
        (37.699, "37.699"),
        (0.0014815, "0.0014815"),
        (0.0001, "1e-04"),
        (1.5e-100, "1.5e-100"),
        (0, "0"),
    )
    for value, expected in captured_from_r:
        assert _r_format_number(value) == expected, value

    tree = {
        "root": {
            "id": 1,
            "split": {"var": "income", "breaks": 1200000.0},
            "kids": [{"id": 2}, {"id": 3}],
        }
    }
    assert dtlabels(tree) == {"2": "income <= 1200000", "3": "income > 1200000"}


def test_wcfuzzyquality_weights_change_fuzzy_indices():
    D = np.array(
        [
            [0.0, 0.2, 0.9],
            [0.2, 0.0, 1.0],
            [0.9, 1.0, 0.0],
        ],
        dtype=float,
    )
    U = np.array(
        [
            [0.95, 0.05],
            [0.55, 0.45],
            [0.10, 0.90],
        ],
        dtype=float,
    )

    unweighted = wcFuzzyQuality(D, U, medoids=[0, 2], m=1.6)["stats"]
    weighted = wcFuzzyQuality(
        D,
        U,
        medoids=[0, 2],
        m=1.6,
        weights=np.array([10.0, 1.0, 1.0], dtype=float),
    )["stats"]

    assert weighted["FPC"] > unweighted["FPC"]
    assert weighted["FPE"] < unweighted["FPE"]
    assert np.isfinite(weighted["XBf"])
    assert np.isfinite(weighted["AMS"])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda D, U: crispness(U[:, 0]), "2D"),
        (lambda D, U: crispness(np.array([[np.nan, 1.0]])), "finite"),
        (lambda D, U: wcFuzzyQuality(D, U, medoids=[0.5, 2]), "integer"),
        (
            lambda D, U: wcFuzzyQuality(D, np.array([[0.0, 0.0], *U[1:]])),
            "positive membership",
        ),
        (lambda D, U: wcFuzzyRange(D, [2.5]), "kvals"),
        (lambda D, U: wcFuzzyRange(D, [2], pam_n_init=1.5), "pam_n_init"),
        (lambda D, U: wcFuzzyRange(D, [2], crisp_labels=1), "crisp_labels"),
        (lambda D, U: fcmdd(D, U, m=1.0), "m must be"),
        (lambda D, U: fcmdd(D, U, m="1.5"), "m must be"),
        (lambda D, U: fcmdd(D, U, max_iter=2.5), "max_iter"),
        (lambda D, U: fcmdd(D, U, alpha=np.nan), "alpha"),
        (lambda D, U: fcmdd(D, U, method="NCdd", dnoise=np.nan), "dnoise"),
        (lambda D, U: fcmdd(D, U, method="NCdd", dnoise=-0.1), "dnoise"),
        (lambda D, U: fcmdd(D, U, method="NCdd", dlambda=np.inf), "dlambda"),
        (lambda D, U: fcmdd(D, U, method="PCMdd", eta=np.array(1.0)), "eta"),
        (
            lambda D, U: fcmdd(D, U, method="PCMdd", eta=np.array([1.0, np.nan])),
            "eta",
        ),
        (
            lambda D, U: fcmdd(D, U, method="PCMdd", eta=np.array([1.0, -0.1])),
            "eta",
        ),
        (
            lambda D, U: fcmdd(D, U, method="PCMdd", eta=np.array([1.0, 0.0])),
            "eta",
        ),
    ],
)
def test_fuzzy_public_boundaries_reject_invalid_inputs(call, match):
    D = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 1.0], [0.9, 1.0, 0.0]], dtype=float)
    U = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]], dtype=float)

    with pytest.raises((TypeError, ValueError), match=match):
        call(D, U)


def test_dtprune_with_min_gain():
    D = np.array(
        [
            [0.0, 0.1, 0.9, 1.0, 1.1, 1.2],
            [0.1, 0.0, 1.0, 1.1, 1.2, 1.3],
            [0.9, 1.0, 0.0, 0.1, 0.8, 0.9],
            [1.0, 1.1, 0.1, 0.0, 0.9, 0.8],
            [1.1, 1.2, 0.8, 0.9, 0.0, 0.1],
            [1.2, 1.3, 0.9, 0.8, 0.1, 0.0],
        ]
    )
    props = {
        "x": np.array([0.1, 0.2, 0.9, 1.0, 1.8, 1.9]),
        "grp": np.array(["A", "A", "B", "B", "C", "C"], dtype=object),
    }
    tree = wcPropertyClustering(D, props, min_size=0.16, max_depth=3)
    pruned = dtprune(tree, k=2, min_gain=0.0)
    assert "fitted" in pruned
    assert pruned["root"] is not tree["root"]
    assert len(dtlabels(tree)) > len(dtlabels(pruned)) == 2
    assert len(np.unique(pruned["fitted"])) == 2
    # R dtlabels reports every non-root node, including internal cut nodes.
    assert len(dtlabels(tree)) == 6
    assert len(np.unique(tree["fitted"])) == 4


def test_wcpropertyclustering_supports_weight_permutation_delegation():
    D = np.array(
        [
            [0.0, 0.1, 1.2, 1.1],
            [0.1, 0.0, 1.1, 1.2],
            [1.2, 1.1, 0.0, 0.2],
            [1.1, 1.2, 0.2, 0.0],
        ]
    )
    props = {
        "x": np.array([0.1, 0.2, 1.0, 1.1]),
        "grp": np.array(["A", "A", "B", "B"], dtype=object),
    }
    tree = wcPropertyClustering(D, props, R=1, weight_permutation="diss")
    assert tree["info"]["weight_permutation"] == "none"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": 2.5}, "max_depth must be an integer"),
        ({"R": True}, "R must be an integer"),
        ({"squared": 1}, "squared must be a boolean"),
        ({"maxcluster": 2.5}, "maxcluster must be an integer"),
        ({"random_state": 1.5}, "random_state must be an integer"),
        ({"min_size": np.nan}, "min_size must be a positive finite number"),
        ({"pval": np.nan}, "pval must be between zero and one"),
        ({"R": -1}, "R must be non-negative"),
    ],
)
def test_wcpropertyclustering_rejects_lossy_controls(kwargs, message):
    D = np.array(
        [[0.0, 0.1, 1.0, 1.1], [0.1, 0.0, 1.1, 1.0], [1.0, 1.1, 0.0, 0.1], [1.1, 1.0, 0.1, 0.0]]
    )
    props = {"x": [0.0, 0.1, 1.0, 1.1]}
    with pytest.raises((TypeError, ValueError), match=message):
        wcPropertyClustering(D, props, **kwargs)


def test_property_tree_cut_and_prune_reject_lossy_controls():
    D = np.array(
        [[0.0, 0.1, 1.0, 1.1], [0.1, 0.0, 1.1, 1.0], [1.0, 1.1, 0.0, 0.1], [1.1, 1.0, 0.1, 0.0]]
    )
    tree = wcPropertyClustering(D, {"x": [0.0, 0.1, 1.0, 1.1]}, R=1, pval=1.0)
    with pytest.raises(TypeError, match="labels must be a boolean"):
        dtcut(tree, 2, labels=1)
    with pytest.raises(ValueError, match="k must be an integer"):
        dtprune(tree, 1.5)
    with pytest.raises(ValueError, match="min_gain must be finite"):
        dtprune(tree, 1, min_gain=np.nan)


def test_wcfuzzyrange_forwards_common_scaled_weights_to_every_stage(monkeypatch):
    """Scale-invariant public weights keep one ratio vector at every stage."""
    import sys
    import types

    import weightedcluster._fallback as fallback_module
    import weightedcluster.fuzzy as fuzzy_module
    import weightedcluster.quality as quality_module

    D = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 1.1, 1.0],
            [1.0, 1.1, 0.0, 0.2],
            [1.1, 1.0, 0.2, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    seen: dict[str, np.ndarray] = {}

    def fake_weighted_medoids(
        D_in,
        k,
        *,
        weights=None,
        max_iter=100,
        n_init=10,
        random_state=None,
        init_medoids=None,
    ):
        seen["seed_weights"] = np.asarray(weights, dtype=float)
        labels0 = np.array([0, 0, 1, 1], dtype=int)
        medoids0 = np.array([0, 2], dtype=int)
        return labels0, medoids0, 0.0

    def fake_fcmdd(D_in, u_init, **kwargs):
        seen["fcmdd_weights"] = np.asarray(kwargs["weights"], dtype=float)
        return {
            "memb": np.asarray(u_init, dtype=float),
            "mobileCenters": [0, 2],
            "functional": 0.0,
            "dnoise": None,
        }

    def fake_wc_fuzzy_quality(D_in, U, *, medoids=None, m=1.5, weights=None):
        seen["quality_weights"] = np.asarray(weights, dtype=float)
        return {"stats": {"ASWw": 0.0, "PBC": 0.0}}

    monkeypatch.setitem(
        sys.modules,
        "weightedcluster._wc_core",
        types.SimpleNamespace(weighted_medoids=fake_weighted_medoids),
    )
    monkeypatch.setattr(fallback_module, "weighted_medoids", fake_weighted_medoids)
    monkeypatch.setattr(fuzzy_module, "fcmdd", fake_fcmdd)
    monkeypatch.setattr(quality_module, "wcFuzzyQuality", fake_wc_fuzzy_quality)

    out = wcFuzzyRange(D, [2], weights=weights, method="FCMdd")

    assert "cluster2" in out["memb"]
    # Every workflow stage must receive the original finite weights.  A common
    # scale is mathematically irrelevant, but pre-normalizing here can erase a
    # positive weight when the input spans the full binary64 range.
    assert np.array_equal(seen["seed_weights"], weights)
    assert np.array_equal(seen["fcmdd_weights"], weights)
    assert np.array_equal(seen["quality_weights"], weights)


def test_wcfuzzyrange_derives_per_k_random_seeds(monkeypatch):
    import sys
    import types

    import weightedcluster._fallback as fallback_module
    import weightedcluster.fuzzy as fuzzy_module
    import weightedcluster.quality as quality_module

    D = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 1.1, 1.0],
            [1.0, 1.1, 0.0, 0.2],
            [1.1, 1.0, 0.2, 0.0],
        ],
        dtype=float,
    )
    calls: list[tuple[int, int, int, np.ndarray]] = []

    def fake_weighted_medoids(
        D_in,
        k,
        *,
        weights=None,
        max_iter=100,
        n_init=10,
        random_state=None,
        init_medoids=None,
    ):
        calls.append(
            (
                int(k),
                int(n_init),
                int(random_state),
                np.asarray(init_medoids, dtype=int).copy(),
            )
        )
        labels0 = np.arange(D_in.shape[0], dtype=int) % int(k)
        medoids0 = np.arange(int(k), dtype=int)
        return labels0, medoids0, 0.0

    def fake_fcmdd(D_in, u_init, **kwargs):
        return {
            "memb": np.asarray(u_init, dtype=float),
            "mobileCenters": list(range(int(u_init.shape[1]))),
            "functional": 0.0,
            "dnoise": None,
        }

    def fake_wc_fuzzy_quality(D_in, U, *, medoids=None, m=1.5, weights=None):
        return {"stats": {"ASWw": 0.0, "PBC": 0.0}}

    monkeypatch.setitem(
        sys.modules,
        "weightedcluster._wc_core",
        types.SimpleNamespace(weighted_medoids=fake_weighted_medoids),
    )
    monkeypatch.setattr(fallback_module, "weighted_medoids", fake_weighted_medoids)
    monkeypatch.setattr(fuzzy_module, "fcmdd", fake_fcmdd)
    monkeypatch.setattr(quality_module, "wcFuzzyQuality", fake_wc_fuzzy_quality)

    out = wcFuzzyRange(D, [2, 3, 4], method="FCMdd", random_state=17)

    assert "cluster2" in out["memb"]
    assert "cluster3" in out["memb"]
    assert "cluster4" in out["memb"]
    base_seed = generate_random_seed(17, 0)
    expected: list[tuple[int, int, int, np.ndarray]] = []
    for index, k in enumerate((2, 3, 4)):
        seed = generate_random_seed(base_seed, 1, index)
        rng = np.random.default_rng(seed)
        expected.extend(
            (k, 1, seed, np.asarray(rng.choice(4, size=k, replace=False), dtype=int))
            for _ in range(2)
        )
    assert len(calls) == len(expected)
    for actual, planned in zip(calls, expected, strict=True):
        assert actual[:3] == planned[:3]
        np.testing.assert_array_equal(actual[3], planned[3])

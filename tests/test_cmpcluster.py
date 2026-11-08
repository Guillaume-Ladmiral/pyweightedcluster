import numpy as np

from weightedcluster import wcCmpCluster


def test_wccmpcluster_linkages_on_distances():
    # Small distance matrix (3 points)
    D = np.array(
        [
            [0.0, 0.2, 1.0],
            [0.2, 0.0, 0.9],
            [1.0, 0.9, 0.0],
        ]
    )
    out = wcCmpCluster(D, maxcluster=3, method=["single", "complete", "average"])  # no 'ward'
    assert "allstats" in out
    for m in ["single", "complete", "average"]:
        assert m in out and "kvals" in out[m] and "clustering" in out[m]


def test_wccmpcluster_ward_with_precomputed_runs():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(X, metric="euclidean"))
    out = wcCmpCluster(D, maxcluster=3, method=["ward"])
    assert "ward.D2" in out
    assert out["ward.D2"]["kvals"] == [2, 3]
    for k in out["ward.D2"]["kvals"]:
        labels = out["ward.D2"]["clustering"][f"cluster{k}"]
        assert len(np.unique(labels)) == k


def test_wccmpcluster_embed_then_ward_runs():
    # Construct a simple ultrametric-ish distance that is Euclidean embeddable
    X = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
    # Build Euclidean distances
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(X, metric="euclidean"))
    out = wcCmpCluster(D, maxcluster=3, method=["ward"], embed_then_ward=True)
    assert "ward.D2" in out
    assert out["ward.D2"]["kvals"] == [2, 3]
    assert set(out["ward.D2"]["clustering"].keys()) == {"cluster2", "cluster3"}


def test_wccmpcluster_all_includes_diana_and_beta_flexible_without_weights():
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1],
            [0.2, 0.0, 1.0, 1.1],
            [1.0, 1.0, 0.0, 0.2],
            [1.1, 1.1, 0.2, 0.0],
        ]
    )
    out = wcCmpCluster(D, maxcluster=3, method="all")
    assert "diana" in out
    assert "beta.flexible" in out
    assert "pam" in out


def test_wccmpcluster_all_excludes_diana_and_beta_flexible_with_weights():
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1],
            [0.2, 0.0, 1.0, 1.1],
            [1.0, 1.0, 0.0, 0.2],
            [1.1, 1.1, 0.2, 0.0],
        ]
    )
    w = np.array([1.0, 1.0, 1.0, 1.0])
    out = wcCmpCluster(D, maxcluster=3, method="all", weights=w)
    assert "diana" not in out
    assert "beta.flexible" not in out
    assert "pam" in out


def test_wccmpcluster_pam_random_state_reproducible():
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1],
            [0.2, 0.0, 1.0, 1.1],
            [1.0, 1.0, 0.0, 0.2],
            [1.1, 1.1, 0.2, 0.0],
        ]
    )

    out1 = wcCmpCluster(D, maxcluster=3, method=["pam"], random_state=123)
    out2 = wcCmpCluster(D, maxcluster=3, method=["pam"], random_state=123)

    assert out1["param"].additions["random_state"] == 123
    assert out2["param"].additions["random_state"] == 123
    assert out1["pam"]["kvals"] == out2["pam"]["kvals"]

    for k in out1["pam"]["kvals"]:
        np.testing.assert_array_equal(
            out1["pam"]["clustering"][f"cluster{k}"],
            out2["pam"]["clustering"][f"cluster{k}"],
        )

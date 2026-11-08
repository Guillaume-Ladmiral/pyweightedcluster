import numpy as np

from weightedcluster import wcCmpCluster


def test_cmpcluster_diana_basic():
    # Small distance matrix (4 points, two obvious groups)
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1],
            [0.2, 0.0, 1.0, 1.1],
            [1.0, 1.0, 0.0, 0.2],
            [1.1, 1.1, 0.2, 0.0],
        ]
    )
    out = wcCmpCluster(D, maxcluster=4, method=["diana"])
    assert "diana" in out
    kvals = out["diana"]["kvals"]
    assert kvals == [2, 3, 4]
    assert out["param"]["kvals"] == kvals
    clust = out["diana"]["clustering"]
    assert list(clust) == ["cluster2", "cluster3", "cluster4"]
    for k in kvals:
        lab = np.asarray(clust[f"cluster{k}"])
        assert lab.shape[0] == D.shape[0]
        assert set(np.unique(lab)).issubset(set(range(1, k + 1)))

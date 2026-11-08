import numpy as np

from weightedcluster import wcCmpCluster


def test_cmpcluster_beta_flexible_basic():
    # Small distance matrix (5 points)
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1, 1.2],
            [0.2, 0.0, 1.0, 1.1, 1.2],
            [1.0, 1.0, 0.0, 0.2, 0.3],
            [1.1, 1.1, 0.2, 0.0, 0.2],
            [1.2, 1.2, 0.3, 0.2, 0.0],
        ]
    )
    out = wcCmpCluster(D, maxcluster=4, method=["beta.flexible"])  # custom linkage
    assert "beta.flexible" in out
    kvals = out["beta.flexible"]["kvals"]
    assert kvals == [2, 3, 4]
    clust = out["beta.flexible"]["clustering"]
    for k in kvals:
        lab = np.asarray(clust[f"cluster{k}"])
        assert lab.shape[0] == D.shape[0]
        assert set(np.unique(lab)).issubset(set(range(1, k + 1)))

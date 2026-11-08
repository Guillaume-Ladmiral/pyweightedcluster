from __future__ import annotations

import numpy as np

from weightedcluster import as_seqtree, dtcut
from weightedcluster._seqdist import sequences_distance


def test_as_seqtree_smoke_and_dtcut_compatibility():
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

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
    Z = linkage(squareform(D, checks=False), method="average")
    tree = as_seqtree(Z, seqdata=X, diss=D, ncluster=3)

    assert "root" in tree and "fitted" in tree and "weights" in tree
    assert np.asarray(tree["fitted"]).shape[0] == X.shape[0]

    labels = dtcut(tree, 2, labels=False)
    assert np.asarray(labels).shape[0] == X.shape[0]

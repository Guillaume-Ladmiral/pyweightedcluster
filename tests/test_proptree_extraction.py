import numpy as np

from weightedcluster import seqpropclust
from weightedcluster._seqdist import sequences_distance


def test_seqpropclust_auto_extraction_smoke():
    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )
    D = sequences_distance(X, metric="HAM", normalize=True)
    tree = seqpropclust(X, D, maxcluster=3)
    # Basic structure
    assert "root" in tree and "data" in tree and "fitted" in tree


def test_seqpropclust_features_present():
    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )
    # Use internal extractor via seqpropclust(prop_only=True)
    props = seqpropclust(X, np.zeros((3, 3)), prop_only=True)
    cols = list(props.columns)
    # R-compatible combined-group names.
    assert any(c.startswith("state.T") for c in cols)
    assert any(c.startswith("spell.") and "_dur_" in c for c in cols)
    assert any(c.startswith("duration.") for c in cols)
    assert any(c.startswith("transition.(") for c in cols)

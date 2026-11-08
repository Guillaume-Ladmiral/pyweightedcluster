import numpy as np

from weightedcluster import _wc_core as _core
from weightedcluster import seqclararange, wcClusterQuality, wcKMedoids
from weightedcluster._seqdist import sequences_distance


def _labels1b_to_0(labels1b: np.ndarray) -> np.ndarray:
    uniq = np.unique(labels1b)
    mapping = {int(u): i for i, u in enumerate(uniq.tolist())}
    return np.array([mapping[int(v)] for v in labels1b], dtype=int)


def test_om_distance_and_core_quality_match_python():
    # Two obvious clusters in sequence space
    X = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
        ]
    )
    # OM distances with indel=1.0, default substitution costs
    D = sequences_distance(X, metric="OM", normalize=True, indel=1.0)
    assert D.shape == (8, 8)
    # Cluster
    res = wcKMedoids(D, 2)
    labels1b = res["clustering"]
    labels0 = _labels1b_to_0(labels1b)

    # Core fast-path cluster quality
    stats_core, asw_core, asw_w_core = _core.cluster_quality(D, labels0)
    # Python wrapper
    q = wcClusterQuality(D, labels1b)
    stats_py = q["stats"]
    # Compare a few main metrics (tolerant)
    for key in ["ASW", "R2", "CH"]:
        v_core = float(stats_core.get(key, np.nan))
        v_py = float(stats_py.get(key, np.nan))
        assert np.isfinite(v_core) and np.isfinite(v_py)
        assert abs(v_core - v_py) <= 1e-6


def test_seqclararange_with_om_smoke():
    X = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ]
    )
    out = seqclararange(
        X,
        R=2,
        sample_size=3,
        kvals=[2],
        seqdist_args={"method": "OM", "indel": 1.0},
    )
    assert out["kvals"] == [2]
    assert "clustering" in out and "stats" in out and "clara" in out

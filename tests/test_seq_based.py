import numpy as np

from weightedcluster import db_index, pbm_index, seqclararange, seqnullcqi, xb_index


def test_seqclararange_smoke():
    # Tiny toy sequences (two clusters apparent by symbol)
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    out = seqclararange(
        X,
        R=3,
        sample_size=3,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        random_state=0,
    )
    assert out["kvals"] == [2]
    assert "clustering" in out and "stats" in out and "clara" in out
    cl = out["clustering"]["cluster2"]
    assert set(np.unique(cl)) == {1, 2}


def test_internal_indices_and_selection():
    import numpy as np

    # Simple Euclidean-like distances for 4 points in two clusters
    D = np.array(
        [
            [0.0, 0.1, 2.0, 2.1],
            [0.1, 0.0, 2.1, 2.0],
            [2.0, 2.1, 0.0, 0.2],
            [2.1, 2.0, 0.2, 0.0],
        ]
    )
    # Perfect 2-cluster labels
    labels = np.array([1, 1, 2, 2])
    # DB should be finite and positive
    v_db = db_index(D, labels)
    v_xb = xb_index(D, labels)
    v_pbm = pbm_index(D, labels)
    assert np.isfinite(v_db)
    assert np.isfinite(v_xb)
    assert np.isfinite(v_pbm)

    # Also ensure seqclararange can select with AMS criterion in fuzzy mode
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )
    out = seqclararange(
        X,
        R=5,
        sample_size=4,
        kvals=[2, 3],
        seqdist_args={"method": "HAM"},
        method="fuzzy",
        m=1.6,
        criteria=("ams",),
        export_membership=True,
        random_state=7,
    )
    # Ensure best selected K is among provided
    assert set(out["kvals"]) == {2, 3}
    # Membership matrix integrity
    memb = out["clara"][2].get("membership")
    if memb is not None:
        import numpy as np

        U = np.array(memb, dtype=float)
        assert U.shape[0] == X.shape[0]
        # Row sums approximately 1 for FCMdd projection
        rs = U.sum(axis=1)
        assert np.allclose(rs, 1.0, atol=1e-6)


def test_seqnullcqi_smoke():
    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    # Seed a trivial clustrange with K=2
    from weightedcluster import wcKMedoids
    from weightedcluster._seqdist import sequences_distance
    from weightedcluster.schemas import ClustrangeResult

    D = sequences_distance(X, metric="hamming", normalize=True)
    res = wcKMedoids(D, 2)
    clustrange = ClustrangeResult({"kvals": [2], "clustering": {"cluster2": res["clustering"]}})

    out = seqnullcqi(X, clustrange, R=2, model="combined", seqdist_args={"method": "HAM"})
    assert "boot" not in out and 2 in out.additions["boot"]
    assert set(out.additions["boot"][2].keys()) >= {"ASW"}

import numpy as np

from weightedcluster import seqclararange


def test_seqclararange_family_structure():
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )
    fam = seqclararange(
        X,
        R=4,
        sample_size=3,
        kvals=[2, 3],
        seqdist_args={"method": "HAM"},
        method="crisp",
        criteria=["distance", "db"],
        random_state=7,
    )
    # Family keys include param, allstats, and each criterion
    assert "param" in fam and "allstats" in fam
    assert all(c in fam for c in ("distance", "db"))
    # Each sub-object should have kvals and stats
    for c in ("distance", "db"):
        sub = fam[c]
        assert "kvals" in sub and "stats" in sub and "clustering" in sub
        assert sub["kvals"] == [2, 3]
    # allstats contains ngroup and criteria fields
    allstats = fam["allstats"]
    assert "ngroup" in allstats and "criteria" in allstats
    # Distance row for K=2 should match sub stats at index 0 for a common stat (e.g., ASW)
    # We just check presence and length consistency to avoid tight coupling
    for c in ("distance", "db"):
        keys = list(fam[c]["stats"].keys())
        assert len(keys) > 0
        for k in fam[c]["kvals"]:
            # ensure we have values per stat for each k
            idx = fam[c]["kvals"].index(k)
            for stat_name in keys:
                assert idx < len(fam[c]["stats"][stat_name])

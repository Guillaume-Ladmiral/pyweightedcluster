import numpy as np

from tests.r_parity_utils import import_optional_or_skip
from weightedcluster import wcKMedoids
from weightedcluster._seqdist import sequences_distance
from weightedcluster.schemas import ClustrangeResult
from weightedcluster.seqnull import seqnullcqi


def test_plot_seqnullcqi_seqdplot_runs():
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)
    from weightedcluster.plot import plot_seqnullcqi

    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )
    D = sequences_distance(X, metric="HAM", normalize=True)
    res = wcKMedoids(D, 2)
    cr = ClustrangeResult({"kvals": [2], "clustering": {"cluster2": res["clustering"]}})
    bcq = seqnullcqi(X, cr, R=2, model="combined", seqdist_args={"method": "HAM"})
    fig, ax = plot_seqnullcqi(bcq, stat="ASW", kind="seqdplot")
    assert fig is not None and ax is not None

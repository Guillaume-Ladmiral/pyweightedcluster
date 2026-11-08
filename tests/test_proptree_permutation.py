import numpy as np
import pandas as pd
import pytest

from weightedcluster.proptree import wcPropertyClustering


def test_proptree_permutation_flag_runs_reproducibly():
    # Small synthetic case: numeric x strongly splits groups
    D = np.array(
        [
            [0.0, 0.2, 1.0, 1.1],
            [0.2, 0.0, 1.0, 1.1],
            [1.0, 1.0, 0.0, 0.3],
            [1.1, 1.1, 0.3, 0.0],
        ]
    )
    props = pd.DataFrame({"x": [0.1, 0.2, 1.0, 1.1], "grp": ["A", "A", "B", "B"]})
    # Enable a few permutations to exercise path
    tree = wcPropertyClustering(
        D,
        props,
        maxcluster=None,
        min_size=0.25,
        max_depth=2,
        perm_R=25,
        perm_n_jobs=1,
        progressbar=False,
        pval=1.0,
        random_state=123,
    )
    tree2 = wcPropertyClustering(
        D,
        props,
        maxcluster=None,
        min_size=0.25,
        max_depth=2,
        perm_R=25,
        perm_n_jobs=1,
        progressbar=False,
        pval=1.0,
        random_state=123,
    )
    assert "root" in tree

    root_split = tree["root"]["split"]
    root_split2 = tree2["root"]["split"]
    assert "pvalue" in root_split
    assert root_split["pvalue"] == pytest.approx(root_split2["pvalue"])
    assert 0.0 < float(root_split["pvalue"]) < 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"perm_n_jobs": 2}, "serial permutation execution"),
        ({"progressbar": True}, "progress reporting is not available"),
    ],
)
def test_proptree_rejects_unsupported_no_effect_execution_controls(kwargs, message):
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    props = pd.DataFrame({"x": [0.0, 1.0]})
    with pytest.raises(NotImplementedError, match=message):
        wcPropertyClustering(D, props, R=1, **kwargs)

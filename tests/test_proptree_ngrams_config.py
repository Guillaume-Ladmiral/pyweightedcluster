import numpy as np
import pytest

from weightedcluster import seqpropclust


def test_seqpropclust_max_k_controls_event_subsequence_length():
    X = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=int,
    )
    max2 = seqpropclust(
        X,
        np.zeros((3, 3)),
        prop_only=True,
        properties=("pattern",),
        pmin_support=0.0,
        max_k=2,
    )
    max4 = seqpropclust(
        X,
        np.zeros((3, 3)),
        prop_only=True,
        properties=("pattern",),
        pmin_support=0.0,
        max_k=4,
    )
    assert set(max2.columns) < set(max4.columns)
    assert "(0)-(1)-(0)" in max4.columns
    assert not any(str(column).startswith("ng") for column in max4.columns)


@pytest.mark.parametrize(
    "obsolete_control",
    [{"feature_include_transitions": False}, {"feature_ngrams_max": 4}],
)
def test_seqpropclust_rejects_removed_no_effect_feature_controls(obsolete_control):
    X = np.array([[0, 0, 1], [1, 1, 0]], dtype=int)

    with pytest.raises(TypeError, match="unexpected seqpropclust keyword"):
        seqpropclust(
            X,
            np.zeros((2, 2)),
            prop_only=True,
            properties=("state",),
            **obsolete_control,
        )

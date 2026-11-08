from __future__ import annotations

import inspect

import numpy as np
import pytest

from weightedcluster import seqpropclust, wcPropertyClustering


def _toy_sequences() -> np.ndarray:
    return np.array(
        [
            [0, 0, 1, 1, 2],
            [0, 1, 1, 2, 2],
            [2, 2, 1, 0, 0],
            [1, 1, 2, 2, 0],
        ],
        dtype=int,
    )


def test_seqpropclust_properties_filter_is_applied():
    X = _toy_sequences()
    props = seqpropclust(
        X, np.zeros((X.shape[0], X.shape[0])), prop_only=True, properties=("state",)
    )
    cols = list(props.columns)
    assert cols
    assert cols == [f"T{i}" for i in range(1, X.shape[1] + 1)]


def test_seqpropclust_pattern_respects_effective_pmin_support_and_max_k():
    X = _toy_sequences()
    D0 = np.zeros((X.shape[0], X.shape[0]))

    props_loose = seqpropclust(
        X,
        D0,
        prop_only=True,
        properties=("pattern",),
        pmin_support=0.0,
        max_k=4,
    )
    props_strict = seqpropclust(
        X,
        D0,
        prop_only=True,
        properties=("pattern",),
        pmin_support=1.0,
        max_k=4,
    )
    props_max2 = seqpropclust(
        X,
        D0,
        prop_only=True,
        properties=("pattern",),
        pmin_support=0.0,
        max_k=2,
    )

    assert props_strict.shape[1] <= props_loose.shape[1]
    assert not any(c.startswith("tri_") for c in props_max2.columns)
    assert not any(c.startswith("ng4_") for c in props_max2.columns)


def test_seqpropclust_combines_other_prop_with_extracted_properties():
    X = _toy_sequences()
    D0 = np.zeros((X.shape[0], X.shape[0]))
    props = seqpropclust(
        X,
        D0,
        prop_only=True,
        properties=("state",),
        other_prop={"extra": [10, 20, 30, 40]},
    )
    assert "other.prop.extra" in props.columns
    assert any(c.startswith("state.T") for c in props.columns)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prop_only": 1}, "prop_only must be a boolean"),
        ({"with_missing": "no", "prop_only": True}, "with_missing must be a boolean"),
        ({"max_k": 2.5, "prop_only": True}, "max_k must be an integer"),
        ({"max_depth": 2.5}, "max_depth must be an integer"),
        ({"R": True}, "R must be an integer"),
        ({"squared": 1}, "squared must be a boolean"),
        ({"weighted": 1}, "weighted must be a boolean"),
        ({"maxcluster": 2.5}, "maxcluster must be an integer"),
        ({"random_state": 1.5}, "random_state must be an integer"),
        ({"pmin_support": np.nan, "prop_only": True}, "pmin_support must be between zero and one"),
        ({"max_k": -2, "prop_only": True}, "max_k must be -1 or a positive integer"),
        ({"min_size": np.nan}, "min_size must be a positive finite number"),
        ({"R": -1}, "R must be non-negative"),
    ],
)
def test_seqpropclust_rejects_lossy_controls(kwargs, message):
    X = _toy_sequences()
    D = np.not_equal(X[:, None, :], X[None, :, :]).mean(axis=2)
    with pytest.raises((TypeError, ValueError), match=message):
        seqpropclust(X, D, properties=("state",), **kwargs)


@pytest.mark.parametrize("public_api", [wcPropertyClustering, seqpropclust])
def test_property_tree_execution_control_signature_contract(public_api):
    parameters = inspect.signature(public_api).parameters

    assert parameters["perm_n_jobs"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["perm_n_jobs"].default is None
    assert parameters["n_jobs"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["n_jobs"].default is None
    assert parameters["progressbar"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["progressbar"].default is False

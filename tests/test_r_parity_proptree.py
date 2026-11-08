from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from weightedcluster import dtcut, dtlabels, dtprune, seqpropclust, wcPropertyClustering
from weightedcluster.plot import plot_proptree_labels, plot_proptree_panel, plot_proptree_tree

pytest_plugins = ["tests.r_parity_utils"]

Y = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 12.0, 12.1, 12.2])
D = np.abs(Y[:, None] - Y[None, :])
GROUP = np.repeat(["A", "B"], 6)
PHASE = np.tile(np.repeat([0, 1], 3), 2)
PROPERTIES = {
    "group": GROUP,
    "phase": PHASE,
}
PROPERTIES_DF = pd.DataFrame(
    {
        "group": pd.Categorical(GROUP),
        "phase": PHASE,
    }
)
WEIGHTS = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2, 1.0, 1.5, 0.75, 2.0, 1.0, 1.25])
TREE_SEQUENCES = np.tile(
    np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
        ],
        dtype=int,
    ),
    (3, 1),
)
EXTRACTION_SEQUENCES = np.array(
    [
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ],
    dtype=int,
)


def _flatten_python_tree(tree: dict) -> np.ndarray:
    rows: list[list[float]] = []

    def walk(node: dict, parent: int | None = None, side: int | None = None) -> None:
        split = node.get("split")
        info = node["info"]
        split_info = {} if split is None else split["info"]
        rows.append(
            [
                float(node["id"]),
                np.nan if parent is None else float(parent),
                np.nan if side is None else float(side),
                float(info["depth"]),
                float(info["splitschedule"]),
                float(info["n"]),
                float(info["vardis"]),
                float(info["medoid"]) + 1.0,
                float(split is not None),
                np.nan if split is None else float(split["varindex"]) + 1.0,
                np.nan if split is None or split.get("breaks") is None else float(split["breaks"]),
                np.nan if split is None else float(split_info["SCres"]),
                np.nan if split is None else float(split_info["lpop"]),
                np.nan if split is None else float(split_info["rpop"]),
                np.nan if split is None else float(split_info["lvar"]),
                np.nan if split is None else float(split_info["rvar"]),
                np.nan if split is None else float(split_info["R2"]),
                (
                    np.nan
                    if split is None or split_info.get("pval") is None
                    else float(split_info["pval"])
                ),
            ]
        )
        if node.get("kids") is not None:
            walk(node["kids"][0], int(node["id"]), 1)
            walk(node["kids"][1], int(node["id"]), 2)

    walk(tree["root"])
    return np.asarray(rows, dtype=float)


def _assert_tree_contract(py: dict, r: dict) -> None:
    np.testing.assert_allclose(
        _flatten_python_tree(py),
        r["nodes"],
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(py["fitted"], r["fitted"])
    assert dtlabels(py) == dict(zip(r["label_names"], r["label_values"], strict=True))
    np.testing.assert_array_equal(dtcut(py, 2, labels=False), r["cut2_ids"])
    assert list(dtcut(py, 2, labels=True)) == r["cut2_labels"]
    py_adjustment = py["adjustment"].stat["t0"].to_dict()
    np.testing.assert_allclose(
        [py_adjustment[name] for name in r["adjustment"]],
        list(r["adjustment"].values()),
        rtol=0.0,
        atol=1e-10,
    )


def test_wcpropertyclustering_matches_live_r_tree_semantics(r_parity_session):
    r = r_parity_session.wc_property_clustering_contract(
        D,
        PROPERTIES,
        categorical=["group"],
        R=1,
        min_size=2,
        max_depth=3,
    )
    py = wcPropertyClustering(
        D,
        PROPERTIES_DF,
        R=1,
        min_size=2,
        max_depth=3,
    )

    assert r["class"] == ["dtclust", "disstree", "list"]
    assert py["class"] == r["class"]
    assert py["root"]["split"]["var"] == "group"
    assert py["root"]["kids"][0]["split"]["var"] == "phase"
    _assert_tree_contract(py, r)


def test_wcpropertyclustering_supports_r_weighted_path(r_parity_session):
    r = r_parity_session.wc_property_clustering_contract(
        D,
        PROPERTIES,
        categorical=["group"],
        weights=WEIGHTS,
        R=1,
        min_size=2,
        max_depth=3,
        weight_permutation="diss",
    )
    py = wcPropertyClustering(
        D,
        PROPERTIES_DF,
        weights=WEIGHTS,
        R=1,
        min_size=2,
        max_depth=3,
        weight_permutation="diss",
    )

    _assert_tree_contract(py, r)


def test_maxcluster_and_dtprune_match_live_r(r_parity_session):
    r = r_parity_session.wc_property_clustering_contract(
        D,
        PROPERTIES,
        categorical=["group"],
        R=1,
        min_size=2,
        max_depth=3,
        maxcluster=3,
    )
    full = wcPropertyClustering(D, PROPERTIES_DF, R=1, min_size=2, max_depth=3)
    via_function = dtprune(full, 3)
    via_argument = wcPropertyClustering(
        D,
        PROPERTIES_DF,
        R=1,
        min_size=2,
        max_depth=3,
        maxcluster=3,
    )

    _assert_tree_contract(via_function, r)
    _assert_tree_contract(via_argument, r)
    np.testing.assert_array_equal(via_function["fitted"], via_argument["fitted"])


def test_seeded_permutation_runs_are_reproducible_in_each_runtime(r_parity_session):
    py_kwargs = {
        "R": 25,
        "pval": 1.0,
        "min_size": 2,
        "max_depth": 3,
        "random_state": 19,
    }
    py_first = wcPropertyClustering(D, PROPERTIES_DF, **py_kwargs)
    py_second = wcPropertyClustering(D, PROPERTIES_DF, **py_kwargs)
    np.testing.assert_allclose(
        _flatten_python_tree(py_first),
        _flatten_python_tree(py_second),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_array_equal(py_first["fitted"], py_second["fitted"])

    r_first = r_parity_session.wc_property_clustering_contract(
        D,
        PROPERTIES,
        categorical=["group"],
        R=25,
        pval=1.0,
        min_size=2,
        max_depth=3,
        seed=19,
    )
    r_second = r_parity_session.wc_property_clustering_contract(
        D,
        PROPERTIES,
        categorical=["group"],
        R=25,
        pval=1.0,
        min_size=2,
        max_depth=3,
        seed=19,
    )
    np.testing.assert_allclose(
        r_first["nodes"], r_second["nodes"], rtol=0.0, atol=0.0, equal_nan=True
    )
    np.testing.assert_array_equal(r_first["fitted"], r_second["fitted"])


def test_seqpropclust_state_and_duration_properties_match_live_r(r_parity_session):
    r = r_parity_session.seqpropclust_properties(
        EXTRACTION_SEQUENCES,
        ["state", "duration"],
    )
    py = seqpropclust(
        EXTRACTION_SEQUENCES,
        np.zeros((len(EXTRACTION_SEQUENCES), len(EXTRACTION_SEQUENCES))),
        properties=("state", "duration"),
        prop_only=True,
    )

    assert list(py.columns) == r["names"]
    for name in r["names"]:
        values = ["<NA>" if pd.isna(value) else str(value) for value in py[name]]
        assert values == r["values"][name]
        if r["classes"][name] == ["factor"]:
            assert isinstance(py[name].dtype, pd.CategoricalDtype)
        else:
            assert np.issubdtype(py[name].dtype, np.integer)


def test_seqpropclust_tree_stage_matches_live_r_seqtree(r_parity_session):
    r = r_parity_session.seqpropclust_contract(
        TREE_SEQUENCES,
        D,
        PROPERTIES,
        categorical=["group"],
        R=1,
        min_size=2,
        max_depth=3,
    )
    py = seqpropclust(
        TREE_SEQUENCES,
        D,
        properties=(),
        other_prop=PROPERTIES_DF,
        R=1,
        min_size=2,
        max_depth=3,
    )

    assert r["class"][:4] == ["seqtreeclust", "dtclust", "seqtree", "disstree"]
    assert py["class"][:4] == r["class"][:4]
    assert py["info"]["method"] == "seqtree"
    _assert_tree_contract(py, r)


def test_seqpropclust_weighted_default_matches_r_diss_permutation(r_parity_session):
    r = r_parity_session.seqpropclust_contract(
        TREE_SEQUENCES,
        D,
        PROPERTIES,
        categorical=["group"],
        weights=WEIGHTS,
        R=1,
        min_size=2,
        max_depth=3,
    )
    py = seqpropclust(
        TREE_SEQUENCES,
        D,
        properties=(),
        other_prop=PROPERTIES_DF,
        weights=WEIGHTS,
        R=1,
        min_size=2,
        max_depth=3,
    )

    assert py["info"]["weight_permutation"] == "diss"
    _assert_tree_contract(py, r)


def test_property_tree_faithful_names_delegate_all_tree_growth_to_traminex():
    source_path = Path(__file__).parents[1] / "src" / "weightedcluster" / "proptree.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    defined = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "_build_tree" not in defined
    assert "_best_split_for_variable" not in defined
    assert "traminex.disstree" in source
    assert "traminex.seqtree" in source


def test_property_tree_plotting_accepts_the_traminex_adapter():
    tree = wcPropertyClustering(D, PROPERTIES_DF, R=1, min_size=2, max_depth=3)
    figures = [
        plot_proptree_labels(tree),
        plot_proptree_tree(tree),
        plot_proptree_panel(tree),
    ]
    assert all(result[0] is not None for result in figures)

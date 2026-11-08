from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from traminex import STSList

from tests.r_parity_utils import import_optional_or_skip
from weightedcluster import (
    as_seqtree,
    dtcut,
    dtlabels,
    dtprune,
    fuzzyseqplot,
    seqclustname,
    wcAggregateCases,
)
from weightedcluster.plot import plot_proptree_tree

pytest_plugins = ["tests.r_parity_utils"]


def test_wc_aggregate_cases_uses_r_aggregate_key_order(r_parity_session):
    rows = [("b", 2), ("a", 1), ("b", 2), ("c", 0), ("a", 1)]
    weights = np.array([1, 2, 3, 4, 5], dtype=float)

    oracle = r_parity_session.wc_aggregate_cases_contract(rows, weights=weights)
    result = wcAggregateCases(rows, weights=weights)

    assert oracle["ok"] is True
    np.testing.assert_array_equal(result["aggIndex"], oracle["aggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["disaggIndex"], oracle["disaggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["aggWeights"], oracle["aggWeights"])
    np.testing.assert_array_equal(result["disaggWeights"], oracle["disaggWeights"])
    assert isinstance(result, Mapping)
    assert list(result) == ["aggIndex", "aggWeights", "disaggIndex", "disaggWeights"]
    assert oracle["class"] == ["wcAggregateCases", "list"]
    assert oracle["names"] == list(result)
    assert str(result).splitlines() == [line.rstrip() for line in oracle["print_lines"]]


@pytest.mark.parametrize(
    ("input_kind", "r_rows", "python_rows"),
    [
        (
            "data.frame",
            [("b", 2), ("a", 1), ("b", 2), ("c", 0), ("a", 1)],
            pd.DataFrame({"state": ["b", "a", "b", "c", "a"], "score": [2, 1, 2, 0, 1]}),
        ),
        (
            "matrix",
            [("b", 2), ("a", 1), ("b", 2), ("c", 0), ("a", 1)],
            np.asarray(
                [("b", 2), ("a", 1), ("b", 2), ("c", 0), ("a", 1)],
                dtype=object,
            ),
        ),
        ("default", ["b", "a", "b", "c", "a"], np.array(["b", "a", "b", "c", "a"])),
    ],
)
def test_wc_aggregate_cases_dispatch_and_unit_weights_match_live_r(
    r_parity_session, input_kind, r_rows, python_rows
):
    before = (
        python_rows.copy(deep=True) if isinstance(python_rows, pd.DataFrame) else python_rows.copy()
    )
    oracle = r_parity_session.wc_aggregate_cases_contract(r_rows, input_kind=input_kind)

    result = wcAggregateCases(python_rows)

    np.testing.assert_array_equal(result["aggIndex"], oracle["aggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["disaggIndex"], oracle["disaggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["aggWeights"], oracle["aggWeights"])
    np.testing.assert_array_equal(result["disaggWeights"], oracle["disaggWeights"])
    if isinstance(python_rows, pd.DataFrame):
        pd.testing.assert_frame_equal(python_rows, before)
    else:
        np.testing.assert_array_equal(python_rows, before)


def test_wc_aggregate_cases_preserves_factor_levels_missing_and_mixed_columns(
    r_parity_session,
):
    frame = pd.DataFrame(
        {
            "category": pd.Categorical(
                ["b", None, "a", None, "b"],
                categories=["b", "a", "unused"],
                ordered=True,
            ),
            "number": [2.0, 1.0, np.nan, 1.0, 2.0],
            "mixed_text": ["x", "2", "x", "2", "x"],
        }
    )
    before = frame.copy(deep=True)
    rows = [tuple(row) for row in frame.astype(object).itertuples(index=False, name=None)]
    rows = [tuple(None if pd.isna(value) else value for value in row) for row in rows]
    oracle = r_parity_session.wc_aggregate_cases_contract(
        rows,
        categories={0: ["b", "a", "unused"]},
        weights=[1, 2, 3, 4, 5],
    )

    result = wcAggregateCases(frame, weights=[1, 2, 3, 4, 5])

    np.testing.assert_array_equal(result["aggIndex"], oracle["aggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["disaggIndex"], oracle["disaggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["aggWeights"], oracle["aggWeights"])
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("weighted", "explicit", "expected"),
    [
        (True, None, [4.0, 2.0, 4.0]),
        (False, None, [2.0, 1.0, 1.0]),
        (True, [2.0, 4.0, 6.0, 8.0], [8.0, 4.0, 8.0]),
    ],
)
def test_wc_aggregate_cases_stslist_weight_dispatch_matches_live_r(
    r_parity_session, weighted, explicit, expected
):
    codes = np.array([[0], [1], [0], [2]], dtype=int)
    sequence = STSList(
        data=codes,
        alphabet=["aa", "bb", "cc"],
        weights=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    oracle = r_parity_session.wc_aggregate_cases_contract(
        [["aa"], ["bb"], ["aa"], ["cc"]],
        input_kind="stslist",
        weights=explicit,
        attached_weights=[1.0, 2.0, 3.0, 4.0],
        weighted=weighted,
    )

    result = wcAggregateCases(sequence, weights=explicit, weighted=weighted)

    np.testing.assert_array_equal(result["aggIndex"], oracle["aggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["disaggIndex"], oracle["disaggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["aggWeights"], oracle["aggWeights"])
    np.testing.assert_array_equal(result["aggWeights"], expected)
    np.testing.assert_array_equal(result["disaggWeights"], oracle["disaggWeights"])


@pytest.mark.parametrize("weights", [[2.0], [2.0, 3.0, 4.0], []])
def test_wc_aggregate_cases_permissive_weight_lengths_match_live_r(r_parity_session, weights):
    rows = ["a", "a", "b"]
    oracle = r_parity_session.wc_aggregate_cases_contract(
        rows, input_kind="default", weights=weights
    )

    result = wcAggregateCases(rows, weights=weights)

    np.testing.assert_array_equal(result["aggIndex"], oracle["aggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["disaggIndex"], oracle["disaggIndex_1b"] - 1)
    np.testing.assert_allclose(
        result["aggWeights"], oracle["aggWeights"], rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_array_equal(result["disaggWeights"], oracle["disaggWeights"])


def test_wc_aggregate_cases_zero_and_one_row_boundaries_match_live_r(r_parity_session):
    zero = r_parity_session.wc_aggregate_cases_contract([], input_kind="default")
    one = r_parity_session.wc_aggregate_cases_contract(["only"], input_kind="default")

    assert zero == {"ok": False, "error": "arguments must have same length"}
    with pytest.raises(ValueError, match="^arguments must have same length$"):
        wcAggregateCases(np.array([], dtype=float))

    result = wcAggregateCases(np.array(["only"]))
    np.testing.assert_array_equal(result["aggIndex"], one["aggIndex_1b"] - 1)
    np.testing.assert_array_equal(result["aggWeights"], one["aggWeights"])
    assert str(result).splitlines() == [line.rstrip() for line in one["print_lines"]]


def test_wc_aggregate_cases_non_numeric_weights_match_live_r_error(r_parity_session):
    oracle = r_parity_session.wc_aggregate_cases_contract(
        ["a", "a"], input_kind="default", weights=["2", "3"]
    )

    assert oracle == {"ok": False, "error": "invalid 'type' (character) of argument"}
    with pytest.raises(TypeError, match="invalid 'type' \\(character\\) of argument"):
        wcAggregateCases(["a", "a"], weights=["2", "3"])


def test_wc_aggregate_cases_signature_exposes_stslist_weight_dispatch():
    signature = inspect.signature(wcAggregateCases)
    assert list(signature.parameters) == ["x", "weights", "weighted"]
    assert signature.parameters["weighted"].default is True


def test_wc_aggregate_cases_rejects_non_boolean_weight_dispatch():
    with pytest.raises(TypeError, match="weighted must be a boolean"):
        wcAggregateCases([[1], [1]], weighted="yes")


def test_seqclustname_first_medoid_ties_and_sps_labels_match_live_r(r_parity_session):
    rows = [["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"]]
    groups = [1, 1, 2, 2]
    diss = np.array(
        [
            [0.0, 1.0, 4.0, 4.0],
            [1.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 1.0],
            [4.0, 4.0, 1.0, 0.0],
        ]
    )
    oracle = r_parity_session.seqclustname_contract(rows, groups, diss)

    result = seqclustname(np.asarray(rows), groups, diss)

    assert oracle["medoid_indices_1b"].tolist() == [1, 3]
    assert oracle["levels"] == ["A/2", "C/2"]
    assert list(result.categories) == oracle["levels"]
    assert result.codes.tolist() == oracle["codes"].tolist()


def _categorical_values(result: pd.Categorical) -> list[str | None]:
    return [None if code == -1 else str(result.categories[code]) for code in result.codes.tolist()]


def test_seqclustname_attached_weights_change_medoids_like_live_r(r_parity_session):
    rows = [
        ["A", "B", "B"],
        ["A", "A", "A"],
        ["B", "B", "B"],
        ["B", "A", "A"],
        ["A", "A", "B"],
        ["B", "B", "A"],
    ]
    groups = [1, 1, 1, 2, 2, 2]
    diss = np.full((6, 6), 10.0)
    np.fill_diagonal(diss, 0.0)
    for start in (0, 3):
        diss[start, start + 1] = diss[start + 1, start] = 1.0
        diss[start, start + 2] = diss[start + 2, start] = 1.0
        diss[start + 1, start + 2] = diss[start + 2, start + 1] = 3.0
    weights = np.array([1.0, 10.0, 1.0, 1.0, 10.0, 1.0])
    sequence = STSList(
        data=np.array(
            [
                [0, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
                [0, 0, 1],
                [1, 1, 0],
            ],
            dtype=int,
        ),
        alphabet=["A", "B"],
        weights=weights,
    )
    unweighted_r = r_parity_session.seqclustname_contract(
        rows, groups, diss, weights=weights, weighted=False
    )
    weighted_r = r_parity_session.seqclustname_contract(
        rows, groups, diss, weights=weights, weighted=True
    )

    unweighted = seqclustname(sequence, groups, diss, weighted=False)
    weighted = seqclustname(sequence, groups, diss, weighted=True)

    assert unweighted_r["levels"] == ["A/1-B/2", "B/1-A/2"]
    assert weighted_r["levels"] == ["A/3", "A/2-B/1"]
    assert list(unweighted.categories) == unweighted_r["levels"]
    assert list(weighted.categories) == weighted_r["levels"]
    assert unweighted.codes.tolist() == unweighted_r["codes"].tolist()
    assert weighted.codes.tolist() == weighted_r["codes"].tolist()


def test_seqclustname_weighted_and_unweighted_percentages_match_live_r(
    r_parity_session,
):
    rows = [["A", "B"], ["A", "A"], ["B", "B"]]
    groups = [1, 2, 2]
    diss = np.array([[0.0, 4.0, 4.0], [4.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    weights = np.array([2.0, 1.0, 7.0])
    sequence = STSList(
        data=np.array([[0, 1], [0, 0], [1, 1]], dtype=int),
        alphabet=["A", "B"],
        weights=weights,
    )
    unweighted_r = r_parity_session.seqclustname_contract(
        rows, groups, diss, weights=weights, weighted=False, perc=True
    )
    weighted_r = r_parity_session.seqclustname_contract(
        rows, groups, diss, weights=weights, weighted=True, perc=True
    )

    unweighted = seqclustname(sequence, groups, diss, weighted=False, perc=True)
    weighted = seqclustname(sequence, groups, diss, weighted=True, perc=True)

    assert unweighted_r["levels"] == ["A/1-B/1 (33.3%)", "A/2 (66.7%)"]
    assert weighted_r["levels"] == ["A/1-B/1 (20%)", "B/2 (80%)"]
    assert list(unweighted.categories) == unweighted_r["levels"]
    assert list(weighted.categories) == weighted_r["levels"]
    assert _categorical_values(unweighted) == unweighted_r["values"]
    assert _categorical_values(weighted) == weighted_r["values"]


@pytest.mark.parametrize(
    ("groups", "levels"),
    [
        ([2, 2, 1, 1], None),
        (["z", "z", "a", "a"], None),
        (
            pd.Categorical(["z", "z", "a", "a"], categories=["z", "unused", "a"]),
            ["z", "unused", "a"],
        ),
    ],
)
def test_seqclustname_group_types_levels_and_empty_categories_match_live_r(
    r_parity_session, groups, levels
):
    rows = [["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"]]
    diss = np.array(
        [[0.0, 1.0, 4.0, 4.0], [1.0, 0.0, 4.0, 4.0], [4.0, 4.0, 0.0, 1.0], [4.0, 4.0, 1.0, 0.0]]
    )
    oracle = r_parity_session.seqclustname_contract(rows, list(groups), diss, group_levels=levels)

    result = seqclustname(pd.DataFrame(rows), groups, diss)

    assert list(result.categories) == oracle["levels"]
    assert result.codes.tolist() == oracle["codes"].tolist()
    assert _categorical_values(result) == oracle["values"]


def test_seqclustname_missing_groups_and_single_group_bug_match_live_r(r_parity_session):
    rows = [["A", "A"], ["B", "B"]]
    diss = np.array([[0.0, 1.0], [1.0, 0.0]])
    missing_r = r_parity_session.seqclustname_contract(rows, ["z", None], diss)
    single_r = r_parity_session.seqclustname_contract(rows, [1, 1], diss)

    assert missing_r == {"ok": False, "error": "'x' and 'w' must have the same length"}
    with pytest.raises(ValueError, match="'x' and 'w' must have the same length"):
        seqclustname(np.asarray(rows), ["z", None], diss)

    single = seqclustname(np.asarray(rows), [1, 1], diss)
    assert list(single.categories) == single_r["levels"] == ["A/2"]
    assert single.codes.tolist() == single_r["codes"].tolist() == [-1, -1]
    assert _categorical_values(single) == single_r["values"] == [None, None]


def test_seqclustname_required_diss_and_shape_validation_match_live_r(r_parity_session):
    rows = [["A", "A"], ["B", "B"]]
    diss = np.array([[0.0, 1.0], [1.0, 0.0]])
    missing_r = r_parity_session.seqclustname_contract(rows, [1, 2], diss, omit_diss=True)
    bad_group_r = r_parity_session.seqclustname_contract(rows, [1, 2, 3], diss)
    nonsquare_r = r_parity_session.seqclustname_contract(rows, [1, 2], np.ones((2, 3)))

    assert 'argument "diss" is missing' in missing_r["error"]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'diss'"):
        seqclustname(np.asarray(rows), [1, 2])  # type: ignore[call-arg]
    assert bad_group_r["ok"] is False
    with pytest.raises(ValueError, match="group must have one value per sequence"):
        seqclustname(np.asarray(rows), [1, 2, 3], diss)
    # R currently recycles a 2x3 matrix with warnings; Python deliberately
    # rejects it at the public boundary instead of propagating malformed data.
    assert nonsquare_r["ok"] is True
    with pytest.raises(ValueError, match="diss must be a square matrix"):
        seqclustname(np.asarray(rows), [1, 2], np.ones((2, 3)))


def test_seqclustname_signature_matches_weightedcluster_2_0():
    signature = inspect.signature(seqclustname)
    assert list(signature.parameters) == ["seqdata", "group", "diss", "weighted", "perc"]
    assert signature.parameters["diss"].default is inspect.Parameter.empty
    assert signature.parameters["weighted"].default is True
    assert signature.parameters["weighted"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["perc"].default is False


@pytest.mark.parametrize(
    ("kwargs", "name"), [({"weighted": "yes"}, "weighted"), ({"perc": 1}, "perc")]
)
def test_seqclustname_rejects_truthy_boolean_controls(kwargs, name):
    rows = np.array([[0, 0], [1, 1]], dtype=int)
    with pytest.raises(TypeError, match=rf"{name} must be a boolean"):
        seqclustname(rows, [1, 2], np.array([[0.0, 1.0], [1.0, 0.0]]), **kwargs)


def test_as_seqtree_hclust_topology_and_predictors_match_live_r(r_parity_session):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    points = np.array([0.0, 0.2, 2.0, 5.0])
    hierarchy_diss = np.abs(points[:, None] - points[None, :])
    diss = hierarchy_diss**2
    rows = [["A", "A"], ["A", "B"], ["B", "B"], ["C", "C"]]
    oracle = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="hclust",
        hierarchy_diss=hierarchy_diss,
        hierarchy_method="average",
        ncluster=3,
    )
    linkage_matrix = linkage(squareform(hierarchy_diss, checks=False), method="average")

    result = as_seqtree(
        linkage_matrix,
        seqdata=np.asarray(rows),
        diss=diss,
        ncluster=3,
    )

    assert oracle["ok"] is True
    assert result["class"] == oracle["class"] == ["seqtree", "disstree", "list"]
    assert list(result["data"].columns) == oracle["data_columns"] == ["Split2", "Split3"]
    np.testing.assert_array_equal(result["data"].to_numpy(), oracle["data_codes"])
    np.testing.assert_array_equal(result["fitted"], oracle["fitted"])


def _flatten_phase1_tree(tree: dict) -> np.ndarray:
    rows: list[list[float]] = []

    def walk(node: dict, parent: int | None = None, side: int | None = None) -> None:
        split = node.get("split")
        split_info = {} if split is None else split["info"]
        rows.append(
            [
                float(node["id"]),
                np.nan if parent is None else float(parent),
                np.nan if side is None else float(side),
                float(node["info"]["depth"]),
                float(node["info"]["splitschedule"]),
                float(node["info"]["n"]),
                float(node["info"]["vardis"]),
                float(node["info"]["medoid"]) + 1.0,
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


def _phase1_python_nodes(tree: dict) -> list[dict]:
    nodes: list[dict] = []

    def walk(node: dict) -> None:
        nodes.append(node)
        if node.get("kids") is not None:
            walk(node["kids"][0])
            walk(node["kids"][1])

    walk(tree["root"])
    return nodes


def _assert_phase1_tree_contract(result: dict, oracle: dict) -> None:
    # These statistics are deterministic compositions of public traminex
    # discrepancy/inertia primitives; R and Python may differ only by rounding.
    np.testing.assert_allclose(
        _flatten_phase1_tree(result),
        oracle["nodes"],
        rtol=0,
        atol=1e-10,
        equal_nan=True,
    )
    nodes = _phase1_python_nodes(result)
    assert len(nodes) == len(oracle["members_1b"])
    for node, members_1b, probability, index, labels in zip(
        nodes,
        oracle["members_1b"],
        oracle["split_prob"],
        oracle["split_index"],
        oracle["split_labels"],
        strict=True,
    ):
        # Directly document R's one-based members versus Python's zero-based members.
        np.testing.assert_array_equal(node["info"]["ind"], members_1b - 1)
        if node["split"] is None:
            assert probability.size == index.size == 0
            assert labels == []
        else:
            np.testing.assert_allclose(node["split"]["prob"], probability, rtol=0, atol=1e-10)
            np.testing.assert_array_equal(node["split"]["index"], index)
            assert [str(value) for value in node["split"]["labels"]] == labels

    np.testing.assert_array_equal(result["fitted"], oracle["fitted"])
    assert dtlabels(result) == dict(zip(oracle["label_names"], oracle["label_values"], strict=True))
    np.testing.assert_array_equal(dtcut(result, 2, labels=False), oracle["cut2_ids"])
    assert list(dtcut(result, 2, labels=True)) == oracle["cut2_labels"]
    python_adjustment = result["adjustment"].stat["t0"].to_dict()
    np.testing.assert_allclose(
        [python_adjustment[name] for name in oracle["adjustment"]],
        list(oracle["adjustment"].values()),
        rtol=0,
        atol=1e-10,
    )


def _phase1_hierarchy_fixture():
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    points = np.array([0.0, 0.2, 2.0, 5.0])
    hierarchy_diss = np.abs(points[:, None] - points[None, :])
    diss = hierarchy_diss**2
    rows = [["A", "A"], ["A", "B"], ["B", "B"], ["C", "C"]]
    linkage_matrix = linkage(squareform(hierarchy_diss, checks=False), method="average")
    return rows, hierarchy_diss, diss, linkage_matrix


def test_as_seqtree_complete_hclust_node_and_result_contract(r_parity_session):
    rows, hierarchy_diss, diss, linkage_matrix = _phase1_hierarchy_fixture()
    oracle = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="hclust",
        hierarchy_diss=hierarchy_diss,
        ncluster=3,
    )

    result = as_seqtree(linkage_matrix, np.asarray(rows), diss, ncluster=3)

    _assert_phase1_tree_contract(result, oracle)
    assert oracle["names"] == ["fitted", "info", "data", "weights", "root"]
    assert set(oracle["names"]) <= set(result)
    assert result["info"]["method"] == oracle["info_method"] == "disstree"
    assert result["info"]["n"] == oracle["info_n"] == 4.0
    assert (
        result["info"]["parameters"]
        == oracle["parameters"]
        == {
            "minSize": 1.0,
            "maxdepth": 3,
            "R": 0,
            "pval": 1.0,
        }
    )
    assert result["info"]["weight_permutation"] == oracle["weight_permutation"] == "diss"
    assert "terms" not in result and oracle["terms_is_null"] is True
    np.testing.assert_array_equal(result["weights"], oracle["weights"])
    assert [
        [str(value) for value in result["data"][column].cat.categories]
        for column in result["data"].columns
    ] == oracle["data_levels"]
    assert any(node["info"]["ind"].size == 1 for node in _phase1_python_nodes(result))
    # Medoids and split-variable fields are converted explicitly in the flattened check.
    np.testing.assert_array_equal(_flatten_phase1_tree(result)[:, 7], oracle["nodes"][:, 7])
    np.testing.assert_array_equal(_flatten_phase1_tree(result)[:, 9], oracle["nodes"][:, 9])


def test_as_seqtree_twins_and_hclust_share_scipy_linkage_representation(r_parity_session):
    rows, hierarchy_diss, diss, linkage_matrix = _phase1_hierarchy_fixture()
    hclust_r = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="hclust",
        hierarchy_diss=hierarchy_diss,
        ncluster=3,
    )
    twins_r = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="twins",
        hierarchy_diss=hierarchy_diss,
        ncluster=3,
    )

    result = as_seqtree(linkage_matrix, np.asarray(rows), diss, ncluster=3)

    np.testing.assert_array_equal(twins_r["data_codes"], hclust_r["data_codes"])
    np.testing.assert_allclose(
        twins_r["nodes"], hclust_r["nodes"], rtol=0, atol=1e-10, equal_nan=True
    )
    _assert_phase1_tree_contract(result, twins_r)


def test_as_seqtree_ncluster_two_preserves_r_colon_edge(r_parity_session):
    rows, hierarchy_diss, diss, linkage_matrix = _phase1_hierarchy_fixture()
    oracle = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="hclust",
        hierarchy_diss=hierarchy_diss,
        ncluster=2,
    )

    result = as_seqtree(linkage_matrix, np.asarray(rows), diss, ncluster=2)

    assert oracle["data_columns"] == ["Split2", "Split3"]
    assert list(result["data"].columns) == oracle["data_columns"]
    assert result["info"]["parameters"]["maxdepth"] == oracle["parameters"]["maxdepth"] == 3
    _assert_phase1_tree_contract(result, oracle)


def test_as_seqtree_default_predictors_advance_past_nonbinary_schedule(r_parity_session):
    rows = [["A"], ["B"], ["C"], ["D"], ["E"]]
    points = np.array([0.0, 0.2, 1.0, 4.0, 5.0])
    diss = np.abs(points[:, None] - points[None, :])
    predictors = {
        "Split2": ["left", "left", "left", "right", "right"],
        "Split3": ["same", "same", "same", "u", "v"],
        "Split4": ["x", "x", "y", "u", "v"],
    }
    frame = pd.DataFrame({name: pd.Categorical(values) for name, values in predictors.items()})
    oracle = r_parity_session.as_seqtree_contract(
        rows, diss, input_kind="default", predictors=predictors
    )

    result = as_seqtree(frame, np.asarray(rows), diss)

    assert result["info"]["parameters"]["maxdepth"] == 4
    assert result["root"]["kids"][0]["split"]["var"] == "Split4"
    assert result["root"]["kids"][0]["info"]["depth"] == 2
    assert result["root"]["kids"][0]["kids"][0]["info"]["depth"] == 4
    _assert_phase1_tree_contract(result, oracle)


def test_as_seqtree_attached_and_unit_weights_match_live_r(r_parity_session):
    rows, hierarchy_diss, diss, linkage_matrix = _phase1_hierarchy_fixture()
    weights = np.array([1.0, 4.0, 2.0, 3.0])
    sequence = STSList(
        data=np.array([[0, 0], [0, 1], [1, 1], [2, 2]], dtype=int),
        alphabet=["A", "B", "C"],
        weights=weights,
    )
    weighted_r = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="hclust",
        hierarchy_diss=hierarchy_diss,
        ncluster=3,
        weights=weights,
        weighted=True,
    )
    unit_r = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="hclust",
        hierarchy_diss=hierarchy_diss,
        ncluster=3,
        weights=weights,
        weighted=False,
    )

    weighted_result = as_seqtree(linkage_matrix, sequence, diss, True, 3)
    unit_result = as_seqtree(linkage_matrix, sequence, diss, False, 3)

    np.testing.assert_array_equal(weighted_result["weights"], weighted_r["weights"])
    np.testing.assert_array_equal(unit_result["weights"], unit_r["weights"])
    _assert_phase1_tree_contract(weighted_result, weighted_r)
    _assert_phase1_tree_contract(unit_result, unit_r)


def test_as_seqtree_validation_and_required_ncluster_contract(r_parity_session):
    rows, hierarchy_diss, diss, linkage_matrix = _phase1_hierarchy_fixture()
    missing_r = r_parity_session.as_seqtree_contract(
        rows,
        diss,
        input_kind="hclust",
        hierarchy_diss=hierarchy_diss,
        ncluster=None,
    )
    empty_r = r_parity_session.as_seqtree_contract(rows, diss, input_kind="default", predictors={})

    assert 'argument "ncluster" is missing' in missing_r["error"]
    with pytest.raises(TypeError, match="ncluster is required"):
        as_seqtree(linkage_matrix, np.asarray(rows), diss)
    assert empty_r == {"ok": False, "error": " [!] ncluster should be bigger than 2"}
    with pytest.raises(ValueError, match="ncluster should be bigger than 2"):
        as_seqtree(pd.DataFrame(index=np.arange(4)), np.asarray(rows), diss)
    with pytest.raises(ValueError, match="same number of rows"):
        as_seqtree(pd.DataFrame({"Split2": [1, 1, 2]}), np.asarray(rows), diss)
    with pytest.raises(ValueError, match="square"):
        as_seqtree(linkage_matrix, np.asarray(rows), np.ones((4, 3)), ncluster=3)
    with pytest.raises(ValueError, match="linkage"):
        as_seqtree(np.ones((2, 4)), np.asarray(rows), diss, ncluster=3)
    with pytest.raises(ValueError, match="between 2 and 4"):
        as_seqtree(linkage_matrix, np.asarray(rows), diss, ncluster=5)
    with pytest.raises(ValueError, match="ncluster must be an integer"):
        as_seqtree(linkage_matrix, np.asarray(rows), diss, ncluster=2.5)
    with pytest.raises(TypeError, match="weighted must be a boolean"):
        as_seqtree(linkage_matrix, np.asarray(rows), diss, weighted="yes", ncluster=3)


def test_as_seqtree_cut_label_prune_and_headless_plot_interoperate():
    rows, _hierarchy_diss, diss, linkage_matrix = _phase1_hierarchy_fixture()
    tree = as_seqtree(linkage_matrix, np.asarray(rows), diss, ncluster=3)

    assert dtcut(tree, 2, labels=False).shape == (4,)
    assert dtlabels(tree)
    pruned = dtprune(tree, 2)
    assert pruned["adjustment"] is not tree["adjustment"]
    assert pruned["info"]["adjustment"] is pruned["adjustment"]
    assert np.unique(pruned["fitted"]).size == 2

    matplotlib = import_optional_or_skip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plot_proptree_tree(tree)
    assert axis.texts
    plt.close(figure)


def test_fuzzyseqplot_canonical_membership_matrix_matches_live_r(r_parity_session):
    rows = [["A", "A"], ["A", "B"], ["B", "B"]]
    memberships = np.array([[0.9, 0.1], [0.4, 0.6], [0.0, 0.8]], dtype=float)
    weights = np.array([2.0, 3.0, 5.0])
    oracle = r_parity_session.fuzzyseqplot_contract(
        rows,
        memberships,
        group_names=["alpha", "beta"],
        weights=weights,
        membership_threshold=0.15,
        members_weighted=False,
        memb_exp=2,
        sortv="membership",
        plot_kwargs={"main": "Pinned", "with.legend": False},
    )
    sequence = STSList(
        data=np.array([[0, 0], [0, 1], [1, 1]], dtype=int),
        alphabet=["A", "B"],
        weights=weights,
        ids=["case-a", "case-b", "case-c"],
    )

    # This also pins the helper's R column-major orientation before comparing Python.
    np.testing.assert_array_equal(
        oracle["rows"],
        np.array([["A", "A"], ["A", "B"], ["A", "B"], ["B", "B"]]),
    )
    assert oracle["groups"] == ["alpha", "alpha", "beta", "beta"]
    np.testing.assert_allclose(oracle["weights"], [0.81, 0.16, 0.36, 0.64], rtol=0, atol=1e-10)
    assert oracle["call_count"] == 1
    assert oracle["returned"] is True
    assert oracle["return_class"] == ["fuzzyseqplot_spy"]
    assert oracle["type"] == "i"
    np.testing.assert_allclose(oracle["sortv"], oracle["weights"], rtol=0, atol=1e-10)
    assert oracle["forwarded_kwargs"] == {"main": "Pinned", "with.legend": False}
    assert not list(r_parity_session.r("ls(pattern='^wc_phase1_fuzzy_', envir=.GlobalEnv)"))

    figure, axes = fuzzyseqplot(
        sequence,
        group=pd.DataFrame(memberships, columns=["alpha", "beta"]),
        membership_threashold=0.15,
        members_weighted=False,
        memb_exp=2,
        sortv="membership",
    )
    assert np.asarray(axes, dtype=object).reshape(-1).size == 2
    figure.canvas.draw()
    import matplotlib.pyplot as plt

    plt.close(figure)


def _phase1_fuzzy_fixture(*, attached_weights: bool = True):
    memberships = np.array([[0.9, 0.1], [0.4, 0.6], [0.0, 0.8]], dtype=float)
    weights = np.array([2.0, 3.0, 5.0]) if attached_weights else None
    sequence = STSList(
        data=np.array([[0, 0], [0, 1], [1, 1]], dtype=int),
        alphabet=["A", "B"],
        weights=weights,
        ids=["case-a", "case-b", "case-c"],
    )
    return sequence, memberships


def test_fuzzyseqplot_preparation_exact_weights_order_ids_and_inclusive_threshold(
    r_parity_session,
):
    from weightedcluster.fuzzy import _prepare_fuzzyseqplot_data

    rows = [["A", "A"], ["A", "B"], ["B", "B"]]
    sequence, memberships = _phase1_fuzzy_fixture()
    original_data = sequence.data.copy()
    original_weights = sequence.weights.copy()
    original_memberships = memberships.copy()

    unweighted_r = r_parity_session.fuzzyseqplot_contract(
        rows,
        memberships,
        group_names=["alpha", "beta"],
        weights=original_weights,
        membership_threshold=0.16,
        members_weighted=False,
        memb_exp=2,
        sortv="membership",
    )
    weighted_r = r_parity_session.fuzzyseqplot_contract(
        rows,
        memberships,
        group_names=["alpha", "beta"],
        weights=original_weights,
        membership_threshold=0.16,
        members_weighted=True,
        memb_exp=2,
    )
    unweighted = _prepare_fuzzyseqplot_data(
        sequence,
        pd.DataFrame(memberships, columns=["alpha", "beta"]),
        membership_threashold=0.16,
        members_weighted=False,
        memb_exp=2,
        sortv="membership",
    )
    weighted = _prepare_fuzzyseqplot_data(
        sequence,
        pd.DataFrame(memberships, columns=["alpha", "beta"]),
        membership_threashold=0.16,
        members_weighted=True,
        memb_exp=2,
    )

    # 0.4**2 remains at the exactly inclusive 0.16 boundary.
    np.testing.assert_allclose(
        unweighted["membership"], [0.81, 0.16, 0.36, 0.64], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        unweighted["plot_weights"], unweighted_r["weights"], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(weighted["plot_weights"], weighted_r["weights"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        weighted["plot_weights"], [1.62, 0.48, 1.08, 3.2], rtol=0, atol=1e-10
    )
    # The exact R vector lives beside the sequence object, never inside it.
    assert unweighted["seqdata"].weights is None
    assert weighted["seqdata"].weights is None
    assert list(unweighted["group"]) == unweighted_r["groups"]
    assert list(unweighted["source_indices"]) == [0, 1, 1, 2]
    assert list(unweighted["source_ids"]) == ["case-a", "case-b", "case-b", "case-c"]
    np.testing.assert_allclose(unweighted["sortv"], unweighted["membership"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(sequence.data, original_data)
    np.testing.assert_array_equal(sequence.weights, original_weights)
    np.testing.assert_array_equal(memberships, original_memberships)


def test_fuzzyseqplot_fanny_equivalent_mapping_and_default_names_match_live_r(
    r_parity_session,
):
    from weightedcluster.fuzzy import _prepare_fuzzyseqplot_data

    rows = [["A", "A"], ["A", "B"], ["B", "B"]]
    sequence, memberships = _phase1_fuzzy_fixture()
    oracle = r_parity_session.fuzzyseqplot_contract(
        rows,
        memberships,
        group_kind="fanny",
        weights=sequence.weights,
        membership_threshold=0.15,
    )
    mapping = {"membership": memberships}

    prepared = _prepare_fuzzyseqplot_data(
        sequence,
        mapping,
        membership_threashold=0.15,
    )

    assert [str(value) for value in prepared["group"]] == oracle["groups"]
    np.testing.assert_allclose(prepared["plot_weights"], oracle["weights"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(prepared["plot_weights"], [1.8, 1.2, 1.8, 4.0], rtol=0, atol=1e-10)
    assert prepared["seqdata"].weights is None
    assert prepared["group_levels"] == (1, 2)

    class FannyEquivalent:
        membership = pd.DataFrame(memberships, columns=["first", "second"])

    protocol = _prepare_fuzzyseqplot_data(
        sequence,
        FannyEquivalent(),
        membership_threashold=0.15,
    )
    assert protocol["group_levels"] == ("first", "second")


def test_fuzzyseqplot_none_and_factor_plot_once_then_raise_exact_upstream_error(
    r_parity_session, monkeypatch
):
    import weightedcluster.fuzzy as fuzzy_module

    rows = [["A", "A"], ["A", "B"], ["B", "B"]]
    sequence, _memberships = _phase1_fuzzy_fixture()
    calls: list[dict] = []

    def recorder(_seqdata, **kwargs):
        calls.append(kwargs)
        return object(), object()

    monkeypatch.setattr(fuzzy_module, "_render_fuzzyseqplot_panels", recorder)
    none_r = r_parity_session.fuzzyseqplot_contract(
        rows,
        None,
        group_kind="none",
        weights=sequence.weights,
        type="d",
    )
    with pytest.raises(ValueError, match="membership matrix with one row per sequence") as error:
        fuzzyseqplot(sequence, None, type="d")
    assert str(error.value) == none_r["error"]
    assert none_r["call_count"] == 1 and none_r["type"] is None
    assert len(calls) == 1 and "type" not in calls[0]

    calls.clear()
    factor = pd.Categorical(["left", "left", "right"])
    factor_r = r_parity_session.fuzzyseqplot_contract(
        rows,
        None,
        group_kind="factor",
        factor_group=list(factor),
        weights=sequence.weights,
        type="d",
    )
    with pytest.warns(RuntimeWarning, match="not meaningful for factors") as warning_records:
        with pytest.raises(ValueError) as factor_error:
            fuzzyseqplot(sequence, factor, type="d")
    assert str(warning_records[0].message) == factor_r["warnings"][0]
    assert str(factor_error.value) == factor_r["error"]
    assert factor_r["call_count"] == 1 and factor_r["type"] is None
    assert len(calls) == 1 and "type" not in calls[0]


def test_fuzzyseqplot_python_boundary_validation_and_absent_weights():
    from weightedcluster.fuzzy import _prepare_fuzzyseqplot_data

    sequence, memberships = _phase1_fuzzy_fixture()
    with pytest.raises(TypeError, match="membership matrix"):
        _prepare_fuzzyseqplot_data(sequence, memberships[:, 0])
    with pytest.raises(ValueError, match="at least one row and one column"):
        _prepare_fuzzyseqplot_data(sequence, np.empty((3, 0)))
    with pytest.raises(ValueError, match="at least one row and one column"):
        _prepare_fuzzyseqplot_data(sequence, np.empty((0, 2)))
    with pytest.raises(ValueError, match="finite"):
        _prepare_fuzzyseqplot_data(
            sequence,
            np.array([[np.nan, 0.0], [0.4, 0.6], [0.0, 0.8]]),
        )
    with pytest.raises(ValueError, match="one row per sequence"):
        _prepare_fuzzyseqplot_data(sequence, memberships[:2])
    with pytest.raises(ValueError, match="membership_threashold must be a finite number"):
        _prepare_fuzzyseqplot_data(sequence, memberships, membership_threashold=np.inf)
    with pytest.raises(ValueError, match="memb_exp must be a finite number"):
        _prepare_fuzzyseqplot_data(sequence, memberships, memb_exp=np.nan)
    exponent_zero = _prepare_fuzzyseqplot_data(sequence, memberships, memb_exp=0)
    np.testing.assert_array_equal(exponent_zero["membership"], np.ones(6))
    without_attached_weights, _ = _phase1_fuzzy_fixture(attached_weights=False)
    absent = _prepare_fuzzyseqplot_data(
        without_attached_weights,
        memberships,
        membership_threashold=0.15,
    )
    np.testing.assert_allclose(absent["plot_weights"], [0.9, 0.4, 0.6, 0.8], rtol=0, atol=1e-10)
    assert absent["seqdata"].weights is None


def test_fuzzyseqplot_permissive_membership_power_and_empty_boundaries_match_live_r(
    r_parity_session,
):
    from weightedcluster.fuzzy import _prepare_fuzzyseqplot_data

    rows = [["A", "A"], ["B", "B"]]
    sequence = STSList(
        data=np.array([[0, 0], [1, 1]], dtype=int),
        alphabet=["A", "B"],
        weights=np.ones(2),
    )
    cases = [
        (
            np.array([[1.2, 0.0], [-0.2, 1.0]]),
            {},
        ),
        (
            np.array([[0.0, 1.0], [0.5, 0.5]]),
            {"memb_exp": -1.0},
        ),
        (
            np.array([[0.2, 0.1], [0.3, 0.4]]),
            {"membership_threshold": 2.0},
        ),
    ]

    for membership, oracle_kwargs in cases:
        oracle = r_parity_session.fuzzyseqplot_contract(rows, membership, **oracle_kwargs)
        python_kwargs = dict(oracle_kwargs)
        if "membership_threshold" in python_kwargs:
            python_kwargs["membership_threashold"] = python_kwargs.pop("membership_threshold")
        prepared = _prepare_fuzzyseqplot_data(sequence, membership, **python_kwargs)

        assert oracle["call_count"] == 1
        assert oracle["error"] is None
        assert oracle["returned"] is True
        decoded = np.asarray(prepared["seqdata"].alphabet, dtype=object)[prepared["seqdata"].data]
        np.testing.assert_array_equal(decoded, oracle["rows"])
        # The exact R-derived vector is carried beside the sequence object, not
        # inside it: it may hold positive infinity, which traminex's STSList
        # refuses. See weightedcluster.fuzzy._copy_sts_rows and TECHNICAL_DEBT
        # item 6.
        np.testing.assert_allclose(
            prepared["plot_weights"],
            oracle["weights"],
            rtol=0,
            atol=1e-10,
            equal_nan=True,
        )
        assert prepared["seqdata"].weights is None
        assert [str(value) for value in prepared["group"]] == oracle["groups"]


@pytest.mark.parametrize("sortv", ["membership", "from.start"])
def test_fuzzyseqplot_forwards_type_sort_and_valid_plot_kwargs(
    r_parity_session, monkeypatch, sortv
):
    import weightedcluster.fuzzy as fuzzy_module

    sequence, memberships = _phase1_fuzzy_fixture()
    calls: list[dict] = []

    def index_spy(seqdata, **kwargs):
        calls.append(
            {
                "weights": np.asarray(seqdata.weights, dtype=float).copy(),
                "ids": list(seqdata.ids),
                **kwargs,
            }
        )
        return kwargs["ax"].figure, kwargs["ax"]

    monkeypatch.setattr(fuzzy_module, "_render_weighted_index_panel", index_spy)
    monkeypatch.setattr(
        fuzzy_module,
        "seqplot",
        lambda *_args, **_kwargs: pytest.fail("index panels must use the narrow adapter"),
    )
    oracle = r_parity_session.fuzzyseqplot_contract(
        [["A", "A"], ["A", "B"], ["B", "B"]],
        memberships,
        group_names=["alpha", "beta"],
        weights=sequence.weights,
        membership_threshold=0.15,
        members_weighted=False,
        memb_exp=2,
        sortv=sortv,
        plot_kwargs={"main": "Pinned", "with.legend": False},
    )
    figure, axes = fuzzyseqplot(
        sequence,
        pd.DataFrame(memberships, columns=["alpha", "beta"]),
        0.15,
        "i",
        False,
        2,
        sortv=sortv,
        main="Pinned",
        with_legend=False,
    )

    assert oracle["forwarded_kwargs"] == {"main": "Pinned", "with.legend": False}
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "Pinned"
    assert [axis.get_title() for axis in np.asarray(axes).reshape(-1)] == ["alpha", "beta"]
    assert len(calls) == 2
    assert [call["legend"] for call in calls] == [False, False]
    if sortv == "membership":
        np.testing.assert_allclose(calls[0]["sortv"], [0.81, 0.16], rtol=0, atol=1e-10)
        np.testing.assert_allclose(calls[1]["sortv"], [0.36, 0.64], rtol=0, atol=1e-10)
    else:
        assert [call["sortv"] for call in calls] == ["from.start", "from.start"]
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_traminex_refuses_a_non_finite_attached_sequence_weight():
    """The sibling contract this repository's fuzzy sidecar exists to respect.

    traminex commit ``1ebb46cd`` rejects non-finite case weights on ``STSList``
    construction and assignment.  Pinning it here means a future change that
    "simplifies" the sidecar by attaching the vector again fails immediately and
    locally, instead of surfacing as a live-R parity failure.
    """
    data = np.array([[0, 0], [1, 1]], dtype=int)
    with pytest.raises(ValueError, match="finite"):
        STSList(data=data, alphabet=["A", "B"], weights=np.array([np.inf, 1.0]))

    finite = STSList(data=data, alphabet=["A", "B"], weights=np.array([1.0, 2.0]))
    np.testing.assert_allclose(finite.weights, [1.0, 2.0])


def test_fuzzyseqplot_default_weighted_index_refuses_an_infinite_membership_weight():
    """A nonfinite requested geometry fails before index-row selection."""
    import matplotlib

    matplotlib.use("Agg")

    from weightedcluster.fuzzy import _prepare_fuzzyseqplot_data

    sequence, _ = _phase1_fuzzy_fixture(attached_weights=False)
    memberships = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=float)

    prepared = _prepare_fuzzyseqplot_data(sequence, memberships, memb_exp=-1.0)
    assert not np.all(np.isfinite(prepared["plot_weights"]))
    assert prepared["seqdata"].weights is None

    with pytest.raises(ValueError, match="finite and non-negative"):
        fuzzyseqplot(sequence, memberships, 0, "i", True, -1.0)


def test_fuzzyseqplot_panels_receive_the_sidecar_weights_not_the_stripped_sequence(monkeypatch):
    """The second `_copy_sts_rows` call site must read the sidecar.

    Preparation now leaves ``seqdata.weights`` as ``None``, so a rendering path
    that still derives panel weights from the sequence object silently falls back
    to ones and drops the fuzzy weighting entirely. Nothing else in the suite
    catches that: an in-memory reversion making the panel path ignore the sidecar
    left all other tests green.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import weightedcluster.fuzzy as fuzzy_module

    sequence, memberships = _phase1_fuzzy_fixture()
    captured: list[np.ndarray | None] = []

    def index_spy(seqdata, **kwargs):
        captured.append(None if seqdata.weights is None else np.asarray(seqdata.weights, float))
        return kwargs["ax"].figure, kwargs["ax"]

    monkeypatch.setattr(fuzzy_module, "_render_weighted_index_panel", index_spy)
    figure, _ = fuzzyseqplot(
        sequence, pd.DataFrame(memberships, columns=["alpha", "beta"]), 0.15, "i", False, 2
    )

    # memb^2 flattened column-major, filtered at the inclusive 0.15 threshold,
    # split across the two membership panels.
    assert len(captured) == 2
    np.testing.assert_allclose(captured[0], [0.81, 0.16], rtol=0, atol=1e-10)
    np.testing.assert_allclose(captured[1], [0.36, 0.64], rtol=0, atol=1e-10)
    plt.close(figure)


def test_fuzzyseqplot_refuses_an_unprobed_plot_type_for_a_non_finite_weight():
    """A plot type whose infinity behaviour is unestablished must refuse, not guess."""
    import matplotlib

    matplotlib.use("Agg")

    sequence, _ = _phase1_fuzzy_fixture(attached_weights=False)
    memberships = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="finite and non-negative"):
        fuzzyseqplot(sequence, memberships, 0, "d", True, -1.0)


def test_fuzzyseqplot_rejects_duplicate_or_unknown_plot_keywords():
    sequence, memberships = _phase1_fuzzy_fixture()
    group = pd.DataFrame(memberships, columns=["alpha", "beta"])

    with pytest.raises(TypeError, match="with_legend was specified more than once"):
        fuzzyseqplot(
            sequence,
            group,
            with_legend=False,
            **{"with.legend": False},
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown_plot_option'"):
        fuzzyseqplot(sequence, group, unknown_plot_option=True)


def test_fuzzyseqplot_headless_index_and_distribution_render_smoke():
    matplotlib = import_optional_or_skip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sequence, memberships = _phase1_fuzzy_fixture()
    for plot_type in ("i", "d"):
        figure, axes = fuzzyseqplot(
            sequence,
            pd.DataFrame(memberships, columns=["alpha", "beta"]),
            membership_threashold=0.15,
            type=plot_type,
        )
        panels = np.asarray(axes, dtype=object).reshape(-1)
        assert len(panels) == 2
        assert [axis.get_title() for axis in panels] == ["alpha", "beta"]
        figure.canvas.draw()
        plt.close(figure)


def test_fuzzyseqplot_signature_matches_weightedcluster_2_0():
    signature = inspect.signature(fuzzyseqplot)
    assert list(signature.parameters) == [
        "seqdata",
        "group",
        "membership_threashold",
        "type",
        "members_weighted",
        "memb_exp",
        "plot_kwargs",
    ]
    assert signature.parameters["group"].default is None
    assert signature.parameters["membership_threashold"].default == 0
    assert signature.parameters["type"].default == "i"
    assert signature.parameters["members_weighted"].default is True
    assert signature.parameters["memb_exp"].default == 1
    assert signature.parameters["plot_kwargs"].kind is inspect.Parameter.VAR_KEYWORD

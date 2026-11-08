import numpy as np
import pytest

from weightedcluster.schemas import (
    ClusteringTable,
    LabeledObjectTable,
    LabeledStatsTable,
    coerce_stats_table,
    rows_to_stats_table,
    rows_to_table,
    validate_clustrange_result,
)


def test_rows_to_table_and_stats_table():
    rows = [
        {"ASW": 0.5, "k": 2},
        {"ASW": 0.4, "k": 3, "DB": 1.2},
    ]
    obj_table = rows_to_table(rows, keys=["k", "ASW", "DB"])
    assert obj_table["k"] == [2, 3]
    assert obj_table["ASW"] == [0.5, 0.4]
    assert obj_table["DB"][0] is None

    stats = rows_to_stats_table(rows, keys=["ASW", "DB"])
    assert np.allclose(stats["ASW"], [0.5, 0.4], equal_nan=True)
    assert np.isnan(stats["DB"][0])
    assert np.allclose(stats["DB"][1], 1.2)


@pytest.mark.parametrize(
    "rows,keys",
    (
        ([{"ASW": 0.5}], ["ASW", "ASW"]),
        ([{1: 0.5, "1": 0.4}], None),
        ([{"ASW": 0.5}], "ASW"),
    ),
)
def test_rows_to_table_rejects_duplicate_string_column_names(rows, keys):
    with pytest.raises(ValueError, match="keys|column"):
        rows_to_table(rows, keys=keys)


def test_coerce_stats_table_rejects_length_mismatch():
    with pytest.raises(ValueError, match="expected 2"):
        coerce_stats_table({"ASW": [0.1], "DB": [1.0, 2.0]}, expected_len=2)


@pytest.mark.parametrize("expected_len", (1.5, True, -1))
def test_coerce_stats_table_rejects_invalid_expected_length(expected_len):
    with pytest.raises(ValueError, match="expected_len"):
        coerce_stats_table({}, expected_len=expected_len)


@pytest.mark.parametrize(
    "stats",
    (
        {1: [0.5], "1": [0.4]},
        {"ASW": [[0.5, 0.4]]},
    ),
)
def test_coerce_stats_table_rejects_colliding_names_or_nonvector_columns(stats):
    with pytest.raises(ValueError, match="column|one-dimensional"):
        coerce_stats_table(stats)


def test_validate_clustrange_result_schema():
    obj = {
        "kvals": [2, 3],
        "clustering": {
            "cluster2": np.array([1, 1, 2, 2]),
            "cluster3": np.array([1, 2, 3, 1]),
        },
        "stats": {"ASW": [0.6, 0.5], "DB": np.array([0.9, 1.1])},
    }
    out = validate_clustrange_result(obj, n_obs=4)
    assert out["kvals"] == [2, 3]
    assert set(out["clustering"].keys()) == {"cluster2", "cluster3"}
    assert out["stats"]["ASW"] == [0.6, 0.5]


def test_validate_clustrange_result_accepts_numpy_kvals_iterable():
    obj = {
        "kvals": np.array([2, 3]),
        "clustering": {
            "cluster2": np.array([1, 1, 2, 2]),
            "cluster3": np.array([1, 2, 3, 1]),
        },
        "stats": {"ASW": [0.6, 0.5]},
    }

    out = validate_clustrange_result(obj, n_obs=4)

    assert out["kvals"] == [2, 3]


@pytest.mark.parametrize(
    "factory,values,row_names",
    (
        (LabeledStatsTable, {"ASW": [0.5, 0.4], "DB": [1.2]}, ["cluster2", "cluster3"]),
        (LabeledObjectTable, {"method": ["ward", "pam"], "k": [2]}, ["first", "second"]),
        (LabeledStatsTable, {"ASW": [0.5, 0.4]}, ["cluster2"]),
        (LabeledObjectTable, {"method": ["ward", "pam"]}, ["first", "first"]),
        (LabeledStatsTable, {"ASW": [0.5, 0.4]}, "12"),
    ),
)
def test_labeled_tables_reject_inconsistent_or_duplicate_row_metadata(factory, values, row_names):
    with pytest.raises(ValueError, match="row|column"):
        factory(values, row_names=row_names)


@pytest.mark.parametrize(
    "values,options,match",
    (
        ({"cluster2": [1, 2], "cluster3": [1]}, {}, "length"),
        ({"cluster2": np.array([[1, 2]])}, {}, "one-dimensional"),
        ({"cluster2": [1, 2]}, {"column_names": ["cluster3"]}, "column_names"),
        (
            {"cluster2": [1, 2]},
            {"factor_levels": {"cluster3": [1, 2, 3]}},
            "factor_levels",
        ),
    ),
)
def test_clustering_table_rejects_inconsistent_payload_or_metadata(values, options, match):
    with pytest.raises(ValueError, match=match):
        ClusteringTable(values, **options)


@pytest.mark.parametrize("kvals", ([2.5], [True], [2, 2], [1]))
def test_validate_clustrange_result_rejects_invalid_kvals(kvals):
    obj = {
        "kvals": kvals,
        "clustering": {"cluster2": np.array([1, 1, 2, 2])},
        "stats": {"ASW": [0.6] * len(kvals)},
    }

    with pytest.raises(ValueError, match="kvals"):
        validate_clustrange_result(obj, n_obs=4)


def test_validate_clustrange_result_requires_one_clustering_column_per_k():
    obj = {
        "kvals": [2, 3],
        "clustering": {"cluster2": np.array([1, 1, 2, 2])},
        "stats": {"ASW": [0.6, 0.5]},
    }

    with pytest.raises(ValueError, match="clustering.*cluster3"):
        validate_clustrange_result(obj, n_obs=4)


@pytest.mark.parametrize("n_obs", (4.5, True, -1))
def test_validate_clustrange_result_rejects_invalid_observation_count(n_obs):
    obj = {
        "kvals": [2],
        "clustering": {"cluster2": np.array([1, 1, 2, 2])},
        "stats": {"ASW": [0.6]},
    }

    with pytest.raises(ValueError, match="n_obs"):
        validate_clustrange_result(obj, n_obs=n_obs)

from __future__ import annotations

import ast
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest_plugins = ["tests.r_parity_utils"]


def _canonical_naive_fixture() -> dict[str, Any]:
    """Small full-rank fixture shared by the Phase 3A oracle and solver tests."""
    n_obs = 36
    x = np.asarray(
        [
            -2.4,
            -1.7,
            -0.9,
            0.2,
            1.1,
            2.3,
            -2.0,
            -1.2,
            -0.3,
            0.7,
            1.6,
            2.6,
        ]
        * 3,
        dtype=float,
    )
    cluster = [
        "late",
        "early",
        "middle",
        "late",
        "middle",
        "early",
        "middle",
        "late",
        "early",
        "early",
        "late",
        "middle",
        "early",
        "middle",
        "late",
        "middle",
        "early",
        "late",
        "late",
        "early",
        "middle",
        "early",
        "middle",
        "late",
        "middle",
        "late",
        "early",
        "late",
        "early",
        "middle",
        "early",
        "late",
        "middle",
        "middle",
        "late",
        "early",
    ]
    group = ["reference", "low", "high"] * 12
    binary = [False, True, False, True, True, False] * 6

    # Exercise model.frame omission without allowing the missing rows to alter
    # the retained cluster order.
    x[8] = np.nan
    cluster[31] = None
    return {
        "formula": "cluster ~ x + group + binary + x:group + I(x^2)",
        "data": {
            "cluster": cluster,
            "x": x,
            "group": group,
            "binary": binary,
        },
        "categorical": {"group": ["reference", "low", "high"]},
        "logical": ["binary"],
        "row_names": [f"case-{100 + index * 3}" for index in range(n_obs)],
    }


def _python_frame(fixture: dict[str, Any]) -> pd.DataFrame:
    data = pd.DataFrame(fixture["data"], index=fixture["row_names"])
    for column, levels in fixture["categorical"].items():
        data[column] = pd.Categorical(
            data[column],
            categories=levels,
            ordered=False,
        )
    return data


def _raw_character_naive_fixture(formula: str) -> dict[str, Any]:
    """Stable fixture whose raw-character levels distinguish ICU from code points."""
    levels = ["a", "A", "b", "B", "ä"]
    cluster_cycle = ["late", "early", "middle"]
    group = [level for block in range(18) for level in levels]
    cluster = [
        cluster_cycle[(block + 2 * level_index) % 3]
        for block in range(18)
        for level_index in range(len(levels))
    ]
    x = [
        0.37 * (((3 * block + 2 * level_index) % 13) - 6)
        for block in range(18)
        for level_index in range(len(levels))
    ]
    n_obs = len(group)
    return {
        "formula": formula,
        "data": {
            "cluster": cluster,
            "group": group,
            "x": x,
        },
        "row_names": [f"character-{index + 1}" for index in range(n_obs)],
    }


def _stable_naive_fixture(case: str) -> dict[str, Any]:
    """Generate one realized F5 table shared unchanged by R and Python."""
    specifications = {
        "balanced_standard": (90, 3101, "normal", [30, 30, 30], [30, 30, 30]),
        "small_scale": (84, 3102, "small", [28, 28, 28], [28, 28, 28]),
        "large_scale": (90, 3103, "large", [31, 30, 29], [30, 30, 30]),
        "factor_imbalanced": (96, 3111, "normal", [38, 31, 27], [52, 28, 16]),
        "wide_uniform": (105, 3105, "uniform", [36, 35, 34], [35, 35, 35]),
    }
    n_obs, seed, scale, cluster_counts, group_counts = specifications[case]
    rng = np.random.default_rng(seed)
    if scale == "small":
        x = rng.normal(0.0, 0.12, n_obs)
    elif scale == "large":
        x = rng.normal(0.0, 9.0, n_obs)
    elif scale == "uniform":
        x = rng.uniform(-25.0, 25.0, n_obs)
    else:
        x = rng.normal(0.0, 1.4, n_obs)
    clusters = np.repeat(["a", "b", "c"], cluster_counts)
    groups = np.repeat(["reference", "low", "high"], group_counts)
    return {
        "formula": "cluster ~ x + group + binary + x:group + I(x^2)",
        "data": {
            "cluster": clusters[rng.permutation(n_obs)].tolist(),
            "x": x,
            "group": groups[rng.permutation(n_obs)].tolist(),
            "binary": (rng.random(n_obs) < 0.47).tolist(),
        },
        "categorical": {"group": ["reference", "low", "high"]},
        "logical": ["binary"],
        "row_names": [f"{case}-{index + 1}" for index in range(n_obs)],
    }


def _canonical_crossed_fixture() -> pd.DataFrame:
    id_effects = np.asarray([-0.16, -0.08, -0.02, 0.04, 0.09, 0.15])
    bootstrap_effects = np.asarray([-0.11, -0.04, 0.01, 0.06, 0.1])
    residual_pattern = np.asarray(
        [
            [-0.030, 0.012, 0.018, -0.009, 0.021],
            [0.016, -0.022, 0.009, 0.027, -0.014],
            [0.011, 0.024, -0.019, 0.006, -0.028],
            [-0.018, 0.005, 0.026, -0.021, 0.013],
            [0.023, -0.011, -0.006, 0.019, -0.017],
            [-0.009, 0.028, -0.014, 0.008, -0.025],
        ]
    )
    rows: list[dict[str, Any]] = []
    for id_index, id_effect in enumerate(id_effects):
        for bootstrap_index, bootstrap_effect in enumerate(bootstrap_effects):
            rows.append(
                {
                    "id": f"id-{id_index + 1}",
                    "bootstrap": f"boot-{bootstrap_index + 1}",
                    "ame": (
                        0.27
                        + id_effect
                        + bootstrap_effect
                        + residual_pattern[id_index, bootstrap_index]
                    ),
                    "sterror": 0.045 + 0.008 * ((2 * id_index + bootstrap_index) % 6),
                }
            )
    frame = pd.DataFrame(rows, index=[f"crossed-{index + 1}" for index in range(len(rows))])
    frame.loc["crossed-12", "ame"] = np.nan
    return frame


def _alternate_crossed_fixture() -> pd.DataFrame:
    frame = _canonical_crossed_fixture()
    positions = np.arange(frame.shape[0], dtype=float)
    frame["ame"] = 0.08 + 0.74 * frame["ame"] + 0.017 * np.cos(1.3 * positions)
    frame["sterror"] = 0.038 + 0.0065 * ((3 * positions.astype(int) + 1) % 7)
    return frame


def _rarcat_shaped_numeric_crossed_fixture() -> pd.DataFrame:
    """Crossed integer groups whose R level order crosses the 9/10 boundary."""
    ids = np.arange(1, 11, dtype=int)
    bootstraps = np.arange(1, 11, dtype=int)
    id_effects = np.linspace(-0.18, 0.16, ids.size)
    bootstrap_effects = 0.09 * np.sin(np.arange(bootstraps.size, dtype=float) + 0.4)
    raw_interaction = np.fromfunction(
        lambda row, column: ((3 * row + 5 * column) % 11) - 5,
        (ids.size, bootstraps.size),
        dtype=float,
    )
    interaction = (
        raw_interaction
        - raw_interaction.mean(axis=0)
        - raw_interaction.mean(axis=1)[:, None]
        + raw_interaction.mean()
    )
    rows: list[dict[str, Any]] = []
    for id_position, id_value in enumerate(ids):
        for bootstrap_position, bootstrap_value in enumerate(bootstraps):
            rows.append(
                {
                    "id": id_value,
                    "bootstrap": bootstrap_value,
                    "ame": (
                        0.24
                        + id_effects[id_position]
                        + bootstrap_effects[bootstrap_position]
                        + 0.004 * interaction[id_position, bootstrap_position]
                    ),
                    "sterror": 0.04 + 0.004 * ((2 * id_position + 3 * bootstrap_position) % 7),
                }
            )
    return pd.DataFrame(
        rows,
        index=[f"numeric-crossed-{position + 1}" for position in range(len(rows))],
    )


def _singular_crossed_fixture() -> pd.DataFrame:
    frame = _canonical_crossed_fixture()
    id_names = sorted(frame["id"].unique())
    bootstrap_names = sorted(frame["bootstrap"].unique())
    id_effects = dict(zip(id_names, [-0.15, -0.08, -0.02, 0.04, 0.09, 0.14], strict=True))
    interaction = np.asarray(
        [[((3 * row + 2 * column) % 7) - 3 for column in range(5)] for row in range(6)],
        dtype=float,
    )
    interaction = (
        interaction
        - interaction.mean(axis=0)
        - interaction.mean(axis=1)[:, None]
        + interaction.mean()
    )
    frame["ame"] = [
        0.25
        + id_effects[row.id]
        + 0.008
        * interaction[
            id_names.index(row.id),
            bootstrap_names.index(row.bootstrap),
        ]
        for row in frame.itertuples()
    ]
    frame["sterror"] = 0.06
    frame.loc["crossed-12", "ame"] = np.nan
    return frame


def _f4_nonconvergent_crossed_fixture() -> pd.DataFrame:
    """Exact verified 3-by-3 fixture that previously leaked ``potrf``."""
    target_weights = np.linspace(0.7, 1.3, 9)
    return pd.DataFrame(
        {
            "id": np.repeat([1, 2, 10], 3),
            "bootstrap": np.tile([1, 2, 10], 3),
            "ame": np.asarray(
                [
                    0.05,
                    0.13,
                    0.21000000000000002,
                    0.16999999999999998,
                    0.25,
                    0.33,
                    0.29,
                    0.37,
                    0.45,
                ]
            ),
            "sterror": 1.0 / np.sqrt(target_weights),
        },
        index=[f"f4-{index + 1}" for index in range(9)],
    )


def _f4_reml_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    frame = _f4_nonconvergent_crossed_fixture()
    id_incidence = np.equal.outer(
        frame["id"].to_numpy(),
        [1, 2, 10],
    ).astype(float)
    bootstrap_incidence = np.equal.outer(
        frame["bootstrap"].to_numpy(),
        [1, 2, 10],
    ).astype(float)
    normalized_weights = 1.0 / np.square(frame["sterror"].to_numpy(dtype=float))
    normalized_weights /= np.mean(normalized_weights)
    return (
        frame["ame"].to_numpy(dtype=float),
        np.ones((frame.shape[0], 1), dtype=float),
        (
            id_incidence @ id_incidence.T,
            bootstrap_incidence @ bootstrap_incidence.T,
            np.diag(1.0 / normalized_weights),
        ),
    )


def _assert_naive_cluster_numeric_contract(actual, expected) -> None:
    assert actual["error"] == expected["error"]
    assert actual["warnings"] == expected["warnings"]
    assert actual["converged"] == expected["converged"]
    assert actual["nobs"] == expected["nobs"]
    assert actual["rank"] == expected["rank"]
    assert actual["df_residual"] == expected["df_residual"]
    assert actual["coefficient_names"] == expected["coefficient_names"]
    assert actual["design_column_names"] == expected["design_column_names"]
    assert actual["ame_factor_names"] == expected["ame_factor_names"]
    # On the canonical fixture, full-precision measurements over all three
    # one-vs-rest fits found MKL/OpenBLAS maxima of 8.53e-14 for coefficients,
    # 7.54e-13 for covariance, 1.11e-13 for AMEs, and 3.94e-6 absolute for
    # margins' SEs. The latter reflects margins 0.3.28's forward numerical
    # Jacobian magnifying sub-ulp prediction-order differences. These limits
    # retain at least 50% headroom over the measured supported profiles.
    np.testing.assert_allclose(
        actual["coefficients"],
        expected["coefficients"],
        rtol=0.0,
        atol=1.3e-13,
    )
    np.testing.assert_allclose(
        actual["covariance"],
        expected["covariance"],
        rtol=3e-12,
        atol=1.2e-12,
    )
    np.testing.assert_allclose(
        actual["ames"],
        expected["ames"],
        rtol=0.0,
        atol=1.7e-13,
    )
    np.testing.assert_allclose(
        actual["standard_errors"],
        expected["standard_errors"],
        rtol=7e-5,
        atol=6e-6,
    )


def _assert_crossed_numeric_contract(actual, expected) -> None:
    # Across the principal, alternate/permuted, and constant-weight fixtures,
    # maxima were 3.6e-15 for normalized weights, 2.2e-9 for the intercept,
    # 2.3e-6 relative for its SE, 4.7e-6 relative for variances, 1e-7 relative
    # for BLUPs (2e-7 on the challenging-weight fixture), 1.3e-10 for the
    # REML criterion, and 3.4e-6 relative for lme4's theta. The variance/theta
    # gap is lme4's optimizer stopping point versus the custom solver's tighter
    # analytic REML score.
    np.testing.assert_allclose(
        actual["normalized_weights"],
        expected["normalized_weights"],
        rtol=4e-16,
        atol=4e-15,
    )
    np.testing.assert_allclose(
        actual["fixed_intercept"],
        expected["fixed_intercept"],
        rtol=0.0,
        atol=2.5e-9,
    )
    np.testing.assert_allclose(
        actual["fixed_standard_error"],
        expected["fixed_standard_error"],
        rtol=2.5e-6,
        atol=1e-10,
    )
    for field in (
        "observation_variance",
        "bootstrap_variance",
        "residual_variance",
    ):
        np.testing.assert_allclose(
            actual[field],
            expected[field],
            rtol=5e-6,
            atol=1e-10,
        )
    for field in (
        "observation_random_effects",
        "bootstrap_random_effects",
    ):
        np.testing.assert_allclose(
            actual[field],
            expected[field],
            rtol=2e-7,
            atol=1e-10,
        )
    np.testing.assert_allclose(
        actual["log_likelihood"],
        expected["log_likelihood"],
        rtol=0.0,
        atol=7e-11,
    )
    np.testing.assert_allclose(
        actual["reml_criterion"],
        expected["reml_criterion"],
        rtol=0.0,
        atol=1.4e-10,
    )
    np.testing.assert_allclose(
        actual["theta"],
        expected["theta"],
        rtol=3.5e-6,
        atol=1e-8,
    )


def test_rarcat_naive_oracle_contract_pins_model_frame_glm_and_margins(
    r_parity_session,
) -> None:
    fixture = _canonical_naive_fixture()
    globals_before = {str(value) for value in r_parity_session.r("ls(envir=.GlobalEnv)")}
    payload = r_parity_session.rarcat_naive_contract(**fixture)
    globals_after = {str(value) for value in r_parity_session.r("ls(envir=.GlobalEnv)")}

    assert globals_after == globals_before
    assert payload["versions"] == {
        "WeightedCluster": "2.0",
        "TraMineR": "2.2.14",
        "margins": "0.3.28",
        "lme4": "2.1.0",
    }
    assert payload["error"] is None
    assert payload["formula"] == fixture["formula"]
    assert payload["lhs"] == "cluster"
    assert payload["rhs"] == "x + group + binary + x:group + I(x^2)"
    assert payload["model_frame"]["column_names"] == [
        "cluster",
        "x",
        "group",
        "binary",
        "I(x^2)",
    ]
    assert payload["model_frame"]["row_names"] == [
        name for index, name in enumerate(fixture["row_names"]) if index not in {8, 31}
    ]
    assert payload["model_frame"]["omitted_row_names"] == [
        fixture["row_names"][8],
        fixture["row_names"][31],
    ]
    assert payload["model_frame"]["term_labels"] == [
        "x",
        "group",
        "binary",
        "I(x^2)",
        "x:group",
    ]
    assert payload["model_frame"]["intercept"] == 1
    assert payload["model_frame"]["factor_levels"] == {
        "cluster": [],
        "x": [],
        "group": ["reference", "low", "high"],
        "binary": [],
        "I(x^2)": [],
    }
    assert payload["cluster_names"] == ["late", "early", "middle"]
    assert payload["factor_names"] == [
        "binary",
        "grouphigh",
        "grouplow",
        "x",
    ]
    assert payload["model_frame"]["design_column_names"] == [
        "(Intercept)",
        "x",
        "grouplow",
        "grouphigh",
        "binaryTRUE",
        "I(x^2)",
        "x:grouplow",
        "x:grouphigh",
    ]
    assert payload["model_frame"]["contrasts"] == {
        "group": "contr.treatment",
        "binary": "contr.treatment",
    }
    assert payload["rarcat"]["class"] == ["rarcat"]
    assert payload["rarcat"]["names"] == [
        "arguments",
        "formula",
        "AMElist",
        "clusterNames",
        "factorName",
        "clustering",
    ]
    assert payload["rarcat"]["cluster_names"] == payload["cluster_names"]
    assert payload["rarcat"]["factor_names"] == payload["factor_names"]

    assert list(payload["clusters"]) == payload["cluster_names"]
    for cluster_name, contract in payload["clusters"].items():
        assert contract["error"] is None, cluster_name
        assert contract["warnings"] == []
        assert contract["converged"] is True
        assert contract["nobs"] == 34
        assert contract["rank"] == len(contract["coefficient_names"])
        assert contract["design_column_names"] == payload["model_frame"]["design_column_names"]
        assert contract["ame_factor_names"] == payload["factor_names"]
        assert np.all(np.isfinite(contract["coefficients"]))
        assert np.all(np.isfinite(contract["ames"]))
        assert np.all(np.asarray(contract["standard_errors"]) > 0.0)
        assert np.any(np.abs(contract["ames"]) > 1e-6)


def test_rarcat_naive_oracle_contract_pins_unused_factor_level_error(
    r_parity_session,
) -> None:
    fixture = _canonical_naive_fixture()
    fixture["categorical"]["group"].append("unused")
    payload = r_parity_session.rarcat_naive_contract(**fixture)

    assert payload["error"] is None
    assert payload["model_frame"]["factor_levels"]["group"] == [
        "reference",
        "low",
        "high",
        "unused",
    ]
    assert payload["factor_names"] == []
    assert payload["rarcat"]["error"] == "factor group has new level unused"
    for contract in payload["clusters"].values():
        assert contract == {
            "error": "factor group has new level unused",
            "warnings": [],
        }


@pytest.mark.parametrize(
    ("cluster_values", "cluster_levels", "expected_names", "expected_class"),
    [
        (
            {"late": 10, "early": -2, "middle": 7},
            None,
            ["10", "-2", "7"],
            ["numeric"],
        ),
        (
            {"late": "late", "early": "early", "middle": "middle"},
            ["middle", "late", "early", "unused"],
            ["late", "early", "middle"],
            ["factor"],
        ),
    ],
)
def test_rarcat_naive_oracle_contract_preserves_numeric_and_factor_cluster_order(
    r_parity_session,
    cluster_values,
    cluster_levels,
    expected_names,
    expected_class,
) -> None:
    fixture = _canonical_naive_fixture()
    fixture["data"]["cluster"] = [
        None if value is None else cluster_values[value] for value in fixture["data"]["cluster"]
    ]
    if cluster_levels is not None:
        fixture["categorical"]["cluster"] = cluster_levels

    payload = r_parity_session.rarcat_naive_contract(**fixture)

    assert payload["error"] is None
    assert payload["model_frame"]["column_classes"]["cluster"] == expected_class
    assert payload["cluster_names"] == expected_names
    assert list(payload["clusters"]) == expected_names
    assert payload["rarcat"]["cluster_names"] == expected_names


def test_rarcat_naive_oracle_contract_pins_no_intercept_formula(r_parity_session) -> None:
    fixture = _canonical_naive_fixture()
    fixture["formula"] = "cluster ~ 0 + x + I(x^2)"
    payload = r_parity_session.rarcat_naive_contract(**fixture)

    assert payload["error"] is None
    assert payload["rhs"] == "0 + x + I(x^2)"
    assert payload["model_frame"]["intercept"] == 0
    assert payload["model_frame"]["term_labels"] == ["x", "I(x^2)"]
    assert payload["model_frame"]["design_column_names"] == ["x", "I(x^2)"]
    assert payload["factor_names"] == ["x"]
    assert payload["rarcat"]["error"] is None


@pytest.mark.parametrize(
    ("formula", "expected_design_names", "expected_factor_names"),
    [
        (
            "cluster ~ group",
            ["(Intercept)", "grouplow", "grouphigh"],
            ["grouphigh", "grouplow"],
        ),
        (
            "cluster ~ 0 + group",
            ["groupreference", "grouplow", "grouphigh"],
            ["grouphigh", "grouplow"],
        ),
        (
            "cluster ~ group + x + group:x + I(x^2)",
            [
                "(Intercept)",
                "grouplow",
                "grouphigh",
                "x",
                "I(x^2)",
                "grouplow:x",
                "grouphigh:x",
            ],
            ["grouphigh", "grouplow", "x"],
        ),
        (
            "cluster ~ 0 + group + x + group:x + I(x^2)",
            [
                "groupreference",
                "grouplow",
                "grouphigh",
                "x",
                "I(x^2)",
                "grouplow:x",
                "grouphigh:x",
            ],
            ["grouphigh", "grouplow", "x"],
        ),
        (
            "cluster ~ binary",
            ["(Intercept)", "binaryTRUE"],
            ["binary"],
        ),
        (
            "cluster ~ 0 + binary",
            ["binaryFALSE", "binaryTRUE"],
            ["binary"],
        ),
        (
            "cluster ~ binary + x + binary:x + I(x^2)",
            ["(Intercept)", "binaryTRUE", "x", "I(x^2)", "binaryTRUE:x"],
            ["binary", "x"],
        ),
        (
            "cluster ~ 0 + binary + x + binary:x + I(x^2)",
            ["binaryFALSE", "binaryTRUE", "x", "I(x^2)", "binaryTRUE:x"],
            ["binary", "x"],
        ),
    ],
)
def test_private_naive_solver_matches_r_categorical_and_logical_contrasts(
    r_parity_session,
    formula,
    expected_design_names,
    expected_factor_names,
) -> None:
    """F2a: intercept choice must not change R contrast/name semantics."""
    from tests._historical_rarcat_solver import (
        _fit_naive_rarcat_models,
        _r_design_column_name,
    )

    fixture = _canonical_naive_fixture()
    fixture["formula"] = formula
    oracle = r_parity_session.rarcat_naive_contract(**fixture)
    result = _fit_naive_rarcat_models(formula, _python_frame(fixture))

    assert oracle["runtime"] == {
        "LC_COLLATE": "C.UTF-8",
        "OutDec": ".",
        "digits": 7,
        "scipen": 0,
    }
    assert oracle["model_frame"]["design_column_names"] == expected_design_names
    assert oracle["factor_names"] == expected_factor_names
    assert result["model_frame"] == oracle["model_frame"]
    assert result["factor_names"] == expected_factor_names
    for cluster_name in oracle["cluster_names"]:
        _assert_naive_cluster_numeric_contract(
            result["clusters"][cluster_name],
            oracle["clusters"][cluster_name],
        )

    assert (
        _r_design_column_name(
            "literal[value]",
            categorical_factors=frozenset(),
        )
        == "literal[value]"
    )


@pytest.mark.parametrize(
    ("formula", "expected_design_names"),
    [
        (
            "cluster ~ group + x",
            ["(Intercept)", "groupA", "groupä", "groupb", "groupB", "x"],
        ),
        (
            "cluster ~ 0 + group + x",
            ["groupa", "groupA", "groupä", "groupb", "groupB", "x"],
        ),
    ],
)
def test_private_naive_solver_matches_r_raw_character_collation(
    r_parity_session,
    formula,
    expected_design_names,
) -> None:
    """F2b: raw characters use R's C.UTF-8/ICU factor order."""
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    fixture = _raw_character_naive_fixture(formula)
    oracle = r_parity_session.rarcat_naive_contract(**fixture)
    data = pd.DataFrame(fixture["data"], index=fixture["row_names"])
    result = _fit_naive_rarcat_models(formula, data)

    expected_levels = ["a", "A", "ä", "b", "B"]
    assert oracle["runtime"]["LC_COLLATE"] == "C.UTF-8"
    assert oracle["model_frame"]["column_classes"]["group"] == ["character"]
    assert oracle["model_frame"]["factor_levels"]["group"] == []
    assert oracle["model_frame"]["generated_factor_levels"]["group"] == expected_levels
    assert oracle["model_frame"]["contrast_matrices"]["group"]["row_names"] == expected_levels
    assert oracle["model_frame"]["contrast_matrices"]["group"]["column_names"] == [
        "A",
        "ä",
        "b",
        "B",
    ]
    np.testing.assert_array_equal(
        oracle["model_frame"]["contrast_matrices"]["group"]["values"],
        np.vstack([np.zeros(4), np.eye(4)]),
    )
    assert oracle["model_frame"]["design_column_names"] == expected_design_names
    assert oracle["factor_names"] == ["groupA", "groupä", "groupb", "groupB", "x"]
    assert result["model_frame"] == oracle["model_frame"]
    assert result["factor_names"] == oracle["factor_names"]
    for cluster_name in oracle["cluster_names"]:
        _assert_naive_cluster_numeric_contract(
            result["clusters"][cluster_name],
            oracle["clusters"][cluster_name],
        )


def test_rarcat_numeric_label_contract_pins_threshold_matrix(r_parity_session) -> None:
    """F3 oracle: pin general numeric rendering rather than the known 1e7 case."""
    values = [
        0.0,
        -0.0,
        1.0,
        -2.0,
        1.5,
        1.234567890123456,
        99_999.0,
        100_000.0,
        999_999.0,
        1_000_000.0,
        9_999_999.0,
        10_000_000.0,
        100_000_000.0,
        1e-3,
        1e-4,
        1e-5,
        9.999999e-5,
        0.0009999999,
        np.inf,
        -np.inf,
        np.nan,
    ]
    expected = [
        "0",
        "0",
        "1",
        "-2",
        "1.5",
        "1.23456789012346",
        "99999",
        "1e+05",
        "999999",
        "1e+06",
        "9999999",
        "1e+07",
        "1e+08",
        "0.001",
        "1e-04",
        "1e-05",
        "9.999999e-05",
        "0.0009999999",
        "Inf",
        "-Inf",
        "NaN",
    ]

    payload = r_parity_session.rarcat_numeric_label_contract(values)

    assert payload["rendered"] == expected
    assert payload["runtime"] == {
        "LC_COLLATE": "C.UTF-8",
        "OutDec": ".",
        "digits": 7,
        "scipen": 0,
    }
    assert payload["is_nan"] == [False] * 20 + [True]
    assert payload["is_na"] == [False] * 20 + [True]


def test_private_solver_uses_central_r_numeric_label_policy(r_parity_session) -> None:
    """F3 solver: names, ordering, collisions, and non-finites stay aligned."""
    from tests._historical_rarcat_solver import (
        _fit_naive_rarcat_models,
        _fit_weighted_crossed_pool,
        _r_value_string,
    )

    characterized_values = [
        1e7,
        -2.0,
        7.0,
        1e-4,
        np.nextafter(1e-4, np.inf),
        1.0,
        np.nextafter(1.0, np.inf),
        np.inf,
        -np.inf,
        np.nan,
    ]
    characterization = r_parity_session.rarcat_numeric_label_contract(characterized_values)
    assert [_r_value_string(value) for value in characterized_values] == characterization[
        "rendered"
    ]
    assert characterization["rendered"][:-1] == [
        "1e+07",
        "-2",
        "7",
        "1e-04",
        "1e-04",
        "1",
        "1",
        "Inf",
        "-Inf",
    ]

    fixture = _canonical_naive_fixture()
    mapping = {"late": 1e7, "early": -2.0, "middle": 7.0}
    fixture["data"]["cluster"] = [
        None if value is None else mapping[value] for value in fixture["data"]["cluster"]
    ]
    oracle = r_parity_session.rarcat_naive_contract(**fixture)
    result = _fit_naive_rarcat_models(fixture["formula"], _python_frame(fixture))
    assert oracle["cluster_names"] == ["1e+07", "-2", "7"]
    assert result["cluster_names"] == oracle["cluster_names"]
    assert list(result["clusters"]) == oracle["cluster_names"]

    nonfinite_fixture = _canonical_naive_fixture()
    nonfinite_mapping = {"late": np.inf, "early": -np.inf, "middle": 7.0}
    nonfinite_fixture["data"]["cluster"] = [
        None if value is None else nonfinite_mapping[value]
        for value in nonfinite_fixture["data"]["cluster"]
    ]
    nonfinite_oracle = r_parity_session.rarcat_naive_contract(**nonfinite_fixture)
    nonfinite_result = _fit_naive_rarcat_models(
        nonfinite_fixture["formula"],
        _python_frame(nonfinite_fixture),
    )
    assert nonfinite_oracle["cluster_names"] == ["Inf", "-Inf", "7"]
    assert nonfinite_result["cluster_names"] == nonfinite_oracle["cluster_names"]

    missing_response_fixture = _canonical_naive_fixture()
    missing_response_fixture["data"]["cluster"][0] = np.nan
    missing_response_oracle = r_parity_session.rarcat_naive_contract(**missing_response_fixture)
    missing_response_result = _fit_naive_rarcat_models(
        missing_response_fixture["formula"],
        _python_frame(missing_response_fixture),
    )
    assert missing_response_result["model_frame"] == missing_response_oracle["model_frame"]
    assert (
        missing_response_fixture["row_names"][0]
        in missing_response_result["model_frame"]["omitted_row_names"]
    )

    collision_fixture = _canonical_naive_fixture()
    collision_mapping = {
        "late": 1.0,
        "early": np.nextafter(1.0, np.inf),
        "middle": 7.0,
    }
    collision_fixture["data"]["cluster"] = [
        None if value is None else collision_mapping[value]
        for value in collision_fixture["data"]["cluster"]
    ]
    with pytest.raises(
        ValueError,
        match="cluster labels must remain unique after R string normalization",
    ):
        _fit_naive_rarcat_models(
            collision_fixture["formula"],
            _python_frame(collision_fixture),
        )

    crossed_numeric = _canonical_crossed_fixture()
    numeric_group_values = [-np.inf, -2.0, 1e-4, 7.0, 1e7, np.inf]
    numeric_group_mapping = dict(
        zip(
            sorted(crossed_numeric["id"].unique()),
            numeric_group_values,
            strict=True,
        )
    )
    crossed_numeric["id"] = crossed_numeric["id"].map(numeric_group_mapping).astype(float)
    crossed_oracle = r_parity_session.rarcat_crossed_pool_contract(crossed_numeric)
    crossed_result = _fit_weighted_crossed_pool(crossed_numeric)
    expected_group_names = ["-Inf", "-2", "1e-04", "7", "1e+07", "Inf"]
    assert crossed_oracle["observation_random_effect_names"] == expected_group_names
    assert crossed_result["observation_random_effect_names"] == expected_group_names
    _assert_crossed_numeric_contract(crossed_result, crossed_oracle)

    crossed_collision = _canonical_crossed_fixture()
    numeric_collision_mapping = dict(
        zip(
            sorted(crossed_collision["id"].unique()),
            [1.0, np.nextafter(1.0, np.inf), 2.0, 3.0, 4.0, 5.0],
            strict=True,
        )
    )
    crossed_collision["id"] = crossed_collision["id"].map(numeric_collision_mapping).astype(float)
    with pytest.raises(
        ValueError,
        match="id labels must remain unique after R string normalization",
    ):
        _fit_weighted_crossed_pool(crossed_collision)

    crossed = _canonical_crossed_fixture()
    crossed.loc[crossed["id"] == "id-1", "id"] = np.nan
    with pytest.raises(ValueError, match="id and bootstrap must not contain missing values"):
        _fit_weighted_crossed_pool(crossed)


def test_private_naive_solver_multi_fixture_convergence_and_tolerance_matrix(
    r_parity_session,
) -> None:
    """F5: derive stable tolerances only after aligning the R IRLS stop rule."""
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    maxima = {
        "coefficient_absolute": 0.0,
        "covariance_absolute": 0.0,
        "ame_absolute": 0.0,
        "ame_se_absolute": 0.0,
        "ame_se_relative": 0.0,
        "deviance_absolute": 0.0,
        "null_deviance_absolute": 0.0,
    }
    cases = (
        "balanced_standard",
        "small_scale",
        "large_scale",
        "factor_imbalanced",
        "wide_uniform",
    )
    for case in cases:
        fixture = _stable_naive_fixture(case)
        data = _python_frame(fixture)
        original = data.copy(deep=True)
        oracle = r_parity_session.rarcat_naive_contract(**fixture)
        result = _fit_naive_rarcat_models(fixture["formula"], data)
        pd.testing.assert_frame_equal(data, original)

        assert oracle["glm_control"] == {
            "epsilon": 1e-8,
            "maxit": 25,
            "criterion": "abs(deviance - previous) / (0.1 + abs(deviance))",
        }
        assert result["model_frame"] == oracle["model_frame"]
        assert result["cluster_names"] == oracle["cluster_names"]
        assert result["factor_names"] == oracle["factor_names"]
        for cluster_name in oracle["cluster_names"]:
            expected = oracle["clusters"][cluster_name]
            actual = result["clusters"][cluster_name]
            assert expected["error"] is actual["error"] is None
            assert expected["warnings"] == actual["warnings"] == []
            assert expected["converged"] is actual["converged"] is True
            assert expected["rank"] == actual["rank"] == len(expected["coefficient_names"])
            assert actual["iterations"] == expected["iterations"]
            assert actual["convergence_control"] == {
                "maxiter": 25,
                "atol": 1e-9,
                "rtol": 1e-8,
                "criterion": "deviance",
            }
            assert actual["fit_history"]["iterations"] == actual["iterations"]
            np.testing.assert_allclose(
                actual["fit_history"]["deviance"][-1],
                actual["deviance"],
                rtol=0.0,
                atol=3e-14,
            )
            assert actual["coefficient_names"] == expected["coefficient_names"]
            assert actual["ame_factor_names"] == expected["ame_factor_names"]
            np.testing.assert_allclose(
                expected["covariance_unscaled"],
                expected["covariance_scaled"],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                actual["covariance_unscaled"],
                actual["covariance_scaled"],
                rtol=0.0,
                atol=0.0,
            )

            pairs = (
                (
                    "coefficient_absolute",
                    actual["coefficients"],
                    expected["coefficients"],
                ),
                (
                    "covariance_absolute",
                    actual["covariance_scaled"],
                    expected["covariance_scaled"],
                ),
                ("ame_absolute", actual["ames"], expected["ames"]),
                (
                    "ame_se_absolute",
                    actual["standard_errors"],
                    expected["standard_errors"],
                ),
                (
                    "deviance_absolute",
                    np.asarray([actual["deviance"]]),
                    np.asarray([expected["deviance"]]),
                ),
                (
                    "null_deviance_absolute",
                    np.asarray([actual["null_deviance"]]),
                    np.asarray([expected["null_deviance"]]),
                ),
            )
            for field, actual_values, expected_values in pairs:
                maxima[field] = max(
                    maxima[field],
                    float(np.max(np.abs(np.asarray(actual_values) - np.asarray(expected_values)))),
                )
            maxima["ame_se_relative"] = max(
                maxima["ame_se_relative"],
                float(
                    np.max(
                        np.abs(
                            (actual["standard_errors"] - expected["standard_errors"])
                            / expected["standard_errors"]
                        )
                    )
                ),
            )

    # These field-specific limits are finalized from MKL and OpenBLAS GREEN
    # measurements, with at least 50% headroom over each observed maximum:
    # coefficient 1.8541e-14, covariance 3.9791e-13, AME 3.0754e-14,
    # AME-SE 3.3281e-7 absolute / 7.8434e-7 relative, and both deviance
    # fields 2.8422e-14. AME-SE remains the sole numerical-differentiation
    # field; every fitted GLM field is at binary64/linear-algebra noise.
    assert maxima["coefficient_absolute"] <= 3e-14
    assert maxima["covariance_absolute"] <= 6e-13
    assert maxima["ame_absolute"] <= 5e-14
    assert maxima["ame_se_absolute"] <= 5e-7
    assert maxima["ame_se_relative"] <= 1.2e-6
    assert maxima["deviance_absolute"] <= 5e-14
    assert maxima["null_deviance_absolute"] <= 5e-14


def test_rarcat_naive_oracle_contract_captures_separation_warnings(
    r_parity_session,
) -> None:
    payload = r_parity_session.rarcat_naive_contract(
        formula="cluster ~ x",
        data={
            "cluster": ["left"] * 8 + ["right"] * 8,
            "x": np.arange(-8.0, 8.0),
        },
        row_names=[f"separation-{index}" for index in range(16)],
    )

    assert payload["error"] is None
    assert payload["cluster_names"] == ["left", "right"]
    assert payload["rarcat"]["error"] is None
    for contract in payload["clusters"].values():
        assert contract["error"] is None
        assert contract["converged"] is False
        assert contract["warnings"] == [
            "glm.fit: algorithm did not converge",
            "glm.fit: fitted probabilities numerically 0 or 1 occurred",
        ]
        assert np.max(np.abs(contract["coefficients"])) > 40.0
        assert np.max(np.abs(contract["ames"])) < 1e-8
        assert np.all(np.asarray(contract["standard_errors"]) > 0.0)


def test_rarcat_source_contract_pins_installed_contradictions(r_parity_session) -> None:
    payload = r_parity_session.rarcat_source_contract()

    assert payload["versions"]["WeightedCluster"] == "2.0"
    assert payload["formals"]["kmedoid"] == "FALSE"
    assert payload["formals"]["cqi"] == '"HC"'
    assert payload["formals"]["robust"] == "TRUE"
    assert payload["help"]["kmedoid_mentions_character"] is True
    assert payload["help"]["kmedoid_mentions_pam_default"] is True
    assert payload["help"]["cqi_mentions_ch_default"] is True
    assert payload["vignette"]["hc_described_as_maximized"] is True
    assert payload["amemat"]["uses_which_min"] is True
    assert payload["rarcat"]["normalizes_inverse_variance_weights"] is True
    assert payload["rarcat"]["crosses_id_and_bootstrap"] is True
    assert payload["rarcat"]["allocates_bootstrap_ranef"] is True
    assert payload["rarcat"]["returns_bootstrap_ranef"] is False
    assert payload["rarcat"]["fisher_se_reapplies_tanh"] is True


def test_private_naive_solver_matches_live_r_glm_and_margins(r_parity_session) -> None:
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    fixture = _canonical_naive_fixture()
    data = _python_frame(fixture)
    original = data.copy(deep=True)
    oracle = r_parity_session.rarcat_naive_contract(**fixture)

    result = _fit_naive_rarcat_models(fixture["formula"], data)

    pd.testing.assert_frame_equal(data, original)
    assert result["formula"] == oracle["formula"]
    assert result["lhs"] == oracle["lhs"]
    assert result["rhs"] == oracle["rhs"]
    assert result["model_frame"] == oracle["model_frame"]
    assert result["cluster_names"] == oracle["cluster_names"]
    assert result["factor_names"] == oracle["factor_names"]
    assert list(result["clusters"]) == list(oracle["clusters"])
    for cluster_name in oracle["cluster_names"]:
        _assert_naive_cluster_numeric_contract(
            result["clusters"][cluster_name],
            oracle["clusters"][cluster_name],
        )


def test_private_naive_solver_matches_live_r_no_intercept(r_parity_session) -> None:
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    fixture = _canonical_naive_fixture()
    fixture["formula"] = "cluster ~ 0 + x + I(x^2)"
    oracle = r_parity_session.rarcat_naive_contract(**fixture)
    result = _fit_naive_rarcat_models(fixture["formula"], _python_frame(fixture))

    assert result["model_frame"] == oracle["model_frame"]
    assert result["factor_names"] == oracle["factor_names"]
    for cluster_name in oracle["cluster_names"]:
        _assert_naive_cluster_numeric_contract(
            result["clusters"][cluster_name],
            oracle["clusters"][cluster_name],
        )


def test_private_naive_solver_rejects_live_r_unused_factor_boundary(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    fixture = _canonical_naive_fixture()
    fixture["categorical"]["group"].append("unused")
    oracle = r_parity_session.rarcat_naive_contract(**fixture)

    assert {contract["error"] for contract in oracle["clusters"].values()} == {
        "factor group has new level unused"
    }
    with pytest.raises(ValueError, match="factor group has new level unused"):
        _fit_naive_rarcat_models(fixture["formula"], _python_frame(fixture))


def test_private_naive_solver_reports_live_r_separation_state() -> None:
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    result = _fit_naive_rarcat_models(
        "cluster ~ x",
        pd.DataFrame(
            {
                "cluster": ["left"] * 8 + ["right"] * 8,
                "x": np.arange(-8.0, 8.0),
            },
            index=[f"separation-{index}" for index in range(16)],
        ),
    )

    for contract in result["clusters"].values():
        assert contract["error"] is None
        assert contract["converged"] is False
        assert contract["warnings"] == [
            "glm.fit: algorithm did not converge",
            "glm.fit: fitted probabilities numerically 0 or 1 occurred",
        ]
        assert np.max(np.abs(contract["coefficients"])) > 40.0


def test_private_naive_solver_rejects_unsupported_formula_functions() -> None:
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    with pytest.raises(ValueError, match=r"unsupported formula function\(s\): log"):
        _fit_naive_rarcat_models(
            "cluster ~ log(x)",
            pd.DataFrame({"cluster": ["a", "b", "a"], "x": [1.0, 2.0, 3.0]}),
        )


def test_private_naive_solver_rejects_live_r_rank_deficiency(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_naive_rarcat_models

    data = {
        "cluster": ["a", "b", "c"] * 8,
        "x": np.arange(24.0),
        "z": 2.0 * np.arange(24.0),
    }
    oracle = r_parity_session.rarcat_naive_contract(
        formula="cluster ~ x + z",
        data=data,
        row_names=[f"rank-{index}" for index in range(24)],
    )

    for contract in oracle["clusters"].values():
        assert contract["rank"] == 2
        assert len(contract["coefficient_names"]) == 3
        assert np.isnan(contract["coefficients"][-1])
        assert len(contract["warnings"]) == 10
        assert set(contract["warnings"]) == {
            'prediction from rank-deficient fit; attr(*, "non-estim") has doubtful cases'
        }
    with pytest.raises(ValueError, match="formula design matrix is rank deficient"):
        _fit_naive_rarcat_models(
            "cluster ~ x + z",
            pd.DataFrame(data),
        )


def test_private_solver_rejects_r_string_label_collisions() -> None:
    from tests._historical_rarcat_solver import (
        _fit_naive_rarcat_models,
        _fit_weighted_crossed_pool,
    )

    with pytest.raises(
        ValueError,
        match="cluster labels must remain unique after R string normalization",
    ):
        _fit_naive_rarcat_models(
            "cluster ~ x",
            pd.DataFrame(
                {
                    "cluster": [1, "1", 2, 1, "1", 2, 1, "1", 2],
                    "x": [-2.1, -1.7, -1.0, -0.2, 0.3, 0.8, 1.2, 1.8, 2.4],
                }
            ),
        )

    colliding_naive_rows = pd.DataFrame(
        {
            "cluster": ["a", "b", "c"] * 3,
            "x": [-2.1, -1.7, -1.0, -0.2, 0.3, 0.8, 1.2, 1.8, 2.4],
        },
        index=[1, "1", "row-3", "row-4", "row-5", "row-6", "row-7", "row-8", "row-9"],
    )
    with pytest.raises(
        ValueError,
        match="data index labels must remain unique after R string normalization",
    ):
        _fit_naive_rarcat_models("cluster ~ x", colliding_naive_rows)

    colliding_groups = _canonical_crossed_fixture()
    colliding_groups["id"] = colliding_groups["id"].replace({"id-1": 1, "id-2": "1"})
    with pytest.raises(
        ValueError,
        match="id labels must remain unique after R string normalization",
    ):
        _fit_weighted_crossed_pool(colliding_groups)

    colliding_bootstraps = _canonical_crossed_fixture()
    colliding_bootstraps["bootstrap"] = colliding_bootstraps["bootstrap"].replace(
        {"boot-1": 1, "boot-2": "1"}
    )
    with pytest.raises(
        ValueError,
        match="bootstrap labels must remain unique after R string normalization",
    ):
        _fit_weighted_crossed_pool(colliding_bootstraps)

    colliding_rows = _canonical_crossed_fixture()
    row_names = list(colliding_rows.index)
    row_names[0] = 1
    row_names[1] = "1"
    colliding_rows.index = row_names
    with pytest.raises(
        ValueError,
        match="frame index labels must remain unique after R string normalization",
    ):
        _fit_weighted_crossed_pool(colliding_rows)


def test_rarcat_crossed_pool_oracle_contract_pins_weighted_lmer(
    r_parity_session,
) -> None:
    frame = _canonical_crossed_fixture()
    payload = r_parity_session.rarcat_crossed_pool_contract(frame)

    assert payload["versions"] == {
        "WeightedCluster": "2.0",
        "TraMineR": "2.2.14",
        "margins": "0.3.28",
        "lme4": "2.1.0",
    }
    assert payload["error"] is None
    assert payload["warnings"] == []
    assert payload["formula"] == "ame ~ (1 | id) + (1 | bootstrap)"
    assert payload["source_row_names"] == list(frame.index)
    assert payload["included_row_names"] == [name for name in frame.index if name != "crossed-12"]
    assert payload["omitted_row_names"] == ["crossed-12"]
    assert payload["nobs"] == 29
    assert payload["fixed_effect_names"] == ["(Intercept)"]
    assert np.isfinite(payload["fixed_intercept"])
    assert payload["fixed_standard_error"] > 0.0
    assert payload["observation_variance"] > 0.0
    assert payload["bootstrap_variance"] > 0.0
    assert payload["residual_variance"] > 0.0
    assert np.isclose(np.mean(payload["normalized_weights"]), 1.0)
    assert np.ptp(payload["normalized_weights"]) > 0.5
    assert payload["observation_random_effect_names"] == [f"id-{index}" for index in range(1, 7)]
    assert payload["bootstrap_random_effect_names"] == [f"boot-{index}" for index in range(1, 6)]
    assert payload["observation_random_effects"].shape == (6,)
    assert payload["bootstrap_random_effects"].shape == (5,)
    assert np.ptp(payload["observation_random_effects"]) > 0.1
    assert np.ptp(payload["bootstrap_random_effects"]) > 0.05
    assert payload["converged"] is True
    assert payload["optimizer_code"] == 0
    assert payload["convergence_messages"] == []
    assert payload["singular"] is False
    assert payload["reml"] is True
    assert np.isfinite(payload["log_likelihood"])
    assert np.isfinite(payload["reml_criterion"])


def test_private_crossed_pool_preserves_f4_fitted_nonconvergence(
    r_parity_session,
) -> None:
    """F4: a valid fitted-but-nonconverged model must never leak ``potrf``."""
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _f4_nonconvergent_crossed_fixture()
    permutation = [7, 0, 5, 2, 8, 3, 1, 6, 4]
    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)
    result = _fit_weighted_crossed_pool(frame)
    repeated = _fit_weighted_crossed_pool(frame)
    permuted = _fit_weighted_crossed_pool(frame.iloc[permutation])

    assert oracle["error"] is None
    assert oracle["optimizer_code"] == 0
    assert oracle["converged"] is False
    assert oracle["singular"] is False
    assert oracle["warnings"] == oracle["convergence_messages"]
    assert len(oracle["convergence_messages"]) == 2
    assert "max|grad|" in oracle["convergence_messages"][0]
    assert "nearly unidentifiable" in oracle["convergence_messages"][1]

    for payload in (result, repeated, permuted):
        assert payload["error"] is None
        assert payload["converged"] is False
        assert payload["singular"] is False
        assert len(payload["convergence_messages"]) == 2
        assert payload["optimizer"]["bounded_domain"] is True
        assert payload["optimizer"]["active_bound_components"] == ["residual"]
        assert np.all(np.isfinite(payload["theta"]))
        assert np.isfinite(payload["reml_criterion"])

    for field in (
        "fixed_intercept",
        "fixed_standard_error",
        "observation_variance",
        "bootstrap_variance",
        "residual_variance",
        "log_likelihood",
        "reml_criterion",
    ):
        assert result[field] == repeated[field]
    np.testing.assert_allclose(
        result["fixed_intercept"],
        permuted["fixed_intercept"],
        rtol=0.0,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        result["fixed_standard_error"],
        permuted["fixed_standard_error"],
        rtol=3e-6,
        atol=1e-12,
    )
    for field in (
        "observation_variance",
        "bootstrap_variance",
        "residual_variance",
    ):
        np.testing.assert_allclose(
            result[field],
            permuted[field],
            rtol=1.1e-5,
            atol=1e-16,
        )
    for field in ("log_likelihood", "reml_criterion"):
        np.testing.assert_allclose(
            result[field],
            permuted[field],
            rtol=5e-10,
            atol=2e-14,
        )
    for name_field, value_field in (
        ("observation_random_effect_names", "observation_random_effects"),
        ("bootstrap_random_effect_names", "bootstrap_random_effects"),
    ):
        expected = dict(zip(result[name_field], result[value_field], strict=True))
        actual = dict(zip(permuted[name_field], permuted[value_field], strict=True))
        assert list(expected) == list(actual) == ["1", "2", "10"]
        np.testing.assert_allclose(
            [expected[name] for name in expected],
            [actual[name] for name in expected],
            rtol=0.0,
            atol=2e-9,
        )


def test_crossed_reml_domain_and_analytic_score_are_controlled() -> None:
    """F4 lower-level contract for interiors, boundaries, and invalid states."""
    from tests._historical_rarcat_solver import (
        _crossed_reml_objective,
        _crossed_reml_state,
    )

    y, X, derivatives = _f4_reml_inputs()
    interior_logs = np.log(np.asarray([0.014, 0.006, 0.001]))
    objective, gradient = _crossed_reml_objective(
        interior_logs,
        y=y,
        X=X,
        covariance_derivatives=derivatives,
    )
    assert np.isfinite(objective)
    assert np.all(np.isfinite(gradient))

    step = 1e-6
    finite_difference = np.empty(3)
    for position in range(3):
        direction = np.zeros(3)
        direction[position] = step
        upper = _crossed_reml_objective(
            interior_logs + direction,
            y=y,
            X=X,
            covariance_derivatives=derivatives,
        )[0]
        lower = _crossed_reml_objective(
            interior_logs - direction,
            y=y,
            X=X,
            covariance_derivatives=derivatives,
        )[0]
        finite_difference[position] = (upper - lower) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-7, atol=2e-8)

    boundary = _crossed_reml_state(
        np.asarray([0.0, 0.006, 0.001]),
        y=y,
        X=X,
        covariance_derivatives=derivatives,
    )
    assert np.isfinite(boundary["objective"])
    assert boundary["gradient"][0] == 0.0

    with pytest.raises(ValueError, match="positive covariance"):
        _crossed_reml_state(
            np.zeros(3),
            y=y,
            X=X,
            covariance_derivatives=derivatives,
        )
    with pytest.raises(ValueError, match="finite REML log-variances"):
        _crossed_reml_objective(
            np.asarray([np.nan, 0.0, 0.0]),
            y=y,
            X=X,
            covariance_derivatives=derivatives,
        )
    for extreme in (-1_000.0, 1_000.0):
        with pytest.raises(ValueError, match="representable REML variance domain"):
            _crossed_reml_objective(
                np.asarray([extreme, 0.0, 0.0]),
                y=y,
                X=X,
                covariance_derivatives=derivatives,
            )
    with pytest.raises(ValueError, match="positive definite"):
        _crossed_reml_state(
            np.ones(3),
            y=y,
            X=X,
            covariance_derivatives=(-np.eye(y.size), np.zeros_like(derivatives[1]), derivatives[2]),
        )


def test_crossed_reml_rejects_malformed_algebra_inputs_before_factorization() -> None:
    from tests._historical_rarcat_solver import _crossed_reml_objective, _crossed_reml_state

    y, X, derivatives = _f4_reml_inputs()
    variances = np.asarray([0.014, 0.006, 0.001])

    with pytest.raises(ValueError, match="y must be a finite one-dimensional"):
        _crossed_reml_state(
            variances,
            y=y[:, None],
            X=X,
            covariance_derivatives=derivatives,
        )
    with pytest.raises(ValueError, match="y must be a finite one-dimensional"):
        _crossed_reml_state(
            variances,
            y=np.where(np.arange(y.size) == 0, np.nan, y),
            X=X,
            covariance_derivatives=derivatives,
        )
    with pytest.raises(ValueError, match="X must be a finite two-dimensional"):
        _crossed_reml_state(
            variances,
            y=y,
            X=np.column_stack([X, np.full(y.size, np.inf)]),
            covariance_derivatives=derivatives,
        )
    with pytest.raises(ValueError, match="X rows must match y"):
        _crossed_reml_state(
            variances,
            y=y,
            X=X[:-1],
            covariance_derivatives=derivatives,
        )
    with pytest.raises(ValueError, match="more observations than fixed effects"):
        _crossed_reml_state(
            variances,
            y=y[:2],
            X=np.eye(2),
            covariance_derivatives=tuple(derivative[:2, :2] for derivative in derivatives),
        )
    with pytest.raises(ValueError, match="exactly three covariance derivatives"):
        _crossed_reml_state(
            variances,
            y=y,
            X=X,
            covariance_derivatives=derivatives[:2],
        )
    with pytest.raises(ValueError, match="covariance derivatives.*shape"):
        _crossed_reml_state(
            variances,
            y=y,
            X=X,
            covariance_derivatives=(derivatives[0][:-1, :-1], *derivatives[1:]),
        )
    with pytest.raises(ValueError, match="covariance derivatives.*finite"):
        _crossed_reml_state(
            variances,
            y=y,
            X=X,
            covariance_derivatives=(
                derivatives[0],
                np.where(np.eye(y.size, dtype=bool), np.nan, derivatives[1]),
                derivatives[2],
            ),
        )
    with pytest.raises(TypeError, match="return_state must be a boolean"):
        _crossed_reml_objective(
            np.log(variances),
            y=y,
            X=X,
            covariance_derivatives=derivatives,
            return_state=1,
        )


def test_private_rarcat_fitters_reject_duplicate_dataframe_columns() -> None:
    from tests._historical_rarcat_solver import (
        _fit_naive_rarcat_models,
        _fit_weighted_crossed_pool,
    )

    naive = pd.DataFrame(
        [["a", "a", 0.0], ["b", "b", 1.0], ["a", "a", 2.0]],
        columns=["cluster", "cluster", "x"],
    )
    with pytest.raises(ValueError, match="data columns must be unique"):
        _fit_naive_rarcat_models("cluster ~ x", naive)

    crossed = _canonical_crossed_fixture()
    crossed.insert(1, "id", crossed["id"], allow_duplicates=True)
    with pytest.raises(ValueError, match="frame columns must be unique"):
        _fit_weighted_crossed_pool(crossed)


@pytest.mark.parametrize(
    ("column", "expected_class", "expected_type", "expected_levels"),
    [
        ("integer", ["integer"], "integer", ["1", "2", "10"]),
        ("numeric", ["numeric"], "double", ["1.5", "2.5", "10.5"]),
        ("character", ["character"], "character", ["1", "10", "2"]),
        ("factor", ["factor"], "integer", ["ten", "one", "two"]),
        ("logical", ["logical"], "logical", ["FALSE", "TRUE"]),
    ],
)
def test_rarcat_crossed_oracle_preserves_group_semantic_types(
    r_parity_session,
    column,
    expected_class,
    expected_type,
    expected_levels,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _canonical_crossed_fixture()
    replacements = {
        "integer": [1, 2, 10, 1, 2, 10],
        "numeric": [1.5, 2.5, 10.5, 1.5, 2.5, 10.5],
        "character": ["1", "2", "10", "1", "2", "10"],
        "factor": pd.Categorical(
            ["one", "two", "ten", "one", "two", "ten"],
            categories=["ten", "one", "two", "unused"],
        ),
        "logical": [False, True, False, True, False, True],
    }
    mapping = dict(zip(sorted(frame["id"].unique()), replacements[column], strict=True))
    frame["id"] = frame["id"].map(mapping)
    if column == "integer":
        frame["id"] = frame["id"].astype(int)
    elif column == "numeric":
        frame["id"] = frame["id"].astype(float)
    elif column == "factor":
        frame["id"] = pd.Categorical(
            frame["id"],
            categories=replacements[column].categories,
        )
    elif column == "logical":
        frame["id"] = frame["id"].astype(bool)

    payload = r_parity_session.rarcat_crossed_pool_contract(frame)
    result = _fit_weighted_crossed_pool(frame)

    metadata = payload["group_metadata"]["id"]
    assert metadata["class"] == expected_class
    assert metadata["type"] == expected_type
    assert metadata["factor_levels"] == expected_levels
    assert metadata["random_effect_names"] == expected_levels
    assert metadata["missing_count"] == 0
    assert result["observation_random_effect_names"] == expected_levels


def test_rarcat_crossed_oracle_preserves_missing_integer_representation(
    r_parity_session,
) -> None:
    frame = _canonical_crossed_fixture()
    mapped = frame["id"].map(
        dict(zip(sorted(frame["id"].unique()), [1, 2, 10, 1, 2, 10], strict=True))
    )
    frame["id"] = pd.Series(mapped, index=frame.index, dtype="Int64")
    frame.loc["crossed-1", "id"] = pd.NA

    payload = r_parity_session.rarcat_crossed_pool_contract(frame)

    metadata = payload["group_metadata"]["id"]
    assert metadata["class"] == ["integer"]
    assert metadata["type"] == "integer"
    assert metadata["missing_count"] == 1
    assert metadata["values"][0] is None
    assert payload["nobs"] == 28


def test_private_crossed_pool_preserves_numeric_group_order_and_blup_alignment(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _rarcat_shaped_numeric_crossed_fixture()
    expected_names = [str(value) for value in range(1, 11)]
    permutation = np.random.default_rng(20260728).permutation(frame.shape[0])
    permuted = frame.iloc[permutation]

    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)
    permuted_oracle = r_parity_session.rarcat_crossed_pool_contract(permuted)
    result = _fit_weighted_crossed_pool(frame)
    permuted_result = _fit_weighted_crossed_pool(permuted)

    for payload in (oracle, permuted_oracle, result, permuted_result):
        assert payload["observation_random_effect_names"] == expected_names
        assert payload["bootstrap_random_effect_names"] == expected_names
    for actual, expected in ((result, oracle), (permuted_result, permuted_oracle)):
        np.testing.assert_allclose(
            actual["normalized_weights"],
            expected["normalized_weights"],
            rtol=4e-16,
            atol=4e-15,
        )
        np.testing.assert_allclose(
            actual["fixed_intercept"],
            expected["fixed_intercept"],
            rtol=0.0,
            atol=2.5e-9,
        )
        for value_field in (
            "observation_random_effects",
            "bootstrap_random_effects",
        ):
            np.testing.assert_allclose(
                actual[value_field],
                expected[value_field],
                rtol=2e-7,
                atol=1e-10,
            )
    for base, shuffled in ((oracle, permuted_oracle), (result, permuted_result)):
        for name_field, value_field in (
            ("observation_random_effect_names", "observation_random_effects"),
            ("bootstrap_random_effect_names", "bootstrap_random_effects"),
        ):
            base_mapping = dict(zip(base[name_field], base[value_field], strict=True))
            shuffled_mapping = dict(zip(shuffled[name_field], shuffled[value_field], strict=True))
            assert list(base_mapping) == expected_names
            assert list(shuffled_mapping) == expected_names
            np.testing.assert_allclose(
                [base_mapping[name] for name in expected_names],
                [shuffled_mapping[name] for name in expected_names],
                rtol=2e-7,
                atol=1e-10,
            )


@pytest.mark.parametrize(
    ("kind", "expected_names"),
    [
        ("character", ["1", "10", "2", "3", "4", "5"]),
        ("categorical", ["5", "2", "4", "1", "10", "3"]),
    ],
)
def test_private_crossed_pool_uses_character_and_declared_factor_order(
    r_parity_session,
    kind,
    expected_names,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _canonical_crossed_fixture()
    raw_names = ["1", "2", "10", "3", "4", "5"]
    mapping = dict(zip(sorted(frame["id"].unique()), raw_names, strict=True))
    frame["id"] = frame["id"].map(mapping)
    if kind == "categorical":
        frame["id"] = pd.Categorical(
            frame["id"],
            categories=["5", "2", "4", "1", "10", "3", "unused"],
        )

    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)
    result = _fit_weighted_crossed_pool(frame)

    assert oracle["observation_random_effect_names"] == expected_names
    assert result["observation_random_effect_names"] == expected_names
    _assert_crossed_numeric_contract(result, oracle)


def test_private_crossed_pool_solver_matches_live_r_weighted_lmer(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _canonical_crossed_fixture()
    original = frame.copy(deep=True)
    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)

    result = _fit_weighted_crossed_pool(frame)

    pd.testing.assert_frame_equal(frame, original)
    for field in (
        "error",
        "warnings",
        "formula",
        "source_row_names",
        "included_row_names",
        "omitted_row_names",
        "nobs",
        "fixed_effect_names",
        "observation_random_effect_names",
        "bootstrap_random_effect_names",
        "singular",
        "reml",
    ):
        assert result[field] == oracle[field], field
    # The custom analytic-score solver and lme4 can straddle their convergence
    # thresholds while retaining the same fitted scientific quantities. The
    # canonical fixture reports code 0 on MKL and code 2 with a 2.07e-7 score
    # on OpenBLAS, so optimizer diagnostics are provenance, not parity fields.
    for field in (
        "fixed_intercept",
        "fixed_standard_error",
        "observation_variance",
        "bootstrap_variance",
        "residual_variance",
        "log_likelihood",
        "reml_criterion",
    ):
        assert np.isfinite(result[field]), field
    _assert_crossed_numeric_contract(result, oracle)


def test_private_crossed_pool_solver_matches_second_live_r_fixture_and_permutation(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _alternate_crossed_fixture().iloc[
        [
            17,
            2,
            26,
            7,
            21,
            0,
            14,
            29,
            5,
            11,
            24,
            9,
            18,
            1,
            27,
            6,
            20,
            4,
            15,
            28,
            8,
            22,
            3,
            16,
            25,
            10,
            19,
            12,
            23,
            13,
        ]
    ]
    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)
    result = _fit_weighted_crossed_pool(frame)

    assert result["included_row_names"] == oracle["included_row_names"]
    assert result["omitted_row_names"] == oracle["omitted_row_names"]
    assert result["observation_random_effect_names"] == oracle["observation_random_effect_names"]
    assert result["bootstrap_random_effect_names"] == oracle["bootstrap_random_effect_names"]
    assert result["converged"] is oracle["converged"] is True
    assert result["singular"] is oracle["singular"] is False
    _assert_crossed_numeric_contract(result, oracle)


def test_private_crossed_pool_inverse_variance_weights_change_fit(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    weighted_frame = _canonical_crossed_fixture()
    constant_frame = weighted_frame.copy(deep=True)
    constant_frame["sterror"] = 0.07
    weighted_oracle = r_parity_session.rarcat_crossed_pool_contract(weighted_frame)
    constant_oracle = r_parity_session.rarcat_crossed_pool_contract(constant_frame)
    weighted = _fit_weighted_crossed_pool(weighted_frame)
    constant = _fit_weighted_crossed_pool(constant_frame)

    _assert_crossed_numeric_contract(weighted, weighted_oracle)
    _assert_crossed_numeric_contract(constant, constant_oracle)
    assert abs(weighted_oracle["fixed_intercept"] - constant_oracle["fixed_intercept"]) > 0.002
    assert abs(weighted["fixed_intercept"] - constant["fixed_intercept"]) > 0.002


def test_private_crossed_pool_solver_matches_live_r_singular_boundary(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _singular_crossed_fixture()
    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)
    result = _fit_weighted_crossed_pool(frame)

    assert oracle["optimizer_code"] == result["optimizer_code"] == 0
    assert oracle["converged"] is result["converged"] is False
    assert oracle["singular"] is result["singular"] is True
    assert (
        oracle["convergence_messages"]
        == result["convergence_messages"]
        == ["boundary (singular) fit: see help('isSingular')"]
    )
    assert oracle["bootstrap_variance"] == result["bootstrap_variance"] == 0.0
    assert oracle["theta"][1] == result["theta"][1] == 0.0
    _assert_crossed_numeric_contract(result, oracle)


def test_private_crossed_pool_solver_reports_challenging_weight_convergence(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _canonical_crossed_fixture()
    frame["sterror"] = np.geomspace(1e-4, 1.0, frame.shape[0])
    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)
    result = _fit_weighted_crossed_pool(frame)

    assert np.max(result["normalized_weights"]) / np.min(result["normalized_weights"]) > 1e7
    assert oracle["converged"] is result["converged"] is True
    assert oracle["convergence_messages"] == result["convergence_messages"] == []
    assert result["optimizer"]["gradient_max_abs"] < 1e-7
    _assert_crossed_numeric_contract(result, oracle)


def test_crossed_pool_standard_error_domain_pins_r_and_python_policy(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    negative = _canonical_crossed_fixture()
    negative.loc["crossed-1", "sterror"] = -0.06
    negative_oracle = r_parity_session.rarcat_crossed_pool_contract(negative)
    negative_result = _fit_weighted_crossed_pool(negative)
    assert negative_oracle["error"] is None
    _assert_crossed_numeric_contract(negative_result, negative_oracle)

    zero = _canonical_crossed_fixture()
    zero.loc["crossed-1", "sterror"] = 0.0
    zero_oracle = r_parity_session.rarcat_crossed_pool_contract(zero)
    assert zero_oracle["error"] == "Downdated VtV is not positive definite"
    assert np.isnan(zero_oracle["normalized_weights"][0])
    assert np.all(zero_oracle["normalized_weights"][1:] == 0.0)
    with pytest.raises(ValueError, match="included standard errors must be nonzero"):
        _fit_weighted_crossed_pool(zero)

    missing = _canonical_crossed_fixture()
    missing.loc["crossed-1", "sterror"] = np.nan
    missing_oracle = r_parity_session.rarcat_crossed_pool_contract(missing)
    assert missing_oracle["error"] == "0 (non-NA) cases"
    assert np.all(np.isnan(missing_oracle["normalized_weights"]))
    with pytest.raises(
        ValueError,
        match="included standard errors must contain only finite values",
    ):
        _fit_weighted_crossed_pool(missing)

    infinite = _canonical_crossed_fixture()
    infinite.loc["crossed-1", "sterror"] = np.inf
    infinite_oracle = r_parity_session.rarcat_crossed_pool_contract(infinite)
    assert infinite_oracle["error"] is None
    assert infinite_oracle["normalized_weights"][0] == 0.0
    with pytest.raises(
        ValueError,
        match="included standard errors must contain only finite values",
    ):
        _fit_weighted_crossed_pool(infinite)


@pytest.mark.parametrize("standard_error", [1e-200, 1e200])
def test_crossed_pool_extreme_standard_error_domain_is_explicit(
    r_parity_session,
    standard_error,
) -> None:
    """Closure: reject R's nonrepresentable weight domain without raw warnings."""
    from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

    frame = _canonical_crossed_fixture()
    frame["sterror"] = standard_error
    oracle = r_parity_session.rarcat_crossed_pool_contract(frame)

    assert oracle["error"] == "0 (non-NA) cases"
    assert np.all(np.isnan(oracle["normalized_weights"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match="safe inverse-variance magnitude domain",
        ):
            _fit_weighted_crossed_pool(frame)
    assert caught == []


def test_phase3a_private_solver_source_and_dependency_boundaries(monkeypatch) -> None:
    """Closure: the retired solver stays test-owned with explicit ICU failure."""
    from weightedcluster import _bayesian_rarcat_design as design

    root = Path(__file__).resolve().parents[1]
    source = (root / "tests/_historical_rarcat_solver.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    import_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_roots.add(node.module.split(".", maxsplit=1)[0])
    assert import_roots.isdisjoint({"rpy2", "subprocess"})
    assert "Rscript" not in source
    assert source.count("def _r_value_string(") == 0

    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = metadata["project"]["dependencies"]
    bayesian_dependencies = metadata["project"]["optional-dependencies"]["bayesian"]
    development_dependencies = metadata["project"]["optional-dependencies"]["dev"]
    assert not any(value.startswith("patsy>=") for value in runtime_dependencies)
    assert not any(value.startswith("statsmodels>=") for value in runtime_dependencies)
    assert any(value.startswith("patsy>=") for value in bayesian_dependencies)
    assert any(value.startswith("patsy>=") for value in development_dependencies)
    assert any(value.startswith("statsmodels>=") for value in development_dependencies)
    assert not any(value.startswith("rpy2") for value in runtime_dependencies)
    assert any(value.startswith("rpy2") for value in development_dependencies)

    design._icu_collation_api.cache_clear()
    monkeypatch.setattr(design.ctypes.util, "find_library", lambda _name: "missing-icu.so")
    monkeypatch.setattr(
        design.ctypes,
        "CDLL",
        lambda _name: (_ for _ in ()).throw(OSError("loader failure")),
    )
    try:
        with pytest.raises(RuntimeError, match="could not load ICU"):
            design._icu_collation_api()
    finally:
        design._icu_collation_api.cache_clear()


def test_phase3a_new_oracles_are_repeatable_and_clean_global_state(
    r_parity_session,
) -> None:
    """Closure: new live-R helpers own copies and remove every temporary global."""
    globals_before = {str(value) for value in r_parity_session.r("ls(envir=.GlobalEnv)")}
    values = [0.0, 1e-4, 1e7, np.inf, -np.inf, np.nan]

    first = r_parity_session.rarcat_numeric_label_contract(values)
    second = r_parity_session.rarcat_numeric_label_contract(values)
    crossed = r_parity_session.rarcat_crossed_pool_contract(_f4_nonconvergent_crossed_fixture())
    globals_after = {str(value) for value in r_parity_session.r("ls(envir=.GlobalEnv)")}

    assert first == second
    assert crossed["error"] is None
    assert globals_after == globals_before


def test_rarcat_fisher_oracle_contract_pins_installed_transform_order(
    r_parity_session,
) -> None:
    payload = r_parity_session.rarcat_fisher_contract(
        values=np.asarray([-0.999999, -0.8, 0.0, 0.8, 0.999999]),
        coefficient=0.73,
        standard_error=0.19,
        observation_variance=0.17,
        bootstrap_variance=0.08,
    )

    assert payload["error"] is None
    assert payload["warnings"] == []
    assert payload["forward"].shape == (5,)
    assert np.all(np.isfinite(payload["forward"]))
    assert np.all(np.diff(payload["forward"]) > 0.0)
    assert -1.0 < payload["coefficient"] < 1.0
    assert payload["standard_error"] > 0.0
    assert payload["observation_standard_deviation"] > 0.0
    assert payload["bootstrap_standard_deviation"] > 0.0
    assert payload["se_reapplies_tanh_to_backtransformed_coefficient"] is True


def test_private_fisher_primitives_match_installed_rarcat_order(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import (
        _fisher_backtransform_pool,
        _fisher_forward,
    )

    values = np.asarray([-0.999999, -0.8, 0.0, 0.8, 0.999999])
    oracle = r_parity_session.rarcat_fisher_contract(
        values=values,
        coefficient=0.73,
        standard_error=0.19,
        observation_variance=0.17,
        bootstrap_variance=0.08,
    )
    result = _fisher_backtransform_pool(
        coefficient=0.73,
        standard_error=0.19,
        observation_variance=0.17,
        bootstrap_variance=0.08,
    )

    np.testing.assert_array_equal(_fisher_forward(values), oracle["forward"])
    for field in (
        "coefficient",
        "standard_error",
        "observation_standard_deviation",
        "bootstrap_standard_deviation",
    ):
        # R and NumPy's scalar tanh differ by at most one binary64 ulp here.
        np.testing.assert_allclose(
            result[field],
            oracle[field],
            rtol=0.0,
            atol=1.2e-16,
        )


def test_fisher_boundary_contract_pins_r_and_rejects_nonfinite_python_domain(
    r_parity_session,
) -> None:
    from tests._historical_rarcat_solver import (
        _fisher_backtransform_pool,
        _fisher_forward,
    )

    oracle = r_parity_session.rarcat_fisher_contract(
        values=np.asarray([-1.0, 1.0, -1.1, 1.1]),
        coefficient=0.73,
        standard_error=0.19,
        observation_variance=0.17,
        bootstrap_variance=0.08,
    )
    assert oracle["error"] is None
    assert oracle["warnings"] == ["NaNs produced"]
    assert np.array_equal(
        np.isneginf(oracle["forward"]),
        [True, False, False, False],
    )
    assert np.array_equal(
        np.isposinf(oracle["forward"]),
        [False, True, False, False],
    )
    assert np.array_equal(
        np.isnan(oracle["forward"]),
        [False, False, True, True],
    )
    for invalid in (-1.0, 1.0, -1.1, 1.1, np.nan, np.inf):
        with pytest.raises(
            ValueError,
            match=r"Fisher values must be finite and strictly inside \(-1, 1\)",
        ):
            _fisher_forward(np.asarray([invalid]))
    with pytest.raises(ValueError, match="Fisher standard error must be finite"):
        _fisher_backtransform_pool(
            coefficient=0.1,
            standard_error=np.inf,
            observation_variance=0.1,
            bootstrap_variance=0.1,
        )
    with pytest.raises(
        ValueError,
        match="Fisher variance components must be finite and non-negative",
    ):
        _fisher_backtransform_pool(
            coefficient=0.1,
            standard_error=0.1,
            observation_variance=-0.1,
            bootstrap_variance=0.1,
        )

from __future__ import annotations

import builtins
import importlib
import inspect
import logging
import sys
import warnings as warnings_module
from dataclasses import replace
from importlib import metadata as importlib_metadata
from types import SimpleNamespace

import numpy as np
import pandas as pd
import patsy
import pytest
import xarray as xr

from weightedcluster import _bayesian_rarcat_partition as b2

_COUNTERFACTUAL_HALF_STEP = 1e-4
_COUNTERFACTUAL_CONVERGENCE_ATOL = 2e-7


@pytest.fixture(scope="module")
def partition_result() -> b2.PartitionPosteriorResult:
    positions = np.asarray([0.0, 0.01, 0.02, 5.0, 5.01, 5.02])
    ids = [f"case-{index}" for index in range(6)]
    return b2.build_partition_posterior(
        dissimilarity=np.abs(positions[:, None] - positions[None, :]),
        reference_labels=["A", "A", "A", "B", "B", "B"],
        observation_ids=ids,
        reference_observation_ids=ids,
        B=8,
        npass=3,
        root_seed=13,
    )


@pytest.fixture
def regression_frame(partition_result: b2.PartitionPosteriorResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": pd.Categorical(
                partition_result.reference_labels,
                categories=partition_result.reference_cluster_order,
                ordered=True,
            ),
            "x": np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
            "group": pd.Categorical(
                [20, 10, 20, 10, 20, 10],
                categories=[20, 10],
                ordered=True,
            ),
            "unused": [np.nan] * 6,
        },
        index=partition_result.observation_ids,
    )


@pytest.fixture(scope="module")
def partition_result_three() -> b2.PartitionPosteriorResult:
    positions = np.asarray([0.0, 0.01, 0.02, 5.0, 5.01, 5.02, 10.0, 10.01, 10.02])
    ids = [f"tri-{index}" for index in range(9)]
    return b2.build_partition_posterior(
        dissimilarity=np.abs(positions[:, None] - positions[None, :]),
        reference_labels=["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        observation_ids=ids,
        reference_observation_ids=ids,
        B=8,
        npass=3,
        root_seed=29,
    )


@pytest.fixture
def regression_frame_three(partition_result_three) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": pd.Categorical(
                partition_result_three.reference_labels,
                categories=partition_result_three.reference_cluster_order,
                ordered=True,
            ),
            "x": np.asarray([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
            "group": pd.Categorical(
                [20, 10, 20, 10, 20, 10, 20, 10, 20],
                categories=[20, 10],
                ordered=True,
            ),
            "flag": [False, False, True, True, False, False, True, True, False],
        },
        index=partition_result_three.observation_ids,
    )


def _module():
    return importlib.import_module("weightedcluster._bayesian_rarcat_conditional")


def test_private_b3_design_entrypoint_exists():
    module = _module()

    assert callable(module.prepare_conditional_design)


def test_design_preserves_identity_factor_order_and_population_scaling(
    partition_result, regression_frame
):
    b3 = _module()

    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result,
        formula="cluster ~ 1 + x + group",
        data=regression_frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )

    assert prepared.formula == "cluster ~ 1 + x + group"
    assert prepared.response_name == "cluster"
    assert prepared.row_ids == tuple(regression_frame.index)
    assert prepared.cluster_order == ("A", "B")
    assert prepared.reference_category == "A"
    assert prepared.base_predictors == ("x", "group")
    assert prepared.continuous_predictors == ("x",)
    assert prepared.categorical_predictors == ("group",)
    assert prepared.factor_levels == (("group", (20, 10)),)
    assert prepared.factor_references == (("group", 20),)
    assert prepared.design_matrix.shape == (6, 3)
    assert prepared.design_rank == 3
    assert prepared.coefficient_coordinates[0] == "(Intercept)"
    assert prepared.engine_to_coefficient[0] == ("Intercept", "(Intercept)")
    transform = prepared.scaling_transforms[0]
    assert transform.predictor == "x"
    np.testing.assert_array_equal(transform.raw_values, regression_frame["x"].to_numpy())
    assert transform.mean == pytest.approx(0.0, abs=1e-12, rel=0.0)
    assert transform.population_sd == pytest.approx(
        np.std(regression_frame["x"], ddof=0), abs=1e-12, rel=0.0
    )
    np.testing.assert_allclose(
        prepared.design_matrix[:, prepared.coefficient_coordinates.index("x")],
        regression_frame["x"].to_numpy() / transform.population_sd,
    )
    assert not prepared.design_matrix.flags.writeable
    assert not transform.raw_values.flags.writeable


def test_design_ignores_unrelated_missing_columns_and_never_mutates_input(
    partition_result, regression_frame
):
    b3 = _module()
    original = regression_frame.copy(deep=True)

    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result,
        formula="cluster ~ x",
        data=regression_frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )

    pd.testing.assert_frame_equal(regression_frame, original)
    assert prepared.design_matrix.shape == (6, 2)


@pytest.mark.parametrize(
    "index",
    [
        ["other-0", "case-1", "case-2", "case-3", "case-4", "case-5"],
        ["case-1", "case-0", "case-2", "case-3", "case-4", "case-5"],
        ["case-0", "case-0", "case-2", "case-3", "case-4", "case-5"],
    ],
)
def test_design_rejects_changed_reordered_or_duplicate_row_identity(
    partition_result, regression_frame, index
):
    b3 = _module()
    changed = regression_frame.copy()
    changed.index = index

    with pytest.raises(ValueError, match="index.*elementwise identical|unique"):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula="cluster ~ x",
            data=changed,
            continuous_predictors=("x",),
            counterfactual_steps={},
        )


def test_design_requires_ordered_categorical_response_with_exact_values_and_order(
    partition_result, regression_frame
):
    b3 = _module()
    cases = []
    plain = regression_frame.copy()
    plain["cluster"] = plain["cluster"].astype(object)
    cases.append(plain)
    unordered = regression_frame.copy()
    unordered["cluster"] = unordered["cluster"].cat.as_unordered()
    cases.append(unordered)
    reversed_order = regression_frame.copy()
    reversed_order["cluster"] = reversed_order["cluster"].cat.reorder_categories(["B", "A"])
    cases.append(reversed_order)
    changed = regression_frame.copy()
    changed["cluster"] = pd.Categorical(
        ["B", "A", "A", "B", "B", "B"], categories=["A", "B"], ordered=True
    )
    cases.append(changed)

    for invalid in cases:
        with pytest.raises(ValueError, match="response"):
            b3.prepare_conditional_design(
                partition_posterior=partition_result,
                formula="cluster ~ x",
                data=invalid,
                continuous_predictors=("x",),
                counterfactual_steps={},
            )


@pytest.mark.parametrize(
    ("continuous", "message"),
    [
        ((), "continuous_predictors"),
        (("x", "x"), "exactly once"),
        (("x", "unused"), "unused"),
        (("group", "x"), "categorical"),
    ],
)
def test_design_requires_exact_raw_numeric_predictor_classification(
    partition_result, regression_frame, continuous, message
):
    b3 = _module()

    with pytest.raises((TypeError, ValueError), match=message):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula="cluster ~ x + group",
            data=regression_frame,
            continuous_predictors=continuous,
            counterfactual_steps={},
        )


@pytest.mark.parametrize("column", ["x", "group"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_design_rejects_missing_or_nonfinite_referenced_model_frame_values(
    partition_result, regression_frame, column, value
):
    b3 = _module()
    changed = regression_frame.copy()
    if column == "group" and np.isinf(value):
        changed[column] = changed[column].astype(object)
    changed.iloc[0, changed.columns.get_loc(column)] = value

    with pytest.raises(ValueError, match="complete|finite"):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula="cluster ~ x + group",
            data=changed,
            continuous_predictors=("x",),
            counterfactual_steps={},
        )


@pytest.mark.parametrize(
    "formula",
    [
        "cluster",
        "cluster + x ~ group",
        "cluster ~",
        "cluster ~ .",
        "cluster ~ np.log(x)",
        "cluster ~ x | group",
        "cluster ~ I(x^0)",
        "cluster ~ I(x^-2)",
        "cluster ~ I(x^1.5)",
        "cluster ~ 2",
    ],
)
def test_design_rejects_unsupported_formula_grammar(partition_result, regression_frame, formula):
    b3 = _module()

    with pytest.raises(ValueError, match="formula|unsupported|power|constant"):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula=formula,
            data=regression_frame,
            continuous_predictors=("x",),
            counterfactual_steps={},
        )


def test_design_rejects_constants_unused_levels_aliases_and_rank_deficiency(
    partition_result, regression_frame
):
    b3 = _module()
    constant = regression_frame.assign(constant=1.0)
    unused_level = regression_frame.copy()
    unused_level["group"] = unused_level["group"].cat.add_categories([30])
    aliased = regression_frame.assign(x_copy=regression_frame["x"])

    with pytest.raises(ValueError, match="constant"):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula="cluster ~ constant",
            data=constant,
            continuous_predictors=("constant",),
            counterfactual_steps={},
        )
    with pytest.raises(ValueError, match="unused.*level"):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula="cluster ~ group",
            data=unused_level,
            continuous_predictors=(),
            counterfactual_steps={},
        )
    with pytest.raises(ValueError, match="rank-deficient|aliased"):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula="cluster ~ x + x_copy",
            data=aliased,
            continuous_predictors=("x", "x_copy"),
            counterfactual_steps={},
        )


def test_design_validates_counterfactual_steps_exactly_for_nonlinear_continuous_terms(
    partition_result, regression_frame
):
    b3 = _module()

    for formula in ("cluster ~ x * group", "cluster ~ I(x^2)"):
        with pytest.raises(ValueError, match="counterfactual.*x"):
            b3.prepare_conditional_design(
                partition_posterior=partition_result,
                formula=formula,
                data=regression_frame,
                continuous_predictors=("x",),
                counterfactual_steps={},
            )
        prepared = b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula=formula,
            data=regression_frame,
            continuous_predictors=("x",),
            counterfactual_steps={"x": 0.01},
        )
        assert prepared.counterfactual_steps == (("x", 0.01),)

    for invalid in (0, -0.1, np.inf, np.nan, True):
        with pytest.raises((TypeError, ValueError), match="positive finite|boolean"):
            b3.prepare_conditional_design(
                partition_posterior=partition_result,
                formula="cluster ~ x * group",
                data=regression_frame,
                continuous_predictors=("x",),
                counterfactual_steps={"x": invalid},
            )
    with pytest.raises(ValueError, match="extra|exactly"):
        b3.prepare_conditional_design(
            partition_posterior=partition_result,
            formula="cluster ~ x",
            data=regression_frame,
            continuous_predictors=("x",),
            counterfactual_steps={"x": 0.1},
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"counts": np.asarray([7], dtype=np.int64)}, "counts"),
        ({"masses": np.asarray([0.5])}, "masses"),
        ({"inverse_index": np.asarray([0, 0, 0, 0, 0, 0, 0, 1])}, "inverse"),
        ({"reference_cluster_order": ("B", "A")}, "cluster order|reference"),
    ],
)
def test_design_rejects_incoherent_b2_result(partition_result, regression_frame, changes, message):
    b3 = _module()
    corrupted = replace(partition_result, **changes)

    with pytest.raises(ValueError, match=message):
        b3.prepare_conditional_design(
            partition_posterior=corrupted,
            formula="cluster ~ x",
            data=regression_frame,
            continuous_predictors=("x",),
            counterfactual_steps={},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 13),
        ("K", 4),
        ("B", 129),
        ("U", 17),
        ("design_width", 5),
    ],
)
def test_private_envelope_rejects_every_upper_boundary(field, value):
    b3 = _module()
    controls = {"n": 12, "K": 3, "B": 128, "U": 16, "design_width": 4}
    controls[field] = value

    with pytest.raises(ValueError, match=field):
        b3.validate_private_envelope(**controls)


@pytest.mark.parametrize("field", ["n", "K", "B", "U", "design_width"])
@pytest.mark.parametrize("value", [True, 1.5, np.nan, np.inf])
def test_private_envelope_rejects_boolean_lossy_or_nonfinite_integers(field, value):
    b3 = _module()
    controls = {"n": 12, "K": 3, "B": 128, "U": 16, "design_width": 4}
    controls[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        b3.validate_private_envelope(**controls)


def test_private_envelope_rejects_unique_partition_fraction_above_quarter():
    b3 = _module()

    with pytest.raises(ValueError, match="U / B"):
        b3.validate_private_envelope(n=12, K=3, B=63, U=16, design_width=4)


def _prepared_linear_design(partition_result, regression_frame):
    return _module().prepare_conditional_design(
        partition_posterior=partition_result,
        formula="cluster ~ 1 + x + group",
        data=regression_frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )


def test_reference_softmax_is_stable_simplex_with_exact_structural_zero():
    b3 = _module()
    design = np.asarray([[1.0, 1000.0], [1.0, -1000.0], [1.0, 0.0]])
    beta_nonreference = np.asarray([[2.0, -3.0], [4.0, -5.0]])

    probabilities, beta_full = b3.reference_softmax_probabilities(design, beta_nonreference)

    assert probabilities.shape == (3, 3)
    assert beta_full.shape == (2, 3)
    np.testing.assert_array_equal(beta_full[:, 0], 0.0)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12)
    assert not probabilities.flags.writeable
    assert not beta_full.flags.writeable


@pytest.mark.parametrize(
    ("design", "beta", "message"),
    [
        (np.ones(3), np.ones((1, 1)), "two-dimensional"),
        (np.ones((3, 2)), np.ones(2), "two-dimensional"),
        (np.ones((3, 2)), np.ones((3, 1)), "incompatible"),
        (np.asarray([[1.0, np.nan]]), np.ones((2, 1)), "finite"),
        (np.ones((1, 2)), np.asarray([[1.0], [np.inf]]), "finite"),
    ],
)
def test_reference_softmax_rejects_malformed_or_nonfinite_inputs(design, beta, message):
    b3 = _module()

    with pytest.raises(ValueError, match=message):
        b3.reference_softmax_probabilities(design, beta)


def test_default_prior_has_frozen_intercept_and_coefficient_scales(
    partition_result, regression_frame
):
    b3 = _module()
    prepared = _prepared_linear_design(partition_result, regression_frame)

    prior = b3.prepare_prior(prepared)

    assert prior.source == "default"
    assert prior.coefficient_coordinates == prepared.coefficient_coordinates
    assert prior.nonreference_cluster_coordinates == ("B",)
    assert prior.mean.shape == (3, 1)
    assert prior.sigma.shape == (3, 1)
    np.testing.assert_array_equal(prior.mean, 0.0)
    assert prior.sigma[prepared.coefficient_coordinates.index("(Intercept)"), 0] == 1.5
    for coordinate in ("x", "group10"):
        assert prior.sigma[prepared.coefficient_coordinates.index(coordinate), 0] == 1.0
    assert not prior.mean.flags.writeable
    assert not prior.sigma.flags.writeable


def test_complete_user_prior_replaces_defaults_and_preserves_nonstring_coordinates(
    partition_result, regression_frame
):
    b3 = _module()
    numeric = regression_frame.copy()
    numeric["cluster"] = pd.Categorical([10, 10, 10, 20, 20, 20], categories=[10, 20], ordered=True)
    numeric_partition = replace(
        partition_result,
        reference_labels=np.asarray([10, 10, 10, 20, 20, 20], dtype=object),
        reference_cluster_order=(10, 20),
        aligned_partitions=np.asarray([[10, 10, 10, 20, 20, 20]] * 8, dtype=object),
        distinct_partitions=np.asarray([[10, 10, 10, 20, 20, 20]], dtype=object),
    )
    prepared = b3.prepare_conditional_design(
        partition_posterior=numeric_partition,
        formula="cluster ~ x",
        data=numeric,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    mean = np.asarray([[7.0], [8.0]])
    sigma = np.asarray([[0.7], [0.8]])

    prior = b3.prepare_prior(
        prepared,
        user_prior={
            "mean": mean,
            "sigma": sigma,
            "coefficient_coordinates": prepared.coefficient_coordinates,
            "nonreference_cluster_coordinates": (20,),
        },
    )

    assert prior.source == "user"
    assert prior.nonreference_cluster_coordinates == (20,)
    np.testing.assert_array_equal(prior.mean, mean)
    np.testing.assert_array_equal(prior.sigma, sigma)
    assert prior.mean[0, 0] != 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        {"mean": np.zeros((1, 1))},
        {"sigma": np.ones((1, 1))},
        {"mean": np.asarray([[0.0], [np.nan], [0.0]])},
        {"sigma": np.asarray([[1.0], [0.0], [1.0]])},
        {"coefficient_coordinates": ("x", "(Intercept)", "group10")},
        {"nonreference_cluster_coordinates": ("A",)},
        {"extra": "forbidden"},
    ],
)
def test_user_prior_rejects_partial_shape_nonfinite_reordered_or_extra_entries(
    partition_result, regression_frame, mutation
):
    b3 = _module()
    prepared = _prepared_linear_design(partition_result, regression_frame)
    specification = {
        "mean": np.zeros((3, 1)),
        "sigma": np.ones((3, 1)),
        "coefficient_coordinates": prepared.coefficient_coordinates,
        "nonreference_cluster_coordinates": ("B",),
    }
    specification.update(mutation)

    with pytest.raises((TypeError, ValueError), match="prior|shape|finite|positive|coordinates"):
        b3.prepare_prior(prepared, user_prior=specification)


def test_direct_pymc_model_has_named_coordinates_one_categorical_and_no_b2_weight(
    partition_result, regression_frame
):
    import pymc as pm

    b3 = _module()
    prepared = _prepared_linear_design(partition_result, regression_frame)
    prior = b3.prepare_prior(prepared)

    model = b3.build_component_model(
        prepared=prepared,
        response_codes=prepared.response_codes,
        prior=prior,
    )

    assert tuple(model.coords["observation"]) == prepared.row_ids
    assert tuple(model.coords["cluster"]) == prepared.cluster_order
    assert tuple(model.coords["nonreference_cluster"]) == ("B",)
    assert tuple(model.coords["coefficient"]) == prepared.coefficient_coordinates
    assert len(model.observed_RVs) == 1
    assert isinstance(model.observed_RVs[0].owner.op, pm.Categorical.rv_op.__class__)
    assert model.observed_RVs[0].name == "cluster_id"
    assert {"design_matrix", "beta_nonreference", "beta_full", "logits", "probability"} <= set(
        model.named_vars
    )
    forbidden = ("weight", "mass", "count", "inverse")
    assert not any(fragment in name for name in model.named_vars for fragment in forbidden)
    assert "partition_posterior" not in inspect.signature(b3.build_component_model).parameters


def test_direct_pymc_model_compiles_and_draws_exact_reference_zeros_and_simplexes(
    partition_result, regression_frame
):
    import pymc as pm

    b3 = _module()
    prepared = _prepared_linear_design(partition_result, regression_frame)
    prior = b3.prepare_prior(prepared)
    model = b3.build_component_model(
        prepared=prepared,
        response_codes=prepared.response_codes,
        prior=prior,
    )

    with model:
        assert np.isfinite(model.compile_logp()(model.initial_point()))
        beta_full, probability = pm.draw(
            [model["beta_full"], model["probability"]], draws=3, random_seed=42
        )

    np.testing.assert_array_equal(beta_full[..., 0], 0.0)
    assert probability.shape == (3, partition_result.n, partition_result.K)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=-1), 1.0, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "codes",
    [
        np.asarray([0, 0, 0, 0, 0, 0]),
        np.asarray([0, 0, 0, 1, 1]),
        np.asarray([0, 0, 0, 1, 1, 2]),
        np.asarray([0, 0, 0, 1, 1, np.nan]),
        np.asarray([False, False, False, True, True, True]),
    ],
)
def test_direct_model_rejects_empty_misaligned_invalid_or_boolean_response_codes(
    partition_result, regression_frame, codes
):
    b3 = _module()
    prepared = _prepared_linear_design(partition_result, regression_frame)
    prior = b3.prepare_prior(prepared)

    with pytest.raises((TypeError, ValueError), match="response_codes|cluster"):
        b3.build_component_model(prepared=prepared, response_codes=codes, prior=prior)


def test_bayesian_dependency_import_is_lazy_and_dependency_specific(
    partition_result, regression_frame, monkeypatch
):
    b3 = _module()
    prepared = _prepared_linear_design(partition_result, regression_frame)
    prior = b3.prepare_prior(prepared)
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pymc" or name.startswith("pymc."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pymc", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="PyMC.*B3"):
        b3.build_component_model(
            prepared=prepared,
            response_codes=prepared.response_codes,
            prior=prior,
        )


def _fixed_beta_draws(prepared, *, clusters):
    rng = np.random.default_rng(991)
    values = rng.normal(
        scale=0.35,
        size=(2, 4, len(prepared.coefficient_coordinates), clusters),
    )
    values[..., 0] = 0.0
    return values


def _test_softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def test_linear_continuous_drawwise_effects_match_independent_original_scale_oracle(
    partition_result_three, regression_frame_three
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x",
        data=regression_frame_three,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)
    x_position = prepared.coefficient_coordinates.index("x")
    logits = np.einsum("np,cdpk->cdnk", prepared.design_matrix, beta)
    probability = _test_softmax(logits)
    beta_x = beta[:, :, x_position, :]
    expected = probability * (
        beta_x[:, :, None, :] - np.sum(probability * beta_x[:, :, None, :], axis=-1, keepdims=True)
    )
    expected /= prepared.scaling_transforms[0].population_sd

    effects = b3.compute_drawwise_effects(
        prepared=prepared,
        beta_full=beta,
        component_identifier="component-000",
    )

    assert len(effects) == 1
    effect = effects[0]
    assert effect.metadata.effect_identifier == "derivative:x"
    assert effect.metadata.source_variable == "x"
    assert effect.metadata.estimand_kind == "derivative"
    assert effect.metadata.method == "analytic_softmax"
    assert effect.metadata.original_scale_units == "probability per original x unit"
    assert effect.metadata.averaging_population == "all 9 observed model-frame rows, equal 1/n"
    np.testing.assert_allclose(effect.observation_effects, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(effect.ame_draws, expected.mean(axis=2), atol=1e-12, rtol=1e-12)
    assert effect.ame_draws.shape == (2, 4, 3)
    assert not effect.observation_effects.flags.writeable
    assert not effect.ame_draws.flags.writeable


def test_multicategory_effects_are_zero_sum_per_observation_and_draw(
    partition_result_three, regression_frame_three
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x + group",
        data=regression_frame_three,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)

    effects = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-001"
    )

    assert {effect.metadata.effect_identifier for effect in effects} == {
        "derivative:x",
        "contrast:group:level-1-vs-0",
    }
    for effect in effects:
        np.testing.assert_allclose(
            effect.observation_effects.sum(axis=-1), 0.0, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(effect.ame_draws.sum(axis=-1), 0.0, atol=1e-12, rtol=1e-12)
        assert effect.metadata.zero_sum_atol == 1e-12
        assert effect.metadata.cluster_order == ("A", "B", "C")
        assert effect.metadata.component_identifier == "component-001"
        assert effect.metadata.chain_coordinates == (0, 1)
        assert effect.metadata.draw_coordinates == (0, 1, 2, 3)


@pytest.mark.parametrize("formula", ["cluster ~ x * group", "cluster ~ I(x^2)"])
def test_counterfactual_continuous_derivatives_converge_to_independent_analytic_oracle(
    partition_result_three, regression_frame_three, formula
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula=formula,
        data=regression_frame_three,
        continuous_predictors=("x",),
        counterfactual_steps={"x": _COUNTERFACTUAL_HALF_STEP},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)
    design = prepared.design_matrix
    probability = _test_softmax(np.einsum("np,cdpk->cdnk", design, beta))
    transform = prepared.scaling_transforms[0]
    x_scaled = (regression_frame_three["x"].to_numpy() - transform.mean) / transform.population_sd
    coefficient_derivative = np.zeros((len(regression_frame_three), 3))
    if "*" in formula:
        x_position = prepared.coefficient_coordinates.index("x")
        interaction_position = next(
            index
            for index, label in enumerate(prepared.coefficient_coordinates)
            if "x:" in label or ":x" in label
        )
        group_indicator = (regression_frame_three["group"].astype(object).to_numpy() == 10)[:, None]
        coefficient_derivative = (
            beta[0, 0, x_position, :][None, :]
            + group_indicator * beta[0, 0, interaction_position, :][None, :]
        ) / transform.population_sd
    else:
        power_position = next(
            index for index, label in enumerate(prepared.coefficient_coordinates) if "x^2" in label
        )
        coefficient_derivative = (
            2.0
            * x_scaled[:, None]
            * beta[0, 0, power_position, :][None, :]
            / transform.population_sd
        )
    p = probability[0, 0]
    oracle = p * (
        coefficient_derivative - np.sum(p * coefficient_derivative, axis=-1, keepdims=True)
    )

    effect = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-002"
    )[0]

    assert effect.metadata.method == "central_counterfactual"
    assert effect.metadata.half_step == _COUNTERFACTUAL_HALF_STEP
    assert effect.metadata.evaluated_offsets == (
        -_COUNTERFACTUAL_HALF_STEP,
        _COUNTERFACTUAL_HALF_STEP,
    )
    assert effect.metadata.denominator == 2 * _COUNTERFACTUAL_HALF_STEP
    np.testing.assert_allclose(
        effect.observation_effects[0, 0],
        oracle,
        atol=_COUNTERFACTUAL_CONVERGENCE_ATOL,
        rtol=0.0,
    )


def test_categorical_main_effect_matches_direct_probability_contrast_with_one_over_n_average(
    partition_result_three, regression_frame_three
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ group",
        data=regression_frame_three,
        continuous_predictors=(),
        counterfactual_steps={},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)
    intercept = beta[:, :, prepared.coefficient_coordinates.index("(Intercept)"), :]
    group = beta[:, :, prepared.coefficient_coordinates.index("group10"), :]
    reference_probability = _test_softmax(intercept)
    comparison_probability = _test_softmax(intercept + group)
    expected = comparison_probability - reference_probability

    effect = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-003"
    )[0]

    assert effect.metadata.estimand_kind == "contrast"
    assert effect.metadata.method == "discrete_counterfactual"
    assert effect.metadata.factor_reference == 20
    assert effect.metadata.factor_comparison == 10
    expected_observations = np.broadcast_to(
        expected[:, :, None, :], (2, 4, len(regression_frame_three), 3)
    )
    np.testing.assert_allclose(effect.observation_effects, expected_observations)
    np.testing.assert_allclose(effect.ame_draws, expected)
    assert np.array_equal(effect.ame_draws, np.sum(effect.observation_effects, axis=2) / 9.0)


def test_categorical_counterfactuals_cover_interactions_booleans_and_nonstring_levels(
    partition_result_three, regression_frame_three
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ group * flag",
        data=regression_frame_three,
        continuous_predictors=(),
        counterfactual_steps={},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)

    effects = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-004"
    )

    by_source = {effect.metadata.source_variable: effect for effect in effects}
    assert set(by_source) == {"group", "flag"}
    assert by_source["group"].metadata.factor_reference == 20
    assert by_source["group"].metadata.factor_comparison == 10
    assert by_source["flag"].metadata.factor_reference is False
    assert by_source["flag"].metadata.factor_comparison is True
    for effect in effects:
        assert np.isfinite(effect.observation_effects).all()
        np.testing.assert_allclose(effect.ame_draws, effect.observation_effects.mean(axis=2))


def test_effect_boundary_has_no_b2_weight_mass_or_count_argument(
    partition_result_three, regression_frame_three
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x",
        data=regression_frame_three,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)
    parameters = inspect.signature(b3.compute_drawwise_effects).parameters

    assert not {"weights", "masses", "counts", "partition_posterior"} & set(parameters)
    first = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-005"
    )
    second = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-005"
    )
    np.testing.assert_array_equal(first[0].ame_draws, second[0].ame_draws)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda beta: beta[..., 1:],
        lambda beta: beta[:, :, :-1, :],
        lambda beta: np.concatenate([beta[:, :, :, :1] + 1.0, beta[:, :, :, 1:]], axis=-1),
        lambda beta: np.where(np.indices(beta.shape)[0] == 0, np.nan, beta),
    ],
)
def test_effects_reject_malformed_nonfinite_or_nonzero_reference_coefficients(
    partition_result_three, regression_frame_three, mutation
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x",
        data=regression_frame_three,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    beta = mutation(_fixed_beta_draws(prepared, clusters=3))

    with pytest.raises(ValueError, match="beta_full|finite|reference"):
        b3.compute_drawwise_effects(
            prepared=prepared, beta_full=beta, component_identifier="component-006"
        )


@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
def test_analytic_effect_survives_a_small_scale_continuous_predictor(
    partition_result_three, regression_frame_three, scale
):
    """Finding F-04: the round-off bound must scale with the effect it guards.

    The analytic derivative is divided by the predictor's population standard
    deviation, so its floating-point zero-sum residual grows as
    ``1 / population_sd``.  A bound that stays at ``100 * eps`` therefore
    rejects an ordinary, correct fit whenever the predictor is measured in
    units that make its standard deviation small.
    """
    b3 = _module()
    frame = regression_frame_three.copy(deep=True)
    frame["x"] = np.asarray(frame["x"], dtype=float) * scale
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x",
        data=frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)

    effect = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-scale"
    )[0]

    # The standardized design is scale-invariant, so the reported effect is
    # exactly the unit-scale effect divided by the predictor's own scale.
    assert prepared.scaling_transforms[0].population_sd == pytest.approx(
        1.2909944487358056 * scale, rel=1e-12
    )
    assert np.isfinite(effect.observation_effects).all()
    # The zero-sum identity is exact on the standardized scale, so at the
    # reported scale it holds to 1e-12 relative to the effect's own magnitude.
    tolerance = 1e-12 / prepared.scaling_transforms[0].population_sd
    np.testing.assert_allclose(
        effect.observation_effects.sum(axis=-1), 0.0, atol=tolerance, rtol=0.0
    )
    np.testing.assert_allclose(effect.ame_draws.sum(axis=-1), 0.0, atol=tolerance, rtol=0.0)
    # The reported effect is the unit-scale effect divided by the predictor's
    # own scale, which is what makes the fixed round-off bound wrong.
    unit_prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x",
        data=regression_frame_three,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    unit_effect = b3.compute_drawwise_effects(
        prepared=unit_prepared,
        beta_full=_fixed_beta_draws(unit_prepared, clusters=3),
        component_identifier="component-unit",
    )[0]
    np.testing.assert_allclose(
        np.asarray(effect.ame_draws) * scale, unit_effect.ame_draws, rtol=1e-9, atol=0.0
    )


def test_materially_non_zero_sum_effect_input_is_still_rejected(
    partition_result_three, regression_frame_three
):
    """Finding F-04: widening the bound must not admit a real error.

    A residual far above round-off has to fail on both the analytic and the
    counterfactual path, whatever the predictor's scale.
    """
    b3 = _module()
    frame = regression_frame_three.copy(deep=True)
    frame["x"] = np.asarray(frame["x"], dtype=float) * 1e-8
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x",
        data=frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    beta = _fixed_beta_draws(prepared, clusters=3)
    honest = b3.compute_drawwise_effects(
        prepared=prepared, beta_full=beta, component_identifier="component-scale"
    )[0]
    corrupted = np.array(honest.observation_effects, dtype=float, copy=True)
    magnitude = float(np.max(np.abs(corrupted)))
    corrupted[..., 0] += magnitude

    metadata = b3._effect_metadata(
        prepared,
        beta,
        component_identifier="component-scale",
        effect_identifier="derivative:x",
        source_variable="x",
        estimand_kind="derivative",
        method="analytic_softmax",
        original_scale_units="probability per original x unit",
    )
    scale = 1.0 / prepared.scaling_transforms[0].population_sd

    with pytest.raises(ValueError, match="non-roundoff zero-sum residual"):
        b3._checked_effect(metadata, corrupted, roundoff_scale=scale)


def _two_partition_result(partition_result):
    first = np.asarray(["A", "A", "A", "B", "B", "B"], dtype=object)
    second = np.asarray(["A", "A", "B", "A", "B", "B"], dtype=object)
    inverse = np.asarray([0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int64)
    aligned = np.asarray([first if index == 0 else second for index in inverse], dtype=object)
    counts = np.asarray([6, 2], dtype=np.int64)
    masses = counts / 8.0
    membership = np.zeros((6, 2), dtype=float)
    coclustering = np.zeros((6, 6), dtype=float)
    for row in aligned:
        for cluster_index, cluster in enumerate(("A", "B")):
            members = np.flatnonzero(row == cluster)
            membership[members, cluster_index] += 1.0 / 8.0
            coclustering[np.ix_(members, members)] += 1.0 / 8.0
    entropy_terms = np.zeros_like(membership)
    positive = membership > 0.0
    entropy_terms[positive] = membership[positive] * np.log(membership[positive])
    return replace(
        partition_result,
        aligned_partitions=aligned,
        distinct_partitions=np.asarray([first, second], dtype=object),
        counts=counts,
        masses=masses,
        inverse_index=inverse,
        mass_mcse=np.sqrt(masses * (1.0 - masses) / 8.0),
        membership_probabilities=membership,
        entropy=-entropy_terms.sum(axis=1),
        co_clustering=coclustering,
    )


def test_b3_seed_manifest_is_named_replayable_and_mass_independent():
    b3 = _module()

    first = b3.derive_b3_seed_manifest(root_seed=8675309, U=2)
    repeated = b3.derive_b3_seed_manifest(root_seed=8675309, U=2)
    extended = b3.derive_b3_seed_manifest(root_seed=8675309, U=3)

    assert first == repeated
    assert first.prior_predictive.name == "prior_predictive"
    assert first.prior_predictive.generated_seed == repeated.prior_predictive.generated_seed
    assert len(first.components) == 2
    for index, component in enumerate(first.components):
        assert component.component_index == index
        assert component.name == f"component-{index:03d}"
        assert len(component.chain_seeds) == 4
        assert len(set(component.chain_seeds)) == 4
        assert component.posterior_predictive.name.endswith("posterior_predictive")
        assert component == extended.components[index]
    all_seeds = [first.prior_predictive.generated_seed]
    for component in first.components:
        all_seeds.extend(component.chain_seeds)
        all_seeds.append(component.posterior_predictive.generated_seed)
    assert len(all_seeds) == len(set(all_seeds))


@pytest.mark.parametrize("value", [True, -1, 1.5, np.nan, np.inf])
def test_b3_seed_manifest_rejects_invalid_root_before_randomness(value):
    b3 = _module()

    with pytest.raises((TypeError, ValueError), match="root_seed"):
        b3.derive_b3_seed_manifest(root_seed=value, U=1)


def test_sampler_configuration_is_the_frozen_four_chain_profile():
    b3 = _module()

    assert b3.SAMPLER_CONFIGURATION == b3.SamplerConfiguration(
        chains=4,
        tune=500,
        draws=500,
        cores=1,
        blas_cores=1,
        target_accept=0.90,
        nuts_sampler="pymc",
    )


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ({"all_finite": False}, "nonfinite"),
        ({"max_rhat": 1.01}, "rhat"),
        ({"min_bulk_ess": 399.9}, "bulk_ess"),
        ({"min_tail_ess": 399.9}, "tail_ess"),
        ({"divergences": 1}, "divergences"),
        ({"bfmi_by_chain": (0.29, 0.9, 0.9, 0.9)}, "bfmi"),
        ({"reference_zero": False}, "reference_zero"),
        ({"probability_simplex": False}, "probability_simplex"),
    ],
)
def test_sampler_diagnostic_policy_exposes_every_material_failure(change, failure):
    b3 = _module()
    values = {
        "all_finite": True,
        "max_rhat": 1.001,
        "min_bulk_ess": 800.0,
        "min_tail_ess": 750.0,
        "divergences": 0,
        "bfmi_by_chain": (0.9, 0.91, 0.92, 0.93),
        "reference_zero": True,
        "probability_simplex": True,
    }
    values.update(change)

    diagnostic = b3.evaluate_sampler_diagnostics(**values)

    assert diagnostic.status == "fail"
    assert failure in diagnostic.failure_reasons
    assert diagnostic.thresholds.rhat_exclusive_max == 1.01
    assert diagnostic.thresholds.bulk_ess_minimum == 400
    assert diagnostic.thresholds.tail_ess_minimum == 400
    assert diagnostic.thresholds.bfmi_each_chain_minimum == 0.30


def test_sampler_diagnostic_policy_passes_exact_supported_values():
    b3 = _module()

    diagnostic = b3.evaluate_sampler_diagnostics(
        all_finite=True,
        max_rhat=1.009,
        min_bulk_ess=400.0,
        min_tail_ess=400.0,
        divergences=0,
        bfmi_by_chain=(0.30, 0.31, 0.32, 0.33),
        reference_zero=True,
        probability_simplex=True,
    )

    assert diagnostic.status == "pass"
    assert diagnostic.failure_reasons == ()


def test_nan_diagnostic_reduction_is_invariant_to_summary_order():
    """Finding F-08: a NaN anywhere must be detected wherever it sits.

    ``summarize_target_draws`` sets R-hat and both effective sample sizes to
    NaN for a constant target.  Python's ``max``/``min`` propagate a NaN only
    when it is the first argument, so reducing the per-target summaries with
    the builtins made the component's verdict depend on the position of the
    constant target in the tuple.
    """
    b3 = _module()
    generator = np.random.default_rng(808)
    healthy = b3.summarize_target_draws(
        generator.normal(size=(4, 500)), target_identifier="coefficient:x:cluster-1"
    )
    constant = b3.summarize_target_draws(
        np.ones((4, 500)), target_identifier="coefficient:z:cluster-1"
    )

    assert np.isnan(constant.rhat)
    assert np.isnan(constant.bulk_ess)
    assert np.isnan(constant.tail_ess)
    assert np.isfinite(healthy.rhat)
    # The defect itself, stated as a fact about the builtins.
    assert max((healthy.rhat, constant.rhat)) == healthy.rhat
    assert np.isnan(max((constant.rhat, healthy.rhat)))

    verdicts = []
    for summaries in ((healthy, constant), (constant, healthy)):
        record = b3.evaluate_sampler_diagnostics(
            all_finite=True,
            max_rhat=b3._reduced_max(summary.rhat for summary in summaries),
            min_bulk_ess=b3._reduced_min(summary.bulk_ess for summary in summaries),
            min_tail_ess=b3._reduced_min(summary.tail_ess for summary in summaries),
            divergences=0,
            bfmi_by_chain=(0.9, 0.9, 0.9, 0.9),
            reference_zero=True,
            probability_simplex=True,
        )
        verdicts.append((record.status, tuple(sorted(record.failure_reasons))))

    assert verdicts[0] == verdicts[1]
    assert verdicts[0] == ("fail", ("bulk_ess", "rhat", "tail_ess"))
    assert np.isnan(b3._reduced_max(()))
    assert np.isnan(b3._reduced_min(()))
    assert b3._reduced_max((1.0, 3.0, 2.0)) == 3.0
    assert b3._reduced_min((1.0, 3.0, 2.0)) == 1.0


def test_component_diagnostics_are_fed_by_the_order_invariant_reduction():
    """Finding F-08: the gate's *inputs* must come from the fixed reduction.

    Pinning the semantics of `_reduced_max` and `_reduced_min` is not enough on
    its own. The defect lived in the call site, and a mutation campaign on
    2026-08-26 confirmed that reverting that one call to the builtins survived
    the whole Bayesian/RARCAT suite unnoticed.

    The check reads the *loaded code object*, not the source file, so it holds
    whether the reduction is changed on disk or replaced at run time.
    """
    b3 = _module()

    names = b3.fit_partition_component.__code__.co_names

    assert "_reduced_max" in names
    assert "_reduced_min" in names
    assert "max" not in names
    assert "min" not in names


def test_growth_diagnostic_handles_pass_mean_mcse_and_constant_paths():
    b3 = _module()
    rng = np.random.default_rng(44)
    stable = rng.normal(size=(4, 500))
    shifted = stable.copy()
    shifted[:, :250] += 2.0
    constant = np.ones((4, 500))

    passed = b3.evaluate_monte_carlo_growth(stable, mcse_mean=0.01)
    mean_failed = b3.evaluate_monte_carlo_growth(shifted, mcse_mean=0.01)
    mcse_failed = b3.evaluate_monte_carlo_growth(stable, mcse_mean=1.0)
    constant_failed = b3.evaluate_monte_carlo_growth(constant, mcse_mean=0.0)

    assert passed.status == "pass"
    assert "mean_change" in mean_failed.failure_reasons
    assert "mcse" in mcse_failed.failure_reasons
    assert constant_failed.constant_nonstructural
    assert "constant_nonstructural" in constant_failed.failure_reasons
    assert not np.isfinite(constant_failed.mean_change_in_sd)


def test_posterior_predictive_diagnostic_retains_counts_and_threshold():
    b3 = _module()
    observed = np.asarray([0, 0, 0, 1, 1, 2])
    good = np.broadcast_to(observed, (4, 10, 6)).copy()
    bad = np.zeros((4, 10, 6), dtype=int)

    passed = b3.evaluate_posterior_predictive(
        observed_codes=observed,
        replicated_codes=good,
        cluster_order=("A", "B", "C"),
        seed=123,
        warnings=(),
    )
    failed = b3.evaluate_posterior_predictive(
        observed_codes=observed,
        replicated_codes=bad,
        cluster_order=("A", "B", "C"),
        seed=123,
        warnings=(),
    )

    assert passed.status == "pass"
    assert passed.observed_counts == (("A", 3), ("B", 2), ("C", 1))
    assert passed.maximum_count_fraction_error == 0.0
    assert failed.status == "fail"
    assert failed.maximum_count_fraction_error > 0.20
    assert "posterior_predictive" in failed.failure_reasons


@pytest.mark.parametrize(
    ("message", "classification", "material"),
    [
        ("NumbaWarning: Cannot cache compiled function", "informational", False),
        ("UserWarning: largest_eigval is NaN", "informational", False),
        ("Initializing NUTS using jitter+adapt_diag...", "informational", False),
        ("Sequential sampling (4 chains in 1 job)", "informational", False),
        (
            "We recommend running at least 4 chains for robust computation of "
            "convergence diagnostics",
            "informational",
            False,
        ),
        # Every string below is constructed verbatim by PyMC 6.0.1 in
        # `pymc/stats/convergence.py`.  Finding F-03: the previous
        # parametrisation asserted materiality against
        # "divergent transition detected", which PyMC never emits, so the
        # test passed whether or not the classifier matched anything real.
        (
            "There were 10 divergences after tuning. Increase `target_accept` or reparameterize.",
            "invalid",
            True,
        ),
        (
            "There was 1 divergence after tuning. Increase `target_accept` or reparameterize.",
            "invalid",
            True,
        ),
        (
            "Chain 0 reached the maximum tree depth. Increase `max_treedepth`, "
            "increase `target_accept` or reparameterize.",
            "invalid",
            True,
        ),
        (
            "The rhat statistic is larger than 1.01 for some parameters. This indicates "
            "problems during sampling. See https://arxiv.org/abs/1903.08008 for details",
            "invalid",
            True,
        ),
        (
            "The effective sample size per chain is smaller than 100 for some parameters. "
            " A higher number is needed for reliable rhat and ess computation. "
            "See https://arxiv.org/abs/1903.08008 for details",
            "invalid",
            True,
        ),
        ("RuntimeWarning: overflow encountered in exp", "invalid", True),
        ("RuntimeWarning: invalid value encountered in divide", "invalid", True),
        ("RuntimeWarning: nonfinite posterior value", "invalid", True),
    ],
)
def test_warning_classification_uses_scientific_materiality(message, classification, material):
    b3 = _module()

    record = b3.classify_warning(message)

    assert record.classification == classification
    assert record.material is material


def test_material_warning_terms_are_an_explicit_allowlist_not_a_blanket_check():
    """`SCIENTIFIC_GOVERNANCE.md` requires an allowlist tied to reliability."""
    b3 = _module()

    terms = b3.MATERIAL_WARNING_TERMS

    assert isinstance(terms, tuple)
    assert all(isinstance(term, str) and term == term.casefold() for term in terms)
    assert len(set(terms)) == len(terms)
    assert not b3.classify_warning("").material


def test_material_warning_gate_sees_a_real_pymc_divergence_through_the_logging_channel():
    """Finding F-03: PyMC 6.0.1 reports sampler diagnostics through `logging`.

    `pymc/stats/convergence.py` builds its divergence text and emits it with
    `logger.warning`/`logger.log`, never through `warnings.warn`, so a
    `warnings.catch_warnings(record=True)` block around the sampling call sees
    nothing at all.  Neal's funnel at a low `target_accept` diverges reliably.
    """
    b3 = _module()
    pm = b3._import_pymc()

    with b3.captured_sampler_warnings() as captured:
        with pm.Model():
            log_scale = pm.Normal("log_scale", mu=0.0, sigma=3.0)
            pm.Normal("theta", mu=0.0, sigma=pm.math.exp(log_scale), shape=9)
            trace = pm.sample(
                draws=200,
                tune=200,
                chains=2,
                cores=1,
                random_seed=[20260826, 20260827],
                target_accept=0.50,
                progressbar=False,
                return_inferencedata=True,
            )

    assert int(trace.sample_stats["diverging"].sum().to_numpy()) > 0
    messages = [record.message for record in captured]
    material = [record for record in captured if record.material]
    assert material, messages
    assert any("divergence" in record.message for record in material), messages
    assert all(record.classification == "invalid" for record in material)


def test_captured_sampler_warnings_stays_empty_and_detaches_its_handler():
    """A quiet block must record nothing, and must leave no handler behind."""
    b3 = _module()
    logger = logging.getLogger("pymc")
    before = list(logger.handlers)

    with b3.captured_sampler_warnings() as captured:
        assert len(logger.handlers) == len(before) + 1
        logger.info("Initializing NUTS using jitter+adapt_diag...")

    assert list(logger.handlers) == before
    assert captured == []

    with b3.captured_sampler_warnings() as raised:
        logger.warning(
            "There were 3 divergences after tuning. Increase `target_accept` or reparameterize."
        )
        warnings_module.warn("largest_eigval is NaN", UserWarning, stacklevel=1)

    assert list(logger.handlers) == before
    assert [record.material for record in raised] == [False, True]


def test_sample_wrapper_passes_fixed_serial_direct_pymc_arguments(monkeypatch):
    b3 = _module()
    calls = []
    sentinel = object()

    class FakeModel:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_pm = SimpleNamespace(
        sample=lambda **kwargs: calls.append(kwargs) or sentinel,
    )
    monkeypatch.setattr(b3, "_import_pymc", lambda: fake_pm)

    result = b3.sample_component_posterior(model=FakeModel(), chain_seeds=(11, 12, 13, 14))

    assert result is sentinel
    assert calls == [
        {
            "draws": 500,
            "tune": 500,
            "chains": 4,
            "cores": 1,
            "blas_cores": 1,
            "random_seed": [11, 12, 13, 14],
            "target_accept": 0.90,
            "nuts_sampler": "pymc",
            "progressbar": False,
            "return_inferencedata": True,
        }
    ]


def test_orchestration_runs_shared_prior_once_and_each_exact_component_once(
    partition_result, regression_frame, monkeypatch
):
    b3 = _module()
    two = _two_partition_result(partition_result)
    prior_calls = []
    component_calls = []

    def fake_prior(**kwargs):
        prior_calls.append(kwargs)
        return b3.PriorPredictiveRecord(status="pass", seed=kwargs["seed"].generated_seed)

    def fake_component(**kwargs):
        component_calls.append(kwargs)
        return b3.ComponentFitRecord(
            component_index=kwargs["component_index"],
            component_identifier=kwargs["component_identifier"],
            partition=np.asarray(kwargs["partition"], dtype=object),
            count=kwargs["count"],
            mass=kwargs["mass"],
            mass_mcse=kwargs["mass_mcse"],
            seeds=kwargs["seeds"],
            status="pass",
            timing_seconds=0.25,
        )

    monkeypatch.setattr(b3, "run_prior_predictive", fake_prior)
    monkeypatch.setattr(b3, "fit_partition_component", fake_component)

    result = b3.fit_conditional_posteriors(
        partition_posterior=two,
        formula="cluster ~ x",
        data=regression_frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
        root_seed=712,
    )

    assert result.status == "pass"
    assert len(prior_calls) == 1
    assert len(component_calls) == len(two.distinct_partitions) == 2
    assert [call["component_index"] for call in component_calls] == [0, 1]
    assert [call["count"] for call in component_calls] == [6, 2]
    assert [call["mass"] for call in component_calls] == [0.75, 0.25]
    assert [component.status for component in result.components] == ["pass", "pass"]
    assert result.mean_component_time_seconds == pytest.approx(0.25, abs=1e-12, rel=0.0)
    assert result.projected_serial_time_u16_seconds == pytest.approx(4.0, abs=1e-12, rel=0.0)
    assert not hasattr(result, "mixed_draws")
    assert not hasattr(result, "total_variance")


def test_orchestration_retains_failed_component_mass_continues_and_never_retries(
    partition_result, regression_frame, monkeypatch
):
    b3 = _module()
    two = _two_partition_result(partition_result)
    attempts = []

    monkeypatch.setattr(
        b3,
        "run_prior_predictive",
        lambda **kwargs: b3.PriorPredictiveRecord(
            status="pass", seed=kwargs["seed"].generated_seed
        ),
    )

    def sometimes_fails(**kwargs):
        attempts.append(kwargs["component_index"])
        if kwargs["component_index"] == 0:
            raise RuntimeError("deterministic sampler failure")
        return b3.ComponentFitRecord(
            component_index=1,
            component_identifier=kwargs["component_identifier"],
            partition=np.asarray(kwargs["partition"], dtype=object),
            count=kwargs["count"],
            mass=kwargs["mass"],
            mass_mcse=kwargs["mass_mcse"],
            seeds=kwargs["seeds"],
            status="pass",
            timing_seconds=0.5,
        )

    monkeypatch.setattr(b3, "fit_partition_component", sometimes_fails)

    result = b3.fit_conditional_posteriors(
        partition_posterior=two,
        formula="cluster ~ x",
        data=regression_frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
        root_seed=713,
    )

    assert result.status == "fail"
    assert attempts == [0, 1]
    assert len(result.components) == 2
    failed = result.components[0]
    assert failed.status == "fail"
    assert failed.count == 6
    assert failed.mass == 0.75
    assert failed.mass_mcse == pytest.approx(np.sqrt(0.75 * 0.25 / 8.0), abs=1e-12, rel=0.0)
    assert failed.exception_type == "RuntimeError"
    assert failed.exception_message == "deterministic sampler failure"
    assert result.components[1].status == "pass"
    np.testing.assert_array_equal(result.masses, two.masses)
    assert result.masses.sum() == 1.0


def test_posterior_predictive_wrapper_uses_all_retained_draws_and_fixed_seed(monkeypatch):
    b3 = _module()
    calls = []
    sentinel = object()

    class FakeModel:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_pm = SimpleNamespace(
        sample_posterior_predictive=lambda trace, **kwargs: calls.append((trace, kwargs))
        or sentinel
    )
    monkeypatch.setattr(b3, "_import_pymc", lambda: fake_pm)
    posterior = object()

    result = b3.sample_component_posterior_predictive(
        model=FakeModel(), posterior=posterior, seed=987
    )

    assert result is sentinel
    assert calls == [
        (
            posterior,
            {
                "var_names": ["cluster_id"],
                "random_seed": 987,
                "progressbar": False,
                "return_inferencedata": True,
            },
        )
    ]


def test_target_summary_retains_94_interval_sign_mcse_ess_rhat_and_growth():
    b3 = _module()
    rng = np.random.default_rng(551)
    values = rng.normal(loc=0.4, scale=0.5, size=(4, 500))

    summary = b3.summarize_target_draws(values, target_identifier="coefficient:x:B")

    flattened = values.reshape(-1)
    assert summary.target_identifier == "coefficient:x:B"
    assert summary.posterior_mean == pytest.approx(np.mean(flattened), abs=1e-12, rel=0.0)
    assert summary.posterior_sd == pytest.approx(np.std(flattened, ddof=1), abs=1e-12, rel=0.0)
    np.testing.assert_allclose(summary.interval_94, np.quantile(flattened, [0.03, 0.97]))
    assert summary.probability_above_zero == pytest.approx(
        np.mean(flattened > 0.0), abs=1e-12, rel=0.0
    )
    assert summary.probability_below_zero == pytest.approx(
        np.mean(flattened < 0.0), abs=1e-12, rel=0.0
    )
    assert summary.mcse_mean > 0.0
    assert summary.bulk_ess >= 400.0
    assert summary.tail_ess >= 400.0
    assert summary.rhat < 1.01
    assert summary.growth.status == "pass"


def test_component_fit_builds_summaries_effects_predictive_and_pass_status(
    partition_result, regression_frame, monkeypatch
):
    b3 = _module()
    prepared = b3.prepare_conditional_design(
        partition_posterior=partition_result,
        formula="cluster ~ x",
        data=regression_frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    prior = b3.prepare_prior(prepared)
    seeds = b3.derive_b3_seed_manifest(root_seed=9911, U=1).components[0]
    rng = np.random.default_rng(882)
    nonreference = rng.normal(size=(4, 500, 2, 1))
    beta_full = np.concatenate([np.zeros((4, 500, 2, 1)), nonreference], axis=-1)
    probability = _test_softmax(np.einsum("np,cdpk->cdnk", prepared.design_matrix, beta_full))
    posterior = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "beta_full": (
                    ("chain", "draw", "coefficient", "cluster"),
                    beta_full,
                ),
                "probability": (
                    ("chain", "draw", "observation", "cluster"),
                    probability,
                ),
            }
        ),
        sample_stats=xr.Dataset(
            {
                "diverging": (
                    ("chain", "draw"),
                    np.zeros((4, 500), dtype=bool),
                )
            }
        ),
    )
    replicated = np.broadcast_to(prepared.response_codes, (4, 500, 6)).copy()
    predictive = SimpleNamespace(
        posterior_predictive=xr.Dataset(
            {"cluster_id": (("chain", "draw", "observation"), replicated)}
        )
    )
    posterior_predictive_calls = []
    monkeypatch.setattr(b3, "build_component_model", lambda **kwargs: object())
    monkeypatch.setattr(b3, "sample_component_posterior", lambda **kwargs: posterior)
    monkeypatch.setattr(
        b3,
        "sample_component_posterior_predictive",
        lambda **kwargs: posterior_predictive_calls.append(kwargs) or predictive,
    )
    monkeypatch.setattr(b3, "_bfmi_values", lambda posterior: (0.9, 0.91, 0.92, 0.93))

    component = b3.fit_partition_component(
        prepared=prepared,
        prior=prior,
        response_codes=prepared.response_codes,
        component_index=0,
        component_identifier="component-000",
        partition=partition_result.distinct_partitions[0],
        count=int(partition_result.counts[0]),
        mass=float(partition_result.masses[0]),
        mass_mcse=float(partition_result.mass_mcse[0]),
        seeds=seeds,
    )

    assert component.status == "pass"
    assert component.sampler_diagnostics.status == "pass"
    assert component.posterior_predictive_diagnostics.status == "pass"
    assert len(posterior_predictive_calls) == 1
    assert posterior_predictive_calls[0]["posterior"] is posterior
    assert posterior_predictive_calls[0]["seed"] == seeds.posterior_predictive.generated_seed
    assert len(component.coefficient_summaries) == 2
    assert len(component.effects) == 1
    assert len(component.effect_summaries) == 2
    assert component.failure_reasons == ()
    assert component.posterior is posterior
    assert component.posterior_predictive is predictive


def _degenerate_partition_posterior(
    labels: list[object], *, observation_ids: list[str], root_seed: int
) -> b2.PartitionPosteriorResult:
    level_positions = {level: index for index, level in enumerate(dict.fromkeys(labels))}
    within_level: dict[object, int] = {level: 0 for level in level_positions}
    positions = []
    for label in labels:
        positions.append(5.0 * level_positions[label] + 0.01 * within_level[label])
        within_level[label] += 1
    coordinates = np.asarray(positions, dtype=float)
    return b2.build_partition_posterior(
        dissimilarity=np.abs(coordinates[:, None] - coordinates[None, :]),
        reference_labels=labels,
        observation_ids=observation_ids,
        reference_observation_ids=observation_ids,
        B=8,
        npass=3,
        root_seed=root_seed,
    )


def _run_production_fixture(
    *,
    labels: list[object],
    predictors: dict[str, object],
    formula: str,
    continuous_predictors: tuple[str, ...],
    counterfactual_steps: dict[str, float],
    root_seed: int,
):
    b3 = _module()
    ids = [f"production-{root_seed}-{index}" for index in range(len(labels))]
    cluster_order = list(dict.fromkeys(labels))
    partition = _degenerate_partition_posterior(
        labels,
        observation_ids=ids,
        root_seed=root_seed + 10_000,
    )
    frame = pd.DataFrame(predictors, index=ids)
    frame.insert(
        0,
        "cluster",
        pd.Categorical(labels, categories=cluster_order, ordered=True),
    )
    return b3.fit_conditional_posteriors(
        partition_posterior=partition,
        formula=formula,
        data=frame,
        continuous_predictors=continuous_predictors,
        counterfactual_steps=counterfactual_steps,
        root_seed=root_seed,
    )


@pytest.fixture(scope="module")
def binary_production_replays():
    arguments = {
        "labels": ["A"] * 4 + ["B"] * 4,
        "predictors": {"x_scaled": np.asarray([-1.8, -1.2, -0.7, -0.2, 0.2, 0.7, 1.2, 1.8])},
        "formula": "cluster ~ 1 + x_scaled",
        "continuous_predictors": ("x_scaled",),
        "counterfactual_steps": {},
        "root_seed": 2_026_080_3,
    }
    return _run_production_fixture(**arguments), _run_production_fixture(**arguments)


def _binary_quadrature_at_order(
    order: int, *, x: np.ndarray, x_predictive: float
) -> dict[str, float]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    intercept = np.sqrt(2.0) * 1.5 * nodes[:, None]
    slope = np.sqrt(2.0) * nodes[None, :]
    response = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    eta = intercept[..., None] + slope[..., None] * x
    log_likelihood = np.sum(
        response * -np.logaddexp(0.0, -eta) + (1 - response) * -np.logaddexp(0.0, eta),
        axis=-1,
    )
    prior_weights = np.outer(weights, weights) / np.pi
    stabilized = np.exp(log_likelihood - np.max(log_likelihood))
    posterior_weights = prior_weights * stabilized
    posterior_weights /= posterior_weights.sum()
    predictive = 1.0 / (1.0 + np.exp(-(intercept + x_predictive * slope)))
    return {
        "intercept": float(np.sum(posterior_weights * intercept)),
        "slope": float(np.sum(posterior_weights * slope)),
        "predictive": float(np.sum(posterior_weights * predictive)),
    }


def test_production_binary_matches_independent_81_by_81_gauss_hermite_reference(
    binary_production_replays,
):
    result, _ = binary_production_replays
    component = result.components[0]
    design = result.design
    transform = design.scaling_transforms[0]
    x = design.design_matrix[:, design.coefficient_coordinates.index("x_scaled")]
    x_predictive = (0.75 - transform.mean) / transform.population_sd
    estimates = {
        order: _binary_quadrature_at_order(order, x=x, x_predictive=x_predictive)
        for order in (41, 61, 81)
    }
    grid_error = max(
        abs(estimates[81][target] - estimates[61][target])
        for target in ("intercept", "slope", "predictive")
    )
    beta = component.posterior.posterior["beta_full"]
    posterior_means = {
        "intercept": float(beta.sel(coefficient="(Intercept)", cluster="B").mean()),
        "slope": float(beta.sel(coefficient="x_scaled", cluster="B").mean()),
    }
    predictive_draws = 1.0 / (
        1.0
        + np.exp(
            -(
                beta.sel(coefficient="(Intercept)", cluster="B").values
                + x_predictive * beta.sel(coefficient="x_scaled", cluster="B").values
            )
        )
    )

    assert grid_error <= 1e-8
    assert (
        max(
            abs(posterior_means[target] - estimates[81][target])
            for target in ("intercept", "slope")
        )
        <= 0.08
    )
    assert abs(float(np.mean(predictive_draws)) - estimates[81]["predictive"]) <= 0.03


def test_bambi_018_same_backend_construction_matches_production_design(
    binary_production_replays,
):
    import bambi as bmb

    result, _ = binary_production_replays
    prepared = result.design
    bambi_model = bmb.Model(
        prepared.formula,
        prepared.scaled_design_frame,
        family="categorical",
        auto_scale=False,
        center_predictors=False,
        dropna=False,
    )
    bambi_model.build()
    bambi_design = bambi_model.components["p"].design
    bambi_columns = tuple(
        label for term in bambi_design.common.terms.values() for label in term.labels
    )

    assert bmb.__version__ == "0.18.0"
    assert list(bambi_design.common.data.index) == list(prepared.row_ids)
    assert bambi_columns == prepared.engine_design_labels
    assert tuple(bambi_model.response_component.term.levels) == prepared.cluster_order
    assert bambi_model.family.likelihood.name == "Categorical"
    assert bambi_model.family.link["p"].name == "softmax"


def test_single_distinct_partition_production_fit_is_exact_replay_and_not_b4(
    binary_production_replays,
):
    first, replay = binary_production_replays

    assert first.status == replay.status == "pass"
    assert first.B == replay.B == 8
    assert first.U == replay.U == 1
    assert len(first.components) == len(replay.components) == 1
    assert first.counts.tolist() == replay.counts.tolist() == [8]
    assert first.masses.tolist() == replay.masses.tolist() == [1.0]
    assert first.seed_manifest == replay.seed_manifest
    assert first.components[0].status == replay.components[0].status == "pass"
    np.testing.assert_array_equal(
        first.components[0].posterior.posterior["beta_full"].values,
        replay.components[0].posterior.posterior["beta_full"].values,
    )
    np.testing.assert_array_equal(
        first.components[0].posterior.posterior["probability"].values,
        replay.components[0].posterior.posterior["probability"].values,
    )
    assert len(first.components[0].effects) == len(replay.components[0].effects) == 1
    np.testing.assert_array_equal(
        first.components[0].effects[0].observation_effects,
        replay.components[0].effects[0].observation_effects,
    )
    np.testing.assert_array_equal(
        first.components[0].effects[0].ame_draws,
        replay.components[0].effects[0].ame_draws,
    )
    for forbidden in (
        "mixture_draws",
        "integrated_posterior",
        "variance_decomposition",
        "serialization_manifest",
        "plot_data",
    ):
        assert not hasattr(first, forbidden)


def test_production_algebraic_predictive_diagnostic_and_timing_gate(
    binary_production_replays,
):
    result, _ = binary_production_replays
    component = result.components[0]
    beta = component.posterior.posterior["beta_full"].values
    probability = component.posterior.posterior["probability"].values

    assert result.prior_predictive.status == "pass"
    assert result.prior_predictive.predictive.prior.sizes["draw"] == 500
    assert result.prior_predictive.coordinates.chain == (0,)
    assert result.prior_predictive.coordinates.draw == tuple(range(500))
    assert result.prior_predictive.coordinates.observation == tuple(result.observation_ids)
    assert result.prior_predictive.coordinates.cluster == result.cluster_order
    assert component.sampler_diagnostics.status == "pass"
    assert component.posterior_predictive_diagnostics.status == "pass"
    assert component.posterior_predictive_diagnostics.coordinates.chain == (0, 1, 2, 3)
    assert component.posterior_predictive_diagnostics.coordinates.draw == tuple(range(500))
    assert component.posterior_predictive_diagnostics.coordinates.observation == tuple(
        result.observation_ids
    )
    assert component.posterior_predictive_diagnostics.coordinates.cluster == result.cluster_order
    assert np.array_equal(beta[..., 0], np.zeros_like(beta[..., 0]))
    np.testing.assert_allclose(probability.sum(axis=-1), 1.0, atol=1e-12, rtol=1e-12)
    for effect in component.effects:
        np.testing.assert_allclose(
            effect.observation_effects.sum(axis=-1), 0.0, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(effect.ame_draws.sum(axis=-1), 0.0, atol=1e-12, rtol=1e-12)
    assert result.mean_component_time_seconds > 0.0
    assert result.projected_serial_time_u16_seconds == pytest.approx(
        16.0 * result.mean_component_time_seconds
    )
    assert result.projected_serial_time_u16_seconds <= 900.0


_B3_SIMULATION_SCENARIOS = (
    (
        "balanced",
        ["A", "B"] * 6,
        {"x": np.linspace(-2.0, 2.0, 12)},
        "cluster ~ 1 + x",
        ("x",),
        {},
    ),
    (
        "rare_cluster",
        ["A"] * 8 + ["B"] * 3 + ["C"],
        {"x": np.linspace(-2.0, 2.0, 12)},
        "cluster ~ 1 + x",
        ("x",),
        {},
    ),
    (
        "separated",
        ["A"] * 6 + ["B"] * 6,
        {"x": np.linspace(-3.0, 3.0, 12)},
        "cluster ~ 1 + x",
        ("x",),
        {},
    ),
    (
        "weak_information",
        ["A", "B"] * 6,
        {"x": np.asarray([-1.0, 1.0, -0.8, 0.8, -0.6, 0.6] * 2)},
        "cluster ~ 1 + x",
        ("x",),
        {},
    ),
    (
        "positive_continuous",
        ["A"] * 6 + ["B"] * 6,
        {"x": np.asarray([-2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])},
        "cluster ~ 1 + x",
        ("x",),
        {},
    ),
    (
        "interaction",
        ["A", "A", "A", "B", "B", "B", "B", "B", "B", "A", "A", "A"],
        {
            "x": np.asarray([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5] * 2),
            "group": pd.Categorical(
                ["left"] * 6 + ["right"] * 6,
                categories=["left", "right"],
                ordered=True,
            ),
        },
        "cluster ~ 1 + x * group",
        ("x",),
        {"x": _COUNTERFACTUAL_HALF_STEP},
    ),
    (
        "categorical_contrast",
        ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        {
            "group": pd.Categorical(
                ["low"] * 4 + ["middle"] * 4 + ["high"] * 4,
                categories=["low", "middle", "high"],
                ordered=True,
            )
        },
        "cluster ~ 1 + group",
        (),
        {},
    ),
)


@pytest.mark.parametrize(
    "scenario,labels,predictors,formula,continuous_predictors,counterfactual_steps",
    _B3_SIMULATION_SCENARIOS,
)
def test_fixed_seed_scientific_simulations_exercise_complete_production_path(
    scenario,
    labels,
    predictors,
    formula,
    continuous_predictors,
    counterfactual_steps,
):
    seed = 2_026_100 + next(
        index for index, values in enumerate(_B3_SIMULATION_SCENARIOS) if values[0] == scenario
    )
    result = _run_production_fixture(
        labels=labels,
        predictors=predictors,
        formula=formula,
        continuous_predictors=continuous_predictors,
        counterfactual_steps=counterfactual_steps,
        root_seed=seed,
    )

    assert result.status == "pass", [component.failure_reasons for component in result.components]
    assert result.U == len(result.components) == 1
    assert result.prior_predictive.status == "pass"
    component = result.components[0]
    assert component.status == "pass"
    assert component.sampler_diagnostics.status == "pass"
    assert component.posterior_predictive_diagnostics.status == "pass"
    assert component.effects
    assert all(
        effect.metadata.averaging_population
        == f"all {result.n} observed model-frame rows, equal 1/n"
        for effect in component.effects
    )
    assert all(np.isfinite(effect.ame_draws).all() for effect in component.effects)
    if scenario == "positive_continuous":
        target = next(
            summary
            for summary in component.effect_summaries
            if summary.target_identifier == "effect:derivative:x:cluster-1"
        )
        assert target.probability_above_zero >= 0.90


def test_recorded_software_versions_cover_every_declared_runtime_dependency():
    """Finding F-15: recorded provenance must not drift from the declared set.

    `patsy` builds the design matrix and every counterfactual, and
    `arviz-stats` computes BFMI, a material diagnostic; neither was recorded,
    so a stored bundle could not show which version produced its design.
    """
    b3 = _module()
    bayesian = importlib.import_module("weightedcluster.bayesian")

    recorded = b3._software_versions()
    names = [name for name, _ in recorded]
    recorded_names = set(names)

    assert len(names) == len(recorded_names)
    assert set(bayesian._RUNTIME_DEPENDENCIES) <= recorded_names
    assert "arviz-stats" in recorded_names
    assert {"pymc", "arviz", "pytensor", "numpy", "scipy", "pandas"} <= recorded_names
    for name, version in recorded:
        assert version == importlib_metadata.version(name)
    assert b3._software_versions() == recorded


def _prepared_with_unused_missing_column(partition_result_three, regression_frame_three):
    b3 = _module()
    frame = regression_frame_three.copy(deep=True)
    frame["unrelated"] = np.nan
    return b3.prepare_conditional_design(
        partition_posterior=partition_result_three,
        formula="cluster ~ x",
        data=frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )


def test_both_design_paths_reject_missing_referenced_values(
    partition_result_three, regression_frame_three
):
    """Finding F-16: the two design builders must agree about missing data.

    `_ordered_design` built the initial matrix with `NA_action="drop"`, so it
    would silently discard a row, while the counterfactual rebuild in
    `_build_from_design_info` uses `"raise"` — two paths that must produce
    identically shaped matrices disagreeing about the same input.  The
    completeness guard in `_validate_model_frame_values` makes the
    disagreement unreachable through the public path today; it is removed
    here rather than left as a trap for the deferred missing-data extension.

    Dropping is still available, but only where it is asked for by name: the
    historical naive RARCAT solver reproduces R's complete-case `glm` and
    passes `na_action="drop"` deliberately.
    """
    design = importlib.import_module("weightedcluster._bayesian_rarcat_design")
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})

    with pytest.raises(patsy.PatsyError, match="missing values"):
        design._ordered_design("x", frame)

    complete = frame.dropna()
    _, design_info, positions, _, _, _ = design._ordered_design("x", complete)
    with pytest.raises(patsy.PatsyError, match="missing values"):
        design._build_from_design_info(design_info, positions, frame)

    dropped, _, _, _, _, _ = design._ordered_design("x", frame, na_action="drop")
    assert np.asarray(dropped).shape[0] == 3
    with pytest.raises(ValueError, match="na_action"):
        design._ordered_design("x", complete, na_action="omit")

    # The documented promise that an unrelated unused column may be missing
    # must survive: the model frame is restricted to referenced columns, so
    # the design keeps every row.
    prepared = _prepared_with_unused_missing_column(partition_result_three, regression_frame_three)
    assert prepared.design_matrix.shape[0] == len(partition_result_three.observation_ids) == 9
    assert prepared.observed_model_frame.columns.tolist() == ["cluster", "x"]

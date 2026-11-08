from __future__ import annotations

import importlib
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weightedcluster import _bayesian_rarcat_conditional as b3
from weightedcluster import _bayesian_rarcat_partition as b2


def _module():
    return importlib.import_module("weightedcluster._bayesian_rarcat_mixture")


def _target_summary(identifier: str, values: np.ndarray) -> b3.TargetSummary:
    flat = np.ravel(values)
    posterior_sd = float(np.std(flat, ddof=1))
    growth = b3.GrowthDiagnostic(
        status="pass",
        posterior_sd=posterior_sd,
        half_mean=float(np.mean(flat[: flat.size // 2])),
        full_mean=float(np.mean(flat)),
        mean_change_in_sd=0.01,
        mcse_mean=0.01,
        mcse_fraction_of_sd=0.01,
        constant_nonstructural=False,
        failure_reasons=(),
        thresholds=b3.DIAGNOSTIC_THRESHOLDS,
    )
    interval = np.quantile(flat, [0.03, 0.97])
    return b3.TargetSummary(
        target_identifier=identifier,
        posterior_mean=float(np.mean(flat)),
        posterior_sd=posterior_sd,
        interval_94=(float(interval[0]), float(interval[1])),
        probability_above_zero=float(np.mean(flat > 0.0)),
        probability_below_zero=float(np.mean(flat < 0.0)),
        mcse_mean=0.01,
        bulk_ess=500.0,
        tail_ess=500.0,
        rhat=1.001,
        growth=growth,
        constant_nonstructural=False,
    )


@pytest.fixture(scope="module")
def phase_pair() -> tuple[b2.PartitionPosteriorResult, b3.ConditionalPosteriorResult]:
    positions = np.asarray([0.0, 0.5, 1.0, 3.0, 3.5, 4.0])
    observation_ids = list(range(100, 106))
    partition = b2.build_partition_posterior(
        dissimilarity=np.abs(positions[:, None] - positions[None, :]),
        reference_labels=[10, 10, 10, 20, 20, 20],
        observation_ids=observation_ids,
        reference_observation_ids=observation_ids,
        B=8,
        npass=3,
        root_seed=4,
    )
    assert partition.counts.tolist() == [1, 7]
    frame = pd.DataFrame(
        {
            "cluster": pd.Categorical(
                partition.reference_labels,
                categories=partition.reference_cluster_order,
                ordered=True,
            ),
            "x": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        },
        index=partition.observation_ids,
    )
    design = b3.prepare_conditional_design(
        partition_posterior=partition,
        formula="cluster ~ x",
        data=frame,
        continuous_predictors=("x",),
        counterfactual_steps={},
    )
    prior = b3.prepare_prior(design)
    seeds = b3.derive_b3_seed_manifest(root_seed=991, U=2)
    chain = np.asarray([0, 1], dtype=np.int64)
    draw = np.asarray([0, 1, 2], dtype=np.int64)
    cluster = np.asarray(design.cluster_order)
    observation = np.asarray(design.row_ids)
    prior_probability = np.full((1, 3, 6, 2), 0.5)
    prior_tree = xr.DataTree.from_dict(
        {
            "prior": xr.Dataset(
                {
                    "probability": (
                        ("chain", "draw", "observation", "cluster"),
                        prior_probability,
                    )
                },
                coords={"chain": [0], "draw": draw, "observation": observation, "cluster": cluster},
            ),
            "prior_predictive": xr.Dataset(
                {
                    "cluster_id": (
                        ("chain", "draw", "observation"),
                        np.zeros((1, 3, 6), dtype=np.int64),
                    )
                },
                coords={"chain": [0], "draw": draw, "observation": observation},
            ),
        }
    )
    predictive_coordinates = b3.PredictiveCoordinates(
        chain=(0,), draw=(0, 1, 2), observation=tuple(observation_ids), cluster=(10, 20)
    )
    prior_record = b3.PriorPredictiveRecord(
        status="pass",
        seed=seeds.prior_predictive.generated_seed,
        predictive=prior_tree,
        finite_probability=True,
        probability_simplex=True,
        mean_cluster_counts=((10, 3.0), (20, 3.0)),
        cluster_order=(10, 20),
        coordinates=predictive_coordinates,
    )
    components = []
    for component_index in range(2):
        base = 4.0 if component_index == 0 else 0.5
        nonreference = np.asarray(
            [
                [[base, base + 0.2], [base + 0.4, base + 0.6], [base + 0.8, base + 1.0]],
                [[base + 0.1, base + 0.3], [base + 0.5, base + 0.7], [base + 0.9, base + 1.1]],
            ],
            dtype=np.float64,
        )
        beta_full = np.zeros((2, 3, 2, 2), dtype=np.float64)
        beta_full[..., 1] = nonreference
        logits = np.einsum("np,cdpk->cdnk", design.design_matrix, beta_full)
        logits -= np.max(logits, axis=-1, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=-1, keepdims=True)
        posterior = xr.DataTree.from_dict(
            {
                "posterior": xr.Dataset(
                    {
                        "beta_full": (
                            ("chain", "draw", "coefficient", "cluster"),
                            beta_full,
                        ),
                        "probability": (
                            ("chain", "draw", "observation", "cluster"),
                            probability,
                        ),
                    },
                    coords={
                        "chain": chain,
                        "draw": draw,
                        "coefficient": np.asarray(design.coefficient_coordinates),
                        "cluster": cluster,
                        "observation": observation,
                    },
                ),
                "sample_stats": xr.Dataset(
                    {"diverging": (("chain", "draw"), np.zeros((2, 3), dtype=bool))},
                    coords={"chain": chain, "draw": draw},
                ),
            }
        )
        replicated = np.broadcast_to(design.response_codes, (2, 3, 6)).copy()
        posterior_predictive = xr.DataTree.from_dict(
            {
                "posterior_predictive": xr.Dataset(
                    {"cluster_id": (("chain", "draw", "observation"), replicated)},
                    coords={"chain": chain, "draw": draw, "observation": observation},
                )
            }
        )
        identifier = f"component-{component_index:03d}"
        effects = b3.compute_drawwise_effects(
            prepared=design,
            beta_full=beta_full,
            component_identifier=identifier,
        )
        coefficient_summaries = tuple(
            _target_summary(
                f"coefficient:{coordinate}:cluster-1", beta_full[:, :, coefficient_index, 1]
            )
            for coefficient_index, coordinate in enumerate(design.coefficient_coordinates)
        )
        effect_summaries = tuple(
            _target_summary(
                f"effect:{effect.metadata.effect_identifier}:cluster-{cluster_index}",
                effect.ame_draws[:, :, cluster_index],
            )
            for effect in effects
            for cluster_index in range(2)
        )
        diagnostic = b3.SamplerDiagnosticRecord(
            status="pass",
            all_finite=True,
            max_rhat=1.001,
            min_bulk_ess=500.0,
            min_tail_ess=500.0,
            divergences=0,
            bfmi_by_chain=(0.9, 0.91),
            minimum_bfmi=0.9,
            reference_zero=True,
            probability_simplex=True,
            failure_reasons=(),
            thresholds=b3.DIAGNOSTIC_THRESHOLDS,
        )
        pp_coordinates = b3.PredictiveCoordinates(
            chain=(0, 1),
            draw=(0, 1, 2),
            observation=tuple(observation_ids),
            cluster=(10, 20),
        )
        predictive_diagnostic = b3.PosteriorPredictiveRecord(
            status="pass",
            seed=seeds.components[component_index].posterior_predictive.generated_seed,
            observed_counts=((10, 3), (20, 3)),
            mean_replicated_counts=((10, 3.0), (20, 3.0)),
            maximum_count_fraction_error=0.0,
            cluster_order=(10, 20),
            coordinates=pp_coordinates,
        )
        components.append(
            b3.ComponentFitRecord(
                component_index=component_index,
                component_identifier=identifier,
                partition=partition.distinct_partitions[component_index],
                count=int(partition.counts[component_index]),
                mass=float(partition.masses[component_index]),
                mass_mcse=float(partition.mass_mcse[component_index]),
                seeds=seeds.components[component_index],
                status="pass",
                posterior=posterior,
                posterior_predictive=posterior_predictive,
                posterior_predictive_diagnostics=predictive_diagnostic,
                sampler_diagnostics=diagnostic,
                coefficient_summaries=coefficient_summaries,
                effects=effects,
                effect_summaries=effect_summaries,
                timing_seconds=0.25 + component_index,
                warnings=(b3.WarningRecord("cache unavailable", "informational", False),),
            )
        )
    conditional = b3.ConditionalPosteriorResult(
        status="pass",
        n=partition.n,
        B=partition.B,
        K=partition.K,
        U=2,
        observation_ids=partition.observation_ids,
        reference_labels=partition.reference_labels,
        cluster_order=partition.reference_cluster_order,
        distinct_partitions=partition.distinct_partitions,
        counts=partition.counts,
        masses=partition.masses,
        mass_mcse=partition.mass_mcse,
        root_seed=991,
        seed_manifest=seeds,
        formula="cluster ~ x",
        design=design,
        prior=prior,
        sampler_configuration=b3.SamplerConfiguration(
            chains=2,
            tune=3,
            draws=3,
            cores=1,
            blas_cores=1,
            target_accept=0.9,
            nuts_sampler="pymc",
        ),
        software_versions=(("fixture", "1"),),
        prior_predictive=prior_record,
        components=tuple(components),
        failures=(),
        mean_component_time_seconds=0.75,
        projected_serial_time_u16_seconds=12.0,
    )
    return partition, conditional


def test_phase_boundary_requires_exact_b2_and_b3_result_types():
    b4 = _module()

    with pytest.raises(TypeError, match="PartitionPosteriorResult"):
        b4.validate_phase_inputs(object(), object())


def test_phase_boundary_accepts_a_complete_coherent_pair_without_mutation(phase_pair):
    b4 = _module()
    partition, conditional = phase_pair
    b2_arrays = {
        name: np.array(getattr(partition, name), copy=True)
        for name in (
            "weights",
            "raw_partitions",
            "aligned_partitions",
            "distinct_partitions",
            "counts",
            "masses",
            "mass_mcse",
        )
    }
    b3_arrays = {
        name: np.array(getattr(conditional, name), copy=True)
        for name in ("distinct_partitions", "counts", "masses", "mass_mcse")
    }

    assert b4.validate_phase_inputs(partition, conditional) == (6, 8, 2, 2)

    for name, expected in b2_arrays.items():
        np.testing.assert_array_equal(getattr(partition, name), expected)
    for name, expected in b3_arrays.items():
        np.testing.assert_array_equal(getattr(conditional, name), expected)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n": 7}, "n"),
        ({"B": 9}, "B"),
        ({"K": 3}, "K"),
        ({"U": 1}, "U"),
        ({"observation_ids": np.asarray(range(6), dtype=object)}, "observation IDs"),
        ({"counts": np.asarray([2, 6], dtype=np.int64)}, "counts"),
        ({"masses": np.asarray([0.25, 0.75])}, "masses"),
    ],
)
def test_phase_boundary_rejects_cross_phase_identity_mismatch(phase_pair, change, message):
    b4 = _module()
    partition, conditional = phase_pair

    with pytest.raises(ValueError, match=message):
        b4.validate_phase_inputs(partition, replace(conditional, **change))


def test_phase_boundary_rejects_b2_reconstruction_and_component_order_corruption(phase_pair):
    b4 = _module()
    partition, conditional = phase_pair

    with pytest.raises(ValueError, match="inverse index"):
        b4.validate_phase_inputs(
            replace(partition, inverse_index=partition.inverse_index[::-1]), conditional
        )
    with pytest.raises(ValueError, match="component index"):
        b4.validate_phase_inputs(
            partition,
            replace(conditional, components=conditional.components[::-1]),
        )


def test_unequal_mass_mixture_uses_exact_per_draw_weights_and_total_variance():
    b4 = _module()
    component_draws = (
        np.asarray([-1.0, 1.0]),
        np.asarray([8.0, 12.0]),
    )
    original = tuple(values.copy() for values in component_draws)

    summary = b4.summarize_scalar_mixture(component_draws, np.asarray([0.9, 0.1]))

    np.testing.assert_array_equal(summary.values, [-1.0, 1.0, 8.0, 12.0])
    np.testing.assert_array_equal(summary.component_indices, [0, 0, 1, 1])
    np.testing.assert_allclose(summary.weights, [0.45, 0.45, 0.05, 0.05])
    np.testing.assert_allclose(summary.component_weight_sums, [0.9, 0.1], atol=1e-15, rtol=0)
    np.testing.assert_allclose(summary.component_means, [0.0, 10.0])
    np.testing.assert_allclose(summary.component_variances, [1.0, 4.0])
    assert summary.posterior_mean == pytest.approx(1.0, abs=1e-12, rel=0.0)
    assert summary.within_component_variance == pytest.approx(1.3, abs=1e-12, rel=0.0)
    assert summary.between_component_variance == pytest.approx(9.0, abs=1e-12, rel=0.0)
    assert summary.total_variance == pytest.approx(10.3, abs=1e-12, rel=0.0)
    assert summary.second_moment_variance == pytest.approx(10.3, abs=1e-12, rel=0.0)
    assert summary.variance_residual == pytest.approx(0.0, abs=1e-12)
    assert summary.posterior_sd == pytest.approx(np.sqrt(10.3), abs=1e-12, rel=0.0)
    assert summary.probability_above_zero == pytest.approx(0.55, abs=1e-12, rel=0.0)
    assert summary.probability_below_zero == pytest.approx(0.45, abs=1e-12, rel=0.0)
    assert summary.interval_94 == (-1.0, 12.0)
    assert summary.support_count == 4
    assert summary.component_count == 2
    for before, after in zip(original, component_draws, strict=True):
        np.testing.assert_array_equal(after, before)


def test_second_moment_identity_survives_a_large_common_offset():
    """Finding F-11: the second moment must be taken about the mixture mean.

    ``sum(m*(v + mu**2)) - M**2`` cancels catastrophically once the component
    means are large relative to their standard deviations, so a correct
    mixture raised either "law-of-total-variance second-moment identity
    failed" or "second-moment variance is materially negative".  Shifting
    every draw by a common constant cannot change any variance, so the
    reported decomposition must be unmoved by the offset.
    """
    b4 = _module()
    generator = np.random.default_rng(4242)
    base = (
        generator.normal(scale=1e-3, size=64),
        generator.normal(scale=1e-3, size=64) + 2e-3,
    )
    masses = np.asarray([0.6, 0.4])
    reference = b4.summarize_scalar_mixture(base, masses)

    for offset in (1e2, 1e4, 1e6, 1e8):
        shifted = b4.summarize_scalar_mixture(tuple(item + offset for item in base), masses)
        # Representing an offset draw costs precision in the deviation itself,
        # so the recoverable agreement is bounded by the offset's own double
        # spacing relative to the mixture's standard deviation.  That loss is
        # in the inputs; the fix is about the identity, which must hold exactly.
        recoverable = 100.0 * float(np.spacing(offset)) / float(np.sqrt(reference.total_variance))

        assert shifted.posterior_mean == pytest.approx(reference.posterior_mean + offset, rel=1e-15)
        assert shifted.within_component_variance == pytest.approx(
            reference.within_component_variance, rel=recoverable
        )
        assert shifted.between_component_variance == pytest.approx(
            reference.between_component_variance, rel=recoverable
        )
        assert shifted.total_variance == pytest.approx(reference.total_variance, rel=recoverable)
        assert shifted.second_moment_variance == pytest.approx(
            shifted.total_variance, abs=1e-12, rel=1e-12
        )
        assert abs(shifted.variance_residual) <= 1e-12


def test_second_moment_of_a_well_conditioned_mixture_matches_the_uncentred_form():
    """The stable form must agree with the algebraically identical old one.

    Where cancellation is not a factor, ``sum(m*(v + (mu - M)**2))`` and
    ``sum(m*(v + mu**2)) - M**2`` are the same number to within double
    round-off, so the F-11 change moves no well-conditioned value.
    """
    b4 = _module()
    generator = np.random.default_rng(11)
    component_draws = (
        generator.normal(size=400),
        generator.normal(loc=1.25, scale=0.5, size=250),
        generator.normal(loc=-0.75, scale=2.0, size=150),
    )
    masses = np.asarray([0.5, 0.3, 0.2])

    summary = b4.summarize_scalar_mixture(component_draws, masses)

    means = np.asarray([float(np.mean(item)) for item in component_draws])
    variances = np.asarray([float(np.var(item, ddof=0)) for item in component_draws])
    uncentred = float(np.sum(masses * (variances + np.square(means)))) - summary.posterior_mean**2

    assert summary.second_moment_variance == pytest.approx(uncentred, rel=1e-13)
    assert summary.second_moment_variance == pytest.approx(summary.total_variance, rel=1e-13)


def test_equal_component_weighting_would_overweight_the_rare_partition():
    b4 = _module()

    summary = b4.summarize_scalar_mixture(
        (np.asarray([0.0, 0.0]), np.asarray([10.0, 10.0])),
        np.asarray([0.9, 0.1]),
    )

    assert summary.posterior_mean == pytest.approx(1.0, abs=1e-12, rel=0.0)
    assert summary.posterior_mean != pytest.approx(5.0)


def test_weighted_empirical_quantile_is_stable_inverse_cdf_without_interpolation():
    b4 = _module()
    values = np.asarray([10.0, -2.0, 5.0, 5.0])
    weights = np.asarray([0.1, 0.4, 0.2, 0.3])

    assert b4.weighted_empirical_quantile(values, weights, 0.03) == -2.0
    assert b4.weighted_empirical_quantile(values, weights, 0.40) == -2.0
    assert b4.weighted_empirical_quantile(values, weights, 0.41) == 5.0
    assert b4.weighted_empirical_quantile(values, weights, 0.90) == 5.0
    assert b4.weighted_empirical_quantile(values, weights, 0.91) == 10.0


@pytest.mark.parametrize("quantile", [-0.01, 1.01, np.nan, True])
def test_weighted_empirical_quantile_rejects_invalid_probabilities(quantile):
    b4 = _module()

    with pytest.raises((TypeError, ValueError), match="q"):
        b4.weighted_empirical_quantile([1.0], [1.0], quantile)


def test_single_component_and_constant_targets_have_exact_degenerate_decomposition():
    b4 = _module()

    single = b4.summarize_scalar_mixture((np.asarray([0.0, 1.0, 2.0, 3.0]),), [1.0])
    constant = b4.summarize_scalar_mixture((np.full((2, 3), 7.0),), [1.0])

    assert single.posterior_mean == pytest.approx(1.5, abs=1e-12, rel=0.0)
    assert single.within_component_variance == pytest.approx(1.25, abs=1e-12, rel=0.0)
    assert single.between_component_variance == 0.0
    assert single.total_variance == pytest.approx(1.25, abs=1e-12, rel=0.0)
    assert single.interval_94 == (0.0, 3.0)
    assert constant.posterior_mean == 7.0
    assert constant.posterior_sd == 0.0
    assert constant.within_component_variance == 0.0
    assert constant.between_component_variance == 0.0
    assert constant.total_variance == 0.0


def test_only_bounded_floating_point_negative_variance_is_projected_to_zero():
    b4 = _module()

    assert b4._project_roundoff_variance(-5e-13, name="test") == (0.0, True)
    assert b4._project_roundoff_variance(0.0, name="test") == (0.0, False)
    with pytest.raises(ValueError, match="negative"):
        b4._project_roundoff_variance(-2e-12, name="test")


@pytest.mark.parametrize(
    ("draws", "masses", "message"),
    [
        ((np.asarray([]),), [1.0], "positive retained draw count"),
        ((np.asarray([np.nan]),), [1.0], "finite"),
        ((np.asarray([1.0]), np.asarray([2.0])), [1.0], "one mass per component"),
        ((np.asarray([1.0]),), [0.0], "positive"),
        ((np.asarray([1.0]),), [0.9], "sum to one"),
    ],
)
def test_scalar_mixture_rejects_invalid_support(draws, masses, message):
    b4 = _module()

    with pytest.raises((TypeError, ValueError), match=message):
        b4.summarize_scalar_mixture(draws, masses)


def test_typed_result_extracts_every_coefficient_and_effect_target_with_provenance(
    phase_pair,
):
    b4 = _module()
    partition, conditional = phase_pair
    beta_before = tuple(
        component.posterior.posterior["beta_full"].values.copy()
        for component in conditional.components
    )

    result = b4.integrate_conditional_posteriors(partition, conditional)

    assert result.status == "pass"
    assert type(result) is b4.BayesRARCATResult
    assert result.partition_posterior is partition
    assert result.conditional_posterior is conditional
    assert (result.n, result.B, result.K, result.U) == (6, 8, 2, 2)
    assert result.formula == "cluster ~ x"
    assert result.cluster_order == (10, 20)
    assert result.schema_name == "weightedcluster.bayes_rarcat.private"
    assert result.schema_version == 1
    assert [summary.descriptor.target_identifier for summary in result.coefficient_summaries] == [
        "coefficient:(Intercept):cluster-1",
        "coefficient:x:cluster-1",
    ]
    assert [summary.descriptor.target_identifier for summary in result.effect_summaries] == [
        "effect:derivative:x:cluster-0",
        "effect:derivative:x:cluster-1",
    ]
    assert len(result.supports) == 4
    for target_index, support in enumerate(result.supports):
        assert support.descriptor.target_index == target_index
        assert support.values.shape == (12,)
        assert support.weights.shape == (12,)
        np.testing.assert_array_equal(support.component_indices, [0] * 6 + [1] * 6)
        assert support.component_identifiers == ("component-000", "component-001")
        np.testing.assert_array_equal(support.chain_coordinates, [0, 0, 0, 1, 1, 1] * 2)
        np.testing.assert_array_equal(support.draw_coordinates, [0, 1, 2, 0, 1, 2] * 2)
        np.testing.assert_allclose(
            [support.weights[:6].sum(), support.weights[6:].sum()],
            partition.masses,
            atol=1e-15,
            rtol=0,
        )
    for component, expected in zip(conditional.components, beta_before, strict=True):
        np.testing.assert_array_equal(component.posterior.posterior["beta_full"].values, expected)
    arrays = [result.observation_ids, result.reference_labels]
    for support in result.supports:
        arrays.extend(
            [
                support.values,
                support.weights,
                support.component_indices,
                support.chain_coordinates,
                support.draw_coordinates,
            ]
        )
    assert all(not array.flags.writeable for array in arrays)


def test_integrated_summary_matches_direct_unequal_mass_target_calculation(phase_pair):
    b4 = _module()
    partition, conditional = phase_pair

    result = b4.integrate_conditional_posteriors(partition, conditional)
    summary = result.coefficient_summaries[0]
    component_values = tuple(
        component.posterior.posterior["beta_full"].values[:, :, 0, 1]
        for component in conditional.components
    )
    expected = b4.summarize_scalar_mixture(component_values, partition.masses)

    assert summary.posterior_mean == expected.posterior_mean
    assert summary.total_variance == expected.total_variance
    assert summary.interval_94 == expected.interval_94
    assert summary.probability_above_zero == expected.probability_above_zero
    assert result.effect_summaries[0].posterior_mean == pytest.approx(
        -result.effect_summaries[1].posterior_mean, abs=1e-12
    )


def test_reference_coefficient_and_target_metadata_corruption_are_rejected(phase_pair):
    b4 = _module()
    partition, conditional = phase_pair
    component = conditional.components[0]
    changed_tree = component.posterior.copy(deep=True)
    changed_tree.posterior["beta_full"].values[0, 0, 0, 0] = 0.01
    changed_component = replace(component, posterior=changed_tree)

    with pytest.raises(ValueError, match="reference coefficient"):
        b4.integrate_conditional_posteriors(
            partition,
            replace(
                conditional,
                components=(changed_component, conditional.components[1]),
            ),
        )
    with pytest.raises(ValueError, match="effect targets"):
        b4.integrate_conditional_posteriors(
            partition,
            replace(
                conditional,
                components=(
                    conditional.components[0],
                    replace(conditional.components[1], effects=()),
                ),
            ),
        )


def test_failed_component_blocks_integration_without_drop_retry_or_renormalization(phase_pair):
    b4 = _module()
    partition, conditional = phase_pair
    failed = replace(
        conditional.components[0],
        status="fail",
        posterior=None,
        posterior_predictive=None,
        sampler_diagnostics=None,
        posterior_predictive_diagnostics=None,
        coefficient_summaries=(),
        effects=(),
        effect_summaries=(),
        failure_reasons=("sampling_exception",),
        exception_type="RuntimeError",
        exception_message="fixture failure",
    )
    blocked_conditional = replace(
        conditional,
        status="fail",
        components=(failed, conditional.components[1]),
        failures=(failed,),
    )

    result = b4.integrate_conditional_posteriors(partition, blocked_conditional)

    assert result.status == "blocked"
    assert result.failures == (failed,)
    assert result.supports == ()
    assert result.coefficient_summaries == ()
    assert result.effect_summaries == ()
    assert result.allocation is None
    np.testing.assert_array_equal(result.masses, [0.125, 0.875])


def test_largest_remainder_allocation_is_stable_complete_and_noninferential(phase_pair):
    b4 = _module()
    partition, conditional = phase_pair

    allocation = b4.largest_remainder_allocation([1 / 3, 1 / 3, 1 / 3], total=2)
    result = b4.integrate_conditional_posteriors(partition, conditional)

    np.testing.assert_array_equal(allocation.allocations, [1, 1, 0])
    np.testing.assert_allclose(allocation.effective_masses, [0.5, 0.5, 0.0])
    assert allocation.max_absolute_mass_error == pytest.approx(1 / 3, abs=1e-12, rel=0.0)
    assert result.allocation.requested_total == 10_000
    np.testing.assert_array_equal(result.allocation.allocations, [1250, 8750])
    assert result.allocation.max_absolute_mass_error == 0.0
    assert np.all(result.allocation.allocations > 0)
    assert result.supports[0].weights[0] == pytest.approx(0.125 / 6.0, abs=1e-12, rel=0.0)


@pytest.mark.parametrize(("masses", "total"), [([0.0, 1.0], 10), ([0.5, 0.5], True)])
def test_largest_remainder_allocation_rejects_invalid_inputs(masses, total):
    b4 = _module()

    with pytest.raises((TypeError, ValueError)):
        b4.largest_remainder_allocation(masses, total=total)


def _assert_scientific_results_equal(left, right):
    assert left.status == right.status
    assert left.failures == right.failures
    assert left.warnings == right.warnings
    assert (left.n, left.B, left.K, left.U, left.formula) == (
        right.n,
        right.B,
        right.K,
        right.U,
        right.formula,
    )
    assert left.cluster_order == right.cluster_order
    np.testing.assert_array_equal(left.observation_ids, right.observation_ids)
    np.testing.assert_array_equal(left.reference_labels, right.reference_labels)
    for left_support, right_support in zip(left.supports, right.supports, strict=True):
        assert left_support.descriptor == right_support.descriptor
        assert left_support.component_identifiers == right_support.component_identifiers
        for name in (
            "values",
            "weights",
            "component_indices",
            "chain_coordinates",
            "draw_coordinates",
            "component_offsets",
        ):
            np.testing.assert_array_equal(getattr(left_support, name), getattr(right_support, name))
    for left_summary, right_summary in zip(
        (*left.coefficient_summaries, *left.effect_summaries),
        (*right.coefficient_summaries, *right.effect_summaries),
        strict=True,
    ):
        assert left_summary.descriptor == right_summary.descriptor
        for name in (
            "support_count",
            "component_count",
            "posterior_mean",
            "posterior_sd",
            "interval_94",
            "probability_above_zero",
            "probability_below_zero",
            "within_component_variance",
            "between_component_variance",
            "total_variance",
            "second_moment_variance",
            "variance_residual",
            "projected_roundoff",
        ):
            assert getattr(left_summary, name) == getattr(right_summary, name)
        for name in (
            "component_means",
            "component_variances",
            "component_masses",
            "component_draw_counts",
        ):
            np.testing.assert_array_equal(getattr(left_summary, name), getattr(right_summary, name))
    for name in (
        "allocations",
        "effective_masses",
        "signed_mass_errors",
        "absolute_mass_errors",
    ):
        np.testing.assert_array_equal(
            getattr(left.allocation, name), getattr(right.allocation, name)
        )


def test_parallel_extraction_is_bitwise_stable_under_reversed_completion(phase_pair, monkeypatch):
    b4 = _module()
    partition, conditional = phase_pair
    serial = b4.integrate_conditional_posteriors(partition, conditional)
    original_inspect = b4._inspect_component
    original_batch = b4.run_task_batch
    batch_calls = []

    def delayed_inspect(component, conditional_result, *, copy_arrays):
        if copy_arrays and component.component_index == 0:
            time.sleep(0.03)
        return original_inspect(component, conditional_result, copy_arrays=copy_arrays)

    def batch_spy(*args, **kwargs):
        batch_calls.append(kwargs.copy())
        return original_batch(*args, **kwargs)

    monkeypatch.setattr(b4, "_inspect_component", delayed_inspect)
    monkeypatch.setattr(b4, "run_task_batch", batch_spy)

    parallel = b4.integrate_conditional_posteriors(partition, conditional, parallel=True, n_jobs=2)

    assert serial.execution_mode == "serial"
    assert parallel.execution_mode == "parallel"
    assert parallel.resolved_workers == 2
    assert batch_calls == [
        {
            "parallel": True,
            "n_jobs": 2,
            "progressbar": False,
            "desc": "Bayesian RARCAT B4 extraction",
            "executor_kind": "thread",
        }
    ]
    _assert_scientific_results_equal(serial, parallel)
    pd.testing.assert_frame_equal(b4.component_mass_data(serial), b4.component_mass_data(parallel))
    pd.testing.assert_frame_equal(
        b4.integrated_target_data(serial), b4.integrated_target_data(parallel)
    )
    pd.testing.assert_frame_equal(
        b4.component_target_data(serial), b4.component_target_data(parallel)
    )
    pd.testing.assert_frame_equal(b4.effect_data(serial), b4.effect_data(parallel))
    assert b4.format_bayes_rarcat_result(serial) == b4.format_bayes_rarcat_result(parallel)


def test_semantic_frames_preserve_stable_order_identity_and_are_fresh(phase_pair):
    b4 = _module()
    result = b4.integrate_conditional_posteriors(*phase_pair)

    mass = b4.component_mass_data(result)
    targets = b4.integrated_target_data(result)
    components = b4.component_target_data(result)
    effects = b4.effect_data(result)

    assert mass.columns.tolist() == [
        "component_index",
        "component_identifier",
        "count",
        "mass",
        "mass_mcse",
        "conditional_draw_count",
        "support_weight_sum",
        "allocation",
        "effective_allocation_mass",
        "allocation_mass_error",
    ]
    assert targets["target_identifier"].tolist() == [
        support.descriptor.target_identifier for support in result.supports
    ]
    assert components[["target_order", "component_index"]].values.tolist() == [
        [target_index, component_index] for target_index in range(4) for component_index in range(2)
    ]
    assert effects["cluster_coordinate"].tolist() == [10, 20]
    assert all(not isinstance(value, str) for value in effects["cluster_coordinate"])
    mass.loc[0, "mass"] = -1.0
    targets.loc[0, "posterior_mean"] = -999.0
    assert result.masses[0] == 0.125
    assert result.coefficient_summaries[0].posterior_mean != -999.0


def test_headless_formatter_is_deterministic_and_imports_no_graphics(phase_pair, monkeypatch):
    b4 = _module()
    result = b4.integrate_conditional_posteriors(*phase_pair)
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"matplotlib", "seaborn"}:
            raise AssertionError("graphical backend imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    before = pd.get_option("display.max_rows")
    first = b4.format_bayes_rarcat_result(result)
    second = b4.format_bayes_rarcat_result(result)

    assert first == second
    assert "status: pass" in first
    assert "n=6 B=8 K=2 U=2" in first
    assert "cluster ~ x" in first
    assert "targets: 4" in first
    assert pd.get_option("display.max_rows") == before


def _conditional_for_partition(partition, template):
    seeds = b3.derive_b3_seed_manifest(
        root_seed=template.root_seed, U=len(partition.distinct_partitions)
    )
    components = []
    for component_index, component_partition in enumerate(partition.distinct_partitions):
        source = template.components[component_index % len(template.components)]
        identifier = f"component-{component_index:03d}"
        effects = tuple(
            replace(
                effect,
                metadata=replace(
                    effect.metadata,
                    component_identifier=identifier,
                ),
            )
            for effect in source.effects
        )
        predictive_diagnostic = replace(
            source.posterior_predictive_diagnostics,
            seed=seeds.components[component_index].posterior_predictive.generated_seed,
        )
        components.append(
            replace(
                source,
                component_index=component_index,
                component_identifier=identifier,
                partition=component_partition,
                count=int(partition.counts[component_index]),
                mass=float(partition.masses[component_index]),
                mass_mcse=float(partition.mass_mcse[component_index]),
                seeds=seeds.components[component_index],
                effects=effects,
                posterior_predictive_diagnostics=predictive_diagnostic,
            )
        )
    prior_predictive = replace(
        template.prior_predictive,
        seed=seeds.prior_predictive.generated_seed,
    )
    mean_time = float(np.mean([component.timing_seconds for component in components]))
    return replace(
        template,
        n=partition.n,
        B=partition.B,
        K=partition.K,
        U=len(partition.distinct_partitions),
        observation_ids=partition.observation_ids,
        reference_labels=partition.reference_labels,
        cluster_order=partition.reference_cluster_order,
        distinct_partitions=partition.distinct_partitions,
        counts=partition.counts,
        masses=partition.masses,
        mass_mcse=partition.mass_mcse,
        seed_manifest=seeds,
        prior_predictive=prior_predictive,
        components=tuple(components),
        failures=(),
        mean_component_time_seconds=mean_time,
        projected_serial_time_u16_seconds=16.0 * mean_time,
    )


@pytest.fixture(scope="module")
def three_component_pair(phase_pair):
    positions = np.asarray([0.0, 0.5, 1.0, 3.0, 3.5, 4.0])
    ids = list(range(100, 106))
    partition = b2.build_partition_posterior(
        dissimilarity=np.abs(positions[:, None] - positions[None, :]),
        reference_labels=[10, 10, 10, 20, 20, 20],
        observation_ids=ids,
        reference_observation_ids=ids,
        B=12,
        npass=3,
        root_seed=6,
    )
    assert partition.counts.tolist() == [10, 1, 1]
    return partition, _conditional_for_partition(partition, phase_pair[1])


@pytest.fixture(scope="module")
def single_component_pair(phase_pair):
    positions = np.asarray([0.0, 0.01, 0.02, 5.0, 5.01, 5.02])
    ids = list(range(100, 106))
    partition = b2.build_partition_posterior(
        dissimilarity=np.abs(positions[:, None] - positions[None, :]),
        reference_labels=[10, 10, 10, 20, 20, 20],
        observation_ids=ids,
        reference_observation_ids=ids,
        B=8,
        npass=3,
        root_seed=13,
    )
    assert partition.counts.tolist() == [8]
    return partition, _conditional_for_partition(partition, phase_pair[1])


def test_three_component_rare_partition_gate_uses_exact_mass_and_direct_variance(
    three_component_pair,
):
    b4 = _module()
    partition, conditional = three_component_pair

    result = b4.integrate_conditional_posteriors(partition, conditional)

    np.testing.assert_array_equal(partition.counts, [10, 1, 1])
    np.testing.assert_allclose(partition.masses, [10 / 12, 1 / 12, 1 / 12])
    np.testing.assert_array_equal(result.allocation.allocations, [8334, 833, 833])
    assert result.allocation.max_absolute_mass_error == pytest.approx(
        2 / 30_000, abs=1e-12, rel=0.0
    )
    for support, summary in zip(
        result.supports,
        (*result.coefficient_summaries, *result.effect_summaries),
        strict=True,
    ):
        component_weight_sums = [
            support.weights[support.component_indices == index].sum() for index in range(3)
        ]
        np.testing.assert_allclose(component_weight_sums, partition.masses, atol=1e-15, rtol=0)
        direct_mean = float(np.sum(support.weights * support.values))
        direct_variance = float(np.sum(support.weights * np.square(support.values - direct_mean)))
        assert summary.posterior_mean == pytest.approx(direct_mean, abs=1e-12)
        assert summary.total_variance == pytest.approx(direct_variance, abs=1e-12)
        assert summary.total_variance == pytest.approx(summary.second_moment_variance, abs=1e-12)
        assert support.weights[0] == pytest.approx((10 / 12) / 6, abs=1e-12, rel=0.0)
        assert support.weights[-1] == pytest.approx((1 / 12) / 6, abs=1e-12, rel=0.0)


def test_single_component_typed_gate_matches_component_empirical_posterior(
    single_component_pair,
):
    b4 = _module()
    partition, conditional = single_component_pair

    result = b4.integrate_conditional_posteriors(partition, conditional)

    assert result.U == 1
    for support, summary in zip(
        result.supports,
        (*result.coefficient_summaries, *result.effect_summaries),
        strict=True,
    ):
        values = support.values
        ordered = np.sort(values, kind="stable")
        lower = ordered[int(np.ceil(0.03 * len(values))) - 1]
        upper = ordered[int(np.ceil(0.97 * len(values))) - 1]
        np.testing.assert_allclose(support.weights, np.full(len(values), 1 / len(values)))
        assert summary.posterior_mean == pytest.approx(np.mean(values), abs=1e-12, rel=0.0)
        assert summary.probability_above_zero == pytest.approx(
            np.mean(values > 0), abs=1e-12, rel=0.0
        )
        assert summary.probability_below_zero == pytest.approx(
            np.mean(values < 0), abs=1e-12, rel=0.0
        )
        assert summary.interval_94 == (lower, upper)
        assert summary.between_component_variance == 0.0
        assert summary.total_variance == pytest.approx(np.var(values, ddof=0), abs=1e-12, rel=0.0)


def test_frozen_b1_count_allocation_replays_through_production_rule():
    b4 = _module()
    counts = np.asarray([51, 26, 32, 1, 2, 2, 2, 1, 6, 2, 1, 1, 1])
    masses = counts / 128.0
    quotas = masses * 10_000
    expected = np.floor(quotas).astype(np.int64)
    remaining = 10_000 - int(expected.sum())
    stable_order = sorted(
        range(len(counts)), key=lambda index: (-(quotas[index] - expected[index]), index)
    )
    for index in stable_order[:remaining]:
        expected[index] += 1

    allocation = b4.largest_remainder_allocation(masses, total=10_000)

    np.testing.assert_array_equal(allocation.allocations, expected)
    assert allocation.allocations.sum() == 10_000
    assert np.all(allocation.allocations > 0)
    assert allocation.max_absolute_mass_error <= 1 / 10_000


def test_mixture_identity_checks_are_the_single_defensive_owner():
    """The mixture module must not carry its own, looser identity rule.

    It used to define `_same_identity` as `type(left) is type(right) and
    bool(left == right)`, a second copy of the rule `bayesian.py` owns. On
    `pd.NA` that raised TypeError, on a two-element array ValueError, and on a
    one-element array it returned True -- accepting an identity the owner
    rejects, the permissive direction for an identity gate (2026-09-21 targeted
    audit, L-4). Both names now resolve to the owner, so the cases fail closed.
    """
    mixture = importlib.import_module("weightedcluster._bayesian_rarcat_mixture")
    bayesian = importlib.import_module("weightedcluster.bayesian")

    assert mixture._same_identity is bayesian._same_identity_value
    assert mixture._same_identity_sequence is bayesian._same_identity_sequence

    for value in (pd.NA, np.array([1, 2]), np.array([1])):
        assert mixture._same_identity(value, value) is False
        assert mixture._same_identity_sequence([value], [value]) is False
    # Ordinary scalar labels are unchanged, including the exact-type rule.
    assert mixture._same_identity("cluster1", "cluster1") is True
    assert mixture._same_identity(1, 1.0) is False

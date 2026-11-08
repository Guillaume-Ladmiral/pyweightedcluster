from __future__ import annotations

import inspect
import itertools
from dataclasses import replace

import numpy as np
import pytest

import weightedcluster
from weightedcluster import _bayesian_rarcat_partition as b2


def _distance_fixture() -> np.ndarray:
    positions = np.asarray([0.0, 0.2, 1.0, 1.2])
    return np.abs(positions[:, None] - positions[None, :])


def _co_clustering(partition: np.ndarray) -> np.ndarray:
    values = np.asarray(partition)
    return values[:, None] == values[None, :]


def _exhaustive_assignment(overlap: np.ndarray) -> dict[str, object]:
    candidates = [
        (float(sum(overlap[row, assignment[row]] for row in range(overlap.shape[0]))), assignment)
        for assignment in itertools.permutations(range(overlap.shape[0]))
    ]
    best_score = max(score for score, _ in candidates)
    optimal = [
        assignment
        for score, assignment in candidates
        if np.isclose(score, best_score, atol=1e-12, rtol=1e-12)
    ]
    chosen = min(optimal)
    runner_up = max(score for score, assignment in candidates if assignment != chosen)
    return {
        "assignment": chosen,
        "best_score": best_score,
        "runner_up_score": runner_up,
        "ambiguous": np.isclose(best_score, runner_up, atol=1e-12, rtol=1e-12),
    }


def test_partition_inputs_validate_identity_reference_order_and_controls():
    validated = b2.validate_partition_inputs(
        dissimilarity=_distance_fixture(),
        reference_labels=np.asarray(["later", "first", "later", "first"], dtype=object),
        observation_ids=["a", "b", "c", "d"],
        reference_observation_ids=["a", "b", "c", "d"],
        B=np.int64(5),
        npass=3.0,
        root_seed=np.int64(712),
    )

    assert validated.n == 4
    assert validated.B == 5
    assert validated.npass == 3
    assert validated.root_seed == 712
    assert validated.K == 2
    assert validated.reference_cluster_order == ("later", "first")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("B", True),
        ("B", 1.5),
        ("B", np.inf),
        ("B", 0),
        ("npass", False),
        ("npass", 2.25),
        ("npass", np.nan),
        ("npass", 0),
        ("root_seed", True),
        ("root_seed", 3.5),
        ("root_seed", np.inf),
        ("root_seed", -1),
    ],
)
def test_partition_inputs_reject_invalid_integer_controls(field: str, value: object):
    kwargs = {
        "dissimilarity": _distance_fixture(),
        "reference_labels": [0, 0, 1, 1],
        "observation_ids": ["a", "b", "c", "d"],
        "reference_observation_ids": ["a", "b", "c", "d"],
        "B": 5,
        "npass": 3,
        "root_seed": 712,
    }
    kwargs[field] = value

    # 2026-09-02 audit WC-090212: root_seed now uses the same strict gate as
    # conditional_root_seed, which raises TypeError for type-family
    # violations (booleans, non-index numerics) and ValueError for range.
    if field == "root_seed" and not isinstance(value, int):
        expected: type[Exception] | tuple[type[Exception], ...] = TypeError
    elif field == "root_seed" and isinstance(value, bool):
        expected = TypeError
    else:
        expected = ValueError
    with pytest.raises(expected, match=field):
        b2.validate_partition_inputs(**kwargs)


@pytest.mark.parametrize(
    "bad_distance",
    [
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([[0.0, 1.0], [2.0, 0.0]]),
        np.asarray([[0.0, -1.0], [-1.0, 0.0]]),
        np.asarray([[0.0, np.inf], [np.inf, 0.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_partition_inputs_enforce_strict_square_distance_contract(bad_distance: np.ndarray):
    with pytest.raises(ValueError):
        b2.validate_partition_inputs(
            dissimilarity=bad_distance,
            reference_labels=[0, 1],
            observation_ids=["a", "b"],
            reference_observation_ids=["a", "b"],
            B=5,
            npass=3,
            root_seed=712,
        )


@pytest.mark.parametrize(
    ("observation_ids", "reference_observation_ids"),
    [
        (["a", "b", "c", "d"], ["b", "a", "c", "d"]),
        (["a", "a", "c", "d"], ["a", "a", "c", "d"]),
        (["a", "b", "c"], ["a", "b", "c"]),
        ([["a"], ["b"], ["c"], ["d"]], [["a"], ["b"], ["c"], ["d"]]),
    ],
)
def test_partition_inputs_reject_misaligned_or_invalid_identifiers(
    observation_ids: object,
    reference_observation_ids: object,
):
    with pytest.raises((TypeError, ValueError), match="identifier|identifiers"):
        b2.validate_partition_inputs(
            dissimilarity=_distance_fixture(),
            reference_labels=[0, 0, 1, 1],
            observation_ids=observation_ids,
            reference_observation_ids=reference_observation_ids,
            B=5,
            npass=3,
            root_seed=712,
        )


@pytest.mark.parametrize(
    "reference_labels",
    [
        [0, 0, 0, 0],
        [0, 1, 2, 3, 4],
        [0, 0, 1],
        [0, None, 1, 1],
        [0, np.nan, 1, 1],
        [0, [0], 1, 1],
    ],
)
def test_partition_inputs_reject_missing_malformed_or_invalid_k_reference(reference_labels: object):
    with pytest.raises((TypeError, ValueError), match="reference|cluster|K|length"):
        b2.validate_partition_inputs(
            dissimilarity=_distance_fixture(),
            reference_labels=reference_labels,
            observation_ids=["a", "b", "c", "d"],
            reference_observation_ids=["a", "b", "c", "d"],
            B=5,
            npass=3,
            root_seed=712,
        )


def test_bayesian_bootstrap_weights_are_exactly_replayable_positive_and_normalized():
    first = b2.draw_bayesian_bootstrap_weights(n_draws=7, n_observations=4, seed=9123)
    second = b2.draw_bayesian_bootstrap_weights(n_draws=7, n_observations=4, seed=9123)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (7, 4)
    assert first.dtype == np.float64
    assert np.isfinite(first).all()
    assert np.all(first > 0.0)
    np.testing.assert_allclose(first.sum(axis=1), 4.0, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("n_draws", "n_observations", "seed"),
    [(0, 4, 1), (2.5, 4, 1), (2, 0, 1), (2, False, 1), (2, 4, -1), (2, 4, True)],
)
def test_bayesian_bootstrap_weights_reject_invalid_controls(
    n_draws: object,
    n_observations: object,
    seed: object,
):
    # WC-090212: booleans hit the strict shared seed gate's TypeError.
    expected = TypeError if isinstance(seed, bool) else ValueError
    with pytest.raises(expected):
        b2.draw_bayesian_bootstrap_weights(
            n_draws=n_draws,
            n_observations=n_observations,
            seed=seed,
        )


def test_seed_streams_are_named_replayable_and_isolated_from_schedule_size():
    first = b2.derive_seed_streams(54321)
    repeated = b2.derive_seed_streams(54321)
    weights_npass_2 = b2.draw_bayesian_bootstrap_weights(
        n_draws=6,
        n_observations=4,
        seed=first.weight_seed_sequence(),
    )
    weights_npass_20 = b2.draw_bayesian_bootstrap_weights(
        n_draws=6,
        n_observations=4,
        seed=repeated.weight_seed_sequence(),
    )

    assert first.weight_stream.name == "bayesian_bootstrap_weights"
    assert first.start_stream.name == "kmedoids_start_schedule"
    assert first.weight_stream.spawn_key != first.start_stream.spawn_key
    assert first == repeated
    np.testing.assert_array_equal(weights_npass_2, weights_npass_20)


def test_replay_start_schedule_is_exact_unique_zero_based_and_seed_isolated_from_B():
    streams_before_small_weights = b2.derive_seed_streams(54321)
    b2.draw_bayesian_bootstrap_weights(
        n_draws=2,
        n_observations=6,
        seed=streams_before_small_weights.weight_seed_sequence(),
    )
    starts_after_small_weights = b2.build_replay_start_schedule(
        n_observations=6,
        k=2,
        npass=5,
        seed=streams_before_small_weights.start_seed_sequence(),
    )
    streams_before_large_weights = b2.derive_seed_streams(54321)
    b2.draw_bayesian_bootstrap_weights(
        n_draws=20,
        n_observations=6,
        seed=streams_before_large_weights.weight_seed_sequence(),
    )
    starts_after_large_weights = b2.build_replay_start_schedule(
        n_observations=6,
        k=2,
        npass=5,
        seed=streams_before_large_weights.start_seed_sequence(),
    )

    np.testing.assert_array_equal(starts_after_small_weights, starts_after_large_weights)
    assert starts_after_small_weights.shape == (5, 2)
    assert np.issubdtype(starts_after_small_weights.dtype, np.integer)
    assert np.all((0 <= starts_after_small_weights) & (starts_after_small_weights < 6))
    assert all(len(set(row)) == 2 for row in starts_after_small_weights.tolist())


@pytest.mark.parametrize(
    ("n_observations", "k", "npass", "seed"),
    [
        (0, 2, 3, 1),
        (4, 1, 3, 1),
        (4, 5, 3, 1),
        (4, 2, 0, 1),
        (4, 2.5, 3, 1),
        (4, 2, True, 1),
        (4, 2, 3, -1),
    ],
)
def test_replay_start_schedule_rejects_invalid_controls(
    n_observations: object,
    k: object,
    npass: object,
    seed: object,
):
    with pytest.raises(ValueError):
        b2.build_replay_start_schedule(
            n_observations=n_observations,
            k=k,
            npass=npass,
            seed=seed,
        )


def test_replay_weighted_partitions_is_exact_and_preserves_medoid_position_levels():
    positions = np.asarray([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
    dissimilarity = np.abs(positions[:, None] - positions[None, :])
    weights = b2.draw_bayesian_bootstrap_weights(n_draws=4, n_observations=6, seed=9123)
    starts = b2.build_replay_start_schedule(n_observations=6, k=2, npass=4, seed=451)

    first = b2.replay_weighted_partitions(
        dissimilarity=dissimilarity,
        k=2,
        weights=weights,
        replay_starts=starts,
    )
    second = b2.replay_weighted_partitions(
        dissimilarity=dissimilarity,
        k=2,
        weights=weights,
        replay_starts=starts,
    )

    np.testing.assert_array_equal(first.partitions, second.partitions)
    np.testing.assert_array_equal(first.raw_medoid_levels, second.raw_medoid_levels)
    assert first.partitions.shape == (4, 6)
    assert first.raw_medoid_levels.shape == (4, 2)
    for partition, levels in zip(
        first.partitions,
        first.raw_medoid_levels,
        strict=True,
    ):
        assert np.unique(partition).size == 2
        assert np.all((1 <= partition) & (partition <= 6))
        assert set(levels.tolist()) == set(np.unique(partition).tolist())


def test_positive_npass_replay_ignores_row_zero_but_retains_build_start():
    positions = np.asarray([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
    dissimilarity = np.abs(positions[:, None] - positions[None, :])
    weights = np.ones((1, 6), dtype=float)
    starts = np.asarray([[0, 1], [0, 3], [2, 5]], dtype=int)
    changed_unused_row = starts.copy()
    changed_unused_row[0] = [4, 5]

    first = b2.replay_weighted_partitions(
        dissimilarity=dissimilarity,
        k=2,
        weights=weights,
        replay_starts=starts,
    )
    changed = b2.replay_weighted_partitions(
        dissimilarity=dissimilarity,
        k=2,
        weights=weights,
        replay_starts=changed_unused_row,
    )

    np.testing.assert_array_equal(first.partitions, changed.partitions)


def test_replay_boundary_passes_only_partition_inputs_to_public_kmedoids(monkeypatch):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def spy(*args: object, **kwargs: object) -> np.ndarray:
        calls.append((args, kwargs))
        return np.asarray([1, 1, 4, 4])

    monkeypatch.setattr(b2, "wcKMedoids", spy)
    result = b2.replay_weighted_partitions(
        dissimilarity=_distance_fixture(),
        k=2,
        weights=np.ones((1, 4)),
        replay_starts=np.asarray([[0, 2], [1, 3]], dtype=int),
    )

    assert result.partitions.tolist() == [[1, 1, 4, 4]]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 2
    np.testing.assert_array_equal(args[0], _distance_fixture())
    assert args[1] == 2
    assert set(kwargs) == {
        "weights",
        "npass",
        "replay_starts",
        "method",
        "cluster_only",
        "random_state",
    }
    assert kwargs["method"] == "PAMonce"
    assert kwargs["cluster_only"] is True
    assert kwargs["random_state"] == 0
    assert not any(
        "reference" in name or "predict" in name or "covariate" in name for name in kwargs
    )


@pytest.mark.parametrize(
    "bad_weights",
    [
        np.ones(4),
        np.ones((0, 4)),
        np.ones((2, 3)),
        np.asarray([[1.0, 1.0, 1.0, 0.0]]),
        np.asarray([[1.0, 1.0, 1.0, np.inf]]),
        np.asarray([[2.0, 2.0, 2.0, 2.0]]),
    ],
)
def test_replay_weighted_partitions_rejects_invalid_weights(bad_weights: np.ndarray):
    with pytest.raises(ValueError, match="weights"):
        b2.replay_weighted_partitions(
            dissimilarity=_distance_fixture(),
            k=2,
            weights=bad_weights,
            replay_starts=np.asarray([[0, 2], [1, 3]], dtype=int),
        )


@pytest.mark.parametrize(
    "bad_starts",
    [
        np.asarray([0, 2]),
        np.asarray([], dtype=int).reshape(0, 2),
        np.asarray([[0, 2, 3]], dtype=int),
        np.asarray([[0.0, 2.0]]),
        np.asarray([[0, 0]], dtype=int),
        np.asarray([[0, 4]], dtype=int),
    ],
)
def test_replay_weighted_partitions_rejects_invalid_start_schedule(bad_starts: np.ndarray):
    with pytest.raises(ValueError, match="replay_starts"):
        b2.replay_weighted_partitions(
            dissimilarity=_distance_fixture(),
            k=2,
            weights=np.ones((1, 4)),
            replay_starts=bad_starts,
        )


@pytest.mark.parametrize(
    "malformed_partition",
    [
        np.asarray([1, 1, 2]),
        np.asarray([1, 1, 1, 1]),
        np.asarray([1.0, 1.0, 2.5, 2.5]),
        np.asarray([1, 1, 5, 5]),
        np.asarray([1.0, 1.0, np.inf, np.inf]),
    ],
)
def test_replay_weighted_partitions_rejects_malformed_draw_with_index(
    monkeypatch,
    malformed_partition: np.ndarray,
):
    monkeypatch.setattr(b2, "wcKMedoids", lambda *args, **kwargs: malformed_partition)

    with pytest.raises(RuntimeError, match="draw 0"):
        b2.replay_weighted_partitions(
            dissimilarity=_distance_fixture(),
            k=2,
            weights=np.ones((1, 4)),
            replay_starts=np.asarray([[0, 2], [1, 3]], dtype=int),
        )


def test_weighted_alignment_is_label_invariant_and_differs_from_unweighted_sensitivity():
    raw = np.asarray([5, 6, 5, 6])
    permuted_raw_labels = np.asarray([99, 42, 99, 42])
    reference = np.asarray(["left", "left", "right", "right"], dtype=object)
    weights = np.asarray([1.0, 10.0, 10.0, 1.0])

    aligned = b2.align_partition(raw, reference, weights=weights)
    permuted = b2.align_partition(permuted_raw_labels, reference, weights=weights)

    np.testing.assert_array_equal(aligned.aligned, permuted.aligned)
    np.testing.assert_array_equal(_co_clustering(aligned.aligned), _co_clustering(raw))
    assert aligned.raw_levels == (5, 6)
    assert permuted.raw_levels == (99, 42)
    assert aligned.weighted_assignment == (1, 0)
    assert aligned.unweighted_assignment == (0, 1)
    assert aligned.weighted_mapping == ((5, "right"), (6, "left"))
    assert aligned.weighted_vs_unweighted_different is True
    assert aligned.weighted_ambiguous is False
    assert aligned.unweighted_ambiguous is True
    assert aligned.weighted_runner_up_margin > 0.0


@pytest.mark.parametrize(
    "overlap",
    [
        np.asarray([[3.0, 1.0], [1.0, 3.0]]),
        np.ones((2, 2), dtype=float),
        np.asarray([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 2.0]]),
        np.asarray(
            [
                [8.0, 2.0, 1.0, 0.0],
                [1.0, 7.0, 2.0, 0.0],
                [0.0, 1.0, 6.0, 3.0],
                [3.0, 0.0, 1.0, 5.0],
            ]
        ),
    ],
)
def test_scalable_assignment_matches_independent_exhaustive_oracle(overlap: np.ndarray):
    expected = _exhaustive_assignment(overlap)
    actual = b2.solve_overlap_assignment(overlap)

    assert actual.assignment == expected["assignment"]
    assert actual.best_score == pytest.approx(expected["best_score"])
    assert actual.runner_up_score == pytest.approx(expected["runner_up_score"])
    assert actual.runner_up_margin == pytest.approx(
        float(expected["best_score"]) - float(expected["runner_up_score"])
    )
    assert actual.ambiguous == bool(expected["ambiguous"])


def test_assignment_tolerance_selects_lexicographically_smallest_near_tied_optimum():
    overlap = np.asarray([[1.0, 1.0 + 2e-13], [1.0, 1.0]])

    solution = b2.solve_overlap_assignment(overlap)

    assert solution.assignment == (0, 1)
    assert solution.ambiguous is True
    assert solution.best_score - solution.selected_score == pytest.approx(2e-13, abs=1e-15)
    assert solution.runner_up_margin == 0.0


def test_assignment_solver_scales_beyond_exhaustive_test_envelope():
    overlap = np.random.default_rng(22).uniform(size=(12, 12))

    solution = b2.solve_overlap_assignment(overlap)

    assert sorted(solution.assignment) == list(range(12))
    assert np.isfinite(solution.best_score)
    assert solution.runner_up_score <= solution.best_score + 1e-12


def test_reference_label_renaming_preserves_aligned_co_clustering():
    raw = np.asarray([4, 4, 2, 2])
    named = b2.align_partition(
        raw,
        np.asarray(["A", "A", "B", "B"], dtype=object),
        weights=np.ones(4),
    )
    renamed = b2.align_partition(
        raw,
        np.asarray(["B", "B", "A", "A"], dtype=object),
        weights=np.ones(4),
    )

    np.testing.assert_array_equal(
        _co_clustering(named.aligned),
        _co_clustering(renamed.aligned),
    )
    assert named.reference_levels == ("A", "B")
    assert renamed.reference_levels == ("B", "A")


@pytest.mark.parametrize(
    ("raw", "reference", "weights"),
    [
        ([1, 1, 1, 1], [0, 0, 1, 1], np.ones(4)),
        ([1, 2, 3, 3], [0, 0, 1, 1], np.ones(4)),
        ([1, 1, None, 2], [0, 0, 1, 1], np.ones(4)),
        ([1, 1, 2], [0, 0, 1, 1], np.ones(4)),
        ([1, 1, 2, 2], [0, 0, 1, 1], np.asarray([1.0, 1.0, 0.0, 2.0])),
    ],
)
def test_alignment_rejects_unalignable_or_invalid_partitions(
    raw: object,
    reference: object,
    weights: np.ndarray,
):
    with pytest.raises((TypeError, ValueError), match="raw|reference|weight|cluster|K|shape"):
        b2.align_partition(raw, reference, weights=weights)


def test_reference_choice_can_change_exact_alignment_compression_not_raw_partition_information():
    raw = np.asarray([5, 6, 5, 6])
    weights = (
        np.asarray([10.0, 1.0, 1.0, 10.0]),
        np.asarray([1.0, 10.0, 10.0, 1.0]),
    )
    crossing_reference = np.asarray(["A", "A", "B", "B"], dtype=object)
    matching_reference = np.asarray(["A", "B", "A", "B"], dtype=object)

    crossing_aligned = np.asarray(
        [
            b2.align_partition(raw, crossing_reference, weights=draw_weights).aligned
            for draw_weights in weights
        ],
        dtype=object,
    )
    matching_aligned = np.asarray(
        [
            b2.align_partition(raw, matching_reference, weights=draw_weights).aligned
            for draw_weights in weights
        ],
        dtype=object,
    )

    assert len({tuple(row) for row in crossing_aligned}) == 2
    assert len({tuple(row) for row in matching_aligned}) == 1
    np.testing.assert_array_equal(_co_clustering(crossing_aligned[0]), _co_clustering(raw))
    np.testing.assert_array_equal(_co_clustering(matching_aligned[1]), _co_clustering(raw))


def test_partition_compression_is_stable_lossless_and_preserves_rare_mass():
    aligned = np.asarray(
        [
            ["A", "A", "B", "B"],
            ["A", "A", "B", "B"],
            ["A", "B", "A", "B"],
            ["A", "A", "B", "B"],
        ],
        dtype=object,
    )

    compressed = b2.compress_aligned_partitions(aligned)

    assert compressed.distinct_partitions.tolist() == [
        ["A", "A", "B", "B"],
        ["A", "B", "A", "B"],
    ]
    assert compressed.counts.tolist() == [3, 1]
    np.testing.assert_array_equal(compressed.inverse_index, [0, 0, 1, 0])
    np.testing.assert_array_equal(
        compressed.distinct_partitions[compressed.inverse_index],
        aligned,
    )
    np.testing.assert_allclose(compressed.masses, [0.75, 0.25])
    np.testing.assert_allclose(
        compressed.mass_mcse,
        np.sqrt(np.asarray([0.75, 0.25]) * np.asarray([0.25, 0.75]) / 4.0),
    )
    assert compressed.counts.sum() == 4
    assert compressed.masses.sum() == pytest.approx(1.0, abs=1e-12, rel=0.0)


@pytest.mark.parametrize(
    "bad_partitions",
    [
        np.asarray(["A", "B"], dtype=object),
        np.empty((0, 2), dtype=object),
        np.empty((2, 0), dtype=object),
        np.asarray([["A", None], ["A", "B"]], dtype=object),
        np.asarray([["A", ["B"]], ["A", "B"]], dtype=object),
    ],
)
def test_partition_compression_rejects_malformed_draw_matrices(bad_partitions: np.ndarray):
    with pytest.raises((TypeError, ValueError), match="partition|label|shape|nonmissing"):
        b2.compress_aligned_partitions(bad_partitions)


def _uniform_diagnostics(count: int) -> tuple[b2.AlignmentResult, ...]:
    base = b2.align_partition([1, 1, 2, 2], ["A", "A", "B", "B"], weights=np.ones(4))
    return (base,) * count


def test_degenerate_partition_posterior_is_surfaced_as_a_warning():
    """Finding F-05: a single-partition posterior carries no typology uncertainty.

    RARCAT exists to show how covariate conclusions change when the typology
    is uncertain, and that uncertainty is carried entirely by the
    between-component variance term.  When the posterior compresses to one
    distinct partition that term is exactly zero for every reported target,
    and nothing in the result said so.
    """
    identical = np.asarray([["A", "A", "B", "B"]] * 8, dtype=object)

    summary = b2.summarize_partition_posterior(
        identical,
        reference_cluster_order=("A", "B"),
        alignment_diagnostics=_uniform_diagnostics(8),
    )

    np.testing.assert_array_equal(summary.compression.masses, [1.0])
    assert len(summary.warnings) == 1
    text = summary.warnings[0]
    assert "single distinct partition" in text
    assert "typology" in text
    assert "zero" in text


def test_near_degenerate_partition_posterior_is_surfaced_as_a_warning():
    """Finding F-05: a dominated posterior is degenerate for practical purposes.

    A largest mass at or above the declared threshold leaves the remaining
    partitions with too little weight for the between-component term to be a
    material share of the total, which the audit measured at 0.06%-3.1%.
    """
    rows = [["A", "A", "B", "B"]] * 39 + [["A", "B", "A", "B"]]
    dominated = np.asarray(rows, dtype=object)

    summary = b2.summarize_partition_posterior(
        dominated,
        reference_cluster_order=("A", "B"),
        alignment_diagnostics=_uniform_diagnostics(40),
    )

    assert summary.compression.masses.tolist() == [0.975, 0.025]
    assert b2.NEGLIGIBLE_TYPOLOGY_MASS == 0.95
    assert len(summary.warnings) == 1
    text = summary.warnings[0]
    assert "0.975" in text
    assert "0.95" in text
    assert "negligible" in text


def test_a_genuinely_uncertain_partition_posterior_warns_about_neither():
    """The two typology warnings must not fire on ordinary competing support."""
    rows = [["A", "A", "B", "B"]] * 6 + [["A", "B", "A", "B"]] * 4
    competing = np.asarray(rows, dtype=object)

    summary = b2.summarize_partition_posterior(
        competing,
        reference_cluster_order=("A", "B"),
        alignment_diagnostics=_uniform_diagnostics(10),
    )

    assert summary.compression.masses.tolist() == [0.6, 0.4]
    assert summary.warnings == ()


def test_partition_summary_computes_membership_entropy_coclustering_and_diagnostics():
    aligned = np.asarray(
        [
            ["A", "A", "B", "B"],
            ["A", "B", "A", "B"],
            ["A", "A", "B", "B"],
            ["A", "A", "B", "B"],
        ],
        dtype=object,
    )
    base = b2.align_partition([1, 1, 2, 2], ["A", "A", "B", "B"], weights=np.ones(4))
    diagnostics = (
        replace(base, weighted_ambiguous=True),
        replace(base, weighted_vs_unweighted_different=True),
        base,
        base,
    )

    summary = b2.summarize_partition_posterior(
        aligned,
        reference_cluster_order=("A", "B"),
        alignment_diagnostics=diagnostics,
    )

    np.testing.assert_allclose(
        summary.membership_probabilities,
        [[1.0, 0.0], [0.75, 0.25], [0.25, 0.75], [0.0, 1.0]],
    )
    np.testing.assert_allclose(summary.membership_probabilities.sum(axis=1), 1.0)
    known_entropy = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    np.testing.assert_allclose(summary.entropy, [0.0, known_entropy, known_entropy, 0.0])
    assert np.all((0.0 <= summary.entropy) & (summary.entropy <= np.log(2.0)))
    np.testing.assert_allclose(summary.co_clustering, summary.co_clustering.T)
    np.testing.assert_allclose(np.diag(summary.co_clustering), 1.0)
    assert summary.co_clustering[0, 1] == pytest.approx(0.75, abs=1e-12, rel=0.0)
    assert np.all((0.0 <= summary.co_clustering) & (summary.co_clustering <= 1.0))
    assert summary.ambiguous_draw_count == 1
    assert summary.ambiguous_draw_rate == pytest.approx(0.25, abs=1e-12, rel=0.0)
    assert summary.weighted_unweighted_difference_count == 1
    assert summary.weighted_unweighted_difference_rate == pytest.approx(0.25, abs=1e-12, rel=0.0)
    assert summary.ambiguity_flags.tolist() == [True, False, False, False]
    assert summary.weighted_unweighted_difference_flags.tolist() == [False, True, False, False]
    assert len(summary.warnings) == 2


def _build_fixture_result(
    *,
    reference_labels: object,
    B: int = 4,
    npass: int = 3,
    root_seed: int = 222,
):
    positions = np.asarray([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
    return b2.build_partition_posterior(
        dissimilarity=np.abs(positions[:, None] - positions[None, :]),
        reference_labels=reference_labels,
        observation_ids=[f"id-{index}" for index in range(6)],
        reference_observation_ids=[f"id-{index}" for index in range(6)],
        B=B,
        npass=npass,
        root_seed=root_seed,
    )


def test_partition_posterior_result_is_complete_replayable_and_shape_documented():
    reference = np.asarray(["A", "A", "A", "B", "B", "B"], dtype=object)

    first = _build_fixture_result(reference_labels=reference)
    repeated = _build_fixture_result(reference_labels=reference)

    assert (first.n, first.B, first.K, first.npass) == (6, 4, 2, 3)
    assert first.root_seed == 222
    assert first.reference_cluster_order == ("A", "B")
    assert first.observation_ids.tolist() == [f"id-{index}" for index in range(6)]
    assert first.weights.shape == (4, 6)
    assert first.replay_starts.shape == (3, 2)
    assert first.raw_medoid_levels.shape == (4, 2)
    assert first.raw_partitions.shape == (4, 6)
    assert first.aligned_partitions.shape == (4, 6)
    assert first.raw_partitions.dtype == np.int64
    assert first.aligned_partitions.dtype == object
    assert len(first.alignment_diagnostics) == 4
    assert first.membership_probabilities.shape == (6, 2)
    assert first.entropy.shape == (6,)
    assert first.co_clustering.shape == (6, 6)
    assert first.counts.sum() == first.B
    assert first.masses.sum() == pytest.approx(1.0, abs=1e-12, rel=0.0)
    np.testing.assert_array_equal(
        first.distinct_partitions[first.inverse_index], first.aligned_partitions
    )
    np.testing.assert_array_equal(first.weights, repeated.weights)
    np.testing.assert_array_equal(first.replay_starts, repeated.replay_starts)
    np.testing.assert_array_equal(first.raw_partitions, repeated.raw_partitions)
    np.testing.assert_array_equal(first.aligned_partitions, repeated.aligned_partitions)
    np.testing.assert_array_equal(first.membership_probabilities, repeated.membership_probabilities)
    np.testing.assert_array_equal(first.entropy, repeated.entropy)
    np.testing.assert_array_equal(first.co_clustering, repeated.co_clustering)
    assert first.seed_streams == repeated.seed_streams
    assert first.warnings == repeated.warnings


def test_partition_orchestration_has_isolated_streams_and_reference_only_affects_alignment():
    matching = _build_fixture_result(
        reference_labels=["A", "A", "A", "B", "B", "B"],
        B=3,
        npass=2,
    )
    changed_npass = _build_fixture_result(
        reference_labels=["A", "A", "A", "B", "B", "B"],
        B=3,
        npass=5,
    )
    changed_B = _build_fixture_result(
        reference_labels=["A", "A", "A", "B", "B", "B"],
        B=6,
        npass=2,
    )
    changed_reference = _build_fixture_result(
        reference_labels=[100, 200, 100, 200, 100, 200],
        B=3,
        npass=2,
    )

    np.testing.assert_array_equal(matching.weights, changed_npass.weights)
    np.testing.assert_array_equal(matching.replay_starts, changed_B.replay_starts)
    np.testing.assert_array_equal(matching.weights, changed_reference.weights)
    np.testing.assert_array_equal(matching.replay_starts, changed_reference.replay_starts)
    np.testing.assert_array_equal(matching.raw_partitions, changed_reference.raw_partitions)
    for original, changed in zip(
        matching.raw_partitions,
        changed_reference.raw_partitions,
        strict=True,
    ):
        np.testing.assert_array_equal(_co_clustering(original), _co_clustering(changed))


def test_partition_orchestration_forbids_predictor_and_user_weight_boundaries():
    parameters = inspect.signature(b2.build_partition_posterior).parameters
    forbidden_fragments = ("predict", "design", "formula", "regression", "coefficient", "pymc")
    assert not any(
        fragment in name.casefold() for name in parameters for fragment in forbidden_fragments
    )
    kwargs = {
        "dissimilarity": _distance_fixture(),
        "reference_labels": [0, 0, 1, 1],
        "observation_ids": ["a", "b", "c", "d"],
        "reference_observation_ids": ["a", "b", "c", "d"],
        "B": 2,
        "npass": 2,
        "root_seed": 1,
    }
    with pytest.raises(TypeError):
        b2.build_partition_posterior(**kwargs, predictors=np.ones((4, 1)))
    with pytest.raises(TypeError):
        b2.build_partition_posterior(**kwargs, weights=np.ones(4))

    assert "rarcat" not in weightedcluster.__all__
    assert "bayes_rarcat" in weightedcluster.__all__
    assert "BayesRARCATResult" in weightedcluster.__all__
    assert not hasattr(weightedcluster, "rarcat")
    assert hasattr(weightedcluster, "bayes_rarcat")


def test_module_documents_reference_choice_sensitivity_without_changing_raw_posterior_claim():
    documentation = (b2.__doc__ or "").casefold()
    assert "reference-choice sensitivity" in documentation
    assert "raw partition posterior" in documentation


def test_partition_seed_gates_reject_integral_floats_like_the_conditional_gate():
    """2026-09-02 audit WC-090212: ``partition_root_seed`` accepted ``712.0``
    while ``conditional_root_seed`` rejected it through ``operator.index``.
    The two public seed gates of ``bayes_rarcat`` must accept the same
    numeric family; the strict gate wins. ``B``/``npass`` keep their pinned
    permissive integral-numeric contract.
    """
    with pytest.raises(TypeError):
        b2.validate_partition_inputs(
            dissimilarity=_distance_fixture(),
            reference_labels=np.asarray(["later", "first", "later", "first"], dtype=object),
            observation_ids=["a", "b", "c", "d"],
            reference_observation_ids=["a", "b", "c", "d"],
            B=2,
            npass=1,
            root_seed=712.0,
        )
    with pytest.raises(TypeError):
        b2.derive_seed_streams(712.0)
    with pytest.raises(TypeError):
        b2.draw_bayesian_bootstrap_weights(n_draws=2, n_observations=3, seed=7.0)
    assert b2.derive_seed_streams(np.int64(712)).root_seed == 712

"""Private predictor-independent partition posterior for Bayesian RARCAT B2.

This module owns only ``q(z | D, r)`` from the modular Bayesian RARCAT design.
Predictors, formulas, regression state, and conditional posterior objects are
deliberately absent from every boundary in this module.

Exact aligned compression has reference-choice sensitivity: changing the
fixed reference partition can change label-oriented aligned rows even when the
raw partition posterior and every label-invariant co-clustering quantity stay
unchanged. The reference is therefore fixed before inspecting B2 results.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ._validation import validate_distance_matrix, validate_integer_control
from .kmedoids import wcKMedoids

_FLOAT_ATOL = 1e-12
_FLOAT_RTOL = 1e-12

# Largest partition mass at or above which the typology-uncertainty term is
# reported as negligible rather than absent.  This is a communication
# threshold, not a gate: it changes no reported number, no status and no
# diagnostic verdict.  0.95 is declared because the remaining partitions then
# carry at most 5% of the posterior weight, and the 2026-08-25 audit measured
# between-component shares of 0.06%-3.1% of total variance at largest masses
# of 0.9375-0.9844 — that is, negligible throughout the range above it.
NEGLIGIBLE_TYPOLOGY_MASS = 0.95


@dataclass(frozen=True)
class SeedStreamIdentifier:
    """Complete replay identifier for one named ``SeedSequence`` child."""

    name: str
    entropy: int
    spawn_key: tuple[int, ...]
    pool_size: int

    def seed_sequence(self) -> np.random.SeedSequence:
        """Recreate the named child stream without mutating stored provenance."""
        return np.random.SeedSequence(
            entropy=self.entropy,
            spawn_key=self.spawn_key,
            pool_size=self.pool_size,
        )


@dataclass(frozen=True)
class SeedStreams:
    """Named, independently replayable B2 random streams."""

    root_seed: int
    weight_stream: SeedStreamIdentifier
    start_stream: SeedStreamIdentifier

    def weight_seed_sequence(self) -> np.random.SeedSequence:
        return self.weight_stream.seed_sequence()

    def start_seed_sequence(self) -> np.random.SeedSequence:
        return self.start_stream.seed_sequence()


@dataclass(frozen=True)
class ValidatedPartitionInputs:
    """Validated B2 orchestration inputs.

    Arrays use object dtype and shape ``(n,)`` so observation and reference
    label identities are retained without string coercion.
    """

    dissimilarity: np.ndarray
    reference_labels: np.ndarray
    observation_ids: np.ndarray
    reference_observation_ids: np.ndarray
    reference_cluster_order: tuple[object, ...]
    n: int
    B: int
    npass: int
    root_seed: int
    K: int


@dataclass(frozen=True)
class RawPartitionReplay:
    """Raw weighted-k-medoids output.

    ``partitions`` has ``int64`` dtype and shape ``(B, n)``. The corresponding
    one-based medoid-position labels are stored in canonical membership order
    in ``raw_medoid_levels`` with ``int64`` dtype and shape ``(B, K)``.
    """

    partitions: np.ndarray
    raw_medoid_levels: np.ndarray


@dataclass(frozen=True)
class AssignmentSolution:
    """Deterministic maximum-overlap assignment diagnostics."""

    assignment: tuple[int, ...]
    selected_score: float
    best_score: float
    runner_up_score: float
    runner_up_margin: float
    ambiguous: bool


@dataclass(frozen=True)
class AlignmentResult:
    """Weighted reference alignment and its unweighted sensitivity result.

    ``aligned`` has object dtype and shape ``(n,)``. Overlap matrices have
    shape ``(K, K)`` in canonical raw-membership row order and stable
    first-observed reference-level column order.
    """

    aligned: np.ndarray
    raw_levels: tuple[object, ...]
    reference_levels: tuple[object, ...]
    weighted_mapping: tuple[tuple[object, object], ...]
    unweighted_mapping: tuple[tuple[object, object], ...]
    weighted_overlap: np.ndarray
    unweighted_overlap: np.ndarray
    weighted_assignment: tuple[int, ...]
    unweighted_assignment: tuple[int, ...]
    weighted_best_score: float
    weighted_runner_up_score: float
    weighted_runner_up_margin: float
    weighted_ambiguous: bool
    unweighted_best_score: float
    unweighted_runner_up_score: float
    unweighted_runner_up_margin: float
    unweighted_ambiguous: bool
    weighted_vs_unweighted_different: bool


@dataclass(frozen=True)
class PartitionCompression:
    """Lossless stable compression of aligned draws.

    ``distinct_partitions`` is object ``(U, n)``; ``counts`` and
    ``inverse_index`` are ``int64`` with shapes ``(U,)`` and ``(B,)``;
    ``masses`` and ``mass_mcse`` are ``float64 (U,)``.
    """

    distinct_partitions: np.ndarray
    counts: np.ndarray
    masses: np.ndarray
    inverse_index: np.ndarray
    mass_mcse: np.ndarray


@dataclass(frozen=True)
class PartitionPosteriorSummary:
    """Derived empirical summaries of aligned partition draws."""

    compression: PartitionCompression
    membership_probabilities: np.ndarray
    entropy: np.ndarray
    co_clustering: np.ndarray
    ambiguity_flags: np.ndarray
    ambiguous_draw_count: int
    ambiguous_draw_rate: float
    weighted_unweighted_difference_flags: np.ndarray
    weighted_unweighted_difference_count: int
    weighted_unweighted_difference_rate: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PartitionPosteriorResult:
    """Complete private B2 result with lossless replay provenance.

    Object identity arrays use object dtype. Weights are ``float64 (B, n)``;
    replay starts and raw medoid levels are ``int64 (npass, K)`` and
    ``int64 (B, K)``; raw and aligned partitions are ``int64 (B, n)`` and
    object ``(B, n)``. Membership, entropy, and co-clustering are float64 with
    shapes ``(n, K)``, ``(n,)``, and ``(n, n)``.
    """

    n: int
    B: int
    K: int
    npass: int
    observation_ids: np.ndarray
    reference_labels: np.ndarray
    reference_cluster_order: tuple[object, ...]
    root_seed: int
    seed_streams: SeedStreams
    weights: np.ndarray
    replay_starts: np.ndarray
    raw_medoid_levels: np.ndarray
    raw_partitions: np.ndarray
    aligned_partitions: np.ndarray
    alignment_diagnostics: tuple[AlignmentResult, ...]
    distinct_partitions: np.ndarray
    counts: np.ndarray
    masses: np.ndarray
    inverse_index: np.ndarray
    mass_mcse: np.ndarray
    membership_probabilities: np.ndarray
    entropy: np.ndarray
    co_clustering: np.ndarray
    ambiguity_flags: np.ndarray
    ambiguous_draw_count: int
    ambiguous_draw_rate: float
    weighted_unweighted_difference_flags: np.ndarray
    weighted_unweighted_difference_count: int
    weighted_unweighted_difference_rate: float
    warnings: tuple[str, ...]


def _validated_integer(value: object, *, name: str, minimum: int) -> int:
    integer = validate_integer_control(value, name=name)
    if integer < minimum:
        qualifier = "nonnegative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return integer


def _exact_seed_integer(value: object, *, name: str) -> int:
    """Require an exact nonnegative integer seed, as the B3 seed gate does.

    2026-09-02 audit, finding WC-090212: ``partition_root_seed`` accepted
    integral floats through :func:`validate_integer_control` while
    ``conditional_root_seed`` rejected them through ``operator.index``. The
    two public seed gates now share the strict contract; ``B`` and ``npass``
    deliberately keep the permissive integral-numeric contract their tests
    pin.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer and cannot be boolean")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an exact integer") from error
    integer = int(integer)
    if integer < 0:
        raise ValueError(f"{name} must be at least 0")
    return integer


def _is_scalar_hashable(value: object) -> bool:
    if np.ndim(value) != 0:
        return False
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_missing(value: object) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else True


def _validated_identity_vector(values: Sequence[object], *, name: str, n: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} identifiers must be one-dimensional")
    if array.shape[0] != n:
        raise ValueError(f"{name} identifiers must have length {n}")
    seen: set[object] = set()
    for value in array.tolist():
        if not _is_scalar_hashable(value):
            raise TypeError(f"{name} identifiers must contain scalar hashable values")
        if _is_missing(value):
            raise ValueError(f"{name} identifiers must be complete")
        if value in seen:
            raise ValueError(f"{name} identifiers must be unique")
        seen.add(value)
    return array


def _validated_reference_labels(
    values: Sequence[object], *, n: int
) -> tuple[np.ndarray, tuple[object, ...]]:
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1:
        raise ValueError("reference labels must be one-dimensional")
    if labels.shape[0] != n:
        raise ValueError(f"reference labels must have length {n}")
    order: list[object] = []
    seen: set[object] = set()
    for value in labels.tolist():
        if not _is_scalar_hashable(value):
            raise TypeError("reference labels must contain scalar hashable values")
        if _is_missing(value):
            raise ValueError("reference labels must be complete and nonmissing")
        if value not in seen:
            seen.add(value)
            order.append(value)
    return labels, tuple(order)


def validate_partition_inputs(
    *,
    dissimilarity: np.ndarray,
    reference_labels: Sequence[object],
    observation_ids: Sequence[object],
    reference_observation_ids: Sequence[object],
    B: object,
    npass: object,
    root_seed: object,
) -> ValidatedPartitionInputs:
    """Validate B2 inputs before any random stream is constructed."""
    validate_distance_matrix(dissimilarity, name="dissimilarity", strict=True)
    n = int(dissimilarity.shape[0])
    ids = _validated_identity_vector(observation_ids, name="dissimilarity", n=n)
    reference_ids = _validated_identity_vector(
        reference_observation_ids,
        name="reference",
        n=n,
    )
    if not all(left == right for left, right in zip(ids, reference_ids, strict=True)):
        raise ValueError(
            "dissimilarity and reference identifiers must be elementwise identical in order"
        )
    labels, reference_order = _validated_reference_labels(reference_labels, n=n)
    K = len(reference_order)
    if not 2 <= K <= n:
        raise ValueError(f"reference labels imply invalid K={K}; expected 2 <= K <= n")

    B_int = _validated_integer(B, name="B", minimum=1)
    npass_int = _validated_integer(npass, name="npass", minimum=1)
    root_seed_int = _exact_seed_integer(root_seed, name="root_seed")
    return ValidatedPartitionInputs(
        dissimilarity=dissimilarity,
        reference_labels=labels,
        observation_ids=ids,
        reference_observation_ids=reference_ids,
        reference_cluster_order=reference_order,
        n=n,
        B=B_int,
        npass=npass_int,
        root_seed=root_seed_int,
        K=K,
    )


def _stream_identifier(name: str, seed: np.random.SeedSequence) -> SeedStreamIdentifier:
    return SeedStreamIdentifier(
        name=name,
        entropy=int(seed.entropy),
        spawn_key=tuple(int(value) for value in seed.spawn_key),
        pool_size=int(seed.pool_size),
    )


def derive_seed_streams(root_seed: object) -> SeedStreams:
    """Derive the named weight and start streams from one nonnegative seed."""
    seed = _exact_seed_integer(root_seed, name="root_seed")
    weight_seed, start_seed = np.random.SeedSequence(seed).spawn(2)
    return SeedStreams(
        root_seed=seed,
        weight_stream=_stream_identifier("bayesian_bootstrap_weights", weight_seed),
        start_stream=_stream_identifier("kmedoids_start_schedule", start_seed),
    )


def _coerce_seed_sequence(seed: object) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    integer = _exact_seed_integer(seed, name="seed")
    return np.random.SeedSequence(integer)


def draw_bayesian_bootstrap_weights(
    *,
    n_draws: object,
    n_observations: object,
    seed: object,
) -> np.ndarray:
    """Draw deterministic ``float64`` Bayesian-bootstrap weights of shape ``(B, n)``.

    Every row is finite, strictly positive, and normalized to ``n`` within
    ``atol=rtol=1e-12``.
    """
    B = _validated_integer(n_draws, name="n_draws", minimum=1)
    n = _validated_integer(n_observations, name="n_observations", minimum=1)
    rng = np.random.default_rng(_coerce_seed_sequence(seed))
    weights = np.asarray(rng.dirichlet(np.ones(n, dtype=float), size=B), dtype=np.float64)
    weights *= float(n) / weights.sum(axis=1, keepdims=True)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("Bayesian bootstrap produced nonpositive or nonfinite weights")
    if not np.allclose(
        weights.sum(axis=1),
        float(n),
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    ):
        raise RuntimeError("Bayesian bootstrap weights did not normalize to n")
    return weights


def build_replay_start_schedule(
    *,
    n_observations: object,
    k: object,
    npass: object,
    seed: object,
) -> np.ndarray:
    """Build a deterministic ``(npass, K)`` zero-based start matrix."""
    n = _validated_integer(n_observations, name="n_observations", minimum=1)
    K = _validated_integer(k, name="k", minimum=2)
    passes = _validated_integer(npass, name="npass", minimum=1)
    if K > n:
        raise ValueError("k cannot exceed n_observations")
    rng = np.random.default_rng(_coerce_seed_sequence(seed))
    return np.asarray(
        [rng.choice(n, size=K, replace=False) for _ in range(passes)],
        dtype=np.int64,
    )


def _validated_weight_matrix(weights: np.ndarray, *, n: int) -> np.ndarray:
    try:
        matrix = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != n:
        raise ValueError(f"weights must have shape (B, {n}) with B positive")
    if not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    if not np.allclose(
        matrix.sum(axis=1),
        float(n),
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    ):
        raise ValueError("weights must sum to n within atol=rtol=1e-12")
    return matrix


def _validated_replay_starts(replay_starts: np.ndarray, *, n: int, K: int) -> np.ndarray:
    starts = np.asarray(replay_starts)
    if starts.ndim != 2 or starts.shape[0] < 1 or starts.shape[1] != K:
        raise ValueError(f"replay_starts must have shape (positive npass, {K})")
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("replay_starts must contain zero-based integer indices")
    starts = starts.astype(np.int64, copy=True)
    if np.any(starts < 0) or np.any(starts >= n):
        raise ValueError("replay_starts contains an out-of-range zero-based index")
    if any(np.unique(row).size != K for row in starts):
        raise ValueError("replay_starts rows must contain unique medoids")
    return starts


def _canonical_raw_levels(partition: np.ndarray) -> np.ndarray:
    levels = np.unique(partition)
    memberships = sorted(
        ((tuple(np.flatnonzero(partition == level).tolist()), int(level)) for level in levels),
        key=lambda item: item[0],
    )
    return np.asarray([level for _, level in memberships], dtype=np.int64)


def replay_weighted_partitions(
    *,
    dissimilarity: np.ndarray,
    k: object,
    weights: np.ndarray,
    replay_starts: np.ndarray,
) -> RawPartitionReplay:
    """Replay raw partitions without generating any random weights or starts."""
    validate_distance_matrix(dissimilarity, name="dissimilarity", strict=True)
    n = int(dissimilarity.shape[0])
    K = _validated_integer(k, name="k", minimum=2)
    if K > n:
        raise ValueError("k cannot exceed the number of observations")
    weight_matrix = _validated_weight_matrix(weights, n=n)
    starts = _validated_replay_starts(replay_starts, n=n, K=K)

    partitions: list[np.ndarray] = []
    raw_levels: list[np.ndarray] = []
    for draw_index, draw_weights in enumerate(weight_matrix):
        result = wcKMedoids(
            dissimilarity,
            K,
            weights=draw_weights,
            npass=starts.shape[0],
            replay_starts=starts,
            method="PAMonce",
            cluster_only=True,
            random_state=0,
        )
        candidate = np.asarray(result)
        if candidate.ndim != 1 or candidate.shape != (n,):
            raise RuntimeError(f"weighted k-medoids draw {draw_index} has shape {candidate.shape}")
        try:
            numeric = np.asarray(candidate, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"weighted k-medoids draw {draw_index} has nonnumeric labels"
            ) from exc
        if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
            raise RuntimeError(
                f"weighted k-medoids draw {draw_index} labels must be finite and integral"
            )
        partition = numeric.astype(np.int64)
        if np.any(partition < 1) or np.any(partition > n):
            raise RuntimeError(
                f"weighted k-medoids draw {draw_index} violates one-based medoid positions"
            )
        levels = _canonical_raw_levels(partition)
        if levels.size != K:
            raise RuntimeError(
                f"weighted k-medoids draw {draw_index} produced {levels.size} clusters; expected {K}"
            )
        partitions.append(partition)
        raw_levels.append(levels)

    return RawPartitionReplay(
        partitions=np.asarray(partitions, dtype=np.int64),
        raw_medoid_levels=np.asarray(raw_levels, dtype=np.int64),
    )


def _maximum_assignment_score(matrix: np.ndarray) -> float:
    if matrix.shape == (0, 0):
        return 0.0
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].sum())


def _score_with_forbidden_edge(
    matrix: np.ndarray,
    *,
    forbidden_row: int,
    forbidden_column: int,
) -> float | None:
    costs = -matrix.copy()
    costs[forbidden_row, forbidden_column] = np.inf
    try:
        rows, columns = linear_sum_assignment(costs)
    except ValueError:
        return None
    if np.isinf(costs[rows, columns]).any():
        return None
    return float(matrix[rows, columns].sum())


def solve_overlap_assignment(overlap: np.ndarray) -> AssignmentSolution:
    """Solve maximum overlap with scalable deterministic tie handling.

    The lexicographically smallest assignment within ``atol=rtol=1e-12`` of
    the Hungarian optimum is selected. The runner-up is the best feasible
    assignment distinct from that selected mapping.
    """
    try:
        matrix = np.asarray(overlap, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("overlap must be a numeric square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("overlap must be a square K x K matrix with K >= 2")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("overlap must contain finite nonnegative values")

    K = matrix.shape[0]
    best_score = _maximum_assignment_score(matrix)
    available_columns = list(range(K))
    chosen: list[int] = []
    prefix_score = 0.0
    for row in range(K):
        selected_column: int | None = None
        for column in available_columns:
            remaining_columns = [value for value in available_columns if value != column]
            residual = matrix[np.ix_(range(row + 1, K), remaining_columns)]
            candidate_score = prefix_score + float(matrix[row, column])
            candidate_score += _maximum_assignment_score(residual)
            if np.isclose(
                candidate_score,
                best_score,
                atol=_FLOAT_ATOL,
                rtol=_FLOAT_RTOL,
            ):
                selected_column = column
                break
        if selected_column is None:  # pragma: no cover - defensive numerical guard
            raise RuntimeError("could not reconstruct the deterministic optimal assignment")
        chosen.append(selected_column)
        available_columns.remove(selected_column)
        prefix_score += float(matrix[row, selected_column])

    assignment = tuple(chosen)
    selected_score = float(sum(matrix[row, column] for row, column in enumerate(assignment)))
    alternatives = [
        score
        for row, column in enumerate(assignment)
        if (
            score := _score_with_forbidden_edge(
                matrix,
                forbidden_row=row,
                forbidden_column=column,
            )
        )
        is not None
    ]
    if not alternatives:  # pragma: no cover - K >= 2 always has a derangement alternative
        raise RuntimeError("could not calculate a distinct runner-up assignment")
    runner_up_score = max(alternatives)
    margin = best_score - runner_up_score
    if margin < 0.0 and np.isclose(margin, 0.0, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL):
        margin = 0.0
    return AssignmentSolution(
        assignment=assignment,
        selected_score=selected_score,
        best_score=best_score,
        runner_up_score=runner_up_score,
        runner_up_margin=margin,
        ambiguous=bool(
            np.isclose(
                best_score,
                runner_up_score,
                atol=_FLOAT_ATOL,
                rtol=_FLOAT_RTOL,
            )
        ),
    )


def _validated_partition_vector(
    values: Sequence[object],
    *,
    name: str,
    expected_n: int | None = None,
) -> tuple[np.ndarray, tuple[object, ...]]:
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1:
        raise ValueError(f"{name} partition must be one-dimensional")
    if expected_n is not None and labels.shape[0] != expected_n:
        raise ValueError(f"{name} partition must have shape ({expected_n},)")
    order: list[object] = []
    seen: set[object] = set()
    for value in labels.tolist():
        if not _is_scalar_hashable(value):
            raise TypeError(f"{name} partition must contain scalar hashable labels")
        if _is_missing(value):
            raise ValueError(f"{name} partition must be complete and nonmissing")
        if value not in seen:
            seen.add(value)
            order.append(value)
    return labels, tuple(order)


def _canonical_membership_levels(
    labels: np.ndarray,
    levels: Sequence[object],
) -> tuple[object, ...]:
    memberships = sorted(
        ((tuple(np.flatnonzero(labels == level).tolist()), level) for level in levels),
        key=lambda item: item[0],
    )
    return tuple(level for _, level in memberships)


def align_partition(
    raw_partition: Sequence[object],
    reference_partition: Sequence[object],
    *,
    weights: np.ndarray,
) -> AlignmentResult:
    """Align one raw partition by weighted overlap and audit unweighted overlap."""
    raw, raw_first_seen = _validated_partition_vector(raw_partition, name="raw")
    reference, reference_levels = _validated_partition_vector(
        reference_partition,
        name="reference",
        expected_n=raw.shape[0],
    )
    K = len(reference_levels)
    if K < 2 or len(raw_first_seen) != K:
        raise ValueError("raw and reference partitions must each contain exactly the same K >= 2")
    raw_levels = _canonical_membership_levels(raw, raw_first_seen)

    alignment_weights = np.asarray(weights, dtype=np.float64)
    if alignment_weights.shape != raw.shape:
        raise ValueError(f"weights must have shape {raw.shape}")
    if not np.isfinite(alignment_weights).all() or np.any(alignment_weights <= 0.0):
        raise ValueError("weights must be finite and strictly positive")

    weighted_overlap = np.zeros((K, K), dtype=np.float64)
    unweighted_overlap = np.zeros((K, K), dtype=np.int64)
    for raw_index, raw_level in enumerate(raw_levels):
        raw_membership = raw == raw_level
        for reference_index, reference_level in enumerate(reference_levels):
            joint = raw_membership & (reference == reference_level)
            weighted_overlap[raw_index, reference_index] = float(alignment_weights[joint].sum())
            unweighted_overlap[raw_index, reference_index] = int(np.count_nonzero(joint))

    weighted = solve_overlap_assignment(weighted_overlap)
    unweighted = solve_overlap_assignment(unweighted_overlap)
    weighted_mapping = tuple(
        (raw_level, reference_levels[weighted.assignment[row]])
        for row, raw_level in enumerate(raw_levels)
    )
    unweighted_mapping = tuple(
        (raw_level, reference_levels[unweighted.assignment[row]])
        for row, raw_level in enumerate(raw_levels)
    )
    mapping = dict(weighted_mapping)
    aligned = np.asarray([mapping[value] for value in raw.tolist()], dtype=object)
    return AlignmentResult(
        aligned=aligned,
        raw_levels=raw_levels,
        reference_levels=reference_levels,
        weighted_mapping=weighted_mapping,
        unweighted_mapping=unweighted_mapping,
        weighted_overlap=weighted_overlap,
        unweighted_overlap=unweighted_overlap,
        weighted_assignment=weighted.assignment,
        unweighted_assignment=unweighted.assignment,
        weighted_best_score=weighted.best_score,
        weighted_runner_up_score=weighted.runner_up_score,
        weighted_runner_up_margin=weighted.runner_up_margin,
        weighted_ambiguous=weighted.ambiguous,
        unweighted_best_score=unweighted.best_score,
        unweighted_runner_up_score=unweighted.runner_up_score,
        unweighted_runner_up_margin=unweighted.runner_up_margin,
        unweighted_ambiguous=unweighted.ambiguous,
        weighted_vs_unweighted_different=(weighted.assignment != unweighted.assignment),
    )


def _readonly_array(values: object, *, dtype: object | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _validated_partition_matrix(partitions: np.ndarray) -> np.ndarray:
    matrix = np.asarray(partitions, dtype=object)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("partitions must have shape (positive B, positive n)")
    for value in matrix.ravel().tolist():
        if not _is_scalar_hashable(value):
            raise TypeError("partition labels must be scalar and hashable")
        if _is_missing(value):
            raise ValueError("partition labels must be complete and nonmissing")
    return matrix


def compress_aligned_partitions(partitions: np.ndarray) -> PartitionCompression:
    """Compress exact aligned rows in stable first-seen order without mass loss."""
    matrix = _validated_partition_matrix(partitions)
    positions: dict[tuple[object, ...], int] = {}
    distinct: list[np.ndarray] = []
    counts: list[int] = []
    inverse: list[int] = []
    for row in matrix:
        key = tuple(row.tolist())
        partition_index = positions.get(key)
        if partition_index is None:
            partition_index = len(distinct)
            positions[key] = partition_index
            distinct.append(row.copy())
            counts.append(0)
        counts[partition_index] += 1
        inverse.append(partition_index)

    B = matrix.shape[0]
    count_array = np.asarray(counts, dtype=np.int64)
    masses = count_array.astype(np.float64) / float(B)
    mass_mcse = np.sqrt(masses * (1.0 - masses) / float(B))
    if int(count_array.sum()) != B or not np.isclose(
        masses.sum(),
        1.0,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    ):
        raise RuntimeError("partition compression failed to preserve empirical mass")
    return PartitionCompression(
        distinct_partitions=_readonly_array(distinct, dtype=object),
        counts=_readonly_array(count_array, dtype=np.int64),
        masses=_readonly_array(masses, dtype=np.float64),
        inverse_index=_readonly_array(inverse, dtype=np.int64),
        mass_mcse=_readonly_array(mass_mcse, dtype=np.float64),
    )


def _validated_reference_order(values: Sequence[object]) -> tuple[object, ...]:
    array, levels = _validated_partition_vector(values, name="reference cluster order")
    if len(levels) < 2 or len(levels) != array.shape[0]:
        raise ValueError("reference cluster order must contain K >= 2 unique labels")
    return levels


def summarize_partition_posterior(
    aligned_partitions: np.ndarray,
    *,
    reference_cluster_order: Sequence[object],
    alignment_diagnostics: Sequence[AlignmentResult],
) -> PartitionPosteriorSummary:
    """Calculate compression, membership, entropy, co-clustering, and diagnostics."""
    matrix = _validated_partition_matrix(aligned_partitions)
    B, n = matrix.shape
    reference_order = _validated_reference_order(reference_cluster_order)
    K = len(reference_order)
    reference_set = set(reference_order)
    for draw_index, row in enumerate(matrix):
        row_levels = set(row.tolist())
        if len(row_levels) != K or row_levels != reference_set:
            raise ValueError(
                f"aligned partition draw {draw_index} must contain every reference cluster"
            )
    diagnostics = tuple(alignment_diagnostics)
    if len(diagnostics) != B or not all(isinstance(item, AlignmentResult) for item in diagnostics):
        raise ValueError("alignment_diagnostics must contain one AlignmentResult per draw")

    compression = compress_aligned_partitions(matrix)
    cluster_columns = {level: index for index, level in enumerate(reference_order)}
    membership_counts = np.zeros((n, K), dtype=np.int64)
    co_clustering_counts = np.zeros((n, n), dtype=np.int64)
    for row in matrix:
        for level in reference_order:
            members = np.flatnonzero(row == level)
            membership_counts[members, cluster_columns[level]] += 1
            co_clustering_counts[np.ix_(members, members)] += 1

    membership = membership_counts.astype(np.float64) / float(B)
    if not np.allclose(
        membership.sum(axis=1),
        1.0,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    ):
        raise RuntimeError("posterior membership rows did not sum to one")
    entropy_terms = np.zeros_like(membership)
    positive = membership > 0.0
    entropy_terms[positive] = membership[positive] * np.log(membership[positive])
    entropy = -entropy_terms.sum(axis=1)
    co_clustering = co_clustering_counts.astype(np.float64) / float(B)

    ambiguity_flags = np.asarray(
        [item.weighted_ambiguous for item in diagnostics],
        dtype=bool,
    )
    difference_flags = np.asarray(
        [item.weighted_vs_unweighted_different for item in diagnostics],
        dtype=bool,
    )
    ambiguous_count = int(ambiguity_flags.sum())
    difference_count = int(difference_flags.sum())
    warnings: list[str] = []
    largest_mass = float(np.max(compression.masses))
    if compression.masses.size == 1:
        warnings.append(
            "the partition posterior compressed to a single distinct partition: the "
            "between-component typology-uncertainty term is exactly zero for every "
            "target, and every reported interval is conditional on that one partition"
        )
    elif largest_mass >= NEGLIGIBLE_TYPOLOGY_MASS:
        warnings.append(
            f"one partition holds mass {largest_mass:.6g} of the partition posterior, at or "
            f"above the declared {NEGLIGIBLE_TYPOLOGY_MASS:g} threshold: the between-component "
            "typology-uncertainty term is negligible and every reported interval is close to "
            "conditional on that one partition"
        )
    if ambiguous_count:
        warnings.append(
            f"{ambiguous_count} of {B} weighted reference alignments were ambiguous within "
            "atol=rtol=1e-12 and resolved lexicographically"
        )
    if difference_count:
        warnings.append(
            f"{difference_count} of {B} weighted alignment mappings differed from the "
            "independently selected unweighted sensitivity mapping"
        )
    return PartitionPosteriorSummary(
        compression=compression,
        membership_probabilities=_readonly_array(membership, dtype=np.float64),
        entropy=_readonly_array(entropy, dtype=np.float64),
        co_clustering=_readonly_array(co_clustering, dtype=np.float64),
        ambiguity_flags=_readonly_array(ambiguity_flags, dtype=bool),
        ambiguous_draw_count=ambiguous_count,
        ambiguous_draw_rate=ambiguous_count / float(B),
        weighted_unweighted_difference_flags=_readonly_array(difference_flags, dtype=bool),
        weighted_unweighted_difference_count=difference_count,
        weighted_unweighted_difference_rate=difference_count / float(B),
        warnings=tuple(warnings),
    )


def build_partition_posterior(
    *,
    dissimilarity: np.ndarray,
    reference_labels: Sequence[object],
    observation_ids: Sequence[object],
    reference_observation_ids: Sequence[object],
    B: object,
    npass: object,
    root_seed: object,
) -> PartitionPosteriorResult:
    """Build the complete private predictor-independent B2 partition posterior."""
    inputs = validate_partition_inputs(
        dissimilarity=dissimilarity,
        reference_labels=reference_labels,
        observation_ids=observation_ids,
        reference_observation_ids=reference_observation_ids,
        B=B,
        npass=npass,
        root_seed=root_seed,
    )
    streams = derive_seed_streams(inputs.root_seed)
    weights = draw_bayesian_bootstrap_weights(
        n_draws=inputs.B,
        n_observations=inputs.n,
        seed=streams.weight_seed_sequence(),
    )
    replay_starts = build_replay_start_schedule(
        n_observations=inputs.n,
        k=inputs.K,
        npass=inputs.npass,
        seed=streams.start_seed_sequence(),
    )
    raw = replay_weighted_partitions(
        dissimilarity=inputs.dissimilarity,
        k=inputs.K,
        weights=weights,
        replay_starts=replay_starts,
    )
    diagnostics = tuple(
        align_partition(
            partition,
            inputs.reference_labels,
            weights=draw_weights,
        )
        for partition, draw_weights in zip(raw.partitions, weights, strict=True)
    )
    aligned = np.asarray([item.aligned for item in diagnostics], dtype=object)
    summary = summarize_partition_posterior(
        aligned,
        reference_cluster_order=inputs.reference_cluster_order,
        alignment_diagnostics=diagnostics,
    )
    compression = summary.compression
    return PartitionPosteriorResult(
        n=inputs.n,
        B=inputs.B,
        K=inputs.K,
        npass=inputs.npass,
        observation_ids=_readonly_array(inputs.observation_ids, dtype=object),
        reference_labels=_readonly_array(inputs.reference_labels, dtype=object),
        reference_cluster_order=inputs.reference_cluster_order,
        root_seed=inputs.root_seed,
        seed_streams=streams,
        weights=_readonly_array(weights, dtype=np.float64),
        replay_starts=_readonly_array(replay_starts, dtype=np.int64),
        raw_medoid_levels=_readonly_array(raw.raw_medoid_levels, dtype=np.int64),
        raw_partitions=_readonly_array(raw.partitions, dtype=np.int64),
        aligned_partitions=_readonly_array(aligned, dtype=object),
        alignment_diagnostics=diagnostics,
        distinct_partitions=compression.distinct_partitions,
        counts=compression.counts,
        masses=compression.masses,
        inverse_index=compression.inverse_index,
        mass_mcse=compression.mass_mcse,
        membership_probabilities=summary.membership_probabilities,
        entropy=summary.entropy,
        co_clustering=summary.co_clustering,
        ambiguity_flags=summary.ambiguity_flags,
        ambiguous_draw_count=summary.ambiguous_draw_count,
        ambiguous_draw_rate=summary.ambiguous_draw_rate,
        weighted_unweighted_difference_flags=summary.weighted_unweighted_difference_flags,
        weighted_unweighted_difference_count=summary.weighted_unweighted_difference_count,
        weighted_unweighted_difference_rate=summary.weighted_unweighted_difference_rate,
        warnings=summary.warnings,
    )

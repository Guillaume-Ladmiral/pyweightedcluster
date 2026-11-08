from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests.r_parity_utils import canonical_partition
from weightedcluster import wcCmpCluster, wcKMedRange
from weightedcluster._fallback import weighted_medoids as pure_weighted_medoids
from weightedcluster._utils import find_medoid
from weightedcluster.kmedoids import (
    _pam_build_initial_medoids,
    _pamonce_from_start,
    wcKMedoids,
)


def _assign(D: np.ndarray, medoids: np.ndarray, weights: np.ndarray):
    distances = D[:, medoids]
    labels = np.argmin(distances, axis=1)
    objective = 0.0
    for i in range(D.shape[0]):
        objective += float(weights[i]) * float(distances[i, labels[i]])
    return labels.astype(int), objective


def _scalar_r_build(D: np.ndarray, k: int, weights: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    nearest = np.full(n, 1.1 * float(np.max(D[np.triu_indices(n, 1)])) + 1.0)
    selected = np.zeros(n, dtype=bool)
    result: list[int] = []
    for _ in range(k):
        best_gain = 0.0
        best_index = -1
        for candidate in range(n):
            if selected[candidate]:
                continue
            gain = 0.0
            for observation in range(n):
                improvement = float(nearest[observation] - D[candidate, observation])
                if improvement > 0.0:
                    gain += float(weights[observation]) * improvement
            if best_gain <= gain:
                best_gain = gain
                best_index = candidate
        selected[best_index] = True
        result.append(best_index)
        for observation in range(n):
            if nearest[observation] > D[best_index, observation]:
                nearest[observation] = D[best_index, observation]
    return np.asarray(result, dtype=int)


def _swap_mutant(
    D: np.ndarray,
    start: np.ndarray,
    weights: np.ndarray,
    *,
    first_improvement: bool = False,
    zero_guard: bool = True,
):
    medoids = start.copy()
    labels, objective = _assign(D, medoids, weights)
    for _ in range(100):
        best: tuple[float, int, int, np.ndarray, float] | None = None
        changed = False
        for slot, current in enumerate(medoids):
            for candidate in range(D.shape[0]):
                if zero_guard and D[current, candidate] <= 0.0:
                    continue
                trial = medoids.copy()
                trial[slot] = candidate
                trial_labels, trial_objective = _assign(D, trial, weights)
                delta = trial_objective - objective
                if delta < -1e-10 and first_improvement:
                    medoids, labels, objective = trial, trial_labels, trial_objective
                    changed = True
                    break
                if best is None or delta < best[0]:
                    best = (delta, slot, candidate, trial_labels, trial_objective)
            if changed:
                break
        if changed:
            continue
        if best is None or best[0] >= -1e-10:
            return labels, medoids, objective
        _, slot, candidate, labels, objective = best
        medoids[slot] = candidate
    raise AssertionError("test mutant did not converge")


def _old_eager_scan_mutant(
    D: np.ndarray, start: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Replay the former backend's result-changing in-scan swap updates."""
    medoids = start.copy()
    labels, objective = _assign(D, medoids, weights)
    for _ in range(100):
        improved = False
        is_medoid = np.zeros(D.shape[0], dtype=bool)
        is_medoid[medoids] = True
        for slot in range(len(medoids)):
            for candidate in range(D.shape[0]):
                if is_medoid[candidate]:
                    continue
                trial = medoids.copy()
                trial[slot] = candidate
                trial_labels, trial_objective = _assign(D, trial, weights)
                if trial_objective + 1e-12 < objective:
                    medoids = trial
                    labels = trial_labels
                    objective = trial_objective
                    is_medoid[:] = False
                    is_medoid[medoids] = True
                    improved = True
        if not improved:
            return labels, medoids, objective
    raise AssertionError("former eager-scan mutant did not converge")


def _tie_heavy(rng: np.random.Generator, n: int, levels: int) -> np.ndarray:
    raw = rng.integers(0, levels, size=(n, n)).astype(float)
    D = (raw + raw.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def test_pam_build_uses_r_scalar_accumulation_and_tie_order():
    rng = np.random.default_rng(4242)
    checked = 0
    for levels in (2, 3, 4):
        for _ in range(60):
            n = int(rng.integers(12, 26))
            D = _tie_heavy(rng, n, levels)
            k = int(rng.integers(2, 6))
            weights = rng.integers(1, 4, size=n).astype(float)
            np.testing.assert_array_equal(
                _pam_build_initial_medoids(D, k, weights),
                _scalar_r_build(D, k, weights),
            )
            checked += 1
    assert checked == 180


def _find_discriminating_case(
    predicate: Callable[[np.ndarray, np.ndarray, np.ndarray], bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260812)
    for _ in range(600):
        n = int(rng.integers(9, 20))
        D = _tie_heavy(rng, n, int(rng.integers(2, 5)))
        weights = rng.integers(0, 5, size=n).astype(float)
        if weights.sum() == 0.0:
            continue
        start = _scalar_r_build(D, int(rng.integers(2, min(5, n))), weights)
        if predicate(D, start, weights):
            return D, start, weights
    raise AssertionError("deterministic search did not find a discriminating case")


def test_pamonce_regression_is_sensitive_to_old_first_improvement_behavior():
    def differs(D, start, weights):
        expected = _pamonce_from_start(D, start, weights)
        mutant = _old_eager_scan_mutant(D, start, weights)
        return not np.array_equal(expected[1][expected[0]], mutant[1][mutant[0]])

    D, start, weights = _find_discriminating_case(differs)
    expected_labels, expected_medoids, expected_objective = _pamonce_from_start(D, start, weights)
    mutant_labels, mutant_medoids, _ = _old_eager_scan_mutant(D, start, weights)
    assert not np.array_equal(expected_medoids[expected_labels], mutant_medoids[mutant_labels])

    labels, medoids, objective = pure_weighted_medoids(
        D,
        len(start),
        weights=weights,
        init_medoids=start,
        n_init=1,
        max_iter=100,
    )
    np.testing.assert_array_equal(medoids[labels], expected_medoids[expected_labels])
    assert objective == pytest.approx(expected_objective)
    try:
        from weightedcluster._wc_core import weighted_medoids as compiled_weighted_medoids
    except ImportError:
        return
    compiled_labels, compiled_medoids, compiled_objective = compiled_weighted_medoids(
        D,
        len(start),
        weights,
        100,
        1,
        0,
        start,
    )
    np.testing.assert_array_equal(
        compiled_medoids[compiled_labels], expected_medoids[expected_labels]
    )
    assert compiled_objective == pytest.approx(expected_objective)


def test_pamonce_regression_is_sensitive_to_zero_distance_candidate_guard():
    def differs(D, start, weights):
        expected = _pamonce_from_start(D, start, weights)
        mutant = _swap_mutant(D, start, weights, zero_guard=False)
        return not np.array_equal(expected[1][expected[0]], mutant[1][mutant[0]])

    D, start, weights = _find_discriminating_case(differs)
    expected = _pamonce_from_start(D, start, weights)
    mutant = _swap_mutant(D, start, weights, zero_guard=False)
    assert not np.array_equal(expected[1][expected[0]], mutant[1][mutant[0]])


def test_initialclust_medoid_positions_are_only_one_based():
    points = np.array([[0.0], [1.0], [4.0], [9.0]])
    D = np.abs(points - points.T)
    from_one_based = wcKMedoids(D, 2, initialclust=[1, 3], cluster_only=True)
    from_zero_based_replay = wcKMedoids(
        D,
        2,
        npass=0,
        replay_starts=[[0, 2]],
        cluster_only=True,
    )
    np.testing.assert_array_equal(from_one_based, from_zero_based_replay)
    with pytest.raises(ValueError, match="starting medoids should be in 1:4"):
        wcKMedoids(D, 2, initialclust=[0, 2])


def test_compiled_backend_matches_explicit_pamonce_when_available():
    core = pytest.importorskip("weightedcluster._wc_core")

    def differs(D, start, weights):
        expected = _pamonce_from_start(D, start, weights)
        mutant = _swap_mutant(D, start, weights, first_improvement=True)
        return not np.array_equal(expected[1][expected[0]], mutant[1][mutant[0]])

    D, start, weights = _find_discriminating_case(differs)
    expected_labels, expected_medoids, expected_objective = _pamonce_from_start(D, start, weights)
    labels, medoids, objective = core.weighted_medoids(
        D,
        len(start),
        weights,
        100,
        1,
        0,
        start,
    )
    np.testing.assert_array_equal(medoids[labels], expected_medoids[expected_labels])
    assert objective == pytest.approx(expected_objective)


def test_discriminating_pamonce_propagates_through_range_and_comparison_wrappers():
    def differs(D, start, weights):
        expected = _pamonce_from_start(D, start, weights)
        mutant = _swap_mutant(D, start, weights, first_improvement=True)
        return canonical_partition(expected[1][expected[0]]) != canonical_partition(
            mutant[1][mutant[0]]
        )

    D, start, weights = _find_discriminating_case(differs)
    expected_labels, expected_medoids, _objective = _pamonce_from_start(D, start, weights)
    expected_partition = canonical_partition(expected_medoids[expected_labels])
    range_replay = {len(start): start.reshape(1, -1)}
    comparison_replay = {len(start): start}

    range_result = wcKMedRange(
        D,
        [len(start)],
        weights=weights,
        npass=0,
        replay_starts=range_replay,
    )
    comparison = wcCmpCluster(
        D,
        weights=weights,
        maxcluster=len(start),
        method="pam",
        replay_starts=comparison_replay,
    )

    assert (
        canonical_partition(range_result["clustering"][f"cluster{len(start)}"])
        == expected_partition
    )
    assert (
        canonical_partition(comparison["pam"]["clustering"][f"cluster{len(start)}"])
        == expected_partition
    )


@pytest.mark.parametrize("method", ["PAMonce", "PAM", "KMedoids"])
@pytest.mark.parametrize("weight_scale", [np.nextafter(0.0, 1.0), np.finfo(float).max])
def test_public_kmedoids_preserves_common_weight_scale_across_methods(
    method: str,
    weight_scale: float,
) -> None:
    distances = np.array(
        [
            [0.0, 17.0, 7.0, 8.0],
            [17.0, 0.0, 7.0, 17.0],
            [7.0, 7.0, 0.0, 25.0],
            [8.0, 17.0, 25.0, 0.0],
        ]
    )

    result = wcKMedoids(
        distances,
        2,
        weights=np.full(4, weight_scale),
        npass=0,
        replay_starts=[[0, 3]],
        method=method,
        cluster_only=True,
    )

    np.testing.assert_array_equal(result, np.array([3, 3, 3, 4]))


@pytest.mark.parametrize("method", ["PAMonce", "PAM", "KMedoids"])
def test_public_kmedoids_preserves_joint_distance_weight_objective_order(
    method: str,
) -> None:
    maximum = np.finfo(float).max
    distances = np.array(
        [
            [0.0, 1e-308, 1.0, maximum],
            [1e-308, 0.0, 1e-323, maximum],
            [1.0, 1e-323, 0.0, 1e-323],
            [maximum, maximum, 1e-323, 0.0],
        ]
    )

    result = wcKMedoids(
        distances,
        2,
        weights=np.ones(4),
        npass=0,
        replay_starts=[[3, 0]],
        method=method,
        cluster_only=True,
    )

    np.testing.assert_array_equal(result, np.array([1, 3, 3, 3]))


@pytest.mark.parametrize("weight_scale", [np.nextafter(0.0, 1.0), np.finfo(float).max])
def test_public_kmedoids_build_preserves_common_weight_scale(weight_scale: float) -> None:
    distances = np.array(
        [
            [0.0, 17.0, 7.0, 8.0],
            [17.0, 0.0, 7.0, 17.0],
            [7.0, 7.0, 0.0, 25.0],
            [8.0, 17.0, 25.0, 0.0],
        ]
    )

    result = wcKMedoids(
        distances,
        2,
        weights=np.full(4, weight_scale),
        npass=1,
        random_state=1,
        method="PAMonce",
        cluster_only=True,
    )

    np.testing.assert_array_equal(result, np.array([3, 3, 3, 4]))


@pytest.mark.parametrize("method", ["PAMonce", "PAM", "KMedoids"])
def test_public_kmedoids_build_handles_finite_maximum_distance(method: str) -> None:
    maximum = np.finfo(float).max
    distances = np.array(
        [
            [0.0, 1e-308, 1.0, maximum],
            [1e-308, 0.0, 1e-323, maximum],
            [1.0, 1e-323, 0.0, 1e-323],
            [maximum, maximum, 1e-323, 0.0],
        ]
    )

    result = wcKMedoids(
        distances,
        2,
        weights=np.ones(4),
        npass=1,
        random_state=1,
        method=method,
        cluster_only=True,
    )

    np.testing.assert_array_equal(result, np.array([1, 3, 3, 3]))


@pytest.mark.parametrize("method", ["PAMonce", "PAM", "KMedoids"])
def test_public_kmedoids_compares_positive_subnormal_objectives_exactly(
    method: str,
) -> None:
    smallest = np.nextafter(0.0, 1.0)
    distances = np.zeros((5, 5), dtype=float)
    distances[:4, :4] = (
        np.array(
            [
                [0.0, 16.0, 5.0, 28.0],
                [16.0, 0.0, 2.0, 6.0],
                [5.0, 2.0, 0.0, 16.0],
                [28.0, 6.0, 16.0, 0.0],
            ]
        )
        * smallest
    )
    distances[:4, 4] = distances[4, :4] = 100.0 * smallest

    result = wcKMedoids(
        distances,
        2,
        weights=np.array([0.49, 0.9, 0.7, 0.49, 1.0]),
        npass=0,
        initialclust=np.array([1, 1, 1, 1, 2]),
        method=method,
        cluster_only=True,
    )

    np.testing.assert_array_equal(result, np.array([3, 3, 3, 3, 5]))


def test_compiled_kmedoids_delegates_positive_subnormal_objective_comparison() -> None:
    core = pytest.importorskip("weightedcluster._wc_core")
    smallest = np.nextafter(0.0, 1.0)
    distances = np.zeros((5, 5), dtype=float)
    distances[:4, :4] = (
        np.array(
            [
                [0.0, 16.0, 5.0, 28.0],
                [16.0, 0.0, 2.0, 6.0],
                [5.0, 2.0, 0.0, 16.0],
                [28.0, 6.0, 16.0, 0.0],
            ]
        )
        * smallest
    )
    distances[:4, 4] = distances[4, :4] = 100.0 * smallest
    weights = np.array([0.49, 0.9, 0.7, 0.49, 1.0])
    initial = np.array([1, 4])

    expected = pure_weighted_medoids(
        distances,
        2,
        weights,
        max_iter=100,
        n_init=1,
        random_state=0,
        init_medoids=initial,
    )
    actual = core.weighted_medoids(
        distances,
        2,
        weights,
        100,
        1,
        0,
        initial,
    )

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_shared_medoid_helper_compares_positive_subnormal_costs_exactly() -> None:
    smallest = np.nextafter(0.0, 1.0)
    distances = (
        np.array(
            [
                [0.0, 16.0, 5.0, 28.0],
                [16.0, 0.0, 2.0, 6.0],
                [5.0, 2.0, 0.0, 16.0],
                [28.0, 6.0, 16.0, 0.0],
            ]
        )
        * smallest
    )

    assert (
        find_medoid(
            distances,
            np.arange(4),
            np.array([0.49, 0.9, 0.7, 0.49]),
        )
        == 2
    )


def test_wc_kmedoids_partitions_when_positive_weight_rows_are_all_zero_distance():
    """2026-09-02 audit WC-090203: valid degenerate input must not crash.

    Every positive-weight observation sits at distance zero from everything,
    and the only positive distances join zero-weight observations. The
    exact-objective range predicate must classify this as ordinary (False)
    instead of reducing an empty exponent array.
    """
    D = np.zeros((4, 4))
    D[2, 3] = D[3, 2] = 7.0
    w = np.array([1.0, 1.0, 0.0, 0.0])
    with pytest.warns(UserWarning, match="ASW can not be computed"):
        result = wcKMedoids(D, k=2, weights=w, method="PAMonce", random_state=0)
    clustering = np.asarray(result["clustering"])
    assert clustering.shape == (4,)
    assert result["info"]["Sum of Distances"] == 0.0


def test_pam_range_predicate_has_single_guarded_implementation():
    """2026-09-02 audit WC-090203: the kmedoids copy had lost the empty-mask guard."""
    from weightedcluster import _fallback
    from weightedcluster import kmedoids as kmedoids_module

    D = np.zeros((4, 4))
    D[2, 3] = D[3, 2] = 7.0
    w = np.array([1.0, 1.0, 0.0, 0.0])
    assert _fallback._pam_requires_exact_objectives(D, w) is False
    assert kmedoids_module._pam_requires_exact_objectives(D, w) is False
    assert (
        kmedoids_module._pam_requires_exact_objectives is _fallback._pam_requires_exact_objectives
    )


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1] / "orig_R_pkg/WeightedCluster/src/PAMonce.h"
    ).is_file(),
    reason="needs the vendored R source orig_R_pkg/, which the published package leaves out",
)
def test_pamonce_swap_tolerance_is_identical_across_python_cpp_and_r():
    """One value, three homes: tie the hand-synchronised C++ literal to it.

    R defines `WEIGHTED_CLUST_TOL -1e-10` (src/PAMonce.h:5). Python owns it as
    `_constants.PAMONCE_SWAP_TOLERANCE`, but `cpp/wc_core.cpp` must carry it as a
    literal because a C++ translation unit cannot import it. Nothing checked the
    two stayed equal: with integer-valued, tie-heavy dissimilarities every swap
    delta is an integer or a half, so no behavioural fixture can separate
    -1e-10 from, say, -1e-3 (2026-09-21 targeted audit, L-1). This reads the
    compiled kernel's actual comparison and R's definition and pins both.
    """
    import re
    from pathlib import Path

    from weightedcluster import _constants, kmedoids

    root = Path(__file__).resolve().parents[1]
    number = r"(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"

    cpp = (root / "cpp" / "wc_core.cpp").read_text(encoding="utf-8")
    cpp_literals = re.findall(r"best_delta\s*<\s*" + number, cpp)
    assert len(cpp_literals) == 1, f"expected one swap-stop comparison, found {cpp_literals}"

    header = (root / "orig_R_pkg/WeightedCluster/src/PAMonce.h").read_text(encoding="utf-8")
    r_literals = re.findall(r"#define\s+WEIGHTED_CLUST_TOL\s+" + number, header)
    assert len(r_literals) == 1, f"expected one WEIGHTED_CLUST_TOL, found {r_literals}"

    expected = _constants.PAMONCE_SWAP_TOLERANCE
    assert float(cpp_literals[0]) == expected
    assert float(r_literals[0]) == expected
    assert kmedoids._PAMONCE_SWAP_TOLERANCE == expected

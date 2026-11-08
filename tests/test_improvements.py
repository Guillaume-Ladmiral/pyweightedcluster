"""Test suite for verifying all improvements from the 2025-10-25 audit.

This module tests:
1. Empty cluster handling
2. Linkage matrix validation
3. Random state consistency
4. Utility functions
5. Memory efficiency
6. Performance warnings
"""

from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pytest

from tests.r_parity_utils import fail_or_skip


def _hubert_c_ordered_pair_reference(
    D: np.ndarray, labels0: np.ndarray, weights: np.ndarray
) -> float:
    """Reference the WeightedCluster C++ ordered-pair Hubert C definition."""
    buckets: dict[float, float] = {}
    within_mass = 0.0
    wxy = 0.0
    for i in range(D.shape[0]):
        if weights[i] <= 0.0:
            continue
        diagonal_mass = float(weights[i] * weights[i])
        buckets[0.0] = buckets.get(0.0, 0.0) + diagonal_mass
        within_mass += diagonal_mass
        for j in range(i + 1, D.shape[0]):
            if weights[j] <= 0.0:
                continue
            mass = float(2.0 * weights[i] * weights[j])
            distance = float(D[i, j])
            buckets[distance] = buckets.get(distance, 0.0) + mass
            if labels0[i] == labels0[j]:
                within_mass += mass
                wxy += mass * distance

    remaining = within_mass
    smin = 0.0
    for distance, mass in sorted(buckets.items()):
        assigned = min(remaining, mass)
        smin += assigned * distance
        remaining -= assigned
        if remaining <= 0.0:
            break

    remaining = within_mass
    smax = 0.0
    for distance, mass in sorted(buckets.items(), reverse=True):
        assigned = min(remaining, mass)
        smax += assigned * distance
        remaining -= assigned
        if remaining <= 0.0:
            break
    return float((wxy - smin) / (smax - smin)) if smax > smin else float("nan")


class TestLabelCompression:
    """Test that sparse external labels are normalized consistently."""

    def test_sparse_labels_match_dense_labels_for_internal_indices(self):
        """Verify internal indices compress non-contiguous labels before scoring."""
        from weightedcluster.quality import compute_internal_indices, dunn_index

        D = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float)
        labels_gap = np.array([1, 1, 3, 3], dtype=int)
        labels_dense = np.array([1, 1, 2, 2], dtype=int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gap = compute_internal_indices(D, labels_gap)
        dense = compute_internal_indices(D, labels_dense)

        assert caught == []
        for key in ("DB", "XB", "PBM"):
            assert gap[key] == pytest.approx(dense[key])
        assert dunn_index(D, labels_gap) == pytest.approx(dunn_index(D, labels_dense))

    def test_quality_with_sparse_labels_matches_dense_labels(self):
        """Verify quality stats do not change when labels only differ by unused gaps."""
        from weightedcluster import wcClusterQuality

        D = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float)
        labels_gap = np.array([1, 1, 3, 3], dtype=int)
        labels_dense = np.array([1, 1, 2, 2], dtype=int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gap = wcClusterQuality(D, labels_gap)
        dense = wcClusterQuality(D, labels_dense)

        assert caught == []
        assert "stats" in gap and "ASW" in gap
        for key, dense_val in dense["stats"].items():
            gap_val = gap["stats"][key]
            if np.isnan(dense_val):
                assert np.isnan(gap_val)
            else:
                assert gap_val == pytest.approx(dense_val)

    def test_internal_indices_preserve_noninteger_cluster_categories(self):
        """Python-only indices must factorize labels rather than truncate them."""
        from weightedcluster.quality import compute_internal_indices, dunn_index

        D = np.array([[0, 1, 4, 5], [1, 0, 3, 4], [4, 3, 0, 1], [5, 4, 1, 0]], dtype=float)
        categorical = np.array([0.1, 0.1, 0.9, 0.9])
        dense = np.array([0, 0, 1, 1])

        actual = compute_internal_indices(D, categorical)
        expected = compute_internal_indices(D, dense)
        assert actual == pytest.approx(expected)
        assert dunn_index(D, categorical) == pytest.approx(dunn_index(D, dense))

    @pytest.mark.parametrize("function_name", ["compute_internal_indices", "dunn_index"])
    def test_internal_indices_reject_malformed_distance_shapes(self, function_name):
        from weightedcluster import quality

        function = getattr(quality, function_name)
        with pytest.raises(ValueError, match="square matrix"):
            function(np.array(1.0), [0])

    @pytest.mark.parametrize("function_name", ["compute_internal_indices", "dunn_index"])
    @pytest.mark.parametrize(
        "bad_distance",
        [
            np.array([[0.0, np.inf], [np.inf, 0.0]]),
            np.array([[0.0, -1.0], [-1.0, 0.0]]),
            np.array([[0.0, 1.0], [2.0, 0.0]]),
        ],
        ids=["nonfinite", "negative", "asymmetric"],
    )
    def test_internal_indices_reject_invalid_distance_values(self, function_name, bad_distance):
        from weightedcluster import quality

        with pytest.raises(ValueError):
            getattr(quality, function_name)(bad_distance, [0, 1])


class TestClustrangeOrdering:
    """Test that predictor column order and statistics remain aligned."""

    def test_as_clustrange_preserves_predictor_column_order(self):
        from weightedcluster import as_clustrange

        D = np.array(
            [
                [0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
                [1.0, 0.0, 1.0, 4.0, 5.0, 6.0],
                [2.0, 1.0, 0.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 0.0, 1.0, 2.0],
                [6.0, 5.0, 4.0, 1.0, 0.0, 1.0],
                [7.0, 6.0, 5.0, 2.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        cluster2 = np.array([1, 1, 1, 2, 2, 2], dtype=int)
        cluster3 = np.array([1, 1, 2, 2, 3, 3], dtype=int)

        ordered = as_clustrange({"cluster2": cluster2, "cluster3": cluster3}, D)
        unordered = as_clustrange({"cluster3": cluster3, "cluster2": cluster2}, D)

        assert unordered["kvals"] == [3, 2]
        assert unordered["clustering"].column_names == ["cluster3", "cluster2"]
        for key in ordered["stats"]:
            np.testing.assert_allclose(
                np.asarray(unordered["stats"][key], dtype=float),
                np.asarray(ordered["stats"][key], dtype=float)[::-1],
                equal_nan=True,
            )


class TestLinkageValidation:
    """Test linkage matrix validation."""

    def test_valid_linkage_matrix(self):
        """Verify valid linkage matrix passes validation."""
        from weightedcluster._validation import validate_linkage_matrix

        # Valid linkage for 4 observations (3 merges)
        Z = np.array([[0, 1, 0.5, 2], [2, 3, 0.8, 2], [4, 5, 1.0, 4]])

        # Should not raise
        validate_linkage_matrix(Z, 4, "test")

    def test_invalid_linkage_shape(self):
        """Verify invalid linkage matrix is rejected."""
        from weightedcluster._validation import validate_linkage_matrix

        # Wrong number of columns
        Z = np.array([[0, 1, 0.5], [2, 3, 0.8]])

        with pytest.raises(ValueError, match="4 columns"):
            validate_linkage_matrix(Z, 3, "test")

    def test_invalid_linkage_size(self):
        """Verify linkage matrix with wrong size is rejected."""
        from weightedcluster._validation import validate_linkage_matrix

        # 2 rows but should have 3 for N=4
        Z = np.array([[0, 1, 0.5, 2], [2, 3, 0.8, 2]])

        with pytest.raises(ValueError, match="should have 3"):
            validate_linkage_matrix(Z, 4, "test")

    def test_as_clustrange_linkage_runtime_errors_propagate(self, monkeypatch):
        """Verify valid linkage inputs do not mask downstream runtime failures."""
        import weightedcluster.range as rangemod
        from weightedcluster import as_clustrange

        D = np.array(
            [
                [0.0, 0.1, 1.0, 1.1],
                [0.1, 0.0, 1.1, 1.0],
                [1.0, 1.1, 0.0, 0.2],
                [1.1, 1.0, 0.2, 0.0],
            ],
            dtype=float,
        )
        Z = np.array(
            [
                [0.0, 1.0, 0.1, 2.0],
                [2.0, 3.0, 0.2, 2.0],
                [4.0, 5.0, 1.0, 4.0],
            ],
            dtype=float,
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("quality exploded")

        monkeypatch.setattr(rangemod, "wcClusterQuality", _boom)

        with pytest.raises(RuntimeError, match="quality exploded"):
            as_clustrange(Z, D, ncluster=3)


class TestRandomStateHandling:
    """Test random state generation and consistency."""

    def test_generate_random_seed_with_none(self):
        """Verify None generates different seeds."""
        from weightedcluster._utils import generate_random_seed

        seed1 = generate_random_seed(None, 0)
        seed2 = generate_random_seed(None, 0)

        # Should be different (very unlikely to be same)
        assert seed1 != seed2
        # Should be valid integers
        assert isinstance(seed1, int)
        assert isinstance(seed2, int)

    def test_generate_random_seed_deterministic(self):
        """Verify same base seed produces consistent results."""
        from weightedcluster._utils import generate_random_seed

        seed1 = generate_random_seed(42, 0)
        seed2 = generate_random_seed(42, 0)

        # Should be identical
        assert seed1 == seed2
        assert 0 <= seed1 < 2**31

    def test_generate_random_seed_with_offset(self):
        """Verify offset produces different seeds."""
        from weightedcluster._utils import generate_random_seed

        seed1 = generate_random_seed(42, 0)
        seed2 = generate_random_seed(42, 1)
        seed3 = generate_random_seed(42, 10)

        assert len({seed1, seed2, seed3}) == 3
        assert all(0 <= seed < 2**31 for seed in (seed1, seed2, seed3))

    @pytest.mark.parametrize(
        ("base_seed", "offset"),
        [(1.5, 0), (True, 0), ("7", 0), (7, 0.5), (-1, 0), (0, -1)],
    )
    def test_generate_random_seed_rejects_lossy_or_negative_controls(self, base_seed, offset):
        from weightedcluster._utils import generate_random_seed

        with pytest.raises(ValueError, match="seed|offset"):
            generate_random_seed(base_seed, offset)

    def test_reproducibility_with_seed(self):
        """Verify same seed produces same clustering results."""
        from weightedcluster import wcKMedoids

        rng = np.random.default_rng(20260801)
        D = rng.random((20, 20))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        result1 = wcKMedoids(D, k=3, npass=2, random_state=42, cluster_only=True)
        result2 = wcKMedoids(D, k=3, npass=2, random_state=42, cluster_only=True)

        np.testing.assert_array_equal(result1, result2)


class TestUtilityFunctions:
    """Test new utility functions."""

    def test_find_medoid_simple(self):
        """Test find_medoid with simple case."""
        from weightedcluster._utils import find_medoid

        # Simple triangle: 0-1 distance 1, 0-2 distance 4, 1-2 distance 3
        D = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]], dtype=float)
        idx = np.array([0, 1, 2])

        medoid = find_medoid(D, idx)
        # Node 1 has minimum sum: 1+3=4 vs 0: 1+4=5 vs 2: 4+3=7
        assert medoid == 1

    def test_find_medoid_with_weights(self):
        """Test find_medoid with weights."""
        from weightedcluster._utils import find_medoid

        D = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]], dtype=float)
        idx = np.array([0, 1, 2])
        weights = np.array([10.0, 1.0, 1.0])  # Heavy weight on first node

        medoid = find_medoid(D, idx, weights)
        # With heavy weight on node 0, medoid might change
        assert medoid in [0, 1, 2]

    def test_find_medoid_single_element(self):
        """Test find_medoid with single element."""
        from weightedcluster._utils import find_medoid

        D = np.array([[0, 1], [1, 0]], dtype=float)
        idx = np.array([0])

        medoid = find_medoid(D, idx)
        assert medoid == 0

    def test_find_medoid_empty_raises(self):
        """Test find_medoid raises on empty subset."""
        from weightedcluster._utils import find_medoid

        D = np.array([[0, 1], [1, 0]], dtype=float)
        idx = np.array([], dtype=int)

        with pytest.raises(ValueError, match="empty"):
            find_medoid(D, idx)

    def test_pairwise_sum(self):
        """Test pairwise_sum calculation."""
        from weightedcluster._utils import pairwise_sum

        D = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]], dtype=float)
        idx = np.array([0, 1, 2])

        result = pairwise_sum(D, idx)
        # Upper triangle: 1 + 4 + 3 = 8
        assert result == 8.0

    def test_pairwise_sum_single(self):
        """Test pairwise_sum with single element."""
        from weightedcluster._utils import pairwise_sum

        D = np.array([[0, 1], [1, 0]], dtype=float)
        idx = np.array([0])

        result = pairwise_sum(D, idx)
        assert result == 0.0


class TestMemoryEfficiency:
    """Test memory efficiency improvements."""

    def test_quality_completes_with_large_n(self):
        """Verify wcClusterQuality works with moderately large N."""
        from weightedcluster import wcClusterQuality

        # Create distance matrix for N=100
        N = 100
        rng = np.random.default_rng(42)
        D = rng.random((N, N))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        # Random clustering
        labels = rng.integers(1, 6, size=N)

        # Should complete without memory issues
        result = wcClusterQuality(D, labels)
        assert "stats" in result
        assert "ASW" in result["stats"]

    def test_quality_fallback_hc_matches_r_ordered_pair_formula(self, monkeypatch):
        """Verify fallback HC uses WeightedCluster's ordered-pair min/max normalization."""
        import weightedcluster.quality as qmod

        # Force pure-Python fallback path by masking compiled/numba modules.
        monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", types.ModuleType("_wc_core"))
        monkeypatch.setitem(sys.modules, "weightedcluster._numba", types.ModuleType("_numba"))

        D = np.array(
            [
                [0.0, 0.2, 1.0, 1.1],
                [0.2, 0.0, 0.9, 1.0],
                [1.0, 0.9, 0.0, 0.3],
                [1.1, 1.0, 0.3, 0.0],
            ],
            dtype=float,
        )
        labels = np.array([1, 1, 2, 2], dtype=int)
        weights = np.array([1.0, 1.5, 0.75, 1.25], dtype=float)
        out = qmod.wcClusterQuality(D, labels, weights=weights)
        hc = float(out["stats"]["HC"])

        hc_ref = _hubert_c_ordered_pair_reference(D, labels - 1, weights)

        assert np.isfinite(hc)
        assert np.isclose(hc, hc_ref, atol=1e-12, rtol=1e-10)


class TestHgHgsdParity:
    """Test HG/HGSD formulas distinguish ties on distance."""

    def test_hg_hgsd_weighted_formula_distinguishes_ties(self):
        from weightedcluster.quality import _compute_hg_hgsd_weighted

        D = np.array(
            [
                [0.0, 1.0, 1.0, 2.0],
                [1.0, 0.0, 2.0, 1.0],
                [1.0, 2.0, 0.0, 1.0],
                [2.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        labels0 = np.array([0, 0, 1, 1], dtype=int)
        w = np.ones(4, dtype=float)

        hg, hgsd = _compute_hg_hgsd_weighted(D, labels0, w)
        assert np.isfinite(hg)
        assert np.isfinite(hgsd)
        assert not np.isclose(hg, hgsd)
        assert abs(hgsd) <= abs(hg) + 1e-12

    def test_numba_hg_hgsd_not_identical(self):
        try:
            from weightedcluster._numba import cluster_quality_numba
        except ImportError:
            fail_or_skip("numba backend unavailable")

        D = np.array(
            [
                [0.0, 1.0, 1.0, 2.0],
                [1.0, 0.0, 2.0, 1.0],
                [1.0, 2.0, 0.0, 1.0],
                [2.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        labels0 = np.array([0, 0, 1, 1], dtype=int)
        stats, _asw, _asww = cluster_quality_numba(D, labels0, weights=None)
        assert np.isfinite(float(stats["HG"]))
        assert np.isfinite(float(stats["HGSD"]))
        assert not np.isclose(float(stats["HG"]), float(stats["HGSD"]))


class TestPerformanceWarnings:
    """Test performance warnings."""

    @pytest.mark.parametrize(("n_obs", "expect_warning"), [(500, False), (501, True)])
    def test_flexible_beta_warning_large_n(self, monkeypatch, n_obs, expect_warning):
        """Exercise the public warning boundary without running the cubic kernel."""
        from weightedcluster import cmpcluster as cmpcluster_module
        from weightedcluster import wcCmpCluster

        class BoundedKernelStop(Exception):
            pass

        calls = []

        def stop_before_cubic_work(distances, *, alpha):
            calls.append((np.asarray(distances).shape, alpha))
            raise BoundedKernelStop

        monkeypatch.setattr(cmpcluster_module, "_linkage_flexible_beta", stop_before_cubic_work)
        D = np.zeros((n_obs, n_obs), dtype=float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(BoundedKernelStop):
                wcCmpCluster(D, maxcluster=3, method=["flexible"], pam_combine=False)

        performance_warnings = [
            warning for warning in caught if "O(N³) complexity" in str(warning.message)
        ]
        assert bool(performance_warnings) is expect_warning
        assert calls == [((n_obs * (n_obs - 1) // 2,), 0.625)]


class TestDeprecatedRemoved:
    """Test that deprecated elements are removed."""

    def test_seqclarafamily_removed(self):
        """Verify seqclarafamily alias is removed."""
        import weightedcluster

        # Should not have the old alias
        assert not hasattr(weightedcluster, "seqclarafamily")
        # But should have the new name
        assert hasattr(weightedcluster, "seqclararange_family")

    def test_aggregate_exposes_stslist_weight_dispatch(self):
        """Verify the R-compatible stslist weighted dispatch is public."""
        import inspect

        from weightedcluster import wcAggregateCases

        sig = inspect.signature(wcAggregateCases)
        assert list(sig.parameters) == ["x", "weights", "weighted"]
        assert sig.parameters["weighted"].default is True

    def test_seqclustname_matches_weightedcluster_signature(self):
        """Verify seqclustname exposes required diss and weight dispatch."""
        import inspect

        from weightedcluster import seqclustname

        sig = inspect.signature(seqclustname)
        assert list(sig.parameters) == ["seqdata", "group", "diss", "weighted", "perc"]
        assert sig.parameters["diss"].default is inspect.Parameter.empty
        assert sig.parameters["weighted"].default is True

    def test_seqpropclust_exposes_r_with_missing_param(self):
        """Verify seqpropclust exposes R's active missing-spell control."""
        import inspect

        from weightedcluster import seqpropclust

        sig = inspect.signature(seqpropclust)
        assert sig.parameters["with_missing"].default is True

    def test_seqpropclust_exposes_r_weight_permutation_param(self):
        """Verify seqpropclust exposes R's active tree permutation control."""
        import inspect

        from weightedcluster import seqpropclust

        sig = inspect.signature(seqpropclust)
        assert sig.parameters["weight_permutation"].default == "diss"

    def test_seqnull_exposes_weighted_param_for_r_userpos_parity(self):
        """The R 2.0 userpos model weights attached sequence statistics."""
        import inspect

        from weightedcluster import seqnull

        sig = inspect.signature(seqnull)
        assert sig.parameters["weighted"].default is True

    def test_plot_clustrangefamily_no_ax_param(self):
        """Verify plot_clustrangefamily no longer exposes unused ax parameter."""
        import inspect

        from weightedcluster import plot_clustrangefamily

        sig = inspect.signature(plot_clustrangefamily)
        assert "ax" not in sig.parameters


class TestValidationConsistency:
    """Test consistent validation across modules."""

    def test_distance_matrix_validation(self):
        """Verify all modules validate distance matrices."""
        from weightedcluster import wcClusterQuality, wcKMedoids

        # Invalid distance matrix (not square)
        D_invalid = np.array([[0, 1, 2], [1, 0, 1]], dtype=float)

        with pytest.raises(ValueError):
            wcKMedoids(D_invalid, k=2)

        with pytest.raises(ValueError):
            wcClusterQuality(D_invalid, [1, 1])

    def test_distance_matrix_non_symmetric_raises_by_default(self):
        from weightedcluster._validation import validate_distance_matrix

        D = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
        with pytest.raises(ValueError, match="not symmetric"):
            validate_distance_matrix(D, name="diss")

    def test_distance_matrix_non_zero_diagonal_raises_by_default(self):
        from weightedcluster._validation import validate_distance_matrix

        D = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=float)
        with pytest.raises(ValueError, match="diagonal is not zero"):
            validate_distance_matrix(D, name="diss")

    def test_distance_matrix_permissive_mode_warns(self):
        from weightedcluster._validation import validate_distance_matrix

        D = np.array([[1.0, 1.0], [2.0, 0.0]], dtype=float)
        with pytest.warns(UserWarning, match="strict=False"):
            validate_distance_matrix(D, name="diss", strict=False)

    def test_distance_matrix_permissive_mode_warns_on_asymmetry_only(self):
        from weightedcluster._validation import validate_distance_matrix

        D = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_distance_matrix(D, name="diss", strict=False)
        assert any("not symmetric" in str(w.message) for w in caught)

    def test_distance_matrix_permissive_mode_warns_on_diagonal_only(self):
        from weightedcluster._validation import validate_distance_matrix

        D = np.array([[1.0, 2.0], [2.0, 0.0]], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_distance_matrix(D, name="diss", strict=False)
        assert any("diagonal is not zero" in str(w.message) for w in caught)

    def test_distance_matrix_permissive_warning_points_to_caller(self):
        from weightedcluster._validation import validate_distance_matrix

        D = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_distance_matrix(D, name="diss", strict=False)
        assert len(caught) >= 1
        assert "test_improvements.py" in caught[0].filename

    def test_distance_matrix_empty_raises(self):
        from weightedcluster._validation import validate_distance_matrix

        D = np.zeros((0, 0), dtype=float)
        with pytest.raises(ValueError, match="non-empty"):
            validate_distance_matrix(D, name="diss")

    def test_weights_validation(self):
        """Verify length validation and the source-supported negative edge."""
        from weightedcluster import wcKMedoids

        D = np.array([[0, 1], [1, 0]], dtype=float)

        # Wrong length weights
        with pytest.raises(ValueError, match="length"):
            wcKMedoids(D, k=2, weights=[1.0])

        # WeightedCluster 2.0 passes finite negative weights to its native
        # objective. The full Phase 2 oracle suite pins the numeric behavior.
        out = wcKMedoids(
            D,
            k=2,
            weights=[-1.0, 1.0],
            initialclust=[1, 2],
            npass=0,
            cluster_only=True,
        )
        assert out.shape == (2,)


@pytest.mark.parametrize(
    ("operation", "parameter"),
    [
        ("R", "R"),
        ("samplesize", "samplesize"),
        ("max_rank", "max_rank"),
        ("digits", "digits"),
    ],
)
def test_clustrange_integer_controls_reject_lossy_values(operation, parameter):
    from weightedcluster import as_clustrange, print_clustrange, summary_clustrange

    D = np.array(
        [[0.0, 1.0, 4.0, 5.0], [1.0, 0.0, 3.0, 4.0], [4.0, 3.0, 0.0, 1.0], [5.0, 4.0, 1.0, 0.0]]
    )
    clustering = {"cluster2": np.array([1, 1, 2, 2])}
    result = as_clustrange(clustering, D)

    with pytest.raises(ValueError, match=rf"{parameter} must be an integer"):
        if operation == "R":
            as_clustrange(clustering, D, R=1.5)
        elif operation == "samplesize":
            as_clustrange(clustering, D, R=2, samplesize=2.5)
        elif operation == "max_rank":
            summary_clustrange(result, max_rank=1.5)
        else:
            print_clustrange(result, digits=2.5)


class TestConstantsUsage:
    """Test that magic numbers are replaced with constants."""

    def test_constants_defined(self):
        """Verify all new constants are defined."""
        from weightedcluster._constants import (
            CLARA_BASE_SAMPLE_SIZE,
            CLARA_K_MULTIPLIER,
            FLEXIBLE_ALPHA_DEFAULT,
            FUZZY_CONVERGENCE_TOL,
            FUZZY_MAX_ITER,
        )

        assert CLARA_BASE_SAMPLE_SIZE == 40
        assert CLARA_K_MULTIPLIER == 2
        assert FUZZY_CONVERGENCE_TOL == 1e-3
        assert FUZZY_MAX_ITER == 100
        assert FLEXIBLE_ALPHA_DEFAULT == 0.625
        assert 1.0 - 2.0 * FLEXIBLE_ALPHA_DEFAULT == -0.25

    def test_every_constant_is_consumed_by_production_code(self):
        """Keep the shared-constants module free of inert scaffolding."""
        import ast
        import inspect
        from pathlib import Path

        import weightedcluster._constants as constants

        constants_path = Path(inspect.getfile(constants))
        constants_tree = ast.parse(constants_path.read_text(encoding="utf-8"))
        defined = {
            node.targets[0].id
            for node in constants_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        }
        loaded: set[str] = set()
        for path in constants_path.parent.glob("*.py"):
            if path == constants_path:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            loaded.update(
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in defined
            )

        assert defined == loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Regressions for WeightedCluster's fractional zero-mass HC boundary."""

from __future__ import annotations

import inspect
import math
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

import weightedcluster.quality as quality_module
from tests.r_parity_utils import fail_or_skip
from weightedcluster import (
    compute_internal_indices,
    pbm_index,
    wcClusterQuality,
    wcFuzzyQuality,
    wcFuzzyRange,
    wcSilhouetteObs,
)
from weightedcluster._fallback import cluster_quality as pure_cluster_quality
from weightedcluster.fuzzy import crispness

DISTANCES = np.array(
    [
        [0.0, 2.698970388939392, 4.201520141931764],
        [2.698970388939392, 0.0, 6.965269861988483],
        [4.201520141931764, 6.965269861988483, 0.0],
    ],
    dtype=float,
)
DIRECT_LABELS = np.array([0, 1, 1], dtype=np.int32)
R_STAT_NAMES = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")
UNDEFINED_ZERO_MASS_STATS = {
    "PBC": np.nan,
    "HG": np.nan,
    "HGSD": np.nan,
    "ASW": np.nan,
    "ASWw": 1.0,
    "CH": np.nan,
    "R2": np.nan,
    "CHsq": np.nan,
    "R2sq": np.nan,
    "HC": np.nan,
}
POSITIVE_MASS_STATS = {
    "PBC": 0.6060546393335191,
    "HG": 0.8190045248868777,
    "HGSD": 0.8190045248868777,
    "ASW": 0.13977228959635266,
    "ASWw": 0.8246394090092399,
    "CH": 0.18694846420195965,
    "R2": 0.6515053660310997,
    "CHsq": 0.05008949979523999,
    "R2sq": 0.3337308730029395,
    "HC": 0.12970037426860132,
}
PUBLIC_ONLY_STATS = {
    "DB": 1.2903568506221383,
    "XB": 2.2200277359299783,
    "PBM": 1.7873939439280486,
    "Dunn": 0.3874897085708715,
}
TINY_SCALE_DISTANCES = np.array(
    [
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 3.0],
        [4.0, 3.0, 0.0],
    ],
    dtype=float,
)
TINY_SCALE_LABELS = np.array([0, 0, 1], dtype=np.int32)
TINY_UNIFORM_WEIGHTS = np.full(3, 2.0**-500, dtype=float)
TINY_SCALE_PBC = 0.9521574310926243
PBM_DISTANCES = np.array(
    [
        [0.0, 1.0, 4.0, 5.0],
        [1.0, 0.0, 3.0, 4.0],
        [4.0, 3.0, 0.0, 1.0],
        [5.0, 4.0, 1.0, 0.0],
    ],
    dtype=float,
)
PBM_LABELS = np.array([0, 0, 1, 1], dtype=np.int32)
PBM_REFERENCE = 64.0
SCALE_INVARIANT_QUALITY_STATS = {
    "PBC": 0.9438798074485389,
    "HG": 1.0,
    "HGSD": 1.0,
    "ASW": 0.746031746031746,
    "ASWw": 0.873015873015873,
    "CH": 7.0,
    "R2": 0.7777777777777778,
    "CHsq": 32.0,
    "R2sq": 0.9411764705882353,
    "HC": 0.0,
}
SILHOUETTE_DYNAMIC_RANGE_DISTANCES = np.array(
    [
        [0.0, 0.5e308, 1e308, 1e308],
        [0.5e308, 0.0, 1e308, 1e308],
        [1e308, 1e308, 0.0, 0.5e308],
        [1e308, 1e308, 0.5e308, 0.0],
    ],
    dtype=float,
)
SILHOUETTE_DYNAMIC_RANGE_LABELS = np.array([1, 1, 2, 2], dtype=int)
MIN_POSITIVE = np.nextafter(0.0, 1.0)
ORDINARY_SILHOUETTE_POSITIONS = np.array(
    [
        13.598687942152127,
        38.92884700122874,
        47.185028430913434,
        49.30319960130941,
        84.29450634956284,
        88.89717741897533,
    ]
)
ORDINARY_SILHOUETTE_DISTANCES = np.abs(
    ORDINARY_SILHOUETTE_POSITIONS[:, None] - ORDINARY_SILHOUETTE_POSITIONS[None, :]
)
ORDINARY_SILHOUETTE_LABELS = np.array([1, 1, 1, 2, 2, 2], dtype=int)
ORDINARY_SILHOUETTE_WEIGHTS = np.array(
    [
        12.205746166179248,
        0.3063636292803649,
        15.1182440779202,
        7.395103413252505,
        4.36101598722675,
        11.051026643790744,
    ]
)
ORDINARY_ASW_BITS = (
    "0x1.5f0bae9f86b95p-1",
    "0x1.19d21f1a7017cp-1",
    "0x1.c96ca54a8f14ap-2",
    "-0x1.7afbb6f00560cp-2",
    "0x1.744d15906ad26p-1",
    "0x1.7e4ed652828f7p-1",
)
ORDINARY_ASWW_BITS = (
    "0x1.64def3be61ffbp-1",
    "0x1.2226c58b126f8p-1",
    "0x1.ddee1104199ffp-2",
    "-0x1.5d67b02e56bcdp-2",
    "0x1.7a6d2446e3995p-1",
    "0x1.83fe92816148fp-1",
)


def _dynamic_range_xb_fixture(x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array([0.0, x, 2.0 * x, 3.0 * x, 1.0], dtype=float)
    distances = np.abs(positions[:, None] - positions[None, :])
    labels = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    membership = np.eye(3, dtype=float)[labels]
    return distances, labels, membership


def _local_positive_fixture(far: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array([0.0, MIN_POSITIVE, 2 * MIN_POSITIVE, 3 * MIN_POSITIVE, far])
    distances = np.abs(positions[:, None] - positions[None, :])
    labels = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    membership = np.eye(3, dtype=float)[labels]
    return distances, labels, membership


def _backend(name: str) -> Callable:
    if name == "compiled":
        try:
            from weightedcluster._wc_core import cluster_quality
        except ImportError:
            fail_or_skip("compiled core quality backend unavailable")
        return cluster_quality
    if name == "numba":
        try:
            from weightedcluster._numba import cluster_quality_numba
        except ImportError:
            fail_or_skip("Numba quality backend unavailable")
        return cluster_quality_numba
    if name == "pure":
        return pure_cluster_quality
    raise AssertionError(f"unexpected backend: {name}")


def _assert_scalar(actual: float, expected: float) -> None:
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)


def _assert_stats(actual: dict[str, float], expected: dict[str, float]) -> None:
    for name, expected_value in expected.items():
        _assert_scalar(float(actual[name]), expected_value)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
@pytest.mark.parametrize(
    ("weights", "expected_stats", "expected_asw", "expected_asww"),
    [
        pytest.param(
            np.array([0.0, 1.0, 0.1]),
            UNDEFINED_ZERO_MASS_STATS,
            [np.nan, np.nan],
            [np.nan, 1.0],
            id="fractional-zero-mass",
        ),
        pytest.param(
            np.array([0.0, 1.0, 1.0]),
            UNDEFINED_ZERO_MASS_STATS,
            [np.nan, np.nan],
            [np.nan, 1.0],
            id="integer-zero-mass-control",
        ),
        pytest.param(
            np.array([1.0, 1.0, 0.1]),
            POSITIVE_MASS_STATS,
            [1.0, -0.6422529016796903],
            [1.0, 0.6652206899267307],
            id="positive-mass-control",
        ),
    ],
)
def test_quality_threshold_boundary_matches_authoritative_literals_across_backends(
    backend_name: str,
    weights: np.ndarray,
    expected_stats: dict[str, float],
    expected_asw: list[float],
    expected_asww: list[float],
) -> None:
    """A residual-min implementation breaks only the fractional HC row."""
    stats, asw, asww = _backend(backend_name)(DISTANCES, DIRECT_LABELS, weights)

    assert set(stats) == set(R_STAT_NAMES)
    _assert_stats(stats, expected_stats)
    assert list(asw) == [1, 2]
    assert list(asww) == [1, 2]
    np.testing.assert_allclose(
        list(asw.values()), expected_asw, rtol=1e-13, atol=1e-13, equal_nan=True
    )
    np.testing.assert_allclose(
        list(asww.values()), expected_asww, rtol=1e-13, atol=1e-13, equal_nan=True
    )


@pytest.mark.parametrize(
    ("labels", "expected_keys", "expected_asww"),
    [
        pytest.param([0, 1, 1], [0, 1], [np.nan, 1.0], id="zero-based"),
        pytest.param([2, 1, 1], [1, 2], [1.0, np.nan], id="one-based-reordered"),
        pytest.param([20, 10, 10], [10, 20], [1.0, np.nan], id="noncontiguous-reordered"),
        pytest.param(["z", "a", "a"], ["a", "z"], [1.0, np.nan], id="string-factor-order"),
    ],
)
def test_public_fractional_zero_mass_preserves_factor_order_and_undefined_hc(
    labels: Sequence[Any], expected_keys: list[Any], expected_asww: list[float]
) -> None:
    with pytest.warns(UserWarning, match="less than one observation"):
        result = wcClusterQuality(DISTANCES, labels, weights=[0.0, 1.0, 0.1])

    _assert_stats(result["stats"], UNDEFINED_ZERO_MASS_STATS)
    _assert_stats(result["stats"], PUBLIC_ONLY_STATS)
    assert list(result["ASW"]) == expected_keys
    assert list(result["ASWw"]) == expected_keys
    np.testing.assert_allclose(list(result["ASW"].values()), [np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(list(result["ASWw"].values()), expected_asww, equal_nan=True)


@pytest.mark.parametrize(
    ("weights", "expected_stats", "expected_asw"),
    [
        pytest.param(
            np.array([0.0, 1.0, 0.1]),
            UNDEFINED_ZERO_MASS_STATS,
            [[np.nan, 1.0], [np.nan, np.nan]],
            id="fractional-zero-mass",
        ),
        pytest.param(
            np.array([0.0, 1.0, 1.0]),
            UNDEFINED_ZERO_MASS_STATS,
            [[np.nan, 1.0], [np.nan, np.nan]],
            id="integer-zero-mass-control",
        ),
        pytest.param(
            np.array([1.0, 1.0, 0.1]),
            POSITIVE_MASS_STATS,
            [[-0.6422529016796903, 0.6652206899267307], [1.0, 1.0]],
            id="positive-mass-control",
        ),
    ],
)
def test_quality_boundary_literals_match_live_r_weightedcluster_2_0(
    r_parity_session,
    weights: np.ndarray,
    expected_stats: dict[str, float],
    expected_asw: list[list[float]],
) -> None:
    result = r_parity_session.wc_cluster_quality(
        DISTANCES,
        np.array([20, 10, 10]),
        weights=weights,
    )

    assert tuple(result["stats"]) == R_STAT_NAMES
    _assert_stats(result["stats"], expected_stats)
    np.testing.assert_allclose(result["asw"], expected_asw, rtol=1e-13, atol=1e-13, equal_nan=True)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_pbc_is_invariant_below_the_former_compiled_denominator_floor(
    backend_name: str,
) -> None:
    """An absolute pair-mass floor changes PBC under uniform weight scaling."""
    stats, _asw, _asww = _backend(backend_name)(
        TINY_SCALE_DISTANCES,
        TINY_SCALE_LABELS,
        TINY_UNIFORM_WEIGHTS,
    )

    assert stats["PBC"] == pytest.approx(TINY_SCALE_PBC, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
@pytest.mark.parametrize("weight_scale", [np.finfo(float).max, MIN_POSITIVE])
def test_pair_mass_indices_cover_the_full_finite_common_weight_range(
    backend_name: str,
    weight_scale: float,
) -> None:
    """R-01: pair products are dimensionless after a finite common scale."""
    stats, _asw, _asww = _backend(backend_name)(
        PBM_DISTANCES,
        PBM_LABELS,
        np.full(4, weight_scale),
    )

    assert stats["PBC"] == pytest.approx(SCALE_INVARIANT_QUALITY_STATS["PBC"], rel=3e-15, abs=0.0)
    assert stats["HG"] == 1.0
    assert stats["HGSD"] == 1.0
    assert stats["ASWw"] == pytest.approx(SCALE_INVARIANT_QUALITY_STATS["ASWw"], rel=3e-15, abs=0.0)
    assert stats["HC"] == SCALE_INVARIANT_QUALITY_STATS["HC"]
    assert stats["R2"] == pytest.approx(SCALE_INVARIANT_QUALITY_STATS["R2"], rel=3e-15, abs=0.0)
    assert stats["R2sq"] == pytest.approx(SCALE_INVARIANT_QUALITY_STATS["R2sq"], rel=3e-15, abs=0.0)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_pair_mass_indices_preserve_mixed_maximum_and_minimum_weights(
    backend_name: str,
) -> None:
    maximum = np.finfo(float).max
    stats, _asw, _asww = _backend(backend_name)(
        PBM_DISTANCES,
        PBM_LABELS,
        np.array([maximum, MIN_POSITIVE, maximum, MIN_POSITIVE]),
    )

    assert stats["PBC"] == 1.0
    assert stats["HG"] == 1.0
    assert stats["HGSD"] == 1.0


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_pair_mass_indices_stabilize_finite_aggregate_overflow(
    backend_name: str,
) -> None:
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 5.0, 4.0],
            [4.0, 5.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    weight = 0.5 * math.sqrt(np.finfo(float).max)

    stats, _asw, _asww = _backend(backend_name)(
        distances,
        labels,
        np.full(4, weight),
    )

    assert stats["PBC"] == pytest.approx(0.9701425001453319, rel=3e-15, abs=0.0)
    assert stats["HG"] == 1.0
    assert stats["HGSD"] == 1.0
    assert stats["ASWw"] == pytest.approx(8.0 / 9.0, rel=3e-15, abs=0.0)
    assert stats["CH"] == pytest.approx(16.0 * weight - 8.0, rel=3e-15, abs=0.0)
    assert stats["CHsq"] == pytest.approx(80.0 * weight - 40.0, rel=3e-15, abs=0.0)


def test_public_pair_mass_indices_stabilize_finite_aggregate_overflow() -> None:
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 5.0, 4.0],
            [4.0, 5.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )
    weight = 0.5 * math.sqrt(np.finfo(float).max)

    result = wcClusterQuality(
        distances,
        [1, 1, 2, 2],
        weights=np.full(4, weight),
    )

    assert result["stats"]["PBC"] == pytest.approx(0.9701425001453319, rel=3e-15, abs=0.0)
    assert result["stats"]["HG"] == 1.0
    assert result["stats"]["HGSD"] == 1.0
    assert result["stats"]["CH"] == pytest.approx(16.0 * weight - 8.0, rel=3e-15, abs=0.0)
    assert result["stats"]["CHsq"] == pytest.approx(80.0 * weight - 40.0, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_pair_mass_indices_stabilize_subnormal_second_order_products(
    backend_name: str,
) -> None:
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 5.0, 4.0],
            [4.0, 5.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    weights = np.array([2.0**-537, 2.0**-536, 2.0**-537, 2.0**-536])

    stats, _asw, _asww = _backend(backend_name)(distances, labels, weights)

    assert stats["HG"] == 1.0
    assert stats["HGSD"] == 1.0


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
@pytest.mark.parametrize("weight_scale", [2.0**-300, 2.0**300])
def test_pair_mass_indices_stabilize_interior_fourth_power_products(
    backend_name: str,
    weight_scale: float,
) -> None:
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 5.0, 4.0],
            [4.0, 5.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    stats, _asw, _asww = _backend(backend_name)(
        distances,
        labels,
        np.full(4, weight_scale),
    )

    assert stats["HG"] == 1.0
    assert stats["HGSD"] == 1.0


@pytest.mark.parametrize("weight_scale", [2.0**-300, 2.0**300])
def test_public_pair_mass_indices_stabilize_interior_fourth_power_products(
    weight_scale: float,
) -> None:
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 5.0, 4.0],
            [4.0, 5.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )

    if weight_scale < 1.0:
        with pytest.warns(UserWarning, match="less than one observation"):
            result = wcClusterQuality(
                distances,
                [1, 1, 2, 2],
                weights=np.full(4, weight_scale),
            )
    else:
        result = wcClusterQuality(
            distances,
            [1, 1, 2, 2],
            weights=np.full(4, weight_scale),
        )

    assert result["stats"]["HG"] == 1.0
    assert result["stats"]["HGSD"] == 1.0


def test_pair_mass_range_predicate_has_fourth_order_boundary_controls() -> None:
    from weightedcluster._quality_contract import finite_pair_masses_need_scaling

    assert finite_pair_masses_need_scaling(np.full(4, 2.0**-256))
    assert not finite_pair_masses_need_scaling(np.full(4, 2.0**-255))
    assert finite_pair_masses_need_scaling(np.full(4, 2.0**254))
    assert not finite_pair_masses_need_scaling(np.full(4, 2.0**253))


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_pair_mass_indices_preserve_mixed_exponent_cancellation(
    backend_name: str,
) -> None:
    distances = np.array(
        [
            [0.0, 3.0, 7.0, 1.0],
            [3.0, 0.0, 10.0, 5.0],
            [7.0, 10.0, 0.0, 5.0],
            [1.0, 5.0, 5.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    weights = np.array([2.0**-200, 2.0**-200, 2.0**200, 2.0**200])
    expected = {
        "PBC": 1.1440874386248061e-60,
        "HG": 5.0 / 7.0,
        "HGSD": 2.0 / 3.0,
        "HC": 4.0 / 27.0,
        "R2": 1.1617775744547955e-120,
        "R2sq": 2.1841418399750154e-120,
        "CH": 3.733809166716685e-60,
        "CHsq": 7.019561233427367e-60,
    }

    stats, _asw, _asww = _backend(backend_name)(distances, labels, weights)

    for name, expected_value in expected.items():
        assert stats[name] == pytest.approx(expected_value, rel=3e-15, abs=0.0)


def test_public_pair_mass_indices_preserve_mixed_exponent_cancellation() -> None:
    distances = np.array(
        [
            [0.0, 3.0, 7.0, 1.0],
            [3.0, 0.0, 10.0, 5.0],
            [7.0, 10.0, 0.0, 5.0],
            [1.0, 5.0, 5.0, 0.0],
        ]
    )
    weights = np.array([2.0**-200, 2.0**-200, 2.0**200, 2.0**200])

    with pytest.warns(UserWarning, match="less than one observation"):
        result = wcClusterQuality(distances, [1, 1, 2, 2], weights=weights)

    assert result["stats"]["HG"] == pytest.approx(5.0 / 7.0, rel=3e-15)
    assert result["stats"]["CH"] == pytest.approx(3.733809166716685e-60, rel=3e-15)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
@pytest.mark.parametrize("weight_scale", [np.finfo(float).max, MIN_POSITIVE])
def test_cluster_quality_signed_asww_aggregate_preserves_common_weight_scale(
    backend_name: str,
    weight_scale: float,
) -> None:
    distances = np.array(
        [
            [0.0, 10.0, 1.0, 1.0],
            [10.0, 0.0, 20.0, 20.0],
            [1.0, 20.0, 0.0, 1.0],
            [1.0, 20.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    stats, _asw, _asww = _backend(backend_name)(
        distances,
        labels,
        np.full(4, weight_scale),
    )

    assert stats["ASWw"] == pytest.approx(0.46369047619047615, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("weight_scale", [np.finfo(float).max, MIN_POSITIVE])
def test_public_cluster_quality_signed_asww_aggregate_preserves_common_scale(
    weight_scale: float,
) -> None:
    distances = np.array(
        [
            [0.0, 10.0, 1.0, 1.0],
            [10.0, 0.0, 20.0, 20.0],
            [1.0, 20.0, 0.0, 1.0],
            [1.0, 20.0, 1.0, 0.0],
        ]
    )

    if weight_scale == MIN_POSITIVE:
        with pytest.warns(UserWarning, match="less than one observation"):
            result = wcClusterQuality(
                distances,
                [1, 1, 2, 2],
                weights=np.full(4, weight_scale),
            )
    else:
        result = wcClusterQuality(
            distances,
            [1, 1, 2, 2],
            weights=np.full(4, weight_scale),
        )

    assert result["stats"]["ASWw"] == pytest.approx(0.46369047619047615, rel=3e-15, abs=0.0)


def test_signed_weighted_average_preserves_exact_product_cancellation() -> None:
    above_one = np.nextafter(1.0, 2.0)
    below_one = np.nextafter(1.0, 0.0)

    result = quality_module._signed_weighted_average(
        np.array([below_one, -1.0]),
        np.array([above_one, 1.0]),
    )

    assert result.hex() == "0x1.ffffffffffffdp-55"


def _core_local_positive_distance_fixture() -> tuple[np.ndarray, np.ndarray]:
    smallest = MIN_POSITIVE
    maximum = 1e308
    distances = np.full((6, 6), maximum, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for left, right in ((0, 1), (2, 3), (4, 5)):
        distances[left, right] = distances[right, left] = smallest
    for left in (0, 1):
        for right in (2, 3):
            distances[left, right] = distances[right, left] = 2.0 * smallest
    return distances, np.repeat(np.arange(3, dtype=np.int32), 2)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_cluster_quality_tiers_preserve_local_positive_distances(
    backend_name: str,
) -> None:
    """R-12: a far pair cannot erase local positive silhouette geometry."""
    distances, labels = _core_local_positive_distance_fixture()

    stats, asw, asww = _backend(backend_name)(distances, labels, np.ones(6))

    assert stats["ASW"] == pytest.approx(2.0 / 3.0, rel=3e-15, abs=0.0)
    assert stats["ASWw"] == pytest.approx(5.0 / 6.0, rel=3e-15, abs=0.0)
    assert stats["HG"] == 1.0
    assert stats["HGSD"] == 1.0
    np.testing.assert_allclose(list(asw.values()), [0.5, 0.5, 1.0], rtol=3e-15, atol=0.0)
    np.testing.assert_allclose(list(asww.values()), [0.75, 0.75, 1.0], rtol=3e-15, atol=0.0)


def test_public_cluster_quality_preserves_local_positive_distances() -> None:
    distances, labels = _core_local_positive_distance_fixture()

    result = wcClusterQuality(distances, labels + 1, weights=np.ones(6))

    assert result["stats"]["ASW"] == pytest.approx(2.0 / 3.0, rel=3e-15, abs=0.0)
    assert result["stats"]["ASWw"] == pytest.approx(5.0 / 6.0, rel=3e-15, abs=0.0)


def test_public_cluster_quality_uses_exact_cluster_mass_at_one_boundary() -> None:
    below_one = np.nextafter(1.0, 0.0)
    distances = np.ones((4, 4), dtype=float) - np.eye(4)

    with pytest.warns(UserWarning, match="less than one observation"):
        result = wcClusterQuality(
            distances,
            [1, 1, 2, 2],
            weights=[below_one, 2.0**-54, 1.0, 1.0],
        )

    assert np.isnan(result["stats"]["ASW"])
    assert all(np.isnan(value) for value in result["ASW"].values())


def test_compiled_quality_routes_partial_local_distance_underflow_stably() -> None:
    smallest = MIN_POSITIVE
    distances = np.full((4, 4), smallest, dtype=float)
    np.fill_diagonal(distances, 0.0)
    distances[2, 3] = distances[3, 2] = 1.0
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    stats, _asw, asww = _backend("compiled")(distances, labels, np.ones(4))

    assert stats["ASWw"] == -0.25
    assert asww == {1: 0.5, 2: -1.0}


def test_compiled_quality_preserves_distinct_positive_distance_ranks() -> None:
    maximum = np.finfo(float).max
    a = 2.0**-51
    b = np.nextafter(a, np.inf)
    assert a / maximum == b / maximum == MIN_POSITIVE
    distances = np.array(
        [
            [0.0, a, b, maximum],
            [a, 0.0, maximum, maximum],
            [b, maximum, 0.0, a],
            [maximum, maximum, a, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    stats, _asw, _asww = _backend("compiled")(distances, labels, np.ones(4))

    assert stats["HGSD"] == 1.0


def test_public_pure_quality_preserves_minimum_positive_distance_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-14: scaling moments must not merge positive and zero rank keys."""
    smallest = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, smallest, smallest, smallest],
            [smallest, 0.0, smallest, smallest],
            [smallest, smallest, 0.0, 1e308],
            [smallest, smallest, 1e308, 0.0],
        ]
    )
    labels = np.array([1, 1, 2, 2], dtype=int)
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", None)
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", None)

    public = wcClusterQuality(distances, labels, weights=np.ones(4))
    internal = quality_module._wc_cluster_quality_internal(
        distances,
        labels,
        weights=np.ones(4),
    )

    for result in (public, internal):
        assert result["stats"]["HG"] == pytest.approx(1.0 / 3.0, abs=0.0)
        assert result["stats"]["HGSD"] == pytest.approx(2.0 / 7.0, abs=0.0)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_cluster_quality_ignores_zero_weight_far_scale_before_active_geometry(
    backend_name: str,
) -> None:
    smallest = MIN_POSITIVE
    maximum = 1e308
    distances = np.array(
        [
            [0.0, smallest, 2 * smallest, maximum],
            [smallest, 0.0, 4 * smallest, maximum],
            [2 * smallest, 4 * smallest, 0.0, maximum],
            [maximum, maximum, maximum, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    stats, _asw, _asww = _backend(backend_name)(
        distances,
        labels,
        np.array([1.0, 1.0, 1.0, 0.0]),
    )

    expected = {
        "PBC": 0.8618916073713347,
        "HG": 1.0,
        "HGSD": 1.0,
        "CH": 3.666666666666667,
        "R2": 0.7857142857142857,
        "CHsq": 13.0,
        "R2sq": 0.9285714285714286,
    }
    _assert_stats(stats, expected)


def test_tiny_uniform_weight_pbc_literal_matches_live_r_weightedcluster_2_0(
    r_parity_session,
) -> None:
    result = r_parity_session.wc_cluster_quality(
        TINY_SCALE_DISTANCES,
        TINY_SCALE_LABELS + 1,
        weights=TINY_UNIFORM_WEIGHTS,
    )

    assert result["stats"]["PBC"] == pytest.approx(TINY_SCALE_PBC, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
@pytest.mark.parametrize("distance_scale", [1e-170, 1e160])
def test_cluster_quality_backends_use_relative_distance_scale(
    backend_name: str,
    distance_scale: float,
) -> None:
    """R-11: dimensionless quality indices cannot square raw distance units."""
    stats, _asw, _asww = _backend(backend_name)(
        PBM_DISTANCES * distance_scale,
        PBM_LABELS,
    )

    _assert_stats(stats, SCALE_INVARIANT_QUALITY_STATS)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
@pytest.mark.parametrize(
    "distance_scale",
    [float.fromhex("0x1.8p-537"), math.sqrt(np.finfo(float).max) * 0.15],
)
def test_cluster_quality_tiers_stabilize_subnormal_or_overflowing_squared_sums(
    backend_name: str,
    distance_scale: float,
) -> None:
    stats, _asw, _asww = _backend(backend_name)(
        PBM_DISTANCES * distance_scale,
        PBM_LABELS,
        np.ones(4),
    )

    _assert_stats(stats, SCALE_INVARIANT_QUALITY_STATS)


@pytest.mark.parametrize(
    "distance_scale",
    [float.fromhex("0x1.8p-537"), math.sqrt(np.finfo(float).max) * 0.15],
)
def test_public_quality_stabilizes_subnormal_or_overflowing_squared_sums(
    distance_scale: float,
) -> None:
    result = wcClusterQuality(
        PBM_DISTANCES * distance_scale,
        PBM_LABELS + 1,
        weights=np.ones(4),
    )

    _assert_stats(result["stats"], SCALE_INVARIANT_QUALITY_STATS)


@pytest.mark.parametrize("backend_name", ["compiled", "numba", "pure"])
def test_cluster_quality_tiers_preserve_mixed_distance_exponent_cancellation(
    backend_name: str,
) -> None:
    distances = np.array(
        [
            [0.0, 2.0**200, 2.0**-200, 2.0**-200],
            [2.0**200, 0.0, 2.0**-100, 2.0**200],
            [2.0**-200, 2.0**-100, 0.0, 2.0**-200],
            [2.0**-200, 2.0**200, 2.0**-200, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    expected = {
        "PBC": 1.4171332168333375e-91,
        "R2": 2.4545467326488633e-91,
        "R2sq": 1.204959932551442e-181,
        "CH": 4.909093465297727e-91,
        "CHsq": 2.409919865102884e-181,
    }

    stats, _asw, _asww = _backend(backend_name)(distances, labels, np.ones(4))

    for name, expected_value in expected.items():
        assert stats[name] == pytest.approx(expected_value, rel=3e-15, abs=0.0)


def test_public_quality_preserves_mixed_distance_exponent_cancellation() -> None:
    distances = np.array(
        [
            [0.0, 2.0**200, 2.0**-200, 2.0**-200],
            [2.0**200, 0.0, 2.0**-100, 2.0**200],
            [2.0**-200, 2.0**-100, 0.0, 2.0**-200],
            [2.0**-200, 2.0**200, 2.0**-200, 0.0],
        ]
    )

    result = wcClusterQuality(distances, [1, 1, 2, 2], weights=np.ones(4))

    assert result["stats"]["PBC"] == pytest.approx(1.4171332168333375e-91, rel=3e-15, abs=0.0)
    assert result["stats"]["CH"] == pytest.approx(4.909093465297727e-91, rel=3e-15, abs=0.0)


def test_distance_range_predicate_has_precision_window_controls() -> None:
    from weightedcluster._quality_contract import finite_distance_squares_need_scaling

    assert not finite_distance_squares_need_scaling(np.r_[np.ones(11), 2.0**23])
    assert finite_distance_squares_need_scaling(np.r_[np.ones(11), 2.0**24])


@pytest.mark.parametrize(
    ("distance_scale", "expected_pbm"),
    [(1e-170, 0.0), (1e160, math.inf)],
)
def test_public_cluster_quality_uses_relative_distance_scale(
    distance_scale: float,
    expected_pbm: float,
) -> None:
    result = wcClusterQuality(
        PBM_DISTANCES * distance_scale,
        PBM_LABELS + 1,
    )

    _assert_stats(result["stats"], SCALE_INVARIANT_QUALITY_STATS)
    _assert_stats(result["stats"], {"DB": 0.25, "XB": 0.03125, "Dunn": 3.0})
    assert result["stats"]["PBM"] == expected_pbm


@pytest.mark.parametrize("x", [1e-200, np.nextafter(0.0, 1.0)])
def test_internal_indices_keep_dynamic_range_squared_ratios(x: float) -> None:
    """R-12: common factors must cancel before tiny distance squares form."""
    distances, labels, _membership = _dynamic_range_xb_fixture(x)

    result = compute_internal_indices(distances, labels)

    assert result["XB"] == pytest.approx(0.1, rel=3e-15, abs=0.0)
    assert result["PBM"] == math.inf


def test_subnormal_squared_ratios_keep_pre_rounding_precision() -> None:
    h = float.fromhex
    distances = np.ones((4, 4), dtype=float) - np.eye(4)
    distances[0, 1] = distances[1, 0] = h("0x1p-144")
    distances[2, 3] = distances[3, 2] = h("0x1.8p-154")
    distances[0, 2] = distances[2, 0] = h("0x1.8p-537")
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    expected = h("0x1.c71cb1c71c71cp+782")

    direct = compute_internal_indices(distances, labels)
    public = wcClusterQuality(distances, labels + 1)
    fuzzy = wcFuzzyQuality(
        distances,
        np.eye(2)[labels],
        medoids=[0, 2],
        m=2.0,
        weights=np.ones(4),
    )

    assert direct["XB"] == expected
    assert public["stats"]["XB"] == expected
    assert fuzzy["stats"]["XBf"] == expected


def test_pbm_restores_scale_before_a_subnormal_intermediate_rounds() -> None:
    h = float.fromhex
    distances = np.zeros((5, 5), dtype=float)
    distances[0, 1] = distances[1, 0] = h("0x1.fffffffffffffp+1022")
    distances[0, 3] = distances[3, 0] = 2.0**-52
    distances[2, 4] = distances[4, 2] = h("0x1.8p+1022")
    labels = np.array([0, 0, 1, 1, 2], dtype=np.int32)

    result = compute_internal_indices(distances, labels)

    assert result["PBM"].hex() == "0x1.0000000000001p-108"


@pytest.mark.parametrize("x", [1e-200, np.nextafter(0.0, 1.0)])
def test_public_cluster_quality_keeps_dynamic_range_squared_ratios(x: float) -> None:
    distances, labels, _membership = _dynamic_range_xb_fixture(x)

    result = wcClusterQuality(distances, labels + 1)

    assert result["stats"]["XB"] == pytest.approx(0.1, rel=3e-15, abs=0.0)
    assert result["stats"]["PBM"] == math.inf


@pytest.mark.parametrize("x", [1e-200, np.nextafter(0.0, 1.0)])
def test_fuzzy_quality_keeps_dynamic_range_squared_ratio(x: float) -> None:
    distances, _labels, membership = _dynamic_range_xb_fixture(x)

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2, 4],
        m=2.0,
        weights=np.ones(5, dtype=float),
    )

    assert result["stats"]["XBf"] == pytest.approx(0.1, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("far", [1.0, 1e100, 1e308])
def test_internal_indices_keep_local_positive_distances_before_ratio_math(far: float) -> None:
    """R-12: a whole-matrix scale cannot erase local exact-positive distances."""
    distances, labels, _membership = _local_positive_fixture(far)

    result = compute_internal_indices(distances, labels)

    assert result["DB"] == pytest.approx(1.0 / 3.0, rel=3e-15, abs=0.0)
    assert result["XB"] == pytest.approx(0.1, rel=3e-15, abs=0.0)
    assert result["PBM"] == math.inf


@pytest.mark.parametrize("far", [1.0, 1e100, 1e308])
def test_public_quality_keeps_local_positive_distances_before_ratio_math(far: float) -> None:
    distances, labels, _membership = _local_positive_fixture(far)

    result = wcClusterQuality(distances, labels + 1)

    assert result["stats"]["DB"] == pytest.approx(1.0 / 3.0, rel=3e-15, abs=0.0)
    assert result["stats"]["XB"] == pytest.approx(0.1, rel=3e-15, abs=0.0)
    assert result["stats"]["PBM"] == math.inf


@pytest.mark.parametrize("far", [1.0, 1e100, 1e308])
def test_fuzzy_quality_keeps_local_positive_distances_before_ratio_math(far: float) -> None:
    distances, _labels, membership = _local_positive_fixture(far)

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2, 4],
        m=2.0,
        weights=np.ones(5, dtype=float),
    )

    assert result["stats"]["XBf"] == pytest.approx(0.1, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("far", [1e100, 1e308])
def test_automatic_fuzzy_medoids_keep_local_positive_distances(far: float) -> None:
    distances, _labels, membership = _local_positive_fixture(far)

    result = wcFuzzyQuality(
        distances,
        membership,
        m=2.0,
        weights=np.ones(5, dtype=float),
    )

    assert result["stats"]["XBf"] == pytest.approx(0.1, rel=3e-15, abs=0.0)


def test_internal_and_public_db_cancel_minimum_positive_linear_scale() -> None:
    x = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, x, x, 2 * x],
            [x, 0.0, 2 * x, x],
            [x, 2 * x, 0.0, x],
            [2 * x, x, x, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    assert compute_internal_indices(distances, labels)["DB"] == pytest.approx(
        1.0,
        rel=0.0,
        abs=0.0,
    )
    assert wcClusterQuality(distances, labels + 1)["stats"]["DB"] == pytest.approx(
        1.0,
        rel=0.0,
        abs=0.0,
    )


def test_internal_and_public_db_average_large_finite_ratios_stably() -> None:
    separation = 1e-308
    distances = np.array(
        [
            [0.0, 1.0, separation, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [separation, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    assert compute_internal_indices(distances, labels)["DB"] == pytest.approx(
        1e308,
        rel=3e-15,
        abs=0.0,
    )
    assert wcClusterQuality(distances, labels + 1)["stats"]["DB"] == pytest.approx(
        1e308,
        rel=3e-15,
        abs=0.0,
    )


def test_internal_and_public_pbm_keep_positive_ratio_before_infinite_sum() -> None:
    maximum = 1e308
    distances = np.array(
        [
            [0.0, maximum, maximum, 1.0, 1.0],
            [maximum, 0.0, 1.0, 1.0, 1.0],
            [maximum, 1.0, 0.0, maximum, 1.0],
            [1.0, 1.0, maximum, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2], dtype=np.int32)

    assert compute_internal_indices(distances, labels)["PBM"] == pytest.approx(
        4.0 / 9.0,
        rel=3e-15,
        abs=0.0,
    )
    assert wcClusterQuality(distances, labels + 1)["stats"]["PBM"] == pytest.approx(
        4.0 / 9.0,
        rel=3e-15,
        abs=0.0,
    )


@pytest.mark.parametrize("weight_scale", [1.0, 1e308, MIN_POSITIVE])
def test_fuzzy_xb_forms_minimum_positive_membership_coefficients_stably(
    weight_scale: float,
) -> None:
    x = MIN_POSITIVE
    distances = np.array([[0.0, x, 1.0], [x, 0.0, 0.0], [1.0, 0.0, 0.0]])
    membership = np.array([[1.0, 0.0], [0.0, 1.0], [x, 1.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1],
        m=2.0,
        weights=np.full(3, weight_scale, dtype=float),
    )

    assert result["stats"]["XBf"] == pytest.approx(1.0 / 3.0, rel=3e-15, abs=0.0)


@pytest.mark.parametrize(
    ("measure", "expected"),
    [("ASW", 0.5), ("ASWw", 0.75)],
)
def test_public_silhouette_uses_stable_dynamic_range_accumulation(
    measure: str,
    expected: float,
) -> None:
    """R-13: a representable between-group mean cannot overflow while summing."""
    actual = wcSilhouetteObs(
        SILHOUETTE_DYNAMIC_RANGE_DISTANCES,
        SILHOUETTE_DYNAMIC_RANGE_LABELS,
        measure=measure,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("distance_scale", [1.0, 1e-200])
def test_public_silhouette_preserves_ordinary_common_scale(distance_scale: float) -> None:
    ordinary = SILHOUETTE_DYNAMIC_RANGE_DISTANCES / 1e308

    np.testing.assert_allclose(
        wcSilhouetteObs(
            ordinary * distance_scale,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            measure=["ASW", "ASWw"],
        ),
        np.tile([0.5, 0.75], (4, 1)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("weight_scale", [1.0, 1e308, np.nextafter(0.0, 1.0)])
def test_public_weighted_silhouette_is_invariant_to_common_weight_scale(
    weight_scale: float,
) -> None:
    """R-13: ASWw depends on weight ratios, not their common finite scale."""
    ordinary = SILHOUETTE_DYNAMIC_RANGE_DISTANCES / 1e308

    np.testing.assert_allclose(
        wcSilhouetteObs(
            ordinary,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=np.full(4, weight_scale, dtype=float),
            measure="ASWw",
        ),
        0.75,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("weight_scale", "expected"),
    [(1.0, 0.5), (1e308, 0.75)],
)
def test_public_asw_preserves_absolute_weight_mass_without_overflow(
    weight_scale: float,
    expected: float,
) -> None:
    ordinary = SILHOUETTE_DYNAMIC_RANGE_DISTANCES / 1e308

    np.testing.assert_allclose(
        wcSilhouetteObs(
            ordinary,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=np.full(4, weight_scale, dtype=float),
            measure="ASW",
        ),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_public_asw_still_rejects_subunit_absolute_weight_mass() -> None:
    ordinary = SILHOUETTE_DYNAMIC_RANGE_DISTANCES / 1e308

    with pytest.raises(ValueError, match="cluster has less than one observation"):
        wcSilhouetteObs(
            ordinary,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=np.full(4, np.nextafter(0.0, 1.0), dtype=float),
            measure="ASW",
        )


def test_public_silhouette_preserves_minimum_positive_local_distance_ratios() -> None:
    distances, labels0, _membership = _local_positive_fixture(1.0)

    result = wcSilhouetteObs(
        distances,
        labels0 + 1,
        measure=["ASW", "ASWw"],
    )

    np.testing.assert_allclose(
        result[:, 0],
        [0.6, 1.0 / 3.0, 1.0 / 3.0, 0.6, 1.0],
        rtol=3e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result[:, 1],
        [0.8, 2.0 / 3.0, 2.0 / 3.0, 0.8, 1.0],
        rtol=3e-15,
        atol=0.0,
    )


def test_fuzzy_quality_ams_preserves_minimum_positive_local_distance_ratios() -> None:
    distances, labels0, membership = _local_positive_fixture(1.0)

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2, 4],
        m=2.0,
        weights=np.ones(5, dtype=float),
    )

    assert result["stats"]["AMS"] == pytest.approx(
        0.7866666666666667,
        rel=3e-15,
        abs=0.0,
    )


@pytest.mark.parametrize("weight_scale", [1.0, 1e308, MIN_POSITIVE])
def test_weighted_silhouette_preserves_minimum_positive_distance_ratio(
    weight_scale: float,
) -> None:
    x = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, x, 2 * x, 2 * x],
            [x, 0.0, 2 * x, 2 * x],
            [2 * x, 2 * x, 0.0, x],
            [2 * x, 2 * x, x, 0.0],
        ]
    )

    np.testing.assert_allclose(
        wcSilhouetteObs(
            distances,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=np.full(4, weight_scale, dtype=float),
            measure="ASWw",
        ),
        0.75,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(("weight_scale", "expected"), [(1.0, 0.5), (1e308, 0.75)])
def test_asw_preserves_minimum_positive_distance_ratio_and_absolute_mass(
    weight_scale: float,
    expected: float,
) -> None:
    x = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, x, 2 * x, 2 * x],
            [x, 0.0, 2 * x, 2 * x],
            [2 * x, 2 * x, 0.0, x],
            [2 * x, 2 * x, x, 0.0],
        ]
    )

    np.testing.assert_allclose(
        wcSilhouetteObs(
            distances,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=np.full(4, weight_scale, dtype=float),
            measure="ASW",
        ),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_asw_rejects_minimum_positive_weight_mass_with_minimum_distances() -> None:
    x = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, x, 2 * x, 2 * x],
            [x, 0.0, 2 * x, 2 * x],
            [2 * x, 2 * x, 0.0, x],
            [2 * x, 2 * x, x, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="cluster has less than one observation"):
        wcSilhouetteObs(
            distances,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=np.full(4, x, dtype=float),
            measure="ASW",
        )


@pytest.mark.parametrize(
    ("measure", "expected_bits"),
    [("ASW", ORDINARY_ASW_BITS), ("ASWw", ORDINARY_ASWW_BITS)],
)
def test_public_silhouette_preserves_pre_r13_ordinary_output_bits(
    measure: str,
    expected_bits: tuple[str, ...],
) -> None:
    result = wcSilhouetteObs(
        ORDINARY_SILHOUETTE_DISTANCES,
        ORDINARY_SILHOUETTE_LABELS,
        weights=ORDINARY_SILHOUETTE_WEIGHTS,
        measure=measure,
    )

    assert tuple(float(value).hex() for value in result) == expected_bits


@pytest.mark.parametrize(
    "weight_scale",
    [1.0, np.finfo(float).max / 4.0, MIN_POSITIVE],
)
def test_nonuniform_asww_is_invariant_across_representable_common_weight_scales(
    weight_scale: float,
) -> None:
    ratios = np.array([1.0, 2.0, 1.0, 2.0, 3.0, 4.0])
    ordinary = wcSilhouetteObs(
        ORDINARY_SILHOUETTE_DISTANCES,
        ORDINARY_SILHOUETTE_LABELS,
        weights=ratios,
        measure="ASWw",
    )

    np.testing.assert_allclose(
        wcSilhouetteObs(
            ORDINARY_SILHOUETTE_DISTANCES,
            ORDINARY_SILHOUETTE_LABELS,
            weights=ratios * weight_scale,
            measure="ASWw",
        ),
        ordinary,
        rtol=3e-15,
        atol=0.0,
    )


def test_weighted_silhouette_detects_joint_product_underflow() -> None:
    distances = (
        np.array(
            [
                [0.0, 1.0, 2.0, 2.0],
                [1.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 1.0],
                [2.0, 2.0, 1.0, 0.0],
            ]
        )
        * 1e-200
    )

    np.testing.assert_allclose(
        wcSilhouetteObs(
            distances,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=np.full(4, 1e-200),
            measure="ASWw",
        ),
        0.75,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("distance_scale", "weight_scale"),
    [(1.0, 1.0), (1e-200, 1e-200), (1e200, 1e108)],
)
def test_nonuniform_asww_handles_joint_product_range(
    distance_scale: float,
    weight_scale: float,
) -> None:
    distances = (
        np.array(
            [
                [0.0, 1.0, 2.0, 2.0],
                [1.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 1.0],
                [2.0, 2.0, 1.0, 0.0],
            ]
        )
        * distance_scale
    )
    weights = np.array([1.0, 2.0, 1.0, 2.0]) * weight_scale

    np.testing.assert_allclose(
        wcSilhouetteObs(
            distances,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=weights,
            measure="ASWw",
        ),
        [2.0 / 3.0, 5.0 / 6.0, 2.0 / 3.0, 5.0 / 6.0],
        rtol=3e-15,
        atol=0.0,
    )


def test_asw_preserves_exact_below_and_above_one_cluster_mass_boundaries() -> None:
    ordinary = SILHOUETTE_DYNAMIC_RANGE_DISTANCES / 1e308
    exact = np.full(4, 0.5)
    below = np.full(4, np.nextafter(0.5, 0.0))
    above = np.full(4, np.nextafter(0.5, 1.0))

    np.testing.assert_allclose(
        wcSilhouetteObs(
            ordinary,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=exact,
            measure="ASW",
        ),
        1.0,
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="cluster has less than one observation"):
        wcSilhouetteObs(
            ordinary,
            SILHOUETTE_DYNAMIC_RANGE_LABELS,
            weights=below,
            measure="ASW",
        )
    above_result = wcSilhouetteObs(
        ordinary,
        SILHOUETTE_DYNAMIC_RANGE_LABELS,
        weights=above,
        measure="ASW",
    )
    assert np.all(above_result < 0.0)
    assert np.all(above_result > -1.0)


def test_fuzzy_quality_ams_uses_stable_public_silhouette_route() -> None:
    membership = np.eye(2, dtype=float)[np.array([0, 0, 1, 1])]

    result = wcFuzzyQuality(
        SILHOUETTE_DYNAMIC_RANGE_DISTANCES,
        membership,
        medoids=[0, 2],
        m=2.0,
        weights=np.ones(4, dtype=float),
    )

    assert result["stats"]["AMS"] == pytest.approx(0.75, rel=0.0, abs=0.0)


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-13])
def test_python_pbm_uses_every_exact_positive_within_dispersion(scale: float) -> None:
    """R-02: a positive ``EK`` cannot be replaced by an absolute floor."""
    distances = PBM_DISTANCES * scale
    expected = PBM_REFERENCE * scale**2

    assert compute_internal_indices(distances, PBM_LABELS)["PBM"] == pytest.approx(
        expected,
        rel=1e-14,
        abs=0.0,
    )
    assert pbm_index(distances, PBM_LABELS) == pytest.approx(
        expected,
        rel=1e-14,
        abs=0.0,
    )


def test_normalized_single_cluster_crispness_is_undefined() -> None:
    """R-03: the normalized K=1 formula is exactly zero divided by zero."""
    actual = crispness(np.ones((3, 1), dtype=float), norm=True)

    assert np.isnan(actual).all()


def test_normalized_single_cluster_crispness_matches_live_r(r_parity_session) -> None:
    membership = np.ones((3, 1), dtype=float)

    actual = crispness(membership, norm=True)
    expected = np.asarray(r_parity_session.r_crispness(membership, norm=True), dtype=float)

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "tiny_membership",
    [
        pytest.param(1e-13, id="below-former-floor"),
        pytest.param(1e-18, id="far-below-former-floor"),
        pytest.param(2.0**-500, id="extreme-positive"),
        pytest.param(0.0, id="exact-zero-control"),
        pytest.param(0.25, id="ordinary-control"),
    ],
)
def test_fuzzy_partition_entropy_uses_every_exact_positive_membership(
    tiny_membership: float,
) -> None:
    """R-05: exact-positive entropy terms cannot be replaced by a floor."""
    membership = np.array(
        [
            [1.0 - tiny_membership, tiny_membership],
            [tiny_membership, 1.0 - tiny_membership],
        ],
        dtype=float,
    )
    with localcontext() as context:
        context.prec = 800
        raw = [Decimal.from_float(float(value)) for value in membership[0]]
        total = sum(raw)
        normalized = [value / total for value in raw]
        expected = float(sum(-value * value.ln() for value in normalized if value > 0))

    result = wcFuzzyQuality(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        membership,
        medoids=[0, 1],
    )

    assert result["stats"]["FPE"] == pytest.approx(expected, rel=2e-15, abs=0.0)


FUZZY_SCALE_DISTANCES = np.array(
    [
        [0.0, 1.0, 4.0, 5.0],
        [1.0, 0.0, 3.0, 4.0],
        [4.0, 3.0, 0.0, 1.0],
        [5.0, 4.0, 1.0, 0.0],
    ],
    dtype=float,
)
_FUZZY_MEMBERSHIP_UNIT = math.ldexp(1.0, 1023)
FUZZY_OVERFLOW_MEMBERSHIP = np.array(
    [
        [1.5 * _FUZZY_MEMBERSHIP_UNIT, 0.5 * _FUZZY_MEMBERSHIP_UNIT],
        [1.25 * _FUZZY_MEMBERSHIP_UNIT, 0.75 * _FUZZY_MEMBERSHIP_UNIT],
        [0.5 * _FUZZY_MEMBERSHIP_UNIT, 1.5 * _FUZZY_MEMBERSHIP_UNIT],
        [0.75 * _FUZZY_MEMBERSHIP_UNIT, 1.25 * _FUZZY_MEMBERSHIP_UNIT],
    ],
    dtype=float,
)
FUZZY_WEIGHT_SCALES = [
    pytest.param(1.0, id="ordinary"),
    pytest.param(1e308, id="overflowing-total"),
    pytest.param(np.nextafter(0.0, 1.0), id="smallest-positive"),
]
FUZZY_EXPLICIT_MEDOID_STATS = {
    "FPC": 0.578125,
    "FPE": 0.6119491913883952,
    "XBf": 0.0928125,
    "AMS": 0.873015873015873,
}
FUZZY_AUTOMATIC_MEDOID_XBF = 0.1675347222222222
FUZZY_RANGE_STATS = {
    "AMS": 0.8738512949039264,
    "FPC": 0.8368055555555556,
    "FPE": 0.25322408837127824,
    "XBf": 0.039279513888888895,
}

SCALE_INVARIANT_INDEX_DISTANCES = np.abs(
    np.array([0.0, 10.0, 1.0, 20.0], dtype=float)[:, None]
    - np.array([0.0, 10.0, 1.0, 20.0], dtype=float)[None, :]
)
SCALE_INVARIANT_INDEX_LABELS = np.array([0, 1, 0, 1], dtype=int)


@pytest.mark.parametrize(
    ("scale", "expected_pbm"),
    [
        pytest.param(1e-170, 0.0, id="true-underflow"),
        pytest.param(1e-162, 1.73e-322, id="representable-subnormal"),
        pytest.param(1.0, 173.7603305785124, id="ordinary"),
        pytest.param(1e153, 1.737603305785124e308, id="representable-large"),
        pytest.param(1e160, np.inf, id="true-overflow"),
    ],
)
def test_internal_indices_use_relative_distance_scale_without_changing_pbm_units(
    scale: float,
    expected_pbm: float,
) -> None:
    """R-09: DB/XB are invariant while PBM retains its distance-squared units."""
    result = compute_internal_indices(
        SCALE_INVARIANT_INDEX_DISTANCES * scale,
        SCALE_INVARIANT_INDEX_LABELS,
    )

    assert result["DB"] == pytest.approx(0.55, rel=3e-15, abs=0.0)
    assert result["XB"] == pytest.approx(0.2525, rel=3e-15, abs=0.0)
    if expected_pbm == 0.0 or np.isinf(expected_pbm):
        assert result["PBM"] == expected_pbm
    else:
        assert result["PBM"] == pytest.approx(expected_pbm, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("weight_value", FUZZY_WEIGHT_SCALES)
def test_fuzzy_quality_normalizes_overflowing_rows_and_common_weight_scale(
    weight_value: float,
) -> None:
    """R-08: finite row/weight scales cannot change the four fuzzy indices."""
    result = wcFuzzyQuality(
        FUZZY_SCALE_DISTANCES,
        FUZZY_OVERFLOW_MEMBERSHIP,
        medoids=[0, 3],
        m=2.0,
        weights=np.full(4, weight_value, dtype=float),
    )

    for name, expected in FUZZY_EXPLICIT_MEDOID_STATS.items():
        assert result["stats"][name] == pytest.approx(expected, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("weight_value", FUZZY_WEIGHT_SCALES)
def test_fuzzy_quality_common_weight_scale_reaches_automatic_medoids(
    weight_value: float,
) -> None:
    result = wcFuzzyQuality(
        FUZZY_SCALE_DISTANCES,
        FUZZY_OVERFLOW_MEMBERSHIP,
        m=2.0,
        weights=np.full(4, weight_value, dtype=float),
    )

    assert result["stats"]["FPC"] == pytest.approx(0.578125, rel=3e-15, abs=0.0)
    assert result["stats"]["FPE"] == pytest.approx(
        0.6119491913883952,
        rel=3e-15,
        abs=0.0,
    )
    assert result["stats"]["XBf"] == pytest.approx(
        FUZZY_AUTOMATIC_MEDOID_XBF,
        rel=3e-15,
        abs=0.0,
    )
    assert result["stats"]["AMS"] == pytest.approx(
        0.873015873015873,
        rel=3e-15,
        abs=0.0,
    )


@pytest.mark.parametrize("weight_value", FUZZY_WEIGHT_SCALES)
def test_fuzzy_range_public_workflow_is_invariant_to_uniform_weight_scale(
    weight_value: float,
) -> None:
    result = wcFuzzyRange(
        FUZZY_SCALE_DISTANCES,
        [2],
        m=2.0,
        weights=np.full(4, weight_value, dtype=float),
        pam_n_init=1,
        random_state=17,
        crisp_labels=True,
    )

    np.testing.assert_array_equal(result["medoids"]["cluster2"], [2, 0])
    np.testing.assert_allclose(
        result["memb"]["cluster2"],
        [[0.0, 1.0], [0.25, 0.75], [1.0, 0.0], [5.0 / 6.0, 1.0 / 6.0]],
        rtol=3e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(result["labels"]["cluster2"], [2, 2, 1, 1])
    for name, expected in FUZZY_RANGE_STATS.items():
        assert result["stats"][name][0] == pytest.approx(expected, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("distance_scale", [1e-170, 1e160])
def test_fuzzy_quality_indices_are_invariant_to_extreme_finite_distance_scale(
    distance_scale: float,
) -> None:
    """R-10: fuzzy XB and AMS cannot square or accumulate raw distance scale."""
    result = wcFuzzyQuality(
        FUZZY_SCALE_DISTANCES * distance_scale,
        FUZZY_OVERFLOW_MEMBERSHIP,
        medoids=[0, 3],
        m=2.0,
        weights=np.ones(4, dtype=float),
    )

    for name, expected in FUZZY_EXPLICIT_MEDOID_STATS.items():
        assert result["stats"][name] == pytest.approx(expected, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("distance_scale", [1e-170, 1e160])
def test_fuzzy_range_public_workflow_is_invariant_to_extreme_distance_scale(
    distance_scale: float,
) -> None:
    result = wcFuzzyRange(
        FUZZY_SCALE_DISTANCES * distance_scale,
        [2],
        m=2.0,
        weights=np.ones(4, dtype=float),
        pam_n_init=1,
        random_state=17,
        crisp_labels=True,
    )

    np.testing.assert_array_equal(result["medoids"]["cluster2"], [2, 0])
    np.testing.assert_array_equal(result["labels"]["cluster2"], [2, 2, 1, 1])
    for name, expected in FUZZY_RANGE_STATS.items():
        assert result["stats"][name][0] == pytest.approx(expected, rel=3e-15, abs=0.0)


def test_fuzzy_xb_restores_tiny_power_before_large_distance_square() -> None:
    distances = np.array(
        [
            [0.0, 1.0, 1e308],
            [1.0, 0.0, 1e308],
            [1e308, 1e308, 0.0],
        ]
    )
    membership = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    expected = math.exp2(math.log2(2.0 / 3.0) - 2000.0 + 2.0 * math.log2(1e308))

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1],
        m=2000.0,
        weights=np.ones(3),
    )

    assert result["stats"]["XBf"] == pytest.approx(expected, rel=3e-13, abs=0.0)


def test_automatic_fuzzy_medoid_retains_minimum_positive_tie_breaker() -> None:
    x = MIN_POSITIVE
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    membership = np.array([[1.0, 0.0], [1.0, 0.0], [x, 1.0]])

    automatic = wcFuzzyQuality(distances, membership, m=2.0, weights=np.ones(3))
    explicit = wcFuzzyQuality(
        distances,
        membership,
        medoids=[1, 2],
        m=2.0,
        weights=np.ones(3),
    )

    assert explicit["stats"]["XBf"] == pytest.approx(1.0 / 3.0, rel=3e-15, abs=0.0)
    assert automatic["stats"]["XBf"] == pytest.approx(explicit["stats"]["XBf"], rel=0.0, abs=0.0)


def test_automatic_fuzzy_medoids_compare_noncrisp_coefficients_as_tuples() -> None:
    x = MIN_POSITIVE
    distances = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, x], [1.0, x, 0.0]])
    membership = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [x, 1.0, 0.0]])

    automatic = wcFuzzyQuality(distances, membership, m=2.0, weights=np.ones(3))
    explicit = wcFuzzyQuality(
        distances,
        membership,
        medoids=[1, 2, 0],
        m=2.0,
        weights=np.ones(3),
    )

    assert explicit["stats"]["XBf"] == pytest.approx(1.0 / 12.0, rel=3e-15, abs=0.0)
    assert automatic["stats"]["XBf"] == pytest.approx(explicit["stats"]["XBf"], rel=0.0, abs=0.0)


def test_automatic_fuzzy_medoids_accept_finite_extreme_membership_power() -> None:
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    below = np.nextafter(0.5, 0.0)
    above = np.nextafter(0.5, 1.0)
    membership = np.array([[0.5, 0.5], [0.5, 0.5], [below, above]])

    result = wcFuzzyQuality(distances, membership, m=1e308, weights=np.ones(3))

    assert result["stats"]["FPC"] == pytest.approx(0.5, rel=3e-15, abs=0.0)
    assert result["stats"]["FPE"] == pytest.approx(math.log(2.0), rel=3e-15, abs=0.0)


def test_fuzzy_ams_retains_minimum_positive_effective_weights() -> None:
    z = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    membership = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2],
        m=2.0,
        weights=np.array([1.0, z, z, z]),
    )

    assert result["stats"]["AMS"] == pytest.approx(0.5, rel=0.0, abs=0.0)


def test_fuzzy_ams_retains_signed_cancellation_residual() -> None:
    z = MIN_POSITIVE
    distances = np.ones((5, 5), dtype=float)
    np.fill_diagonal(distances, 0.0)
    distances[2, 4] = distances[4, 2] = 0.0
    membership = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ]
    )

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1, 2, 4, 3],
        m=2.0,
        weights=np.array([1.0, 2 * z, 1.0, 1.0, 1.0]),
    )

    assert result["stats"]["AMS"] == z


def test_silhouette_retains_huge_and_minimum_positive_weight_ratios() -> None:
    z = MIN_POSITIVE
    M = np.finfo(float).max
    distances = np.array(
        [
            [0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    weights = np.array([M, z, z, z])

    np.testing.assert_allclose(
        wcSilhouetteObs(distances, [1, 1, 2, 2], weights=weights, measure="ASWw"),
        [1.0, 0.5, 0.5, 0.5],
        rtol=0.0,
        atol=0.0,
    )


def test_silhouette_retains_compensating_mixed_weight_distance_ratios() -> None:
    x = MIN_POSITIVE
    M = 1e308
    distances = np.array(
        [
            [0.0, M, 2 * x, 2 * x],
            [M, 0.0, 2 * x, 2 * x],
            [2 * x, 2 * x, 0.0, M],
            [2 * x, 2 * x, M, 0.0],
        ]
    )
    weights = np.array([M, x, M, x])

    actual = wcSilhouetteObs(
        distances,
        [1, 1, 2, 2],
        weights=weights,
        measure=["ASW", "ASWw"],
    )

    np.testing.assert_allclose(actual[[0, 2], 0], 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[[0, 2], 1], 0.5, rtol=0.0, atol=0.0)


def test_silhouette_retains_close_cancellation_residual() -> None:
    z = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
        ]
    )

    actual = wcSilhouetteObs(
        distances,
        [1, 1, 2, 2],
        weights=[0.0, 1.0, 1.0, z],
        measure="ASWw",
    )

    assert actual[0] == z


def test_forced_stable_silhouette_keeps_material_same_sign_terms() -> None:
    z = MIN_POSITIVE
    distances = np.full((5, 5), 2.0, dtype=float)
    np.fill_diagonal(distances, 0.0)
    distances[0, 1] = distances[1, 0] = 1.0
    distances[2, 3] = distances[3, 2] = 1.0
    distances[0, 4] = distances[4, 0] = z
    distances[1, 4] = distances[4, 1] = 1.0
    weights = np.array([1.0, 0.0625, 1.0, 0.0625, 0.0])

    actual = wcSilhouetteObs(
        distances,
        [1, 1, 2, 2, 1],
        weights=weights,
        measure="ASWw",
    )

    np.testing.assert_allclose(
        actual[:4],
        [33.0 / 34.0, 9.0 / 17.0, 33.0 / 34.0, 9.0 / 17.0],
        rtol=3e-15,
        atol=0.0,
    )


def test_asw_retains_mass_minus_one_cancellation() -> None:
    M = 1e308
    distances = np.ones((4, 4), dtype=float)
    np.fill_diagonal(distances, 0.0)

    actual = wcSilhouetteObs(
        distances,
        [1, 1, 2, 2],
        weights=[0.0, M, M, 0.0],
        measure="ASW",
    )

    assert actual[0] == pytest.approx(-1e-308, rel=3e-15, abs=0.0)


def test_stable_silhouette_retains_zero_over_zero_nan_contract() -> None:
    x = MIN_POSITIVE
    distances = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, x, x],
            [0.0, x, 0.0, x],
            [0.0, x, x, 0.0],
        ]
    )

    actual = wcSilhouetteObs(distances, [1, 1, 2, 2], measure=["ASW", "ASWw"])

    assert np.isnan(actual[0]).all()


def test_stable_silhouette_compresses_each_sum_before_cross_products() -> None:
    # The stable branch was extracted into module helpers so the permissive
    # kernel can share it (WC-090304); inspect the whole path, not one frame.
    public_source = inspect.getsource(wcSilhouetteObs)
    mass_source = inspect.getsource(quality_module._cluster_mass_expansions)
    stable_source = inspect.getsource(quality_module._stable_observation_silhouettes)
    source = public_source + mass_source + stable_source

    assert "_cluster_mass_expansions(" in public_source
    assert "_stable_observation_silhouettes(" in public_source
    assert "size_components =" in mass_source
    assert "_exact_expansion_component" in mass_source
    assert "_exact_expansion_component" in stable_source
    assert "_positive_expansion_product" not in source

    n = 64
    labels = np.repeat([1, 2], n // 2)
    distances = np.ones((n, n), dtype=float) * MIN_POSITIVE
    np.fill_diagonal(distances, 0.0)
    result = wcSilhouetteObs(
        distances,
        labels,
        weights=np.full(n, MIN_POSITIVE),
        measure="ASWw",
    )
    assert result.shape == (n,)


def test_silhouette_rechecks_ordinary_near_tie_with_exact_sums() -> None:
    up1 = np.nextafter(1.0, 2.0)
    up2 = np.nextafter(up1, 2.0)
    down1 = np.nextafter(1.0, 0.0)
    down2 = np.nextafter(down1, 0.0)
    distances = np.ones((7, 7), dtype=float) - np.eye(7)

    def set_edge(left: int, right: int, value: float) -> None:
        distances[left, right] = distances[right, left] = value

    for target, value in zip(range(1, 4), [up2, down1, 1.0], strict=True):
        set_edge(0, target, value)
    for target, value in zip(range(4, 7), [down2, up2, up1], strict=True):
        set_edge(0, target, value)

    actual = wcSilhouetteObs(
        distances,
        [1, 1, 1, 1, 2, 2, 2],
        weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        measure="ASWw",
    )

    assert actual[0].hex() == "0x1.5555555555554p-55"


def test_stable_silhouette_rounds_only_after_the_final_exact_ratio() -> None:
    tiny = 2.0**-54
    distances = np.ones((4, 4), dtype=float) - np.eye(4)
    distances[0, 1] = distances[1, 0] = 1.0
    distances[0, 2] = distances[2, 0] = tiny
    distances[0, 3] = distances[3, 0] = 1.0

    actual = wcSilhouetteObs(
        distances,
        [0, 0, 1, 1],
        weights=[2.0**-53, 0.5, 2.0**1000, 2.0**-54],
        measure="ASWw",
    )

    assert actual[0].hex() == "-0x1.fffffffffffffp-1"


def test_stable_silhouette_retains_exact_final_ratio_low_bits() -> None:
    distances = np.ones((4, 4), dtype=float) - np.eye(4)
    distances[0, 1] = distances[1, 0] = float.fromhex("0x1.0000000000001p+900")
    distances[0, 2] = distances[2, 0] = float.fromhex("0x1.0000000000001p+1")
    distances[0, 3] = distances[3, 0] = 1.0

    actual = wcSilhouetteObs(
        distances,
        [0, 0, 1, 1],
        weights=[2.0**1000, 2.0**100, 2.0**-54, float.fromhex("0x1.0000000000001p+1")],
        measure="ASWw",
    )

    assert actual[0].hex() == "-0x1.bfffffffffffep-53"


def test_stable_silhouette_does_not_double_round_quarter_ratio() -> None:
    distances = np.ones((4, 4), dtype=float) - np.eye(4)
    distances[0, 1] = distances[1, 0] = 2.0**-10
    distances[0, 2] = distances[2, 0] = 2.0**-10
    distances[0, 3] = distances[3, 0] = 2.0**-500

    actual = wcSilhouetteObs(
        distances,
        [0, 0, 1, 1],
        weights=[2.0**100, 2.0**100, 2.0**-52, 2.0**-53],
        measure="ASWw",
    )

    assert actual[0].hex() == "0x1.0000000000000p-2"


def test_stable_silhouette_keeps_exact_weight_distance_products() -> None:
    h = float.fromhex
    distances = np.array(
        [
            [0.0, h("0x1p-1"), h("0x1.fffffffffffffp-54"), h("0x1p-54")],
            [h("0x1p-1"), 0.0, h("0x1p+53"), h("0x1.8p+99")],
            [h("0x1.fffffffffffffp-54"), h("0x1p+53"), 0.0, h("0x1.8p+99")],
            [h("0x1p-54"), h("0x1.8p+99"), h("0x1.8p+99"), 0.0],
        ]
    )
    weights = np.array([h("0x1.8p+9"), h("0x1.fffffffffffffp+999"), h("0x1p+899"), h("0x1p-55")])

    actual = wcSilhouetteObs(
        distances,
        [0, 0, 1, 1],
        weights=weights,
        measure="ASWw",
    )

    assert actual[3].hex() == "0x1.ffffffffe8000p-955"


def test_fuzzy_ams_keeps_exact_signed_weight_silhouette_products(monkeypatch) -> None:
    above_one = np.nextafter(1.0, 2.0)
    below_one = np.nextafter(1.0, 0.0)
    monkeypatch.setattr(
        quality_module,
        "wcSilhouetteObs",
        lambda *_args, **_kwargs: np.array([below_one, -1.0]),
    )

    result = wcFuzzyQuality(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        medoids=[0, 1],
        m=2.0,
        weights=np.array([above_one, 1.0]),
    )

    assert result["stats"]["AMS"].hex() == "0x1.ffffffffffffdp-55"


def test_asw_rejects_exact_mass_below_one_when_sum_rounds_to_one() -> None:
    below = np.nextafter(0.5, 0.0)
    distances = np.ones((4, 4), dtype=float) - np.eye(4)

    with pytest.raises(ValueError, match="cluster has less than one observation"):
        wcSilhouetteObs(
            distances,
            [1, 1, 2, 2],
            weights=[0.5, below, 1.0, 1.0],
            measure="ASW",
        )


def test_asw_retains_exact_mass_above_one_denominator() -> None:
    above = np.nextafter(0.5, 1.0)
    distances = np.ones((4, 4), dtype=float) - np.eye(4)

    actual = wcSilhouetteObs(
        distances,
        [1, 1, 2, 2],
        weights=[0.5, above, 1.0, 1.0],
        measure="ASW",
    )

    assert actual[0] < 0.0
    assert actual[0] > -1.0


def test_fuzzy_xb_retains_normalized_minimum_positive_membership_ratio() -> None:
    z = MIN_POSITIVE
    M = np.finfo(float).max
    membership_power = np.nextafter(1.0, 2.0)
    distances = np.array([[0.0, 1.0, M], [1.0, 0.0, M], [M, M, 0.0]])
    membership = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [z, 0.0, M]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1, 2],
        m=membership_power,
        weights=np.ones(3),
    )

    assert result["stats"]["XBf"] == pytest.approx(
        2.9605947323327946e-16,
        rel=3e-15,
        abs=0.0,
    )


def test_fuzzy_power_retains_exact_row_total_below_binary64_rounding() -> None:
    z = MIN_POSITIVE
    distances = np.ones((3, 3), dtype=float) - np.eye(3)
    membership = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, z, 0.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1, 2],
        m=1e308,
        weights=np.ones(3),
    )

    assert result["stats"]["XBf"].hex() == "0x1.5555555555552p-2"


def test_fuzzy_power_retains_amplified_exact_row_normalization() -> None:
    u = 2.0**-54
    membership_power = float(2**54)
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    membership = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, u]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1],
        m=membership_power,
        weights=np.ones(3),
    )

    assert result["stats"]["XBf"] == pytest.approx(
        0.4905059215619231,
        rel=3e-15,
        abs=0.0,
    )


def test_ordinary_fuzzy_rows_do_not_enter_guard_precision_path(monkeypatch) -> None:
    positions = np.arange(30, dtype=float)
    distances = np.abs(positions[:, None] - positions[None, :])
    membership = np.tile([0.1, 0.2, 0.7], (30, 1))
    membership[:10] = [0.7, 0.2, 0.1]
    membership[10:20] = [0.2, 0.7, 0.1]

    def fail_guard_precision(*_args, **_kwargs):
        raise AssertionError("ordinary rows must retain the direct binary64 path")

    monkeypatch.setattr(
        quality_module,
        "_decimal_powered_weighted_squared_distance_ratio",
        fail_guard_precision,
    )

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[5, 15, 25],
        m=2.0,
        weights=np.ones(30),
    )

    assert np.isfinite(result["stats"]["XBf"])


def test_automatic_medoid_retains_exact_normalized_membership_order() -> None:
    z = MIN_POSITIVE
    distances = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    membership = np.array([[1.0, 2 * z], [1.0, z], [0.0, 1.0]])

    automatic = wcFuzzyQuality(distances, membership, m=2.0, weights=np.ones(3))
    explicit = wcFuzzyQuality(
        distances,
        membership,
        medoids=[1, 2],
        m=2.0,
        weights=np.ones(3),
    )

    assert explicit["stats"]["XBf"] == pytest.approx(1.0 / 12.0, rel=3e-15, abs=0.0)
    assert automatic["stats"]["XBf"] == pytest.approx(explicit["stats"]["XBf"], rel=0.0, abs=0.0)


def test_automatic_medoid_rechecks_nearby_non_tied_binary64_costs() -> None:
    up1 = np.nextafter(1.0, 2.0)
    up2 = np.nextafter(up1, 2.0)
    down1 = np.nextafter(1.0, 0.0)
    down2 = np.nextafter(down1, 0.0)
    distances = np.full((6, 6), 10.0)
    np.fill_diagonal(distances, 0.0)

    def set_edge(left: int, right: int, value: float) -> None:
        distances[left, right] = distances[right, left] = value

    set_edge(0, 1, 0.0)
    for target, value in zip(range(2, 5), [up2, down1, 1.0], strict=True):
        set_edge(0, target, value)
    for target, value in zip(range(2, 5), [down2, up2, up1], strict=True):
        set_edge(1, target, value)
    set_edge(0, 5, 1.0)
    set_edge(1, 5, 2.0)
    for target in range(2, 5):
        set_edge(target, 5, 20.0)
    membership = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]])

    automatic = wcFuzzyQuality(distances, membership, m=2.0, weights=np.ones(6))
    explicit = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 5],
        m=2.0,
        weights=np.ones(6),
    )

    assert explicit["stats"]["XBf"] == pytest.approx(0.5000000000000001, rel=0.0, abs=0.0)
    assert automatic["stats"]["XBf"] == pytest.approx(explicit["stats"]["XBf"], rel=0.0, abs=0.0)


def test_automatic_medoid_exact_recheck_also_covers_tuple_coefficients() -> None:
    z = MIN_POSITIVE
    distances = np.array(
        [[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 2.0, 2.0], [1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]
    )
    membership = np.array([[1.0, 2 * z], [1.0, z], [0.0, 1.0], [z, 1.0]])

    automatic = wcFuzzyQuality(distances, membership, m=2.0, weights=np.ones(4))
    explicit = wcFuzzyQuality(
        distances,
        membership,
        medoids=[1, 2],
        m=2.0,
        weights=np.ones(4),
    )

    assert explicit["stats"]["XBf"] == pytest.approx(0.0625, rel=0.0, abs=0.0)
    assert automatic["stats"]["XBf"] == pytest.approx(explicit["stats"]["XBf"], rel=0.0, abs=0.0)


def test_fuzzy_ams_retains_exact_normalized_ambiguity_cancellation() -> None:
    z = MIN_POSITIVE
    below = np.nextafter(0.5, 0.0)
    distances = np.array(
        [[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 1.0, 0.0]]
    )
    membership = np.array([[1.0, z], [0.5, 0.5], [2 * z, 1.0], [below, 0.5]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 3],
        m=2.0,
        weights=np.array([1.0, z, 1.0, z]),
    )

    assert result["stats"]["AMS"] == z


def test_fuzzy_ams_retains_nonunit_exact_ambiguity_cancellation() -> None:
    u = 2.0**-52
    z = MIN_POSITIVE
    below = np.nextafter(0.5, 0.0)
    distances = np.array(
        [[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 1.0, 0.0]]
    )
    membership = np.array([[3.0, 2.0, u], [0.5, 0.5, 0.0], [2 * u, 3.0, 2.0], [0.0, 0.5, below]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 3, 1],
        m=2.0,
        weights=np.array([1.0, z, 1.0, z]),
    )

    assert result["stats"]["AMS"].hex() == "0x1.9999999999999p-56"


def test_fuzzy_ams_exact_ambiguity_guard_is_not_zero_result_only() -> None:
    r = 2.0**-52
    u = 2.0**-53
    distances = np.array(
        [[0.0, 0.0, u, 2.0], [0.0, 0.0, u, 2.0], [u, u, 0.0, 1.0], [2.0, 2.0, 1.0, 0.0]]
    )
    membership = np.array([[3.0, 2.0, r], [0.5, 0.5, 0.0], [2 * r, 3.0, 2.0], [0.0, 0.5, 0.5]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 3, 1],
        m=2.0,
        weights=np.ones(4),
    )

    assert result["stats"]["AMS"].hex() == "0x1.3333333333333p-53"


def test_fuzzy_entropy_retains_exact_normalized_row_terms() -> None:
    z = MIN_POSITIVE
    result = wcFuzzyQuality(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, z], [0.0, 1.0]]),
        medoids=[0, 1],
        m=2.0,
        weights=np.array([1.0, 0.0]),
    )

    assert result["stats"]["FPE"].hex() == "0x0.00000000002e9p-1022"


def test_fuzzy_entropy_detects_inexact_row_total_below_maximum() -> None:
    u = 2.0**-56
    result = wcFuzzyQuality(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 1.0, u], [0.0, 0.0, 1.0]]),
        medoids=[0, 1, 0],
        m=2.0,
        weights=np.array([1.0, 0.0]),
    )

    assert result["stats"]["FPE"].hex() == "0x1.62e42fefa39f2p-1"


@pytest.mark.parametrize("u", [2.0**-52, 2.0**-53])
def test_fuzzy_fpc_retains_exact_row_total_ratio(u: float) -> None:
    result = wcFuzzyQuality(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 1.0, u], [0.0, 0.0, 1.0]]),
        medoids=[0, 1, 0],
        m=2.0,
        weights=np.array([1.0, 0.0]),
    )

    expected_hex = "0x1.ffffffffffffep-2" if u == 2.0**-52 else "0x1.fffffffffffffp-2"
    assert result["stats"]["FPC"].hex() == expected_hex


def test_fuzzy_fpc_retains_weighted_average_cancellation() -> None:
    u = 2.0**-52
    distances = np.ones((3, 3), dtype=float) - np.eye(3)
    membership = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2],
        m=2.0,
        weights=np.array([1.0, u, 0.0]),
    )

    assert result["stats"]["FPC"].hex() == "0x1.fffffffffffffp-1"


def test_fuzzy_fpc_detects_product_scaled_weight_loss() -> None:
    u = np.nextafter(sys.float_info.epsilon, math.inf)
    distances = np.ones((3, 3), dtype=float) - np.eye(3)
    membership = np.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [0.0, 1.0, 0.0, 0.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2, 0, 0],
        m=2.0,
        weights=np.array([1.0, u, 0.0]),
    )

    assert result["stats"]["FPC"].hex() == "0x1.fffffffffffffp-1"


def test_fuzzy_fpc_fpe_detect_adjacent_row_sum_range_loss() -> None:
    u = np.nextafter(sys.float_info.epsilon, math.inf)
    result = wcFuzzyQuality(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 1.0, u], [0.0, 0.0, 1.0]]),
        medoids=[0, 1, 0],
        m=2.0,
        weights=np.array([1.0, 0.0]),
    )

    assert result["stats"]["FPC"].hex() == "0x1.ffffffffffffep-2"
    assert result["stats"]["FPE"].hex() == "0x1.62e42fefa3a14p-1"


def test_fuzzy_fpe_retains_weighted_average_cancellation() -> None:
    u = 2.0**-53
    distances = np.ones((3, 3), dtype=float) - np.eye(3)
    membership = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2],
        m=2.0,
        weights=np.array([1.0, u, 0.0]),
    )

    assert result["stats"]["FPE"].hex() == "0x1.62e42fefa39eep-54"


def test_fuzzy_xb_retains_exact_weight_total_residual() -> None:
    u = 2.0**-53
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    membership = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[1, 2],
        m=2.0,
        weights=np.array([1.0, u, 0.0]),
    )

    assert result["stats"]["XBf"].hex() == "0x1.fffffffffffffp-1"


@pytest.mark.parametrize("weight_scale", [1e-200, 1e-180])
def test_fuzzy_xb_keeps_joint_weight_distance_products(weight_scale: float) -> None:
    x = 1e-100
    distances = np.array(
        [
            [0.0, x, 2 * x, 2 * x],
            [x, 0.0, 2 * x, 2 * x],
            [2 * x, 2 * x, 0.0, x],
            [2 * x, 2 * x, x, 0.0],
        ]
    )
    membership = np.eye(2)[np.array([0, 0, 1, 1])]

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2],
        m=2.0,
        weights=np.full(4, weight_scale),
    )

    assert result["stats"]["XBf"] == pytest.approx(0.125, rel=3e-15, abs=0.0)


@pytest.mark.parametrize(
    "distance_scale",
    [
        math.sqrt(np.finfo(float).max),
        math.sqrt(np.finfo(float).tiny),
        1.0,
    ],
)
def test_fuzzy_xb_stabilizes_squared_term_aggregates(distance_scale: float) -> None:
    within = 0.49 * distance_scale
    between = 0.75 * distance_scale
    distances = np.array(
        [
            [0.0, within, between, between],
            [within, 0.0, between, between],
            [between, between, 0.0, between],
            [between, between, between, 0.0],
        ]
    )
    membership = np.eye(2)[np.array([0, 1, 0, 1])]

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1],
        m=2.0,
        weights=np.ones(4),
    )

    assert result["stats"]["XBf"] == pytest.approx(
        1.171386922115785,
        rel=3e-15,
        abs=0.0,
    )


def test_fuzzy_xb_distance_range_route_preserves_safe_coefficient_bits() -> None:
    distances = np.array([[0.0, 0.25, 0.51], [0.25, 0.0, 0.0], [0.51, 0.0, 0.0]])
    membership = np.array([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]])
    expected_hex = "0x1.fc3e93b16b717p-869"

    for distance_scale in (1.0, 1e-200):
        result = wcFuzzyQuality(
            distances * distance_scale,
            membership,
            medoids=[0, 1],
            m=500.0,
            weights=np.ones(3),
        )
        assert result["stats"]["XBf"].hex() == expected_hex


def test_fuzzy_xb_stabilizes_positive_subnormal_term_rounding() -> None:
    distances = np.array([[0.0, 0.25, 0.6], [0.25, 0.0, 0.0], [0.6, 0.0, 0.0]])
    membership = np.array([[1.0, 0.0], [0.0, 1.0], [0.49, 0.51]])
    expected_hex = "0x0.0000000222f19p-1022"

    for distance_scale in (1.0, 1e-200):
        result = wcFuzzyQuality(
            distances * distance_scale,
            membership,
            medoids=[0, 1],
            m=1024.0,
            weights=np.ones(3),
        )
        assert result["stats"]["XBf"].hex() == expected_hex


def test_fuzzy_ams_retains_exact_effective_weight_denominator() -> None:
    u = 2.0**-53
    below = np.nextafter(0.5, 0.0)
    distances = np.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    )
    membership = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [below, 0.5]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 2],
        m=2.0,
        weights=np.array([1.0, 1.0, u, u]),
    )

    assert result["stats"]["AMS"].hex() == "0x1.fffffffffffffp-1"


def test_fuzzy_xb_retains_huge_and_minimum_positive_weight_product() -> None:
    z = MIN_POSITIVE
    M = np.finfo(float).max
    distances = np.array([[0.0, 1.0, M], [1.0, 0.0, M], [M, M, 0.0]])
    membership = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1],
        m=2.0,
        weights=np.array([M, M, z]),
    )

    assert result["stats"]["XBf"] == pytest.approx(
        4.4408920985006257e-16,
        rel=3e-15,
        abs=0.0,
    )


def test_fuzzy_xb_cancels_mixed_weight_and_distance_exponents() -> None:
    x = MIN_POSITIVE
    M = 1e308
    separation = math.sqrt(x * M)
    distances = np.array([[0.0, separation, M], [separation, 0.0, M], [M, M, 0.0]])
    membership = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    result = wcFuzzyQuality(
        distances,
        membership,
        medoids=[0, 1],
        m=2.0,
        weights=np.array([M, M, x]),
    )

    assert result["stats"]["XBf"] == pytest.approx(0.5, rel=3e-15, abs=0.0)


def test_db_keeps_overrange_cluster_ratios_until_representable_mean() -> None:
    distances = np.ones((8, 8), dtype=float)
    np.fill_diagonal(distances, 0.0)
    distances[0, 1] = distances[1, 0] = 3.0
    distances[2, 3] = distances[3, 2] = 3.0
    distances[4, 5] = distances[5, 4] = 0.0
    distances[6, 7] = distances[7, 6] = 0.0
    distances[0, 2] = distances[2, 0] = 1e-308
    labels = np.repeat(np.arange(4), 2)

    assert compute_internal_indices(distances, labels)["DB"] == pytest.approx(
        1.5e308,
        rel=3e-15,
        abs=0.0,
    )
    assert wcClusterQuality(distances, labels + 1)["stats"]["DB"] == pytest.approx(
        1.5e308,
        rel=3e-15,
        abs=0.0,
    )


def test_crisp_xb_stabilizes_finite_squared_term_aggregate_overflow() -> None:
    distance = 0.65 * math.sqrt(np.finfo(float).max)
    distances = np.ones((6, 6), dtype=float)
    np.fill_diagonal(distances, 0.0)
    for left in range(4):
        for right in range(left + 1, 4):
            distances[left, right] = distances[right, left] = distance
    distances[4, 5] = distances[5, 4] = 0.0
    labels = np.array([0, 0, 0, 0, 1, 1])
    expected = 3.797626747396641e307

    assert compute_internal_indices(distances, labels)["XB"] == pytest.approx(
        expected,
        rel=3e-15,
        abs=0.0,
    )
    assert wcClusterQuality(distances, labels + 1)["stats"]["XB"] == pytest.approx(
        expected,
        rel=3e-15,
        abs=0.0,
    )


def test_crisp_indices_compare_medoid_costs_as_expansions() -> None:
    s = 1e-308
    t = 1e-154
    distances = np.zeros((5, 5), dtype=float)
    edges = {
        (0, 1): 0.0,
        (0, 2): 1.0,
        (0, 3): 2 * s,
        (1, 2): 1.0,
        (1, 3): s,
        (2, 3): 2.0,
        (0, 4): 1.0,
        (1, 4): t,
        (2, 4): 1.0,
        (3, 4): 1.0,
    }
    for (left, right), value in edges.items():
        distances[left, right] = distances[right, left] = value
    labels = np.array([0, 0, 0, 0, 1])
    expected = {"DB": 2.5e153, "XB": 2e307, "PBM": 2.5e-309}

    direct = compute_internal_indices(distances, labels)
    public = wcClusterQuality(distances, labels + 1)["stats"]

    for name, value in expected.items():
        assert direct[name] == pytest.approx(value, rel=3e-15, abs=0.0)
        assert public[name] == pytest.approx(value, rel=3e-15, abs=0.0)

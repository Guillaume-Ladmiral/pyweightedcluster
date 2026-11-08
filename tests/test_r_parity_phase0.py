from __future__ import annotations

import numpy as np
import pytest

from tests.r_parity_utils import fail_or_skip, fail_parity
from weightedcluster import as_clustrange, wcClusterQuality, wcKMedoids, wcSilhouetteObs

pytest_plugins = ["tests.r_parity_utils"]

REFERENCE_STATS = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")


def canonical_partition(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    groups = []
    for c in np.unique(arr):
        groups.append(tuple(int(i) for i in np.where(arr == c)[0]))
    return tuple(sorted(groups))


def _check_close(
    py_v: float, r_v: float, *, metric: str, atol: float = 1e-6, rtol: float = 1e-5
) -> None:
    if np.isfinite(py_v) and np.isfinite(r_v) and np.isclose(py_v, r_v, atol=atol, rtol=rtol):
        return
    if (not np.isfinite(py_v)) and (not np.isfinite(r_v)):
        return
    fail_parity(
        f"Parity gap for {metric}: python={py_v:.12g}, r={r_v:.12g}, atol={atol}, rtol={rtol}"
    )


def _check_matrix_close(
    py_m: np.ndarray, r_m: np.ndarray, *, label: str, atol: float = 1e-6, rtol: float = 1e-5
) -> None:
    if py_m.shape != r_m.shape:
        fail_parity(f"{label} shape mismatch: python={py_m.shape}, r={r_m.shape}")
        return
    if np.allclose(py_m, r_m, atol=atol, rtol=rtol, equal_nan=True):
        return
    max_abs = float(np.nanmax(np.abs(py_m - r_m)))
    fail_parity(
        f"{label} mismatch: max_abs_diff={max_abs:.6g}, atol={atol}, rtol={rtol}, "
        f"python_shape={py_m.shape}, r_shape={r_m.shape}"
    )


def _direct_cluster_quality(
    backend: str, D: np.ndarray, labels_1b: np.ndarray, weights: np.ndarray
) -> tuple[dict, dict, dict]:
    _levels, labels0 = np.unique(labels_1b, return_inverse=True)
    labels0 = labels0.astype(np.int32)
    if backend == "core":
        try:
            from weightedcluster._wc_core import cluster_quality
        except ImportError:
            fail_or_skip("compiled core quality backend unavailable")
        return cluster_quality(D, labels0, weights=weights)
    if backend == "numba":
        try:
            from weightedcluster._numba import cluster_quality_numba
        except ImportError:
            fail_or_skip("numba quality backend unavailable")
        return cluster_quality_numba(D, labels0, weights=weights)
    if backend == "pure":
        from weightedcluster._fallback import cluster_quality

        return cluster_quality(D, labels0, weights=weights)
    raise ValueError(f"unknown quality backend {backend!r}")


def test_phase0_parity_wckmedoids_partition_and_stats(r_parity_session):
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
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    initial_medoids_1b = np.array([1, 5], dtype=int)  # 1-based for R, accepted by Python too

    py = wcKMedoids(
        D,
        2,
        weights=weights,
        npass=1,
        initialclust=initial_medoids_1b,
        random_state=42,
    )
    r = r_parity_session.wc_kmedoids(
        D,
        k=2,
        weights=weights,
        initialclust=initial_medoids_1b,
        method="PAMonce",
        npass=0,
    )

    py_part = canonical_partition(np.asarray(py["clustering"], dtype=int))
    r_part = canonical_partition(np.asarray(r["clustering"], dtype=int))
    if py_part != r_part:
        fail_parity(f"wcKMedoids partition mismatch: python={py_part}, r={r_part}")

    for key in REFERENCE_STATS:
        py_v = float(py["stats"].get(key, np.nan))
        r_v = float(r["stats"].get(key, np.nan))
        _check_close(py_v, r_v, metric=f"wcKMedoids.stats[{key}]")

    py_asw = np.full((2, 2), np.nan, dtype=float)
    for i, c in enumerate(sorted(py["ASW"])):
        py_asw[i, 0] = float(py["ASW"].get(c, np.nan))
        py_asw[i, 1] = float(py["ASWw"].get(c, np.nan))
    _check_matrix_close(py_asw, r["asw"], label="wcKMedoids.ASW")


def test_phase0_parity_wcclusterquality_stats_and_asw(r_parity_session):
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
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    labels_1b = np.array([1, 1, 1, 2, 2, 2], dtype=int)

    py = wcClusterQuality(D, labels_1b, weights=weights)
    r = r_parity_session.wc_cluster_quality(D, labels_1b, weights=weights)

    for key in REFERENCE_STATS:
        py_v = float(py["stats"].get(key, np.nan))
        r_v = float(r["stats"].get(key, np.nan))
        _check_close(py_v, r_v, metric=f"wcClusterQuality.stats[{key}]")

    py_asw = np.full((2, 2), np.nan, dtype=float)
    for i, c in enumerate((1, 2)):
        py_asw[i, 0] = float(py["ASW"].get(c, np.nan))
        py_asw[i, 1] = float(py["ASWw"].get(c, np.nan))
    _check_matrix_close(py_asw, r["asw"], label="wcClusterQuality.ASW")


@pytest.mark.parametrize("backend", ["core", "numba", "pure"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unit_weights", "nonuniform_weights"])
def test_phase0_direct_cluster_quality_backends_match_live_r(
    r_parity_session, backend: str, weighted: bool
):
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
    weights = (
        np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
        if weighted
        else np.ones(6, dtype=float)
    )
    labels = np.array([2, 2, 2, 7, 7, 7], dtype=int)

    stats, asw, asww = _direct_cluster_quality(backend, D, labels, weights)
    r = r_parity_session.wc_cluster_quality(D, labels, weights=weights)

    for key in REFERENCE_STATS:
        _check_close(
            float(stats.get(key, np.nan)),
            float(r["stats"].get(key, np.nan)),
            metric=f"{backend}.cluster_quality[{key}]",
            atol=1e-10,
            rtol=1e-9,
        )
    py_asw = np.asarray(
        [[float(asw[c]), float(asww[c])] for c in sorted(asw)],
        dtype=float,
    )
    _check_matrix_close(
        py_asw,
        r["asw"],
        label=f"{backend}.cluster_quality.ASW",
        atol=1e-10,
        rtol=1e-9,
    )


@pytest.mark.parametrize("backend", ["core", "numba", "pure"])
def test_phase0_direct_cluster_quality_singleton_matches_live_r(r_parity_session, backend: str):
    D = np.array(
        [
            [0.0, 1.0, 2.0, 5.0, 6.0],
            [1.0, 0.0, 1.0, 4.0, 5.0],
            [2.0, 1.0, 0.0, 3.0, 4.0],
            [5.0, 4.0, 3.0, 0.0, 1.0],
            [6.0, 5.0, 4.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([2, 2, 2, 7, 11], dtype=int)
    weights = np.ones(5, dtype=float)
    stats, asw, asww = _direct_cluster_quality(backend, D, labels, weights)
    r = r_parity_session.wc_cluster_quality(D, labels, weights=weights)

    for key in REFERENCE_STATS:
        _check_close(
            float(stats.get(key, np.nan)),
            float(r["stats"].get(key, np.nan)),
            metric=f"{backend}.singleton.cluster_quality[{key}]",
            atol=1e-10,
            rtol=1e-9,
        )
    py_asw = np.asarray(
        [[float(asw[c]), float(asww[c])] for c in sorted(asw)],
        dtype=float,
    )
    _check_matrix_close(
        py_asw,
        r["asw"],
        label=f"{backend}.singleton.cluster_quality.ASW",
        atol=1e-10,
        rtol=1e-9,
    )


@pytest.mark.parametrize("backend", ["core", "numba", "pure"])
def test_phase0_zero_mass_cluster_quality_backends_match_live_r(r_parity_session, backend: str):
    D = np.array(
        [
            [0.0, 1.0, 3.0, 4.0],
            [1.0, 0.0, 2.0, 3.0],
            [3.0, 2.0, 0.0, 1.0],
            [4.0, 3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([1, 1, 2, 2], dtype=int)
    weights = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    stats, asw, asww = _direct_cluster_quality(backend, D, labels, weights)
    expected = r_parity_session.wc_cluster_quality(D, labels, weights=weights)

    for key in REFERENCE_STATS:
        _check_close(
            float(stats.get(key, np.nan)),
            float(expected["stats"].get(key, np.nan)),
            metric=f"{backend}.zero_mass.cluster_quality[{key}]",
            atol=1e-12,
            rtol=0.0,
        )
    actual_asw = np.asarray(
        [[float(asw[c]), float(asww[c])] for c in sorted(asw)],
        dtype=float,
    )
    _check_matrix_close(
        actual_asw,
        expected["asw"],
        label=f"{backend}.zero_mass.cluster_quality.ASW",
        atol=1e-12,
        rtol=0.0,
    )


def test_phase0_parity_wcsilhouetteobs(r_parity_session):
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
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    labels_1b = np.array([1, 1, 1, 2, 2, 2], dtype=int)

    py_asw = np.asarray(wcSilhouetteObs(D, labels_1b, weights=weights, measure="ASW"), dtype=float)
    py_asww = np.asarray(
        wcSilhouetteObs(D, labels_1b, weights=weights, measure="ASWw"), dtype=float
    )
    r_asw = r_parity_session.wc_silhouette_obs(D, labels_1b, weights=weights, measure="ASW")
    r_asww = r_parity_session.wc_silhouette_obs(D, labels_1b, weights=weights, measure="ASWw")

    _check_matrix_close(py_asw.reshape(-1, 1), r_asw.reshape(-1, 1), label="wcSilhouetteObs.ASW")
    _check_matrix_close(py_asww.reshape(-1, 1), r_asww.reshape(-1, 1), label="wcSilhouetteObs.ASWw")


def test_phase0_parity_as_clustrange_default_stats(r_parity_session):
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
    weights = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], dtype=float)
    cluster2 = np.array([1, 1, 1, 2, 2, 2], dtype=int)
    cluster3 = np.array([1, 1, 2, 2, 3, 3], dtype=int)

    py = as_clustrange(
        {"cluster2": cluster2, "cluster3": cluster3},
        D,
        weights=weights,
    )
    r = r_parity_session.as_clustrange_default(
        D,
        {"cluster2": cluster2, "cluster3": cluster3},
        weights=weights,
    )

    py_kvals = np.asarray(py["kvals"], dtype=int)
    r_kvals = np.asarray(r["kvals"], dtype=int)
    if not np.array_equal(py_kvals, r_kvals):
        fail_parity(f"as_clustrange kvals mismatch: python={py_kvals}, r={r_kvals}")

    r_stats_names = list(r["stats_names"])
    py_stats_matrix = np.column_stack(
        [np.asarray(py["stats"].get(name, []), dtype=float).reshape(-1) for name in r_stats_names]
    )
    _check_matrix_close(py_stats_matrix, r["stats_matrix"], label="as_clustrange.stats")

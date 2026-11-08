from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from weightedcluster import wcCmpCluster

from .r_parity_utils import canonical_partition, fail_parity

pytest_plugins = ["tests.r_parity_utils"]

REFERENCE_STATS = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")


def is_strict_parity_mode() -> bool:
    return str(os.environ.get("WC_PARITY_STRICT", "0")).lower() in {"1", "true", "yes", "on"}


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


def _distance_fixture() -> np.ndarray:
    X = np.array(
        [
            [0.0, 0.0],
            [0.3, 0.1],
            [1.0, 0.2],
            [3.0, 3.1],
            [3.2, 2.9],
            [3.8, 3.6],
            [7.0, 0.2],
            [7.4, 0.4],
        ],
        dtype=float,
    )
    return squareform(pdist(X, metric="euclidean"))


@pytest.mark.parametrize(
    ("method_py", "method_r"),
    [
        ("ward.D2", "ward.D2"),
        ("diana", "diana"),
        ("beta.flexible", "beta.flexible"),
    ],
)
def test_phase2_parity_wccmpcluster_methods(r_parity_session, method_py: str, method_r: str):
    D = _distance_fixture()
    maxcluster = 4

    py = wcCmpCluster(D, maxcluster=maxcluster, method=[method_py], pam_combine=False)
    r = r_parity_session.wc_cmp_cluster(
        D,
        maxcluster=maxcluster,
        method=method_r,
        pam_combine=False,
    )

    py_key = str(method_py)
    if py_key not in py:
        fail_parity(f"Python wcCmpCluster missing method key '{py_key}'")
        return

    py_kvals = np.asarray(py[py_key]["kvals"], dtype=int).reshape(-1)
    r_kvals = np.asarray(r["kvals"], dtype=int).reshape(-1)
    if not np.array_equal(py_kvals, r_kvals):
        fail_parity(f"kvals mismatch for {method_py}: python={py_kvals}, r={r_kvals}")
        return

    for k in py_kvals.tolist():
        py_labels = np.asarray(py[py_key]["clustering"][f"cluster{int(k)}"], dtype=int).reshape(-1)
        r_labels = np.asarray(r["clustering"][f"cluster{int(k)}"], dtype=int).reshape(-1)
        py_part = canonical_partition(py_labels)
        r_part = canonical_partition(r_labels)
        if py_part != r_part:
            fail_parity(
                f"Partition mismatch for {method_py} cluster{k}: python={py_part}, r={r_part}"
            )

    py_stats = py[py_key]["stats"]
    r_names = list(r["stats_names"])
    r_mat = np.asarray(r["stats_matrix"], dtype=float)
    py_k_to_i = {int(k): i for i, k in enumerate(py_kvals.tolist())}
    r_k_to_i = {int(k): i for i, k in enumerate(r_kvals.tolist())}

    for metric in REFERENCE_STATS:
        if metric not in py_stats or metric not in r_names:
            continue
        r_col = r_names.index(metric)
        py_vals = np.asarray(py_stats[metric], dtype=float).reshape(-1)
        for k in py_kvals.tolist():
            py_v = float(py_vals[py_k_to_i[int(k)]])
            r_v = float(r_mat[r_k_to_i[int(k)], r_col])
            _check_close(py_v, r_v, metric=f"wcCmpCluster[{method_py}].stats[{metric}].k{k}")


def test_phase2_parity_wccmpcluster_diana_equal_split_levels(r_parity_session):
    rng = np.random.default_rng(44_713)
    points = rng.uniform(-3.0, 3.0, size=(13, 6))
    distances = squareform(pdist(points))

    py = wcCmpCluster(distances, maxcluster=5, method="diana", pam_combine=False)
    r = r_parity_session.wc_cmp_cluster(
        distances,
        maxcluster=5,
        method="diana",
        pam_combine=False,
    )

    for k in range(2, 6):
        py_labels = np.asarray(py["diana"]["clustering"][f"cluster{k}"], dtype=int)
        r_labels = np.asarray(r["clustering"][f"cluster{k}"], dtype=int)
        assert canonical_partition(py_labels) == canonical_partition(r_labels)

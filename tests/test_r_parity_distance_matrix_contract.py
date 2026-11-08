"""One distance-matrix contract for the four public entry points (F-06, F-07).

The 2026-08-24 whole-codebase audit found three inconsistencies at once:

* an asymmetric or non-zero-diagonal square matrix raised `ValueError` from
  `wcClusterQuality`, `as_clustrange` and `wcKMedoids`, while R computed a
  value from it;
* `wcSilhouetteObs` accepted the same input and returned **different numbers
  from R**, with no error and no warning (F-06, S2); and
* a matrix with negative entries was accepted by `wcClusterQuality`, where it
  agrees with R exactly, and rejected by `as_clustrange` (F-07).

`ECOSYSTEM.md` makes WeightedCluster 2.0 the source of truth, so the contract
follows R: **only the lower triangle is data**, exactly as `as.dist` reads it.
Python adds one thing R does not — a `UserWarning` when the discarded half
disagrees with the lower triangle, because silently ignoring half a caller's
matrix is the failure mode this contract exists to make visible.

One qualification, measured rather than assumed. Every other helper in
`r_parity_utils` hands R `as.dist(D)`, so none of them can observe what R does
with a **bare matrix** — and R does not treat the two forms alike.

Measured (`test_python_follows_rs_dist_reading_where_rs_matrix_path_disagrees`):
on a symmetric matrix R's two input forms agree everywhere. On a *non-symmetric*
matrix they disagree, in all four entry points, and only in the
silhouette-derived quantities — `ASW` and `ASWw` for `wcClusterQuality` and
`as.clustrange`, the whole vector for `wcSilhouetteObs`, and the partition for
`wcKMedoids`. The mechanism is visible in the R source: the ordered-pair
statistics (`PBC`, `HG`, `HGSD`, `CH`, `R2`, `CHsq`, `R2sq`, `HC`) iterate
`j > i` and so read one triangle, while the silhouette loop runs over every
`j != i` and therefore reads both.

Python reduces to the lower triangle first, so it reproduces R's `dist` reading
in every cell. That is a deliberate divergence from R's *matrix* path, recorded
in `docs/R_MIGRATION_GUIDE.md` and pinned below so it cannot drift silently.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from weightedcluster import as_clustrange, wcClusterQuality, wcKMedoids, wcSilhouetteObs

from .r_parity_utils import canonical_partition

pytest_plugins = ["tests.r_parity_utils"]

#: R's ten `wcClusterQuality` statistics, in the order R returns them. Python's
#: public result adds four Python-only indices, so the comparison selects these.
R_STAT_ORDER = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")

BASE = np.array(
    [
        [0.0, 1.0, 4.0, 5.0],
        [1.0, 0.0, 3.0, 4.0],
        [4.0, 3.0, 0.0, 1.0],
        [5.0, 4.0, 1.0, 0.0],
    ]
)
CLUSTERING = np.array([1, 1, 2, 2])


def _asymmetric() -> np.ndarray:
    """Upper triangle corrupted; the lower triangle still carries `BASE`."""
    D = BASE.copy()
    D[0, 1] = 99.0
    return D


def _nonzero_diagonal() -> np.ndarray:
    D = BASE.copy()
    D[1, 1] = 3.0
    return D


def _negative() -> np.ndarray:
    D = BASE.copy()
    D[0, 1] = D[1, 0] = -2.0
    return D


@pytest.mark.parametrize(
    "factory", (_asymmetric, _nonzero_diagonal), ids=("asymmetric", "diagonal")
)
def test_all_four_entry_points_match_r_given_a_dist_object(r_parity_session, factory):
    """Python's reduction reproduces R exactly when R is handed `as.dist(D)`.

    Named for what it proves. The helpers used here pass `as.dist(D)` to R, so
    this establishes agreement with R's `dist` reading — the idiomatic input, and
    what `TraMineR::seqdist` returns. R's *matrix* path is a separate question,
    settled by `test_python_follows_rs_dist_reading_where_rs_matrix_path_disagrees`.
    """
    D = factory()

    r_quality = r_parity_session.wc_cluster_quality(D, CLUSTERING)
    r_silhouette = r_parity_session.wc_silhouette_obs(D, CLUSTERING)
    r_range = r_parity_session.as_clustrange_default(D, {"cluster2": CLUSTERING})
    r_medoids = r_parity_session.wc_kmedoids(D, k=2, initialclust=np.array([1, 3]))

    with pytest.warns(UserWarning, match="lower triangle"):
        quality = wcClusterQuality(D, CLUSTERING)
    with pytest.warns(UserWarning, match="lower triangle"):
        silhouette = wcSilhouetteObs(D, CLUSTERING)
    with pytest.warns(UserWarning, match="lower triangle"):
        clustrange = as_clustrange({"cluster2": CLUSTERING}, D)
    with pytest.warns(UserWarning, match="lower triangle"):
        medoids = wcKMedoids(D, 2, initialclust=np.array([1, 3]))

    for name, r_value in r_quality["stats"].items():
        assert np.isclose(
            float(quality["stats"][name]), float(r_value), rtol=0.0, atol=1e-12, equal_nan=True
        ), name
    np.testing.assert_allclose(silhouette, r_silhouette, rtol=0.0, atol=1e-12)
    for column, name in enumerate(r_range["stats_names"]):
        np.testing.assert_allclose(
            np.asarray(clustrange["stats"][name], dtype=float),
            r_range["stats_matrix"][:, column],
            rtol=0.0,
            atol=1e-12,
            err_msg=name,
        )
    assert canonical_partition(np.asarray(medoids["clustering"])) == canonical_partition(
        r_medoids["clustering"]
    )


def test_python_follows_rs_dist_reading_where_rs_matrix_path_disagrees(r_parity_session):
    """Pin the one place Python deliberately does not follow R: R's matrix path.

    R is not self-consistent here. Given a non-symmetric square **matrix**, R's
    silhouette loop reads both triangles while its ordered-pair statistics read
    one, so R's own answer differs between `D` and `as.dist(D)` in all four entry
    points — but only in the silhouette-derived components: `ASW` and `ASWw` for
    `wcClusterQuality` and `as.clustrange`, the whole vector for
    `wcSilhouetteObs`, and the partition for `wcKMedoids`. `PBC` and the other
    ordered-pair statistics agree either way.

    Python reduces to the lower triangle everywhere, so it reproduces R's `dist`
    reading in all eight cells.

    This test states that division explicitly, per component. It fails if Python
    starts following R's matrix path, if R's two paths ever converge, or if the
    set of affected components changes — any of which means the documented
    contract has to be revisited.
    """
    #: Which components of each R result change when R is handed the bare matrix
    #: instead of `as.dist` of it. Empty means R's two input forms agree.
    expected_matrix_vs_dist_differences = {
        ("wcClusterQuality", "symmetric"): (),
        ("wcClusterQuality", "asymmetric"): ("ASW", "ASWw"),
        ("as.clustrange", "symmetric"): (),
        ("as.clustrange", "asymmetric"): ("ASW", "ASWw"),
        ("wcSilhouetteObs", "symmetric"): (),
        ("wcSilhouetteObs", "asymmetric"): ("<whole vector>",),
        ("wcKMedoids", "symmetric"): (),
        ("wcKMedoids", "asymmetric"): ("<partition>",),
    }

    def differing_components(function_name: str, r_matrix, r_dist) -> tuple[str, ...]:
        if np.allclose(r_matrix, r_dist, rtol=0.0, atol=1e-12, equal_nan=True):
            return ()
        if function_name in ("wcClusterQuality", "as.clustrange"):
            return tuple(
                name
                for index, name in enumerate(R_STAT_ORDER)
                if not (np.isnan(r_matrix[index]) and np.isnan(r_dist[index]))
                and r_matrix[index] != r_dist[index]
            )
        return ("<whole vector>",) if function_name == "wcSilhouetteObs" else ("<partition>",)

    python_value = {
        "wcClusterQuality": lambda D: np.asarray(
            [wcClusterQuality(D, CLUSTERING)["stats"][name] for name in R_STAT_ORDER], dtype=float
        ),
        "wcSilhouetteObs": lambda D: np.asarray(wcSilhouetteObs(D, CLUSTERING), dtype=float),
        "as.clustrange": lambda D: np.asarray(
            [as_clustrange({"cluster2": CLUSTERING}, D)["stats"][name][0] for name in R_STAT_ORDER],
            dtype=float,
        ),
        "wcKMedoids": lambda D: np.asarray(
            wcKMedoids(D, 2, initialclust=np.array([1, 3]), cluster_only=True), dtype=float
        ),
    }

    cells = 0
    for input_name, D in (("symmetric", BASE), ("asymmetric", _asymmetric())):
        paths = r_parity_session.wc_matrix_versus_dist_paths(D, CLUSTERING)
        for function_name, forms in paths.items():
            cells += 1
            r_matrix, r_dist = forms["matrix"], forms["dist"]

            measured = differing_components(function_name, r_matrix, r_dist)
            assert measured == expected_matrix_vs_dist_differences[(function_name, input_name)], (
                f"R's matrix and dist paths for {function_name} on {input_name} input "
                f"differ in {measured}, expected "
                f"{expected_matrix_vs_dist_differences[(function_name, input_name)]} "
                f"(matrix={r_matrix}, dist={r_dist})"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                value = python_value[function_name](D)

            if function_name == "wcKMedoids":
                assert canonical_partition(value.astype(int)) == canonical_partition(
                    r_dist.astype(int)
                ), f"{function_name}/{input_name}: python must follow R's dist reading"
            else:
                np.testing.assert_allclose(
                    value,
                    r_dist,
                    rtol=0.0,
                    atol=1e-12,
                    err_msg=f"{function_name}/{input_name}: python must follow R's dist reading",
                )

    assert cells == 8


def test_a_symmetric_matrix_is_accepted_without_a_warning():
    """The warning must fire only when data is actually discarded."""
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            wcClusterQuality(BASE, CLUSTERING)
            wcSilhouetteObs(BASE, CLUSTERING)
            as_clustrange({"cluster2": CLUSTERING}, BASE)
            wcKMedoids(BASE, 2, initialclust=np.array([1, 3]))


def test_negative_distances_follow_r_where_the_computation_is_defined(r_parity_session):
    """`wcClusterQuality` and `as_clustrange` must agree with each other and R.

    `wcKMedoids` is the one documented narrowing: its compiled medoid search
    requires finite, non-negative distances, while R's PAM returns a partition.
    """
    D = _negative()

    r_quality = r_parity_session.wc_cluster_quality(D, CLUSTERING)
    r_range = r_parity_session.as_clustrange_default(D, {"cluster2": CLUSTERING})

    quality = wcClusterQuality(D, CLUSTERING)
    clustrange = as_clustrange({"cluster2": CLUSTERING}, D)

    for name, r_value in r_quality["stats"].items():
        assert np.isclose(
            float(quality["stats"][name]), float(r_value), rtol=0.0, atol=1e-12, equal_nan=True
        ), name
    for column, name in enumerate(r_range["stats_names"]):
        np.testing.assert_allclose(
            np.asarray(clustrange["stats"][name], dtype=float),
            r_range["stats_matrix"][:, column],
            rtol=0.0,
            atol=1e-12,
            err_msg=name,
        )

    with pytest.raises(ValueError, match="non-negative"):
        wcKMedoids(D, 2, initialclust=np.array([1, 3]))

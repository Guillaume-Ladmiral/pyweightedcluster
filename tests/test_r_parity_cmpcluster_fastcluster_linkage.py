"""`wcCmpCluster` linkage parity against R's own public wrapper (finding F-01).

`orig_R_pkg/WeightedCluster/NAMESPACE` contains ``importFrom("fastcluster",
"hclust")``, so ``R/clustrangefamily.R:27``'s unqualified ``hclust`` call runs
**fastcluster's** implementation.  The 2026-08-24 whole-codebase audit found
that Python reproduced ``stats::hclust`` instead, which returns different
partitions whenever the dissimilarities contain equal values.

The fixtures here are therefore unnormalised Hamming distances between
categorical state sequences: integer-valued, so ties are the normal case in this
project's domain rather than an edge case.  The suite's existing `wcCmpCluster`
fixture is eight continuous two-dimensional points whose Euclidean distances
carry no discriminating tie, which is why it could not see the defect.

The oracle is R's **public** ``WeightedCluster::wcCmpCluster``, never
``stats::hclust`` and never an unqualified ``hclust``, which resolves to
``stats::hclust`` in ``.GlobalEnv``.
"""

from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import wcCmpCluster

from .r_parity_utils import canonical_partition

pytest_plugins = ["tests.r_parity_utils"]

R_QUALITY_NAMES = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")
HCLUST_METHODS = (
    "ward.D",
    "ward.D2",
    "single",
    "complete",
    "average",
    "mcquitty",
    "median",
    "centroid",
)
MAXCLUSTER = 4


def _tie_rich_dissimilarity(seed: int, *, n_seq: int = 12, n_time: int = 8, n_states: int = 4):
    """Unnormalised Hamming distances between random categorical trajectories.

    Computed here rather than through the distance layer so the fixture stays
    independent of traminex: the property under test is the linkage, not the
    distance.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_states, size=(n_seq, n_time))
    D = (X[:, None, :] != X[None, :, :]).sum(axis=2).astype(float)
    assert np.array_equal(D, D.T)
    assert np.all(np.diag(D) == 0.0)
    # The fixture is only meaningful if it actually contains ties.
    upper = D[np.triu_indices(n_seq, k=1)]
    assert np.unique(upper).size < upper.size
    return D


@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("seed", (11, 22, 33))
def test_cmp_cluster_hclust_methods_match_live_r_on_tie_rich_dissimilarities(
    r_parity_session,
    seed: int,
    weighted: bool,
):
    D = _tie_rich_dissimilarity(seed)
    weights = None
    if weighted:
        weights = np.asarray(
            np.random.default_rng(seed + 1000).integers(1, 6, size=D.shape[0]), dtype=float
        )

    expected = r_parity_session.wc_cmp_cluster_complete_contract(
        D,
        maxcluster=MAXCLUSTER,
        method="all",
        weights=weights,
        pam_combine=False,
    )
    assert expected["error"] is None

    actual = wcCmpCluster(
        D,
        maxcluster=MAXCLUSTER,
        method="all",
        weights=weights,
        pam_combine=False,
    )

    # R drops `diana` and `beta.flexible` when weights are supplied; Python must
    # select the same family.
    assert actual["param"]["all.methods"] == expected["param_all_methods"]

    case = f"seed={seed} weighted={weighted}"
    for method_name in HCLUST_METHODS:
        method_actual = actual[method_name]
        method_expected = expected["methods"][method_name]
        for column, k in enumerate(range(2, MAXCLUSTER + 1)):
            python_partition = canonical_partition(method_actual["clustering"][f"cluster{k}"])
            r_partition = canonical_partition(method_expected["clustering"][:, column])
            assert python_partition == r_partition, f"{method_name} k={k} {case}"
        np.testing.assert_allclose(
            np.column_stack([method_actual["stats"][name] for name in R_QUALITY_NAMES]),
            method_expected["stats"],
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
            err_msg=f"{method_name} statistics, {case}",
        )


@pytest.mark.parametrize("method", HCLUST_METHODS)
def test_linkage_reproduces_fastcluster_merge_order_and_heights(r_parity_session, method: str):
    """Compare the hierarchy itself, not only the partitions it is cut into.

    Two hierarchies can agree at every cut this project takes and still differ,
    so the merge order and the merge heights are compared directly against
    ``fastcluster::hclust`` — the binding WeightedCluster's namespace holds.
    """
    from weightedcluster._fastcluster_linkage import linkage_fastcluster

    D = _tie_rich_dissimilarity(44, n_seq=10)
    members = np.asarray(np.random.default_rng(7).integers(1, 5, size=D.shape[0]), dtype=float)
    for weights in (None, members):
        expected = r_parity_session.fastcluster_hclust_contract(
            D,
            method=method,
            maxcluster=5,
            members=weights,
        )
        hierarchy = linkage_fastcluster(D, method=method.lower(), members=weights)

        np.testing.assert_allclose(
            hierarchy[:, 2], expected["height"], rtol=0.0, atol=1e-9, err_msg=method
        )
        for column, k in enumerate(range(2, 6)):
            python_partition = canonical_partition(_cut(hierarchy, k))
            r_partition = canonical_partition(expected["cuts"][:, column])
            assert python_partition == r_partition, f"{method} k={k}"


def test_nan_dissimilarities_are_refused_as_r_refuses_them(r_parity_session):
    """R raises `NaN dissimilarity value.`; the port must too.

    The public distance contract (findings F-06, F-07) accepts non-finite
    entries so `wcClusterQuality` can propagate them as R does, so the refusal
    has to live where R's lives — inside the linkage.
    """
    from weightedcluster import wcCmpCluster
    from weightedcluster._fastcluster_linkage import linkage_fastcluster

    D = _tie_rich_dissimilarity(11, n_seq=6)
    D[0, 1] = D[1, 0] = np.nan

    r_result = r_parity_session.wc_cmp_cluster_complete_contract(
        D, maxcluster=3, method="complete", pam_combine=False
    )
    assert r_result["error"] == "NaN dissimilarity value."

    with pytest.raises(ValueError, match="NaN dissimilarity value"):
        linkage_fastcluster(D, method="complete")
    with pytest.raises(ValueError, match="NaN dissimilarity value"):
        wcCmpCluster(D, maxcluster=3, method="complete", pam_combine=False)


def _cut(hierarchy: np.ndarray, k: int) -> np.ndarray:
    from weightedcluster._validation import cut_linkage_by_merge_count

    return cut_linkage_by_merge_count(hierarchy, n_clusters=k)

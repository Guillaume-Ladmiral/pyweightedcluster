from __future__ import annotations

import itertools
import json
import sys
import types

import numpy as np
import pytest

from tests.r_parity_utils import fail_or_skip
from weightedcluster import get_backend_info
from weightedcluster._fallback import weighted_medoids as py_weighted_medoids
from weightedcluster._seqdist import sequences_distance

R_QUALITY_KEYS = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")


def _implementation(info: dict, capability: str, backend: str, symbol: str | None = None) -> dict:
    matches = [
        item
        for item in info["capabilities"][capability]["implementations"]
        if item["backend"] == backend and (symbol is None or item["symbol"] == symbol)
    ]
    assert len(matches) == 1
    return matches[0]


def test_backend_info_preserves_legacy_strings_and_adds_json_safe_capabilities():
    info = get_backend_info()

    assert set(info) == {"compiled_core", "numba", "pure_python", "capabilities"}
    assert info["compiled_core"] in {"available", "unavailable"}
    assert isinstance(info["numba"], str)
    assert info["pure_python"] == "available"
    capabilities = info["capabilities"]
    assert list(capabilities) == sorted(capabilities)
    assert set(capabilities) == {
        "weighted_medoids",
        "cluster_quality",
        "observation_silhouette",
        "seqnull_stateindep",
        "seqnull_markov",
        "seqnull_spell_models",
        "seqnull_userpos_time_varying",
        "sequence_distances",
        # Finding F-10: the hierarchical-linkage selection had no registry
        # entry, so get_backend_info() under-reported which implementation
        # produced a caller's hierarchy.
        "hierarchical_linkage",
    }
    assert json.loads(json.dumps(info, sort_keys=True)) == info
    for capability in capabilities.values():
        assert isinstance(capability["owner"], str)
        assert isinstance(capability["implementations"], list)
        assert isinstance(capability["public_dispatch"], list)
        assert isinstance(capability["selection"], str)
        for implementation in capability["implementations"]:
            assert set(implementation) == {
                "backend",
                "symbol",
                "importable",
                "callable",
                "publicly_selected",
            }
            assert isinstance(implementation["importable"], bool)
            assert isinstance(implementation["callable"], bool)
            assert isinstance(implementation["publicly_selected"], bool)


def test_backend_info_reports_missing_core_module_per_symbol(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", None)

    info = get_backend_info()

    assert info["compiled_core"] == "unavailable"
    for capability in (
        "weighted_medoids",
        "cluster_quality",
    ):
        implementation = _implementation(info, capability, "compiled")
        assert implementation["importable"] is False
        assert implementation["callable"] is False


def test_backend_info_distinguishes_importable_core_from_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_core = types.ModuleType("weightedcluster._wc_core")

    def _must_not_execute(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("get_backend_info executed a kernel")

    fake_core.weighted_medoids = _must_not_execute
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", fake_core)

    info = get_backend_info()

    assert info["compiled_core"] == "available"
    assert _implementation(info, "weighted_medoids", "compiled")["callable"] is True
    quality = _implementation(info, "cluster_quality", "compiled")
    assert quality["importable"] is True
    assert quality["callable"] is False


def test_backend_info_reports_absent_numba_without_affecting_other_backends(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", None)

    info = get_backend_info()

    assert info["numba"] == "unavailable"
    for capability in (
        "cluster_quality",
        "seqnull_stateindep",
        "seqnull_markov",
    ):
        implementation = _implementation(info, capability, "numba")
        assert implementation["importable"] is False
        assert implementation["callable"] is False
    assert _implementation(info, "weighted_medoids", "pure_python")["callable"] is True


def test_backend_info_reports_public_selection_and_ownership_truthfully():
    info = get_backend_info()
    capabilities = info["capabilities"]

    medoids = capabilities["weighted_medoids"]
    assert medoids["public_dispatch"] == ["compiled", "pure_python"]
    assert [item["backend"] for item in medoids["implementations"]] == [
        "compiled",
        "pure_python",
    ]
    assert capabilities["observation_silhouette"]["public_dispatch"] == ["python"]
    assert capabilities["observation_silhouette"]["selection"] == "python_only"
    assert not any(
        item["backend"] == "compiled"
        for name in (
            "seqnull_stateindep",
            "seqnull_markov",
            "seqnull_spell_models",
            "seqnull_userpos_time_varying",
        )
        for item in capabilities[name]["implementations"]
    )
    distances = capabilities["sequence_distances"]
    assert distances["owner"] == distances["provider"] == "traminex"
    assert distances["selection"] == "delegated"


def _toy_sequences() -> np.ndarray:
    # Three separated groups to avoid tie-heavy medoid choices.
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [2, 2, 2, 2],
            [2, 2, 2, 1],
            [2, 2, 1, 1],
        ],
        dtype=int,
    )


def _canonical_partition(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    groups = []
    for c in np.unique(arr):
        groups.append(tuple(int(i) for i in np.where(arr == c)[0]))
    return tuple(sorted(groups))


def _assert_close_or_nan(a: float, b: float, *, atol: float = 1e-6, rtol: float = 1e-5) -> None:
    if np.isnan(a) and np.isnan(b):
        return
    assert np.isclose(float(a), float(b), atol=atol, rtol=rtol), (a, b)


def _cluster_quality_with_backend(
    *,
    backend: str,
    D: np.ndarray,
    labels0: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict, dict, dict]:
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
            fail_or_skip("numba backend unavailable")
        return cluster_quality_numba(D, labels0, weights=weights)

    if backend == "pure":
        from weightedcluster._fallback import cluster_quality

        return cluster_quality(D, labels0, weights=weights)

    raise ValueError(f"unknown backend {backend!r}")


def _weighted_medoids_with_backend(
    *, backend: str, D: np.ndarray, k: int, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    if backend == "pure":
        return py_weighted_medoids(D, k, weights=weights, max_iter=80, n_init=5, random_state=37)
    if backend == "core":
        try:
            from weightedcluster._wc_core import weighted_medoids
        except ImportError:
            fail_or_skip("compiled core medoids backend unavailable")
        return weighted_medoids(D, k, weights=weights, max_iter=80, n_init=5, random_state=37)
    raise ValueError(f"unknown backend {backend!r}")


def _available_weighted_medoids_results(
    *, D: np.ndarray, k: int, weights: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    results: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for backend in ("pure", "core"):
        try:
            results[backend] = _weighted_medoids_with_backend(
                backend=backend,
                D=D,
                k=k,
                weights=weights,
            )
        except pytest.skip.Exception:
            continue
    return results


def _available_quality_results(
    *,
    D: np.ndarray,
    labels0: np.ndarray,
    weights: np.ndarray,
) -> dict[str, tuple[dict, dict, dict]]:
    results: dict[str, tuple[dict, dict, dict]] = {}
    for backend in ("pure", "numba", "core"):
        try:
            results[backend] = _cluster_quality_with_backend(
                backend=backend,
                D=D,
                labels0=labels0,
                weights=weights,
            )
        except pytest.skip.Exception:
            continue
    return results


def test_backend_parity_weighted_medoids_core_vs_pure():
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    w = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.1, 0.95], dtype=float)

    labels_p, meds_p, obj_p = _weighted_medoids_with_backend(backend="pure", D=D, k=3, weights=w)
    labels_c, meds_c, obj_c = _weighted_medoids_with_backend(backend="core", D=D, k=3, weights=w)

    _assert_close_or_nan(float(obj_c), float(obj_p), atol=1e-8, rtol=1e-6)
    assert _canonical_partition(labels_c) == _canonical_partition(labels_p)
    assert meds_p.shape == meds_c.shape == (3,)


@pytest.mark.parametrize("key", R_QUALITY_KEYS)
@pytest.mark.parametrize("weighted", [False, True], ids=["unit_weights", "nonuniform_weights"])
def test_backend_parity_cluster_quality_core_vs_pure(key: str, weighted: bool):
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    weights = (
        np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.1, 0.95], dtype=float)
        if weighted
        else np.ones(9, dtype=float)
    )

    stats_pure, _asw_pure, _asww_pure = _cluster_quality_with_backend(
        backend="pure", D=D, labels0=labels0, weights=weights
    )
    stats_core, _asw_core, _asww_core = _cluster_quality_with_backend(
        backend="core", D=D, labels0=labels0, weights=weights
    )

    _assert_close_or_nan(
        float(stats_core.get(key, np.nan)),
        float(stats_pure.get(key, np.nan)),
        atol=1e-10,
        rtol=1e-9,
    )


@pytest.mark.parametrize("weighted", [False, True], ids=["unit_weights", "nonuniform_weights"])
def test_backend_parity_cluster_quality_numba_vs_pure(weighted: bool):
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    weights = (
        np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.1, 0.95], dtype=float)
        if weighted
        else np.ones(9, dtype=float)
    )

    stats_pure, _asw_pure, _asww_pure = _cluster_quality_with_backend(
        backend="pure", D=D, labels0=labels0, weights=weights
    )
    stats_nb, _asw_nb, _asww_nb = _cluster_quality_with_backend(
        backend="numba", D=D, labels0=labels0, weights=weights
    )

    for key in R_QUALITY_KEYS:
        _assert_close_or_nan(
            float(stats_nb.get(key, np.nan)),
            float(stats_pure.get(key, np.nan)),
            atol=1e-10,
            rtol=1e-9,
        )


def test_backend_parity_weighted_medoids_all_available_backends_align():
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    weights = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.1, 0.95], dtype=float)
    results = _available_weighted_medoids_results(D=D, k=3, weights=weights)

    assert "pure" in results
    assert len(results) >= 2
    for left, right in itertools.combinations(sorted(results), 2):
        labels_l, meds_l, obj_l = results[left]
        labels_r, meds_r, obj_r = results[right]
        _assert_close_or_nan(float(obj_l), float(obj_r), atol=1e-8, rtol=1e-6)
        assert _canonical_partition(labels_l) == _canonical_partition(labels_r), (left, right)
        assert meds_l.shape == meds_r.shape == (3,)


def test_backend_parity_cluster_quality_all_available_backends_align():
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    weights = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.1, 0.95], dtype=float)
    results = _available_quality_results(
        D=D,
        labels0=labels0,
        weights=weights,
    )

    assert "pure" in results
    assert len(results) >= 2
    for left, right in itertools.combinations(sorted(results), 2):
        stats_left, asw_left, asww_left = results[left]
        stats_right, asw_right, asww_right = results[right]
        for key in R_QUALITY_KEYS:
            _assert_close_or_nan(
                float(stats_left.get(key, np.nan)),
                float(stats_right.get(key, np.nan)),
                atol=1e-10,
                rtol=1e-9,
            )
        assert asw_left.keys() == asw_right.keys()
        assert asww_left.keys() == asww_right.keys()
        np.testing.assert_allclose(list(asw_left.values()), list(asw_right.values()), atol=1e-10)
        np.testing.assert_allclose(list(asww_left.values()), list(asww_right.values()), atol=1e-10)


@pytest.mark.parametrize("backend", ["pure", "numba", "core"])
def test_backend_parity_cluster_quality_singleton_cluster(backend: str):
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
    labels0 = np.array([0, 0, 0, 1, 2], dtype=np.int32)
    weights = np.ones(5, dtype=float)
    stats, asw, asww = _cluster_quality_with_backend(
        backend=backend, D=D, labels0=labels0, weights=weights
    )
    core_stats, core_asw, core_asww = _cluster_quality_with_backend(
        backend="core", D=D, labels0=labels0, weights=weights
    )

    for key in R_QUALITY_KEYS:
        _assert_close_or_nan(float(stats[key]), float(core_stats[key]), atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(list(asw.values()), list(core_asw.values()), atol=1e-10)
    np.testing.assert_allclose(list(asww.values()), list(core_asww.values()), atol=1e-10)


@pytest.mark.parametrize("backend", ["pure", "numba", "core"])
def test_backend_parity_cluster_quality_zero_dissimilarity_nan_contract(backend: str):
    D = np.zeros((4, 4), dtype=float)
    labels0 = np.array([0, 0, 1, 1], dtype=np.int32)
    stats, _asw, _asww = _cluster_quality_with_backend(
        backend=backend,
        D=D,
        labels0=labels0,
        weights=np.ones(4, dtype=float),
    )

    for key in ("PBC", "HG", "CH", "CHsq", "HC"):
        assert np.isnan(float(stats[key])), (backend, key, stats[key])
    assert float(stats["HGSD"]) == 0.0


@pytest.mark.parametrize("backend", ["pure", "numba", "core"])
def test_direct_quality_backends_reject_malformed_kernel_inputs(backend: str):
    D = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="contiguous zero-based"):
        _cluster_quality_with_backend(
            backend=backend,
            D=D,
            labels0=np.array([0, 2]),
            weights=np.ones(2),
        )
    with pytest.raises(ValueError, match="finite"):
        _cluster_quality_with_backend(
            backend=backend,
            D=D,
            labels0=np.array([0, 1]),
            weights=np.array([1.0, np.nan]),
        )


@pytest.mark.parametrize("backend", ["pure", "core"])
def test_direct_medoid_backends_reject_nonfinite_kernel_inputs(backend: str):
    D = np.array([[0.0, np.inf], [np.inf, 0.0]])

    with pytest.raises(ValueError, match="finite"):
        _weighted_medoids_with_backend(
            backend=backend,
            D=D,
            k=2,
            weights=np.ones(2),
        )


def _extreme_range_case(label_base: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The WC-090202/WC-090301 audit reproduction, shared by its three tests.

    Active distances 1e-16/2e-16/3e-16 among three unit-weight observations,
    with a zero-weight fourth observation carrying 1e308 to everyone.
    """
    D = np.zeros((4, 4))
    D[0, 1] = D[1, 0] = 1e-16
    D[0, 2] = D[2, 0] = 2e-16
    D[1, 2] = D[2, 1] = 3e-16
    for j in range(3):
        D[3, j] = D[j, 3] = 1e308
    labels = np.array([0, 0, 1, 1]) + int(label_base)
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    return D, labels, weights


def _zero_over_zero_case(label_base: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The WC-090303 live-R-pinned reproduction, shared by its two tests.

    Observation 0 sits at distance zero from its own cluster and from every
    member of the other cluster, so its silhouette is 0/0 = NaN
    (audit/whole-codebase-audit-and-fix-2026-09-03/sweepB_case5b_liveR.log).
    """
    D = np.zeros((4, 4))
    D[1, 2] = D[2, 1] = 5.0
    D[1, 3] = D[3, 1] = 5.0
    D[2, 3] = D[3, 2] = 1.0
    labels = np.array([0, 0, 1, 1]) + int(label_base)
    weights = np.ones(4)
    return D, labels, weights


def test_numba_quality_matches_pure_when_zero_weight_rows_carry_extreme_distances():
    """2026-09-02 audit WC-090202: the Numba wrapper rescaled by the whole-matrix
    maximum (including zero-weight rows) before rank bucketing, collapsing
    distinct active distances and corrupting HGSD/PBC/CH/R2. The escape hatch
    already guarantees the positive-weight submatrix is range-safe, so the raw
    distances must reach the kernel unscaled, as the compiled and pure kernels do.
    """
    from weightedcluster._fallback import cluster_quality as pure_cluster_quality

    try:
        from weightedcluster._numba import cluster_quality_numba
    except ImportError:
        fail_or_skip("numba backend unavailable")

    D, labels, weights = _extreme_range_case()
    numba_stats, _numba_asw, _numba_asww = cluster_quality_numba(D, labels, weights=weights)
    pure_stats, _pure_asw, _pure_asww = pure_cluster_quality(D, labels, weights=weights)
    for name in ("PBC", "HG", "HGSD", "CH", "R2", "CHsq", "R2sq", "HC"):
        assert numba_stats[name] == pytest.approx(pure_stats[name], rel=1e-12, abs=1e-15), name
    assert numba_stats["HGSD"] == 1.0
    assert np.isfinite(numba_stats["PBC"])


def test_permissive_pure_quality_asw_survives_zero_weight_extreme_distance_row():
    """2026-09-03 audit WC-090301: the permissive pure kernel scaled its
    silhouette-loop distances by the whole-matrix maximum, so a zero-weight
    row carrying a far distance collapsed the active contributions to zero and
    returned ASW/ASWw = NaN where the compiled, Numba, and validated pure
    kernels return the R value. The scale decision must read only the
    distances the silhouette loop actually accumulates (rows with positive
    weight against columns with nonzero weight), as the compiled kernel's
    active-submatrix rule and the WC-090202 Numba fix already do.
    """
    from weightedcluster._fallback import cluster_quality as pure_cluster_quality
    from weightedcluster._fallback import cluster_quality_r_permissive

    D, labels, weights = _extreme_range_case()
    perm_stats, perm_asw, perm_asww = cluster_quality_r_permissive(D, labels, weights=weights)
    pure_stats, pure_asw, pure_asww = pure_cluster_quality(D, labels, weights=weights)
    for name in ("ASW", "ASWw"):
        assert perm_stats[name] == pytest.approx(pure_stats[name], rel=1e-12), name
    assert perm_stats["ASW"] == pytest.approx(0.7222222222222222)
    assert perm_stats["ASWw"] == pytest.approx(0.8611111111111112)
    for cluster in (1, 2):
        assert perm_asw[cluster] == pytest.approx(pure_asw[cluster], rel=1e-12)
        assert perm_asww[cluster] == pytest.approx(pure_asww[cluster], rel=1e-12)


def test_public_pure_tier_asw_survives_zero_weight_extreme_distance_row(monkeypatch):
    """WC-090301, public reachability: with the compiled and Numba tiers
    unavailable, ``wcClusterQuality`` selects the permissive pure kernel, so
    the corrupted ASW/ASWw reached the public API. The public pure branch must
    agree with the default branch on this input.
    """
    from weightedcluster import wcClusterQuality

    D, labels, weights = _extreme_range_case(label_base=1)
    reference = wcClusterQuality(D, labels, weights=weights)
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", None)
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", None)
    pure_branch = wcClusterQuality(D, labels, weights=weights)
    for name in ("ASW", "ASWw"):
        assert pure_branch["stats"][name] == pytest.approx(
            reference["stats"][name], rel=1e-12
        ), name
    assert np.isfinite(pure_branch["stats"]["ASW"])


def test_all_tiers_propagate_genuine_zero_over_zero_silhouettes_like_r():
    """2026-09-03 audit WC-090303: an observation at distance zero from every
    member of another cluster, with own-cluster mean zero, has silhouette
    0/0 = NaN. Live R (4.6.1 / WeightedCluster 2.0) propagates that NaN into
    the overall and per-cluster ASW/ASWw, and so do the compiled and
    permissive paths — but ``cluster_asw_core_compatible`` masked non-finite
    observation silhouettes, so the Numba and validated pure tiers returned
    finite aggregates and the public answer depended on which backend was
    installed. Live-R pinned values: stats ASW = ASWw = NaN; per-cluster ASW
    {NaN, 0.6}; ASWw {NaN, 0.8}.
    """
    from weightedcluster import _fallback

    kernels = [
        ("valid", _fallback.cluster_quality),
        ("perm", _fallback.cluster_quality_r_permissive),
    ]
    try:
        from weightedcluster import _wc_core as core
    except ImportError:
        fail_or_skip("compiled core quality backend unavailable")
    else:
        kernels.append(("core", core.cluster_quality))
    try:
        from weightedcluster import _numba
    except ImportError:
        fail_or_skip("numba backend unavailable")
    else:
        kernels.append(("numba", _numba.cluster_quality_numba))

    D, labels, weights = _zero_over_zero_case()
    for tag, kernel in kernels:
        stats, asw, asww = kernel(D, labels, weights=weights)
        assert np.isnan(stats["ASW"]), tag
        assert np.isnan(stats["ASWw"]), tag
        assert np.isnan(asw[1]), tag
        assert asw[2] == pytest.approx(0.6), tag
        assert np.isnan(asww[1]), tag
        assert asww[2] == pytest.approx(0.8), tag


def test_public_asw_zero_over_zero_result_is_scale_invariant():
    """WC-090303, public consequence: the same partition of the same geometry
    must not flip between NaN and a finite ASW under a common positive
    distance rescaling that only moves the input into the range-risk path.
    """
    from weightedcluster import wcClusterQuality

    D, labels, _weights = _zero_over_zero_case(label_base=1)
    ordinary = wcClusterQuality(D, labels)
    huge = wcClusterQuality(D * 1e305, labels)
    assert np.isnan(ordinary["stats"]["ASW"])
    assert np.isnan(huge["stats"]["ASW"])
    assert np.isnan(ordinary["stats"]["ASWw"])
    assert np.isnan(huge["stats"]["ASWw"])


def _perfect_clusters_at_scale(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Two perfectly separated clusters: zero within, ``scale`` between."""
    D = np.zeros((4, 4))
    for i in (0, 1):
        for j in (2, 3):
            D[i, j] = D[j, i] = scale
    return D, np.array([0, 0, 1, 1])


def test_exact_range_risk_statistics_divide_by_zero_like_r():
    """2026-09-06 finding WC-090306: R's kernel divides CH's zero within-cluster
    dispersion in double arithmetic and returns ``Inf`` (live R 4.6.1:
    audit/whole-codebase-audit-and-fix-2026-09-03/wc090304_liveR_zero_residual.log),
    and so do the float tiers on ordinary scales — but the exact range-risk
    path guarded ``ss_res == 0`` into NaN, so a range-risk input with perfect
    clusters returned NaN CH/CHsq from every tier that delegates to it,
    including the public compiled route.
    """
    from weightedcluster import wcClusterQuality
    from weightedcluster._fallback import _exact_pair_rank_statistics

    D, labels = _perfect_clusters_at_scale(1.0)
    ordinary = wcClusterQuality(D, labels + 1)["stats"]
    assert ordinary["CH"] == np.inf and ordinary["CHsq"] == np.inf
    exact = _exact_pair_rank_statistics(D, labels, np.ones(4))
    assert exact["CH"] == np.inf and exact["CHsq"] == np.inf
    assert exact["R2"] == pytest.approx(1.0) and exact["R2sq"] == pytest.approx(1.0)

    D_big, labels = _perfect_clusters_at_scale(1e308 / 4)
    public = wcClusterQuality(D_big, labels + 1)["stats"]
    assert public["CH"] == np.inf and public["CHsq"] == np.inf
    assert public["R2"] == pytest.approx(1.0) and public["R2sq"] == pytest.approx(1.0)


_WC090304_CASES = {
    # kind 2 of the sweep-B corpus: mixed binary64 extremes; CH/R2/CHsq/R2sq
    # lost cancellation in the float moments (CHsq/R2sq collapsed to 0.0).
    "mixed-extremes": (
        [
            [0.0, 1e12, 1.0, 1.0],
            [1e12, 0.0, 0.0, 1e12],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1e12, 0.0, 0.0],
        ],
        [2, 0, 1, 0],
        [1.0, 1.0, 1.0, 1.0],
    ),
    # kind 4: subnormal distances; ASW halved and ASWw pushed to ~1 by the
    # float silhouette loop.
    "subnormal-weighted": (
        [
            [0.0, 1e-300, 1.0, 5e-324, 5e-324],
            [1e-300, 0.0, 5e-324, 5e-324, 5e-324],
            [1.0, 5e-324, 0.0, 1.0, 5e-324],
            [5e-324, 5e-324, 1.0, 0.0, 5e-324],
            [5e-324, 5e-324, 5e-324, 5e-324, 0.0],
        ],
        [0, 2, 1, 2, 2],
        [1.0, 1000.0, 1000.0, 0.001, 1000.0],
    ),
    "subnormal-unit": (
        [
            [0.0, 5e-324, 5e-324, 1.0, 1.0, 1.0, 1e-300],
            [5e-324, 0.0, 1.0, 1e-300, 1e-300, 1.0, 5e-324],
            [5e-324, 1.0, 0.0, 1e-300, 1e-300, 5e-324, 5e-324],
            [1.0, 1e-300, 1e-300, 0.0, 1.0, 5e-324, 1.0],
            [1.0, 1e-300, 1e-300, 1.0, 0.0, 1.0, 5e-324],
            [1.0, 1.0, 5e-324, 5e-324, 1.0, 0.0, 1e-300],
            [1e-300, 5e-324, 5e-324, 1.0, 5e-324, 1e-300, 0.0],
        ],
        [2, 2, 1, 0, 3, 0, 3],
        [1.0] * 7,
    ),
}

_ALL_TEN = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")


def _assert_same_statistics(left: dict, right: dict, tag: str) -> None:
    for name in _ALL_TEN:
        a, b = left[name], right[name]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), (tag, name, a, b)
        elif np.isinf(a) or np.isinf(b):
            assert a == b, (tag, name, a, b)
        else:
            assert a == pytest.approx(b, rel=1e-9, abs=1e-300), (tag, name, a, b)


@pytest.mark.parametrize("case", sorted(_WC090304_CASES))
def test_permissive_kernel_matches_validated_kernel_on_range_risk_inputs(case):
    """2026-09-06 finding WC-090304 (the demonstrated 2026-09-02 Suspected
    lead (a)): on validated-domain inputs whose distances or pair masses leave
    the binary64 range, the permissive kernel kept R's float arithmetic while
    the compiled, Numba, and validated tiers route to exact statistics — so the
    publicly selected pure tier disagreed with the other three. The kernel now
    takes the same exact route on that domain and keeps its bit-identical R
    mirror on ordinary inputs. Inputs are the smallest divergent cases of the
    sweep-B corpus (audit/.../wc090304_corpus_cases.json).
    """
    from weightedcluster._fallback import cluster_quality, cluster_quality_r_permissive

    D, labels, w = (np.asarray(v, dtype=float) for v in _WC090304_CASES[case])
    labels = labels.astype(int)
    perm_stats, perm_asw, perm_asww = cluster_quality_r_permissive(D, labels, weights=w)
    valid_stats, valid_asw, valid_asww = cluster_quality(D, labels, weights=w)
    _assert_same_statistics(perm_stats, valid_stats, case)
    for cluster in valid_asw:
        for name, left, right in (("ASW", perm_asw, valid_asw), ("ASWw", perm_asww, valid_asww)):
            a, b = left[cluster], right[cluster]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9), (
                case,
                name,
                cluster,
            )


def test_public_pure_tier_matches_compiled_tier_on_range_risk_inputs(monkeypatch):
    """WC-090304, public consequence: with the compiled core and Numba absent,
    ``wcClusterQuality`` selects the permissive kernel, so the public answer on
    these inputs depended on backend availability."""
    from weightedcluster import wcClusterQuality

    D, labels, w = (np.asarray(v, dtype=float) for v in _WC090304_CASES["subnormal-weighted"])
    labels = labels.astype(int) + 1
    reference = wcClusterQuality(D, labels, weights=w)["stats"]
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", None)
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", None)
    pure = wcClusterQuality(D, labels, weights=w)["stats"]
    _assert_same_statistics(pure, reference, "public-pure-vs-compiled")


def _zero_mass_other_cluster_case(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 2026-08-21 F-05 reproduction (zero-mass second cluster) at ``scale``."""
    x = np.array([0.0, 0.2, 1.0, 1.3, 2.4, 2.6])
    D = np.abs(x[:, None] - x[None, :]) * scale
    return D, np.array([0, 0, 0, 1, 1, 1]), np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("scale", [1.0, 1e-320])
def test_stable_silhouettes_keep_r_dbl_max_sentinel_for_zero_mass_other_clusters(scale):
    """2026-09-07 finding WC-090307: R leaves ``b_ik`` at ``DBL_MAX`` when every
    other cluster has zero mass, so each positive-weight observation gets
    silhouette 1 and the aggregate ASWw is 1 (live R 4.6.1 at both scales:
    audit/whole-codebase-audit-and-fix-2026-09-03/wc090307_liveR_sentinel_range_risk.log).
    The 2026-08-24 F-05 fix restored that sentinel only in the ordinary float
    branch of ``wcSilhouetteObs``; the exact stable branch — which every tier
    takes on range-risk inputs — returned NaN instead, so the public answer
    flipped from 1 to NaN under a positive rescaling of the distances.
    """
    from weightedcluster import wcClusterQuality, wcSilhouetteObs
    from weightedcluster._fallback import cluster_quality, cluster_quality_r_permissive

    D, labels, w = _zero_mass_other_cluster_case(scale)
    observed = wcSilhouetteObs(D, labels + 1, weights=w, measure="ASWw")
    np.testing.assert_array_equal(observed[:3], [1.0, 1.0, 1.0])
    assert np.all(np.isnan(observed[3:]))

    with pytest.warns(UserWarning, match="ASW can not be computed"):
        public = wcClusterQuality(D, labels + 1, weights=w)
    assert public["stats"]["ASWw"] == 1.0
    assert public["ASWw"][1] == 1.0 and np.isnan(public["ASWw"][2])

    for tier in (cluster_quality, cluster_quality_r_permissive):
        stats, _asw, asww = tier(D, labels, weights=w)
        assert stats["ASWw"] == 1.0, tier.__name__
        assert asww[1] == 1.0 and np.isnan(asww[2]), tier.__name__


# --------------------------------------------------------------------------
# Sweep-B cross-tier corpus (240 cases).
#
# The CHANGELOG's WC-090304 entry claims "the 240-case cross-tier corpus shows
# zero divergences across all four tiers after the change". Until 2026-09-21
# that claim rested on a generator that lived only in the gitignored evidence
# directory, so the claim was true but not re-runnable from the tracked tree
# (2026-09-19 audit, finding -003 — which also reported the below-one-mass ASW
# domain as unpinned; it is not, the sweep classifies it deliberately below).
# The generator is reproduced here verbatim, seed included, so the claim is now
# enforced by the suite instead of asserted in prose.
# --------------------------------------------------------------------------

_SWEEP_B_SEED = 20260903
_SWEEP_B_CASES = 240
#: Permissive-tier ASW/ASWw differences that are the documented F-19 contract:
#: the permissive kernel is the R bootstrap kernel and keeps R's unguarded
#: native ASW where every cluster mass is below one, while the other tiers
#: embed R's wrapper-level NA. Measured on 2026-09-21.
_SWEEP_B_EXPECTED_BELOW_ONE = 18


def _sweep_b_case(rng, index: int):
    """Reproduce one sweep-B case. Identical to the 2026-09-03 generator."""
    n = int(rng.integers(3, 12))
    kind = index % 6
    if kind == 0:
        matrix = rng.integers(0, 8, size=(n, n)).astype(float)
    elif kind == 1:
        matrix = rng.random((n, n))
    elif kind == 2:
        matrix = rng.choice([0.0, 1.0, 1e-12, 1e12], size=(n, n))
    elif kind == 3:
        matrix = np.round(rng.random((n, n)) * 3) / 3
    elif kind == 4:
        matrix = rng.choice([5e-324, 1e-300, 1.0], size=(n, n))
    else:
        matrix = rng.choice([0.0, 1e308 / n, 1.0], size=(n, n))
    D = np.tril(matrix, -1)
    D = D + D.T
    labels = rng.integers(0, min(3, n - 1) + 1, size=n)
    labels = np.unique(labels, return_inverse=True)[1]
    if labels.max() == 0:
        labels[0] = 1
        labels = np.unique(labels, return_inverse=True)[1]
    wkind = index % 4
    if wkind == 0:
        w = np.ones(n)
    elif wkind == 1:
        w = rng.integers(0, 4, size=n).astype(float)
    elif wkind == 2:
        w = rng.choice([1e-3, 1.0, 1e3], size=n)
    else:
        w = rng.random(n) * 10
    for cluster in range(labels.max() + 1):
        if w[labels == cluster].sum() <= 0.0:
            w[np.flatnonzero(labels == cluster)[0]] = 1.0
    return D, labels.astype(int), w


def _sweep_b_close(a: float, b: float) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isnan(a) != np.isnan(b):
        return False
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-300)


def test_sweep_b_cross_tier_corpus_has_zero_unexpected_divergences():
    """Re-run the full 240-case corpus the CHANGELOG's WC-090304 claim rests on.

    The compiled core is the reference; the Numba, validated pure, and
    permissive tiers must agree with it on all ten R statistics. The only
    permitted differences are the permissive tier's ASW/ASWw on cases where
    every cluster's weighted size is below one, which is the deliberately
    unguarded R bootstrap behaviour recorded at finding F-19.
    """
    try:
        from weightedcluster import _wc_core
    except ImportError:  # pragma: no cover - compiled core is the reference
        pytest.skip("compiled core unavailable; it is this sweep's reference tier")
    from weightedcluster import _fallback
    from weightedcluster._quality_contract import has_cluster_weight_below_one

    tiers = [
        ("pure", _fallback.cluster_quality),
        ("perm", _fallback.cluster_quality_r_permissive),
    ]
    try:
        from weightedcluster import _numba

        tiers.insert(0, ("numba", _numba.cluster_quality_numba))
    except ImportError:  # pragma: no cover - optional extra
        pass

    rng = np.random.default_rng(_SWEEP_B_SEED)
    divergences: list[str] = []
    below_one_differences = 0
    for index in range(_SWEEP_B_CASES):
        D, labels, w = _sweep_b_case(rng, index)
        below_one = has_cluster_weight_below_one(labels, w)
        reference = _wc_core.cluster_quality(D, labels, weights=w)[0]
        for tier, kernel in tiers:
            stats = kernel(D, labels, weights=w)[0]
            for name in _ALL_TEN:
                if _sweep_b_close(reference[name], stats[name]):
                    continue
                if tier == "perm" and below_one and name in ("ASW", "ASWw"):
                    below_one_differences += 1
                    continue
                divergences.append(
                    f"case={index} kind={index % 6} wkind={index % 4} tier={tier} "
                    f"stat={name} core={reference[name]!r} {tier}={stats[name]!r}"
                )

    assert divergences == [], "\n".join(divergences)
    assert below_one_differences == _SWEEP_B_EXPECTED_BELOW_ONE

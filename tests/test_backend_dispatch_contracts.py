from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import textwrap
import types
from collections.abc import Callable

import numpy as np
import pytest

from tests.r_parity_utils import fail_or_skip
from weightedcluster._backend_contract import CAPABILITY_REGISTRY

EXPECTED_CAPABILITIES = {
    "weighted_medoids",
    "cluster_quality",
    "observation_silhouette",
    "seqnull_stateindep",
    "seqnull_markov",
    "seqnull_spell_models",
    "seqnull_userpos_time_varying",
    "sequence_distances",
    "hierarchical_linkage",
}


def _module(name: str, **symbols: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for symbol, value in symbols.items():
        setattr(module, symbol, value)
    return module


def _fallback_module(**symbols: object) -> types.ModuleType:
    """Mock ``weightedcluster._fallback`` keeping the real shared dyadic helpers.

    ``kmedoids`` imports its exact-objective helpers from ``_fallback``
    (2026-09-02 audit, finding WC-090203 deduplicated them), so reloading
    ``kmedoids`` against a mocked fallback must still provide them.
    """
    from weightedcluster import _fallback as real_fallback

    helpers: dict[str, object] = {
        name: getattr(real_fallback, name)
        for name in (
            "_component_less",
            "_component_to_float",
            "_pam_requires_exact_objectives",
            "_product_component",
            "_sum_components",
            "_weighted_medoids_objective_component",
        )
    }
    helpers.update(symbols)
    return _module("weightedcluster._fallback", **helpers)


def _distance_fixture() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.2, 3.0, 3.2],
            [0.2, 0.0, 2.8, 3.0],
            [3.0, 2.8, 0.0, 0.3],
            [3.2, 3.0, 0.3, 0.0],
        ],
        dtype=float,
    )


def _medoid_spy(
    calls: list[str], backend: str
) -> Callable[..., tuple[np.ndarray, np.ndarray, float]]:
    def _weighted_medoids(D: np.ndarray, k: int, **_kwargs: object):
        calls.append(backend)
        labels = np.arange(D.shape[0], dtype=np.int32) % int(k)
        medoids = np.arange(int(k), dtype=np.int32)
        return labels, medoids, float(backend == "pure")

    return _weighted_medoids


def _failing_medoid(backend: str, exception: type[Exception]) -> Callable[..., object]:
    def _weighted_medoids(*_args: object, **_kwargs: object) -> object:
        raise exception(f"{backend} medoid called")

    return _weighted_medoids


def _fake_quality(_D: np.ndarray, _labels: np.ndarray, **_kwargs: object) -> dict:
    names = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")
    return {"stats": {name: 0.0 for name in names}, "ASW": {}, "ASWw": {}}


def _quality_kernel_spy(calls: list[str], backend: str) -> Callable[..., tuple[dict, dict, dict]]:
    def _cluster_quality(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        calls.append(backend)
        names = ("PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC")
        return ({name: 0.0 for name in names}, {1: 0.0, 2: 0.0}, {1: 0.0, 2: 0.0})

    return _cluster_quality


def _run_call_time_medoid_consumer(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    D = _distance_fixture()
    weights = np.ones(D.shape[0], dtype=float)
    if name == "wcFuzzyRange":
        fuzzy = importlib.import_module("weightedcluster.fuzzy")
        quality = importlib.import_module("weightedcluster.quality")

        def _fake_fcmdd(_D: np.ndarray, u_init: np.ndarray, **_kwargs: object) -> dict[str, object]:
            return {
                "memb": u_init,
                "mobileCenters": list(range(u_init.shape[1])),
                "functional": 0.0,
                "dnoise": None,
            }

        monkeypatch.setattr(fuzzy, "fcmdd", _fake_fcmdd)
        monkeypatch.setattr(quality, "wcFuzzyQuality", _fake_quality)
        fuzzy.wcFuzzyRange(D, [2], weights=weights, random_state=19)
        return

    clara = importlib.import_module("weightedcluster.seqclararange")
    X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=int)
    monkeypatch.setattr(
        clara,
        "sequences_distance_between",
        lambda left, right, **_kwargs: np.abs(
            np.arange(left.shape[0], dtype=float)[:, None]
            - np.arange(right.shape[0], dtype=float)[None, :]
        ),
    )
    common = dict(
        Ds=D,
        k=2,
        sample_weights=weights,
        pam_n_init=1,
        seed_run=23,
        X=X,
        Xs=X,
        idx_global=np.arange(X.shape[0]),
        metric="HAM",
        indel=1.0,
        substitution_costs=None,
        timecost=None,
    )
    if name == "seqclararange_crisp":
        clara._cluster_crisp(**common)
        return
    if name == "seqclararange_fuzzy":
        fuzzy = importlib.import_module("weightedcluster.fuzzy")

        def _fake_fcmdd(_D: np.ndarray, u_init: np.ndarray, **_kwargs: object) -> dict[str, object]:
            return {
                "memb": u_init,
                "mobileCenters": list(range(u_init.shape[1])),
                "functional": 0.0,
                "dnoise": None,
            }

        monkeypatch.setattr(fuzzy, "fcmdd", _fake_fcmdd)
        clara._cluster_fuzzy(
            **common,
            method="fuzzy",
            m=1.5,
            dnoise=None,
            dlambda=None,
            eta=None,
        )
        return
    raise AssertionError(f"unknown medoid consumer {name!r}")


def test_declarative_capability_registry_matches_phase4_matrix():
    assert set(CAPABILITY_REGISTRY) == EXPECTED_CAPABILITIES
    medoids = CAPABILITY_REGISTRY["weighted_medoids"]
    assert medoids["owner"] == "pyweightedcluster"
    assert medoids["public_dispatch"] == ("compiled", "pure_python")
    assert medoids["selection"] == "conditional"
    assert {item["backend"] for item in medoids["implementations"]} == {
        "compiled",
        "pure_python",
    }
    # 2026-09-02 audit WC-090204: kmedoids-facing consumers are conditional
    # (only PAMonce with non-negative ordinary-range weights uses the
    # import-time binding); wcFuzzyRange selects at call time.
    assert {item["selection"] for item in medoids["public_consumers"]} == {
        "conditional",
        "call_time",
    }

    assert CAPABILITY_REGISTRY["cluster_quality"]["public_dispatch"] == (
        "compiled",
        "numba",
        "pure_python",
    )
    assert CAPABILITY_REGISTRY["observation_silhouette"]["selection"] == "python_only"
    assert CAPABILITY_REGISTRY["seqnull_spell_models"]["public_dispatch"] == ("python",)
    assert CAPABILITY_REGISTRY["seqnull_userpos_time_varying"]["selection"] == "python_only"
    assert CAPABILITY_REGISTRY["sequence_distances"]["owner"] == "traminex"
    assert CAPABILITY_REGISTRY["sequence_distances"]["selection"] == "delegated"

    # Finding F-10: the hierarchical-linkage selection had no registry entry, so
    # it was invisible to the ownership and truthfulness checks and absent from
    # get_backend_info()['capabilities'].
    linkage = CAPABILITY_REGISTRY["hierarchical_linkage"]
    assert linkage["owner"] == "pyweightedcluster"
    assert linkage["public_dispatch"] == ("python",)
    assert linkage["selection"] == "python_only"
    assert {item["symbol"] for item in linkage["implementations"]} == {
        "weightedcluster._fastcluster_linkage.linkage_fastcluster",
        "weightedcluster._cmpcluster_algorithms._diana_merge_structure",
        "weightedcluster.cmpcluster._linkage_flexible_beta",
    }


def test_every_unregistered_third_party_selection_carries_a_recorded_reason():
    """Finding F-10 asked for a decision on each unregistered selection.

    The registry says which implementation produced a caller's numbers, so a
    selection that cannot change a returned value stays out of it — but the
    reason is recorded rather than left silent, and pinned here so the list
    cannot drift without a deliberate edit.
    """
    from weightedcluster._backend_contract import UNREGISTERED_SELECTIONS

    assert set(UNREGISTERED_SELECTIONS) == {
        "matplotlib.pyplot (weightedcluster.fuzzy)",
        "seaborn (weightedcluster.plot)",
        "mplcursors (weightedcluster.clustassoc)",
        "tqdm (weightedcluster._parallel)",
        "threadpoolctl (weightedcluster._threads)",
        "pymc (weightedcluster._bayesian_rarcat_conditional)",
        "arviz / arviz_stats (weightedcluster._bayesian_rarcat_*)",
        "process / thread / serial executor (weightedcluster._parallel)",
    }
    assert all(reason.strip() for reason in UNREGISTERED_SELECTIONS.values())
    # `fastcluster` is absent from this list because finding F-11 removed the
    # branch that selected it; nothing selects it any more.
    assert not any("fastcluster" in name for name in UNREGISTERED_SELECTIONS)


@pytest.mark.parametrize("consumer", ["wcKMedoids", "wcKMedRange"])
@pytest.mark.parametrize(
    ("core_available", "expected"),
    [(True, "compiled"), (False, "pure_python")],
    ids=["compiled_selected", "pure_on_importerror"],
)
def test_public_kmedoids_consumers_bind_compiled_then_pure_at_import_time(
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    core_available: bool,
    expected: str,
):
    kmedoids = importlib.import_module("weightedcluster.kmedoids")
    calls: list[str] = []
    core = _module(
        "weightedcluster._wc_core",
        weighted_medoids=_medoid_spy(calls, "compiled"),
    )
    fallback = _fallback_module(
        weighted_medoids=_medoid_spy(calls, "pure_python"),
    )
    try:
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "weightedcluster._wc_core", core if core_available else None)
            mp.setitem(sys.modules, "weightedcluster._fallback", fallback)
            importlib.reload(kmedoids)
            mp.setattr(kmedoids, "wcClusterQuality", _fake_quality)
            if consumer == "wcKMedoids":
                kmedoids.wcKMedoids(_distance_fixture(), 2, cluster_only=True, random_state=31)
            else:
                kmedoids.wcKMedRange(_distance_fixture(), [2], R=1, random_state=31)
    finally:
        importlib.reload(kmedoids)

    assert calls == [expected]


def test_kmedoids_import_time_route_catches_only_importerror(monkeypatch: pytest.MonkeyPatch):
    kmedoids = importlib.import_module("weightedcluster.kmedoids")
    fallback_calls: list[str] = []
    broken_core = _module("weightedcluster._wc_core")

    def _broken_attribute(name: str) -> object:
        if name == "weighted_medoids":
            raise RuntimeError("compiled symbol resolution failed")
        raise AttributeError(name)

    broken_core.__getattr__ = _broken_attribute  # type: ignore[attr-defined]
    fallback = _fallback_module(
        weighted_medoids=_medoid_spy(fallback_calls, "pure_python"),
    )
    try:
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "weightedcluster._wc_core", broken_core)
            mp.setitem(sys.modules, "weightedcluster._fallback", fallback)
            with pytest.raises(RuntimeError, match="symbol resolution failed"):
                importlib.reload(kmedoids)
    finally:
        importlib.reload(kmedoids)
    assert fallback_calls == []


def test_kmedoids_import_time_fallback_is_not_excluded_from_coverage():
    import inspect

    kmedoids = importlib.import_module("weightedcluster.kmedoids")
    source = inspect.getsource(kmedoids)

    assert "except ImportError:  # pragma: no cover - fallback" not in source


@pytest.mark.parametrize("expected", ["compiled", "numba", "pure_python"])
def test_cluster_quality_selects_exactly_one_backend_tier(
    monkeypatch: pytest.MonkeyPatch, expected: str
):
    quality = importlib.import_module("weightedcluster.quality")
    calls: list[str] = []

    def _fail(backend: str) -> Callable[..., object]:
        def _unexpected(*_args: object, **_kwargs: object) -> object:
            raise AssertionError(f"unselected {backend} quality tier executed")

        return _unexpected

    core_kernel = (
        _quality_kernel_spy(calls, "compiled") if expected == "compiled" else _fail("compiled")
    )
    numba_kernel = _quality_kernel_spy(calls, "numba") if expected == "numba" else _fail("numba")
    core = _module("weightedcluster._wc_core", cluster_quality=core_kernel)
    numba = _module("weightedcluster._numba", cluster_quality_numba=numba_kernel)

    monkeypatch.setitem(
        sys.modules, "weightedcluster._wc_core", core if expected == "compiled" else None
    )
    monkeypatch.setitem(
        sys.modules, "weightedcluster._numba", numba if expected == "numba" else None
    )
    original_pure = quality._pure_public_quality_result

    def _pure(**kwargs: object) -> dict:
        if expected != "pure_python":
            raise AssertionError("unselected pure quality tier executed")
        calls.append("pure_python")
        return original_pure(**kwargs)

    monkeypatch.setattr(quality, "_pure_public_quality_result", _pure)
    quality.wcClusterQuality(_distance_fixture(), [1, 1, 2, 2])

    assert calls == [expected]


@pytest.mark.parametrize("consumer", ["wcFuzzyRange"])
@pytest.mark.parametrize(
    ("core_available", "expected"),
    [(True, "compiled"), (False, "pure_python")],
    ids=["compiled_selected", "pure_on_importerror"],
)
def test_call_time_medoid_consumers_select_compiled_then_pure(
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    core_available: bool,
    expected: str,
):
    calls: list[str] = []
    core = _module(
        "weightedcluster._wc_core",
        weighted_medoids=_medoid_spy(calls, "compiled"),
    )
    fallback = _module(
        "weightedcluster._fallback",
        weighted_medoids=_medoid_spy(calls, "pure_python"),
    )
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", core if core_available else None)
    monkeypatch.setitem(sys.modules, "weightedcluster._fallback", fallback)

    _run_call_time_medoid_consumer(consumer, monkeypatch)

    assert calls == [expected, expected]


@pytest.mark.parametrize("consumer", ["wcFuzzyRange"])
def test_call_time_medoid_routes_propagate_selected_kernel_errors(
    monkeypatch: pytest.MonkeyPatch, consumer: str
):
    core = _module(
        "weightedcluster._wc_core",
        weighted_medoids=_failing_medoid("compiled", RuntimeError),
    )
    fallback = _module(
        "weightedcluster._fallback",
        weighted_medoids=_failing_medoid("pure", AssertionError),
    )
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", core)
    monkeypatch.setitem(sys.modules, "weightedcluster._fallback", fallback)

    with pytest.raises(RuntimeError, match="compiled medoid called"):
        _run_call_time_medoid_consumer(consumer, monkeypatch)


def test_clara_crisp_reaches_the_medoid_kernel_through_wckmedoids(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLARA's crisp branch no longer selects the medoid kernel itself (F-05).

    R seeds every CLARA iteration from a Ward hierarchy and hands it to
    `wcKMedoids` (R/clara_clustrange.R:101,:125). `weightedcluster.kmedoids`
    binds the compiled-then-pure medoid kernel at import time, so the selection
    now happens there rather than at each call of this branch. Before finding
    F-05 the branch ran random-restart PAM and selected the kernel directly.
    """
    kmedoids = importlib.import_module("weightedcluster.kmedoids")
    calls: list[str] = []
    real_kernel = kmedoids._weighted_medoids

    def _spy(*args: object, **kwargs: object):
        calls.append("kmedoids")
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(kmedoids, "_weighted_medoids", _spy)
    _run_call_time_medoid_consumer("seqclararange_crisp", monkeypatch)

    assert calls == ["kmedoids"]


def test_clara_fuzzy_starts_from_the_ward_cut_without_a_medoid_kernel(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLARA's fuzzy branch selects no medoid kernel at all (F-05).

    Its starting membership is R's `as.memb(cutree(hc, k))`
    (R/clara_clustrange.R:112); per-cluster medoids come from
    `weightedcluster._utils.find_medoid`, and the search itself is `fcmdd`.
    """
    kmedoids = importlib.import_module("weightedcluster.kmedoids")
    calls: list[str] = []
    core = _module("weightedcluster._wc_core", weighted_medoids=_medoid_spy(calls, "compiled"))
    fallback = _module("weightedcluster._fallback", weighted_medoids=_medoid_spy(calls, "pure"))
    monkeypatch.setitem(sys.modules, "weightedcluster._wc_core", core)
    monkeypatch.setitem(sys.modules, "weightedcluster._fallback", fallback)
    monkeypatch.setattr(
        kmedoids,
        "_weighted_medoids",
        lambda *args, **kwargs: calls.append("kmedoids"),
    )

    _run_call_time_medoid_consumer("seqclararange_fuzzy", monkeypatch)

    assert calls == []


def test_seqnull_routes_numba_only_for_supported_paths(monkeypatch: pytest.MonkeyPatch):
    seqnull_module = importlib.import_module("weightedcluster.seqnull")
    calls: list[str] = []

    def _probs(
        _freqs: np.ndarray,
        _global: np.ndarray,
        _time_varying: bool,
        n: int,
        t: int,
        _seed: int,
    ) -> np.ndarray:
        calls.append("numba_stateindep")
        return np.zeros((n, t), dtype=np.int32)

    def _markov(
        _transition: np.ndarray,
        _begin: np.ndarray,
        n: int,
        t: int,
        _seed: int,
    ) -> np.ndarray:
        calls.append("numba_markov")
        return np.zeros((n, t), dtype=np.int32)

    fake_numba = _module(
        "weightedcluster._numba",
        seqnull_random_from_probs_numba=_probs,
        seqnull_markov_numba=_markov,
    )
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", fake_numba)
    monkeypatch.setattr(
        seqnull_module,
        "_seqnull_spell",
        lambda X, **_kwargs: calls.append("python_spell") or np.zeros_like(X),
    )
    monkeypatch.setattr(
        seqnull_module,
        "_seqnull_pos",
        lambda X, **_kwargs: calls.append("python_position") or np.zeros_like(X),
    )
    X = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=int)
    static = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    varying = np.repeat(static[:, :, None], X.shape[1] - 1, axis=2)

    for model in ("combined", "duration", "sequencing"):
        seqnull_module.seqnull(X, model=model, random_state=5)
    seqnull_module.seqnull(X, model="stateindep", random_state=6)
    seqnull_module.seqnull(X, model="markov", random_state=7)
    seqnull_module.seqnull(X, model="userpos", trate="trate", time_varying=False, random_state=8)
    seqnull_module.seqnull(X, model="userpos", trate=static, time_varying=False, random_state=9)
    seqnull_module.seqnull(X, model="userpos", trate="trate", random_state=10)
    seqnull_module.seqnull(X, model="userpos", trate=varying, random_state=11)

    assert calls == [
        "python_spell",
        "python_spell",
        "python_spell",
        "numba_stateindep",
        "numba_markov",
        "numba_markov",
        "numba_markov",
        "python_position",
        "python_position",
    ]


@pytest.mark.parametrize("model", ["stateindep", "markov", "userpos"])
def test_seqnull_supported_numba_routes_fall_back_only_on_importerror(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    seqnull_module = importlib.import_module("weightedcluster.seqnull")
    python_calls: list[str] = []
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", None)
    monkeypatch.setattr(
        seqnull_module,
        "_seqnull_pos",
        lambda X, **_kwargs: python_calls.append(model) or np.zeros_like(X),
    )
    X = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)
    kwargs = {"trate": "trate", "time_varying": False} if model == "userpos" else {}

    seqnull_module.seqnull(X, model=model, random_state=41, **kwargs)

    assert python_calls == [model]


def test_seqnull_selected_numba_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch):
    seqnull_module = importlib.import_module("weightedcluster.seqnull")

    def _raise(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("numba execution failed")

    fake_numba = _module(
        "weightedcluster._numba",
        seqnull_random_from_probs_numba=_raise,
        seqnull_markov_numba=_raise,
    )
    monkeypatch.setitem(sys.modules, "weightedcluster._numba", fake_numba)
    monkeypatch.setattr(
        seqnull_module,
        "_seqnull_pos",
        lambda *_args, **_kwargs: pytest.fail("Python fallback hid a Numba runtime error"),
    )
    X = np.array([[0, 1], [1, 0]], dtype=int)

    with pytest.raises(RuntimeError, match="numba execution failed"):
        seqnull_module.seqnull(X, model="stateindep", random_state=43)


def test_seqnull_stateindep_distribution_and_markov_structure_invariants():
    from weightedcluster import seqnull

    n = 4000
    stateindep_source = np.column_stack(
        [
            np.r_[np.zeros(3 * n // 4, dtype=int), np.ones(n // 4, dtype=int)],
            np.r_[np.zeros(n // 4, dtype=int), np.ones(3 * n // 4, dtype=int)],
        ]
    )
    generated = seqnull(stateindep_source, model="stateindep", random_state=47)
    assert generated.shape == stateindep_source.shape
    assert np.issubdtype(generated.dtype, np.integer)
    assert set(np.unique(generated)).issubset({0, 1})
    assert abs(float(np.mean(generated[:, 0] == 1)) - 0.25) < 0.04
    assert abs(float(np.mean(generated[:, 1] == 1)) - 0.75) < 0.04

    alternating = np.tile(np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=int), (500, 1))
    markov = seqnull(alternating, model="markov", random_state=53)
    assert markov.shape == alternating.shape
    assert np.all(markov[:, 1:] != markov[:, :-1])


def test_numba_seqnull_generators_reject_invalid_direct_probabilities():
    try:
        from weightedcluster._numba import (
            seqnull_markov_numba,
            seqnull_random_from_probs_numba,
        )
    except ImportError:
        fail_or_skip("numba backend unavailable")

    with pytest.raises(ValueError, match="finite"):
        seqnull_random_from_probs_numba(
            np.array([[0.5, 0.5], [np.nan, np.nan]]),
            np.array([0.5, 0.5]),
            True,
            2,
            2,
            7,
        )
    with pytest.raises(ValueError, match="sum to one"):
        seqnull_markov_numba(
            np.array([[0.2, 0.2], [0.3, 0.3]]),
            np.array([0.5, 0.5]),
            2,
            2,
            7,
        )


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        ("stateindep", {}),
        ("markov", {}),
        ("userpos", {"trate": "trate", "time_varying": False}),
        (
            "userpos",
            {
                "trate": np.array(
                    [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]],
                    dtype=float,
                ),
                "time_varying": False,
            },
        ),
    ],
    ids=["stateindep", "markov", "userpos-trate-global", "userpos-matrix-global"],
)
def test_seqnull_supported_numba_and_python_paths_are_value_equivalent(
    monkeypatch: pytest.MonkeyPatch, model: str, kwargs: dict[str, object]
):
    from weightedcluster import seqnull

    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 2],
            [1, 1, 2, 2],
            [2, 2, 0, 0],
        ],
        dtype=int,
    )
    numba_result = seqnull(X, model=model, random_state=71, **kwargs)
    with monkeypatch.context() as blocked:
        blocked.setitem(sys.modules, "weightedcluster._numba", None)
        python_result = seqnull(X, model=model, random_state=71, **kwargs)
    np.testing.assert_array_equal(numba_result, python_result)


def test_real_absent_numba_subprocess_uses_seeded_python_without_changing_numpy():
    script = textwrap.dedent("""
        import builtins
        import json
        import sys
        import numpy as np
        from weightedcluster import seqnull

        version_before = np.__version__
        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "weightedcluster._numba" or (level and name == "_numba"):
                raise ImportError("Numba deliberately unavailable for contract test")
            return original_import(name, globals, locals, fromlist, level)

        sys.modules.pop("weightedcluster._numba", None)
        builtins.__import__ = blocked_import
        X = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=int)
        first = seqnull(X, model="stateindep", random_state=61)
        second = seqnull(X, model="stateindep", random_state=61)
        print(json.dumps({
            "numpy_before": version_before,
            "numpy_after": np.__version__,
            "same": bool(np.array_equal(first, second)),
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "alphabet": sorted(int(v) for v in np.unique(first)),
        }, sort_keys=True))
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["numpy_before"] == result["numpy_after"] == np.__version__
    assert result["same"] is True
    assert result["shape"] == [4, 3]
    assert result["dtype"].startswith("int")
    assert set(result["alphabet"]).issubset({0, 1})


def test_registry_pure_quality_symbol_is_the_publicly_selected_implementation():
    """2026-09-02 audit WC-090204: the registry declared _fallback.cluster_quality
    as the pure tier, but the public pure branch selects
    cluster_quality_r_permissive; the validated kernel is only the Numba
    escape-hatch target. The registry's purpose is to say which implementation
    produced a caller's numbers, so it must name the selected one.
    """
    entry = CAPABILITY_REGISTRY["cluster_quality"]
    pure = [item for item in entry["implementations"] if item["backend"] == "pure_python"]
    assert len(pure) == 1
    assert pure[0]["symbol"] == "weightedcluster._fallback.cluster_quality_r_permissive"

    from weightedcluster import _fallback, quality

    source = inspect.getsource(quality._pure_public_quality_result)
    assert "cluster_quality_r_permissive" in source
    assert callable(_fallback.cluster_quality_r_permissive)


def test_registry_medoid_consumer_selection_is_conditional_not_import_time():
    """2026-09-02 audit WC-090204: only PAMonce with non-negative ordinary-range
    weights uses the import-time-bound kernel; methods 1/2 run module-local
    searches, negative weights use the local PAMonce port, and range-risk
    inputs import the pure kernel at call time (kmedoids.py exact branch).
    """
    consumers = {
        item["symbol"]: item for item in CAPABILITY_REGISTRY["weighted_medoids"]["public_consumers"]
    }
    for name in (
        "weightedcluster.kmedoids.wcKMedoids",
        "weightedcluster.kmedoids.wcKMedRange",
    ):
        assert consumers[name]["selection"] == "conditional", name
        assert "PAMonce" in consumers[name]["condition"], name

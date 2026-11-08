from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from weightedcluster import seqclararange
from weightedcluster._seqdist import normalize_seqdist_config
from weightedcluster._validation import validate_seqdist_args


def _toy_sequences() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=int,
    )


def test_final_seqclararange_assembly_has_no_ignored_medoid_callback():
    runtime = importlib.import_module("weightedcluster._seqclara_runtime")

    parameters = inspect.signature(runtime.assemble_seqclararange_result_from_cache).parameters
    assert "compute_full_cluster_medoids_fn" not in parameters


def test_seqclararange_cache_has_no_unreachable_warm_start_scaffolding():
    runtime = importlib.import_module("weightedcluster._seqclara_runtime")
    public_module = importlib.import_module("weightedcluster.seqclararange")

    assert not hasattr(runtime, "map_warm_start_medoids")
    assert not hasattr(runtime, "compute_full_cluster_medoids")
    assert not hasattr(public_module, "_map_warm_start_medoids")
    assert not hasattr(public_module, "_compute_full_cluster_medoids")
    assert "warm_full" not in inspect.signature(runtime.run_seqclara_iterations_for_k).parameters
    assert "init_medoids_sub" not in inspect.signature(public_module._cluster_crisp).parameters
    assert "init_medoids_sub" not in inspect.signature(public_module._cluster_fuzzy).parameters
    assert (
        "compute_full_cluster_medoids_fn"
        not in inspect.signature(runtime.run_seqclara_cache).parameters
    )


def test_seqclararange_multi_criteria_uses_shared_run_cache(monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    real = scmod._run_seqclara_cache
    calls = {"n": 0}

    def _spy(context):
        calls["n"] += 1
        return real(context)

    monkeypatch.setattr(scmod, "_run_seqclara_cache", _spy)

    X = _toy_sequences()
    out = seqclararange(
        X,
        R=3,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria=["distance", "db"],
        random_state=31,
    )

    assert calls["n"] == 1
    assert "distance" in out and "db" in out


def test_seqclararange_distance_criterion_defers_quality_to_final_assembly(monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    real = scmod.wcClusterQuality
    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(scmod, "wcClusterQuality", _spy)

    X = _toy_sequences()
    out = seqclararange(
        X,
        R=3,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        random_state=32,
    )

    assert calls["n"] == 1
    # The R-compatible stats table retains seqclararange's medoid-criterion DB;
    # the extended quality table exposes the published partition-level index.
    assert out["stats"]["DB"][0] == pytest.approx(out["clara"][2]["db"])
    assert np.isfinite(out["quality_stats"]["DB"][0])


def test_reaggregate_sampled_cases_collapses_duplicate_draws():
    scmod = importlib.import_module("weightedcluster.seqclararange")

    class _FakeRng:
        def choice(self, *args, **kwargs):
            return np.array([2, 2, 1, 2, 1], dtype=int)

    Xagg = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ],
        dtype=int,
    )
    Xs, idx_global, sample_weights = scmod._reaggregate_sampled_cases(
        rng_local=_FakeRng(),
        Xagg=Xagg,
        agg_index=np.array([10, 11, 12], dtype=int),
        probs=np.array([0.2, 0.3, 0.5], dtype=float),
        sample_size=5,
        min_unique=2,
    )

    # wcAggregateCases follows R's aggregate-key order, so the sampled case id
    # 1 precedes id 2 even though id 2 appeared first in the random draw.
    np.testing.assert_array_equal(Xs, np.array([[1, 1], [2, 2]], dtype=int))
    np.testing.assert_array_equal(idx_global, np.array([11, 12], dtype=int))
    np.testing.assert_array_equal(sample_weights, np.array([2.0, 3.0], dtype=float))


def test_cluster_crisp_passes_sample_weights(monkeypatch):
    """The CLARA sample weights must reach both stages of R's initialization.

    R passes `ac2$aggWeights` to the Ward hierarchy and again to `wcKMedoids`
    (R/clara_clustrange.R:101,:125). Before finding F-05 this branch ran
    random-restart PAM, so the seam this test watches was `weighted_medoids`;
    it now watches the two stages that actually run.
    """
    scmod = importlib.import_module("weightedcluster.seqclararange")
    kmedoids_module = importlib.import_module("weightedcluster.kmedoids")
    seen: dict[str, np.ndarray] = {}

    real_ward = scmod._ward_initial_hierarchy

    def spy_ward(Ds, *, weights):
        seen["ward_weights"] = np.asarray(weights, dtype=float)
        return real_ward(Ds, weights=weights)

    def fake_kmedoids(D_in, k, *, weights=None, cluster_only=False, **kwargs):
        seen["weights"] = np.asarray(weights, dtype=float)
        return np.array([1, 2], dtype=int)

    monkeypatch.setattr(scmod, "_ward_initial_hierarchy", spy_ward)
    monkeypatch.setattr(kmedoids_module, "wcKMedoids", fake_kmedoids)
    monkeypatch.setattr(
        scmod,
        "sequences_distance_between",
        lambda X, Y, **kwargs: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
    )

    scmod._cluster_crisp(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        2,
        np.array([4.0, 1.0], dtype=float),
        2,
        17,
        np.array([[0], [1]], dtype=int),
        np.array([[0], [1]], dtype=int),
        np.array([0, 1], dtype=int),
        "HAM",
        1.0,
        None,
        None,
    )

    np.testing.assert_array_equal(seen["weights"], np.array([4.0, 1.0], dtype=float))
    np.testing.assert_array_equal(seen["ward_weights"], np.array([4.0, 1.0], dtype=float))


def test_cluster_crisp_breaks_distance_ties_in_r_medoid_order(monkeypatch):
    """A case equidistant from two medoids must join the one R lists first.

    R orders the medoid columns by first appearance in the PAM membership,
    `mysample[ac2$aggIndex[unique(clustering)]]` (R/clara_clustrange.R:127),
    and assigns each case to the first minimum, `which.min` (:193). Python
    sorted the columns with `np.unique`, so on tie-heavy sequence data the
    public partition and DB differed from R (2026-09-21 targeted audit, M-1).

    The medoid search is replaced so its membership is known; the distances
    are real Hamming distances, so the tie below is genuine, not stipulated.
    """
    scmod = importlib.import_module("weightedcluster.seqclararange")
    kmedoids_module = importlib.import_module("weightedcluster.kmedoids")

    # Subsample positions 2 then 1 (one-based) appear first; sorted order would
    # be 1 then 2. R's column order is therefore [row 1, row 0], zero-based.
    monkeypatch.setattr(
        kmedoids_module,
        "wcKMedoids",
        lambda *args, **kwargs: np.array([2, 2, 1], dtype=int),
    )
    sequences = np.array([[0, 0], [1, 1], [0, 1]], dtype=int)
    # Row 2 is at Hamming distance 1 from both medoids, [0, 0] and [1, 1].
    labels, _, medoids_global, distances = scmod._cluster_crisp(
        np.zeros((3, 3), dtype=float),
        2,
        np.ones(3, dtype=float),
        1,
        0,
        sequences,
        sequences,
        np.array([0, 1, 2], dtype=int),
        "HAM",
        1.0,
        None,
        None,
        init_membership_sub=np.array([1, 1, 2], dtype=int),
    )

    assert distances[2, 0] == distances[2, 1], "the fixture must contain a real tie"
    np.testing.assert_array_equal(medoids_global, np.array([1, 0], dtype=int))
    # The tie resolves to R's first medoid, sequence [1, 1] at row 1 -- not to
    # row 0, which is where ascending column order would send it.
    assert medoids_global[labels[2]] == 1


def test_cluster_fuzzy_passes_sample_weights(monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    fuzzy_module = importlib.import_module("weightedcluster.fuzzy")
    seen: dict[str, np.ndarray] = {}

    real_ward = scmod._ward_initial_hierarchy

    def spy_ward(Ds, *, weights):
        seen["seed_weights"] = np.asarray(weights, dtype=float)
        return real_ward(Ds, weights=weights)

    def fake_fcmdd(D_in, u_init, **kwargs):
        seen["fcmdd_weights"] = np.asarray(kwargs["weights"], dtype=float)
        return {
            "memb": np.asarray(u_init, dtype=float),
            "mobileCenters": [0, 1],
            "functional": 0.0,
            "dnoise": None,
        }

    monkeypatch.setattr(scmod, "_ward_initial_hierarchy", spy_ward)
    monkeypatch.setattr(fuzzy_module, "fcmdd", fake_fcmdd)
    monkeypatch.setattr(
        scmod,
        "sequences_distance_between",
        lambda X, Y, **kwargs: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
    )

    scmod._cluster_fuzzy(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        2,
        np.array([4.0, 1.0], dtype=float),
        "fuzzy",
        2,
        17,
        np.array([[0], [1]], dtype=int),
        np.array([[0], [1]], dtype=int),
        np.array([0, 1], dtype=int),
        "HAM",
        1.0,
        None,
        None,
        1.6,
        None,
        None,
        None,
    )

    np.testing.assert_array_equal(seen["seed_weights"], np.array([4.0, 1.0], dtype=float))
    np.testing.assert_array_equal(seen["fcmdd_weights"], np.array([4.0, 1.0], dtype=float))


def test_validate_seqdist_args_delegates_to_normalizer(monkeypatch):
    vmod = importlib.import_module("weightedcluster._validation")
    seen: dict[str, object] = {}

    def fake_normalize(seqdist_args, *, default_method="HAM"):
        seen["seqdist_args"] = seqdist_args
        seen["default_method"] = default_method
        return {
            "method": "LCS",
            "metric": "LCS",
            "indel": 1.0,
            "substitution_costs": None,
            "timecost": None,
        }

    monkeypatch.setattr(vmod, "normalize_seqdist_config", fake_normalize)

    out = validate_seqdist_args({"method": "HAM"}, default_method="LCS")

    assert seen["seqdist_args"] == {"method": "HAM"}
    assert seen["default_method"] == "LCS"
    assert out["metric"] == "LCS"


@pytest.mark.parametrize(
    "config",
    (
        {"method": "HAM", "metod": "OM"},
        {"method": "OMSPELL", "tpow": 2.0},
        {"method": "HAM", "metric": "OM"},
        {"method": "OM", "sm": 2.0, "substitution_costs": 3.0},
        {"method": "OMSPELL", "timecost": 0.5, "time_cost": 0.7},
    ),
)
def test_seqdist_config_rejects_unknown_or_conflicting_controls(config):
    with pytest.raises(ValueError, match="seqdist_args"):
        normalize_seqdist_config(config)


@pytest.mark.parametrize(
    "config",
    (
        {"method": "OM", "indel": np.nan},
        {"method": "OM", "indel": np.inf},
        {"method": "OMSPELL", "timecost": np.nan},
        {"method": "OMSPELL", "timecost": np.inf},
    ),
)
def test_seqdist_config_rejects_nonfinite_cost_controls(config):
    with pytest.raises(ValueError, match="finite"):
        normalize_seqdist_config(config)


@pytest.mark.parametrize("kvals", ([2.5], [True], [2, 2]))
def test_seqclararange_rejects_nonintegral_boolean_or_duplicate_kvals(kvals, monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    monkeypatch.setattr(
        scmod,
        "sequences_distance",
        lambda *args, **kwargs: pytest.fail("distance computation preceded kvals validation"),
    )

    with pytest.raises(ValueError, match="kvals"):
        seqclararange(_toy_sequences(), R=1, sample_size=4, kvals=kvals)


@pytest.mark.parametrize("pam_n_init", (1.5, True, "2"))
def test_seqclararange_rejects_lossy_pam_n_init(pam_n_init, monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    monkeypatch.setattr(
        scmod,
        "sequences_distance",
        lambda *args, **kwargs: pytest.fail("distance computation preceded control validation"),
    )

    with pytest.raises(ValueError, match="pam_n_init"):
        seqclararange(
            _toy_sequences(),
            R=1,
            sample_size=4,
            kvals=[2],
            pam_n_init=pam_n_init,
        )


@pytest.mark.parametrize("n_jobs", (1.5, True, "2", 0))
def test_seqclararange_rejects_invalid_worker_counts_before_distance(n_jobs, monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    monkeypatch.setattr(
        scmod,
        "sequences_distance",
        lambda *args, **kwargs: pytest.fail("distance computation preceded control validation"),
    )

    with pytest.raises(ValueError, match="n_jobs"):
        seqclararange(
            _toy_sequences(),
            R=1,
            sample_size=4,
            kvals=[2],
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize(
    "control,value",
    (
        ("stability", 1),
        ("parallel", "false"),
        ("progressbar", 0),
        ("keep_diss", "false"),
        ("export_membership", 1),
    ),
)
def test_seqclararange_rejects_truthy_boolean_controls(control, value, monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    monkeypatch.setattr(
        scmod,
        "sequences_distance",
        lambda *args, **kwargs: pytest.fail("distance computation preceded control validation"),
    )

    with pytest.raises(TypeError, match=control):
        seqclararange(
            _toy_sequences(),
            R=1,
            sample_size=4,
            kvals=[2],
            **{control: value},
        )


@pytest.mark.parametrize(
    "options,match",
    (
        ({"method": "fuzzy", "m": np.nan}, "m"),
        ({"method": "fuzzy", "m": "1.5"}, "m"),
        ({"method": "noise", "dnoise": np.nan}, "dnoise"),
        ({"method": "noise", "dnoise": -0.1}, "dnoise"),
        ({"method": "noise", "dlambda": np.inf}, "dlambda"),
        ({"method": "pcmdd", "eta": np.array([1.0, np.nan])}, "eta"),
        ({"method": "representativeness", "max_dist": "1"}, "max_dist"),
    ),
)
def test_seqclararange_rejects_invalid_method_parameters_before_distance(
    options, match, monkeypatch
):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    monkeypatch.setattr(
        scmod,
        "sequences_distance",
        lambda *args, **kwargs: pytest.fail("distance computation preceded control validation"),
    )

    with pytest.raises((TypeError, ValueError), match=match):
        seqclararange(
            _toy_sequences(),
            R=1,
            sample_size=4,
            kvals=[2],
            **options,
        )


@pytest.mark.parametrize("raw_key", (2.5, True, "2"))
def test_seqclararange_replay_membership_keys_are_lossless_integers(raw_key, monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    monkeypatch.setattr(
        scmod,
        "sequences_distance",
        lambda *args, **kwargs: pytest.fail("distance computation preceded replay validation"),
    )

    with pytest.raises(ValueError, match="replay_initial_memberships.*integer"):
        seqclararange(
            _toy_sequences(),
            R=1,
            sample_size=4,
            kvals=[2],
            replay_initial_memberships={raw_key: [None]},
        )


def test_seqclararange_replay_memberships_must_be_a_mapping(monkeypatch):
    scmod = importlib.import_module("weightedcluster.seqclararange")
    monkeypatch.setattr(
        scmod,
        "sequences_distance",
        lambda *args, **kwargs: pytest.fail("distance computation preceded replay validation"),
    )

    with pytest.raises(TypeError, match="replay_initial_memberships.*mapping"):
        seqclararange(
            _toy_sequences(),
            R=1,
            sample_size=4,
            kvals=[2],
            replay_initial_memberships=[(2, [None])],
        )

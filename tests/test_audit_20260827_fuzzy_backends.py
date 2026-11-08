from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from weightedcluster.fuzzy import fcmdd

pytest_plugins = ["tests.r_parity_utils"]


AUDIT_DISTANCES = np.array(
    [
        [
            0.0,
            2.708752363664987,
            3.571920116102849,
            3.5945315442990227,
            3.904702187780351,
            3.736609269898084,
        ],
        [
            2.708752363664987,
            0.0,
            2.199716055091518,
            2.8453313734941017,
            2.656139494885924,
            2.3856393258549353,
        ],
        [
            3.571920116102849,
            2.199716055091518,
            0.0,
            1.178716278194888,
            2.4207530499706906,
            2.3357784158939943,
        ],
        [
            3.5945315442990227,
            2.8453313734941017,
            1.178716278194888,
            0.0,
            1.9622167762540534,
            3.4254175253919157,
        ],
        [
            3.904702187780351,
            2.656139494885924,
            2.4207530499706906,
            1.9622167762540534,
            0.0,
            4.363455175601341,
        ],
        [
            3.736609269898084,
            2.3856393258549353,
            2.3357784158939943,
            3.4254175253919157,
            4.363455175601341,
            0.0,
        ],
    ],
    dtype=float,
)


def _run_isolated_fuzzy_tier(tier: str) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    code = textwrap.dedent("""
        import json
        import os
        import sys
        import types

        import numpy as np

        tier = os.environ["WC_AUDIT_FUZZY_TIER"]
        calls = []
        if tier == "pure":
            sys.modules["weightedcluster._wc_core"] = None
            from weightedcluster._fallback import weighted_medoids as actual_kernel
            backend_name = "pure_python"
            module_name = "weightedcluster._fallback"
        else:
            from weightedcluster._wc_core import weighted_medoids as actual_kernel
            backend_name = "compiled"
            module_name = "weightedcluster._wc_core"

        def observed_kernel(*args, **kwargs):
            calls.append(
                {
                    "n_init": int(kwargs["n_init"]),
                    "random_state": int(kwargs["random_state"]),
                    "init_medoids": np.asarray(kwargs["init_medoids"], dtype=int).tolist(),
                }
            )
            return actual_kernel(*args, **kwargs)

        observed_module = types.ModuleType(module_name)
        observed_module.weighted_medoids = observed_kernel
        sys.modules[module_name] = observed_module

        from weightedcluster.fuzzy import wcFuzzyRange

        distances = np.array(
            [
                [0.0, 2.708752363664987, 3.571920116102849, 3.5945315442990227, 3.904702187780351, 3.736609269898084],
                [2.708752363664987, 0.0, 2.199716055091518, 2.8453313734941017, 2.656139494885924, 2.3856393258549353],
                [3.571920116102849, 2.199716055091518, 0.0, 1.178716278194888, 2.4207530499706906, 2.3357784158939943],
                [3.5945315442990227, 2.8453313734941017, 1.178716278194888, 0.0, 1.9622167762540534, 3.4254175253919157],
                [3.904702187780351, 2.656139494885924, 2.4207530499706906, 1.9622167762540534, 0.0, 4.363455175601341],
                [3.736609269898084, 2.3856393258549353, 2.3357784158939943, 3.4254175253919157, 4.363455175601341, 0.0],
            ],
            dtype=float,
        )

        def serialise(result):
            return {
                "kvals": result["kvals"],
                "medoids": {key: value.tolist() for key, value in result["medoids"].items()},
                "labels": {key: value.tolist() for key, value in result["labels"].items()},
                "memb": {key: value.tolist() for key, value in result["memb"].items()},
                "stats": result["stats"],
            }

        cases = [
            {"name": "audit", "kvals": [2], "pam_n_init": 2, "weights": None},
            {"name": "single", "kvals": [2, 3], "pam_n_init": 1, "weights": None},
            {
                "name": "weighted_multi",
                "kvals": [2, 3],
                "pam_n_init": 4,
                "weights": np.array([1.0, 0.5, 2.0, 1.5, 0.75, 3.0]),
            },
        ]
        results = {}
        for case in cases:
            output = wcFuzzyRange(
                distances,
                case["kvals"],
                method="FCMdd",
                m=1.6,
                weights=case["weights"],
                pam_n_init=case["pam_n_init"],
                random_state=2,
                crisp_labels=True,
            )
            results[case["name"]] = serialise(output)
        repeated = wcFuzzyRange(
            distances,
            [2],
            method="FCMdd",
            m=1.6,
            pam_n_init=2,
            random_state=2,
            crisp_labels=True,
        )
        print(
            json.dumps(
                {
                    "tier": backend_name,
                    "calls": calls,
                    "results": results,
                    "repeat": serialise(repeated),
                },
                sort_keys=True,
            )
        )
        """)
    environment = os.environ.copy()
    environment["WC_AUDIT_FUZZY_TIER"] = tier
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _assert_nested_numerically_equal(left: object, right: object) -> None:
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_numerically_equal(left[key], right[key])
        return
    if isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_numerically_equal(left_item, right_item)
        return
    if isinstance(left, float):
        assert isinstance(right, (int, float))
        np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-15)
        return
    assert left == right


def test_wcfuzzyrange_seeded_results_are_backend_independent_and_repeatable():
    compiled = _run_isolated_fuzzy_tier("compiled")
    pure = _run_isolated_fuzzy_tier("pure")

    assert compiled["tier"] == "compiled"
    assert pure["tier"] == "pure_python"
    assert compiled["calls"]
    assert pure["calls"]
    assert len(compiled["calls"]) == len(pure["calls"]) == 14
    assert all(call["n_init"] == 1 for call in compiled["calls"])
    assert all(call["n_init"] == 1 for call in pure["calls"])
    assert all(call["init_medoids"] for call in compiled["calls"])
    assert all(call["init_medoids"] for call in pure["calls"])
    assert compiled["calls"] == pure["calls"]
    assert compiled["repeat"] == compiled["results"]["audit"]
    assert pure["repeat"] == pure["results"]["audit"]
    _assert_nested_numerically_equal(compiled["results"], pure["results"])


def test_wcfuzzyrange_equal_pam_objectives_keep_first_planned_start(monkeypatch):
    import types

    import weightedcluster.fuzzy as fuzzy_module
    import weightedcluster.quality as quality_module
    from weightedcluster._utils import generate_random_seed

    calls: list[np.ndarray] = []

    def tied_kernel(D, k, **kwargs):
        start = np.asarray(kwargs["init_medoids"], dtype=int)
        calls.append(start.copy())
        labels = np.ones(D.shape[0], dtype=int)
        labels[start[0]] = 0
        return labels, start, 1.0

    def passthrough_fcmdd(D, u_init, **kwargs):
        return {
            "memb": np.asarray(u_init, dtype=float),
            "mobileCenters": [0, 1],
            "functional": 1.0,
            "dnoise": None,
        }

    def fixed_quality(D, membership, **kwargs):
        return {"stats": {"FPC": 0.0}}

    monkeypatch.setitem(
        sys.modules,
        "weightedcluster._wc_core",
        types.SimpleNamespace(weighted_medoids=tied_kernel),
    )
    monkeypatch.setattr(fuzzy_module, "fcmdd", passthrough_fcmdd)
    monkeypatch.setattr(quality_module, "wcFuzzyQuality", fixed_quality)

    result = fuzzy_module.wcFuzzyRange(
        AUDIT_DISTANCES,
        [2],
        pam_n_init=2,
        random_state=23,
        crisp_labels=True,
    )

    seed = generate_random_seed(generate_random_seed(23, 0), 1, 0)
    rng = np.random.default_rng(seed)
    expected_starts = [rng.choice(6, size=2, replace=False) for _ in range(2)]
    assert len(calls) == 2
    for actual, expected in zip(calls, expected_starts, strict=True):
        np.testing.assert_array_equal(actual, expected)
    expected_labels = np.full(6, 2, dtype=int)
    expected_labels[expected_starts[0][0]] = 1
    np.testing.assert_array_equal(result["labels"]["cluster2"], expected_labels)


def test_wcfuzzyrange_preserves_every_finite_positive_weight_across_public_stages(
    monkeypatch,
):
    """R-08: public orchestration must not erase a positive ratio before kernels."""
    import types

    import weightedcluster.fuzzy as fuzzy_module
    import weightedcluster.quality as quality_module

    maximum = np.finfo(float).max
    minimum = np.nextafter(0.0, 1.0)
    raw_weights = np.array([maximum, minimum, maximum, minimum])
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 3.0, 4.0],
            [4.0, 3.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )
    observed: dict[str, list[np.ndarray]] = {"pam": [], "fcmdd": [], "quality": []}

    def observed_kernel(D, k, **kwargs):
        observed["pam"].append(np.asarray(kwargs["weights"], dtype=float).copy())
        return np.array([0, 0, 1, 1]), np.array([0, 2]), 1.0

    def observed_fcmdd(D, u_init, **kwargs):
        observed["fcmdd"].append(np.asarray(kwargs["weights"], dtype=float).copy())
        return {
            "memb": np.asarray(u_init, dtype=float),
            "mobileCenters": [0, 2],
            "functional": 1.0,
            "dnoise": None,
        }

    def observed_quality(D, membership, **kwargs):
        forwarded = np.asarray(kwargs["weights"], dtype=float).copy()
        observed["quality"].append(forwarded)
        return {"stats": {"positive_weights": float(np.count_nonzero(forwarded > 0.0))}}

    monkeypatch.setitem(
        sys.modules,
        "weightedcluster._wc_core",
        types.SimpleNamespace(weighted_medoids=observed_kernel),
    )
    monkeypatch.setattr(fuzzy_module, "_pam_requires_exact_objectives", lambda _D, _w: False)
    monkeypatch.setattr(fuzzy_module, "fcmdd", observed_fcmdd)
    monkeypatch.setattr(quality_module, "wcFuzzyQuality", observed_quality)

    result = fuzzy_module.wcFuzzyRange(
        distances,
        [2],
        weights=raw_weights,
        pam_n_init=2,
        random_state=7,
    )

    assert result["stats"]["positive_weights"] == [4.0]
    assert len(observed["pam"]) == 2
    for stage_weights in sum(observed.values(), []):
        np.testing.assert_array_equal(stage_weights, raw_weights)
        assert np.count_nonzero(stage_weights > 0.0) == raw_weights.size


def test_wcfuzzyrange_extreme_mixed_weights_does_not_crash_pam_boundary() -> None:
    """R-08: exponent-safe forwarding must remain safe at the PAM boundary."""
    script = textwrap.dedent("""
        import json
        import numpy as np
        from weightedcluster import wcFuzzyRange

        maximum = np.finfo(float).max
        minimum = np.nextafter(0.0, 1.0)
        distances = np.array(
            [
                [0.0, 1.0, 4.0, 5.0],
                [1.0, 0.0, 3.0, 4.0],
                [4.0, 3.0, 0.0, 1.0],
                [5.0, 4.0, 1.0, 0.0],
            ]
        )
        result = wcFuzzyRange(
            distances,
            [2],
            method="FCMdd",
            m=2.0,
            weights=np.array([maximum, minimum, maximum, minimum]),
            pam_n_init=1,
            random_state=1,
        )
        print(json.dumps({"medoids": result["medoids"]["cluster2"].tolist()}))
        """)
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"})

    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert json.loads(completed.stdout)["medoids"] == [2, 0]


def test_exact_pam_swap_delta_includes_the_removed_medoid_cost() -> None:
    from weightedcluster._fallback import _weighted_medoids_exact

    smallest = np.nextafter(0.0, 1.0)
    distances = np.array([[0.0, 10.0, 1.0], [10.0, 0.0, 9.0], [1.0, 9.0, 0.0]])

    labels, medoids, objective = _weighted_medoids_exact(
        distances,
        2,
        np.array([np.finfo(float).max, smallest, smallest]),
        max_iter=1,
        n_init=1,
        random_state=0,
        init_medoids=np.array([0, 1]),
    )

    np.testing.assert_array_equal(labels, [0, 1, 0])
    np.testing.assert_array_equal(medoids, [0, 1])
    assert objective == smallest


def test_exact_pam_preserves_common_weight_scale_swap_order() -> None:
    from weightedcluster._fallback import weighted_medoids

    distances = np.array(
        [
            [0.0, 17.0, 7.0, 8.0],
            [17.0, 0.0, 7.0, 17.0],
            [7.0, 7.0, 0.0, 25.0],
            [8.0, 17.0, 25.0, 0.0],
        ]
    )
    start = np.array([0, 3])
    unit = weighted_medoids(
        distances,
        2,
        weights=np.ones(4),
        max_iter=10,
        n_init=1,
        init_medoids=start,
    )
    smallest = np.nextafter(0.0, 1.0)
    scaled = weighted_medoids(
        distances,
        2,
        weights=np.full(4, smallest),
        max_iter=10,
        n_init=1,
        init_medoids=start,
    )

    np.testing.assert_array_equal(unit[0], [0, 0, 0, 1])
    np.testing.assert_array_equal(unit[1], [2, 3])
    assert unit[2] == 14.0
    np.testing.assert_array_equal(scaled[0], unit[0])
    np.testing.assert_array_equal(scaled[1], unit[1])
    assert scaled[2] == 14.0 * smallest


def test_exact_pam_predicate_includes_distance_product_range() -> None:
    from weightedcluster._fallback import weighted_medoids

    maximum = np.finfo(float).max
    distances = np.array(
        [
            [0.0, 1e-308, 1.0, maximum],
            [1e-308, 0.0, 1e-323, maximum],
            [1.0, 1e-323, 0.0, 1e-323],
            [maximum, maximum, 1e-323, 0.0],
        ]
    )

    labels, medoids, objective = weighted_medoids(
        distances,
        2,
        weights=np.ones(4),
        max_iter=10,
        n_init=1,
        init_medoids=np.array([3, 0]),
    )

    np.testing.assert_array_equal(labels, [1, 0, 0, 0])
    np.testing.assert_array_equal(medoids, [2, 0])
    assert objective == 2e-323


def test_public_fuzzy_range_huge_distances_does_not_crash_pam_boundary() -> None:
    script = textwrap.dedent("""
        import json
        import numpy as np
        from weightedcluster import wcFuzzyRange

        distances = (np.ones((4, 4)) - np.eye(4)) * 1e308
        result = wcFuzzyRange(
            distances,
            [2],
            method="FCMdd",
            m=2.0,
            weights=np.ones(4),
            pam_n_init=1,
            random_state=1,
        )
        print(json.dumps({"count": len(result["medoids"]["cluster2"])}))
        """)
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"})

    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert json.loads(completed.stdout) == {"count": 2}


def test_compiled_pam_never_leaves_first_overflowing_run_unselected() -> None:
    from weightedcluster._wc_core import weighted_medoids

    distances = (np.ones((4, 4)) - np.eye(4)) * 1e308

    labels, medoids, objective = weighted_medoids(
        distances,
        2,
        weights=np.ones(4),
        max_iter=1,
        n_init=1,
        random_state=1,
    )

    assert labels.shape == (4,)
    assert medoids.shape == (2,)
    assert np.isinf(objective)


def test_wcfuzzyrange_ranks_overflowing_multistart_objectives_exactly(monkeypatch) -> None:
    import weightedcluster._fallback as fallback_module
    import weightedcluster.fuzzy as fuzzy_module
    import weightedcluster.quality as quality_module

    positions = np.array([0.0, 100.0, 1.0, 2.0])
    distances = np.abs(positions[:, None] - positions[None, :])
    starts = (np.array([0, 3]), np.array([0, 1]))

    def fixed_plan(*_args, **_kwargs):
        return starts

    def no_swap_kernel(D, k, **kwargs):
        medoids = np.asarray(kwargs["init_medoids"], dtype=int)
        nearest = D[:, medoids]
        labels = np.argmin(nearest, axis=1)
        return labels, medoids, float("inf")

    selected: list[np.ndarray] = []

    def passthrough_fcmdd(D, membership, **kwargs):
        selected.append(np.asarray(membership, dtype=float).copy())
        return {
            "memb": np.asarray(membership, dtype=float),
            "mobileCenters": [0, 1],
            "functional": 1.0,
            "dnoise": None,
        }

    monkeypatch.setattr(fuzzy_module, "_weighted_pam_start_plan", fixed_plan)
    monkeypatch.setattr(fallback_module, "weighted_medoids", no_swap_kernel)
    monkeypatch.setattr(fuzzy_module, "fcmdd", passthrough_fcmdd)
    monkeypatch.setattr(
        quality_module,
        "wcFuzzyQuality",
        lambda *_args, **_kwargs: {"stats": {"FPC": 1.0}},
    )

    fuzzy_module.wcFuzzyRange(
        distances,
        [2],
        weights=np.full(4, np.finfo(float).max),
        pam_n_init=2,
        random_state=0,
    )

    expected_labels = np.argmin(distances[:, starts[1]], axis=1)
    np.testing.assert_array_equal(np.argmax(selected[0], axis=1), expected_labels)


def test_explicit_medoid_start_remains_exact_across_actual_kernels():
    from weightedcluster._wc_core import weighted_medoids as compiled

    from weightedcluster._fallback import weighted_medoids as pure

    start = np.array([0, 2], dtype=int)
    kwargs = {
        "weights": np.array([1.0, 0.5, 2.0, 1.5, 0.75, 3.0]),
        "max_iter": 50,
        "n_init": 1,
        "random_state": 2,
        "init_medoids": start,
    }
    compiled_result = compiled(AUDIT_DISTANCES, 2, **kwargs)
    pure_result = pure(AUDIT_DISTANCES, 2, **kwargs)

    np.testing.assert_array_equal(compiled_result[0], pure_result[0])
    np.testing.assert_array_equal(compiled_result[1], pure_result[1])
    assert compiled_result[2] == pure_result[2]


def _scale_fixture() -> tuple[np.ndarray, np.ndarray]:
    points = np.array([0.0, 1.0, 9.0])
    distances = np.abs(points[:, None] - points[None, :])
    initial = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    return distances, initial


@pytest.mark.parametrize("scale", [1e-18, 1e-13, 1e13])
def test_fcmdd_positive_distance_memberships_are_scale_invariant(scale: float):
    distances, initial = _scale_fixture()
    reference = fcmdd(distances, initial, method="FCMdd", m=1.5, max_iter=3, alpha=0.0)
    scaled = fcmdd(
        distances * scale,
        initial,
        method="FCMdd",
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )

    assert scaled["mobileCenters"] == reference["mobileCenters"]
    np.testing.assert_allclose(scaled["memb"], reference["memb"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(
        scaled["functional"], reference["functional"] * scale, rtol=2e-14, atol=0.0
    )


def test_fcmdd_exact_zero_ties_split_membership_only_between_zero_medoids():
    distances = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [2.0, 2.0, 0.0]], dtype=float)
    initial = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    result = fcmdd(distances, initial, method="FCMdd", m=1.5, max_iter=1, alpha=0.0)

    assert result["mobileCenters"] == [0, 1]
    np.testing.assert_array_equal(result["memb"][:2], np.full((2, 2), 0.5))
    np.testing.assert_array_equal(result["memb"].sum(axis=1), np.ones(3))


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("NCdd", {"dnoise": 4.0}),
        ("PCMdd", {"eta": np.array([2.0, 5.0])}),
    ],
)
@pytest.mark.parametrize("scale", [1e-18, 1e-13, 1e13])
def test_fuzzy_variants_preserve_memberships_under_common_distance_scale(
    method: str, kwargs: dict[str, object], scale: float
):
    distances, initial = _scale_fixture()
    reference = fcmdd(distances, initial, method=method, m=1.5, max_iter=3, alpha=0.0, **kwargs)
    scaled_kwargs = {
        key: np.asarray(value) * scale if key == "eta" else float(value) * scale
        for key, value in kwargs.items()
    }
    scaled = fcmdd(
        distances * scale,
        initial,
        method=method,
        m=1.5,
        max_iter=3,
        alpha=0.0,
        **scaled_kwargs,
    )

    assert scaled["mobileCenters"] == reference["mobileCenters"]
    np.testing.assert_allclose(scaled["memb"], reference["memb"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(
        scaled["functional"], reference["functional"] * scale, rtol=2e-14, atol=0.0
    )


def test_adaptive_noise_is_invariant_to_common_weight_scale_below_absolute_epsilon():
    distances, initial = _scale_fixture()
    weights = np.array([1.0, 2.0, 1.0])
    reference = fcmdd(
        distances,
        initial,
        weights=weights,
        method="NCdd",
        dlambda=0.7,
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )
    weight_scale = 1e-15
    scaled = fcmdd(
        distances,
        initial,
        weights=weights * weight_scale,
        method="NCdd",
        dlambda=0.7,
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )

    np.testing.assert_allclose(scaled["dnoise"], reference["dnoise"], rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(scaled["memb"], reference["memb"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(
        scaled["functional"],
        reference["functional"] * weight_scale,
        rtol=2e-14,
        atol=0.0,
    )


def test_automatic_noise_preserves_representable_extreme_distance_scale():
    distances, initial = _scale_fixture()
    reference = fcmdd(
        distances,
        initial,
        method="NCdd",
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )
    distance_scale = 1e307
    scaled = fcmdd(
        distances * distance_scale,
        initial,
        method="NCdd",
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )

    assert np.isfinite(scaled["dnoise"])
    assert np.isfinite(scaled["functional"])
    assert scaled["mobileCenters"] == reference["mobileCenters"]
    np.testing.assert_allclose(
        scaled["dnoise"], reference["dnoise"] * distance_scale, rtol=2e-14, atol=0.0
    )
    np.testing.assert_allclose(scaled["memb"], reference["memb"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(
        scaled["functional"],
        reference["functional"] * distance_scale,
        rtol=2e-14,
        atol=0.0,
    )


def test_adaptive_noise_preserves_overflowing_finite_weight_sum_scale():
    distances, initial = _scale_fixture()
    weights = np.array([1.0, 1.0, 0.01])
    reference = fcmdd(
        distances,
        initial,
        weights=weights,
        method="NCdd",
        dlambda=0.7,
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )
    weight_scale = 1e308
    scaled_weights = weights * weight_scale
    assert np.all(np.isfinite(scaled_weights))
    with np.errstate(over="ignore"):
        assert np.isinf(scaled_weights.sum())
    scaled = fcmdd(
        distances,
        initial,
        weights=scaled_weights,
        method="NCdd",
        dlambda=0.7,
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )

    assert np.isfinite(scaled["dnoise"])
    assert np.isfinite(scaled["functional"])
    assert scaled["mobileCenters"] == reference["mobileCenters"]
    np.testing.assert_allclose(scaled["dnoise"], reference["dnoise"], rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(scaled["memb"], reference["memb"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(
        scaled["functional"],
        reference["functional"] * weight_scale,
        rtol=2e-14,
        atol=0.0,
    )


def test_adaptive_noise_preserves_subnormal_distance_and_huge_weight_scales():
    distances, initial = _scale_fixture()
    weights = np.array([1.0, 1.0, 0.01])
    reference = fcmdd(
        distances,
        initial,
        weights=weights,
        method="NCdd",
        dlambda=0.7,
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )
    scaled = fcmdd(
        distances * 1e-308,
        initial,
        weights=weights * 1e308,
        method="NCdd",
        dlambda=0.7,
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )

    assert scaled["dnoise"] == pytest.approx(3.77860696517413e-309, rel=2e-14, abs=0.0)
    assert scaled["functional"] == pytest.approx(0.003771084334395957, rel=2e-14)
    assert scaled["mobileCenters"] == reference["mobileCenters"]
    np.testing.assert_allclose(scaled["memb"], reference["memb"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(scaled["functional"], reference["functional"], rtol=2e-14)


def test_candidate_costs_preserve_positive_order_when_products_underflow():
    from weightedcluster.fuzzy import _stable_candidate_costs

    ordinary = _stable_candidate_costs(
        np.array([1.0]),
        1.0,
        np.array([1.0]),
        np.array([[7.0, 4.0]]),
    )
    scaled = _stable_candidate_costs(
        np.array([1.0]),
        1.0,
        np.array([1e-308]),
        np.array([[7e-308, 4e-308]]),
    )

    assert int(np.argmin(ordinary)) == int(np.argmin(scaled)) == 1
    np.testing.assert_allclose(scaled / scaled.min(), ordinary / ordinary.min(), rtol=2e-14)


def test_fcmdd_rechecks_nearly_tied_ordinary_candidate_costs_exactly() -> None:
    h = float.fromhex
    distances = np.array(
        [
            [
                0.0,
                h("0x1.0000000000002p+0"),
                h("0x1.0000000000002p+0"),
                h("0x1p+0"),
                h("0x1.ffffffffffffep-1"),
                h("0x1.fffffffffffffp-1"),
            ],
            [
                h("0x1.0000000000002p+0"),
                0.0,
                h("0x1p+0"),
                h("0x1.0000000000002p+0"),
                h("0x1.0000000000001p+0"),
                h("0x1.fffffffffffffp-1"),
            ],
            [
                h("0x1.0000000000002p+0"),
                h("0x1p+0"),
                0.0,
                h("0x1.fffffffffffffp-1"),
                h("0x1.0000000000001p+0"),
                h("0x1.fffffffffffffp-1"),
            ],
            [
                h("0x1p+0"),
                h("0x1.0000000000002p+0"),
                h("0x1.fffffffffffffp-1"),
                0.0,
                h("0x1.0000000000001p+0"),
                h("0x1.fffffffffffffp-1"),
            ],
            [
                h("0x1.ffffffffffffep-1"),
                h("0x1.0000000000001p+0"),
                h("0x1.0000000000001p+0"),
                h("0x1.0000000000001p+0"),
                0.0,
                h("0x1.fffffffffffffp-1"),
            ],
            [h("0x1.fffffffffffffp-1")] * 5 + [0.0],
        ]
    )
    initial = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]])

    exact_costs = [
        sum(Fraction.from_float(float(value)) for value in distances[:5, candidate])
        for candidate in range(5)
    ]
    assert exact_costs[0] - exact_costs[4] == Fraction(1, 2**52)

    from weightedcluster.fuzzy import _stable_candidate_costs

    costs = _stable_candidate_costs(
        initial[:, 0],
        2.0,
        np.ones(6),
        distances[:, :5],
    )
    assert int(np.argmin(costs)) == 4

    result = fcmdd(
        distances,
        initial,
        weights=np.ones(6),
        method="FCMdd",
        m=2.0,
        max_iter=1,
        alpha=0.0,
    )

    assert result["mobileCenters"] == [4, 5]


def test_public_fcmdd_preserves_medoids_when_distance_and_weights_are_subnormal():
    distances, initial = _scale_fixture()
    weights = np.array([1.0, 1.0, 0.01])
    reference = fcmdd(
        distances,
        initial,
        weights=weights,
        method="FCMdd",
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )
    scaled = fcmdd(
        distances * 1e-308,
        initial,
        weights=weights * 1e-308,
        method="FCMdd",
        m=1.5,
        max_iter=3,
        alpha=0.0,
    )

    assert scaled["mobileCenters"] == reference["mobileCenters"]
    np.testing.assert_allclose(scaled["memb"], reference["memb"], rtol=2e-14, atol=1e-15)


def test_positive_product_and_objective_classify_float_boundaries_safely():
    from weightedcluster.fuzzy import _stable_fuzzy_functional, _stable_positive_product

    maximum = np.finfo(float).max
    below_one = np.nextafter(1.0, 0.0)
    above_one = np.nextafter(1.0, 2.0)
    smallest = np.nextafter(0.0, 1.0)

    assert _stable_positive_product(maximum, below_one) < maximum
    assert _stable_positive_product(maximum, 1.0) == maximum
    assert np.isinf(_stable_positive_product(maximum, above_one))
    assert _stable_positive_product(smallest, 1.0) == smallest
    assert _stable_positive_product(smallest, 2.0) == 2.0 * smallest
    assert _stable_positive_product(smallest, 0.5) == 0.0

    distances = np.array([[maximum]])
    weights = np.array([1.0])
    below = _stable_fuzzy_functional(distances, np.array([[below_one]]), 1.0, weights)
    exact = _stable_fuzzy_functional(distances, np.array([[1.0]]), 1.0, weights)
    above = _stable_fuzzy_functional(distances, np.array([[above_one]]), 1.0, weights)
    assert below < maximum
    assert exact == maximum
    assert np.isinf(above)


@pytest.mark.filterwarnings("error")
def test_public_fcmdd_normalises_minsub_and_max_without_invalid_intermediates():
    smallest = np.nextafter(0.0, 1.0)
    maximum = np.finfo(float).max
    distances = np.array(
        [
            [0.0, smallest, maximum],
            [smallest, 0.0, maximum],
            [maximum, maximum, 0.0],
        ]
    )
    initial = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    result = fcmdd(distances, initial, method="FCMdd", m=1.5, max_iter=1, alpha=0.0)

    assert result["mobileCenters"] == [0, 2]
    assert np.all(np.isfinite(result["memb"]))
    np.testing.assert_array_equal(result["memb"].sum(axis=1), np.ones(3))
    np.testing.assert_array_equal(result["memb"][1], np.array([1.0, 0.0]))


def test_adaptive_noise_ignores_zero_weight_maximum_when_scaling_fallback():
    from weightedcluster.fuzzy import _stable_adaptive_noise_distance

    smallest = np.nextafter(0.0, 1.0)
    maximum = np.finfo(float).max
    distances = np.array([[maximum], [smallest], [2.0 * smallest]])
    weights = np.array([0.0, maximum, maximum])

    result = _stable_adaptive_noise_distance(distances, weights, 1, 1.0)

    assert result == 2.0 * smallest


def test_adaptive_noise_stabilizes_positive_subnormal_weighted_distance() -> None:
    from weightedcluster.fuzzy import _stable_adaptive_noise_distance

    distance = np.array([[0.6]])
    expected_hex = float(0.6).hex()
    for weight in (1.0, np.finfo(float).tiny):
        result = _stable_adaptive_noise_distance(
            distance,
            np.array([weight]),
            1,
            1.0,
        )
        assert result.hex() == expected_hex


def test_public_adaptive_noise_preserves_common_tiny_weight_scale() -> None:
    distances = np.array([[0.0, 0.6], [0.6, 0.0]])
    initial = np.ones((2, 1), dtype=float)
    expected_hex = "0x1.3333333333333p-2"

    for weight_scale in (1.0, np.finfo(float).tiny):
        result = fcmdd(
            distances,
            initial,
            weights=np.full(2, weight_scale),
            method="NCdd",
            m=2.0,
            dlambda=1.0,
            max_iter=1,
        )
        assert result["dnoise"].hex() == expected_hex


def test_fuzzy_functional_stabilizes_positive_subnormal_membership_power() -> None:
    from weightedcluster.fuzzy import _stable_fuzzy_functional

    result = _stable_fuzzy_functional(
        np.array([[0.5]]),
        np.array([[0.49]]),
        1024.0,
        np.ones(1),
    )

    assert result.hex() == "0x0.000000008e6efp-1022"


def test_public_pcmdd_functional_retains_subnormal_membership_power() -> None:
    exponent = 1024.0
    eta = np.array([0.5 / ((1.0 / 0.49 - 1.0) ** 1023)])
    result = fcmdd(
        np.array([[0.0, 0.5], [0.5, 0.0]]),
        np.ones((2, 1)),
        method="PCMdd",
        m=exponent,
        eta=eta,
        weights=np.ones(2),
        max_iter=1,
    )

    assert result["memb"][1, 0].hex() == "0x1.f5c28f5c28f5dp-2"
    assert result["functional"].hex() == "0x0.000000008e6efp-1022"


def test_seqclara_fuzzy_db_preserves_full_finite_common_weight_scale() -> None:
    from weightedcluster.seqclararange import _compute_clara_criteria

    distances = np.array(
        [[0.0, 1.0, 4.0, 5.0], [1.0, 0.0, 3.0, 4.0], [4.0, 3.0, 0.0, 1.0], [5.0, 4.0, 1.0, 0.0]]
    )
    medoids = np.array([0, 2])
    labels = np.array([0, 0, 1, 1])
    membership = np.eye(2)[labels]

    for weight_scale in (1.0, np.finfo(float).max, np.nextafter(0.0, 1.0)):
        criteria = _compute_clara_criteria(
            method_l="fuzzy",
            D_full=distances,
            D_to_m=distances[:, medoids],
            labels_full=labels,
            medoids_global_idx=medoids,
            memb_full_all=membership,
            fuzzifier_m=2.0,
            dnoise_value=None,
            case_weights=np.full(4, weight_scale),
            requested_criteria=["db"],
        )
        assert criteria["db"] == 0.25


def test_seqclara_fuzzy_criteria_preserve_ratios_and_true_ieee_boundaries() -> None:
    from weightedcluster.seqclararange import _compute_clara_criteria

    distances = np.array(
        [[0.0, 1.0, 4.0, 5.0], [1.0, 0.0, 3.0, 4.0], [4.0, 3.0, 0.0, 1.0], [5.0, 4.0, 1.0, 0.0]]
    )
    medoids = np.array([0, 2])
    labels = np.array([0, 0, 1, 1])
    membership = np.eye(2)[labels]

    expected = {
        1.0: (2.0, 0.5, 1.0),
        np.finfo(float).max: (float("inf"), np.finfo(float).max / 2.0, 0.0),
        np.nextafter(0.0, 1.0): (1e-323, 0.0, float("inf")),
    }
    for weight_scale, (distance, xb, pbm) in expected.items():
        criteria = _compute_clara_criteria(
            method_l="fuzzy",
            D_full=distances,
            D_to_m=distances[:, medoids],
            labels_full=labels,
            medoids_global_idx=medoids,
            memb_full_all=membership,
            fuzzifier_m=2.0,
            dnoise_value=None,
            case_weights=np.full(4, weight_scale),
            requested_criteria=["distance", "xb", "pbm", "ams"],
        )
        assert criteria["distance"] == distance
        assert criteria["xb"] == xb
        assert criteria["pbm"] == pbm
        assert criteria["ams"] == pytest.approx(0.8666666666666667, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("weight_scale", [1.0, np.nextafter(0.0, 1.0)])
def test_seqclara_fuzzy_ams_preserves_confidence_weight_products(
    weight_scale: float,
) -> None:
    from weightedcluster.seqclararange import _compute_clara_criteria

    distances_to_medoids = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 1.0], [0.8, 1.0]])
    memberships = np.array([[0.5, 0.5], [0.5, 0.5], [0.75, 0.25], [0.625, 0.375]])
    distances = np.zeros((4, 4), dtype=float)
    distances[0, 1] = distances[1, 0] = 1.0

    criteria = _compute_clara_criteria(
        method_l="fuzzy",
        D_full=distances,
        D_to_m=distances_to_medoids,
        labels_full=np.array([0, 1, 0, 0]),
        medoids_global_idx=np.array([0, 1]),
        memb_full_all=memberships,
        fuzzifier_m=2.0,
        dnoise_value=None,
        case_weights=np.full(4, weight_scale),
        requested_criteria=["ams"],
    )

    # Preserve the exact binary64 silhouette from the 0.8 input rather than
    # replacing it with the decimal-real shorthand 0.2.
    assert criteria["ams"].hex() == "0x1.9999999999999p-2"


@pytest.mark.parametrize(
    ("weight_scale", "expected_hex"),
    [
        (1.0, "0x1.b6db6db6db6dcp-2"),
        (np.nextafter(0.0, 1.0), "0x1.b6db6db6db6dbp-2"),
    ],
)
def test_seqclara_fuzzy_db_preserves_membership_weight_products(
    weight_scale: float,
    expected_hex: str,
) -> None:
    from weightedcluster.seqclararange import _compute_clara_criteria

    distances_to_medoids = np.array([[0.0, 4.0], [4.0, 0.0], [1.0, 3.0], [3.0, 1.0]])
    memberships = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]])
    distances = np.zeros((4, 4), dtype=float)
    distances[0, 1] = distances[1, 0] = 4.0

    criteria = _compute_clara_criteria(
        method_l="fuzzy",
        D_full=distances,
        D_to_m=distances_to_medoids,
        labels_full=np.array([0, 1, 0, 1]),
        medoids_global_idx=np.array([0, 1]),
        memb_full_all=memberships,
        fuzzifier_m=2.0,
        dnoise_value=None,
        case_weights=np.full(4, weight_scale),
        requested_criteria=["db"],
    )

    # Cluster diameters are 5/7 and 1, with medoid separation 4.
    assert criteria["db"].hex() == expected_hex


def test_seqclara_fuzzy_xb_and_pbm_keep_subnormal_functional_ratios() -> None:
    from weightedcluster.seqclararange import _compute_clara_criteria

    minimum_positive = np.nextafter(0.0, 1.0)
    distances = np.array([[0.0, minimum_positive], [minimum_positive, 0.0]])
    criteria = _compute_clara_criteria(
        method_l="fuzzy",
        D_full=distances,
        D_to_m=distances,
        labels_full=np.array([0, 1]),
        medoids_global_idx=np.array([0, 1]),
        memb_full_all=np.full((2, 2), 0.5),
        fuzzifier_m=2.0,
        dnoise_value=None,
        case_weights=np.ones(2),
        requested_criteria=["distance", "xb", "pbm"],
    )

    assert criteria == {"distance": 0.0, "xb": 0.5, "pbm": 0.25}


def test_seqclara_crisp_xb_and_pbm_keep_subnormal_functional_ratios() -> None:
    from weightedcluster.seqclararange import _compute_clara_criteria

    minimum_positive = np.nextafter(0.0, 1.0)
    distances = np.array(
        [
            [0.0, minimum_positive, 2.0 * minimum_positive],
            [minimum_positive, 0.0, minimum_positive],
            [2.0 * minimum_positive, minimum_positive, 0.0],
        ]
    )
    criteria = _compute_clara_criteria(
        method_l="pam",
        D_full=distances,
        D_to_m=distances[:, [0, 2]],
        labels_full=np.array([0, 0, 1]),
        medoids_global_idx=np.array([0, 2]),
        memb_full_all=None,
        fuzzifier_m=2.0,
        dnoise_value=None,
        case_weights=np.array([minimum_positive, 0.5, minimum_positive]),
        requested_criteria=["distance", "xb", "pbm"],
    )

    assert criteria == {"distance": 0.0, "xb": 0.25, "pbm": 4.0}


def test_seqclara_fuzzy_xb_keeps_membership_power_before_materialization() -> None:
    from weightedcluster.seqclararange import _compute_clara_criteria

    minimum_positive = np.nextafter(0.0, 1.0)
    distances = np.array(
        [
            [0.0, minimum_positive, 1.0],
            [minimum_positive, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    criteria = _compute_clara_criteria(
        method_l="fuzzy",
        D_full=distances,
        D_to_m=distances[:, [0, 1]],
        labels_full=np.array([0, 1, 0]),
        medoids_global_idx=np.array([0, 1]),
        memb_full_all=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        fuzzifier_m=2000.0,
        dnoise_value=None,
        case_weights=np.ones(3),
        requested_criteria=["distance", "xb", "pbm"],
    )

    assert criteria["distance"] == 0.0
    assert criteria["xb"].hex() == "0x1.0000000000000p-925"
    assert criteria["pbm"] == 0.0


def test_candidate_costs_preserve_order_when_membership_power_log_overflows():
    from weightedcluster.fuzzy import _stable_candidate_costs

    costs = _stable_candidate_costs(
        np.array([0.1]),
        1e308,
        np.array([1.0]),
        np.array([[7.0, 4.0]]),
    )

    assert np.all(np.isfinite(costs))
    assert int(np.argmin(costs)) == 1
    np.testing.assert_allclose(costs / costs.min(), np.array([1.75, 1.0]), rtol=2e-14)


def test_candidate_cost_fast_path_keeps_blas_without_term_matrix_allocation():
    import inspect

    from weightedcluster.fuzzy import _stable_candidate_costs

    source = inspect.getsource(_stable_candidate_costs)
    assert "ordinary_costs = ordinary_weights @ candidate_distances" in source
    assert "ordinary_weights[:, None]" not in source


def test_pcmdd_eta_coercion_overflow_raises_documented_value_error():
    distances, initial = _scale_fixture()

    with pytest.raises(ValueError, match="eta must be a finite positive vector"):
        fcmdd(
            distances,
            initial,
            method="PCMdd",
            eta=[10**1000, 1.0],
        )


@pytest.mark.parametrize("scale", [1e-18, 1e-13, 1.0, 1e13])
def test_fcmdd_scaled_boundary_matches_live_r(r_parity_session, scale: float):
    distances, initial = _scale_fixture()
    scaled_distances = distances * scale
    python_result = fcmdd(
        scaled_distances,
        initial,
        method="FCMdd",
        m=1.5,
        max_iter=100,
    )
    r_result = r_parity_session.r_wfcmdd(
        scaled_distances,
        initial,
        method="FCMdd",
        m=1.5,
    )

    np.testing.assert_array_equal(
        np.asarray(python_result["mobileCenters"], dtype=int),
        np.asarray(r_result["mobile_centers_1b"], dtype=int) - 1,
    )
    np.testing.assert_allclose(python_result["memb"], r_result["memb"], rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(
        python_result["functional"], r_result["functional"], rtol=2e-13, atol=0.0
    )


def test_fuzzy_clara_full_membership_respects_fuzziness_below_absolute_floor(
    monkeypatch,
):
    """R-04: full-data memberships use the exact positive ``m - 1``."""
    import importlib

    import weightedcluster.fuzzy as fuzzy_module

    seqclara_module = importlib.import_module("weightedcluster.seqclararange")

    distance_delta = 5e-13
    full_to_medoids = np.array(
        [
            [1.0, 1.0 + distance_delta],
            [1.0 + distance_delta, 1.0],
        ],
        dtype=float,
    )

    def fake_fcmdd(_distances, membership, **_kwargs):
        return {
            "memb": np.asarray(membership, dtype=float),
            "mobileCenters": [0, 1],
            "functional": 0.0,
            "dnoise": None,
        }

    monkeypatch.setattr(fuzzy_module, "fcmdd", fake_fcmdd)
    monkeypatch.setattr(
        seqclara_module,
        "sequences_distance_between",
        lambda *_args, **_kwargs: full_to_medoids.copy(),
    )

    _labels, membership, _medoids, _distances, _noise = seqclara_module._cluster_fuzzy(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        2,
        np.ones(2, dtype=float),
        "fuzzy",
        1,
        0,
        np.array([[0], [1]], dtype=int),
        np.array([[0], [1]], dtype=int),
        np.array([0, 1], dtype=int),
        "HAM",
        1.0,
        None,
        None,
        1.0 + distance_delta,
        None,
        None,
        None,
        init_memberships_sub=[np.eye(2, dtype=float)],
    )

    expected = np.array(
        [
            [0.7310585786299557, 0.26894142137004423],
            [0.26894142137004423, 0.7310585786299557],
        ]
    )
    np.testing.assert_allclose(membership, expected, rtol=2e-14, atol=0.0)

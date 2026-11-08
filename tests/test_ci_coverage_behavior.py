from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.r_parity_utils import import_optional_or_skip
from weightedcluster._fallback import weighted_medoids
from weightedcluster._threads import limit_threads
from weightedcluster._validation import (
    validate_distance_matrix,
    validate_kvals,
    validate_linkage_matrix,
)
from weightedcluster.plot import (
    _seqnull_extract_null_matrix,
    _seqnull_extract_observed_vector,
    _seqnull_standardize_by_k,
    plot_bootclustrange,
    plot_clustrange,
)
from weightedcluster.range import as_clustrange
from weightedcluster.seqclustname import seqclustname

REPO_ROOT = Path(__file__).resolve().parents[1]


def _distance_matrix() -> np.ndarray:
    points = np.array([0.0, 0.2, 2.0, 2.3])
    return np.abs(points[:, None] - points[None, :])


def test_behavior_tests_do_not_mutate_numpy_legacy_global_rng():
    legacy_calls = ("np.random.seed(", "np.random.rand(", "np.random.randint(")

    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8")
        for call in legacy_calls:
            assert call not in text, f"{path.relative_to(REPO_ROOT)} uses {call}"


def test_optional_seaborn_loader_is_measured_in_both_supported_states(monkeypatch):
    import builtins
    import inspect

    import weightedcluster.plot as plot_module

    sentinel = object()
    monkeypatch.setitem(sys.modules, "seaborn", sentinel)
    assert plot_module._get_sns() is sentinel

    monkeypatch.delitem(sys.modules, "seaborn", raising=False)
    real_import = builtins.__import__

    def import_without_seaborn(name, *args, **kwargs):
        if name == "seaborn":
            raise ImportError("forced absent seaborn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_seaborn)
    assert plot_module._get_sns() is None
    assert "pragma: no cover" not in inspect.getsource(plot_module._get_sns)


def test_c_and_en_us_collation_have_the_distinct_order_that_requires_pinning():
    script = (
        "import locale; locale.setlocale(locale.LC_COLLATE, ''); "
        "print(','.join(sorted(['a', 'B'], key=locale.strxfrm)))"
    )
    orders = {}
    for name in ("C.UTF-8", "en_US.UTF-8"):
        environment = dict(os.environ, LC_ALL=name, LANG=name)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"locale {name} is not installed")
        orders[name] = completed.stdout.strip()
    assert orders["C.UTF-8"] == "B,a"
    assert orders["en_US.UTF-8"] == "a,B"


@pytest.mark.skipif(
    not (REPO_ROOT / ".github" / "workflows" / "ci.yml").is_file(),
    reason="needs .github/workflows/ci.yml, which the published package leaves out",
)
def test_ci_declares_real_backend_tiers_and_pins_strict_locale():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "backend: [compiled, pure, numba]" in workflow
    assert 'mv "$core_path" "$core_path.disabled"' in workflow
    assert 'mv "$core_path.disabled" "$core_path"' in workflow
    assert "verified backend tier" in workflow
    assert "LC_ALL: C.UTF-8" in workflow
    assert "WC_PARITY_STRICT: '1'" in workflow
    assert "-rsxX -p no:randomly" in workflow
    assert "--junitxml=strict-parity.xml" in workflow
    assert "zero skips/xfails" in workflow


def test_pure_fallback_medoids_validation_and_fixed_initialization():
    D = _distance_matrix()

    with pytest.raises(ValueError, match="square"):
        weighted_medoids(np.ones((2, 3)), 2)
    with pytest.raises(ValueError, match="k out of range"):
        weighted_medoids(D, 1)
    with pytest.raises(ValueError, match=r"shape \(k,\)"):
        weighted_medoids(D, 2, init_medoids=np.array([[0, 2]]))
    with pytest.raises(ValueError, match="out of range"):
        weighted_medoids(D, 2, init_medoids=np.array([0, 4]))
    with pytest.raises(ValueError, match="unique"):
        weighted_medoids(D, 2, init_medoids=np.array([0, 0]))

    labels, medoids, objective = weighted_medoids(
        D,
        2,
        weights=[1.0, -1.0, 2.0, 1.0],
        n_init=2,
        random_state=7,
        init_medoids=np.array([0, 2]),
    )
    assert labels.shape == (4,)
    assert medoids.shape == (2,)
    assert objective >= 0.0


def test_thread_limits_restore_present_and_absent_environment(monkeypatch):
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "9")

    with limit_threads(omp=2, openblas=3, blas=4):
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["MKL_NUM_THREADS"] == "9"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
        assert os.environ["BLAS_NUM_THREADS"] == "4"

    assert "OMP_NUM_THREADS" not in os.environ
    assert os.environ["MKL_NUM_THREADS"] == "9"
    assert "OPENBLAS_NUM_THREADS" not in os.environ
    assert "BLAS_NUM_THREADS" not in os.environ


def test_seqclustname_requires_diss_and_returns_factor_equivalent_categories():
    X = np.array([[0, 0], [0, 1], [2, 2], [2, 1]])
    groups = np.array([1, 1, 2, 2])
    D = _distance_matrix()

    with pytest.raises(TypeError, match="required positional argument: 'diss'"):
        seqclustname(X, groups)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="group must have one value per sequence"):
        seqclustname(X, groups[:3], D)
    with pytest.raises(ValueError, match="diss must be a square matrix"):
        seqclustname(X, groups, D[:3, :3])

    result = seqclustname(X, groups, D, weighted=False, perc=True)
    assert isinstance(result, pd.Categorical)
    assert result.codes.tolist() == [0, 0, 1, 1]
    assert all(str(label).endswith("(50%)") for label in result.categories)


def test_validation_error_and_warning_contracts():
    with pytest.raises(TypeError, match="numpy array"):
        validate_distance_matrix([[0.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be 2D"):
        validate_distance_matrix(np.array([0.0]))
    with pytest.raises(ValueError, match="must be square"):
        validate_distance_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-empty"):
        validate_distance_matrix(np.empty((0, 0)))

    asymmetric = np.array([[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="not symmetric"):
        validate_distance_matrix(asymmetric)
    with pytest.warns(UserWarning, match="not symmetric"):
        validate_distance_matrix(asymmetric, strict=False)
    nonzero_diagonal = np.array([[0.5, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="diagonal is not zero"):
        validate_distance_matrix(nonzero_diagonal)
    with pytest.warns(UserWarning, match="diagonal is not zero"):
        validate_distance_matrix(nonzero_diagonal, strict=False)
    with pytest.raises(ValueError, match="finite"):
        validate_distance_matrix(np.array([[0.0, np.inf], [np.inf, 0.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        validate_distance_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]]))


def test_label_k_and_linkage_validation_contracts():
    with pytest.raises(ValueError, match="non-empty"):
        validate_kvals([])
    assert validate_kvals(np.array([3, 2]), max_k=3) == [3, 2]
    with pytest.raises(ValueError, match="must be integers"):
        validate_kvals(["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be integers"):
        validate_kvals([2.9])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be >= 2"):
        validate_kvals([1])
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_kvals([4], max_k=3)
    assert validate_kvals([3, 2], max_k=3) == [3, 2]
    with pytest.raises(ValueError, match="duplicates"):
        validate_kvals([3, 2, 3], max_k=3)

    with pytest.raises(ValueError, match="must be 2D"):
        validate_linkage_matrix(np.array([1.0, 2.0]), 2)
    with pytest.raises(ValueError, match="4 columns"):
        validate_linkage_matrix(np.ones((1, 3)), 2)
    with pytest.raises(ValueError, match="rows"):
        validate_linkage_matrix(np.ones((2, 4)), 2)
    validate_linkage_matrix(np.array([[0.0, 1.0, 1.0, 2.0]]), 2)


def test_as_clustrange_uses_predictor_columns_and_rejects_duplicate_group_counts():
    D = _distance_matrix()
    with pytest.raises(ValueError, match="duplicate 'row.names'"):
        as_clustrange(
            {
                "clusterbad": [1, 1, 2, 2],
                object(): [1, 1, 2, 2],
                "cluster2": [1, 1, 2, 2],
            },
            D,
        )

    class BadArray:
        def __array__(self):
            raise TypeError("not array-like")

    with pytest.raises(NotImplementedError, match="predictor frames"):
        as_clustrange(BadArray(), D)
    with pytest.raises(ValueError, match="different number of elements"):
        as_clustrange(np.ones((2, 3)), D)


def test_seqnull_plot_extractors_accept_all_supported_storage_shapes():
    kvals = [2, 3]
    obs = _seqnull_extract_observed_vector({"clustrange": {"stats": {"ASW": [0.1]}}}, "ASW", kvals)
    np.testing.assert_allclose(obs[:1], [0.1])
    assert np.isnan(obs[1])

    nested_obs = _seqnull_extract_observed_vector(
        {"clustrange": {"stats": {2: {"ASW": 0.2}, "3": {"ASW": 0.3}}}},
        "ASW",
        kvals,
    )
    np.testing.assert_allclose(nested_obs, [0.2, 0.3])
    assert np.isnan(_seqnull_extract_observed_vector({}, "ASW", kvals)).all()

    np.testing.assert_array_equal(
        _seqnull_extract_null_matrix({"stats": {"ASW": [1.0, 2.0]}}, "ASW", [2]),
        [[1.0], [2.0]],
    )
    np.testing.assert_array_equal(
        _seqnull_extract_null_matrix({"stats": {"ASW": [1.0, 2.0]}}, "ASW", kvals),
        [[1.0, 2.0]],
    )
    np.testing.assert_array_equal(
        _seqnull_extract_null_matrix({"stats": {"ASW": [1.0, 2.0, 3.0]}}, "ASW", kvals),
        [[1.0], [2.0], [3.0]],
    )
    np.testing.assert_array_equal(
        _seqnull_extract_null_matrix({"stats": {"ASW": [[1.0, 2.0], [3.0, 4.0]]}}, "ASW", kvals),
        [[1.0, 2.0], [3.0, 4.0]],
    )
    np.testing.assert_array_equal(
        _seqnull_extract_null_matrix(
            {"stats": {"ASW": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]}},
            "ASW",
            [2, 3, 4],
        ),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )

    for key in ("boot_matrices",):
        np.testing.assert_array_equal(
            _seqnull_extract_null_matrix({key: {"ASW": [1.0, 2.0]}}, "ASW", [2]),
            [[1.0], [2.0]],
        )
        np.testing.assert_array_equal(
            _seqnull_extract_null_matrix({key: {"ASW": [1.0, 2.0]}}, "ASW", kvals),
            [[1.0, 2.0]],
        )
        np.testing.assert_array_equal(
            _seqnull_extract_null_matrix({key: {"ASW": [1.0, 2.0, 3.0]}}, "ASW", kvals),
            [[1.0], [2.0], [3.0]],
        )
        np.testing.assert_array_equal(
            _seqnull_extract_null_matrix({key: {"ASW": [[1.0, 2.0], [3.0, 4.0]]}}, "ASW", kvals),
            [[1.0, 2.0], [3.0, 4.0]],
        )
        np.testing.assert_array_equal(
            _seqnull_extract_null_matrix(
                {key: {"ASW": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]}},
                "ASW",
                [2, 3, 4],
            ),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )

    ragged = _seqnull_extract_null_matrix(
        {"boot": {2: {"ASW": [1.0, 2.0]}, 3: {"ASW": [3.0]}}}, "ASW", kvals
    )
    np.testing.assert_allclose(ragged[0], [1.0, 3.0])
    assert np.isnan(ragged[1, 1])
    assert _seqnull_extract_null_matrix({}, "ASW", kvals).shape == (0, 2)


def test_seqnull_plot_standardization_and_max_t_edge_contracts():
    obs = np.array([1.0, 2.0, np.nan])
    null = np.array([[np.nan, 1.0, 2.0], [np.nan, np.nan, 2.0], [np.nan, np.nan, 2.0]])
    obs_n, null_n = _seqnull_standardize_by_k(obs, null)
    assert np.isnan(obs_n[0]) and np.isnan(null_n[:, 0]).all()
    # WeightedCluster's normstatcqi uses mean/sd without na.rm and preserves
    # undefined constant-column normalization.
    assert np.isnan(obs_n[1]) and np.isnan(null_n[:, 1]).all()
    assert np.isnan(obs_n[2])
    assert np.isnan(null_n[:, 2]).all()

    raw_obs, raw_null = _seqnull_standardize_by_k(obs, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(raw_obs, obs)
    np.testing.assert_array_equal(raw_null, [1.0, 2.0])


def test_clustrange_plotting_covers_owned_and_supplied_axes(monkeypatch):
    plt = import_optional_or_skip("matplotlib.pyplot")
    range_obj = {
        "kvals": [2, 3],
        "stats": {"ASW": [0.2, 0.4], "R2": [0.3, 0.5], "CH": [2.0, 3.0]},
    }

    monkeypatch.setattr("weightedcluster.plot._get_sns", lambda: None)
    fig_owned, ax_owned = plot_clustrange(range_obj, stat="all", norm="range")
    assert ax_owned.get_xlabel() == "N clusters"

    fig_supplied, supplied = plt.subplots()
    returned_fig, returned_ax = plot_clustrange(range_obj, stat="noCH", norm="none", ax=supplied)
    assert returned_fig is fig_supplied
    assert returned_ax is supplied

    fig_list, _ = plot_clustrange(range_obj, stat=["ASW"], norm="zscoremed")
    fig_empty, ax_empty = plot_bootclustrange({"kvals": [], "boot": {}}, stat="ASW")
    assert ax_empty.texts[0].get_text() == "No bootstrap data for ASW"

    boot_obj = {
        "kvals": [2, 3],
        "boot": {2: {"ASW": [0.1, 0.2]}, 3: {"ASW": [0.3, 0.4]}},
    }
    fallback_figures = []
    for kind in ("line", "box", "density"):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_bootclustrange(boot_obj, stat="ASW", kind=kind, ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        fallback_figures.append(fig)

    for fig in (fig_owned, fig_supplied, fig_list, fig_empty, *fallback_figures):
        plt.close(fig)

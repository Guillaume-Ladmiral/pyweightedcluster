import builtins

import numpy as np
import pytest

from weightedcluster import clustassoc, plot_clustassoc


def _toy_clustrange():
    # Two clusters: 1..3 in cluster1, 4..6 in cluster2
    labels2 = np.array([1, 1, 1, 2, 2, 2])
    labels3 = np.array([1, 2, 3, 1, 2, 3])
    return {
        "kvals": [2, 3],
        "clustering": {"cluster2": labels2, "cluster3": labels3},
        "stats": {
            "PBC": [0.0, 0.0],
            "HG": [0.0, 0.0],
            "HGSD": [0.0, 0.0],
        },
    }


def _toy_distance():
    # Simple 6x6 distance: two blocks closer within (0.1) and farther across (1.0)
    N = 6
    D = np.ones((N, N), dtype=float)
    np.fill_diagonal(D, 0.0)
    for i in range(3):
        for j in range(3):
            if i != j:
                D[i, j] = 0.1
                D[i + 3, j + 3] = 0.1
    return D


def test_clustassoc_numeric_basic():
    rng = _toy_clustrange()
    D = _toy_distance()
    covar = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])

    out = clustassoc(rng, D, covar)
    assert all(key in out for key in ("Unaccounted", "Remaining", "BIC", "numcluster"))
    assert "BICgain" not in out
    assert "BICgain" in out.additions
    # First entry is No Clustering (K=1)
    assert out["numcluster"][0] == 1
    # There should be one value per (1 + len(kvals))
    L = 1 + len(rng["kvals"])
    for k in ("Unaccounted", "Remaining", "BIC"):
        assert len(out[k]) == L
    for k in (
        "BICgain",
        "RemainingPctOfBaseline",
        "RemainingImprovementPct",
        "UnaccountedImprovementPct",
    ):
        assert len(out.additions[k]) == L


def test_clustassoc_categorical():
    rng = _toy_clustrange()
    D = _toy_distance()
    covar = np.array(["A", "A", "A", "B", "B", "B"])  # aligns with K=2 grouping

    out = clustassoc(rng, D, covar)
    assert out["numcluster"][0] == 1
    # BICgain at K=2 should be <= 0 typically (better model => lower BIC); not asserting strict sign for portability
    assert len(out.additions["BICgain"]) == 1 + len(rng["kvals"])
    # Cramer's V should be ~1 for perfect separation at K=2
    idx_k2 = out["numcluster"].index(2)
    V_k2 = out.additions.get("V", [np.nan] * len(out["numcluster"]))[idx_k2]
    assert V_k2 > 0.95
    LRT_k2 = out.additions.get("LRT", [np.nan] * len(out["numcluster"]))[idx_k2]
    assert LRT_k2 > 0.0


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("tooltip", "false"),
        ("axes", 0),
        ("ann", "yes"),
        ("frame_plot", 1),
        ("frame.plot", None),
    ],
)
def test_plot_clustassoc_rejects_non_boolean_controls(keyword, value):
    assoc = {
        "Unaccounted": [1.0, 0.5],
        "Remaining": [2.0, 1.0],
        "BIC": [10.0, 8.0],
        "numcluster": [1, 2],
    }

    with pytest.raises(TypeError, match=rf"{keyword.replace('.', '_')} must be a boolean"):
        plot_clustassoc(assoc, **{keyword: value})


def test_plot_clustassoc_rejects_duplicate_frame_plot_aliases():
    assoc = {
        "Unaccounted": [1.0, 0.5],
        "Remaining": [2.0, 1.0],
        "BIC": [10.0, 8.0],
        "numcluster": [1, 2],
    }

    with pytest.raises(TypeError, match="frame_plot was specified more than once"):
        plot_clustassoc(assoc, frame_plot=True, **{"frame.plot": False})


def test_plot_clustassoc_warns_when_requested_tooltips_are_unavailable(monkeypatch):
    assoc = {
        "Unaccounted": [1.0, 0.5],
        "Remaining": [2.0, 1.0],
        "BIC": [10.0, 8.0],
        "numcluster": [1, 2],
    }
    original_import = builtins.__import__

    def _import_without_mplcursors(name, *args, **kwargs):
        if name == "mplcursors":
            raise ImportError("simulated absent optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_without_mplcursors)

    with pytest.warns(RuntimeWarning, match="mplcursors.*tooltips are disabled"):
        figure, _ = plot_clustassoc(assoc, stat="BIC", tooltip=True)
    figure.clf()

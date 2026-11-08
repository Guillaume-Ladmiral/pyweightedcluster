from __future__ import annotations

import numpy as np
import pytest

from tests.r_parity_utils import import_optional_or_skip


def _toy_sc_obj() -> dict:
    return {
        "kvals": [2, 3, 4],
        "stats": {
            "ARI>0.8": [1.0, 2.0, 3.0],
            "JC>0.8": [1.0, 1.5, 2.0],
        },
        "quality_stats": {
            "ARImean": [0.4, 0.5, 0.6],
            "JCmean": [0.3, 0.35, 0.45],
        },
    }


def test_plot_seqclararange_stability_reuses_passed_axes():
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from weightedcluster.plot import plot_seqclararange_stability

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    out_fig, (out_ax1, out_ax2) = plot_seqclararange_stability(_toy_sc_obj(), ax=(ax1, ax2))

    assert out_fig is fig
    assert out_ax1 is ax1
    assert out_ax2 is ax2
    assert len(ax1.lines) >= 2
    assert len(ax2.lines) >= 2


def test_plot_seqclararange_stability_rejects_single_axis():
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from weightedcluster.plot import plot_seqclararange_stability

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    with pytest.raises(ValueError, match="sequence of two Axes"):
        plot_seqclararange_stability(_toy_sc_obj(), ax=ax)


def test_plot_seqclararange_stability_uses_real_seqclararange_output():
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)

    from weightedcluster import seqclararange
    from weightedcluster.plot import plot_seqclararange_stability

    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )
    out = seqclararange(
        X,
        R=3,
        sample_size=3,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria="distance",
        stability=True,
        random_state=7,
    )
    fig, (ax1, ax2) = plot_seqclararange_stability(out)

    assert fig is ax1.figure is ax2.figure
    assert len(ax1.lines) >= 2
    assert len(ax2.lines) >= 2
    assert all(len(line.get_xdata()) == 1 for line in ax2.lines if len(line.get_xdata()) > 0)


def test_plot_seqclararange_stability_does_not_fall_back_to_stats_for_means(monkeypatch):
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)
    import weightedcluster.plot as plotmod

    monkeypatch.setattr(plotmod, "_get_sns", lambda: None)
    sc_obj = {
        "kvals": [2, 3],
        "stats": {
            "ARI>0.8": [1.0, 2.0],
            "JC>0.8": [0.5, 1.0],
            "ARImean": [9.0, 9.5],
            "JCmean": [8.0, 8.5],
        },
        "quality_stats": {},
    }

    fig, (_ax1, ax2) = plotmod.plot_seqclararange_stability(sc_obj)

    assert fig is ax2.figure
    assert len(ax2.lines) == 2
    for line in ax2.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        assert np.all(np.isnan(y))

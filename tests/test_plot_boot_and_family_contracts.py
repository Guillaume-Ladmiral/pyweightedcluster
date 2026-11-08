from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.r_parity_utils import import_optional_or_skip


def test_plot_bootclustrange_line_uses_cluster_counts_on_x_axis():
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)

    from weightedcluster.plot import plot_bootclustrange

    boot_obj = {
        "kvals": [2, 3],
        "boot": {
            2: {"ASW": [0.1, 0.2, 0.3]},
            3: {"ASW": [0.4, 0.5, 0.6]},
        },
    }
    fig, ax = plot_bootclustrange(boot_obj, stat="ASW", kind="line")

    assert fig is ax.figure
    assert ax.get_xlabel() == "N clusters"
    assert any(np.array_equal(np.asarray(line.get_xdata()), np.array([2, 3])) for line in ax.lines)


def test_plot_bootclustrange_rejects_unknown_kind():
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)

    from weightedcluster.plot import plot_bootclustrange

    boot_obj = {
        "kvals": [2],
        "boot": {
            2: {"ASW": [0.1, 0.2]},
        },
    }
    with pytest.raises(ValueError, match="kind must be one of"):
        plot_bootclustrange(boot_obj, stat="ASW", kind="unknown")


def test_plot_bootclustrange_box_avoids_seaborn_matplotlib_deprecation(monkeypatch):
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)

    import weightedcluster.plot as plotmod

    class DeprecatedSeabornPath:
        @staticmethod
        def boxplot(**_kwargs):
            raise AssertionError("box rendering must not use Seaborn's deprecated vert path")

    monkeypatch.setattr(plotmod, "_get_sns", lambda: DeprecatedSeabornPath())
    boot_obj = {
        "kvals": [2, 4],
        "boot": {
            2: {"ASW": [0.1, 0.2, 0.3]},
            4: {"ASW": [0.4, 0.5, 0.6]},
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", mpl.MatplotlibDeprecationWarning)
        fig, ax = plotmod.plot_bootclustrange(boot_obj, stat="ASW", kind="box")

    assert fig is ax.figure
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["2", "4"]


def test_plot_proptree_panel_propagates_semantic_data_errors(monkeypatch):
    mpl = import_optional_or_skip("matplotlib")
    mpl.use("Agg", force=True)

    import weightedcluster.plot as plotmod

    def _boom(*args, **kwargs):
        raise RuntimeError("plot data exploded")

    monkeypatch.setattr(plotmod, "proptree_plot_data", _boom)

    tree = {
        "root": {
            "id": 1,
            "kids": None,
            "split": None,
        },
        "data": None,
    }

    with pytest.raises(RuntimeError, match="plot data exploded"):
        plotmod.plot_proptree_panel(tree)

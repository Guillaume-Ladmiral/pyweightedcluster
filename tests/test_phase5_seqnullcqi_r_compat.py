from __future__ import annotations

import numpy as np
import pytest

from tests.r_parity_utils import import_optional_or_skip
from weightedcluster import seqnullcqi, wcKMedoids
from weightedcluster._seqdist import sequences_distance
from weightedcluster.plot import plot_seqnullcqi, print_seqnullcqi
from weightedcluster.schemas import ClustrangeResult


def _toy_sequences() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )


def _toy_clustrange_without_stats(X: np.ndarray) -> dict:
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    return ClustrangeResult({"kvals": [2], "clustering": {"cluster2": labels}})


def test_seqnullcqi_returns_r_style_stats_and_metadata():
    X = _toy_sequences()
    cr = _toy_clustrange_without_stats(X)
    out = seqnullcqi(
        X,
        cr,
        R=3,
        model="combined",
        seqdist_args={"method": "HAM"},
        random_state=41,
    )

    assert out["R"] == 3
    assert out["kmedoid"] is False
    assert out["hclust.method"] == "ward.D"
    assert out["seqdist.args"]["method"] == "HAM"
    assert out["seqdata"].shape == (3 * X.shape[0], X.shape[1])

    assert "ASW" in out["stats"]
    mat = np.asarray(out["stats"]["ASW"], dtype=float)
    assert mat.shape == (3, 1)

    assert "ASW" in out.additions["boot"][2]
    assert len(out.additions["boot"][2]["ASW"]) == 3
    assert "boot_matrices" not in out

    assert "ASW" in out["clustrange"]["stats"]
    assert len(out["clustrange"]["stats"]["ASW"]) == 1


def test_seqnullcqi_supports_kmedoid_reclustering_flag():
    X = _toy_sequences()
    cr = _toy_clustrange_without_stats(X)
    out = seqnullcqi(
        X,
        cr,
        R=2,
        model="combined",
        seqdist_args={"method": "HAM"},
        kmedoid=True,
        random_state=42,
    )
    assert out["kmedoid"] is True
    assert np.asarray(out["stats"]["ASW"], dtype=float).shape == (2, 1)


def test_seqnullcqi_requires_kvals_field():
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    cr = ClustrangeResult({"clustering": {"cluster2": labels}})

    with pytest.raises(ValueError, match="contain kvals"):
        seqnullcqi(
            X,
            cr,
            R=1,
            model="combined",
            seqdist_args={"method": "HAM"},
            random_state=42,
        )


def test_seqnullcqi_accepts_clustrange_stats_grouped_by_k():
    X = _toy_sequences()
    D = sequences_distance(X, metric="HAM", normalize=True)
    labels = wcKMedoids(D, 2)["clustering"]
    cr = ClustrangeResult(
        {
            "kvals": [2],
            "clustering": {"cluster2": labels},
            "stats": {2: {"ASW": 0.123}},
        }
    )

    out = seqnullcqi(
        X,
        cr,
        R=1,
        model="combined",
        seqdist_args={"method": "HAM"},
        random_state=44,
    )

    assert out["clustrange"]["stats"]["ASW"] == pytest.approx([0.123])
    assert "CH" in out["clustrange"]["stats"]


def test_seqnullcqi_plot_and_print_support_r_style_type_names():
    import_optional_or_skip("matplotlib")

    X = _toy_sequences()
    cr = _toy_clustrange_without_stats(X)
    out = seqnullcqi(
        X,
        cr,
        R=2,
        model="combined",
        seqdist_args={"method": "HAM"},
        random_state=43,
    )

    fig1, ax1 = plot_seqnullcqi(out, stat="ASW", type="boxplot")
    assert fig1 is not None and ax1 is not None

    fig2, ax2 = plot_seqnullcqi(out, stat="ASW", kind="seqdplot")
    assert fig2 is not None and ax2 is not None

    tab = print_seqnullcqi(out, norm=True, quant=0.9, digits=2)
    assert "ASW" in tab.columns
    assert "Null Max-T 0.9 interval" in tab.index


def test_plot_seqnullcqi_beanplot_alias_follows_installed_line_branch():
    import_optional_or_skip("matplotlib")

    bcq = {
        "clustrange": {"kvals": [2], "stats": {"ASW": [0.2]}},
        "stats": {"ASW": np.array([[0.1], [0.15]], dtype=float)},
    }

    figure, axes = plot_seqnullcqi(bcq, stat="ASW", kind="beanplot")
    assert len(figure.axes) == 2
    assert len(axes) == 2


def test_print_seqnullcqi_boot_matrices_fallback_and_plot_line_support():
    import_optional_or_skip("matplotlib")

    bcq = {
        "clustrange": {"kvals": [2, 3], "stats": {"ASW": [0.2, 0.3]}},
        "stats": {},
        "boot_matrices": {
            "ASW": np.array(
                [
                    [0.10, 0.25],
                    [0.15, 0.30],
                    [0.05, 0.35],
                ],
                dtype=float,
            )
        },
    }

    tab = print_seqnullcqi(bcq, norm=False, quant=0.9, digits=2)
    assert tab.loc["Null Max-T 0.9 interval", "ASW"] != ""

    fig, ax = plot_seqnullcqi(bcq, stat="ASW", kind="line")
    assert fig is not None and ax is not None


def test_print_seqnullcqi_hc_interval_uses_installed_max_t_tail():
    bcq = {
        "clustrange": {"kvals": [2, 3], "stats": {"HC": [0.55, 0.45]}},
        "stats": {
            "HC": np.array(
                [
                    [0.90, 0.80],
                    [0.70, 0.60],
                    [0.50, 0.40],
                ],
                dtype=float,
            )
        },
    }
    tab = print_seqnullcqi(bcq, norm=False, quant=0.5, digits=1)
    assert tab.loc["Null Max-T 0.5 interval", "HC"] == "[0.6; 0.8]"


def test_seqnullcqi_plot_print_accept_k_by_r_stats_matrix():
    import_optional_or_skip("matplotlib")

    # stats["ASW"] is provided as K x R and should be interpreted as R x K.
    bcq = {
        "clustrange": {"kvals": [2, 3], "stats": {"ASW": [0.2, 0.3]}},
        "stats": {
            "ASW": np.array(
                [
                    [0.10, 0.11, 0.12],
                    [0.20, 0.21, 0.22],
                ],
                dtype=float,
            )
        },
    }

    tab = print_seqnullcqi(bcq, norm=False, quant=0.9, digits=2)
    assert tab.loc["Null Max-T 0.9 interval", "ASW"] != ""

    fig, ax = plot_seqnullcqi(bcq, stat="ASW", kind="line")
    assert fig is not None and ax is not None


@pytest.mark.parametrize("R", [1.5, True, "1"])
def test_seqnullcqi_rejects_lossy_replicate_controls_even_with_replay(R):
    X = _toy_sequences()
    cr = _toy_clustrange_without_stats(X)

    with pytest.raises(ValueError, match="R must be an integer"):
        seqnullcqi(
            X,
            cr,
            R=R,
            replay_null_sequences=X.reshape(1, *X.shape),
            random_state=45,
        )


@pytest.mark.parametrize(
    ("kwargs", "message", "error_type"),
    [
        ({"kmedoid": "false"}, "kmedoid must be a boolean", TypeError),
        ({"model": "unknown"}, "unknown model", ValueError),
        ({"not_a_seqnull_control": 1}, "unsupported seqnull arguments", TypeError),
        (
            {"replay_kmedoid_starts": [{}]},
            "replay_kmedoid_starts requires kmedoid=True",
            ValueError,
        ),
    ],
)
def test_seqnullcqi_replay_does_not_bypass_scientific_control_validation(
    kwargs, message, error_type
):
    X = _toy_sequences()
    cr = _toy_clustrange_without_stats(X)

    with pytest.raises(error_type, match=message):
        seqnullcqi(
            X,
            cr,
            R=1,
            replay_null_sequences=X.reshape(1, *X.shape),
            random_state=46,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("seqdist_args", "message", "error_type"),
    [
        (
            {"method": "HAM", "metric": "LCS"},
            "conflicting aliases.*method.*metric",
            ValueError,
        ),
        ({"normalize": "false"}, "normalize must be a boolean", TypeError),
        (
            {"normalize": True, "norm": "none"},
            "conflicting aliases.*norm.*normalize",
            ValueError,
        ),
        (
            {"sm": np.ones((2, 2)), "subm": np.ones((2, 2))},
            "conflicting aliases.*sm.*subm",
            ValueError,
        ),
    ],
)
def test_seqnullcqi_rejects_ambiguous_distance_controls(seqdist_args, message, error_type):
    X = _toy_sequences()
    cr = _toy_clustrange_without_stats(X)

    with pytest.raises(error_type, match=message):
        seqnullcqi(
            X,
            cr,
            R=1,
            seqdist_args=seqdist_args,
            replay_null_sequences=X.reshape(1, *X.shape),
            random_state=47,
        )


def test_seqnullcqi_rejects_fractional_kvals_in_mutated_clustrange():
    X = _toy_sequences()
    cr = _toy_clustrange_without_stats(X)
    cr["kvals"] = [2.5]

    with pytest.raises(ValueError, match="All kvals must be integers"):
        seqnullcqi(
            X,
            cr,
            R=1,
            replay_null_sequences=X.reshape(1, *X.shape),
            random_state=48,
        )

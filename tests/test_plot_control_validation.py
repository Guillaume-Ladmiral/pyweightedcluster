from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weightedcluster.plot import (
    clustrange_plot_data,
    clustrangefamily_plot_data,
    plot_bootclustrange,
    plot_clustrange,
    plot_clustrangefamily,
    plot_seqclararange_stability,
    plot_seqnullcqi,
    print_bootclustrange,
    print_seqnullcqi,
)


def _range_object() -> dict:
    return {"kvals": [2, 3], "stats": {"ASW": [0.2, 0.3]}}


def _family_object() -> dict:
    stats = {"ASW": [0.2, 0.3]}
    return {
        "single": {"stats": stats},
        "pam-single": {"stats": stats},
        "param": {
            "all.methods": ["single", "pam-single"],
            "method": ["single"],
            "pam.combine": True,
            "kvals": [2, 3],
        },
    }


def _seqnull_object() -> dict:
    return {
        "clustrange": {"kvals": [2], "stats": {"ASW": [0.2]}},
        "stats": {"ASW": np.array([[0.1], [0.15]], dtype=float)},
        "R": 2,
    }


@pytest.mark.parametrize("digits", [1.5, True, "2"])
def test_print_adapters_reject_lossy_digits(digits):
    boot = {
        "kvals": [2],
        "boot": None,
        "stats": pd.DataFrame({"ASW": [0.123]}),
    }

    with pytest.raises(ValueError, match="digits must be an integer"):
        print_bootclustrange(boot, digits=digits)
    with pytest.raises(ValueError, match="digits must be an integer"):
        print_seqnullcqi(_seqnull_object(), digits=digits)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rows": 1.5}, "rows must be an integer"),
        ({"cols": True}, "cols must be an integer"),
        ({"pam_combine": "no"}, "pam_combine must be a boolean"),
        ({"withlegend": 1}, "withlegend must be a boolean"),
        ({"group": "typo"}, "group must be 'stat' or 'method'"),
        ({"legend_prop": 1.2}, "legend_prop must be between zero and one"),
    ],
)
def test_family_plot_data_rejects_invalid_controls(kwargs, message):
    error_type = TypeError if "boolean" in message else ValueError
    with pytest.raises(error_type, match=message):
        clustrangefamily_plot_data(_family_object(), stat="ASW", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message", "error_type"),
    [
        ({"conf_int": "0.9"}, "conf_int must be a finite number", TypeError),
        ({"ci_alpha": True}, "ci_alpha must be a finite number", TypeError),
        ({"conf_int": np.nan}, "conf_int must be a finite number", ValueError),
    ],
)
def test_clustrange_plot_data_rejects_non_numeric_probabilities(kwargs, message, error_type):
    with pytest.raises(error_type, match=message):
        clustrange_plot_data(_range_object(), stat="ASW", **kwargs)


def test_plot_clustrange_rejects_truthy_legend_control():
    with pytest.raises(TypeError, match="withlegend must be a boolean"):
        plot_clustrange(_range_object(), stat="ASW", withlegend="false")


def test_plot_adapters_reject_nonfinite_or_non_numeric_line_widths():
    with pytest.raises(ValueError, match="lwd must be a finite number"):
        plot_clustrange(_range_object(), stat="ASW", lwd=np.inf)
    with pytest.raises(TypeError, match="lwd must be a finite number"):
        plot_clustrangefamily(_family_object(), stat="ASW", lwd="1")


@pytest.mark.parametrize(
    ("kwargs", "message", "error_type"),
    [
        ({"withlegend": "false"}, "withlegend must be a boolean", TypeError),
        ({"conf_int": "0.9"}, "conf_int must be a finite number", TypeError),
        ({"lwd": np.inf}, "lwd must be a finite number", ValueError),
    ],
)
def test_empty_boot_plot_does_not_bypass_declared_control_validation(kwargs, message, error_type):
    with pytest.raises(error_type, match=message):
        plot_bootclustrange({"kvals": [], "boot": {}}, stat="ASW", **kwargs)


def test_legacy_seqclara_plot_does_not_ignore_invalid_normalization():
    with pytest.raises(ValueError, match="norm must be one of"):
        plot_seqclararange_stability(
            {"kvals": [], "stats": {}, "quality_stats": {}},
            norm="bogus",
        )


@pytest.mark.parametrize(
    ("call", "message", "error_type"),
    [
        (
            lambda: plot_seqnullcqi(_seqnull_object(), norm="false"),
            "norm must be a boolean",
            TypeError,
        ),
        (
            lambda: plot_seqnullcqi(_seqnull_object(), quant=np.nan),
            "quant must be a finite number",
            ValueError,
        ),
        (
            lambda: plot_seqnullcqi(_seqnull_object(), alpha=2.0),
            "alpha must be between zero and one",
            ValueError,
        ),
        (
            lambda: print_seqnullcqi(_seqnull_object(), norm=1),
            "norm must be a boolean",
            TypeError,
        ),
    ],
)
def test_seqnull_presentation_rejects_invalid_controls(call, message, error_type):
    with pytest.raises(error_type, match=message):
        call()

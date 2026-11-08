from __future__ import annotations

import importlib

import numpy as np
import pytest

from weightedcluster import seqclararange, seqclararange_family

_seqclararange_mod = importlib.import_module("weightedcluster.seqclararange")


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


def test_seqclararange_rejects_unknown_criteria():
    X = _toy_sequences()
    with pytest.raises(ValueError, match="Unknown criteria"):
        seqclararange(
            X,
            R=2,
            sample_size=4,
            kvals=[2],
            seqdist_args={"method": "HAM"},
            criteria="not_a_real_criterion",
            random_state=7,
        )


def test_seqclararange_family_rejects_unknown_criteria():
    X = _toy_sequences()
    with pytest.raises(ValueError, match="Unknown criteria"):
        seqclararange_family(
            X,
            R=2,
            sample_size=4,
            kvals=[2],
            seqdist_args={"method": "HAM"},
            criteria=["distance", "bad_criterion"],
            random_state=7,
        )


def test_objective_direction_matches_package_criterion_semantics_case_insensitively():
    minimized = ("distance", "db", "xb", "hc")
    maximized = ("pbm", "ams", "pbc", "hg", "hgsd", "asw", "asww", "ch", "r2", "chsq", "r2sq")

    for criterion in minimized:
        lower = _seqclararange_mod._objective_value(criterion.upper(), {criterion: 0.2})
        higher = _seqclararange_mod._objective_value(criterion.upper(), {criterion: 0.8})
        assert lower < higher, criterion

    for criterion in maximized:
        lower = _seqclararange_mod._objective_value(criterion.upper(), {criterion: 0.2})
        higher = _seqclararange_mod._objective_value(criterion.upper(), {criterion: 0.8})
        assert higher < lower, criterion


def test_hc_selection_chooses_the_run_with_lower_value():
    runs = [
        {"criteria": {"hc": 0.8}},
        {"criteria": {"hc": 0.2}},
    ]

    assert _seqclararange_mod._select_best_run_index(runs, "HC") == 1


def test_objective_value_maps_non_finite_criteria_to_infinity():
    criteria = (
        "distance",
        "db",
        "xb",
        "hc",
        "pbm",
        "ams",
        "pbc",
        "hg",
        "hgsd",
        "asw",
        "asww",
        "ch",
        "r2",
        "chsq",
        "r2sq",
    )

    for criterion in criteria:
        for value in (np.nan, np.inf, -np.inf):
            assert _seqclararange_mod._objective_value(
                criterion.upper(), {criterion: value}
            ) == float("inf")


def test_seqclararange_accepts_r_quality_criteria():
    X = _toy_sequences()
    for crit in ("asw", "asww", "pbc", "hg", "hgsd", "ch", "r2", "chsq", "r2sq", "hc"):
        out = seqclararange(
            X,
            R=2,
            sample_size=4,
            kvals=[2],
            seqdist_args={"method": "HAM"},
            criteria=crit,
            random_state=9,
        )
        assert out["kvals"] == [2]
        assert "cluster2" in out["clustering"]


def test_seqclararange_family_accepts_r_quality_criteria():
    X = _toy_sequences()
    fam = seqclararange_family(
        X,
        R=2,
        sample_size=4,
        kvals=[2],
        seqdist_args={"method": "HAM"},
        criteria=["asw", "ch"],
        random_state=10,
    )
    assert "asw" in fam
    assert "ch" in fam


def test_compute_clara_criteria_uses_medoid_distances():
    D_full = np.array(
        [
            [0.0, 0.2, 1.0, 1.1],
            [0.2, 0.0, 0.9, 1.0],
            [1.0, 0.9, 0.0, 0.2],
            [1.1, 1.0, 0.2, 0.0],
        ],
        dtype=float,
    )
    medoids = np.array([0, 2], dtype=int)
    D_to_m = D_full[:, medoids]
    labels = np.array([0, 0, 1, 1], dtype=int)
    crits = _seqclararange_mod._compute_clara_criteria(
        method_l="crisp",
        D_full=D_full,
        D_to_m=D_to_m,
        labels_full=labels,
        medoids_global_idx=medoids,
        memb_full_all=None,
        fuzzifier_m=1.5,
        dnoise_value=None,
    )
    # mean of nearest-medoid distances: [0, 0.2, 0, 0.2]
    assert crits["distance"] == pytest.approx(0.1, rel=1e-9, abs=1e-12)
    # min medoid separation is D(0,2)=1.0 -> XB = 0.1 / 1.0
    assert crits["xb"] == pytest.approx(0.1, rel=1e-9, abs=1e-12)
    # PBM = ((1/k) * max_sep / mean_diss)^2 = ((1/2) * 1 / 0.1)^2 = 25
    assert crits["pbm"] == pytest.approx(25.0, rel=1e-9, abs=1e-12)

from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import clustassoc


@pytest.fixture(scope="module")
def clustassoc_r_session():
    from .r_parity_utils import create_r_parity_session, fail_or_skip

    session, reason = create_r_parity_session()
    if session is None:
        fail_or_skip(reason or "R parity session unavailable")
    return session


_LABELS2 = np.repeat([1, 2], 6)
_LABELS3 = np.repeat([1, 2, 3], 4)
_CLUSTRANGE = {
    "kvals": [2, 3],
    "clustering": {"cluster2": _LABELS2, "cluster3": _LABELS3},
    "stats": {
        "PBC": [0.0, 0.0],
        "HG": [0.0, 0.0],
        "HGSD": [0.0, 0.0],
    },
}
_NUMERIC = np.array([0, 0.2, 0.1, 0.4, 1.1, 1.0, 1.3, 1.4, 2.2, 2.0, 2.3, 2.4])
_CATEGORICAL = np.array(["a", "a", "b", "a", "b", "b", "a", "b", "a", "b", "b", "b"])
_WEIGHTS = np.array([0.5, 1, 1.5, 2, 1, 2, 1, 1.5, 0.75, 1.25, 2.25, 0.8])


def _distance_fixture() -> np.ndarray:
    position = np.array([0, 0.1, 0.4, 0.7, 1.2, 1.5, 1.9, 2.4, 3.0, 3.3, 3.9, 4.4])
    group = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    return np.abs(position[:, None] - position[None, :]) + 0.15 * (group[:, None] != group[None, :])


def _core_table(result) -> np.ndarray:
    return np.column_stack(
        [result["Unaccounted"], result["Remaining"], result["BIC"], result["numcluster"]]
    )


@pytest.mark.parametrize("categorical", [False, True], ids=["numeric", "categorical"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_clustassoc_core_and_model_diagnostics_match_r(clustassoc_r_session, categorical, weighted):
    D = _distance_fixture()
    covar = _CATEGORICAL if categorical else _NUMERIC
    weights = _WEIGHTS if weighted else None

    expected = clustassoc_r_session.r_clustassoc_values(
        D,
        _CLUSTRANGE,
        covar,
        categorical=categorical,
        weights=weights,
    )
    actual = clustassoc(_CLUSTRANGE, D, covar, weights=weights)
    repeated = clustassoc(_CLUSTRANGE, D, covar, weights=weights)

    assert "clustassoc" in expected["class"]
    assert expected["colnames"] == ["Unaccounted", "Remaining", "BIC", "numcluster"]
    assert expected["rownames"] == ["No Clustering", "cluster2", "cluster3"]
    np.testing.assert_array_equal(expected["numcluster"], [1, 2, 3])
    assert actual.additions["row_names"] == expected["rownames"]
    assert actual["numcluster"] == expected["numcluster"].tolist()
    np.testing.assert_allclose(_core_table(actual), expected["core"], rtol=0, atol=1e-7)

    actual_diagnostics = np.column_stack(
        [
            actual["BIC"],
            actual.additions["BICLogLik"],
            actual.additions["BICDf"],
            actual.additions["BICNobs"],
            np.logical_not(actual.additions["BICConverged"]).astype(float),
        ]
    )
    np.testing.assert_allclose(
        actual_diagnostics,
        expected["model_diagnostics"],
        rtol=0,
        atol=1e-7,
    )
    for key in actual:
        np.testing.assert_array_equal(actual[key], repeated[key])


def test_clustassoc_categorical_separation_tracks_r_nnet_convergence(clustassoc_r_session):
    D = np.ones((6, 6), dtype=float)
    np.fill_diagonal(D, 0)
    D[:3, :3] = 0.1
    D[3:, 3:] = 0.1
    np.fill_diagonal(D, 0)
    clustrange = {
        "kvals": [2, 3],
        "clustering": {
            "cluster2": np.array([1, 1, 1, 2, 2, 2]),
            "cluster3": np.array([1, 2, 3, 1, 2, 3]),
        },
        "stats": {
            "PBC": [0.0, 0.0],
            "HG": [0.0, 0.0],
            "HGSD": [0.0, 0.0],
        },
    }
    covar = np.array(["a", "a", "a", "b", "b", "b"])

    expected = clustassoc_r_session.r_clustassoc_values(
        D,
        clustrange,
        covar,
        categorical=True,
    )
    actual = clustassoc(clustrange, D, covar)

    # cluster2 is collinear with covar: its covariate term adds no rank to the design.
    # R's dissmfacw then gives a finite value of an arbitrary QR direction, which R's
    # clustassoc reports; traminex gives NaN for such a term (3a3a5149, a DIVERGENCE of its
    # ledger), and so does this port, a deliberate departure from WeightedCluster
    # (docs/R_MIGRATION_GUIDE.md; owner's request of 2026-11-07, recorded option (a)).
    observed = _core_table(actual)[:, :2]
    assert np.all(np.isfinite(expected["core"][1, :2]))
    assert np.all(np.isnan(observed[1]))
    np.testing.assert_allclose(
        observed[[0, 2]],
        expected["core"][[0, 2], :2],
        rtol=0,
        atol=1e-10,
    )
    # nnet stops a separated fit once its objective crosses abstol=1e-4;
    # the faithful local optimizer may cross at a nearby BFGS step.
    np.testing.assert_allclose(actual["BIC"], expected["core"][:, 2], rtol=0, atol=2e-4)
    assert actual.additions["BICConverged"] == [True, True, True]


def _random_bic_case(rng: np.random.Generator, weighted: bool):
    n = int(rng.integers(24, 56))
    k = int(rng.integers(2, 5))
    c = int(rng.integers(2, 4))
    labels = rng.integers(1, k + 1, size=n)
    shift = rng.uniform(0.0, 1.5)
    scores = np.array(
        [[shift * ((label + j) % c) * rng.uniform(0.5, 1.5) for j in range(c)] for label in labels]
    )
    probabilities = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    covar = np.array([f"c{rng.choice(c, p=p)}" for p in probabilities])
    weights = rng.uniform(0.2, 3.0, size=n).round(3) if weighted else np.ones(n)
    return labels, covar, weights


def test_categorical_bic_is_the_converged_multinom_value_on_a_random_domain(clustassoc_r_session):
    """2026-09-21 targeted audit, L-5, re-measured 2026-11-07 on 60 random cases.

    Python computes the factor-only multinomial likelihood exactly; R's ``multinom``
    stops at its default ``reltol`` (1e-8), above that minimum, by up to 3.2e-6
    without an empty cell and 2.4e-3 with one. With ``reltol=1e-15`` R reaches
    Python's value within 4.5e-13 and 2.7e-6. So R's BIC is never below Python's,
    and the bounds below are the measured spreads with a margin.
    """
    import rpy2.robjects as ro

    from weightedcluster.clustassoc import _bic_multinomial

    r = clustassoc_r_session.r
    rng = np.random.default_rng(20261107)
    for case in range(24):
        labels, covar, weights = _random_bic_case(rng, weighted=bool(case % 2))
        fit = _bic_multinomial(covar, labels, weights)
        r.assign("bic_covar", ro.StrVector(list(covar)))
        r.assign("bic_labels", ro.IntVector([int(v) for v in labels]))
        r.assign("bic_weights", ro.FloatVector([float(v) for v in weights]))
        r(
            "bic_frame <- data.frame(covar = factor(bic_covar), clustf = factor(bic_labels), "
            "w = bic_weights)"
        )
        default, tight = (
            float(
                r(
                    "BIC(suppressWarnings(nnet::multinom(covar ~ clustf, data = bic_frame, "
                    f"weights = w, trace = FALSE{options})))"
                )[0]
            )
            for options in ("", ", reltol = 1e-15, maxit = 10000")
        )
        empty_cell = bool(
            np.any(
                np.array(
                    [
                        [np.sum((labels == g) & (covar == v)) for v in np.unique(covar)]
                        for g in np.unique(labels)
                    ]
                )
                == 0
            )
        )
        assert default - fit.bic >= -1e-9, case
        assert tight - fit.bic >= -1e-9, case
        assert abs(tight - fit.bic) <= (1e-5 if empty_cell else 1e-9), case
        assert abs(default - fit.bic) <= (5e-3 if empty_cell else 1e-5), case

from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import bootclustrange, seqnull, seqnullcqi
from weightedcluster._seqdist import sequences_distance
from weightedcluster.schemas import ClustrangeResult
from weightedcluster.seqnull import (
    _encode_states,
)

pytest_plugins = ["tests.r_parity_utils"]


# A fixed (N=12, T=4, K=3) dataset. Chosen so that every state appears as a
# transition *source* (no all-zero seqtrate row) and the three blocks give clean
# cluster columns -- so neither the seqnull transition-rate anchor nor the
# bootclustrange subsample loop can hit a degenerate case.
_X = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 1],
        [2, 2, 1, 1],
        [2, 1, 1, 0],
    ],
    dtype=int,
)
_CLUSTER2 = np.array([1] * 8 + [2] * 4, dtype=int)
_CLUSTER3 = np.array([1] * 4 + [2] * 4 + [3] * 4, dtype=int)


def _clustrange_dict() -> dict:
    return ClustrangeResult(
        {
            "kvals": [2, 3],
            "clustering": {"cluster2": _CLUSTER2, "cluster3": _CLUSTER3},
        }
    )


def test_r_oracle_exposes_bootstrap_null_functions(r_parity_session):
    # The functions this phase pins must be reachable in the installed oracle.
    assert r_parity_session.r_internal_exists("seqnull")
    assert r_parity_session.r_internal_exists("bootclustrange")
    assert r_parity_session.r_internal_exists("seqnullcqi")


_SEQNULL_MODELS = ["combined", "duration", "sequencing", "stateindep", "markov", "userpos"]
_R_SEQNULL_MODELS = {
    "combined": "combined",
    "duration": "duration",
    "sequencing": "sequencing",
    "stateindep": "stateindep",
    "markov": "Markov",  # R capitalises Markov
    "userpos": "userpos",
}


def test_seqnull_markov_model_matches_r_seqtrate(r_parity_session):
    # The load-bearing teeth: seqnull(model="markov") draws from the GLOBAL
    # transition-rate matrix, which R builds with seqtrate(). Pin the production
    # source to R's, exactly. (Deterministic; no RNG involved.)
    #
    # This used to compare seqnull._global_transition, a local reimplementation
    # that the production path had already stopped using. Finding F-16 of the
    # 2026-08-24 audit deleted it as a superseded duplicate of a statistic
    # ECOSYSTEM.md assigns to traminex, so the comparison now runs against the
    # traminex call seqnull actually makes.
    from traminex import seqdef, seqtrate

    _Xc, states = _encode_states(_X)
    k = int(states.size)
    py_tr = np.asarray(
        seqtrate(seqdef(_X), time_varying=False, weighted=False, as_dataframe=False),
        dtype=float,
    )
    r_tr = r_parity_session.r_seqtrate_global(_X)
    assert py_tr.shape == (k, k)
    assert np.allclose(py_tr, r_tr, atol=1e-12, rtol=0.0)


def test_seqnull_stateindep_start_dist_matches_r_seqstatd(r_parity_session):
    # seqnull(model="stateindep") draws position 0 from the t0 marginal, which R
    # builds with seqstatd()$Frequencies[,1]. Pin the production source exactly;
    # as above, the local duplicate this used to compare is gone (F-16).
    from traminex import seqdef, seqstatd

    py_f0 = np.asarray(seqstatd(seqdef(_X), weighted=False)["Frequencies"], dtype=float)[0]
    r_f0 = r_parity_session.r_seqstatd_first_freq(_X)
    assert np.allclose(py_f0, r_f0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("model", _SEQNULL_MODELS)
def test_seqnull_structural_contract_matches_r(r_parity_session, model):
    # Structural parity across ALL six model branches: Python and R both return an
    # (N, T) null object whose states are a subset of the original alphabet.
    py = seqnull(_X, model=model, random_state=20260615)
    assert py.shape == _X.shape
    assert set(np.unique(py)).issubset(set(np.unique(_X)))

    shape, alpha = r_parity_session.r_seqnull_structural(
        _X, model=_R_SEQNULL_MODELS[model], seed=20260615
    )
    assert shape == _X.shape
    orig_alpha = {str(s) for s in np.unique(_X)}
    assert set(alpha).issubset(orig_alpha)


@pytest.mark.parametrize("model", _SEQNULL_MODELS)
def test_seqnull_is_seed_reproducible(model):
    # Python determinism (no R): same random_state -> identical output.
    a = seqnull(_X, model=model, random_state=99)
    b = seqnull(_X, model=model, random_state=99)
    assert np.array_equal(a, b)


R_CLUSTRANGE_STAT_COLUMNS = (
    "PBC",
    "HG",
    "HGSD",
    "ASW",
    "ASWw",
    "CH",
    "R2",
    "CHsq",
    "R2sq",
    "HC",
)


def test_bootclustrange_exposes_only_installed_result_fields(r_parity_session):
    out = bootclustrange(
        _clustrange_dict(),
        _X,
        R=4,
        sample_size=10,
        seqdist_args={"method": "LCS"},
        sampling="random",
        random_state=7,
    )
    assert list(out) == ["clustering", "kvals", "meant", "stderr", "boot", "stats"]
    assert out.r_class == ["bootclustrange", "clustrange", "list"]
    assert "baseline_stats" not in list(out)
    assert tuple(out.additions["baseline_stats"]) == R_CLUSTRANGE_STAT_COLUMNS
    assert out["stats"] is out["meant"]


def test_bootclustrange_structural_contract_matches_r(r_parity_session):
    # Structural parity for the stochastic bootstrap summary.
    out = bootclustrange(
        _clustrange_dict(),
        _X,
        R=4,
        sample_size=10,
        seqdist_args={"method": "LCS"},
        sampling="random",
        random_state=7,
    )
    contract = r_parity_session.r_bootclustrange_contract(
        _X,
        {"cluster2": _CLUSTER2, "cluster3": _CLUSTER3},
        kvals=[2, 3],
        r_iter=4,
        sample_size=10,
        method="LCS",
        seed=7,
    )
    assert tuple(contract["meant_cols"]) == R_CLUSTRANGE_STAT_COLUMNS
    assert tuple(contract["stderr_cols"]) == R_CLUSTRANGE_STAT_COLUMNS
    assert list(contract["kvals"]) == [2, 3]
    assert list(out["kvals"]) == [2, 3]
    # R: boot is a list of length numclust; each boot[[i]] is R x 10.
    assert contract["n_boot_lists"] == 2
    assert contract["boot1_dim"] == [4, 10]
    # Python: R's 10 stats are a subset of the per-stat meant/stderr keys, each of
    # length == #kvals; boot[k][stat] has length R.
    for stat in R_CLUSTRANGE_STAT_COLUMNS:
        assert stat in out["meant"] and stat in out["stderr"]
        assert len(out["meant"][stat]) == 2
        assert len(out["stderr"][stat]) == 2
        for k in out["kvals"]:
            assert len(out["boot"][int(k)][stat]) == 4


def test_bootclustrange_is_seed_reproducible():
    kw = dict(
        R=5,
        sample_size=10,
        seqdist_args={"method": "LCS"},
        sampling="random",
        random_state=13,
    )
    a = bootclustrange(_clustrange_dict(), _X, **kw)
    b = bootclustrange(_clustrange_dict(), _X, **kw)
    assert list(a["kvals"]) == list(b["kvals"]) == [2, 3]
    for stat in a["meant"]:
        assert np.allclose(
            np.asarray(a["meant"][stat], dtype=float),
            np.asarray(b["meant"][stat], dtype=float),
            equal_nan=True,
        )


def test_seqnullcqi_structural_contract_matches_r(r_parity_session):
    cr = _clustrange_dict()
    out = seqnullcqi(
        _X,
        cr,
        R=3,
        model="combined",
        seqdist_args={"method": "LCS"},
        random_state=11,
    )
    contract = r_parity_session.r_seqnullcqi_contract(
        _X,
        {"cluster2": _CLUSTER2, "cluster3": _CLUSTER3},
        r_iter=3,
        model="combined",
        method="LCS",
        seed=11,
    )
    # R metadata defaults.
    assert contract["R"] == 3 and out["R"] == 3
    assert contract["kmedoid"] is False and out["kmedoid"] is False
    assert contract["hclust_method"] == "ward.D" and out["hclust.method"] == "ward.D"
    # R: stats is a named list of R x K matrices over the 10 stat columns.
    assert tuple(contract["stat_names"]) == R_CLUSTRANGE_STAT_COLUMNS
    assert contract["stat_shape"] == [3, 2]  # [R, K]
    assert contract["seqdata_shape"] == [3 * _X.shape[0], _X.shape[1]]
    # Python: R's 10 stats are a subset of the per-stat null matrices, each R x K.
    for stat in R_CLUSTRANGE_STAT_COLUMNS:
        assert stat in out["stats"]
        assert np.asarray(out["stats"][stat], dtype=float).shape == (3, 2)
    assert np.asarray(out["seqdata"]).shape == (3 * _X.shape[0], _X.shape[1])


def test_seqnullcqi_observed_clustrange_matches_r_as_clustrange(r_parity_session):
    # Deterministic anchor (load-bearing, exact): the observed/baseline clustrange
    # stats seqnullcqi reports must equal R as.clustrange on the same observed
    # distance matrix (the input typology's quality, independent of the bootstrap).
    D_obs = sequences_distance(_X, metric="LCS", normalize=True)
    out = seqnullcqi(
        _X,
        _clustrange_dict(),
        R=2,
        model="combined",
        seqdist_args={"method": "LCS"},
        random_state=11,
    )
    r = r_parity_session.as_clustrange_default(
        D_obs,
        {"cluster2": _CLUSTER2, "cluster3": _CLUSTER3},
    )
    r_idx = {name: j for j, name in enumerate(r["stats_names"])}
    obs = out["clustrange"]["stats"]
    for ki in range(len(out["clustrange"]["kvals"])):
        for stat in R_CLUSTRANGE_STAT_COLUMNS:
            py_val = float(np.asarray(obs[stat], dtype=float)[ki])
            r_val = float(r["stats_matrix"][ki, r_idx[stat]])
            assert np.isclose(py_val, r_val, atol=1e-8, rtol=1e-6), stat


def test_seqnullcqi_is_seed_reproducible():
    kw = dict(R=3, model="combined", seqdist_args={"method": "LCS"}, random_state=21)
    a = seqnullcqi(_X, _clustrange_dict(), **kw)
    b = seqnullcqi(_X, _clustrange_dict(), **kw)
    assert np.array_equal(np.asarray(a["seqdata"]), np.asarray(b["seqdata"]))
    for stat in a["stats"]:
        assert np.allclose(
            np.asarray(a["stats"][stat], dtype=float),
            np.asarray(b["stats"][stat], dtype=float),
            equal_nan=True,
        )

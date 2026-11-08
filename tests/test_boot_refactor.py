from __future__ import annotations

import warnings

import numpy as np
import pytest
from traminex import STSList

from weightedcluster._boot_controls import (
    _BootstrapControls,
    _build_bootstrap_worker_context,
    _prepare_bootstrap_controls,
    _strata_allocation,
)
from weightedcluster._boot_inputs import _prepare_bootstrap_inputs
from weightedcluster._boot_result import _BOOT_R_STATS, _build_bootclustrange_result
from weightedcluster.schemas import SeqClaraRangeResult


def _sequence_matrix() -> np.ndarray:
    return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=int)


def _plain_clustering() -> dict[str, np.ndarray]:
    return {
        "two": np.array([0, 0, 1, 1], dtype=int),
        "three": np.array([0, 1, 2, 2], dtype=int),
    }


def _control_inputs(*, R: int = 2, sample_size: int = 6):
    return _prepare_bootstrap_inputs(
        {
            "two": np.array([0, 0, 0, 1, 1, 1]),
            "three": np.array([0, 0, 1, 1, 2, 2]),
        },
        np.array([[0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1]], dtype=int),
        R=R,
        sample_size=sample_size,
        seqdist_args={"method": "HAM"},
    )


def test_prepare_bootstrap_inputs_copies_array_and_sts_inputs_with_attached_weights():
    sequence = _sequence_matrix()
    array_inputs = _prepare_bootstrap_inputs(
        _plain_clustering(), sequence, R=2, sample_size=3, seqdist_args=None
    )

    sequence[0, 0] = 9
    assert array_inputs.sequence_matrix.tolist() == [[0, 0], [0, 1], [1, 1], [1, 0]]
    assert array_inputs.attached_weights is None

    weights = np.array([1.0, 2.0, 3.0, 4.0])
    sts = STSList(data=_sequence_matrix(), alphabet=["A", "B"], weights=weights)
    sts_inputs = _prepare_bootstrap_inputs(
        _plain_clustering(), sts, R=2, sample_size=3, seqdist_args=None
    )

    sts.data[0, 0] = 9
    weights[0] = 9.0
    assert sts_inputs.sequence_matrix.tolist() == [[0, 0], [0, 1], [1, 1], [1, 0]]
    assert sts_inputs.attached_weights.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_prepare_bootstrap_inputs_accepts_plain_and_nested_clustering_mappings():
    plain = _prepare_bootstrap_inputs(
        _plain_clustering(), _sequence_matrix(), R=2, sample_size=3, seqdist_args=None
    )
    nested = _prepare_bootstrap_inputs(
        {"kvals": [99], "clustering": _plain_clustering()},
        _sequence_matrix(),
        R=2,
        sample_size=3,
        seqdist_args=None,
    )

    assert plain.kvals == [2, 3]
    assert plain.clustering.column_names == ["cluster2", "cluster3"]
    assert plain.clustering["cluster2"].tolist() == [0, 0, 1, 1]
    assert nested.kvals == [2, 3]
    assert nested.clustering.column_names == ["cluster2", "cluster3"]
    assert nested.clustering["cluster3"].tolist() == [0, 1, 2, 2]


def test_prepare_bootstrap_inputs_uses_final_seqclararange_clara_medoids():
    sequence_result = SeqClaraRangeResult(
        {
            "clustering": _plain_clustering(),
            "clara": {2: {"medoids": [1, 3]}, 3: {"medoids": [0, 2, 3]}},
        }
    )

    inputs = _prepare_bootstrap_inputs(
        sequence_result, _sequence_matrix(), R=2, sample_size=3, seqdist_args=None
    )

    assert inputs.kvals == [2, 3]
    assert inputs.medoids.tolist() == [0, 2, 3]


def test_prepare_bootstrap_inputs_validates_in_existing_order_with_exact_messages():
    with pytest.raises(ValueError, match="^seqdata must be a two-dimensional sequence matrix$"):
        _prepare_bootstrap_inputs(
            np.array(1), np.array([0, 1]), R=True, sample_size=True, seqdist_args=[]
        )
    with pytest.raises(
        ValueError, match="^object must be a clustering frame or seqclararange result$"
    ):
        _prepare_bootstrap_inputs(
            np.array(1), _sequence_matrix(), R=True, sample_size=True, seqdist_args=[]
        )
    with pytest.raises(ValueError, match="^R must be an integer$"):
        _prepare_bootstrap_inputs(
            _plain_clustering(), _sequence_matrix(), R=True, sample_size=True, seqdist_args=[]
        )
    with pytest.raises(ValueError, match="^sample_size must be an integer$"):
        _prepare_bootstrap_inputs(
            _plain_clustering(), _sequence_matrix(), R=1, sample_size=True, seqdist_args=[]
        )
    with pytest.raises(TypeError, match="^seqdist_args must be a mapping or None$"):
        _prepare_bootstrap_inputs(
            _plain_clustering(), _sequence_matrix(), R=1, sample_size=1, seqdist_args=[]
        )


def test_prepare_bootstrap_inputs_defaults_and_copies_seqdist_args_into_record_fields():
    provided = {"method": "HAM", "normalize": True}
    defaulted = _prepare_bootstrap_inputs(
        _plain_clustering(), _sequence_matrix(), R=2, sample_size=3, seqdist_args=None
    )
    configured = _prepare_bootstrap_inputs(
        _plain_clustering(), _sequence_matrix(), R=2, sample_size=3, seqdist_args=provided
    )

    provided["method"] = "OM"
    assert defaulted.seqdist_args == {"method": "LCS"}
    assert configured.seqdist_args == {"method": "HAM", "normalize": True}
    assert configured.sequence_matrix.shape == (4, 2)
    assert configured.attached_weights is None
    assert configured.kvals == [2, 3]
    assert configured.clustering.column_names == ["cluster2", "cluster3"]
    assert configured.medoids is None
    assert configured.R == 2
    assert configured.sample_size == 3


def test_prepare_bootstrap_controls_resolves_clustering_and_explicit_strata_with_r_allocation():
    clustering_controls = _prepare_bootstrap_controls(
        _control_inputs(), sampling="clustering", strata=None, replay_samples=None
    )
    explicit_controls = _prepare_bootstrap_controls(
        _control_inputs(sample_size=5),
        sampling="strata",
        strata=["east", "east", "east", "west", "west", "west"],
        replay_samples=None,
    )

    assert clustering_controls.sampling == "strata"
    assert [indices.tolist() for indices in clustering_controls.strata_idx.values()] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert {
        key: values.tolist() for key, values in clustering_controls.expected_unique.items()
    } == {2: [0, 1], 3: [0, 1, 2]}
    assert explicit_controls.sampling == "strata"
    allocation, correction = _strata_allocation(explicit_controls.strata_idx, 5)
    assert allocation.tolist() == [2, 2]
    assert correction == 1


def test_prepare_bootstrap_controls_routes_medoids_and_unknown_sampling_and_checks_population_limits():
    medoid_inputs = _control_inputs(sample_size=4)
    medoid_inputs.medoids = np.array([0, 4], dtype=int)

    assert (
        _prepare_bootstrap_controls(
            medoid_inputs, sampling="medoids", strata=None, replay_samples=None
        ).sampling
        == "medoids"
    )
    assert (
        _prepare_bootstrap_controls(
            _control_inputs(), sampling="source-style", strata=None, replay_samples=None
        ).sampling
        == "source-style"
    )
    with pytest.raises(ValueError, match="cannot take a sample larger than the population"):
        _prepare_bootstrap_controls(
            _control_inputs(sample_size=7), sampling="random", strata=None, replay_samples=None
        )
    with pytest.raises(ValueError, match="cannot take a sample larger than the population"):
        _prepare_bootstrap_controls(
            _control_inputs(sample_size=9), sampling="medoids", strata=None, replay_samples=None
        )


def test_prepare_bootstrap_controls_rejects_cluster_and_medoid_tail_size_boundaries():
    # Catches a regression that allows a sample unable to cover three clusters.
    with pytest.raises(ValueError, match="need at least 3"):
        _prepare_bootstrap_controls(
            _control_inputs(sample_size=2), sampling="random", strata=None, replay_samples=None
        )

    medoid_inputs = _control_inputs(sample_size=1)
    medoid_inputs.medoids = np.array([0, 4], dtype=int)
    # Catches a regression that permits a forced-medoid prefix longer than the sample.
    with pytest.raises(ValueError, match="negative 'size' argument"):
        _prepare_bootstrap_controls(
            medoid_inputs, sampling="medoids", strata=None, replay_samples=None
        )

    # Catches a regression that routes a one-per-cluster clustering sample to workers.
    with pytest.raises(ValueError, match="minimum value of 6"):
        _prepare_bootstrap_controls(
            _control_inputs(sample_size=4),
            sampling="clustering",
            strata=None,
            replay_samples=None,
        )


@pytest.mark.parametrize(
    ("replay", "message"),
    [
        (np.array([[0, 1, 2]], dtype=int), "shape"),
        (np.array([[0.0, 1.5, 2.0, 3.0, 4.0, 5.0]]), "integer"),
        (np.array([[0, 1, 2, 3, 4, 6]], dtype=int), "out-of-bounds"),
        (np.array([[0, 0, 1, 2, 4, 5]], dtype=int), "duplicate"),
        (np.array([[0, 1, 2, 3, 4, 4]], dtype=int), "duplicate"),
    ],
)
def test_prepare_bootstrap_controls_rejects_invalid_replay_shape_values_and_duplicates(
    replay, message
):
    with pytest.raises(ValueError, match=message):
        _prepare_bootstrap_controls(
            _control_inputs(R=1), sampling="random", strata=None, replay_samples=replay
        )


def test_prepare_bootstrap_controls_validates_strata_and_medoid_replays_and_defensively_copies():
    with pytest.raises(ValueError, match="stratified rounding allocation"):
        _prepare_bootstrap_controls(
            _control_inputs(R=1, sample_size=5),
            sampling="strata",
            strata=["east", "east", "east", "east", "west", "west"],
            replay_samples=np.array([[0, 1, 2, 3, 4]], dtype=int),
        )

    valid_strata = _prepare_bootstrap_controls(
        _control_inputs(R=1, sample_size=5),
        sampling="strata",
        strata=["east", "east", "east", "west", "west", "west"],
        replay_samples=np.array([[0, 1, 2, 3, 4]], dtype=int),
    )
    # Catches a regression that rejects either legal direction of a +1 correction.
    assert valid_strata.replay_samples.tolist() == [[0, 1, 2, 3, 4]]

    medoid_inputs = _control_inputs(R=1, sample_size=4)
    medoid_inputs.medoids = np.array([0, 4], dtype=int)
    replay = np.array([[0, 4, 0, 2]], dtype=int)
    controls = _prepare_bootstrap_controls(
        medoid_inputs, sampling="medoids", strata=None, replay_samples=replay
    )
    replay[0, 0] = 5
    assert controls.replay_samples.tolist() == [[0, 4, 0, 2]]
    with pytest.raises(ValueError, match="begin with the medoids"):
        _prepare_bootstrap_controls(
            medoid_inputs,
            sampling="medoids",
            strata=None,
            replay_samples=np.array([[4, 0, 1, 2]], dtype=int),
        )

    with pytest.raises(ValueError, match="preserve every cluster count"):
        _prepare_bootstrap_controls(
            _control_inputs(R=1, sample_size=4),
            sampling="random",
            strata=None,
            replay_samples=np.array([[0, 1, 2, 3]], dtype=int),
        )


def test_build_bootstrap_worker_context_preserves_contract_shapes_without_mutating_inputs():
    inputs = _control_inputs(R=1)
    replay = np.array([[0, 1, 2, 3, 4, 5]], dtype=int)
    controls = _prepare_bootstrap_controls(
        inputs, sampling="random", strata=None, replay_samples=replay
    )
    context = _build_bootstrap_worker_context(inputs, controls, seed=91)

    assert list(context) == [
        "seed",
        "X",
        "N",
        "sample_size",
        "sampling",
        "strata_idx",
        "medoids_idx",
        "seqdist_args",
        "kvals",
        "clustering",
        "expected_unique",
        "replay_samples",
    ]
    assert context["seed"] == 91
    assert context["N"] == 6
    assert context["kvals"] == [2, 3]
    assert context["replay_samples"].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert context["X"].shape == (6, 2)
    assert context["clustering"]["cluster3"].shape == (6,)
    assert inputs.seqdist_args == {"method": "HAM"}
    assert inputs.kvals == [2, 3]
    assert inputs.sequence_matrix.tolist() == [[0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1]]


def test_build_bootclustrange_result_preserves_literal_result_contract_and_copies_arrays():
    inputs = _prepare_bootstrap_inputs(
        _plain_clustering(),
        _sequence_matrix(),
        R=2,
        sample_size=3,
        seqdist_args={"method": "HAM"},
    )
    controls = _BootstrapControls(
        sampling="random",
        strata_idx=None,
        expected_unique={2: np.array([0, 1]), 3: np.array([0, 1, 2])},
        replay_samples=None,
    )
    baseline_stats = {"source": "baseline"}
    results = [
        {
            "stats": {
                "PBC": [1.0, 11.0],
                "HG": [2.0, 12.0],
                "HGSD": [3.0, 13.0],
                "ASW": [4.0, 14.0],
                "ASWw": [5.0, 15.0],
                "CH": [6.0, 16.0],
                "R2": [7.0, 17.0],
                "CHsq": [8.0, 18.0],
                "R2sq": [9.0, 19.0],
                "HC": [10.0, 20.0],
            },
            "sample": np.array([0, 1, 3]),
            "distance": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "clustering": np.array([[0, 0], [0, 1], [1, 2]]),
        },
        {
            "stats": {
                "PBC": [3.0, 15.0],
                "HG": [4.0, 16.0],
                "HGSD": [5.0, 17.0],
                "ASW": [6.0, 18.0],
                "ASWw": [7.0, 19.0],
                "CH": [8.0, 20.0],
                "R2": [9.0, 21.0],
                "CHsq": [10.0, 22.0],
                "R2sq": [11.0, 23.0],
                "HC": [12.0, 24.0],
            },
            "sample": np.array([0, 2, 3]),
            "distance": np.array([[0.0, 2.0], [2.0, 0.0]]),
            "clustering": np.array([[0, 0], [1, 1], [1, 2]]),
        },
    ]

    result = _build_bootclustrange_result(
        inputs,
        controls,
        baseline_stats=baseline_stats,
        results=results,
    )

    assert _BOOT_R_STATS == (
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
    assert list(result) == ["clustering", "kvals", "meant", "stderr", "boot", "stats"]
    assert result.r_class == ["bootclustrange", "clustrange", "list"]
    assert result["stats"] is result["meant"]
    assert result["clustering"].column_names == ["cluster2", "cluster3"]
    assert result["meant"].row_names == ["cluster2", "cluster3"]
    assert result["meant"].column_names == [
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
    ]
    assert result["stderr"].row_names == ["cluster2", "cluster3"]
    assert result["stderr"].column_names == [
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
    ]
    assert result["boot"][2].row_names == ["1", "2"]
    assert result["boot"][3].row_names == ["1", "2"]
    assert result["boot"][2].column_names == [
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
    ]
    assert result["boot"][3].column_names == [
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
    ]
    assert result["boot"][2] == {
        "PBC": [1.0, 3.0],
        "HG": [2.0, 4.0],
        "HGSD": [3.0, 5.0],
        "ASW": [4.0, 6.0],
        "ASWw": [5.0, 7.0],
        "CH": [6.0, 8.0],
        "R2": [7.0, 9.0],
        "CHsq": [8.0, 10.0],
        "R2sq": [9.0, 11.0],
        "HC": [10.0, 12.0],
    }
    assert result["boot"][3] == {
        "PBC": [11.0, 15.0],
        "HG": [12.0, 16.0],
        "HGSD": [13.0, 17.0],
        "ASW": [14.0, 18.0],
        "ASWw": [15.0, 19.0],
        "CH": [16.0, 20.0],
        "R2": [17.0, 21.0],
        "CHsq": [18.0, 22.0],
        "R2sq": [19.0, 23.0],
        "HC": [20.0, 24.0],
    }
    assert result["meant"] == {
        "PBC": [2.0, 13.0],
        "HG": [3.0, 14.0],
        "HGSD": [4.0, 15.0],
        "ASW": [5.0, 16.0],
        "ASWw": [6.0, 17.0],
        "CH": [7.0, 18.0],
        "R2": [8.0, 19.0],
        "CHsq": [9.0, 20.0],
        "R2sq": [10.0, 21.0],
        "HC": [11.0, 22.0],
    }
    assert result["stderr"] == pytest.approx(
        {
            "PBC": [1.4142135623730951, 2.8284271247461903],
            "HG": [1.4142135623730951, 2.8284271247461903],
            "HGSD": [1.4142135623730951, 2.8284271247461903],
            "ASW": [1.4142135623730951, 2.8284271247461903],
            "ASWw": [1.4142135623730951, 2.8284271247461903],
            "CH": [1.4142135623730951, 2.8284271247461903],
            "R2": [1.4142135623730951, 2.8284271247461903],
            "CHsq": [1.4142135623730951, 2.8284271247461903],
            "R2sq": [1.4142135623730951, 2.8284271247461903],
            "HC": [1.4142135623730951, 2.8284271247461903],
        }
    )
    assert list(result.additions) == [
        "sampling",
        "R",
        "seqdist.args",
        "baseline_stats",
        "samples",
        "distances",
        "subset_clusterings",
        "weights",
    ]
    assert result.additions["sampling"] == "random"
    assert result.additions["R"] == 2
    assert result.additions["seqdist.args"] == {"method": "HAM"}
    assert result.additions["baseline_stats"] is baseline_stats
    assert [sample.tolist() for sample in result.additions["samples"]] == [
        [0, 1, 3],
        [0, 2, 3],
    ]
    assert [distance.tolist() for distance in result.additions["distances"]] == [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 2.0], [2.0, 0.0]],
    ]
    assert [clustering.tolist() for clustering in result.additions["subset_clusterings"]] == [
        [[0, 0], [0, 1], [1, 2]],
        [[0, 0], [1, 1], [1, 2]],
    ]
    assert result.additions["weights"] is None

    inputs.clustering["cluster2"][0] = 9
    results[0]["sample"][0] = 9
    results[0]["distance"][0, 0] = 9.0
    results[0]["clustering"][0, 0] = 9
    assert result["clustering"]["cluster2"].tolist() == [0, 0, 1, 1]
    assert result.additions["samples"][0].tolist() == [0, 1, 3]
    assert result.additions["distances"][0].tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert result.additions["subset_clusterings"][0].tolist() == [
        [0, 0],
        [0, 1],
        [1, 2],
    ]


def test_build_bootclustrange_result_uses_nan_sample_standard_deviation_for_one_replicate():
    inputs = _prepare_bootstrap_inputs(
        {"two": np.array([0, 0, 1, 1])},
        _sequence_matrix(),
        R=1,
        sample_size=3,
        seqdist_args=None,
    )
    controls = _BootstrapControls(
        sampling="random",
        strata_idx=None,
        expected_unique={2: np.array([0, 1])},
        replay_samples=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _build_bootclustrange_result(
            inputs,
            controls,
            baseline_stats={},
            results=[
                {
                    "stats": {
                        "PBC": [1.0],
                        "HG": [2.0],
                        "HGSD": [3.0],
                        "ASW": [4.0],
                        "ASWw": [5.0],
                        "CH": [6.0],
                        "R2": [7.0],
                        "CHsq": [8.0],
                        "R2sq": [9.0],
                        "HC": [10.0],
                    },
                    "sample": np.array([0, 1, 3]),
                    "distance": np.array([[0.0, 1.0], [1.0, 0.0]]),
                    "clustering": np.array([[0], [0], [1]]),
                }
            ],
        )

    assert result["boot"][2].row_names == ["1"]
    assert result["stderr"].column_names == [
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
    ]
    assert np.isnan(result["stderr"]["PBC"][0])
    assert np.isnan(result["stderr"]["HG"][0])
    assert np.isnan(result["stderr"]["HGSD"][0])
    assert np.isnan(result["stderr"]["ASW"][0])
    assert np.isnan(result["stderr"]["ASWw"][0])
    assert np.isnan(result["stderr"]["CH"][0])
    assert np.isnan(result["stderr"]["R2"][0])
    assert np.isnan(result["stderr"]["CHsq"][0])
    assert np.isnan(result["stderr"]["R2sq"][0])
    assert np.isnan(result["stderr"]["HC"][0])

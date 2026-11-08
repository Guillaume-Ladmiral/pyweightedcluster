from __future__ import annotations

import numpy as np
import pytest
from traminex import STSList, seqdef

from weightedcluster import seqnull


def _toy_sequences() -> np.ndarray:
    return np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )


def test_seqnull_combined_rejects_ignored_begin_argument():
    X = _toy_sequences()
    with pytest.raises(ValueError, match="begin is ignored"):
        seqnull(X, model="combined", begin="ofreq", random_state=1)


def test_seqnull_stateindep_rejects_false_time_varying():
    X = _toy_sequences()
    with pytest.raises(ValueError, match="time_varying must be True or omitted"):
        seqnull(X, model="stateindep", time_varying=False, random_state=2)


def test_seqnull_markov_rejects_true_time_varying():
    X = _toy_sequences()
    with pytest.raises(ValueError, match="time_varying must be False or omitted"):
        seqnull(X, model="markov", time_varying=True, random_state=3)


def test_seqnull_userpos_rejects_spell_only_arguments():
    X = _toy_sequences()
    with pytest.raises(ValueError, match="imp_trans is only supported by spell-based models"):
        seqnull(X, model="userpos", imp_trans={0: [1]}, random_state=4)


@pytest.mark.parametrize(
    "model",
    ["combined", "duration", "sequencing", "stateindep", "markov", "userpos"],
)
def test_seqnull_one_column_input_has_stable_error(model):
    with pytest.raises(ValueError, match="at least two"):
        seqnull(np.array([[0], [1]], dtype=int), model=model, random_state=5)


def test_seqnull_zero_rows_and_zero_columns_return_immutable_empty_copies():
    for source in (
        np.empty((0, 3), dtype=int),
        np.empty((3, 0), dtype=int),
    ):
        actual = seqnull(source, model="combined", random_state=6)
        assert actual.shape == source.shape
        assert actual is not source

    sts = STSList(
        data=np.empty((0, 3), dtype=np.int32),
        alphabet=["A", "B"],
        labels=["Alpha", "Beta"],
    )
    actual_sts = seqnull(sts, model="combined", random_state=7)
    assert actual_sts is not sts
    assert actual_sts.data.shape == (0, 3)
    assert actual_sts.alphabet == sts.alphabet


def test_seqnull_rejects_numpy_and_sts_missing_states():
    with pytest.raises(ValueError, match="missing data"):
        seqnull(np.array([[0.0, np.nan], [1.0, 0.0]]), random_state=8)
    missing_sts = seqdef([[1, None, 2], [1, 2, 2]])
    with pytest.raises(ValueError, match="missing data"):
        seqnull(missing_sts, random_state=9)


@pytest.mark.parametrize("model", ["combined", "duration", "sequencing"])
def test_seqnull_spell_models_accept_variable_lengths_without_mutating_input(model):
    source = seqdef([[1, 1, 2, None], [2, 3, None, None]])
    before = source.copy()
    actual = seqnull(source, model=model, random_state=10)
    assert actual.data.shape == source.data.shape
    assert np.all(actual.data < 3)
    np.testing.assert_array_equal(source.data, before.data)
    assert source.alphabet == before.alphabet


@pytest.mark.parametrize("model", ["stateindep", "markov", "userpos"])
def test_seqnull_positional_models_reject_variable_lengths(model):
    source = seqdef([[1, 1, 2, None], [2, 3, None, None]])
    with pytest.raises(ValueError, match="equal full sequence lengths"):
        seqnull(source, model=model, random_state=11)


def test_seqnull_replay_plan_validation_is_strict_and_non_mutating():
    X = _toy_sequences()
    original = [{"kind": "sample.int", "values": [1, 2, 1]}]
    with pytest.raises(ValueError, match="exhausted"):
        seqnull(X, model="stateindep", replay_plan=original, random_state=12)
    assert original == [{"kind": "sample.int", "values": [1, 2, 1]}]

    unused = [
        {"kind": "sample.int", "values": [1, 2, 1]},
        {"kind": "sample.int", "values": [1, 1]},
        {"kind": "sample.int", "values": [1]},
        {"kind": "sample.int", "values": [1]},
    ]
    with pytest.raises(ValueError, match="unused draw"):
        seqnull(
            np.array([[0, 0], [1, 1], [0, 1]], dtype=int),
            model="stateindep",
            replay_plan=unused,
            random_state=13,
        )


@pytest.mark.parametrize("plan", ["bad", [1], 3])
def test_seqnull_replay_plan_requires_a_sequence_of_mappings(plan):
    with pytest.raises(TypeError, match="replay_plan must be a sequence of mappings"):
        seqnull(_toy_sequences(), model="stateindep", replay_plan=plan, random_state=13)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"begin": np.array([1.0, 0.0])}, "begin must have length 3"),
        ({"begin": "unknown"}, "unknown begin"),
        ({"trate": "unknown"}, "unknown trate"),
        ({"trate": np.ones((2, 2))}, "shape"),
        ({"trate": np.ones((3, 3, 1))}, "shape"),
        ({"trate": np.ones(3)}, "2D or 3D"),
    ],
)
def test_seqnull_userpos_rejects_malformed_probability_inputs(kwargs, message):
    X = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=int)
    with pytest.raises(ValueError, match=message):
        seqnull(X, model="userpos", random_state=14, **kwargs)


def test_seqnull_unknown_model_and_non_2d_inputs_are_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        seqnull(_toy_sequences(), model="unknown", random_state=15)
    with pytest.raises(ValueError, match="2D"):
        seqnull(np.array([0, 1, 0], dtype=int), random_state=16)


@pytest.mark.parametrize(
    ("kwargs", "message", "error_type"),
    [
        ({"weighted": 1}, "weighted must be a boolean", TypeError),
        ({"time_varying": "false"}, "time_varying must be a boolean or None", TypeError),
        (
            {"imp_trans_limit": "0"},
            "imp_trans_limit must be a finite number",
            TypeError,
        ),
        (
            {"imp_trans_limit": np.nan},
            "imp_trans_limit must be a finite number",
            ValueError,
        ),
    ],
)
def test_seqnull_rejects_silent_boolean_and_numeric_control_coercions(kwargs, message, error_type):
    with pytest.raises(error_type, match=message):
        seqnull(_toy_sequences(), model="userpos", random_state=17, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "unknown"}, "unknown model"),
        ({"weighted": "yes"}, "weighted must be a boolean"),
        ({"random_state": 1.5}, "random_state must be an integer"),
    ],
)
def test_seqnull_empty_inputs_do_not_bypass_control_validation(kwargs, message):
    error_type = TypeError if "boolean" in message else ValueError
    with pytest.raises(error_type, match=message):
        seqnull(np.empty((0, 3), dtype=int), **kwargs)


@pytest.mark.parametrize(
    "model",
    ["combined", "duration", "sequencing", "stateindep", "markov", "userpos"],
)
def test_seqnull_public_random_path_never_mutates_numpy_input(model):
    source = np.array(
        [[0, 0, 1, 1], [0, 1, 1, 2], [1, 1, 2, 2], [2, 2, 0, 0]],
        dtype=int,
    )
    before = source.copy()
    seqnull(source, model=model, random_state=17)
    np.testing.assert_array_equal(source, before)

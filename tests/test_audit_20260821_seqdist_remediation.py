from __future__ import annotations

import numpy as np
import pytest

from weightedcluster import (
    bootclustrange,
    seqclararange,
    seqclararange_family,
    seqnullcqi,
)
from weightedcluster._seqdist import sequences_distance, sequences_distance_between
from weightedcluster.schemas import ClustrangeResult

SEQUENCES = np.array(
    [
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 0],
    ],
    dtype=int,
)


def _observed_nonfinite_costs(*, time_varying: bool) -> np.ndarray:
    fixed = np.ones((3, 3), dtype=float)
    np.fill_diagonal(fixed, 0.0)
    if not time_varying:
        fixed[0, 1] = fixed[1, 0] = np.inf
        return fixed
    costs = np.repeat(fixed[None, :, :], SEQUENCES.shape[1], axis=0)
    costs[1, 0, 1] = costs[1, 1, 0] = np.inf
    return costs


def _unreachable_nonfinite_costs(*, time_varying: bool) -> np.ndarray:
    fixed = np.ones((3, 3), dtype=float)
    np.fill_diagonal(fixed, 0.0)
    fixed[2, :2] = np.inf
    fixed[:2, 2] = np.inf
    if not time_varying:
        return fixed
    return np.repeat(fixed[None, :, :], SEQUENCES.shape[1], axis=0)


@pytest.mark.parametrize("metric", ["HAM", "DHD"])
@pytest.mark.parametrize("time_varying", [False, True], ids=["fixed", "position"])
def test_observed_nonfinite_costs_fail_before_low_level_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    time_varying: bool,
) -> None:
    from weightedcluster import _seqdist

    calls: list[str] = []

    def _unexpected_within(*args, **kwargs):
        calls.append("within")
        raise AssertionError("provider dispatch must follow observed-cost validation")

    def _unexpected_between(*args, **kwargs):
        calls.append("between")
        raise AssertionError("provider dispatch must follow observed-cost validation")

    monkeypatch.setattr(_seqdist, "_tx_pairwise", _unexpected_within)
    monkeypatch.setattr(_seqdist, "_tx_pairwise_between", _unexpected_between)
    costs = _observed_nonfinite_costs(time_varying=time_varying)

    with pytest.raises(
        ValueError,
        match=rf"{metric} substitution costs for observed states must be finite",
    ):
        sequences_distance(
            SEQUENCES,
            metric=metric,
            normalize=False,
            substitution_costs=costs,
        )
    with pytest.raises(
        ValueError,
        match=rf"{metric} substitution costs for observed states must be finite",
    ):
        sequences_distance_between(
            SEQUENCES[:3],
            SEQUENCES[3:],
            metric=metric,
            normalize=False,
            substitution_costs=costs,
        )

    assert calls == []


@pytest.mark.parametrize("metric", ["HAM", "DHD"])
@pytest.mark.parametrize("time_varying", [False, True], ids=["fixed", "position"])
def test_unreachable_nonfinite_costs_remain_accepted(
    metric: str,
    time_varying: bool,
) -> None:
    costs = _unreachable_nonfinite_costs(time_varying=time_varying)

    within = sequences_distance(
        SEQUENCES,
        metric=metric,
        normalize=False,
        substitution_costs=costs,
    )
    between = sequences_distance_between(
        SEQUENCES[:3],
        SEQUENCES[3:],
        metric=metric,
        normalize=False,
        substitution_costs=costs,
    )

    assert np.isfinite(within).all()
    assert np.isfinite(between).all()


def test_observed_nonfinite_costs_fail_at_every_public_workflow_boundary() -> None:
    costs = _observed_nonfinite_costs(time_varying=False)
    seqdist_args = {"method": "HAM", "sm": costs}
    clustering = {"cluster2": np.array([1, 1, 1, 2, 2, 2], dtype=int)}
    clustrange = ClustrangeResult({"kvals": [2], "clustering": clustering})
    calls = [
        lambda: seqclararange(
            SEQUENCES,
            R=1,
            sample_size=6,
            kvals=[2],
            seqdist_args=seqdist_args,
            random_state=8,
        ),
        lambda: seqclararange_family(
            SEQUENCES,
            R=1,
            sample_size=6,
            kvals=[2],
            criteria=["distance", "ams"],
            seqdist_args=seqdist_args,
            random_state=8,
        ),
        lambda: bootclustrange(
            clustering,
            SEQUENCES,
            R=1,
            sample_size=6,
            seqdist_args=seqdist_args,
            random_state=8,
        ),
        lambda: seqnullcqi(
            SEQUENCES,
            clustrange,
            R=1,
            seqdist_args=seqdist_args,
            random_state=8,
        ),
    ]

    for call in calls:
        with pytest.raises(
            ValueError,
            match="HAM substitution costs for observed states must be finite",
        ) as error:
            call()
        assert "diss must contain" not in str(error.value)


@pytest.mark.parametrize("metric", ["HAM", "DHD"])
@pytest.mark.parametrize("time_varying", [False, True], ids=["fixed", "position"])
def test_unreachable_nonfinite_costs_give_the_distances_of_any_finite_value(
    metric: str,
    time_varying: bool,
) -> None:
    # traminex's explicit-cost DHD refuses any non-finite cost, as R's DHD does, so the
    # unreachable entries are handed to it as 0; no distance reads them, so no value moves.
    costs = _unreachable_nonfinite_costs(time_varying=time_varying)
    finite = np.where(np.isfinite(costs), costs, 7.0)
    for normalize in (False, True):
        np.testing.assert_array_equal(
            sequences_distance(
                SEQUENCES, metric=metric, normalize=normalize, substitution_costs=costs
            ),
            sequences_distance(
                SEQUENCES, metric=metric, normalize=normalize, substitution_costs=finite
            ),
        )
        np.testing.assert_array_equal(
            sequences_distance_between(
                SEQUENCES[:3],
                SEQUENCES[3:],
                metric=metric,
                normalize=normalize,
                substitution_costs=costs,
            ),
            sequences_distance_between(
                SEQUENCES[:3],
                SEQUENCES[3:],
                metric=metric,
                normalize=normalize,
                substitution_costs=finite,
            ),
        )

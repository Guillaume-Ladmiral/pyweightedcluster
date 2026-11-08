"""Public Bayesian RARCAT orchestration and stable result identity."""

from __future__ import annotations

import importlib
import operator
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._parallel import validate_parallel_controls

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._bayesian_rarcat_conditional import (
        ComponentFitRecord,
        ConditionalPosteriorResult,
        PreparedConditionalDesign,
        PriorSpecification,
    )
    from ._bayesian_rarcat_mixture import (
        AllocationRecord,
        IntegratedTargetSummary,
        WeightedTargetSupport,
    )
    from ._bayesian_rarcat_partition import PartitionPosteriorResult

_RESPONSE_NAME = re.compile(r"[A-Za-z_]\w*\Z")
_RUNTIME_DEPENDENCIES = ("pymc", "arviz", "xarray", "h5netcdf", "h5py", "patsy")
_INSTALL_INSTRUCTION = "Install pyweightedcluster[bayesian]"


@dataclass(frozen=True)
class BayesRARCATResult:
    """Complete result of the modular Bayesian RARCAT estimator.

    The object retains the predictor-independent partition posterior, every
    conditional component, exact mixture support, diagnostics, failures,
    seeds, and private schema provenance. Array fields are read-only.
    """

    status: str
    failures: tuple[ComponentFitRecord, ...]
    warnings: tuple[object, ...]
    partition_posterior: PartitionPosteriorResult
    conditional_posterior: ConditionalPosteriorResult
    n: int
    B: int
    K: int
    U: int
    formula: str
    observation_ids: np.ndarray
    reference_labels: np.ndarray
    cluster_order: tuple[object, ...]
    masses: np.ndarray
    design: PreparedConditionalDesign
    prior: PriorSpecification
    software_versions: tuple[tuple[str, str], ...]
    b2_root_seed: int
    b3_root_seed: int
    supports: tuple[WeightedTargetSupport, ...]
    coefficient_summaries: tuple[IntegratedTargetSummary, ...]
    effect_summaries: tuple[IntegratedTargetSummary, ...]
    allocation: AllocationRecord | None
    schema_name: str
    schema_version: int
    execution_mode: str
    resolved_workers: int


def _require_bayesian_dependencies() -> None:
    for dependency in _RUNTIME_DEPENDENCIES:
        try:
            importlib.import_module(dependency)
        except ImportError as error:
            missing = getattr(error, "name", None) or dependency
            raise ImportError(
                f"Bayesian RARCAT requires the optional dependency {missing!r}. "
                f"{_INSTALL_INSTRUCTION}"
            ) from error


def _same_identity_value(left: object, right: object) -> bool:
    """Exact identity of two labels: same type, and ``==`` yields a true bool.

    The single owner of this rule. ``_bayesian_rarcat_mixture`` used to carry a
    second copy, ``type(left) is type(right) and bool(left == right)``, which
    raised on ``pd.NA`` and on arrays and accepted a one-element array as
    identical -- the permissive direction, for an identity gate (2026-09-21
    targeted audit, L-4). A comparison that raises, or that returns anything but
    a bool, is not an identity.
    """
    if type(left) is not type(right):
        return False
    try:
        equal = left == right
    except Exception:
        return False
    return isinstance(equal, (bool, np.bool_)) and bool(equal)


def _same_identity_sequence(left: Sequence[object], right: Sequence[object]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        _same_identity_value(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _validate_checkable_envelope_caps(
    *,
    dissimilarity,
    reference_labels,
    B,
    conditional_module,
) -> None:
    """Reject envelope-cap violations knowable before the B2 bootstrap.

    2026-09-02 audit, finding WC-090211: out-of-envelope ``n``, ``K``, and
    ``B`` previously ran the complete B2 bootstrap before B3's
    ``validate_private_envelope`` rejected them, so the wasted work scaled
    with the invalid input. Only the caps are enforced here, with B3's exact
    message format; minima, malformed inputs, ``U``, and ``design_width``
    keep their existing owners and error text. Inputs this best-effort check
    cannot interpret fall through to B2's own validation unchanged.
    """
    matrix = np.asarray(dissimilarity)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        return
    n = int(matrix.shape[0])
    K = int(len(reference_labels.cat.categories))
    try:
        b_int = int(operator.index(B))
    except TypeError:
        try:
            numeric = float(B)
        except (TypeError, ValueError):
            return
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            return
        b_int = int(numeric)
    envelope = conditional_module._BAYES_RARCAT_ENVELOPE
    for name, actual in (("n", n), ("K", K), ("B", b_int)):
        maximum = envelope[name][1]
        if actual > maximum:
            raise ValueError(f"{name}={actual} exceeds the private maximum {maximum}")
    if K > n:
        raise ValueError("K cannot exceed n")


def _reference_labels(*, formula: str, data, conditional_module):
    response_name, _, _ = conditional_module._validated_formula(formula)
    if _RESPONSE_NAME.fullmatch(response_name) is None:
        raise ValueError("formula response must be one untransformed data column")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if list(data.columns).count(response_name) != 1:
        raise ValueError("formula response must identify exactly one existing data column")
    response = data[response_name]
    if not isinstance(response.dtype, pd.CategoricalDtype) or not response.cat.ordered:
        raise ValueError("formula response must be an ordered categorical")
    if response.isna().any():
        raise ValueError("formula response must be complete")
    categories = tuple(response.cat.categories.tolist())
    observed = tuple(pd.unique(response.astype(object)).tolist())
    if not _same_identity_sequence(categories, observed):
        raise ValueError("formula response categories must equal stable first-observed label order")
    return response


def _require_distinct_root_seeds(
    partition_root_seed: object, conditional_root_seed: object
) -> None:
    """Refuse equal root seeds for the two replay domains (2026-09-21 targeted audit, L-3).

    B2 takes its bootstrap-weight stream and B3 its prior-predictive stream from
    ``SeedSequence(root).spawn(2)[0]``, so one integer passed as both roots gives
    both the same bits. The derivation itself is left as it is, because recorded
    replays depend on it (docs/maintainers/SCIENTIFIC_GOVERNANCE.md); refusing equal
    roots keeps every replay with distinct roots unchanged. Values that are not
    integers are left to each domain's own seed check, which runs later.
    """
    try:
        partition, conditional = operator.index(partition_root_seed), operator.index(
            conditional_root_seed
        )
    except TypeError:
        return
    if isinstance(partition_root_seed, bool) or isinstance(conditional_root_seed, bool):
        return
    if partition == conditional:
        raise ValueError(
            "partition_root_seed and conditional_root_seed must differ: they seed separate "
            "replay domains whose first streams coincide for equal roots"
        )


def bayes_rarcat(
    dissimilarity,
    *,
    formula: str,
    data,
    B: int,
    npass: int,
    partition_root_seed: int,
    conditional_root_seed: int,
    continuous_predictors,
    counterfactual_steps,
    user_prior=None,
    parallel: bool = False,
    n_jobs: int | None = None,
) -> BayesRARCATResult:
    """Fit the supported modular Bayesian RARCAT estimator.

    The partition and conditional root seeds are separate replay domains, so they
    must differ: each domain draws its first stream from
    ``SeedSequence(root).spawn(2)[0]``, and equal roots would give the bootstrap
    weights and the prior-predictive draws the same bits. ``parallel`` controls only
    deterministic mixture post-processing.

    The enforced public envelope is ``2 <= n <= 12``, ``2 <= K <= 3`` with
    ``K <= n``, ``1 <= B <= 128``, ``1 <= U <= 16``, ``U / B <= 0.25``, and
    ``1 <= design_width <= 4``. The dimensions knowable up front — ``n``,
    ``K``, and ``B`` — are rejected before the B2 partition bootstrap runs;
    ``U`` and ``design_width`` are only determined during B3 design
    preparation and are rejected there, before any sampling.

    The envelope is enforced, not statistically calibrated: the frozen prior,
    sampler profile and diagnostic thresholds were calibrated at ``n >= 36``,
    and no valid calibration evidence exists inside ``n <= 12``. A preserved
    outcome-selected small-n study cannot support unconditional recovery,
    coverage, direction, or magnitude conclusions; the estimator algebra was
    checked separately. See
    ``BAYESIAN_RARCAT_SMALL_N_EVIDENCE_CORRECTION_2026_08_27.md`` and the
    "Scientific interpretation and limits" section of
    ``docs/vignettes/bayesian_rarcat.qmd``.

    ``parallel`` must be a boolean. ``n_jobs`` is ``None`` or a positive
    non-boolean integer. Both controls are rejected before partition or
    conditional-posterior fitting when invalid.
    """
    validate_parallel_controls(parallel, n_jobs)
    _require_bayesian_dependencies()

    from . import _bayesian_rarcat_conditional as conditional_module
    from . import _bayesian_rarcat_mixture as mixture_module
    from . import _bayesian_rarcat_partition as partition_module

    reference_labels = _reference_labels(
        formula=formula,
        data=data,
        conditional_module=conditional_module,
    )
    _validate_checkable_envelope_caps(
        dissimilarity=dissimilarity,
        reference_labels=reference_labels,
        B=B,
        conditional_module=conditional_module,
    )
    _require_distinct_root_seeds(partition_root_seed, conditional_root_seed)
    observation_ids = data.index
    partition = partition_module.build_partition_posterior(
        dissimilarity=dissimilarity,
        reference_labels=reference_labels,
        observation_ids=observation_ids,
        reference_observation_ids=observation_ids,
        B=B,
        npass=npass,
        root_seed=partition_root_seed,
    )
    conditional = conditional_module.fit_conditional_posteriors(
        partition_posterior=partition,
        formula=formula,
        data=data,
        continuous_predictors=continuous_predictors,
        counterfactual_steps=counterfactual_steps,
        root_seed=conditional_root_seed,
        user_prior=user_prior,
    )
    return mixture_module.integrate_conditional_posteriors(
        partition,
        conditional,
        parallel=parallel,
        n_jobs=n_jobs,
    )


__all__ = ["BayesRARCATResult", "bayes_rarcat"]

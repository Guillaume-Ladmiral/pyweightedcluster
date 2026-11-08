from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal, localcontext
from numbers import Real
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from traminex import STSList, seqdef

from ._constants import FUZZY_CONVERGENCE_TOL, FUZZY_MAX_ITER
from ._fallback import (
    _component_less,
    _pam_requires_exact_objectives,
    _weighted_medoids_objective_component,
)
from ._utils import generate_random_seed
from ._validation import validate_distance_matrix, validate_integer_control, validate_kvals


def crispness(membership: np.ndarray, *, norm: bool = True) -> np.ndarray:
    """Crispness of fuzzy memberships per observation.

    Returns sum_j u_ij^2, optionally normalized to [0,1] as in R. The
    normalized one-cluster result is undefined and contains ``nan`` values,
    matching R's exact zero-over-zero formula.
    """
    U = np.asarray(membership, dtype=float)
    if U.ndim != 2 or U.shape[1] == 0:
        raise ValueError("membership must be a 2D matrix with at least one column")
    if not np.all(np.isfinite(U)):
        raise ValueError("membership must contain finite values")
    vals = (U * U).sum(axis=1)
    if not norm:
        return vals
    K = U.shape[1]
    if K == 1:
        return np.full(vals.shape, np.nan, dtype=float)
    return (vals - 1.0 / K) / (1.0 - 1.0 / K)


_FUZZY_MEMBERSHIP_ERROR = (
    " [!] The group argument should be a membership matrix with one row per sequence."
)


def seqplot(*args: Any, **kwargs: Any):
    """Lazily resolve the optional traminex plotting dependency.

    traminex rewrites an optional-dependency ``ImportError`` to name a
    *traminex* extra. Re-raising it unchanged told a ``pyweightedcluster`` user
    without matplotlib to run ``pip install 'traminex[plot]'``, which is
    actionable in the wrong package's namespace — finding F-21 of the
    2026-08-24 whole-codebase audit. Naming this package's own extra also
    insulates the message from future changes to the sibling's text.
    """
    try:
        import matplotlib  # noqa: F401 -- validate this package's optional extra eagerly
        from traminex import seqplot as traminex_seqplot
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for fuzzyseqplot; install this package's "
            "'plot' extra (pip install 'pyweightedcluster[plot]')"
        ) from exc

    return traminex_seqplot(*args, **kwargs)


def _copy_sts_rows(
    seqdata: STSList,
    positions: np.ndarray,
    *,
    weights: np.ndarray | None,
) -> STSList:
    """Take STS rows while preserving sequence metadata and duplicate IDs.

    ``weights`` may be ``None``, and must be when the vector to carry contains a
    non-finite or negative value. ``traminex.sequences.STSList`` requires finite,
    non-negative case weights, and has done since its canonical ``seqdef``
    constructor; traminex commit ``1ebb46cd`` extended the same rule to direct
    construction and to later assignment.  The fuzzy membership power can
    legitimately produce positive infinity -- R's ``fuzzyseqplot`` does the same
    -- so that vector travels beside the sequence object as ``plot_weights``
    rather than inside it.  See ``docs/maintainers/TECHNICAL_DEBT.md`` item 6.
    """
    indexes = np.asarray(positions, dtype=int).reshape(-1)
    ids = None if seqdata.ids is None else [seqdata.ids[index] for index in indexes]
    return STSList(
        data=seqdata.data[indexes].copy(),
        alphabet=list(seqdata.alphabet),
        labels=None if seqdata.labels is None else list(seqdata.labels),
        cpal=None if seqdata.cpal is None else list(seqdata.cpal),
        xtlab=None if seqdata.xtlab is None else list(seqdata.xtlab),
        xtstep=seqdata.xtstep,
        tick_last=seqdata.tick_last,
        nr=seqdata.nr,
        void=seqdata.void,
        nr_code=seqdata.nr_code,
        void_code=seqdata.void_code,
        weights=None if weights is None else np.asarray(weights, dtype=float).copy(),
        start=seqdata.start,
        ids=ids,
        missing_color=seqdata.missing_color,
    )


def _membership_and_names(group: Any) -> tuple[np.ndarray, list[Any]]:
    """Extract a membership matrix and its ordered display names."""
    names: Sequence[Any] | None = None
    membership = group
    if isinstance(group, Mapping):
        if "membership" not in group:
            raise TypeError(_FUZZY_MEMBERSHIP_ERROR)
        membership = group["membership"]
        names = group.get("membership_names", group.get("group_names"))
    elif not isinstance(group, (np.ndarray, pd.DataFrame)) and hasattr(group, "membership"):
        membership = group.membership
        names = getattr(group, "membership_names", getattr(group, "group_names", None))

    if isinstance(membership, pd.DataFrame):
        if names is None:
            names = list(membership.columns)
        matrix = membership.to_numpy(dtype=float, copy=True)
    else:
        try:
            matrix = np.asarray(membership, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError(_FUZZY_MEMBERSHIP_ERROR) from error
        matrix = matrix.copy()
    if matrix.ndim != 2:
        raise TypeError(_FUZZY_MEMBERSHIP_ERROR)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("group membership must have at least one row and one column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("group membership values must be finite")
    if names is None:
        group_names = list(range(1, matrix.shape[1] + 1))
    else:
        group_names = list(names)
        if len(group_names) != matrix.shape[1]:
            raise ValueError("membership names must match the number of membership columns")
    return matrix, group_names


def _prepare_fuzzyseqplot_data(
    seqdata: Any,
    group: Any,
    *,
    membership_threashold: float = 0,
    members_weighted: bool = True,
    memb_exp: float = 1,
    sortv: Any = None,
) -> dict[str, Any]:
    """Prepare R-faithful plot-driving data without rendering pixels.

    Memberships are powered, flattened in column-major order, and filtered
    inclusively. A mapping with a ``membership`` entry (or an object exposing a
    ``membership`` attribute) is the Python fanny-equivalent protocol.
    """
    sequence = seqdata.copy() if isinstance(seqdata, STSList) else seqdef(seqdata)
    matrix, group_names = _membership_and_names(group)
    if matrix.shape[0] != sequence.nseq:
        raise ValueError(_FUZZY_MEMBERSHIP_ERROR)
    if not isinstance(members_weighted, (bool, np.bool_)):
        raise TypeError("members_weighted must be a boolean")
    if not isinstance(membership_threashold, Real) or isinstance(
        membership_threashold, (bool, np.bool_)
    ):
        raise TypeError("membership_threashold must be a finite number")
    threshold = float(membership_threashold)
    if not np.isfinite(threshold):
        raise ValueError("membership_threashold must be a finite number")
    if not isinstance(memb_exp, Real) or isinstance(memb_exp, (bool, np.bool_)):
        raise TypeError("memb_exp must be a finite number")
    exponent = float(memb_exp)
    if not np.isfinite(exponent):
        raise ValueError("memb_exp must be a finite number")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        powered = np.power(matrix, exponent)
    memberships = powered.ravel(order="F")
    source_indices = np.tile(np.arange(sequence.nseq, dtype=int), matrix.shape[1])
    groups = np.repeat(np.asarray(group_names, dtype=object), sequence.nseq)
    retained = memberships >= threshold
    attached = (
        np.ones(sequence.nseq, dtype=float)
        if sequence.weights is None
        else np.asarray(sequence.weights, dtype=float).copy()
    )
    repeated_weights = np.tile(attached, matrix.shape[1])
    fuzzy_weights = memberships * repeated_weights if members_weighted else memberships.copy()
    complete_plot_weights = fuzzy_weights.copy()
    retained_indices = source_indices[retained]
    retained_memberships = memberships[retained].copy()
    retained_plot_weights = fuzzy_weights[retained].copy()
    # The exact R-derived vector travels beside the sequence object, never
    # inside it: it may hold positive infinity, which traminex's STSList
    # refuses.  See _copy_sts_rows and TECHNICAL_DEBT item 6.
    expanded = _copy_sts_rows(
        sequence,
        retained_indices,
        weights=None,
    )
    source_ids = (
        retained_indices.astype(object)
        if sequence.ids is None
        else np.asarray([sequence.ids[index] for index in retained_indices], dtype=object)
    )
    resolved_sortv = (
        retained_memberships.copy() if isinstance(sortv, str) and sortv == "membership" else sortv
    )
    return {
        "seqdata": expanded,
        "plot_weights": retained_plot_weights,
        "complete_plot_weights": complete_plot_weights,
        "group": groups[retained].copy(),
        "group_levels": tuple(group_names),
        "membership": retained_memberships,
        "source_indices": retained_indices.copy(),
        "source_ids": source_ids,
        "sortv": resolved_sortv,
    }


def _categorical_values(group: Any, n_obs: int) -> tuple[np.ndarray | None, list[Any]]:
    if group is None:
        return None, ["Unique group"]
    if isinstance(group, pd.Series) and isinstance(group.dtype, pd.CategoricalDtype):
        categorical = group.array
    elif isinstance(group, pd.Categorical):
        categorical = group
    else:
        raise TypeError(_FUZZY_MEMBERSHIP_ERROR)
    if len(categorical) != n_obs:
        raise ValueError("group must contain one value per sequence")
    values = np.asarray(categorical.astype(object), dtype=object)
    levels = [level for level in categorical.categories if bool(np.any(values == level))]
    return values, levels


_INDEX_PLOT_TYPES = frozenset({"i", "I"})


def _validate_fuzzy_plot_weights(weights: np.ndarray) -> None:
    """Fail closed before a requested fuzzy weighting can reach selection."""
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(
            "fuzzyseqplot weighted rendering requires plot weights to be finite and non-negative"
        )


def _weighted_index_row_edges(weights: np.ndarray) -> np.ndarray:
    """Return R-direction row boundaries for weighted index cells.

    Representable weights retain their raw scale.  Only when adding ``-0.5``
    would erase positive, very small cells do we divide every weight by one
    common scale. If the axis still cannot represent every positive cell, the
    geometry is refused rather than silently erasing a row.
    """
    values = np.asarray(weights, dtype=float).reshape(-1)
    _validate_fuzzy_plot_weights(values)
    try:
        total = math.fsum(values.tolist())
    except OverflowError as error:
        raise ValueError("cumulative plot weight is not representable") from error
    if not math.isfinite(total):
        raise ValueError("cumulative plot weight is not representable")
    edges = -0.5 + np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    heights = np.diff(edges)
    if np.all(heights[values > 0.0] > 0.0):
        return edges
    scale = float(np.max(values, initial=0.0))
    if scale == 0.0:
        return edges
    normalized = -0.5 + np.concatenate(([0.0], np.cumsum(values / scale, dtype=float)))
    if not np.all(np.diff(normalized)[values > 0.0] > 0.0):
        raise ValueError("positive plot weight is not representable in axis geometry")
    return normalized


def _resolve_sequence_state_code(
    seqdata: STSList,
    code_attr: str,
    symbol_attr: str,
) -> int | None:
    """Resolve an explicit missing/void code or its alphabet symbol."""
    code = getattr(seqdata, code_attr, None)
    if code is not None:
        return int(code)
    symbol = getattr(seqdata, symbol_attr, None)
    if symbol in seqdata.alphabet:
        return int(seqdata.alphabet.index(symbol))
    return None


def _render_weighted_index_panel(
    seqdata: STSList,
    *,
    ax,
    cmap: Any = None,
    highlight: Sequence[int] | None = None,
    legend: bool = False,
    sortv: Any = None,
    sampling: int | None = None,
    sample_meth: str = "systematic",
    weighted: bool = True,
    random_state: int | np.random.Generator | None = None,
    missing_alpha: float = 0.3,
    void_alpha: float = 0.2,
    apply_hatching: bool = False,
):
    """Draw the fuzzy-only variable-height index cells.

    Traminex remains responsible for row selection, state palettes, time-axis
    ticks, labels, and legends. This adapter owns only the row geometry that
    WeightedCluster's fuzzy membership weights require.
    """
    try:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle
        from traminex.visualization import state_legend
        from traminex.visualization.palette import ensure_cmap
        from traminex.visualization.series import select_index_rows
        from traminex.visualization.ticks import apply_time_axis
    except ImportError as error:
        raise ImportError("matplotlib required for fuzzyseqplot") from error

    selection = select_index_rows(
        seqdata,
        sortv=sortv,
        sampling=sampling,
        sample_meth=sample_meth,
        weighted=weighted,
        random_state=random_state,
    )
    data = selection.display_rows
    row_edges = _weighted_index_row_edges(selection.weights)
    n_rows, length = data.shape
    x_edges = np.arange(length + 1, dtype=float) - 0.5

    if n_rows > 0 and length > 0:
        resolved_cmap = ensure_cmap(seqdata, cmap)
        state_count = max(len(seqdata.alphabet), 1)
        state_colors = np.asarray(
            [resolved_cmap(index / max(state_count - 1, 1)) for index in range(state_count)],
            dtype=float,
        )
        for code_attr, symbol_attr, alpha in (
            ("nr_code", "nr", missing_alpha),
            ("void_code", "void", void_alpha),
        ):
            code = _resolve_sequence_state_code(seqdata, code_attr, symbol_attr)
            if code is not None and 0 <= int(code) < state_count:
                opacity = float(alpha)
                state_colors[int(code), :3] = opacity * state_colors[int(code), :3] + (
                    1.0 - opacity
                )
                state_colors[int(code), 3] = 1.0
        weighted_cmap = ListedColormap(state_colors)
        ax.pcolormesh(
            x_edges,
            row_edges,
            data,
            shading="flat",
            cmap=weighted_cmap,
            vmin=-0.5,
            vmax=state_count - 0.5,
        )
    else:
        ax.set_xlim(-0.5, max(length - 0.5, 0.5))

    apply_time_axis(ax, seqdata, length)
    centers = (row_edges[:-1] + row_edges[1:]) / 2.0
    ax.set_yticks(centers)
    ax.set_yticklabels([str(label) for label in selection.labels])
    upper = float(row_edges[-1]) if row_edges.size > 1 else 0.5
    if upper == -0.5:
        upper = 0.5
    ax.set_ylim(upper, -0.5)

    if highlight is not None:
        for index in highlight:
            if 0 <= index < n_rows:
                ax.axhline(row_edges[index], color="red", linewidth=1.5)
                ax.axhline(row_edges[index + 1], color="red", linewidth=1.5)
    if apply_hatching:
        hatch_styles = {}
        for code_attr, symbol_attr, edgecolor, hatch in (
            ("nr_code", "nr", "gray", "///"),
            ("void_code", "void", "lightgray", "..."),
        ):
            code = _resolve_sequence_state_code(seqdata, code_attr, symbol_attr)
            if code is not None:
                hatch_styles[code] = (edgecolor, hatch)
        for row in range(n_rows):
            for column in range(length):
                style = hatch_styles.get(int(data[row, column]))
                if style is not None:
                    edgecolor, hatch = style
                    ax.add_patch(
                        Rectangle(
                            (column - 0.5, row_edges[row]),
                            1.0,
                            row_edges[row + 1] - row_edges[row],
                            fill=False,
                            edgecolor=edgecolor,
                            hatch=hatch,
                            linewidth=0,
                        )
                    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Sequence index")
    ax.set_title("Sequence index plot")
    if legend:
        state_legend(seqdata, ax=ax)
    return ax.figure, ax


def _render_fuzzyseqplot_panels(
    seqdata: Any,
    *,
    group: Any = None,
    group_levels: Sequence[Any] | None = None,
    plot_weights: np.ndarray | None = None,
    complete_plot_weights: np.ndarray | None = None,
    type: str = "i",
    sortv: Any = None,
    main: Any = None,
    with_legend: bool | None = None,
    **plot_kwargs: Any,
):
    """Orchestrate group panels using the appropriate public traminex facilities.

    ``plot_weights`` is the fuzzy sidecar described in :func:`_copy_sts_rows`.
    When it is supplied it, rather than ``seqdata.weights``, is the source of
    each panel's case weights. Non-index plots continue through public
    :func:`traminex.seqplot`; fuzzy index cells use the narrow local weighted
    adapter while traminex still owns selection and presentation utilities.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib required for fuzzyseqplot") from error

    sequence = seqdata.copy() if isinstance(seqdata, STSList) else seqdef(seqdata)
    if group is None:
        group_values = None
        levels = ["Unique group"] if group_levels is None else list(group_levels)
    else:
        group_values = np.asarray(group, dtype=object).reshape(-1)
        if group_values.shape != (sequence.nseq,):
            raise ValueError("group must contain one value per sequence")
        levels = list(pd.unique(group_values)) if group_levels is None else list(group_levels)
    if not levels:
        raise ValueError("group must contain at least one plotted level")
    if not isinstance(type, str):
        raise TypeError("type must be a string")
    if with_legend is not None and not isinstance(with_legend, (bool, np.bool_)):
        raise TypeError("with_legend must be a boolean")
    if with_legend is not None and "legend" in plot_kwargs:
        raise TypeError("with_legend and legend cannot both be specified")

    index_weighted = plot_kwargs.get("weighted", True)
    if type in _INDEX_PLOT_TYPES and not isinstance(index_weighted, (bool, np.bool_)):
        raise TypeError("weighted must be a boolean")
    explicitly_unweighted_index = type in _INDEX_PLOT_TYPES and not bool(index_weighted)

    sidecar = None
    if plot_weights is not None:
        sidecar = np.asarray(plot_weights, dtype=float).reshape(-1)
        if sidecar.shape != (sequence.nseq,):
            raise ValueError("plot_weights must contain one value per sequence")
        if not explicitly_unweighted_index:
            _validate_fuzzy_plot_weights(sidecar)

    complete_sidecar = None
    if complete_plot_weights is not None:
        complete_sidecar = np.asarray(complete_plot_weights, dtype=float).reshape(-1)
        if not explicitly_unweighted_index:
            _validate_fuzzy_plot_weights(complete_sidecar)
    panel_positions = [
        (
            np.arange(sequence.nseq, dtype=int)
            if group_values is None
            else np.flatnonzero(group_values == level)
        )
        for level in levels
    ]
    if type in _INDEX_PLOT_TYPES and not explicitly_unweighted_index and sidecar is not None:
        for positions in panel_positions:
            _weighted_index_row_edges(sidecar[positions])

    figsize = plot_kwargs.pop("figsize", (8.0, max(3.0, 3.0 * len(levels))))
    figure, axes = plt.subplots(len(levels), 1, figsize=figsize, squeeze=False)
    panels = axes.reshape(-1)
    for level, positions, axis in zip(levels, panel_positions, panels, strict=True):
        if sidecar is not None:
            panel_weights = sidecar[positions]
        elif sequence.weights is None:
            panel_weights = np.ones(positions.size, dtype=float)
        else:
            panel_weights = np.asarray(sequence.weights, dtype=float)[positions]
        attachable = np.all(np.isfinite(panel_weights)) and np.all(panel_weights >= 0.0)
        panel_sequence = _copy_sts_rows(
            sequence,
            positions,
            weights=panel_weights if attachable else None,
        )
        panel_sortv = sortv
        if sortv is not None and not isinstance(sortv, str):
            values = np.asarray(sortv)
            if values.shape == (sequence.nseq,):
                panel_sortv = values[positions]
        call_kwargs = dict(plot_kwargs)
        if panel_sortv is not None:
            call_kwargs["sortv"] = panel_sortv
        if with_legend is not None and type in _INDEX_PLOT_TYPES:
            call_kwargs["legend"] = bool(with_legend)
        if type in _INDEX_PLOT_TYPES:
            _render_weighted_index_panel(panel_sequence, ax=axis, **call_kwargs)
        else:
            seqplot(panel_sequence, type=type, ax=axis, **call_kwargs)
        if with_legend is False and axis.get_legend() is not None:
            axis.get_legend().remove()
        axis.set_title(str(level))
    if main is not None:
        figure.suptitle(str(main))
    figure.tight_layout(rect=(0, 0, 1, 0.96) if main is not None else None)
    returned_axes = panels[0] if len(panels) == 1 else panels
    return figure, returned_axes


def fuzzyseqplot(
    seqdata: Any,
    group: Any = None,
    membership_threashold: float = 0,
    type: str = "i",
    members_weighted: bool = True,
    memb_exp: float = 1,
    **plot_kwargs: Any,
):
    """Plot sequences using WeightedCluster 2.0 fuzzy-membership semantics.

    ``membership_threashold`` intentionally preserves the upstream spelling.
    Membership-bearing mappings/objects provide the Python fanny-equivalent
    protocol. Non-index rendering is delegated to public
    :func:`traminex.seqplot`; weighted index cells are drawn locally from
    public traminex selection, palette, tick, label, and legend facilities.
    R's ``main`` plot title is retained; ``with.legend`` maps to
    ``with_legend`` (the literal dotted spelling is also accepted through a
    keyword mapping). Index keyword ``weighted=False`` explicitly requests
    unit selection and row geometry; it is distinct from
    ``members_weighted=False``, which retains membership-only R weights.
    """
    sortv = plot_kwargs.pop("sortv", None)
    if "with.legend" in plot_kwargs:
        if "with_legend" in plot_kwargs:
            raise TypeError("with_legend was specified more than once")
        plot_kwargs["with_legend"] = plot_kwargs.pop("with.legend")
    is_factor = isinstance(group, pd.Categorical) or (
        isinstance(group, pd.Series) and isinstance(group.dtype, pd.CategoricalDtype)
    )
    if group is None or is_factor:
        sequence = seqdata.copy() if isinstance(seqdata, STSList) else seqdef(seqdata)
        categorical, levels = _categorical_values(group, sequence.nseq)
        # WeightedCluster 2.0 omits ``type`` on this first call, then continues.
        _render_fuzzyseqplot_panels(
            sequence,
            group=categorical,
            group_levels=levels,
            sortv=sortv,
            **plot_kwargs,
        )
        if is_factor:
            warnings.warn("‘^’ not meaningful for factors", RuntimeWarning, stacklevel=2)
        raise ValueError(_FUZZY_MEMBERSHIP_ERROR)

    prepared = _prepare_fuzzyseqplot_data(
        seqdata,
        group,
        membership_threashold=membership_threashold,
        members_weighted=members_weighted,
        memb_exp=memb_exp,
        sortv=sortv,
    )
    return _render_fuzzyseqplot_panels(
        prepared["seqdata"],
        group=prepared["group"],
        group_levels=prepared["group_levels"],
        plot_weights=prepared["plot_weights"],
        complete_plot_weights=prepared["complete_plot_weights"],
        type=type,
        sortv=prepared["sortv"],
        **plot_kwargs,
    )


def wcFuzzyRange(
    D: np.ndarray,
    kvals: Sequence[int],
    *,
    method: str = "FCMdd",
    m: float = 1.5,
    weights: Optional[np.ndarray] = None,
    dnoise: Optional[float] = None,
    dlambda: Optional[float] = None,
    eta: Optional[np.ndarray] = None,
    pam_n_init: int = 2,
    random_state: Optional[int] = None,
    crisp_labels: bool = False,
) -> Dict:
    """Run fuzzy c-medoids variants over a range of K on a precomputed distance matrix.

    Parameters
    - D: (N, N) symmetric distance matrix
    - kvals: iterable of K values to evaluate
    - method: 'FCMdd' | 'PCMdd'. The noise variants 'NCdd' and 'HNCdd' are
      refused here; call `fcmdd` directly for them (see Notes).
    - m: fuzziness parameter (>1)
    - weights: optional length-N weights
    - eta: PCMdd's length-K typicality scales (required for PCMdd)
    - dnoise/dlambda: accepted for signature compatibility only; they apply to
      the noise variants, which this function refuses
    - pam_n_init: number of k-medoids initializations per K (used to seed fuzzy runs)
    - random_state: base seed for reproducibility; per-K seeds are derived from it

    Returns a dict with keys: 'kvals', 'memb', 'medoids', 'stats'.
    The returned stats table contains fuzzy quality metrics (`FPC`, `FPE`, `XBf`, `AMS`)
    computed with observation weights when `weights` is provided. FPE uses
    every exact positive membership; an exact-zero entropy term contributes
    zero without evaluating its logarithm. The original validated finite
    weight vector reaches PAM, FCMdd, medoids, and every quality statistic
    unchanged. Each stage uses local exact or relative accumulators when raw
    binary64 products would lose range; no shared normalized vector is
    materialized. Dimensionless fuzzy quality calculations also use stable
    finite-distance ratios.

    Notes
    NCdd and HNCdd append a noise column, so their membership has K + 1
    columns but only K medoids. `wcFuzzyQuality` requires one medoid per
    membership column, and the noise column has none. There is also no R-defined
    meaning for FPC, FPE or XBf on a noise-bearing membership: R has no
    `wcFuzzyRange`, and its CLARA noise branch keeps the noise column for
    `mean_diss` and PBM but drops it for DB and AMS
    (`R/clara_clustrange.R:179-189`). Choosing a treatment here would invent
    semantics, so these methods are refused rather than silently reinterpreted.
    They remain fully available through `fcmdd` (2026-09-21 targeted audit, M-2).

    Example
    >>> from weightedcluster._seqdist import sequences_distance
    >>> import numpy as np
    >>> X = np.array([[0,0,0,0],[0,0,0,1],[1,1,1,1],[1,1,1,0]], dtype=int)
    >>> D = sequences_distance(X, metric="HAM", normalize=True)
    >>> fz = wcFuzzyRange(D, [2,3], method="FCMdd", m=1.6, random_state=42)
    """
    # Same normalisation as fcmdd (`str(method).upper()`), so every spelling
    # fcmdd would accept as a noise variant is refused here, before any work.
    if str(method).upper() in {"NCDD", "HNCDD"}:
        raise NotImplementedError(
            f"wcFuzzyRange does not support method={method!r}: the noise variants "
            "return a K+1-column membership, and the fuzzy quality indices "
            "(FPC, FPE, XBf, AMS) have no defined treatment of the noise column. "
            "Call weightedcluster.fuzzy.fcmdd directly for NCdd or HNCdd."
        )
    D = np.asarray(D, dtype=float)
    validate_distance_matrix(D, name="D")
    N = D.shape[0]
    w = None if weights is None else np.asarray(weights, dtype=float)
    if w is not None and w.shape != (N,):
        raise ValueError("weights must be length N")
    if w is not None:
        if not np.all(np.isfinite(w)) or np.any(w < 0.0) or not np.any(w > 0.0):
            raise ValueError("weights must be finite, non-negative, and sum to a positive value")
        # Keep the caller's finite representation through every stage.  A
        # binary64 division by the largest entry can turn a mathematical
        # positive (for example min-subnormal / max-finite) into exact zero,
        # irreversibly changing PAM, fuzzy-medoid, and quality objectives.
        # The stage-specific stable accumulators remove common scales without
        # materialising those unrepresentable ratios.
        w = w.copy()
    kvals_list = validate_kvals(kvals, max_k=N)
    pam_n_init_int = validate_integer_control(pam_n_init, name="pam_n_init")
    if pam_n_init_int < 1:
        raise ValueError("pam_n_init must be at least one")
    if not isinstance(crisp_labels, (bool, np.bool_)):
        raise TypeError("crisp_labels must be a boolean")
    memb_map: Dict[str, np.ndarray] = {}
    med_map: Dict[str, np.ndarray] = {}
    labels_map: Dict[str, np.ndarray] = {}
    stats_rows = []
    exact_pam_objectives = w is not None and _pam_requires_exact_objectives(D, w)
    if exact_pam_objectives:
        from ._fallback import weighted_medoids as _weighted_medoids  # type: ignore
    else:
        try:
            from ._wc_core import weighted_medoids as _weighted_medoids
        except ImportError:
            from ._fallback import weighted_medoids as _weighted_medoids  # type: ignore
    from .quality import wcFuzzyQuality

    base_seed = generate_random_seed(random_state, 0)
    for k_idx, k in enumerate(kvals_list):
        seed_k = generate_random_seed(base_seed, 1, k_idx)
        k_int = int(k)
        start_plan = _weighted_pam_start_plan(
            D,
            k_int,
            w,
            pam_n_init_int,
            np.random.default_rng(seed_k),
        )
        labels0: np.ndarray | None = None
        best_objective = float("inf")
        best_objective_component: tuple[int, int] | None = None
        for start in start_plan:
            candidate_labels, candidate_medoids, candidate_objective = _weighted_medoids(
                D,
                k_int,
                weights=w,
                max_iter=50,
                n_init=1,
                random_state=int(seed_k),
                init_medoids=start,
            )
            candidate_component = (
                _weighted_medoids_objective_component(D, candidate_medoids, w)
                if exact_pam_objectives and w is not None
                else None
            )
            exact_better = bool(
                candidate_component is not None
                and (
                    best_objective_component is None
                    or _component_less(candidate_component, best_objective_component)
                )
            )
            if (
                labels0 is None
                or exact_better
                or (candidate_component is None and candidate_objective < best_objective)
            ):
                labels0 = np.asarray(candidate_labels, dtype=int)
                best_objective = float(candidate_objective)
                best_objective_component = candidate_component
        assert labels0 is not None
        u0 = np.zeros((N, k_int), dtype=float)
        for c in range(k_int):
            u0[labels0 == c, c] = 1.0
        fz = fcmdd(
            D,
            u0,
            weights=w,
            method=str(method),
            m=float(m),
            dnoise=dnoise,
            dlambda=dlambda,
            eta=eta,
            alpha=FUZZY_CONVERGENCE_TOL,
            max_iter=FUZZY_MAX_ITER,
        )
        U = np.asarray(fz["memb"], dtype=float)
        med = np.asarray(fz["mobileCenters"], dtype=int)
        memb_map[f"cluster{k}"] = U
        med_map[f"cluster{k}"] = med
        if crisp_labels:
            labels1b = (np.argmax(U, axis=1) + 1).astype(int)
            labels_map[f"cluster{k}"] = labels1b
        q = wcFuzzyQuality(D, U, medoids=med, m=float(m), weights=w)
        s = q["stats"].copy()
        s["k"] = float(k)
        stats_rows.append(s)
    keys = sorted(stats_rows[0].keys()) if stats_rows else []
    stats = {key: [row.get(key) for row in stats_rows] for key in keys}
    out = {"kvals": kvals_list, "memb": memb_map, "medoids": med_map, "stats": stats}
    if crisp_labels:
        out["labels"] = labels_map
    return out


def _weighted_pam_start_plan(
    diss: np.ndarray,
    k: int,
    weights: np.ndarray | None,
    n_init: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """Generate uniform PAM starts independently of the selected kernel.

    The compiled and pure kernels both historically sampled medoid indices
    uniformly without replacement, while applying ``weights`` to the PAM
    objective.  Keeping those responsibilities separate preserves that
    weighting contract and removes the kernels' incompatible RNG streams.
    """
    n_observations = int(diss.shape[0])
    if weights is not None and weights.shape != (n_observations,):
        raise ValueError("weights must be length N")
    return tuple(
        np.asarray(rng.choice(n_observations, size=k, replace=False), dtype=int)
        for _ in range(n_init)
    )


def _normalise_inverse_distances(distances: np.ndarray, exponent: float) -> np.ndarray:
    """Normalise inverse-distance powers while preserving exact-zero ties."""
    membership = np.zeros_like(distances, dtype=float)
    zero_mask = distances == 0.0
    zero_rows = zero_mask.any(axis=1)
    if np.any(zero_rows):
        zero_counts = zero_mask[zero_rows].sum(axis=1, keepdims=True)
        membership[zero_rows] = zero_mask[zero_rows] / zero_counts

    positive_rows = ~zero_rows
    if np.any(positive_rows):
        positive_distances = distances[positive_rows]
        mantissas, binary_exponents = np.frexp(positive_distances)
        reference_columns = np.argmax(positive_distances, axis=1)
        reference_mantissas = mantissas[np.arange(mantissas.shape[0]), reference_columns][:, None]
        reference_exponents = binary_exponents[
            np.arange(binary_exponents.shape[0]), reference_columns
        ][:, None]
        # ``log1p`` preserves ratios that differ by only a few ulps.  That is
        # load-bearing when m is valid but very close to one: the exponent can
        # magnify a lost ratio bit into a visible membership error.
        log_distances = np.log1p((mantissas - reference_mantissas) / reference_mantissas) + (
            binary_exponents - reference_exponents
        ) * math.log(2.0)
        shifted_logs = log_distances - log_distances.min(axis=1, keepdims=True)
        with np.errstate(over="ignore", under="ignore"):
            scores = np.exp(-exponent * shifted_logs)
        membership[positive_rows] = scores / scores.sum(axis=1, keepdims=True)
    return membership


def _possibilistic_memberships(
    distances: np.ndarray,
    eta: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Compute ``1 / (1 + (distance / eta) ** exponent)`` stably."""
    membership = np.ones_like(distances, dtype=float)
    positive = distances > 0.0
    log_ratio = np.broadcast_to(-np.log(eta), distances.shape).copy()
    log_ratio[positive] += np.log(distances[positive])
    with np.errstate(over="ignore", under="ignore"):
        powered_log_ratio = exponent * log_ratio[positive]
        membership[positive] = np.exp(-np.logaddexp(0.0, powered_log_ratio))
    return membership


def _stable_nonnegative_mean(values: np.ndarray) -> float:
    """Return a mean without overflowing the reduction of finite values."""
    with np.errstate(over="ignore", invalid="ignore"):
        ordinary = float(np.mean(values))
    if np.isfinite(ordinary):
        return ordinary
    scale = float(np.max(values))
    if scale == 0.0:
        return 0.0
    return _stable_positive_product(scale, float(np.mean(values / scale)))


def _binary_product_components(*values: float) -> tuple[float, int]:
    """Represent a positive product as ``mantissa * 2**exponent``.

    Renormalising after every multiplication prevents an overflowing or
    underflowing intermediate from deciding whether the final product is
    representable.
    """
    mantissa = 1.0
    exponent = 0
    for value in values:
        value_mantissa, value_exponent = math.frexp(float(value))
        mantissa *= value_mantissa
        mantissa, adjustment = math.frexp(mantissa)
        exponent += value_exponent + adjustment
    return mantissa, exponent


def _binary_power_components(value: float, exponent: float) -> tuple[float, int]:
    """Represent a positive floating-point power without losing a subnormal."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        powered = float(np.float64(value) ** exponent)
    if np.isfinite(powered) and powered >= np.finfo(float).tiny:
        return math.frexp(powered)

    with localcontext() as context:
        context.prec = 200
        decimal_value = Decimal.from_float(float(value))
        decimal_exponent = Decimal.from_float(float(exponent))
        log_two = Decimal(2).ln()
        power_log2 = decimal_exponent * decimal_value.ln() / log_two
        power_exponent = int(power_log2.to_integral_value(rounding=ROUND_FLOOR)) + 1
        power_mantissa = float(context.power(Decimal(2), power_log2 - Decimal(power_exponent)))
    return power_mantissa, power_exponent


def _combine_binary_components(*components: tuple[float, int]) -> tuple[float, int]:
    """Combine already-normalised positive binary components."""
    mantissa = 1.0
    exponent = 0
    for value_mantissa, value_exponent in components:
        if value_mantissa == 0.0:
            return 0.0, 0
        mantissa *= value_mantissa
        mantissa, adjustment = math.frexp(mantissa)
        exponent += value_exponent + adjustment
    return mantissa, exponent


def _binary_components_to_float(mantissa: float, exponent: int) -> float:
    """Round a binary representation at the actual float boundary."""
    if mantissa == 0.0:
        return 0.0
    try:
        return float(math.ldexp(mantissa, exponent))
    except OverflowError:
        return float("inf")


def _sum_binary_component_pair(
    components: list[tuple[float, int]],
) -> tuple[float, int]:
    """Accumulate non-negative terms while retaining binary components."""
    positive = [(mantissa, exponent) for mantissa, exponent in components if mantissa > 0.0]
    if not positive:
        return 0.0, 0
    maximum_exponent = max(exponent for _, exponent in positive)
    scaled_total = math.fsum(
        math.ldexp(mantissa, exponent - maximum_exponent) for mantissa, exponent in positive
    )
    total_mantissa, adjustment = math.frexp(scaled_total)
    return total_mantissa, maximum_exponent + adjustment


def _sum_binary_components(components: list[tuple[float, int]]) -> float:
    """Accumulate non-negative binary terms with one final range decision."""
    return _binary_components_to_float(*_sum_binary_component_pair(components))


def _stable_positive_product(*values: float) -> float:
    """Multiply non-negative finite values when the final result is representable."""
    if any(value == 0.0 for value in values):
        return 0.0
    return _binary_components_to_float(*_binary_product_components(*values))


def _preserve_near_tie_candidate_order(
    costs: np.ndarray,
    membership: np.ndarray,
    exponent: float,
    weights: np.ndarray,
    candidate_distances: np.ndarray,
) -> np.ndarray:
    """Recheck only candidate costs indistinguishable at dot-product precision."""
    if costs.size < 2 or not bool(np.all(np.isfinite(costs))):
        return costs
    positive_rows = np.flatnonzero((membership > 0.0) & (weights > 0.0))
    if positive_rows.size == 0:
        return costs
    ordered = np.sort(costs)
    scale = max(float(ordered[1]), float(ordered[0]), np.finfo(float).tiny)
    rounding_bound = 8.0 * max(1, int(positive_rows.size)) * np.finfo(float).eps * scale
    if float(ordered[1] - ordered[0]) > rounding_bound:
        return costs
    near = np.flatnonzero(costs <= float(ordered[0] + rounding_bound))
    if near.size < 2:
        return costs

    exact_costs: dict[int, Decimal] = {}
    with localcontext() as context:
        context.prec = 2500
        decimal_exponent = Decimal.from_float(float(exponent))
        for column in near:
            total = Decimal(0)
            for row in positive_rows:
                distance = float(candidate_distances[row, column])
                if distance == 0.0:
                    continue
                membership_value = float(membership[row])
                if membership_value == 1.0:
                    powered = Decimal(1)
                else:
                    powered = context.power(
                        Decimal.from_float(membership_value),
                        decimal_exponent,
                    )
                total += (
                    powered * Decimal.from_float(float(weights[row])) * Decimal.from_float(distance)
                )
            exact_costs[int(column)] = total

    exact_best = min(near, key=lambda column: (exact_costs[int(column)], int(column)))
    ordinary_best = int(np.argmin(costs))
    if int(exact_best) == ordinary_best:
        return costs
    adjusted = costs.copy()
    minimum = float(costs[ordinary_best])
    if minimum > 0.0:
        adjusted[int(exact_best)] = np.nextafter(minimum, -np.inf)
    else:
        adjusted[ordinary_best] = np.nextafter(0.0, np.inf)
        adjusted[int(exact_best)] = 0.0
    return adjusted


def _stable_candidate_costs(
    membership: np.ndarray,
    exponent: float,
    weights: np.ndarray,
    candidate_distances: np.ndarray,
) -> np.ndarray:
    """Compute medoid candidate costs up to a common positive scale."""
    with np.errstate(over="ignore", invalid="ignore"):
        ordinary_weights = (membership**exponent) * weights
        ordinary_costs = ordinary_weights @ candidate_distances
    positive_rows = (membership > 0.0) & (weights > 0.0)
    lost_positive_weight = positive_rows & (
        (ordinary_weights == 0.0) | ~np.isfinite(ordinary_weights)
    )
    exceptional = bool(np.any(lost_positive_weight)) or not bool(
        np.all(np.isfinite(ordinary_costs))
    )
    if not exceptional:
        for column in np.flatnonzero(ordinary_costs == 0.0):
            if np.any(candidate_distances[positive_rows, column] > 0.0):
                exceptional = True
                break
    if not exceptional:
        return _preserve_near_tie_candidate_order(
            ordinary_costs,
            membership,
            exponent,
            weights,
            candidate_distances,
        )

    positive_indices = np.flatnonzero(positive_rows)
    if positive_indices.size == 0:
        return np.zeros(candidate_distances.shape[1], dtype=float)

    candidate_components: list[tuple[float, int]] = []
    for column in range(candidate_distances.shape[1]):
        terms = []
        for row in positive_indices:
            distance = float(candidate_distances[row, column])
            if distance == 0.0:
                continue
            terms.append(
                _combine_binary_components(
                    _binary_power_components(float(membership[row]), exponent),
                    math.frexp(float(weights[row])),
                    math.frexp(distance),
                )
            )
        candidate_components.append(_sum_binary_component_pair(terms))

    positive_costs = [component for component in candidate_components if component[0] > 0.0]
    if not positive_costs:
        return np.zeros(candidate_distances.shape[1], dtype=float)
    minimum_mantissa, minimum_exponent = min(
        positive_costs, key=lambda component: (component[1], component[0])
    )
    scaled_costs = np.zeros(candidate_distances.shape[1], dtype=float)
    for column, (mantissa, binary_exponent) in enumerate(candidate_components):
        if mantissa == 0.0:
            continue
        try:
            scaled_costs[column] = math.ldexp(
                mantissa / minimum_mantissa,
                binary_exponent - minimum_exponent,
            )
        except OverflowError:
            scaled_costs[column] = float("inf")
    return _preserve_near_tie_candidate_order(
        scaled_costs,
        membership,
        exponent,
        weights,
        candidate_distances,
    )


def _stable_adaptive_noise_distance(
    distances: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    dlambda: float,
) -> float:
    """Compute WeightedCluster's adaptive noise distance without ``inf / inf``."""
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_distances = distances * weights[:, None]
        numerator = float(weighted_distances.sum())
        denominator = float(n_clusters) * float(weights.sum())
        ordinary = float(dlambda) * (numerator / denominator)
    positive_term = (distances > 0.0) & (weights[:, None] > 0.0)
    lost_positive_term = positive_term & (weighted_distances == 0.0)
    rounded_subnormal_term = positive_term & (
        (weighted_distances > 0.0) & (weighted_distances < np.finfo(float).tiny)
    )
    if (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > 0.0
        and np.isfinite(ordinary)
        and not np.any(lost_positive_term)
        and not np.any(rounded_subnormal_term)
    ):
        return ordinary

    if dlambda == 0.0:
        return 0.0
    numerator_components = []
    for row in np.flatnonzero(weights > 0.0):
        for distance in distances[row]:
            if distance > 0.0:
                numerator_components.append(
                    _binary_product_components(float(distance), float(weights[row]))
                )
    numerator_mantissa, numerator_exponent = _sum_binary_component_pair(numerator_components)
    if numerator_mantissa == 0.0:
        return 0.0
    weight_components = [math.frexp(float(weight)) for weight in weights if weight > 0.0]
    denominator_components = _combine_binary_components(
        _sum_binary_component_pair(weight_components),
        math.frexp(float(n_clusters)),
    )
    denominator_mantissa, denominator_exponent = denominator_components
    ratio_mantissa, adjustment = math.frexp(numerator_mantissa / denominator_mantissa)
    ratio_components = (
        ratio_mantissa,
        numerator_exponent - denominator_exponent + adjustment,
    )
    return _binary_components_to_float(
        *_combine_binary_components(
            math.frexp(float(dlambda)),
            ratio_components,
        )
    )


def _stable_fuzzy_functional(
    distances: np.ndarray,
    membership: np.ndarray,
    exponent: float,
    weights: np.ndarray,
) -> float:
    """Sum positive fuzzy objective terms with a binary-scaled fallback."""
    with np.errstate(over="ignore", invalid="ignore"):
        powered_membership = membership**exponent
        ordinary_terms = distances * powered_membership * weights[:, None]
        ordinary = float(ordinary_terms.sum())

    positive = (distances > 0.0) & (membership > 0.0) & (weights[:, None] > 0.0)
    lost_positive_term = positive & ((ordinary_terms == 0.0) | ~np.isfinite(ordinary_terms))
    rounded_subnormal_power = (membership > 0.0) & (
        (powered_membership > 0.0) & (powered_membership < np.finfo(float).tiny)
    )
    rounded_subnormal_term = positive & (
        (ordinary_terms > 0.0) & (ordinary_terms < np.finfo(float).tiny)
    )
    if (
        np.isfinite(ordinary)
        and not np.any(lost_positive_term)
        and not np.any(rounded_subnormal_power)
        and not np.any(rounded_subnormal_term)
    ):
        return ordinary

    if not np.any(positive):
        return 0.0
    row_indices, column_indices = np.where(positive)
    components = []
    for row, column in zip(row_indices, column_indices, strict=True):
        components.append(
            _combine_binary_components(
                math.frexp(float(distances[row, column])),
                _binary_power_components(float(membership[row, column]), exponent),
                math.frexp(float(weights[row])),
            )
        )
    return _sum_binary_components(components)


def fcmdd(
    D: np.ndarray,
    u_init: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    method: str = "FCMdd",
    m: float = 1.5,
    dnoise: Optional[float] = None,
    dlambda: Optional[float] = None,
    eta: Optional[np.ndarray] = None,
    alpha: float = 1e-3,
    max_iter: int = 100,
) -> Dict:
    """Fuzzy (and noise) c-medoids on a precomputed distance matrix.

    Parameters
    ----------
    D : np.ndarray
        (N, N) distance matrix
    u_init : np.ndarray
        (N, K) initial membership matrix for mobile clusters
    weights : np.ndarray, optional
        Length-N observation weights
    method : str, default='FCMdd'
        Fuzzy clustering variant: 'FCMdd', 'NCdd', 'HNCdd', or 'PCMdd'
    m : float, default=1.5
        Fuzziness exponent (must be > 1). HNCdd accepts any finite value and uses 1,
        as R's ``wfcmdd`` does.
    dnoise : float, optional
        Fixed noise distance for NCdd/HNCdd methods
    dlambda : float, optional
        Adaptive noise parameter for NCdd (updates dnoise each iteration)
    eta : np.ndarray, optional
        Length-K scale parameters for PCMdd method
    alpha : float, default=1e-3
        Convergence tolerance threshold
    max_iter : int, default=100
        Maximum number of iterations

    Returns
    -------
    dict
        Results with keys:
        - 'memb': (N, K) or (N, K+1) membership matrix
        - 'mobileCenters': list of medoid indices
        - 'dnoise': noise distance (or None)
        - 'functional': objective function value

    Notes
    -----
    Faithful to R ``WeightedCluster:::wfcmdd``: each cluster's medoid is chosen
    only among points not fully assigned (membership == 1) to another cluster
    (R's ``apply(u[, -k], 1, max) < 1``). This matters mainly when ``u_init`` is an
    arbitrary crisp partition rather than a k-medoids solution. The only deviation
    is a guard for a degenerate empty-cluster init (which R errors on): there the
    candidate set falls back to all not-yet-chosen points so the run proceeds.
    """
    D = np.asarray(D, dtype=float)
    validate_distance_matrix(D, name="D")
    N = D.shape[0]
    # Own the working matrix: the PCMdd branch updates columns in place.
    u = np.array(u_init, dtype=float, copy=True)
    if u.ndim != 2 or u.shape[0] != N or u.shape[1] == 0:
        raise ValueError("u_init must be (N,K) with K >= 1")
    if not np.all(np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("u_init must contain finite non-negative membership values")
    K = u.shape[1]
    w = np.ones(N, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (N,):
        raise ValueError("weights must be length N")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or not np.any(w > 0.0):
        raise ValueError("weights must be finite, non-negative, and sum to a positive value")
    method_up = str(method).upper()
    if method_up not in {"FCMDD", "NCDD", "HNCDD", "PCMDD"}:
        raise NotImplementedError("Supported methods: FCMdd, NCdd, HNCdd, PCMdd")
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Real):
        raise ValueError("m must be a finite number greater than one")
    m_value = float(m)
    if method_up == "HNCDD":
        # R's wfcmdd sets m <- 1 in its HNCdd branch (wfcmdd.r:24-28), so HNCdd's
        # medoid search and functional use u^1 whatever m the caller passes; only
        # FCMdd, NCdd and PCMdd need m > 1 (2026-09-21 targeted audit, M-3).
        if not np.isfinite(m_value):
            raise ValueError("m must be a finite number")
        m_value = 1.0
    elif not np.isfinite(m_value) or m_value <= 1.0:
        raise ValueError("m must be a finite number greater than one")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or alpha_value < 0.0:
        raise ValueError("alpha must be a finite non-negative number")
    max_iter_int = validate_integer_control(max_iter, name="max_iter")
    if max_iter_int < 1:
        raise ValueError("max_iter must be at least one")
    use_noise = method_up == "NCDD" or method_up == "HNCDD"
    if use_noise and dnoise is not None:
        if isinstance(dnoise, (bool, np.bool_)) or not isinstance(dnoise, Real):
            raise ValueError("dnoise must be a finite non-negative number")
        dnoise = float(dnoise)
        if not np.isfinite(dnoise) or dnoise < 0.0:
            raise ValueError("dnoise must be a finite non-negative number")
    if use_noise and dlambda is not None:
        if isinstance(dlambda, (bool, np.bool_)) or not isinstance(dlambda, Real):
            raise ValueError("dlambda must be a finite non-negative number")
        dlambda = float(dlambda)
        if not np.isfinite(dlambda) or dlambda < 0.0:
            raise ValueError("dlambda must be a finite non-negative number")
    if method_up == "HNCDD" and dnoise is None:
        raise ValueError("HNCdd requires dnoise")
    if use_noise and (dnoise is None and dlambda is None and method_up != "HNCDD"):
        # fallback heuristic for NCdd: use mean distance
        dnoise = _stable_nonnegative_mean(D)
    # Optionally append noise column to membership matrix for NCdd/HNCdd
    if use_noise:
        u = np.hstack([u, np.zeros((N, 1), dtype=float)])
    K_nc = u.shape[1]
    # PCMdd requires eta per cluster
    if method_up == "PCMDD":
        if eta is None:
            raise ValueError("PCMdd requires eta (length K)")
        try:
            eta = np.asarray(eta, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("eta must be a finite positive vector of length K") from exc
        if eta.ndim != 1 or eta.shape != (K,) or not np.all(np.isfinite(eta)) or np.any(eta <= 0.0):
            raise ValueError("eta must contain finite positive values and have length K")
    # Medoids indices
    med = np.full(K, -1, dtype=int)
    u_prev = np.zeros_like(u)
    # main loop
    dist2med = np.zeros((N, K_nc), dtype=float)
    if use_noise:
        dist2med[:, -1] = float(dnoise if dnoise is not None else 0.0)
    for iteration in range(1, max_iter_int + 1):
        # 1) choose medoids for mobile clusters
        for k in range(K):
            # Candidate rule, faithful to R wfcmdd (fuzzyfunc/wfcmdd line 63):
            #   candidates <- which(apply(u[, -k], 1, max) < 1 & !(1:n %in% med[0:(k-1)]))
            # A point may be cluster k's medoid only if it is NOT fully assigned
            # (membership == 1) to some *other* cluster, and is not already a
            # medoid of clusters 0..k-1. The "max over other columns < 1" test
            # spans every other column of u, including the appended noise column
            # for NCdd/HNCdd, matching R's u[, -k]. Omitting it lets fcmdd pull a
            # medoid from outside the cluster on the first sweep (when a crisp
            # init makes other columns exactly 1), diverging from R.
            others = np.delete(u, k, axis=1)
            if others.shape[1] == 0:
                eligible = np.ones(N, dtype=bool)
            else:
                eligible = others.max(axis=1) < 1.0
            if k > 0:
                eligible &= ~np.isin(np.arange(N), med[:k])
            candidates = np.where(eligible)[0]
            if candidates.size == 0:
                # Degenerate init: every point is fully committed to another
                # cluster, i.e. column k is empty in a crisp init. R's wfcmdd would
                # error here, but the Python CLARA pipeline can hand fcmdd such an
                # init on tiny subsamples. Fall back to all not-yet-chosen points
                # (the pre-fix behaviour) so the run proceeds; healthy inits are
                # unaffected because the restriction above is non-empty whenever
                # cluster k has at least one member.
                all_idx = np.arange(N)
                candidates = all_idx[~np.isin(all_idx, med[:k])] if k > 0 else all_idx
            # objective for each candidate j: sum_i (u_i,k^m * w_i * D_{i,j})
            uk = u[:, k]
            costs = _stable_candidate_costs(uk, m_value, w, D[:, candidates])
            j_best = int(candidates[int(np.argmin(costs))])
            med[k] = j_best
            dist2med[:, k] = D[:, j_best]
        # 1.b adaptive dnoise for NCdd with dlambda
        if use_noise and dlambda is not None:
            dnoise = _stable_adaptive_noise_distance(
                dist2med[:, :K],
                w,
                K,
                float(dlambda),
            )
            dist2med[:, -1] = dnoise
        # 2) update membership
        if method_up in ("FCMDD", "NCDD"):
            u = _normalise_inverse_distances(dist2med, 1.0 / (m_value - 1.0))
        elif method_up == "HNCDD":
            # Hard assignment to argmin over K+noise
            u.fill(0.0)
            mins = dist2med.argmin(axis=1)
            for i in range(N):
                u[i, mins[i]] = 1.0
        elif method_up == "PCMDD":
            # Possibilistic C-Medoids memberships
            u = _possibilistic_memberships(
                dist2med[:, :K],
                eta,
                1.0 / (m_value - 1.0),
            )
        # 3) check convergence
        # Preserve R's three-sweep convergence warm-up when the requested
        # budget permits it, but never exceed the public max_iter bound.
        converged = iteration > 2 and np.max(np.abs(u - u_prev)) <= alpha_value
        u_prev = u.copy()
        if converged:
            break
    # functional value
    # For FCMdd/NCdd: sum(dist2med * (u^m) * weights)
    func = _stable_fuzzy_functional(dist2med, u, m_value, w)
    res = {
        "dnoise": (
            None
            if not use_noise
            else float(dnoise) if dnoise is not None else float(dist2med[:, -1].mean())
        ),
        "memb": u,
        "mobileCenters": med.tolist(),
        "functional": func,
    }
    return res

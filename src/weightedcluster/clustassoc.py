from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from traminex import dissmfacw

from .schemas import ClustassocResult


@dataclass(frozen=True)
class _BICFit:
    bic: float
    loglik: float
    df: int
    nobs: float
    converged: bool


def _factor_design(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """R ``model.matrix(~ factor(labels))`` without its column names."""
    _, codes = np.unique(np.asarray(labels).reshape(-1), return_inverse=True)
    n_levels = int(codes.max() + 1) if codes.size else 0
    columns = [np.ones(codes.size, dtype=float)]
    columns.extend((codes == level).astype(float) for level in range(1, n_levels))
    return np.column_stack(columns), codes, n_levels


def _bic_lm(y: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> _BICFit:
    """Exact ``BIC(lm(y ~ factor(labels), weights=))`` sufficient statistics.

    R's ``logLik.lm`` counts the fitted coefficient rank plus the Gaussian
    variance parameter, excludes zero-weight rows from ``nobs``, and includes
    ``sum(log(weights))`` in the weighted Gaussian log-likelihood.
    """
    X, _, _ = _factor_design(labels)
    sqrt_weights = np.sqrt(weights)
    X_weighted = X * sqrt_weights[:, None]
    y_weighted = y * sqrt_weights
    coefficients, _, rank, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    residuals = y - X @ coefficients

    positive = weights > 0
    nobs = int(np.count_nonzero(positive))
    if nobs == 0:
        raise ValueError("weights must contain at least one positive value")
    retained_weights = weights[positive]
    rss = float(np.sum(retained_weights * residuals[positive] ** 2))
    if rss < 0:
        raise ValueError("weighted residual sum of squares must be non-negative")
    if rss == 0:
        loglik = float("inf")
    else:
        loglik = 0.5 * (
            float(np.log(retained_weights).sum())
            - nobs * (np.log(2.0 * np.pi) + 1.0 - np.log(float(nobs)) + np.log(rss))
        )
    df = int(rank) + 1
    bic = -2.0 * loglik + np.log(float(nobs)) * df
    return _BICFit(float(bic), float(loglik), df, float(nobs), True)


def _multinomial_loss_gradient(
    parameters: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
) -> tuple[float, np.ndarray]:
    """Baseline-class multinomial negative log-likelihood and gradient."""
    coefficients = parameters.reshape(X.shape[1], n_classes - 1)
    eta = X @ coefficients
    row_max = np.maximum(0.0, np.max(eta, axis=1))
    exp_eta = np.exp(eta - row_max[:, None])
    denominator = np.exp(-row_max) + exp_eta.sum(axis=1)
    log_denominator = row_max + np.log(denominator)

    selected_eta = np.zeros(y.size, dtype=float)
    nonbaseline = y > 0
    selected_eta[nonbaseline] = eta[nonbaseline, y[nonbaseline] - 1]
    loss = float(np.sum(weights * (log_denominator - selected_eta)))

    probabilities = exp_eta / denominator[:, None]
    targets = np.zeros_like(probabilities)
    targets[nonbaseline, y[nonbaseline] - 1] = 1.0
    gradient = X.T @ (weights[:, None] * (probabilities - targets))
    return loss, gradient.reshape(-1)


class _NnetAbstolReached(Exception):
    """Internal control flow for nnet's absolute objective tolerance."""


def _nnet_boundary_loss(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
) -> tuple[float, bool]:
    """Follow ``nnet::multinom`` at a separated likelihood boundary.

    A factor-only multinomial model has a closed-form likelihood unless a cell
    probability lies on the zero boundary. R's nnet fit stops such separated
    models when the objective first crosses ``abstol=1e-4`` (rather than at the
    unattainable infinite-coefficient MLE). SciPy's BFGS uses a different line
    search, but the same zero start, objective, gradient, maximum iteration
    count, and absolute stop reproduce R's BIC within that stated tolerance.
    """
    initial = np.zeros(X.shape[1] * (n_classes - 1), dtype=float)
    stopped: dict[str, float] = {}

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return _multinomial_loss_gradient(values, X, y, weights, n_classes)

    def callback(values: np.ndarray) -> None:
        loss, _ = objective(values)
        if loss <= 1e-4:
            stopped["loss"] = loss
            raise _NnetAbstolReached

    try:
        result = minimize(
            objective,
            initial,
            method="BFGS",
            jac=True,
            callback=callback,
            options={"gtol": 1e-12, "maxiter": 100},
        )
    except _NnetAbstolReached:
        return float(stopped["loss"]), True
    return float(result.fun), bool(result.success or result.fun <= 1e-4)


def _bic_multinomial(y: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> _BICFit:
    """Faithful ``BIC(nnet::multinom(y ~ factor(labels), weights=))``.

    With a factor as the sole predictor, the maximum likelihood is the entropy
    of the weighted category counts within each factor level. This computes
    that likelihood directly, then uses nnet's documented parameter count and
    frequency-weight ``nobs=sum(weights)``. Boundary fits use the nnet-style
    finite convergence rule above instead of silently substituting a penalized
    logistic-regression approximation.
    """
    categories, y_codes = np.unique(np.asarray(y).astype(str), return_inverse=True)
    if categories.size < 2:
        raise ValueError("categorical covar must contain at least two levels")
    X, label_codes, n_levels = _factor_design(labels)
    n_classes = int(categories.size)
    counts = np.zeros((n_levels, n_classes), dtype=float)
    np.add.at(counts, (label_codes, y_codes), weights)
    row_weights = counts.sum(axis=1)
    if np.any(row_weights <= 0):
        raise ValueError("each cluster level must have positive total weight")

    boundary = bool(np.any(counts == 0))
    if boundary:
        loss, converged = _nnet_boundary_loss(X, y_codes, weights, n_classes)
    else:
        probabilities = counts / row_weights[:, None]
        positive = counts > 0
        loss = -float(np.sum(counts[positive] * np.log(probabilities[positive])))
        converged = True

    df = int((n_classes - 1) * n_levels)
    nobs = float(weights.sum())
    if nobs <= 0:
        raise ValueError("weights must have a positive sum")
    loglik = -loss
    bic = 2.0 * loss + np.log(nobs) * df
    return _BICFit(float(bic), float(loglik), df, nobs, converged)


def _categorical_association(
    covar: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Weighted Cramer's V and likelihood-ratio statistic extension fields."""
    categories, y_codes = np.unique(covar.astype(str), return_inverse=True)
    _, label_codes, n_levels = _factor_design(labels)
    table = np.zeros((n_levels, len(categories)), dtype=float)
    np.add.at(table, (label_codes, y_codes), weights)
    row_sums = table.sum(axis=1)
    column_sums = table.sum(axis=0)
    total = float(table.sum())
    expected = np.outer(row_sums, column_sums) / total
    valid = expected > 0
    chi2 = float(np.sum((table[valid] - expected[valid]) ** 2 / expected[valid]))
    denominator = total * float(max(1, min(n_levels - 1, len(categories) - 1)))
    cramers_v = float(np.sqrt(chi2 / denominator)) if denominator > 0 else float("nan")
    positive = valid & (table > 0)
    likelihood_ratio = float(
        2.0 * np.sum(table[positive] * np.log(table[positive] / expected[positive]))
    )
    return cramers_v, likelihood_ratio


def clustassoc(
    clustrange: Mapping[str, Any],
    diss: np.ndarray,
    covar: Sequence,
    *,
    weights: Optional[Sequence[float]] = None,
) -> ClustassocResult:
    """Association between cluster solutions and a covariate.

    The four WeightedCluster core fields are reproduced from the installed R
    2.0 implementation: ``Remaining`` and ``Unaccounted`` delegate to
    :func:`traminex.dissmfacw`, while ``BIC`` follows R ``lm`` for numeric
    covariates and ``nnet::multinom`` for categorical covariates. Results are
    aligned to ``No Clustering`` followed by ``clustrange['kvals']``.

    Iteration exposes the exact four R data-frame columns. Established Python
    diagnostics remain available by key as off-table additions.
    """
    shape_error = (
        " [!] diss should be a dissimilarity matrix or a dist object and "
        "covar should be a variable with one value per observation."
    )
    D = np.asarray(diss, dtype=float)
    covar_categories: list[Any] | None = None
    if isinstance(covar, pd.Categorical):
        covar_categories = list(covar.categories)
    elif isinstance(covar, pd.Series) and isinstance(covar.dtype, pd.CategoricalDtype):
        covar_categories = list(covar.cat.categories)
    cov = np.asarray(covar)
    if cov.ndim != 1:
        raise ValueError(shape_error)
    original_n = int(cov.shape[0])
    if D.ndim != 2 or D.shape != (original_n, original_n):
        raise ValueError(shape_error)

    if weights is None:
        full_weights = np.ones(original_n, dtype=float)
    else:
        full_weights = np.asarray(weights, dtype=float).reshape(-1)
        if full_weights.shape != (original_n,):
            raise ValueError("NA/NaN/Inf in foreign function call (arg 1)")
        if not np.all(np.isfinite(full_weights)):
            raise ValueError("NA/NaN/Inf in foreign function call (arg 1)")
        if np.any(full_weights < 0):
            warnings.warn("NaNs produced", RuntimeWarning, stacklevel=2)
            raise ValueError("NA/NaN/Inf in foreign function call (arg 1)")

    complete = ~pd.isna(cov)
    if not np.any(complete):
        raise ValueError("subscript out of bounds")
    keep = np.flatnonzero(complete)

    is_numeric = bool(
        pd.api.types.is_numeric_dtype(cov.dtype) and not pd.api.types.is_bool_dtype(cov.dtype)
    )
    if is_numeric:
        cov_model = cov[keep].astype(float)
        if not np.all(np.isfinite(cov_model)):
            raise ValueError("NA/NaN/Inf in foreign function call (arg 1)")
        cov_diss = cov_model
    else:
        cov_model = cov[keep].astype(str)
        observed = set(cov_model.tolist())
        levels = (
            [str(value) for value in covar_categories]
            if covar_categories is not None
            else list(dict.fromkeys(cov_model.tolist()))
        )
        if len(observed) < 2:
            raise ValueError("contrasts can be applied only to factors with 2 or more levels")
        empty_levels = [level for level in levels if level not in observed]
        cov_diss = cov_model

    if not isinstance(clustrange, Mapping):
        raise ValueError("object of type 'closure' is not subsettable")
    stats = clustrange.get("stats")
    if stats is None:
        raise ValueError("attempt to set 'colnames' on an object with less than two dimensions")
    if isinstance(stats, pd.DataFrame):
        stats_columns = [str(value) for value in stats.columns]
        stats_nrow = int(stats.shape[0])
        stats_row_names = [str(value) for value in stats.index]
    elif isinstance(stats, Mapping):
        stats_columns = [str(value) for value in stats]
        stats_lengths = [len(np.asarray(value).reshape(-1)) for value in stats.values()]
        stats_nrow = stats_lengths[0] if stats_lengths else 0
        if any(length != stats_nrow for length in stats_lengths):
            raise ValueError("arguments imply differing number of rows")
        explicit_row_names = getattr(stats, "row_names", None)
        stats_row_names = (
            [str(value) for value in explicit_row_names] if explicit_row_names is not None else []
        )
    else:
        stats_array = np.asarray(stats)
        if stats_array.ndim < 2:
            raise ValueError("attempt to set 'colnames' on an object with less than two dimensions")
        stats_nrow, stats_ncol = stats_array.shape[:2]
        stats_columns = [str(index + 1) for index in range(int(stats_ncol))]
        stats_row_names = []
    if len(stats_columns) < 3:
        raise ValueError("undefined columns selected")

    kvals = [int(value) for value in clustrange.get("kvals", [])]
    clustering = clustrange.get("clustering", {})
    if not isinstance(clustering, Mapping) or len(clustering) == 0:
        raise ValueError("argument of length 0")
    clustering_items = list(clustering.items())
    if stats_nrow != len(clustering_items):
        raise ValueError(f"replacement has {len(clustering_items)} rows, data has {stats_nrow}")
    if len(kvals) != stats_nrow:
        raise ValueError(f"replacement has {len(kvals) + 1} rows, data has {stats_nrow + 1}")
    if not stats_row_names:
        stats_row_names = [str(name) for name, _ in clustering_items]
    if len(stats_row_names) != stats_nrow:
        raise ValueError("invalid 'row.names' length")

    labels_by_k: list[np.ndarray] = []
    for _, values in clustering_items:
        labels = np.asarray(values).reshape(-1)
        if labels.shape[0] != original_n:
            raise ValueError("variable lengths differ (found for 'covar')")
        labels_by_k.append(labels[keep])

    D = D[np.ix_(keep, keep)]
    w = full_weights[keep]
    all_zero_weights = bool(float(w.sum()) == 0.0)
    if not is_numeric:
        for _ in range(len(labels_by_k) + 1):
            for level in empty_levels:
                warnings.warn(
                    f"group ‘{level}’ is empty",
                    RuntimeWarning,
                    stacklevel=2,
                )

    baseline_labels = np.ones(len(keep), dtype=int)
    if all_zero_weights:
        remaining: List[float] = [float("nan")] * (len(labels_by_k) + 1)
        unaccounted: List[float] = [1.0] + [float("nan")] * len(labels_by_k)
        zero_weight_bic = float("nan") if is_numeric else float("-inf")
        bic_fits = [
            _BICFit(zero_weight_bic, float("nan"), 0, 0.0, True)
            for _ in range(len(labels_by_k) + 1)
        ]
    else:
        null_discrepancy = dissmfacw(
            D,
            factors={"covar": cov_diss},
            weights=w,
            nperm=1,
        )
        remaining = [float(null_discrepancy.pseudo_R2[0])]
        unaccounted = [1.0]
        bic_fits = [
            (
                _bic_lm(cov_model, baseline_labels, w)
                if is_numeric
                else _bic_multinomial(cov_model, baseline_labels, w)
            )
        ]
    v_values: List[float] = [float("nan")]
    lrt_values: List[float] = [float("nan")]

    if all_zero_weights:
        v_values.extend([float("nan")] * len(labels_by_k))
        lrt_values.extend([float("nan")] * len(labels_by_k))
    else:
        for labels in labels_by_k:
            discrepancy = dissmfacw(
                D,
                factors={"clustf": labels.astype(str), "covar": cov_diss},
                weights=w,
                nperm=1,
            )
            covariate_r2 = float(discrepancy.pseudo_R2[1])
            denominator = float(discrepancy.pseudo_R2[2] - discrepancy.pseudo_R2[0])
            remaining.append(covariate_r2)
            with np.errstate(divide="ignore", invalid="ignore"):
                unaccounted.append(float(np.divide(covariate_r2, denominator)))

            if is_numeric:
                bic_fits.append(_bic_lm(cov_model, labels, w))
                v_values.append(float("nan"))
                lrt_values.append(float("nan"))
            else:
                bic_fits.append(_bic_multinomial(cov_model, labels, w))
                cramers_v, likelihood_ratio = _categorical_association(cov_model, labels, w)
                v_values.append(cramers_v)
                lrt_values.append(likelihood_ratio)

    bic_values = [fit.bic for fit in bic_fits]
    base_remaining = remaining[0]
    base_unaccounted = unaccounted[0]
    base_bic = bic_values[0]
    remaining_pct = [
        float(100.0 * value / base_remaining) if base_remaining != 0 else float("nan")
        for value in remaining
    ]
    remaining_improvement = [
        (
            float(100.0 * (base_remaining - value) / base_remaining)
            if base_remaining != 0
            else float("nan")
        )
        for value in remaining
    ]
    unaccounted_improvement = [
        (
            float(100.0 * (base_unaccounted - value) / base_unaccounted)
            if base_unaccounted != 0
            else float("nan")
        )
        for value in unaccounted
    ]

    row_names = ["No Clustering"] + stats_row_names
    return ClustassocResult(
        {
            "Unaccounted": unaccounted,
            "Remaining": remaining,
            "BIC": bic_values,
            "numcluster": [1] + kvals,
        },
        row_names=row_names,
        additions={
            "BICgain": [float(value - base_bic) for value in bic_values],
            "V": v_values,
            "LRT": lrt_values,
            "RemainingPctOfBaseline": remaining_pct,
            "RemainingImprovementPct": remaining_improvement,
            "UnaccountedImprovementPct": unaccounted_improvement,
            "BICLogLik": [fit.loglik for fit in bic_fits],
            "BICDf": [fit.df for fit in bic_fits],
            "BICNobs": [fit.nobs for fit in bic_fits],
            "BICConverged": [fit.converged for fit in bic_fits],
            "row_names": row_names,
        },
    )


def _clustassoc_plot_data(
    assoc: Mapping[str, Sequence[Any]],
    *,
    stat: str | Sequence[str] = ("Unaccounted", "Remaining", "BIC"),
    type: str = "b",
) -> dict[str, Any]:
    """Return exact source-driven ``plot.clustassoc`` arrays and labels."""
    selected = [stat] if isinstance(stat, str) else list(stat)
    name = str(selected[0]) if selected else ""
    if name not in assoc or name not in {"Unaccounted", "Remaining", "BIC", "numcluster"}:
        raise ValueError("undefined columns selected")
    if name == "numcluster":
        # R permits selecting this existing column even though it is not documented.
        pass
    plot_type = str(type)
    if len(plot_type) > 1:
        warnings.warn(
            f"plot type '{plot_type}' will be truncated to first character",
            RuntimeWarning,
            stacklevel=2,
        )
        plot_type = plot_type[0]
    if plot_type not in {"p", "l", "b", "o", "h", "s", "S", "n"}:
        suffix = f" '{plot_type}'" if plot_type else ""
        raise ValueError(f"invalid plot type{suffix}")
    titles = {
        "Unaccounted": "Unreproduced proportion of the association",
        "Remaining": "Unexplained association",
        "BIC": "BIC",
        "numcluster": "Number of clusters",
    }
    x = np.asarray(assoc["numcluster"], dtype=float).reshape(-1)
    y = np.asarray(assoc[name], dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("'x' and 'y' lengths differ")
    return {
        "stat": name,
        "x": x,
        "y": y,
        "xlab": "Number of clusters",
        "ylab": titles[name],
        "type": plot_type,
    }


def _r_plot_marker(value: Any) -> Any:
    if isinstance(value, (int, np.integer)):
        return {
            0: "s",
            1: "o",
            2: "^",
            3: "+",
            4: "x",
            5: "D",
            6: "v",
        }.get(int(value), "o")
    text = str(value)
    return text[0] if text else "o"


def _require_plot_bool(value: Any, *, name: str) -> bool:
    """Return an exact boolean plotting control without truthiness coercion."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def plot_clustassoc(
    assoc: Mapping[str, Sequence[Any]],
    *,
    stat: str | Sequence[str] = ("Unaccounted", "Remaining", "BIC"),
    type: str = "b",
    ax=None,
    tooltip: bool = True,
    kind: str | None = None,
    **plot_kwargs: Any,
):
    """Render the installed R 2.0 ``plot.clustassoc`` semantic mapping."""
    import matplotlib.pyplot as plt  # local import to avoid hard dep at import-time

    tooltip = _require_plot_bool(tooltip, name="tooltip")
    if kind is not None:
        kind_map = {"line": "b", "scatter": "p"}
        if kind not in kind_map:
            raise ValueError("kind must be 'line' or 'scatter'")
        if type != "b":
            raise TypeError("type and kind were both specified")
        type = kind_map[kind]
    semantic = _clustassoc_plot_data(assoc, stat=stat, type=type)
    x = semantic["x"]
    y = semantic["y"]
    plot_type = semantic["type"]

    if "xlab" in plot_kwargs:
        raise TypeError('formal argument "xlab" matched by multiple actual arguments')
    if "ylab" in plot_kwargs:
        raise TypeError('formal argument "ylab" matched by multiple actual arguments')
    main = plot_kwargs.pop("main", None)
    xlim = plot_kwargs.pop("xlim", None)
    ylim = plot_kwargs.pop("ylim", None)
    log = str(plot_kwargs.pop("log", ""))
    axes = _require_plot_bool(plot_kwargs.pop("axes", True), name="axes")
    ann = _require_plot_bool(plot_kwargs.pop("ann", True), name="ann")
    if "frame_plot" in plot_kwargs and "frame.plot" in plot_kwargs:
        raise TypeError("frame_plot was specified more than once")
    frame_plot = _require_plot_bool(
        plot_kwargs.pop("frame_plot", plot_kwargs.pop("frame.plot", True)),
        name="frame_plot",
    )

    aliases = {
        "col": "color",
        "lwd": "linewidth",
        "cex": "markersize",
    }
    for source, target in aliases.items():
        if source in plot_kwargs:
            if target in plot_kwargs:
                raise TypeError(f"{target} was specified more than once")
            value = plot_kwargs.pop(source)
            plot_kwargs[target] = 6.0 * float(value) if source == "cex" else value
    if "lty" in plot_kwargs:
        if "linestyle" in plot_kwargs:
            raise TypeError("linestyle was specified more than once")
        lty = plot_kwargs.pop("lty")
        plot_kwargs["linestyle"] = (
            {
                0: "None",
                1: "-",
                2: "--",
                3: ":",
                4: "-.",
            }.get(int(lty), "-")
            if isinstance(lty, (int, np.integer))
            else str(lty)
        )
    if "pch" in plot_kwargs:
        if "marker" in plot_kwargs:
            raise TypeError("marker was specified more than once")
        plot_kwargs["marker"] = _r_plot_marker(plot_kwargs.pop("pch"))

    allowed = {
        "alpha",
        "color",
        "label",
        "linestyle",
        "linewidth",
        "marker",
        "markersize",
        "markeredgecolor",
        "markerfacecolor",
        "zorder",
    }
    unexpected = next((key for key in plot_kwargs if key not in allowed), None)
    if unexpected is not None:
        raise TypeError(f"unexpected keyword argument '{unexpected}'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        own = True
    else:
        fig, own = ax.figure, False

    style = dict(plot_kwargs)
    artists: list[Any] = []
    if plot_type in {"b", "o"}:
        style.setdefault("marker", "o")
        artists.extend(ax.plot(x, y, **style))
    elif plot_type == "l":
        style.pop("marker", None)
        artists.extend(ax.plot(x, y, **style))
    elif plot_type == "p":
        style.setdefault("marker", "o")
        style["linestyle"] = "None"
        artists.extend(ax.plot(x, y, **style))
    elif plot_type == "h":
        color = style.pop("color", None)
        linewidth = style.pop("linewidth", 1.5)
        if style:
            unexpected = next(iter(style))
            raise TypeError(f"unexpected keyword argument '{unexpected}'")
        artists.append(ax.vlines(x, 0.0, y, color=color, linewidth=linewidth))
    elif plot_type in {"s", "S"}:
        style.pop("marker", None)
        style["drawstyle"] = "steps-post" if plot_type == "s" else "steps-pre"
        artists.extend(ax.plot(x, y, **style))

    if ann:
        ax.set_xlabel(str(semantic["xlab"]))
        ax.set_ylabel(semantic["ylab"] or "")
        if main is not None:
            ax.set_title(str(main))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if "x" in log:
        ax.set_xscale("log")
    if "y" in log:
        ax.set_yscale("log")
    if not axes:
        ax.set_axis_off()
    if not frame_plot:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if tooltip and artists:
        try:
            import mplcursors  # type: ignore

            cursor = mplcursors.cursor(artists[0], hover=True)

            @cursor.connect("add")
            def _on_add(sel):  # pragma: no cover - ui
                idx = int(sel.target.index) if hasattr(sel.target, "index") else sel.index
                k = x[idx] if idx < len(x) else None
                value = y[idx] if idx < len(y) else None
                sel.annotation.set(text=f"K={k}, {semantic['stat']}={value:.3g}")

        except ImportError:
            warnings.warn(
                "mplcursors is not installed; requested clustassoc tooltips are disabled",
                RuntimeWarning,
                stacklevel=2,
            )
    if own:
        fig.tight_layout()
    return fig, ax

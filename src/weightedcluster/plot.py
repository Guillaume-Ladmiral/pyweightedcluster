from __future__ import annotations

from numbers import Real
from statistics import NormalDist
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._validation import validate_integer_control

# For type hints without importing matplotlib at runtime
if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import matplotlib.pyplot as plt  # type: ignore

# Lazy seaborn import to avoid import-time heavy dependencies (matplotlib)
# Returns seaborn module or None


def _get_sns():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except ImportError:
        return None


def _require_plot_bool(value: object, *, name: str) -> bool:
    """Return an exact Python/NumPy boolean presentation control."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _require_finite_real(value: object, *, name: str) -> float:
    """Return a finite, non-boolean real presentation control."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _require_probability(
    value: object,
    *,
    name: str,
    endpoints: bool,
) -> float:
    """Return a finite probability with an explicit endpoint contract."""
    numeric = _require_finite_real(value, name=name)
    valid = 0.0 <= numeric <= 1.0 if endpoints else 0.0 < numeric < 1.0
    if not valid:
        raise ValueError(f"{name} must be between zero and one")
    return numeric


def _seqnull_extract_observed_vector(bcq: Dict, stat: str, kvals: List[int]) -> np.ndarray:
    cstats = bcq.get("clustrange", {}).get("stats", {})
    if isinstance(cstats, dict):
        vals = cstats.get(stat)
        if isinstance(vals, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(vals, dtype=float).reshape(-1)
            return np.asarray(
                [float(arr[i]) if i < arr.size else np.nan for i in range(len(kvals))],
                dtype=float,
            )
        if cstats and all(isinstance(v, dict) for v in cstats.values()):
            out = []
            for k in kvals:
                sv = cstats.get(int(k), cstats.get(str(int(k)), {}))
                out.append(float(sv.get(stat, np.nan)) if isinstance(sv, dict) else np.nan)
            return np.asarray(out, dtype=float)
    return np.full(len(kvals), np.nan, dtype=float)


def _seqnull_extract_null_matrix(bcq: Dict, stat: str, kvals: List[int]) -> np.ndarray:
    stats_obj = bcq.get("stats", {})
    if isinstance(stats_obj, dict) and stat in stats_obj:
        mat = np.asarray(stats_obj[stat], dtype=float)
        if mat.ndim == 1:
            if len(kvals) == 1:
                return mat.reshape(-1, 1)
            if mat.size == len(kvals):
                return mat.reshape(1, -1)
            return mat.reshape(-1, 1)
        if mat.ndim == 2:
            if mat.shape[1] == len(kvals):
                return mat
            if mat.shape[0] == len(kvals):
                return mat.T

    boot_mats = bcq.get("boot_matrices", {})
    if isinstance(boot_mats, dict) and stat in boot_mats:
        mat = np.asarray(boot_mats[stat], dtype=float)
        if mat.ndim == 1:
            if len(kvals) == 1:
                return mat.reshape(-1, 1)
            if mat.size == len(kvals):
                return mat.reshape(1, -1)
            return mat.reshape(-1, 1)
        if mat.ndim == 2:
            if mat.shape[1] == len(kvals):
                return mat
            if mat.shape[0] == len(kvals):
                return mat.T

    boot = bcq.get("boot", {})
    cols = []
    max_len = 0
    for k in kvals:
        vec = np.asarray(boot.get(int(k), {}).get(stat, []), dtype=float).reshape(-1)
        cols.append(vec)
        max_len = max(max_len, int(vec.size))
    if max_len <= 0:
        return np.empty((0, len(kvals)), dtype=float)
    mat = np.full((max_len, len(kvals)), np.nan, dtype=float)
    for j, vec in enumerate(cols):
        if vec.size > 0:
            mat[: vec.size, j] = vec
    return mat


def _seqnull_standardize_by_k(
    obs: np.ndarray, null_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    obs_n = np.asarray(obs, dtype=float).copy()
    null_n = np.asarray(null_mat, dtype=float).copy()
    if null_n.ndim != 2 or null_n.shape[1] == 0:
        return obs_n, null_n
    for j in range(null_n.shape[1]):
        col = null_n[:, j]
        mu = float(np.mean(col))
        sd = float(np.std(col, ddof=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            null_n[:, j] = (col - mu) / sd
            obs_n[j] = (obs_n[j] - mu) / sd
    return obs_n, null_n


def _normalize_clustrange_values(values: np.ndarray, norm: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if numeric.size <= 1 and norm != "none":
        return np.full_like(numeric, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if norm == "none":
            return numeric.copy()
        if norm == "range":
            return (numeric - np.min(numeric)) / (np.max(numeric) - np.min(numeric))
        if norm == "zscore":
            return (numeric - np.mean(numeric)) / np.std(numeric, ddof=1)
        if norm == "zscoremed":
            median = np.median(numeric)
            return (numeric - median) / np.median(np.abs(numeric - median))
    raise ValueError("norm must be one of 'none', 'range', 'zscore', or 'zscoremed'")


def clustrange_plot_data(
    range_obj: Dict,
    *,
    stat: str | Sequence[str] = "noCH",
    norm: str = "none",
    col: Optional[Sequence[object]] = None,
    conf_int: float = 0.9,
    ci_method: str = "none",
    ci_alpha: float = 0.3,
    line: str = "t0",
) -> Dict[str, object]:
    """Build the semantic arrays consumed by WeightedCluster ``plot.clustrange``."""
    conf_int_value = _require_probability(conf_int, name="conf_int", endpoints=False)
    ci_alpha_value = _require_probability(ci_alpha, name="ci_alpha", endpoints=True)
    kvals = np.asarray(range_obj.get("kvals", []), dtype=int)
    stats = range_obj.get("stats", {})
    all_names = [str(name) for name in stats]
    normalized = str(norm).lower()
    if normalized not in {"none", "range", "zscore", "zscoremed"}:
        raise ValueError("norm must be one of 'none', 'range', 'zscore', or 'zscoremed'")

    if isinstance(stat, str):
        if stat == "all":
            names = all_names
            if normalized == "none":
                normalized = "range"
        elif stat == "noCH":
            names = [name for name in all_names if "CH" not in name]
        else:
            names = [stat]
    else:
        requested = [str(name) for name in stat]
        names = [name for name in requested if name != "RHC"]
        if "RHC" in requested:
            names.append("RHC")
    unknown = [name for name in names if name != "RHC" and name not in stats]
    if unknown:
        raise ValueError(f"unknown statistic {unknown[0]}")
    if "RHC" in names and "HC" not in stats:
        raise ValueError("RHC requires the HC statistic")
    if not names:
        raise ValueError("at least one statistic must be selected")
    if col is not None and len(col) < len(names):
        raise ValueError("You should specify at least one color per quality measure to plot")
    ci = str(ci_method).lower()
    if ci not in {"none", "norm", "perc"}:
        raise ValueError("ci_method must be one of 'none', 'norm', or 'perc'")
    line_name = str(line).lower()
    if line_name not in {"t0", "mean", "median"}:
        raise ValueError("line must be one of 't0', 'mean', or 'median'")

    raw = np.column_stack(
        [
            (
                1.0 - np.asarray(stats["HC"], dtype=float)
                if name == "RHC"
                else np.asarray(stats[name], dtype=float)
            )
            for name in names
        ]
    )
    plot_ci = ci != "none" and range_obj.get("boot") is not None
    lower = upper = None
    values = raw.copy()
    if plot_ci:
        boot = range_obj["boot"]
        lower = np.empty_like(raw)
        upper = np.empty_like(raw)
        for stat_index, name in enumerate(names):
            source = "HC" if name == "RHC" else name
            columns = []
            for k in kvals:
                block = boot[int(k)]
                column = np.asarray(block[source], dtype=float)
                columns.append(1.0 - column if name == "RHC" else column)
            replicates = np.column_stack(columns)
            normalized_replicates = _normalize_clustrange_values(replicates, normalized)
            if line_name == "t0":
                values[:, stat_index] = normalized_replicates[0, :]
            elif line_name == "mean":
                values[:, stat_index] = np.mean(normalized_replicates, axis=0)
            else:
                values[:, stat_index] = np.median(normalized_replicates, axis=0)
            probabilities = (
                (1.0 - conf_int_value) / 2.0,
                1.0 - (1.0 - conf_int_value) / 2.0,
            )
            if ci == "norm":
                quantiles = np.asarray(
                    [NormalDist().inv_cdf(probability) for probability in probabilities],
                    dtype=float,
                )
                center = np.mean(normalized_replicates, axis=0)
                spread = np.std(normalized_replicates, axis=0, ddof=1)
                lower[:, stat_index] = center + quantiles[0] * spread
                upper[:, stat_index] = center + quantiles[1] * spread
            else:
                with np.errstate(invalid="ignore"):
                    lower[:, stat_index], upper[:, stat_index] = np.quantile(
                        normalized_replicates,
                        probabilities,
                        axis=0,
                    )
    else:
        for stat_index in range(raw.shape[1]):
            values[:, stat_index] = _normalize_clustrange_values(
                raw[:, stat_index],
                normalized,
            )

    if col is None:
        set3 = [
            "#8DD3C7",
            "#FFFFB3",
            "#BEBADA",
            "#FB8072",
            "#80B1D3",
            "#FDB462",
            "#B3DE69",
            "#FCCDE5",
            "#D9D9D9",
            "#BC80BD",
            "#CCEBC5",
            "#FFED6F",
        ]
        palette_size = max(3, min(12, len(all_names) + 1))
        base_colors = (set3[:1] + set3[2:palette_size])[: len(all_names)]
        color_map = dict(zip(all_names, base_colors, strict=True))
        if "HC" in color_map:
            color_map["RHC"] = color_map["HC"]
        selected_colors = [color_map[name] for name in names]
    else:
        selected_colors = list(col)[: len(names)]

    def _r_round(value: float) -> str:
        numeric = float(value)
        if np.isnan(numeric):
            return "NaN"
        if np.isposinf(numeric):
            return "Inf"
        if np.isneginf(numeric):
            return "-Inf"
        return format(float(np.round(numeric, 2)), "g")

    return {
        "kvals": kvals,
        "stat_names": names,
        "values": values,
        "lower": lower,
        "upper": upper,
        "norm": normalized,
        "colors": selected_colors,
        "with_intervals": bool(plot_ci),
        "ci_alpha": ci_alpha_value,
        "legend_labels": [
            f"{name} ( {_r_round(np.min(values[:, index]))} / "
            f"{_r_round(np.max(values[:, index]))} )"
            for index, name in enumerate(names)
        ],
    }


def plot_clustrange(
    range_obj: Dict,
    *,
    stat: str | Sequence[str] = "noCH",
    legendpos: str = "bottomright",
    norm: str = "none",
    withlegend: bool = True,
    lwd: float = 1,
    col: Optional[Sequence[object]] = None,
    ylab: str = "Indicators",
    xlab: str = "N clusters",
    conf_int: float = 0.9,
    ci_method: str = "none",
    ci_alpha: float = 0.3,
    line: str = "t0",
    ax: Optional[plt.Axes] = None,
):
    """Render a semantic ``plot.clustrange`` translation on Matplotlib."""
    withlegend = _require_plot_bool(withlegend, name="withlegend")
    lwd_value = _require_finite_real(lwd, name="lwd")
    if lwd_value < 0.0:
        raise ValueError("lwd must be non-negative")
    data = clustrange_plot_data(
        range_obj,
        stat=stat,
        norm=norm,
        col=col,
        conf_int=conf_int,
        ci_method=ci_method,
        ci_alpha=ci_alpha,
        line=line,
    )
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig, own_fig = ax.figure, False
    kvals = np.asarray(data["kvals"], dtype=int)
    values = np.asarray(data["values"], dtype=float)
    lower = data["lower"]
    upper = data["upper"]
    colors = data["colors"]
    for index, _name in enumerate(data["stat_names"]):
        color = None if colors is None else colors[index]
        ax.plot(
            kvals,
            values[:, index],
            color=color,
            linewidth=lwd_value,
            label=data["legend_labels"][index],
        )
        if lower is not None and upper is not None:
            ax.fill_between(
                kvals,
                np.asarray(lower)[:, index],
                np.asarray(upper)[:, index],
                color=color,
                alpha=float(data["ci_alpha"]),
                linewidth=0,
            )
    if withlegend:
        positions = {
            "bottomright": "lower right",
            "bottomleft": "lower left",
            "topright": "upper right",
            "topleft": "upper left",
            "bottom": "lower center",
            "top": "upper center",
        }
        ax.legend(loc=positions.get(str(legendpos), str(legendpos)))
    ax.set_xlabel(str(xlab))
    ax.set_ylabel(str(ylab))
    if own_fig:
        fig.tight_layout()
    return fig, ax


class BootClustrangePrintResult(pd.DataFrame):
    """Invisible typed translation of ``print.bootclustrange``'s table."""

    _metadata = ["visible"]

    @property
    def _constructor(self):
        return BootClustrangePrintResult


def print_bootclustrange(
    object: Dict,
    *,
    digits: int = 2,
    bootstat: str | Sequence[str] = "mean",
) -> BootClustrangePrintResult:
    """Return the exact table selected by R's ``print.bootclustrange``."""
    digits_value = validate_integer_control(digits, name="digits")
    if isinstance(bootstat, str):
        selection = bootstat
    else:
        choices = [str(value) for value in bootstat]
        if len(choices) != 1:
            raise ValueError(f"'length = {len(choices)}' in coercion to 'logical(1)'")
        selection = choices[0]
    if object.get("boot") is not None and selection != "t0":
        field = "meant" if selection == "mean" else "stderr"
    else:
        field = "stats"
    table = object[field]
    row_names = getattr(
        table,
        "row_names",
        [f"cluster{k}" for k in object.get("kvals", [])],
    )
    result = BootClustrangePrintResult(table, index=row_names).round(digits_value)
    result.visible = False
    return result


def _legacy_boot_distribution_plot(
    boot_obj: Dict,
    *,
    stat: str,
    kind: str,
    ax: Optional[plt.Axes],
):
    """Retain the pre-2.0 Python box/density convenience off the R path."""
    kvals = boot_obj.get("kvals", [])
    boot = boot_obj.get("boot", {})
    rows = []
    for k in kvals:
        series = boot.get(int(k), {}).get(stat, [])
        for v in series:
            rows.append({"k": int(k), stat: float(v)})
    df = pd.DataFrame(rows)
    if df.empty:
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No bootstrap data for {stat}", ha="center")
        return ax.figure, ax
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig, own_fig = ax.figure, False
    kind_l = str(kind).lower()
    if kind_l == "box":
        groups = list(df.groupby("k", sort=False))
        positions = np.arange(len(groups), dtype=float)
        ax.boxplot(
            [group[stat].to_numpy(dtype=float) for _, group in groups],
            positions=positions,
            widths=0.6,
        )
        ax.set_xticks(positions, labels=[str(k) for k, _ in groups])
    elif kind_l == "density":
        sns = _get_sns()
        if sns is not None:
            # density per K via facetting would be ideal; for simplicity, overlay kde by K
            for k, sub in df.groupby("k"):
                sns.kdeplot(sub[stat], ax=ax, label=f"K={k}")
            ax.legend()
        else:
            ax.hist(df[stat], bins=20, alpha=0.5)
    else:
        raise ValueError("kind must be one of 'line', 'box', or 'density'")
    ax.set_xlabel("Number of clusters (K)" if kind_l in ("line", "box") else stat)
    ax.set_ylabel(stat)
    ax.set_title(f"Bootstrap {stat} by K")
    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_bootclustrange(
    boot_obj: Dict,
    *,
    stat: str | Sequence[str] = "noCH",
    legendpos: str = "bottomright",
    norm: str = "none",
    withlegend: bool = True,
    lwd: float = 1,
    col: Optional[Sequence[object]] = None,
    ylab: str = "Indicators",
    xlab: str = "N clusters",
    conf_int: float = 0.95,
    ci_method: str = "perc",
    ci_alpha: float = 0.3,
    line: str = "median",
    ax: Optional[plt.Axes] = None,
    kind: Optional[str] = None,
):
    """Translate ``plot.bootclustrange`` through the clustrange plot contract.

    ``kind='box'`` and ``kind='density'`` remain Python-only compatibility
    conveniences. ``None`` and ``kind='line'`` use the installed 2.0 S3
    semantics and defaults.
    """
    withlegend = _require_plot_bool(withlegend, name="withlegend")
    lwd_value = _require_finite_real(lwd, name="lwd")
    if lwd_value < 0.0:
        raise ValueError("lwd must be non-negative")
    conf_int_value = _require_probability(conf_int, name="conf_int", endpoints=False)
    ci_alpha_value = _require_probability(ci_alpha, name="ci_alpha", endpoints=True)
    norm_value = str(norm).lower()
    if norm_value not in {"none", "range", "zscore", "zscoremed"}:
        raise ValueError("norm must be one of 'none', 'range', 'zscore', or 'zscoremed'")
    ci_method_value = str(ci_method).lower()
    if ci_method_value not in {"none", "norm", "perc"}:
        raise ValueError("ci_method must be one of 'none', 'norm', or 'perc'")
    line_value = str(line).lower()
    if line_value not in {"t0", "mean", "median"}:
        raise ValueError("line must be one of 't0', 'mean', or 'median'")
    if kind is not None and str(kind).lower() not in {"line", "box", "density"}:
        raise ValueError("kind must be one of 'line', 'box', or 'density'")
    if kind is not None and str(kind).lower() in {"box", "density"}:
        if not isinstance(stat, str):
            raise ValueError("legacy distribution plots require one statistic")
        return _legacy_boot_distribution_plot(
            boot_obj,
            stat=stat,
            kind=str(kind).lower(),
            ax=ax,
        )
    if not boot_obj.get("kvals") or not boot_obj.get("boot"):
        import matplotlib.pyplot as plt

        if ax is None:
            _figure, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No bootstrap data for {stat}", ha="center")
        return ax.figure, ax
    if kind is None and isinstance(stat, str) and stat not in {"all", "noCH"}:
        # Installed plot.clustrange drops a single selected column to a vector
        # and then calls apply(), which raises this source-level error.
        raise ValueError("dim(X) must have a positive length")

    semantic_object = boot_obj
    if not isinstance(boot_obj.get("stats"), Mapping):
        # Backward-compatible support for the historical minimal
        # ``{"kvals", "boot"}`` plotting payload.
        if not isinstance(stat, str) or stat in {"all", "noCH"}:
            raise ValueError("minimal bootstrap plot data require one statistic")
        semantic_object = dict(boot_obj)
        semantic_object["stats"] = {
            stat: [
                float(np.mean(np.asarray(boot_obj["boot"][int(k)][stat], dtype=float)))
                for k in boot_obj.get("kvals", [])
            ]
        }
    return plot_clustrange(
        semantic_object,
        stat=stat,
        legendpos=legendpos,
        norm=norm_value,
        withlegend=withlegend,
        lwd=lwd_value,
        col=col,
        ylab=ylab,
        xlab=xlab,
        conf_int=conf_int_value,
        ci_method=ci_method_value,
        ci_alpha=ci_alpha_value,
        line=line_value,
        ax=ax,
    )


def _seqnullcqi_plot_data(
    bcq: Dict,
    *,
    stat: str = "ASW",
    kind: str = "line",
    type: Optional[str] = None,
    quant: float = 0.95,
    norm: bool = False,
    legendpos: str = "topright",
    alpha: float = 0.2,
):
    """Return source-faithful semantic arrays for ``plot.seqnullcqi``."""
    norm = _require_plot_bool(norm, name="norm")
    quant_value = _require_probability(quant, name="quant", endpoints=True)
    alpha_value = _require_probability(alpha, name="alpha", endpoints=True)
    kvals = [int(k) for k in bcq.get("clustrange", {}).get("kvals", [])]
    plot_type = str(type if type is not None else kind).lower()
    if plot_type == "box":
        plot_type = "boxplot"
    if plot_type not in {"density", "boxplot", "seqdplot"}:
        # The installed method takes type[1] without match.arg; every other
        # spelling follows its ordinary line branch.
        plot_type = "line"

    if plot_type == "seqdplot":
        return {
            "type": "seqdplot",
            "panel_count": 1,
            "seqdata": np.asarray(bcq.get("seqdata", np.empty((0, 0)))).copy(),
        }

    if stat == "all":
        all_stats = [
            name
            for name in (bcq.get("clustrange", {}).get("stats", {}) or {})
            if isinstance(name, str)
        ]
        if not all_stats:
            all_stats = [name for name in (bcq.get("stats", {}) or {}) if isinstance(name, str)]
        standardized: Dict[str, np.ndarray] = {}
        for st in all_stats:
            obs = _seqnull_extract_observed_vector(bcq, st, kvals)
            null_mat = _seqnull_extract_null_matrix(bcq, st, kvals)
            combined = np.vstack([obs.reshape(1, -1), null_mat])
            with np.errstate(divide="ignore", invalid="ignore"):
                standardized[st] = (obs - np.mean(combined, axis=0)) / np.std(
                    combined, axis=0, ddof=1
                )
        return {
            "type": "line",
            "panel_count": 1,
            "kvals": np.asarray(kvals, dtype=int),
            "all_standardized": standardized,
            "labels": all_stats,
        }

    available = bcq.get("clustrange", {}).get("stats", {}) or {}
    if stat not in available:
        raise ValueError("undefined columns selected")

    obs = _seqnull_extract_observed_vector(bcq, stat, kvals)
    null_mat = _seqnull_extract_null_matrix(bcq, stat, kvals)
    standardized_obs, standardized_null = _seqnull_standardize_by_k(obs, null_mat)
    alpha_q = (1.0 - quant_value) / 2.0

    def _band(values: np.ndarray) -> np.ndarray:
        return np.quantile(values, [alpha_q, 1.0 - alpha_q], axis=0)

    def _reference(values: np.ndarray) -> float:
        row_extreme = np.min(values, axis=1) if str(stat) == "HC" else np.max(values, axis=1)
        probability = alpha_q if str(stat) == "HC" else 1.0 - alpha_q
        return float(np.quantile(row_extreme, probability))

    if plot_type == "density":
        density_obs, density_null = (
            (standardized_obs, standardized_null) if norm else (obs, null_mat)
        )
        max_t = np.max(density_null, axis=1)
        return {
            "type": "density",
            "panel_count": 1,
            "kvals": np.asarray(kvals, dtype=int),
            "observed": density_obs,
            "null": density_null,
            "max_t": max_t,
            "max_t_interval": np.quantile(max_t, [alpha_q, 1.0 - alpha_q]),
            "labels": [int(k) for k in kvals],
            "norm": norm,
        }

    return {
        "type": plot_type,
        "panel_count": 2,
        "kvals": np.asarray(kvals, dtype=int),
        "raw_observed": obs,
        "raw_null": null_mat,
        "raw_quantile_band": _band(null_mat),
        "raw_reference": _reference(null_mat),
        "standardized_observed": standardized_obs,
        "standardized_null": standardized_null,
        "standardized_quantile_band": _band(standardized_null),
        "standardized_reference": _reference(standardized_null),
        "labels": [f"Raw {stat}", f"Standardized {stat}"],
        "alpha": alpha_value,
    }


def plot_seqnullcqi(
    bcq: Dict,
    *,
    stat: str = "ASW",
    kind: str = "line",
    type: Optional[str] = None,
    quant: float = 0.95,
    norm: bool = False,
    legendpos: str = "topright",
    alpha: float = 0.2,
    ax: Optional["plt.Axes"] = None,
):
    """Render the semantic translation of WeightedCluster ``plot.seqnullcqi``."""
    import matplotlib.pyplot as plt

    data = _seqnullcqi_plot_data(
        bcq,
        stat=stat,
        kind=kind,
        type=type,
        quant=quant,
        norm=norm,
        legendpos=legendpos,
        alpha=alpha,
    )
    legend_locations = {
        "topright": "upper right",
        "topleft": "upper left",
        "bottomright": "lower right",
        "bottomleft": "lower left",
    }
    legend_loc = legend_locations.get(str(legendpos).lower(), "best")

    if data["type"] == "seqdplot":
        if ax is None:
            fig, axis = plt.subplots(figsize=(8, 5))
        else:
            fig, axis = ax.figure, ax
        sequences = np.asarray(data["seqdata"])
        if sequences.ndim == 2 and sequences.shape[0]:
            states = np.unique(sequences)
            proportions = np.column_stack([np.mean(sequences == state, axis=0) for state in states])
            bottom = np.zeros(sequences.shape[1], dtype=float)
            for index, state in enumerate(states):
                axis.bar(
                    np.arange(sequences.shape[1]),
                    proportions[:, index],
                    bottom=bottom,
                    label=f"state {state}",
                )
                bottom += proportions[:, index]
            axis.legend(loc=legend_loc)
        axis.set_xlabel("Time")
        axis.set_ylabel("Proportion")
        fig.tight_layout()
        return fig, axis

    kvals = np.asarray(data["kvals"], dtype=int)
    if stat == "all":
        if ax is None:
            fig, axis = plt.subplots(figsize=(8, 5))
        else:
            fig, axis = ax.figure, ax
        for name, values in data["all_standardized"].items():
            axis.plot(kvals, values, marker="o", label=name)
        axis.set_title("Standardized quality measure")
        axis.set_xlabel("Number of clusters")
        axis.set_ylabel("Normalized quality measure")
        axis.legend(loc=legend_loc)
        fig.tight_layout()
        return fig, axis

    if data["type"] == "density":
        if ax is None:
            fig, axis = plt.subplots(figsize=(8, 5))
        else:
            fig, axis = ax.figure, ax
        max_t = np.asarray(data["max_t"], dtype=float)
        sns = _get_sns()
        if sns is None or max_t.size < 2 or np.all(max_t == max_t[0]):
            axis.hist(max_t, bins=max(1, min(20, max_t.size)), alpha=0.5)
        else:
            sns.kdeplot(max_t, ax=axis)
        interval = np.asarray(data["max_t_interval"], dtype=float)
        axis.axvspan(interval[0], interval[1], color="lightgray", alpha=0.25)
        for kval, observed in zip(kvals, data["observed"], strict=True):
            axis.axvline(observed, color="black", linewidth=1.2)
            axis.text(observed, 0.0, str(int(kval)), rotation=90, va="bottom")
        axis.set_xlabel(stat if not norm else f"Standardized {stat}")
        axis.set_ylabel("Density")
        fig.tight_layout()
        return fig, axis

    if ax is not None:
        raise ValueError("line and boxplot seqnullcqi plots require two panels")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes_array = axes.reshape(-1)
    panels = (
        (
            data["raw_observed"],
            data["raw_null"],
            data["raw_quantile_band"],
            data["raw_reference"],
            data["labels"][0],
        ),
        (
            data["standardized_observed"],
            data["standardized_null"],
            data["standardized_quantile_band"],
            data["standardized_reference"],
            data["labels"][1],
        ),
    )
    for axis, (observed, null_values, band, reference, label) in zip(
        axes_array, panels, strict=True
    ):
        null_array = np.asarray(null_values, dtype=float)
        if data["type"] == "boxplot":
            axis.boxplot(
                [null_array[:, index] for index in range(null_array.shape[1])],
                positions=np.arange(1, len(kvals) + 1),
            )
            axis.plot(
                np.arange(1, len(kvals) + 1),
                observed,
                color="black",
                marker="o",
                linewidth=2,
            )
            axis.set_xticks(np.arange(1, len(kvals) + 1), kvals)
        else:
            axis.plot(kvals, observed, color="black", marker="o", linewidth=2)
            for row in null_array:
                axis.plot(kvals, row, color="gray", linewidth=1, alpha=float(alpha))
            band_array = np.asarray(band, dtype=float)
            axis.fill_between(
                kvals,
                band_array[0],
                band_array[1],
                color="lightgray",
                alpha=0.15,
            )
        axis.axhline(reference, linestyle=":", color="#E41A1C", linewidth=2)
        axis.set_title(label)
        axis.set_xlabel("Number of clusters")
        axis.set_ylabel(label)
    fig.tight_layout()
    return fig, axes_array


class SeqNullCQIPrintResult(pd.DataFrame):
    """Invisible, typed translation of the table returned by R's print method."""

    _metadata = ["headers", "visible"]

    @property
    def _constructor(self):
        return SeqNullCQIPrintResult


def _r_round_string(value: float, digits: int) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "NaN"
    if np.isposinf(numeric):
        return "Inf"
    if np.isneginf(numeric):
        return "-Inf"
    rounded = round(numeric, int(digits))
    if rounded == 0:
        rounded = 0.0
    rendered = format(rounded, f".{max(0, int(digits))}f")
    rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _seqnullcqi_header_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return f"<{'x'.join(str(part) for part in value.shape)}>"
    return str(value)


def print_seqnullcqi(
    bcq: Dict, *, norm: bool = True, quant: float = 0.95, digits: int = 2
) -> SeqNullCQIPrintResult:
    """Return an R-style summary table for seqnullcqi."""
    norm = _require_plot_bool(norm, name="norm")
    quant_value = _require_probability(quant, name="quant", endpoints=True)
    digits_value = validate_integer_control(digits, name="digits")
    kvals = [int(k) for k in bcq.get("clustrange", {}).get("kvals", [])]
    stats_obj = bcq.get("stats", {})
    if isinstance(stats_obj, dict):
        stat_names = list(stats_obj.keys())
    else:
        stat_names = []
    if not stat_names:
        cstats = bcq.get("clustrange", {}).get("stats", {})
        if isinstance(cstats, dict):
            stat_names = list(cstats.keys())

    stats_table = bcq.get("clustrange", {}).get("stats", {})
    row_names = getattr(stats_table, "row_names", None)
    row_idx = (
        [str(value) for value in row_names]
        if row_names and len(row_names) == len(kvals)
        else [f"cluster{k}" for k in kvals]
    )
    interval_row = f"Null Max-T {quant_value:g} interval"
    out = SeqNullCQIPrintResult("", index=row_idx + ["", interval_row], columns=stat_names)

    for st in stat_names:
        obs = _seqnull_extract_observed_vector(bcq, st, kvals)
        null_mat = _seqnull_extract_null_matrix(bcq, st, kvals)
        if norm:
            obs_use, null_use = _seqnull_standardize_by_k(obs, null_mat)
        else:
            obs_use, null_use = obs, null_mat
        for i, row_name in enumerate(row_idx):
            v = obs_use[i] if i < obs_use.size else np.nan
            out.loc[row_name, st] = _r_round_string(v, digits_value)
        alpha_q = (1.0 - quant_value) / 2.0
        max_t = np.max(null_use, axis=1)
        lo, hi = np.quantile(max_t, [alpha_q, 1.0 - alpha_q])
        out.loc[interval_row, st] = (
            f"[{_r_round_string(lo, digits_value)}; {_r_round_string(hi, digits_value)}]"
        )
    nullmodel = bcq.get("nullmodel", {})
    nullmodel_text = ", ".join(
        f"{name} = {_seqnullcqi_header_value(value)}" for name, value in nullmodel.items()
    )
    seqdist_args = bcq.get("seqdist.args", {})
    seqdist_text = ", ".join(
        f"{name}={_seqnullcqi_header_value(value)}" for name, value in seqdist_args.items()
    )
    out.headers = [
        "Parametric bootstrap cluster analysis validation",
        f"Sequence analysis null model: {nullmodel_text}",
        f"Number of bootstraps: {int(bcq.get('R', 0))}",
        (
            "Clustering method: PAM/K-Medoid"
            if bool(bcq.get("kmedoid", False))
            else f"Clustering method: hclust with {bcq.get('hclust.method', 'ward.D')}"
        ),
        f"Seqdist arguments: {seqdist_text}",
    ]
    out.visible = False
    return out


def _seqclararange_plot_data(
    sc_obj: Dict,
    *,
    stat: str,
    norm: str = "none",
) -> Dict[str, object]:
    """Translate every successful R ``plot.seqclararange`` mode to arrays."""
    normalization = str(norm)
    if normalization not in {"none", "range", "zscore", "zscoremed"}:
        raise ValueError("norm must be one of 'none', 'range', 'zscore', or 'zscoremed'")
    if not hasattr(sc_obj, "r_class") or "seqclararange" not in sc_obj.r_class:
        raise TypeError("x should be a seqclararange object")
    kvals = np.asarray(sc_obj.get("kvals", []), dtype=int)
    clara = sc_obj.get("clara", {})
    mode = str(stat)

    def _arimatrix(k: int) -> np.ndarray:
        matrix = np.asarray(clara[int(k)].get("arimatrix"), dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 2 or np.any(np.isnan(matrix)):
            raise ValueError(
                "stability evaluation is not available for this object; "
                "use stability=True in seqclararange"
            )
        return matrix

    if mode in {"CQI", "CQImv"}:
        if mode == "CQImv":
            # WeightedCluster 2.0 first drops SASW and then indexes it. Preserve
            # that source-observed failure instead of inventing a value.
            raise ValueError("subscript out of bounds: SASW")
        stats = sc_obj.get("stats", {})
        names = ["PBM", "DB", "XB", "AMS", "PBMK"]
        values = np.column_stack(
            [
                np.asarray(stats.get("PBM", []), dtype=float),
                np.asarray(stats.get("DB", []), dtype=float),
                np.asarray(stats.get("XB", []), dtype=float),
                np.asarray(stats.get("AMS", []), dtype=float),
                np.asarray(stats.get("PBM", []), dtype=float) * np.sqrt(kvals),
            ]
        )
        values = np.column_stack(
            [_normalize_clustrange_values(values[:, column], normalization) for column in range(5)]
        )
        return {
            "stat": mode,
            "kvals": kvals,
            "values": values,
            "series": names,
            "title": "Cluster Quality Indices",
            "xlab": "Number of clusters",
            "ylab": mode,
        }

    matrices = [_arimatrix(int(k)) for k in kvals]
    if mode == "ari.boxplot":
        # WeightedCluster 2.0 indexes every matrix with clara[[1]]$iteration.
        # That silently selects ARI or JC according to the first K.  Use the
        # iteration selected for each K instead; a direct installed-R capture
        # and this deliberate correction are pinned separately in parity tests.
        values = [
            (
                matrix[:, int(clara[int(k)].get("iteration", 1)) - 1]
                if int(clara[int(k)].get("iteration", 1)) <= matrix.shape[1]
                else np.asarray([], dtype=float)
            )
            for k, matrix in zip(kvals, matrices, strict=True)
        ]
        if any(value.size == 0 for value in values):
            raise ValueError("subscript out of bounds")
        return {
            "stat": mode,
            "kvals": kvals,
            "values": values,
            "series": [str(k) for k in kvals],
            "title": "",
            "xlab": "Number of clusters",
            "ylab": mode,
        }
    if mode == "stability":
        thresholds = ((0, 0.9), (0, 0.8), (0, 0.7), (1, 0.5), (1, 2 / 3), (1, 0.8))
        values = np.asarray(
            [
                [np.sum(matrix[:, column] >= threshold) for column, threshold in thresholds]
                for matrix in matrices
            ],
            dtype=float,
        )
        names = [
            "Strong Recovery(ARI > 0.9)",
            "Good Recovery (ARI > 0.8)",
            "Weak Recovery (ARI > 0.7)",
            "Minimal Recovery (JC > 0.5)",
            "Weak Recovery (JC > 0.66)",
            "Strong Recovery (JC > 0.8)",
        ]
    elif mode == "stabmean":
        rows: list[list[float]] = []
        for k, matrix in zip(kvals, matrices, strict=True):
            objectives = np.asarray(clara[int(k)]["iter.objective"], dtype=float)
            cutoff = float(np.quantile(objectives, 0.2))
            bottom = objectives <= cutoff
            rows.append(
                [
                    float(np.mean(matrix[:, 0])),
                    float(np.mean(matrix[bottom, 0])),
                    float(np.mean(matrix[:, 1])),
                    float(np.mean(matrix[bottom, 1])),
                ]
            )
        values = np.asarray(rows, dtype=float)
        names = [
            "Average ARI",
            "Trimmed average ARI (20%)",
            "Average JC",
            "Trimmed average JC (20%)",
        ]
    else:
        raise ValueError("stat must be one of CQI, CQImv, ari.boxplot, stability, or stabmean")
    return {
        "stat": mode,
        "kvals": kvals,
        "values": values,
        "series": names,
        "title": "",
        "xlab": "Number of clusters",
        "ylab": mode,
    }


def plot_seqclararange_stability(
    sc_obj: Dict,
    *,
    stat: Optional[str] = None,
    norm: str = "none",
    main: Optional[str] = None,
    xlab: str = "Number of clusters",
    ylab: Optional[str] = None,
    legendpos: str = "upper right",
    ax: Optional[Sequence[plt.Axes] | plt.Axes] = None,
):
    """Plot stability vs K from seqclararange output.

    Counts are read from ``stats`` and means are read from ``quality_stats``.
    Returns (fig, (ax1, ax2)).

    Parameters
    ----------
    sc_obj : dict
        seqclararange output object.
    ax : None | sequence of 2 matplotlib Axes
        If provided, must be `(ax1, ax2)` and both axes are reused.
    """
    normalization = str(norm)
    if normalization not in {"none", "range", "zscore", "zscoremed"}:
        raise ValueError("norm must be one of 'none', 'range', 'zscore', or 'zscoremed'")
    if stat is not None:
        data = _seqclararange_plot_data(sc_obj, stat=str(stat), norm=normalization)
        import matplotlib.pyplot as plt

        if ax is None:
            fig, target = plt.subplots(figsize=(8, 5))
            own_fig = True
        elif isinstance(ax, plt.Axes):
            fig, target, own_fig = ax.figure, ax, False
        else:
            supplied = list(ax)
            if len(supplied) != 1 or not isinstance(supplied[0], plt.Axes):
                raise ValueError("ax must contain exactly one matplotlib Axes for stat modes")
            target = supplied[0]
            fig, own_fig = target.figure, False
        if str(stat) == "ari.boxplot":
            target.boxplot(data["values"], labels=[str(k) for k in data["kvals"]])
        else:
            values = np.asarray(data["values"], dtype=float)
            for column, label in enumerate(data["series"]):
                target.plot(data["kvals"], values[:, column], marker="o", label=str(label))
            target.legend(loc=str(legendpos))
        target.set_title(str(data["title"] if main is None else main))
        target.set_xlabel(str(xlab))
        target.set_ylabel(str(data["ylab"] if ylab is None else ylab))
        if own_fig:
            fig.tight_layout()
        return fig, (target,)

    kvals = sc_obj.get("kvals", [])
    stats = sc_obj.get("stats", {})
    quality_stats = sc_obj.get("quality_stats", {})

    def _get_series(table: Dict, *, key: str):
        if isinstance(table, dict):
            vals = table.get(key, None)
            if vals is not None:
                vals_list = list(vals)
                if len(vals_list) == len(kvals):
                    return vals_list
        return [np.nan] * len(kvals)

    # Build DF with stability columns
    df = pd.DataFrame(
        {
            "k": kvals,
            "ARI>0.8": _get_series(stats, key="ARI>0.8"),
            "JC>0.8": _get_series(stats, key="JC>0.8"),
            "ARImean": _get_series(quality_stats, key="ARImean"),
            "JCmean": _get_series(quality_stats, key="JCmean"),
        }
    )
    import matplotlib.pyplot as plt

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        own_fig = True
    else:
        if isinstance(ax, plt.Axes):
            raise ValueError(
                "ax must be None or a sequence of two Axes (ax1, ax2); single axis is unsupported"
            )
        try:
            ax1, ax2 = ax  # type: ignore[misc]
        except (TypeError, ValueError) as e:
            raise ValueError("ax must be a sequence of two Axes (ax1, ax2)") from e
        if not isinstance(ax1, plt.Axes) or not isinstance(ax2, plt.Axes):
            raise ValueError("ax must contain two matplotlib Axes")
        if ax1.figure is not ax2.figure:
            raise ValueError("ax1 and ax2 must belong to the same figure")
        fig = ax1.figure
        own_fig = False
    # Counts
    sns = _get_sns()
    if sns is not None:
        sns.lineplot(data=df, x="k", y="ARI>0.8", marker="o", ax=ax1, label="ARI>0.8")
        sns.lineplot(data=df, x="k", y="JC>0.8", marker="o", ax=ax1, label="JC>0.8")
    else:
        ax1.plot(df["k"], df["ARI>0.8"], marker="o", label="ARI>0.8")
        ax1.plot(df["k"], df["JC>0.8"], marker="o", label="JC>0.8")
    ax1.set_title("Stability counts ≥ 0.8")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Count")
    ax1.legend()
    # Means
    if sns is not None:
        sns.lineplot(data=df, x="k", y="ARImean", marker="o", ax=ax2, label="ARImean")
        sns.lineplot(data=df, x="k", y="JCmean", marker="o", ax=ax2, label="JCmean")
    else:
        ax2.plot(df["k"], df["ARImean"], marker="o", label="ARImean")
        ax2.plot(df["k"], df["JCmean"], marker="o", label="JCmean")
    ax2.set_title("Stability means")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Mean")
    ax2.legend()
    if own_fig:
        fig.tight_layout()
    return fig, (ax1, ax2)


def proptree_plot_data(tree: Dict) -> dict[str, pd.DataFrame]:
    """Return renderer-independent property-tree nodes, edges, and leaves."""
    from .proptree import dtlabels

    if not isinstance(tree, dict) or not isinstance(tree.get("root"), dict):
        raise TypeError("tree must be a property-tree mapping")
    labels = dtlabels(tree)
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []

    def walk(node: dict, parent: int | None = None, side: int | None = None) -> None:
        split = node.get("split")
        info = node["info"]
        node_id = int(node["id"])
        node_rows.append(
            {
                "id": node_id,
                "parent": parent,
                "side": side,
                "depth": int(info["depth"]),
                "schedule": int(info["splitschedule"]),
                "n": float(info["n"]),
                "vardis": float(info["vardis"]),
                "medoid": int(info["medoid"]),
                "is_leaf": node.get("kids") is None,
                "split_variable": None if split is None else str(split["var"]),
                "split_type": None if split is None else str(split["type"]),
                "split_gain": np.nan if split is None else float(split["gain"]),
                "split_pvalue": (
                    np.nan
                    if split is None or split.get("pvalue") is None
                    else float(split["pvalue"])
                ),
            }
        )
        if parent is not None:
            edge_rows.append(
                {
                    "parent": int(parent),
                    "child": node_id,
                    "side": int(side),
                    "label": labels.get(str(node_id), ""),
                }
            )
        if node.get("kids") is not None:
            walk(node["kids"][0], node_id, 1)
            walk(node["kids"][1], node_id, 2)

    walk(tree["root"])
    fitted = np.asarray(tree.get("fitted", []), dtype=int).reshape(-1)
    weights = np.asarray(
        tree.get("weights", np.ones(fitted.size, dtype=float)),
        dtype=float,
    ).reshape(-1)
    leaf_rows = []
    for leaf_id in sorted(int(value) for value in np.unique(fitted)):
        mask = fitted == leaf_id
        leaf_rows.append(
            {
                "id": leaf_id,
                "label": labels.get(str(leaf_id), ""),
                "count": int(mask.sum()),
                "weight": float(weights[mask].sum()),
            }
        )
    return {
        "nodes": pd.DataFrame(node_rows),
        "edges": pd.DataFrame(edge_rows),
        "leaves": pd.DataFrame(leaf_rows),
    }


def plot_proptree_labels(tree: Dict, *, ax: Optional[plt.Axes] = None):
    """Simple display of leaf labels produced by dtlabels()."""
    leaves = proptree_plot_data(tree)["leaves"]
    items = [(str(row.id), str(row.label)) for row in leaves.itertuples()]
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        own = True
    else:
        fig, own = ax.figure, False
    ax.axis("off")
    y = 1.0
    dy = 1.0 / max(1, len(items))
    for k, v in items:
        ax.text(0.05, y, f"Leaf {k}: {v}", transform=ax.transAxes)
        y -= dy
    if own:
        fig.tight_layout()
    return fig, ax


def plot_proptree_tree(tree: Dict, *, ax: Optional[plt.Axes] = None):
    """Draw a simple binary tree diagram for a property tree.

    This is a lightweight layout that places nodes by depth and order.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        own = True
    else:
        fig, own = ax.figure, False
    ax.axis("off")

    # Collect nodes by depth
    levels: Dict[int, list] = {}
    edges: list[tuple] = []

    def walk(node, depth=0, order_list=None):
        if order_list is None:
            order_list = []
        levels.setdefault(depth, []).append(node)
        if node.get("kids") is not None:
            edges.append((node, node["kids"][0]))
            edges.append((node, node["kids"][1]))
            walk(node["kids"][0], depth + 1)
            walk(node["kids"][1], depth + 1)

    walk(tree["root"])

    # Assign positions
    y_max = max(levels.keys()) if levels else 0
    pos: Dict[int, tuple[float, float]] = {}
    for depth in range(y_max + 1):
        row = levels.get(depth, [])
        n = max(1, len(row))
        xs = np.linspace(0.1, 0.9, n)
        y = 1.0 - (depth / max(1, y_max)) * 0.9
        for i, node in enumerate(row):
            pos[id(node)] = (xs[i], y)

    # Draw edges
    for parent, child in edges:
        x1, y1 = pos[id(parent)]
        x2, y2 = pos[id(child)]
        ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1)

    # Draw nodes and labels
    for _depth, row in levels.items():
        for node in row:
            x, y = pos[id(node)]
            # Node label
            if node.get("kids") is None:
                # leaf label
                ax.text(
                    x,
                    y,
                    f"Leaf {node['id']}",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
                )
            else:
                sp = node.get("split", {})
                if sp.get("type") == "num":
                    label = f"{sp.get('var')} <= {sp.get('threshold'):.3g} (gain={sp.get('gain', 0):.3g})"
                else:
                    label = (
                        f"{sp.get('var')} in {sp.get('levels', [])} (gain={sp.get('gain', 0):.3g})"
                    )
                pv = sp.get("pvalue")
                if pv is not None:
                    label += f"\np={pv:.3g}"
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )

    if own:
        fig.tight_layout()
    return fig, ax


def plot_proptree_panel(
    tree: Dict, *, ax_tree: Optional[plt.Axes] = None, ax_bar: Optional[plt.Axes] = None
):
    """Panel plot: left a tree diagram, right a bar chart of leaf sizes.

    Returns (fig, (ax_tree, ax_bar)).
    """
    leaves = proptree_plot_data(tree)["leaves"]

    if ax_tree is None or ax_bar is None:
        import matplotlib.pyplot as plt

        fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
        own = True
    else:
        fig = ax_tree.figure
        ax_t, ax_b = ax_tree, ax_bar
        own = False

    # Left: tree
    plot_proptree_tree(tree, ax=ax_t)

    # Right: bar of leaf sizes
    lids = [str(value) for value in leaves["id"]]
    vals = [int(value) for value in leaves["count"]]
    ax_b.bar(lids, vals, color="steelblue")
    ax_b.set_title("Leaf sizes (counts)")
    ax_b.set_xlabel("Leaf ID")
    ax_b.set_ylabel("Count")
    ax_b.grid(axis="y", alpha=0.2)

    if own:
        fig.tight_layout()
    return fig, (ax_t, ax_b)


def _family_dimensions(
    nplots: int,
    *,
    rows: int | None,
    cols: int | None,
) -> tuple[int, int]:
    """Return validated panel rows and columns without lossy coercion."""
    ncols = min(nplots, 2) if cols is None else validate_integer_control(cols, name="cols")
    if ncols < 1:
        raise ValueError("rows and cols must be positive")
    nrows = (
        int(np.ceil(nplots / ncols))
        if rows is None
        else validate_integer_control(rows, name="rows")
    )
    if nrows < 1:
        raise ValueError("rows and cols must be positive")
    return nrows, ncols


def _family_layout(
    nplots: int,
    *,
    rows: int | None,
    cols: int | None,
    withlegend: bool,
    legend_prop: float | None,
) -> dict[str, object]:
    """Translate ``TraMineRInternalLayout(..., axes='all')``."""

    nrows, ncols = _family_dimensions(nplots, rows=rows, cols=cols)
    laymat = np.arange(1, nrows * ncols + 1, dtype=int).reshape(
        (nrows, ncols),
        order="C",
    )
    freecells = nrows * ncols - nplots
    heights = np.repeat(1.0 / nrows, nrows)
    widths = np.repeat(1.0 / ncols, ncols)
    legpos: str | None = None
    if withlegend:
        if freecells == 0:
            proportion = 0.15 if legend_prop is None else legend_prop
            heights = np.concatenate([np.repeat((1.0 - proportion) / nrows, nrows), [proportion]])
            laymat = np.vstack([laymat, np.repeat(nplots + 1, ncols).reshape(1, -1)])
            legpos = "bottom"
        else:
            legpos = "center"
    return {
        "laymat": laymat,
        "heights": heights,
        "widths": widths,
        "legpos": legpos,
    }


def clustrangefamily_plot_data(
    fam_obj: Dict,
    *,
    group: str = "stat",
    method: str | Sequence[str] = "all",
    pam_combine: bool = False,
    stat: str | Sequence[str] = "noCH",
    norm: str = "none",
    withlegend: bool = True,
    col: Optional[Sequence[object]] = None,
    legend_prop: float | None = None,
    rows: int | None = None,
    cols: int | None = None,
    main: str | Sequence[str] | None = None,
    xlab: str = "",
    ylab: str = "",
) -> Dict[str, object]:
    """Build semantic panel data for ``plot.clustrangefamily``."""

    pam_combine = _require_plot_bool(pam_combine, name="pam_combine")
    withlegend = _require_plot_bool(withlegend, name="withlegend")
    legend_prop_value = (
        None
        if legend_prop is None
        else _require_probability(legend_prop, name="legend_prop", endpoints=False)
    )
    group_name = str(group)
    if group_name not in {"stat", "method"}:
        raise ValueError("group must be 'stat' or 'method'")
    params = fam_obj.get("param", {})
    if "all.methods" in params:
        all_method_names = list(params.get("all.methods", []))
        original_methods = list(params.get("method", []))
    else:
        all_method_names = [name for name in fam_obj if name not in {"allstats", "param"}]
        original_methods = list(all_method_names)
    requested_methods = [method] if isinstance(method, str) else list(method)
    if len(requested_methods) == 1 and requested_methods[0] == "all":
        methods = all_method_names if pam_combine else original_methods
    elif len(requested_methods) == 1 and requested_methods[0] == "pam.combine":
        if not bool(params.get("pam.combine", False)):
            raise ValueError("you should have used pam.combine when computing the clustering")
        methods = [name for name in all_method_names if "pam" in name]
    else:
        methods = [str(value) for value in requested_methods]
    unknown_methods = [name for name in methods if name not in all_method_names]
    if unknown_methods:
        raise ValueError(f"unknown clustering method {unknown_methods[0]}")
    if not methods:
        raise ValueError("at least one clustering method must be selected")

    all_stats = list(fam_obj[all_method_names[0]]["stats"])
    requested_stats = [stat] if isinstance(stat, str) else list(stat)
    if len(requested_stats) == 1:
        if requested_stats[0] == "all":
            stats = all_stats
        elif requested_stats[0] == "noCH":
            stats = [name for name in all_stats if "CH" not in name]
        elif requested_stats[0] in (*all_stats, "RHC"):
            stats = [str(requested_stats[0])]
        else:
            raise ValueError(f"unknown statistic {requested_stats[0]}")
    else:
        stats = [str(name) for name in requested_stats if str(name) in (*all_stats, "RHC")]
    if not stats:
        raise ValueError("at least one statistic must be selected")
    normalization = str(norm)
    if normalization not in {"none", "range", "zscore", "zscoremed"}:
        raise ValueError("norm must be one of 'none', 'range', 'zscore', or 'zscoremed'")
    kvals = np.asarray(params.get("kvals", []), dtype=int)

    if group_name == "stat":
        values = np.empty((len(kvals), len(methods), len(stats)), dtype=float)
        for stat_index, stat_name in enumerate(stats):
            block = np.column_stack(
                [
                    (
                        1.0 - np.asarray(fam_obj[name]["stats"]["HC"], dtype=float)
                        if stat_name == "RHC"
                        else np.asarray(
                            fam_obj[name]["stats"][stat_name],
                            dtype=float,
                        )
                    )
                    for name in methods
                ]
            )
            for method_index in range(block.shape[1]):
                block[:, method_index] = _normalize_clustrange_values(
                    block[:, method_index],
                    normalization,
                )
            values[:, :, stat_index] = block
        if col is None:
            if len(methods) > 8:
                raise ValueError("Too many clustering methods, manually specify the colors")
            dark2 = [
                "#1B9E77",
                "#D95F02",
                "#7570B3",
                "#E7298A",
                "#66A61E",
                "#E6AB02",
                "#A6761D",
                "#666666",
            ]
            colors = dark2[: len(methods)]
        else:
            if len(col) != len(methods):
                raise ValueError("specify at least one color per quality measure to plot")
            colors = list(col)
        title_keys = stats
    else:
        values = np.empty((len(kvals), len(stats), len(methods)), dtype=float)
        for method_index, method_name in enumerate(methods):
            block = np.column_stack(
                [
                    (
                        1.0 - np.asarray(fam_obj[method_name]["stats"]["HC"], dtype=float)
                        if stat_name == "RHC"
                        else np.asarray(
                            fam_obj[method_name]["stats"][stat_name],
                            dtype=float,
                        )
                    )
                    for stat_name in stats
                ]
            )
            for stat_index in range(block.shape[1]):
                block[:, stat_index] = _normalize_clustrange_values(
                    block[:, stat_index],
                    normalization,
                )
            values[:, :, method_index] = block
        if col is None:
            set3 = [
                "#8DD3C7",
                "#FFFFB3",
                "#BEBADA",
                "#FB8072",
                "#80B1D3",
                "#FDB462",
                "#B3DE69",
                "#FCCDE5",
                "#D9D9D9",
                "#BC80BD",
                "#CCEBC5",
                "#FFED6F",
            ]
            palette_size = max(3, min(12, len(all_stats) + 1))
            base_colors = (set3[:1] + set3[2:palette_size])[: len(all_stats)]
            color_map = dict(zip(all_stats, base_colors, strict=True))
            if "HC" in color_map:
                color_map["RHC"] = color_map["HC"]
            colors = [color_map[name] for name in stats]
        else:
            if len(col) != len(stats):
                raise ValueError("specify at least one color per quality measure to plot")
            colors = list(col)
        title_keys = methods

    if main is None:
        titles = list(title_keys)
    else:
        supplied_titles = [main] if isinstance(main, str) else list(main)
        if len(supplied_titles) == 1 and len(title_keys) > 1:
            titles = [f"{supplied_titles[0]} - {key}" for key in title_keys]
        elif len(supplied_titles) != len(title_keys):
            raise ValueError("provide one title per plot")
        else:
            titles = [str(value) for value in supplied_titles]
    layout = _family_layout(
        len(title_keys),
        rows=rows,
        cols=cols,
        withlegend=withlegend,
        legend_prop=legend_prop_value,
    )
    return {
        "group": group_name,
        "methods": methods,
        "stats": stats,
        "kvals": kvals,
        "titles": titles,
        "values": values,
        "colors": colors,
        "withlegend": withlegend,
        "xlab": str(xlab),
        "ylab": str(ylab),
        **layout,
    }


def plot_clustrangefamily(
    fam_obj: Dict,
    *,
    group: str = "stat",
    method: str | Sequence[str] = "all",
    pam_combine: bool = False,
    stat: str | Sequence[str] = "noCH",
    norm: str = "none",
    withlegend: bool = True,
    lwd: float = 1,
    col: Optional[Sequence[object]] = None,
    legend_prop: float | None = None,
    rows: int | None = None,
    cols: int | None = None,
    main: str | Sequence[str] | None = None,
    xlab: str = "",
    ylab: str = "",
):
    """Render the ``clustrangefamily`` semantic panel translation."""

    lwd_value = _require_finite_real(lwd, name="lwd")
    if lwd_value < 0.0:
        raise ValueError("lwd must be non-negative")
    data = clustrangefamily_plot_data(
        fam_obj,
        group=group,
        method=method,
        pam_combine=pam_combine,
        stat=stat,
        norm=norm,
        withlegend=withlegend,
        col=col,
        legend_prop=legend_prop,
        rows=rows,
        cols=cols,
        main=main,
        xlab=xlab,
        ylab=ylab,
    )
    import matplotlib.pyplot as plt

    panel_count = len(data["titles"])
    nrows, ncols = _family_dimensions(panel_count, rows=rows, cols=cols)
    fig, axes_array = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False,
    )
    axes = list(axes_array.reshape(-1))
    values = np.asarray(data["values"], dtype=float)
    kvals = np.asarray(data["kvals"], dtype=int)
    if data["group"] == "stat":
        for panel_index, _stat_name in enumerate(data["stats"]):
            axis = axes[panel_index]
            for series_index, method_name in enumerate(data["methods"]):
                axis.plot(
                    kvals,
                    values[:, series_index, panel_index],
                    color=data["colors"][series_index],
                    linewidth=lwd_value,
                    label=method_name,
                )
            axis.set_title(data["titles"][panel_index])
    else:
        for panel_index, _method_name in enumerate(data["methods"]):
            axis = axes[panel_index]
            for series_index, stat_name in enumerate(data["stats"]):
                axis.plot(
                    kvals,
                    values[:, series_index, panel_index],
                    color=data["colors"][series_index],
                    linewidth=lwd_value,
                    label=stat_name,
                )
            axis.set_title(data["titles"][panel_index])
    for axis in axes[:panel_count]:
        axis.set_xlabel(data["xlab"])
        axis.set_ylabel(data["ylab"])
        if data["withlegend"]:
            axis.legend()
    for axis in axes[panel_count:]:
        axis.set_visible(False)
    fig.tight_layout()
    return fig, axes[:panel_count]

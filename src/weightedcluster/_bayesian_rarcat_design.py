from __future__ import annotations

import ctypes
import ctypes.util
import re
from functools import cmp_to_key, lru_cache
from typing import Any

import numpy as np
import pandas as pd
import patsy

_R_POWER_PATTERN = re.compile(r"I\(\s*([A-Za-z_]\w*)\s*\^\s*(\d+)\s*\)")
_CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_]\w*\b")
_SIMPLE_RESPONSE_PATTERN = re.compile(r"[A-Za-z_]\w*")
_PATSY_CATEGORICAL_COLUMN_PATTERN = re.compile(r"^([A-Za-z_]\w*)\[(?:T\.)?([^\]]+)\]$")


def _split_formula(formula: str) -> tuple[str, str]:
    if not isinstance(formula, str) or formula.count("~") != 1:
        raise ValueError("formula must be a two-sided formula string")
    lhs, rhs = (part.strip() for part in formula.split("~", maxsplit=1))
    if _SIMPLE_RESPONSE_PATTERN.fullmatch(lhs) is None:
        raise ValueError("formula response must be one untransformed column name")
    if not rhs:
        raise ValueError("formula must contain a right-hand side")
    calls = set(_CALL_PATTERN.findall(rhs))
    unsupported_calls = calls - {"I"}
    if unsupported_calls:
        names = ", ".join(sorted(unsupported_calls))
        raise ValueError(f"unsupported formula function(s): {names}")
    return lhs, rhs


def _translate_formula_rhs(rhs: str) -> str:
    translated = _R_POWER_PATTERN.sub(r"I(\1 ** \2)", rhs)
    if "^" in translated:
        raise ValueError("formula powers are supported only as I(column^integer)")
    return translated


def _r_expression_name(name: str) -> str:
    return re.sub(r"\s*\*\*\s*", "^", name)


def _r_level_name(value: str) -> str:
    if value == "True":
        return "TRUE"
    if value == "False":
        return "FALSE"
    return value


def _r_design_column_name(
    name: str,
    *,
    categorical_factors: frozenset[str],
) -> str:
    if name == "Intercept":
        return "(Intercept)"

    parts: list[str] = []
    for part in name.split(":"):
        match = _PATSY_CATEGORICAL_COLUMN_PATTERN.fullmatch(part)
        if match is not None and match.group(1) in categorical_factors:
            part = match.group(1) + _r_level_name(match.group(2))
        parts.append(part)
    return _r_expression_name(":".join(parts))


def _r_value_string(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "NaN"
        if np.isposinf(numeric):
            return "Inf"
        if np.isneginf(numeric):
            return "-Inf"
        if numeric == 0.0:
            return "0"
        fixed = np.format_float_positional(
            numeric,
            precision=15,
            unique=False,
            fractional=False,
            trim="-",
        )
        scientific = np.format_float_scientific(
            numeric,
            precision=14,
            unique=False,
            trim="-",
            exp_digits=2,
        ).replace(".e", "e")
        return scientific if len(scientific) < len(fixed) else fixed
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    return str(value)


@lru_cache(maxsize=1)
def _icu_collation_api() -> tuple[Any, Any, Any]:
    """Load the ICU entry points used by R's Ubuntu C.UTF-8 collation."""
    library_name = ctypes.util.find_library("icui18n")
    if library_name is None:
        raise RuntimeError("ICU is required for R-compatible C.UTF-8 collation")
    try:
        library = ctypes.CDLL(library_name)
    except OSError as error:
        raise RuntimeError(
            f"could not load ICU for R-compatible C.UTF-8 collation: {library_name!r}"
        ) from error
    version_match = re.search(r"\.so\.(\d+)", library_name)
    suffix = "" if version_match is None else "_" + version_match.group(1)

    def symbol(name: str) -> Any:
        for candidate in (name + suffix, name):
            try:
                return getattr(library, candidate)
            except AttributeError:
                continue
        raise RuntimeError(f"ICU library {library_name!r} lacks the required {name} entry point")

    open_collator = symbol("ucol_open")
    compare_utf8 = symbol("ucol_strcollUTF8")
    close_collator = symbol("ucol_close")
    open_collator.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
    open_collator.restype = ctypes.c_void_p
    compare_utf8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    compare_utf8.restype = ctypes.c_int32
    close_collator.argtypes = [ctypes.c_void_p]
    close_collator.restype = None
    return open_collator, compare_utf8, close_collator


def _r_sort_strings(values: list[str]) -> list[str]:
    """Sort strings as R 4.6 does under the supported Ubuntu C.UTF-8 locale."""
    open_collator, compare_utf8, close_collator = _icu_collation_api()
    status = ctypes.c_int32(0)
    collator = open_collator(b"", ctypes.byref(status))
    if not collator or status.value > 0:
        raise RuntimeError("ICU could not create the root collator required for R C.UTF-8 ordering")

    def compare(left: str, right: str) -> int:
        left_bytes = left.encode("utf-8")
        right_bytes = right.encode("utf-8")
        comparison_status = ctypes.c_int32(0)
        result = compare_utf8(
            collator,
            left_bytes,
            len(left_bytes),
            right_bytes,
            len(right_bytes),
            ctypes.byref(comparison_status),
        )
        if comparison_status.value > 0:
            raise RuntimeError("ICU failed while applying R C.UTF-8 collation")
        return int(result)

    try:
        return sorted(values, key=cmp_to_key(compare))
    finally:
        close_collator(collator)


def _r_character_levels(values: list[Any], *, label: str) -> list[Any]:
    rendered_to_value: dict[str, Any] = {}
    for value in values:
        rendered = _r_value_string(value)
        if rendered in rendered_to_value:
            raise ValueError(f"{label} must remain unique after R string normalization")
        rendered_to_value[rendered] = value
    return [rendered_to_value[name] for name in _r_sort_strings(list(rendered_to_value))]


def _ordered_design(
    rhs: str,
    data: pd.DataFrame,
    *,
    na_action: str = "raise",
) -> tuple[np.ndarray, patsy.DesignInfo, list[int], list[str], list[str], int]:
    """Build one design matrix in the canonical R term and column order.

    ``na_action`` defaults to ``"raise"`` so that this builder and the
    counterfactual rebuild in :func:`_build_from_design_info` agree about
    missing data: they must produce identically shaped matrices, and silently
    discarding a row would break the one-data-row-per-dissimilarity-row
    contract (finding F-16 of the 2026-08-25 audit).  The historical naive
    RARCAT solver passes ``"drop"`` deliberately, because it reproduces R's
    complete-case ``glm`` behaviour and is validated against it.
    """
    if na_action not in {"raise", "drop"}:
        raise ValueError("na_action must be 'raise' or 'drop'")
    matrix = patsy.dmatrix(
        rhs,
        data,
        NA_action=na_action,
        return_type="dataframe",
        eval_env=-1,
    )
    description = patsy.ModelDesc.from_formula("~ " + rhs)
    terms = list(description.rhs_termlist)
    intercept_terms = [term for term in terms if not term.factors]
    non_intercept_terms = [term for term in terms if term.factors]
    ordered_terms = intercept_terms + sorted(
        non_intercept_terms,
        key=lambda term: len(term.factors),
    )
    ordered_positions: list[int] = []
    term_labels: list[str] = []
    for term in ordered_terms:
        term_name = term.name()
        term_slice = matrix.design_info.term_name_slices[term_name]
        ordered_positions.extend(range(term_slice.start, term_slice.stop))
        if term.factors:
            term_labels.append(_r_expression_name(term_name))
    categorical_factors = frozenset(
        factor_info.factor.name()
        for factor_info in matrix.design_info.factor_infos.values()
        if factor_info.type == "categorical"
    )
    column_names = [
        _r_design_column_name(
            matrix.design_info.column_names[position],
            categorical_factors=categorical_factors,
        )
        for position in ordered_positions
    ]
    intercept = int(bool(intercept_terms))
    return (
        np.asarray(matrix, dtype=float)[:, ordered_positions],
        matrix.design_info,
        ordered_positions,
        column_names,
        term_labels,
        intercept,
    )


def _build_from_design_info(
    design_info: patsy.DesignInfo,
    ordered_positions: list[int],
    data: pd.DataFrame,
) -> np.ndarray:
    matrix = patsy.build_design_matrices(
        [design_info],
        data,
        NA_action="raise",
        return_type="matrix",
    )[0]
    return np.asarray(matrix, dtype=float)[:, ordered_positions]


def _base_predictors(rhs: str, data: pd.DataFrame) -> list[str]:
    identifiers = _IDENTIFIER_PATTERN.findall(rhs)
    result: list[str] = []
    for identifier in identifiers:
        if identifier in data.columns and identifier not in result:
            result.append(identifier)
    return result


def _r_design_data(rhs: str, source_data: pd.DataFrame) -> pd.DataFrame:
    """Give patsy R-compatible generated levels without changing source classes."""
    design_data = source_data.copy(deep=True)
    for variable in _base_predictors(rhs, source_data):
        series = source_data[variable]
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_numeric_dtype(series.dtype)
        ):
            continue
        observed = list(pd.unique(series.dropna()))
        if not observed:
            continue
        levels = _r_character_levels(
            observed,
            label=f"factor {variable} levels",
        )
        design_data[variable] = pd.Categorical(
            series,
            categories=levels,
            ordered=False,
        )
    return design_data

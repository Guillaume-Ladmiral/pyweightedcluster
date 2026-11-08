from __future__ import annotations

import importlib
import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np
import pandas as pd
import pytest

_HAS_RPY2 = False
_RPY2_IMPORT_ERROR = ""

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import PackageNotInstalledError, importr

    _HAS_RPY2 = True
except Exception as e:  # pragma: no cover - optional dependency
    _RPY2_IMPORT_ERROR = str(e)
    ro = None  # type: ignore[assignment]
    PackageNotInstalledError = Exception  # type: ignore[assignment]
    importr = None  # type: ignore[assignment]


# Oracle provenance is exact: every shared live-R parity test targets these two
# releases, with punctuation normalized only because R renders hyphens as dots.
#
# TraMineR moved 2.2-13 -> 2.2-14 on 2026-09-21 by maintainer decision: 2.2-14 is
# now the canonical companion oracle across the project family. traminex adopted
# it first (its own goldens were regenerated under 2.2-14), and the installed R
# library carries it, so 2.2-13 is no longer obtainable as a live target here.
EXPECTED_WC_VERSION = "2.0"
EXPECTED_TRAMINER_VERSION = "2.2-14"


def exact_oracle_version_problems(versions: dict[str, str]) -> list[str]:
    """Return exact-target mismatches, normalizing only hyphen/dot punctuation."""
    expected = {
        "WeightedCluster": EXPECTED_WC_VERSION,
        "TraMineR": EXPECTED_TRAMINER_VERSION,
    }
    problems: list[str] = []
    for package, target in expected.items():
        actual = str(versions.get(package, ""))
        if actual.replace("-", ".") != target.replace("-", "."):
            problems.append(f"{package} {actual or '<missing>'}; expected {target}")
    return problems


def is_strict_parity_mode() -> bool:
    return str(os.environ.get("WC_PARITY_STRICT", "0")).lower() in {"1", "true", "yes", "on"}


def fail_or_xfail(message: str) -> None:
    """Report an *unobtainable oracle*: xfail normally, hard-fail when strict.

    Reserved for provenance -- a live R whose version is not the pinned target,
    so the comparison cannot be run against the intended oracle at all. It must
    not be used for a value mismatch; use :func:`fail_parity` for that.
    """
    if is_strict_parity_mode():
        pytest.fail(message)
    pytest.xfail(message)


def fail_parity(message: str) -> NoReturn:
    """Report a Python-versus-R value mismatch: a failure in every mode.

    Against the pinned oracle, a disagreement is a defect, never an expected
    failure. Routing value comparators through :func:`fail_or_xfail` let a
    real divergence end an ordinary ``pytest -q`` run as "xfailed" with exit
    status 0, visible only to the strict gate (2026-09-21 targeted audit, L-2).
    """
    pytest.fail(message)


def enforce_exact_oracle_versions(versions: dict[str, str]) -> None:
    """Gate a loaded oracle: xfail normally and hard-fail in strict mode."""
    problems = exact_oracle_version_problems(versions)
    if problems:
        fail_or_xfail("Live R oracle version mismatch: " + "; ".join(problems))


def fail_or_skip(message: str, *, allow_module_level: bool = False) -> None:
    """Honest prerequisite gate: hard-fail under strict, otherwise skip.

    Mirrors ``tests/traminex_parity_utils.py``'s import gate. Distinct from
    :func:`fail_or_xfail` (which xfails individual in-body parity checks): a
    missing oracle must not be silently green, so under ``WC_PARITY_STRICT=1`` it
    fails the test rather than skipping it (design D-2 / issue I-2).
    """
    if is_strict_parity_mode():
        pytest.fail(message)
    pytest.skip(message, allow_module_level=allow_module_level)


def import_optional_or_skip(module_name: str, *, allow_module_level: bool = False) -> Any:
    """Import a test prerequisite, failing rather than skipping in strict mode."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        fail_or_skip(
            f"optional test dependency {module_name!r} is unavailable: {exc}",
            allow_module_level=allow_module_level,
        )


def canonical_partition(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    groups = []
    for c in np.unique(arr):
        groups.append(tuple(int(i) for i in np.where(arr == c)[0]))
    return tuple(sorted(groups))


@dataclass
class RParitySession:
    r: Any

    def _set_distance_matrix(self, D: np.ndarray) -> None:
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("distance matrix must be square")
        self.r.assign(
            "D",
            self.r.matrix(
                ro.FloatVector(D.ravel(order="F")),
                nrow=int(D.shape[0]),
                ncol=int(D.shape[1]),
            ),
        )

    def _set_weights(self, weights: np.ndarray | None) -> None:
        if weights is None:
            self.r("weights <- NULL")
            return
        w = np.asarray(weights, dtype=float).reshape(-1)
        self.r.assign("weights", ro.FloatVector(w))

    def _set_int_vector_or_null(self, name: str, values: np.ndarray | None) -> None:
        if values is None:
            self.r(f"{name} <- NULL")
            return
        arr = np.asarray(values, dtype=int).reshape(-1)
        self.r.assign(name, ro.IntVector(arr))

    def _named_numeric_vector(self, expr: str) -> dict[str, float]:
        names = [str(x) for x in self.r(f"names({expr})")]
        vals = np.asarray(self.r(f"as.numeric({expr})"), dtype=float).reshape(-1)
        return {str(k): float(v) for k, v in zip(names, vals, strict=False)}

    def _matrix(self, expr: str) -> np.ndarray:
        nrow = int(self.r(f"nrow({expr})")[0])
        ncol = int(self.r(f"ncol({expr})")[0])
        # rpy2 may expose R's backing memory directly.  Every contract removes
        # its temporary R objects before returning, so retain an owned NumPy
        # copy instead of a view whose contents can be reclaimed by a later
        # live-R parity call.
        vals = np.asarray(self.r(f"as.numeric({expr})"), dtype=float).reshape(-1).copy()
        return vals.reshape((nrow, ncol), order="F")

    def _assign_matrix(self, name: str, M: np.ndarray) -> None:
        M = np.asarray(M, dtype=float)
        if M.ndim != 2:
            raise ValueError("expected a 2D matrix")
        self.r.assign(
            name,
            self.r.matrix(
                ro.FloatVector(M.ravel(order="F")),
                nrow=int(M.shape[0]),
                ncol=int(M.shape[1]),
            ),
        )

    def wc_aggregate_cases_contract(
        self,
        rows: Sequence[Sequence[Any]] | Sequence[Any],
        *,
        input_kind: str = "data.frame",
        weights: Sequence[Any] | np.ndarray | None = None,
        attached_weights: Sequence[Any] | np.ndarray | None = None,
        weighted: bool = True,
        categories: dict[int, Sequence[Any]] | None = None,
    ) -> dict[str, Any]:
        """Return normalized live-R ``wcAggregateCases`` semantics."""
        token = uuid.uuid4().hex
        x_name = f"wc_phase1_x_{token}"
        weights_name = f"wc_phase1_weights_{token}"
        attached_name = f"wc_phase1_attached_{token}"
        result_name = f"wc_phase1_result_{token}"
        level_names: list[str] = []

        def _r_vector(values: Sequence[Any]):
            non_missing = [value for value in values if value is not None]
            if all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in non_missing
            ):
                return ro.FloatVector(
                    [ro.NA_Real if value is None else float(value) for value in values]
                )
            return ro.StrVector(
                [ro.NA_Character if value is None else str(value) for value in values]
            )

        try:
            if input_kind == "default":
                self.r.assign(x_name, _r_vector(list(rows)))
            else:
                raw_rows = np.asarray(rows, dtype=object)
                row_values = [list(row) for row in rows]  # type: ignore[arg-type]
                ncol = (
                    int(raw_rows.shape[1])
                    if raw_rows.ndim == 2
                    else (len(row_values[0]) if row_values else 0)
                )
                columns = {
                    f"V{column + 1}": _r_vector([row[column] for row in row_values])
                    for column in range(ncol)
                }
                frame = ro.DataFrame(columns)
                self.r.assign(x_name, frame)
                for column, levels in (categories or {}).items():
                    level_name = f"wc_phase1_levels_{column}_{token}"
                    level_names.append(level_name)
                    self.r.assign(level_name, _r_vector(list(levels)))
                    self.r(
                        f"{x_name}[[{int(column) + 1}]] <- factor({x_name}[[{int(column) + 1}]], "
                        f"levels=get('{level_name}', envir=.GlobalEnv))"
                    )
                if input_kind == "matrix":
                    self.r(f"{x_name} <- as.matrix({x_name})")
            if weights is None:
                self.r(f"{weights_name} <- NULL")
            else:
                self.r.assign(weights_name, _r_vector(np.asarray(weights).reshape(-1)))
            if attached_weights is None:
                self.r(f"{attached_name} <- NULL")
            else:
                self.r.assign(
                    attached_name,
                    _r_vector(np.asarray(attached_weights).reshape(-1)),
                )
            if input_kind == "stslist":
                self.r(
                    f"{x_name} <- suppressMessages(TraMineR::seqdef(as.matrix({x_name}), "
                    f"weights=get('{attached_name}', envir=.GlobalEnv)))"
                )
            self.r(
                f"{result_name} <- local({{ "
                f"x <- get('{x_name}', envir=.GlobalEnv); "
                f"weights <- get('{weights_name}', envir=.GlobalEnv); "
                "value <- tryCatch(WeightedCluster::wcAggregateCases(x, weights=weights, "
                f"weighted={str(bool(weighted)).upper()}), "
                "error=function(e) e); "
                "if (inherits(value, 'error')) list(ok=FALSE, error=conditionMessage(value)) "
                "else list(ok=TRUE, class=class(value), names=names(value), "
                "aggIndex=as.integer(value$aggIndex), "
                "aggWeights=as.numeric(value$aggWeights), "
                "disaggIndex=as.integer(value$disaggIndex), "
                "disaggWeights=as.numeric(value$disaggWeights), "
                "print=capture.output(print(value))) })"
            )
            ok = bool(self.r(f"{result_name}$ok")[0])
            if not ok:
                return {"ok": False, "error": str(self.r(f"{result_name}$error")[0])}
            return {
                "ok": True,
                "class": [str(value) for value in self.r(f"{result_name}$class")],
                "names": [str(value) for value in self.r(f"{result_name}$names")],
                "aggIndex_1b": np.asarray(self.r(f"{result_name}$aggIndex"), dtype=int).reshape(-1),
                "aggWeights": np.asarray(self.r(f"{result_name}$aggWeights"), dtype=float).reshape(
                    -1
                ),
                "disaggIndex_1b": np.asarray(
                    self.r(f"{result_name}$disaggIndex"), dtype=int
                ).reshape(-1),
                "disaggWeights": np.asarray(
                    self.r(f"{result_name}$disaggWeights"), dtype=float
                ).reshape(-1),
                "print_lines": [str(value) for value in self.r(f"{result_name}$print")],
            }
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(
                    f"'{name}'"
                    for name in (
                        x_name,
                        weights_name,
                        attached_name,
                        result_name,
                        *level_names,
                    )
                )
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def seqclustname_contract(
        self,
        rows: Sequence[Sequence[Any]],
        group: Sequence[Any],
        diss: np.ndarray | None,
        *,
        weights: Sequence[float] | np.ndarray | None = None,
        weighted: bool = True,
        perc: bool = False,
        group_levels: Sequence[Any] | None = None,
        omit_diss: bool = False,
    ) -> dict[str, Any]:
        """Return normalized live-R ``seqclustname`` factor semantics."""
        token = uuid.uuid4().hex
        seq_name = f"wc_phase1_seq_{token}"
        group_name = f"wc_phase1_group_{token}"
        diss_name = f"wc_phase1_diss_{token}"
        weights_name = f"wc_phase1_seqweights_{token}"
        levels_name = f"wc_phase1_grouplevels_{token}"
        result_name = f"wc_phase1_seqclustname_{token}"

        def _r_vector(values: Sequence[Any]):
            non_missing = [value for value in values if value is not None]
            if all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in non_missing
            ):
                return ro.FloatVector(
                    [ro.NA_Real if value is None else float(value) for value in values]
                )
            return ro.StrVector(
                [ro.NA_Character if value is None else str(value) for value in values]
            )

        row_values = [list(row) for row in rows]
        nrow = len(row_values)
        ncol = len(row_values[0]) if row_values else 0
        flattened = [row_values[row][column] for column in range(ncol) for row in range(nrow)]
        cleanup = (
            seq_name,
            group_name,
            diss_name,
            weights_name,
            levels_name,
            result_name,
        )
        try:
            self.r.assign(seq_name, _r_vector(flattened))
            self.r(f"{seq_name} <- matrix({seq_name}, nrow={nrow}, ncol={ncol})")
            self.r.assign(group_name, _r_vector(list(group)))
            if group_levels is not None:
                self.r.assign(levels_name, _r_vector(list(group_levels)))
                self.r(f"{group_name} <- factor({group_name}, levels={levels_name})")
            if weights is None:
                self.r(f"{weights_name} <- NULL")
            else:
                self.r.assign(
                    weights_name,
                    ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
                )
            self.r(
                f"{seq_name} <- suppressMessages(TraMineR::seqdef({seq_name}, "
                f"weights={weights_name}))"
            )
            if diss is None:
                self.r(f"{diss_name} <- NULL")
            else:
                matrix = np.asarray(diss, dtype=float)
                self.r.assign(
                    diss_name,
                    self.r.matrix(
                        ro.FloatVector(matrix.ravel(order="F")),
                        nrow=int(matrix.shape[0]),
                        ncol=int(matrix.shape[1]) if matrix.ndim == 2 else 1,
                    ),
                )
            call = (
                f"WeightedCluster::seqclustname({seq_name}, {group_name}, "
                f"weighted={str(bool(weighted)).upper()}, perc={str(bool(perc)).upper()})"
                if omit_diss
                else f"WeightedCluster::seqclustname({seq_name}, {group_name}, {diss_name}, "
                f"weighted={str(bool(weighted)).upper()}, perc={str(bool(perc)).upper()})"
            )
            self.r(
                f"{result_name} <- local({{ value <- tryCatch({call}, error=function(e) e); "
                "if (inherits(value, 'error')) list(ok=FALSE, error=conditionMessage(value)) "
                "else { ww <- if ("
                f"{str(bool(weighted)).upper()}) attr({seq_name}, 'weights') else NULL; "
                f"medoids <- TraMineR::disscenter({diss_name}, group={group_name}, "
                "medoids.index='first', allcenter=FALSE, weights=ww); "
                "list(ok=TRUE, class=class(value), levels=levels(value), "
                "values=ifelse(is.na(value), '<NA>', as.character(value)), "
                "codes=ifelse(is.na(value), -1L, as.integer(value)-1L), "
                "medoids=as.integer(unname(medoids)), medoid.names=names(medoids)) } })"
            )
            ok = bool(self.r(f"{result_name}$ok")[0])
            if not ok:
                return {"ok": False, "error": str(self.r(f"{result_name}$error")[0])}
            values = [str(value) for value in self.r(f"{result_name}$values")]
            medoid_names = (
                []
                if bool(self.r(f"is.null({result_name}$medoid.names)")[0])
                else [str(value) for value in self.r(f"{result_name}$medoid.names")]
            )
            return {
                "ok": True,
                "class": [str(value) for value in self.r(f"{result_name}$class")],
                "levels": [str(value) for value in self.r(f"{result_name}$levels")],
                "values": [None if value == "<NA>" else value for value in values],
                "codes": np.asarray(self.r(f"{result_name}$codes"), dtype=int).reshape(-1),
                "medoid_indices_1b": np.asarray(
                    self.r(f"{result_name}$medoids"), dtype=int
                ).reshape(-1),
                "medoid_names": medoid_names,
            }
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def as_seqtree_contract(
        self,
        rows: Sequence[Sequence[Any]],
        diss: np.ndarray,
        *,
        input_kind: str,
        hierarchy_diss: np.ndarray | None = None,
        predictors: dict[str, Sequence[Any]] | None = None,
        ncluster: int | None = None,
        weights: Sequence[float] | np.ndarray | None = None,
        weighted: bool = True,
        hierarchy_method: str = "average",
    ) -> dict[str, Any]:
        """Return normalized live-R ``as.seqtree`` semantics for one dispatch."""
        token = uuid.uuid4().hex
        seq_name = f"wc_phase1_tree_seq_{token}"
        diss_name = f"wc_phase1_tree_diss_{token}"
        hierarchy_name = f"wc_phase1_tree_hierarchy_{token}"
        predictor_name = f"wc_phase1_tree_predictor_{token}"
        weights_name = f"wc_phase1_tree_weights_{token}"
        tree_name = f"wc_phase1_tree_result_{token}"
        error_name = f"wc_phase1_tree_error_{token}"
        counter_name = f"wc_phase1_tree_counter_{token}"
        counter_env_name = f"wc_phase1_tree_counter_env_{token}"
        method_name = f"wc_phase1_tree_method_{token}"
        row_values = [list(row) for row in rows]
        nrow = len(row_values)
        ncol = len(row_values[0]) if row_values else 0
        flattened = [row_values[row][column] for column in range(ncol) for row in range(nrow)]
        cleanup = [
            seq_name,
            diss_name,
            hierarchy_name,
            predictor_name,
            weights_name,
            tree_name,
            error_name,
            counter_name,
            counter_env_name,
            method_name,
            "tree_nodes",
            "tree_labels",
            "tree_max_k",
            "tree_cut_ids",
            "tree_cut_labels",
            "tree_adjustment",
            "tree_members",
            "tree_split_prob",
            "tree_split_index",
            "tree_split_labels",
        ]
        try:
            self.r.assign(seq_name, ro.StrVector([str(value) for value in flattened]))
            self.r(f"{seq_name} <- matrix({seq_name}, nrow={nrow}, ncol={ncol})")
            if weights is None:
                self.r(f"{weights_name} <- NULL")
            else:
                self.r.assign(
                    weights_name,
                    ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
                )
            self.r(
                f"{seq_name} <- suppressMessages(TraMineR::seqdef({seq_name}, "
                f"weights={weights_name}))"
            )
            D = np.asarray(diss, dtype=float)
            self.r.assign(
                diss_name,
                self.r.matrix(
                    ro.FloatVector(D.ravel(order="F")),
                    nrow=int(D.shape[0]),
                    ncol=int(D.shape[1]) if D.ndim == 2 else 1,
                ),
            )
            if input_kind in {"hclust", "twins"}:
                H = D if hierarchy_diss is None else np.asarray(hierarchy_diss, dtype=float)
                self.r.assign(
                    hierarchy_name,
                    self.r.matrix(
                        ro.FloatVector(H.ravel(order="F")),
                        nrow=int(H.shape[0]),
                        ncol=int(H.shape[1]),
                    ),
                )
                self.r.assign(method_name, ro.StrVector([hierarchy_method]))
                if input_kind == "hclust":
                    self.r(
                        f"{predictor_name} <- hclust(as.dist({hierarchy_name}), "
                        f"method=as.character({method_name}))"
                    )
                else:
                    self.r(
                        f"{predictor_name} <- cluster::agnes(as.dist({hierarchy_name}), "
                        f"diss=TRUE, method=as.character({method_name}))"
                    )
            elif input_kind == "default":
                self.r(f"{predictor_name} <- data.frame(row.names=seq_len({nrow}))")
                for column_index, (column, values) in enumerate((predictors or {}).items()):
                    value_name = f"wc_phase1_tree_column_{column_index}_{token}"
                    label_name = f"wc_phase1_tree_label_{column_index}_{token}"
                    cleanup.extend([value_name, label_name])
                    self.r.assign(
                        value_name,
                        ro.StrVector([str(value) for value in values]),
                    )
                    self.r.assign(label_name, ro.StrVector([str(column)]))
                    self.r(
                        f"{predictor_name}[[as.character({label_name})]] <- "
                        f"factor({value_name})"
                    )
            else:
                raise ValueError(f"unknown as.seqtree input_kind: {input_kind}")

            self.r(
                f"{counter_env_name} <- getFromNamespace('.localstuffDissTree', 'TraMineR'); "
                f"{counter_name} <- {counter_env_name}$DTNnodeCounter; "
                f"{counter_env_name}$DTNnodeCounter <- as.integer(1)"
            )
            args = (
                f"list(object={predictor_name}, seqdata={seq_name}, diss={diss_name}, "
                f"weighted={str(bool(weighted)).upper()})"
            )
            if input_kind in {"hclust", "twins"} and ncluster is not None:
                args = args[:-1] + f", ncluster=as.integer({int(ncluster)}))"
            self.r(
                f"{error_name} <- NULL; {tree_name} <- tryCatch("
                f"do.call(WeightedCluster::as.seqtree, {args}), "
                f"error=function(e) {{ {error_name} <<- conditionMessage(e); NULL }})"
            )
            if bool(self.r(f"is.null({tree_name})")[0]):
                return {"ok": False, "error": str(self.r(error_name)[0])}

            contract = self._tree_contract(tree_name)
            data_columns = [str(value) for value in self.r(f"names({tree_name}$data)")]
            data_levels = [
                [str(value) for value in self.r(f"levels({tree_name}$data[[{index}]])")]
                for index in range(1, len(data_columns) + 1)
            ]
            contract.update(
                {
                    "ok": True,
                    "data_columns": data_columns,
                    "data_levels": data_levels,
                    "data_codes": self._matrix(
                        f"vapply({tree_name}$data, as.integer, integer(nrow({tree_name}$data)))"
                    ),
                    "weights": np.asarray(
                        self.r(f"as.numeric({tree_name}$weights)"), dtype=float
                    ).reshape(-1),
                    "info_method": str(self.r(f"{tree_name}$info$method")[0]),
                    "info_n": float(self.r(f"{tree_name}$info$n")[0]),
                    "parameters": {
                        "minSize": float(self.r(f"{tree_name}$info$parameters$minSize")[0]),
                        "maxdepth": int(self.r(f"{tree_name}$info$parameters$maxdepth")[0]),
                        "R": int(self.r(f"{tree_name}$info$parameters$R")[0]),
                        "pval": float(self.r(f"{tree_name}$info$parameters$pval")[0]),
                    },
                    "weight_permutation": str(self.r(f"{tree_name}$info$weight.permutation")[0]),
                    "terms_is_null": bool(self.r(f"is.null({tree_name}$terms)")[0]),
                }
            )
            return contract
        finally:
            if bool(self.r("exists('" + counter_name + "', envir=.GlobalEnv, inherits=FALSE)")[0]):
                self.r(f"{counter_env_name}$DTNnodeCounter <- as.integer({counter_name})")
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def fuzzyseqplot_contract(
        self,
        rows: Sequence[Sequence[Any]],
        membership: np.ndarray | None,
        *,
        group_kind: str = "matrix",
        group_names: Sequence[Any] | None = None,
        factor_group: Sequence[Any] | None = None,
        weights: Sequence[float] | np.ndarray | None = None,
        membership_threshold: float = 0.0,
        type: str = "i",
        members_weighted: bool = True,
        memb_exp: float = 1.0,
        sortv: Any = None,
        plot_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record exact live-R ``fuzzyseqplot`` delegation without graphics."""
        token = uuid.uuid4().hex
        seq_name = f"wc_phase1_fuzzy_seq_{token}"
        weights_name = f"wc_phase1_fuzzy_weights_{token}"
        group_name = f"wc_phase1_fuzzy_group_{token}"
        names_name = f"wc_phase1_fuzzy_names_{token}"
        kwargs_name = f"wc_phase1_fuzzy_kwargs_{token}"
        result_name = f"wc_phase1_fuzzy_result_{token}"
        row_values = [list(row) for row in rows]
        nrow = len(row_values)
        ncol = len(row_values[0]) if row_values else 0
        flattened = [row_values[row][column] for column in range(ncol) for row in range(nrow)]
        cleanup = (
            seq_name,
            weights_name,
            group_name,
            names_name,
            kwargs_name,
            result_name,
        )

        def _r_atomic(value: Any):
            values = (
                list(value)
                if isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, str)
                else [value]
            )
            if all(isinstance(item, (bool, np.bool_)) for item in values):
                return ro.BoolVector([bool(item) for item in values])
            if all(
                isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))
                for item in values
            ):
                return ro.IntVector([int(item) for item in values])
            if all(isinstance(item, (int, float, np.integer, np.floating)) for item in values):
                return ro.FloatVector([float(item) for item in values])
            return ro.StrVector([str(item) for item in values])

        def _atomic_value(expr: str) -> Any:
            kind = str(self.r(f"typeof({expr})")[0])
            values = self.r(expr)
            if kind == "logical":
                normalized = [bool(value) for value in values]
            elif kind == "integer":
                normalized = [int(value) for value in values]
            elif kind == "double":
                normalized = [float(value) for value in values]
            else:
                normalized = [str(value) for value in values]
            return normalized[0] if len(normalized) == 1 else normalized

        try:
            self.r.assign(seq_name, ro.StrVector([str(value) for value in flattened]))
            self.r(f"{seq_name} <- matrix({seq_name}, nrow={nrow}, ncol={ncol})")
            if weights is None:
                self.r(f"{weights_name} <- NULL")
            else:
                self.r.assign(
                    weights_name,
                    ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
                )
            self.r(
                f"{seq_name} <- suppressMessages(TraMineR::seqdef({seq_name}, "
                f"weights={weights_name}))"
            )

            if group_kind in {"matrix", "fanny"}:
                if membership is None:
                    raise ValueError("membership is required for matrix and fanny contracts")
                matrix = np.asarray(membership, dtype=float)
                self.r.assign(
                    group_name,
                    self.r.matrix(
                        ro.FloatVector(matrix.ravel(order="F")),
                        nrow=int(matrix.shape[0]),
                        ncol=int(matrix.shape[1]),
                    ),
                )
                if group_names is not None:
                    self.r.assign(names_name, ro.StrVector([str(value) for value in group_names]))
                    self.r(f"colnames({group_name}) <- {names_name}")
                if group_kind == "fanny":
                    self.r(
                        f"{group_name} <- structure(list(membership={group_name}), class='fanny')"
                    )
            elif group_kind == "factor":
                self.r.assign(
                    group_name,
                    ro.StrVector([str(value) for value in (factor_group or [])]),
                )
                self.r(f"{group_name} <- factor({group_name})")
            elif group_kind == "none":
                self.r(f"{group_name} <- NULL")
            elif group_kind == "vector":
                self.r.assign(
                    group_name,
                    ro.FloatVector(np.asarray(membership, dtype=float).reshape(-1)),
                )
            else:
                raise ValueError(f"unknown fuzzyseqplot group_kind: {group_kind}")

            forwarded = dict(plot_kwargs or {})
            if sortv is not None:
                forwarded["sortv"] = sortv
            self.r.assign(
                kwargs_name,
                ro.ListVector([(str(name), _r_atomic(value)) for name, value in forwarded.items()]),
            )
            self.r(
                f"{result_name} <- local({{ "
                "calls <- list(); warnings <- character(); error <- NULL; "
                "parent <- environment(WeightedCluster::fuzzyseqplot); "
                "child <- new.env(parent=parent); "
                "child$calls <- list(); "
                "child$seqplot <- eval(quote(function(seqdata, group=NULL, ...) { "
                "dots <- list(...); calls[[length(calls)+1L]] <<- list("
                "rows=as.matrix(seqdata), weights=attr(seqdata, 'weights'), "
                "group=group, dots=dots); structure(list(spy=TRUE), class='fuzzyseqplot_spy') "
                "}), envir=child); "
                "fun <- WeightedCluster::fuzzyseqplot; environment(fun) <- child; "
                f"args <- c(list(seqdata={seq_name}, group={group_name}, "
                f"membership.threashold={float(membership_threshold)!r}, "
                f"type={type!r}, members.weighted={str(bool(members_weighted)).upper()}, "
                f"memb.exp={float(memb_exp)!r}), {kwargs_name}); "
                "value <- tryCatch(withCallingHandlers(do.call(fun, args), "
                "warning=function(w) warnings <<- c(warnings, conditionMessage(w))), "
                "error=function(e) { error <<- conditionMessage(e); NULL }); "
                "calls <- child$calls; "
                "call <- if (length(calls)) calls[[length(calls)]] else NULL; "
                "list(call_count=length(calls), call=call, warnings=warnings, error=error, "
                "returned=!is.null(value), return_class=if (is.null(value)) character() "
                "else class(value)) })"
            )

            call_count = int(self.r(f"{result_name}$call_count")[0])
            error_is_null = bool(self.r(f"is.null({result_name}$error)")[0])
            contract: dict[str, Any] = {
                "call_count": call_count,
                "warnings": [str(value) for value in self.r(f"{result_name}$warnings")],
                "error": None if error_is_null else str(self.r(f"{result_name}$error")[0]),
                "returned": bool(self.r(f"{result_name}$returned")[0]),
                "return_class": [str(value) for value in self.r(f"{result_name}$return_class")],
            }
            if call_count == 0:
                contract.update(
                    {
                        "rows": np.empty((0, ncol), dtype=object),
                        "weights": np.empty(0, dtype=float),
                        "groups": [],
                        "type": None,
                        "sortv": None,
                        "forwarded_kwargs": {},
                    }
                )
                return contract

            call_expr = f"{result_name}$call"
            call_nrow = int(self.r(f"nrow({call_expr}$rows)")[0])
            call_ncol = int(self.r(f"ncol({call_expr}$rows)")[0])
            raw_rows = [str(value) for value in self.r(f"as.character({call_expr}$rows)")]
            dots_names = (
                []
                if bool(self.r(f"is.null(names({call_expr}$dots))")[0])
                else [str(value) for value in self.r(f"names({call_expr}$dots)")]
            )
            forwarded_kwargs = {
                name: _atomic_value(f"{call_expr}$dots[[{name!r}]]")
                for name in dots_names
                if name not in {"type", "sortv"}
            }
            contract.update(
                {
                    "rows": np.asarray(raw_rows, dtype=object).reshape(
                        (call_nrow, call_ncol), order="F"
                    ),
                    "weights": (
                        np.empty(0, dtype=float)
                        if bool(self.r(f"is.null({call_expr}$weights)")[0])
                        else np.asarray(self.r(f"as.numeric({call_expr}$weights)"), dtype=float)
                    ),
                    "groups": (
                        []
                        if bool(self.r(f"is.null({call_expr}$group)")[0])
                        else [str(value) for value in self.r(f"as.character({call_expr}$group)")]
                    ),
                    "type": (
                        _atomic_value(f"{call_expr}$dots[['type']]")
                        if "type" in dots_names
                        else None
                    ),
                    "sortv": (
                        _atomic_value(f"{call_expr}$dots[['sortv']]")
                        if "sortv" in dots_names
                        else None
                    ),
                    "forwarded_kwargs": forwarded_kwargs,
                }
            )
            return contract
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def r_internal_exists(self, name: str) -> bool:
        self.r.assign("nm", ro.StrVector([str(name)]))
        return bool(self.r('exists(nm, where = getNamespace("WeightedCluster"))')[0])

    def r_crispness(self, U: np.ndarray, *, norm: bool = True) -> np.ndarray:
        self._assign_matrix("U", U)
        self.r.assign("norm", ro.BoolVector([bool(norm)]))
        vals = self.r("WeightedCluster:::crispness(U, norm = as.logical(norm))")
        return np.asarray(vals, dtype=float).reshape(-1)

    def r_wfcmdd(
        self,
        D: np.ndarray,
        memb: np.ndarray,
        *,
        method: str = "FCMdd",
        m: float = 1.5,
        weights: np.ndarray | None = None,
        dnoise: float | None = None,
    ) -> dict[str, Any]:
        self._set_distance_matrix(D)
        self._assign_matrix("memb", memb)
        self._set_weights(weights)
        self.r.assign("method", ro.StrVector([str(method)]))
        self.r.assign("m", ro.FloatVector([float(m)]))
        if dnoise is None:
            self.r("dnoise <- NULL")
        else:
            self.r.assign("dnoise", ro.FloatVector([float(dnoise)]))
        self.r(
            "res <- WeightedCluster:::wfcmdd(D, memb = memb, weights = weights, "
            "method = as.character(method), m = m, dnoise = dnoise)"
        )
        return {
            "memb": self._matrix("res$memb"),
            "mobile_centers_1b": np.asarray(
                self.r("as.integer(res$mobileCenters)"), dtype=int
            ).reshape(-1),
            "functional": float(np.asarray(self.r("as.numeric(res$functional)"))[0]),
        }

    def r_seqclararange_stats_contract(
        self, X: np.ndarray, *, kvals, r_iter: int = 5, sample_size: int = 6, seed: int = 0
    ) -> tuple[list[str], np.ndarray]:
        Xi = np.asarray(X, dtype=int)
        self._assign_matrix("Xmat", Xi.astype(float))
        self.r.assign("kvals", ro.IntVector([int(k) for k in kvals]))
        self.r.assign("Riter", ro.IntVector([int(r_iter)]))
        self.r.assign("ssize", ro.IntVector([int(sample_size)]))
        self.r("suppressMessages(library(WeightedCluster))")
        self.r("storage.mode(Xmat) <- 'integer'")
        # Seed R's global RNG: seqclararange draws random subsamples, so without a
        # seed this driver depends on whatever RNG state earlier tests left in the
        # shared rpy2 session (order-dependent flakiness). The dataset/sample_size
        # the caller supplies must also keep every subsample large enough that an
        # internal cutree(., k=max(kvals)) is feasible.
        self.r(f"set.seed({int(seed)})")
        self.r("sd <- suppressMessages(TraMineR::seqdef(Xmat))")
        self.r(
            "cr <- suppressMessages(seqclararange(sd, R = as.integer(Riter), "
            "sample.size = as.integer(ssize), kvals = as.integer(kvals), "
            "seqdist.args = list(method = 'LCS'), method = 'crisp', "
            "criteria = 'distance'))"
        )
        cols = [str(x) for x in self.r("colnames(cr$stats)")]
        kv = np.asarray(self.r("as.integer(cr$kvals)"), dtype=int).reshape(-1)
        return cols, kv

    def seqclararange_complete_contract(
        self,
        X: np.ndarray,
        *,
        kvals: Sequence[int],
        R: int = 2,
        sample_size: int | None = None,
        method: str = "crisp",
        criteria: Sequence[str] | str = ("distance",),
        stability: bool = False,
        dnoise: float | None = None,
        m: float = 1.5,
        keep_diss: bool = False,
        max_dist: float | None = None,
        weights: Sequence[float] | np.ndarray | None = None,
        replay_samples_1b: Sequence[Sequence[int]] | np.ndarray | None = None,
        seqdist_method: str = "LCS",
        seqdist_norm: str = "none",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Execute installed ``seqclararange`` with exact sample replay.

        The installed function itself remains the oracle. Only base
        ``sample.int``'s probability-bearing CLARA draw is replaced inside a
        protected local scope; future's RNG bookkeeping continues through the
        original base function. The same R Ward/fanny producers are captured as
        replayable initial memberships for Python.
        """
        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_seqclara_{key}_{token}"
            for key in (
                "X",
                "kvals",
                "criteria",
                "weights",
                "samples",
                "contract",
            )
        }
        Xi = np.asarray(X, dtype=int)
        self._assign_matrix(names["X"], Xi.astype(float))
        self.r.assign(names["kvals"], ro.IntVector([int(value) for value in kvals]))
        criteria_values = [criteria] if isinstance(criteria, str) else list(criteria)
        self.r.assign(
            names["criteria"],
            ro.StrVector([str(value) for value in criteria_values]),
        )
        if weights is None:
            self.r(f"{names['weights']} <- NULL")
        else:
            self.r.assign(
                names["weights"],
                ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
            )
        replay = None
        if replay_samples_1b is not None:
            replay = np.asarray(replay_samples_1b, dtype=int)
            if replay.ndim != 2 or replay.shape[0] != int(R):
                raise ValueError("replay_samples_1b must have shape (R, sample_size)")
            if sample_size is not None and replay.shape[1] != int(sample_size):
                raise ValueError("replay sample width must equal sample_size")
            sample_size = int(replay.shape[1])
            self._assign_matrix(names["samples"], replay)
        else:
            self.r(f"{names['samples']} <- NULL")
        sample_expr = (
            "40L + 2L * max(as.integer(kvals_local))"
            if sample_size is None
            else f"{int(sample_size)}L"
        )
        dnoise_expr = "NULL" if dnoise is None else repr(float(dnoise))
        max_dist_expr = "NULL" if max_dist is None else repr(float(max_dist))
        cleanup = list(names.values())
        try:
            self.r(f"""
                {names['contract']} <- local({{
                  suppressMessages(library(WeightedCluster))
                  suppressMessages(library(TraMineR))
                  X_local <- get('{names['X']}', envir=.GlobalEnv)
                  storage.mode(X_local) <- 'integer'
                  kvals_local <- get('{names['kvals']}', envir=.GlobalEnv)
                  criteria_local <- get('{names['criteria']}', envir=.GlobalEnv)
                  weights_local <- get('{names['weights']}', envir=.GlobalEnv)
                  samples_local <- get('{names['samples']}', envir=.GlobalEnv)
                  sd_local <- suppressMessages(TraMineR::seqdef(X_local))
                  if (!is.null(weights_local)) attr(sd_local, 'weights') <- weights_local
                  ac_local <- wcAggregateCases(sd_local)
                  ag_local <- sd_local[ac_local$aggIndex, ]
                  sample_size_local <- {sample_expr}

                  init_local <- setNames(vector('list', length(kvals_local)),
                                         as.character(kvals_local))
                  if (!is.null(samples_local)) {{
                    for (ki in seq_along(kvals_local)) {{
                      init_local[[ki]] <- vector('list', {int(R)}L)
                    }}
                    for (loop in seq_len({int(R)}L)) {{
                      mysample <- as.integer(samples_local[loop, ])
                      ac2 <- wcAggregateCases(data.frame(id=mysample))
                      data_subset <- ag_local[mysample[ac2$aggIndex], ]
                      diss <- suppressMessages(TraMineR::seqdist(
                        data_subset, method='{str(seqdist_method)}',
                        norm='{str(seqdist_norm)}'
                      ))
                      hc <- fastcluster::hclust(
                        as.dist(diss), method='ward.D', members=ac2$aggWeights
                      )
                      for (ki in seq_along(kvals_local)) {{
                        ward <- vegclust::as.memb(
                          stats::cutree(hc, k=kvals_local[ki])
                        )
                        if ('{str(method)}' %in% c('fuzzy', 'noise')) {{
                          fanny_fit <- cluster::fanny(
                            diss, kvals_local[ki], diss=TRUE, memb.exp={float(m)},
                            iniMem.p=ward, tol=0.00001
                          )
                          fanny_membership <- as.matrix(fanny_fit$membership)
                        }} else {{
                          fanny_membership <- as.matrix(ward)
                        }}
                        init_local[[ki]][[loop]] <- list(
                          ward=as.matrix(ward),
                          fanny=fanny_membership,
                          sample.unique=as.integer(mysample[ac2$aggIndex]),
                          sample.weights=as.numeric(ac2$aggWeights)
                        )
                      }}
                    }}
                  }}

                  state <- new.env(parent=emptyenv())
                  state$i <- 0L
                  state$error <- NULL
                  original_sample_int <- base::sample.int
                  old_plan <- future::plan(future::sequential)
                  on.exit(future::plan(old_plan), add=TRUE)
                  patched <- !is.null(samples_local)
                  if (patched) {{
                    unlockBinding('sample.int', baseenv())
                    assign('sample.int', function(n, size, replace=FALSE, prob=NULL, ...) {{
                      if (!is.null(prob)) {{
                        state$i <- state$i + 1L
                        if (state$i > nrow(samples_local)) stop('replay plan exhausted')
                        return(as.integer(samples_local[state$i, seq_len(size)]))
                      }}
                      original_sample_int(n, size, replace, prob, ...)
                    }}, envir=baseenv())
                    lockBinding('sample.int', baseenv())
                  }}
                  set.seed({int(seed)})
                  value <- tryCatch(
                    suppressMessages(WeightedCluster::seqclararange(
                      sd_local,
                      R={int(R)}L,
                      sample.size=sample_size_local,
                      kvals=as.integer(kvals_local),
                      seqdist.args=list(
                        method='{str(seqdist_method)}',
                        norm='{str(seqdist_norm)}'
                      ),
                      method='{str(method)}',
                      m={float(m)},
                      criteria=as.character(criteria_local),
                      stability={str(bool(stability)).upper()},
                      dnoise={dnoise_expr},
                      parallel=FALSE,
                      progressbar=FALSE,
                      keep.diss={str(bool(keep_diss)).upper()},
                      max.dist={max_dist_expr}
                    )),
                    error=function(e) {{
                      state$error <- conditionMessage(e)
                      NULL
                    }},
                    finally={{
                      if (patched) {{
                        unlockBinding('sample.int', baseenv())
                        assign('sample.int', original_sample_int, envir=baseenv())
                        lockBinding('sample.int', baseenv())
                      }}
                    }}
                  )
                  list(
                    value=value,
                    error=state$error,
                    replay.consumed=state$i,
                    init=init_local,
                    aggregation=ac_local,
                    probs=as.numeric(ac_local$aggWeights / nrow(sd_local)),
                    sample.size=sample_size_local
                  )
                }})
                """)
            contract_name = names["contract"]
            error = (
                None
                if bool(self.r(f"is.null({contract_name}$error)")[0])
                else str(self.r(f"as.character({contract_name}$error)")[0])
            )
            base: dict[str, Any] = {
                "error": error,
                "replay_consumed": int(
                    np.asarray(self.r(f"as.integer({contract_name}$replay.consumed)"))[0]
                ),
                "sample_size": int(
                    np.asarray(self.r(f"as.integer({contract_name}$sample.size)"))[0]
                ),
                "agg_index_0b": np.asarray(
                    self.r(f"as.integer({contract_name}$aggregation$aggIndex)"),
                    dtype=int,
                )
                - 1,
                "agg_weights": np.asarray(
                    self.r(f"as.numeric({contract_name}$aggregation$aggWeights)"),
                    dtype=float,
                ),
                "disagg_index_0b": np.asarray(
                    self.r(f"as.integer({contract_name}$aggregation$disaggIndex)"),
                    dtype=int,
                )
                - 1,
                "probs": np.asarray(
                    self.r(f"as.numeric({contract_name}$probs)"),
                    dtype=float,
                ),
                "initial_memberships": {},
            }
            for ki, kval in enumerate(kvals, start=1):
                plans: list[dict[str, np.ndarray]] = []
                if replay is not None:
                    for iteration in range(1, int(R) + 1):
                        prefix = f"{contract_name}$init[[{ki}]][[{iteration}]]"
                        plans.append(
                            {
                                "ward": self._matrix(f"{prefix}$ward"),
                                "fanny": self._matrix(f"{prefix}$fanny"),
                                "sample_unique_indices_0b": np.asarray(
                                    self.r(f"as.integer({prefix}$sample.unique)"),
                                    dtype=int,
                                )
                                - 1,
                                "sample_weights": np.asarray(
                                    self.r(f"as.numeric({prefix}$sample.weights)"),
                                    dtype=float,
                                ),
                            }
                        )
                base["initial_memberships"][int(kval)] = plans
            if error is not None:
                return base

            value_expr = f"{contract_name}$value"
            family_requested = len(criteria_values) > 1
            is_family = bool(self.r(f"inherits({value_expr}, 'seqclarafamily')")[0])
            range_expr = (
                f"{value_expr}[[{repr(str(criteria_values[0]))}]]" if is_family else value_expr
            )
            base["family_requested"] = family_requested
            base["family_source_bug"] = bool(family_requested and not is_family)
            base.update(
                {
                    "class": [str(value) for value in self.r(f"class({range_expr})")],
                    "names": [str(value) for value in self.r(f"names({range_expr})")],
                    "kvals": np.asarray(
                        self.r(f"as.integer({range_expr}$kvals)"),
                        dtype=int,
                    ),
                    "stats": self._matrix(f"as.matrix({range_expr}$stats)"),
                    "stats_row_names": [
                        str(value) for value in self.r(f"rownames({range_expr}$stats)")
                    ],
                    "stats_column_names": [
                        str(value) for value in self.r(f"colnames({range_expr}$stats)")
                    ],
                    "clustering": {},
                    "clara": {},
                    "plot_errors": {},
                }
            )
            for index, kval in enumerate(kvals, start=1):
                clara_expr = f"{range_expr}$clara[[{index}]]"
                arimatrix = np.asarray(self.r(f"{clara_expr}$arimatrix"), dtype=float)
                if bool(self.r(f"is.matrix({clara_expr}$arimatrix)")[0]):
                    arimatrix = self._matrix(f"{clara_expr}$arimatrix")
                clustering = np.asarray(self.r(f"{clara_expr}$clustering"), dtype=float)
                if bool(self.r(f"is.matrix({clara_expr}$clustering)")[0]):
                    clustering = self._matrix(f"{clara_expr}$clustering")
                entry: dict[str, Any] = {
                    "class": [str(value) for value in self.r(f"class({clara_expr})")],
                    "names": [str(value) for value in self.r(f"names({clara_expr})")],
                    "medoids_0b": np.asarray(
                        self.r(f"as.integer({clara_expr}$medoids)"),
                        dtype=int,
                    )
                    - 1,
                    "clustering": clustering,
                    "evol.diss": np.asarray(
                        self.r(f"as.numeric({clara_expr}$evol.diss)"),
                        dtype=float,
                    ),
                    "iter.objective": np.asarray(
                        self.r(f"as.numeric({clara_expr}$iter.objective)"),
                        dtype=float,
                    ),
                    "objective": float(
                        np.asarray(self.r(f"as.numeric({clara_expr}$objective)"))[0]
                    ),
                    "iteration": int(np.asarray(self.r(f"as.integer({clara_expr}$iteration)"))[0]),
                    "arimatrix": arimatrix,
                    "criteria": str(self.r(f"as.character({clara_expr}$criteria)")[0]),
                    "method": str(self.r(f"as.character({clara_expr}$method)")[0]),
                    "summary": np.asarray(
                        self.r(
                            f"as.numeric(c({clara_expr}$avg_dist, {clara_expr}$pbm, "
                            f"{clara_expr}$db, {clara_expr}$xb, {clara_expr}$ams, "
                            f"{clara_expr}$ari08, {clara_expr}$jc08, "
                            f"{clara_expr}$R, {clara_expr}$k))"
                        ),
                        dtype=float,
                    ),
                }
                for field in ("diss", "representativeness"):
                    if bool(self.r(f"!is.null({clara_expr}${field})")[0]):
                        entry[field] = self._matrix(f"as.matrix({clara_expr}${field})")
                base["clara"][int(kval)] = entry
                if bool(self.r(f"is.data.frame({range_expr}$clustering)")[0]):
                    base["clustering"][f"cluster{int(kval)}"] = np.asarray(
                        self.r(f"as.numeric({range_expr}$clustering[[{index}]])"),
                        dtype=float,
                    )
                else:
                    base["clustering"][f"cluster{int(kval)}"] = self._matrix(
                        f"as.matrix({range_expr}$clustering[[{index}]])"
                    )

            for plot_stat in ("CQI", "CQImv", "ari.boxplot", "stability", "stabmean"):
                self.r(f"""
                    {contract_name}$plot.error <- local({{
                      filename <- tempfile(fileext='.pdf')
                      grDevices::pdf(filename)
                      on.exit({{ grDevices::dev.off(); unlink(filename) }}, add=TRUE)
                      tryCatch({{
                        suppressWarnings(plot({range_expr}, stat='{plot_stat}'))
                        NA_character_
                      }}, error=function(e) conditionMessage(e))
                    }})
                    """)
                plot_error = str(self.r(f"as.character({contract_name}$plot.error)")[0])
                base["plot_errors"][plot_stat] = None if plot_error == "NA" else plot_error
            if is_family:
                base["family_class"] = [str(value) for value in self.r(f"class({value_expr})")]
                base["family_names"] = [str(value) for value in self.r(f"names({value_expr})")]
                base["family_param"] = {
                    "criteria": [
                        str(value) for value in self.r(f"as.character({value_expr}$param$criteria)")
                    ],
                    "all.criterias": [
                        str(value)
                        for value in self.r(f"as.character({value_expr}$param$all.criterias)")
                    ],
                    "kvals": np.asarray(
                        self.r(f"as.integer({value_expr}$param$kvals)"),
                        dtype=int,
                    ),
                    "method": str(self.r(f"as.character({value_expr}$param$method)")[0]),
                    "stability": bool(self.r(f"as.logical({value_expr}$param$stability)")[0]),
                }
                base["allstats_names"] = [
                    str(value) for value in self.r(f"names({value_expr}$allstats)")
                ]
                base["allstats_ngroup"] = np.asarray(
                    self.r(f"as.integer({value_expr}$allstats$ngroup)"),
                    dtype=int,
                )
                base["allstats_criteria"] = [
                    str(value) for value in self.r(f"as.character({value_expr}$allstats$criteria)")
                ]
            return base
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def r_seqtrate_global(self, X: np.ndarray) -> np.ndarray:
        """R seqtrate(seqdef(X)) -- the global transition-rate matrix (K x K).

        This is exactly the matrix R's seqnull(model="Markov") uses as its
        probability model (seqnullpos prob="trate", time.varying=FALSE).
        """
        Xi = np.asarray(X, dtype=int)
        n_states = int(np.unique(Xi).size)
        self._assign_matrix("Xmat", Xi.astype(float))
        self.r("storage.mode(Xmat) <- 'integer'")
        self.r("sd <- suppressMessages(TraMineR::seqdef(Xmat))")
        vals = np.asarray(self.r("as.numeric(TraMineR::seqtrate(sd))"), dtype=float)
        return vals.reshape((n_states, n_states), order="F")

    def r_seqstatd_first_freq(self, X: np.ndarray) -> np.ndarray:
        """R seqstatd(seqdef(X))$Frequencies[,1] -- the begin='freq' start dist."""
        Xi = np.asarray(X, dtype=int)
        self._assign_matrix("Xmat", Xi.astype(float))
        self.r("storage.mode(Xmat) <- 'integer'")
        self.r("sd <- suppressMessages(TraMineR::seqdef(Xmat))")
        vals = self.r("as.numeric(TraMineR::seqstatd(sd)$Frequencies[,1])")
        return np.asarray(vals, dtype=float).reshape(-1)

    def r_seqnull_structural(
        self, X: np.ndarray, *, model: str, seed: int = 0
    ) -> tuple[tuple[int, int], list[str]]:
        """Run R seqnull(sd, model=) -> (shape, alphabet-of-result).

        Returns the (nrow, ncol) of the generated null object and its alphabet
        (a subset of the original states). seqnull is stochastic -> seed R first.
        """
        Xi = np.asarray(X, dtype=int)
        self._assign_matrix("Xmat", Xi.astype(float))
        self.r("storage.mode(Xmat) <- 'integer'")
        self.r("sd <- suppressMessages(TraMineR::seqdef(Xmat))")
        self.r.assign("model", ro.StrVector([str(model)]))
        self.r(f"set.seed({int(seed)})")
        self.r("ns <- suppressMessages(WeightedCluster::seqnull(sd, model=as.character(model)))")
        shape = (int(self.r("nrow(ns)")[0]), int(self.r("ncol(ns)")[0]))
        alpha = [str(x) for x in self.r("TraMineR::alphabet(ns)")]
        return shape, alpha

    def seqnull_complete_contract(
        self,
        X: np.ndarray,
        *,
        model: str = "combined",
        imp_trans: Mapping[object, Sequence[object] | object] | None = None,
        imp_trans_limit: float = -1.0,
        trate: str | np.ndarray = "trate",
        begin: str | Sequence[float] | np.ndarray = "freq",
        time_varying: bool = True,
        weighted: bool = True,
        weights: Sequence[float] | np.ndarray | None = None,
        alphabet: Sequence[object] | None = None,
        states: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run installed ``seqnull`` while recording its exact random draws.

        Copies the three installed closures into a child lexical environment and
        shadows only ``sample``, ``sample.int``, and ``runif`` there. Base and
        package namespace bindings are never changed. The recorded values form a
        deterministic replay queue for Python; the installed function still
        computes every probability and assembles the oracle result.
        """
        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_seqnull_{key}_{token}"
            for key in (
                "X",
                "weights",
                "alphabet",
                "states",
                "labels",
                "imp",
                "trate",
                "begin",
                "contract",
            )
        }
        Xi = np.asarray(X)
        if Xi.ndim != 2:
            raise ValueError("X must be two-dimensional")
        self._assign_matrix(names["X"], np.asarray(Xi, dtype=float))
        self.r(f"storage.mode({names['X']}) <- 'integer'")

        def assign_optional_vector(name: str, values: object, *, string: bool = False) -> None:
            if values is None:
                self.r(f"{name} <- NULL")
            elif string:
                self.r.assign(name, ro.StrVector([str(value) for value in values]))  # type: ignore[arg-type]
            else:
                self.r.assign(
                    name,
                    ro.FloatVector(np.asarray(values, dtype=float).reshape(-1)),
                )

        assign_optional_vector(names["weights"], weights)
        assign_optional_vector(names["alphabet"], alphabet, string=True)
        assign_optional_vector(names["states"], states, string=True)
        assign_optional_vector(names["labels"], labels, string=True)

        if imp_trans is None:
            self.r(f"{names['imp']} <- NULL")
        else:
            imp_items: dict[str, Any] = {}
            for key, raw_values in imp_trans.items():
                if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
                    values = [raw_values]
                else:
                    values = list(raw_values)
                imp_items[str(key)] = ro.StrVector([str(value) for value in values])
            self.r.assign(names["imp"], ro.ListVector(imp_items))

        if isinstance(trate, str):
            self.r.assign(names["trate"], ro.StrVector([str(trate)]))
        else:
            trate_array = np.asarray(trate, dtype=float)
            self.r.assign(
                names["trate"],
                ro.FloatVector(trate_array.reshape(-1, order="F")),
            )
            dims = ",".join(str(int(value)) for value in trate_array.shape)
            self.r(f"dim({names['trate']}) <- c({dims})")
        if isinstance(begin, str):
            self.r.assign(names["begin"], ro.StrVector([str(begin)]))
        else:
            self.r.assign(
                names["begin"],
                ro.FloatVector(np.asarray(begin, dtype=float).reshape(-1)),
            )

        cleanup = list(names.values())
        try:
            self.r(f"""
                {names['contract']} <- local({{
                  suppressMessages(library(WeightedCluster))
                  suppressMessages(library(TraMineR))
                  X_local <- get('{names['X']}', envir=.GlobalEnv)
                  weights_local <- get('{names['weights']}', envir=.GlobalEnv)
                  alphabet_local <- get('{names['alphabet']}', envir=.GlobalEnv)
                  states_local <- get('{names['states']}', envir=.GlobalEnv)
                  labels_local <- get('{names['labels']}', envir=.GlobalEnv)
                  imp_local <- get('{names['imp']}', envir=.GlobalEnv)
                  trate_local <- get('{names['trate']}', envir=.GlobalEnv)
                  begin_local <- get('{names['begin']}', envir=.GlobalEnv)
                  seqdef_args <- list(data=X_local)
                  if (!is.null(alphabet_local)) seqdef_args$alphabet <- alphabet_local
                  if (!is.null(states_local)) seqdef_args$states <- states_local
                  if (!is.null(labels_local)) seqdef_args$labels <- labels_local
                  if (!is.null(weights_local)) seqdef_args$weights <- weights_local
                  sd_local <- suppressMessages(do.call(TraMineR::seqdef, seqdef_args))

                  replay_env <- new.env(parent=environment(WeightedCluster:::seqnull))
                  replay_env$draws <- list()
                  replay_env$record <- function(kind, values, prob=NULL, x=NULL) {{
                    replay_env$draws[[length(replay_env$draws) + 1L]] <- list(
                      kind=kind,
                      values=values,
                      prob=prob,
                      x=x
                    )
                    values
                  }}
                  replay_env$sample <- function(x, size, replace=FALSE, prob=NULL, ...) {{
                    value <- base::sample(x, size=size, replace=replace, prob=prob, ...)
                    replay_env$record('sample', value, prob=prob, x=x)
                  }}
                  replay_env$sample.int <- function(
                      n, size=n, replace=FALSE, prob=NULL, useHash=(n > 1e7), ...) {{
                    value <- base::sample.int(
                      n, size=size, replace=replace, prob=prob, useHash=useHash, ...
                    )
                    replay_env$record('sample.int', value, prob=prob, x=seq_len(n))
                  }}
                  replay_env$runif <- function(n, min=0, max=1) {{
                    value <- stats::runif(n, min=min, max=max)
                    replay_env$record('runif', value)
                  }}
                  for (function_name in c('seqnull', 'seqnullspell', 'seqnullpos')) {{
                    fn <- get(function_name, envir=asNamespace('WeightedCluster'))
                    environment(fn) <- replay_env
                    assign(function_name, fn, envir=replay_env)
                  }}

                  state <- new.env(parent=emptyenv())
                  state$error <- NULL
                  set.seed({int(seed)})
                  value <- tryCatch(
                    suppressMessages(replay_env$seqnull(
                      sd_local,
                      model='{str(model)}',
                      imp.trans=imp_local,
                      imp.trans.limit={float(imp_trans_limit)!r},
                      trate=trate_local,
                      begin=begin_local,
                      time.varying={str(bool(time_varying)).upper()},
                      weighted={str(bool(weighted)).upper()}
                    )),
                    error=function(e) {{
                      state$error <- conditionMessage(e)
                      NULL
                    }}
                  )
                  list(
                    value=value,
                    error=state$error,
                    draws=replay_env$draws
                  )
                }})
                """)
            contract = names["contract"]
            error = (
                None
                if bool(self.r(f"is.null({contract}$error)")[0])
                else str(self.r(f"as.character({contract}$error)")[0])
            )
            draw_count = int(self.r(f"length({contract}$draws)")[0])
            replay_plan: list[dict[str, Any]] = []
            for index in range(1, draw_count + 1):
                draw = f"{contract}$draws[[{index}]]"
                kind = str(self.r(f"as.character({draw}$kind)")[0])
                value_type = str(self.r(f"typeof({draw}$values)")[0])
                if value_type == "character":
                    values: np.ndarray = np.asarray(
                        self.r(f"as.character({draw}$values)"), dtype=object
                    )
                elif value_type in {"integer", "logical"}:
                    values = np.asarray(self.r(f"as.integer({draw}$values)"), dtype=int)
                else:
                    values = np.asarray(self.r(f"as.numeric({draw}$values)"), dtype=float)
                prob = (
                    None
                    if bool(self.r(f"is.null({draw}$prob)")[0])
                    else np.asarray(self.r(f"as.numeric({draw}$prob)"), dtype=float)
                )
                replay_plan.append({"kind": kind, "values": values, "prob": prob})

            result: dict[str, Any] = {
                "error": error,
                "replay_plan": replay_plan,
                "replay_consumed": draw_count,
            }
            if error is not None:
                return result
            value = f"{contract}$value"
            raw = np.asarray(self.r(f"as.character(as.matrix({value}))"), dtype=object)
            nrow = int(self.r(f"nrow({value})")[0])
            ncol = int(self.r(f"ncol({value})")[0])
            result.update(
                {
                    "sequences": raw.reshape((nrow, ncol), order="F"),
                    "class": [str(item) for item in self.r(f"class({value})")],
                    "alphabet": [str(item) for item in self.r(f"TraMineR::alphabet({value})")],
                    "states": [
                        str(item) for item in self.r(f"as.character(attr({value}, 'states'))")
                    ],
                    "labels": [
                        str(item) for item in self.r(f"as.character(attr({value}, 'labels'))")
                    ],
                    "weights": (
                        None
                        if bool(self.r(f"is.null(attr({value}, 'weights'))")[0])
                        else np.asarray(
                            self.r(f"as.numeric(attr({value}, 'weights'))"), dtype=float
                        )
                    ),
                    "lengths": np.asarray(
                        self.r(f"as.integer(TraMineR::seqlength({value}))"), dtype=int
                    ),
                }
            )
            return result
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def seqnullcqi_complete_contract(
        self,
        X: np.ndarray,
        cluster_columns: Mapping[str, np.ndarray],
        null_sequences: Sequence[np.ndarray] | np.ndarray,
        *,
        model: str = "combined",
        seqdist_method: str = "LCS",
        seqdist_norm: str = "maxlength",
        seqdist_indel: float | None = None,
        seqdist_sm: np.ndarray | None = None,
        kmedoid: bool = False,
        hclust_method: str = "ward.D",
        trate: str | np.ndarray = "trate",
        begin: str | Sequence[float] | np.ndarray = "freq",
        time_varying: bool = True,
        weighted: bool = True,
        print_norm: bool = True,
        print_quant: float = 0.95,
        print_digits: int = 2,
        plot_stat: str = "ASW",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Execute installed ``seqnullcqi`` with exact null-sequence replay.

        ``seqnull`` and, on the k-medoids route, only its random-start producer
        are shadowed in child lexical environments. Installed distance,
        hierarchy/k-medoids, CQI, result, print, and plot code remains
        authoritative. Wrappers around the two reclustering calls record their
        exact partitions.
        """
        Xi = np.asarray(X, dtype=int)
        replay = np.asarray(null_sequences, dtype=int)
        if Xi.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if replay.ndim != 3 or replay.shape[1:] != Xi.shape:
            raise ValueError("null_sequences must have shape (R, N, T)")
        if replay.shape[0] < 2:
            raise ValueError("live R seqnullcqi replay requires at least two iterations")
        if not cluster_columns:
            raise ValueError("cluster_columns must not be empty")

        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_seqnullcqi_{key}_{token}"
            for key in (
                "X",
                "clusters",
                "nulls",
                "trate",
                "begin",
                "seqdist_sm",
                "starts",
                "contract",
            )
        }
        self._assign_matrix(names["X"], Xi.astype(float))
        self.r(f"storage.mode({names['X']}) <- 'integer'")
        ordered_clusters = sorted(
            (
                (int(str(key).replace("cluster", "")), np.asarray(value, dtype=int))
                for key, value in cluster_columns.items()
            ),
            key=lambda item: item[0],
        )
        cluster_matrix = np.column_stack([values for _, values in ordered_clusters])
        self._assign_matrix(names["clusters"], cluster_matrix.astype(float))
        self.r(f"storage.mode({names['clusters']}) <- 'integer'")
        self.r.assign(
            names["nulls"],
            ro.IntVector(replay.reshape(-1, order="F")),
        )
        self.r(
            f"dim({names['nulls']}) <- c({replay.shape[0]}, {replay.shape[1]}, {replay.shape[2]})"
        )
        if isinstance(trate, str):
            self.r.assign(names["trate"], ro.StrVector([str(trate)]))
        else:
            trate_array = np.asarray(trate, dtype=float)
            self.r.assign(
                names["trate"],
                ro.FloatVector(trate_array.reshape(-1, order="F")),
            )
            dims = ",".join(str(int(value)) for value in trate_array.shape)
            self.r(f"dim({names['trate']}) <- c({dims})")
        if isinstance(begin, str):
            self.r.assign(names["begin"], ro.StrVector([str(begin)]))
        else:
            self.r.assign(
                names["begin"],
                ro.FloatVector(np.asarray(begin, dtype=float).reshape(-1)),
            )
        if seqdist_sm is None:
            self.r(f"{names['seqdist_sm']} <- NULL")
        else:
            self._assign_matrix(names["seqdist_sm"], np.asarray(seqdist_sm, dtype=float))

        kvals = [item[0] for item in ordered_clusters]
        if kmedoid:
            starts = ro.ListVector(
                [
                    (
                        str(iteration + 1),
                        ro.ListVector(
                            [
                                (
                                    str(kval),
                                    ro.IntVector(
                                        [
                                            (iteration * max(kvals) + offset) % Xi.shape[0] + 1
                                            for offset in range(kval)
                                        ]
                                    ),
                                )
                                for kval in kvals
                            ]
                        ),
                    )
                    for iteration in range(replay.shape[0])
                ]
            )
            self.r.assign(names["starts"], starts)
        else:
            self.r(f"{names['starts']} <- NULL")
        cleanup = list(names.values())
        try:
            self.r(f"""
                {names['contract']} <- local({{
                  suppressMessages(library(WeightedCluster))
                  suppressMessages(library(TraMineR))
                  suppressMessages(library(future))
                  X_local <- get('{names['X']}', envir=.GlobalEnv)
                  cluster_matrix <- get('{names['clusters']}', envir=.GlobalEnv)
                  null_array <- get('{names['nulls']}', envir=.GlobalEnv)
                  trate_local <- get('{names['trate']}', envir=.GlobalEnv)
                  begin_local <- get('{names['begin']}', envir=.GlobalEnv)
                  seqdist_sm_local <- get(
                    '{names['seqdist_sm']}', envir=.GlobalEnv
                  )
                  starts_local <- get('{names['starts']}', envir=.GlobalEnv)
                  storage.mode(X_local) <- 'integer'
                  storage.mode(cluster_matrix) <- 'integer'
                  sd_local <- suppressMessages(TraMineR::seqdef(X_local))
                  kvals_local <- as.integer(c({",".join(str(value) for value in kvals)}))
                  clustering_local <- as.data.frame(cluster_matrix)
                  names(clustering_local) <- paste0('cluster', kvals_local)
                  for (column in seq_along(clustering_local)) {{
                    clustering_local[[column]] <- factor(clustering_local[[column]])
                  }}
                  seqdist_local <- list(
                    method='{str(seqdist_method)}',
                    norm='{str(seqdist_norm)}'
                  )
                  if (!is.null(seqdist_sm_local)) {{
                    seqdist_local$sm <- seqdist_sm_local
                  }}
                  if (!is.null({("NULL" if seqdist_indel is None else repr(float(seqdist_indel)))})) {{
                    seqdist_local$indel <- {
                        "NULL" if seqdist_indel is None else repr(float(seqdist_indel))
                    }
                  }}
                  observed_args <- seqdist_local
                  observed_args$seqdata <- sd_local
                  observed_diss <- suppressMessages(do.call(TraMineR::seqdist, observed_args))
                  clustrange_local <- WeightedCluster::as.clustrange(
                    clustering_local, diss=observed_diss, R=1
                  )

                  fixed <- vector('list', dim(null_array)[1])
                  for (iteration in seq_len(dim(null_array)[1])) {{
                    replay_matrix <- null_array[iteration, , ]
                    dim(replay_matrix) <- c(nrow(X_local), ncol(X_local))
                    fixed[[iteration]] <- suppressMessages(TraMineR::seqdef(
                      replay_matrix,
                      alphabet=attr(sd_local, 'alphabet'),
                      states=attr(sd_local, 'alphabet'),
                      labels=attr(sd_local, 'labels')
                    ))
                  }}

                  state <- new.env(parent=emptyenv())
                  state$seqnull_iteration <- 0L
                  state$distance_iteration <- 0L
                  state$cluster_iteration <- 0L
                  state$calls <- vector('list', length(fixed))
                  state$partitions <- vector('list', length(fixed))
                  state$starts <- vector('list', length(fixed))
                  state$hierarchies <- vector('list', length(fixed))
                  state$distances <- vector('list', length(fixed))
                  for (iteration in seq_along(fixed)) {{
                    state$partitions[[iteration]] <- list()
                    state$starts[[iteration]] <- list()
                  }}

                  replay_env <- new.env(parent=environment(WeightedCluster:::seqnullcqi))
                  replay_env$state <- state
                  replay_env$fixed <- fixed
                  replay_env$seqnull <- function(seqdata, model, ...) {{
                    iteration <- as.integer(get(
                      'loop', envir=parent.frame(), inherits=TRUE
                    ))
                    state$seqnull_iteration <- state$seqnull_iteration + 1L
                    state$calls[[iteration]] <- c(
                      list(model=model), list(...)
                    )
                    replay_env$fixed[[iteration]]
                  }}
                  original_seqdist <- TraMineR::seqdist
                  replay_env$seqdist <- function(...) {{
                    iteration <- as.integer(get(
                      'loop', envir=parent.frame(), inherits=TRUE
                    ))
                    state$distance_iteration <- state$distance_iteration + 1L
                    value <- original_seqdist(...)
                    state$distances[[iteration]] <- as.matrix(value)
                    value
                  }}
                  original_as_clustrange <- WeightedCluster::as.clustrange
                  replay_env$as.clustrange <- function(...) {{
                    iteration <- as.integer(get(
                      'loop', envir=parent.frame(), inherits=TRUE
                    ))
                    state$cluster_iteration <- state$cluster_iteration + 1L
                    args <- list(...)
                    if (length(args) && inherits(args[[1]], 'hclust')) {{
                      state$hierarchies[[iteration]] <- list(
                        merge=args[[1]]$merge,
                        height=args[[1]]$height,
                        order=args[[1]]$order
                      )
                    }}
                    value <- original_as_clustrange(...)
                    state$partitions[[iteration]] <- lapply(
                      value$clustering, function(column) as.integer(column)
                    )
                    value
                  }}
                  original_kmedrange <- WeightedCluster::wcKMedRange
                  kmedrange_env <- new.env(parent=environment(original_kmedrange))
                  kmedrange_env$state <- state
                  kmedrange_env$starts <- starts_local
                  kmedrange_env$current_iteration <- 1L
                  kmedrange_env$original_kmedoids <- WeightedCluster::wcKMedoids
                  kmedoids_env <- new.env(
                    parent=environment(kmedrange_env$original_kmedoids)
                  )
                  kmedoids_env$state <- state
                  kmedoids_env$starts <- starts_local
                  kmedoids_env$current_iteration <- 1L
                  kmedoids_env$current_k <- 2L
                  kmedoids_env$sample.int <- eval(quote(function(
                      n, size=n, replace=FALSE, prob=NULL, ...
                  ) {{
                    key <- as.character(as.integer(current_k))
                    initial <- starts[[current_iteration]][[key]]
                    state$starts[[current_iteration]][[key]] <- rbind(
                      state$starts[[current_iteration]][[key]],
                      as.integer(initial)
                    )
                    as.integer(initial)
                  }}), envir=kmedoids_env)
                  kmedoids_local <- kmedrange_env$original_kmedoids
                  environment(kmedoids_local) <- kmedoids_env
                  kmedrange_env$kmedoids_env <- kmedoids_env
                  kmedrange_env$kmedoids_local <- kmedoids_local
                  kmedrange_env$wcKMedoids <- eval(quote(function(diss, k, ...) {{
                    kmedoids_env$current_iteration <- current_iteration
                    kmedoids_env$current_k <- as.integer(k)
                    kmedoids_local(diss=diss, k=k, ...)
                  }}), envir=kmedrange_env)
                  kmedrange_local <- original_kmedrange
                  environment(kmedrange_local) <- kmedrange_env
                  replay_env$wcKMedRange <- function(...) {{
                    iteration <- as.integer(get(
                      'loop', envir=parent.frame(), inherits=TRUE
                    ))
                    state$cluster_iteration <- state$cluster_iteration + 1L
                    kmedrange_env$current_iteration <- iteration
                    value <- kmedrange_local(...)
                    state$partitions[[iteration]] <- lapply(
                      value$clustering, function(column) as.integer(column)
                    )
                    value
                  }}
                  seqnullcqi_local <- WeightedCluster:::seqnullcqi
                  environment(seqnullcqi_local) <- replay_env

                  old_plan <- future::plan(future::sequential)
                  on.exit(future::plan(old_plan), add=TRUE)
                  forwarded <- list(
                    trate=trate_local,
                    begin=begin_local,
                    time.varying={str(bool(time_varying)).upper()},
                    weighted={str(bool(weighted)).upper()}
                  )
                  call_args <- c(
                    list(
                      seqdata=sd_local,
                      clustrange=clustrange_local,
                      R=as.integer(length(fixed)),
                      model='{str(model)}',
                      seqdist.args=seqdist_local,
                      kmedoid={str(bool(kmedoid)).upper()},
                      hclust.method='{str(hclust_method)}',
                      parallel=FALSE,
                      progressbar=FALSE
                    ),
                    forwarded
                  )
                  error <- NULL
                  value <- tryCatch(
                    {{
                      set.seed({int(seed)})
                      suppressMessages(do.call(seqnullcqi_local, call_args))
                    }},
                    error=function(e) {{
                      error <<- conditionMessage(e)
                      NULL
                    }}
                  )
                  if (is.null(value)) {{
                    return(list(error=error, state=state))
                  }}

                  print_error <- NULL
                  print_visible <- NULL
                  print_capture <- tryCatch(
                    capture.output(
                      print_visible <- withVisible(print(
                        value,
                        norm={str(bool(print_norm)).upper()},
                        quant={float(print_quant)!r},
                        digits={int(print_digits)}
                      ))
                    ),
                    error=function(e) {{
                      print_error <<- conditionMessage(e)
                      character()
                    }}
                  )
                  print_table <- if (is.null(print_visible)) NULL else print_visible$value

                  stat_local <- '{str(plot_stat)}'
                  orig_raw <- as.numeric(value$clustrange$stats[, stat_local])
                  null_raw <- as.matrix(value$stats[[stat_local]])
                  normalized <- WeightedCluster:::normstatcqi(
                    value, stat=stat_local, norm=TRUE
                  )
                  alpha_local <- (1-{float(print_quant)!r})/2
                  per_k_quantiles <- apply(
                    null_raw, 2, stats::quantile,
                    probs=c(alpha_local, 1-alpha_local)
                  )
                  max_t <- apply(null_raw, 1, max)
                  max_t_interval <- stats::quantile(
                    max_t, probs=c(alpha_local, 1-alpha_local)
                  )
                  plot_threshold <- if (stat_local == 'HC') {{
                    stats::quantile(apply(null_raw, 1, min), alpha_local)
                  }} else {{
                    stats::quantile(apply(null_raw, 1, max), 1-alpha_local)
                  }}
                  all_standardized <- list()
                  for (stat_name in colnames(value$clustrange$stats)) {{
                    observed <- as.numeric(value$clustrange$stats[, stat_name])
                    combined <- rbind(observed, value$stats[[stat_name]])
                    all_standardized[[stat_name]] <- vapply(
                      seq_along(observed),
                      function(index) {{
                        (observed[index] - mean(combined[, index])) /
                          stats::sd(combined[, index])
                      }},
                      numeric(1)
                    )
                  }}

                  plot_errors <- list()
                  for (plot_type in c('line', 'density', 'boxplot', 'seqdplot', 'unknown')) {{
                    filename <- tempfile(fileext='.pdf')
                    grDevices::pdf(filename)
                    plot_errors[[plot_type]] <- tryCatch(
                      {{
                        capture.output(plot(
                          value, stat=stat_local, type=plot_type,
                          quant={float(print_quant)!r}, norm=TRUE
                        ))
                        NA_character_
                      }},
                      error=function(e) conditionMessage(e)
                    )
                    grDevices::dev.off()
                    unlink(filename)
                  }}
                  filename <- tempfile(fileext='.pdf')
                  grDevices::pdf(filename)
                  plot_errors[['invalid_stat']] <- tryCatch(
                    {{
                      plot(value, stat='not_a_stat', type='line')
                      NA_character_
                    }},
                    error=function(e) conditionMessage(e)
                  )
                  grDevices::dev.off()
                  unlink(filename)

                  list(
                    error=error,
                    value=value,
                    state=state,
                    print=list(
                      output=print_capture,
                      table=print_table,
                      visible=if (is.null(print_visible)) NA else print_visible$visible,
                      error=print_error
                    ),
                    semantic=list(
                      observed.distance=as.matrix(observed_diss),
                      orig.raw=orig_raw,
                      null.raw=null_raw,
                      orig.normalized=normalized$origstat,
                      null.normalized=normalized$nullstat,
                      per.k.quantiles=per_k_quantiles,
                      max.t=max_t,
                      max.t.interval=max_t_interval,
                      plot.threshold=plot_threshold,
                      all.standardized=all_standardized
                    ),
                    plot.errors=plot_errors
                  )
                }})
                """)
            contract = names["contract"]
            error = (
                None
                if bool(self.r(f"is.null({contract}$error)")[0])
                else str(self.r(f"as.character({contract}$error)")[0])
            )
            result: dict[str, Any] = {"error": error}
            if error is not None:
                return result

            value = f"{contract}$value"
            result.update(
                {
                    "class": [str(item) for item in self.r(f"class({value})")],
                    "names": [str(item) for item in self.r(f"names({value})")],
                    "R": int(self.r(f"as.integer({value}$R)")[0]),
                    "kmedoid": bool(self.r(f"as.logical({value}$kmedoid)")[0]),
                    "hclust_method": str(self.r(f"as.character({value}$hclust.method)")[0]),
                    "seqdist_names": [str(item) for item in self.r(f"names({value}$seqdist.args)")],
                    "seqdist_method": str(self.r(f"as.character({value}$seqdist.args$method)")[0]),
                    "seqdist_norm": str(self.r(f"as.character({value}$seqdist.args$norm)")[0]),
                    "seqdist_indel": (
                        None
                        if bool(self.r(f"is.null({value}$seqdist.args$indel)")[0])
                        else float(self.r(f"as.numeric({value}$seqdist.args$indel)")[0])
                    ),
                    "seqdist_sm": (
                        None
                        if bool(self.r(f"is.null({value}$seqdist.args$sm)")[0])
                        else self._matrix(f"as.matrix({value}$seqdist.args$sm)")
                    ),
                    "nullmodel_names": [str(item) for item in self.r(f"names({value}$nullmodel)")],
                    "nullmodel_model": str(self.r(f"as.character({value}$nullmodel$model)")[0]),
                    "nullmodel_time_varying": bool(
                        self.r(f"as.logical({value}$nullmodel$time.varying)")[0]
                    ),
                    "nullmodel_weighted": bool(
                        self.r(f"as.logical({value}$nullmodel$weighted)")[0]
                    ),
                    "stats_names": [str(item) for item in self.r(f"names({value}$stats)")],
                    "stats": {},
                    "partitions": [],
                    "distances": [],
                    "hierarchies": [],
                    "kmedoid_starts_0b": [],
                }
            )
            clustrange = f"{value}$clustrange"
            result["kvals"] = np.asarray(self.r(f"as.integer({clustrange}$kvals)"), dtype=int)
            result["clustrange_stats_names"] = [
                str(item) for item in self.r(f"colnames({clustrange}$stats)")
            ]
            result["clustrange_stats"] = self._matrix(f"as.matrix({clustrange}$stats)")
            result["clustrange_stats_row_names"] = [
                str(item) for item in self.r(f"rownames({clustrange}$stats)")
            ]
            result["clustrange_clustering"] = {
                int(kval): np.asarray(
                    self.r(f"as.integer({clustrange}$clustering[[{position}]])"),
                    dtype=int,
                )
                for position, kval in enumerate(kvals, start=1)
            }
            raw_sequences = np.asarray(
                self.r(f"as.character(as.matrix({value}$seqdata))"), dtype=object
            )
            result["sequences"] = raw_sequences.reshape(
                (replay.shape[0] * replay.shape[1], replay.shape[2]), order="F"
            )
            for stat in result["stats_names"]:
                result["stats"][stat] = self._matrix(f"as.matrix({value}$stats[[{repr(stat)}]])")
            for iteration in range(1, replay.shape[0] + 1):
                partition_map: dict[int, np.ndarray] = {}
                start_map: dict[int, np.ndarray] = {}
                for position, kval in enumerate(kvals, start=1):
                    partition_map[int(kval)] = np.asarray(
                        self.r(
                            f"as.integer({contract}$state$partitions[[{iteration}]][[{position}]])"
                        ),
                        dtype=int,
                    )
                    start_expr = f"{contract}$state$starts[[{iteration}]][[{repr(str(kval))}]]"
                    if bool(self.r(f"!is.null({start_expr})")[0]):
                        start_map[int(kval)] = (
                            self._matrix(f"as.matrix({start_expr})").astype(int) - 1
                        )
                result["partitions"].append(partition_map)
                result["kmedoid_starts_0b"].append(start_map)
                result["distances"].append(
                    self._matrix(f"as.matrix({contract}$state$distances[[{iteration}]])")
                )
                hierarchy_expr = f"{contract}$state$hierarchies[[{iteration}]]"
                if bool(self.r(f"is.null({hierarchy_expr})")[0]):
                    result["hierarchies"].append(None)
                else:
                    result["hierarchies"].append(
                        {
                            "merge": self._matrix(f"as.matrix({hierarchy_expr}$merge)").astype(int),
                            "height": np.asarray(
                                self.r(f"as.numeric({hierarchy_expr}$height)"),
                                dtype=float,
                            ),
                            "order_1b": np.asarray(
                                self.r(f"as.integer({hierarchy_expr}$order)"),
                                dtype=int,
                            ),
                        }
                    )

            print_expr = f"{contract}$print"
            print_error = (
                None
                if bool(self.r(f"is.null({print_expr}$error)")[0])
                else str(self.r(f"as.character({print_expr}$error)")[0])
            )
            print_table_is_null = bool(self.r(f"is.null({print_expr}$table)")[0])
            if print_table_is_null:
                print_table = None
            else:
                print_nrow = int(self.r(f"nrow({print_expr}$table)")[0])
                print_ncol = int(self.r(f"ncol({print_expr}$table)")[0])
                print_values = np.asarray(
                    self.r(f"as.character(as.matrix({print_expr}$table))"),
                    dtype=object,
                )
                print_table = print_values.reshape((print_nrow, print_ncol), order="F")
            result["print"] = {
                "output": [str(item) for item in self.r(f"{print_expr}$output")],
                "error": print_error,
                "visible": (
                    None
                    if bool(self.r(f"is.na({print_expr}$visible)")[0])
                    else bool(self.r(f"as.logical({print_expr}$visible)")[0])
                ),
                "table": print_table,
                "row_names": (
                    []
                    if print_table_is_null
                    else [str(item) for item in self.r(f"rownames({print_expr}$table)")]
                ),
                "column_names": (
                    []
                    if print_table_is_null
                    else [str(item) for item in self.r(f"colnames({print_expr}$table)")]
                ),
            }
            semantic = f"{contract}$semantic"
            result["semantic"] = {
                "observed_distance": self._matrix(f"as.matrix({semantic}$observed.distance)"),
                "orig_raw": np.asarray(self.r(f"as.numeric({semantic}$orig.raw)"), dtype=float),
                "null_raw": self._matrix(f"as.matrix({semantic}$null.raw)"),
                "orig_normalized": np.asarray(
                    self.r(f"as.numeric({semantic}$orig.normalized)"), dtype=float
                ),
                "null_normalized": self._matrix(f"as.matrix({semantic}$null.normalized)"),
                "per_k_quantiles": self._matrix(f"as.matrix({semantic}$per.k.quantiles)"),
                "max_t": np.asarray(self.r(f"as.numeric({semantic}$max.t)"), dtype=float),
                "max_t_interval": np.asarray(
                    self.r(f"as.numeric({semantic}$max.t.interval)"), dtype=float
                ),
                "plot_threshold": float(self.r(f"as.numeric({semantic}$plot.threshold)")[0]),
                "all_standardized": {
                    stat: np.asarray(
                        self.r(f"as.numeric({semantic}$all.standardized[[{repr(stat)}]])"),
                        dtype=float,
                    )
                    for stat in result["stats_names"]
                },
            }
            result["plot_errors"] = {}
            for plot_type in (
                "line",
                "density",
                "boxplot",
                "seqdplot",
                "unknown",
                "invalid_stat",
            ):
                raw_error = str(
                    self.r(f"as.character({contract}$plot.errors[[{repr(plot_type)}]])")[0]
                )
                result["plot_errors"][plot_type] = (
                    None if raw_error in {"NA", "NA_character_"} else raw_error
                )
            return result
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def bootclustrange_complete_contract(
        self,
        X: np.ndarray,
        cluster_columns: Mapping[str, np.ndarray],
        replay_samples_1b: Sequence[Sequence[int]] | np.ndarray,
        *,
        sampling: str = "random",
        strata: Sequence[int] | np.ndarray | None = None,
        medoids_1b: Sequence[int] | np.ndarray | None = None,
        object_kind: str = "frame",
        seqdist_method: str = "LCS",
        seqdist_norm: str = "none",
        seqdist_indel: float | None = None,
        seqdist_sm: np.ndarray | None = None,
        weights: Sequence[float] | np.ndarray | None = None,
        print_digits: int = 2,
        print_bootstat: str = "mean",
        plot_stat: str | Sequence[str] = "noCH",
        plot_norm: str = "none",
        plot_conf_int: float = 0.95,
        plot_ci_method: str = "perc",
        plot_line: str = "median",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Execute installed ``bootclustrange`` with exact subsample replay.

        The installed function, public TraMineR distance producer,
        ``as.clustrange``, print method, and plot method remain authoritative.
        Only the random sample primitives are shadowed in a child lexical
        environment so Python and R consume identical one-based subsamples.
        """
        Xi = np.asarray(X, dtype=int)
        samples = np.asarray(replay_samples_1b, dtype=int)
        if Xi.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if samples.ndim != 2 or samples.shape[0] < 1:
            raise ValueError("replay_samples_1b must have shape (R, sample_size)")
        if not cluster_columns:
            raise ValueError("cluster_columns must not be empty")
        if object_kind not in {"frame", "seqclararange"}:
            raise ValueError("object_kind must be 'frame' or 'seqclararange'")

        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_bootclustrange_{key}_{token}"
            for key in (
                "X",
                "clusters",
                "samples",
                "strata",
                "medoids",
                "weights",
                "seqdist_sm",
                "plot_stat",
                "contract",
            )
        }
        cleanup = list(names.values())
        self._assign_matrix(names["X"], Xi.astype(float))
        self.r(f"storage.mode({names['X']}) <- 'integer'")
        ordered_clusters = sorted(
            (
                (int(str(key).replace("cluster", "")), np.asarray(value, dtype=int))
                for key, value in cluster_columns.items()
            ),
            key=lambda item: item[0],
        )
        cluster_matrix = np.column_stack([values for _, values in ordered_clusters])
        self._assign_matrix(names["clusters"], cluster_matrix.astype(float))
        self.r(f"storage.mode({names['clusters']}) <- 'integer'")
        self._assign_matrix(names["samples"], samples.astype(float))
        self.r(f"storage.mode({names['samples']}) <- 'integer'")
        if strata is None:
            self.r(f"{names['strata']} <- NULL")
        else:
            self.r.assign(
                names["strata"],
                ro.IntVector(np.asarray(strata, dtype=int).reshape(-1)),
            )
        if medoids_1b is None:
            self.r(f"{names['medoids']} <- NULL")
        else:
            self.r.assign(
                names["medoids"],
                ro.IntVector(np.asarray(medoids_1b, dtype=int).reshape(-1)),
            )
        if weights is None:
            self.r(f"{names['weights']} <- NULL")
        else:
            self.r.assign(
                names["weights"],
                ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
            )
        if seqdist_sm is None:
            self.r(f"{names['seqdist_sm']} <- NULL")
        else:
            self._assign_matrix(names["seqdist_sm"], np.asarray(seqdist_sm, dtype=float))
        plot_stats = [plot_stat] if isinstance(plot_stat, str) else list(plot_stat)
        self.r.assign(
            names["plot_stat"],
            ro.StrVector([str(value) for value in plot_stats]),
        )

        kvals = [item[0] for item in ordered_clusters]
        indel_expr = "NULL" if seqdist_indel is None else repr(float(seqdist_indel))
        try:
            self.r(f"""
                {names['contract']} <- local({{
                  suppressMessages(library(WeightedCluster))
                  suppressMessages(library(TraMineR))
                  suppressMessages(library(future))
                  X_local <- get('{names['X']}', envir=.GlobalEnv)
                  cluster_matrix <- get('{names['clusters']}', envir=.GlobalEnv)
                  samples_local <- get('{names['samples']}', envir=.GlobalEnv)
                  strata_local <- get('{names['strata']}', envir=.GlobalEnv)
                  medoids_local <- get('{names['medoids']}', envir=.GlobalEnv)
                  weights_local <- get('{names['weights']}', envir=.GlobalEnv)
                  seqdist_sm_local <- get(
                    '{names['seqdist_sm']}', envir=.GlobalEnv
                  )
                  plot_stat_local <- get('{names['plot_stat']}', envir=.GlobalEnv)
                  storage.mode(X_local) <- 'integer'
                  storage.mode(cluster_matrix) <- 'integer'
                  storage.mode(samples_local) <- 'integer'
                  clustering_local <- as.data.frame(cluster_matrix)
                  names(clustering_local) <- paste0(
                    'cluster', c({",".join(str(value) for value in kvals)})
                  )
                  sd_local <- suppressMessages(TraMineR::seqdef(X_local))
                  if (!is.null(weights_local)) {{
                    attr(sd_local, 'weights') <- weights_local
                  }}
                  if ('{object_kind}' == 'seqclararange') {{
                    clara_local <- vector('list', ncol(clustering_local))
                    clara_local[[length(clara_local)]] <- list(medoids=medoids_local)
                    object_local <- list(
                      clustering=clustering_local,
                      clara=clara_local
                    )
                    class(object_local) <- c(
                      'seqclararange', 'clustrange', 'list'
                    )
                  }} else {{
                    object_local <- clustering_local
                  }}
                  seqdist_local <- list(
                    method='{str(seqdist_method)}',
                    norm='{str(seqdist_norm)}'
                  )
                  if (!is.null(seqdist_sm_local)) {{
                    seqdist_local$sm <- seqdist_sm_local
                  }}
                  if (!is.null({indel_expr})) {{
                    seqdist_local$indel <- {indel_expr}
                  }}

                  state <- new.env(parent=emptyenv())
                  state$samples <- lapply(
                    seq_len(nrow(samples_local)),
                    function(iteration) as.integer(samples_local[iteration, ])
                  )
                  state$seqdist.calls <- vector('list', nrow(samples_local))
                  state$as.clustrange <- vector('list', nrow(samples_local))
                  replay_env <- new.env(
                    parent=environment(WeightedCluster::bootclustrange)
                  )
                  replay_env$state <- state
                  replay_env$samples_local <- samples_local
                  replay_env$strata_local <- strata_local
                  replay_env$medoids_local <- medoids_local
                  replay_env$sampling_local <- '{str(sampling)}'
                  replay_env$sample_size_local <- ncol(samples_local)
                  replay_env$original_sample_int <- base::sample.int
                  replay_env$original_sample <- base::sample
                  replay_env$sample.int <- eval(quote(function(
                    n, size=n, replace=FALSE, prob=NULL, ...
                  ) {{
                    iteration <- tryCatch(
                      as.integer(get(
                        'loop', envir=parent.frame(), inherits=TRUE
                      )),
                      error=function(e) NA_integer_
                    )
                    if (is.na(iteration)) {{
                      return(original_sample_int(
                        n, size=size, replace=replace, prob=prob, ...
                      ))
                    }}
                    if (!replace && size > n) {{
                      stop(
                        "cannot take a sample larger than the population ",
                        "when 'replace = FALSE'"
                      )
                    }}
                    planned <- as.integer(samples_local[iteration, ])
                    state$samples[[iteration]] <- planned
                    if (
                      sampling_local == 'medoids' &&
                      !is.null(medoids_local)
                    ) {{
                      forced <- as.integer(medoids_local)
                      if (
                        length(planned) < length(forced) ||
                        !identical(planned[seq_along(forced)], forced)
                      ) {{
                        stop('medoid replay must begin with installed medoids')
                      }}
                      planned <- planned[-seq_along(forced)]
                    }}
                    if (length(planned) != size) {{
                      stop('replayed sample has the wrong size')
                    }}
                    planned
                  }}), envir=replay_env)
                  replay_env$sample <- eval(quote(function(
                    x, size, replace=FALSE, prob=NULL, ...
                  ) {{
                    iteration <- tryCatch(
                      as.integer(get(
                        'loop', envir=parent.frame(), inherits=TRUE
                      )),
                      error=function(e) NA_integer_
                    )
                    if (is.na(iteration)) {{
                      if (missing(size)) {{
                        return(original_sample(
                          x, replace=replace, prob=prob, ...
                        ))
                      }}
                      return(original_sample(
                        x, size=size, replace=replace, prob=prob, ...
                      ))
                    }}
                    planned <- as.integer(samples_local[iteration, ])
                    state$samples[[iteration]] <- planned
                    parent <- parent.frame()
                    if (
                      exists('correction', envir=parent, inherits=FALSE) &&
                      exists('tosample', envir=parent, inherits=FALSE) &&
                      length(x) == length(get(
                        'tosample', envir=parent, inherits=FALSE
                      )) &&
                      size == abs(get(
                        'correction', envir=parent, inherits=FALSE
                      ))
                    ) {{
                      base_counts <- as.integer(get(
                        'tosample', envir=parent, inherits=FALSE
                      ))
                      strata_levels <- names(get(
                        'tosample', envir=parent, inherits=FALSE
                      ))
                      planned_counts <- as.integer(table(factor(
                        strata_local[planned], levels=strata_levels
                      )))
                      delta <- planned_counts - base_counts
                      direction <- sign(get(
                        'correction', envir=parent, inherits=FALSE
                      ))
                      picks <- which(delta == direction)
                      if (length(picks) != size) {{
                        stop('replayed correction does not match strata counts')
                      }}
                      return(as.integer(picks))
                    }}
                    chosen <- planned[planned %in% as.integer(x)]
                    if (length(chosen) != size) {{
                      stop('replayed stratum has the wrong size')
                    }}
                    chosen
                  }}), envir=replay_env)
                  original_seqdist <- TraMineR::seqdist
                  replay_env$seqdist <- function(...) {{
                    iteration <- as.integer(get(
                      'loop', envir=parent.frame(), inherits=TRUE
                    ))
                    args <- list(...)
                    value <- original_seqdist(...)
                    state$seqdist.calls[[iteration]] <- list(
                      names=names(args),
                      method=args$method,
                      norm=args$norm,
                      seqdata=as.matrix(args$seqdata),
                      distance=as.matrix(value)
                    )
                    value
                  }}
                  original_as_clustrange <- WeightedCluster::as.clustrange
                  replay_env$as.clustrange <- function(...) {{
                    iteration <- as.integer(get(
                      'loop', envir=parent.frame(), inherits=TRUE
                    ))
                    args <- list(...)
                    value <- original_as_clustrange(...)
                    state$as.clustrange[[iteration]] <- list(
                      clustering=as.matrix(args[[1]]),
                      stats=as.matrix(value$stats)
                    )
                    value
                  }}
                  boot_local <- WeightedCluster::bootclustrange
                  old_plan <- future::plan(future::sequential)
                  on.exit(future::plan(old_plan), add=TRUE)
                  base_env <- baseenv()
                  patched_bindings <- list(
                    sample=list(env=base_env, value=base::sample),
                    sample.int=list(env=base_env, value=base::sample.int)
                  )
                  replace_binding <- function(name, env, value) {{
                    unlockBinding(name, env)
                    assign(name, value, envir=env)
                    lockBinding(name, env)
                  }}
                  restore_bindings <- function() {{
                    for (name in names(patched_bindings)) {{
                      binding <- patched_bindings[[name]]
                      replace_binding(name, binding$env, binding$value)
                    }}
                  }}
                  replace_binding('sample', base_env, replay_env$sample)
                  replace_binding(
                    'sample.int', base_env, replay_env$sample.int
                  )
                  error <- NULL
                  value <- tryCatch(
                    {{
                      set.seed({int(seed)})
                      suppressMessages(boot_local(
                        object_local,
                        sd_local,
                        seqdist.args=seqdist_local,
                        R=as.integer(nrow(samples_local)),
                        sample.size=as.integer(ncol(samples_local)),
                        parallel=FALSE,
                        progressbar=FALSE,
                        sampling='{str(sampling)}',
                        strata=strata_local
                      ))
                    }},
                    error=function(e) {{
                      error <<- conditionMessage(e)
                      NULL
                    }},
                    finally=restore_bindings()
                  )
                  if (is.null(value)) {{
                    return(list(error=error, state=state))
                  }}
                  for (iteration in seq_len(nrow(samples_local))) {{
                    planned <- as.integer(samples_local[iteration, ])
                    direct_args <- seqdist_local
                    direct_args$seqdata <- sd_local[planned, ]
                    direct_distance <- suppressMessages(do.call(
                      TraMineR::seqdist, direct_args
                    ))
                    direct_range <- WeightedCluster::as.clustrange(
                      clustering_local[planned, , drop=FALSE],
                      diss=direct_distance
                    )
                    state$seqdist.calls[[iteration]] <- list(
                      names=names(direct_args),
                      method=direct_args$method,
                      norm=direct_args$norm,
                      seqdata=as.matrix(direct_args$seqdata),
                      distance=as.matrix(direct_distance)
                    )
                    state$as.clustrange[[iteration]] <- list(
                      clustering=as.matrix(
                        clustering_local[planned, , drop=FALSE]
                      ),
                      stats=as.matrix(direct_range$stats)
                    )
                    replayed_stats <- do.call(rbind, lapply(
                      value$boot,
                      function(block) block[iteration, , drop=FALSE]
                    ))
                    if (!isTRUE(all.equal(
                      unname(as.matrix(direct_range$stats)),
                      unname(as.matrix(replayed_stats)),
                      tolerance=1e-12
                    ))) {{
                      stop('installed bootclustrange did not consume replay samples')
                    }}
                  }}

                  print_visible <- NULL
                  print_error <- NULL
                  print_output <- tryCatch(
                    capture.output(
                      print_visible <- withVisible(print(
                        value,
                        digits={int(print_digits)},
                        bootstat='{str(print_bootstat)}'
                      ))
                    ),
                    error=function(e) {{
                      print_error <<- conditionMessage(e)
                      character()
                    }}
                  )

                  stat_requested <- as.character(plot_stat_local)
                  normalized <- '{str(plot_norm)}'
                  if (length(stat_requested) == 1L) {{
                    if (stat_requested == 'all') {{
                      selected <- value$stats
                      if (normalized == 'none') normalized <- 'range'
                    }} else if (stat_requested == 'noCH') {{
                      selected <- value$stats[
                        , -grep('CH', colnames(value$stats)), drop=FALSE
                      ]
                    }} else {{
                      selected <- value$stats[
                        , stat_requested, drop=FALSE
                      ]
                    }}
                  }} else {{
                    selected <- value$stats[
                      , stat_requested[stat_requested != 'RHC'], drop=FALSE
                    ]
                    if ('RHC' %in% stat_requested) {{
                      selected[, 'RHC'] <- 1 - value$stats[, 'HC']
                    }}
                  }}
                  probabilities <- c(
                    (1-{float(plot_conf_int)!r})/2,
                    1-(1-{float(plot_conf_int)!r})/2
                  )
                  plot_values <- selected
                  lower <- upper <- NULL
                  plot_ci <- '{str(plot_ci_method)}' != 'none' &&
                    !is.null(value$boot)
                  if (plot_ci) {{
                    lower <- selected
                    upper <- selected
                    for (stat_name in colnames(selected)) {{
                      source_name <- if (stat_name == 'RHC') 'HC' else stat_name
                      samples_by_k <- sapply(
                        value$boot, function(block) block[, source_name]
                      )
                      if (stat_name == 'RHC') {{
                        samples_by_k <- 1 - samples_by_k
                      }}
                      normalized_samples <- WeightedCluster:::normalize.values.matrix(
                        samples_by_k, norm=normalized
                      )
                      if ('{str(plot_line)}' == 't0') {{
                        plot_values[, stat_name] <- normalized_samples[1, ]
                      }} else if ('{str(plot_line)}' == 'mean') {{
                        plot_values[, stat_name] <- apply(
                          normalized_samples, 2, mean
                        )
                      }} else {{
                        plot_values[, stat_name] <- apply(
                          normalized_samples, 2, median
                        )
                      }}
                      if ('{str(plot_ci_method)}' == 'norm') {{
                        bounds <- apply(
                          normalized_samples, 2,
                          function(column) mean(column) +
                            stats::qnorm(probabilities) *
                            sqrt(stats::var(column))
                        )
                      }} else {{
                        bounds <- apply(
                          normalized_samples, 2,
                          function(column) stats::quantile(
                            column, probabilities
                          )
                        )
                      }}
                      lower[, stat_name] <- bounds[1, ]
                      upper[, stat_name] <- bounds[2, ]
                    }}
                  }} else {{
                    plot_values <- WeightedCluster:::normalize.values.all(
                      selected, normalized
                    )
                  }}
                  labels <- paste(
                    colnames(plot_values), '(',
                    round(apply(plot_values, 2, min), 2), '/',
                    round(apply(plot_values, 2, max), 2), ')'
                  )

                  plot_error <- NULL
                  filename <- tempfile(fileext='.pdf')
                  grDevices::pdf(filename)
                  tryCatch(
                    plot(
                      value,
                      stat=stat_requested,
                      norm='{str(plot_norm)}',
                      conf.int={float(plot_conf_int)!r},
                      ci.method='{str(plot_ci_method)}',
                      line='{str(plot_line)}'
                    ),
                    error=function(e) plot_error <<- conditionMessage(e)
                  )
                  grDevices::dev.off()
                  unlink(filename)

                  list(
                    error=error,
                    value=value,
                    state=state,
                    print=list(
                      output=print_output,
                      table=if (is.null(print_visible)) NULL else print_visible$value,
                      visible=if (is.null(print_visible)) NA else print_visible$visible,
                      error=print_error
                    ),
                    plot=list(
                      stat.names=colnames(plot_values),
                      values=as.matrix(plot_values),
                      lower=if (is.null(lower)) NULL else as.matrix(lower),
                      upper=if (is.null(upper)) NULL else as.matrix(upper),
                      norm=normalized,
                      labels=labels,
                      xlab='N clusters',
                      ylab='Indicators',
                      error=plot_error
                    )
                  )
                }})
                """)
            contract = names["contract"]
            error = (
                None
                if bool(self.r(f"is.null({contract}$error)")[0])
                else str(self.r(f"as.character({contract}$error)")[0])
            )
            result: dict[str, Any] = {"error": error}
            if error is not None:
                return result

            value = f"{contract}$value"
            result.update(
                {
                    "names": [str(item) for item in self.r(f"names({value})")],
                    "class": [str(item) for item in self.r(f"class({value})")],
                    "kvals": np.asarray(self.r(f"as.integer({value}$kvals)"), dtype=int).copy(),
                    "clustering_names": [
                        str(item) for item in self.r(f"colnames({value}$clustering)")
                    ],
                    "clustering": self._matrix(f"as.matrix({value}$clustering)"),
                    "meant_names": [str(item) for item in self.r(f"colnames({value}$meant)")],
                    "meant_rows": [str(item) for item in self.r(f"rownames({value}$meant)")],
                    "meant": self._matrix(f"as.matrix({value}$meant)"),
                    "stderr": self._matrix(f"as.matrix({value}$stderr)"),
                    "stats": self._matrix(f"as.matrix({value}$stats)"),
                    "boot": [
                        self._matrix(f"as.matrix({value}$boot[[{position}]])")
                        for position in range(1, len(kvals) + 1)
                    ],
                    "boot_names": [
                        [str(item) for item in self.r(f"colnames({value}$boot[[{position}]])")]
                        for position in range(1, len(kvals) + 1)
                    ],
                    "samples_1b": [
                        np.asarray(
                            self.r(f"as.integer({contract}$state$samples[[{iteration}]])"),
                            dtype=int,
                        ).copy()
                        for iteration in range(1, samples.shape[0] + 1)
                    ],
                    "seqdist_calls": [],
                    "as_clustrange_calls": [],
                }
            )
            for iteration in range(1, samples.shape[0] + 1):
                seqdist_call = f"{contract}$state$seqdist.calls[[{iteration}]]"
                result["seqdist_calls"].append(
                    {
                        "names": [str(item) for item in self.r(f"{seqdist_call}$names")],
                        "method": str(self.r(f"as.character({seqdist_call}$method)")[0]),
                        "norm": str(self.r(f"as.character({seqdist_call}$norm)")[0]),
                        "seqdata": np.asarray(
                            self.r(f"as.character(as.matrix({seqdist_call}$seqdata))"),
                            dtype=object,
                        )
                        .reshape((samples.shape[1], Xi.shape[1]), order="F")
                        .copy(),
                        "distance": self._matrix(f"as.matrix({seqdist_call}$distance)"),
                    }
                )
                cluster_call = f"{contract}$state$as.clustrange[[{iteration}]]"
                result["as_clustrange_calls"].append(
                    {
                        "clustering": self._matrix(f"as.matrix({cluster_call}$clustering)").astype(
                            int
                        ),
                        "stats": self._matrix(f"as.matrix({cluster_call}$stats)"),
                    }
                )

            print_expr = f"{contract}$print"
            print_table = self.r(f"{print_expr}$table")
            result["print"] = {
                "output": [str(item) for item in self.r(f"{print_expr}$output")],
                "error": (
                    None
                    if bool(self.r(f"is.null({print_expr}$error)")[0])
                    else str(self.r(f"as.character({print_expr}$error)")[0])
                ),
                "visible": (
                    None
                    if bool(self.r(f"is.na({print_expr}$visible)")[0])
                    else bool(self.r(f"as.logical({print_expr}$visible)")[0])
                ),
                "table": (
                    None
                    if print_table is ro.NULL
                    else self._matrix(f"as.matrix({print_expr}$table)")
                ),
                "row_names": (
                    []
                    if print_table is ro.NULL
                    else [str(item) for item in self.r(f"rownames({print_expr}$table)")]
                ),
                "column_names": (
                    []
                    if print_table is ro.NULL
                    else [str(item) for item in self.r(f"colnames({print_expr}$table)")]
                ),
            }
            plot_expr = f"{contract}$plot"
            result["plot"] = {
                "stat_names": [str(item) for item in self.r(f"{plot_expr}$stat.names")],
                "values": self._matrix(f"as.matrix({plot_expr}$values)"),
                "lower": (
                    None
                    if bool(self.r(f"is.null({plot_expr}$lower)")[0])
                    else self._matrix(f"as.matrix({plot_expr}$lower)")
                ),
                "upper": (
                    None
                    if bool(self.r(f"is.null({plot_expr}$upper)")[0])
                    else self._matrix(f"as.matrix({plot_expr}$upper)")
                ),
                "norm": str(self.r(f"as.character({plot_expr}$norm)")[0]),
                "labels": [str(item) for item in self.r(f"{plot_expr}$labels")],
                "xlab": str(self.r(f"as.character({plot_expr}$xlab)")[0]),
                "ylab": str(self.r(f"as.character({plot_expr}$ylab)")[0]),
                "error": (
                    None
                    if bool(self.r(f"is.null({plot_expr}$error)")[0])
                    else str(self.r(f"as.character({plot_expr}$error)")[0])
                ),
            }
            return result
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def r_bootclustrange_contract(
        self,
        X: np.ndarray,
        cluster_columns: dict[str, np.ndarray],
        *,
        kvals,
        r_iter: int = 4,
        sample_size: int = 10,
        method: str = "LCS",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run R bootclustrange (sampling='random') -> structural summary."""
        Xi = np.asarray(X, dtype=int)
        self._assign_matrix("Xmat", Xi.astype(float))
        self.r("storage.mode(Xmat) <- 'integer'")
        self.r("sd <- suppressMessages(TraMineR::seqdef(Xmat))")
        self.r.assign("Riter", ro.IntVector([int(r_iter)]))
        self.r.assign("ssize", ro.IntVector([int(sample_size)]))
        self.r.assign("meth", ro.StrVector([str(method)]))
        # Build the clustering data.frame column-by-column.
        self.r("cl <- NULL")
        for key, values in cluster_columns.items():
            col = np.asarray(values, dtype=int).reshape(-1)
            self.r.assign("tmpcol", ro.IntVector(col))
            self.r.assign("tmpname", ro.StrVector([str(key)]))
            self.r(
                "if (is.null(cl)) { cl <- data.frame(tmp=as.integer(tmpcol)); "
                "names(cl) <- as.character(tmpname) } else { "
                "cl[[as.character(tmpname)]] <- as.integer(tmpcol) }"
            )
        self.r(f"set.seed({int(seed)})")
        self.r(
            "bc <- suppressMessages(WeightedCluster::bootclustrange(cl, seqdata=sd, "
            "seqdist.args=list(method=as.character(meth)), R=as.integer(Riter), "
            "sample.size=as.integer(ssize), sampling='random'))"
        )
        n_boot = int(self.r("length(bc$boot)")[0])
        boot1_dim = [int(v) for v in self.r("dim(bc$boot[[1]])")]
        return {
            "kvals": np.asarray(self.r("as.integer(bc$kvals)"), dtype=int).reshape(-1),
            "meant_cols": [str(x) for x in self.r("colnames(bc$meant)")],
            "stderr_cols": [str(x) for x in self.r("colnames(bc$stderr)")],
            "n_boot_lists": n_boot,
            "boot1_dim": boot1_dim,  # [r_iter, 10]
            "names": [str(x) for x in self.r("names(bc)")],
        }

    def r_seqnullcqi_contract(
        self,
        X: np.ndarray,
        cluster_columns: dict[str, np.ndarray],
        *,
        r_iter: int = 3,
        model: str = "combined",
        method: str = "LCS",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run R seqnullcqi -> structural summary (hclust path, kmedoid=FALSE)."""
        Xi = np.asarray(X, dtype=int)
        self._assign_matrix("Xmat", Xi.astype(float))
        self.r("storage.mode(Xmat) <- 'integer'")
        self.r("sd <- suppressMessages(TraMineR::seqdef(Xmat))")
        self.r.assign("Riter", ro.IntVector([int(r_iter)]))
        self.r.assign("model", ro.StrVector([str(model)]))
        self.r.assign("meth", ro.StrVector([str(method)]))
        self.r("cl <- NULL")
        for key, values in cluster_columns.items():
            col = np.asarray(values, dtype=int).reshape(-1)
            self.r.assign("tmpcol", ro.IntVector(col))
            self.r.assign("tmpname", ro.StrVector([str(key)]))
            self.r(
                "if (is.null(cl)) { cl <- data.frame(tmp=as.integer(tmpcol)); "
                "names(cl) <- as.character(tmpname) } else { "
                "cl[[as.character(tmpname)]] <- as.integer(tmpcol) }"
            )
        self.r("diss <- suppressMessages(TraMineR::seqdist(sd, method=as.character(meth)))")
        self.r("cr <- as.clustrange(cl, diss=diss, R=1)")
        self.r(f"set.seed({int(seed)})")
        self.r(
            "bcq <- suppressMessages(WeightedCluster::seqnullcqi(sd, cr, "
            "R=as.integer(Riter), model=as.character(model), "
            "seqdist.args=list(method=as.character(meth))))"
        )
        stat_names = [str(x) for x in self.r("names(bcq$stats)")]
        return {
            "names": [str(x) for x in self.r("names(bcq)")],
            "R": int(self.r("bcq$R")[0]),
            "kmedoid": bool(self.r("bcq$kmedoid")[0]),
            "hclust_method": str(self.r("bcq$hclust.method")[0]),
            "stat_names": stat_names,
            "stat_shape": [int(v) for v in self.r("dim(bcq$stats$ASW)")],  # [R, K]
            "seqdata_shape": [int(v) for v in self.r("dim(bcq$seqdata)")],  # [R*N, T]
            "clustrange_stat_cols": [str(x) for x in self.r("colnames(bcq$clustrange$stats)")],
        }

    def rarcat_naive_contract(
        self,
        *,
        formula: str,
        data: Mapping[str, Sequence[Any]],
        categorical: Mapping[str, Sequence[Any]] | None = None,
        logical: Sequence[str] = (),
        row_names: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Return the installed RARCAT model-frame, GLM, and margins contract."""
        token = uuid.uuid4().hex
        prefix = f"wc_phase3a_naive_{token}"
        data_name = f"{prefix}_data"
        formula_name = f"{prefix}_formula"
        column_names_name = f"{prefix}_column_names"
        row_names_name = f"{prefix}_row_names"
        contract_name = f"{prefix}_contract"
        column_objects: list[str] = []
        cleanup = [
            data_name,
            formula_name,
            column_names_name,
            row_names_name,
            contract_name,
        ]
        categorical = {} if categorical is None else dict(categorical)
        logical_names = {str(name) for name in logical}
        column_names = [str(name) for name in data]
        if not column_names:
            raise ValueError("data must contain at least one column")
        column_lengths = {len(data[name]) for name in data}
        if len(column_lengths) != 1:
            raise ValueError("all data columns must have the same length")
        n_rows = column_lengths.pop()
        if row_names is None:
            row_names = [str(index + 1) for index in range(n_rows)]
        if len(row_names) != n_rows:
            raise ValueError("row_names must match the data row count")

        def _character_vector(values: Sequence[Any]):
            return ro.StrVector(
                [
                    (
                        ro.NA_Character
                        if value is None or (isinstance(value, float) and np.isnan(value))
                        else str(value)
                    )
                    for value in values
                ]
            )

        try:
            for index, column_name in enumerate(column_names):
                object_name = f"{prefix}_column_{index}"
                cleanup.append(object_name)
                values = list(data[column_name])
                if column_name in categorical:
                    levels_name = f"{prefix}_levels_{index}"
                    cleanup.append(levels_name)
                    self.r.assign(object_name, _character_vector(values))
                    self.r.assign(
                        levels_name,
                        ro.StrVector([str(value) for value in categorical[column_name]]),
                    )
                    self.r(
                        f"{object_name} <- factor({object_name}, "
                        f"levels={levels_name}, exclude=NULL)"
                    )
                elif column_name in logical_names:
                    self.r.assign(
                        object_name,
                        ro.BoolVector(
                            [
                                (
                                    ro.NA_Logical
                                    if value is None
                                    or (isinstance(value, float) and np.isnan(value))
                                    else bool(value)
                                )
                                for value in values
                            ]
                        ),
                    )
                else:
                    non_missing = [
                        value
                        for value in values
                        if value is not None and not (isinstance(value, float) and np.isnan(value))
                    ]
                    numeric = all(
                        isinstance(value, (int, float, np.integer, np.floating))
                        and not isinstance(value, (bool, np.bool_))
                        for value in non_missing
                    )
                    if numeric:
                        self.r.assign(
                            object_name,
                            ro.FloatVector(
                                [
                                    (
                                        ro.NA_Real
                                        if value is None
                                        or (isinstance(value, float) and np.isnan(value))
                                        else float(value)
                                    )
                                    for value in values
                                ]
                            ),
                        )
                    else:
                        self.r.assign(object_name, _character_vector(values))
                column_objects.append(object_name)

            self.r.assign(column_names_name, ro.StrVector(column_names))
            self.r.assign(row_names_name, ro.StrVector([str(value) for value in row_names]))
            self.r.assign(formula_name, ro.StrVector([str(formula)]))
            self.r(
                f"{data_name} <- data.frame("
                + ",".join(column_objects)
                + ", check.names=FALSE, stringsAsFactors=FALSE)"
            )
            self.r(f"names({data_name}) <- {column_names_name}")
            self.r(f"row.names({data_name}) <- {row_names_name}")
            self.r(f"""
                {contract_name} <- local({{
                  capture_condition <- function(fun) {{
                    captured_warnings <- character()
                    captured_error <- NULL
                    value <- tryCatch(
                      withCallingHandlers(
                        fun(),
                        warning=function(w) {{
                          captured_warnings <<- c(captured_warnings, conditionMessage(w))
                          invokeRestart("muffleWarning")
                        }}
                      ),
                      error=function(e) {{
                        captured_error <<- conditionMessage(e)
                        NULL
                      }}
                    )
                    list(value=value, warnings=captured_warnings, error=captured_error)
                  }}
                  formula_object <- as.formula(as.character({formula_name}))
                  model_frame_capture <- capture_condition(
                    function() model.frame(formula_object, {data_name})
                  )
                  if (is.null(model_frame_capture$value)) {{
                    list(
                      error=model_frame_capture$error,
                      warnings=model_frame_capture$warnings,
                      formula=formula_object
                    )
                  }} else {{
                    source_model_frame <- model_frame_capture$value
                    modelDF <- source_model_frame
                    lhs <- deparse(formula_object[[2]])
                    rhs <- deparse(formula_object[[3]])
                    clustering <- modelDF[, lhs]
                    cluster_names <- as.character(unique(clustering))
                    membership_formula <- as.formula(paste("membership~", rhs))
                    fits <- setNames(vector("list", length(cluster_names)), cluster_names)
                    for (cluster_name in cluster_names) {{
                      modelDF$membership <- clustering == cluster_name
                      fit_capture <- capture_condition(
                        function() glm(membership_formula, modelDF, family="binomial")
                      )
                      if (is.null(fit_capture$value)) {{
                        fits[[cluster_name]] <- list(
                          fit=NULL,
                          margins=NULL,
                          warnings=fit_capture$warnings,
                          error=fit_capture$error
                        )
                      }} else {{
                        margins_capture <- capture_condition(
                          function() summary(margins::margins(fit_capture$value))
                        )
                        fits[[cluster_name]] <- list(
                          fit=fit_capture$value,
                          margins=margins_capture$value,
                          warnings=c(fit_capture$warnings, margins_capture$warnings),
                          error=margins_capture$error
                        )
                      }}
                    }}
                    rarcat_capture <- capture_condition(
                      function() WeightedCluster::rarcat(
                        formula_object,
                        {data_name},
                        diag(nrow({data_name})),
                        robust=FALSE
                      )
                    )
                    list(
                      error=NULL,
                      warnings=model_frame_capture$warnings,
                      formula=formula_object,
                      lhs=lhs,
                      rhs=rhs,
                      model_frame=source_model_frame,
                      source_row_names=row.names({data_name}),
                      cluster_names=cluster_names,
                      fits=fits,
                      rarcat=rarcat_capture
                    )
                  }}
                }})
                """)

            versions = {
                package: str(self.r(f'as.character(packageVersion("{package}"))')[0])
                for package in ("WeightedCluster", "TraMineR", "margins", "lme4")
            }
            error = (
                None
                if bool(self.r(f"is.null({contract_name}$error)")[0])
                else str(self.r(f"as.character({contract_name}$error)")[0])
            )
            payload: dict[str, Any] = {
                "versions": versions,
                "error": error,
                "formula": str(formula),
                "warnings": [str(value) for value in self.r(f"{contract_name}$warnings")],
                "glm_control": {
                    "epsilon": float(self.r("glm.control()$epsilon")[0]),
                    "maxit": int(self.r("glm.control()$maxit")[0]),
                    "criterion": ("abs(deviance - previous) / (0.1 + abs(deviance))"),
                },
            }
            if error is not None:
                return payload

            model_frame = f"{contract_name}$model_frame"
            retained_rows = [str(value) for value in self.r(f"row.names({model_frame})")]
            source_rows = [str(value) for value in self.r(f"{contract_name}$source_row_names")]
            model_frame_columns = [str(value) for value in self.r(f"names({model_frame})")]
            factor_levels = {
                column: [
                    str(value)
                    for value in self.r(f"as.character(levels({model_frame}[[{index}]]))")
                ]
                for index, column in enumerate(model_frame_columns, start=1)
            }
            first_fit = f"{contract_name}$fits[[1]]$fit"
            design = f"model.matrix({first_fit})"
            contrast_names = [
                str(value) for value in self.r(f"as.character(names(attr({design}, 'contrasts')))")
            ]
            contrasts = {
                name: str(self.r(f"as.character(attr({design}, 'contrasts')[[{index}]])")[0])
                for index, name in enumerate(contrast_names, start=1)
            }
            fit_xlevel_names = [str(value) for value in self.r(f"names({first_fit}$xlevels)")]
            generated_level_names = list(dict.fromkeys([*contrast_names, *fit_xlevel_names]))
            level_expressions = {
                name: (
                    f"{first_fit}$xlevels[[{name!r}]]"
                    if name in fit_xlevel_names
                    else (
                        f"levels(factor({model_frame}[[match("
                        f"{name!r}, names({model_frame}))]]))"
                    )
                )
                for name in generated_level_names
            }
            generated_factor_levels = {
                name: [str(value) for value in self.r(f"as.character({level_expressions[name]})")]
                for name in generated_level_names
            }
            contrast_matrices: dict[str, dict[str, Any]] = {}
            for name in generated_level_names:
                matrix_expression = (
                    "contrasts(factor("
                    f"{model_frame}[[match({name!r}, names({model_frame}))]], "
                    f"levels={level_expressions[name]}), contrasts=TRUE)"
                )
                contrast_matrices[name] = {
                    "row_names": [str(value) for value in self.r(f"rownames({matrix_expression})")],
                    "column_names": [
                        str(value) for value in self.r(f"colnames({matrix_expression})")
                    ],
                    "values": self._matrix(matrix_expression).tolist(),
                }
            payload.update(
                {
                    "lhs": str(self.r(f"{contract_name}$lhs")[0]),
                    "rhs": str(self.r(f"{contract_name}$rhs")[0]),
                    "runtime": {
                        "LC_COLLATE": str(self.r("Sys.getlocale('LC_COLLATE')")[0]),
                        "OutDec": str(self.r("getOption('OutDec')")[0]),
                        "digits": int(self.r("getOption('digits')")[0]),
                        "scipen": int(self.r("getOption('scipen')")[0]),
                    },
                    "cluster_names": [
                        str(value) for value in self.r(f"{contract_name}$cluster_names")
                    ],
                    "model_frame": {
                        "column_names": model_frame_columns,
                        "column_classes": {
                            column: [
                                str(value) for value in self.r(f"class({model_frame}[[{index}]])")
                            ]
                            for index, column in enumerate(model_frame_columns, start=1)
                        },
                        "row_names": retained_rows,
                        "omitted_row_names": [
                            row for row in source_rows if row not in set(retained_rows)
                        ],
                        "response": [
                            str(value) for value in self.r(f"as.character({model_frame}[[1]])")
                        ],
                        "term_labels": [
                            str(value)
                            for value in self.r(f"attr(terms({model_frame}), 'term.labels')")
                        ],
                        "intercept": int(self.r(f"attr(terms({model_frame}), 'intercept')")[0]),
                        "factor_levels": factor_levels,
                        "design_column_names": [
                            str(value) for value in self.r(f"colnames({design})")
                        ],
                        "contrasts": contrasts,
                        "generated_factor_levels": generated_factor_levels,
                        "contrast_matrices": contrast_matrices,
                    },
                }
            )

            cluster_payload: dict[str, Any] = {}
            all_factor_names: list[str] = []
            for position, cluster_name in enumerate(payload["cluster_names"], start=1):
                item = f"{contract_name}$fits[[{position}]]"
                cluster_error = (
                    None
                    if bool(self.r(f"is.null({item}$error)")[0])
                    else str(self.r(f"as.character({item}$error)")[0])
                )
                contract: dict[str, Any] = {
                    "error": cluster_error,
                    "warnings": [str(value) for value in self.r(f"{item}$warnings")],
                }
                if cluster_error is None:
                    fit = f"{item}$fit"
                    margins = f"{item}$margins"
                    ame_factor_names = [
                        str(value) for value in self.r(f"as.character({margins}$factor)")
                    ]
                    for factor_name in ame_factor_names:
                        if factor_name not in all_factor_names:
                            all_factor_names.append(factor_name)
                    contract.update(
                        {
                            "converged": bool(self.r(f"{fit}$converged")[0]),
                            "nobs": int(self.r(f"stats::nobs({fit})")[0]),
                            "rank": int(self.r(f"{fit}$rank")[0]),
                            "df_residual": int(self.r(f"{fit}$df.residual")[0]),
                            "iterations": int(self.r(f"{fit}$iter")[0]),
                            "deviance": float(self.r(f"{fit}$deviance")[0]),
                            "null_deviance": float(self.r(f"{fit}$null.deviance")[0]),
                            "dispersion": float(self.r(f"summary({fit})$dispersion")[0]),
                            "coefficient_names": [
                                str(value) for value in self.r(f"names(coef({fit}))")
                            ],
                            "coefficients": np.asarray(
                                self.r(f"as.numeric(coef({fit}))"), dtype=float
                            )
                            .reshape(-1)
                            .copy(),
                            "covariance": self._matrix(f"as.matrix(vcov({fit}))"),
                            "covariance_unscaled": self._matrix(
                                f"as.matrix(summary({fit})$cov.unscaled)"
                            ),
                            "covariance_scaled": self._matrix(
                                f"as.matrix(summary({fit})$cov.scaled)"
                            ),
                            "design_column_names": [
                                str(value) for value in self.r(f"colnames(model.matrix({fit}))")
                            ],
                            "ame_factor_names": ame_factor_names,
                            "ames": np.asarray(self.r(f"as.numeric({margins}$AME)"), dtype=float)
                            .reshape(-1)
                            .copy(),
                            "standard_errors": np.asarray(
                                self.r(f"as.numeric({margins}$SE)"), dtype=float
                            )
                            .reshape(-1)
                            .copy(),
                            "z_values": np.asarray(self.r(f"as.numeric({margins}$z)"), dtype=float)
                            .reshape(-1)
                            .copy(),
                            "p_values": np.asarray(self.r(f"as.numeric({margins}$p)"), dtype=float)
                            .reshape(-1)
                            .copy(),
                        }
                    )
                cluster_payload[str(cluster_name)] = contract
            payload["factor_names"] = all_factor_names
            payload["clusters"] = cluster_payload

            rarcat_capture = f"{contract_name}$rarcat"
            rarcat_error = (
                None
                if bool(self.r(f"is.null({rarcat_capture}$error)")[0])
                else str(self.r(f"as.character({rarcat_capture}$error)")[0])
            )
            rarcat_payload: dict[str, Any] = {
                "error": rarcat_error,
                "warnings": [str(value) for value in self.r(f"{rarcat_capture}$warnings")],
            }
            if rarcat_error is None:
                value = f"{rarcat_capture}$value"
                rarcat_payload.update(
                    {
                        "class": [str(item) for item in self.r(f"class({value})")],
                        "names": [str(item) for item in self.r(f"names({value})")],
                        "cluster_names": [str(item) for item in self.r(f"{value}$clusterNames")],
                        "factor_names": [str(item) for item in self.r(f"{value}$factorName")],
                    }
                )
            payload["rarcat"] = rarcat_payload
            return payload
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def rarcat_numeric_label_contract(self, values: Sequence[float]) -> dict[str, Any]:
        """Return live-R numeric string rendering and factor-level evidence."""
        token = uuid.uuid4().hex
        values_name = f"wc_phase3a_labels_values_{token}"
        contract_name = f"wc_phase3a_labels_contract_{token}"
        cleanup = [values_name, contract_name]
        try:
            self.r.assign(values_name, ro.FloatVector([float(value) for value in values]))
            self.r(f"""
                {contract_name} <- local({{
                  values <- {values_name}
                  list(
                    rendered=as.character(values),
                    is_nan=is.nan(values),
                    is_na=is.na(values),
                    factor_levels=levels(factor(values, exclude=NULL)),
                    LC_COLLATE=Sys.getlocale("LC_COLLATE"),
                    OutDec=getOption("OutDec"),
                    digits=getOption("digits"),
                    scipen=getOption("scipen")
                  )
                }})
                """)
            is_nan = [bool(value) for value in self.r(f"{contract_name}$is_nan")]
            is_na = [bool(value) for value in self.r(f"{contract_name}$is_na")]
            rendered_values = list(self.r(f"{contract_name}$rendered"))
            rendered = [
                "NaN" if nan else None if missing else str(value)
                for value, nan, missing in zip(
                    rendered_values,
                    is_nan,
                    is_na,
                    strict=True,
                )
            ]
            return {
                "rendered": rendered,
                "is_nan": is_nan,
                "is_na": is_na,
                "factor_levels": [str(value) for value in self.r(f"{contract_name}$factor_levels")],
                "runtime": {
                    "LC_COLLATE": str(self.r(f"{contract_name}$LC_COLLATE")[0]),
                    "OutDec": str(self.r(f"{contract_name}$OutDec")[0]),
                    "digits": int(self.r(f"{contract_name}$digits")[0]),
                    "scipen": int(self.r(f"{contract_name}$scipen")[0]),
                },
            }
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in cleanup)
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def rarcat_source_contract(self) -> dict[str, Any]:
        """Return machine-readable contradictions in the installed R sources."""
        token = uuid.uuid4().hex
        contract_name = f"wc_phase3a_source_{token}"
        try:
            self.r(f"""
                {contract_name} <- local({{
                  rarcat_body <- paste(deparse(body(WeightedCluster::rarcat)), collapse="\\n")
                  amemat_body <- paste(
                    deparse(body(getFromNamespace("amemat", "WeightedCluster"))),
                    collapse="\\n"
                  )
                  rarcat_compact <- gsub("[[:space:]]", "", rarcat_body)
                  help_text <- paste(
                    capture.output(
                      tools::Rd2txt(
                        utils:::.getHelpFile(help("rarcat", package="WeightedCluster"))
                      )
                    ),
                    collapse="\\n"
                  )
                  vignette_path <- system.file(
                    "doc", "RARCAT.Rmd", package="WeightedCluster"
                  )
                  vignette_text <- if (nzchar(vignette_path)) {{
                    paste(readLines(vignette_path, warn=FALSE), collapse="\\n")
                  }} else {{
                    ""
                  }}
                  formal_values <- lapply(
                    formals(WeightedCluster::rarcat),
                    function(value) paste(deparse(value), collapse=" ")
                  )
                  fisher_assignment <- regexpr(
                    "output$coefficients[1]<-tanh(output$coefficients[1])",
                    rarcat_compact,
                    fixed=TRUE
                  )[[1]]
                  fisher_se <- regexpr(
                    paste0(
                      "output$coefficients[2]<-output$coefficients[2]*",
                      "(1-tanh(output$coefficients[1])^2)"
                    ),
                    rarcat_compact,
                    fixed=TRUE
                  )[[1]]
                  list(
                    formals=formal_values,
                    help=list(
                      kmedoid_mentions_character=grepl(
                        "clustering algorithm as a character string",
                        help_text,
                        fixed=TRUE
                      ),
                      kmedoid_mentions_pam_default=grepl(
                        'By default "pam"',
                        help_text,
                        fixed=TRUE
                      ),
                      cqi_mentions_ch_default=(
                        grepl('"CH"', help_text, fixed=TRUE) &&
                        grepl("Calinski-Harabasz", help_text, fixed=TRUE) &&
                        grepl("index) by default", help_text, fixed=TRUE)
                      )
                    ),
                    vignette=list(
                      hc_described_as_maximized=grepl(
                        "maximizing the HC index",
                        vignette_text,
                        fixed=TRUE
                      )
                    ),
                    amemat=list(
                      uses_which_min=grepl("which.min", amemat_body, fixed=TRUE)
                    ),
                    rarcat=list(
                      normalizes_inverse_variance_weights=grepl(
                        "prep$stweight<-prep$weight/mean(prep$weight)",
                        rarcat_compact,
                        fixed=TRUE
                      ),
                      crosses_id_and_bootstrap=grepl(
                        "ame~(1|id)+(1|bootstrap)",
                        rarcat_compact,
                        fixed=TRUE
                      ),
                      allocates_bootstrap_ranef=grepl(
                        "bootstrap.ranef<-matrix(",
                        rarcat_compact,
                        fixed=TRUE
                      ),
                      returns_bootstrap_ranef=grepl(
                        "ret$bootstrap.ranef",
                        rarcat_compact,
                        fixed=TRUE
                      ),
                      fisher_se_reapplies_tanh=(
                        fisher_assignment > 0 &&
                        fisher_se > fisher_assignment
                      )
                    )
                  )
                }})
                """)
            formals_expr = f"{contract_name}$formals"
            formal_names = [str(value) for value in self.r(f"names({formals_expr})")]
            return {
                "versions": {
                    package: str(self.r(f'as.character(packageVersion("{package}"))')[0])
                    for package in ("WeightedCluster", "TraMineR", "margins", "lme4")
                },
                "formals": {
                    name: str(self.r(f"as.character({formals_expr}[[{index}]])")[0])
                    for index, name in enumerate(formal_names, start=1)
                },
                "help": {
                    key: bool(self.r(f"{contract_name}$help${key}")[0])
                    for key in (
                        "kmedoid_mentions_character",
                        "kmedoid_mentions_pam_default",
                        "cqi_mentions_ch_default",
                    )
                },
                "vignette": {
                    "hc_described_as_maximized": bool(
                        self.r(f"{contract_name}$vignette$hc_described_as_maximized")[0]
                    )
                },
                "amemat": {
                    "uses_which_min": bool(self.r(f"{contract_name}$amemat$uses_which_min")[0])
                },
                "rarcat": {
                    key: bool(self.r(f"{contract_name}$rarcat${key}")[0])
                    for key in (
                        "normalizes_inverse_variance_weights",
                        "crosses_id_and_bootstrap",
                        "allocates_bootstrap_ranef",
                        "returns_bootstrap_ranef",
                        "fisher_se_reapplies_tanh",
                    )
                },
            }
        finally:
            self.r(
                f"rm(list=intersect('{contract_name}', ls(envir=.GlobalEnv)), " "envir=.GlobalEnv)"
            )

    def rarcat_crossed_pool_contract(self, frame: Any) -> dict[str, Any]:
        """Fit RARCAT's isolated weighted crossed random-intercept model."""
        required = ("id", "bootstrap", "ame", "sterror")
        if not all(column in frame for column in required):
            raise ValueError("frame must contain id, bootstrap, ame, and sterror")
        token = uuid.uuid4().hex
        prefix = f"wc_phase3a_crossed_{token}"
        names = {
            "id": f"{prefix}_id",
            "id_levels": f"{prefix}_id_levels",
            "bootstrap": f"{prefix}_bootstrap",
            "bootstrap_levels": f"{prefix}_bootstrap_levels",
            "ame": f"{prefix}_ame",
            "sterror": f"{prefix}_sterror",
            "rows": f"{prefix}_rows",
            "data": f"{prefix}_data",
            "contract": f"{prefix}_contract",
        }

        def _is_missing(value: Any) -> bool:
            missing = pd.isna(value)
            return bool(missing) if isinstance(missing, (bool, np.bool_)) else False

        def _string_vector(values):
            return ro.StrVector(
                [ro.NA_Character if _is_missing(value) else str(value) for value in values]
            )

        def _float_vector(values):
            return ro.FloatVector(
                [(ro.NA_Real if _is_missing(value) else float(value)) for value in values]
            )

        def _assign_group(column: str) -> None:
            series = frame[column]
            object_name = names[column]
            levels_name = names[f"{column}_levels"]
            if isinstance(series.dtype, pd.CategoricalDtype):
                self.r.assign(object_name, _string_vector(series))
                self.r.assign(
                    levels_name,
                    ro.StrVector([str(value) for value in series.cat.categories]),
                )
                ordered = "TRUE" if series.cat.ordered else "FALSE"
                self.r(
                    f"{object_name} <- factor({object_name}, levels={levels_name}, "
                    f"ordered={ordered})"
                )
            elif pd.api.types.is_bool_dtype(series.dtype):
                self.r.assign(
                    object_name,
                    ro.BoolVector(
                        [ro.NA_Logical if _is_missing(value) else bool(value) for value in series]
                    ),
                )
            elif pd.api.types.is_integer_dtype(series.dtype):
                self.r.assign(
                    object_name,
                    ro.IntVector(
                        [ro.NA_Integer if _is_missing(value) else int(value) for value in series]
                    ),
                )
            elif pd.api.types.is_numeric_dtype(series.dtype):
                self.r.assign(object_name, _float_vector(series))
            else:
                self.r.assign(object_name, _string_vector(series))

        try:
            _assign_group("id")
            _assign_group("bootstrap")
            self.r.assign(names["ame"], _float_vector(frame["ame"]))
            self.r.assign(names["sterror"], _float_vector(frame["sterror"]))
            self.r.assign(names["rows"], ro.StrVector([str(value) for value in frame.index]))
            self.r(f"""
                {names["data"]} <- data.frame(
                  id={names["id"]},
                  bootstrap={names["bootstrap"]},
                  ame={names["ame"]},
                  sterror={names["sterror"]},
                  row.names={names["rows"]},
                  stringsAsFactors=FALSE
                )
                {names["contract"]} <- local({{
                  capture_condition <- function(fun) {{
                    captured_warnings <- character()
                    captured_error <- NULL
                    value <- tryCatch(
                      withCallingHandlers(
                        fun(),
                        warning=function(w) {{
                          captured_warnings <<- c(captured_warnings, conditionMessage(w))
                          invokeRestart("muffleWarning")
                        }}
                      ),
                      error=function(e) {{
                        captured_error <<- conditionMessage(e)
                        NULL
                      }}
                    )
                    list(value=value, warnings=captured_warnings, error=captured_error)
                  }}
                  prep <- subset({names["data"]}, !is.na(ame))
                  prep$weight <- 1 / prep$sterror^2
                  prep$stweight <- prep$weight / mean(prep$weight)
                  fit <- capture_condition(
                    function() lme4::lmer(
                      ame ~ (1 | id) + (1 | bootstrap),
                      weights=prep$stweight,
                      data=prep,
                      control=lme4::lmerControl()
                    )
                  )
                  list(
                    error=fit$error,
                    warnings=fit$warnings,
                    prep=prep,
                    fit=fit$value
                  )
                }})
                """)
            contract = names["contract"]
            error = (
                None
                if bool(self.r(f"is.null({contract}$error)")[0])
                else str(self.r(f"as.character({contract}$error)")[0])
            )
            payload: dict[str, Any] = {
                "versions": {
                    package: str(self.r(f'as.character(packageVersion("{package}"))')[0])
                    for package in ("WeightedCluster", "TraMineR", "margins", "lme4")
                },
                "error": error,
                "warnings": [str(value) for value in self.r(f"{contract}$warnings")],
                "formula": "ame ~ (1 | id) + (1 | bootstrap)",
                "source_row_names": [str(value) for value in frame.index],
            }
            prep = f"{contract}$prep"
            included_rows = [str(value) for value in self.r(f"row.names({prep})")]
            included_set = set(included_rows)
            payload.update(
                {
                    "included_row_names": included_rows,
                    "omitted_row_names": [
                        str(value) for value in frame.index if str(value) not in included_set
                    ],
                    "normalized_weights": np.asarray(
                        self.r(f"as.numeric({prep}$stweight)"), dtype=float
                    )
                    .reshape(-1)
                    .copy(),
                }
            )
            group_metadata: dict[str, Any] = {}
            for column in ("id", "bootstrap"):
                source_column = f"{names['data']}${column}"
                if bool(self.r(f"is.factor({source_column})")[0]):
                    levels_expression = f"levels(droplevels({source_column}))"
                else:
                    levels_expression = f"levels(factor({source_column}))"
                values: list[str | None] = []
                for position in range(1, len(frame) + 1):
                    if bool(self.r(f"is.na({source_column}[{position}])")[0]):
                        values.append(None)
                    else:
                        values.append(str(self.r(f"as.character({source_column}[{position}])")[0]))
                group_metadata[column] = {
                    "class": [str(value) for value in self.r(f"class({source_column})")],
                    "type": str(self.r(f"typeof({source_column})")[0]),
                    "ordered": bool(self.r(f"is.ordered({source_column})")[0]),
                    "factor_levels": [
                        str(value) for value in self.r(f"as.character({levels_expression})")
                    ],
                    "values": values,
                    "missing_count": int(self.r(f"sum(is.na({source_column}))")[0]),
                }
            payload["group_metadata"] = group_metadata
            if error is not None:
                return payload

            fit = f"{contract}$fit"
            fitted_rows = [str(value) for value in self.r(f"row.names(model.frame({fit}))")]
            fitted_set = set(fitted_rows)
            prep_rows = [str(value) for value in self.r(f"row.names({prep})")]
            fitted_positions = [
                position + 1
                for position, row_name in enumerate(prep_rows)
                if row_name in fitted_set
            ]
            payload["included_row_names"] = fitted_rows
            payload["omitted_row_names"] = [
                str(value) for value in frame.index if str(value) not in fitted_set
            ]
            payload["normalized_weights"] = (
                np.asarray(
                    self.r(
                        f"as.numeric({prep}$stweight[c({','.join(map(str, fitted_positions))})])"
                    ),
                    dtype=float,
                )
                .reshape(-1)
                .copy()
            )
            convergence_messages = [
                str(value)
                for value in self.r(f"as.character(unlist({fit}@optinfo$conv$lme4$messages))")
            ]
            optimizer_code = int(self.r(f"as.integer({fit}@optinfo$conv$opt)")[0])
            fixed_table = f"coef(summary({fit}))"
            id_random = f"lme4::ranef({fit})$id"
            bootstrap_random = f"lme4::ranef({fit})$bootstrap"
            payload.update(
                {
                    "nobs": int(self.r(f"stats::nobs({fit})")[0]),
                    "fixed_effect_names": [
                        str(value) for value in self.r(f"row.names({fixed_table})")
                    ],
                    "fixed_intercept": float(self.r(f"{fixed_table}[1, 1]")[0]),
                    "fixed_standard_error": float(self.r(f"{fixed_table}[1, 2]")[0]),
                    "observation_variance": float(
                        self.r(f"as.numeric(lme4::VarCorr({fit})$id)")[0]
                    ),
                    "bootstrap_variance": float(
                        self.r(f"as.numeric(lme4::VarCorr({fit})$bootstrap)")[0]
                    ),
                    "residual_variance": float(self.r(f"sigma({fit})^2")[0]),
                    "observation_random_effect_names": [
                        str(value) for value in self.r(f"row.names({id_random})")
                    ],
                    "observation_random_effects": np.asarray(
                        self.r(f"as.numeric({id_random}[, 1])"), dtype=float
                    )
                    .reshape(-1)
                    .copy(),
                    "bootstrap_random_effect_names": [
                        str(value) for value in self.r(f"row.names({bootstrap_random})")
                    ],
                    "bootstrap_random_effects": np.asarray(
                        self.r(f"as.numeric({bootstrap_random}[, 1])"), dtype=float
                    )
                    .reshape(-1)
                    .copy(),
                    "optimizer_code": optimizer_code,
                    "convergence_messages": convergence_messages,
                    "converged": optimizer_code == 0 and not convergence_messages,
                    "singular": bool(self.r(f"lme4::isSingular({fit})")[0]),
                    "reml": bool(self.r(f"lme4::isREML({fit})")[0]),
                    "log_likelihood": float(self.r(f"as.numeric(logLik({fit}))")[0]),
                    "reml_criterion": float(self.r(f"as.numeric(lme4::REMLcrit({fit}))")[0]),
                    "theta": np.asarray(
                        self.r(f"as.numeric(lme4::getME({fit}, 'theta'))"), dtype=float
                    )
                    .reshape(-1)
                    .copy(),
                }
            )
            derivatives = f"{fit}@optinfo$derivs"
            has_derivatives = not bool(self.r(f"is.null({derivatives})")[0])
            if has_derivatives:
                gradient = (
                    np.asarray(
                        self.r(f"as.numeric({derivatives}$gradient)"),
                        dtype=float,
                    )
                    .reshape(-1)
                    .copy()
                )
                hessian = self._matrix(f"{derivatives}$Hessian")
                if np.all(np.isfinite(hessian)):
                    hessian_eigenvalues = (
                        np.asarray(
                            self.r(
                                f"as.numeric(eigen({derivatives}$Hessian, "
                                "symmetric=TRUE, only.values=TRUE)$values)"
                            ),
                            dtype=float,
                        )
                        .reshape(-1)
                        .copy()
                    )
                else:
                    hessian_eigenvalues = None
            else:
                gradient = None
                hessian = None
                hessian_eigenvalues = None
            raw_message = (
                None
                if bool(self.r(f"is.null({fit}@optinfo$message)")[0])
                else str(self.r(f"as.character({fit}@optinfo$message)")[0])
            )
            payload["architecture"] = {
                "optimizer": str(self.r(f"as.character({fit}@optinfo$optimizer)")[0]),
                "algorithm": str(
                    self.r(
                        "as.character(environment(lme4:::nloptwrap)" "$defaultControl$algorithm)"
                    )[0]
                ),
                "control": {
                    "xtol_abs": float(
                        self.r(
                            "as.numeric(environment(lme4:::nloptwrap)" "$defaultControl$xtol_abs)"
                        )[0]
                    ),
                    "ftol_abs": float(
                        self.r(
                            "as.numeric(environment(lme4:::nloptwrap)" "$defaultControl$ftol_abs)"
                        )[0]
                    ),
                    "maxeval": int(
                        self.r(
                            "as.integer(environment(lme4:::nloptwrap)" "$defaultControl$maxeval)"
                        )[0]
                    ),
                },
                "restart_edge": bool(self.r("isTRUE(lme4::lmerControl()$restart_edge)")[0]),
                "boundary_tolerance": float(
                    self.r("as.numeric(lme4::lmerControl()$boundary.tol)")[0]
                ),
                "singular_tolerance": float(self.r("lme4:::getSingTol()")[0]),
                "theta_lower": [float(value) for value in self.r(f"as.numeric({fit}@lower)")],
                "theta_upper": [
                    float(value) for value in self.r(f"as.numeric(attr({fit}, 'upper'))")
                ],
                "theta_names": [
                    str(value) for value in self.r(f"names(lme4::getME({fit}, 'theta'))")
                ],
                "function_evaluations": int(self.r(f"as.integer({fit}@optinfo$feval)")[0]),
                "raw_message": raw_message,
                "gradient": gradient,
                "hessian": hessian,
                "hessian_eigenvalues": hessian_eigenvalues,
            }
            payload["group_metadata"]["id"]["random_effect_names"] = list(
                payload["observation_random_effect_names"]
            )
            payload["group_metadata"]["bootstrap"]["random_effect_names"] = list(
                payload["bootstrap_random_effect_names"]
            )
            return payload
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in names.values())
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def rarcat_fisher_contract(
        self,
        *,
        values: np.ndarray,
        coefficient: float,
        standard_error: float,
        observation_variance: float,
        bootstrap_variance: float,
    ) -> dict[str, Any]:
        """Apply the installed RARCAT Fisher-transform statements in order."""
        token = uuid.uuid4().hex
        names = {
            "values": f"wc_phase3a_fisher_values_{token}",
            "coefficient": f"wc_phase3a_fisher_coefficient_{token}",
            "standard_error": f"wc_phase3a_fisher_standard_error_{token}",
            "observation_variance": f"wc_phase3a_fisher_observation_variance_{token}",
            "bootstrap_variance": f"wc_phase3a_fisher_bootstrap_variance_{token}",
            "contract": f"wc_phase3a_fisher_contract_{token}",
        }
        try:
            self.r.assign(
                names["values"],
                ro.FloatVector(np.asarray(values, dtype=float).reshape(-1)),
            )
            for key, value in (
                ("coefficient", coefficient),
                ("standard_error", standard_error),
                ("observation_variance", observation_variance),
                ("bootstrap_variance", bootstrap_variance),
            ):
                self.r.assign(names[key], ro.FloatVector([float(value)]))
            self.r(f"""
                {names["contract"]} <- local({{
                  captured_warnings <- character()
                  captured_error <- NULL
                  value <- tryCatch(
                    withCallingHandlers(
                      {{
                        forward <- atanh({names["values"]})
                        transformed_coefficient <- {names["coefficient"]}
                        transformed_standard_error <- {names["standard_error"]}
                        transformed_observation_variance <- {names["observation_variance"]}
                        transformed_bootstrap_variance <- {names["bootstrap_variance"]}
                        transformed_coefficient <- tanh(transformed_coefficient)
                        transformed_standard_error <- transformed_standard_error * (
                          1 - tanh(transformed_coefficient)^2
                        )
                        transformed_observation_variance <- tanh(
                          transformed_observation_variance
                        )
                        transformed_bootstrap_variance <- tanh(
                          transformed_bootstrap_variance
                        )
                        list(
                          forward=forward,
                          coefficient=transformed_coefficient,
                          standard_error=transformed_standard_error,
                          observation_standard_deviation=sqrt(
                            transformed_observation_variance
                          ),
                          bootstrap_standard_deviation=sqrt(
                            transformed_bootstrap_variance
                          )
                        )
                      }},
                      warning=function(w) {{
                        captured_warnings <<- c(captured_warnings, conditionMessage(w))
                        invokeRestart("muffleWarning")
                      }}
                    ),
                    error=function(e) {{
                      captured_error <<- conditionMessage(e)
                      NULL
                    }}
                  )
                  rarcat_compact <- gsub(
                    "[[:space:]]",
                    "",
                    paste(deparse(body(WeightedCluster::rarcat)), collapse="\\n")
                  )
                  coefficient_assignment <- regexpr(
                    "output$coefficients[1]<-tanh(output$coefficients[1])",
                    rarcat_compact,
                    fixed=TRUE
                  )[[1]]
                  se_assignment <- regexpr(
                    paste0(
                      "output$coefficients[2]<-output$coefficients[2]*",
                      "(1-tanh(output$coefficients[1])^2)"
                    ),
                    rarcat_compact,
                    fixed=TRUE
                  )[[1]]
                  list(
                    value=value,
                    warnings=captured_warnings,
                    error=captured_error,
                    se_reapplies_tanh_to_backtransformed_coefficient=(
                      coefficient_assignment > 0 &&
                      se_assignment > coefficient_assignment
                    )
                  )
                }})
                """)
            contract = names["contract"]
            error = (
                None
                if bool(self.r(f"is.null({contract}$error)")[0])
                else str(self.r(f"as.character({contract}$error)")[0])
            )
            payload: dict[str, Any] = {
                "error": error,
                "warnings": [str(value) for value in self.r(f"{contract}$warnings")],
                "se_reapplies_tanh_to_backtransformed_coefficient": bool(
                    self.r(f"{contract}$se_reapplies_tanh_to_backtransformed_coefficient")[0]
                ),
            }
            if error is not None:
                return payload
            value = f"{contract}$value"
            payload.update(
                {
                    "forward": np.asarray(self.r(f"as.numeric({value}$forward)"), dtype=float)
                    .reshape(-1)
                    .copy(),
                    "coefficient": float(self.r(f"{value}$coefficient")[0]),
                    "standard_error": float(self.r(f"{value}$standard_error")[0]),
                    "observation_standard_deviation": float(
                        self.r(f"{value}$observation_standard_deviation")[0]
                    ),
                    "bootstrap_standard_deviation": float(
                        self.r(f"{value}$bootstrap_standard_deviation")[0]
                    ),
                }
            )
            return payload
        finally:
            self.r(
                "rm(list=intersect(c("
                + ",".join(f"'{name}'" for name in names.values())
                + "), ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def oracle_versions(self) -> dict[str, str]:
        """Return the live R package versions as rendered by ``packageVersion``.

        R's ``packageVersion`` returns a ``numeric_version``; ``as.character`` of it
        can render a hyphenated release with dots. The exact gate normalizes only
        that punctuation before comparing with the pinned target.
        """
        wc = str(list(self.r('as.character(packageVersion("WeightedCluster"))'))[0])
        tm = str(list(self.r('as.character(packageVersion("TraMineR"))'))[0])
        return {"WeightedCluster": wc, "TraMineR": tm}

    def wc_kmedoids(
        self,
        D: np.ndarray,
        *,
        k: int,
        weights: np.ndarray | None = None,
        initialclust: np.ndarray | None = None,
        method: str = "PAMonce",
        npass: int = 0,
    ) -> dict[str, Any]:
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self._set_int_vector_or_null("initialclust", initialclust)
        self.r.assign("k", ro.IntVector([int(k)]))
        self.r.assign("method", ro.StrVector([str(method)]))
        self.r.assign("npass", ro.IntVector([int(npass)]))
        self.r(
            "res <- wcKMedoids(as.dist(D), k=as.integer(k), weights=weights, "
            "initialclust=initialclust, method=as.character(method), npass=as.integer(npass))"
        )
        return {
            "clustering": np.asarray(self.r("as.integer(res$clustering)"), dtype=int).reshape(-1),
            "info": self._named_numeric_vector("res$info"),
            "stats": self._named_numeric_vector("res$stats"),
            "asw": self._matrix("as.matrix(res$ASW)"),
        }

    def wc_kmedoids_method(
        self,
        D: np.ndarray,
        *,
        k: int,
        initialclust: np.ndarray,
        method: str,
        weights: np.ndarray | None = None,
        npass: int = 0,
    ) -> dict[str, Any]:
        """Run an installed-R k-medoids method with fixed initial medoids.

        WeightedCluster 2.0 exposes three method names, but its compiled
        ``PAM`` branch raises an implementation error. Returning that condition
        as data lets the parity test pin the installed behavior without hiding
        or xfail-ing it.
        """
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self._set_int_vector_or_null("initialclust", np.asarray(initialclust, dtype=int))
        self.r.assign("k", ro.IntVector([int(k)]))
        self.r.assign("method", ro.StrVector([str(method)]))
        self.r.assign("npass", ro.IntVector([int(npass)]))
        self.r(
            "res <- tryCatch("
            "wcKMedoids(as.dist(D), k=as.integer(k), weights=weights, "
            "initialclust=initialclust, method=as.character(method), "
            "npass=as.integer(npass)), "
            "error=function(e) conditionMessage(e))"
        )
        if bool(self.r("is.character(res)")[0]):
            return {"error": str(self.r("res")[0])}
        return {
            "clustering": np.asarray(self.r("as.integer(res$clustering)"), dtype=int).reshape(-1),
            "stats": self._named_numeric_vector("res$stats"),
        }

    def wc_kmedrange_contract(
        self,
        D: np.ndarray,
        *,
        kvals: Sequence[int],
        weights: np.ndarray | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Return the installed-R ``wcKMedRange(R=1)`` structural contract."""
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self.r.assign("kvals", ro.IntVector([int(k) for k in kvals]))
        self.r.assign("seed", ro.IntVector([int(seed)]))
        self.r("set.seed(as.integer(seed))")
        self.r("res <- wcKMedRange(as.dist(D), kvals=as.integer(kvals), " "weights=weights, R=1)")
        return {
            "class": [str(x) for x in self.r("class(res)")],
            "names": [str(x) for x in self.r("names(res)")],
            "kvals": [int(x) for x in self.r("as.integer(res$kvals)")],
            "clustering_names": [str(x) for x in self.r("colnames(res$clustering)")],
            "stats_names": [str(x) for x in self.r("colnames(res$stats)")],
            "stats_shape": [int(x) for x in self.r("dim(res$stats)")],
        }

    def _clustrange_result_payload(
        self,
        result_name: str,
        *,
        digits: int = 2,
        bootstat: str | Sequence[str] = "t0",
        max_rank: int = 1,
    ) -> dict[str, Any]:
        """Normalize an installed-R ``clustrange`` without losing table labels."""

        def _optional_matrix(field: str) -> np.ndarray | None:
            if bool(self.r(f"is.null({result_name}${field})")[0]):
                return None
            return self._matrix(f"as.matrix({result_name}${field})")

        def _optional_names(expr: str) -> list[str]:
            value = self.r(expr)
            if value is ro.NULL:
                return []
            return [str(item) for item in value]

        summary_name = f"{result_name}_summary"
        print_error_name = f"{result_name}_print_error"
        print_lines_name = f"{result_name}_print_lines"
        boot_values = [bootstat] if isinstance(bootstat, str) else list(bootstat)
        boot_expr = (
            "c("
            + ",".join("'" + str(value).replace("'", "\\'") + "'" for value in boot_values)
            + ")"
        )
        self.r(
            f"{summary_name} <- summary({result_name}, max.rank=as.integer({int(max_rank)})); "
            f"{print_error_name} <- NULL; "
            f"{print_lines_name} <- tryCatch("
            f"capture.output(print({result_name}, digits=as.integer({int(digits)}), "
            f"bootstat={boot_expr})), "
            f"error=function(e) {{ {print_error_name} <<- conditionMessage(e); character() }})"
        )
        try:
            boot = []
            if not bool(self.r(f"is.null({result_name}$boot)")[0]):
                boot = [
                    self._matrix(f"as.matrix({result_name}$boot[[{index + 1}]])")
                    for index in range(int(self.r(f"length({result_name}$boot)")[0]))
                ]
            print_error = self.r(print_error_name)
            clustering_is_numeric = bool(
                self.r(
                    f"all(vapply({result_name}$clustering, "
                    "function(column) is.numeric(column) || is.integer(column), "
                    "logical(1)))"
                )[0]
            )
            return {
                "class": [str(value) for value in self.r(f"class({result_name})")],
                "names": [str(value) for value in self.r(f"names({result_name})")],
                "R": (
                    None
                    if bool(self.r(f"is.null({result_name}$R)")[0])
                    else int(self.r(f"as.integer({result_name}$R)")[0])
                ),
                "kvals": [int(value) for value in self.r(f"as.integer({result_name}$kvals)")],
                "clustering": self._matrix(
                    "do.call(cbind, lapply("
                    f"{result_name}$clustering, "
                    "function(column) as.integer(factor(column))))"
                ),
                "clustering_numeric": (
                    self._matrix(f"as.matrix({result_name}$clustering)")
                    if clustering_is_numeric
                    else None
                ),
                "clustering_column_names": _optional_names(f"colnames({result_name}$clustering)"),
                "clustering_factor_levels": [
                    [
                        str(level)
                        for level in self.r(
                            f"levels(factor({result_name}$clustering[[{index + 1}]]))"
                        )
                    ]
                    for index in range(int(self.r(f"ncol({result_name}$clustering)")[0]))
                ],
                "stats": self._matrix(f"as.matrix({result_name}$stats)"),
                "stats_column_names": _optional_names(f"colnames({result_name}$stats)"),
                "stats_row_names": _optional_names(f"rownames({result_name}$stats)"),
                "boot": boot,
                "meant": _optional_matrix("meant"),
                "meant_row_names": _optional_names(f"rownames({result_name}$meant)"),
                "stderr": _optional_matrix("stderr"),
                "stderr_row_names": _optional_names(f"rownames({result_name}$stderr)"),
                "summary": self._matrix(f"as.matrix({summary_name})"),
                "summary_row_names": _optional_names(f"rownames({summary_name})"),
                "summary_column_names": _optional_names(f"colnames({summary_name})"),
                "print_lines": [str(value) for value in self.r(print_lines_name)],
                "print_error": (None if print_error is ro.NULL else str(print_error[0])),
            }
        finally:
            self.r(
                "rm(list=intersect(ls(), c("
                + ",".join(
                    f"'{name}'" for name in (summary_name, print_error_name, print_lines_name)
                )
                + ")))"
            )

    def _clustrange_plot_payload(
        self,
        result_name: str,
        *,
        stat: str | Sequence[str] = "noCH",
        norm: str = "none",
        conf_int: float = 0.9,
        ci_method: str = "none",
        ci_alpha: float = 0.3,
        line: str = "t0",
    ) -> dict[str, Any]:
        """Capture installed rendering conditions plus source-exact plot arrays."""
        token = uuid.uuid4().hex
        stat_name = f"wc_phase2_clustrange_plot_stat_{token}"
        payload_name = f"wc_phase2_clustrange_plot_payload_{token}"
        render_name = f"wc_phase2_clustrange_plot_render_{token}"
        requested = [stat] if isinstance(stat, str) else list(stat)
        self.r.assign(stat_name, ro.StrVector([str(value) for value in requested]))
        try:
            self.r(f"""
                {payload_name} <- local({{
                  x <- get('{result_name}', envir=.GlobalEnv)
                  requested <- get('{stat_name}', envir=.GlobalEnv)
                  effective_norm <- '{str(norm)}'
                  all_names <- colnames(x$stats)
                  if (length(requested) == 1L && requested == 'all') {{
                    selected <- all_names
                    if (effective_norm == 'none') effective_norm <- 'range'
                  }} else if (length(requested) == 1L && requested == 'noCH') {{
                    selected <- all_names[-grep('CH', all_names)]
                  }} else {{
                    selected <- requested[requested != 'RHC']
                    if ('RHC' %in% requested) selected <- c(selected, 'RHC')
                  }}
                  base_names <- selected[selected != 'RHC']
                  values <- as.matrix(x$stats[, base_names, drop=FALSE])
                  if ('RHC' %in% selected) {{
                    values <- cbind(values, RHC=1 - x$stats[, 'HC'])
                  }}
                  colnames(values) <- selected
                  lower <- NULL
                  upper <- NULL
                  plot_ci <- '{str(ci_method)}' != 'none' && !is.null(x$boot)
                  normalize_one <- get('normalize.values',
                                       envir=asNamespace('WeightedCluster'))
                  normalize_matrix <- get('normalize.values.matrix',
                                           envir=asNamespace('WeightedCluster'))
                  if (plot_ci) {{
                    lower <- matrix(NA_real_, nrow=nrow(values), ncol=ncol(values),
                                    dimnames=dimnames(values))
                    upper <- lower
                    confvec <- c((1 - {float(conf_int)}) / 2,
                                 1 - (1 - {float(conf_int)}) / 2)
                    for (index in seq_along(selected)) {{
                      name <- selected[index]
                      source <- if (name == 'RHC') 'HC' else name
                      block <- sapply(x$boot, function(item) item[, source])
                      if (name == 'RHC') block <- 1 - block
                      normalized_block <- normalize_matrix(block, effective_norm)
                      if ('{str(line)}' == 't0') {{
                        values[, index] <- normalized_block[1L, ]
                      }} else if ('{str(line)}' == 'mean') {{
                        values[, index] <- apply(normalized_block, 2L, mean)
                      }} else if ('{str(line)}' == 'median') {{
                        values[, index] <- apply(normalized_block, 2L, median)
                      }} else {{
                        stop('invalid line')
                      }}
                      if ('{str(ci_method)}' == 'norm') {{
                        bounds <- apply(normalized_block, 2L, function(column)
                          mean(column) + qnorm(confvec) * sqrt(var(column)))
                      }} else if ('{str(ci_method)}' == 'perc') {{
                        bounds <- apply(normalized_block, 2L, function(column)
                          quantile(column, confvec))
                      }} else {{
                        stop('invalid ci.method')
                      }}
                      lower[, index] <- bounds[1L, ]
                      upper[, index] <- bounds[2L, ]
                    }}
                  }} else {{
                    for (index in seq_len(ncol(values))) {{
                      values[, index] <- normalize_one(values[, index], effective_norm)
                    }}
                  }}
                  palette <- RColorBrewer::brewer.pal(length(all_names) + 1L, 'Set3')[-2L]
                  names(palette) <- all_names
                  palette['RHC'] <- palette['HC']
                  colors <- unname(palette[selected])
                  legend_labels <- paste(colnames(values), '(',
                    round(apply(values, 2L, min), 2), '/',
                    round(apply(values, 2L, max), 2), ')')
                  list(stat_names=selected, norm=effective_norm, values=values,
                       lower=lower, upper=upper, plot_ci=plot_ci,
                       ci_alpha={float(ci_alpha)}, colors=colors,
                       legend_labels=legend_labels)
                }})
                {render_name} <- local({{
                  grDevices::pdf(NULL)
                  on.exit(grDevices::dev.off())
                  tryCatch({{
                    plot(get('{result_name}', envir=.GlobalEnv),
                         stat=get('{stat_name}', envir=.GlobalEnv),
                         norm='{str(norm)}', conf.int={float(conf_int)},
                         ci.method='{str(ci_method)}', ci.alpha={float(ci_alpha)},
                         line='{str(line)}')
                    NULL
                  }}, error=function(e) conditionMessage(e))
                }})
                """)
            render = self.r(render_name)
            lower = (
                None
                if bool(self.r(f"is.null({payload_name}$lower)")[0])
                else self._matrix(f"{payload_name}$lower")
            )
            upper = (
                None
                if bool(self.r(f"is.null({payload_name}$upper)")[0])
                else self._matrix(f"{payload_name}$upper")
            )
            return {
                "stat_names": [str(value) for value in self.r(f"{payload_name}$stat_names")],
                "norm": str(self.r(f"{payload_name}$norm")[0]),
                "values": self._matrix(f"{payload_name}$values"),
                "lower": lower,
                "upper": upper,
                "with_intervals": bool(self.r(f"{payload_name}$plot_ci")[0]),
                "ci_alpha": float(self.r(f"{payload_name}$ci_alpha")[0]),
                "colors": [str(value) for value in self.r(f"{payload_name}$colors")],
                "legend_labels": [str(value) for value in self.r(f"{payload_name}$legend_labels")],
                "render_error": None if render is ro.NULL else str(render[0]),
            }
        finally:
            self.r(
                "rm(list=intersect(ls(), c("
                + ",".join(f"'{name}'" for name in (stat_name, payload_name, render_name))
                + ")))"
            )

    def _clustrangefamily_plot_payload(
        self,
        result_name: str,
        *,
        group: str = "stat",
        method: str | Sequence[str] = "all",
        pam_combine: bool = False,
        stat: str | Sequence[str] = "noCH",
        norm: str = "none",
        withlegend: bool = True,
        col: Sequence[str] | None = None,
        legend_prop: float | None = None,
        rows: int | None = None,
        cols: int | None = None,
        main: str | Sequence[str] | None = None,
        xlab: str = "",
        ylab: str = "",
    ) -> dict[str, Any]:
        """Capture source-exact family plot arrays, layout, and conditions."""

        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_clustrangefamily_plot_{key}_{token}"
            for key in (
                "method",
                "stat",
                "col",
                "main",
                "payload",
                "data_error",
                "render_error",
            )
        }
        method_values = [method] if isinstance(method, str) else list(method)
        stat_values = [stat] if isinstance(stat, str) else list(stat)
        self.r.assign(names["method"], ro.StrVector([str(value) for value in method_values]))
        self.r.assign(names["stat"], ro.StrVector([str(value) for value in stat_values]))
        if col is None:
            self.r(f"{names['col']} <- NULL")
        else:
            self.r.assign(names["col"], ro.StrVector([str(value) for value in col]))
        if main is None:
            self.r(f"{names['main']} <- NULL")
        else:
            main_values = [main] if isinstance(main, str) else list(main)
            self.r.assign(
                names["main"],
                ro.StrVector([str(value) for value in main_values]),
            )
        row_expr = "NA" if rows is None else f"as.integer({int(rows)})"
        col_expr = "NA" if cols is None else f"as.integer({int(cols)})"
        legend_expr = "NA" if legend_prop is None else str(float(legend_prop))
        try:
            self.r(f"""
                {names['data_error']} <- NULL
                {names['payload']} <- tryCatch(local({{
                  x <- get('{result_name}', envir=.GlobalEnv)
                  requested_method <- get('{names['method']}', envir=.GlobalEnv)
                  requested_stat <- get('{names['stat']}', envir=.GlobalEnv)
                  supplied_col <- get('{names['col']}', envir=.GlobalEnv)
                  supplied_main <- get('{names['main']}', envir=.GlobalEnv)
                  allstats <- colnames(x[[1]]$stats)
                  if (length(requested_stat) == 1L) {{
                    if (requested_stat == 'all') {{
                      selected_stat <- allstats
                    }} else if (requested_stat == 'noCH') {{
                      selected_stat <- allstats[-grep('CH', allstats)]
                    }} else if (!requested_stat %in% c(allstats, 'RHC')) {{
                      stop(paste(' [!] unknow statistic', requested_stat))
                    }} else {{
                      selected_stat <- requested_stat
                    }}
                  }} else {{
                    selected_stat <- requested_stat[
                      requested_stat %in% c(allstats, 'RHC')]
                  }}
                  if (length(requested_method) == 1L &&
                      requested_method == 'all') {{
                    selected_method <- if ({'TRUE' if pam_combine else 'FALSE'})
                      x$param$all.methods else x$param$method
                  }} else if (length(requested_method) == 1L &&
                             requested_method == 'pam.combine') {{
                    if (!x$param$pam.combine) {{
                      stop(' [!] you should have used pam.combine when computing the clustering.')
                    }}
                    selected_method <- grep('pam', x$param$all.methods, value=TRUE)
                  }} else {{
                    selected_method <- requested_method
                  }}
                  titles <- if ('{str(group)}' == 'stat')
                    selected_stat else selected_method
                  if (is.null(supplied_main)) {{
                    plot_titles <- titles
                  }} else if (length(supplied_main) == 1L &&
                             length(titles) > 1L) {{
                    plot_titles <- paste(supplied_main, titles, sep=' - ')
                  }} else if (length(supplied_main) != length(titles)) {{
                    stop(' [!] You should provide one title (main argument) per plot.')
                  }} else {{
                    plot_titles <- supplied_main
                  }}
                  names(plot_titles) <- titles
                  normalize_all <- get('normalize.values.all',
                    envir=asNamespace('WeightedCluster'))
                  if ('{str(group)}' == 'stat') {{
                    values <- array(NA_real_,
                      dim=c(length(x$param$kvals), length(selected_method),
                            length(selected_stat)),
                      dimnames=list(as.character(x$param$kvals),
                                    selected_method, selected_stat))
                    for (stat_index in seq_along(selected_stat)) {{
                      stat_name <- selected_stat[stat_index]
                      block <- matrix(NA_real_, nrow=length(x$param$kvals),
                                      ncol=length(selected_method),
                                      dimnames=list(NULL, selected_method))
                      for (method_name in selected_method) {{
                        block[, method_name] <- if (stat_name == 'RHC')
                          1 - x[[method_name]]$stats[, 'HC']
                        else x[[method_name]]$stats[, stat_name]
                      }}
                      values[, , stat_index] <- normalize_all(block, '{str(norm)}')
                    }}
                    if (is.null(supplied_col)) {{
                      if (length(selected_method) > 8L) {{
                        stop('[!] Too many clustering methods, you need to manually specify the colors (col argument).')
                      }}
                      palette <- RColorBrewer::brewer.pal(
                        length(selected_method), 'Dark2')
                      names(palette) <- selected_method
                      colors <- unname(palette[selected_method])
                    }} else {{
                      if (length(supplied_col) != length(selected_method)) {{
                        stop(' [!] You should specify at least one color per quality measure to plot.')
                      }}
                      colors <- unname(supplied_col)
                    }}
                    panel_count <- length(selected_stat)
                  }} else {{
                    values <- array(NA_real_,
                      dim=c(length(x$param$kvals), length(selected_stat),
                            length(selected_method)),
                      dimnames=list(as.character(x$param$kvals),
                                    selected_stat, selected_method))
                    for (method_index in seq_along(selected_method)) {{
                      method_name <- selected_method[method_index]
                      block <- matrix(NA_real_, nrow=length(x$param$kvals),
                                      ncol=length(selected_stat),
                                      dimnames=list(NULL, selected_stat))
                      for (stat_name in selected_stat) {{
                        block[, stat_name] <- if (stat_name == 'RHC')
                          1 - x[[method_name]]$stats[, 'HC']
                        else x[[method_name]]$stats[, stat_name]
                      }}
                      values[, , method_index] <- normalize_all(block, '{str(norm)}')
                    }}
                    if (is.null(supplied_col)) {{
                      palette <- RColorBrewer::brewer.pal(
                        length(allstats) + 1L, 'Set3')[-2L]
                      names(palette) <- allstats
                      palette['RHC'] <- palette['HC']
                      colors <- unname(palette[selected_stat])
                    }} else {{
                      if (length(supplied_col) != length(selected_stat)) {{
                        stop(' [!] You should specify at least one color per quality measure to plot.')
                      }}
                      colors <- unname(supplied_col)
                    }}
                    panel_count <- length(selected_method)
                  }}
                  layout_fun <- get('TraMineRInternalLayout',
                    envir=asNamespace('WeightedCluster'))
                  layout_value <- layout_fun(
                    panel_count, {row_expr}, {col_expr},
                    {'TRUE' if withlegend else 'FALSE'}, axes='all',
                    {legend_expr})
                  list(group='{str(group)}', methods=selected_method,
                       stats=selected_stat, titles=unname(plot_titles),
                       values=values, colors=colors,
                       laymat=layout_value$laymat,
                       heights=layout_value$heights,
                       widths=layout_value$widths,
                       legpos=layout_value$legpos,
                       withlegend={'TRUE' if withlegend else 'FALSE'},
                       xlab='{str(xlab)}', ylab='{str(ylab)}')
                }}), error=function(e) {{
                  {names['data_error']} <<- conditionMessage(e); NULL
                }})
                {names['render_error']} <- local({{
                  grDevices::pdf(NULL)
                  on.exit(grDevices::dev.off())
                  tryCatch({{
                    plot(get('{result_name}', envir=.GlobalEnv),
                         group='{str(group)}',
                         method=get('{names['method']}', envir=.GlobalEnv),
                         pam.combine={'TRUE' if pam_combine else 'FALSE'},
                         stat=get('{names['stat']}', envir=.GlobalEnv),
                         norm='{str(norm)}',
                         withlegend={'TRUE' if withlegend else 'FALSE'},
                         col=get('{names['col']}', envir=.GlobalEnv),
                         legend.prop={legend_expr}, rows={row_expr}, cols={col_expr},
                         main=get('{names['main']}', envir=.GlobalEnv),
                         xlab='{str(xlab)}', ylab='{str(ylab)}')
                    NULL
                  }}, error=function(e) conditionMessage(e))
                }})
                """)
            data_error = self.r(names["data_error"])
            render_error = self.r(names["render_error"])
            result: dict[str, Any] = {
                "data_error": (None if data_error is ro.NULL else str(data_error[0])),
                "render_error": (None if render_error is ro.NULL else str(render_error[0])),
            }
            if result["data_error"] is not None:
                return result
            dims = [int(value) for value in self.r(f"dim({names['payload']}$values)")]
            values = np.asarray(
                self.r(f"as.numeric({names['payload']}$values)"), dtype=float
            ).reshape(tuple(dims), order="F")
            result.update(
                {
                    "group": str(self.r(f"{names['payload']}$group")[0]),
                    "methods": [str(value) for value in self.r(f"{names['payload']}$methods")],
                    "stats": [str(value) for value in self.r(f"{names['payload']}$stats")],
                    "titles": [str(value) for value in self.r(f"{names['payload']}$titles")],
                    "values": values,
                    "colors": [str(value) for value in self.r(f"{names['payload']}$colors")],
                    "laymat": self._matrix(f"{names['payload']}$laymat"),
                    "heights": [float(value) for value in self.r(f"{names['payload']}$heights")],
                    "widths": [float(value) for value in self.r(f"{names['payload']}$widths")],
                    "legpos": str(self.r(f"{names['payload']}$legpos")[0]),
                    "withlegend": bool(self.r(f"{names['payload']}$withlegend")[0]),
                    "xlab": str(self.r(f"{names['payload']}$xlab")[0]),
                    "ylab": str(self.r(f"{names['payload']}$ylab")[0]),
                }
            )
            return result
        finally:
            self.r(
                "rm(list=intersect(ls(), c("
                + ",".join(f"'{name}'" for name in names.values())
                + ")))"
            )

    def wc_kmedrange_complete_contract(
        self,
        D: np.ndarray,
        *,
        kvals: Sequence[int],
        weights: np.ndarray | None = None,
        R: int = 1,
        samplesize: int | None = None,
        distance_kind: str = "dist",
        initial_kind: str = "hclust",
        method: str | int = "PAMonce",
        npass: int = 1,
        debuglevel: int = 0,
        replay_samples_1b: Sequence[Sequence[int]] | np.ndarray | None = None,
        digits: int = 2,
        bootstat: str | Sequence[str] = "t0",
        max_rank: int = 1,
        plot_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture complete ``wcKMedRange`` data with optional exact sample replay."""
        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_kmedrange_{key}_{token}"
            for key in (
                "D",
                "diss",
                "weights",
                "kvals",
                "initial",
                "method",
                "samples",
                "warnings",
                "error",
                "result",
                "before_diss",
                "before_weights",
                "before_kvals",
                "before_initial",
            )
        }
        if distance_kind not in {"matrix", "dist"}:
            raise ValueError("distance_kind must be 'matrix' or 'dist'")
        if initial_kind not in {"hclust", "twins", "none"}:
            raise ValueError("initial_kind must be hclust, twins, or none")
        D_array = np.asarray(D, dtype=float)
        replay = None
        if replay_samples_1b is not None:
            replay = np.asarray(replay_samples_1b, dtype=int)
            if replay.ndim != 2 or replay.shape[0] != int(R):
                raise ValueError("replay_samples_1b must have shape (R, samplesize)")
            if samplesize is not None and replay.shape[1] != int(samplesize):
                raise ValueError("replay sample width must equal samplesize")
            samplesize = int(replay.shape[1])
        try:
            self._assign_matrix(names["D"], D_array)
            self.r(
                f"{names['diss']} <- "
                + (names["D"] if distance_kind == "matrix" else f"as.dist({names['D']})")
            )
            if weights is None:
                self.r(f"{names['weights']} <- NULL")
            else:
                self.r.assign(
                    names["weights"],
                    ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
                )
            self.r.assign(names["kvals"], ro.IntVector([int(k) for k in kvals]))
            if initial_kind == "hclust":
                self.r(f"{names['initial']} <- hclust(as.dist({names['D']}), method='single')")
            elif initial_kind == "twins":
                self.r(
                    f"{names['initial']} <- cluster::agnes("
                    f"as.dist({names['D']}), diss=TRUE, method='average')"
                )
            else:
                self.r(f"{names['initial']} <- NULL")
            if isinstance(method, str):
                self.r.assign(names["method"], ro.StrVector([str(method)]))
            else:
                self.r.assign(names["method"], ro.IntVector([int(method)]))
            if replay is not None:
                self._assign_matrix(names["samples"], replay)
                self.r(
                    f"{names['samples']} <- matrix(as.integer({names['samples']}), "
                    f"nrow=nrow({names['samples']}), ncol=ncol({names['samples']}))"
                )
            sample_arg = "NULL" if samplesize is None else f"as.integer({int(samplesize)})"
            self.r(
                f"{names['warnings']} <- character(); {names['error']} <- NULL; "
                f"{names['before_diss']} <- {names['diss']}; "
                f"{names['before_weights']} <- {names['weights']}; "
                f"{names['before_kvals']} <- {names['kvals']}; "
                f"{names['before_initial']} <- {names['initial']}; "
                f"{names['result']} <- withCallingHandlers(tryCatch({{ "
                + (
                    (
                        f"local({{ value <- WeightedCluster::wcKMedRange("
                        f"{names['diss']}, kvals={names['kvals']}, "
                        f"weights={names['weights']}, R=1, "
                        f"samplesize={sample_arg}, initialclust={names['initial']}, "
                        f"method={names['method']}, npass=as.integer({int(npass)}), "
                        f"debuglevel=as.integer({int(debuglevel)})); "
                        f"value$R <- as.integer({int(R)}); "
                        "stat_names <- colnames(value$stats); "
                        "boot <- vector('list', length(value$kvals)); "
                        "for (column in seq_along(value$kvals)) { "
                        f"  block <- matrix(NA_real_, nrow=as.integer({int(R)}) + 1L, "
                        "                  ncol=length(stat_names), "
                        "                  dimnames=list(NULL, stat_names)); "
                        "  labels0 <- as.integer(factor(value$clustering[, column])) - 1L; "
                        f"  full_weights <- if (is.null({names['weights']})) "
                        f"    rep(1.0, nrow({names['D']})) else {names['weights']}; "
                        "  quality_t0 <- .Call("
                        "    get('C_RClusterQual', envir=asNamespace('WeightedCluster')), "
                        f"    {names['diss']}, labels0, as.double(full_weights), "
                        "    as.integer(max(labels0) + 1L), "
                        f"    as.integer(inherits({names['diss']}, 'dist'))); "
                        "  block[1L, ] <- as.numeric(quality_t0[[1L]]); "
                        f"  for (replicate in seq_len(as.integer({int(R)}))) {{ "
                        f"    counts <- tabulate({names['samples']}[replicate, ], "
                        f"                       nbins=nrow({names['D']})); "
                        "    labels0 <- as.integer(factor(value$clustering[, column])) - 1L; "
                        "    quality <- .Call("
                        "      get('C_RClusterQual', envir=asNamespace('WeightedCluster')), "
                        f"      {names['diss']}, labels0, as.double(counts), "
                        "      as.integer(max(labels0) + 1L), "
                        f"      as.integer(inherits({names['diss']}, 'dist'))); "
                        "    block[replicate + 1L, ] <- as.numeric(quality[[1L]]); "
                        "  }; "
                        "  boot[[column]] <- block "
                        "}; "
                        "class(boot) <- 'clustrangeboot'; value$boot <- boot; "
                        "value$stats <- as.data.frame(t(vapply(boot, "
                        "function(block) block[1L, ], numeric(length(stat_names))))); "
                        "colnames(value$stats) <- stat_names; "
                        "rownames(value$stats) <- paste0('cluster', value$kvals); value })"
                    )
                    if replay is not None
                    else (
                        f"WeightedCluster::wcKMedRange("
                        f"{names['diss']}, kvals={names['kvals']}, "
                        f"weights={names['weights']}, R=as.integer({int(R)}), "
                        f"samplesize={sample_arg}, initialclust={names['initial']}, "
                        f"method={names['method']}, npass=as.integer({int(npass)}), "
                        f"debuglevel=as.integer({int(debuglevel)}))"
                    )
                )
                + f" }}, error=function(e) {{ {names['error']} <<- "
                "conditionMessage(e); NULL }), "
                f"warning=function(w) {{ {names['warnings']} <<- "
                f"c({names['warnings']}, conditionMessage(w)); "
                "invokeRestart('muffleWarning') })"
            )
            error = self.r(names["error"])
            error_text = None if error is ro.NULL else str(error[0])
            result: dict[str, Any] = {
                "error": error_text,
                "warnings": [str(value) for value in self.r(names["warnings"])],
                "inputs_unchanged": bool(
                    self.r(
                        f"identical({names['diss']}, {names['before_diss']}) && "
                        f"identical({names['weights']}, {names['before_weights']}) && "
                        f"identical({names['kvals']}, {names['before_kvals']}) && "
                        f"identical({names['initial']}, {names['before_initial']})"
                    )[0]
                ),
            }
            if error_text is None:
                result.update(
                    self._clustrange_result_payload(
                        names["result"],
                        digits=digits,
                        bootstat=bootstat,
                        max_rank=max_rank,
                    )
                )
                if plot_options is not None:
                    result["plot"] = self._clustrange_plot_payload(
                        names["result"],
                        **dict(plot_options),
                    )
            return result
        finally:
            self.r(
                "rm(list=intersect(ls(), c("
                + ",".join(f"'{name}'" for name in names.values())
                + ")))"
            )

    def as_clustrange_complete_contract(
        self,
        D: np.ndarray,
        cluster_columns: Mapping[str, Sequence[Any]] | None = None,
        *,
        weights: np.ndarray | None = None,
        factor_levels: Mapping[str, Sequence[Any]] | None = None,
        R: int = 1,
        samplesize: int | None = None,
        distance_kind: str = "dist",
        input_kind: str = "default",
        ncluster: int = 3,
        linkage_method: str = "single",
        dt_properties: Mapping[str, Sequence[Any]] | None = None,
        dt_categorical: Sequence[str] = (),
        replay_samples_1b: Sequence[Sequence[int]] | np.ndarray | None = None,
        digits: int = 2,
        bootstat: str | Sequence[str] = "t0",
        max_rank: int = 1,
        plot_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture all ``as.clustrange`` dispatch/result/presentation fields."""
        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_as_clustrange_{key}_{token}"
            for key in (
                "D",
                "diss",
                "weights",
                "object",
                "samples",
                "warnings",
                "error",
                "result",
                "before_diss",
                "before_weights",
                "before_object",
            )
        }
        if distance_kind not in {"matrix", "dist"}:
            raise ValueError("distance_kind must be 'matrix' or 'dist'")
        if input_kind not in {"default", "hclust", "twins", "dtclust"}:
            raise ValueError("input_kind must be default, hclust, twins, or dtclust")
        D_array = np.asarray(D, dtype=float)
        replay = None
        if replay_samples_1b is not None:
            replay = np.asarray(replay_samples_1b, dtype=int)
            if replay.ndim != 2 or replay.shape[0] != int(R):
                raise ValueError("replay_samples_1b must have shape (R, samplesize)")
            if samplesize is not None and replay.shape[1] != int(samplesize):
                raise ValueError("replay sample width must equal samplesize")
            samplesize = int(replay.shape[1])
        try:
            self._assign_matrix(names["D"], D_array)
            self.r(
                f"{names['diss']} <- "
                + (names["D"] if distance_kind == "matrix" else f"as.dist({names['D']})")
            )
            if weights is None:
                self.r(f"{names['weights']} <- NULL")
            else:
                self.r.assign(
                    names["weights"],
                    ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
                )
            if input_kind == "default":
                columns = list((cluster_columns or {}).items())
                self.r(f"{names['object']} <- data.frame(row.names=seq_len(nrow({names['D']})))")
                for index, (column_name, values) in enumerate(columns):
                    value_name = f"{names['object']}_column_{index}"
                    levels_name = f"{names['object']}_levels_{index}"
                    values_list = list(values)
                    levels = (factor_levels or {}).get(str(column_name))
                    if levels is not None:
                        self.r.assign(
                            value_name,
                            ro.StrVector([str(value) for value in values_list]),
                        )
                        self.r.assign(
                            levels_name,
                            ro.StrVector([str(value) for value in levels]),
                        )
                        self.r(
                            f"{names['object']}[['{str(column_name)}']] <- factor("
                            f"{value_name}, levels={levels_name})"
                        )
                    elif all(
                        isinstance(value, (int, np.integer))
                        and not isinstance(value, (bool, np.bool_))
                        for value in values_list
                    ):
                        self.r.assign(
                            value_name,
                            ro.IntVector([int(value) for value in values_list]),
                        )
                        self.r(
                            f"{names['object']}[['{str(column_name)}']] <- "
                            f"as.integer({value_name})"
                        )
                    else:
                        self.r.assign(
                            value_name,
                            ro.StrVector([str(value) for value in values_list]),
                        )
                        self.r(f"{names['object']}[['{str(column_name)}']] <- {value_name}")
            elif input_kind == "hclust":
                self.r(
                    f"{names['object']} <- hclust(as.dist({names['D']}), "
                    f"method='{str(linkage_method)}')"
                )
            elif input_kind == "twins":
                self.r(
                    f"{names['object']} <- cluster::agnes(as.dist({names['D']}), "
                    f"diss=TRUE, method='{str(linkage_method)}')"
                )
            else:
                if not dt_properties:
                    raise ValueError("dt_properties are required for dtclust input")
                properties_name = f"{names['object']}_properties"
                self.r(
                    f"{properties_name} <- data.frame(" f"row.names=seq_len(nrow({names['D']})))"
                )
                categorical = {str(value) for value in dt_categorical}
                for index, (column_name, values) in enumerate(dt_properties.items()):
                    value_name = f"{properties_name}_column_{index}"
                    values_list = list(values)
                    if str(column_name) in categorical:
                        self.r.assign(
                            value_name,
                            ro.StrVector([str(value) for value in values_list]),
                        )
                        self.r(
                            f"{properties_name}[['{str(column_name)}']] <- " f"factor({value_name})"
                        )
                    else:
                        self.r.assign(
                            value_name,
                            ro.FloatVector([float(value) for value in values_list]),
                        )
                        self.r(
                            f"{properties_name}[['{str(column_name)}']] <- "
                            f"as.numeric({value_name})"
                        )
                self.r(
                    f"{names['object']} <- WeightedCluster::wcPropertyClustering("
                    f"{names['diss']}, {properties_name}, R=1, min.size=2, "
                    "max.depth=3, pval=1, weight.permutation='none')"
                )
            if replay is not None:
                self._assign_matrix(names["samples"], replay)
                self.r(
                    f"{names['samples']} <- matrix(as.integer({names['samples']}), "
                    f"nrow=nrow({names['samples']}), ncol=ncol({names['samples']}))"
                )
            sample_arg = "NULL" if samplesize is None else f"as.integer({int(samplesize)})"
            ncluster_arg = (
                "" if input_kind == "default" else f", ncluster=as.integer({int(ncluster)})"
            )
            base_call = (
                f"WeightedCluster::as.clustrange({names['object']}, "
                f"diss={names['diss']}, weights={names['weights']}, R=1, "
                f"samplesize={sample_arg}{ncluster_arg})"
            )
            direct_call = (
                f"WeightedCluster::as.clustrange({names['object']}, "
                f"diss={names['diss']}, weights={names['weights']}, "
                f"R=as.integer({int(R)}), samplesize={sample_arg}{ncluster_arg})"
            )
            replay_call = (
                f"local({{ value <- {base_call}; stat_names <- colnames(value$stats); "
                "boot <- vector('list', length(value$kvals)); "
                "for (column in seq_along(value$kvals)) { "
                f"  block <- matrix(NA_real_, nrow=as.integer({int(R)}) + 1L, "
                "                  ncol=length(stat_names), "
                "                  dimnames=list(NULL, stat_names)); "
                "  labels0 <- as.integer(factor(value$clustering[, column])) - 1L; "
                f"  full_weights <- if (is.null({names['weights']})) "
                f"    rep(1.0, nrow({names['D']})) else {names['weights']}; "
                "  quality_t0 <- .Call("
                "    get('C_RClusterQual', envir=asNamespace('WeightedCluster')), "
                f"    {names['diss']}, labels0, as.double(full_weights), "
                "    as.integer(max(labels0) + 1L), "
                f"    as.integer(inherits({names['diss']}, 'dist'))); "
                "  block[1L, ] <- as.numeric(quality_t0[[1L]]); "
                f"  for (replicate in seq_len(as.integer({int(R)}))) {{ "
                f"    counts <- tabulate({names['samples']}[replicate, ], "
                f"                       nbins=nrow({names['D']})); "
                "    labels0 <- as.integer(factor(value$clustering[, column])) - 1L; "
                "    quality <- .Call("
                "      get('C_RClusterQual', envir=asNamespace('WeightedCluster')), "
                f"      {names['diss']}, labels0, as.double(counts), "
                "      as.integer(max(labels0) + 1L), "
                f"      as.integer(inherits({names['diss']}, 'dist'))); "
                "    block[replicate + 1L, ] <- as.numeric(quality[[1L]]); "
                "  }; "
                "  boot[[column]] <- block "
                "}; "
                "class(boot) <- 'clustrangeboot'; value$boot <- boot; "
                "value$stats <- as.data.frame(t(vapply(boot, "
                "function(block) block[1L, ], numeric(length(stat_names))))); "
                "colnames(value$stats) <- stat_names; "
                "rownames(value$stats) <- paste0('cluster', value$kvals); "
                "value$meant <- as.data.frame(t(vapply(boot, colMeans, "
                "numeric(length(stat_names))))); "
                "value$stderr <- as.data.frame(t(vapply(boot, "
                "function(block) apply(block, 2L, function(x) sqrt(var(x))), "
                "numeric(length(stat_names))))); "
                "colnames(value$meant) <- stat_names; "
                "colnames(value$stderr) <- stat_names; "
                "rownames(value$meant) <- paste0('cluster', value$kvals); "
                "rownames(value$stderr) <- paste0('cluster', value$kvals); value })"
            )
            self.r(
                f"{names['warnings']} <- character(); {names['error']} <- NULL; "
                f"{names['before_diss']} <- {names['diss']}; "
                f"{names['before_weights']} <- {names['weights']}; "
                f"{names['before_object']} <- {names['object']}; "
                f"{names['result']} <- withCallingHandlers(tryCatch({{ "
                + (replay_call if replay is not None else direct_call)
                + f" }}, error=function(e) {{ {names['error']} <<- "
                "conditionMessage(e); NULL }), "
                f"warning=function(w) {{ {names['warnings']} <<- "
                f"c({names['warnings']}, conditionMessage(w)); "
                "invokeRestart('muffleWarning') })"
            )
            error = self.r(names["error"])
            error_text = None if error is ro.NULL else str(error[0])
            result: dict[str, Any] = {
                "error": error_text,
                "warnings": [str(value) for value in self.r(names["warnings"])],
                "inputs_unchanged": bool(
                    self.r(
                        f"identical({names['diss']}, {names['before_diss']}) && "
                        f"identical({names['weights']}, {names['before_weights']}) && "
                        f"identical({names['object']}, {names['before_object']})"
                    )[0]
                ),
            }
            if error_text is None:
                result.update(
                    self._clustrange_result_payload(
                        names["result"],
                        digits=digits,
                        bootstat=bootstat,
                        max_rank=max_rank,
                    )
                )
                if plot_options is not None:
                    result["plot"] = self._clustrange_plot_payload(
                        names["result"],
                        **dict(plot_options),
                    )
            return result
        finally:
            self.r(
                "rm(list=grep("
                f"'{token}', ls(envir=.GlobalEnv), value=TRUE), "
                "envir=.GlobalEnv)"
            )

    def _set_property_frame(
        self,
        properties: dict[str, np.ndarray],
        *,
        categorical: Sequence[str] = (),
        r_name: str = "properties",
    ) -> None:
        lengths = {int(np.asarray(values).size) for values in properties.values()}
        if len(lengths) > 1:
            raise ValueError("property columns must have equal lengths")
        n_rows = next(iter(lengths), 0)
        self.r.assign("property_nrow", ro.IntVector([n_rows]))
        self.r(f"{r_name} <- data.frame(row.names=seq_len(as.integer(property_nrow)))")
        categorical_names = {str(name) for name in categorical}
        for name, values in properties.items():
            arr = np.asarray(values).reshape(-1)
            self.r.assign("property_name", ro.StrVector([str(name)]))
            if str(name) in categorical_names or arr.dtype.kind in {"O", "S", "U", "b"}:
                self.r.assign(
                    "property_values",
                    ro.StrVector(
                        [
                            (
                                ro.NA_Character
                                if value is None
                                or (
                                    isinstance(value, (float, np.floating))
                                    and np.isnan(float(value))
                                )
                                else str(value)
                            )
                            for value in arr
                        ]
                    ),
                )
                self.r(
                    f"{r_name}[[as.character(property_name)]] <- "
                    "factor(as.character(property_values))"
                )
            else:
                self.r.assign(
                    "property_values",
                    ro.FloatVector(
                        [
                            (
                                ro.NA_Real
                                if value is None
                                or (
                                    isinstance(value, (float, np.floating))
                                    and np.isnan(float(value))
                                )
                                else float(value)
                            )
                            for value in arr
                        ]
                    ),
                )
                self.r(f"{r_name}[[as.character(property_name)]] <- " "as.numeric(property_values)")

    def _tree_contract(self, r_name: str = "tree") -> dict[str, Any]:
        self.r(
            f"tree_nodes <- local({{ st <- {r_name}; rows <- list(); "
            "walk <- function(node, parent=NA_integer_, side=NA_integer_) { "
            "sp <- node$split; has <- !is.null(sp); "
            "val <- c(id=node$id, parent=parent, side=side, "
            "depth=node$info$depth, schedule=node$info$splitschedule, "
            "n=node$info$n, vardis=node$info$vardis, medoid=node$info$medoid, "
            "has_split=as.integer(has), "
            "varindex=if (has) sp$varindex else NA_real_, "
            "breaks=if (has && !is.null(sp$breaks)) sp$breaks else NA_real_, "
            "SCres=if (has) sp$info$SCres else NA_real_, "
            "lpop=if (has) sp$info$lpop else NA_real_, "
            "rpop=if (has) sp$info$rpop else NA_real_, "
            "lvar=if (has) sp$info$lvar else NA_real_, "
            "rvar=if (has) sp$info$rvar else NA_real_, "
            "R2=if (has) sp$info$R2 else NA_real_, "
            "pval=if (has && !is.null(sp$info$pval)) sp$info$pval else NA_real_); "
            "rows[[length(rows)+1L]] <<- val; "
            "if (!is.null(node$kids)) { walk(node$kids[[1]], node$id, 1L); "
            "walk(node$kids[[2]], node$id, 2L) } }; "
            "walk(st$root); do.call(rbind, rows) })"
        )
        self.r(f"tree_labels <- WeightedCluster:::dtlabels({r_name})")
        self.r(
            f"tree_members <- local({{ st <- {r_name}; rows <- list(); "
            "walk <- function(node) { rows[[length(rows)+1L]] <<- as.integer(node$info$ind); "
            "if (!is.null(node$kids)) { walk(node$kids[[1]]); walk(node$kids[[2]]) } }; "
            "walk(st$root); rows })"
        )
        self.r(
            f"tree_split_prob <- local({{ st <- {r_name}; rows <- list(); "
            "walk <- function(node) { rows[[length(rows)+1L]] <<- "
            "if (is.null(node$split)) numeric() else as.numeric(node$split$prob); "
            "if (!is.null(node$kids)) { walk(node$kids[[1]]); walk(node$kids[[2]]) } }; "
            "walk(st$root); rows })"
        )
        self.r(
            f"tree_split_index <- local({{ st <- {r_name}; rows <- list(); "
            "walk <- function(node) { rows[[length(rows)+1L]] <<- "
            "if (is.null(node$split) || is.null(node$split$index)) integer() "
            "else as.integer(node$split$index); "
            "if (!is.null(node$kids)) { walk(node$kids[[1]]); walk(node$kids[[2]]) } }; "
            "walk(st$root); rows })"
        )
        self.r(
            f"tree_split_labels <- local({{ st <- {r_name}; rows <- list(); "
            "walk <- function(node) { rows[[length(rows)+1L]] <<- "
            "if (is.null(node$split) || is.null(node$split$labels)) character() "
            "else as.character(node$split$labels); "
            "if (!is.null(node$kids)) { walk(node$kids[[1]]); walk(node$kids[[2]]) } }; "
            "walk(st$root); rows })"
        )
        self.r(f"tree_max_k <- length(unique(as.integer({r_name}$fitted[, 1])))")
        self.r(
            f"tree_cut_ids <- if (tree_max_k >= 2L) "
            f"WeightedCluster:::dtcut({r_name}, 2L, labels=FALSE) else "
            f"as.integer({r_name}$fitted[, 1])"
        )
        self.r(
            f"tree_cut_labels <- if (tree_max_k >= 2L) "
            f"as.character(WeightedCluster:::dtcut({r_name}, 2L, labels=TRUE)) else "
            'rep("All", length(tree_cut_ids))'
        )
        self.r(
            f"tree_adjustment <- if (!is.null({r_name}$info$adjustment)) "
            f"setNames({r_name}$info$adjustment$stat[, 1], "
            f"rownames({r_name}$info$adjustment$stat)) else "
            "setNames(numeric(), character())"
        )
        tree_label_names = self.r("names(tree_labels)")
        return {
            "class": [str(x) for x in self.r(f"class({r_name})")],
            "names": [str(x) for x in self.r(f"names({r_name})")],
            "nodes": self._matrix("as.matrix(tree_nodes)"),
            "members_1b": [
                np.asarray(value, dtype=int).reshape(-1) for value in self.r("tree_members")
            ],
            "split_prob": [
                np.asarray(value, dtype=float).reshape(-1) for value in self.r("tree_split_prob")
            ],
            "split_index": [
                np.asarray(value, dtype=int).reshape(-1) for value in self.r("tree_split_index")
            ],
            "split_labels": [
                [str(item) for item in value] for value in self.r("tree_split_labels")
            ],
            "fitted": np.asarray(self.r(f"as.integer({r_name}$fitted[, 1])"), dtype=int).reshape(
                -1
            ),
            "label_names": (
                [] if tree_label_names is ro.NULL else [str(x) for x in tree_label_names]
            ),
            "label_values": [str(x) for x in self.r("as.character(tree_labels)")],
            "cut2_ids": np.asarray(self.r("as.integer(tree_cut_ids)"), dtype=int).reshape(-1),
            "cut2_labels": [str(x) for x in self.r("as.character(tree_cut_labels)")],
            "adjustment": self._named_numeric_vector("tree_adjustment"),
        }

    def seqpropclust_complete_contract(
        self,
        X: np.ndarray,
        *,
        properties: Sequence[str],
        other_properties: Mapping[str, Sequence[Any]] | None = None,
        categorical: Sequence[str] = (),
        factor_levels: Mapping[str, Sequence[Any]] | None = None,
        D: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        prop_only: bool = True,
        pmin_support: float = 0.05,
        max_k: int = -1,
        with_missing: bool = True,
        row_names: Sequence[Any] | None = None,
        R: int = 1,
        pval: float = 0.01,
        weight_permutation: str = "diss",
        min_size: float = 0.01,
        max_depth: int = 5,
        maxcluster: int | None = None,
        squared: bool = False,
        first: str | None = None,
        seed: int = 1,
    ) -> dict[str, Any]:
        """Return the complete live-R sequence-property extraction contract."""
        token = uuid.uuid4().hex
        names = {
            key: f"wc_seqprop_{key}_{token}"
            for key in (
                "values",
                "matrix",
                "sequence",
                "properties",
                "other",
                "weights",
                "rows",
                "result",
                "before_sequence",
                "before_other",
                "diss",
                "first",
            )
        }
        column_names: list[str] = []

        def _r_vector(values: Sequence[Any]):
            normalized = list(values)
            non_missing = [
                value
                for value in normalized
                if value is not None
                and not (isinstance(value, (float, np.floating)) and np.isnan(float(value)))
            ]
            if all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in non_missing
            ):
                return ro.FloatVector(
                    [
                        (
                            ro.NA_Real
                            if value is None
                            or (isinstance(value, (float, np.floating)) and np.isnan(float(value)))
                            else float(value)
                        )
                        for value in normalized
                    ]
                )
            return ro.StrVector(
                [
                    (
                        ro.NA_Character
                        if value is None
                        or (isinstance(value, (float, np.floating)) and np.isnan(float(value)))
                        else str(value)
                    )
                    for value in normalized
                ]
            )

        array = np.asarray(X, dtype=object)
        if array.ndim != 2:
            raise ValueError("X must be two-dimensional")
        flat_column_major = [
            array[row, column] for column in range(array.shape[1]) for row in range(array.shape[0])
        ]
        try:
            self.r.assign(names["values"], _r_vector(flat_column_major))
            self.r(
                f"{names['matrix']} <- matrix({names['values']}, "
                f"nrow={array.shape[0]}, ncol={array.shape[1]})"
            )
            if row_names is not None:
                self.r.assign(names["rows"], _r_vector(list(row_names)))
                self.r(f"rownames({names['matrix']}) <- as.character({names['rows']})")
            else:
                self.r(f"{names['rows']} <- NULL")
            if weights is None:
                self.r(f"{names['weights']} <- NULL")
            else:
                self.r.assign(
                    names["weights"],
                    ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
                )
            self.r(
                f"{names['sequence']} <- suppressMessages(TraMineR::seqdef("
                f"{names['matrix']}, weights={names['weights']}))"
            )
            self.r.assign(names["properties"], ro.StrVector([str(x) for x in properties]))

            if other_properties is None:
                self.r(f"{names['other']} <- NULL")
            else:
                other_lengths = {
                    len(np.asarray(values).reshape(-1)) for values in other_properties.values()
                }
                if len(other_lengths) > 1:
                    raise ValueError("other property columns must have equal lengths")
                other_nrow = next(iter(other_lengths), 0)
                self.r(f"{names['other']} <- data.frame(" f"row.names=seq_len({other_nrow}))")
                categorical_names = {str(value) for value in categorical}
                levels_by_name = {
                    str(key): list(value) for key, value in (factor_levels or {}).items()
                }
                for index, (column, values) in enumerate(other_properties.items()):
                    value_name = f"wc_seqprop_column_{index}_{token}"
                    column_names.append(value_name)
                    self.r.assign(value_name, _r_vector(list(values)))
                    self.r.assign(
                        names["values"] + "_column_name",
                        ro.StrVector([str(column)]),
                    )
                    if str(column) in categorical_names:
                        level_name = f"wc_seqprop_levels_{index}_{token}"
                        column_names.append(level_name)
                        levels = levels_by_name.get(str(column))
                        if levels is None:
                            self.r(
                                f"{names['other']}[[as.character("
                                f"{names['values']}_column_name)]] <- factor({value_name})"
                            )
                        else:
                            self.r.assign(level_name, _r_vector(levels))
                            self.r(
                                f"{names['other']}[[as.character("
                                f"{names['values']}_column_name)]] <- "
                                f"factor({value_name}, levels={level_name})"
                            )
                    else:
                        self.r(
                            f"{names['other']}[[as.character("
                            f"{names['values']}_column_name)]] <- {value_name}"
                        )
                if row_names is not None and other_nrow == len(row_names):
                    self.r(f"rownames({names['other']}) <- as.character({names['rows']})")

            self.r(f"{names['before_sequence']} <- {names['sequence']}")
            self.r(f"{names['before_other']} <- {names['other']}")
            distance = (
                np.zeros((array.shape[0], array.shape[0]), dtype=float)
                if D is None
                else np.asarray(D, dtype=float)
            )
            self.r.assign(
                names["diss"],
                self.r.matrix(
                    ro.FloatVector(distance.ravel(order="F")),
                    nrow=int(distance.shape[0]),
                    ncol=int(distance.shape[1]),
                ),
            )
            if first is None:
                self.r(f"{names['first']} <- NULL")
            else:
                self.r.assign(names["first"], ro.StrVector([str(first)]))
            maxcluster_expr = "NULL" if maxcluster is None else str(int(maxcluster))
            self.r(f"set.seed({int(seed)})")
            self.r(
                f"{names['result']} <- tryCatch(suppressMessages("
                "WeightedCluster::seqpropclust("
                f"{names['sequence']}, as.dist({names['diss']}), "
                f"properties=as.character({names['properties']}), "
                f"other.prop={names['other']}, "
                f"prop.only={str(bool(prop_only)).upper()}, "
                f"pmin.support={float(pmin_support)}, max.k={int(max_k)}, "
                f"with.missing={str(bool(with_missing)).upper()}, "
                f"R={int(R)}, pval={float(pval)}, "
                f"weight.permutation='{str(weight_permutation)}', "
                f"min.size={float(min_size)}, max.depth={int(max_depth)}, "
                f"maxcluster={maxcluster_expr}, squared={str(bool(squared)).upper()}, "
                f"first={names['first']})), "
                "error=function(e) e)"
            )
            unchanged = bool(
                self.r(
                    f"identical({names['sequence']}, {names['before_sequence']}) && "
                    f"identical({names['other']}, {names['before_other']})"
                )[0]
            )
            if bool(self.r(f"inherits({names['result']}, 'error')")[0]):
                return {
                    "ok": False,
                    "error": str(self.r(f"conditionMessage({names['result']})")[0]),
                    "inputs_unchanged": unchanged,
                }
            if not prop_only:
                payload = self._tree_contract(names["result"])
                data_names = [str(value) for value in self.r(f"names({names['result']}$data)")]
                data_levels: list[list[str]] = []
                for index in range(1, len(data_names) + 1):
                    levels = self.r(f"levels({names['result']}$data[[{index}]])")
                    data_levels.append(
                        [] if levels is ro.NULL else [str(value) for value in levels]
                    )
                payload.update(
                    {
                        "ok": True,
                        "inputs_unchanged": unchanged,
                        "data_names": data_names,
                        "data_classes": [
                            [
                                str(value)
                                for value in self.r(f"class({names['result']}$data[[{index}]])")
                            ]
                            for index in range(1, len(data_names) + 1)
                        ],
                        "data_levels": data_levels,
                        "weights": np.asarray(
                            self.r(f"as.numeric({names['result']}$weights)"),
                            dtype=float,
                        ).reshape(-1),
                        "parameter_names": [
                            str(value)
                            for value in self.r(f"names({names['result']}$info$parameters)")
                        ],
                        "parameter_values": np.asarray(
                            self.r(
                                f"as.numeric(unlist({names['result']}$info$parameters, "
                                "use.names=FALSE))"
                            ),
                            dtype=float,
                        ).reshape(-1),
                        "prune": (
                            None
                            if self.r(f"{names['result']}$info$prune") is ro.NULL
                            else int(self.r(f"{names['result']}$info$prune")[0])
                        ),
                    }
                )
                return payload

            result_names = [str(value) for value in self.r(f"names({names['result']})")]
            columns: list[dict[str, Any]] = []
            for index, column in enumerate(result_names, start=1):
                levels = self.r(f"levels({names['result']}[[{index}]])")
                columns.append(
                    {
                        "name": column,
                        "class": [
                            str(value) for value in self.r(f"class({names['result']}[[{index}]])")
                        ],
                        "typeof": str(self.r(f"typeof({names['result']}[[{index}]])")[0]),
                        "levels": [] if levels is ro.NULL else [str(value) for value in levels],
                        "values": [
                            str(value)
                            for value in self.r(
                                f"ifelse(is.na({names['result']}[[{index}]]), "
                                f"'<NA>', as.character({names['result']}[[{index}]]))"
                            )
                        ],
                        "numeric": np.asarray(
                            self.r(f"as.numeric({names['result']}[[{index}]])"),
                            dtype=float,
                        ).reshape(-1),
                    }
                )
            return {
                "ok": True,
                "class": [str(value) for value in self.r(f"class({names['result']})")],
                "names": result_names,
                "row_names": [str(value) for value in self.r(f"rownames({names['result']})")],
                "columns": columns,
                "inputs_unchanged": unchanged,
            }
        finally:
            cleanup = [
                *names.values(),
                *column_names,
                names["values"] + "_column_name",
                "tree_nodes",
                "tree_labels",
                "tree_members",
                "tree_split_prob",
                "tree_split_index",
                "tree_split_labels",
                "tree_max_k",
                "tree_cut_ids",
                "tree_cut_labels",
                "tree_adjustment",
            ]
            self.r(
                "rm(list=intersect(ls(envir=.GlobalEnv), c("
                + ",".join(f"'{name}'" for name in cleanup)
                + ")), envir=.GlobalEnv)"
            )

    def wc_property_clustering_contract(
        self,
        D: np.ndarray,
        properties: dict[str, np.ndarray],
        *,
        categorical: Sequence[str] = (),
        weights: np.ndarray | None = None,
        R: int = 1,
        pval: float = 0.01,
        min_size: float = 0.05,
        max_depth: int = 5,
        maxcluster: int | None = None,
        weight_permutation: str = "replicate",
        seed: int = 1,
    ) -> dict[str, Any]:
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self._set_property_frame(properties, categorical=categorical)
        self.r.assign("tree_R", ro.IntVector([int(R)]))
        self.r.assign("tree_pval", ro.FloatVector([float(pval)]))
        self.r.assign("tree_min_size", ro.FloatVector([float(min_size)]))
        self.r.assign("tree_max_depth", ro.IntVector([int(max_depth)]))
        self.r.assign("tree_seed", ro.IntVector([int(seed)]))
        self.r.assign("tree_weight_permutation", ro.StrVector([str(weight_permutation)]))
        self._set_int_vector_or_null(
            "tree_maxcluster",
            None if maxcluster is None else np.asarray([maxcluster], dtype=int),
        )
        self.r("set.seed(as.integer(tree_seed))")
        self.r(
            "tree_args <- list(diss=as.dist(D), properties=properties, "
            "maxcluster=tree_maxcluster, R=as.integer(tree_R), "
            "pval=as.numeric(tree_pval), "
            "min.size=as.numeric(tree_min_size), max.depth=as.integer(tree_max_depth), "
            "weight.permutation=as.character(tree_weight_permutation)); "
            "if (!is.null(weights)) tree_args$weights <- weights; "
            "tree <- do.call(WeightedCluster::wcPropertyClustering, tree_args)"
        )
        return self._tree_contract()

    def wc_property_clustering_complete_contract(
        self,
        D: np.ndarray,
        properties: dict[str, np.ndarray],
        *,
        categorical: Sequence[str] = (),
        factor_levels: Mapping[str, Sequence[Any]] | None = None,
        weights: np.ndarray | None = None,
        R: int = 1,
        pval: float = 0.01,
        min_size: float = 0.05,
        max_depth: int = 5,
        maxcluster: int | None = None,
        weight_permutation: str = "replicate",
        squared: bool = False,
        first: str | None = None,
        seed: int = 1,
    ) -> dict[str, Any]:
        """Return complete live-R property-tree semantics and failures."""
        token = uuid.uuid4().hex
        property_name = f"wc_property_frame_{token}"
        result_name = f"wc_property_result_{token}"
        before_diss = f"wc_property_before_diss_{token}"
        before_properties = f"wc_property_before_properties_{token}"
        before_weights = f"wc_property_before_weights_{token}"
        first_name = f"wc_property_first_{token}"
        level_names: list[str] = []
        fixed_tree_bindings = [
            "tree_nodes",
            "tree_labels",
            "tree_members",
            "tree_split_prob",
            "tree_split_index",
            "tree_split_labels",
            "tree_max_k",
            "tree_cut_ids",
            "tree_cut_labels",
            "tree_adjustment",
            "property_name",
            "property_values",
            "property_nrow",
        ]
        try:
            self._set_distance_matrix(D)
            self._set_weights(weights)
            self._set_property_frame(
                properties,
                categorical=categorical,
                r_name=property_name,
            )
            for index, (name, levels) in enumerate((factor_levels or {}).items()):
                level_name = f"wc_property_levels_{index}_{token}"
                level_names.append(level_name)
                self.r.assign(level_name, ro.StrVector([str(value) for value in levels]))
                self.r.assign("property_name", ro.StrVector([str(name)]))
                self.r(
                    f"{property_name}[[as.character(property_name)]] <- "
                    f"factor({property_name}[[as.character(property_name)]], "
                    f"levels={level_name})"
                )
            if first is None:
                self.r(f"{first_name} <- NULL")
            else:
                self.r.assign(first_name, ro.StrVector([str(first)]))
            self.r(f"{before_diss} <- D")
            self.r(f"{before_properties} <- {property_name}")
            self.r(f"{before_weights} <- weights")
            maxcluster_expr = "NULL" if maxcluster is None else str(int(maxcluster))
            self.r(f"set.seed({int(seed)})")
            self.r(
                f"{result_name} <- tryCatch(suppressMessages("
                "WeightedCluster::wcPropertyClustering("
                f"as.dist(D), {property_name}, maxcluster={maxcluster_expr}, "
                f"R={int(R)}, pval={float(pval)}, min.size={float(min_size)}, "
                f"max.depth={int(max_depth)}, "
                f"weight.permutation='{str(weight_permutation)}', "
                f"squared={str(bool(squared)).upper()}, first={first_name}, "
                "weights=weights)), error=function(e) e)"
            )
            unchanged = bool(
                self.r(
                    f"identical(D, {before_diss}) && "
                    f"identical({property_name}, {before_properties}) && "
                    f"identical(weights, {before_weights})"
                )[0]
            )
            if bool(self.r(f"inherits({result_name}, 'error')")[0]):
                return {
                    "ok": False,
                    "error": str(self.r(f"conditionMessage({result_name})")[0]),
                    "inputs_unchanged": unchanged,
                }

            payload = self._tree_contract(result_name)
            data_names = [str(value) for value in self.r(f"names({result_name}$data)")]
            payload.update(
                {
                    "ok": True,
                    "inputs_unchanged": unchanged,
                    "data_names": data_names,
                    "data_classes": [
                        [str(value) for value in self.r(f"class({result_name}$data[[{index}]])")]
                        for index in range(1, len(data_names) + 1)
                    ],
                    "data_levels": [
                        (
                            []
                            if self.r(f"levels({result_name}$data[[{index}]])") is ro.NULL
                            else [
                                str(value)
                                for value in self.r(f"levels({result_name}$data[[{index}]])")
                            ]
                        )
                        for index in range(1, len(data_names) + 1)
                    ],
                    "weights": np.asarray(
                        self.r(f"as.numeric({result_name}$weights)"),
                        dtype=float,
                    ).reshape(-1),
                    "info_names": [str(value) for value in self.r(f"names({result_name}$info)")],
                    "parameter_names": [
                        str(value) for value in self.r(f"names({result_name}$info$parameters)")
                    ],
                    "parameter_values": np.asarray(
                        self.r(
                            f"as.numeric(unlist({result_name}$info$parameters, " "use.names=FALSE))"
                        ),
                        dtype=float,
                    ).reshape(-1),
                    "prune": (
                        None
                        if self.r(f"{result_name}$info$prune") is ro.NULL
                        else int(self.r(f"{result_name}$info$prune")[0])
                    ),
                }
            )
            return payload
        finally:
            cleanup = [
                "D",
                "weights",
                property_name,
                result_name,
                before_diss,
                before_properties,
                before_weights,
                first_name,
                *level_names,
                *fixed_tree_bindings,
            ]
            self.r(
                "rm(list=intersect(ls(envir=.GlobalEnv), c("
                + ",".join(f"'{name}'" for name in cleanup)
                + ")), envir=.GlobalEnv)"
            )

    def dtcut_complete_contract(
        self,
        D: np.ndarray,
        properties: dict[str, np.ndarray],
        *,
        categorical: Sequence[str] = (),
        weights: np.ndarray | None = None,
        min_size: float = 0.05,
        max_depth: int = 5,
        k_values: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Execute every requested live-R ``dtcut`` labels branch."""
        token = uuid.uuid4().hex
        property_name = f"wc_dtcut_properties_{token}"
        tree_name = f"wc_dtcut_tree_{token}"
        before_name = f"wc_dtcut_before_{token}"
        try:
            self._set_distance_matrix(D)
            self._set_weights(weights)
            self._set_property_frame(
                properties,
                categorical=categorical,
                r_name=property_name,
            )
            self.r(
                f"{tree_name} <- suppressMessages("
                "WeightedCluster::wcPropertyClustering("
                f"as.dist(D), {property_name}, R=1L, "
                f"min.size={float(min_size)}, max.depth={int(max_depth)}, "
                "weights=weights))"
            )
            self.r(f"{before_name} <- {tree_name}")
            max_k = int(self.r(f"length(unique(as.integer({tree_name}$fitted[, 1])))")[0])
            requested = (
                [float(value) for value in k_values]
                if k_values is not None
                else [float(value) for value in range(1, max_k + 2)]
            )
            cuts: dict[str, dict[str, Any]] = {}
            for index, k in enumerate(requested):
                for labels in (False, True):
                    value_name = f"wc_dtcut_value_{index}_{int(labels)}_{token}"
                    self.r(
                        f"{value_name} <- tryCatch("
                        f"WeightedCluster:::dtcut({tree_name}, {float(k)}, "
                        f"labels={str(labels).upper()}), error=function(e) e)"
                    )
                    key = f"{k:g}:{str(labels).lower()}"
                    if bool(self.r(f"inherits({value_name}, 'error')")[0]):
                        cuts[key] = {
                            "ok": False,
                            "error": str(self.r(f"conditionMessage({value_name})")[0]),
                        }
                    else:
                        levels = self.r(f"levels({value_name})")
                        cuts[key] = {
                            "ok": True,
                            "class": [str(value) for value in self.r(f"class({value_name})")],
                            "values": [
                                str(value)
                                for value in self.r(
                                    f"ifelse(is.na({value_name}), '<NA>', "
                                    f"as.character({value_name}))"
                                )
                            ],
                            "numeric": np.asarray(
                                self.r(f"as.numeric({value_name})"),
                                dtype=float,
                            ).reshape(-1),
                            "levels": (
                                [] if levels is ro.NULL else [str(value) for value in levels]
                            ),
                        }
                    self.r(f"rm({value_name}, envir=.GlobalEnv)")

            invalid: dict[str, str] = {}
            for kind, expression in {
                "null": "NULL",
                "list": "list()",
                "fitted_only": "list(fitted=matrix(1:2, ncol=1))",
            }.items():
                value_name = f"wc_dtcut_invalid_{kind}_{token}"
                self.r(
                    f"{value_name} <- tryCatch("
                    f"WeightedCluster:::dtcut({expression}, 2L), error=function(e) e)"
                )
                invalid[kind] = str(self.r(f"conditionMessage({value_name})")[0])
                self.r(f"rm({value_name}, envir=.GlobalEnv)")

            return {
                "max_k": max_k,
                "cuts": cuts,
                "invalid": invalid,
                "inputs_unchanged": bool(self.r(f"identical({tree_name}, {before_name})")[0]),
            }
        finally:
            cleanup = [
                "D",
                "weights",
                "property_name",
                "property_values",
                "property_nrow",
                property_name,
                tree_name,
                before_name,
            ]
            self.r(
                "rm(list=intersect(ls(envir=.GlobalEnv), c("
                + ",".join(f"'{name}'" for name in cleanup)
                + ")), envir=.GlobalEnv)"
            )

    def seqpropclust_contract(
        self,
        X: np.ndarray,
        D: np.ndarray,
        other_properties: dict[str, np.ndarray],
        *,
        categorical: Sequence[str] = (),
        weights: np.ndarray | None = None,
        R: int = 1,
        min_size: float = 0.01,
        max_depth: int = 5,
        maxcluster: int | None = None,
        seed: int = 1,
    ) -> dict[str, Any]:
        X_arr = np.asarray(X)
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self._set_property_frame(
            other_properties, categorical=categorical, r_name="other_properties"
        )
        if X_arr.dtype.kind in {"O", "S", "U"}:
            self.r.assign("sequence_values", ro.StrVector([str(x) for x in X_arr.ravel()]))
        else:
            self.r.assign("sequence_values", ro.IntVector(np.asarray(X_arr, dtype=int).ravel()))
        self.r.assign("sequence_nrow", ro.IntVector([int(X_arr.shape[0])]))
        self.r.assign("sequence_ncol", ro.IntVector([int(X_arr.shape[1])]))
        self.r.assign("tree_R", ro.IntVector([int(R)]))
        self.r.assign("tree_min_size", ro.FloatVector([float(min_size)]))
        self.r.assign("tree_max_depth", ro.IntVector([int(max_depth)]))
        self.r.assign("tree_seed", ro.IntVector([int(seed)]))
        self._set_int_vector_or_null(
            "tree_maxcluster",
            None if maxcluster is None else np.asarray([maxcluster], dtype=int),
        )
        self.r(
            "sequence_matrix <- matrix(sequence_values, nrow=as.integer(sequence_nrow), "
            "ncol=as.integer(sequence_ncol), byrow=TRUE); "
            "sequence_data <- suppressMessages(TraMineR::seqdef(sequence_matrix, "
            "weights=weights)); "
            "set.seed(as.integer(tree_seed)); "
            "tree <- suppressMessages(WeightedCluster::seqpropclust("
            "sequence_data, as.dist(D), properties=character(), "
            "other.prop=other_properties, R=as.integer(tree_R), "
            "min.size=as.numeric(tree_min_size), max.depth=as.integer(tree_max_depth), "
            "maxcluster=tree_maxcluster))"
        )
        return self._tree_contract()

    def seqpropclust_properties(
        self,
        X: np.ndarray,
        properties: Sequence[str],
    ) -> dict[str, Any]:
        X_arr = np.asarray(X)
        if X_arr.dtype.kind in {"O", "S", "U"}:
            self.r.assign("sequence_values", ro.StrVector([str(x) for x in X_arr.ravel()]))
        else:
            self.r.assign("sequence_values", ro.IntVector(np.asarray(X_arr, dtype=int).ravel()))
        self.r.assign("sequence_nrow", ro.IntVector([int(X_arr.shape[0])]))
        self.r.assign("sequence_ncol", ro.IntVector([int(X_arr.shape[1])]))
        self.r.assign("property_groups", ro.StrVector([str(x) for x in properties]))
        self.r(
            "sequence_matrix <- matrix(sequence_values, nrow=as.integer(sequence_nrow), "
            "ncol=as.integer(sequence_ncol), byrow=TRUE); "
            "sequence_data <- suppressMessages(TraMineR::seqdef(sequence_matrix)); "
            "sequence_diss <- as.dist(matrix(0, nrow=as.integer(sequence_nrow), "
            "ncol=as.integer(sequence_nrow))); "
            "property_frame <- suppressMessages(WeightedCluster::seqpropclust("
            "sequence_data, sequence_diss, properties=as.character(property_groups), "
            "prop.only=TRUE))"
        )
        names = [str(x) for x in self.r("names(property_frame)")]
        values: dict[str, list[str]] = {}
        classes: dict[str, list[str]] = {}
        for index, name in enumerate(names, start=1):
            self.r.assign("property_index", ro.IntVector([index]))
            values[name] = [
                str(x)
                for x in self.r(
                    'ifelse(is.na(property_frame[[property_index]]), "<NA>", '
                    "as.character(property_frame[[property_index]]))"
                )
            ]
            classes[name] = [str(x) for x in self.r("class(property_frame[[property_index]])")]
        return {"names": names, "values": values, "classes": classes}

    def wc_cluster_quality(
        self, D: np.ndarray, clustering: np.ndarray, *, weights: np.ndarray | None = None
    ) -> dict[str, Any]:
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self._set_int_vector_or_null("clustering", clustering)
        self.r("cq <- wcClusterQuality(as.dist(D), clustering=clustering, weights=weights)")
        return {
            "stats": self._named_numeric_vector("cq$stats"),
            "asw": self._matrix("as.matrix(cq$ASW)"),
        }

    def wc_silhouette_obs(
        self,
        D: np.ndarray,
        clustering: np.ndarray,
        *,
        measure: str = "ASW",
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self._set_int_vector_or_null("clustering", clustering)
        self.r.assign("measure", ro.StrVector([str(measure)]))
        vals = self.r(
            "wcSilhouetteObs(as.dist(D), clustering=clustering, weights=weights, measure=as.character(measure))"
        )
        return np.asarray(vals, dtype=float).reshape(-1)

    def wc_matrix_versus_dist_paths(
        self,
        D: np.ndarray,
        clustering: Sequence[int],
        *,
        k: int = 2,
        initialclust: Sequence[int] = (1, 3),
    ) -> dict[str, dict[str, np.ndarray]]:
        """Call each entry point twice: with a bare matrix, and with ``as.dist``.

        Every other helper in this module hands R ``as.dist(D)``, which is the
        idiomatic input and what ``TraMineR::seqdist`` produces.  That is the
        right oracle for the ordinary contract, but it means no helper here can
        observe what R does when a caller passes a **bare matrix** — and R does
        not treat the two forms alike for every function.  This helper exists to
        measure that difference so the divergence it creates can be pinned
        rather than assumed.
        """
        self._set_distance_matrix(D)
        self._set_int_vector_or_null("clustering", clustering)
        self.r.assign("k", ro.IntVector([int(k)]))
        self.r.assign("initialclust", ro.IntVector([int(value) for value in initialclust]))
        expressions = {
            "wcClusterQuality": "as.numeric(wcClusterQuality({d}, cl)$stats)",
            "wcSilhouetteObs": "as.numeric(wcSilhouetteObs({d}, cl))",
            "as.clustrange": (
                "as.numeric(as.clustrange(data.frame(cluster2=factor(cl))," " diss={d})$stats[1, ])"
            ),
            "wcKMedoids": (
                "as.numeric(wcKMedoids({d}, k=as.integer(k),"
                " initialclust=initialclust, cluster.only=TRUE))"
            ),
        }
        self.r("cl <- clustering")
        out: dict[str, dict[str, np.ndarray]] = {}
        for name, template in expressions.items():
            out[name] = {
                form: np.asarray(self.r(template.format(d=expression)), dtype=float).reshape(-1)
                for form, expression in (("matrix", "D"), ("dist", "as.dist(D)"))
            }
        return out

    def wc_cluster_quality_complete_contract(
        self,
        D: np.ndarray,
        clustering: Sequence[Any],
        *,
        weights: np.ndarray | None = None,
        distance_kind: str = "dist",
        factor_levels: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Capture the complete live-R ``wcClusterQuality`` result and conditions."""
        D_array = np.asarray(D, dtype=float)
        if D_array.ndim != 2:
            raise ValueError("R quality contract expects a two-dimensional distance input")
        self._assign_matrix("D", D_array)
        self._set_weights(weights)
        cluster_values = list(clustering)
        if cluster_values and all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in cluster_values
        ):
            self.r.assign("phase2_clustering_raw", ro.IntVector(cluster_values))
        elif cluster_values and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in cluster_values
        ):
            self.r.assign("phase2_clustering_raw", ro.FloatVector(cluster_values))
        else:
            self.r.assign(
                "phase2_clustering_raw",
                ro.StrVector([str(value) for value in cluster_values]),
            )
        if factor_levels is None:
            self.r("phase2_clustering <- phase2_clustering_raw")
        else:
            self.r.assign(
                "phase2_factor_levels",
                ro.StrVector([str(value) for value in factor_levels]),
            )
            self.r(
                "phase2_clustering <- factor(as.character(phase2_clustering_raw), "
                "levels=as.character(phase2_factor_levels))"
            )
        if distance_kind not in {"matrix", "dist"}:
            raise ValueError("distance_kind must be 'matrix' or 'dist'")
        self.r("phase2_diss <- " + ("D" if distance_kind == "matrix" else "as.dist(D)"))
        self.r(
            "phase2_warnings <- character(); phase2_error <- NULL; "
            "phase2_diss_before <- phase2_diss; "
            "phase2_clustering_before <- phase2_clustering; "
            "phase2_weights_before <- weights; "
            "phase2_result <- withCallingHandlers("
            "  tryCatch("
            "    WeightedCluster::wcClusterQuality("
            "      phase2_diss, phase2_clustering, weights=weights), "
            "    error=function(e) { phase2_error <<- conditionMessage(e); NULL }"
            "  ), "
            "  warning=function(w) { "
            "    phase2_warnings <<- c(phase2_warnings, conditionMessage(w)); "
            '    invokeRestart("muffleWarning") '
            "  }"
            ")"
        )
        try:
            error = self.r("phase2_error")
            error_text = None if error is ro.NULL else str(error[0])
            warnings = [str(value) for value in self.r("phase2_warnings")]
            result: dict[str, Any] = {
                "error": error_text,
                "warnings": warnings,
                "inputs_unchanged": bool(
                    self.r(
                        "identical(phase2_diss, phase2_diss_before) && "
                        "identical(phase2_clustering, phase2_clustering_before) && "
                        "identical(weights, phase2_weights_before)"
                    )[0]
                ),
            }
            if error_text is not None:
                return result
            result.update(
                {
                    "class": [str(value) for value in self.r("class(phase2_result)")],
                    "names": [str(value) for value in self.r("names(phase2_result)")],
                    "stats_names": [str(value) for value in self.r("names(phase2_result$stats)")],
                    "stats": np.asarray(
                        self.r("as.numeric(phase2_result$stats)"), dtype=float
                    ).reshape(-1),
                    "asw_shape": [int(value) for value in self.r("dim(phase2_result$ASW)")],
                    "asw_row_names": [
                        str(value) for value in self.r("rownames(phase2_result$ASW)")
                    ],
                    "asw_column_names": [
                        str(value) for value in self.r("colnames(phase2_result$ASW)")
                    ],
                    "asw": self._matrix("as.matrix(phase2_result$ASW)"),
                }
            )
            return result
        finally:
            self.r(
                "rm(list=intersect(ls(), c("
                "'D', 'weights', 'phase2_clustering_raw', 'phase2_factor_levels', "
                "'phase2_clustering', 'phase2_diss', 'phase2_warnings', "
                "'phase2_error', 'phase2_diss_before', "
                "'phase2_clustering_before', 'phase2_weights_before', "
                "'phase2_result')))"
            )

    def wc_silhouette_obs_complete_contract(
        self,
        D: np.ndarray,
        clustering: Sequence[Any],
        *,
        measure: str | Sequence[str] = "ASW",
        weights: np.ndarray | None = None,
        distance_kind: str = "dist",
        factor_levels: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Capture complete live-R ``wcSilhouetteObs`` values and conditions."""
        D_array = np.asarray(D, dtype=float)
        if D_array.ndim != 2:
            raise ValueError("R silhouette contract expects a two-dimensional distance input")
        self._assign_matrix("D", D_array)
        self._set_weights(weights)
        cluster_values = list(clustering)
        if cluster_values and all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in cluster_values
        ):
            self.r.assign("phase2_clustering_raw", ro.IntVector(cluster_values))
        else:
            self.r.assign(
                "phase2_clustering_raw",
                ro.StrVector([str(value) for value in cluster_values]),
            )
        if factor_levels is None:
            self.r("phase2_clustering <- phase2_clustering_raw")
        else:
            self.r.assign(
                "phase2_factor_levels",
                ro.StrVector([str(value) for value in factor_levels]),
            )
            self.r(
                "phase2_clustering <- factor(as.character(phase2_clustering_raw), "
                "levels=as.character(phase2_factor_levels))"
            )
        if distance_kind not in {"matrix", "dist"}:
            raise ValueError("distance_kind must be 'matrix' or 'dist'")
        self.r("phase2_diss <- " + ("D" if distance_kind == "matrix" else "as.dist(D)"))
        measures = [measure] if isinstance(measure, str) else list(measure)
        self.r.assign(
            "phase2_measure",
            ro.StrVector([str(value) for value in measures]),
        )
        self.r(
            "phase2_warnings <- character(); phase2_error <- NULL; "
            "phase2_diss_before <- phase2_diss; "
            "phase2_clustering_before <- phase2_clustering; "
            "phase2_weights_before <- weights; "
            "phase2_result <- withCallingHandlers("
            "  tryCatch("
            "    WeightedCluster::wcSilhouetteObs("
            "      phase2_diss, phase2_clustering, weights=weights, "
            "      measure=phase2_measure), "
            "    error=function(e) { phase2_error <<- conditionMessage(e); NULL }"
            "  ), "
            "  warning=function(w) { "
            "    phase2_warnings <<- c(phase2_warnings, conditionMessage(w)); "
            '    invokeRestart("muffleWarning") '
            "  }"
            ")"
        )
        try:
            error = self.r("phase2_error")
            error_text = None if error is ro.NULL else str(error[0])
            result: dict[str, Any] = {
                "error": error_text,
                "warnings": [str(value) for value in self.r("phase2_warnings")],
                "inputs_unchanged": bool(
                    self.r(
                        "identical(phase2_diss, phase2_diss_before) && "
                        "identical(phase2_clustering, phase2_clustering_before) && "
                        "identical(weights, phase2_weights_before)"
                    )[0]
                ),
            }
            if error_text is not None:
                return result
            result.update(
                {
                    "class": [str(value) for value in self.r("class(phase2_result)")],
                    "shape": [
                        int(value)
                        for value in self.r(
                            "if (is.null(dim(phase2_result))) "
                            "length(phase2_result) else dim(phase2_result)"
                        )
                    ],
                    "column_names": [
                        str(value)
                        for value in self.r(
                            "if (is.null(dim(phase2_result))) character() "
                            "else colnames(phase2_result)"
                        )
                    ],
                    "values": np.asarray(
                        self.r("as.numeric(as.matrix(phase2_result))"), dtype=float
                    ),
                }
            )
            return result
        finally:
            self.r(
                "rm(list=intersect(ls(), c("
                "'D', 'weights', 'phase2_clustering_raw', 'phase2_factor_levels', "
                "'phase2_clustering', 'phase2_diss', 'phase2_measure', "
                "'phase2_warnings', 'phase2_error', 'phase2_diss_before', "
                "'phase2_clustering_before', 'phase2_weights_before', "
                "'phase2_result')))"
            )

    def wc_kmedoids_complete_contract(
        self,
        D: np.ndarray,
        *,
        k: int,
        weights: np.ndarray | None = None,
        npass: int = 1,
        initialclust: Sequence[Any] | np.ndarray | None = None,
        initial_kind: str = "medoids",
        method: str | int = "PAMonce",
        cluster_only: bool = False,
        debuglevel: int = 0,
        distance_kind: str = "dist",
        seed: int = 0,
        replay_starts_1b: Sequence[Sequence[int]] | np.ndarray | None = None,
        digits: int = 7,
    ) -> dict[str, Any]:
        """Capture complete installed-R ``wcKMedoids`` semantics and presentation."""
        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_kmedoids_{key}_{token}"
            for key in (
                "D",
                "weights",
                "initial_raw",
                "initial",
                "method",
                "diss",
                "fun",
                "body",
                "replay",
                "state",
                "warnings",
                "error",
                "debug",
                "result",
                "diss_before",
                "initial_before",
                "weights_before",
            )
        }
        D_array = np.asarray(D, dtype=float)
        if D_array.ndim != 2:
            raise ValueError("R k-medoids contract expects a two-dimensional distance input")
        if distance_kind not in {"matrix", "dist"}:
            raise ValueError("distance_kind must be 'matrix' or 'dist'")
        if initial_kind not in {"medoids", "membership", "factor", "hclust", "twins"}:
            raise ValueError("initial_kind must be medoids, membership, factor, hclust, or twins")

        try:
            self._assign_matrix(names["D"], D_array)
            if weights is None:
                self.r(f"{names['weights']} <- NULL")
            else:
                self.r.assign(
                    names["weights"],
                    ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
                )
            if initial_kind in {"hclust", "twins"}:
                if initial_kind == "hclust":
                    self.r(
                        f"{names['initial']} <- hclust(as.dist({names['D']}), " "method='single')"
                    )
                else:
                    self.r(
                        f"{names['initial']} <- cluster::agnes("
                        f"as.dist({names['D']}), diss=TRUE, method='average')"
                    )
            elif initialclust is None:
                self.r(f"{names['initial']} <- NULL")
            else:
                initial_values = list(np.asarray(initialclust, dtype=object).reshape(-1))
                if initial_kind == "factor":
                    self.r.assign(
                        names["initial_raw"],
                        ro.StrVector([str(value) for value in initial_values]),
                    )
                    self.r(f"{names['initial']} <- factor({names['initial_raw']})")
                elif all(
                    isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                    for value in initial_values
                ):
                    self.r.assign(
                        names["initial"],
                        ro.IntVector([int(value) for value in initial_values]),
                    )
                elif all(
                    isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, (bool, np.bool_))
                    for value in initial_values
                ):
                    self.r.assign(
                        names["initial"],
                        ro.FloatVector([float(value) for value in initial_values]),
                    )
                else:
                    self.r.assign(
                        names["initial"],
                        ro.StrVector([str(value) for value in initial_values]),
                    )
            if isinstance(method, str):
                self.r.assign(names["method"], ro.StrVector([method]))
            else:
                self.r.assign(names["method"], ro.IntVector([int(method)]))
            self.r(
                f"{names['diss']} <- "
                + (names["D"] if distance_kind == "matrix" else f"as.dist({names['D']})")
            )

            call_name = "WeightedCluster::wcKMedoids"
            if replay_starts_1b is not None:
                replay = np.asarray(replay_starts_1b, dtype=int)
                if replay.ndim != 2:
                    raise ValueError("replay_starts_1b must be a two-dimensional plan")
                self._assign_matrix(names["replay"], replay - 1)
                self.r(
                    f"{names['state']} <- new.env(parent=emptyenv()); "
                    f"{names['state']}$cursor <- 0L; "
                    f"{names['state']}$starts <- matrix("
                    f"as.integer({names['replay']}), "
                    f"nrow=nrow({names['replay']}), ncol=ncol({names['replay']}))"
                )
                self.r(
                    f"{names['fun']} <- WeightedCluster::wcKMedoids; "
                    f"{names['body']} <- as.list(body({names['fun']})); "
                    f"{names['body']}[[5]] <- substitute("
                    "InternalRandomSample <- function() { "
                    f"state <- get('{names['state']}', envir=.GlobalEnv); "
                    "state$cursor <- state$cursor + 1L; "
                    "if (state$cursor > nrow(state$starts)) "
                    "stop('replay plan exhausted'); "
                    "as.integer(state$starts[state$cursor, ]) "
                    "}); "
                    f"body({names['fun']}) <- as.call({names['body']})"
                )
                call_name = names["fun"]

            self.r(f"set.seed({int(seed)})")
            self.r(
                f"{names['warnings']} <- character(); {names['error']} <- NULL; "
                f"{names['diss_before']} <- {names['diss']}; "
                f"{names['initial_before']} <- {names['initial']}; "
                f"{names['weights_before']} <- {names['weights']}; "
                f"{names['debug']} <- capture.output("
                f"{names['result']} <- withCallingHandlers("
                "tryCatch("
                f"{call_name}("
                f"{names['diss']}, k=as.integer({int(k)}), "
                f"weights={names['weights']}, npass=as.integer({int(npass)}), "
                f"initialclust={names['initial']}, method={names['method']}, "
                f"cluster.only={str(bool(cluster_only)).upper()}, "
                f"debuglevel=as.integer({int(debuglevel)})), "
                f"error=function(e) {{ {names['error']} <<- conditionMessage(e); NULL }}), "
                f"warning=function(w) {{ {names['warnings']} <<- "
                f"c({names['warnings']}, conditionMessage(w)); "
                "invokeRestart('muffleWarning') }))"
            )
            error = self.r(names["error"])
            error_text = None if error is ro.NULL else str(error[0])
            result: dict[str, Any] = {
                "error": error_text,
                "warnings": [str(value) for value in self.r(names["warnings"])],
                "debug_lines": [str(value) for value in self.r(names["debug"])],
                "inputs_unchanged": bool(
                    self.r(
                        f"identical({names['diss']}, {names['diss_before']}) && "
                        f"identical({names['initial']}, {names['initial_before']}) && "
                        f"identical({names['weights']}, {names['weights_before']})"
                    )[0]
                ),
                "replay_consumed": (
                    int(self.r(f"{names['state']}$cursor")[0])
                    if replay_starts_1b is not None
                    else None
                ),
            }
            if error_text is not None:
                return result
            result_names = self.r(f"names({names['result']})")
            result.update(
                {
                    "class": [str(value) for value in self.r(f"class({names['result']})")],
                    "names": (
                        [] if result_names is ro.NULL else [str(value) for value in result_names]
                    ),
                }
            )
            if cluster_only:
                result["clustering"] = np.asarray(
                    self.r(f"as.integer({names['result']})"), dtype=int
                ).reshape(-1)
                return result
            result.update(
                {
                    "clustering": np.asarray(
                        self.r(f"as.integer({names['result']}$clustering)"), dtype=int
                    ).reshape(-1),
                    "info_names": [
                        str(value) for value in self.r(f"names({names['result']}$info)")
                    ],
                    "info": np.asarray(
                        self.r(f"as.numeric({names['result']}$info)"), dtype=float
                    ).reshape(-1),
                    "stats_names": [
                        str(value) for value in self.r(f"names({names['result']}$stats)")
                    ],
                    "stats": np.asarray(
                        self.r(f"as.numeric({names['result']}$stats)"), dtype=float
                    ).reshape(-1),
                    "asw_row_names": [
                        str(value) for value in self.r(f"rownames({names['result']}$ASW)")
                    ],
                    "asw_column_names": [
                        str(value) for value in self.r(f"colnames({names['result']}$ASW)")
                    ],
                    "asw": self._matrix(f"as.matrix({names['result']}$ASW)"),
                    "print_lines": [
                        str(value)
                        for value in self.r(
                            f"capture.output(print({names['result']}, "
                            f"digits=as.integer({int(digits)})))"
                        )
                    ],
                }
            )
            return result
        finally:
            self.r(
                "rm(list=intersect(ls(), c("
                + ",".join(f"'{name}'" for name in names.values())
                + ")))"
            )

    def as_clustrange_default(
        self,
        D: np.ndarray,
        cluster_columns: dict[str, np.ndarray],
        *,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        self._set_distance_matrix(D)
        self._set_weights(weights)

        self.r("obj <- NULL")
        for key, values in cluster_columns.items():
            col = np.asarray(values, dtype=int).reshape(-1)
            self.r.assign("tmpcol", ro.IntVector(col))
            self.r.assign("tmpname", ro.StrVector([str(key)]))
            self.r(
                "if (is.null(obj)) { "
                "  obj <- data.frame(tmp = as.integer(tmpcol)); "
                "  names(obj) <- as.character(tmpname) "
                "} else { "
                "  obj[[as.character(tmpname)]] <- as.integer(tmpcol) "
                "}"
            )

        self.r("cr <- as.clustrange(obj, diss=as.dist(D), weights=weights, R=1)")
        return {
            "kvals": np.asarray(self.r("as.integer(cr$kvals)"), dtype=int).reshape(-1),
            "stats_names": [str(x) for x in self.r("colnames(cr$stats)")],
            "stats_matrix": self._matrix("as.matrix(cr$stats)"),
        }

    def r_clustassoc_values(
        self,
        D: np.ndarray,
        clustrange: dict[str, Any],
        covar: np.ndarray,
        *,
        categorical: bool,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run installed WeightedCluster 2.0 ``clustassoc`` and model diagnostics."""
        kvals = [int(value) for value in clustrange["kvals"]]
        columns = {f"cluster{k}": clustrange["clustering"][f"cluster{k}"] for k in kvals}
        result = self.clustassoc_complete_contract(
            D,
            columns,
            covar,
            kvals=kvals,
            categorical=categorical,
            weights=weights,
            distance_kind="dist",
        )
        if result["error"] is not None:
            raise RuntimeError(result["error"])
        return {
            "class": result["class"],
            "colnames": result["column_names"],
            "rownames": result["row_names"],
            "numcluster": np.asarray(result["table"][:, 3], dtype=int),
            "core": result["table"],
            "model_diagnostics": result["model_diagnostics"],
        }

    def clustassoc_complete_contract(
        self,
        D: np.ndarray,
        cluster_columns: Mapping[str, Sequence[Any]],
        covar: Sequence[Any] | np.ndarray,
        *,
        kvals: Sequence[int] | None = None,
        stats_row_names: Sequence[str] | None = None,
        categorical: bool = False,
        covar_levels: Sequence[Any] | None = None,
        weights: Sequence[float] | np.ndarray | None = None,
        distance_kind: str = "matrix",
        malformed: str | None = None,
        plot_stat: str | Sequence[str] | None = None,
        plot_type: str = "b",
        plot_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture exact installed ``clustassoc`` and ``plot.clustassoc`` semantics."""
        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_clustassoc_{key}_{token}"
            for key in (
                "D",
                "diss",
                "weights",
                "covar",
                "levels",
                "clustering",
                "stats",
                "kvals",
                "object",
                "result",
                "warnings",
                "error",
                "before_D",
                "before_diss",
                "before_weights",
                "before_covar",
                "before_object",
                "direct",
                "model_diag",
                "plot_stat",
                "plot_type",
                "plot_args",
                "plot_warnings",
                "plot_error",
            )
        }
        if distance_kind not in {"matrix", "dist"}:
            raise ValueError("distance_kind must be 'matrix' or 'dist'")
        allowed_malformed = {
            None,
            "missing_stats",
            "stats_two_columns",
            "missing_clustering",
            "clustering_row_mismatch",
            "stats_row_mismatch",
            "kvals_mismatch",
        }
        if malformed not in allowed_malformed:
            raise ValueError(f"unsupported malformed contract: {malformed}")

        def _r_vector(values: Sequence[Any] | np.ndarray):
            flattened = list(np.asarray(values, dtype=object).reshape(-1))
            non_missing = [
                value
                for value in flattened
                if value is not None
                and not (isinstance(value, (float, np.floating)) and np.isnan(float(value)))
            ]
            numeric = all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in non_missing
            )
            if numeric:
                return ro.FloatVector(
                    [
                        (
                            ro.NA_Real
                            if value is None
                            or (isinstance(value, (float, np.floating)) and np.isnan(float(value)))
                            else float(value)
                        )
                        for value in flattened
                    ]
                )
            return ro.StrVector(
                [
                    (
                        ro.NA_Character
                        if value is None
                        or (isinstance(value, (float, np.floating)) and np.isnan(float(value)))
                        else str(value)
                    )
                    for value in flattened
                ]
            )

        def _assign_plot_value(name: str, value: Any) -> None:
            if isinstance(value, (str, np.str_)):
                self.r.assign(name, ro.StrVector([str(value)]))
            elif isinstance(value, (bool, np.bool_)):
                self.r.assign(name, ro.BoolVector([bool(value)]))
            elif np.isscalar(value):
                self.r.assign(name, ro.FloatVector([float(value)]))
            else:
                self.r.assign(name, _r_vector(value))

        columns = list(cluster_columns.items())
        resolved_kvals = (
            [int(value) for value in kvals]
            if kvals is not None
            else list(range(2, 2 + len(columns)))
        )
        row_names = (
            [str(value) for value in stats_row_names]
            if stats_row_names is not None
            else [str(name) for name, _ in columns]
        )
        try:
            D_array = np.asarray(D, dtype=float)
            self._assign_matrix(names["D"], D_array)
            self.r(
                f"{names['diss']} <- "
                + (names["D"] if distance_kind == "matrix" else f"as.dist({names['D']})")
            )
            if weights is None:
                self.r(f"{names['weights']} <- NULL")
            else:
                self.r.assign(names["weights"], _r_vector(weights))
            self.r.assign(names["covar"], _r_vector(covar))
            if categorical:
                if covar_levels is None:
                    self.r(f"{names['covar']} <- factor({names['covar']})")
                else:
                    self.r.assign(names["levels"], _r_vector(covar_levels))
                    self.r(
                        f"{names['covar']} <- factor({names['covar']}, "
                        f"levels={names['levels']})"
                    )
            self.r(
                f"{names['clustering']} <- data.frame(" f"row.names=seq_len(nrow({names['D']})))"
            )
            for index, (column_name, values) in enumerate(columns):
                value_name = f"{names['clustering']}_column_{index}"
                self.r.assign(value_name, _r_vector(values))
                self.r(f"{names['clustering']}[['{str(column_name)}']] <- {value_name}")
            self.r.assign(names["kvals"], ro.IntVector(resolved_kvals))
            self.r.assign(
                names["stats"],
                self.r.matrix(
                    ro.FloatVector([0.0] * (max(1, len(columns)) * 3)),
                    nrow=max(1, len(columns)),
                    ncol=3,
                ),
            )
            self.r(f"{names['stats']} <- as.data.frame({names['stats']})")
            if row_names:
                row_name_var = f"{names['stats']}_rownames"
                self.r.assign(row_name_var, ro.StrVector(row_names))
                self.r(f"rownames({names['stats']}) <- {row_name_var}")
            self.r(
                f"{names['object']} <- list(kvals={names['kvals']}, "
                f"clustering={names['clustering']}, stats={names['stats']})"
            )
            if malformed == "missing_stats":
                self.r(f"{names['object']}$stats <- NULL")
            elif malformed == "stats_two_columns":
                self.r(f"{names['object']}$stats <- {names['object']}$stats[, 1:2]")
            elif malformed == "missing_clustering":
                self.r(f"{names['object']}$clustering <- NULL")
            elif malformed == "clustering_row_mismatch":
                self.r(
                    f"{names['object']}$clustering <- "
                    f"{names['object']}$clustering[-1, , drop=FALSE]"
                )
            elif malformed == "stats_row_mismatch":
                self.r(f"{names['object']}$stats <- " f"{names['object']}$stats[1, , drop=FALSE]")
            elif malformed == "kvals_mismatch":
                self.r(f"{names['object']}$kvals <- {names['object']}$kvals[1]")

            self.r(
                f"{names['before_D']} <- {names['D']}; "
                f"{names['before_diss']} <- {names['diss']}; "
                f"{names['before_weights']} <- {names['weights']}; "
                f"{names['before_covar']} <- {names['covar']}; "
                f"{names['before_object']} <- {names['object']}; "
                f"{names['warnings']} <- character(); {names['error']} <- NULL; "
                f"{names['result']} <- withCallingHandlers(tryCatch("
                f"WeightedCluster::clustassoc({names['object']}, {names['diss']}, "
                f"{names['covar']}, weights={names['weights']}), "
                f"error=function(e) {{ {names['error']} <<- conditionMessage(e); NULL }}), "
                f"warning=function(w) {{ {names['warnings']} <<- "
                f"c({names['warnings']}, conditionMessage(w)); "
                "invokeRestart('muffleWarning') })"
            )
            error = self.r(names["error"])
            error_text = None if error is ro.NULL else str(error[0])
            result: dict[str, Any] = {
                "error": error_text,
                "warnings": [str(value) for value in self.r(names["warnings"])],
                "inputs_unchanged": bool(
                    self.r(
                        f"identical({names['D']}, {names['before_D']}) && "
                        f"identical({names['diss']}, {names['before_diss']}) && "
                        f"identical({names['weights']}, {names['before_weights']}) && "
                        f"identical({names['covar']}, {names['before_covar']}) && "
                        f"identical({names['object']}, {names['before_object']})"
                    )[0]
                ),
            }
            if error_text is not None:
                return result

            self.r(
                f"{names['direct']} <- local({{ "
                f"null <- suppressWarnings(TraMineR::dissmfacw("
                f"{names['diss']} ~ {names['covar']}, R=1, data=NULL, "
                f"weights={names['weights']})); "
                f"out <- matrix(NA_real_, nrow=nrow({names['result']}), ncol=2); "
                "out[1, ] <- c(1, null$mfac[1, 3]); "
                f"for (i in seq_len(ncol({names['object']}$clustering))) {{ "
                f"  clustf <- factor({names['object']}$clustering[, i]); "
                f"  mm <- suppressWarnings(TraMineR::dissmfacw("
                f"    {names['diss']} ~ clustf + {names['covar']}, R=1, "
                f"    data=NULL, weights={names['weights']})); "
                "  out[i + 1, ] <- c(mm$mfac[2, 3] / "
                "    (mm$mfac[3, 3] - mm$mfac[1, 3]), mm$mfac[2, 3]) "
                "}; out })"
            )
            self.r(
                f"{names['model_diag']} <- local({{ mods <- list(); "
                f"if (is.numeric({names['covar']})) {{ "
                f"  mods[[1]] <- lm({names['covar']} ~ 1, weights={names['weights']}); "
                f"  for (i in seq_len(ncol({names['object']}$clustering))) {{ "
                f"    clustf <- factor({names['object']}$clustering[, i]); "
                f"    mods[[i + 1]] <- lm({names['covar']} ~ clustf, "
                f"      weights={names['weights']}) "
                "  } "
                "} else { "
                f"  mods[[1]] <- suppressWarnings(nnet::multinom({names['covar']} ~ 1, "
                f"    weights={names['weights']}, trace=FALSE)); "
                f"  for (i in seq_len(ncol({names['object']}$clustering))) {{ "
                f"    clustf <- factor({names['object']}$clustering[, i]); "
                f"    mods[[i + 1]] <- suppressWarnings(nnet::multinom("
                f"      {names['covar']} ~ clustf, weights={names['weights']}, "
                "      trace=FALSE)) "
                "  } "
                "}; t(vapply(mods, function(mod) { ll <- logLik(mod); "
                "c(BIC=BIC(mod), logLik=as.numeric(ll), df=attr(ll, 'df'), "
                "nobs=attr(ll, 'nobs'), convergence="
                "if (is.null(mod$convergence)) 0 else mod$convergence) "
                "}, numeric(5))) })"
            )
            result.update(
                {
                    "class": [str(value) for value in self.r(f"class({names['result']})")],
                    "column_names": [
                        str(value) for value in self.r(f"colnames({names['result']})")
                    ],
                    "row_names": [str(value) for value in self.r(f"rownames({names['result']})")],
                    "table": self._matrix(f"as.matrix({names['result']})"),
                    "direct_association": self._matrix(names["direct"]),
                    "model_diagnostics": self._matrix(names["model_diag"]),
                }
            )

            if plot_stat is not None:
                selected = [plot_stat] if isinstance(plot_stat, str) else list(plot_stat)
                self.r.assign(names["plot_stat"], ro.StrVector([str(value) for value in selected]))
                self.r.assign(names["plot_type"], ro.StrVector([str(plot_type)]))
                self.r(f"{names['plot_args']} <- list()")
                for index, (key, value) in enumerate((plot_options or {}).items()):
                    value_name = f"{names['plot_args']}_value_{index}"
                    _assign_plot_value(value_name, value)
                    self.r(f"{names['plot_args']}[['{str(key)}']] <- {value_name}")
                self.r(
                    f"{names['plot_warnings']} <- character(); "
                    f"{names['plot_error']} <- NULL; "
                    "withCallingHandlers(tryCatch(local({ "
                    "  target <- tempfile(fileext='.pdf'); "
                    "  grDevices::pdf(target); on.exit({ grDevices::dev.off(); "
                    "    unlink(target) }, add=TRUE); "
                    f"  do.call(graphics::plot, c(list(x={names['result']}, "
                    f"    stat={names['plot_stat']}, type={names['plot_type']}), "
                    f"    {names['plot_args']})); TRUE "
                    "}), error=function(e) { "
                    f"{names['plot_error']} <<- conditionMessage(e); FALSE }}), "
                    f"warning=function(w) {{ {names['plot_warnings']} <<- "
                    f"c({names['plot_warnings']}, conditionMessage(w)); "
                    "invokeRestart('muffleWarning') })"
                )
                plot_error = self.r(names["plot_error"])
                plot_error_text = None if plot_error is ro.NULL else str(plot_error[0])
                titles_name = f"{names['plot_args']}_titles"
                self.r(
                    f"{titles_name} <- c("
                    "residualProp='Unreproduced proportion of the association', "
                    "residual='Unexplained association', BIC='BIC', "
                    "BICgain='Gain in BIC')"
                )
                selected_name = str(selected[0]) if selected else ""
                ylab = self.r(f"unname({titles_name}[as.character({names['plot_stat']}[1])])")
                result["plot"] = {
                    "error": plot_error_text,
                    "warnings": [str(value) for value in self.r(names["plot_warnings"])],
                    "stat": selected_name,
                    "x": np.asarray(
                        self.r(f"as.numeric({names['result']}$numcluster)"),
                        dtype=float,
                    ).reshape(-1),
                    "y": (
                        None
                        if plot_error_text is not None
                        else np.asarray(
                            self.r(
                                f"as.numeric({names['result']}[, "
                                f"as.character({names['plot_stat']}[1])])"
                            ),
                            dtype=float,
                        ).reshape(-1)
                    ),
                    "xlab": "Number of clusters",
                    "ylab": (
                        None
                        if len(ylab) == 0 or bool(self.r(f"is.na({ylab.r_repr()})")[0])
                        else str(ylab[0])
                    ),
                    "type": str(plot_type),
                    "options": dict(plot_options or {}),
                    "rendered": plot_error_text is None,
                }
            return result
        finally:
            self.r(
                "rm(list=grep("
                f"'{token}', ls(envir=.GlobalEnv), value=TRUE), "
                "envir=.GlobalEnv)"
            )

    def wc_cmp_cluster(
        self,
        D: np.ndarray,
        *,
        maxcluster: int,
        method: str,
        weights: np.ndarray | None = None,
        pam_combine: bool = False,
    ) -> dict[str, Any]:
        self._set_distance_matrix(D)
        self._set_weights(weights)
        self.r.assign("maxcluster", ro.IntVector([int(maxcluster)]))
        self.r.assign("method", ro.StrVector([str(method)]))
        self.r.assign("pam_combine", ro.BoolVector([bool(pam_combine)]))
        # Compute a method-specific reference directly to avoid wrapper-specific
        # issues in some WeightedCluster versions.
        self.r("dd <- as.dist(D)")
        self.r("meth <- tolower(as.character(method))")
        # `fastcluster::hclust` is qualified explicitly. An unqualified `hclust`
        # in a session with `library(WeightedCluster)` attached resolves to
        # `stats::hclust`, while the wrapper this helper stands in for resolves
        # it from WeightedCluster's own namespace, where NAMESPACE binds it to
        # fastcluster. Certifying `wcCmpCluster` against the wrong `hclust` is
        # finding F-03 of the 2026-08-24 whole-codebase audit, and it concealed
        # the S1 partition divergence recorded as F-01.
        self.r(
            "if (meth %in% c('ward', 'ward.d', 'ward.d2', 'single', 'complete', "
            "'average', 'mcquitty', 'median', 'centroid')) { "
            "  hc <- fastcluster::hclust(dd, method=as.character(method), members=weights); "
            "  cr <- as.clustrange(hc, diss=dd, weights=weights, ncluster=as.integer(maxcluster)); "
            "} else if (meth == 'diana') { "
            "  hc <- diana(D, diss=TRUE); "
            "  cr <- as.clustrange(hc, diss=dd, weights=weights, ncluster=as.integer(maxcluster)); "
            "} else if (meth == 'beta.flexible' || meth == 'flexible') { "
            "  hc <- agnes(D, diss=TRUE, method='flexible', par.method=0.625); "
            "  cr <- as.clustrange(hc, diss=dd, weights=weights, ncluster=as.integer(maxcluster)); "
            "} else if (meth == 'pam') { "
            "  cr <- wcKMedRange(dd, kvals=2:as.integer(maxcluster), weights=weights); "
            "} else { "
            "  stop(paste('Unsupported method for parity helper:', as.character(method))); "
            "}"
        )

        kvals = np.asarray(self.r("as.integer(cr[['kvals']])"), dtype=int).reshape(-1)
        clustering: dict[str, np.ndarray] = {}
        for k in kvals.tolist():
            lbl = np.asarray(
                self.r(f"as.integer(cr[['clustering']][['cluster{int(k)}']])"),
                dtype=int,
            ).reshape(-1)
            clustering[f"cluster{int(k)}"] = lbl

        return {
            "kvals": kvals,
            "clustering": clustering,
            "stats_names": [str(x) for x in self.r("colnames(cr[['stats']])")],
            "stats_matrix": self._matrix("as.matrix(cr[['stats']])"),
        }

    def fastcluster_hclust_contract(
        self,
        D: np.ndarray,
        *,
        method: str,
        maxcluster: int,
        members: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Capture ``fastcluster::hclust`` merge heights and merge-count cuts.

        This is the binding ``WeightedCluster``'s NAMESPACE holds, so it is the
        oracle for every Python route that corresponds to an unqualified
        ``hclust`` call inside the R package — ``wcCmpCluster``,
        ``seqclararange``'s CLARA initialization and ``seqnullcqi``'s
        reclustering.

        It was previously named ``hclust_contract`` and called ``stats::hclust``.
        Finding F-03 of the 2026-08-24 whole-codebase audit established that
        ``stats::hclust`` is a different implementation with a different
        equal-distance merge rule, so certifying those routes against it could
        not support the grade it was cited for. A claim genuinely about
        ``stats::hclust`` needs its own helper; nothing in this suite makes one.
        """
        token = uuid.uuid4().hex
        names = {
            key: f"wc_hclust_{key}_{token}" for key in ("D", "method", "members", "result", "cuts")
        }
        self._assign_matrix(names["D"], np.asarray(D, dtype=float))
        self.r.assign(names["method"], ro.StrVector([str(method)]))
        if members is None:
            self.r(f"{names['members']} <- NULL")
        else:
            self.r.assign(
                names["members"],
                ro.FloatVector(np.asarray(members, dtype=float).reshape(-1)),
            )
        try:
            self.r(
                f"{names['result']} <- fastcluster::hclust(as.dist({names['D']}), "
                f"method=as.character({names['method']}), members={names['members']}); "
                f"{names['cuts']} <- sapply(2:{int(maxcluster)}, function(k) "
                f"stats::cutree({names['result']}, k=k))"
            )
            return {
                "height": np.asarray(
                    self.r(f"as.numeric({names['result']}$height)"), dtype=float
                ).reshape(-1),
                "cuts": self._matrix(f"as.matrix({names['cuts']})"),
            }
        finally:
            self.r(
                "rm(list=intersect(ls(envir=.GlobalEnv), c("
                + ",".join(f"'{name}'" for name in names.values())
                + ")), envir=.GlobalEnv)"
            )

    def seqclararange_ari_boxplot_data(
        self,
        *,
        kvals: Sequence[int],
        arimatrices: Sequence[np.ndarray],
        iterations: Sequence[int],
    ) -> dict[int, np.ndarray]:
        """Capture the data frame passed by installed R to ``boxplot``.

        The installed S3 function body is invoked directly.  Only its lexical
        ``boxplot`` dependency is replaced with a recorder, so this is an
        oracle capture rather than a Python reconstruction of the R indexing.
        """

        kval_values = [int(value) for value in kvals]
        iteration_values = [int(value) for value in iterations]
        if not (len(kval_values) == len(arimatrices) == len(iteration_values)):
            raise ValueError("kvals, arimatrices, and iterations must have equal lengths")
        token = uuid.uuid4().hex
        names = {
            key: f"wc_ari_boxplot_{key}_{token}"
            for key in ("kvals", "iterations", "object", "function", "environment", "data")
        }
        matrix_names: list[str] = []
        self.r.assign(names["kvals"], ro.IntVector(kval_values))
        self.r.assign(names["iterations"], ro.IntVector(iteration_values))
        for index, matrix in enumerate(arimatrices):
            matrix_name = f"wc_ari_boxplot_matrix_{index}_{token}"
            matrix_names.append(matrix_name)
            self._assign_matrix(matrix_name, np.asarray(matrix, dtype=float))
        clara_expression = ",".join(
            f"list(arimatrix={matrix_name}, iteration={names['iterations']}[[{index + 1}]], "
            f"iter.objective=seq_len(nrow({matrix_name})))"
            for index, matrix_name in enumerate(matrix_names)
        )
        cleanup_names = [*names.values(), *matrix_names]
        try:
            self.r(
                f"{names['object']} <- list(kvals={names['kvals']}, "
                f"clara=list({clara_expression})); "
                f"class({names['object']}) <- 'seqclararange'; "
                f"{names['function']} <- getS3method('plot', 'seqclararange'); "
                f"{names['environment']} <- new.env(parent=environment({names['function']})); "
                f"{names['environment']}$captured <- NULL; "
                f"evalq(boxplot <- function(formula, data, ...) {{ "
                "captured <<- data; invisible(NULL) }, "
                f"envir={names['environment']}); "
                f"environment({names['function']}) <- {names['environment']}; "
                f"{names['function']}({names['object']}, stat='ari.boxplot'); "
                f"{names['data']} <- {names['environment']}$captured"
            )
            captured = self._matrix(f"as.matrix({names['data']}[, c('kvals', 'ari'), drop=FALSE])")
            return {
                kval: np.asarray(captured[captured[:, 0] == kval, 1], dtype=float)
                for kval in kval_values
            }
        finally:
            self.r(
                "rm(list=intersect(ls(envir=.GlobalEnv), c("
                + ",".join(f"'{name}'" for name in cleanup_names)
                + ")), envir=.GlobalEnv)"
            )

    def wc_cmp_cluster_complete_contract(
        self,
        D: np.ndarray,
        *,
        maxcluster: int,
        method: str | Sequence[str] = "all",
        weights: np.ndarray | None = None,
        pam_combine: bool = True,
        distance_kind: str = "matrix",
        replay_starts_1b: Mapping[int, Sequence[int]] | None = None,
        max_rank: int = 1,
        plot_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture the installed ``wcCmpCluster`` family contract.

        Explicit one-based starts replace the plain ``pam`` branch after the
        public wrapper returns.  The replacement uses installed
        ``wcKMedoids`` and ``as.clustrange`` primitives, so Python can consume
        the same deterministic plan without relying on cross-language RNG
        equivalence.
        """

        if distance_kind not in {"matrix", "dist"}:
            raise ValueError("distance_kind must be 'matrix' or 'dist'")
        token = uuid.uuid4().hex
        names = {
            key: f"wc_phase2_cmpcluster_{key}_{token}"
            for key in (
                "D",
                "diss",
                "weights",
                "method",
                "result",
                "warnings",
                "error",
                "before_diss",
                "before_weights",
                "summary",
                "print_lines",
                "print_error",
                "hc_backup",
            )
        }
        replay_names: dict[int, str] = {}
        self._assign_matrix(names["D"], np.asarray(D, dtype=float))
        self.r(
            f"{names['diss']} <- "
            + (f"as.dist({names['D']})" if distance_kind == "dist" else f"{names['D']}")
        )
        if weights is None:
            self.r(f"{names['weights']} <- NULL")
        else:
            self.r.assign(
                names["weights"],
                ro.FloatVector(np.asarray(weights, dtype=float).reshape(-1)),
            )
        methods = [method] if isinstance(method, str) else list(method)
        self.r.assign(names["method"], ro.StrVector([str(value) for value in methods]))
        had_global_hc = bool(self.r("exists('hc', envir=.GlobalEnv, inherits=FALSE)")[0])
        if had_global_hc:
            self.r(
                f"{names['hc_backup']} <- get('hc', envir=.GlobalEnv); "
                "rm('hc', envir=.GlobalEnv)"
            )
        for k, starts in (replay_starts_1b or {}).items():
            replay_name = f"wc_phase2_cmpcluster_start_{int(k)}_{token}"
            replay_names[int(k)] = replay_name
            self.r.assign(
                replay_name,
                ro.IntVector(np.asarray(starts, dtype=int).reshape(-1)),
            )

        replay_expr = ""
        if replay_names:
            starts_expr = ",".join(
                f"'{int(k)}'=get('{name}', envir=.GlobalEnv)" for k, name in replay_names.items()
            )
            replay_expr = (
                f"; starts <- list({starts_expr}); kvals <- value$param$kvals; "
                "clustering <- lapply(kvals, function(k) "
                "WeightedCluster::wcKMedoids("
                f"{names['diss']}, k=as.integer(k), weights={names['weights']}, "
                "initialclust=starts[[as.character(k)]], cluster.only=TRUE)); "
                "names(clustering) <- paste0('cluster', kvals); "
                "base <- WeightedCluster::as.clustrange("
                f"as.data.frame(clustering), diss={names['diss']}, "
                f"weights={names['weights']}); "
                "pam <- list(R=1, kvals=as.integer(kvals), "
                "clustering=base$clustering, stats=base$stats); "
                "class(pam) <- c('clustrange', 'list'); value$pam <- pam; "
                "allstats <- lapply(value$param$all.methods, function(name) "
                "cbind(value[[name]]$stats, method=name, ngroup=kvals)); "
                "names(allstats) <- value$param$all.methods; "
                "value$allstats <- do.call(rbind, allstats)"
            )

        try:
            self.r(
                f"{names['warnings']} <- character(); {names['error']} <- NULL; "
                f"{names['before_diss']} <- {names['diss']}; "
                f"{names['before_weights']} <- {names['weights']}; "
                f"{names['result']} <- withCallingHandlers(tryCatch(local({{ "
                f"value <- WeightedCluster::wcCmpCluster({names['diss']}, "
                f"weights={names['weights']}, maxcluster=as.integer({int(maxcluster)}), "
                f"method={names['method']}, "
                f"pam.combine={'TRUE' if pam_combine else 'FALSE'})"
                f"{replay_expr}; value }}), "
                f"error=function(e) {{ {names['error']} <<- conditionMessage(e); NULL }}), "
                f"warning=function(w) {{ {names['warnings']} <<- "
                f"c({names['warnings']}, conditionMessage(w)); "
                "invokeRestart('muffleWarning') })"
            )
            error = self.r(names["error"])
            error_text = None if error is ro.NULL else str(error[0])
            payload: dict[str, Any] = {
                "error": error_text,
                "warnings": [str(value) for value in self.r(names["warnings"])],
                "inputs_unchanged": bool(
                    self.r(
                        f"identical({names['diss']}, {names['before_diss']}) && "
                        f"identical({names['weights']}, {names['before_weights']})"
                    )[0]
                ),
            }
            if error_text is not None:
                return payload
            if bool(self.r(f"is.null({names['result']}$allstats)")[0]):
                payload.update(
                    {
                        "class": [str(value) for value in self.r(f"class({names['result']})")],
                        "names": [str(value) for value in self.r(f"names({names['result']})")],
                        "param_names": [
                            str(value) for value in self.r(f"names({names['result']}$param)")
                        ],
                        "param_method": [
                            str(value) for value in self.r(f"{names['result']}$param$method")
                        ],
                        "param_pam_combine": bool(
                            self.r(f"{names['result']}$param$pam.combine")[0]
                        ),
                        "param_all_methods": [],
                        "param_kvals": [
                            int(value)
                            for value in self.r(f"as.integer({names['result']}$param$kvals)")
                        ],
                        "methods": {},
                        "allstats_column_names": [],
                        "allstats_row_names": [],
                        "allstats_values": np.empty((0, 10), dtype=float),
                        "allstats_method": [],
                        "allstats_ngroup": [],
                        "summary": None,
                        "summary_row_names": [],
                        "summary_column_names": [],
                        "print_lines": [],
                        "print_error": "family contains no clustering methods",
                    }
                )
                return payload

            self.r(
                f"{names['summary']} <- summary({names['result']}, "
                f"max.rank=as.integer({int(max_rank)})); "
                f"{names['print_error']} <- NULL; "
                f"{names['print_lines']} <- tryCatch("
                f"capture.output(print({names['result']}, "
                f"max.rank=as.integer({int(max_rank)}))), "
                f"error=function(e) {{ {names['print_error']} <<- "
                "conditionMessage(e); character() })"
            )
            method_names = [str(value) for value in self.r(f"{names['result']}$param$all.methods")]
            method_payload: dict[str, Any] = {}
            for method_name in method_names:
                escaped = method_name.replace("\\", "\\\\").replace("'", "\\'")
                expr = f"{names['result']}[['{escaped}']]"
                method_payload[method_name] = {
                    "class": [str(value) for value in self.r(f"class({expr})")],
                    "names": [str(value) for value in self.r(f"names({expr})")],
                    "R": (
                        None
                        if bool(self.r(f"is.null({expr}$R)")[0])
                        else int(self.r(f"as.integer({expr}$R)")[0])
                    ),
                    "kvals": [int(value) for value in self.r(f"as.integer({expr}$kvals)")],
                    "clustering_column_names": [
                        str(value) for value in self.r(f"colnames({expr}$clustering)")
                    ],
                    "clustering": self._matrix(
                        f"as.matrix(data.frame(lapply({expr}$clustering, as.integer)))"
                    ),
                    "stats_column_names": [
                        str(value) for value in self.r(f"colnames({expr}$stats)")
                    ],
                    "stats_row_names": [str(value) for value in self.r(f"rownames({expr}$stats)")],
                    "stats": self._matrix(f"as.matrix({expr}$stats)"),
                }
            print_error = self.r(names["print_error"])
            payload.update(
                {
                    "class": [str(value) for value in self.r(f"class({names['result']})")],
                    "names": [str(value) for value in self.r(f"names({names['result']})")],
                    "param_names": [
                        str(value) for value in self.r(f"names({names['result']}$param)")
                    ],
                    "param_method": [
                        str(value) for value in self.r(f"{names['result']}$param$method")
                    ],
                    "param_pam_combine": bool(self.r(f"{names['result']}$param$pam.combine")[0]),
                    "param_all_methods": method_names,
                    "param_kvals": [
                        int(value) for value in self.r(f"as.integer({names['result']}$param$kvals)")
                    ],
                    "methods": method_payload,
                    "allstats_column_names": [
                        str(value) for value in self.r(f"colnames({names['result']}$allstats)")
                    ],
                    "allstats_row_names": [
                        str(value) for value in self.r(f"rownames({names['result']}$allstats)")
                    ],
                    "allstats_values": self._matrix(
                        f"as.matrix({names['result']}$allstats[, 1:10])"
                    ),
                    "allstats_method": [
                        str(value) for value in self.r(f"{names['result']}$allstats$method")
                    ],
                    "allstats_ngroup": [
                        int(value)
                        for value in self.r(f"as.integer({names['result']}$allstats$ngroup)")
                    ],
                    "summary": np.asarray(
                        self.r(f"as.matrix({names['summary']})"), dtype=str
                    ).reshape(
                        (
                            int(self.r(f"nrow({names['summary']})")[0]),
                            int(self.r(f"ncol({names['summary']})")[0]),
                        ),
                        order="F",
                    ),
                    "summary_row_names": [
                        str(value) for value in self.r(f"rownames({names['summary']})")
                    ],
                    "summary_column_names": [
                        str(value) for value in self.r(f"colnames({names['summary']})")
                    ],
                    "print_lines": [str(value) for value in self.r(names["print_lines"])],
                    "print_error": (None if print_error is ro.NULL else str(print_error[0])),
                }
            )
            if plot_options is not None:
                payload["plot"] = self._clustrangefamily_plot_payload(
                    names["result"],
                    **dict(plot_options),
                )
            return payload
        finally:
            if had_global_hc:
                self.r(f"assign('hc', {names['hc_backup']}, envir=.GlobalEnv)")
            cleanup_names = [*names.values(), *replay_names.values()]
            self.r(
                "rm(list=intersect(ls(), c("
                + ",".join(f"'{name}'" for name in cleanup_names)
                + ")))"
            )


def create_r_parity_session() -> tuple[RParitySession | None, str | None]:
    if not _HAS_RPY2:
        return None, f"rpy2 not available: {_RPY2_IMPORT_ERROR}"

    for package in ("WeightedCluster", "TraMineR"):
        try:
            importr(package)
        except PackageNotInstalledError:
            return None, f"R package '{package}' not installed"
        except Exception as e:
            return None, f"Cannot load R package '{package}': {e}"

    return RParitySession(r=ro.r), None


@pytest.fixture(scope="module")
def r_parity_session() -> RParitySession:
    session, reason = create_r_parity_session()
    if session is None:
        fail_or_skip(reason or "R parity session unavailable")
    enforce_exact_oracle_versions(session.oracle_versions())
    return session

"""Deterministic, development-only Phase 3A differential conformance harness."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tests.r_parity_utils import (
    create_r_parity_session,
    enforce_exact_oracle_versions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "tests" / "fixtures" / "rarcat_conformance"
_DOMAINS = {"formula", "labels", "glm_ame", "crossed_reml"}
_SPLITS = {"development", "validation"}


def corpus_path(domain: str, split: str) -> Path:
    """Return the checked-in manifest path for one domain and split."""
    if domain not in _DOMAINS:
        raise ValueError(f"unknown conformance domain: {domain!r}")
    if split not in _SPLITS:
        raise ValueError(f"unknown conformance split: {split!r}")
    return CORPUS_ROOT / f"{domain}_{split}.json"


def _float_token(value: float) -> dict[str, str]:
    bits = np.asarray([float(value)], dtype=np.float64).view(np.uint64)[0]
    return {"float64_bits": f"0x{int(bits):016x}"}


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "kind": "dataframe",
            "index": [_canonical(item) for item in value.index.tolist()],
            "index_name": _canonical(value.index.name),
            "columns": [
                {
                    "name": _canonical(column),
                    "dtype": str(value[column].dtype),
                    "categorical": (
                        {
                            "categories": [
                                _canonical(item) for item in value[column].cat.categories.tolist()
                            ],
                            "ordered": bool(value[column].cat.ordered),
                        }
                        if isinstance(value[column].dtype, pd.CategoricalDtype)
                        else None
                    ),
                    "values": [_canonical(item) for item in value[column].tolist()],
                }
                for column in value.columns
            ],
        }
    if isinstance(value, pd.Series):
        return _canonical(value.to_frame())
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            data = [_canonical(item) for item in value.reshape(-1).tolist()]
        else:
            contiguous = np.ascontiguousarray(value)
            data = {
                "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                "byteorder": contiguous.dtype.byteorder,
            }
        return {
            "kind": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": data,
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(nested) for key, nested in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _float_token(float(value))
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def input_digest(value: Any) -> str:
    """Hash a materialized input without losing binary64 or dtype semantics."""
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def manifest_identity(manifest: Mapping[str, Any]) -> str:
    """Return the canonical identity, excluding the identity field itself."""
    content = {key: value for key, value in manifest.items() if key != "corpus_id"}
    payload = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_manifest(domain: str, split: str) -> dict[str, Any]:
    """Load and verify one immutable checked-in corpus manifest."""
    path = corpus_path(domain, split)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = manifest_identity(manifest)
    if manifest.get("corpus_id") != actual:
        raise ValueError(
            f"manifest identity mismatch for {path}: "
            f"recorded {manifest.get('corpus_id')!r}, actual {actual!r}"
        )
    return manifest


def materialize_formula_case(manifest: Mapping[str, Any], case_id: int) -> dict[str, Any]:
    """Materialize one structured-name case from a frozen formula manifest."""
    classes = list(manifest["level_classes"].items())
    modes = list(manifest["generator"]["modes"])
    auxiliary_cases = list(manifest["auxiliary_cases"].items())
    structured_count = len(classes) * len(modes)
    count = structured_count + len(auxiliary_cases)
    if case_id < 0 or case_id >= count:
        raise IndexError(f"formula case_id must be in [0, {count})")
    if case_id < structured_count:
        level_class, level = classes[case_id // len(modes)]
        mode = modes[case_id % len(modes)]
    else:
        mode, formula = auxiliary_cases[case_id - structured_count]
        level_class = "auxiliary"
        level = None
    seed = int(manifest["generator"]["seed"]) + case_id
    rng = np.random.default_rng(seed)
    n = int(manifest["generator"]["n_rows"])
    clusters = np.tile(np.asarray(["a", "b", "c"], dtype=object), n // 3)
    groups = np.tile(
        np.asarray(["reference", "middle", str(level)], dtype=object),
        n // 3,
    )
    permutation = rng.permutation(n)
    x = rng.normal(0.0, 0.7, n)
    frame = pd.DataFrame(
        {
            "cluster": clusters[permutation],
            "group": pd.Categorical(
                groups[permutation],
                categories=["reference", "middle", str(level)],
                ordered=False,
            ),
            "x": x,
            "flag": (np.arange(n) % 2 == 0),
            "raw_char": np.tile(["alpha", "beta", "gamma"], n // 3),
        },
        index=[f"{manifest['split']}-formula-{case_id}-{row + 1}" for row in range(n)],
    )
    formulas = {
        "main_intercept": "cluster ~ group + x",
        "main_no_intercept": "cluster ~ 0 + group + x",
        "interaction_intercept": "cluster ~ group + x + group:x",
        "interaction_no_intercept": "cluster ~ 0 + group + x + group:x",
    }
    if level_class != "auxiliary":
        formula = formulas[mode]
    return {
        "case_id": f"{manifest['split']}-formula-{case_id:03d}",
        "level_class": level_class,
        "level": None if level is None else str(level),
        "mode": mode,
        "formula": formula,
        "frame": frame,
        "categorical": {"group": ["reference", "middle", str(level)]},
        "logical": ("flag",),
        "manifest_id": manifest["corpus_id"],
    }


def _label_random_bits(seed: int, finite_count: int) -> tuple[np.ndarray, list[str]]:
    generator = np.random.Generator(np.random.PCG64(seed))
    accepted: list[np.ndarray] = []
    rejected: list[str] = []
    remaining = finite_count
    while remaining:
        raw = generator.bit_generator.random_raw(remaining)
        values = raw.view(np.float64)
        finite = np.isfinite(values)
        selected = raw[finite].copy()
        accepted.append(selected)
        remaining -= int(selected.size)
        rejected.extend(f"0x{int(bits):016x}" for bits in raw[~finite])
    return np.concatenate(accepted), rejected


def materialize_label_values(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the exact finite-bit sweep specified by a label manifest."""
    bits, rejected = _label_random_bits(
        int(manifest["generator"]["seed"]),
        int(manifest["generator"]["random_finite_count"]),
    )
    return {
        "random_finite_bits": bits,
        "random_finite_values": bits.view(np.float64),
        "rejected_bit_patterns": rejected,
        "special_values": np.asarray(manifest["special_values"], dtype=float),
        "manifest_id": manifest["corpus_id"],
    }


def _balanced_values(labels: list[str], counts: list[int]) -> np.ndarray:
    return np.concatenate(
        [np.repeat(label, count) for label, count in zip(labels, counts, strict=True)]
    )


def materialize_glm_case(manifest: Mapping[str, Any], case_id: int) -> dict[str, Any]:
    """Materialize one deterministic, full-rank GLM/AME corpus candidate."""
    count = int(manifest["accepted_case_count"])
    if case_id < 0 or case_id >= count:
        raise IndexError(f"GLM case_id must be in [0, {count})")
    accepted_spec = manifest["accepted_case_ids"]
    candidate_indices = accepted_spec.get("candidate_indices")
    candidate_index = (
        int(candidate_indices[case_id])
        if candidate_indices is not None
        else int(accepted_spec["start"]) + case_id
    )
    seed = int(manifest["generator"]["seed"]) + candidate_index
    rng = np.random.default_rng(seed)
    scale_bands = list(manifest["generator"]["scale_bands"].items())
    scale_band, scale = scale_bands[candidate_index % len(scale_bands)]
    balance_names = list(manifest["generator"]["balance_counts"])
    cluster_balance = balance_names[(candidate_index // len(scale_bands)) % len(balance_names)]
    factor_balance = balance_names[
        (candidate_index // (len(scale_bands) * len(balance_names))) % len(balance_names)
    ]
    balance_counts = manifest["generator"]["balance_counts"]
    clusters = _balanced_values(
        ["a", "b", "c"],
        [int(value) for value in balance_counts[cluster_balance]],
    )
    groups = _balanced_values(
        ["reference", "low", "high"],
        [int(value) for value in balance_counts[factor_balance]],
    )
    n = int(clusters.size)
    clusters = clusters[rng.permutation(n)]
    groups = groups[rng.permutation(n)]
    x = rng.normal(0.0, float(scale), n)
    binary = rng.random(n) < 0.47
    frame = pd.DataFrame(
        {
            "cluster": clusters,
            "x": x,
            "group": pd.Categorical(
                groups,
                categories=["reference", "low", "high"],
                ordered=False,
            ),
            "binary": binary,
        },
        index=[f"{manifest['split']}-glm-{candidate_index}-{row + 1}" for row in range(n)],
    )
    return {
        "case_id": f"{manifest['split']}-glm-{candidate_index:03d}",
        "candidate_index": candidate_index,
        "seed": seed,
        "scale_band": scale_band,
        "scale": float(scale),
        "cluster_balance": cluster_balance,
        "factor_balance": factor_balance,
        "formula": str(manifest["generator"]["formula"]),
        "frame": frame,
        "categorical": {"group": ["reference", "low", "high"]},
        "logical": ("binary",),
        "manifest_id": manifest["corpus_id"],
    }


def materialize_reml_case(manifest: Mapping[str, Any], case_id: int) -> dict[str, Any]:
    """Materialize one deterministic crossed-intercept REML corpus candidate."""
    count = int(manifest["accepted_case_count"])
    if case_id < 0 or case_id >= count:
        raise IndexError(f"REML case_id must be in [0, {count})")
    accepted_spec = manifest["accepted_case_ids"]
    candidate_indices = accepted_spec.get("candidate_indices")
    candidate_index = (
        int(candidate_indices[case_id])
        if candidate_indices is not None
        else int(accepted_spec["start"]) + case_id
    )
    seed = int(manifest["generator"]["seed"]) + candidate_index
    rng = np.random.default_rng(seed)
    regimes = list(manifest["generator"]["regimes"])
    regime = regimes[candidate_index % len(regimes)]
    balanced = candidate_index % 2 == 0
    n_id = 3 + (candidate_index % 4)
    n_boot = 3 + ((candidate_index // 4) % 4)
    settings = {
        "stable_interior": (0.12, 0.08, 0.04),
        "near_zero_component": (1e-8, 0.08, 0.04),
        "singular": (0.0, 0.0, 0.04),
        "nonconverged": (0.12, 0.08, 2e-4),
        "near_zero_residual": (0.12, 0.08, 1e-9),
    }
    id_scale, bootstrap_scale, noise_scale = settings[regime]
    id_effect = rng.normal(0.0, id_scale, n_id)
    bootstrap_effect = rng.normal(0.0, bootstrap_scale, n_boot)
    cells = [
        (id_position, bootstrap_position)
        for id_position in range(n_id)
        for bootstrap_position in range(n_boot)
        if balanced or (id_position + 2 * bootstrap_position + candidate_index) % 5 != 0
    ]
    weight_patterns = list(manifest["generator"]["weight_patterns"])
    weight_pattern = weight_patterns[candidate_index % len(weight_patterns)]
    if weight_pattern == "constant":
        standard_errors = np.repeat(0.05, len(cells))
    elif weight_pattern == "nonconstant":
        standard_errors = rng.uniform(0.035, 0.09, len(cells))
    elif weight_pattern == "moderate_ratio":
        standard_errors = np.geomspace(0.025, 0.15, len(cells))
        standard_errors = standard_errors[rng.permutation(len(cells))]
    else:
        standard_errors = np.geomspace(0.005, 0.2, len(cells))
        standard_errors = standard_errors[rng.permutation(len(cells))]
    rows = []
    for position, (id_position, bootstrap_position) in enumerate(cells):
        rows.append(
            {
                "id": id_position + 1,
                "bootstrap": bootstrap_position + 1,
                "ame": (
                    0.2
                    + id_effect[id_position]
                    + bootstrap_effect[bootstrap_position]
                    + rng.normal(0.0, noise_scale)
                ),
                "sterror": float(standard_errors[position]),
            }
        )
    frame = pd.DataFrame(
        rows,
        index=[f"{manifest['split']}-reml-{candidate_index}-{row + 1}" for row in range(len(rows))],
    )
    if candidate_index % 11 == 0:
        frame.iloc[candidate_index % len(frame), frame.columns.get_loc("ame")] = np.nan
    permuted = candidate_index % 3 == 1
    if permuted:
        frame = frame.iloc[rng.permutation(len(frame))].copy()
    return {
        "case_id": f"{manifest['split']}-reml-{candidate_index:03d}",
        "candidate_index": candidate_index,
        "seed": seed,
        "regime": regime,
        "balanced": balanced,
        "n_id": n_id,
        "n_bootstrap": n_boot,
        "weight_pattern": weight_pattern,
        "missing_ame_count": int(frame["ame"].isna().sum()),
        "permuted": permuted,
        "frame": frame,
        "manifest_id": manifest["corpus_id"],
    }


def _captured(callable_, *args, **kwargs) -> dict[str, Any]:
    try:
        return {"error": None, "value": callable_(*args, **kwargs)}
    except Exception as error:  # development evidence must retain exact failure
        return {"error": f"{type(error).__name__}: {error}", "value": None}


class ConformanceHarness:
    """Run one realized input through the pinned R oracle and private Python."""

    def __init__(self, oracle_session):
        self.oracle = oracle_session

    def oracle_snapshot(self) -> dict[str, Any]:
        versions = {
            package: str(self.oracle.r(f'as.character(packageVersion("{package}"))[1]')[0])
            for package in (
                "WeightedCluster",
                "TraMineR",
                "margins",
                "lme4",
            )
        }
        return {
            "versions": {
                "R": str(self.oracle.r("as.character(getRversion())")[0]),
                **versions,
            },
            "runtime": {
                "LC_COLLATE": str(self.oracle.r("Sys.getlocale('LC_COLLATE')")[0]),
                "OutDec": str(self.oracle.r("getOption('OutDec')")[0]),
                "digits": int(self.oracle.r("getOption('digits')")[0]),
                "scipen": int(self.oracle.r("getOption('scipen')")[0]),
            },
        }

    def r_global_names(self, prefix: str) -> list[str]:
        self.oracle.r.assign("wc_conformance_prefix", str(prefix))
        try:
            return [
                str(value)
                for value in self.oracle.r(
                    "sort(ls(envir=.GlobalEnv)[startsWith("
                    "ls(envir=.GlobalEnv), wc_conformance_prefix)])"
                )
            ]
        finally:
            self.oracle.r(
                "rm(list=intersect('wc_conformance_prefix', "
                "ls(envir=.GlobalEnv)), envir=.GlobalEnv)"
            )

    def run_naive(
        self,
        case: Mapping[str, Any],
        *,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        from tests._historical_rarcat_solver import _fit_naive_rarcat_models

        source = case["frame"]
        before = copy.deepcopy(source)
        input_id = input_digest(source)
        oracle_value = self.oracle.rarcat_naive_contract(
            formula=str(case["formula"]),
            data={column: source[column].tolist() for column in source.columns},
            categorical=case.get("categorical"),
            logical=case.get("logical", ()),
            row_names=list(source.index),
        )
        python_value = _captured(
            _fit_naive_rarcat_models,
            str(case["formula"]),
            source.copy(deep=True),
        )
        if not source.equals(before):
            raise AssertionError(f"conformance input mutated: {case['case_id']}")
        oracle_payload = {
            **oracle_value,
            "source_row_names": [str(value) for value in source.index],
        }
        return {
            "domain": "formula",
            "case_id": str(case["case_id"]),
            "corpus_id": str(manifest["corpus_id"]),
            "input_id": input_id,
            "oracle_input_id": input_digest(source),
            "python_input_id": input_digest(source),
            "oracle_snapshot": self.oracle_snapshot(),
            "oracle": oracle_payload,
            "python": python_value,
            "replay": {
                "domain": "formula",
                "split": manifest["split"],
                "case_id": case["case_id"],
            },
        }

    def run_labels(
        self,
        values: np.ndarray,
        *,
        corpus_id: str,
        case_id: str,
    ) -> dict[str, Any]:
        from tests._historical_rarcat_solver import _r_value_string

        values = np.asarray(values, dtype=np.float64)
        before = values.view(np.uint64).copy()
        oracle_value = self.oracle.rarcat_numeric_label_contract(values)
        python_rendered = [_r_value_string(value) for value in values]
        if not np.array_equal(values.view(np.uint64), before):
            raise AssertionError(f"conformance input mutated: {case_id}")
        mismatches = [
            {
                "position": position,
                "bits": f"0x{int(before[position]):016x}",
                "value": _float_token(float(values[position])),
                "oracle": oracle_value["rendered"][position],
                "python": python_rendered[position],
            }
            for position in range(len(values))
            if oracle_value["rendered"][position] != python_rendered[position]
        ]
        return {
            "domain": "labels",
            "case_id": str(case_id),
            "corpus_id": str(corpus_id),
            "input_id": input_digest(before),
            "oracle_input_id": input_digest(before),
            "python_input_id": input_digest(before),
            "oracle_snapshot": self.oracle_snapshot(),
            "oracle": oracle_value,
            "python": {"rendered": python_rendered},
            "mismatch_count": len(mismatches),
            "counterexamples": mismatches[:25],
            "replay": {
                "domain": "labels",
                "case_id": str(case_id),
                "bit_patterns": [item["bits"] for item in mismatches[:25]],
            },
        }

    def run_crossed(
        self,
        case: Mapping[str, Any],
        *,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        from tests._historical_rarcat_solver import _fit_weighted_crossed_pool

        source = case["frame"]
        before = copy.deepcopy(source)
        input_id = input_digest(source)
        oracle_value = _captured(
            self.oracle.rarcat_crossed_pool_contract,
            source.copy(deep=True),
        )
        python_value = _captured(
            _fit_weighted_crossed_pool,
            source.copy(deep=True),
        )
        if not source.equals(before):
            raise AssertionError(f"conformance input mutated: {case['case_id']}")
        return {
            "domain": "crossed_reml",
            "case_id": str(case["case_id"]),
            "corpus_id": str(manifest["corpus_id"]),
            "input_id": input_id,
            "oracle_input_id": input_digest(source),
            "python_input_id": input_digest(source),
            "oracle_snapshot": self.oracle_snapshot(),
            "oracle": oracle_value,
            "python": python_value,
            "replay": {
                "domain": "crossed_reml",
                "split": manifest["split"],
                "case_id": case["case_id"],
            },
        }


def characterize_case(
    harness: ConformanceHarness,
    domain: str,
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one frozen candidate using only live-R structural evidence."""
    source = case["frame"]
    before = copy.deepcopy(source)
    input_id = input_digest(source)
    problems: list[str] = []
    structural: dict[str, Any] = {}
    if domain in {"formula", "glm_ame"}:
        captured = _captured(
            harness.oracle.rarcat_naive_contract,
            formula=str(case["formula"]),
            data={column: source[column].tolist() for column in source.columns},
            categorical=case.get("categorical"),
            logical=case.get("logical", ()),
            row_names=list(source.index),
        )
        if captured["error"] is not None:
            problems.append(captured["error"])
        else:
            oracle = captured["value"]
            if oracle["error"] is not None:
                problems.append(f"model-frame error: {oracle['error']}")
            if oracle["warnings"]:
                problems.extend(f"model-frame warning: {item}" for item in oracle["warnings"])
            rarcat_error = oracle.get("rarcat", {}).get("error")
            if rarcat_error is not None:
                problems.append(f"rarcat error: {rarcat_error}")
            cluster_states = {}
            for cluster_name, cluster in oracle.get("clusters", {}).items():
                cluster_problems = []
                if cluster["error"] is not None:
                    cluster_problems.append(f"error: {cluster['error']}")
                cluster_problems.extend(f"warning: {item}" for item in cluster["warnings"])
                if cluster.get("converged") is not True:
                    cluster_problems.append("GLM did not converge")
                full_rank = cluster.get("rank") == len(cluster.get("coefficient_names", ()))
                if not full_rank:
                    cluster_problems.append("GLM design is rank deficient")
                if cluster_problems:
                    problems.extend(f"cluster {cluster_name}: {item}" for item in cluster_problems)
                cluster_states[str(cluster_name)] = {
                    "error": cluster["error"],
                    "warnings": list(cluster["warnings"]),
                    "converged": cluster.get("converged"),
                    "iterations": cluster.get("iterations"),
                    "rank": cluster.get("rank"),
                    "full_rank": full_rank,
                    "coefficient_names": cluster.get("coefficient_names"),
                    "ame_factor_names": cluster.get("ame_factor_names"),
                }
            structural = {
                "formula": str(case["formula"]),
                "rarcat_error": rarcat_error,
                "model_frame": (
                    None
                    if oracle["error"] is not None
                    else {
                        "row_names": oracle["model_frame"]["row_names"],
                        "omitted_row_names": oracle["model_frame"]["omitted_row_names"],
                        "term_labels": oracle["model_frame"]["term_labels"],
                        "intercept": oracle["model_frame"]["intercept"],
                        "design_column_names": oracle["model_frame"]["design_column_names"],
                    }
                ),
                "cluster_states": cluster_states,
            }
    elif domain == "crossed_reml":
        captured = _captured(
            harness.oracle.rarcat_crossed_pool_contract,
            source.copy(deep=True),
        )
        if captured["error"] is not None:
            problems.append(captured["error"])
            structural = {"error": captured["error"]}
        else:
            oracle = captured["value"]
            if oracle["error"] is not None:
                problems.append(f"lme4 error: {oracle['error']}")
            structural = {
                "error": oracle["error"],
                "warnings": list(oracle["warnings"]),
                "included_row_names": oracle.get("included_row_names"),
                "omitted_row_names": oracle.get("omitted_row_names"),
                "optimizer_code": oracle.get("optimizer_code"),
                "convergence_messages": oracle.get("convergence_messages"),
                "converged": oracle.get("converged"),
                "singular": oracle.get("singular"),
                "theta_length": (
                    None if oracle.get("theta") is None else int(len(oracle["theta"]))
                ),
                "observation_group_count": len(oracle.get("observation_random_effect_names", [])),
                "bootstrap_group_count": len(oracle.get("bootstrap_random_effect_names", [])),
            }
    else:
        raise ValueError(f"oracle characterization is unsupported for {domain!r}")
    if not source.equals(before):
        raise AssertionError(f"conformance input mutated: {case['case_id']}")
    return {
        "domain": domain,
        "split": manifest["split"],
        "case_id": str(case["case_id"]),
        "corpus_id": manifest["corpus_id"],
        "input_id": input_id,
        "accepted": not problems,
        "rejection_reasons": problems,
        "case_metadata": {
            key: value
            for key, value in case.items()
            if key not in {"frame", "categorical", "logical"}
        },
        "structural": structural,
        "oracle_snapshot": harness.oracle_snapshot(),
        "replay": {
            "domain": domain,
            "split": manifest["split"],
            "case_id": str(case["case_id"]),
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isnan(numeric):
            return {"nonfinite": "NaN"}
        if np.isposinf(numeric):
            return {"nonfinite": "Inf"}
        if np.isneginf(numeric):
            return {"nonfinite": "-Inf"}
        return numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    return str(value)


def format_counterexample(result: Mapping[str, Any]) -> str:
    """Format the smallest retained mismatch as a directly replayable JSON line."""
    counterexamples = list(result.get("counterexamples", ()))
    if not counterexamples:
        raise ValueError("result does not contain a minimized counterexample")
    return json.dumps(
        {
            "domain": result["domain"],
            "case_id": result["case_id"],
            "corpus_id": result["corpus_id"],
            "input_id": result["input_id"],
            "counterexample": _jsonable(counterexamples[0]),
            "replay": _jsonable(result["replay"]),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def write_machine_summary(path: Path, results: list[Mapping[str, Any]]) -> None:
    """Write auditable test output outside the committed corpus manifests."""
    path.write_text(
        json.dumps(
            {"schema_version": 1, "results": _jsonable(results)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_corpus(domain: str, split: str, summary_path: Path) -> dict[str, int]:
    manifest = load_manifest(domain, split)
    session, reason = create_r_parity_session()
    if session is None:
        raise RuntimeError(reason or "R parity session unavailable")
    enforce_exact_oracle_versions(session.oracle_versions())
    harness = ConformanceHarness(session)
    if domain == "labels":
        realized = materialize_label_values(manifest)
        result = harness.run_labels(
            realized["random_finite_values"],
            corpus_id=manifest["corpus_id"],
            case_id=f"{split}-binary64-sweep",
        )
        results = [result]
        counts = {
            "accepted": int(result["mismatch_count"] == 0),
            "rejected": int(result["mismatch_count"] != 0),
            "mismatches": int(result["mismatch_count"]),
        }
        if result["mismatch_count"]:
            print(format_counterexample(result))
    else:
        case_count = int(
            manifest["expected_case_count"]
            if domain == "formula"
            else manifest["accepted_case_count"]
        )
        materializers = {
            "formula": materialize_formula_case,
            "glm_ame": materialize_glm_case,
            "crossed_reml": materialize_reml_case,
        }
        results = [
            characterize_case(
                harness,
                domain,
                manifest,
                materializers[domain](manifest, case_id),
            )
            for case_id in range(case_count)
        ]
        counts = {
            "accepted": sum(bool(result["accepted"]) for result in results),
            "rejected": sum(not bool(result["accepted"]) for result in results),
            "mismatches": 0,
        }
    write_machine_summary(summary_path, results)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-corpus-id",
        nargs=2,
        metavar=("DOMAIN", "SPLIT"),
    )
    parser.add_argument("--run", nargs=2, metavar=("DOMAIN", "SPLIT"))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.print_corpus_id:
        domain, split = args.print_corpus_id
        print(load_manifest(domain, split)["corpus_id"])
        return 0
    if args.run:
        if args.summary is None:
            parser.error("--summary is required with --run")
        domain, split = args.run
        counts = _run_corpus(domain, split, args.summary)
        print(
            json.dumps(
                {
                    "domain": domain,
                    "split": split,
                    "summary": str(args.summary),
                    **counts,
                },
                sort_keys=True,
            )
        )
        return 0
    parser.error("choose --print-corpus-id or --run")


if __name__ == "__main__":
    raise SystemExit(main())

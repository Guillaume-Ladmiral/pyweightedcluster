from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ._parallel import resolve_n_jobs
from ._validation import validate_integer_control, validate_kvals


def _require_boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _finite_real(value: object, *, name: str, non_negative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if non_negative and numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric


def stability_metrics(
    best_clustering: np.ndarray,
    all_clusterings: List[np.ndarray],
    *,
    weights: Optional[np.ndarray] = None,
    fuzzy: bool = False,
) -> Dict[str, object]:
    """Compute R's weighted ARI/JC matrix against the selected solution."""

    def _scores(table: np.ndarray) -> tuple[float, float]:
        tab = np.asarray(table, dtype=float)
        if tab.shape == (1, 1):
            return 1.0, 1.0

        def choose2(values: np.ndarray) -> np.ndarray:
            return values * (values - 1.0) / 2.0

        a = float(np.sum(choose2(tab)))
        ab = float(np.sum(choose2(np.sum(tab, axis=1))))
        ac = float(np.sum(choose2(np.sum(tab, axis=0))))
        total = float(np.sum(choose2(np.asarray([np.sum(tab)]))))
        expected = (ab * ac / total) if total != 0 else float("nan")
        denom = ((ab + ac) / 2.0) - expected
        ari = (a - expected) / denom if denom != 0 else 1.0
        n11 = float(np.sum(tab**2))
        n01 = float(np.sum(np.sum(tab, axis=0) ** 2))
        n10 = float(np.sum(np.sum(tab, axis=1) ** 2))
        jc_denom = n01 + n10 - n11
        jc = n11 / jc_denom if jc_denom != 0 else 1.0
        return float(ari), float(jc)

    best = np.asarray(best_clustering)
    n_obs = int(best.shape[0])
    resolved_weights = (
        np.ones(n_obs, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    rows: list[tuple[float, float]] = []
    for clustering in all_clusterings:
        current = np.asarray(clustering)
        if fuzzy:
            left = current * resolved_weights[:, None]
            right = best * resolved_weights[:, None]
            table = left.T @ right
        else:
            current_labels = current.astype(int).reshape(-1)
            best_labels = best.astype(int).reshape(-1)
            current_levels = np.unique(current_labels)
            best_levels = np.unique(best_labels)
            table = np.zeros((current_levels.size, best_levels.size), dtype=float)
            for row, current_level in enumerate(current_levels):
                for column, best_level in enumerate(best_levels):
                    table[row, column] = np.sum(
                        resolved_weights[
                            (current_labels == current_level) & (best_labels == best_level)
                        ]
                    )
        rows.append(_scores(table))
    arimatrix = np.asarray(rows, dtype=float).reshape((-1, 2))
    return {
        "ARI>0.8": int(np.sum(arimatrix[:, 0] >= 0.8)),
        "JC>0.8": int(np.sum(arimatrix[:, 1] >= 0.8)),
        "ARImean": float(np.mean(arimatrix[:, 0])),
        "JCmean": float(np.mean(arimatrix[:, 1])),
        "arimatrix": arimatrix,
    }


def build_seqclararange_context(
    seqdata: Any,
    *,
    R: int,
    sample_size: Optional[int],
    kvals: Sequence[int],
    seqdist_args: Optional[Dict],
    method: str,
    criteria_norm: Sequence[str],
    stability: bool,
    dnoise: Optional[float],
    m: float,
    dlambda: Optional[float],
    eta: Optional[np.ndarray],
    pam_n_init: int,
    parallel: bool,
    n_jobs: Optional[int],
    progressbar: bool,
    keep_diss: bool,
    max_dist: Optional[float],
    random_state: Optional[int],
    export_membership: bool,
    replay_samples: Optional[np.ndarray],
    replay_initial_memberships: Optional[Mapping[int, Sequence[object]]],
    clara_base_sample_size: int,
    clara_k_multiplier: int,
    aggregate_cases_fn: Callable[..., Dict[str, object]],
    validate_seqdist_args_fn: Callable[..., Dict[str, object]],
    sequences_distance_fn: Callable[..., np.ndarray],
    generate_random_seed_fn: Callable[..., int],
) -> Dict[str, object]:
    """Prepare immutable inputs shared by single-criterion and family runs."""
    # Unguarded, as at the four sibling import sites: traminex is a hard runtime
    # dependency and the removed handler's `STSList = ()` sentinel would have
    # made this isinstance check silently False (finding F-30).
    from traminex import STSList

    sequence_object = seqdata if isinstance(seqdata, STSList) else None
    X = np.asarray(sequence_object.data if sequence_object is not None else seqdata).copy()
    if X.ndim != 2:
        raise ValueError("seqdata must be (N, T)")
    n_obs = int(X.shape[0])
    if n_obs == 0:
        raise ValueError("seqdata must contain at least one sequence")
    R_int = validate_integer_control(R, name="R")
    if R_int < 1:
        raise ValueError("R must be a positive integer")
    kvals_list = validate_kvals(kvals)
    stability_value = _require_boolean(stability, name="stability")
    parallel_value = _require_boolean(parallel, name="parallel")
    progressbar_value = _require_boolean(progressbar, name="progressbar")
    keep_diss_value = _require_boolean(keep_diss, name="keep_diss")
    export_membership_value = _require_boolean(export_membership, name="export_membership")
    pam_n_init_int = validate_integer_control(pam_n_init, name="pam_n_init")
    if pam_n_init_int < 1:
        raise ValueError("pam_n_init must be at least one")
    if n_jobs is not None:
        resolve_n_jobs(n_jobs)
    replay_array = None
    if replay_samples is not None:
        replay_array = np.asarray(replay_samples)
        if replay_array.ndim != 2 or replay_array.shape[0] != R_int:
            raise ValueError("replay_samples must have shape (R, sample_size)")
        if not np.issubdtype(replay_array.dtype, np.integer):
            raise ValueError("replay_samples must contain zero-based integer indices")
        replay_array = replay_array.astype(int, copy=True)
        if sample_size is None:
            sample_size = int(replay_array.shape[1])
    if sample_size is None:
        ss = clara_base_sample_size + clara_k_multiplier * max(kvals_list)
        sample_size = int(ss)
    sample_size_int = validate_integer_control(sample_size, name="sample_size")
    if sample_size_int < 1:
        raise ValueError("sample_size must be a positive integer")
    if max(kvals_list) > sample_size_int:
        raise ValueError("More clusters than the size of the sample requested")
    method_l = str(method).lower()
    max_dist_value = max_dist
    if method_l == "representativeness":
        if max_dist is None:
            raise ValueError(
                "You need to set max_dist to the theoretical maximum distance value "
                "when using representativeness method"
            )
        max_dist_value = _finite_real(max_dist, name="max_dist")
        if max_dist_value <= 0.0:
            raise ValueError("max_dist must be finite and positive")
    dnoise_value = dnoise
    dlambda_value = dlambda
    if method_l in ("noise", "hncdd") and dnoise is not None:
        dnoise_value = _finite_real(dnoise, name="dnoise", non_negative=True)
    if method_l in ("noise", "hncdd") and dlambda is not None:
        dlambda_value = _finite_real(dlambda, name="dlambda", non_negative=True)
    if method_l == "noise" and dnoise is None and dlambda is None:
        raise ValueError("Must provide a value for dnoise or dlambda")
    if method_l == "hncdd" and dnoise is None:
        raise ValueError("HNCdd requires dnoise")
    m_value = _finite_real(m, name="m")
    if method_l in ("fuzzy", "noise", "hncdd", "pcmdd") and m_value <= 1.0:
        raise ValueError("m must be a finite number greater than 1")
    eta_value = eta
    if method_l == "pcmdd" and eta is not None:
        try:
            eta_value = np.asarray(eta, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("eta must be a finite non-negative vector") from exc
        if eta_value.ndim != 1 or not np.all(np.isfinite(eta_value)) or np.any(eta_value < 0.0):
            raise ValueError("eta must be a finite non-negative vector")
        if len(kvals_list) != 1 or eta_value.shape != (kvals_list[0],):
            raise ValueError("eta must have length k for the requested PCMdd solution")

    ac = aggregate_cases_fn(sequence_object if sequence_object is not None else X)
    Xagg = X[ac["aggIndex"], :]
    probs = np.asarray(ac["aggWeights"], dtype=float) / float(n_obs)
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or float(np.sum(probs)) <= 0:
        raise ValueError("sequence weights must define finite non-negative sampling probabilities")
    if replay_array is not None:
        if replay_array.shape[1] != sample_size_int:
            raise ValueError("replay_samples width must equal sample_size")
        if np.any(replay_array < 0) or np.any(replay_array >= Xagg.shape[0]):
            raise ValueError("replay_samples contains an out-of-range zero-based index")

    replay_memberships: dict[int, list[object]] = {}
    if replay_initial_memberships is not None:
        if not isinstance(replay_initial_memberships, Mapping):
            raise TypeError("replay_initial_memberships must be a mapping keyed by integer k")
        for raw_k, plans in replay_initial_memberships.items():
            try:
                k = validate_integer_control(raw_k, name="replay_initial_memberships key")
            except ValueError as exc:
                raise ValueError(
                    "replay_initial_memberships keys must be lossless integers"
                ) from exc
            if k not in kvals_list:
                raise ValueError("replay_initial_memberships contains a k outside kvals")
            try:
                iteration_plans = list(plans)
            except TypeError as exc:
                raise TypeError(
                    "each replay_initial_memberships entry must be an iterable of R plans"
                ) from exc
            if len(iteration_plans) != R_int:
                raise ValueError("each replay_initial_memberships entry must contain R plans")
            replay_memberships[k] = iteration_plans
    dist_cfg = validate_seqdist_args_fn(seqdist_args, default_method="LCS")
    norm_value = str(dist_cfg["norm"])
    if norm_value not in {"none", "length", "maxlength"}:
        raise ValueError("seqdist_args['norm'] must be one of none, length, or maxlength")
    normalize_distances = norm_value == "length"
    distance_scale = (
        1.0 / float(X.shape[1]) if norm_value == "maxlength" and X.shape[1] > 0 else 1.0
    )
    metric = str(dist_cfg["metric"])
    indel = float(dist_cfg["indel"])
    substitution_costs = dist_cfg["substitution_costs"]
    timecost = dist_cfg["timecost"]
    D_full = distance_scale * sequences_distance_fn(
        Xagg,
        metric=metric,
        normalize=normalize_distances,
        indel=indel,
        substitution_costs=substitution_costs,
        timecost=timecost,
    )
    return {
        "X": X,
        "R": R_int,
        "sample_size": sample_size_int,
        "kvals": kvals_list,
        "criteria_norm": list(criteria_norm),
        "stability": stability_value,
        "method": str(method),
        "method_l": method_l,
        "dnoise": dnoise_value,
        "m": m_value,
        "dlambda": dlambda_value,
        "eta": eta_value,
        "pam_n_init": pam_n_init_int,
        "parallel": parallel_value,
        "n_jobs": n_jobs,
        "progressbar": progressbar_value,
        "keep_diss": keep_diss_value,
        "max_dist": max_dist_value,
        "export_membership": export_membership_value,
        "replay_samples": replay_array,
        "replay_initial_memberships": replay_memberships,
        "base_seed": generate_random_seed_fn(random_state, 0),
        "ac": ac,
        "Xagg": Xagg,
        "probs": probs,
        "metric": metric,
        "indel": indel,
        "substitution_costs": substitution_costs,
        "timecost": timecost,
        "normalize_distances": normalize_distances,
        "distance_scale": distance_scale,
        "D_full": D_full,
        "n_obs": n_obs,
        "agg_weights": np.asarray(ac["aggWeights"], dtype=float),
    }


def run_seqclara_iterations_for_k(
    *,
    k: int,
    context: Dict[str, object],
    reaggregate_sampled_cases_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    cluster_crisp_fn: Callable[
        ..., tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]
    ],
    cluster_fuzzy_fn: Callable[
        ..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float]]
    ],
    compute_clara_criteria_fn: Callable[..., Dict[str, float]],
    sequences_distance_fn: Callable[..., np.ndarray],
    run_task_batch_fn: Callable[..., list[Dict[str, object]]],
    generate_random_seed_fn: Callable[..., int],
    fuzzy_methods: Sequence[str],
) -> List[Dict[str, object]]:
    """Generate and evaluate all CLARA iterations for one k."""
    Xagg = np.asarray(context["Xagg"])
    idx_agg_full = np.arange(Xagg.shape[0], dtype=int)
    probs = np.asarray(context["probs"], dtype=float)
    analysis_weights = np.asarray(context["agg_weights"], dtype=float)
    metric = str(context["metric"])
    indel = float(context["indel"])
    substitution_costs = context["substitution_costs"]
    timecost = context["timecost"]
    normalize_distances = bool(context["normalize_distances"])
    distance_scale = float(context["distance_scale"])
    D_full = np.asarray(context["D_full"], dtype=float)
    method_l = str(context["method_l"])
    pam_n_init = int(context["pam_n_init"])
    sample_size = int(context["sample_size"])
    base_seed = int(context["base_seed"])
    m = float(context["m"])
    dnoise = context["dnoise"]
    dlambda = context["dlambda"]
    eta = context["eta"]
    parallel = bool(context["parallel"])
    n_jobs = context["n_jobs"]
    progressbar = bool(context["progressbar"])
    n_iter = int(context["R"])
    search_criteria = tuple(
        dict.fromkeys(("distance", "db", "xb", "pbm", "ams", *context["criteria_norm"]))
    )
    replay_samples = context.get("replay_samples")
    replay_initial_memberships = dict(context.get("replay_initial_memberships", {}))

    def _one_iter(iter_idx: int) -> Dict[str, object]:
        seed_run = generate_random_seed_fn(base_seed, 4, iter_idx)
        rng_local = np.random.default_rng(seed_run)
        sampled_replay = (
            None
            if replay_samples is None
            else np.asarray(replay_samples, dtype=int)[int(iter_idx)].copy()
        )
        Xs, idx_global, sample_weights = reaggregate_sampled_cases_fn(
            rng_local=rng_local,
            Xagg=Xagg,
            agg_index=idx_agg_full,
            probs=probs,
            sample_size=sample_size,
            min_unique=int(k),
            sampled_idx=sampled_replay,
        )
        Ds = distance_scale * sequences_distance_fn(
            Xs,
            metric=metric,
            normalize=normalize_distances,
            indel=indel,
            substitution_costs=substitution_costs,
            timecost=timecost,
        )
        memb_full = None
        memb_full_all = None
        dnoise_eff = None
        iteration_init = None
        if int(k) in replay_initial_memberships:
            iteration_init = replay_initial_memberships[int(k)][int(iter_idx)]
        if method_l in ("crisp", "representativeness"):
            crisp_init = None
            if iteration_init is not None:
                if isinstance(iteration_init, Mapping):
                    crisp_init = iteration_init.get("ward", iteration_init.get("crisp"))
                else:
                    crisp_init = iteration_init
            if crisp_init is not None:
                crisp_array = np.asarray(crisp_init)
                if crisp_array.ndim == 2:
                    crisp_init = np.argmax(crisp_array, axis=1) + 1
            labels_full, memb_full, medoids_global_idx, D_full_to_m = cluster_crisp_fn(
                Ds,
                int(k),
                sample_weights,
                pam_n_init,
                seed_run,
                Xagg,
                Xs,
                idx_global,
                metric,
                indel,
                substitution_costs,
                timecost,
                (None if crisp_init is None else np.asarray(crisp_init).reshape(-1)),
                normalize_distances,
                distance_scale,
            )
        elif method_l in fuzzy_methods:
            fuzzy_inits = None
            if iteration_init is not None:
                if isinstance(iteration_init, Mapping):
                    fuzzy_inits = [
                        np.asarray(iteration_init[name], dtype=float)
                        for name in ("ward", "fanny")
                        if name in iteration_init
                    ]
                else:
                    candidate = np.asarray(iteration_init, dtype=float)
                    fuzzy_inits = [candidate]
            (
                labels_full,
                memb_full_all,
                medoids_global_idx,
                D_full_to_m,
                dnoise_eff,
            ) = cluster_fuzzy_fn(
                Ds,
                int(k),
                sample_weights,
                method_l,
                pam_n_init,
                seed_run,
                Xagg,
                Xs,
                idx_global,
                metric,
                indel,
                substitution_costs,
                timecost,
                m,
                dnoise,
                dlambda,
                eta,
                fuzzy_inits,
                normalize_distances,
                distance_scale,
            )
            memb_full = (
                memb_full_all[:, : int(k)] if method_l in ("noise", "hncdd") else memb_full_all
            )
        else:
            raise NotImplementedError(f"Unsupported method: {context['method']}")

        crits = compute_clara_criteria_fn(
            method_l=method_l,
            D_full=D_full,
            D_to_m=D_full_to_m,
            labels_full=labels_full,
            medoids_global_idx=medoids_global_idx,
            memb_full_all=(memb_full_all if method_l in fuzzy_methods else None),
            fuzzifier_m=m,
            dnoise_value=dnoise_eff,
            case_weights=analysis_weights,
            requested_criteria=search_criteria,
        )
        return {
            "labels_full": np.asarray(labels_full, dtype=int),
            "medoids_global_idx": np.asarray(medoids_global_idx, dtype=int),
            "memb_full": None if memb_full is None else np.asarray(memb_full, dtype=float),
            "memb_full_all": (
                None if memb_full_all is None else np.asarray(memb_full_all, dtype=float)
            ),
            "dnoise_value": dnoise_eff,
            "criteria": crits,
            "sample_draw": (
                None if sampled_replay is None else np.asarray(sampled_replay, dtype=int)
            ),
            "sample_unique_indices": np.asarray(idx_global, dtype=int),
            "sample_weights": np.asarray(sample_weights, dtype=float),
            "initial_membership": iteration_init,
            "distances_to_medoids": np.asarray(D_full_to_m, dtype=float),
        }

    if n_iter <= 0:
        return []
    return run_task_batch_fn(
        _one_iter,
        list(range(n_iter)),
        parallel=parallel,
        n_jobs=n_jobs,
        progressbar=progressbar,
        desc=f"K={k}",
        executor_kind="thread",
    )


def select_best_run_index(
    runs: Sequence[Dict[str, object]],
    primary: str,
    objective_value_fn: Callable[[str, Dict[str, float]], float],
) -> Optional[int]:
    """Return the index of the best run for a given criterion."""
    if not runs:
        return None
    objectives = [
        objective_value_fn(str(primary), dict(run["criteria"]))  # type: ignore[arg-type]
        for run in runs
    ]
    return int(np.argmin(np.asarray(objectives, dtype=float)))


def run_seqclara_cache(
    context: Dict[str, object],
    *,
    run_seqclara_iterations_for_k_fn: Callable[..., List[Dict[str, object]]],
) -> Dict[int, List[Dict[str, object]]]:
    """Run CLARA once and keep every evaluated iteration for later selection."""
    runs_by_k: Dict[int, List[Dict[str, object]]] = {}
    for k in context["kvals"]:
        k_int = int(k)
        runs = run_seqclara_iterations_for_k_fn(
            k=k_int,
            context=context,
        )
        runs_by_k[k_int] = runs
    return runs_by_k


def assemble_seqclararange_result_from_cache(
    context: Dict[str, object],
    runs_by_k: Dict[int, List[Dict[str, object]]],
    *,
    primary: str,
    select_best_run_index_fn: Callable[..., Optional[int]],
    stability_metrics_fn: Callable[[np.ndarray, List[np.ndarray]], Dict[str, float]],
    compute_clara_criteria_fn: Callable[..., Dict[str, float]],
    sequences_distance_between_fn: Callable[..., np.ndarray],
    wc_cluster_quality_fn: Callable[..., Dict[str, object]],
    wc_fuzzy_quality_fn: Callable[..., Dict[str, object]],
    rows_to_stats_table_fn: Callable[..., Dict[str, List[float]]],
    seqclara_stats_core: Sequence[str],
    seqclara_medoid_criteria: Sequence[str],
    seqclara_fuzzy_methods: Sequence[str],
) -> Dict[str, object]:
    """Select the best run per k from a cached CLARA iteration matrix."""
    Xagg = np.asarray(context["Xagg"])
    ac = context["ac"]
    agg_index = np.asarray(ac["aggIndex"], dtype=int)
    disagg_index = np.asarray(ac["disaggIndex"], dtype=int)
    agg_weights = np.asarray(context["agg_weights"], dtype=float)
    kvals = [int(k) for k in context["kvals"]]
    D_full = np.asarray(context["D_full"], dtype=float)
    metric = str(context["metric"])
    indel = float(context["indel"])
    substitution_costs = context["substitution_costs"]
    timecost = context["timecost"]
    normalize_distances = bool(context["normalize_distances"])
    distance_scale = float(context["distance_scale"])
    keep_diss = bool(context["keep_diss"])
    max_dist = context["max_dist"]
    method_l_global = str(context["method_l"])
    compute_stability = bool(context["stability"])
    export_membership = bool(context["export_membership"])
    m = float(context["m"])
    sample_size = int(context["sample_size"])
    R = int(context["R"])

    out_clustering: Dict[str, np.ndarray] = {}
    stats_rows: List[Dict[str, float]] = []
    quality_rows: List[Dict[str, float]] = []
    clara_info: Dict[int, Dict[str, object]] = {}

    for k_position, k in enumerate(kvals, start=1):
        runs = runs_by_k.get(int(k), [])
        best_idx = select_best_run_index_fn(runs, primary)
        if best_idx is None:
            continue
        best_run = runs[best_idx]
        best_labels = np.asarray(best_run["labels_full"], dtype=int)
        best_medoids = np.asarray(best_run["medoids_global_idx"], dtype=int)
        best_memb_full = best_run["memb_full"]
        best_memb_full_all = best_run["memb_full_all"]
        best_dnoise_value = best_run.get("dnoise_value")
        best_criteria_vals = dict(best_run["criteria"])  # type: ignore[arg-type]
        best_iter = int(best_idx + 1)
        run_primary_vals = [
            float(dict(run["criteria"]).get(primary, np.nan))  # type: ignore[arg-type]
            for run in runs
        ]

        labels_disaggregated = best_labels[disagg_index]
        fuzzy_stability = (
            method_l_global in seqclara_fuzzy_methods and best_memb_full_all is not None
        )
        stability_best = (
            np.asarray(best_memb_full_all, dtype=float) if fuzzy_stability else best_labels
        )
        stability_all = (
            [np.asarray(run["memb_full_all"], dtype=float) for run in runs]
            if fuzzy_stability
            else [np.asarray(run["labels_full"], dtype=int) for run in runs]
        )
        stab = (
            stability_metrics_fn(
                stability_best,
                stability_all,
                weights=agg_weights,
                fuzzy=fuzzy_stability,
            )
            if compute_stability
            else {}
        )

        cq = wc_cluster_quality_fn(D_full, best_labels + 1, weights=agg_weights)
        report_D_to_m = distance_scale * sequences_distance_between_fn(
            Xagg,
            Xagg[best_medoids],
            metric=metric,
            normalize=normalize_distances,
            indel=indel,
            substitution_costs=substitution_costs,
            timecost=timecost,
        )
        best_criteria_vals.update(
            compute_clara_criteria_fn(
                method_l=method_l_global,
                D_full=D_full,
                D_to_m=report_D_to_m,
                labels_full=best_labels,
                medoids_global_idx=best_medoids,
                memb_full_all=(
                    np.asarray(best_memb_full_all, dtype=float)
                    if best_memb_full_all is not None and method_l_global in seqclara_fuzzy_methods
                    else None
                ),
                fuzzifier_m=m,
                dnoise_value=(float(best_dnoise_value) if best_dnoise_value is not None else None),
                case_weights=agg_weights,
                requested_criteria=seqclara_medoid_criteria,
            )
        )
        s_core = {
            "Avg dist": float(best_criteria_vals.get("distance", np.nan)),
            "PBM": float(best_criteria_vals.get("pbm", np.nan)),
            "DB": float(best_criteria_vals.get("db", np.nan)),
            "XB": float(best_criteria_vals.get("xb", np.nan)),
            "AMS": float(best_criteria_vals.get("ams", np.nan)),
            "ARI>0.8": float(stab.get("ARI>0.8", np.nan)),
            "JC>0.8": float(stab.get("JC>0.8", np.nan)),
            "Best iter": float(best_iter),
        }
        stats_rows.append(s_core)
        qrow = dict(cq["stats"])
        if method_l_global in ("fuzzy", "pcmdd") and best_memb_full_all is not None:
            fq = wc_fuzzy_quality_fn(
                D_full,
                np.asarray(best_memb_full_all, dtype=float),
                medoids=best_medoids,
                m=m,
                weights=agg_weights,
            )
            qrow.update(fq["stats"])
        qrow["AMS"] = float(best_criteria_vals.get("ams", np.nan))
        qrow["k"] = float(k)
        if compute_stability:
            qrow.update({key: value for key, value in stab.items() if key != "arimatrix"})
        quality_rows.append(qrow)

        medoids_original = agg_index[best_medoids]
        Dm_aggregated = None
        if best_medoids.size > 0 and (
            keep_diss
            or (
                method_l_global == "representativeness"
                and max_dist is not None
                and np.isfinite(max_dist)
                and float(max_dist) > 0
            )
        ):
            Dm_aggregated = distance_scale * sequences_distance_between_fn(
                Xagg,
                Xagg[best_medoids],
                metric=metric,
                normalize=normalize_distances,
                indel=indel,
                substitution_costs=substitution_costs,
                timecost=timecost,
            )

        rep_matrix = None
        if (
            method_l_global == "representativeness"
            and max_dist is not None
            and np.isfinite(max_dist)
            and float(max_dist) > 0
            and Dm_aggregated is not None
        ):
            rep_matrix = 1.0 - (Dm_aggregated[disagg_index] / float(max_dist))

        if primary in ("ams", "pbm"):
            evol_vals = np.maximum.accumulate(np.asarray(run_primary_vals, dtype=float))
        else:
            evol_vals = np.minimum.accumulate(np.asarray(run_primary_vals, dtype=float))

        if method_l_global in seqclara_fuzzy_methods and best_memb_full_all is not None:
            clara_clustering_array = np.asarray(best_memb_full_all, dtype=float)[disagg_index]
            clara_clustering = clara_clustering_array.tolist()
        else:
            clara_clustering_array = (labels_disaggregated + 1).astype(int)
            clara_clustering = clara_clustering_array.tolist()

        if method_l_global == "crisp":
            out_clustering[f"cluster{k}"] = clara_clustering_array
        elif method_l_global == "representativeness":
            out_clustering[f"cluster{k}"] = np.asarray(rep_matrix, dtype=float)
        else:
            out_clustering[f"cluster{k}"] = clara_clustering_array

        from .schemas import SeqClaraResult

        clara_info[int(k)] = SeqClaraResult(
            {
                "medoids_sub_idx": best_medoids.tolist(),
                "medoids": medoids_original.tolist(),
                "clustering": clara_clustering,
                "evol.diss": evol_vals.tolist(),
                "iter.objective": [float(v) for v in run_primary_vals],
                "objective": float(best_criteria_vals.get(primary, np.nan)),
                "iteration": best_iter,
                "criteria": primary,
                "method": method_l_global,
                "avg_dist": float(best_criteria_vals.get("distance", np.nan)),
                "pbm": float(best_criteria_vals.get("pbm", np.nan)),
                "db": float(best_criteria_vals.get("db", np.nan)),
                "xb": float(best_criteria_vals.get("xb", np.nan)),
                "ams": float(best_criteria_vals.get("ams", np.nan)),
                "ari08": float(stab.get("ARI>0.8", np.nan)),
                "jc08": float(stab.get("JC>0.8", np.nan)),
                "arimatrix": (
                    np.asarray(stab["arimatrix"], dtype=float)
                    if compute_stability
                    else np.asarray(np.nan)
                ),
                "k": int(k_position),
                "ngroup": int(k),
                "representativeness": rep_matrix,
                "subsample_size": sample_size,
                "R": R,
                "stability": dict(stab) if compute_stability else None,
                "iterations": [
                    {
                        "criteria": dict(run["criteria"]),
                        "sample_draw": run.get("sample_draw"),
                        "sample_unique_indices": run.get("sample_unique_indices"),
                        "sample_weights": run.get("sample_weights"),
                        "initial_membership": run.get("initial_membership"),
                        "medoids": np.asarray(run["medoids_global_idx"], dtype=int),
                        "clustering": np.asarray(run["labels_full"], dtype=int),
                    }
                    for run in runs
                ],
            }
        )
        if keep_diss and Dm_aggregated is not None:
            clara_info[int(k)]["diss"] = Dm_aggregated[disagg_index].tolist()
        if export_membership and method_l_global in seqclara_fuzzy_methods:
            if best_memb_full_all is not None:
                clara_info[int(k)]["membership"] = np.asarray(best_memb_full_all).tolist()
            elif best_memb_full is not None:
                clara_info[int(k)]["membership"] = np.asarray(best_memb_full).tolist()

    from .schemas import LabeledStatsTable, SeqClaraRangeResult

    stats_raw = rows_to_stats_table_fn(stats_rows, keys=seqclara_stats_core)
    stats = LabeledStatsTable(
        stats_raw,
        row_names=[f"cluster{k}" for k in kvals],
    )
    qkeys = sorted(quality_rows[0].keys()) if quality_rows else []
    quality_raw = rows_to_stats_table_fn(quality_rows, keys=qkeys)
    quality_stats = LabeledStatsTable(
        quality_raw,
        row_names=[f"cluster{k}" for k in kvals],
    )
    return SeqClaraRangeResult(
        {
            "kvals": kvals,
            "clara": clara_info,
            "clustering": out_clustering,
            "stats": stats,
            "quality_stats": quality_stats,
        }
    )


def assemble_seqclararange_family_from_cache(
    context: Dict[str, object],
    runs_by_k: Dict[int, List[Dict[str, object]]],
    *,
    assemble_seqclararange_result_from_cache_fn: Callable[..., Dict[str, object]],
    rows_to_table_fn: Callable[..., Dict[str, List[object]]],
) -> Dict[str, object]:
    """Build the family view by re-selecting across one cached iteration matrix."""
    crits = list(context["criteria_norm"])
    ret: Dict[str, object] = {
        "param": {
            "criteria": crits,
            "pam.combine": False,
            "all.criterias": crits,
            "kvals": list(context["kvals"]),
            "method": str(context["method"]),
            "stability": bool(context["stability"]),
        }
    }
    allstats_rows: list[dict[str, object]] = []
    for c in crits:
        out_c = assemble_seqclararange_result_from_cache_fn(context, runs_by_k, primary=str(c))
        ret[str(c)] = out_c
        kvals_list = out_c.get("kvals", [])
        stats_c = out_c.get("stats", {})
        keys = list(stats_c.keys())
        for i, k in enumerate(kvals_list):
            row = {key: stats_c.get(key, [None])[i] for key in keys}
            row["ngroup"] = int(k)
            row["criteria"] = str(c)
            allstats_rows.append(row)
    all_keys = list(allstats_rows[0]) if allstats_rows else []
    ret["allstats"] = rows_to_table_fn(allstats_rows, keys=all_keys)
    from .schemas import SeqClaraRangeFamilyResult

    return SeqClaraRangeFamilyResult(ret)

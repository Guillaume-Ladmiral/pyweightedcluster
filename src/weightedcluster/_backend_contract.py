"""Declarative operation-level backend and ownership contract.

This module intentionally performs no imports or capability probing.  It is the
static source of truth for which implementations exist, which ones public APIs
actually select, and when selection occurs.  Runtime availability is added by
``get_backend_info`` rather than inferred here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

CAPABILITY_REGISTRY: Final[dict[str, dict[str, object]]] = {
    "weighted_medoids": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "compiled",
                "symbol": "weightedcluster._wc_core.weighted_medoids",
            },
            {
                "backend": "pure_python",
                "symbol": "weightedcluster._fallback.weighted_medoids",
            },
        ),
        "public_dispatch": ("compiled", "pure_python"),
        "selection": "conditional",
        "public_consumers": (
            # 2026-09-02 audit, finding WC-090204: these consumer selections
            # were declared "import_time", which described only the binding of
            # the module-level kernel name, not the selection a call performs.
            {
                "symbol": "weightedcluster.kmedoids.wcKMedoids",
                "selection": "conditional",
                "condition": (
                    "method=PAMonce with non-negative ordinary-range weights "
                    "uses the import-time compiled-then-pure binding; "
                    "range-risk inputs select the pure kernel at call time; "
                    "KMedoids/PAM and negative-weight PAMonce run "
                    "module-local searches outside this capability"
                ),
            },
            {
                "symbol": "weightedcluster.kmedoids.wcKMedRange",
                "selection": "conditional",
                "condition": (
                    "as wcKMedoids (method=PAMonce by default), which it " "calls per requested k"
                ),
                "via": "weightedcluster.kmedoids.wcKMedoids",
            },
            {
                "symbol": "weightedcluster.fuzzy.wcFuzzyRange",
                "selection": "call_time",
            },
            {
                # CLARA's crisp branch reaches this kernel through wcKMedoids.
                # It used to call the kernel directly with random restarts;
                # finding F-05 replaced that with R's deterministic
                # Ward-seeded PAM, which moved the selection.
                "symbol": "weightedcluster.seqclararange._cluster_crisp",
                "selection": "conditional",
                "condition": "as wcKMedoids (CLARA passes non-negative weights)",
                "via": "weightedcluster.kmedoids.wcKMedoids",
            },
        ),
        # CLARA's fuzzy branch no longer selects this capability at all: since
        # finding F-05 it starts from R's Ward cut and picks per-cluster medoids
        # with weightedcluster._utils.find_medoid, then hands over to fcmdd.
    },
    "cluster_quality": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "compiled",
                "symbol": "weightedcluster._wc_core.cluster_quality",
            },
            {
                "backend": "numba",
                "symbol": "weightedcluster._numba.cluster_quality_numba",
            },
            {
                # The publicly selected pure implementation (quality.py's
                # _pure_public_quality_result). The validated sibling kernel
                # _fallback.cluster_quality is reached publicly only as the
                # Numba wrapper's range-risk escape target and as the compiled
                # kernel's delegation target; naming the selected function here
                # was finding WC-090204 of the 2026-09-02 audit.
                "backend": "pure_python",
                "symbol": "weightedcluster._fallback.cluster_quality_r_permissive",
            },
        ),
        "public_dispatch": ("compiled", "numba", "pure_python"),
        "selection": "call_time",
        "public_consumers": (
            {
                "symbol": "weightedcluster.quality.wcClusterQuality",
                "selection": "call_time",
            },
        ),
    },
    "observation_silhouette": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "python",
                "symbol": "weightedcluster.quality.wcSilhouetteObs",
            },
        ),
        "public_dispatch": ("python",),
        "selection": "python_only",
        "public_consumers": (
            {
                "symbol": "weightedcluster.quality.wcSilhouetteObs",
                "selection": "python_only",
            },
        ),
    },
    "seqnull_stateindep": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "numba",
                "symbol": "weightedcluster._numba.seqnull_random_from_probs_numba",
            },
            {
                "backend": "python",
                "symbol": "weightedcluster.seqnull._seqnull_pos",
            },
        ),
        "public_dispatch": ("numba", "python"),
        "selection": "call_time",
        "public_consumers": (
            {
                "symbol": "weightedcluster.seqnull.seqnull",
                "selection": "call_time",
                "condition": "model=stateindep",
            },
        ),
    },
    "seqnull_markov": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "numba",
                "symbol": "weightedcluster._numba.seqnull_markov_numba",
            },
            {
                "backend": "python",
                "symbol": "weightedcluster.seqnull._seqnull_pos",
            },
        ),
        "public_dispatch": ("numba", "python"),
        "selection": "conditional",
        "public_consumers": (
            {
                "symbol": "weightedcluster.seqnull.seqnull",
                "selection": "call_time",
                "condition": "model=markov or supported static model=userpos",
            },
        ),
    },
    "seqnull_spell_models": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "python",
                "symbol": "weightedcluster.seqnull._seqnull_spell",
            },
        ),
        "public_dispatch": ("python",),
        "selection": "python_only",
        "public_consumers": (
            {
                "symbol": "weightedcluster.seqnull.seqnull",
                "selection": "python_only",
                "condition": "model in {combined,duration,sequencing}",
            },
        ),
    },
    "seqnull_userpos_time_varying": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "python",
                "symbol": "weightedcluster.seqnull._seqnull_pos",
            },
        ),
        "public_dispatch": ("python",),
        "selection": "python_only",
        "public_consumers": (
            {
                "symbol": "weightedcluster.seqnull.seqnull",
                "selection": "python_only",
                "condition": "model=userpos with time-varying transitions",
            },
        ),
    },
    "sequence_distances": {
        "owner": "traminex",
        "provider": "traminex",
        "implementations": (
            {
                "backend": "traminex",
                "symbol": "traminex.distances.pairwise",
            },
            {
                "backend": "traminex",
                "symbol": "traminex.distances.pairwise_between_sets",
            },
        ),
        "public_dispatch": ("traminex",),
        "selection": "delegated",
        "public_consumers": (
            {
                "symbol": "weightedcluster._seqdist.sequences_distance",
                "selection": "delegated",
            },
            {
                "symbol": "weightedcluster._seqdist.sequences_distance_between",
                "selection": "delegated",
            },
        ),
    },
    "hierarchical_linkage": {
        "owner": "pyweightedcluster",
        "implementations": (
            {
                "backend": "python",
                "symbol": "weightedcluster._fastcluster_linkage.linkage_fastcluster",
            },
            {
                "backend": "python",
                "symbol": "weightedcluster._cmpcluster_algorithms._diana_merge_structure",
            },
            {
                "backend": "python",
                "symbol": "weightedcluster.cmpcluster._linkage_flexible_beta",
            },
        ),
        "public_dispatch": ("python",),
        "selection": "python_only",
        "public_consumers": (
            {
                "symbol": "weightedcluster.cmpcluster.wcCmpCluster",
                "selection": "call_time",
            },
            {
                "symbol": "weightedcluster.seqclararange._ward_initial_hierarchy",
                "selection": "call_time",
            },
            {
                "symbol": "weightedcluster.seqnull._recluster_seqnull_distance",
                "selection": "call_time",
            },
            {
                "symbol": "weightedcluster.kmedoids.wcKMedoids",
                "selection": "caller_supplied",
            },
        ),
    },
}

#: Third-party and executor selections deliberately **not** in the registry,
#: with the reason for each. Finding F-10 of the 2026-08-24 whole-codebase audit
#: asked for an explicit decision on every unregistered selection rather than
#: silence, so this records one.
#:
#: The registry's purpose is to let ``get_backend_info()`` say which
#: implementation produced a caller's numbers. A selection that cannot change a
#: returned value therefore does not belong in it. Registering such selections
#: would make the registry a general dependency inventory, which
#: ``pyproject.toml`` already is.
#:
#: **These decisions are deliberately not surfaced through**
#: ``get_backend_info()``. Finding F-12 of the 2026-08-25 Bayesian RARCAT audit
#: observed that a user cannot read them at runtime, and checked whether that
#: made the accessor untruthful; it does not. ``get_backend_info()`` reports
#: backend *selections*, every entry here is a declared *non*-selection with a
#: stated reason, and omitting a non-selection from a selection registry is
#: consistent rather than false. Adding a key would change public API — five
#: test modules and ``docs/API_INDEX.md`` consume its exact shape — for an
#: informational gain already available here and in ``pyproject.toml``. The
#: dictionary is pinned by ``tests/test_backend_dispatch_contracts.py``.
UNREGISTERED_SELECTIONS: Final[dict[str, str]] = {
    "matplotlib.pyplot (weightedcluster.fuzzy)": (
        "presentation only; no returned value depends on it"
    ),
    "seaborn (weightedcluster.plot)": "presentation only; changes styling, not values",
    "mplcursors (weightedcluster.clustassoc)": (
        "presentation only; adds hover tooltips and warns when absent"
    ),
    "tqdm (weightedcluster._parallel)": (
        "progress reporting only; warns when absent and requested"
    ),
    "threadpoolctl (weightedcluster._threads)": (
        "performance only; limits native thread pools without changing results, "
        "and warns when absent and requested"
    ),
    "pymc (weightedcluster._bayesian_rarcat_conditional)": (
        "a hard requirement of the bayesian extra rather than a selection: there "
        "is no alternative implementation to choose between, and its absence "
        "raises rather than routing elsewhere"
    ),
    "arviz / arviz_stats (weightedcluster._bayesian_rarcat_*)": (
        "as for pymc: required by the bayesian extra, with no alternative"
    ),
    "process / thread / serial executor (weightedcluster._parallel)": (
        "scheduling only; run_task_batch reassembles results by submission index, "
        "so the executor cannot reorder or alter output"
    ),
}


def build_capability_info(
    resolve_symbol: Callable[[str], tuple[bool, bool]],
) -> dict[str, dict[str, object]]:
    """Materialize deterministic JSON-safe capability data.

    ``resolve_symbol`` supplies ``(module_importable, symbol_callable)`` for a
    fully qualified symbol.  Keeping resolution outside this function makes
    the schema builder deterministic and side-effect free.
    """
    capabilities: dict[str, dict[str, object]] = {}
    for name in sorted(CAPABILITY_REGISTRY):
        contract = CAPABILITY_REGISTRY[name]
        public_dispatch = [str(item) for item in contract["public_dispatch"]]
        implementations = []
        declared = sorted(
            contract["implementations"],
            key=lambda item: (str(item["backend"]), str(item["symbol"])),
        )
        for implementation in declared:
            backend = str(implementation["backend"])
            symbol = str(implementation["symbol"])
            importable, symbol_callable = resolve_symbol(symbol)
            implementations.append(
                {
                    "backend": backend,
                    "symbol": symbol,
                    "importable": bool(importable),
                    "callable": bool(symbol_callable),
                    "publicly_selected": backend in public_dispatch,
                }
            )

        capability: dict[str, object] = {
            "owner": str(contract["owner"]),
            "implementations": implementations,
            "public_dispatch": public_dispatch,
            "selection": str(contract["selection"]),
            "public_consumers": [dict(item) for item in contract["public_consumers"]],
        }
        if "provider" in contract:
            capability["provider"] = str(contract["provider"])
        capabilities[name] = capability
    return capabilities

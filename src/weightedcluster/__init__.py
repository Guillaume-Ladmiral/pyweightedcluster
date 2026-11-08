from __future__ import annotations

# The compatibility call must execute between its own stdlib-only import and
# every package import that can bind traminex symbols.
# ruff: noqa: E402,I001

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from ._traminex_compatibility import verify_traminex_compatibility as _verify_traminex_compatibility

_verify_traminex_compatibility()
del _verify_traminex_compatibility

from ._backend_contract import build_capability_info
from .aggregate import wcAggregateCases
from .bayesian import BayesRARCATResult, bayes_rarcat
from .boot import bootclustrange
from .clustassoc import clustassoc, plot_clustassoc
from .cmpcluster import (
    print_clustrangefamily,
    summary_clustrangefamily,
    wcCmpCluster,
)
from .fuzzy import crispness, fuzzyseqplot, wcFuzzyRange
from .kmedoids import print_kmedoids, wcKMedoids, wcKMedRange
from .plot import (
    clustrange_plot_data,
    clustrangefamily_plot_data,
    plot_bootclustrange,
    plot_clustrange,
    plot_clustrangefamily,
    plot_proptree_labels,
    plot_proptree_panel,
    plot_proptree_tree,
    plot_seqclararange_stability,
    plot_seqnullcqi,
    print_bootclustrange,
    print_seqnullcqi,
    proptree_plot_data,
)
from .proptree import dtcut, dtlabels, dtprune, seqpropclust, wcPropertyClustering
from .quality import (
    compute_internal_indices,
    db_index,
    dunn_index,
    pbm_index,
    wcClusterQuality,
    wcFuzzyQuality,
    wcSilhouetteObs,
    xb_index,
)
from .range import as_clustrange, print_clustrange, summary_clustrange
from .seqclararange import seqclararange, seqclararange_family
from .seqclustname import seqclustname
from .seqnull import seqnull, seqnullcqi
from .seqtree import as_seqtree

# Single source of truth: the version lives in pyproject.toml and is read from
# the installed distribution metadata. Do not hardcode a literal here (that is
# what caused README/package version drift).
try:
    __version__ = _dist_version("pyweightedcluster")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "unknown"


def get_backend_info() -> dict[str, object]:
    """Report backend availability per operation without executing kernels.

    The legacy top-level strings report package/module availability.  The
    additive ``capabilities`` mapping describes fully qualified symbols,
    symbol callability, real public dispatch, selection timing, and ownership.

    Returns
    -------
    dict
        Backend status with keys:
        - 'compiled_core' : str, 'available' or 'unavailable'
        - 'numba' : str, 'available (vX.X.X)' or 'unavailable'
        - 'pure_python' : str, always 'available'
        - 'capabilities' : dict, operation-level JSON-safe diagnostics

    Examples
    --------
    >>> import weightedcluster
    >>> info = weightedcluster.get_backend_info()
    >>> info['compiled_core'] in {'available', 'unavailable'}
    True
    >>> info['numba'] == 'unavailable' or info['numba'].startswith('available (v')
    True
    >>> info['pure_python']
    'available'
    >>> info['capabilities']['sequence_distances']['owner']
    'traminex'
    """
    module_cache: dict[str, object | None] = {}

    def _load_module(module_name: str) -> object | None:
        if module_name not in module_cache:
            try:
                module_cache[module_name] = importlib.import_module(module_name)
            except ImportError:
                module_cache[module_name] = None
        return module_cache[module_name]

    def _resolve_symbol(symbol: str) -> tuple[bool, bool]:
        module_name, separator, attribute = symbol.rpartition(".")
        if not separator:
            return False, False
        module = _load_module(module_name)
        if module is None:
            return False, False
        return True, callable(getattr(module, attribute, None))

    compiled_module = _load_module(f"{__name__}._wc_core")
    numba_module = _load_module("numba")
    info: dict[str, object] = {
        "compiled_core": "available" if compiled_module is not None else "unavailable",
        "numba": (
            f"available (v{getattr(numba_module, '__version__', 'unknown')})"
            if numba_module is not None
            else "unavailable"
        ),
        "pure_python": "available",
    }
    info["capabilities"] = build_capability_info(_resolve_symbol)
    return info


__all__ = [
    "wcAggregateCases",
    "BayesRARCATResult",
    "bayes_rarcat",
    "bootclustrange",
    "wcCmpCluster",
    "wcKMedoids",
    "wcKMedRange",
    "print_kmedoids",
    "print_clustrange",
    "summary_clustrange",
    "print_clustrangefamily",
    "summary_clustrangefamily",
    "clustrange_plot_data",
    "clustrangefamily_plot_data",
    "plot_bootclustrange",
    "plot_clustrange",
    "plot_seqclararange_stability",
    "plot_seqnullcqi",
    "print_bootclustrange",
    "print_seqnullcqi",
    "plot_clustrangefamily",
    "wcClusterQuality",
    "wcSilhouetteObs",
    "compute_internal_indices",
    "db_index",
    "dunn_index",
    "xb_index",
    "pbm_index",
    "as_clustrange",
    "seqclararange",
    "seqclararange_family",
    "seqclustname",
    "seqnull",
    "seqnullcqi",
    "as_seqtree",
    "clustassoc",
    "plot_clustassoc",
    "wcPropertyClustering",
    "seqpropclust",
    "dtcut",
    "dtlabels",
    "dtprune",
    "wcFuzzyQuality",
    "wcFuzzyRange",
    "crispness",
    "fuzzyseqplot",
    "plot_proptree_labels",
    "plot_proptree_tree",
    "plot_proptree_panel",
    "proptree_plot_data",
    "__version__",
    "get_backend_info",
]

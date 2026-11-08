"""Regression contracts for the 2026-08-24 whole-codebase audit's remediation.

One module for the findings whose fix is a small, self-contained behaviour or
structure change. The larger findings have their own modules:
`test_r_parity_cmpcluster_fastcluster_linkage.py` (F-01, F-03, F-11) and
`test_r_parity_distance_matrix_contract.py` (F-06, F-07).
"""

from __future__ import annotations

import ast
import builtins
import importlib
import re
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "src" / "weightedcluster"


def _source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def test_absent_matplotlib_names_this_package_s_extra_not_traminex_s():
    """Finding F-21: the message was actionable in the wrong package.

    traminex rewrites an optional-dependency `ImportError` to name a *traminex*
    extra. `fuzzy.seqplot` re-raised it unchanged, so a `pyweightedcluster` user
    without matplotlib was told to install `traminex[plot]` even though this
    package declares its own `plot` extra with matplotlib in it.
    """
    fuzzy = importlib.import_module("weightedcluster.fuzzy")
    real_import = builtins.__import__

    def _without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    # traminex resolves `seqplot` lazily, so blocking matplotlib at import time
    # reproduces the real absent-dependency path without uninstalling anything.
    import sys

    removed = {
        key: sys.modules.pop(key)
        for key in [k for k in sys.modules if k == "matplotlib" or k.startswith("matplotlib.")]
    }
    builtins.__import__ = _without_matplotlib
    try:
        with pytest.raises(ImportError) as raised:
            fuzzy.seqplot(None)
    finally:
        builtins.__import__ = real_import
        sys.modules.update(removed)

    message = str(raised.value)
    assert "pyweightedcluster[plot]" in message
    assert "traminex[plot]" not in message
    assert raised.value.__cause__ is not None


def test_absent_tqdm_warns_instead_of_silently_dropping_the_progress_bar():
    """Finding F-28: optional dependencies degraded inconsistently.

    `mplcursors` and `threadpoolctl` both warn when absent; `tqdm` returned the
    unwrapped iterable, so an explicit `progressbar=True` did nothing and said
    nothing.
    """
    parallel = importlib.import_module("weightedcluster._parallel")
    real_import = builtins.__import__

    def _without_tqdm(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("No module named 'tqdm'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _without_tqdm
    try:
        with pytest.warns(RuntimeWarning, match="tqdm is not installed"):
            wrapped = parallel._wrap_progress([1, 2, 3], enabled=True, total=3, desc="probe")
    finally:
        builtins.__import__ = real_import
    assert list(wrapped) == [1, 2, 3]


def test_a_progress_bar_that_was_not_requested_stays_silent():
    parallel = importlib.import_module("weightedcluster._parallel")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert list(parallel._wrap_progress([1, 2], enabled=False, total=2, desc="probe")) == [1, 2]


def test_stslist_is_imported_unguarded_at_every_site():
    """Finding F-30: two of six `STSList` import sites were guarded.

    traminex is a hard runtime dependency, so the handler could not run in a
    supported installation — and its `STSList = ()` sentinel would have made
    `isinstance` silently `False`, quietly failing to recognise sequence objects
    rather than failing loudly.
    """
    guarded: list[str] = []
    sites: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "STSList" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "traminex":
                continue
            if not any(alias.name == "STSList" for alias in node.names):
                continue
            sites.append(f"{path.name}:{node.lineno}")
        if re.search(r"except ImportError:.*\n\s*STSList = \(\)", text):
            guarded.append(path.name)

    assert len(sites) == 6, sites
    assert guarded == []


def test_the_non_finite_silhouette_contract_has_exactly_one_live_call_site():
    """Finding F-18: two of its three call sites could never do work.

    `wcClusterQuality` routes every non-finite matrix to
    `_pure_public_quality_result` before the compiled or Numba branch runs, and
    the contract returns immediately for a wholly finite matrix, so the calls at
    the end of those two branches were no-ops. Removing them makes the routing
    guard the single place the non-finite contract is decided.
    """
    text = _source("quality.py")
    calls = [
        line
        for line in text.splitlines()
        if "_apply_nonfinite_silhouette_contract(" in line and not line.strip().startswith("def ")
    ]
    assert len(calls) == 1, calls
    assert "not np.all(np.isfinite(D))" in text


def test_the_unguarded_quality_kernel_says_so_at_its_definition():
    """Finding F-19: the name is the inverse of R's, so it needs the contrast.

    R's `wcClusterQualityInternal` holds the small-cluster ASW guard and
    `clustrangeboot` bypasses it; here the guard is in the public
    `wcClusterQuality` and this kernel is the permissive one. The behaviour is
    correct and must not change.
    """
    quality = importlib.import_module("weightedcluster.quality")
    doc = quality._wc_cluster_quality_internal.__doc__ or ""
    assert "unguarded" in doc.lower()
    assert "wcClusterQualityInternal" in doc
    assert "must not change" in doc


def test_the_four_shadowed_package_attributes_are_documented():
    """Finding F-17: four attributes are functions where a reader expects modules.

    The count is measured over every submodule rather than sampled — an earlier
    five-name probe reported three and missed `seqclustname`.
    """
    import weightedcluster

    shadowed = []
    for path in sorted(PACKAGE.glob("*.py")):
        stem = path.stem
        if stem == "__init__":
            continue
        attribute = getattr(weightedcluster, stem, None)
        if attribute is None:
            continue
        module = importlib.import_module(f"weightedcluster.{stem}")
        if attribute is not module:
            shadowed.append(stem)

    assert sorted(shadowed) == ["clustassoc", "seqclararange", "seqclustname", "seqnull"]

    api_index = (REPO_ROOT / "docs" / "API_INDEX.md").read_text(encoding="utf-8")
    for name in shadowed:
        assert f"`weightedcluster.{name}`" in api_index, name
    assert "package attributes are functions, not modules" in api_index

    # The documented consequence must be real: a shadowed module is unreachable
    # by attribute access, an unshadowed one is not.
    with pytest.raises(AttributeError):
        weightedcluster.seqnull.seqnullcqi  # noqa: B018
    assert weightedcluster.boot.bootclustrange is not None


def test_the_superseded_seqnull_statistics_helpers_are_gone():
    """Finding F-16: they duplicated statistics `ECOSYSTEM.md` assigns to traminex.

    `_freqs_per_time` and `_global_transition` recomputed per-time state
    frequencies and the global transition matrix locally, while the production
    `seqnull` path already obtains both from traminex's `seqstatd` and
    `seqtrate`. They were reachable only from tests.
    """
    seqnull_source = _source("seqnull.py")
    assert "_freqs_per_time" not in seqnull_source
    assert "_global_transition" not in seqnull_source
    assert "seqstatd" in seqnull_source
    assert "seqtrate" in seqnull_source

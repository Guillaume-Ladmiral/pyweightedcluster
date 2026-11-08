from __future__ import annotations

import ast
import importlib
import inspect
import json
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pytest

import weightedcluster

PKG_DIR = Path(weightedcluster.__file__).resolve().parent
FOREIGN_CORE_DISTANCE_SYMBOLS = {
    "seq_hamming",
    "seq_hamming_between",
    "seq_om",
    "seq_om_between",
    "seq_lcs",
    "seq_lcs_between",
    "seq_dhd",
    "seq_dhd_between",
}


def _imported_modules(mod) -> set[str]:
    """Set of imported module/symbol dotted names in a module's own source."""
    tree = ast.parse(inspect.getsource(mod))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            for alias in node.names:
                names.add(f"{base}.{alias.name}" if base else alias.name)
    return names


def _defined_function_names() -> set[str]:
    """All function names defined anywhere in the installed package source."""
    defined: set[str] = set()
    for path in PKG_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
    return defined


@pytest.mark.parametrize("modname", ["weightedcluster.costs", "weightedcluster._omspell"])
def test_removed_distance_cost_modules_stay_absent(modname):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(modname)


def test_no_local_weighted_pairwise_sum_definition():
    # The faithful weighted-inertia maths lives in traminex.diss; the former
    # local copies were deleted (migration Phase 2.5). They must not return.
    defined = _defined_function_names()
    assert "weighted_pairwise_sum" not in defined
    assert "_weighted_pairwise_sum" not in defined


def test_seqdist_delegates_distances_to_traminex():
    from weightedcluster import _seqdist

    imports = _imported_modules(_seqdist)
    # delegates to the traminex distance engine ...
    assert any(m == "traminex.distances" or m.startswith("traminex.distances.") for m in imports)
    # ... and does not re-import any local distance kernel
    banned_leaf = {"_wc_core", "_fallback", "_numba"}
    leaves = {m.rsplit(".", 1)[-1] for m in imports}
    assert not (leaves & banned_leaf)


@pytest.mark.parametrize(
    ("metric", "provider_method"),
    [
        ("HAM", "DHD"),
        ("OM", "OM"),
        ("LCS", "LCS"),
        ("DHD", "DHD"),
        ("OMSPELL", "OMSPELL"),
    ],
)
def test_every_supported_pairwise_distance_calls_traminex(
    monkeypatch: pytest.MonkeyPatch, metric: str, provider_method: str
):
    from weightedcluster import _seqdist

    calls: list[dict[str, object]] = []

    def _pairwise(X: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append(dict(kwargs))
        return np.zeros((X.shape[0], X.shape[0]), dtype=float)

    monkeypatch.setattr(_seqdist, "_tx_pairwise", _pairwise)
    X = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)

    result = _seqdist.sequences_distance(X, metric=metric)

    assert result.shape == (2, 2)
    assert [call["method"] for call in calls] == [provider_method]


@pytest.mark.parametrize(
    ("metric", "provider_method"),
    [
        ("HAM", "DHD"),
        ("OM", "OM"),
        ("LCS", "LCS"),
        ("DHD", "DHD"),
        ("OMSPELL", "OMSPELL"),
    ],
)
def test_every_supported_between_set_distance_calls_traminex(
    monkeypatch: pytest.MonkeyPatch, metric: str, provider_method: str
):
    from weightedcluster import _seqdist

    calls: list[dict[str, object]] = []

    def _between(X: np.ndarray, Y: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append(dict(kwargs))
        return np.zeros((X.shape[0], Y.shape[0]), dtype=float)

    monkeypatch.setattr(_seqdist, "_tx_pairwise_between", _between)
    X = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)
    Y = np.array([[0, 0, 1]], dtype=int)

    result = _seqdist.sequences_distance_between(X, Y, metric=metric)

    assert result.shape == (2, 1)
    assert [call["method"] for call in calls] == [provider_method]


def test_high_level_seqdist_entry_points_call_exact_public_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import traminex

    boot_module = importlib.import_module("weightedcluster.boot")
    seqnull_module = importlib.import_module("weightedcluster.seqnull")
    calls: list[tuple[np.ndarray, str, dict[str, object]]] = []

    def _seqdist_spy(seqdata, *, method, **kwargs):
        array = np.asarray(seqdata, dtype=int)
        calls.append((array.copy(), str(method), dict(kwargs)))
        return np.zeros((array.shape[0], array.shape[0]), dtype=float)

    assert boot_module.traminex_seqdist is traminex.seqdist
    assert seqnull_module.traminex_seqdist is traminex.seqdist
    monkeypatch.setattr(boot_module, "traminex_seqdist", _seqdist_spy)
    monkeypatch.setattr(seqnull_module, "traminex_seqdist", _seqdist_spy)
    sequences = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
    costs = np.array([[0.0, 1.0], [1.0, 0.0]])

    boot_module._boot_distance_matrix(
        sequences,
        seqdist_args={"method": "HAM", "sm": costs, "norm": "maxlength"},
    )
    seqnull_module._seqnull_distance_matrix(
        sequences,
        seqdist_args={
            "metric": "DHD",
            "substitution_costs": costs,
            "normalize": False,
        },
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], sequences)
    np.testing.assert_array_equal(calls[1][0], sequences)
    assert calls[0][1] == "HAM"
    assert calls[0][2]["norm"] == "maxlength"
    np.testing.assert_array_equal(calls[0][2]["subm"], costs)
    assert calls[1][1] == "DHD"
    assert calls[1][2]["norm"] == "none"
    np.testing.assert_array_equal(calls[1][2]["subm"], costs)


def test_traminex_provider_version_and_editable_source_are_exact() -> None:
    import traminex

    distribution = metadata.distribution("traminex")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    source = Path(inspect.getsourcefile(traminex.seqdist) or "").resolve()

    assert metadata.version("traminex") == "0.2.2"
    assert distribution.metadata["Name"] == "traminex"
    assert traminex.seqdist.__module__ == "traminex.distances"
    if direct_url.get("dir_info", {}).get("editable") is True:
        editable_root = Path(unquote(urlparse(str(direct_url["url"])).path)).resolve()
        assert source.is_relative_to(editable_root)
    else:
        assert source.is_relative_to(Path(distribution.locate_file("")).resolve())


def test_traminex_compatibility_check_precedes_delegated_imports() -> None:
    assert not hasattr(weightedcluster, "verify_traminex_compatibility")
    assert "verify_traminex_compatibility" not in weightedcluster.__all__
    package_init = PKG_DIR / "__init__.py"
    tree = ast.parse(package_init.read_text(encoding="utf-8"), filename=str(package_init))
    statements = tree.body
    compatibility_import = next(
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.ImportFrom) and node.module == "_traminex_compatibility"
    )
    verification_call = next(
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_verify_traminex_compatibility"
    )
    delegated_imports = [
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.ImportFrom)
        and node.module not in {"__future__", "_traminex_compatibility"}
        and node.level == 1
    ]

    assert compatibility_import < verification_call < min(delegated_imports)

    compatibility_source = (PKG_DIR / "_traminex_compatibility.py").read_text(encoding="utf-8")
    compatibility_tree = ast.parse(compatibility_source)
    imported = {
        alias.name
        for node in ast.walk(compatibility_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "traminex" not in imported


def test_production_python_does_not_consume_foreign_core_distance_symbols():
    offenders: dict[str, list[str]] = {}
    for path in PKG_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        present = sorted(symbol for symbol in FOREIGN_CORE_DISTANCE_SYMBOLS if symbol in text)
        if present:
            offenders[path.name] = present

    assert offenders == {}


def test_seqdist_dependencies_are_public_traminex_submodule_contracts():
    import traminex
    import traminex.distances as distances

    required = {"pairwise", "pairwise_between_sets"}
    assert required <= set(distances.__all__)
    assert all(callable(getattr(distances, name)) for name in required)
    assert required.isdisjoint(traminex.__all__)


def test_seqnull_uses_public_traminex_sequence_statistics_and_transitions():
    seqnull_module = importlib.import_module("weightedcluster.seqnull")
    imports = _imported_modules(seqnull_module)
    assert {
        "traminex.STSList",
        "traminex.seqdss",
        "traminex.seqdur",
        "traminex.seqlength",
        "traminex.seqstatd",
        "traminex.seqstatf",
        "traminex.seqtrate",
    } <= imports


def test_clustassoc_delegates_discrepancy_to_traminex():
    from weightedcluster import clustassoc as public_clustassoc

    module = importlib.import_module(public_clustassoc.__module__)
    imports = _imported_modules(module)
    assert "traminex.dissmfacw" in imports
    assert "traminex.diss.weighted_pairwise_sum" not in imports
    assert "_weighted_within_and_total" not in _defined_function_names()


def test_property_tree_growth_and_pruned_adjustment_delegate_to_traminex():
    from weightedcluster import proptree

    imports = _imported_modules(proptree)
    assert {
        "traminex.disstree.disstree",
        "traminex.seqtree.seqtree",
        "traminex.diss.dissassoc",
        "traminex.seqST",
        "traminex.seqdss",
        "traminex.seqdur",
        "traminex.seqeapplysub",
        "traminex.seqecreate",
        "traminex.seqefsub",
        "traminex.seqici",
        "traminex.seqient",
        "traminex.seqistatd",
        "traminex.seqtransn",
    } <= imports
    assert "traminex.diss.weighted_pairwise_sum" not in imports
    defined = _defined_function_names()
    assert "_build_tree" not in defined
    assert "_best_split_for_variable" not in defined


@pytest.mark.parametrize(
    "modname",
    [
        "weightedcluster._utils",
        "weightedcluster.quality",
    ],
)
def test_diss_sites_consume_traminex_weighted_pairwise_sum(modname):
    mod = importlib.import_module(modname)
    imports = _imported_modules(mod)
    assert "traminex.diss.weighted_pairwise_sum" in imports


def test_as_seqtree_uses_only_public_traminex_tree_math_primitives():
    from weightedcluster import seqtree

    imports = _imported_modules(seqtree)
    assert {
        "traminex.dissassoc",
        "traminex.disscenter",
        "traminex.disstreeleaf",
        "traminex.dissvar",
        "traminex.weighted_inertia",
    } <= imports
    assert "traminex.diss.weighted_pairwise_sum" not in imports
    source_tree = ast.parse(inspect.getsource(seqtree))
    local_names = {
        node.name
        for node in ast.walk(source_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"weighted_inertia", "dissvar", "dissassoc"}.isdisjoint(local_names)


def test_fuzzyseqplot_delegates_general_rendering_and_narrow_index_facilities_to_traminex():
    from weightedcluster import fuzzy

    imports = _imported_modules(fuzzy)
    assert {
        "traminex.STSList",
        "traminex.seqdef",
        "traminex.seqplot",
        "traminex.visualization.state_legend",
        "traminex.visualization.palette.ensure_cmap",
        "traminex.visualization.series.select_index_rows",
        "traminex.visualization.ticks.apply_time_axis",
    } <= imports
    public_source = inspect.getsource(fuzzy.fuzzyseqplot)
    renderer_source = inspect.getsource(fuzzy._render_fuzzyseqplot_panels)
    index_adapter_source = inspect.getsource(fuzzy._render_weighted_index_panel)
    assert "seqplot(" in renderer_source
    assert "_render_weighted_index_panel(" in renderer_source
    assert ".pcolormesh(" in index_adapter_source
    assert ".pcolormesh(" not in public_source + renderer_source
    assert ".bar(" not in public_source + renderer_source + index_adapter_source
    assert ".imshow(" not in public_source + renderer_source + index_adapter_source


def test_base_package_import_does_not_require_optional_matplotlib():
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class BlockMatplotlib(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                    raise ModuleNotFoundError("matplotlib intentionally unavailable")
                return None

        sys.meta_path.insert(0, BlockMatplotlib())
        import weightedcluster

        assert "matplotlib" not in sys.modules
        assert callable(weightedcluster.fuzzyseqplot)
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_surface_does_not_export_sequences_distance():
    assert not hasattr(weightedcluster, "sequences_distance")
    assert "sequences_distance" not in weightedcluster.__all__

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_native_build_surface_is_pybind11_only():
    pyproject_text = _read(REPO_ROOT / "pyproject.toml")
    cmake_text = _read(REPO_ROOT / "CMakeLists.txt")

    assert "pybind11>=2.11" in pyproject_text
    assert "pybind11_add_module(_wc_core MODULE cpp/wc_core.cpp)" in cmake_text
    assert "WC_USE_NANOBIND" not in cmake_text
    assert "nanobind" not in cmake_text.lower()
    assert "wc_core_nb.cpp" not in cmake_text


def test_orphan_nanobind_source_is_removed():
    assert not (REPO_ROOT / "cpp" / "wc_core_nb.cpp").exists()


def test_required_pandas_is_not_treated_as_optional_in_property_trees():
    pyproject_text = _read(REPO_ROOT / "pyproject.toml")
    proptree_text = _read(REPO_ROOT / "src" / "weightedcluster" / "proptree.py")

    assert '"pandas>=2.1"' in pyproject_text
    assert "import pandas as pd" in proptree_text
    assert "if pd is None" not in proptree_text
    assert "except ImportError:  # pragma: no cover" not in proptree_text


def test_traminex_content_commitment_is_packaged_and_complete():
    manifest_path = REPO_ROOT / "src" / "weightedcluster" / "_traminex_compatibility.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["algorithm"] == "sha256(sorted(relative-posix-path + NUL + bytes + NUL))"
    assert manifest["distribution"] == "traminex"
    assert manifest["distribution_version"] == "0.2.2"
    assert manifest["source_revision"] == "2ed89d77a75d67a4ac319c05da463a1758244503"
    assert manifest["python_file_count"] == len(manifest["python_files"]) == 83
    assert manifest["python_files"] == sorted(manifest["python_files"])
    assert manifest["source_tree_sha256"] == (
        "e2ccb374b64010843e58b334299f9902edada5e5056128567afa728bbe39aabb"
    )

    pyproject_text = _read(REPO_ROOT / "pyproject.toml")
    assert '"src/weightedcluster/_traminex_compatibility.json"' in pyproject_text

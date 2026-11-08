from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAMINEX_SPEC = find_spec("traminex")
assert TRAMINEX_SPEC is not None
assert TRAMINEX_SPEC.submodule_search_locations is not None
TRAMINEX_ROOT = Path(next(iter(TRAMINEX_SPEC.submodule_search_locations))).resolve()


def _compatibility_module():
    try:
        return importlib.import_module("weightedcluster._traminex_compatibility")
    except ModuleNotFoundError:
        pytest.fail("weightedcluster._traminex_compatibility is missing", pytrace=False)


def _digest(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _commitment(root: Path, *, version: str = "0.0.8") -> dict[str, object]:
    paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py"))
    return {
        "schema_version": 1,
        "algorithm": "sha256(sorted(relative-posix-path + NUL + bytes + NUL))",
        "distribution": "traminex",
        "distribution_version": version,
        "source_revision": "0" * 40,
        "python_file_count": len(paths),
        "python_files": paths,
        "source_tree_sha256": _digest(root, paths),
    }


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "traminex"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "provider.py").write_text("def delegated():\n    return 1\n", encoding="utf-8")
    return root


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    compatibility,
    root: Path,
    commitment: dict[str, object],
    *,
    version: str = "0.0.8",
) -> None:
    monkeypatch.setattr(compatibility, "_VERIFIED", False)
    monkeypatch.setattr(compatibility, "_read_commitment", lambda: dict(commitment))
    monkeypatch.setattr(compatibility, "_distribution_version", lambda _name: version)
    monkeypatch.setattr(
        compatibility,
        "_find_package_spec",
        lambda _name: SimpleNamespace(
            origin=str(root / "__init__.py"),
            submodule_search_locations=[str(root)],
        ),
    )


def test_matching_package_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    _configure(monkeypatch, compatibility, root, commitment)

    compatibility.verify_traminex_compatibility()

    assert compatibility._VERIFIED is True


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_python_inventory_or_content_mismatch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    if mutation == "changed":
        (root / "provider.py").write_text("def delegated():\n    return 2\n", encoding="utf-8")
    elif mutation == "missing":
        (root / "provider.py").unlink()
    else:
        (root / "extra.py").write_text("EXTRA = True\n", encoding="utf-8")
    _configure(monkeypatch, compatibility, root, commitment)

    with pytest.raises(compatibility.TraminexCompatibilityError, match=mutation):
        compatibility.verify_traminex_compatibility()

    assert compatibility._VERIFIED is False


def test_same_version_behavior_change_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root, version="0.0.8")
    (root / "provider.py").write_text(
        "def delegated():\n    return 'different'\n", encoding="utf-8"
    )
    _configure(monkeypatch, compatibility, root, commitment, version="0.0.8")

    with pytest.raises(compatibility.TraminexCompatibilityError, match="digest"):
        compatibility.verify_traminex_compatibility()


def test_symlinked_python_source_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    target = tmp_path / "outside.py"
    target.write_text((root / "provider.py").read_text(encoding="utf-8"), encoding="utf-8")
    (root / "provider.py").unlink()
    (root / "provider.py").symlink_to(target)
    _configure(monkeypatch, compatibility, root, commitment)

    with pytest.raises(compatibility.TraminexCompatibilityError, match="symlink"):
        compatibility.verify_traminex_compatibility()


def test_symlinked_package_root_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compatibility = _compatibility_module()
    real_root = _package(tmp_path / "real")
    commitment = _commitment(real_root)
    linked_root = tmp_path / "traminex"
    linked_root.symlink_to(real_root, target_is_directory=True)
    _configure(monkeypatch, compatibility, linked_root, commitment)

    with pytest.raises(compatibility.TraminexCompatibilityError, match="symlink"):
        compatibility.verify_traminex_compatibility()


def test_package_root_replaced_by_symlink_after_validation_fails_without_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    replacement = tmp_path / "identical-traminex"
    shutil.copytree(root, replacement)
    displaced = tmp_path / "original-traminex"
    real_package_root_from_spec = compatibility._package_root_from_spec
    replaced = False

    def _replace_validated_root(spec: object | None):
        nonlocal replaced
        validated_root = real_package_root_from_spec(spec)
        compatibility.os.rename(validated_root.path, displaced)
        validated_root.path.symlink_to(replacement, target_is_directory=True)
        replaced = True
        return validated_root

    _configure(monkeypatch, compatibility, root, commitment)
    monkeypatch.setattr(compatibility, "_package_root_from_spec", _replace_validated_root)

    with pytest.raises(compatibility.TraminexCompatibilityError, match="package root"):
        compatibility.verify_traminex_compatibility()

    assert replaced is True
    assert compatibility._VERIFIED is False


def test_irrelevant_cache_compiled_and_metadata_files_are_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "provider.cpython-312.pyc").write_bytes(b"compiled-cache")
    (root / "native.so").write_bytes(b"compiled-extension")
    metadata = root / "traminex-0.0.8.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Version: different text\n", encoding="utf-8")
    _configure(monkeypatch, compatibility, root, commitment)

    compatibility.verify_traminex_compatibility()


def test_absent_distribution_fails_and_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    _configure(monkeypatch, compatibility, root, commitment)

    def _missing(_name: str) -> str:
        raise compatibility.PackageNotFoundError("traminex")

    monkeypatch.setattr(compatibility, "_distribution_version", _missing)
    with pytest.raises(compatibility.TraminexCompatibilityError, match="distribution is absent"):
        compatibility.verify_traminex_compatibility()

    assert compatibility._VERIFIED is False


def test_absent_package_fails_and_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    _configure(monkeypatch, compatibility, root, commitment)
    monkeypatch.setattr(compatibility, "_find_package_spec", lambda _name: None)

    with pytest.raises(compatibility.TraminexCompatibilityError, match="package is absent"):
        compatibility.verify_traminex_compatibility()

    assert compatibility._VERIFIED is False


def test_distribution_version_mismatch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    _configure(monkeypatch, compatibility, root, commitment, version="0.0.9")

    with pytest.raises(compatibility.TraminexCompatibilityError, match="version"):
        compatibility.verify_traminex_compatibility()


def test_failed_check_is_retried_but_success_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    calls = 0

    def _version(_name: str) -> str:
        nonlocal calls
        calls += 1
        return "0.0.9" if calls == 1 else "0.0.8"

    _configure(monkeypatch, compatibility, root, commitment)
    monkeypatch.setattr(compatibility, "_distribution_version", _version)

    with pytest.raises(compatibility.TraminexCompatibilityError):
        compatibility.verify_traminex_compatibility()
    compatibility.verify_traminex_compatibility()
    compatibility.verify_traminex_compatibility()

    assert calls == 2
    assert compatibility._VERIFIED is True


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("schema_bool", "schema"),
        ("schema_float", "schema"),
        ("empty_version", "distribution"),
        ("numeric_version", "distribution"),
        ("short_revision", "source revision"),
        ("uppercase_revision", "source revision"),
        ("nonhex_revision", "source revision"),
        ("short_digest", "SHA-256"),
        ("uppercase_digest", "SHA-256"),
        ("nonhex_digest", "SHA-256"),
        ("count_bool", "count"),
        ("count_float", "count"),
        ("empty_files", "Python-file"),
        ("unsorted_files", "unique and sorted"),
        ("duplicate_files", "unique and sorted"),
        ("absolute_path", "unsafe path"),
        ("parent_path", "unsafe path"),
        ("backslash_path", "unsafe path"),
        ("nul_path", "unsafe path"),
        ("control_path", "unsafe path"),
        ("surrogate_path", "unsafe path"),
        ("wrong_algorithm", "algorithm"),
    ],
)
def test_malformed_commitment_fails_stably_without_caching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    if case == "schema_bool":
        commitment["schema_version"] = True
    elif case == "schema_float":
        commitment["schema_version"] = 1.0
    elif case == "empty_version":
        commitment["distribution_version"] = ""
    elif case == "numeric_version":
        commitment["distribution_version"] = 8
    elif case == "short_revision":
        commitment["source_revision"] = "0" * 39
    elif case == "uppercase_revision":
        commitment["source_revision"] = "A" * 40
    elif case == "nonhex_revision":
        commitment["source_revision"] = "g" * 40
    elif case == "short_digest":
        commitment["source_tree_sha256"] = "0" * 63
    elif case == "uppercase_digest":
        commitment["source_tree_sha256"] = "A" * 64
    elif case == "nonhex_digest":
        commitment["source_tree_sha256"] = "g" * 64
    elif case == "count_bool":
        commitment["python_file_count"] = True
    elif case == "count_float":
        commitment["python_file_count"] = 2.0
    elif case == "empty_files":
        commitment["python_files"] = []
        commitment["python_file_count"] = 0
    elif case == "unsorted_files":
        commitment["python_files"] = ["provider.py", "__init__.py"]
    elif case == "duplicate_files":
        commitment["python_files"] = ["__init__.py", "__init__.py"]
    else:
        unsafe_paths = {
            "absolute_path": "/provider.py",
            "parent_path": "../provider.py",
            "backslash_path": "folder\\provider.py",
            "nul_path": "provider\0.py",
            "control_path": "provider\n.py",
            "surrogate_path": "provider\ud800.py",
        }
        if case in unsafe_paths:
            commitment["python_files"] = [unsafe_paths[case]]
            commitment["python_file_count"] = 1
        else:
            assert case == "wrong_algorithm"
            commitment["algorithm"] = "sha1"
    _configure(monkeypatch, compatibility, root, commitment)

    with pytest.raises(compatibility.TraminexCompatibilityError, match=message):
        compatibility.verify_traminex_compatibility()

    assert compatibility._VERIFIED is False


def test_pathname_replacement_during_source_read_fails_without_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    root = _package(tmp_path)
    commitment = _commitment(root)
    provider = root / "provider.py"
    provider_identity = provider.stat().st_ino
    replacement = root / "replacement.tmp"
    replacement.write_bytes(provider.read_bytes())
    real_fdopen = compatibility.os.fdopen
    replaced = False

    def _replace_pathname(descriptor: int, mode: str):
        nonlocal replaced
        if not replaced and compatibility.os.fstat(descriptor).st_ino == provider_identity:
            compatibility.os.replace(replacement, provider)
            replaced = True
        return real_fdopen(descriptor, mode)

    _configure(monkeypatch, compatibility, root, commitment)
    monkeypatch.setattr(compatibility.os, "fdopen", _replace_pathname)

    with pytest.raises(compatibility.TraminexCompatibilityError, match="changed during"):
        compatibility.verify_traminex_compatibility()

    assert replaced is True
    assert compatibility._VERIFIED is False


def test_dirty_temporary_copy_fails_without_mutating_real_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatibility = _compatibility_module()
    manifest = json.loads(
        (REPO_ROOT / "src/weightedcluster/_traminex_compatibility.json").read_text(encoding="utf-8")
    )
    copied_root = tmp_path / "traminex"
    shutil.copytree(TRAMINEX_ROOT, copied_root)
    changed = copied_root / str(manifest["python_files"][0])
    changed.write_bytes(changed.read_bytes() + b"\n# dirty temporary copy\n")
    _configure(
        monkeypatch,
        compatibility,
        copied_root,
        manifest,
        version=str(manifest["distribution_version"]),
    )

    with pytest.raises(compatibility.TraminexCompatibilityError, match="digest"):
        compatibility.verify_traminex_compatibility()


def test_clean_runner_rejects_changed_source_before_importing_delegated_symbols(
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "traminex"
    shutil.copytree(TRAMINEX_ROOT, copied_root)
    marker = tmp_path / "traminex-imported"
    (copied_root / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "raise AssertionError('delegated traminex symbols imported before verification')\n",
        encoding="utf-8",
    )
    dist_info = tmp_path / "traminex-0.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: traminex\nVersion: 0.2.3\n",
        encoding="utf-8",
    )
    script = (
        "import sys\n"
        "sys.meta_path[:] = [finder for finder in sys.meta_path "
        "if not ('editable' in getattr(finder, '__module__', '').lower() "
        "and 'traminex' in getattr(finder, '__module__', '').lower())]\n"
        "import weightedcluster\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO_ROOT / "src")]),
    }

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "TraminexCompatibilityError" in completed.stderr
    assert "delegated traminex symbols imported" not in completed.stderr
    assert not marker.exists()


def test_clean_runner_accepts_matching_temporary_sibling_copy(tmp_path: Path) -> None:
    copied_root = tmp_path / "traminex"
    shutil.copytree(TRAMINEX_ROOT, copied_root)
    dist_info = tmp_path / "traminex-0.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: traminex\nVersion: 0.2.3\n",
        encoding="utf-8",
    )
    script = (
        "import sys\n"
        "sys.meta_path[:] = [finder for finder in sys.meta_path "
        "if not ('editable' in getattr(finder, '__module__', '').lower() "
        "and 'traminex' in getattr(finder, '__module__', '').lower())]\n"
        "import weightedcluster\n"
        "from weightedcluster._traminex_compatibility import "
        "verify_traminex_compatibility\n"
        "verify_traminex_compatibility()\n"
        "print('clean-runner commitment verified')\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO_ROOT / "src")]),
    }

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean-runner commitment verified"

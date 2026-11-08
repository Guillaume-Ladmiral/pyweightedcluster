"""Fail-closed verification of the delegated traminex source commitment.

This module is deliberately standard-library-only and must not import
``traminex``.  Package initialization calls the verifier before importing any
weightedcluster module that binds delegated traminex symbols.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import unicodedata
from dataclasses import dataclass
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path, PurePosixPath
from typing import Any

_COMMITMENT_ALGORITHM = "sha256(sorted(relative-posix-path + NUL + bytes + NUL))"
_COMMITMENT_PATH = Path(__file__).with_name("_traminex_compatibility.json")
_EXCLUDED_DIRECTORY_SUFFIXES = (".dist-info", ".egg-info")
_VERIFIED = False


class TraminexCompatibilityError(ImportError):
    """Raised when installed traminex does not match the reviewed commitment."""


@dataclass(frozen=True)
class _PackageRoot:
    path: Path
    identity: tuple[int, int, int]


def _read_commitment() -> dict[str, Any]:
    try:
        payload = json.loads(_COMMITMENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraminexCompatibilityError(
            f"cannot read the packaged traminex commitment at {_COMMITMENT_PATH}"
        ) from exc
    if not isinstance(payload, dict):
        raise TraminexCompatibilityError("the packaged traminex commitment is not an object")
    return payload


def _distribution_version(distribution_name: str) -> str:
    return metadata.version(distribution_name)


def _find_package_spec(package_name: str) -> importlib.machinery.ModuleSpec | None:
    return importlib.util.find_spec(package_name)


def _validated_commitment(payload: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
    required = {
        "schema_version",
        "algorithm",
        "distribution",
        "distribution_version",
        "source_revision",
        "python_file_count",
        "python_files",
        "source_tree_sha256",
    }
    missing_keys = sorted(required.difference(payload))
    if missing_keys:
        raise TraminexCompatibilityError(
            f"traminex commitment is missing keys: {', '.join(missing_keys)}"
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise TraminexCompatibilityError(
            f"unsupported traminex commitment schema: {payload['schema_version']!r}"
        )
    if type(payload["algorithm"]) is not str or payload["algorithm"] != _COMMITMENT_ALGORITHM:
        raise TraminexCompatibilityError(
            f"unsupported traminex commitment algorithm: {payload['algorithm']!r}"
        )
    distribution_name = payload["distribution"]
    expected_version = payload["distribution_version"]
    expected_digest = payload["source_tree_sha256"]
    expected_files = payload["python_files"]
    if (
        type(distribution_name) is not str
        or distribution_name != "traminex"
        or type(expected_version) is not str
        or not expected_version.strip()
    ):
        raise TraminexCompatibilityError("invalid traminex distribution commitment")
    source_revision = payload["source_revision"]
    if (
        type(source_revision) is not str
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise TraminexCompatibilityError("invalid traminex source revision commitment")
    if (
        type(expected_digest) is not str
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise TraminexCompatibilityError("invalid traminex SHA-256 commitment")
    if (
        type(expected_files) is not list
        or not expected_files
        or not all(type(relative) is str for relative in expected_files)
    ):
        raise TraminexCompatibilityError("invalid traminex Python-file commitment")
    paths = tuple(expected_files)
    if list(paths) != sorted(paths) or len(paths) != len(set(paths)):
        raise TraminexCompatibilityError("traminex Python-file commitment is not unique and sorted")
    if (
        isinstance(payload["python_file_count"], bool)
        or not isinstance(payload["python_file_count"], int)
        or payload["python_file_count"] != len(paths)
    ):
        raise TraminexCompatibilityError("traminex Python-file commitment count is inconsistent")
    for relative in paths:
        posix_path = PurePosixPath(relative)
        try:
            relative.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TraminexCompatibilityError(
                f"unsafe path in traminex Python-file commitment: {relative!r}"
            ) from exc
        if (
            posix_path.is_absolute()
            or ".." in posix_path.parts
            or "\\" in relative
            or any(unicodedata.category(character) in {"Cc", "Cs"} for character in relative)
            or relative != posix_path.as_posix()
            or posix_path.suffix != ".py"
        ):
            raise TraminexCompatibilityError(
                f"unsafe path in traminex Python-file commitment: {relative!r}"
            )
    return distribution_name, expected_version, paths, expected_digest


def _require_safe_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise TraminexCompatibilityError(f"traminex package root is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise TraminexCompatibilityError(
                f"cannot inspect traminex package-root component: {current}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise TraminexCompatibilityError(
                f"traminex package root contains a symlink component: {current}"
            )
    if not stat.S_ISDIR(path.stat(follow_symlinks=False).st_mode):
        raise TraminexCompatibilityError(f"traminex package root is not a directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise TraminexCompatibilityError(
            f"traminex package root is not canonical: {path} resolves to {resolved}"
        )
    return resolved


def _directory_identity(path: Path) -> tuple[int, int, int]:
    try:
        root_stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TraminexCompatibilityError(f"cannot inspect traminex package root: {path}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise TraminexCompatibilityError(f"traminex package root is not a directory: {path}")
    return root_stat.st_dev, root_stat.st_ino, root_stat.st_mode


def _package_root_from_spec(spec: object | None) -> _PackageRoot:
    if spec is None:
        raise TraminexCompatibilityError("the traminex package is absent")
    locations = getattr(spec, "submodule_search_locations", None)
    if locations is None:
        raise TraminexCompatibilityError("the installed traminex object is not a package")
    location_strings = list(locations)
    if len(location_strings) != 1:
        raise TraminexCompatibilityError(
            "the traminex package must resolve to exactly one source directory"
        )
    root = _require_safe_directory(Path(location_strings[0]))
    origin_value = getattr(spec, "origin", None)
    if not isinstance(origin_value, str):
        raise TraminexCompatibilityError("the traminex package has no concrete __init__.py")
    origin = Path(origin_value)
    try:
        origin_mode = origin.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise TraminexCompatibilityError("the traminex package __init__.py is missing") from exc
    if stat.S_ISLNK(origin_mode):
        raise TraminexCompatibilityError("the traminex package __init__.py is a symlink")
    if origin.resolve(strict=True) != root / "__init__.py":
        raise TraminexCompatibilityError(
            "the traminex package origin lies outside its resolved package root"
        )
    initial_identity = _directory_identity(root)
    # Close the validation-to-snapshot gap as far as pathname APIs permit: the
    # complete component check is repeated around the identity snapshot.
    root = _require_safe_directory(root)
    final_identity = _directory_identity(root)
    if final_identity != initial_identity:
        raise TraminexCompatibilityError(
            "traminex package root changed during compatibility verification"
        )
    return _PackageRoot(root, final_identity)


def _verify_package_root(root: _PackageRoot) -> Path:
    checked_path = _require_safe_directory(root.path)
    if checked_path != root.path or _directory_identity(checked_path) != root.identity:
        raise TraminexCompatibilityError(
            "traminex package root changed during compatibility verification"
        )
    return checked_path


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _python_source_inventory(
    root: Path,
) -> tuple[
    tuple[str, ...],
    dict[str, Path],
    dict[str, tuple[int, int, int, int, int, int]],
]:
    paths: dict[str, Path] = {}
    identities: dict[str, tuple[int, int, int, int, int, int]] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise TraminexCompatibilityError(
                f"cannot enumerate traminex package directory: {directory}"
            ) from exc
        for entry in entries:
            entry_path = Path(entry.path)
            if entry.is_symlink():
                raise TraminexCompatibilityError(
                    f"traminex package contains a symlink: {entry_path.relative_to(root).as_posix()}"
                )
            if entry.is_dir(follow_symlinks=False):
                if entry.name == "__pycache__" or entry.name.endswith(_EXCLUDED_DIRECTORY_SUFFIXES):
                    continue
                pending.append(entry_path)
                continue
            if entry.is_file(follow_symlinks=False) and entry.name.endswith(".py"):
                relative = entry_path.relative_to(root).as_posix()
                paths[relative] = entry_path
                identities[relative] = _file_identity(entry.stat(follow_symlinks=False))
    inventory = tuple(sorted(paths))
    return inventory, paths, identities


def _path_identity(path: Path, relative: str) -> tuple[int, int, int, int, int, int]:
    try:
        return _file_identity(path.stat(follow_symlinks=False))
    except OSError as exc:
        raise TraminexCompatibilityError(
            f"traminex Python source changed during compatibility verification: {relative}"
        ) from exc


def _verify_package_tree(
    package_root: _PackageRoot,
    expected_files: tuple[str, ...],
    expected_digest: str,
) -> None:
    root = _verify_package_root(package_root)
    actual_files, source_paths, initial_identities = _python_source_inventory(root)
    missing = sorted(set(expected_files).difference(actual_files))
    extra = sorted(set(actual_files).difference(expected_files))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing Python source: {', '.join(missing)}")
        if extra:
            details.append(f"extra Python source: {', '.join(extra)}")
        raise TraminexCompatibilityError(
            "traminex source inventory mismatch; " + "; ".join(details)
        )

    digest = hashlib.sha256()
    for relative in actual_files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            descriptor = os.open(
                source_paths[relative],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise TraminexCompatibilityError(
                f"cannot read traminex Python source: {relative}"
            ) from exc
        try:
            with os.fdopen(descriptor, "rb") as source:
                before_read = _file_identity(os.fstat(source.fileno()))
                if not stat.S_ISREG(before_read[2]):
                    raise TraminexCompatibilityError(
                        f"traminex Python source is not a regular file: {relative}"
                    )
                if before_read != initial_identities[relative]:
                    raise TraminexCompatibilityError(
                        "traminex Python source changed during compatibility verification: "
                        f"{relative}"
                    )
                digest.update(source.read())
                after_read = _file_identity(os.fstat(source.fileno()))
                if after_read != before_read:
                    raise TraminexCompatibilityError(
                        "traminex Python source changed during compatibility verification: "
                        f"{relative}"
                    )
        except OSError as exc:
            raise TraminexCompatibilityError(
                f"cannot read traminex Python source: {relative}"
            ) from exc
        if _path_identity(source_paths[relative], relative) != initial_identities[relative]:
            raise TraminexCompatibilityError(
                "traminex Python source changed during compatibility verification: " f"{relative}"
            )
        digest.update(b"\0")

    final_files, _, final_identities = _python_source_inventory(root)
    if final_files != actual_files or final_identities != initial_identities:
        raise TraminexCompatibilityError(
            "traminex Python source changed during final compatibility verification"
        )
    _verify_package_root(package_root)
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        raise TraminexCompatibilityError(
            "traminex changed-source digest mismatch: "
            f"expected {expected_digest}, found {actual_digest}"
        )


def verify_traminex_compatibility() -> None:
    """Verify the installed provider against the reviewed source commitment.

    Only a successful result is cached. Every failure remains retryable after
    the environment is repaired within the same process.
    """

    global _VERIFIED
    if _VERIFIED:
        return

    commitment = _read_commitment()
    distribution_name, expected_version, expected_files, expected_digest = _validated_commitment(
        commitment
    )
    try:
        actual_version = _distribution_version(distribution_name)
    except PackageNotFoundError as exc:
        raise TraminexCompatibilityError("the traminex distribution is absent") from exc
    if actual_version != expected_version:
        raise TraminexCompatibilityError(
            "traminex distribution version mismatch: "
            f"expected {expected_version}, found {actual_version}"
        )

    package_root = _package_root_from_spec(_find_package_spec(distribution_name))
    _verify_package_tree(package_root, expected_files, expected_digest)
    _VERIFIED = True


if __name__ == "__main__":  # pragma: no cover - exercised by install smoke commands
    verify_traminex_compatibility()
    print("verified traminex source commitment")

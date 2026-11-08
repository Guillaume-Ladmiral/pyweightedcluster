from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from weightedcluster import _bayesian_rarcat_mixture as b4
from weightedcluster import _bayesian_rarcat_storage as storage

_FIXTURE_PATH = Path(__file__).with_name("test_bayesian_rarcat_b4_mixture.py")
_FIXTURE_SPEC = importlib.util.spec_from_file_location("_b4_mixture_test_fixtures", _FIXTURE_PATH)
if _FIXTURE_SPEC is None or _FIXTURE_SPEC.loader is None:
    raise RuntimeError("B4 mixture fixture module could not be loaded")
_FIXTURE_MODULE = importlib.util.module_from_spec(_FIXTURE_SPEC)
sys.modules[_FIXTURE_SPEC.name] = _FIXTURE_MODULE
_FIXTURE_SPEC.loader.exec_module(_FIXTURE_MODULE)
phase_pair = _FIXTURE_MODULE.phase_pair
_assert_scientific_results_equal = _FIXTURE_MODULE._assert_scientific_results_equal


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _rehash_manifest(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["payloads"]:
        payload = bundle / entry["path"]
        entry["size"] = payload.stat().st_size
        entry["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest_path.write_bytes(_canonical_json(manifest))


def _assert_tree_equal(left, right):
    assert tuple(left.groups) == tuple(right.groups)
    for group in left.groups:
        left_dataset = left[group].dataset
        right_dataset = right[group].dataset
        assert tuple(left_dataset.variables) == tuple(right_dataset.variables)
        for name in left_dataset.variables:
            left_value = left_dataset[name]
            right_value = right_dataset[name]
            assert left_value.dims == right_value.dims
            assert left_value.dtype == right_value.dtype
            np.testing.assert_array_equal(left_value.values, right_value.values)


def _assert_readonly_arrays(value: object, seen: set[int] | None = None) -> None:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if type(value) is np.ndarray:
        assert not value.flags.writeable
    elif is_dataclass(value):
        for field in fields(value):
            if field.name != "design_info":
                _assert_readonly_arrays(getattr(value, field.name), seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_readonly_arrays(item, seen)


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        True,
        -12345678901234567890,
        -0.0,
        1.25,
        float("inf"),
        float("-inf"),
        float("nan"),
        "1",
        np.bool_(True),
        np.int8(-7),
        np.uint64(2**63 + 5),
        np.float32(1.5),
        np.float64(np.inf),
        np.str_("coordinate"),
    ],
)
def test_tagged_scalar_codec_round_trips_python_and_numpy_identity(value):
    encoded = storage._encode_scalar(value)
    decoded = storage._decode_scalar(encoded)

    assert type(decoded) is type(value)
    if isinstance(value, float) and np.isnan(value):
        assert np.isnan(decoded)
    elif isinstance(value, np.floating) and np.isnan(value):
        assert np.isnan(decoded)
        assert decoded.dtype == value.dtype
    else:
        assert decoded == value
    if isinstance(value, np.generic):
        assert decoded.dtype == value.dtype


def test_tagged_scalar_codec_has_no_bool_int_float_string_or_numpy_collisions():
    values = [1, "1", True, 1.0, np.int64(1), np.float32(1.0), np.str_("1")]
    encoded = [json.dumps(storage._encode_scalar(value), sort_keys=True) for value in values]

    assert len(set(encoded)) == len(values)
    with pytest.raises(TypeError, match="unsupported scalar"):
        storage._encode_scalar(object())
    with pytest.raises(ValueError, match="scalar tag"):
        storage._decode_scalar({"tag": "invented"})


def test_private_bundle_round_trips_complete_typed_state_and_datatrees(phase_pair, tmp_path):
    result = b4.integrate_conditional_posteriors(*phase_pair, parallel=True, n_jobs=2)
    bundle = tmp_path / "b4-result"

    written = storage.write_bayes_rarcat_result(result, bundle)
    loaded = storage.read_bayes_rarcat_result(bundle)

    assert written == bundle
    assert type(loaded) is b4.BayesRARCATResult
    assert type(loaded.partition_posterior) is type(result.partition_posterior)
    assert type(loaded.conditional_posterior) is type(result.conditional_posterior)
    assert loaded.design is loaded.conditional_posterior.design
    assert loaded.prior is loaded.conditional_posterior.prior
    assert loaded.execution_mode == "parallel"
    assert loaded.resolved_workers == 2
    _assert_scientific_results_equal(result, loaded)
    for name in (
        "weights",
        "replay_starts",
        "raw_medoid_levels",
        "raw_partitions",
        "aligned_partitions",
        "distinct_partitions",
        "counts",
        "masses",
        "inverse_index",
        "mass_mcse",
        "membership_probabilities",
        "entropy",
        "co_clustering",
    ):
        np.testing.assert_array_equal(
            getattr(loaded.partition_posterior, name),
            getattr(result.partition_posterior, name),
        )
    np.testing.assert_array_equal(loaded.design.design_matrix, result.design.design_matrix)
    np.testing.assert_array_equal(loaded.design.response_codes, result.design.response_codes)
    assert loaded.design.observed_model_frame.equals(result.design.observed_model_frame)
    assert loaded.design.factor_levels == result.design.factor_levels
    assert loaded.conditional_posterior.seed_manifest == result.conditional_posterior.seed_manifest
    _assert_tree_equal(
        result.conditional_posterior.prior_predictive.predictive,
        loaded.conditional_posterior.prior_predictive.predictive,
    )
    for original, restored in zip(
        result.conditional_posterior.components,
        loaded.conditional_posterior.components,
        strict=True,
    ):
        _assert_tree_equal(original.posterior, restored.posterior)
        _assert_tree_equal(original.posterior_predictive, restored.posterior_predictive)
        for original_effect, restored_effect in zip(
            original.effects, restored.effects, strict=True
        ):
            np.testing.assert_array_equal(
                original_effect.observation_effects, restored_effect.observation_effects
            )
            np.testing.assert_array_equal(original_effect.ame_draws, restored_effect.ame_draws)
    _assert_readonly_arrays(loaded)

    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["schema_name"] == b4.PRIVATE_SCHEMA_NAME
    assert manifest["schema_version"] == b4.PRIVATE_SCHEMA_VERSION
    paths = [entry["path"] for entry in manifest["payloads"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert len([path for path in paths if path.endswith(".nc")]) == 5
    assert not any(Path(path).suffix in {".pkl", ".pickle", ".joblib", ".npz"} for path in paths)
    for path in paths:
        if path.endswith(".npy"):
            array = np.load(bundle / path, allow_pickle=False)
            assert array.dtype != object
    assert [path.name for path in tmp_path.iterdir()] == [bundle.name]


def test_writer_refuses_blocked_nonfinite_missing_parent_existing_and_symlink_destinations(
    phase_pair, tmp_path
):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    failed_component = replace(
        phase_pair[1].components[0], status="fail", failure_reasons=("divergences",)
    )
    failed_conditional = replace(
        phase_pair[1],
        status="fail",
        components=(failed_component, phase_pair[1].components[1]),
        failures=(failed_component,),
    )
    blocked = b4.integrate_conditional_posteriors(phase_pair[0], failed_conditional)
    broken_support = replace(result.supports[0], values=result.supports[0].values.copy())
    broken_support.values[0] = np.nan
    nonfinite = replace(result, supports=(broken_support, *result.supports[1:]))

    with pytest.raises(ValueError, match="passing"):
        storage.write_bayes_rarcat_result(blocked, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    with pytest.raises(ValueError, match="coherent|finite"):
        storage.write_bayes_rarcat_result(nonfinite, tmp_path / "nonfinite")
    assert not (tmp_path / "nonfinite").exists()
    with pytest.raises(FileNotFoundError, match="parent"):
        storage.write_bayes_rarcat_result(result, tmp_path / "missing" / "bundle")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="destination"):
        storage.write_bayes_rarcat_result(result, existing)
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="destination"):
        storage.write_bayes_rarcat_result(result, symlink)


@pytest.mark.parametrize(
    "corruption", ["hash", "missing", "extra", "schema", "traversal", "symlink"]
)
def test_reader_rejects_manifest_and_payload_corruption(phase_pair, tmp_path, corruption):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    bundle = tmp_path / f"bundle-{corruption}"
    storage.write_bayes_rarcat_result(result, bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first_payload = bundle / manifest["payloads"][0]["path"]

    if corruption == "hash":
        first_payload.write_bytes(first_payload.read_bytes() + b"corrupt")
    elif corruption == "missing":
        first_payload.unlink()
    elif corruption == "extra":
        (bundle / "extra.npy").write_bytes(b"not allowed")
    elif corruption == "schema":
        manifest["schema_version"] += 1
        manifest_path.write_bytes(_canonical_json(manifest))
    elif corruption == "traversal":
        manifest["payloads"][0]["path"] = "../escape.npy"
        manifest_path.write_bytes(_canonical_json(manifest))
    elif corruption == "symlink":
        original = first_payload.read_bytes()
        first_payload.unlink()
        outside = tmp_path / "outside"
        outside.write_bytes(original)
        first_payload.symlink_to(outside)

    with pytest.raises(
        (ValueError, FileNotFoundError),
        match="schema|path|hash|size|missing|extra|symlink|inventory",
    ):
        storage.read_bayes_rarcat_result(bundle)


def test_reader_rejects_noncanonical_and_duplicate_key_json(phase_pair, tmp_path):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    bundle = tmp_path / "bundle-json"
    storage.write_bayes_rarcat_result(result, bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(ValueError, match="canonical"):
        storage.read_bayes_rarcat_result(bundle)

    manifest_path.write_text('{"schema_name":"x","schema_name":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        storage.read_bayes_rarcat_result(bundle)


@pytest.mark.parametrize("corruption", ["shape", "dtype", "truncated_tree", "metadata_json"])
def test_reader_rejects_rehashed_structural_payload_corruption(phase_pair, tmp_path, corruption):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    bundle = tmp_path / f"bundle-structural-{corruption}"
    storage.write_bayes_rarcat_result(result, bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    array_path = bundle / next(
        entry["path"] for entry in manifest["payloads"] if entry["path"].endswith(".npy")
    )
    tree_path = bundle / next(
        entry["path"] for entry in manifest["payloads"] if entry["path"].endswith(".nc")
    )

    if corruption in {"shape", "dtype"}:
        values = np.load(array_path, allow_pickle=False)
        changed = values.reshape(-1) if corruption == "shape" else values.astype(np.float32)
        with array_path.open("wb") as handle:
            np.save(handle, changed, allow_pickle=False)
    elif corruption == "truncated_tree":
        tree_path.write_bytes(tree_path.read_bytes()[:64])
    else:
        metadata_path = bundle / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata_path.write_text(json.dumps(metadata, indent=2))
    _rehash_manifest(bundle)

    with pytest.raises(ValueError, match="shape|dtype|DataTree|NetCDF|canonical"):
        storage.read_bayes_rarcat_result(bundle)


def test_reader_rejects_a_symlinked_bundle_root(phase_pair, tmp_path):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    bundle = tmp_path / "real-bundle"
    storage.write_bayes_rarcat_result(result, bundle)
    link = tmp_path / "bundle-link"
    link.symlink_to(bundle, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        storage.read_bayes_rarcat_result(link)


def test_writer_failure_is_atomic_and_preserves_its_temporary_directory_in_place(
    phase_pair, tmp_path, monkeypatch
):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    unrelated = tmp_path / ".bundle.tmp-unrelated"
    unrelated.mkdir()

    def fail_validation(path):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="forced"):
        storage.write_bayes_rarcat_result(result, destination)

    assert not destination.exists()
    assert unrelated.is_dir()
    preserved = [path for path in tmp_path.glob(".bundle.tmp-*") if path != unrelated]
    assert len(preserved) == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [unrelated.name, preserved[0].name]
    )


def test_noreplace_publication_moves_an_absent_directory_atomically(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "marker").write_text("complete", encoding="utf-8")
    destination = tmp_path / "destination"
    monkeypatch.setattr(storage, "_validate_written_bundle", lambda path: path)

    restored = storage._publish_directory_noreplace(temporary, destination)

    assert restored == destination
    assert not temporary.exists()
    assert (destination / "marker").read_text(encoding="utf-8") == "complete"


def test_noreplace_publication_refuses_empty_and_symlink_destinations(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_validate_written_bundle", lambda path: path)
    existing = tmp_path / "existing"
    existing.mkdir()
    temporary = tmp_path / "temporary-existing"
    temporary.mkdir()

    with pytest.raises(FileExistsError, match="destination"):
        storage._publish_directory_noreplace(temporary, existing)

    assert temporary.is_dir()
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    linked_temporary = tmp_path / "temporary-linked"
    linked_temporary.mkdir()
    with pytest.raises(FileExistsError, match="destination"):
        storage._publish_directory_noreplace(linked_temporary, linked)
    assert linked_temporary.is_dir()
    assert linked.is_symlink()


def test_noreplace_publication_refuses_destination_created_in_final_window(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    destination = tmp_path / "destination"
    real_rename = storage._rename_directory_noreplace

    def create_destination_then_rename(source, target):
        target.mkdir()
        real_rename(source, target)

    monkeypatch.setattr(storage, "_validate_written_bundle", lambda path: path)
    monkeypatch.setattr(storage, "_rename_directory_noreplace", create_destination_then_rename)

    with pytest.raises(FileExistsError, match="destination"):
        storage._publish_directory_noreplace(temporary, destination)

    assert temporary.is_dir()
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_two_synchronized_writers_cannot_replace_the_winning_directory(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "writer").write_text("first", encoding="utf-8")
    (second / "writer").write_text("second", encoding="utf-8")
    destination = tmp_path / "destination"
    barrier = Barrier(2)
    real_rename = storage._rename_directory_noreplace

    def synchronized_rename(source, target):
        barrier.wait(timeout=5)
        real_rename(source, target)

    monkeypatch.setattr(storage, "_validate_written_bundle", lambda path: path)
    monkeypatch.setattr(storage, "_rename_directory_noreplace", synchronized_rename)

    def publish(source):
        try:
            storage._publish_directory_noreplace(source, destination)
        except FileExistsError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (first, second)))

    assert sorted(outcomes) == ["lost", "won"]
    assert (destination / "writer").read_text(encoding="utf-8") in {"first", "second"}
    assert sum(path.exists() for path in (first, second)) == 1


def test_postpublication_validation_failure_preserves_owned_directory_in_place(
    tmp_path, monkeypatch
):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    destination = tmp_path / "destination"

    def fail_validation(path):
        raise ValueError("forced published validation failure")

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)

    with pytest.raises(ValueError, match="published validation"):
        storage._publish_directory_noreplace(temporary, destination)

    assert not temporary.exists()
    assert destination.is_dir()
    assert list(tmp_path.glob(".destination.failed-*")) == []


def test_postpublication_failure_preserves_original_exception_and_notes_path(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    destination = tmp_path / "destination"
    original_error = ValueError("forced published validation failure")

    def fail_validation(path):
        raise original_error

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)

    with pytest.raises(ValueError) as exc_info:
        storage._publish_directory_noreplace(temporary, destination)

    assert exc_info.value is original_error
    assert exc_info.value.__notes__ == [
        "Failure handling left the pathname untouched after observing the "
        f"writer artifact at {destination}; its identity may change concurrently"
    ]
    assert destination.is_dir()


def test_prepublication_write_failure_preserves_original_exception_and_notes_path(
    phase_pair, tmp_path, monkeypatch
):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    destination = tmp_path / "bundle"
    original_error = ValueError("forced temporary validation failure")

    def fail_validation(path):
        raise original_error

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)

    with pytest.raises(ValueError) as exc_info:
        storage.write_bayes_rarcat_result(result, destination)

    artifact = next(tmp_path.glob(".bundle.tmp-*"))
    assert exc_info.value is original_error
    assert exc_info.value.__notes__ == [
        "Failure handling left the pathname untouched after observing the "
        f"writer artifact at {artifact}; its identity may change concurrently"
    ]


def test_postpublication_failure_never_deletes_an_unrelated_replacement(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "writer").write_text("ours", encoding="utf-8")
    destination = tmp_path / "destination"
    moved_writer = tmp_path / "moved-writer"
    original_error = ValueError("forced validation after replacement")

    def replace_then_fail(path):
        path.rename(moved_writer)
        path.mkdir()
        (path / "owner").write_text("unrelated", encoding="utf-8")
        raise original_error

    monkeypatch.setattr(storage, "_validate_written_bundle", replace_then_fail)

    with pytest.raises(ValueError, match="after replacement") as exc_info:
        storage._publish_directory_noreplace(temporary, destination)

    assert exc_info.value is original_error
    assert not hasattr(exc_info.value, "__notes__")
    assert (destination / "owner").read_text(encoding="utf-8") == "unrelated"
    assert (moved_writer / "writer").read_text(encoding="utf-8") == "ours"


def test_failure_artifact_check_cannot_mask_the_original_exception(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    destination = tmp_path / "destination"
    original_error = ValueError("forced published validation failure")

    def fail_validation(path):
        raise original_error

    def fail_artifact_check(source, *, expected_identity):
        raise RuntimeError("forced artifact-check failure")

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    monkeypatch.setattr(storage, "_failure_artifact_if_owned", fail_artifact_check)

    with pytest.raises(ValueError) as exc_info:
        storage._publish_directory_noreplace(temporary, destination)

    assert exc_info.value is original_error
    assert exc_info.value.__notes__ == [
        "Could not confirm the in-place failure artifact associated with "
        f"{destination}: RuntimeError: forced artifact-check failure"
    ]


def test_failure_disposition_never_relocates_an_unrelated_replacement(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "payload").write_text("writer", encoding="utf-8")
    destination = tmp_path / "destination"
    moved_writer = tmp_path / "moved-writer"
    original_error = ValueError("forced published validation failure")
    real_identity = storage._entry_identity
    real_rename = storage._rename_directory_noreplace
    replacement_injected = False
    cleanup_rename_calls = 0

    def fail_validation(path):
        raise original_error

    def replace_after_identity_read(path):
        nonlocal replacement_injected
        identity = real_identity(path)
        if path == destination and not replacement_injected:
            path.rename(moved_writer)
            path.mkdir()
            (path / "owner").write_text("unrelated", encoding="utf-8")
            replacement_injected = True
        return identity

    def reject_cleanup_rename(source, target):
        nonlocal cleanup_rename_calls
        cleanup_rename_calls += 1
        if cleanup_rename_calls > 1:
            raise AssertionError("failure disposition must not rename a public pathname")
        real_rename(source, target)

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    monkeypatch.setattr(storage, "_entry_identity", replace_after_identity_read)
    monkeypatch.setattr(storage, "_rename_directory_noreplace", reject_cleanup_rename)

    with pytest.raises(ValueError) as exc_info:
        storage._publish_directory_noreplace(temporary, destination)

    assert exc_info.value is original_error
    assert exc_info.value.__notes__ == [
        "Failure handling left the pathname untouched after observing the "
        f"writer artifact at {destination}; its identity may change concurrently"
    ]
    assert cleanup_rename_calls == 1
    assert (moved_writer / "payload").read_text(encoding="utf-8") == "writer"
    assert (destination / "owner").read_text(encoding="utf-8") == "unrelated"
    assert list(tmp_path.glob(".destination.failed-*")) == []


def test_noreplace_publication_fails_closed_when_kernel_primitive_is_unavailable(
    tmp_path, monkeypatch
):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    destination = tmp_path / "destination"
    monkeypatch.setattr(storage, "_get_renameat2", lambda: None)

    with pytest.raises(NotImplementedError, match="Linux.*renameat2|renameat2.*unavailable"):
        storage._rename_directory_noreplace(temporary, destination)

    assert temporary.is_dir()
    assert not destination.exists()


@pytest.mark.parametrize("error_number", [errno.ENOSYS, errno.EOPNOTSUPP])
def test_noreplace_maps_only_explicit_unsupported_errnos_to_not_implemented(
    tmp_path, monkeypatch, error_number
):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"

    def unsupported_rename(*args):
        storage.ctypes.set_errno(error_number)
        return -1

    monkeypatch.setattr(storage, "_get_renameat2", lambda: unsupported_rename)

    with pytest.raises(NotImplementedError, match="kernel or filesystem"):
        storage._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_noreplace_preserves_einval_for_invalid_descendant_topology(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    child = source / "child"
    child.mkdir()
    moved = child / "moved"

    with pytest.raises(OSError) as exc_info:
        storage._rename_directory_noreplace(source, moved)

    assert exc_info.value.errno == errno.EINVAL
    assert source.is_dir()
    assert child.is_dir()
    assert not moved.exists()


def test_failure_disposition_never_invokes_recursive_deletion(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "writer").write_text("ours", encoding="utf-8")
    destination = tmp_path / "destination"

    def fail_validation(path):
        raise ValueError("forced published validation failure")

    def forbidden_delete(*args, **kwargs):
        raise AssertionError("failure artifacts must not be deleted by pathname")

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    monkeypatch.setattr(storage.os, "unlink", forbidden_delete)
    monkeypatch.setattr(storage.os, "rmdir", forbidden_delete)

    with pytest.raises(ValueError, match="published validation"):
        storage._publish_directory_noreplace(temporary, destination)

    assert (destination / "writer").read_text(encoding="utf-8") == "ours"
    assert list(tmp_path.glob(".destination.failed-*")) == []


def test_failure_disposition_performs_no_second_rename(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "writer").write_text("ours", encoding="utf-8")
    destination = tmp_path / "destination"
    moved_writer = tmp_path / "moved-writer"
    real_rename = storage._rename_directory_noreplace
    rename_calls = 0

    def fail_validation(path):
        raise ValueError("forced published validation failure")

    def count_publication_rename(source, target):
        nonlocal rename_calls
        rename_calls += 1
        if rename_calls == 2:
            source.rename(moved_writer)
            source.mkdir()
            (source / "owner").write_text("unrelated", encoding="utf-8")
        real_rename(source, target)

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    monkeypatch.setattr(storage, "_rename_directory_noreplace", count_publication_rename)

    with pytest.raises(ValueError, match="published validation"):
        storage._publish_directory_noreplace(temporary, destination)

    assert rename_calls == 1
    assert (destination / "writer").read_text(encoding="utf-8") == "ours"
    assert not moved_writer.exists()
    assert list(tmp_path.glob(".destination.failed-*")) == []


def test_writer_outer_cleanup_leaves_recreated_temporary_path_untouched(
    phase_pair, tmp_path, monkeypatch
):
    result = b4.integrate_conditional_posteriors(*phase_pair)
    destination = tmp_path / "bundle"
    temporary = tmp_path / ".bundle.tmp-controlled"
    original_validate = storage._validate_written_bundle
    original_error = ValueError("forced failure after publication")

    def controlled_mkdtemp(*, prefix, dir):
        assert prefix == ".bundle.tmp-"
        assert Path(dir) == tmp_path
        temporary.mkdir()
        return str(temporary)

    def recreate_temporary_then_fail(path):
        if path == destination:
            temporary.mkdir()
            (temporary / "owner").write_text("unrelated", encoding="utf-8")
            raise original_error
        return original_validate(path)

    monkeypatch.setattr(storage.tempfile, "mkdtemp", controlled_mkdtemp)
    monkeypatch.setattr(storage, "_validate_written_bundle", recreate_temporary_then_fail)

    with pytest.raises(ValueError, match="after publication") as exc_info:
        storage.write_bayes_rarcat_result(result, destination)

    assert exc_info.value is original_error
    assert exc_info.value.__notes__ == [
        "Failure handling left the pathname untouched after observing the "
        f"writer artifact at {destination}; its identity may change concurrently"
    ]
    assert destination.is_dir()
    assert (temporary / "owner").read_text(encoding="utf-8") == "unrelated"
    assert list(tmp_path.glob(".bundle.failed-*")) == []


def test_final_root_rmdir_cannot_delete_a_rebound_unrelated_directory(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    writer_identity = storage._entry_identity(temporary)
    destination = tmp_path / "destination"
    moved_writer = tmp_path / "moved-writer"
    original_rmdir = storage.os.rmdir
    injected_identity: tuple[int, int] | None = None

    def fail_validation(path):
        raise ValueError("forced published validation failure")

    def replace_at_final_rmdir(path, *, dir_fd=None):
        nonlocal injected_identity
        candidate = Path(path)
        if dir_fd is None and candidate.name.startswith(".destination.failed-"):
            candidate.rename(moved_writer)
            candidate.mkdir()
            status = candidate.stat()
            injected_identity = (status.st_dev, status.st_ino)
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    monkeypatch.setattr(storage.os, "rmdir", replace_at_final_rmdir)

    with pytest.raises(ValueError, match="published validation"):
        storage._publish_directory_noreplace(temporary, destination)

    assert injected_identity is None
    assert storage._entry_identity(destination) == writer_identity
    assert list(tmp_path.glob(".destination.failed-*")) == []


def test_final_child_unlink_cannot_delete_a_rebound_unrelated_file(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "payload").write_text("ours", encoding="utf-8")
    destination = tmp_path / "destination"
    moved_writer_file = tmp_path / "moved-writer-file"
    original_unlink = storage.os.unlink
    injected = False

    def fail_validation(path):
        raise ValueError("forced published validation failure")

    def replace_at_final_unlink(path, *, dir_fd=None):
        nonlocal injected
        if dir_fd is not None and path == "payload":
            root = Path(f"/proc/self/fd/{dir_fd}").resolve()
            (root / path).rename(moved_writer_file)
            (root / path).write_text("unrelated", encoding="utf-8")
            injected = True
        return original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    monkeypatch.setattr(storage.os, "unlink", replace_at_final_unlink)

    with pytest.raises(ValueError, match="published validation"):
        storage._publish_directory_noreplace(temporary, destination)

    assert not injected
    assert (destination / "payload").read_text(encoding="utf-8") == "ours"
    assert list(tmp_path.glob(".destination.failed-*")) == []


def test_final_child_rmdir_cannot_delete_a_rebound_unrelated_directory(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "payload-dir").mkdir()
    destination = tmp_path / "destination"
    moved_writer_directory = tmp_path / "moved-writer-directory"
    original_rmdir = storage.os.rmdir
    injected_identity: tuple[int, int] | None = None

    def fail_validation(path):
        raise ValueError("forced published validation failure")

    def replace_at_final_rmdir(path, *, dir_fd=None):
        nonlocal injected_identity
        if dir_fd is not None and path == "payload-dir":
            root = Path(f"/proc/self/fd/{dir_fd}").resolve()
            (root / path).rename(moved_writer_directory)
            (root / path).mkdir()
            status = (root / path).stat()
            injected_identity = (status.st_dev, status.st_ino)
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(storage, "_validate_written_bundle", fail_validation)
    monkeypatch.setattr(storage.os, "rmdir", replace_at_final_rmdir)

    with pytest.raises(ValueError, match="published validation"):
        storage._publish_directory_noreplace(temporary, destination)

    assert injected_identity is None
    assert (destination / "payload-dir").is_dir()
    assert list(tmp_path.glob(".destination.failed-*")) == []

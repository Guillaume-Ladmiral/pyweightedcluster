"""Private versioned persistence for the Bayesian RARCAT B4 result.

The bundle is an internal integrity-checked format, not a public compatibility
promise or an authenticity mechanism.  It intentionally uses canonical JSON,
non-object NumPy payloads, and one NetCDF file per retained DataTree.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from functools import cache
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from xarray import DataTree

from . import _bayesian_rarcat_conditional as _b3
from . import _bayesian_rarcat_mixture as _b4
from . import _bayesian_rarcat_partition as _b2

_METADATA_PATH = "metadata.json"
_MANIFEST_PATH = "manifest.json"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


@cache
def _get_renameat2() -> Callable[[int, bytes, int, bytes, int], int] | None:
    """Return the process libc ``renameat2`` binding on supported Linux hosts."""
    if sys.platform != "linux":
        return None
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError:
        return None
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    return renameat2


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing an existing entry.

    This guarantee currently requires Linux ``renameat2`` with
    ``RENAME_NOREPLACE`` support. A missing primitive or explicit unsupported
    errno maps to ``NotImplementedError``; other syscall errnos remain
    ``OSError``. Every failure is fail-closed.
    """
    renameat2 = _get_renameat2()
    if renameat2 is None:
        raise NotImplementedError(
            "Linux renameat2 with RENAME_NOREPLACE is unavailable; "
            "no portable atomic no-replace directory publication is claimed"
        )
    ctypes.set_errno(0)
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "bundle destination already exists",
            destination,
        )
    unsupported_errors = {errno.ENOSYS}
    if hasattr(errno, "EOPNOTSUPP"):
        unsupported_errors.add(errno.EOPNOTSUPP)
    if error_number in unsupported_errors:
        raise NotImplementedError(
            "renameat2 RENAME_NOREPLACE is unavailable on this Linux kernel or filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _encode_scalar(value: object) -> dict[str, object]:
    """Encode one supported scalar without identity or dtype collisions."""
    if value is None:
        return {"tag": "none"}
    if type(value) is bool:
        return {"tag": "python_bool", "value": value}
    if type(value) is int:
        return {"tag": "python_int", "value": str(value)}
    if type(value) is float:
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "positive_infinity" if value > 0.0 else "negative_infinity"
        else:
            encoded = value.hex()
        return {"tag": "python_float", "value": encoded}
    if type(value) is str:
        return {"tag": "python_str", "value": value}
    if isinstance(value, np.generic) and value.dtype.kind in "biufU":
        return {
            "tag": "numpy_scalar",
            "dtype": value.dtype.str,
            "bytes": value.tobytes().hex(),
        }
    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


def _decode_scalar(encoded: object) -> object:
    if type(encoded) is not dict or "tag" not in encoded:
        raise ValueError("scalar tag object is malformed")
    tag = encoded["tag"]
    if tag == "none" and set(encoded) == {"tag"}:
        return None
    if tag == "python_bool" and set(encoded) == {"tag", "value"}:
        if type(encoded["value"]) is not bool:
            raise ValueError("python_bool scalar tag is malformed")
        return encoded["value"]
    if tag == "python_int" and set(encoded) == {"tag", "value"}:
        text = encoded["value"]
        if type(text) is not str or not text or text.strip() != text:
            raise ValueError("python_int scalar tag is malformed")
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError("python_int scalar tag is malformed") from error
        if str(value) != text:
            raise ValueError("python_int scalar tag is noncanonical")
        return value
    if tag == "python_float" and set(encoded) == {"tag", "value"}:
        text = encoded["value"]
        if text == "nan":
            return float("nan")
        if text == "positive_infinity":
            return float("inf")
        if text == "negative_infinity":
            return float("-inf")
        if type(text) is not str:
            raise ValueError("python_float scalar tag is malformed")
        try:
            value = float.fromhex(text)
        except ValueError as error:
            raise ValueError("python_float scalar tag is malformed") from error
        if not np.isfinite(value) or value.hex() != text:
            raise ValueError("python_float scalar tag is noncanonical")
        return value
    if tag == "python_str" and set(encoded) == {"tag", "value"}:
        if type(encoded["value"]) is not str:
            raise ValueError("python_str scalar tag is malformed")
        return encoded["value"]
    if tag == "numpy_scalar" and set(encoded) == {"tag", "dtype", "bytes"}:
        try:
            dtype = np.dtype(encoded["dtype"])
            raw = bytes.fromhex(encoded["bytes"])
        except (TypeError, ValueError) as error:
            raise ValueError("numpy scalar tag is malformed") from error
        if dtype.kind not in "biufU" or dtype.hasobject or len(raw) != dtype.itemsize:
            raise ValueError("numpy scalar tag uses an unsupported dtype")
        return np.frombuffer(raw, dtype=dtype, count=1)[0]
    raise ValueError(f"unsupported scalar tag: {tag!r}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_canonical_json(path: Path) -> object:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed JSON payload: {path.name}") from error
    if _canonical_json_bytes(value) != raw:
        raise ValueError(f"JSON payload is not canonical: {path.name}")
    return value


_DATACLASS_TYPES = {
    cls.__name__: cls
    for cls in (
        _b2.SeedStreamIdentifier,
        _b2.SeedStreams,
        _b2.AlignmentResult,
        _b2.PartitionPosteriorResult,
        _b3.ScalingTransform,
        _b3.PreparedConditionalDesign,
        _b3.PriorSpecification,
        _b3.EffectMetadata,
        _b3.DrawwiseEffect,
        _b3.SamplerConfiguration,
        _b3.B3SeedIdentifier,
        _b3.ComponentSeedManifest,
        _b3.B3SeedManifest,
        _b3.DiagnosticThresholds,
        _b3.SamplerDiagnosticRecord,
        _b3.GrowthDiagnostic,
        _b3.WarningRecord,
        _b3.PredictiveCoordinates,
        _b3.PosteriorPredictiveRecord,
        _b3.PriorPredictiveRecord,
        _b3.TargetSummary,
        _b3.ComponentFitRecord,
        _b3.ConditionalPosteriorResult,
        _b4.EffectTargetMetadata,
        _b4.TargetDescriptor,
        _b4.WeightedTargetSupport,
        _b4.IntegratedTargetSummary,
        _b4.AllocationRecord,
        _b4.BayesRARCATResult,
    )
}


def _tree_signature(tree: DataTree) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for group in tree.groups:
        dataset = tree[group].dataset
        variables = [
            {
                "name": name,
                "dims": list(dataset[name].dims),
                "shape": list(dataset[name].shape),
                "dtype": dataset[name].dtype.str,
            }
            for name in dataset.variables
        ]
        groups.append({"group": group, "variables": variables})
    return {"groups": groups}


class _WriteContext:
    def __init__(self, root: Path):
        self.root = root
        self.array_count = 0
        self.tree_count = 0
        self.object_count = 0
        self.seen: dict[int, str] = {}
        self.owned_objects: list[object] = []
        (root / "arrays").mkdir()
        (root / "trees").mkdir()

    def _object_id(self, value: object) -> tuple[str, bool]:
        existing = self.seen.get(id(value))
        if existing is not None:
            return existing, True
        object_id = f"o{self.object_count:06d}"
        self.object_count += 1
        self.seen[id(value)] = object_id
        self.owned_objects.append(value)
        return object_id, False

    def _encode_array(self, value: np.ndarray) -> dict[str, object]:
        object_id, repeated = self._object_id(value)
        if repeated:
            return {"kind": "ref", "object_id": object_id}
        if value.dtype.hasobject:
            return {
                "kind": "identity_array",
                "object_id": object_id,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "values": [_encode_scalar(item) for item in value.ravel(order="C")],
            }
        if value.dtype.kind not in "biufSU":
            raise TypeError(f"unsupported NumPy array dtype: {value.dtype}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError("scientific NumPy arrays must be finite")
        relative = f"arrays/{self.array_count:06d}.npy"
        self.array_count += 1
        with (self.root / relative).open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        return {
            "kind": "numeric_array",
            "object_id": object_id,
            "path": relative,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }

    def _encode_axis(self, axis: pd.Index) -> dict[str, object]:
        return {
            "values": [_encode_scalar(value) for value in axis.tolist()],
            "dtype": str(axis.dtype),
            "name": _encode_scalar(axis.name),
        }

    def _encode_frame(self, frame: pd.DataFrame) -> dict[str, object]:
        object_id, repeated = self._object_id(frame)
        if repeated:
            return {"kind": "ref", "object_id": object_id}
        series: list[dict[str, object]] = []
        for position in range(frame.shape[1]):
            column = frame.iloc[:, position]
            if isinstance(column.dtype, pd.CategoricalDtype):
                series.append(
                    {
                        "kind": "categorical",
                        "categories": [
                            _encode_scalar(value) for value in column.cat.categories.tolist()
                        ],
                        "categories_dtype": str(column.cat.categories.dtype),
                        "ordered": bool(column.cat.ordered),
                        "codes": self._encode_array(
                            column.cat.codes.to_numpy(dtype=np.int64, copy=True)
                        ),
                    }
                )
            else:
                values = column.to_numpy(copy=True)
                series.append(
                    {
                        "kind": "primitive",
                        "dtype": str(column.dtype),
                        "values": self._encode_array(values),
                    }
                )
        return {
            "kind": "dataframe",
            "object_id": object_id,
            "index": self._encode_axis(frame.index),
            "columns": self._encode_axis(frame.columns),
            "series": series,
        }

    def _encode_tree(self, tree: DataTree) -> dict[str, object]:
        object_id, repeated = self._object_id(tree)
        if repeated:
            return {"kind": "ref", "object_id": object_id}
        relative = f"trees/{self.tree_count:06d}.nc"
        self.tree_count += 1
        tree.to_netcdf(self.root / relative)
        return {
            "kind": "datatree",
            "object_id": object_id,
            "path": relative,
            "signature": _tree_signature(tree),
        }

    def encode(self, value: object) -> object:
        if value is None or type(value) in {bool, int, float, str} or isinstance(value, np.generic):
            return {"kind": "scalar", "value": _encode_scalar(value)}
        if type(value) is np.ndarray:
            return self._encode_array(value)
        if type(value) is tuple:
            return {"kind": "tuple", "items": [self.encode(item) for item in value]}
        if type(value) is list:
            return {"kind": "list", "items": [self.encode(item) for item in value]}
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("serialized mappings must have string keys")
            return {
                "kind": "mapping",
                "items": [[key, self.encode(item)] for key, item in value.items()],
            }
        if type(value) is pd.DataFrame:
            return self._encode_frame(value)
        if type(value) is DataTree:
            return self._encode_tree(value)
        if is_dataclass(value):
            dataclass_type = type(value)
            if _DATACLASS_TYPES.get(dataclass_type.__name__) is not dataclass_type:
                raise TypeError(f"unsupported dataclass type: {dataclass_type.__name__}")
            object_id, repeated = self._object_id(value)
            if repeated:
                return {"kind": "ref", "object_id": object_id}
            encoded_fields = []
            for field in fields(value):
                if dataclass_type is _b3.PreparedConditionalDesign and field.name == "design_info":
                    encoded = {"kind": "design_info_omitted"}
                else:
                    encoded = self.encode(getattr(value, field.name))
                encoded_fields.append([field.name, encoded])
            return {
                "kind": "dataclass",
                "object_id": object_id,
                "type": dataclass_type.__name__,
                "fields": encoded_fields,
            }
        raise TypeError(f"unsupported serialized type: {type(value).__name__}")


def _validated_shape(value: object) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("serialized array shape is malformed")
    return tuple(value)


def _materialize_tree(tree: DataTree) -> DataTree:
    tree.load()
    tree.close()
    for group in tree.groups:
        dataset = tree[group].dataset
        for variable in dataset.variables.values():
            values = variable.values
            if isinstance(values, np.ndarray):
                try:
                    values.setflags(write=False)
                except ValueError:
                    pass
    return tree


class _ReadContext:
    def __init__(self, root: Path, *, allowed_paths: set[str]):
        self.root = root
        self.objects: dict[str, object] = {}
        self.allowed_paths = allowed_paths
        self.used_paths: set[str] = set()

    def _payload_path(self, value: object, kind: str) -> Path:
        relative = _validate_relative_payload(value, kind)
        if relative not in self.allowed_paths:
            raise ValueError("metadata references a payload absent from the inventory")
        if relative in self.used_paths:
            raise ValueError("metadata references a payload more than once")
        self.used_paths.add(relative)
        return self.root / relative

    def _register(self, object_id: object, value: object) -> object:
        if type(object_id) is not str or not object_id.startswith("o"):
            raise ValueError("serialized object identifier is malformed")
        if object_id in self.objects:
            raise ValueError("duplicate serialized object identifier")
        self.objects[object_id] = value
        return value

    def _decode_axis(self, encoded: object) -> pd.Index:
        if type(encoded) is not dict or set(encoded) != {"values", "dtype", "name"}:
            raise ValueError("serialized pandas axis is malformed")
        values = [_decode_scalar(item) for item in encoded["values"]]
        dtype = encoded["dtype"]
        if type(dtype) is not str:
            raise ValueError("serialized pandas axis dtype is malformed")
        try:
            array = np.asarray(values, dtype=np.dtype(dtype))
            axis = pd.Index(array, name=_decode_scalar(encoded["name"]))
        except (TypeError, ValueError) as error:
            raise ValueError("serialized pandas axis cannot be reconstructed") from error
        if str(axis.dtype) != dtype:
            raise ValueError("serialized pandas axis dtype changed")
        return axis

    def _decode_frame(self, encoded: dict[str, object]) -> pd.DataFrame:
        required = {"kind", "object_id", "index", "columns", "series"}
        if set(encoded) != required or type(encoded["series"]) is not list:
            raise ValueError("serialized DataFrame is malformed")
        index = self._decode_axis(encoded["index"])
        columns = self._decode_axis(encoded["columns"])
        if len(encoded["series"]) != len(columns):
            raise ValueError("serialized DataFrame column count is inconsistent")
        decoded_series: list[pd.Series] = []
        for item in encoded["series"]:
            if type(item) is not dict or "kind" not in item:
                raise ValueError("serialized DataFrame series is malformed")
            if item["kind"] == "categorical":
                if set(item) != {
                    "kind",
                    "categories",
                    "categories_dtype",
                    "ordered",
                    "codes",
                }:
                    raise ValueError("serialized categorical series is malformed")
                categories = [_decode_scalar(value) for value in item["categories"]]
                categories_array = np.asarray(categories, dtype=np.dtype(item["categories_dtype"]))
                codes = self.decode(item["codes"])
                if type(codes) is not np.ndarray or codes.dtype != np.dtype("int64"):
                    raise ValueError("serialized categorical codes are malformed")
                values = pd.Categorical.from_codes(
                    codes,
                    categories=categories_array,
                    ordered=item["ordered"],
                )
                decoded_series.append(pd.Series(values, index=index))
            elif item["kind"] == "primitive":
                if set(item) != {"kind", "dtype", "values"}:
                    raise ValueError("serialized primitive series is malformed")
                values = self.decode(item["values"])
                if type(values) is not np.ndarray:
                    raise ValueError("serialized primitive series values are malformed")
                try:
                    series = pd.Series(values, index=index, dtype=item["dtype"])
                except (TypeError, ValueError) as error:
                    raise ValueError("serialized primitive dtype cannot be restored") from error
                if str(series.dtype) != item["dtype"]:
                    raise ValueError("serialized primitive dtype changed")
                decoded_series.append(series)
            else:
                raise ValueError("serialized DataFrame series kind is unsupported")
        frame = pd.concat(decoded_series, axis=1) if decoded_series else pd.DataFrame(index=index)
        frame.columns = columns
        frame.index = index
        return self._register(encoded["object_id"], frame)

    def _decode_array(self, encoded: dict[str, object]) -> np.ndarray:
        shape = _validated_shape(encoded.get("shape"))
        dtype_text = encoded.get("dtype")
        if type(dtype_text) is not str:
            raise ValueError("serialized array dtype is malformed")
        try:
            dtype = np.dtype(dtype_text)
        except TypeError as error:
            raise ValueError("serialized array dtype is unsupported") from error
        if encoded["kind"] == "identity_array":
            if set(encoded) != {
                "kind",
                "object_id",
                "dtype",
                "shape",
                "values",
            } or dtype != np.dtype(object):
                raise ValueError("serialized identity array is malformed")
            values = [_decode_scalar(item) for item in encoded["values"]]
            if len(values) != int(np.prod(shape, dtype=np.int64)):
                raise ValueError("serialized identity array length is inconsistent")
            array = np.asarray(values, dtype=object).reshape(shape)
        elif encoded["kind"] == "numeric_array":
            if set(encoded) != {
                "kind",
                "object_id",
                "path",
                "dtype",
                "shape",
            }:
                raise ValueError("serialized numeric array is malformed")
            try:
                array = np.load(self._payload_path(encoded["path"], "array"), allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError("numeric array payload cannot be loaded safely") from error
            if array.dtype.hasobject:
                raise ValueError("object-dtype NumPy payloads are forbidden")
            if array.dtype != dtype or array.shape != shape:
                raise ValueError("numeric array payload dtype or shape is inconsistent")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError("scientific NumPy arrays must be finite")
            array = np.array(array, copy=True)
        else:
            raise ValueError("serialized array kind is unsupported")
        array.setflags(write=False)
        return self._register(encoded["object_id"], array)

    def _decode_tree(self, encoded: dict[str, object]) -> DataTree:
        if set(encoded) != {"kind", "object_id", "path", "signature"}:
            raise ValueError("serialized DataTree is malformed")
        try:
            import arviz as az

            tree = _materialize_tree(
                az.from_netcdf(self._payload_path(encoded["path"], "datatree"))
            )
        except Exception as error:
            raise ValueError("DataTree NetCDF payload cannot be loaded") from error
        if _tree_signature(tree) != encoded["signature"]:
            raise ValueError("DataTree groups, variables, dimensions, or dtypes changed")
        return self._register(encoded["object_id"], tree)

    def decode(self, encoded: object) -> object:
        if type(encoded) is not dict or type(encoded.get("kind")) is not str:
            raise ValueError("serialized metadata node is malformed")
        kind = encoded["kind"]
        if kind == "scalar" and set(encoded) == {"kind", "value"}:
            return _decode_scalar(encoded["value"])
        if kind == "ref" and set(encoded) == {"kind", "object_id"}:
            try:
                return self.objects[encoded["object_id"]]
            except (KeyError, TypeError) as error:
                raise ValueError("serialized reference is unknown or forward") from error
        if kind in {"identity_array", "numeric_array"}:
            return self._decode_array(encoded)
        if kind in {"tuple", "list"} and set(encoded) == {"kind", "items"}:
            if type(encoded["items"]) is not list:
                raise ValueError("serialized sequence is malformed")
            values = [self.decode(item) for item in encoded["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "mapping" and set(encoded) == {"kind", "items"}:
            result: dict[str, object] = {}
            for item in encoded["items"]:
                if type(item) is not list or len(item) != 2 or type(item[0]) is not str:
                    raise ValueError("serialized mapping item is malformed")
                if item[0] in result:
                    raise ValueError("serialized mapping key is duplicated")
                result[item[0]] = self.decode(item[1])
            return result
        if kind == "dataframe":
            return self._decode_frame(encoded)
        if kind == "datatree":
            return self._decode_tree(encoded)
        if kind == "design_info_omitted" and set(encoded) == {"kind"}:
            return None
        if kind == "dataclass":
            if set(encoded) != {"kind", "object_id", "type", "fields"}:
                raise ValueError("serialized dataclass is malformed")
            dataclass_type = _DATACLASS_TYPES.get(encoded["type"])
            if dataclass_type is None or type(encoded["fields"]) is not list:
                raise ValueError("serialized dataclass type is unsupported")
            expected_names = [field.name for field in fields(dataclass_type)]
            if any(type(item) is not list or len(item) != 2 for item in encoded["fields"]):
                raise ValueError("serialized dataclass fields are inconsistent")
            actual_names = [item[0] for item in encoded["fields"]]
            if actual_names != expected_names:
                raise ValueError("serialized dataclass fields are inconsistent")
            values = {name: self.decode(item) for name, item in encoded["fields"]}
            try:
                value = dataclass_type(**values)
            except (TypeError, ValueError) as error:
                raise ValueError("serialized dataclass cannot be reconstructed") from error
            return self._register(encoded["object_id"], value)
        raise ValueError(f"unsupported serialized metadata kind: {kind!r}")


def _compare_designs(
    stored: _b3.PreparedConditionalDesign, rebuilt: _b3.PreparedConditionalDesign
) -> None:
    scalar_fields = (
        "formula",
        "response_name",
        "translated_rhs",
        "row_ids",
        "cluster_order",
        "reference_category",
        "base_predictors",
        "continuous_predictors",
        "categorical_predictors",
        "nonlinear_continuous_predictors",
        "counterfactual_steps",
        "factor_levels",
        "factor_references",
        "term_labels",
        "engine_design_labels",
        "coefficient_coordinates",
        "engine_to_coefficient",
        "design_rank",
        "ordered_positions",
    )
    for name in scalar_fields:
        if getattr(stored, name) != getattr(rebuilt, name):
            raise ValueError(f"rebuilt B3 design field changed: {name}")
    for name in ("design_matrix", "response_codes"):
        left = getattr(stored, name)
        right = getattr(rebuilt, name)
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise ValueError(f"rebuilt B3 design array changed: {name}")
    if len(stored.scaling_transforms) != len(rebuilt.scaling_transforms):
        raise ValueError("rebuilt B3 scaling transforms changed")
    for left, right in zip(stored.scaling_transforms, rebuilt.scaling_transforms, strict=True):
        if (
            left.predictor != right.predictor
            or left.mean != right.mean
            or left.population_sd != right.population_sd
            or left.raw_values.dtype != right.raw_values.dtype
            or not np.array_equal(left.raw_values, right.raw_values)
        ):
            raise ValueError("rebuilt B3 scaling transform changed")
    try:
        pd.testing.assert_frame_equal(
            stored.observed_model_frame,
            rebuilt.observed_model_frame,
            check_exact=True,
            check_categorical=True,
        )
        pd.testing.assert_frame_equal(
            stored.scaled_design_frame,
            rebuilt.scaled_design_frame,
            check_exact=True,
            check_categorical=True,
        )
    except AssertionError as error:
        raise ValueError("rebuilt B3 design frames changed") from error


def _rebuild_result_design(result: _b4.BayesRARCATResult) -> _b4.BayesRARCATResult:
    stored = result.conditional_posterior.design
    rebuilt = _b3.prepare_conditional_design(
        partition_posterior=result.partition_posterior,
        formula=result.conditional_posterior.formula,
        data=stored.observed_model_frame,
        continuous_predictors=stored.continuous_predictors,
        counterfactual_steps=dict(stored.counterfactual_steps),
    )
    _compare_designs(stored, rebuilt)
    conditional = replace(result.conditional_posterior, design=rebuilt)
    return replace(
        result,
        conditional_posterior=conditional,
        design=rebuilt,
        prior=conditional.prior,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_kind(path: str) -> str:
    if path == _METADATA_PATH:
        return "metadata"
    suffix = PurePosixPath(path).suffix
    if suffix == ".npy":
        return "array"
    if suffix == ".nc":
        return "datatree"
    raise ValueError(f"unexpected payload extension: {path}")


def _inventory(root: Path) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.name != _MANIFEST_PATH
    ):
        relative = path.relative_to(root).as_posix()
        payloads.append(
            {
                "kind": _payload_kind(relative),
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return payloads


def _manifest(result: _b4.BayesRARCATResult, root: Path) -> dict[str, object]:
    try:
        package_version = importlib_metadata.version("pyweightedcluster")
    except importlib_metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "schema_name": _b4.PRIVATE_SCHEMA_NAME,
        "schema_version": _b4.PRIVATE_SCHEMA_VERSION,
        "package_version": package_version,
        "software_versions": [list(item) for item in result.software_versions],
        "payloads": _inventory(root),
    }


def _validate_relative_payload(path: object, kind: object) -> str:
    if type(path) is not str or type(kind) is not str or not path:
        raise ValueError("manifest payload path/kind is malformed")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("manifest payload path is absolute or traverses the bundle")
    expected_kind = _payload_kind(path)
    if expected_kind != kind:
        raise ValueError("manifest payload extension does not match its kind")
    return path


def _validated_manifest(bundle: Path) -> dict[str, object]:
    manifest_path = bundle / _MANIFEST_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("bundle manifest is missing or a symlink")
    manifest = _read_canonical_json(manifest_path)
    if type(manifest) is not dict or set(manifest) != {
        "schema_name",
        "schema_version",
        "package_version",
        "software_versions",
        "payloads",
    }:
        raise ValueError("bundle manifest has an invalid structure")
    if (
        manifest["schema_name"] != _b4.PRIVATE_SCHEMA_NAME
        or manifest["schema_version"] != _b4.PRIVATE_SCHEMA_VERSION
    ):
        raise ValueError("unknown private bundle schema name or version")
    if type(manifest["package_version"]) is not str:
        raise ValueError("manifest package version is malformed")
    software_versions = manifest["software_versions"]
    if type(software_versions) is not list or any(
        type(item) is not list or len(item) != 2 or any(type(value) is not str for value in item)
        for item in software_versions
    ):
        raise ValueError("manifest software versions are malformed")
    payloads = manifest["payloads"]
    if type(payloads) is not list:
        raise ValueError("manifest payload inventory is malformed")
    paths: list[str] = []
    for entry in payloads:
        if type(entry) is not dict or set(entry) != {"kind", "path", "size", "sha256"}:
            raise ValueError("manifest payload inventory entry is malformed")
        path = _validate_relative_payload(entry["path"], entry["kind"])
        if (
            type(entry["size"]) is not int
            or entry["size"] < 0
            or type(entry["sha256"]) is not str
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
        ):
            raise ValueError("manifest payload size or hash is malformed")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("manifest payload paths are duplicated or not canonical")
    if paths.count(_METADATA_PATH) != 1:
        raise ValueError("manifest inventory must contain metadata.json exactly once")

    for item in bundle.rglob("*"):
        if item.is_symlink():
            raise ValueError("bundle payload symlinks are forbidden")
    actual = sorted(
        item.relative_to(bundle).as_posix()
        for item in bundle.rglob("*")
        if item.is_file() and item.name != _MANIFEST_PATH
    )
    if actual != paths:
        raise ValueError("bundle has missing or extra payload files")
    bundle_resolved = bundle.resolve(strict=True)
    for entry in payloads:
        payload = bundle / entry["path"]
        resolved = payload.resolve(strict=True)
        if bundle_resolved not in resolved.parents:
            raise ValueError("bundle payload resolves outside the bundle")
        if payload.stat().st_size != entry["size"]:
            raise ValueError("bundle payload byte size mismatch")
        if _sha256(payload) != entry["sha256"]:
            raise ValueError("bundle payload SHA-256 hash mismatch")
    return manifest


def _assert_tree_round_trip(original: DataTree, restored: DataTree) -> None:
    if tuple(original.groups) != tuple(restored.groups):
        raise ValueError("DataTree groups changed during NetCDF round trip")
    for group in original.groups:
        left = original[group].dataset
        right = restored[group].dataset
        if tuple(left.variables) != tuple(right.variables):
            raise ValueError("DataTree variables changed during NetCDF round trip")
        for name in left.variables:
            left_value = left[name]
            right_value = right[name]
            if left_value.dtype.kind in "fc" and right_value.dtype.kind in "fc":
                values_equal = np.array_equal(left_value.values, right_value.values, equal_nan=True)
            else:
                values_equal = np.array_equal(left_value.values, right_value.values)
            if (
                left_value.dims != right_value.dims
                or left_value.dtype != right_value.dtype
                or left_value.shape != right_value.shape
                or not values_equal
            ):
                raise ValueError(
                    "DataTree values, dimensions, coordinates, or dtypes changed during round trip"
                )


def _assert_result_tree_round_trip(
    original: _b4.BayesRARCATResult, restored: _b4.BayesRARCATResult
) -> None:
    _assert_tree_round_trip(
        original.conditional_posterior.prior_predictive.predictive,
        restored.conditional_posterior.prior_predictive.predictive,
    )
    for left, right in zip(
        original.conditional_posterior.components,
        restored.conditional_posterior.components,
        strict=True,
    ):
        _assert_tree_round_trip(left.posterior, right.posterior)
        _assert_tree_round_trip(left.posterior_predictive, right.posterior_predictive)


def read_bayes_rarcat_result(path: os.PathLike[str] | str) -> _b4.BayesRARCATResult:
    """Read and fully validate one private B4 directory bundle."""
    bundle = Path(path)
    if bundle.is_symlink():
        raise ValueError("bundle path cannot be a symlink")
    if not bundle.is_dir():
        raise FileNotFoundError("bundle path must be an existing directory")
    manifest = _validated_manifest(bundle)
    allowed_paths = {entry["path"] for entry in manifest["payloads"]}
    metadata = _read_canonical_json(bundle / _METADATA_PATH)
    context = _ReadContext(bundle, allowed_paths=allowed_paths)
    decoded = context.decode(metadata)
    used_expected = allowed_paths - {_METADATA_PATH}
    if context.used_paths != used_expected:
        raise ValueError("bundle inventory contains unreferenced or missing payloads")
    if type(decoded) is not _b4.BayesRARCATResult:
        raise ValueError("bundle metadata does not contain a BayesRARCATResult")
    rebuilt = _rebuild_result_design(decoded)
    if manifest["software_versions"] != [list(item) for item in rebuilt.software_versions]:
        raise ValueError("manifest software versions do not match the typed result")
    return _b4.validate_bayes_rarcat_result(rebuilt)


def _validate_written_bundle(path: Path) -> _b4.BayesRARCATResult:
    return read_bayes_rarcat_result(path)


def _entry_identity(path: Path) -> tuple[int, int]:
    status = path.lstat()
    return status.st_dev, status.st_ino


def _failure_artifact_if_owned(
    source: Path,
    *,
    expected_identity: tuple[int, int],
) -> Path | None:
    """Return a matching failure artifact without mutating its pathname."""
    try:
        current_identity = _entry_identity(source)
    except FileNotFoundError:
        return None
    if current_identity != expected_identity:
        return None
    return source


def _note_failure_artifact_disposition(
    error: BaseException,
    source: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Preserve the primary error while reporting an in-place artifact."""
    try:
        artifact = _failure_artifact_if_owned(
            source,
            expected_identity=expected_identity,
        )
    except BaseException as artifact_error:
        error.add_note(
            "Could not confirm the in-place failure artifact associated with "
            f"{source}: "
            f"{type(artifact_error).__name__}: {artifact_error}"
        )
        return
    if artifact is None:
        return
    note = (
        "Failure handling left the pathname untouched after observing the "
        f"writer artifact at {artifact}; its identity may change concurrently"
    )
    if note not in getattr(error, "__notes__", ()):
        error.add_note(note)


def _publish_directory_noreplace(
    temporary: Path,
    destination: Path,
) -> _b4.BayesRARCATResult:
    """Publish and validate a directory atomically without replacement.

    Successful visibility is a same-filesystem atomic Linux rename. The
    destination is never replaced. If published validation fails and the
    recorded identity remains at the destination, it is left in place and its
    exact path is added to the original exception.
    """
    temporary = Path(temporary)
    destination = Path(destination)
    if temporary.is_symlink():
        raise ValueError("temporary bundle cannot be a symlink")
    if not temporary.is_dir():
        raise FileNotFoundError("temporary bundle must be an existing directory")
    if not destination.parent.is_dir():
        raise FileNotFoundError("bundle parent must be an existing directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("bundle destination must not already exist")
    temporary_identity = _entry_identity(temporary)
    if temporary_identity[0] != destination.parent.stat().st_dev:
        raise OSError(errno.EXDEV, "bundle publication requires one filesystem", destination)

    _rename_directory_noreplace(temporary, destination)
    try:
        return _validate_written_bundle(destination)
    except BaseException as error:
        _note_failure_artifact_disposition(
            error,
            destination,
            expected_identity=temporary_identity,
        )
        raise


def write_bayes_rarcat_result(result: _b4.BayesRARCATResult, path: os.PathLike[str] | str) -> Path:
    """Atomically write one passing private B4 bundle without replacement.

    Publication uses Linux ``renameat2(RENAME_NOREPLACE)`` on one filesystem.
    Unsupported platforms and filesystems fail closed instead of weakening the
    no-replace guarantee. The published bundle is validated before return.
    When a recorded writer identity remains at a writer-managed pathname after
    failure, it is left in place and the path is noted on the original
    exception. Failure handling never renames, deletes, or overwrites an entry,
    because a pathname identity check cannot make a later cleanup mutation safe
    against rebinding by an unrelated writer.

    Not reached from ``bayes_rarcat``: this is intentional private API, called
    by the verification suite and by callers outside the public path, and it
    has no reference in ``src/`` other than this definition. See finding F-09
    of the 2026-08-25 audit, which re-adjudicated the earlier reference counts.
    """
    result = _b4.validate_bayes_rarcat_result(result)
    if result.status != "pass":
        raise ValueError("only a passing B4 result can be serialized")
    destination = Path(path)
    parent = destination.parent
    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError("bundle parent must be an existing directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("bundle destination must not already exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent.resolve()))
    temporary_identity = _entry_identity(temporary)
    try:
        context = _WriteContext(temporary)
        metadata = context.encode(result)
        (temporary / _METADATA_PATH).write_bytes(_canonical_json_bytes(metadata))
        manifest = _manifest(result, temporary)
        (temporary / _MANIFEST_PATH).write_bytes(_canonical_json_bytes(manifest))
        restored = _validate_written_bundle(temporary)
        _assert_result_tree_round_trip(result, restored)
        _publish_directory_noreplace(temporary, destination)
    except BaseException as error:
        _note_failure_artifact_disposition(
            error,
            temporary,
            expected_identity=temporary_identity,
        )
        raise
    return destination

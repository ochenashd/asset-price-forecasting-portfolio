"""Lossless, data-only archives for the existing partial keyed capture.

The original bridge retains a probability dtype descriptor and Python float
values, not the original ndarray or prediction-label dtype. It retains no
observation/availability/decision clocks. This codec preserves those absences;
row identities never supply missing physical clocks or session identities.
"""
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
import hashlib
import math
from numbers import Integral
from types import MappingProxyType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .bundle_verification import VerificationLimits, _parse, _temporal
from .protocol import ProtocolDiagnostic, ResearchProtocolError, _json, _key
from .receipt import ResearchFitReceiptV1
from src.evaluation.execution.xgboost_bridge import KeyedXGBoostPredictions, KeyedXGBoostRow, _index
from src.evaluation.execution.pipeline_handoff import CaptureReceipt, capture_fingerprint
from src.evaluation.execution.prediction_export import ProbabilityRow, _utc_instant

SCHEMA_VERSION = "apf-keyed-prediction-artifact-v1"
MISSING_DETAILS = ("feature_observation_timestamps", "feature_availability_timestamps",
                   "historical_decision_eligibility", "exchange_session_ids", "prediction_array_dtype")
INDEX_FIELDS = ("held_out_index", "training_source_index", "fitting_index", "validation_index")


class PredictionArtifactError(ResearchProtocolError):
    """Safe diagnostics never include artifact contents or private contexts."""


def _require(value, code):
    if not value:
        raise PredictionArtifactError((ProtocolDiagnostic(code, "prediction_capture", "Prediction artifact validation failed."),))


@dataclass(frozen=True)
class PredictionArtifactLimits:
    verification: VerificationLimits = VerificationLimits()
    max_rows: int = 100000
    max_index_values: int = 500000

    def __post_init__(self):
        if type(self.verification) is not VerificationLimits or any(type(x) is not int or x <= 0 for x in (self.max_rows, self.max_index_values)):
            raise ValueError("Positive codec limits are required.")


@dataclass(frozen=True)
class PredictionCaptureArtifact:
    data: bytes = field(repr=False)
    sha256: str
    fingerprint: str


@dataclass(frozen=True)
class DecodedPredictionCapture:
    capture: KeyedXGBoostPredictions = field(repr=False)
    fitting_attempt: CaptureReceipt

    @property
    def fingerprint(self):
        return capture_fingerprint(self.capture)


@dataclass(frozen=True)
class PredictionVerificationReport:
    status: str
    fingerprint: str | None = None
    diagnostics: tuple[ProtocolDiagnostic, ...] = ()
    decision_chronology: str = "incomplete"
    unavailable_details: tuple[str, ...] = MISSING_DETAILS
    historical_pit_status: str = "unknown"
    execution_eligibility_status: str = "unassessed"


def _zone(tz):
    if tz is None:
        return None
    if type(tz) is ZoneInfo:
        return {"backend": "zoneinfo", "key": tz.key}
    if tz is timezone.utc:
        return {"backend": "utc"}
    if type(tz) is timezone:
        offset = tz.utcoffset(None)
        microseconds = (offset.days * 86400 + offset.seconds) * 1000000 + offset.microseconds
        return {"backend": "fixed", "microseconds": microseconds, "name": tz.tzname(None)}
    if type(tz).__module__.startswith("pytz") and isinstance(getattr(tz, "zone", None), str):
        return {"backend": "pytz", "key": tz.zone}
    _require(False, "unsupported_timezone_backend")


def _unzone(value):
    if value is None:
        return None
    _require(type(value) is dict, "invalid_timezone")
    if value == {"backend": "utc"}:
        return timezone.utc
    if set(value) == {"backend", "key"} and value["backend"] in {"zoneinfo", "pytz"}:
        _require(type(value["key"]) is str and 0 < len(value["key"]) <= 128, "invalid_timezone")
        if value["backend"] == "zoneinfo":
            return ZoneInfo(value["key"])
        import pytz
        return pytz.timezone(value["key"])
    _require(set(value) == {"backend", "microseconds", "name"} and value["backend"] == "fixed"
             and type(value["microseconds"]) is int and abs(value["microseconds"]) < 86400000000
             and type(value["name"]) is str, "invalid_timezone")
    return timezone(timedelta(microseconds=value["microseconds"]), value["name"])


def _value(value):
    if value is None:
        return {"type": "none"}
    if isinstance(value, datetime):
        _require(value == value, "missing_timestamp")
        result = {"type": "timestamp" if type(value) is pd.Timestamp else "datetime", "iso": value.isoformat(),
                  "zone": _zone(value.tzinfo), "fold": value.fold}
        _require(type(value) in {pd.Timestamp, datetime}, "unsupported_datetime_type")
        if type(value) is pd.Timestamp:
            result["unit"] = np.datetime_data(value.asm8.dtype)[0]
        return result
    if isinstance(value, np.integer):
        return {"type": "numpy_integer", "dtype": str(value.dtype), "value": str(value)}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": str(value)}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value": value.hex()}
    if type(value) is tuple:
        return {"type": "tuple", "values": [_value(x) for x in value]}
    _require(False, "unsupported_typed_identity")


def _unvalue(raw):
    _require(type(raw) is dict and type(raw.get("type")) is str, "invalid_typed_identity")
    kind = raw["type"]
    if kind == "none":
        _require(set(raw) == {"type"}, "invalid_typed_identity")
        return None
    if kind in {"datetime", "timestamp"}:
        expected = {"type", "iso", "zone", "fold"} | ({"unit"} if kind == "timestamp" else set())
        _require(set(raw) == expected and type(raw["iso"]) is str and type(raw["fold"]) is int and raw["fold"] in {0, 1}, "invalid_timestamp")
        tz = _unzone(raw["zone"])
        value = pd.Timestamp(raw["iso"]) if kind == "timestamp" else datetime.fromisoformat(raw["iso"])
        _require(value == value and (value.tzinfo is None) == (tz is None), "invalid_timestamp")
        if tz is not None:
            value = value.tz_convert(tz) if kind == "timestamp" else value.astimezone(tz)
        value = value.replace(fold=raw["fold"])
        if kind == "timestamp":
            _require(raw["unit"] in {"s", "ms", "us", "ns"}, "unsupported_timestamp_unit")
            value = value.as_unit(raw["unit"])
        _require(_value(value) == raw, "timestamp_representation_mismatch")
        return value
    if kind == "tuple":
        _require(set(raw) == {"type", "values"} and type(raw["values"]) is list, "invalid_typed_identity")
        return tuple(_unvalue(x) for x in raw["values"])
    _require(set(raw) == {"type", "value"} | ({"dtype"} if kind == "numpy_integer" else set()), "invalid_typed_identity")
    if kind in {"integer", "numpy_integer"}:
        _require(type(raw["value"]) is str and len(raw["value"]) <= 32, "invalid_integer_identity")
        value = int(raw["value"])
        if kind == "numpy_integer":
            _require(type(raw["dtype"]) is str and len(raw["dtype"]) <= 16, "unsupported_identity_dtype")
            dtype = np.dtype(raw["dtype"])
            _require(dtype.kind in "iu" and dtype.itemsize <= 8 and np.iinfo(dtype).min <= value <= np.iinfo(dtype).max, "unsupported_identity_dtype")
            value = np.asarray(value, dtype=dtype)[()]
    elif kind == "string":
        _require(type(raw["value"]) is str, "invalid_string_identity")
        value = raw["value"]
    elif kind == "bool":
        _require(type(raw["value"]) is bool, "invalid_typed_identity")
        value = raw["value"]
    elif kind == "float":
        value = _unfloat(raw["value"])
    else:
        _require(False, "unsupported_typed_identity")
    _require(_value(value) == raw, "noncanonical_typed_identity")
    return value


def _unfloat(text):
    _require(type(text) is str and len(text) <= 32, "invalid_numeric_value")
    value = float.fromhex(text)
    _require(math.isfinite(value) and value.hex() == text, "noncanonical_or_nonfinite_numeric_value")
    return value


def _offset(value):
    if value is None:
        return None
    _require(type(value) is str and 0 < len(value) <= 64, "unsupported_index_frequency")
    offset = pd.tseries.frequencies.to_offset(value)
    allowed = (pd.offsets.Day, pd.offsets.Hour, pd.offsets.Minute, pd.offsets.Second, pd.offsets.Milli,
               pd.offsets.Micro, pd.offsets.Nano, pd.offsets.BusinessDay, pd.offsets.Week, pd.offsets.MonthBegin, pd.offsets.MonthEnd)
    _require(type(offset) in allowed and offset.freqstr == value, "unsupported_index_frequency")
    return offset


def _index_data(index):
    if index is None:
        return None
    base = {"name": _value(index.name), "dtype": str(index.dtype)}
    if type(index) is pd.RangeIndex:
        return {**base, "type": "range", "start": str(index.start), "stop": str(index.stop), "step": str(index.step)}
    if type(index) is pd.DatetimeIndex:
        frequency = index.freqstr
        _require(index.freq == _offset(frequency), "unsupported_index_frequency_parameters")
        unit = np.datetime_data(index.dtype)[0] if index.tz is None else index.dtype.unit
        return {**base, "type": "datetime", "zone": _zone(index.tz), "unit": unit, "frequency": frequency,
                "values": [_value(x) for x in index]}
    _require(type(index) is pd.Index, "unsupported_index_class")
    dtype = index.dtype
    storage = None
    if isinstance(dtype, pd.StringDtype):
        _require(dtype.storage == "python", "unsupported_string_storage")
        storage = "python"
    else:
        _require(str(dtype) in {"object", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}, "unsupported_index_dtype")
    return {**base, "type": "index", "string_storage": storage, "values": [_value(x) for x in index]}


def _index_from(raw, limits, *, empty=False):
    if raw is None:
        return None
    _require(type(raw) is dict and "type" in raw and "name" in raw and type(raw.get("dtype")) is str, "invalid_index")
    name = _unvalue(raw["name"])
    if raw["type"] == "range":
        _require(set(raw) == {"type", "name", "dtype", "start", "stop", "step"}, "invalid_index")
        values = tuple(int(raw[key]) for key in ("start", "stop", "step"))
        _require(all(-(2**63) <= x < 2**63 for x in values) and values[2] != 0, "invalid_range_index")
        _require(len(range(*values)) <= limits.max_index_values, "index_allocation_limit")
        index = pd.RangeIndex(*values, name=name)
    else:
        _require(type(raw.get("values")) is list and len(raw["values"]) <= limits.max_index_values, "index_allocation_limit")
        if raw["type"] == "datetime":
            _require(set(raw) == {"type", "name", "dtype", "zone", "unit", "frequency", "values"}
                     and raw["unit"] in {"s", "ms", "us", "ns"}, "unsupported_datetime_index")
            tz = _unzone(raw["zone"])
            dtype = pd.DatetimeTZDtype(unit=raw["unit"], tz=tz) if tz is not None else np.dtype("datetime64[" + raw["unit"] + "]")
            index = pd.DatetimeIndex([_unvalue(x) for x in raw["values"]], dtype=dtype, name=name, freq=_offset(raw["frequency"]))
        else:
            _require(raw["type"] == "index" and set(raw) == {"type", "name", "dtype", "string_storage", "values"}, "unsupported_index_schema")
            dtype = raw["dtype"]
            allowed = {"object", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                       "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}
            if dtype == "string":
                _require(raw["string_storage"] == "python", "unsupported_string_storage")
                dtype = pd.StringDtype(storage="python")
            else:
                _require(dtype in allowed and raw["string_storage"] is None, "unsupported_index_dtype")
            index = pd.Index([_unvalue(x) for x in raw["values"]], dtype=dtype, name=name, tupleize_cols=False)
    _index(index, "decoded_index", allow_empty=empty)
    _require(_index_data(index) == raw, "index_representation_mismatch")
    return index


def _capture_data(capture):
    result = {name: _index_data(getattr(capture, name)) for name in INDEX_FIELDS}
    for name in ("ticker", "asset_id", "model_id", "split_id", "feature_names", "estimator_classes", "estimator_type",
                 "probability_dtype", "actual_prediction_generated_at", "missing_execution_metadata", "execution_ready"):
        result[name] = _value(getattr(capture, name))
    result["maximum_probability_sum_error"] = capture.maximum_probability_sum_error.hex()
    result["probability_shape"] = [len(capture.records), 3]
    result["probability_columns"] = list(capture.estimator_classes)
    result["records"] = [{"row_id": _value(row.feature_row_id), "predicted_class": row.predicted_class,
                          "probabilities": [[label, row.probabilities[label].hex()] for label in capture.estimator_classes],
                          "probability_sum_error": row.probability_sum_error.hex()} for row in capture.records]
    return result


def _validate_capture(capture, attempt, limits):
    _require(type(capture) is KeyedXGBoostPredictions and type(attempt) is CaptureReceipt, "unsupported_capture_contract")
    _require(0 < len(capture.records) <= limits.max_rows and sum(len(getattr(capture, key)) for key in INDEX_FIELDS
             if getattr(capture, key) is not None) <= limits.max_index_values, "capture_allocation_limit")
    for key in INDEX_FIELDS:
        index = getattr(capture, key)
        if index is not None:
            _index(index, key, allow_empty=key == "validation_index")
    _require(capture.held_out_index is not None and len(capture.records) == len(capture.held_out_index), "row_count_mismatch")
    _require(type(capture.estimator_classes) is tuple and len(capture.estimator_classes) == 3
             and all(type(x) is int for x in capture.estimator_classes) and set(capture.estimator_classes) == {0, 1, 2}, "invalid_class_order")
    _require(type(capture.feature_names) is tuple and bool(capture.feature_names) and len(set(capture.feature_names)) == len(capture.feature_names)
             and all(type(x) is str and x.strip() and not x.lower().startswith(("target", "future_")) and x.lower() != "y_true" for x in capture.feature_names), "invalid_features")
    _require(type(capture.ticker) is str and bool(capture.ticker.strip()) and type(capture.estimator_type) is str and bool(capture.estimator_type.strip()), "invalid_capture_identity")
    _require(capture.asset_id is None or (isinstance(capture.asset_id, Integral) and not isinstance(capture.asset_id, bool))
             or (type(capture.asset_id) is str and bool(capture.asset_id.strip())), "invalid_asset_identity")
    _require(all(value is None or (type(value) is str and bool(value.strip())) for value in (capture.model_id, capture.split_id)), "invalid_model_or_split")
    _require(type(capture.execution_ready) is bool and capture.execution_ready is False, "unsupported_execution_claim")
    _require(type(capture.missing_execution_metadata) is tuple and all(type(x) is str and x.strip() for x in capture.missing_execution_metadata)
             and len(set(capture.missing_execution_metadata)) == len(capture.missing_execution_metadata), "invalid_missing_metadata")
    _require(type(capture.probability_dtype) is str and len(capture.probability_dtype) <= 16, "unsupported_probability_dtype")
    dtype = np.dtype(capture.probability_dtype)
    _require(dtype.kind in "fiu" and dtype.itemsize <= 8 and str(dtype) == capture.probability_dtype, "unsupported_probability_dtype")
    for index_value, row in zip(capture.held_out_index, capture.records):
        _require(type(row) is KeyedXGBoostRow and _value(index_value) == _value(row.feature_row_id), "record_row_alignment")
        _require(type(row.predicted_class) is int and row.predicted_class in {0, 1, 2}, "invalid_prediction_class")
        _require(tuple(row.probabilities) == capture.estimator_classes and all(type(x) is int for x in row.probabilities), "probability_column_mismatch")
        values = tuple(row.probabilities[label] for label in capture.estimator_classes)
        _require(all(type(x) is float for x in values), "unsupported_probability_scalar")
        ProbabilityRow("codec-validation", values)
        # Re-encoding the descriptor must represent exactly these retained values.
        rebuilt = np.asarray(values, dtype=dtype)
        _require(all(float(x).hex() == value.hex() for x, value in zip(rebuilt, values)), "probability_dtype_value_mismatch")
        error = abs(math.fsum(values) - 1.0)
        _require(type(row.probability_sum_error) is float and error.hex() == row.probability_sum_error.hex(), "probability_sum_error_mismatch")
    _require(type(capture.maximum_probability_sum_error) is float and capture.maximum_probability_sum_error.hex() ==
             max(row.probability_sum_error for row in capture.records).hex(), "maximum_probability_error_mismatch")
    generated = capture.actual_prediction_generated_at
    if generated is not None:
        _utc_instant(generated)
    source, fitting, validation = (getattr(capture, name) for name in INDEX_FIELDS[1:])
    _require(all(x is None for x in (source, fitting, validation)) or all(x is not None for x in (source, fitting, validation)), "partial_split_membership")
    if source is not None:
        _require(fitting.isin(source).all() and validation.isin(source).all()
                 and fitting.intersection(validation).empty and source.intersection(capture.held_out_index).empty, "overlapping_or_unknown_membership")
    _require(all(type(getattr(attempt, item.name)) is str and bool(getattr(attempt, item.name).strip()) for item in fields(CaptureReceipt)), "invalid_attempt_binding")
    _require(attempt.capture_fingerprint == capture_fingerprint(capture), "capture_fingerprint_mismatch")
    _require(capture.model_id is None or capture.model_id == attempt.fitted_model_id, "capture_model_binding_mismatch")
    if type(capture.asset_id) is str:
        _require(capture.asset_id == attempt.stable_asset_id, "capture_asset_binding_mismatch")


def serialize_prediction_capture(capture, *, fitting_attempt, limits=PredictionArtifactLimits()):
    """Encode an existing capture and its explicit fresh-attempt binding only."""
    try:
        _validate_capture(capture, fitting_attempt, limits)
        raw = {"schema_version": SCHEMA_VERSION, "serialization": "canonical-json-v1",
               "capture_fingerprint": capture_fingerprint(capture), "fitting_attempt": {item.name: getattr(fitting_attempt, item.name) for item in fields(CaptureReceipt)},
               "capture": _capture_data(capture), "unavailable_details": list(MISSING_DETAILS)}
        data = _json(raw).encode()
        _require(len(data) <= min(limits.verification.archive.max_artifact_bytes, limits.verification.archive.max_total_bytes), "artifact_size_limit")
        artifact = PredictionCaptureArtifact(data, hashlib.sha256(data).hexdigest(), raw["capture_fingerprint"])
        decoded = deserialize_prediction_capture(artifact, limits=limits)
        for name in INDEX_FIELDS:
            original, restored = getattr(capture, name), getattr(decoded.capture, name)
            _require((original is None and restored is None) or (original is not None and restored is not None
                     and type(original) is type(restored) and original.identical(restored)), "unsupported_index_roundtrip")
        _require(_capture_data(decoded.capture) == raw["capture"], "capture_field_roundtrip_mismatch")
        return artifact
    except PredictionArtifactError:
        raise
    except Exception:
        _require(False, "unsupported_or_malformed_capture")


def deserialize_prediction_capture(artifact_bytes, *, expected_sha256=None, limits=PredictionArtifactLimits()):
    """Decode fixed data contracts; never construct an estimator to rebuild rows."""
    try:
        declared_fp = None
        if type(artifact_bytes) is PredictionCaptureArtifact:
            data, digest, declared_fp = artifact_bytes.data, artifact_bytes.sha256, artifact_bytes.fingerprint
            _require(expected_sha256 is None or expected_sha256 == digest, "artifact_digest_mismatch")
        else:
            data, digest = artifact_bytes, expected_sha256
        _require(type(data) is bytes and type(digest) is str, "explicit_artifact_digest_required")
        _require(len(data) <= min(limits.verification.archive.max_artifact_bytes, limits.verification.archive.max_total_bytes), "artifact_size_limit")
        _require(hashlib.sha256(data).hexdigest() == digest, "artifact_digest_mismatch")
        raw = _parse(data, limits.verification)
        _require(type(raw) is dict and set(raw) == {"schema_version", "serialization", "capture_fingerprint", "fitting_attempt", "capture", "unavailable_details"}, "invalid_artifact_fields")
        _require(raw["schema_version"] == SCHEMA_VERSION and raw["serialization"] == "canonical-json-v1", "unsupported_capture_schema")
        _require(_json(raw).encode() == data and raw["unavailable_details"] == list(MISSING_DETAILS), "noncanonical_capture_artifact")
        values = raw["capture"]
        expected = {item.name for item in fields(KeyedXGBoostPredictions)} | {"probability_shape", "probability_columns"}
        _require(type(values) is dict and set(values) == expected, "invalid_capture_fields")
        records = values["records"]
        _require(type(records) is list and 0 < len(records) <= limits.max_rows
                 and values["probability_shape"] == [len(records), 3]
                 and all(type(x) is int for x in values["probability_shape"]), "invalid_probability_shape")
        # Account for every index before creating pandas objects or arrays.
        count = 0
        for name in INDEX_FIELDS:
            item = values[name]
            if item is not None:
                if item.get("type") == "range":
                    count += len(range(*(int(item[key]) for key in ("start", "stop", "step"))))
                else:
                    count += len(item["values"])
        _require(count <= limits.max_index_values, "index_allocation_limit")
        indices = {name: _index_from(values[name], limits, empty=name == "validation_index") for name in INDEX_FIELDS}
        attributes = {name: _unvalue(value) for name, value in values.items() if name not in INDEX_FIELDS + ("records", "maximum_probability_sum_error", "probability_shape", "probability_columns")}
        classes = attributes["estimator_classes"]
        _require(type(classes) is tuple and values["probability_columns"] == list(classes)
                 and all(type(x) is int for x in values["probability_columns"]), "probability_column_mismatch")
        rows = []
        for row in records:
            _require(type(row) is dict and set(row) == {"row_id", "predicted_class", "probabilities", "probability_sum_error"}, "invalid_record_fields")
            pairs = row["probabilities"]
            _require(type(pairs) is list and len(pairs) == 3 and all(type(pair) is list and len(pair) == 2 and type(pair[0]) is int for pair in pairs)
                     and tuple(pair[0] for pair in pairs) == classes, "probability_column_mismatch")
            rows.append(KeyedXGBoostRow(_unvalue(row["row_id"]), row["predicted_class"],
                MappingProxyType({label: _unfloat(value) for label, value in pairs}), _unfloat(row["probability_sum_error"])))
        capture = KeyedXGBoostPredictions(**indices, **attributes, records=tuple(rows), maximum_probability_sum_error=_unfloat(values["maximum_probability_sum_error"]))
        binding = raw["fitting_attempt"]
        _require(type(binding) is dict and set(binding) == {item.name for item in fields(CaptureReceipt)}, "invalid_attempt_fields")
        attempt = CaptureReceipt(**binding)
        _validate_capture(capture, attempt, limits)
        _require(capture_fingerprint(capture) == raw["capture_fingerprint"] and (declared_fp is None or declared_fp == raw["capture_fingerprint"]), "capture_fingerprint_mismatch")
        _require(_capture_data(capture) == values, "capture_field_roundtrip_mismatch")
        return DecodedPredictionCapture(capture, attempt)
    except PredictionArtifactError:
        raise
    except Exception:
        _require(False, "malformed_prediction_artifact")


def verify_prediction_capture_artifact(artifact_bytes, receipt, *, expected_sha256=None, limits=PredictionArtifactLimits()):
    """Check receipt/manifest association, not model existence or decision timing."""
    try:
        decoded = deserialize_prediction_capture(artifact_bytes, expected_sha256=expected_sha256, limits=limits)
        _require(type(receipt) is ResearchFitReceiptV1, "invalid_receipt")
        receipt = ResearchFitReceiptV1.from_json(receipt.canonical_json())
        capture, attempt = decoded.capture, decoded.fitting_attempt
        _require((attempt.run_id, attempt.fitted_model_id, attempt.stable_asset_id, attempt.capture_fingerprint) ==
                 (receipt.attempt_id, receipt.fitted_model_id, receipt.asset_id, receipt.prediction_capture_fingerprint), "stale_or_unrelated_fitting_attempt")
        _require(capture.ticker == receipt.ticker and capture.asset_id == receipt.asset_id, "receipt_asset_mismatch")
        _require(capture.feature_names == receipt.selected_features and capture.estimator_classes == (0, 1, 2)
                 and capture.estimator_type == "xgboost.sklearn.XGBClassifier", "receipt_features_or_classes_mismatch")
        _require(capture.model_id in {None, receipt.fitted_model_id} and capture.split_id in {None, receipt.temporal_manifest_id}, "receipt_model_or_manifest_mismatch")
        manifest = _temporal(receipt.temporal_manifest_json.encode(), limits.verification)
        _require(manifest.split_id == receipt.temporal_manifest_id and manifest.model_id == receipt.fitted_model_id
                 and manifest.asset_id == receipt.asset_id and manifest.experiment_id == receipt.experiment_id
                 and manifest.feature_names == receipt.selected_features, "receipt_temporal_manifest_mismatch")
        _require((manifest.target_type, manifest.target_horizon, manifest.target_threshold, manifest.uses_early_stopping) ==
                 (receipt.target_configuration.task_type, receipt.target_configuration.horizon,
                  receipt.target_configuration.threshold, receipt.boosting_metadata["uses_early_stopping"]), "receipt_temporal_configuration_mismatch")
        bindings = receipt.row_identity_bindings
        _require(all(type(row) is tuple and len(row) == 4 for row in bindings), "invalid_receipt_row_bindings")
        clocks = {_key(row): clock for row, _, clock, _ in bindings}
        _require(len(clocks) == len(bindings) == len(manifest.source_rows), "receipt_source_identity_mismatch")
        for manifest_field, partition_field in (("fitting_ids", "fitting"), ("inner_purged_ids", "inner_purge"),
                ("validation_ids", "validation"), ("outer_purged_ids", "outer_purge"), ("held_out_ids", "held_out")):
            _require(tuple(_utc_instant(value).isoformat() for value in getattr(manifest, manifest_field)) ==
                     tuple(_utc_instant(clocks[_key(value)]).isoformat() for value in getattr(receipt.partitions, partition_field)), "receipt_frozen_membership_mismatch")
        if manifest.actual_fit_completed_at is None:
            return PredictionVerificationReport("incomplete", decoded.fingerprint,
                (ProtocolDiagnostic("missing_manifest_fit_completion", "prediction_capture", "Manifest fit completion is unknown."),))
        _require(_utc_instant(manifest.actual_fit_completed_at) == _utc_instant(receipt.actual_fit_completed_at)
                 and _utc_instant(receipt.actual_fit_completed_at) >= _utc_instant(manifest.model_information_cutoff), "receipt_fit_completion_mismatch")
        if any(getattr(capture, name) is None for name in INDEX_FIELDS[1:]) or capture.actual_prediction_generated_at is None:
            return PredictionVerificationReport("incomplete", decoded.fingerprint,
                (ProtocolDiagnostic("missing_retained_split_or_generation_metadata", "prediction_capture", "Required capture metadata is absent."),))
        allowed = set(manifest.fitting_ids + manifest.inner_purged_ids + manifest.validation_ids)
        source = tuple(row.row_id for row in manifest.source_rows if row.row_id in allowed)
        for name, expected in (("held_out_index", manifest.held_out_ids), ("training_source_index", source),
                               ("fitting_index", manifest.fitting_ids), ("validation_index", manifest.validation_ids)):
            actual = getattr(capture, name)
            _require(len(actual) == len(expected), "receipt_membership_mismatch")
            for left, right in zip(actual, expected):
                if isinstance(right, datetime) and right.utcoffset() is not None:
                    _require(isinstance(left, datetime) and _utc_instant(left).isoformat() == _utc_instant(right).isoformat(), "receipt_membership_mismatch")
                else:
                    _require(_value(left) == _value(right), "receipt_membership_mismatch")
        generated = _utc_instant(capture.actual_prediction_generated_at)
        _require(generated == _utc_instant(receipt.actual_prediction_generated_at)
                 and generated >= _utc_instant(receipt.actual_fit_completed_at)
                 and generated >= _utc_instant(manifest.model_information_cutoff)
                 and generated >= max(_utc_instant(row.features_available_at) for row in manifest.source_rows), "invalid_generation_chronology")
        return PredictionVerificationReport("verified", decoded.fingerprint)
    except PredictionArtifactError as error:
        status = "unsupported" if any(x.code.startswith("unsupported_") for x in error.diagnostics) else "failed"
        return PredictionVerificationReport(status, diagnostics=error.diagnostics)
    except NotImplementedError:
        return PredictionVerificationReport("unsupported", diagnostics=(ProtocolDiagnostic("unsupported_temporal_manifest", "prediction_capture", "Temporal schema is unsupported."),))
    except Exception:
        return PredictionVerificationReport("failed", diagnostics=(ProtocolDiagnostic("invalid_receipt_or_temporal_binding", "prediction_capture", "Receipt association failed."),))

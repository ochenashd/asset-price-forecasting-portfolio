"""Frozen research intent and local artifact validation, not PIT certification.

V1 supports one primary Parquet observation artifact and optional referenced
Parquet artifacts. It neither computes features nor fits or executes models.
Training identity binds a declared raw source footprint, not fitted matrices;
the future input adapter must witness the transformation and fitted-state link.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.evaluation.execution.prediction_export import _utc_instant


SCHEMA_VERSION = "apf-research-protocol-v1"
ResearchRowIdentity = str | int | date | datetime | np.integer
_SECRET_KEYS = {"password", "secret", "token", "api_key", "trusted_keys", "hmac_key", "private_key", "signing_key"}
_DATA_KEYS = {"target", "target_return", "y_true", "y_test", "held_out_labels", "heldout_labels",
              "future_prices", "eval_set", "training_data", "validation_data", "test_data"}


@dataclass(frozen=True)
class ProtocolDiagnostic:
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class ProtocolValidationReport:
    diagnostics: tuple[ProtocolDiagnostic, ...] = ()
    verified_artifacts: tuple[str, ...] = ()
    historical_point_in_time_verified: bool = field(default=False, init=False)
    execution_eligible: bool = field(default=False, init=False)

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "verified_artifacts", tuple(self.verified_artifacts))

    @property
    def valid(self):
        return not self.diagnostics

    def require_valid(self):
        if not self.valid:
            raise ResearchProtocolError(self.diagnostics)
        return self


class ResearchProtocolError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(f"{item.code}: {item.field}" for item in self.diagnostics))


def _fail(code, field, message):
    raise ResearchProtocolError((ProtocolDiagnostic(code, field, message),))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _identifier(value):
    """Keep opaque identifier types and local representations distinct."""
    if isinstance(value, bool) or value is None or value is pd.NaT:
        _fail("invalid_identity", "identity", "Missing and boolean identities are forbidden.")
    if isinstance(value, str) and value.strip():
        return ["str", value]
    if isinstance(value, np.integer):
        return ["numpy_integer", str(value.dtype), str(int(value))]
    if type(value) is int:
        return ["int", str(value)]
    if isinstance(value, datetime) and not pd.isna(value):
        zone = getattr(value.tzinfo, "key", None) or getattr(value.tzinfo, "zone", None)
        return ["timestamp" if isinstance(value, pd.Timestamp) else "datetime", value.isoformat(), zone, value.fold]
    if type(value) is date:
        return ["date", value.isoformat()]
    _fail("invalid_identity", "identity", "Use a string, integer, date or nonmissing datetime identity.")


def _key(value):
    return _json(_identifier(value))


def _immutable(value, configuration=False):
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) and type(key) is not int:
                _fail("invalid_configuration", "parameters", "Configuration keys must be strings or integer class labels.")
            if isinstance(key, str) and (key.lower() in _SECRET_KEYS or key.lower().endswith(("_password", "_secret", "_token", "_api_key"))):
                _fail("secret_material", "parameters", "Authentication material is not accepted.")
            if isinstance(key, str) and key.lower() in _DATA_KEYS:
                _fail("fitted_data_in_recipe", "parameters", "Recipes cannot contain target arrays or fitted/evaluation datasets.")
            result[key] = _immutable(item, configuration=True)
        return MappingProxyType(result)
    if isinstance(value, (tuple, list)):
        return tuple(_immutable(item, configuration) for item in value)
    if configuration and isinstance(value, str) and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
        _fail("absolute_configuration_path", "parameters", "Use relative locators or opaque references, not machine-specific paths.")
    _encode(value)
    return value


class _Section:
    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (Mapping, tuple, list)):
                object.__setattr__(self, field.name, _immutable(value))


@dataclass(frozen=True)
class ColumnSpec(_Section):
    name: str
    arrow_type: str
    nullable: bool = True


@dataclass(frozen=True)
class ResearchArtifactRef(_Section):
    relative_path: str
    artifact_id: str
    role: str
    sha256: str
    byte_length: int
    columns: tuple[ColumnSpec, ...]
    schema_sha256: str
    row_id_column: str
    original_row_ids: tuple[ResearchRowIdentity, ...]
    serialization_format: str = "parquet"
    snapshot_reference: str | None = None
    acquisition_reference: str | None = None
    label_column: str | None = None


@dataclass(frozen=True)
class SourceObservation(_Section):
    source_id: ResearchRowIdentity
    artifact_id: str
    original_label: object
    index_dtype: str
    observation_at: datetime
    session_id: str | None = None
    session_ordinal: int | None = None


@dataclass(frozen=True)
class EngineeredRow(_Section):
    row_id: ResearchRowIdentity
    source_id: ResearchRowIdentity


@dataclass(frozen=True)
class ObservationUniverse(_Section):
    observations: tuple[SourceObservation, ...]
    engineered_rows: tuple[EngineeredRow, ...]
    excluded_rows: tuple[tuple[ResearchRowIdentity, str], ...]


@dataclass(frozen=True)
class ComputationWindow(_Section):
    historical_prefix_ids: tuple[ResearchRowIdentity, ...]
    feature_endpoint: ResearchRowIdentity
    label_tail_endpoint: ResearchRowIdentity
    evaluation_source_start: ResearchRowIdentity
    evaluation_source_end: ResearchRowIdentity


@dataclass(frozen=True)
class TransformationSpec(_Section):
    normalization: Mapping[str, object]
    sort_policy: str
    duplicate_policy: str
    missing_values: Mapping[str, object]
    indicators: Mapping[str, object]
    correlation: Mapping[str, object]
    feature_selection: Mapping[str, object]
    scaler: Mapping[str, object]
    raw_input_columns: tuple[str, ...]
    feature_order: tuple[str, ...]
    implementation_reference: str
    calendar_timezone: str


@dataclass(frozen=True)
class TargetSpec(_Section):
    task_type: str
    horizon: int
    horizon_unit: str
    maturity: tuple[tuple[ResearchRowIdentity, ResearchRowIdentity], ...]
    price_column: str = "close"
    class_encoding: tuple[tuple[int, str], ...] = ((0, "Down"), (1, "Neutral"), (2, "Up"))
    threshold: float | None = 0.005


@dataclass(frozen=True)
class FrozenPartitionSpec(_Section):
    fitting: tuple[ResearchRowIdentity, ...]
    inner_purge: tuple[ResearchRowIdentity, ...]
    validation: tuple[ResearchRowIdentity, ...]
    outer_purge: tuple[ResearchRowIdentity, ...]
    held_out: tuple[ResearchRowIdentity, ...]
    training_start: ResearchRowIdentity
    validation_start: ResearchRowIdentity | None
    test_start: ResearchRowIdentity


@dataclass(frozen=True)
class EvaluationSpec(_Section):
    prediction_start: ResearchRowIdentity
    prediction_end: ResearchRowIdentity
    execution_start: datetime
    execution_end: datetime
    terminal_session: str
    terminal_policy: str
    holding_horizon: int
    holding_horizon_unit: str
    horizons_equivalent: bool
    reporting_assumptions: tuple[str, ...]
    execution_parameters: Mapping[str, object]
    performance_parameters: Mapping[str, object]
    market_snapshot_reference: str | None = None
    evidence_reference: str | None = None
    held_out_labels_sha256: str | None = None
    calendar_reference: str | None = None


@dataclass(frozen=True)
class LearningSpec(_Section):
    model_type: str
    effective_parameters: Mapping[str, object]
    random_seeds: Mapping[str, object]
    training_policy: str
    validation_policy: str
    early_stopping: Mapping[str, object]
    declared_parameters: Mapping[str, object]


@dataclass(frozen=True)
class EnvironmentSpec(_Section):
    source_revision: str
    dirty_tree: bool
    dirty_policy: str
    source_digest: str | None
    python_version: str
    package_versions: Mapping[str, object]
    dependency_lock_sha256: str | None
    hardware_threads: Mapping[str, object]


@dataclass(frozen=True)
class ProvenanceLimitations(_Section):
    artifact_evidence: str
    point_in_time_status: str
    assumptions: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    execution_status: str = "unassessed"
    evidence_origin: str = "unknown"
    historical_evidence_references: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingInputSpec(_Section):
    """Declared raw footprint; no full-artifact or held-out-label hash in its ID.

    values_sha256 uses hash_training_rows with source IDs and raw_input_columns
    order. Artifact validation checks this footprint without engineering features.
    """
    source_ids: tuple[ResearchRowIdentity, ...]
    values_sha256: str


@dataclass(frozen=True)
class ResearchProtocolV1(_Section):
    asset_id: str
    experiment_label: str
    artifacts: tuple[ResearchArtifactRef, ...]
    universe: ObservationUniverse
    computation: ComputationWindow
    transformations: TransformationSpec
    target: TargetSpec
    partitions: FrozenPartitionSpec
    evaluation: EvaluationSpec
    learning: LearningSpec
    environment: EnvironmentSpec
    provenance: ProvenanceLimitations
    training_inputs: TrainingInputSpec
    schema_version: str = SCHEMA_VERSION

    def canonical_json(self):
        validate_research_protocol(self).require_valid()
        return _json(_encode(self))

    @property
    def protocol_id(self):
        return "apf-research-v1-" + hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def training_input_id(self):
        validate_research_protocol(self).require_valid()
        p = self.partitions
        return "apf-training-input-v1-" + _digest(_encode((
            self.asset_id, self.training_inputs, self.transformations, self.learning, self.environment,
            self.universe.observations[:len(self.training_inputs.source_ids)],
            self.target.task_type, self.target.horizon, self.target.horizon_unit, self.target.threshold,
            self.target.class_encoding, self.target.price_column,
            tuple((row, end) for row, end in self.target.maturity if _key(row) in {_key(x) for x in p.fitting + p.validation}),
            p.fitting, p.validation)))

    @classmethod
    def from_json(cls, text):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    _fail("invalid_serialization", "json", "Duplicate JSON fields are forbidden.")
                result[key] = value
            return result
        try:
            result = _decode(json.loads(text, object_pairs_hook=unique))
        except ResearchProtocolError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            _fail("invalid_serialization", "json", type(exc).__name__)
        if not isinstance(result, cls):
            _fail("invalid_serialization", "json", "Expected ResearchProtocolV1.")
        validate_research_protocol(result).require_valid()
        if result.canonical_json() != text:
            _fail("noncanonical_serialization", "json", "Only canonical manifests are accepted.")
        return result


_TYPES = {cls.__name__: cls for cls in (
    ColumnSpec, ResearchArtifactRef, SourceObservation, EngineeredRow, ObservationUniverse, ComputationWindow,
    TransformationSpec, TargetSpec, FrozenPartitionSpec, EvaluationSpec, LearningSpec, EnvironmentSpec,
    ProvenanceLimitations, TrainingInputSpec, ResearchProtocolV1)}
_OPAQUE_FIELDS = {"source_id", "row_id", "original_label", "feature_endpoint", "label_tail_endpoint",
                  "evaluation_source_start", "evaluation_source_end", "training_start", "validation_start",
                  "test_start", "prediction_start", "prediction_end"}
_OPAQUE_SEQUENCES = {"original_row_ids", "historical_prefix_ids", "source_ids", "fitting", "inner_purge",
                     "validation", "outer_purge", "held_out", "maturity", "excluded_rows"}


def _encode(value, opaque=False):
    if type(value).__name__ in _TYPES and type(value) is _TYPES[type(value).__name__]:
        return {"type": type(value).__name__, "fields": {
            f.name: _encode(getattr(value, f.name), f.name in _OPAQUE_FIELDS | _OPAQUE_SEQUENCES) for f in fields(value)}}
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["pd.NA"]
    if value is pd.NaT:
        return ["NaT"]
    if isinstance(value, datetime):
        if opaque:
            return ["opaque", _identifier(value)]
        try:
            return ["instant", _utc_instant(value).isoformat()]
        except ValueError:
            _fail("invalid_timestamp", "timestamp", "Physical clocks must be nonmissing and timezone-aware.")
    if type(value) is date:
        return ["date", value.isoformat()]
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            _fail("unsupported_value", "serialization", "Extended numpy scalar precision is unsupported; no coercion is performed.")
        return ["numpy", str(value.dtype), _encode(scalar, opaque)]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, Mapping):
        checked = _immutable(value)
        order = sorted(checked, key=lambda key: _json(_encode(key, opaque=True)))
        return ["map", [[_encode(key, opaque=True), _encode(checked[key])] for key in order]]
    if isinstance(value, (tuple, list)):
        return ["tuple", [_encode(item, opaque) for item in value]]
    _fail("unsupported_value", "serialization", "Unsupported manifest value type.")


def _decode(value):
    if isinstance(value, dict):
        if set(value) != {"type", "fields"} or value["type"] not in _TYPES:
            _fail("invalid_serialization", "type", "Unknown contract type.")
        cls = _TYPES[value["type"]]
        if set(value["fields"]) != {f.name for f in fields(cls)}:
            _fail("invalid_serialization", "fields", "Exact versioned fields are required.")
        return cls(**{key: _decode(item) for key, item in value["fields"].items()})
    tag, *parts = value
    if tag in {"none", "pd.NA", "NaT"}:
        return {"none": None, "pd.NA": pd.NA, "NaT": pd.NaT}[tag]
    if tag == "opaque":
        kind, text, zone, fold = parts[0]
        result = pd.Timestamp(text) if kind == "timestamp" else datetime.fromisoformat(text)
        if zone:
            result = result.astimezone(ZoneInfo(zone))
        return result.replace(fold=fold)
    if tag == "instant":
        return pd.Timestamp(parts[0])
    if tag == "date":
        return date.fromisoformat(parts[0])
    if tag == "bool":
        return parts[0]
    if tag in {"int", "float", "decimal", "str"}:
        return {"int": int, "float": float.fromhex, "decimal": Decimal, "str": str}[tag](parts[0])
    if tag == "numpy":
        return np.asarray(_decode(parts[1]), dtype=parts[0])[()]
    if tag == "map":
        pairs = [(_decode(key), _decode(item)) for key, item in parts[0]]
        if len({_key(key) for key, _ in pairs}) != len(pairs):
            _fail("invalid_serialization", "map", "Duplicate mapping fields.")
        return dict(pairs)
    if tag == "tuple":
        return tuple(_decode(item) for item in parts[0])
    _fail("invalid_serialization", "type", "Unsupported value tag.")


def hash_training_rows(rows):
    """Hash ordered (source ID, raw values in declared column order) JSON lines."""
    digest = hashlib.sha256()
    for source_id, values in rows:
        digest.update((_json([_identifier(source_id), _encode(tuple(values))]) + "\n").encode())
    return digest.hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _locator(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return bool(path.parts) and not path.is_absolute() and str(path) == value and all(part not in {".", ".."} for part in path.parts)


def validate_research_protocol(protocol):
    """Validate declarations and exact memberships; do not infer percentages."""
    diagnostics = []
    def add(code, field, message):
        diagnostics.append(ProtocolDiagnostic(code, field, message))
    try:
        if not isinstance(protocol, ResearchProtocolV1):
            _fail("invalid_contract", "protocol", "ResearchProtocolV1 is required.")
        r = protocol
        _encode(r)
        if r.schema_version != SCHEMA_VERSION:
            add("unsupported_version", "schema_version", "Unsupported research schema.")
        for name in ("asset_id", "experiment_label"):
            if not isinstance(getattr(r, name), str) or not getattr(r, name).strip():
                add("missing_identity", name, "Explicit identity is required.")
        expected_types = {"universe": ObservationUniverse, "computation": ComputationWindow,
            "transformations": TransformationSpec, "target": TargetSpec, "partitions": FrozenPartitionSpec,
            "evaluation": EvaluationSpec, "learning": LearningSpec, "environment": EnvironmentSpec,
            "provenance": ProvenanceLimitations, "training_inputs": TrainingInputSpec}
        for name, cls in expected_types.items():
            if not isinstance(getattr(r, name), cls):
                _fail("invalid_contract", name, "Typed research section required.")
        if not r.artifacts or not all(isinstance(a, ResearchArtifactRef) for a in r.artifacts):
            _fail("invalid_artifact", "artifacts", "Explicit artifact contracts are required.")
        if len({a.artifact_id for a in r.artifacts}) != len(r.artifacts) or len({a.relative_path for a in r.artifacts}) != len(r.artifacts):
            add("duplicate_artifact", "artifacts", "Artifact IDs and locators must be unique.")
        for a in r.artifacts:
            if (not _locator(a.relative_path) or not isinstance(a.artifact_id, str) or not a.artifact_id.strip()
                    or a.role not in {"training_ohlcv", "external_features", "execution_market", "auxiliary"}
                    or not _sha(a.sha256) or not _sha(a.schema_sha256) or type(a.byte_length) is not int or a.byte_length < 1
                    or a.serialization_format != "parquet"):
                add("invalid_artifact", a.artifact_id, "Require contained relative Parquet locator, hashes and byte length.")
            names = [c.name for c in a.columns if isinstance(c, ColumnSpec)]
            if (len(names) != len(a.columns) or not names or len(set(names)) != len(names) or a.row_id_column not in names
                    or (a.label_column is not None and a.label_column not in names)
                    or any(not _text(c.name) or not _text(c.arrow_type) or type(c.nullable) is not bool for c in a.columns)
                    or any(reference is not None and not _text(reference) for reference in (a.snapshot_reference, a.acquisition_reference))):
                add("invalid_schema", a.artifact_id, "Explicit unique column schema and row identity column required.")
            ids = [_key(x) for x in a.original_row_ids]
            if not ids or len(set(ids)) != len(ids):
                add("duplicate_identity", a.artifact_id, "Artifact row identities must be unique and nonempty.")
        primary = [a for a in r.artifacts if a.role == "training_ohlcv"]
        if len(primary) != 1:
            _fail("invalid_artifact", "training_ohlcv", "V1 requires one primary observation artifact.")
        observations = r.universe.observations
        if not observations or not all(isinstance(x, SourceObservation) for x in observations):
            _fail("invalid_universe", "observations", "Explicit original observations required.")
        source = [_key(row.source_id) for row in observations]
        if len(set(source)) != len(source):
            _fail("duplicate_identity", "observations", "Duplicate source identity.")
        if source != [_key(x) for x in primary[0].original_row_ids] or any(row.artifact_id != primary[0].artifact_id for row in observations):
            add("source_order_mismatch", "observations", "Original artifact order and source membership must match exactly.")
        clocks = [_utc_instant(row.observation_at) for row in observations]
        if (any(a >= b for a, b in zip(clocks, clocks[1:])) or any(not isinstance(row.index_dtype, str) or not row.index_dtype for row in observations)
                or any(row.session_id is not None and (not isinstance(row.session_id, str) or not row.session_id) for row in observations)
                or any(row.session_ordinal is not None and (type(row.session_ordinal) is not int or row.session_ordinal < 0) for row in observations)):
            add("invalid_chronology", "observations", "Explicit original types and chronological physical clocks required.")
        positions = {key: i for i, key in enumerate(source)}
        rows = r.universe.engineered_rows
        if not rows or not all(isinstance(row, EngineeredRow) for row in rows):
            _fail("invalid_universe", "engineered_rows", "Explicit engineered-row mapping required.")
        row_ids = [_key(row.row_id) for row in rows]
        mapped = [_key(row.source_id) for row in rows]
        if len(set(row_ids)) != len(row_ids) or len(set(mapped)) != len(mapped) or any(key not in positions for key in mapped):
            _fail("invalid_mapping", "engineered_rows", "V1 requires unique one-to-one source mappings.")
        if [positions[key] for key in mapped] != sorted(positions[key] for key in mapped):
            add("source_order_mismatch", "engineered_rows", "Engineered order must preserve original source order.")
        excluded = [_key(key) for key, reason in r.universe.excluded_rows]
        if (len(set(excluded)) != len(excluded) or excluded != [key for key in source if key not in set(mapped)]
                or any(not isinstance(reason, str) or not reason.strip() for _, reason in r.universe.excluded_rows)):
            add("invalid_exclusions", "excluded_rows", "Account for every unmapped row with an explicit reason.")
        w, t, p, e = r.computation, r.target, r.partitions, r.evaluation
        if t.task_type not in {"5d_class", "5d_return", "1d_return"} or type(t.horizon) is not int or t.horizon < 1 or t.horizon_unit != "observations":
            _fail("invalid_target", "target", "Explicit supported task and positive observation horizon required.")
        if ((t.task_type == "5d_class" and (t.class_encoding != ((0, "Down"), (1, "Neutral"), (2, "Up"))
             or any(type(label) is not int for label, _ in t.class_encoding)
             or isinstance(t.threshold, bool) or not isinstance(t.threshold, (int, float)) or not math.isfinite(t.threshold) or t.threshold < 0))
                or (t.task_type != "5d_class" and (t.threshold is not None or t.class_encoding))
                or (t.task_type == "1d_return" and t.horizon != 1)):
            add("invalid_target", "target", "Task, threshold and class encoding are inconsistent.")
        row_positions = dict(zip(row_ids, (positions[key] for key in mapped)))
        maturity = {_key(key): _key(end) for key, end in t.maturity}
        if len(maturity) != len(t.maturity) or [_key(key) for key, _ in t.maturity] != row_ids:
            _fail("invalid_maturity", "target.maturity", "Every engineered row requires exactly one original maturity ID.")
        for row_id, end in maturity.items():
            if end not in positions or positions[end] != row_positions[row_id] + t.horizon:
                add("invalid_maturity", "target.maturity", "Maturity must use the unfiltered original observation horizon.")
        endpoints = [w.feature_endpoint, w.label_tail_endpoint, w.evaluation_source_start, w.evaluation_source_end]
        if any(_key(key) not in positions for key in endpoints):
            _fail("invalid_window", "computation", "Unknown original endpoint.")
        feature_end, tail_end, eval_start, eval_end = [positions[_key(key)] for key in endpoints]
        if ([_key(x) for x in w.historical_prefix_ids] != source[:feature_end + 1]
                or feature_end < max(row_positions.values()) or tail_end < max(positions.get(x, len(source)) for x in maturity.values())
                or not 0 <= eval_start <= feature_end <= tail_end <= eval_end):
            add("invalid_window", "computation", "Freeze the full recursive prefix, evaluation scope and complete label tail.")
        groups = [tuple(_key(x) for x in group) for group in (p.fitting, p.inner_purge, p.validation, p.outer_purge, p.held_out)]
        flattened = sum(groups, ())
        if not p.fitting or not p.held_out or len(set(flattened)) != len(flattened) or set(flattened) != set(row_ids):
            _fail("invalid_partitions", "partitions", "Complete disjoint fitting, purged, validation and held-out membership required.")
        if any(tuple(key for key in row_ids if key in group) != group for group in groups):
            add("partition_order_mismatch", "partitions", "Partition order must preserve original order.")
        if (_key(p.training_start) != row_ids[0] or _key(p.test_start) != groups[-1][0]
                or bool(p.validation) != (p.validation_start is not None)
                or (p.validation and _key(p.validation_start) != groups[2][0])):
            _fail("boundary_mismatch", "partitions", "Freeze the exact original split boundaries.")
        test_pos = row_positions[_key(p.test_start)]
        val_pos = row_positions[_key(p.validation_start)] if p.validation else None
        expected = [[], [], [], [], []]
        for row_id in row_ids:
            pos, end = row_positions[row_id], positions.get(maturity[row_id], len(source))
            role = 4 if pos >= test_pos else 3 if end >= test_pos else 2 if val_pos is not None and pos >= val_pos else 1 if val_pos is not None and end >= val_pos else 0
            expected[role].append(row_id)
        if any(tuple(group) != actual for group, actual in zip(expected, groups)):
            add("maturity_partition_mismatch", "partitions", "Membership violates the fixed strict maturity boundaries.")
        if (_key(e.prediction_start) != groups[-1][0] or _key(e.prediction_end) != groups[-1][-1]
                or _utc_instant(e.execution_start) > clocks[row_positions[groups[-1][0]]]
                or _utc_instant(e.execution_end) < clocks[row_positions[groups[-1][-1]]]
                or _utc_instant(e.execution_start) >= _utc_instant(e.execution_end)
                or not _text(e.terminal_session) or e.terminal_policy not in {"mark_to_market", "liquidate_at_final_close"}
                or type(e.holding_horizon) is not int or e.holding_horizon < 1 or e.holding_horizon_unit != "sessions"
                or type(e.horizons_equivalent) is not bool or not e.reporting_assumptions
                or any(not _text(item) for item in e.reporting_assumptions)):
            add("invalid_evaluation", "evaluation", "Explicit coherent prediction/execution windows and horizon units required.")
        if e.horizons_equivalent:
            if e.holding_horizon != t.horizon:
                add("horizon_unit_mismatch", "evaluation", "Claimed equivalent horizons differ.")
            ordinals = [row.session_ordinal for row in observations]
            sessions = [row.session_id for row in observations]
            if (any(type(x) is not int for x in ordinals) or any(not x for x in sessions)
                    or len(set(sessions)) != len(sessions) or any(b - a != 1 for a, b in zip(ordinals, ordinals[1:]))
                    or not e.calendar_reference or _utc_instant(e.execution_end) < clocks[tail_end]):
                add("unknown_session_equivalence", "observations", "Equivalence requires explicit consecutive independent session mappings.")
        required_execution = {"position_policy", "initial_capital", "allocation_fraction", "fee_rate", "slippage_rate",
                              "allow_short", "initial_margin_fraction", "maintenance_margin_fraction", "max_short_exposure"}
        if (not required_execution.issubset(e.execution_parameters)
                or e.execution_parameters.get("position_policy") not in {"fixed_horizon", "replace_next_open"}
                or type(e.execution_parameters.get("allow_short")) is not bool
                or not {"periods_per_year", "annual_risk_free_rate", "calendar_days_per_year"}.issubset(e.performance_parameters)
                or (e.held_out_labels_sha256 is not None and not _sha(e.held_out_labels_sha256))):
            add("invalid_evaluation", "evaluation.parameters", "Bind complete execution and performance settings; no execution is authorized.")
        else:
            x, perf = e.execution_parameters, e.performance_parameters
            numbers = ("initial_capital", "allocation_fraction", "fee_rate", "slippage_rate",
                       "initial_margin_fraction", "maintenance_margin_fraction", "max_short_exposure")
            if (any(isinstance(x[key], bool) or not isinstance(x[key], (int, float)) or not math.isfinite(x[key]) for key in numbers)
                    or x["initial_capital"] <= 0 or not 0 < x["allocation_fraction"] <= 1
                    or not 0 <= x["fee_rate"] < 1 or not 0 <= x["slippage_rate"] < 1
                    or not 0 < x["maintenance_margin_fraction"] <= x["initial_margin_fraction"]
                    or not 0 < x["max_short_exposure"] <= 1
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in perf.values())
                    or perf["periods_per_year"] <= 0 or perf["calendar_days_per_year"] <= 0 or perf["annual_risk_free_rate"] <= -1):
                add("invalid_evaluation", "evaluation.parameters", "Execution and annualization values must be finite and economically admissible.")
        tr = r.transformations
        if (not tr.feature_order or len(set(tr.feature_order)) != len(tr.feature_order)
                or not tr.raw_input_columns or len(set(tr.raw_input_columns)) != len(tr.raw_input_columns)
                or set(tr.raw_input_columns) - {c.name for c in primary[0].columns}
                or set(tr.raw_input_columns) - {"open", "high", "low", "close", "volume", "adj_close"}
                or t.price_column not in tr.raw_input_columns or any(name.startswith(("target", "future_")) for name in tr.raw_input_columns + tr.feature_order)
                or any(not _text(value) for value in (tr.implementation_reference, tr.calendar_timezone, tr.sort_policy, tr.duplicate_policy))
                or PurePosixPath(tr.implementation_reference).is_absolute() or PureWindowsPath(tr.implementation_reference).is_absolute()
                or any(getattr(tr, name).get("scope") != "fitting" for name in ("correlation", "feature_selection", "scaler"))
                or any(not getattr(tr, name) for name in ("normalization", "missing_values", "indicators", "correlation", "feature_selection", "scaler"))):
            add("invalid_transformations", "transformations", "Freeze effective raw inputs, feature order and complete recipe.")
        learning = r.learning
        if (not _text(learning.model_type) or not learning.effective_parameters or not learning.random_seeds
                or not _text(learning.training_policy) or not _text(learning.validation_policy)
                or any(not isinstance(key, str) for key in learning.effective_parameters)
                or type(learning.early_stopping.get("enabled")) is not bool
                or (learning.early_stopping.get("enabled") and (not p.validation
                    or type(learning.early_stopping.get("rounds")) is not int or learning.early_stopping["rounds"] < 1
                    or learning.early_stopping.get("information_scope") != "validation"))):
            add("invalid_learning", "learning", "Explicit effective model, validation and early-stopping recipe required.")
        last_used = max(max(positions[maturity[key]], row_positions[key]) for key in groups[0] + groups[2])
        if ([_key(x) for x in r.training_inputs.source_ids] != source[:last_used + 1]
                or not _sha(r.training_inputs.values_sha256)):
            add("invalid_training_footprint", "training_inputs", "Bind exactly the raw prefix used by fitting and validation, including maturity.")
        env = r.environment
        if (not _text(env.source_revision) or not _text(env.python_version) or not env.package_versions or not env.hardware_threads
                or any(not _text(key) or (value is not None and not _text(value)) for key, value in env.package_versions.items())
                or type(env.dirty_tree) is not bool or env.dirty_policy not in {"reject", "bind_digest"}
                or (env.dirty_tree and (env.dirty_policy != "bind_digest" or not _sha(env.source_digest)))
                or (env.source_digest is not None and not _sha(env.source_digest))
                or (env.dependency_lock_sha256 is not None and not _sha(env.dependency_lock_sha256))):
            add("invalid_environment", "environment", "Explicit code/environment and dirty-source policy required.")
        pr = r.provenance
        if (pr.artifact_evidence != "declared_pending_verification" or pr.point_in_time_status not in {"unknown", "caller_declared", "evidence_referenced"}
                or pr.execution_status not in {"unassessed", "unknown", "ineligible"}
                or pr.evidence_origin not in {"unknown", "historical_provider", "synthetic_fixture"}
                or (pr.point_in_time_status == "unknown" and not pr.missing_evidence)
                or (pr.point_in_time_status == "evidence_referenced" and not pr.historical_evidence_references)
                or any(not isinstance(item, str) or not item.strip() for item in pr.assumptions + pr.missing_evidence + pr.historical_evidence_references)):
            add("unsupported_provenance_claim", "provenance", "Contracts cannot certify historical validity or execution eligibility.")
    except ResearchProtocolError as exc:
        diagnostics.extend(exc.diagnostics)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        add("invalid_contract", "protocol", type(exc).__name__)
    return ProtocolValidationReport(tuple(diagnostics))


def _open_contained(root, locator):
    """Open relative to directory descriptors; reject every symlink component."""
    if not _locator(locator):
        _fail("unsafe_locator", "artifact", "A canonical relative locator is required.")
    if (any(not hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
            or os.open not in getattr(os, "supports_dir_fd", ())):
        _fail("unsupported_platform", "artifact", "Secure directory-relative file opening is unavailable.")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = PurePosixPath(locator).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            _fail("unsafe_artifact", "artifact", "Only regular local files are supported.")
        return os.fdopen(descriptor, "rb")
    finally:
        os.close(directory)


def validate_research_artifacts(protocol, research_root, *, _verified_consumer=None):
    """Verify a bounded-memory read image before Parquet decoding.

    Without an internal consumer, RAM is bounded by a 1 MiB copy buffer plus one
    Arrow batch. A private temporary
    file holds the verified read image and is deleted on exit. Original artifacts
    are never written, repaired or selected by modification time. Path opening
    requires POSIX dir_fd/O_NOFOLLOW; unsupported platforms fail closed.
    """
    declaration = validate_research_protocol(protocol)
    if not declaration.valid:
        return declaration
    diagnostics, verified = [], []
    try:
        if not Path(research_root).is_absolute():
            _fail("invalid_root", "research_root", "The trusted root must be explicit and absolute.")
        root = Path(research_root).resolve(strict=True)
        if not root.is_dir():
            _fail("invalid_root", "research_root", "Explicit trusted directory required.")
    except (OSError, ValueError, TypeError):
        return ProtocolValidationReport((ProtocolDiagnostic("invalid_root", "research_root", "Research root is unavailable."),))
    for artifact in protocol.artifacts:
        try:
            with _open_contained(root, artifact.relative_path) as source, tempfile.TemporaryFile() as image:
                digest, size = hashlib.sha256(), 0
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > artifact.byte_length:
                        _fail("byte_length_mismatch", artifact.artifact_id, "Artifact exceeds declared immutable length.")
                    digest.update(block)
                    image.write(block)
                if size != artifact.byte_length:
                    _fail("byte_length_mismatch", artifact.artifact_id, "Artifact length changed.")
                if digest.hexdigest() != artifact.sha256:
                    _fail("sha256_mismatch", artifact.artifact_id, "Artifact bytes changed.")
                image.seek(0)
                import pyarrow.parquet as pq
                parquet = pq.ParquetFile(image)
                schema = parquet.schema_arrow
                actual = tuple(ColumnSpec(field.name, str(field.type), field.nullable) for field in schema)
                if actual != artifact.columns or hashlib.sha256(schema.serialize().to_pybytes()).hexdigest() != artifact.schema_sha256:
                    _fail("schema_mismatch", artifact.artifact_id, "Original schema differs from its frozen declaration.")
                if parquet.metadata.num_rows != len(artifact.original_row_ids):
                    _fail("row_count_mismatch", artifact.artifact_id, "Original row count changed.")
                primary = artifact.role == "training_ohlcv"
                columns = [artifact.row_id_column]
                if artifact.label_column and artifact.label_column not in columns:
                    columns.append(artifact.label_column)
                if primary:
                    columns += [name for name in protocol.transformations.raw_input_columns if name not in columns]
                row_number, training_digest = 0, hashlib.sha256()
                training_keys = {_key(x) for x in protocol.training_inputs.source_ids}
                for batch in parquet.iter_batches(batch_size=4096, columns=columns):
                    for values in batch.to_pylist():
                        row_id = values[artifact.row_id_column]
                        if _key(row_id) != _key(artifact.original_row_ids[row_number]):
                            _fail("source_order_mismatch", artifact.artifact_id, "Artifact identity/order differs from its declaration.")
                        if primary:
                            observation = protocol.universe.observations[row_number]
                            label_name = artifact.label_column or artifact.row_id_column
                            if (_encode(values[label_name], opaque=True) != _encode(observation.original_label, opaque=True)
                                    or observation.index_dtype != str(schema.field(label_name).type)):
                                _fail("source_label_mismatch", artifact.artifact_id, "Original labels or their stored types differ from declarations.")
                        if primary and _key(row_id) in training_keys:
                            raw = tuple(values[name] for name in protocol.transformations.raw_input_columns)
                            training_digest.update((_json([_identifier(row_id), _encode(raw)]) + "\n").encode())
                        row_number += 1
                if primary and training_digest.hexdigest() != protocol.training_inputs.values_sha256:
                    _fail("training_values_mismatch", artifact.artifact_id, "Declared raw training footprint differs from verified bytes.")
                # Internal consumers materialize owned data before this verified
                # image closes. Nothing is returned by the reader on any failure.
                if _verified_consumer is not None:
                    _verified_consumer(artifact, parquet)
                verified.append(artifact.artifact_id)
        except ResearchProtocolError as exc:
            diagnostics.extend(exc.diagnostics)
        except FileNotFoundError:
            diagnostics.append(ProtocolDiagnostic("missing_artifact", artifact.artifact_id, "Explicit artifact is missing."))
        except OSError:
            diagnostics.append(ProtocolDiagnostic("unsafe_or_unreadable_artifact", artifact.artifact_id, "Artifact opening failed; no alternative source selected."))
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            diagnostics.append(ProtocolDiagnostic("artifact_decode_failure", artifact.artifact_id, type(exc).__name__))
    return ProtocolValidationReport(tuple(diagnostics), tuple(verified))


@dataclass(frozen=True)
class VerifiedResearchArtifact:
    """Owned immutable Arrow storage decoded from one verified private image.

    Physical columns, including explicit label columns, are returned with Arrow
    extension dtypes. Each public frame is a separate mutable copy; schema facts
    remain available through reference. No source-path reopen is performed.
    """
    reference: ResearchArtifactRef
    _table: object = field(repr=False)

    def to_pandas(self):
        return self._table.to_pandas(ignore_metadata=True, types_mapper=pd.ArrowDtype).copy(deep=True)


@dataclass(frozen=True)
class VerifiedResearchArtifacts:
    report: ProtocolValidationReport
    artifacts: tuple[VerifiedResearchArtifact, ...]


def read_verified_research_artifacts(protocol, research_root):
    """Return owned decoded images only after every artifact passes validation.

    Hashing uses bounded buffers, while materialized tables require memory
    proportional to decoded data. Verification describes the captured read
    image, not an atomic snapshot of a concurrently changing filesystem.
    """
    images = []
    def consume(reference, parquet):
        images.append(VerifiedResearchArtifact(reference, parquet.read()))
    report = validate_research_artifacts(protocol, research_root, _verified_consumer=consume)
    report.require_valid()
    return VerifiedResearchArtifacts(report, tuple(images))

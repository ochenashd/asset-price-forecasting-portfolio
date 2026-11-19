"""Read explicit local Parquet evidence without cleaning or execution certification.

Source bytes are read once: the hash and decoded observations refer to the same
buffer. Row ordinals distinguish physical observations within that artifact;
they are never exchange-session clocks or joins to independently filtered data.
All columns and duplicate rows survive. Date labels remain labels, not closes.

Artifact facts establish local file identity and decoded contents only. Provider,
acquisition, adjustment and asset claims are declarations, even when accompanied
by an evidence reference. No reference is fetched or independently authenticated.
Unknown fields stay unknown. No loader, database, network or engine is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SourceValue:
    """Immutable tagged scalar/nested value; missing sentinels are not imputed.

    Floating point values use hexadecimal representations, preserving signed
    zero and exact finite values. Schema and numpy tags preserve dtype evidence.
    to_python returns a fresh decoded value, never mutable snapshot storage.
    """

    kind: str
    text: str | None = None
    children: tuple[SourceValue, ...] = ()

    def to_python(self):
        if self.kind == "none":
            return None
        if self.kind == "pd.NA":
            return pd.NA
        if self.kind == "NaT":
            return pd.NaT
        if self.kind == "bool":
            return self.text == "true"
        if self.kind == "int":
            return int(self.text)
        if self.kind == "float":
            return float.fromhex(self.text)
        if self.kind == "str":
            return self.text
        if self.kind == "bytes":
            return bytes.fromhex(self.text)
        if self.kind == "decimal":
            return Decimal(self.text)
        if self.kind == "timestamp":
            result = pd.Timestamp(self.text)
            if self.children:
                result = result.tz_convert(self.children[0].text)
            return result
        if self.kind in {"datetime", "date", "time"}:
            constructor = {"datetime": datetime, "date": date, "time": time}[self.kind]
            return constructor.fromisoformat(self.text)
        if self.kind == "timedelta_ns":
            return pd.Timedelta(int(self.text), unit="ns")
        if self.kind == "timedelta_us":
            return timedelta(microseconds=int(self.text))
        if self.kind.startswith("numpy:"):
            return np.asarray(self.children[0].to_python(), dtype=self.kind[6:])[()]
        if self.kind in {"tuple", "list", "array"}:
            values = tuple(child.to_python() for child in self.children)
            return values if self.kind == "tuple" else list(values)
        if self.kind == "mapping":
            return {pair.children[0].to_python(): pair.children[1].to_python() for pair in self.children}
        raise ValueError(f"Unsupported encoded source value: {self.kind}")


def _freeze(value) -> SourceValue:
    if value is None:
        return SourceValue("none")
    if value is pd.NA:
        return SourceValue("pd.NA")
    if value is pd.NaT:
        return SourceValue("NaT")
    if isinstance(value, pd.Timestamp):
        zone_name = getattr(value.tz, "key", None) or getattr(value.tz, "zone", None)
        zone = (SourceValue("str", zone_name),) if zone_name else ()
        return SourceValue("timestamp", value.isoformat(), zone)
    if isinstance(value, pd.Timedelta):
        return SourceValue("timedelta_ns", str(value.value))
    if isinstance(value, np.generic):
        return SourceValue(f"numpy:{value.dtype}", children=(_freeze(value.item()),))
    if isinstance(value, bool):
        return SourceValue("bool", "true" if value else "false")
    if isinstance(value, int):
        return SourceValue("int", str(value))
    if isinstance(value, float):
        return SourceValue("float", value.hex())
    if isinstance(value, str):
        return SourceValue("str", value)
    if isinstance(value, bytes):
        return SourceValue("bytes", value.hex())
    if isinstance(value, Decimal):
        return SourceValue("decimal", str(value))
    if isinstance(value, (datetime, date, time)):
        return SourceValue(type(value).__name__, value.isoformat())
    if isinstance(value, timedelta):
        micros = (value.days * 86400 + value.seconds) * 1000000 + value.microseconds
        return SourceValue("timedelta_us", str(micros))
    if isinstance(value, Mapping):
        return SourceValue("mapping", children=tuple(
            SourceValue("pair", children=(_freeze(key), _freeze(item))) for key, item in value.items()))
    if isinstance(value, (tuple, list, np.ndarray)):
        kind = "tuple" if isinstance(value, tuple) else "array" if isinstance(value, np.ndarray) else "list"
        return SourceValue(kind, children=tuple(_freeze(item) for item in value))
    raise ValueError(f"Unsupported source value type; refusing lossy conversion: {type(value).__name__}")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a non-empty string.")


@dataclass(frozen=True)
class Declaration:
    """A caller claim, not verification; None means explicitly unknown.

    Caller inputs cannot promote themselves to artifact_verified. Evidence
    references are opaque declarations and are never opened by this reader.
    """

    value: SourceValue | object = None
    evidence_status: Literal["caller_declared", "unknown"] = "unknown"
    evidence_reference: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_status not in {"caller_declared", "unknown"}:
            raise ValueError("Only caller_declared or unknown provenance is accepted from callers.")
        if (self.value is None) != (self.evidence_status == "unknown"):
            raise ValueError("Unknown declarations require None; known values require caller_declared status.")
        if self.evidence_reference is not None:
            _identifier(self.evidence_reference, "evidence_reference")
        if isinstance(self.value, SourceValue):
            raise ValueError("Supply original declaration values, not pre-encoded source values.")
        object.__setattr__(self, "value", _freeze(self.value))


def _validate_declarations(instance) -> None:
    for descriptor in fields(instance):
        claim = getattr(instance, descriptor.name)
        if not isinstance(claim, Declaration):
            raise ValueError(f"{descriptor.name} requires an explicit Declaration.")


@dataclass(frozen=True)
class AssetMetadata:
    asset_id: Declaration = field(default_factory=Declaration)
    symbol: Declaration = field(default_factory=Declaration)
    instrument_type: Declaration = field(default_factory=Declaration)
    exchange: Declaration = field(default_factory=Declaration)
    timezone: Declaration = field(default_factory=Declaration)
    currency: Declaration = field(default_factory=Declaration)

    def __post_init__(self) -> None:
        _validate_declarations(self)
        for descriptor in fields(self):
            claim = getattr(self, descriptor.name)
            if claim.evidence_status != "unknown":
                _identifier(claim.value.to_python(), descriptor.name)


@dataclass(frozen=True)
class AcquisitionMetadata:
    provider: Declaration = field(default_factory=Declaration)
    acquired_at: Declaration = field(default_factory=Declaration)
    requested_start: Declaration = field(default_factory=Declaration)
    requested_end: Declaration = field(default_factory=Declaration)
    interval: Declaration = field(default_factory=Declaration)
    library_version: Declaration = field(default_factory=Declaration)
    auto_adjust: Declaration = field(default_factory=Declaration)
    repair: Declaration = field(default_factory=Declaration)
    actions_requested: Declaration = field(default_factory=Declaration)
    request_parameters: Declaration = field(default_factory=Declaration)

    def __post_init__(self) -> None:
        _validate_declarations(self)
        for name in ("provider", "interval", "library_version"):
            claim = getattr(self, name)
            if claim.evidence_status != "unknown":
                _identifier(claim.value.to_python(), name)
        for name in ("requested_start", "requested_end"):
            claim = getattr(self, name)
            if claim.evidence_status != "unknown" and not isinstance(claim.value.to_python(), (str, date)):
                raise ValueError(f"{name} requires a declared date label or datetime, not a constructed exchange clock.")
        if (self.request_parameters.evidence_status != "unknown"
                and not isinstance(self.request_parameters.value.to_python(), Mapping)):
            raise ValueError("request_parameters requires a declared mapping.")
        for name in ("auto_adjust", "repair", "actions_requested"):
            claim = getattr(self, name)
            if claim.evidence_status != "unknown" and type(claim.value.to_python()) is not bool:
                raise ValueError(f"{name} requires an explicit boolean declaration.")
        if self.acquired_at.evidence_status != "unknown":
            value = self.acquired_at.value.to_python()
            if (not isinstance(value, datetime) or value != value or value.tzinfo is None
                    or value.utcoffset() is None):
                raise ValueError("Known acquisition time requires an aware datetime, never an inferred filename clock.")


@dataclass(frozen=True)
class SnapshotDeclarations:
    asset: AssetMetadata = field(default_factory=AssetMetadata)
    acquisition: AcquisitionMetadata = field(default_factory=AcquisitionMetadata)
    price_adjustment_basis: Declaration = field(default_factory=Declaration)
    volume_adjustment_basis: Declaration = field(default_factory=Declaration)
    corporate_action_evidence: Declaration = field(default_factory=Declaration)
    corporate_action_eligibility: Declaration = field(default_factory=Declaration)

    def __post_init__(self) -> None:
        if not isinstance(self.asset, AssetMetadata) or not isinstance(self.acquisition, AcquisitionMetadata):
            raise ValueError("Asset and acquisition metadata require validated contracts.")
        for name in ("price_adjustment_basis", "volume_adjustment_basis", "corporate_action_evidence",
                     "corporate_action_eligibility"):
            claim = getattr(self, name)
            if not isinstance(claim, Declaration):
                raise ValueError(f"{name} requires a Declaration.")
            if claim.evidence_status != "unknown":
                _identifier(claim.value.to_python(), name)


@dataclass(frozen=True)
class AxisSchema:
    axis_type: str
    dtype: str
    names: tuple[SourceValue, ...]
    levels: tuple[tuple[SourceValue, ...], ...]
    level_dtypes: tuple[str, ...]
    codes: tuple[tuple[int, ...], ...]
    frequency: str | None


def _axis_schema(axis: pd.Index) -> AxisSchema:
    multi = isinstance(axis, pd.MultiIndex)
    return AxisSchema(type(axis).__name__, str(axis.dtype), tuple(_freeze(name) for name in axis.names),
                      tuple(tuple(_freeze(value) for value in level) for level in axis.levels) if multi else (),
                      tuple(str(level.dtype) for level in axis.levels) if multi else (),
                      tuple(tuple(int(code) for code in level) for level in axis.codes) if multi else (),
                      getattr(axis, "freqstr", None))


@dataclass(frozen=True)
class ColumnSchema:
    position: int
    label: SourceValue
    dtype: str
    categories: tuple[SourceValue, ...] = ()
    categories_ordered: bool | None = None


@dataclass(frozen=True)
class SourceRow:
    source_row_id: str
    ordinal: int
    session_label: SourceValue
    values: tuple[SourceValue, ...]


@dataclass(frozen=True)
class ArtifactFacts:
    """Locally verified bytes/schema; artifact_id remains a caller's logical alias."""

    artifact_id: str
    sha256: str
    byte_count: int
    file_format: Literal["parquet"]
    parquet_version: str
    arrow_schema: bytes
    index_schema: AxisSchema
    column_axis_schema: AxisSchema
    columns: tuple[ColumnSchema, ...]
    row_count: int
    evidence_status: Literal["artifact_verified"] = field(default="artifact_verified", init=False)
    artifact_id_evidence_status: Literal["caller_declared"] = field(default="caller_declared", init=False)


@dataclass(frozen=True)
class SnapshotDiagnostic:
    code: str
    message: str
    row_ordinals: tuple[int, ...] = ()
    column_positions: tuple[int, ...] = ()


def _canonical(value):
    if is_dataclass(value):
        return {descriptor.name: _canonical(getattr(value, descriptor.name))
                for descriptor in fields(value) if descriptor.name != "source_path"}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable source evidence, always uncertified for execution.

    source_path is a local locator, excluded from logical identity. artifact_id
    must be a stable caller-supplied identity, not automatically derived from a
    filename. Modification times are neither read nor hashed. No export to engine
    contracts occurs. A calendar and corporate-action eligibility assessment are
    still required, regardless of caller claims supplied here.
    """

    source_path: str
    artifact: ArtifactFacts
    declarations: SnapshotDeclarations
    rows: tuple[SourceRow, ...]
    diagnostics: tuple[SnapshotDiagnostic, ...]
    execution_ready: Literal[False] = field(default=False, init=False)
    provenance_independently_verified: Literal[False] = field(default=False, init=False)

    def canonical_json(self) -> str:
        return json.dumps({"schema_version": 1, "snapshot": _canonical(self)},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

    @property
    def snapshot_id(self) -> str:
        return "apf-market-v1-" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _missing(value) -> bool:
    if isinstance(value, (Mapping, tuple, list, np.ndarray)):
        return False
    return bool(pd.isna(value))


def _diagnostics(frame: pd.DataFrame, rows: tuple[SourceRow, ...],
                 declarations: SnapshotDeclarations) -> tuple[SnapshotDiagnostic, ...]:
    result = []
    if not rows:
        result.append(SnapshotDiagnostic("empty_artifact", "The source contains no observations."))
    groups: dict[SourceValue, list[int]] = {}
    for row in rows:
        groups.setdefault(row.session_label, []).append(row.ordinal)
    for ordinals in groups.values():
        if len(ordinals) > 1:
            result.append(SnapshotDiagnostic("duplicate_session_labels", "Source session labels are duplicated.",
                                             tuple(ordinals)))
            if any(rows[index].values != rows[ordinals[0]].values for index in ordinals[1:]):
                result.append(SnapshotDiagnostic("conflicting_duplicate_observations",
                                                 "Duplicate labels contain different source values.", tuple(ordinals)))
    missing_labels = tuple(i for i, label in enumerate(frame.index) if _missing(label))
    if missing_labels:
        result.append(SnapshotDiagnostic("missing_session_labels", "Original session labels are missing.", missing_labels))

    multi = isinstance(frame.columns, pd.MultiIndex)
    names = [str(name).lower() for name in frame.columns.names]
    price_level = names.index("price") if names.count("price") == 1 else None
    ticker_level = names.index("ticker") if names.count("ticker") == 1 else None
    if multi:
        if ticker_level is None or price_level is None or frame.columns.nlevels != 2:
            result.append(SnapshotDiagnostic("ambiguous_ticker_identities",
                                             "MultiIndex needs explicit Price/Ticker levels for quality interpretation; columns are preserved."))
        elif len(set(frame.columns.get_level_values(ticker_level))) != 1:
            result.append(SnapshotDiagnostic("ambiguous_ticker_identities",
                                             "More than one source ticker is present; no asset is selected."))
        elif declarations.asset.symbol.evidence_status != "unknown":
            if declarations.asset.symbol.value.to_python() != frame.columns.get_level_values(ticker_level)[0]:
                result.append(SnapshotDiagnostic("declared_symbol_mismatch", "Declared symbol differs from source ticker level."))
    interpreted: dict[SourceValue, dict[str, list[int]]] = {}
    for position, label in enumerate(frame.columns):
        if multi and (price_level is None or ticker_level is None or frame.columns.nlevels != 2):
            continue
        name = label[price_level] if multi else label
        ticker = _freeze(label[ticker_level]) if multi else _freeze(None)
        normalized = str(name).lower().strip().replace(" ", "_")
        interpreted.setdefault(ticker, {}).setdefault(normalized, []).append(position)
    if not multi and not interpreted:
        interpreted[_freeze(None)] = {}
    for columns in interpreted.values():
        for name in ("open", "high", "low", "close", "volume"):
            positions = columns.get(name, [])
            if not positions:
                result.append(SnapshotDiagnostic("missing_ohlcv_fields", f"Source lacks required OHLCV field: {name}."))
            elif len(positions) > 1:
                result.append(SnapshotDiagnostic("ambiguous_ohlcv_fields", f"Multiple source columns match {name}.",
                                                 column_positions=tuple(positions)))
            for position in positions:
                missing = tuple(i for i in range(len(frame)) if _missing(frame.iat[i, position]))
                if missing:
                    result.append(SnapshotDiagnostic("missing_ohlcv", f"Missing source {name} values are preserved.",
                                                     missing, (position,)))
        repair_columns = columns.get("repaired?", []) + columns.get("repaired", [])
        if not repair_columns:
            result.append(SnapshotDiagnostic("unknown_repair_status", "No source repair flag is present."))
        for position in repair_columns:
            repaired, unknown = [], []
            for i in range(len(frame)):
                value = frame.iat[i, position]
                if isinstance(value, (bool, np.bool_)):
                    if value:
                        repaired.append(i)
                else:
                    unknown.append(i)
            if repaired:
                result.append(SnapshotDiagnostic("repaired_observations", "Source marks observations as repaired.",
                                                 tuple(repaired), (position,)))
            if unknown:
                result.append(SnapshotDiagnostic("unknown_repair_status", "Missing or non-boolean repair flags are not verified observations.",
                                                 tuple(unknown), (position,)))
    for descriptor in fields(declarations.acquisition):
        if getattr(declarations.acquisition, descriptor.name).evidence_status == "unknown":
            result.append(SnapshotDiagnostic("missing_acquisition_metadata", f"Unknown acquisition field: {descriptor.name}."))
    for descriptor in fields(declarations.asset):
        if getattr(declarations.asset, descriptor.name).evidence_status == "unknown":
            result.append(SnapshotDiagnostic("missing_asset_metadata", f"Unknown asset field: {descriptor.name}."))
    for name in ("price_adjustment_basis", "volume_adjustment_basis"):
        claim = getattr(declarations, name)
        if claim.evidence_status == "unknown" or claim.value.to_python() == "unknown":
            result.append(SnapshotDiagnostic("unknown_adjustment_basis", f"Unknown {name}."))
    if declarations.corporate_action_evidence.evidence_status == "unknown":
        result.append(SnapshotDiagnostic("missing_corporate_action_evidence",
                                         "Action columns alone do not prove complete corporate-action coverage or eligibility."))
    return tuple(result)


def read_market_snapshot(path: str | Path, *, artifact_id: str, expected_sha256: str | None = None,
                         declarations: SnapshotDeclarations | None = None) -> MarketSnapshot:
    """Read only this local Parquet file; preserve data issues as diagnostics.

    Hash mismatches, undecodable Parquet, unsupported scalar types and invalid
    declaration contracts fail explicitly. Market-quality issues are retained,
    not cleaned. This function performs no writes and consults no configuration.
    """
    _identifier(artifact_id, "artifact_id")
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in expected_sha256)):
            raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters.")
    declarations = SnapshotDeclarations() if declarations is None else declarations
    if not isinstance(declarations, SnapshotDeclarations):
        raise ValueError("declarations requires SnapshotDeclarations.")
    source = Path(path)
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("Source SHA-256 mismatch; artifact rejected before decoding.")
    parquet = pq.ParquetFile(BytesIO(payload))
    frame = pd.read_parquet(BytesIO(payload), engine="pyarrow")
    columns = tuple(ColumnSchema(
        position, _freeze(label), str(frame.dtypes.iloc[position]),
        tuple(_freeze(value) for value in frame.iloc[:, position].cat.categories)
        if isinstance(frame.dtypes.iloc[position], pd.CategoricalDtype) else (),
        bool(frame.iloc[:, position].cat.ordered)
        if isinstance(frame.dtypes.iloc[position], pd.CategoricalDtype) else None,
    ) for position, label in enumerate(frame.columns))
    rows = tuple(SourceRow(f"{digest}:{ordinal}", ordinal, _freeze(frame.index[ordinal]),
                           tuple(_freeze(frame.iat[ordinal, column]) for column in range(len(frame.columns))))
                 for ordinal in range(len(frame)))
    artifact = ArtifactFacts(artifact_id, digest, len(payload), "parquet", parquet.metadata.format_version,
                             parquet.schema_arrow.serialize().to_pybytes(), _axis_schema(frame.index),
                             _axis_schema(frame.columns), columns, len(frame))
    return MarketSnapshot(str(source.absolute()), artifact, declarations, rows,
                          _diagnostics(frame, rows, declarations))

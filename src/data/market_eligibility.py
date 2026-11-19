"""Pure offline eligibility checks; no loader, file access, engine or provider calls.

Trust boundary: an external evidence issuer must authenticate a versioned bundle
bound to the exact reader snapshot. HMAC-SHA256 verifies authenticity against
separately supplied trusted keys, never a boolean supplied in the evidence.
Keys belong to trusted application configuration, not an untrusted submitter.
This authenticates the issuer's statement, not its factual accuracy. Production
trust roots and source attestations are NOT supplied by APF. Synthetic keys in
tests model this boundary; they do not certify real-world provider data.

V1 requires the snapshot and supplied calendar to cover the same explicit window
exactly. Row/session mappings and column positions are signed, never reconstructed
from positions or naive date labels. Calendar schedules, holidays and action
coverage rely on the trusted issuer's evidence; internal consistency is checked
locally. Every source observation and diagnostic remains in the original snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import hmac
import json
import math
from numbers import Real
from typing import Literal, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from .market_snapshot import AcquisitionMetadata, MarketSnapshot, SnapshotDiagnostic
from src.evaluation.execution.adapter import AssetSpec, ObservedMarketBar, PriceProvenance, SessionCalendar, TradingSession
from src.evaluation.execution.contracts import MarketBar
from src.evaluation.execution.prediction_export import _utc_instant


FIELDS = ("open", "high", "low", "close", "volume")
Status = Literal["eligible", "ineligible", "unknown"]


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a non-empty identifier.")


def _aware(value):
    return (isinstance(value, datetime) and value == value and value.tzinfo is not None
            and value.utcoffset() is not None)


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _unknown(value):
    return value is None or (isinstance(value, str) and value.strip() in {"", "unknown"})


@dataclass(frozen=True)
class FieldObservation:
    """Signed per-field source lineage; None explicitly means unknown."""

    name: str
    column_position: int
    observed: bool | None
    repaired: bool | None
    filled: bool | None

    def __post_init__(self):
        if self.name not in FIELDS or type(self.column_position) is not int or self.column_position < 0:
            raise ValueError("Field observation requires a named OHLCV field and nonnegative column position.")
        if any(value is not None and type(value) is not bool for value in (self.observed, self.repaired, self.filled)):
            raise ValueError("Observation flags must be explicit booleans or None.")


@dataclass(frozen=True)
class RowObservation:
    source_row_id: str
    session_id: str
    fields: tuple[FieldObservation, ...]

    def __post_init__(self):
        _identifier(self.source_row_id, "source_row_id")
        _identifier(self.session_id, "session_id")
        object.__setattr__(self, "fields", tuple(self.fields))
        if not all(isinstance(item, FieldObservation) for item in self.fields):
            raise ValueError("Row observations require FieldObservation records.")


@dataclass(frozen=True)
class CorporateAction:
    kind: str
    effective_at: datetime

    def __post_init__(self):
        _identifier(self.kind, "action kind")
        if not _aware(self.effective_at):
            raise ValueError("Action effective time must be explicit and timezone-aware.")


@dataclass(frozen=True)
class CorporateActionCoverage:
    asset_id: str
    source_id: str
    reference_id: str
    covered_from: datetime
    covered_through: datetime
    complete: bool | None
    events: tuple[CorporateAction, ...] = ()

    def __post_init__(self):
        for name in ("asset_id", "source_id", "reference_id"):
            _identifier(getattr(self, name), name)
        if (not _aware(self.covered_from) or not _aware(self.covered_through)
                or _utc_instant(self.covered_from) > _utc_instant(self.covered_through)):
            raise ValueError("Action coverage requires ordered aware interval endpoints.")
        if self.complete is not None and type(self.complete) is not bool:
            raise ValueError("Action coverage completeness must be a boolean or None.")
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(item, CorporateAction) for item in self.events):
            raise ValueError("Action events require CorporateAction records.")


@dataclass(frozen=True)
class MarketEligibilityEvidence:
    """One external statement covering every required scope, not an engine input.

    A source issuer may reference a separate authoritative calendar/action source.
    Those references are retained but never downloaded. Acquisition values remain
    caller-declared in SnapshotDeclarations; authentication of this separate bundle
    is reported separately and never rewrites the snapshot's evidence statuses.
    """

    artifact_id: str
    source_sha256: str
    snapshot_id: str
    provider_id: str
    asset: AssetSpec | None
    listing_exchange: str | None
    exchange_timezone: str | None
    acquisition: AcquisitionMetadata | None
    calendar: SessionCalendar | None
    calendar_exchange: str | None
    start_session_id: str | None
    end_session_id: str | None
    observations: tuple[RowObservation, ...]
    price_adjustment_basis: str | None
    volume_adjustment_basis: str | None
    corporate_actions: CorporateActionCoverage | None

    def __post_init__(self):
        for name in ("artifact_id", "snapshot_id", "provider_id"):
            _identifier(getattr(self, name), name)
        if not _sha256(self.source_sha256):
            raise ValueError("Evidence source_sha256 must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "observations", tuple(self.observations))
        if not all(isinstance(item, RowObservation) for item in self.observations):
            raise ValueError("Evidence observations require immutable RowObservation records.")
        for value, expected in ((self.asset, AssetSpec), (self.acquisition, AcquisitionMetadata),
                                (self.calendar, SessionCalendar), (self.corporate_actions, CorporateActionCoverage)):
            if value is not None and not isinstance(value, expected):
                raise ValueError("Evidence requires the existing validated metadata contract types.")
        if self.calendar is not None and not all(isinstance(item, TradingSession) for item in self.calendar.sessions):
            raise ValueError("Calendar sessions require existing TradingSession records.")
        if self.calendar is not None:
            for value in (self.calendar.calendar_id, self.calendar.source_id,
                          *(item.session_id for item in self.calendar.sessions)):
                if value is not None and not isinstance(value, str):
                    raise ValueError("Calendar identities must be strings or explicitly missing.")
            if any(clock is not None and not isinstance(clock, datetime)
                   for item in self.calendar.sessions for clock in (item.open_timestamp, item.close_timestamp, item.next_open_timestamp)):
                raise ValueError("Calendar clocks must be datetime values or explicitly missing.")
        if self.asset is not None and any(value is not None and not isinstance(value, str)
                                           for value in (self.asset.asset_id, self.asset.symbol, self.asset.instrument_type,
                                                         self.asset.calendar_id, self.asset.quote_currency)):
            raise ValueError("Asset identity fields must be strings or explicitly missing.")
        for name in ("listing_exchange", "exchange_timezone", "calendar_exchange", "start_session_id",
                     "end_session_id", "price_adjustment_basis", "volume_adjustment_basis"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string or explicitly missing.")


def _canonical(value):
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, datetime):
        if not _aware(value):
            # Invalid calendar times still need to be signable for diagnostics;
            # they are never interpreted as exchange-local timestamps.
            return {"naive_datetime": value.isoformat()}
        return {"utc_datetime": value.astimezone(timezone.utc).isoformat()}
    return value


def attestation_message(evidence: MarketEligibilityEvidence, *, issuer_id: str, reference_id: str) -> bytes:
    """Canonical bytes for an offline issuer to sign; keys are never serialized."""
    _identifier(issuer_id, "issuer_id")
    _identifier(reference_id, "reference_id")
    if not isinstance(evidence, MarketEligibilityEvidence):
        raise ValueError("A MarketEligibilityEvidence payload is required.")
    return json.dumps({"schema_version": 1, "issuer_id": issuer_id, "reference_id": reference_id,
                       "evidence": _canonical(evidence)}, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class EvidenceAttestation:
    issuer_id: str
    reference_id: str
    evidence: MarketEligibilityEvidence
    signature: str | None = None

    def __post_init__(self):
        attestation_message(self.evidence, issuer_id=self.issuer_id, reference_id=self.reference_id)


@dataclass(frozen=True)
class EligibilityReason:
    code: str
    outcome: Literal["unknown", "ineligible"]
    message: str
    evidence_status: Literal["artifact_verified", "authenticated_external", "caller_declared", "unknown"]
    session_id: str | None = None
    source_row_id: str | None = None
    field: str | None = None


@dataclass(frozen=True)
class EvidenceReference:
    source_id: str
    reference_id: str
    evidence_status: Literal["artifact_verified", "authenticated_external", "caller_declared", "unknown"]
    content_sha256: str | None = None


@dataclass(frozen=True)
class EligibilityReport:
    status: Status
    snapshot: MarketSnapshot
    reasons: tuple[EligibilityReason, ...]
    evidence_references: tuple[EvidenceReference, ...]
    authenticated_external_evidence: bool
    market: tuple[ObservedMarketBar, ...]
    asset: AssetSpec | None
    calendar: SessionCalendar | None

    @property
    def bars(self) -> tuple[MarketBar, ...]:
        return tuple(item.bar for item in self.market)

    @property
    def snapshot_diagnostics(self) -> tuple[SnapshotDiagnostic, ...]:
        return self.snapshot.diagnostics


def validate_market_eligibility(snapshot: MarketSnapshot, attestation: EvidenceAttestation | None = None, *,
                                trusted_keys: Mapping[str, bytes] | None = None) -> EligibilityReport:
    """Return eligible only for complete, consistent, authenticated evidence.

    Ineligible defects take priority over missing evidence. Unknown/ineligible
    reports contain no market bars, including when other rows are valid. No file
    is reopened: the reader's source digest is matched to the signed digest and
    full snapshot ID. The caller must use snapshots captured by the reader.
    """
    if not isinstance(snapshot, MarketSnapshot):
        raise ValueError("A reader-produced immutable MarketSnapshot is required.")
    if attestation is not None and not isinstance(attestation, EvidenceAttestation):
        raise ValueError("attestation requires EvidenceAttestation or None.")
    reasons = []
    references = [EvidenceReference("local_snapshot_reader", snapshot.artifact.artifact_id,
                                    "artifact_verified", snapshot.artifact.sha256)]

    def add(code, outcome, message, status="unknown", session=None, row=None, field=None):
        reasons.append(EligibilityReason(code, outcome, message, status, session, row, field))

    def compare(claim, value, name):
        original = claim.value.to_python()
        if (claim.evidence_status != "unknown" and not _unknown(value)
                and not _unknown(original) and original != value):
            add("declaration_conflict", "ineligible", f"Snapshot declaration conflicts with external {name}.",
                "caller_declared", field=name)

    for group in (snapshot.declarations.asset, snapshot.declarations.acquisition, snapshot.declarations):
        for descriptor in fields(group):
            claim = getattr(group, descriptor.name)
            if getattr(claim, "evidence_reference", None):
                references.append(EvidenceReference("snapshot_caller", claim.evidence_reference, "caller_declared"))
    for diagnostic in snapshot.diagnostics:
        if diagnostic.code in {"duplicate_session_labels", "conflicting_duplicate_observations", "missing_ohlcv",
                                "missing_ohlcv_fields", "repaired_observations", "ambiguous_ohlcv_fields",
                                "ambiguous_ticker_identities", "missing_session_labels", "empty_artifact"}:
            add(diagnostic.code, "ineligible", diagnostic.message, "artifact_verified")
    if not _sha256(snapshot.artifact.sha256) or snapshot.artifact.row_count != len(snapshot.rows):
        add("invalid_artifact_facts", "ineligible", "Snapshot source facts are inconsistent.", "artifact_verified")
    for ordinal, row in enumerate(snapshot.rows):
        if (row.ordinal != ordinal or row.source_row_id != f"{snapshot.artifact.sha256}:{ordinal}"
                or len(row.values) != len(snapshot.artifact.columns)):
            add("invalid_row_identity", "ineligible", "Source row identity or width is inconsistent.", "artifact_verified", row=row.source_row_id)

    authenticated = False
    evidence = None if attestation is None else attestation.evidence
    evidence_status = "unknown" if evidence is None else "caller_declared"
    if attestation is None:
        for scope in ("asset", "acquisition", "calendar", "observations", "adjustments", "corporate_actions"):
            add("missing_" + scope + "_evidence", "unknown", f"Missing externally authenticated {scope} evidence.")
    else:
        message = attestation_message(evidence, issuer_id=attestation.issuer_id, reference_id=attestation.reference_id)
        key = None if trusted_keys is None else trusted_keys.get(attestation.issuer_id)
        if key is not None and (not isinstance(key, bytes) or len(key) < 32):
            raise ValueError("Trusted HMAC keys must be configured separately and contain at least 32 bytes.")
        if key is None:
            add("untrusted_issuer", "unknown", "No trusted verification key is configured for this issuer.", "caller_declared")
        elif attestation.signature is None:
            add("missing_authentication", "unknown", "An unsigned caller declaration is not authenticated evidence.", "caller_declared")
        elif (not _sha256(attestation.signature)
              or not hmac.compare_digest(hmac.new(key, message, hashlib.sha256).hexdigest(), attestation.signature)):
            add("invalid_authentication", "ineligible", "External evidence authentication failed.", "caller_declared")
        else:
            authenticated = True
            evidence_status = "authenticated_external"
        references.append(EvidenceReference(attestation.issuer_id, attestation.reference_id, evidence_status,
                                            hashlib.sha256(message).hexdigest()))
        if evidence.artifact_id != snapshot.artifact.artifact_id or evidence.source_sha256 != snapshot.artifact.sha256:
            add("source_identity_mismatch", "ineligible", "Evidence does not identify these exact source bytes and artifact.", evidence_status)
        if evidence.snapshot_id != snapshot.snapshot_id:
            add("snapshot_identity_mismatch", "ineligible", "Evidence does not bind the exact observations and snapshot declarations.", evidence_status)

    asset = None if evidence is None else evidence.asset
    calendar = None if evidence is None else evidence.calendar
    if evidence is not None:
        if asset is None:
            add("missing_asset_evidence", "unknown", "Stable asset identity is missing.", evidence_status)
        else:
            for name in ("asset_id", "symbol", "calendar_id", "quote_currency"):
                if not isinstance(getattr(asset, name), str) or getattr(asset, name).strip() in {"", "unknown"}:
                    add("missing_asset_identity", "unknown", f"Missing asset {name}.", evidence_status, field=name)
            if asset.instrument_type in {None, "", "unknown"}:
                add("missing_instrument_type", "unknown", "Instrument type is unknown.", evidence_status)
            elif asset.instrument_type not in {"equity", "etf"}:
                add("unsupported_instrument", "ineligible", "Only equity/ETF execution is supported.", evidence_status)
            for name, value in (("asset_id", asset.asset_id), ("symbol", asset.symbol),
                                ("instrument_type", asset.instrument_type), ("currency", asset.quote_currency)):
                compare(getattr(snapshot.declarations.asset, name), value, name)
        for name in ("listing_exchange", "exchange_timezone"):
            value = getattr(evidence, name)
            if not isinstance(value, str) or value.strip() in {"", "unknown"}:
                add("missing_asset_evidence", "unknown", f"Missing {name}.", evidence_status, field=name)
        if not _unknown(evidence.exchange_timezone):
            try:
                ZoneInfo(evidence.exchange_timezone)
            except (ZoneInfoNotFoundError, ValueError, TypeError):
                add("invalid_exchange_timezone", "ineligible", "Exchange timezone must be a valid IANA zone.", evidence_status)
        compare(snapshot.declarations.asset.exchange, evidence.listing_exchange, "exchange")
        compare(snapshot.declarations.asset.timezone, evidence.exchange_timezone, "timezone")
        acquisition = evidence.acquisition
        if acquisition is None:
            add("missing_acquisition_evidence", "unknown", "Acquisition metadata is missing.", evidence_status)
        else:
            for descriptor in fields(acquisition):
                claim = getattr(acquisition, descriptor.name)
                value = claim.value.to_python()
                if claim.evidence_status == "unknown" or (isinstance(value, str) and value.strip() in {"", "unknown"}):
                    add("missing_acquisition_metadata", "unknown", f"Unknown acquisition {descriptor.name}.", evidence_status, field=descriptor.name)
                else:
                    compare(getattr(snapshot.declarations.acquisition, descriptor.name), claim.value.to_python(), descriptor.name)
            if not _unknown(acquisition.provider.value.to_python()) and acquisition.provider.value.to_python() != evidence.provider_id:
                add("provider_mismatch", "ineligible", "Acquisition and evidence provider identities differ.", evidence_status)
            if not _unknown(acquisition.interval.value.to_python()) and acquisition.interval.value.to_python() != "1d":
                add("unsupported_interval", "ineligible", "Only daily session OHLCV is supported.", evidence_status)
            if acquisition.auto_adjust.value.to_python() is True:
                add("adjusted_acquisition", "ineligible", "Auto-adjusted acquisition cannot supply unadjusted fills.", evidence_status)
        for name, allowed in (("price_adjustment_basis", "unadjusted"),
                              ("volume_adjustment_basis", "unadjusted_share_count")):
            value = getattr(evidence, name)
            if value is None or value.strip() in {"", "unknown"}:
                add("unknown_adjustment_basis", "unknown", f"Unknown {name}.", evidence_status, field=name)
            elif value != allowed:
                add("incompatible_adjustment_basis", "ineligible", f"{name} must be {allowed}.", evidence_status, field=name)
            compare(getattr(snapshot.declarations, name), value, name)

    sessions = ()
    if evidence is not None and calendar is None:
        add("missing_calendar_evidence", "unknown", "An independent complete calendar is required.", evidence_status)
    elif calendar is not None:
        if _unknown(calendar.source_id) or _unknown(calendar.calendar_id) or not calendar.sessions:
            add("missing_calendar_evidence", "unknown", "Calendar identity or sessions are missing.", evidence_status)
        if calendar.declared_complete is not True:
            add("incomplete_calendar", "unknown", "Calendar completeness is not established.", evidence_status)
        if asset is not None and not _unknown(asset.calendar_id) and calendar.calendar_id != asset.calendar_id:
            add("calendar_asset_mismatch", "ineligible", "Asset and calendar IDs differ.", evidence_status)
        if _unknown(evidence.calendar_exchange):
            add("missing_calendar_exchange", "unknown", "Calendar exchange identity is missing.", evidence_status)
        elif evidence.calendar_exchange != evidence.listing_exchange and evidence.listing_exchange:
            add("calendar_exchange_mismatch", "ineligible", "Calendar does not identify the listing exchange.", evidence_status)
        valid_times = True
        identifiers = set()
        for session in calendar.sessions:
            if not session.session_id or session.session_id in identifiers:
                add("duplicate_calendar_session", "ineligible", "Calendar session IDs must be unique and nonmissing.", evidence_status)
            identifiers.add(session.session_id)
            clocks = (session.open_timestamp, session.close_timestamp, session.next_open_timestamp)
            if not all(_aware(value) for value in clocks):
                valid_times = False
                add("invalid_calendar_timestamp", "ineligible", "Calendar clocks must be timezone-aware.", evidence_status, session.session_id)
            elif not _utc_instant(session.open_timestamp) < _utc_instant(session.close_timestamp) < _utc_instant(session.next_open_timestamp):
                valid_times = False
                add("invalid_calendar_timing", "ineligible", "Require open < close < independently declared next open.", evidence_status, session.session_id)
            elif any(clock.astimezone(timezone.utc).astimezone(clock.tzinfo).replace(tzinfo=None)
                     != clock.replace(tzinfo=None) for clock in clocks):
                valid_times = False
                add("nonexistent_local_time", "ineligible", "Calendar clock fails timezone round-trip validation.", evidence_status, session.session_id)
        if valid_times:
            sessions = tuple(sorted(calendar.sessions, key=lambda item: _utc_instant(item.open_timestamp)))
            for previous, current in zip(sessions, sessions[1:]):
                if _utc_instant(previous.next_open_timestamp) != _utc_instant(current.open_timestamp):
                    add("calendar_gap_or_overlap", "ineligible", "Calendar adjacency does not match its next-opening cutoff.", evidence_status, current.session_id)
            if _unknown(evidence.start_session_id) or _unknown(evidence.end_session_id):
                add("missing_evaluation_window", "unknown", "Evaluation session endpoints must be explicitly supplied.", evidence_status)
            elif sessions and (evidence.start_session_id != sessions[0].session_id or evidence.end_session_id != sessions[-1].session_id):
                add("window_mismatch", "ineligible", "Explicit evaluation endpoints must match the supplied complete calendar.", evidence_status)
        references.append(EvidenceReference(calendar.source_id, calendar.calendar_id, evidence_status))

    if evidence is not None:
        coverage = evidence.corporate_actions
        if coverage is None:
            add("missing_corporate_action_evidence", "unknown", "Action columns are not proof of complete action coverage.", evidence_status)
        else:
            references.append(EvidenceReference(coverage.source_id, coverage.reference_id, evidence_status))
            if asset is not None and not _unknown(asset.asset_id) and coverage.asset_id != asset.asset_id:
                add("action_asset_mismatch", "ineligible", "Corporate-action evidence identifies another asset.", evidence_status)
            if coverage.complete is not True:
                add("incomplete_corporate_action_coverage", "unknown", "Corporate-action coverage is incomplete or unknown.", evidence_status)
            if sessions and (_utc_instant(coverage.covered_from) > _utc_instant(sessions[0].open_timestamp)
                             or _utc_instant(coverage.covered_through) < _utc_instant(sessions[-1].close_timestamp)):
                add("incomplete_corporate_action_coverage", "unknown", "Corporate-action evidence does not cover the entire evaluation interval.", evidence_status)
            for event in coverage.events:
                if not _utc_instant(coverage.covered_from) <= _utc_instant(event.effective_at) <= _utc_instant(coverage.covered_through):
                    add("inconsistent_action_evidence", "ineligible", "An action lies outside its evidence coverage.", evidence_status)
                if sessions and _utc_instant(sessions[0].open_timestamp) <= _utc_instant(event.effective_at) <= _utc_instant(sessions[-1].close_timestamp):
                    add("unsupported_corporate_action", "ineligible", "Corporate actions require accounting absent from this engine.", evidence_status)
            compare(snapshot.declarations.corporate_action_eligibility, "eligible", "corporate_action_eligibility")
        if (sessions and evidence.acquisition is not None
                and evidence.acquisition.acquired_at.evidence_status != "unknown"
                and _utc_instant(evidence.acquisition.acquired_at.value.to_python()) < _utc_instant(sessions[-1].close_timestamp)):
            add("acquisition_precedes_observations", "ineligible", "Acquisition cannot predate the last complete session close.", evidence_status)

    rows = {row.source_row_id: row for row in snapshot.rows}
    aligned = {}
    seen_rows = set()
    column_map = {}
    names = [str(name.to_python()).lower() for name in snapshot.artifact.column_axis_schema.names]
    multi = snapshot.artifact.column_axis_schema.axis_type == "MultiIndex"
    for observation in (() if evidence is None else evidence.observations):
        row = rows.get(observation.source_row_id)
        if row is None:
            add("unknown_source_row", "ineligible", "Evidence references a row outside this artifact.", evidence_status, observation.session_id, observation.source_row_id)
            continue
        if row.source_row_id in seen_rows or observation.session_id in aligned:
            add("duplicate_session_alignment", "ineligible", "Each source row and session require exactly one mapping.", evidence_status, observation.session_id, row.source_row_id)
        seen_rows.add(row.source_row_id)
        aligned[observation.session_id] = observation
        provided = {item.name for item in observation.fields}
        if provided != set(FIELDS):
            add("missing_observation_flags", "unknown", "Every OHLCV field needs explicit observation lineage.", evidence_status, observation.session_id, row.source_row_id)
        if len(provided) != len(observation.fields):
            add("duplicate_field_evidence", "ineligible", "Field evidence is duplicated.", evidence_status, observation.session_id, row.source_row_id)
        for item in observation.fields:
            if item.observed is None or item.repaired is None or item.filled is None:
                add("unknown_observation_status", "unknown", "Observed/repaired/filled flags must all be known.", evidence_status, observation.session_id, row.source_row_id, item.name)
            if item.observed is False or item.repaired is True or item.filled is True:
                add("unobserved_or_repaired_field", "ineligible", "OHLCV must be observed and never repaired or filled.", evidence_status, observation.session_id, row.source_row_id, item.name)
            if item.column_position >= len(snapshot.artifact.columns):
                add("invalid_column_mapping", "ineligible", "Field column position is outside the source schema.", evidence_status, observation.session_id, row.source_row_id, item.name)
                continue
            label = snapshot.artifact.columns[item.column_position].label.to_python()
            if multi:
                if len(names) != 2 or names.count("price") != 1 or names.count("ticker") != 1:
                    add("ambiguous_column_mapping", "ineligible", "MultiIndex interpretation requires exact named Price/Ticker levels.", evidence_status)
                    continue
                if asset is not None and not _unknown(asset.symbol) and label[names.index("ticker")] != asset.symbol:
                    add("source_ticker_mismatch", "ineligible", "Source ticker differs from authenticated asset.", evidence_status)
                label = label[names.index("price")]
            if not isinstance(label, str) or label.strip().lower() != item.name:
                add("invalid_column_mapping", "ineligible", "Signed field positions must identify the corresponding OHLCV columns.", evidence_status, observation.session_id, row.source_row_id, item.name)
            if item.name in column_map and column_map[item.name] != item.column_position:
                add("inconsistent_column_mapping", "ineligible", "Field schema must remain identical across source rows.", evidence_status)
            column_map[item.name] = item.column_position
    if evidence is not None and seen_rows != set(rows):
        add("missing_row_alignment", "unknown", "Every snapshot row requires an explicit source-to-session mapping.", evidence_status)
    expected_sessions = {session.session_id for session in sessions}
    if sessions:
        for missing in sorted(expected_sessions - aligned.keys()):
            add("missing_expected_session", "ineligible", "An expected market session has no observation.", evidence_status, missing)
        for extra in sorted(aligned.keys() - expected_sessions):
            add("unexpected_session", "ineligible", "Source session is outside the explicit evaluation calendar.", evidence_status, extra)

    # Check source flags/actions independently of declarations and signed lineage.
    for column in snapshot.artifact.columns:
        label = column.label.to_python()
        if multi and "price" in names:
            label = label[names.index("price")]
        if not isinstance(label, str):
            continue
        label = label.lower().strip().replace(" ", "_")
        for row in snapshot.rows:
            if column.position >= len(row.values):
                continue
            value = row.values[column.position].to_python()
            if label in {"dividends", "stock_splits", "capital_gains"} and isinstance(value, (Real, Decimal)) and value != 0 and math.isfinite(value):
                add("source_corporate_action", "ineligible", "Source reports a corporate action unsupported by the engine.", "artifact_verified", row=row.source_row_id)
            elif label in {"dividends", "stock_splits", "capital_gains"} and (
                    isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal)) or not math.isfinite(value)):
                add("unknown_source_action_value", "unknown", "Source action information is missing or uninterpretable.", "artifact_verified", row=row.source_row_id)
            flag_field = next((name for name in FIELDS if label in {name + "_observed", name + "_filled", name + "_repaired"}), None)
            global_flag = label in {"observed", "observed?", "repaired", "repaired?", "filled", "filled?", "synthetic", "synthetic?"}
            if flag_field or global_flag:
                if not isinstance(value, (bool, np.bool_)):
                    add("unknown_source_quality_flag", "unknown", "A source quality flag is missing or not boolean.", "artifact_verified", row=row.source_row_id, field=flag_field)
                elif ((label.endswith("_observed") or label in {"observed", "observed?"}) and not value) or (
                        not (label.endswith("_observed") or label in {"observed", "observed?"}) and value):
                    add("source_quality_flag", "ineligible", "Source quality flags contradict observed execution eligibility.", "artifact_verified", row=row.source_row_id, field=flag_field)

    values_by_session = {}
    for sid, observation in aligned.items():
        row = rows[observation.source_row_id]
        values = {}
        for item in observation.fields:
            if item.column_position >= len(row.values):
                continue
            value = row.values[item.column_position].to_python()
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal))
                    or not math.isfinite(value) or (value < 0 if item.name == "volume" else value <= 0)):
                add("invalid_ohlcv", "ineligible", "OHLC must be finite and positive; volume finite and nonnegative.",
                    "artifact_verified", sid, row.source_row_id, item.name)
                continue
            # Existing bars accept Python int/float. Preserve integer share
            # counts exactly rather than rounding large volumes through float.
            converted = value.item() if isinstance(value, np.generic) else value
            if isinstance(converted, Decimal):
                converted = float(converted)
            if converted != value:
                add("lossy_numeric_conversion", "ineligible", "Source value cannot be represented exactly by the existing numeric bar contract.", "artifact_verified", sid, row.source_row_id, item.name)
                continue
            values[item.name] = converted
        if set(values) == set(FIELDS):
            if not values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]:
                add("invalid_ohlc_relationship", "ineligible", "Require low <= open/close <= high.", "artifact_verified", sid, row.source_row_id)
            values_by_session[sid] = values
    status = "ineligible" if any(item.outcome == "ineligible" for item in reasons) else "unknown" if reasons else "eligible"
    market = ()
    if status == "eligible":
        provenance = PriceProvenance(evidence.provider_id, snapshot.snapshot_id, "unadjusted", True, True)
        market = tuple(ObservedMarketBar(asset.asset_id, MarketBar(
            asset.symbol, session.session_id, session.open_timestamp, session.close_timestamp,
            values_by_session[session.session_id]["open"], values_by_session[session.session_id]["close"],
            True, True, values_by_session[session.session_id]["high"], values_by_session[session.session_id]["low"],
            values_by_session[session.session_id]["volume"]), provenance) for session in sessions)
    return EligibilityReport(status, snapshot, tuple(reasons), tuple(references), authenticated,
                             market, asset, calendar)

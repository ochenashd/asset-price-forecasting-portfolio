"""Offline keyed alignment into existing MarketBar and Signal contracts.

The calendar is independently supplied and attested complete by its caller. No
calendar/provider is downloaded or inferred from observed rows. Each calendar
session declares the following session's open, even at the window boundary, so
signal eligibility never depends on appending a future price row. Calendar and
price provenance remain declarations, not independently authenticated evidence.

Only unadjusted, wholly observed, corporate-action-free equity/ETF windows are
supported. Missing rows and filled prices fail closed, rather than dropping
sessions, retrying orders or reconstructing prices. Invalid input raises an
AdapterValidationError with immutable structured diagnostics; no partial engine
inputs are returned. This module neither runs the engine nor generates forecasts.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal, Mapping, Sequence

from .contracts import MarketBar, Signal
from .prediction_export import ExportedPrediction


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    symbol: str
    instrument_type: Literal["equity", "etf"]
    calendar_id: str
    quote_currency: str


@dataclass(frozen=True)
class TradingSession:
    """Independent schedule metadata, not an alternative market-price record."""

    session_id: str
    open_timestamp: datetime
    close_timestamp: datetime
    next_open_timestamp: datetime


@dataclass(frozen=True)
class SessionCalendar:
    """A caller-certified complete window, with a stable schedule/source identity.

    Adjacency is checked against next_open_timestamp; omission from both a
    dishonest calendar and its price feed cannot be independently detected here.
    """

    calendar_id: str
    source_id: str
    sessions: tuple[TradingSession, ...]
    declared_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))


@dataclass(frozen=True)
class PriceProvenance:
    source_id: str
    snapshot_id: str
    adjustment_basis: Literal["unadjusted"]
    all_ohlcv_observed: bool
    corporate_actions_free: bool


@dataclass(frozen=True)
class ObservedMarketBar:
    """Asset/source wrapper around the existing execution price contract."""

    asset_id: str
    bar: MarketBar
    provenance: PriceProvenance


@dataclass(frozen=True)
class AlignmentEntry:
    prediction: ExportedPrediction
    feature_available_at: datetime
    signal: Signal
    next_open_timestamp: datetime


@dataclass(frozen=True)
class AlignmentManifest:
    """Immutable lineage; model availability is caller-declared, not verified."""

    asset: AssetSpec
    calendar: SessionCalendar
    market: tuple[ObservedMarketBar, ...]
    alignments: tuple[AlignmentEntry, ...]
    signal_mode: Literal["long_only", "short_enabled"]
    provenance_independently_verified: bool = False


@dataclass(frozen=True)
class AdapterDiagnostic:
    code: str
    message: str
    session_id: str | None = None
    prediction_id: str | None = None


class AdapterValidationError(ValueError):
    """Fail-closed validation error with machine-readable rejection information."""

    def __init__(self, diagnostic: AdapterDiagnostic):
        self.diagnostics = (diagnostic,)
        super().__init__(f"{diagnostic.code}: {diagnostic.message}")


@dataclass(frozen=True)
class AdaptedExecutionInputs:
    bars: tuple[MarketBar, ...]
    signals: tuple[Signal, ...]
    manifest: AlignmentManifest
    rejection_diagnostics: tuple[AdapterDiagnostic, ...] = ()


def _reject(code: str, message: str, session_id: str | None = None,
            prediction_id: str | None = None) -> None:
    raise AdapterValidationError(AdapterDiagnostic(code, message, session_id, prediction_id))


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _reject("missing_identifier", f"{name} requires a non-empty string identifier.")


def _utc(value: datetime, name: str, session_id: str | None = None,
         prediction_id: str | None = None) -> datetime:
    if (not isinstance(value, datetime) or value != value or value.tzinfo is None
            or value.utcoffset() is None):
        _reject("invalid_timestamp", f"{name} requires an explicit timezone-aware datetime.",
                session_id, prediction_id)
    return value.astimezone(timezone.utc)


def _observed_number(value: float, name: str, session_id: str, *, volume: bool = False) -> None:
    # Match engine-supported numeric price types; never coerce strings or repair
    # missing values. A zero volume is valid, while equity prices must be positive.
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value < 0 if volume else value <= 0)):
        _reject("invalid_ohlcv", f"{name} requires an observed finite {'non-negative volume' if volume else 'positive price'}.",
                session_id)


def adapt_execution_inputs(
    asset: AssetSpec, calendar: SessionCalendar, market: Sequence[ObservedMarketBar],
    predictions: Sequence[ExportedPrediction], *,
    feature_availability: Mapping[str, datetime],
    signal_mode: Literal["long_only", "short_enabled"],
) -> AdaptedExecutionInputs:
    """Align exact asset/session keys; preserve every execution session.

    feature_availability is keyed by prediction ID and must match that set
    exactly. Observation must equal its independently supplied session close;
    availability may be later, and eligibility must strictly follow availability.
    Historical model information must strictly precede eligibility. Live records
    use their actual generation time as the Signal decision, never replay time.

    Valid rows are chronologically ordered by the independent calendar. Invalid
    rows are never silently skipped. Absence of a forecast produces no Signal;
    class 1 is an explicit Flat instruction. Class 0 maps to Flat only when the
    caller explicitly selects long_only. No threshold strategy is applied.
    """
    if not isinstance(asset, AssetSpec):
        _reject("invalid_contract", "asset requires AssetSpec.")
    for name in ("asset_id", "symbol", "calendar_id", "quote_currency"):
        _identifier(getattr(asset, name), name)
    if asset.instrument_type not in {"equity", "etf"}:
        _reject("unsupported_instrument", "Only one equity or ETF is supported.")
    if signal_mode not in {"long_only", "short_enabled"}:
        _reject("invalid_signal_mode", "Choose explicit long_only or short_enabled mode.")
    if not isinstance(calendar, SessionCalendar):
        _reject("invalid_contract", "calendar requires SessionCalendar.")
    _identifier(calendar.calendar_id, "calendar_id")
    _identifier(calendar.source_id, "calendar_source_id")
    if calendar.calendar_id != asset.calendar_id:
        _reject("calendar_mismatch", "Asset and independently supplied calendar must agree.")
    if calendar.declared_complete is not True or not calendar.sessions:
        _reject("incomplete_calendar", "An independently supplied complete non-empty calendar is required.")
    calendar_by_id: dict[str, TradingSession] = {}
    for session in calendar.sessions:
        if not isinstance(session, TradingSession):
            _reject("invalid_contract", "Calendar entries require TradingSession.")
        _identifier(session.session_id, "session_id")
        if session.session_id in calendar_by_id:
            _reject("duplicate_session", "Duplicate calendar session identifiers are forbidden.", session.session_id)
        opened = _utc(session.open_timestamp, "calendar open", session.session_id)
        closed = _utc(session.close_timestamp, "calendar close", session.session_id)
        following = _utc(session.next_open_timestamp, "next opening cutoff", session.session_id)
        if not opened < closed < following:
            _reject("invalid_calendar_timing", "Require open < close < next opening cutoff.", session.session_id)
        calendar_by_id[session.session_id] = session
    sessions = tuple(sorted(calendar.sessions, key=lambda row: _utc(row.open_timestamp, "calendar open")))
    for previous, current in zip(sessions, sessions[1:]):
        if _utc(previous.next_open_timestamp, "next opening cutoff") != _utc(current.open_timestamp, "calendar open"):
            _reject("calendar_gap_or_overlap", "Adjacent sessions must match the independently declared next open.",
                    current.session_id)
    ordered_calendar = SessionCalendar(calendar.calendar_id, calendar.source_id, sessions, True)
    observed_by_id: dict[str, ObservedMarketBar] = {}
    for observed in market:
        if not isinstance(observed, ObservedMarketBar) or not isinstance(observed.bar, MarketBar):
            _reject("invalid_contract", "Market inputs require ObservedMarketBar wrapping MarketBar.")
        if observed.asset_id != asset.asset_id or observed.bar.symbol != asset.symbol:
            _reject("asset_mismatch", "Market asset identifier and symbol must match the single asset.")
        bar = observed.bar
        _identifier(bar.session_id, "market_session_id")
        if bar.session_id in observed_by_id:
            _reject("duplicate_session", "Duplicate market sessions are forbidden.", bar.session_id)
        if bar.session_id not in calendar_by_id:
            _reject("unexpected_market_session", "Market session is outside the independent calendar.", bar.session_id)
        session = calendar_by_id[bar.session_id]
        for name in ("open_timestamp", "close_timestamp"):
            if _utc(getattr(bar, name), name, bar.session_id) != _utc(getattr(session, name), name):
                _reject("market_calendar_timing_mismatch", "Observed bar timestamps must match their calendar session.",
                        bar.session_id)
        provenance = observed.provenance
        if not isinstance(provenance, PriceProvenance):
            _reject("invalid_contract", "Market data requires explicit PriceProvenance.", bar.session_id)
        _identifier(provenance.source_id, "price_source_id")
        _identifier(provenance.snapshot_id, "price_snapshot_id")
        if provenance.adjustment_basis != "unadjusted":
            _reject("unsupported_adjustment_basis", "Only explicitly unadjusted execution prices are supported.", bar.session_id)
        if (provenance.all_ohlcv_observed is not True or bar.open_observed is not True
                or bar.close_observed is not True):
            _reject("synthetic_execution_price", "All OHLCV must be observed, never filled or synthetic.", bar.session_id)
        if provenance.corporate_actions_free is not True:
            _reject("unsupported_corporate_actions", "The existing engine requires a corporate-action-free window.", bar.session_id)
        for name in ("open_price", "close_price", "high", "low", "volume"):
            _observed_number(getattr(bar, name), name, bar.session_id, volume=name == "volume")
        if not bar.low <= min(bar.open_price, bar.close_price) <= max(bar.open_price, bar.close_price) <= bar.high:
            _reject("inconsistent_ohlc", "Observed OHLC must satisfy low <= open/close <= high.", bar.session_id)
        observed_by_id[bar.session_id] = observed
    missing = calendar_by_id.keys() - observed_by_id.keys()
    if missing:
        _reject("missing_market_sessions", "Every expected calendar session requires independently observed OHLCV.",
                sorted(missing)[0])
    prediction_by_session: dict[str, ExportedPrediction] = {}
    prediction_ids: set[str] = set()
    feature_row_ids: set = set()
    for prediction in predictions:
        if not isinstance(prediction, ExportedPrediction):
            _reject("invalid_contract", "Predictions require validated ExportedPrediction records.")
        identity = prediction.identity
        if identity.asset_id != asset.asset_id:
            _reject("asset_mismatch", "Prediction asset must match the single execution asset.",
                    identity.session_id, identity.prediction_id)
        if identity.session_id not in calendar_by_id:
            _reject("unknown_prediction_session", "Prediction session must exist in the independent calendar.",
                    identity.session_id, identity.prediction_id)
        if (identity.prediction_id in prediction_ids or identity.session_id in prediction_by_session
                or identity.feature_row_id in feature_row_ids):
            _reject("duplicate_prediction", "Prediction, source-row and asset/session identities must be unique.",
                    identity.session_id, identity.prediction_id)
        prediction_ids.add(identity.prediction_id)
        feature_row_ids.add(identity.feature_row_id)
        prediction_by_session[identity.session_id] = prediction
    if not isinstance(feature_availability, Mapping) or set(feature_availability) != prediction_ids:
        _reject("feature_availability_keys", "Feature availability keys must match prediction identifiers exactly.")
    alignments: list[AlignmentEntry] = []
    for session in sessions:
        prediction = prediction_by_session.get(session.session_id)
        if prediction is None:
            continue
        identity, provenance = prediction.identity, prediction.provenance
        pid, sid = identity.prediction_id, identity.session_id
        observed = _utc(identity.feature_observation_timestamp, "feature observation", sid, pid)
        closed = _utc(session.close_timestamp, "session close")
        if observed != closed:
            _reject("feature_session_mismatch", "Feature observation must equal its declared session close.", sid, pid)
        available_at = feature_availability[pid]
        available = _utc(available_at, "actual feature availability", sid, pid)
        eligible = _utc(identity.historical_decision_eligibility_timestamp, "decision eligibility", sid, pid)
        if available < closed or eligible <= available:
            _reject("feature_not_available", "Require session close <= feature availability < decision eligibility.", sid, pid)
        if (_utc(provenance.model_information_cutoff, "model information cutoff") >= eligible
                or _utc(provenance.historical_model_available_at, "model availability") > eligible):
            _reject("model_not_available", "Model information must precede decision eligibility and availability cannot follow it.", sid, pid)
        decision = identity.historical_decision_eligibility_timestamp
        if prediction.mode == "live_generation":
            decision = prediction.actual_prediction_generated_at
        if _utc(decision, "effective decision", sid, pid) >= _utc(session.next_open_timestamp, "next opening cutoff"):
            _reject("missed_next_open", "Decision must precede the next independently scheduled opening cutoff.", sid, pid)
        direction = {0: "flat" if signal_mode == "long_only" else "short", 1: "flat", 2: "long"}[prediction.predicted_class]
        signal = Signal(sid, decision, direction)
        alignments.append(AlignmentEntry(prediction, available_at, signal, session.next_open_timestamp))
    ordered_market = tuple(observed_by_id[session.session_id] for session in sessions)
    return AdaptedExecutionInputs(
        bars=tuple(observed.bar for observed in ordered_market),
        signals=tuple(entry.signal for entry in alignments),
        manifest=AlignmentManifest(asset, ordered_calendar, ordered_market, tuple(alignments), signal_mode),
    )

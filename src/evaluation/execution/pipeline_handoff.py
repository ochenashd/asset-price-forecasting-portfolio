"""Readiness only: no fitting, inference, signals, orchestration or execution.

Receipts bind caller-supplied capture/state bytes to an explicit fitting attempt.
They cannot prove that a caller supplied the true estimator or a fresh receipt.
Replay clocks remain declarations. Market eligibility authenticates an issuer's
statement, not its real-world truth. Secrets are neither hashed into identities
nor included in repr/report output; this module never serializes the context.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

from src.data.market_snapshot import MarketSnapshot
from src.data.market_eligibility import EligibilityReport, EvidenceAttestation, validate_market_eligibility
from .adapter import AssetSpec, SessionCalendar
from .contracts import ExecutionConfig
from .metrics import PerformanceConfig
from .model_provenance import SourceRow, FittedStateAvailability, FrozenModelManifest, freeze_model_manifest
from .orchestration import PredictionTiming
from .prediction_export import FeatureRowIdentifier, ProbabilityRow, _utc_instant
from .xgboost_bridge import KeyedXGBoostPredictions

if TYPE_CHECKING:
    from src.models.ml import TemporalSplit


def _canonical(value):
    if isinstance(value, datetime):
        value = value.astimezone(timezone.utc) if value.utcoffset() is not None else value
        return {"datetime": value.isoformat()}
    if isinstance(value, Mapping):
        return [[_canonical(key), _canonical(item)] for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _aware(value) -> bool:
    return isinstance(value, datetime) and value == value and value.utcoffset() is not None


def _same_calendar(left: SessionCalendar | None, right: SessionCalendar | None) -> bool:
    """Match stable calendar identity and physical clocks, never context secrets."""
    if not isinstance(left, SessionCalendar) or not isinstance(right, SessionCalendar):
        return False
    def identity(calendar):
        return (calendar.calendar_id, calendar.source_id, calendar.declared_complete,
                tuple((session.session_id, _utc_instant(session.open_timestamp),
                       _utc_instant(session.close_timestamp), _utc_instant(session.next_open_timestamp))
                      for session in calendar.sessions))
    try:
        return identity(left) == identity(right)
    except (ValueError, TypeError, AttributeError):
        return False


@dataclass(frozen=True)
class TargetConfiguration:
    target_type: str
    horizon: int
    threshold: float | None

    def __post_init__(self):
        if (not isinstance(self.target_type, str) or not self.target_type.strip()
                or isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral) or self.horizon <= 0):
            raise ValueError("An explicit task and positive integer observation horizon are required.")
        if self.threshold is not None and (isinstance(self.threshold, bool)
                or not isinstance(self.threshold, Real) or not math.isfinite(self.threshold) or self.threshold < 0):
            raise ValueError("A supplied threshold must be finite and nonnegative.")
        if self.target_type == "5d_class" and self.threshold is None:
            raise ValueError("Classification requires an explicit threshold.")


@dataclass(frozen=True)
class FittedModelBinding:
    """Content identity of supplied estimator/preprocessing bytes and experiment.

    The caller must serialize the actual fitted state without retraining it.
    A descriptive model name or Python object address is insufficient. No pickle
    is deserialized and no estimator method is called by this layer.
    """

    experiment_id: str
    feature_names: tuple[str, ...]
    target: TargetConfiguration
    fitted_state_names: tuple[str, ...]
    uses_early_stopping: bool
    estimator_state: bytes = field(repr=False)
    preprocessing_state: bytes = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "fitted_state_names", tuple(self.fitted_state_names))
        if (not isinstance(self.experiment_id, str) or not self.experiment_id.strip()
                or not isinstance(self.estimator_state, bytes) or not self.estimator_state
                or not isinstance(self.preprocessing_state, bytes) or not self.preprocessing_state
                or not isinstance(self.target, TargetConfiguration) or not isinstance(self.uses_early_stopping, bool)):
            raise ValueError("Explicit experiment, target and serialized fitted states are required.")

    @property
    def model_id(self) -> str:
        return "apf-fitted-model-v1-" + _digest((
            self.experiment_id, self.feature_names, self.target.target_type, self.target.horizon,
            self.target.threshold, self.fitted_state_names, self.uses_early_stopping, hashlib.sha256(self.estimator_state).hexdigest(),
            hashlib.sha256(self.preprocessing_state).hexdigest()))


def capture_fingerprint(capture: KeyedXGBoostPredictions) -> str:
    """Bind supplied values and index metadata; no external capture lookup occurs."""
    indices = tuple(None if index is None else (tuple(index), str(index.dtype), index.name,
                    getattr(index, "freqstr", None)) for index in (
                        capture.held_out_index, capture.training_source_index, capture.fitting_index, capture.validation_index))
    records = tuple((row.feature_row_id, row.predicted_class,
                     tuple((label, row.probabilities[label]) for label in capture.estimator_classes), row.probability_sum_error)
                    for row in capture.records)
    return _digest((capture.ticker, capture.asset_id, capture.model_id, capture.split_id, indices,
                    capture.feature_names, capture.estimator_classes, capture.estimator_type,
                    capture.probability_dtype, records, capture.actual_prediction_generated_at,
                    capture.maximum_probability_sum_error, capture.missing_execution_metadata, capture.execution_ready))


@dataclass(frozen=True)
class CaptureReceipt:
    run_id: str
    capture_fingerprint: str
    fitted_model_id: str
    stable_asset_id: str


def bind_fresh_capture(capture: KeyedXGBoostPredictions, *, run_id: str,
                       model: FittedModelBinding, asset: AssetSpec) -> CaptureReceipt:
    """Caller invokes at fresh inference, not by retrieving a persistent capture.

    Run identity is supplied by the fitting attempt. This binding is not an
    authentication mechanism against a caller deliberately reissuing receipts.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("An explicit fitting-attempt identity is required.")
    return CaptureReceipt(run_id, capture_fingerprint(capture), model.model_id, asset.asset_id)


@dataclass(frozen=True)
class FrozenTrainingMetadata:
    """Original observation order differs from the post-feature-filter universe.

    Maturity values reference original observation IDs. They are checked against
    that unfiltered order, never reconstructed from surviving feature rows.
    """

    original_observation_ids: tuple[FeatureRowIdentifier, ...]
    source_row_ids: tuple[FeatureRowIdentifier, ...]
    temporal_split: "TemporalSplit"
    held_out_ids: tuple[FeatureRowIdentifier, ...]
    inner_purged_ids: tuple[FeatureRowIdentifier, ...] | None
    outer_purged_ids: tuple[FeatureRowIdentifier, ...] | None
    target_maturity: Mapping[FeatureRowIdentifier, FeatureRowIdentifier] | None
    feature_names: tuple[str, ...]
    target: TargetConfiguration
    training_start_id: FeatureRowIdentifier
    validation_start_id: FeatureRowIdentifier | None
    test_start_id: FeatureRowIdentifier
    actual_fit_completed_at: datetime | None = None

    def __post_init__(self):
        for name in ("original_observation_ids", "source_row_ids", "held_out_ids", "inner_purged_ids",
                     "outer_purged_ids", "feature_names"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.target_maturity is not None:
            object.__setattr__(self, "target_maturity", MappingProxyType(dict(self.target_maturity)))


@dataclass(frozen=True)
class HistoricalAvailability:
    """Explicit replay declarations; observations include referenced maturity IDs."""

    declaration_reference: str
    observation_times: Mapping[FeatureRowIdentifier, datetime]
    feature_available_at: Mapping[FeatureRowIdentifier, datetime | None]
    label_available_at: Mapping[FeatureRowIdentifier, datetime | None]
    feature_selection_available_at: datetime | None
    fitted_states: tuple[FittedStateAvailability, ...] | None
    uses_early_stopping: bool | None
    early_stopping_available_at: datetime | None
    fitting_latency: timedelta | None
    inference_latency: timedelta | None

    def __post_init__(self):
        for name in ("observation_times", "feature_available_at", "label_available_at"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.fitted_states is not None:
            object.__setattr__(self, "fitted_states", tuple(self.fitted_states))


@dataclass(frozen=True)
class ExternalExecutionContext:
    snapshot: MarketSnapshot | None
    attestation: EvidenceAttestation | None
    calendar: SessionCalendar | None
    row_timings: tuple[PredictionTiming, ...] | None
    execution_config: ExecutionConfig | None
    performance_config: PerformanceConfig | None
    signal_mode: str | None
    trusted_keys: Mapping[str, bytes] | None = field(repr=False)

    def __post_init__(self):
        if self.trusted_keys is not None:
            object.__setattr__(self, "trusted_keys", MappingProxyType(dict(self.trusted_keys)))
        if self.row_timings is not None:
            object.__setattr__(self, "row_timings", tuple(self.row_timings))


@dataclass(frozen=True)
class ReadinessBlocker:
    code: str
    field: str
    explanation: str


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    blockers: tuple[ReadinessBlocker, ...]
    run_id: str
    capture_fingerprint: str | None
    model_id: str | None
    stable_asset_id: str | None
    frozen_membership_valid: bool
    manifest_id: str | None
    market_eligibility_status: str | None
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class HandoffPayload:
    """Arguments retained for a subsequent explicit hook; never dispatched here.

    Existing capture indices retain pandas' mutable metadata. A later consumer
    must revalidate this payload; readiness describes the checked inputs only.
    Context contains runtime secrets and must not be persisted or serialized.
    """

    run_id: str
    original_capture_fingerprint: str
    capture: KeyedXGBoostPredictions
    manifest: FrozenModelManifest
    asset_mapping: Mapping[str, str]
    context: ExternalExecutionContext
    eligibility: EligibilityReport


@dataclass(frozen=True)
class HandoffReadiness:
    report: ReadinessReport
    payload: HandoffPayload | None


def assess_pipeline_handoff(*, capture: KeyedXGBoostPredictions | None, current_run_id: str,
                            receipt: CaptureReceipt | None, asset: AssetSpec | None,
                            model: FittedModelBinding | None, metadata: FrozenTrainingMetadata | None,
                            availability: HistoricalAvailability | None,
                            context: ExternalExecutionContext | None) -> HandoffReadiness:
    """Return blockers or a bound payload, without creating signals or executing.

    Missing bridge IDs can be enriched only by explicit receipt-bound identities.
    Existing conflicting IDs are rejected. No target values are accepted.
    Market eligibility is decided by the existing authoritative validator.
    """
    blockers = []
    fingerprint = manifest = eligibility = None
    membership_valid = False
    limitations = (
        "Serialized fitted state and capture/run association are caller-supplied, not independently witnessed.",
        "Historical availability is an explicit replay declaration, not proof of historical model existence.",
        "Ready is not execution certification; downstream adapter, orchestrator and engine validation remains required.",
        "No new ledger, prediction, signal, database record or tracking artifact is produced.",
    )

    def add(code, name, message):
        blockers.append(ReadinessBlocker(code, name, message))

    def finish(payload=None):
        return HandoffReadiness(ReadinessReport(
            payload is not None and not blockers, tuple(blockers), current_run_id, fingerprint,
            model.model_id if isinstance(model, FittedModelBinding) else None,
            asset.asset_id if isinstance(asset, AssetSpec) else None, membership_valid,
            manifest.split_id if manifest is not None else None,
            eligibility.status if eligibility is not None else None, limitations), payload)

    if capture is None:
        add("capture_disabled", "capture", "A fresh individual XGBoost capture is required.")
    elif not isinstance(capture, KeyedXGBoostPredictions):
        add("manifest_mismatch", "capture", "The existing keyed capture contract is required.")
    if not isinstance(model, FittedModelBinding):
        add("missing_model_id", "model", "Supply serialized fitted-state identity and experiment metadata.")
    if not isinstance(asset, AssetSpec) or not isinstance(asset.asset_id, str) or not asset.asset_id.strip():
        add("missing_stable_asset_id", "asset", "Supply an explicit stable asset; a database ID is insufficient.")
    if metadata is None:
        add("manifest_mismatch", "metadata", "Actual frozen training memberships are required.")
    elif metadata.outer_purged_ids is None:
        add("missing_outer_purge_membership", "outer_purged_ids", "Declare the ordered outer purge, including an explicit empty tuple.")
    if metadata is not None and metadata.target_maturity is None:
        add("missing_target_maturity", "target_maturity", "Original keyed maturity metadata is required.")
    if availability is None:
        add("unknown_feature_availability", "availability", "Explicit historical availability declarations are required.")
    if context is None or context.snapshot is None or context.attestation is None or not context.trusted_keys:
        add("missing_market_evidence", "context", "Snapshot, signed evidence and externally configured trust are required.")
    if blockers:
        return finish()
    try:
        fingerprint = capture_fingerprint(capture)
        if (not isinstance(current_run_id, str) or not current_run_id.strip() or receipt is None
                or receipt.run_id != current_run_id or receipt.capture_fingerprint != fingerprint):
            add("stale_capture", "receipt", "Receipt must bind this exact capture to the current fitting attempt.")
        if receipt is not None and (receipt.fitted_model_id != model.model_id or receipt.stable_asset_id != asset.asset_id):
            add("manifest_mismatch", "receipt", "Receipt model/asset binding conflicts with supplied identities.")
        if capture.model_id is not None and capture.model_id != model.model_id:
            add("manifest_mismatch", "model_id", "Existing capture model identity cannot be overwritten.")
        if (capture.asset_id is not None and capture.asset_id != asset.asset_id):
            add("manifest_mismatch", "asset_id", "An existing database ID needs a separately explicit mapping; it is not a stable asset ID.")
        if capture.ticker != asset.symbol:
            add("manifest_mismatch", "ticker", "The supplied security must match the captured symbol.")
        if metadata.target.target_type != "5d_class":
            add("unsupported_target_type", "target", "This handoff supports the existing classification export only.")
        if metadata.target != model.target:
            add("manifest_mismatch", "target", "Actual target configuration differs from fitted-state identity.")
        if capture.feature_names != metadata.feature_names or capture.feature_names != model.feature_names:
            add("manifest_mismatch", "feature_names", "Selected feature order must agree across all supplied identities.")
        split = metadata.temporal_split
        for name, actual, expected in (
            ("held_out_ids", capture.held_out_index, metadata.held_out_ids),
            ("fitting_index", capture.fitting_index, split.fitting_index),
            ("validation_index", capture.validation_index, split.validation_index),
            ("training_source_index", capture.training_source_index, split.source_index)):
            if actual is None or tuple(actual) != tuple(expected) or not actual.is_unique or actual.hasnans:
                add("manifest_mismatch", name, "Ordered frozen membership differs from the fresh capture.")
        if tuple(row.feature_row_id for row in capture.records) != metadata.held_out_ids:
            add("manifest_mismatch", "records", "Prediction keys/order must match held-out membership exactly.")
        inner = tuple(key for key in split.source_index if key not in set(tuple(split.fitting_index) + tuple(split.validation_index)))
        if metadata.inner_purged_ids != inner:
            add("manifest_mismatch", "inner_purged_ids", "Explicit inner-purged membership differs from the actual split.")
        originals = metadata.original_observation_ids
        if (len(set(originals)) != len(originals) or not originals
                or tuple(key for key in originals if key in set(metadata.source_row_ids)) != metadata.source_row_ids):
            add("manifest_mismatch", "original_observation_ids", "Original observations must uniquely contain the ordered model source universe.")
        positions = {key: i for i, key in enumerate(originals)}
        original_clocks = tuple(availability.observation_times.get(key) for key in originals)
        if (not all(_aware(value) for value in original_clocks)
                or any(_utc_instant(left) >= _utc_instant(right) for left, right in zip(original_clocks, original_clocks[1:]))):
            add("unknown_feature_availability", "observation_times", "The original observation calendar needs explicit chronological aware clocks.")
        if set(metadata.target_maturity) != set(metadata.source_row_ids):
            add("missing_target_maturity", "target_maturity", "Maturity keys must cover the frozen source universe exactly.")
        source_rows = []
        for key in metadata.source_row_ids:
            end = metadata.target_maturity.get(key)
            if end is None:
                add("missing_target_maturity", "target_maturity", "A source-row maturity identity is missing.")
                continue
            position = positions.get(key, -1) + metadata.target.horizon
            if position >= len(originals) or position < 0 or originals[position] != end:
                add("manifest_mismatch", "target_maturity", "Provided maturity does not match the declared original-observation horizon.")
            observed, target_end = availability.observation_times.get(key), availability.observation_times.get(end)
            feature = availability.feature_available_at.get(key)
            label = availability.label_available_at.get(key)
            if not _aware(observed) or not _aware(target_end) or not _aware(feature):
                add("unknown_feature_availability", "source_rows", "Observation, maturity and feature availability need explicit aware clocks.")
                continue
            if key not in metadata.held_out_ids and not _aware(label):
                add("unknown_label_availability", "label_available_at", "Every fitting/validation/purged candidate needs label availability.")
                continue
            source_rows.append(SourceRow(key, observed, feature, target_end, label))
        if not availability.declaration_reference:
            add("unknown_feature_availability", "declaration_reference", "Replay assumptions need an explicit source reference.")
        if not _aware(availability.feature_selection_available_at):
            add("unknown_feature_availability", "feature_selection_available_at", "Selection availability is required.")
        if (availability.fitted_states is None or tuple(state.name for state in availability.fitted_states) != model.fitted_state_names):
            add("unknown_feature_availability", "fitted_states", "Declare every fitted preprocessing state's availability.")
        if not isinstance(availability.uses_early_stopping, bool) or (
                availability.uses_early_stopping and not _aware(availability.early_stopping_available_at)):
            add("missing_early_stopping_information", "early_stopping", "Declare actual early-stopping usage and all used information clocks.")
        if availability.uses_early_stopping != model.uses_early_stopping:
            add("manifest_mismatch", "uses_early_stopping", "Availability declarations must match fitted-model usage.")
        if availability.fitting_latency is None or availability.inference_latency is None:
            add("unknown_model_availability", "latency", "Explicit replay fitting and inference latency are required.")
        if blockers:
            return finish()
        manifest = freeze_model_manifest(
            temporal_split=split, source_rows=source_rows, held_out_ids=metadata.held_out_ids,
            outer_purged_ids=metadata.outer_purged_ids, asset_id=asset.asset_id, model_id=model.model_id,
            experiment_id=model.experiment_id, training_start=availability.observation_times[metadata.training_start_id],
            validation_start=availability.observation_times[metadata.validation_start_id] if metadata.validation_start_id is not None else None,
            test_start=availability.observation_times[metadata.test_start_id], target_type=metadata.target.target_type,
            target_horizon=metadata.target.horizon, target_threshold=metadata.target.threshold,
            feature_names=metadata.feature_names, feature_selection_available_at=availability.feature_selection_available_at,
            fitted_states=availability.fitted_states, uses_early_stopping=availability.uses_early_stopping,
            early_stopping_available_at=availability.early_stopping_available_at,
            fitting_latency=availability.fitting_latency, inference_latency=availability.inference_latency,
            actual_fit_completed_at=metadata.actual_fit_completed_at)
        manifest.to_prediction_provenance()
        membership_valid = True
        if capture.split_id is not None and capture.split_id != manifest.split_id:
            add("manifest_mismatch", "split_id", "Existing split identity differs from the full frozen manifest.")
        if (metadata.actual_fit_completed_at is not None and capture.actual_prediction_generated_at is not None
                and _utc_instant(capture.actual_prediction_generated_at) < _utc_instant(metadata.actual_fit_completed_at)):
            add("manifest_mismatch", "actual_prediction_generated_at", "Actual inference cannot precede observed fit completion.")
        generated = capture.actual_prediction_generated_at
        if generated is not None and (not _aware(generated) or _utc_instant(generated) < max(
                _utc_instant(manifest.model_information_cutoff),
                max(_utc_instant(row.observation_at) for row in manifest.source_rows if row.row_id in metadata.held_out_ids))):
            add("manifest_mismatch", "actual_prediction_generated_at", "Actual generation must follow all observed prediction information.")
        if (len(capture.estimator_classes) != 3 or set(capture.estimator_classes) != {0, 1, 2}
                or any(isinstance(label, bool) or not isinstance(label, Integral) for label in capture.estimator_classes)):
            add("manifest_mismatch", "estimator_classes", "Explicit classes must contain exactly 0, 1 and 2.")
        for record in capture.records:
            if isinstance(record.predicted_class, bool) or not isinstance(record.predicted_class, Integral) or record.predicted_class not in {0, 1, 2}:
                add("manifest_mismatch", "predicted_class", "Predicted classes must retain integer labels.")
            if set(record.probabilities) != {0, 1, 2}:
                add("manifest_mismatch", "probabilities", "Every class must have exactly one probability.")
            ProbabilityRow("readiness-validation", tuple(record.probabilities[label] for label in capture.estimator_classes))
        eligibility = validate_market_eligibility(context.snapshot, context.attestation, trusted_keys=context.trusted_keys)
        if eligibility.status != "eligible":
            add("market_" + eligibility.status, "market_evidence", "The authoritative eligibility validator did not authorize market inputs.")
        if not _same_calendar(context.calendar, eligibility.calendar):
            add("missing_calendar", "calendar", "An explicit calendar matching authenticated evidence is required.")
        if eligibility.asset != asset:
            add("manifest_mismatch", "asset", "Authenticated market asset differs from supplied stable identity.")
        if context.row_timings is None:
            add("missing_session_mapping", "row_timings", "Complete explicit held-out row/session mappings are required.")
        config = context.execution_config
        if config is None or context.performance_config is None:
            add("missing_execution_configuration", "configuration", "Explicit execution and performance configurations are required.")
        if blockers:
            return finish()
        replace(context.performance_config)
        numeric = (config.initial_capital, config.allocation_fraction, config.fee_rate, config.slippage_rate,
                   config.max_short_exposure, config.initial_margin_fraction, config.maintenance_margin_fraction)
        if (any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in numeric)
                or config.initial_capital <= 0 or not 0 < config.allocation_fraction <= 1
                or not 0 <= config.fee_rate < 1 or not 0 <= config.slippage_rate < 1
                or not 0 < config.max_short_exposure <= 1
                or not 0 < config.maintenance_margin_fraction <= config.initial_margin_fraction
                or not isinstance(config.allow_short, bool) or isinstance(config.holding_sessions, bool)
                or not isinstance(config.holding_sessions, int)):
            add("window_mismatch", "execution_config", "Execution capital, costs, sizing and margin assumptions must be valid.")
        if (config.symbol != asset.symbol or config.instrument_type != asset.instrument_type
                or context.signal_mode not in {"long_only", "short_enabled"}
                or config.allow_short != (context.signal_mode == "short_enabled")
                or config.holding_sessions != manifest.target_horizon):
            add("window_mismatch", "execution_config", "Asset, signal mode and holding horizon must agree with the manifest.")
        sessions = context.calendar.sessions
        session_by_id = {session.session_id: session for session in sessions}
        endpoint = session_by_id.get(config.terminal_session_id)
        if (endpoint is None or not sessions or _utc_instant(sessions[0].close_timestamp) != _utc_instant(manifest.test_start)
                or config.terminal_policy not in {"mark_to_market", "liquidate_at_final_close"}
                or config.position_policy not in {"fixed_horizon", "replace_next_open"}):
            add("window_mismatch", "terminal_session_id", "Declare a supported fixed evaluation window matching the test start.")
        timings = {row.feature_row_id: row for row in context.row_timings}
        if len(timings) != len(context.row_timings) or set(timings) != set(metadata.held_out_ids):
            add("missing_session_mapping", "row_timings", "Timing keys must match held-out membership exactly once.")
        for row in context.row_timings:
            session = session_by_id.get(row.session_id)
            source = {row.row_id: row for row in manifest.source_rows}.get(row.feature_row_id)
            if session is None or source is None or endpoint is None:
                add("window_mismatch", "row_timings", "A prediction references an unavailable session/window.")
                continue
            decision = row.historical_decision_eligibility_timestamp
            manifest.validate_decision(decision)
            if (_utc_instant(row.feature_observation_timestamp) != _utc_instant(source.observation_at)
                    or _utc_instant(source.observation_at) != _utc_instant(session.close_timestamp)
                    or _utc_instant(row.feature_available_at) != _utc_instant(source.features_available_at)
                    or not _utc_instant(session.close_timestamp) <= _utc_instant(row.feature_available_at)
                    < _utc_instant(decision) < _utc_instant(session.next_open_timestamp)
                    or _utc_instant(manifest.model_information_cutoff) >= _utc_instant(decision)
                    or _utc_instant(session.close_timestamp) > _utc_instant(endpoint.close_timestamp)):
                add("window_mismatch", "row_timings", "Source clocks, decision eligibility and next-open window must agree.")
        if blockers:
            return finish()
        bound_capture = replace(capture, asset_id=asset.asset_id, model_id=model.model_id, split_id=manifest.split_id,
                                held_out_index=capture.held_out_index.copy(), training_source_index=capture.training_source_index.copy(),
                                fitting_index=capture.fitting_index.copy(), validation_index=capture.validation_index.copy())
        return finish(HandoffPayload(current_run_id, fingerprint, bound_capture, manifest,
                                     MappingProxyType({asset.asset_id: asset.asset_id}), context, eligibility))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        add("manifest_mismatch", "contracts", f"Existing contract or supplied metadata rejected: {exc}")
        return finish()

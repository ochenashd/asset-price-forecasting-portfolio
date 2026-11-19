"""Opt-in historical replay composition; no training, ingestion or evidence issuance.

Model availability is caller-declared replay provenance. Market authentication
authenticates an issuer's statement, not its real-world truth. This module never
promotes synthetic trust into provider certification or changes execution rules.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from typing import Mapping, Sequence

from src.data.market_eligibility import (
    EligibilityReport, EvidenceAttestation, validate_market_eligibility,
)
from src.data.market_snapshot import MarketSnapshot
from .adapter import AdaptedExecutionInputs, AdapterValidationError, adapt_execution_inputs
from .contracts import ExecutionConfig, ExecutionResult
from .engine import ExecutionEngineV1
from .metrics import ExecutionPerformanceReport, PerformanceConfig, calculate_performance
from .model_provenance import FrozenModelManifest
from .prediction_export import (
    ExportedPrediction, FeatureRowIdentifier, PredictionIdentity, PredictedClassRow,
    PredictiveTargetMetadata, ProbabilityRow, export_predictions, _utc_instant,
)
from .xgboost_bridge import KeyedXGBoostPredictions


@dataclass(frozen=True)
class PredictionTiming:
    """Explicit row/session association; timestamps are never inferred from IDs."""

    feature_row_id: FeatureRowIdentifier
    session_id: str
    feature_observation_timestamp: datetime
    feature_available_at: datetime
    historical_decision_eligibility_timestamp: datetime


class OrchestrationRejected(ValueError):
    """Structured fail-closed diagnostic, retaining the original eligibility report."""

    def __init__(self, stage: str, code: str, message: str,
                 eligibility_report: EligibilityReport | None = None, diagnostics: tuple = ()):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.eligibility_report = eligibility_report
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class PreparedReplay:
    """Shared validated inputs; provenance reports retain their original claims."""

    model_manifest: FrozenModelManifest
    predictions: tuple[ExportedPrediction, ...]
    eligibility: EligibilityReport
    adapted: AdaptedExecutionInputs


@dataclass(frozen=True)
class OrchestratedReplay:
    prepared: PreparedReplay
    execution_config: ExecutionConfig
    execution: ExecutionResult
    performance: ExecutionPerformanceReport


@dataclass(frozen=True)
class PolicyComparison:
    """Both full results survive a risk stop; no shortened comparison is invented."""

    fixed_horizon: OrchestratedReplay
    replace_next_open: OrchestratedReplay
    comparable: bool
    non_comparable_reasons: tuple[str, ...]


def _reject(stage: str, code: str, message: str) -> None:
    raise OrchestrationRejected(stage, code, message)


def _prediction_id(manifest: FrozenModelManifest, row_id: FeatureRowIdentifier) -> str:
    # Include the identity type so integer 1 and string "1" cannot collide.
    if isinstance(row_id, datetime):
        value = row_id.astimezone(timezone.utc) if row_id.utcoffset() is not None else row_id
        identity = ["datetime", value.isoformat()]
    elif isinstance(row_id, str):
        identity = ["string", row_id]
    else:
        identity = ["integer", int(row_id)]
    payload = json.dumps([manifest.asset_id, manifest.model_id, manifest.split_id, identity],
                         separators=(",", ":"), ensure_ascii=True)
    return "apf-prediction-v1-" + hashlib.sha256(payload.encode()).hexdigest()


def _prepare(*, capture: KeyedXGBoostPredictions, manifest: FrozenModelManifest,
             asset_mapping: Mapping[str | int, str], row_timings: Sequence[PredictionTiming],
             snapshot: MarketSnapshot, attestation: EvidenceAttestation | None,
             trusted_keys: Mapping[str, bytes], execution_config: ExecutionConfig,
             performance_config: PerformanceConfig, signal_mode: str) -> PreparedReplay:
    if not isinstance(capture, KeyedXGBoostPredictions) or not isinstance(manifest, FrozenModelManifest):
        _reject("identity", "invalid_contract", "Validated bridge and frozen manifest contracts are required.")
    if manifest.target_type != "5d_class":
        _reject("identity", "unsupported_target", "Execution export supports classification only.")
    if (not isinstance(asset_mapping, Mapping) or any(isinstance(key, bool) for key in asset_mapping)
            or isinstance(capture.asset_id, bool) or not isinstance(capture.asset_id, (str, Integral))
            or asset_mapping.get(capture.asset_id) != manifest.asset_id):
        _reject("identity", "asset_mismatch", "An explicit capture-to-stable-asset mapping is required.")
    if capture.model_id != manifest.model_id:
        _reject("identity", "model_mismatch", "Capture model identity must match the manifest.")
    if capture.split_id is not None and capture.split_id != manifest.split_id:
        _reject("identity", "split_mismatch", "Capture split identity must match the frozen manifest.")
    if capture.feature_names != manifest.feature_names:
        _reject("identity", "feature_mismatch", "Selected feature order differs from the manifest.")
    source_ids = tuple(row.row_id for row in manifest.source_rows)
    permitted = set(manifest.fitting_ids + manifest.inner_purged_ids + manifest.validation_ids)
    expected_source = tuple(key for key in source_ids if key in permitted)
    for actual, expected in ((capture.held_out_index, manifest.held_out_ids),
                             (capture.fitting_index, manifest.fitting_ids),
                             (capture.validation_index, manifest.validation_ids),
                             (capture.training_source_index, expected_source)):
        if actual is None or tuple(actual) != expected or not actual.is_unique or actual.hasnans:
            _reject("identity", "membership_mismatch", "Frozen ordered memberships must match exactly.")
    if tuple(row.feature_row_id for row in capture.records) != manifest.held_out_ids:
        _reject("identity", "record_mismatch", "Prediction records must retain exact held-out order and count.")
    timings = tuple(row_timings)
    if not all(isinstance(row, PredictionTiming) for row in timings):
        _reject("identity", "invalid_timing", "Explicit PredictionTiming records are required.")
    timing_by_id = {row.feature_row_id: row for row in timings}
    if len(timing_by_id) != len(timings) or set(timing_by_id) != set(manifest.held_out_ids):
        _reject("identity", "timing_membership", "Timing keys must match held-out membership exactly once.")
    rows = {row.row_id: row for row in manifest.source_rows}
    identities, labels, probabilities, availability = [], [], [], {}
    for record in capture.records:
        timing = timing_by_id[record.feature_row_id]
        source = rows[record.feature_row_id]
        if (_utc_instant(timing.feature_observation_timestamp) != _utc_instant(source.observation_at)
                or _utc_instant(timing.feature_available_at) != _utc_instant(source.features_available_at)):
            _reject("temporal", "source_clock_mismatch", "Timing clocks differ from frozen source provenance.")
        manifest.validate_decision(timing.historical_decision_eligibility_timestamp)
        if _utc_instant(manifest.model_information_cutoff) >= _utc_instant(timing.historical_decision_eligibility_timestamp):
            _reject("temporal", "information_not_available", "Model information must strictly precede the decision.")
        pid = _prediction_id(manifest, record.feature_row_id)
        identities.append(PredictionIdentity(
            manifest.asset_id, pid, timing.session_id, record.feature_row_id,
            timing.feature_observation_timestamp, timing.historical_decision_eligibility_timestamp))
        labels.append(PredictedClassRow(pid, record.predicted_class))
        if set(record.probabilities) != {0, 1, 2}:
            _reject("export", "class_mismatch", "Probabilities require exactly classes 0, 1 and 2.")
        probabilities.append(ProbabilityRow(pid, tuple(record.probabilities[c] for c in capture.estimator_classes)))
        availability[pid] = timing.feature_available_at
    predictions = export_predictions(
        identities, labels, probabilities, estimator_classes=capture.estimator_classes,
        provenance=manifest.to_prediction_provenance(),
        target_metadata=PredictiveTargetMetadata(manifest.target_type, manifest.target_horizon,
                                                manifest.target_threshold),
        mode="historical_replay", actual_prediction_generated_at=capture.actual_prediction_generated_at)
    eligibility = validate_market_eligibility(snapshot, attestation, trusted_keys=trusted_keys)
    if eligibility.status != "eligible":
        raise OrchestrationRejected("market", eligibility.status,
                                    "Market eligibility did not authorize execution.", eligibility)
    asset, calendar = eligibility.asset, eligibility.calendar
    if (asset is None or calendar is None or asset.asset_id != manifest.asset_id
            or asset.symbol != capture.ticker or execution_config.symbol != asset.symbol
            or execution_config.instrument_type != asset.instrument_type):
        _reject("configuration", "asset_mismatch", "Execution asset identity is inconsistent.")
    if (signal_mode not in {"long_only", "short_enabled"}
            or execution_config.allow_short != (signal_mode == "short_enabled")):
        _reject("configuration", "signal_mode_mismatch", "Signal mapping and shorting configuration must agree.")
    if execution_config.holding_sessions != manifest.target_horizon:
        _reject("configuration", "horizon_mismatch", "This replay requires matching target and holding horizons.")
    session_ids = tuple(session.session_id for session in calendar.sessions)
    if (execution_config.terminal_session_id is None
            or execution_config.terminal_session_id not in session_ids):
        _reject("configuration", "missing_endpoint", "A declared terminal session in the validated calendar is required.")
    terminal_index = session_ids.index(execution_config.terminal_session_id)
    if (_utc_instant(calendar.sessions[0].close_timestamp) != _utc_instant(manifest.test_start)
            or any(timing.session_id not in session_ids[:terminal_index + 1] for timing in timings)):
        _reject("configuration", "window_mismatch", "Market start and prediction sessions must match the frozen test window.")
    # Validate configuration before run(), without changing any supplied rules.
    ExecutionEngineV1(execution_config)
    replace(performance_config)
    adapted = adapt_execution_inputs(
        asset, calendar, eligibility.market, predictions,
        feature_availability=availability, signal_mode=signal_mode)
    return PreparedReplay(manifest, predictions, eligibility, adapted)


def _validated_inputs(**kwargs) -> PreparedReplay:
    try:
        return _prepare(**kwargs)
    except OrchestrationRejected:
        raise
    except AdapterValidationError as exc:
        raise OrchestrationRejected("adapter", exc.diagnostics[0].code, str(exc),
                                    diagnostics=exc.diagnostics) from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise OrchestrationRejected("validation", "invalid_input", str(exc)) from exc


def _execute(prepared: PreparedReplay, config: ExecutionConfig,
             performance_config: PerformanceConfig) -> OrchestratedReplay:
    result = ExecutionEngineV1(config).run(prepared.adapted.bars, prepared.adapted.signals)
    return OrchestratedReplay(prepared, config, result, calculate_performance(result, performance_config))


def run_execution_replay(*, capture: KeyedXGBoostPredictions, manifest: FrozenModelManifest,
                         asset_mapping: Mapping[str | int, str], row_timings: Sequence[PredictionTiming],
                         snapshot: MarketSnapshot, attestation: EvidenceAttestation | None,
                         trusted_keys: Mapping[str, bytes], execution_config: ExecutionConfig,
                         performance_config: PerformanceConfig, signal_mode: str) -> OrchestratedReplay:
    """Validate a frozen historical replay completely before invoking execution.

    Integer database IDs require an explicit asset_mapping entry. Model identity
    is mandatory; a missing capture split ID is bound by exact frozen membership.
    The endpoint and matching target/holding horizons are deliberate V1 limits.
    No target values are accepted. Exceptions retain market rejection evidence.
    """
    prepared = _validated_inputs(
        capture=capture, manifest=manifest, asset_mapping=asset_mapping, row_timings=row_timings,
        snapshot=snapshot, attestation=attestation, trusted_keys=trusted_keys,
        execution_config=execution_config, performance_config=performance_config, signal_mode=signal_mode)
    return _execute(prepared, execution_config, performance_config)


def compare_position_policies(*, capture: KeyedXGBoostPredictions, manifest: FrozenModelManifest,
                              asset_mapping: Mapping[str | int, str], row_timings: Sequence[PredictionTiming],
                              snapshot: MarketSnapshot, attestation: EvidenceAttestation | None,
                              trusted_keys: Mapping[str, bytes], execution_config: ExecutionConfig,
                              performance_config: PerformanceConfig, signal_mode: str) -> PolicyComparison:
    """Prepare once and change only position_policy; preserve partial risk outcomes."""
    if execution_config.position_policy not in {"fixed_horizon", "replace_next_open"}:
        _reject("configuration", "invalid_policy", "A supported base position policy is required.")
    fixed = replace(execution_config, position_policy="fixed_horizon")
    replacement = replace(execution_config, position_policy="replace_next_open")
    prepared = _validated_inputs(
        capture=capture, manifest=manifest, asset_mapping=asset_mapping, row_timings=row_timings,
        snapshot=snapshot, attestation=attestation, trusted_keys=trusted_keys,
        execution_config=fixed, performance_config=performance_config, signal_mode=signal_mode)
    first, second = (_execute(prepared, config, performance_config) for config in (fixed, replacement))
    reasons = tuple(f"{run.execution_config.position_policy}: {run.execution.run_status} or incomplete window"
                    for run in (first, second) if not run.performance.completeness.full_window_comparison_eligible)
    return PolicyComparison(first, second, not reasons, reasons)

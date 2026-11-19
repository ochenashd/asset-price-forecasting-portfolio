"""Frozen temporal research provenance; no fitting, targets or execution inputs.

Row labels are opaque identities. All information clocks must be supplied as
aware datetimes, independently of those labels. Availability is caller-declared,
not independently authenticated. A replay availability clock does not establish
that a model actually existed then. Only the explicitly supplied frozen source
universe is hashed; an external dataset is neither read nor resplit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Sequence

from .prediction_export import FeatureRowIdentifier, ModelProvenance, _identifier, _timestamp, _utc_instant

if TYPE_CHECKING:
    from src.models.ml import TemporalSplit


def _row_id(value: FeatureRowIdentifier) -> None:
    if isinstance(value, str):
        _identifier(value, "source row identity")
    elif isinstance(value, datetime):
        if value != value:
            raise ValueError("Missing source row identity.")
    elif isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Source row identities require strings, integers or datetimes.")


@dataclass(frozen=True)
class SourceRow:
    """Explicit observation/availability clocks; no realized target values.

    Target end is observation maturity, while label availability may be later.
    Neither is inferred by shifting a filtered index. Held-out labels may remain
    unknown; all rows participating in maturity purging require both clocks.
    Naive datetime row IDs are allowed only as opaque identities.
    """

    row_id: FeatureRowIdentifier
    observation_at: datetime
    features_available_at: datetime
    target_end_at: datetime | None = None
    labels_available_at: datetime | None = None

    def __post_init__(self) -> None:
        _row_id(self.row_id)
        _timestamp(self.observation_at, "observation_at")
        _timestamp(self.features_available_at, "features_available_at")
        if _utc_instant(self.features_available_at) < _utc_instant(self.observation_at):
            raise ValueError("Features cannot be available before observation.")
        if self.target_end_at is not None:
            _timestamp(self.target_end_at, "target_end_at")
            if _utc_instant(self.target_end_at) <= _utc_instant(self.observation_at):
                raise ValueError("Target maturity must follow observation.")
        if self.labels_available_at is not None:
            _timestamp(self.labels_available_at, "labels_available_at")
            if self.target_end_at is None or _utc_instant(self.labels_available_at) < _utc_instant(self.target_end_at):
                raise ValueError("Label availability must include target maturity.")


@dataclass(frozen=True)
class FittedStateAvailability:
    """Named fitted preprocessing state, declared even for unsupervised fitting."""

    name: str
    available_at: datetime

    def __post_init__(self) -> None:
        _identifier(self.name, "fitted state name")
        _timestamp(self.available_at, "fitted state availability")


def _canonical(value):
    if isinstance(value, datetime):
        # Equivalent aware clocks serialize in UTC, retaining pandas nanoseconds.
        # A naive row identity stays opaque and is never localized.
        if value.utcoffset() is not None:
            value = value.astimezone(timezone.utc)
        return {"datetime": value.isoformat()}
    if isinstance(value, timedelta):
        return {"microseconds": (value.days * 86400 + value.seconds) * 1000000 + value.microseconds}
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return value


@dataclass(frozen=True)
class FrozenModelManifest:
    """Immutable, ordered memberships and clocks with a versioned SHA-256 ID.

    training_start and validation_start are observation boundaries, not claims
    about fitting wall-clock time. fitting/validation rows are the actual inputs
    after both maturity purges. Latencies are explicit nonnegative replay
    assumptions. Actual fit completion is retained separately and does not
    backdate or authenticate replay decisions.
    """

    asset_id: str
    model_id: str
    experiment_id: str
    source_rows: tuple[SourceRow, ...]
    fitting_ids: tuple[FeatureRowIdentifier, ...]
    inner_purged_ids: tuple[FeatureRowIdentifier, ...]
    outer_purged_ids: tuple[FeatureRowIdentifier, ...]
    validation_ids: tuple[FeatureRowIdentifier, ...]
    held_out_ids: tuple[FeatureRowIdentifier, ...]
    training_start: datetime
    validation_start: datetime | None
    test_start: datetime
    target_type: str
    target_horizon: int
    target_threshold: float | None
    feature_names: tuple[str, ...]
    feature_selection_available_at: datetime
    fitted_states: tuple[FittedStateAvailability, ...]
    uses_early_stopping: bool
    early_stopping_available_at: datetime | None
    fitting_latency: timedelta
    inference_latency: timedelta
    actual_fit_completed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("asset_id", "model_id", "experiment_id"):
            _identifier(getattr(self, name), name)
        for name in ("source_rows", "fitting_ids", "inner_purged_ids", "outer_purged_ids",
                     "validation_ids", "held_out_ids", "feature_names", "fitted_states"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes, dict, set)):
                raise ValueError(f"{name} requires an ordered sequence.")
            object.__setattr__(self, name, tuple(value))
        if not self.source_rows or not all(isinstance(row, SourceRow) for row in self.source_rows):
            raise ValueError("Explicit validated source rows are required.")
        ids = tuple(row.row_id for row in self.source_rows)
        clocks = tuple(row.observation_at for row in self.source_rows)
        if len(set(ids)) != len(ids) or any(_utc_instant(a) >= _utc_instant(b) for a, b in zip(clocks, clocks[1:])):
            raise ValueError("Source rows require unique identities and strictly chronological observations.")
        groups = (self.fitting_ids, self.inner_purged_ids, self.outer_purged_ids,
                  self.validation_ids, self.held_out_ids)
        flattened = tuple(item for group in groups for item in group)
        for item in flattened:
            _row_id(item)
        if len(set(flattened)) != len(flattened) or set(flattened) != set(ids):
            raise ValueError("Membership must be complete, disjoint and contain no duplicates or unknown rows.")
        for group in groups:
            members = set(group)
            if tuple(item for item in ids if item in members) != group:
                raise ValueError("Membership order must preserve source chronology.")
        if not self.fitting_ids or not self.held_out_ids:
            raise ValueError("Non-empty fitting and held-out memberships are required.")
        for name in ("training_start", "test_start", "feature_selection_available_at"):
            _timestamp(getattr(self, name), name)
        if self.validation_start is not None:
            _timestamp(self.validation_start, "validation_start")
        rows = {row.row_id: row for row in self.source_rows}
        if (_utc_instant(self.training_start) != _utc_instant(clocks[0])
                or _utc_instant(self.test_start) != _utc_instant(rows[self.held_out_ids[0]].observation_at)):
            raise ValueError("Training and test boundaries must match frozen observations.")
        if bool(self.validation_ids) != (self.validation_start is not None):
            raise ValueError("Validation membership requires an explicit boundary and vice versa.")
        if (self.validation_ids
                and _utc_instant(self.validation_start) != _utc_instant(rows[self.validation_ids[0]].observation_at)):
            raise ValueError("Validation boundary must match the first validation observation.")
        expected = {"fitting": [], "inner": [], "outer": [], "validation": [], "test": []}
        for row in self.source_rows:
            if _utc_instant(row.observation_at) >= _utc_instant(self.test_start):
                expected["test"].append(row.row_id)
                continue
            if row.target_end_at is None or row.labels_available_at is None:
                raise ValueError("Every training candidate requires maturity and label availability.")
            if _utc_instant(row.target_end_at) >= _utc_instant(self.test_start):
                role = "outer"
            elif self.validation_start is not None and _utc_instant(row.observation_at) >= _utc_instant(self.validation_start):
                role = "validation"
            elif self.validation_start is not None and _utc_instant(row.target_end_at) >= _utc_instant(self.validation_start):
                role = "inner"
            else:
                role = "fitting"
            expected[role].append(row.row_id)
        if any(tuple(expected[key]) != group for key, group in zip(
                ("fitting", "inner", "outer", "validation", "test"), groups)):
            raise ValueError("Membership violates fixed boundaries or target-maturity purging.")
        if self.target_type not in {"5d_class", "5d_return", "1d_return"}:
            raise ValueError("Unsupported configured target type.")
        if (isinstance(self.target_horizon, bool) or not isinstance(self.target_horizon, Integral)
                or self.target_horizon <= 0 or (self.target_type == "1d_return" and self.target_horizon != 1)):
            raise ValueError("Target horizon must match the explicit configured task.")
        if self.target_type == "5d_class":
            if (isinstance(self.target_threshold, bool) or not isinstance(self.target_threshold, Real)
                    or not math.isfinite(self.target_threshold) or self.target_threshold < 0):
                raise ValueError("Classification requires its finite nonnegative configured threshold.")
            object.__setattr__(self, "target_threshold", float(self.target_threshold))
        elif self.target_threshold is not None:
            raise ValueError("An unused regression classification threshold must be None.")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Unique ordered feature names are required.")
        for name in self.feature_names:
            _identifier(name, "feature name")
            if name.lower().startswith(("target", "future_")) or name.lower() == "y_true":
                raise ValueError("Future-derived metadata cannot be a predictor.")
        fitting_information = max((max(rows[key].features_available_at, rows[key].labels_available_at, key=_utc_instant)
                                   for key in self.fitting_ids), key=_utc_instant)
        if _utc_instant(self.feature_selection_available_at) < _utc_instant(fitting_information):
            raise ValueError("Feature selection availability must include all fitting inputs.")
        if not all(isinstance(state, FittedStateAvailability) for state in self.fitted_states):
            raise ValueError("Fitted states require validated availability declarations.")
        if len({state.name for state in self.fitted_states}) != len(self.fitted_states):
            raise ValueError("Duplicate fitted state names.")
        fitting_features_cutoff = max((rows[key].features_available_at for key in self.fitting_ids), key=_utc_instant)
        if any(_utc_instant(state.available_at) < _utc_instant(fitting_features_cutoff) for state in self.fitted_states):
            raise ValueError("Fitted preprocessing state cannot precede its fitting features.")
        if not isinstance(self.uses_early_stopping, bool):
            raise ValueError("Early-stopping usage requires an explicit boolean.")
        if self.uses_early_stopping:
            _timestamp(self.early_stopping_available_at, "early_stopping_available_at")
            if (not self.validation_ids
                    or _utc_instant(self.early_stopping_available_at) < _utc_instant(self.validation_information_cutoff)):
                raise ValueError("Early stopping must include all validation information.")
        elif self.early_stopping_available_at is not None:
            raise ValueError("Unused early-stopping information must be None.")
        for latency in (self.fitting_latency, self.inference_latency):
            if type(latency) is not timedelta or latency < timedelta(0):
                raise ValueError("Replay latencies require explicit nonnegative timedeltas.")
        if self.actual_fit_completed_at is not None:
            _timestamp(self.actual_fit_completed_at, "actual_fit_completed_at")
            if _utc_instant(self.actual_fit_completed_at) < _utc_instant(self.model_information_cutoff):
                raise ValueError("Actual fit completion precedes model information.")

    @property
    def validation_information_cutoff(self) -> datetime | None:
        rows = {row.row_id: row for row in self.source_rows}
        return max((max(rows[key].features_available_at, rows[key].labels_available_at, key=_utc_instant)
                    for key in self.validation_ids), key=_utc_instant, default=None)

    @property
    def model_information_cutoff(self) -> datetime:
        rows = {row.row_id: row for row in self.source_rows}
        clocks = [self.feature_selection_available_at]
        for key in self.fitting_ids + self.validation_ids:
            clocks.extend((rows[key].features_available_at, rows[key].labels_available_at))
        clocks.extend(state.available_at for state in self.fitted_states)
        if self.uses_early_stopping:
            clocks.append(self.early_stopping_available_at)
        return max(clocks, key=_utc_instant)

    @property
    def earliest_historical_decision(self) -> datetime:
        # Latencies are elapsed durations, including across local DST changes.
        cutoff = self.model_information_cutoff
        return (cutoff.astimezone(timezone.utc) + self.fitting_latency + self.inference_latency).astimezone(cutoff.tzinfo)

    def validate_decision(self, decision_at: datetime) -> None:
        """Validate replay eligibility; actual live availability needs actual clocks."""
        _timestamp(decision_at, "decision eligibility")
        if _utc_instant(decision_at) < _utc_instant(self.earliest_historical_decision):
            raise ValueError("Decision precedes model availability including replay latency.")

    def canonical_json(self) -> str:
        """Versioned deterministic serialization; preserves row and feature order."""
        return json.dumps({"schema_version": 1, "manifest": _canonical(self)},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

    @property
    def split_id(self) -> str:
        """Content ID covers the entire manifest, including model availability."""
        return "apf-temporal-v1-" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_prediction_provenance(self) -> ModelProvenance:
        """Supply the existing export contract without declaring live existence."""
        rows = {row.row_id: row for row in self.source_rows}
        return ModelProvenance(
            model_id=self.model_id, split_id=self.split_id,
            model_information_cutoff=self.model_information_cutoff,
            historical_model_available_at=self.earliest_historical_decision,
            training_labels_available_at=max((rows[key].labels_available_at for key in self.fitting_ids), key=_utc_instant),
            feature_selection_information_available_at=self.feature_selection_available_at,
            uses_validation=bool(self.validation_ids), uses_early_stopping=self.uses_early_stopping,
            validation_labels_available_at=max((rows[key].labels_available_at for key in self.validation_ids),
                                              key=_utc_instant, default=None),
            early_stopping_information_available_at=self.early_stopping_available_at,
            actual_fit_completed_at=self.actual_fit_completed_at,
        )


def freeze_model_manifest(*, temporal_split: TemporalSplit, source_rows: Sequence[SourceRow],
                          held_out_ids: Sequence[FeatureRowIdentifier],
                          outer_purged_ids: Sequence[FeatureRowIdentifier], **metadata) -> FrozenModelManifest:
    """Freeze existing memberships without importing training or applying fractions.

    source_rows must describe exactly the frozen evaluation universe before
    outer purging. Inner purged IDs are the complement in TemporalSplit.source_index;
    their count and maturity are checked. Extra future rows must be deliberately
    excluded by the caller rather than silently discarded here.
    """
    source = tuple(temporal_split.source_index)
    fitting = tuple(temporal_split.fitting_index)
    validation = tuple(temporal_split.validation_index)
    if len(set(source)) != len(source) or not set(fitting + validation).issubset(source):
        raise ValueError("Inconsistent TemporalSplit source membership.")
    retained = set(fitting + validation)
    inner = tuple(key for key in source if key not in retained)
    if temporal_split.purged_count != len(inner):
        raise ValueError("TemporalSplit purged count is inconsistent.")
    manifest = FrozenModelManifest(
        source_rows=tuple(source_rows), fitting_ids=fitting, validation_ids=validation,
        inner_purged_ids=inner, outer_purged_ids=tuple(outer_purged_ids),
        held_out_ids=tuple(held_out_ids), **metadata,
    )
    source_members = set(source)
    expected_source = tuple(row.row_id for row in manifest.source_rows if row.row_id in source_members)
    if source != expected_source:
        raise ValueError("TemporalSplit source order must match the frozen universe.")
    return manifest

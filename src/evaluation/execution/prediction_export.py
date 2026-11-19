"""Strict, immutable historical prediction exports, independent of execution.

No positional DataFrame joins, target arrays, realized prices or estimator calls
are accepted. All three input row collections are explicitly keyed. The caller
must supply observation and eligibility clocks; a source row label, including a
date-only label, never supplies either clock implicitly. Probability sums use
absolute tolerance 3 * 2**-23 (about 3.58e-7), with no relative tolerance,
clipping or normalization. This budgets one float32 machine epsilon per class
for three-class estimator arithmetic; values outside that budget still fail.

Provenance is a declaration, not proof that the underlying training procedure
honored its cutoffs. Callers must include every fitted input and preprocessing
operation in the model information cutoff, as well as the named supervised
information sources. Historical availability is a replay assumption, separate
from actual fit completion and batch-generation timestamps. This module writes
no artifacts and creates no trading orders.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, Mapping, Sequence


PROBABILITY_SUM_TOLERANCE = 3 * 2**-23
REQUIRED_CLASSES = (0, 1, 2)
GenerationMode = Literal["historical_replay", "live_generation"]
FeatureRowIdentifier = str | int | datetime


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string identifier.")


def _timestamp(value: datetime, name: str) -> None:
    """Require actual aware datetime values; never parse or localize row labels."""
    if (not isinstance(value, datetime) or value != value or value.tzinfo is None
            or value.utcoffset() is None):
        raise ValueError(f"{name} must be a non-missing timezone-aware datetime.")


def _utc_instant(value: datetime) -> datetime:
    """Compare physical instants without changing the caller's reporting clock."""
    _timestamp(value, "absolute timestamp")
    return value.astimezone(timezone.utc)


def _label(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in REQUIRED_CLASSES:
        raise ValueError("Class labels must be integers 0, 1 or 2, never booleans or coerced returns.")
    return int(value)


def _probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probabilities must be finite numeric values in [0, 1].")
    return float(value)


@dataclass(frozen=True)
class PredictionIdentity:
    """Explicit source identity; feature_row_id is opaque, never a timing proxy.

    Row identifiers preserve string, integer or datetime labels unchanged. A
    naive datetime is permitted ONLY as an opaque original row identifier, not
    as an observation or eligibility timestamp. Assets can share session/row
    labels; each asset may have at most one prediction per source row/session.
    """

    asset_id: str
    prediction_id: str
    session_id: str
    feature_row_id: FeatureRowIdentifier
    feature_observation_timestamp: datetime
    historical_decision_eligibility_timestamp: datetime

    def __post_init__(self) -> None:
        for name in ("asset_id", "prediction_id", "session_id"):
            _identifier(getattr(self, name), name)
        row = self.feature_row_id
        if isinstance(row, str):
            _identifier(row, "feature_row_id")
        elif isinstance(row, datetime):
            if row != row:
                raise ValueError("feature_row_id must not be missing.")
        elif isinstance(row, bool) or not isinstance(row, Integral):
            raise ValueError("feature_row_id must be a non-missing string, integer or datetime label.")
        _timestamp(self.feature_observation_timestamp, "feature_observation_timestamp")
        _timestamp(self.historical_decision_eligibility_timestamp, "historical_decision_eligibility_timestamp")
        if _utc_instant(self.historical_decision_eligibility_timestamp) <= _utc_instant(self.feature_observation_timestamp):
            raise ValueError("Decision eligibility must be strictly after feature observation.")


@dataclass(frozen=True)
class ModelProvenance:
    """Information availability includes labels used for selection/early stopping.

    Component clocks mean availability, not merely target-end date labels.
    model_information_cutoff must also bound all fitted features, scaling and
    preprocessing state. historical_model_available_at additionally declares
    assumed fit latency. Unused validation/early-stopping clocks must be None;
    used clocks must be supplied, and early stopping requires validation.
    """

    model_id: str
    split_id: str
    model_information_cutoff: datetime
    historical_model_available_at: datetime
    training_labels_available_at: datetime
    feature_selection_information_available_at: datetime
    uses_validation: bool
    uses_early_stopping: bool
    validation_labels_available_at: datetime | None = None
    early_stopping_information_available_at: datetime | None = None
    actual_fit_completed_at: datetime | None = None

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model_id")
        _identifier(self.split_id, "split_id")
        if not isinstance(self.uses_validation, bool) or not isinstance(self.uses_early_stopping, bool):
            raise ValueError("Validation and early-stopping usage must be explicit booleans.")
        if self.uses_early_stopping and not self.uses_validation:
            raise ValueError("Early stopping requires declared validation information.")
        for name in ("model_information_cutoff", "historical_model_available_at",
                     "training_labels_available_at", "feature_selection_information_available_at"):
            _timestamp(getattr(self, name), name)
        sources = [self.training_labels_available_at, self.feature_selection_information_available_at]
        for used, name in ((self.uses_validation, "validation_labels_available_at"),
                           (self.uses_early_stopping, "early_stopping_information_available_at")):
            value = getattr(self, name)
            if used:
                _timestamp(value, name)
                sources.append(value)
            elif value is not None:
                raise ValueError(f"{name} must be None when that information source is unused.")
        if any(_utc_instant(value) > _utc_instant(self.model_information_cutoff) for value in sources):
            raise ValueError("Model information cutoff must include every declared information source.")
        if (self.uses_early_stopping
                and _utc_instant(self.early_stopping_information_available_at) < _utc_instant(self.validation_labels_available_at)):
            raise ValueError("Early-stopping information cannot precede its validation labels.")
        if _utc_instant(self.historical_model_available_at) < _utc_instant(self.model_information_cutoff):
            raise ValueError("Historical model availability cannot precede its information cutoff.")
        if self.actual_fit_completed_at is not None:
            _timestamp(self.actual_fit_completed_at, "actual_fit_completed_at")
            if _utc_instant(self.actual_fit_completed_at) < _utc_instant(self.model_information_cutoff):
                raise ValueError("Actual fitting cannot finish before its declared information cutoff.")


@dataclass(frozen=True)
class PredictiveTargetMetadata:
    """Descriptive existing classification definition, never realized target values."""

    target_type: Literal["5d_class"]
    horizon_observations: int
    threshold: float

    def __post_init__(self) -> None:
        if self.target_type != "5d_class":
            raise ValueError("This export requires the configured 5d_class task.")
        if (isinstance(self.horizon_observations, bool) or not isinstance(self.horizon_observations, Integral)
                or self.horizon_observations <= 0):
            raise ValueError("Target horizon must be a positive observation count.")
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, Real)
                or not math.isfinite(self.threshold) or self.threshold < 0):
            raise ValueError("Target threshold must be finite and non-negative.")


@dataclass(frozen=True)
class PredictedClassRow:
    prediction_id: str
    predicted_class: int

    def __post_init__(self) -> None:
        _identifier(self.prediction_id, "prediction_id")
        object.__setattr__(self, "predicted_class", _label(self.predicted_class))


@dataclass(frozen=True)
class ProbabilityRow:
    """Exactly three probabilities in the separately declared estimator class order."""

    prediction_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.prediction_id, "prediction_id")
        if isinstance(self.values, (Mapping, str, bytes)):
            raise ValueError("Probability columns require an ordered numeric sequence, not a mapping or string.")
        values = tuple(_probability(value) for value in self.values)
        if len(values) != 3:
            raise ValueError("Exactly three probability columns are required.")
        if abs(math.fsum(values) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError("Probability sums must be within absolute tolerance 3 * 2**-23 of 1.")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ExportedPrediction:
    """Immutable keyed prediction with separate replay and actual-generation clocks.

    Live generation requires an actual fit-completion clock and generation no
    earlier than eligibility. Its actual generated timestamp must govern any
    future live adapter; historical eligibility must never backdate a live order.
    Historical batch generation may occur long after the simulated decisions.
    Probability ties/argmax do not replace the supplied predicted class.
    """

    identity: PredictionIdentity
    predicted_class: int
    probabilities: Mapping[int, float]
    provenance: ModelProvenance
    target_metadata: PredictiveTargetMetadata
    mode: GenerationMode
    actual_prediction_generated_at: datetime | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.identity, PredictionIdentity) or not isinstance(self.provenance, ModelProvenance)
                or not isinstance(self.target_metadata, PredictiveTargetMetadata)):
            raise ValueError("Identity, model provenance and target metadata require validated contracts.")
        if self.mode not in {"historical_replay", "live_generation"}:
            raise ValueError("Unsupported prediction generation mode.")
        object.__setattr__(self, "predicted_class", _label(self.predicted_class))
        if not isinstance(self.probabilities, Mapping):
            raise ValueError("Probabilities require an explicit class-to-probability mapping.")
        classes = tuple(_label(value) for value in self.probabilities)
        if len(classes) != 3 or set(classes) != set(REQUIRED_CLASSES):
            raise ValueError("Probability mapping requires exactly the classes 0, 1 and 2.")
        values = ProbabilityRow(self.identity.prediction_id,
                                tuple(self.probabilities[label] for label in REQUIRED_CLASSES)).values
        object.__setattr__(self, "probabilities", MappingProxyType(dict(zip(REQUIRED_CLASSES, values))))
        identity, provenance = self.identity, self.provenance
        decision = identity.historical_decision_eligibility_timestamp
        if _utc_instant(decision) < _utc_instant(provenance.historical_model_available_at):
            raise ValueError("Decision precedes declared model-information availability.")
        generated = self.actual_prediction_generated_at
        if self.mode == "live_generation" and (generated is None or provenance.actual_fit_completed_at is None):
            raise ValueError("Live generation requires actual fit-completion and prediction-generation timestamps.")
        if generated is not None:
            _timestamp(generated, "actual_prediction_generated_at")
            if _utc_instant(generated) < max(_utc_instant(identity.feature_observation_timestamp),
                                             _utc_instant(provenance.model_information_cutoff)):
                raise ValueError("Actual generation cannot precede its observed information.")
            if (provenance.actual_fit_completed_at is not None
                    and _utc_instant(generated) < _utc_instant(provenance.actual_fit_completed_at)):
                raise ValueError("Prediction generation cannot precede actual fit completion.")
            if self.mode == "live_generation" and _utc_instant(generated) < _utc_instant(decision):
                raise ValueError("Live generation cannot precede declared decision eligibility.")


def export_predictions(
    identities: Sequence[PredictionIdentity], predictions: Sequence[PredictedClassRow],
    probability_rows: Sequence[ProbabilityRow], *, estimator_classes: Sequence[int],
    provenance: ModelProvenance, target_metadata: PredictiveTargetMetadata,
    mode: GenerationMode = "historical_replay", actual_prediction_generated_at: datetime | None = None,
) -> tuple[ExportedPrediction, ...]:
    """Join exact keyed row sets, preserving identity order without positional inference.

    estimator_classes is mandatory and must describe the probability column
    order supplied by the estimator. Nothing is padded, truncated, sorted by
    probability, imputed or recomputed from target values. All collections must
    be non-empty and contain the same unique prediction IDs exactly once.
    """
    if isinstance(estimator_classes, (Mapping, str, bytes)):
        raise ValueError("Estimator classes require an explicit ordered class sequence.")
    classes = tuple(_label(label) for label in estimator_classes)
    if len(classes) != 3 or set(classes) != set(REQUIRED_CLASSES):
        raise ValueError("Estimator classes must contain exactly 0, 1 and 2, once each.")
    if not identities or len(identities) != len(predictions) or len(identities) != len(probability_rows):
        raise ValueError("Non-empty identity, prediction and probability row counts must match exactly.")
    for rows, expected_type in ((identities, PredictionIdentity), (predictions, PredictedClassRow),
                                (probability_rows, ProbabilityRow)):
        if not all(isinstance(row, expected_type) for row in rows):
            raise ValueError("Each input collection requires its validated keyed row contract.")
        if len({row.prediction_id for row in rows}) != len(rows):
            raise ValueError("Duplicate prediction identifiers are forbidden in every collection.")
    for name in ("session_id", "feature_row_id", "feature_observation_timestamp"):
        if len({(row.asset_id, getattr(row, name)) for row in identities}) != len(identities):
            raise ValueError(f"Duplicate asset/{name} identities are forbidden.")
    predicted_by_id = {row.prediction_id: row.predicted_class for row in predictions}
    probabilities_by_id = {row.prediction_id: row.values for row in probability_rows}
    identity_ids = {row.prediction_id for row in identities}
    if identity_ids != predicted_by_id.keys() or identity_ids != probabilities_by_id.keys():
        raise ValueError("Prediction, identity and probability identifiers must match exactly.")
    return tuple(ExportedPrediction(
        identity=identity, predicted_class=predicted_by_id[identity.prediction_id],
        probabilities=dict(zip(classes, probabilities_by_id[identity.prediction_id])),
        provenance=provenance, target_metadata=target_metadata, mode=mode,
        actual_prediction_generated_at=actual_prediction_generated_at,
    ) for identity in identities)

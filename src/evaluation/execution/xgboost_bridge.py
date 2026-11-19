"""Bind individual estimator outputs to their exact held-out inference index.

This partial capture is deliberately not ExportedPrediction: source row labels
are not exchange sessions or market-close clocks. No target arrays, predictor
values, prices, orders or returns are accepted. Model/split IDs are optional
caller-supplied identifiers, never guessed from an estimator type or row number.

Arrays must be the row-preserving outputs of the SAME inference boundary and
unchanged input index. Array lengths alone cannot authenticate an arbitrary
caller's ordering. Already keyed Series/DataFrames must match the index exactly;
independently filtered or reordered keyed outputs are rejected, not positionally
joined. Probability values are promoted to Python float without renormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd

from .prediction_export import FeatureRowIdentifier, ProbabilityRow

if TYPE_CHECKING:
    from src.models.ml import TemporalSplit


@dataclass(frozen=True)
class KeyedXGBoostRow:
    feature_row_id: FeatureRowIdentifier
    predicted_class: int
    probabilities: Mapping[int, float]
    probability_sum_error: float


@dataclass(frozen=True)
class KeyedXGBoostPredictions:
    """Independent copies of index metadata and immutable keyed value records.

    Index copies preserve dtype, name, timezone, frequency and order. These are
    pandas Index objects, so their metadata can be edited by a consumer without
    changing the original input. No execution-readiness claim is made.
    """

    ticker: str
    asset_id: str | int | None
    held_out_index: pd.Index
    feature_names: tuple[str, ...]
    estimator_classes: tuple[int, ...]
    records: tuple[KeyedXGBoostRow, ...]
    estimator_type: str
    model_id: str | None
    split_id: str | None
    training_source_index: pd.Index | None
    fitting_index: pd.Index | None
    validation_index: pd.Index | None
    probability_dtype: str
    maximum_probability_sum_error: float
    actual_prediction_generated_at: datetime | None
    missing_execution_metadata: tuple[str, ...]
    execution_ready: bool = False


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an explicit non-empty string.")


def _index(value: pd.Index, name: str, *, allow_empty: bool = False) -> pd.Index:
    if (not isinstance(value, pd.Index) or isinstance(value, pd.MultiIndex)
            or (not allow_empty and value.empty) or value.hasnans or not value.is_unique):
        raise ValueError(f"{name} requires a unique non-missing single-level row index.")
    for label in value:
        if isinstance(label, str):
            _identifier(label, name)
        elif not isinstance(label, datetime) and (isinstance(label, bool) or not isinstance(label, Integral)):
            raise ValueError(f"{name} row identifiers must be strings, integers or datetimes.")
    return value.copy(deep=True)


def _class(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in {0, 1, 2}:
        raise ValueError("Estimator classes and predictions require integer labels 0, 1 and 2.")
    return int(value)


def capture_xgboost_predictions(
    *, test_index: pd.Index, feature_names: Sequence[str], model_feature_names: Sequence[str],
    predictions: np.ndarray | pd.Series, probabilities: np.ndarray | pd.DataFrame,
    estimator: object, ticker: str, asset_id: str | int | None = None,
    temporal_split: TemporalSplit | None = None, model_id: str | None = None,
    split_id: str | None = None, actual_prediction_generated_at: datetime | None = None,
) -> KeyedXGBoostPredictions:
    """Validate and bind before any legacy flattening, truncation or padding.

    Ordered model_feature_names come from the fitted wrapper; estimator feature
    names are also checked when exposed. temporal_split is preserved as metadata,
    not recomputed. It cannot contain held-out rows. Unknown temporal/session/
    source provenance stays explicitly missing until a later audited export.
    """
    _identifier(ticker, "ticker")
    if asset_id is not None:
        if isinstance(asset_id, str):
            _identifier(asset_id, "asset_id")
        elif isinstance(asset_id, bool) or not isinstance(asset_id, Integral):
            raise ValueError("asset_id must retain its supplied string/integer identity or None.")
    for value, name in ((model_id, "model_id"), (split_id, "split_id")):
        if value is not None:
            _identifier(value, name)
    index = _index(test_index, "test_index")
    names = tuple(feature_names)
    fitted_names = tuple(model_feature_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Feature names must be explicit, non-empty and unique.")
    if any(name.lower().startswith(("target", "future_")) or name.lower() == "y_true" for name in names):
        raise ValueError("Target and future-derived metadata cannot enter the bridge feature schema.")
    if names != fitted_names:
        raise ValueError("Held-out feature order must match the fitted model exactly.")
    estimator_names = getattr(estimator, "feature_names_in_", None)
    if estimator_names is not None and tuple(estimator_names) != names:
        raise ValueError("Estimator feature names disagree with the held-out feature order.")
    raw_classes = getattr(estimator, "classes_", None)
    if raw_classes is None:
        raise ValueError("Estimator must expose explicit classes_.")
    classes = tuple(_class(label) for label in raw_classes)
    if len(classes) != 3 or set(classes) != {0, 1, 2}:
        raise ValueError("Estimator classes_ must contain exactly 0, 1 and 2 once each.")
    for values in (predictions, probabilities):
        if isinstance(values, (pd.Series, pd.DataFrame)) and not values.index.identical(test_index):
            raise ValueError("Keyed estimator outputs must match the exact held-out row index.")
    if isinstance(probabilities, pd.DataFrame):
        if tuple(_class(label) for label in probabilities.columns) != classes:
            raise ValueError("Keyed probability columns must match explicit estimator class order.")
    labels = np.asarray(predictions)
    values = np.asarray(probabilities)
    if labels.ndim != 1 or labels.shape[0] != len(index):
        raise ValueError("Prediction count and one-dimensional shape must match test rows exactly.")
    if values.ndim != 2 or values.shape != (len(index), 3):
        raise ValueError("Probability shape must be exactly (test rows, three classes).")
    if values.dtype.kind not in "fiu":
        raise ValueError("Probability values must have a numeric non-boolean dtype.")
    records = []
    for row_id, label, row in zip(index, labels, values):
        # Reuse the public probability validator. This local validation marker
        # is not an exported prediction ID and never participates in alignment.
        preserved = ProbabilityRow("bridge-value-validation", tuple(row)).values
        records.append(KeyedXGBoostRow(
            row_id, _class(label), MappingProxyType(dict(zip(classes, preserved))),
            abs(math.fsum(preserved) - 1.0),
        ))
    source = fitting = validation = None
    if temporal_split is not None:
        source = _index(temporal_split.source_index, "training_source_index")
        fitting = _index(temporal_split.fitting_index, "fitting_index")
        validation = _index(temporal_split.validation_index, "validation_index", allow_empty=True)
        if (not fitting.isin(source).all() or not validation.isin(source).all()
                or not fitting.intersection(validation).empty or not source.intersection(index).empty):
            raise ValueError("Split metadata must be disjoint from held-out rows and internally consistent.")
    generated = actual_prediction_generated_at
    if generated is not None and (not isinstance(generated, datetime) or generated != generated
                                  or generated.tzinfo is None or generated.utcoffset() is None):
        raise ValueError("Actual prediction-generation time must be an aware datetime or unknown.")
    missing = ["exchange_session_ids", "feature_observation_timestamps", "feature_availability_timestamps",
               "historical_decision_eligibility", "model_information_cutoffs", "historical_model_availability",
               "source_data_provenance", "verified_calendar"]
    if asset_id is None:
        missing.append("asset_id")
    if model_id is None:
        missing.append("model_artifact_id")
    if split_id is None:
        missing.append("split_id")
    return KeyedXGBoostPredictions(
        ticker, asset_id, index, names, classes, tuple(records),
        f"{type(estimator).__module__}.{type(estimator).__qualname__}", model_id, split_id,
        source, fitting, validation, str(values.dtype),
        max(record.probability_sum_error for record in records), generated, tuple(missing),
    )

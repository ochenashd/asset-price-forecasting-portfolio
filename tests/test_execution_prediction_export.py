"""Deterministic keyed prediction, information-maturity and clock contracts."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.execution.prediction_export import (
    PROBABILITY_SUM_TOLERANCE, ExportedPrediction, ModelProvenance, PredictedClassRow,
    PredictionIdentity, PredictiveTargetMetadata, ProbabilityRow, export_predictions,
)


BASE = datetime(2020, 1, 6, 16, tzinfo=timezone.utc)
TARGET = PredictiveTargetMetadata("5d_class", 5, 0.005)


def provenance(**changes):
    fields = dict(
        model_id="xgboost-artifact-1", split_id="frozen-split-1",
        model_information_cutoff=BASE - timedelta(days=1),
        historical_model_available_at=BASE - timedelta(hours=1),
        training_labels_available_at=BASE - timedelta(days=7),
        feature_selection_information_available_at=BASE - timedelta(days=7),
        uses_validation=True, uses_early_stopping=True,
        validation_labels_available_at=BASE - timedelta(days=2),
        early_stopping_information_available_at=BASE - timedelta(days=1),
    )
    return ModelProvenance(**(fields | changes))


def identity(i=0, **changes):
    observed = BASE + timedelta(days=i)
    fields = dict(asset_id="asset-1", prediction_id=f"prediction-{i}",
                  session_id=f"session-{i}", feature_row_id=f"original-row-{i}",
                  feature_observation_timestamp=observed,
                  historical_decision_eligibility_timestamp=observed + timedelta(minutes=1))
    return PredictionIdentity(**(fields | changes))


def export(identities=None, predictions=None, probability_rows=None, **changes):
    identities = [identity(0), identity(1)] if identities is None else identities
    predictions = [PredictedClassRow("prediction-0", 2), PredictedClassRow("prediction-1", 0)] if predictions is None else predictions
    probability_rows = [ProbabilityRow("prediction-0", (0.1, 0.2, 0.7)),
                        ProbabilityRow("prediction-1", (0.6, 0.3, 0.1))] if probability_rows is None else probability_rows
    options = dict(estimator_classes=(0, 1, 2), provenance=provenance(), target_metadata=TARGET)
    return export_predictions(identities, predictions, probability_rows, **(options | changes))


def test_keyed_rows_preserve_observation_and_model_identity_without_positional_joins():
    sources = [identity(0), identity(1)]
    result = export(sources,
                    [PredictedClassRow("prediction-1", 0), PredictedClassRow("prediction-0", 2)],
                    [ProbabilityRow("prediction-1", (0.6, 0.3, 0.1)),
                     ProbabilityRow("prediction-0", (0.1, 0.2, 0.7))])
    assert isinstance(result, tuple)
    assert [row.identity for row in result] == sources
    assert [row.predicted_class for row in result] == [2, 0]
    assert dict(result[0].probabilities) == {0: 0.1, 1: 0.2, 2: 0.7}
    assert dict(result[1].probabilities) == {0: 0.6, 1: 0.3, 2: 0.1}
    assert result[0].provenance.model_id == "xgboost-artifact-1"
    assert result[0].provenance.split_id == "frozen-split-1"
    assert result[0].target_metadata is TARGET


def test_shuffled_estimator_classes_are_mapped_explicitly_to_probability_columns():
    rows = [ProbabilityRow("prediction-0", (0.7, 0.1, 0.2)),
            ProbabilityRow("prediction-1", (0.1, 0.6, 0.3))]
    mapped = export(probability_rows=rows, estimator_classes=np.array([2, 0, 1], dtype=np.int64))
    baseline = export()
    assert [dict(row.probabilities) for row in mapped] == [dict(row.probabilities) for row in baseline]
    assert [row.predicted_class for row in mapped] == [2, 0]


@pytest.mark.parametrize("classes", [(0, 1), (0, 1, 2, 3), (0, 0, 2), (0, 1, 3),
                                     (False, 1, 2), (0.0, 1, 2), ("0", 1, 2), ()])
def test_invalid_or_ambiguous_estimator_class_order_is_rejected(classes):
    with pytest.raises(ValueError, match="[Cc]lass"):
        export(estimator_classes=classes)


@pytest.mark.parametrize("collection", ["identities", "predictions", "probability_rows"])
def test_duplicate_prediction_ids_are_rejected_in_every_input_collection(collection):
    options = {
        "identities": [identity(0), identity(1)],
        "predictions": [PredictedClassRow("prediction-0", 2), PredictedClassRow("prediction-1", 0)],
        "probability_rows": [ProbabilityRow("prediction-0", (0.1, 0.2, 0.7)),
                             ProbabilityRow("prediction-1", (0.6, 0.3, 0.1))],
    }
    options[collection][1] = replace(options[collection][1], prediction_id="prediction-0")
    with pytest.raises(ValueError, match="Duplicate prediction"):
        export(**options)


@pytest.mark.parametrize("field", ["session_id", "feature_row_id", "feature_observation_timestamp"])
def test_duplicate_asset_source_identities_are_rejected(field):
    first, second = identity(0), identity(1)
    second = replace(second, **{field: getattr(first, field)})
    with pytest.raises(ValueError, match="Duplicate asset"):
        export(identities=[first, second])


def test_different_assets_can_share_original_row_and_session_labels():
    second = replace(identity(0), asset_id="asset-2", prediction_id="prediction-1")
    result = export(identities=[identity(0), second])
    assert [row.identity.asset_id for row in result] == ["asset-1", "asset-2"]


@pytest.mark.parametrize("field", ["asset_id", "prediction_id", "session_id", "feature_row_id"])
@pytest.mark.parametrize("missing", [None, "", " "])
def test_missing_identifiers_are_rejected(field, missing):
    with pytest.raises(ValueError, match=field):
        identity(**{field: missing})


@pytest.mark.parametrize("collection", ["identities", "predictions", "probability_rows"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_missing_or_extra_rows_are_rejected_without_truncation(collection, count):
    if collection == "identities":
        rows = [identity(i) for i in range(count)]
    elif collection == "predictions":
        rows = [PredictedClassRow(f"prediction-{i}", 2) for i in range(count)]
    else:
        rows = [ProbabilityRow(f"prediction-{i}", (0.1, 0.2, 0.7)) for i in range(count)]
    with pytest.raises(ValueError, match="row counts"):
        export(**{collection: rows})


@pytest.mark.parametrize("collection", ["predictions", "probability_rows"])
def test_equal_counts_with_different_keys_are_not_assumed_aligned(collection):
    if collection == "predictions":
        rows = [PredictedClassRow("prediction-0", 2), PredictedClassRow("wrong-id", 0)]
    else:
        rows = [ProbabilityRow("prediction-0", (0.1, 0.2, 0.7)),
                ProbabilityRow("wrong-id", (0.6, 0.3, 0.1))]
    with pytest.raises(ValueError, match="identifiers must match exactly"):
        export(**{collection: rows})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.01, 1.01, True, "0.7", None])
def test_invalid_probability_values_are_rejected(bad):
    with pytest.raises(ValueError, match="Probabilities"):
        ProbabilityRow("prediction-0", (0.1, 0.2, bad))


@pytest.mark.parametrize("values", [(), (0.4, 0.6), (0.1, 0.2, 0.3, 0.4)])
def test_probability_column_counts_must_be_exactly_three(values):
    with pytest.raises(ValueError, match="Exactly three"):
        ProbabilityRow("prediction-0", values)


@pytest.mark.parametrize("values", [(0, 0, 0), (0.1, 0.1, 0.7), (0.2, 0.2, 0.7)])
def test_incorrect_probability_sums_are_not_normalized(values):
    with pytest.raises(ValueError, match="absolute tolerance"):
        ProbabilityRow("prediction-0", values)


def test_probability_tolerance_is_absolute_documented_and_does_not_change_values():
    assert PROBABILITY_SUM_TOLERANCE == 3 * 2**-23
    values = (0.1, 0.2, 0.7 + 0.5 * PROBABILITY_SUM_TOLERANCE)
    row = ProbabilityRow("prediction-0", values)
    assert row.values == values
    with pytest.raises(ValueError, match="absolute tolerance"):
        ProbabilityRow("prediction-0", (0.1, 0.2, 0.7 + 2 * PROBABILITY_SUM_TOLERANCE))


@pytest.mark.parametrize("label", [0.001, -0.001, 0.0, 2.0, -1, 3, True, "2", None])
def test_predicted_labels_are_validated_without_return_or_integer_coercion(label):
    with pytest.raises(ValueError, match="integer"):
        PredictedClassRow("prediction-0", label)


@pytest.mark.parametrize("label", [0, 1, 2, np.int64(2)])
def test_declared_integer_classes_are_supported(label):
    assert PredictedClassRow("prediction-0", label).predicted_class == label


@pytest.mark.parametrize("field", ["feature_observation_timestamp", "historical_decision_eligibility_timestamp"])
@pytest.mark.parametrize("invalid", [None, pd.NaT, "2020-01-06", datetime(2020, 1, 6)])
def test_feature_and_decision_clocks_must_be_explicit_aware_datetimes(field, invalid):
    with pytest.raises(ValueError, match="timezone-aware"):
        identity(**{field: invalid})


def test_original_naive_date_label_is_preserved_but_never_used_as_market_close():
    source_label = pd.Timestamp("2020-01-06")
    row = identity(feature_row_id=source_label)
    record = export(identities=[row], predictions=[PredictedClassRow("prediction-0", 2)],
                    probability_rows=[ProbabilityRow("prediction-0", (0.1, 0.2, 0.7))])[0]
    assert record.identity.feature_row_id is source_label
    assert record.identity.feature_observation_timestamp == BASE
    assert record.identity.feature_observation_timestamp != source_label
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(row, feature_observation_timestamp=source_label)


@pytest.mark.parametrize("offset", [timedelta(0), -timedelta(minutes=1)])
def test_decisions_must_follow_feature_observation(offset):
    with pytest.raises(ValueError, match="strictly after"):
        identity(historical_decision_eligibility_timestamp=BASE + offset)


@pytest.mark.parametrize("field", ["training_labels_available_at", "feature_selection_information_available_at",
                                   "validation_labels_available_at", "early_stopping_information_available_at"])
def test_information_cutoff_must_cover_every_supervised_information_source(field):
    with pytest.raises(ValueError, match="every declared information source"):
        provenance(**{field: BASE})


@pytest.mark.parametrize("field", ["model_information_cutoff", "historical_model_available_at",
                                   "training_labels_available_at", "feature_selection_information_available_at",
                                   "validation_labels_available_at", "early_stopping_information_available_at"])
@pytest.mark.parametrize("bad", [None, datetime(2020, 1, 1)])
def test_missing_or_naive_provenance_clocks_are_rejected(field, bad):
    with pytest.raises(ValueError, match="timezone-aware"):
        provenance(**{field: bad})


def test_training_label_maturity_alone_cannot_establish_model_availability():
    with pytest.raises(ValueError, match="every declared information source"):
        provenance(model_information_cutoff=BASE - timedelta(days=7))
    later_validation = provenance(model_information_cutoff=BASE + timedelta(days=1),
                                   validation_labels_available_at=BASE + timedelta(days=1),
                                   early_stopping_information_available_at=BASE + timedelta(days=1),
                                   historical_model_available_at=BASE + timedelta(days=1))
    with pytest.raises(ValueError, match="Decision precedes"):
        export(provenance=later_validation)


def test_early_stopping_availability_is_not_inferred_from_training_or_validation_end():
    later_stopping = provenance(model_information_cutoff=BASE + timedelta(hours=1),
                                early_stopping_information_available_at=BASE + timedelta(hours=1),
                                historical_model_available_at=BASE + timedelta(hours=1))
    with pytest.raises(ValueError, match="Decision precedes"):
        export(provenance=later_stopping)


def test_decision_after_information_cutoff_but_before_model_availability_is_rejected():
    delayed_fit = provenance(historical_model_available_at=BASE + timedelta(minutes=2))
    with pytest.raises(ValueError, match="Decision precedes"):
        export(provenance=delayed_fit)


def test_decision_at_declared_availability_is_eligible():
    available = provenance(historical_model_available_at=BASE + timedelta(minutes=1))
    assert len(export(provenance=available)) == 2


def test_model_availability_cannot_precede_declared_information():
    with pytest.raises(ValueError, match="cannot precede"):
        provenance(historical_model_available_at=BASE - timedelta(days=2))


@pytest.mark.parametrize("options", [
    {"uses_validation": False}, {"uses_early_stopping": False},
    {"uses_validation": True, "validation_labels_available_at": None},
    {"uses_early_stopping": True, "early_stopping_information_available_at": None},
    {"uses_validation": 1}, {"uses_early_stopping": "true"},
])
def test_usage_flags_cannot_hide_required_or_inconsistent_information(options):
    with pytest.raises(ValueError):
        provenance(**options)


def test_explicitly_disabled_validation_and_early_stopping_require_no_fake_timestamps():
    unused = provenance(uses_validation=False, uses_early_stopping=False,
                         validation_labels_available_at=None, early_stopping_information_available_at=None)
    assert len(export(provenance=unused)) == 2


def test_historical_replay_keeps_actual_later_generation_separate_from_eligibility():
    actual_fit = datetime(2026, 1, 1, tzinfo=timezone.utc)
    generated = actual_fit + timedelta(minutes=2)
    records = export(provenance=provenance(actual_fit_completed_at=actual_fit),
                     actual_prediction_generated_at=generated)
    assert all(record.mode == "historical_replay" for record in records)
    assert all(record.actual_prediction_generated_at == generated for record in records)
    assert records[0].identity.historical_decision_eligibility_timestamp == BASE + timedelta(minutes=1)
    assert records[0].identity.historical_decision_eligibility_timestamp < actual_fit


def test_unknown_historical_generation_time_remains_unknown():
    assert all(record.actual_prediction_generated_at is None for record in export())


def test_live_generation_preserves_actual_late_time_without_backdating():
    generated = BASE + timedelta(days=2)
    records = export(provenance=provenance(actual_fit_completed_at=BASE - timedelta(hours=1)),
                     mode="live_generation", actual_prediction_generated_at=generated)
    assert all(record.actual_prediction_generated_at == generated for record in records)
    assert records[0].identity.historical_decision_eligibility_timestamp < generated


@pytest.mark.parametrize("generated,actual_fit", [(None, None), (BASE + timedelta(days=2), None),
                                                  (None, BASE - timedelta(hours=1))])
def test_live_generation_requires_actual_fit_and_generation_clocks(generated, actual_fit):
    with pytest.raises(ValueError, match="Live generation requires"):
        export(provenance=provenance(actual_fit_completed_at=actual_fit), mode="live_generation",
               actual_prediction_generated_at=generated)


@pytest.mark.parametrize("generated", [datetime(2020, 1, 8), pd.NaT, "2020-01-08"])
def test_invalid_actual_generation_clocks_are_not_localized(generated):
    with pytest.raises(ValueError, match="timezone-aware"):
        export(actual_prediction_generated_at=generated)


def test_generation_cannot_precede_feature_observation_or_actual_fit():
    with pytest.raises(ValueError, match="observed information"):
        export(actual_prediction_generated_at=BASE - timedelta(minutes=1))
    with pytest.raises(ValueError, match="actual fit completion"):
        export(provenance=provenance(actual_fit_completed_at=BASE + timedelta(days=3)),
               actual_prediction_generated_at=BASE + timedelta(days=2))
    with pytest.raises(ValueError, match="decision eligibility"):
        export(identities=[identity()], predictions=[PredictedClassRow("prediction-0", 2)],
               probability_rows=[ProbabilityRow("prediction-0", (0.1, 0.2, 0.7))], mode="live_generation",
               provenance=provenance(actual_fit_completed_at=BASE - timedelta(hours=1)),
               actual_prediction_generated_at=BASE)


def test_unknown_generation_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        export(mode="backdated_live")


@pytest.mark.parametrize("forbidden", ["target", "target_return", "future_close", "future_realized_price"])
def test_export_api_does_not_accept_realized_target_or_future_price_inputs(forbidden):
    with pytest.raises(TypeError, match="unexpected keyword"):
        export(**{forbidden: [0.9, -0.9]})


def test_predicted_class_is_preserved_instead_of_recomputed_from_probabilities():
    records = export(predictions=[PredictedClassRow("prediction-0", 0), PredictedClassRow("prediction-1", 2)])
    assert [record.predicted_class for record in records] == [0, 2]
    assert records[0].probabilities[2] == 0.7


def test_descriptive_target_metadata_retains_configured_observation_horizon():
    metadata = PredictiveTargetMetadata("5d_class", 10, 0.01)
    assert all(record.target_metadata == metadata for record in export(target_metadata=metadata))
    with pytest.raises(ValueError, match="configured 5d_class"):
        PredictiveTargetMetadata("5d_return", 5, 0.005)


def test_exports_copy_mutable_probabilities_and_are_immutable():
    values = [0.1, 0.2, 0.7]
    row = ProbabilityRow("prediction-0", values)
    values[0] = 0.9
    assert row.values == (0.1, 0.2, 0.7)
    source = {0: 0.1, 1: 0.2, 2: 0.7}
    record = ExportedPrediction(identity(), 2, source, provenance(), TARGET, "historical_replay")
    source[0] = 0.9
    assert record.probabilities[0] == 0.1
    with pytest.raises(TypeError):
        record.probabilities[0] = 0.5
    with pytest.raises(FrozenInstanceError):
        record.predicted_class = 1


@pytest.mark.parametrize("mapping", [{0: 0.5, 1: 0.5}, {0: 0.1, 1: 0.2, 2: 0.6, 3: 0.1},
                                     {"0": 0.1, 1: 0.2, 2: 0.7}, {False: 0.1, 1: 0.2, 2: 0.7}])
def test_direct_export_construction_cannot_bypass_class_mapping_validation(mapping):
    with pytest.raises(ValueError):
        ExportedPrediction(identity(), 2, mapping, provenance(), TARGET, "historical_replay")


def test_reordering_identity_input_preserves_keyed_content_without_changing_source_rows():
    first, second = export(), export(identities=[identity(1), identity(0)])
    assert {row.identity.prediction_id: row for row in first} == {
        row.identity.prediction_id: row for row in second}


def test_early_stopping_information_cannot_be_declared_before_its_validation_labels():
    with pytest.raises(ValueError, match="cannot precede its validation"):
        provenance(early_stopping_information_available_at=BASE - timedelta(days=3))


def test_probability_rows_do_not_treat_dictionary_keys_as_probability_columns():
    with pytest.raises(ValueError, match="ordered numeric sequence"):
        ProbabilityRow("prediction-0", {0.1: "down", 0.2: "neutral", 0.7: "up"})
    with pytest.raises(ValueError, match="ordered class sequence"):
        export(estimator_classes={2: "up", 0: "down", 1: "neutral"})

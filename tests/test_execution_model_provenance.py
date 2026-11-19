"""Independent frozen memberships and explicitly declared information clocks.

Synthetic observations occur at 16:00 UTC; this is a fixture convention, not an
inferred exchange closing time. Two-observation maturity is supplied before any
filtering. No model is fitted and no held-out target value is accepted.
"""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest

from src.models.ml import TemporalSplit
from src.evaluation.execution.model_provenance import (
    FittedStateAvailability, SourceRow, freeze_model_manifest,
)
from src.evaluation.execution.prediction_export import (
    ExportedPrediction, PredictionIdentity, PredictiveTargetMetadata,
)


BASE = datetime(2024, 1, 1, 16, tzinfo=timezone.utc)
IDS = pd.date_range("2024-01-01", periods=16, name="original_row")


def clock(day, minutes=0):
    return BASE + timedelta(days=day, minutes=minutes)


def source_rows():
    return [SourceRow(IDS[i], clock(i), clock(i, 1),
                      clock(i + 2) if i < 12 else None,
                      clock(i + 2, 2) if i < 12 else None) for i in range(16)]


def manifest(**changes):
    options = dict(
        temporal_split=TemporalSplit(IDS[:10], IDS[:4], IDS[6:10], 2),
        source_rows=source_rows(), held_out_ids=IDS[12:], outer_purged_ids=IDS[10:12],
        asset_id="synthetic-equity", model_id="model-v1", experiment_id="research-v1",
        training_start=clock(0), validation_start=clock(6), test_start=clock(12),
        target_type="5d_class", target_horizon=2, target_threshold=0.01,
        feature_names=["lag_return", "volatility"],
        feature_selection_available_at=clock(5, 3),
        fitted_states=[FittedStateAvailability("robust_scaler", clock(5, 4))],
        uses_early_stopping=True, early_stopping_available_at=clock(11, 3),
        fitting_latency=timedelta(minutes=10), inference_latency=timedelta(minutes=1),
    )
    return freeze_model_manifest(**(options | changes))


def test_frozen_memberships_and_both_maturity_purges():
    result = manifest()
    assert result.fitting_ids == tuple(IDS[:4])
    assert result.inner_purged_ids == tuple(IDS[4:6])
    assert result.validation_ids == tuple(IDS[6:10])
    assert result.outer_purged_ids == tuple(IDS[10:12])
    assert result.held_out_ids == tuple(IDS[12:])
    assert result.training_start == clock(0)
    assert result.validation_start == clock(6)
    assert result.test_start == clock(12)
    # Equality to either boundary is purged, not admitted as known beforehand.
    rows = {row.row_id: row for row in result.source_rows}
    assert rows[IDS[4]].target_end_at == result.validation_start
    assert rows[IDS[10]].target_end_at == result.test_start


def test_validation_labels_and_early_stopping_determine_information_cutoff():
    result = manifest()
    provenance = result.to_prediction_provenance()
    assert provenance.training_labels_available_at == clock(5, 2)
    assert provenance.validation_labels_available_at == clock(11, 2)
    assert result.validation_information_cutoff == clock(11, 2)
    assert result.model_information_cutoff == clock(11, 3)
    assert result.earliest_historical_decision == clock(11, 14)
    assert provenance.actual_fit_completed_at is None
    assert provenance.split_id == result.split_id


def test_without_early_stopping_validation_information_remains_conservatively_included():
    result = manifest(uses_early_stopping=False, early_stopping_available_at=None)
    assert result.model_information_cutoff == clock(11, 2)


def test_delayed_preprocessing_state_extends_model_availability():
    result = manifest(fitted_states=[FittedStateAvailability("scaler", clock(13))])
    assert result.model_information_cutoff == clock(13)
    with pytest.raises(ValueError, match="precedes model availability"):
        result.validate_decision(clock(12))


def test_model_latency_is_distinct_from_information_and_actual_fit_completion():
    result = manifest(actual_fit_completed_at=clock(100))
    assert result.model_information_cutoff == clock(11, 3)
    assert result.earliest_historical_decision == clock(11, 14)
    assert result.to_prediction_provenance().actual_fit_completed_at == clock(100)
    result.validate_decision(clock(11, 14))
    with pytest.raises(ValueError, match="precedes model availability"):
        result.validate_decision(clock(11, 14) - timedelta(microseconds=1))


@pytest.mark.parametrize("name", ["training_start", "test_start", "validation_start",
                                  "feature_selection_available_at", "early_stopping_available_at"])
@pytest.mark.parametrize("invalid", [None, datetime(2024, 1, 1), pd.NaT])
def test_missing_and_naive_manifest_clocks_fail(name, invalid):
    with pytest.raises(ValueError):
        manifest(**{name: invalid})


@pytest.mark.parametrize("name", ["observation_at", "features_available_at", "target_end_at", "labels_available_at"])
def test_naive_row_availability_cannot_be_inferred_from_identity(name):
    with pytest.raises(ValueError):
        replace(source_rows()[0], **{name: datetime(2024, 1, 1)})


@pytest.mark.parametrize("name", ["target_end_at", "labels_available_at"])
def test_training_candidate_requires_maturity_and_label_availability(name):
    rows = source_rows()
    rows[0] = replace(rows[0], **{name: None, **({"labels_available_at": None} if name == "target_end_at" else {})})
    with pytest.raises(ValueError, match="requires maturity|require maturity"):
        manifest(source_rows=rows)


@pytest.mark.parametrize("changes", [
    {"features_available_at": clock(-1)}, {"target_end_at": clock(0)},
    {"labels_available_at": clock(1)},
])
def test_row_clocks_respect_feature_observation_and_label_maturity(changes):
    with pytest.raises(ValueError):
        replace(source_rows()[0], **changes)


@pytest.mark.parametrize("changes", [
    {"feature_selection_available_at": clock(5)},
    {"early_stopping_available_at": clock(11)},
    {"uses_early_stopping": False}, {"uses_early_stopping": 1},
    {"actual_fit_completed_at": clock(5)},
    {"fitting_latency": timedelta(seconds=-1)}, {"inference_latency": None},
])
def test_inconsistent_information_and_latency_are_rejected(changes):
    with pytest.raises(ValueError):
        manifest(**changes)


@pytest.mark.parametrize("changes", [
    {"held_out_ids": IDS[11:]}, {"held_out_ids": IDS[13:]},
    {"held_out_ids": [IDS[12], IDS[12], *IDS[13:]]},
    {"outer_purged_ids": IDS[9:12]}, {"outer_purged_ids": []},
    {"held_out_ids": IDS[12:][::-1]},
])
def test_overlapping_missing_duplicate_and_reordered_membership_fails(changes):
    with pytest.raises(ValueError, match="Membership|membership"):
        manifest(**changes)


@pytest.mark.parametrize("split", [
    TemporalSplit(IDS[:10], IDS[:5], IDS[6:10], 1),
    TemporalSplit(IDS[:10], IDS[:4], IDS[5:10], 1),
    TemporalSplit(IDS[:10], IDS[:4], IDS[6:10], 3),
    TemporalSplit(IDS[:10][::-1], IDS[:4], IDS[6:10], 2),
    TemporalSplit(IDS[:10], IDS[:4], IDS[3:10], 0),
    TemporalSplit(IDS[:10], IDS[:4], IDS[6:11], 2),
])
def test_temporal_split_inconsistency_and_unpurged_labels_fail(split):
    with pytest.raises(ValueError):
        manifest(temporal_split=split)


def test_original_maturity_remains_valid_when_observation_rows_are_missing():
    # Remove an interior fitting row without shifting any future observation.
    rows = [row for row in source_rows() if row.row_id != IDS[1]]
    result = manifest(source_rows=rows, temporal_split=TemporalSplit(
        IDS[:10].delete(1), IDS[:4].delete(1), IDS[6:10], 2))
    assert result.fitting_ids == (IDS[0], IDS[2], IDS[3])
    assert result.inner_purged_ids == tuple(IDS[4:6])
    assert result.source_rows[0].target_end_at == clock(2)


def test_no_validation_is_supported_without_fabricating_clocks():
    result = manifest(
        temporal_split=TemporalSplit(IDS[:10], IDS[:10], IDS[:0], 0), validation_start=None,
        uses_early_stopping=False, early_stopping_available_at=None,
        feature_selection_available_at=clock(11, 3),
        fitted_states=[FittedStateAvailability("robust_scaler", clock(11, 4))])
    assert result.validation_ids == result.inner_purged_ids == ()
    assert result.validation_information_cutoff is None
    assert not result.to_prediction_provenance().uses_validation


@pytest.mark.parametrize("change", [
    {"feature_names": ["volatility", "lag_return"]}, {"target_horizon": 5},
    {"target_type": "5d_return", "target_threshold": None}, {"target_threshold": 0.02},
    {"model_id": "model-v2"},
    {"experiment_id": "experiment-v2"}, {"asset_id": "other-equity"},
    {"feature_selection_available_at": clock(5, 4)},
    {"early_stopping_available_at": clock(11, 4)},
    {"fitting_latency": timedelta(minutes=11)},
    {"actual_fit_completed_at": clock(20)},
    {"fitted_states": [FittedStateAvailability("scaler", clock(5, 5))]},
])
def test_relevant_configuration_and_availability_changes_change_identity(change):
    assert manifest(**change).split_id != manifest().split_id


def test_membership_changes_change_identity_even_when_boundaries_do_not():
    original = manifest()
    rows = [row for row in source_rows() if row.row_id != IDS[1]]
    changed = manifest(source_rows=rows, temporal_split=TemporalSplit(
        IDS[:10].delete(1), IDS[:4].delete(1), IDS[6:10], 2))
    assert changed.split_id != original.split_id


def test_identical_manifests_have_canonical_serialization_and_ids():
    first, second = manifest(), manifest()
    assert first.canonical_json() == second.canonical_json()
    assert first.split_id == second.split_id
    assert len(first.split_id.removeprefix("apf-temporal-v1-")) == 64
    assert json.loads(first.canonical_json())["schema_version"] == 1
    assert "labels_available_at" in first.canonical_json()


def test_submicrosecond_availability_changes_are_not_lost_in_hashing():
    first = manifest(early_stopping_available_at=pd.Timestamp(clock(11, 3)))
    second = manifest(early_stopping_available_at=pd.Timestamp(clock(11, 3)) + pd.Timedelta(1, "ns"))
    assert first.split_id != second.split_id


def test_equivalent_aware_availability_clocks_have_identical_hashes():
    offset = timezone(timedelta(hours=2))
    first = manifest()
    second = manifest(early_stopping_available_at=clock(11, 3).astimezone(offset),
                      feature_selection_available_at=clock(5, 3).astimezone(offset))
    assert first == second
    assert first.split_id == second.split_id


def test_future_data_outside_frozen_universe_cannot_change_manifest():
    external = source_rows()
    frozen = manifest(source_rows=external[:16])
    identity, serialized = frozen.split_id, frozen.canonical_json()
    external.append(SourceRow("future-row", clock(30), clock(30, 1)))
    assert frozen.split_id == identity and frozen.canonical_json() == serialized
    assert manifest(source_rows=external[:16]).split_id == identity
    with pytest.raises(ValueError, match="Membership"):
        manifest(source_rows=external)


def test_inputs_are_copied_and_manifest_and_children_are_immutable():
    rows, features = source_rows(), ["lag_return", "volatility"]
    split = TemporalSplit(IDS[:10].copy(), IDS[:4].copy(), IDS[6:10].copy(), 2)
    result = manifest(source_rows=rows, feature_names=features, temporal_split=split)
    identifier = result.split_id
    rows.clear()
    features.reverse()
    split.source_index.name = "changed-name"
    assert result.split_id == identifier
    assert result.feature_names == ("lag_return", "volatility")
    with pytest.raises(FrozenInstanceError):
        result.asset_id = "changed"
    with pytest.raises(FrozenInstanceError):
        result.source_rows[0].observation_at = clock(100)


def test_held_out_target_values_are_not_an_accepted_input():
    with pytest.raises(TypeError):
        manifest(target=[0, 1, 2])
    with pytest.raises(TypeError):
        manifest(target_return=[0.1, -0.1])


def test_held_out_and_purged_information_does_not_enter_model_cutoff():
    rows = source_rows()
    for index in (4, 5, 10, 11, 12, 13, 14, 15):
        rows[index] = replace(rows[index], target_end_at=rows[index].target_end_at or clock(20),
                              labels_available_at=clock(100), features_available_at=clock(100))
    assert manifest(source_rows=rows).model_information_cutoff == clock(11, 3)


@pytest.mark.parametrize("target_type,horizon", [("5d_return", 2), ("1d_return", 1)])
def test_configured_regression_metadata_remains_supported(target_type, horizon):
    changes = {}
    if horizon == 1:
        changes = dict(source_rows=[replace(row, target_end_at=clock(i + 1),
                                           labels_available_at=clock(i + 1, 2)) if i < 12 else row
                                    for i, row in enumerate(source_rows())],
                       temporal_split=TemporalSplit(IDS[:11], IDS[:5], IDS[6:11], 1),
                       outer_purged_ids=IDS[11:12])
    result = manifest(target_type=target_type, target_horizon=horizon, target_threshold=None, **changes)
    assert result.target_type == target_type
    assert result.to_prediction_provenance().model_id == "model-v1"


def test_prediction_export_accepts_manifest_provenance_and_rejects_early_decisions():
    result = manifest()
    identity = PredictionIdentity("synthetic-equity", "pred-12", "session-12", IDS[12],
                                  clock(12), clock(12, 1))
    exported = ExportedPrediction(identity, 2, {0: 0.1, 1: 0.2, 2: 0.7},
                                  result.to_prediction_provenance(),
                                  PredictiveTargetMetadata("5d_class", 2, 0.01), "historical_replay")
    assert exported.provenance.split_id == result.split_id
    delayed = manifest(fitting_latency=timedelta(days=3))
    with pytest.raises(ValueError, match="Decision precedes"):
        replace(exported, provenance=delayed.to_prediction_provenance())


@pytest.mark.parametrize("change", [
    {"target_horizon": 0}, {"target_horizon": True}, {"target_type": "unknown"},
    {"target_threshold": None}, {"target_threshold": -0.1}, {"target_threshold": float("nan")},
    {"target_threshold": True}, {"target_type": "5d_return"},
    {"target_type": "1d_return", "target_horizon": 2},
    {"feature_names": ["target_end_timestamp"]}, {"feature_names": ["target"]},
    {"feature_names": ["x", "x"]}, {"feature_names": []},
    {"fitted_states": [FittedStateAvailability("a", clock(5)), FittedStateAvailability("a", clock(6))]},
    {"fitted_states": [FittedStateAvailability("scaler", clock(0))]},
])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        manifest(**change)

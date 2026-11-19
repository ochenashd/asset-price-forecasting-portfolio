"""Local post-fit consistency receipts, not independently witnessed training.

Only real fitted sklearn Random Forest/RobustScaler and XGBoost classifier
objects are supported in V1. No pickle, fitting, inference or external I/O is
used. Only local code-provenance bytes are read; market artifacts are not
reopened. Caller-supplied event metadata cannot establish historical existence.
"""

from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .protocol import (
    ResearchProtocolError, ProtocolDiagnostic, ResearchProtocolV1, ResearchArtifactRef,
    FrozenPartitionSpec, TargetSpec, EnvironmentSpec, _Section, _encode, _decode, _json,
    _digest, _key, validate_research_protocol,
)
from .input_adapter import PreparedResearchInputs, _check_recipe, _view_digest
from src.evaluation.execution.model_provenance import FrozenModelManifest
from src.evaluation.execution.pipeline_handoff import (
    FittedModelBinding, TargetConfiguration, CaptureReceipt, capture_fingerprint,
    _digest as binding_digest,
)
from src.evaluation.execution.prediction_export import _utc_instant, ProbabilityRow


_LIMITATIONS = (
    "Local supplied artifact/state consistency only; no independent fitting witness.",
    "Actual split, attempt association and completion timestamps are supplied by the caller.",
    "Temporal availability and latencies are historical replay declarations, not evidence of model existence.",
    "Historical price vintage remains unknown; execution eligibility is unassessed.",
)


class ResearchReceiptError(ResearchProtocolError):
    """Safe structured failure; no object repr, model bytes or secret values."""


def _fail(code, field):
    raise ResearchReceiptError((ProtocolDiagnostic(code, field, "Post-fit consistency validation failed."),))


def _require(condition, code, field):
    if not condition:
        _fail(code, field)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _state(value):
    """Canonical supported learned state; structured padding is never hashed."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            _require(all(isinstance(item, str) for item in value.flat), "unsupported_serialization", "fitted_state")
            return {"dtype": value.dtype.str, "shape": tuple(value.shape), "strings": tuple(value.flat)}
        if value.dtype.names:
            return {"dtype": value.dtype.descr, "shape": tuple(value.shape),
                    "fields": {name: _state(value[name]) for name in value.dtype.names}}
        return {"dtype": value.dtype.str, "shape": tuple(value.shape),
                "data_hex": np.ascontiguousarray(value).tobytes().hex()}
    if isinstance(value, Mapping):
        return {key: _state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_state(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return value
    _fail("unsupported_serialization", "fitted_state")


def _bytes(value):
    return _json(_encode(_state(value))).encode("utf-8")


def _ids(values):
    return tuple(_key(value) for value in values)


def _instants(values):
    return tuple(_utc_instant(value).isoformat() for value in values)


@dataclass(frozen=True)
class ResearchFitAttempt:
    """Caller-declared association with the actual wrapper split; not a witness."""
    attempt_id: str
    ticker: str
    asset_id: str
    experiment_id: str
    protocol_id: str
    training_input_id: str
    temporal_manifest_id: str
    temporal_split: object = field(repr=False)
    actual_fit_completed_at: datetime


@dataclass(frozen=True)
class InspectedResearchFit:
    """Owned serialized binding; never contains a live estimator reference."""
    binding: FittedModelBinding = field(repr=False)
    selector_state_sha256: str
    scaler_state_sha256: str
    estimator_state_sha256: str
    preprocessing_state_sha256: str
    effective_parameters_json: str
    boosting_rounds: int
    best_iteration: int | None
    best_score: str | None
    # Exact images from this inspection; defaults preserve construction callers.
    selector_state: bytes = field(default=b"", repr=False)
    scaler_state: bytes = field(default=b"", repr=False)
    native_booster_json: bytes = field(default=b"", repr=False)


@dataclass(frozen=True)
class ResearchFitReceiptV1(_Section):
    """Immutable content record. Deserialization verifies structure, not events."""
    protocol_id: str
    training_input_id: str
    source_artifacts: tuple[ResearchArtifactRef, ...]
    partitions: FrozenPartitionSpec
    row_identity_bindings: tuple
    selected_features: tuple[str, ...]
    feature_selection_settings: Mapping
    state_digests: Mapping[str, str]
    preprocessing_binding_json: str
    fitted_model_id: str
    target_configuration: TargetSpec
    effective_parameters: Mapping
    boosting_metadata: Mapping
    attempt_id: str
    ticker: str
    asset_id: str
    experiment_id: str
    temporal_manifest_id: str
    temporal_manifest_json: str
    prediction_capture_fingerprint: str
    actual_fit_completed_at: datetime
    actual_prediction_generated_at: datetime
    code_dependency_provenance: EnvironmentSpec
    implementation_reference: str
    receipt_implementation_sha256: str
    limitations: tuple[str, ...]
    schema_version: str = "apf-research-fit-receipt-v1"
    historical_point_in_time_verified: bool = field(default=False, init=False)
    execution_eligible: bool = field(default=False, init=False)
    independently_witnessed_training: bool = field(default=False, init=False)
    historical_point_in_time_status: str = field(default="unknown", init=False)
    execution_eligibility_status: str = field(default="unassessed", init=False)

    def __post_init__(self):
        super().__post_init__()
        _require(self.schema_version == "apf-research-fit-receipt-v1", "unsupported_receipt_version", "schema_version")
        _require(self.limitations == _LIMITATIONS, "missing_provenance", "limitations")
        for name in ("protocol_id", "training_input_id", "fitted_model_id", "attempt_id", "ticker", "asset_id", "experiment_id",
                     "temporal_manifest_id", "prediction_capture_fingerprint", "implementation_reference"):
            _require(isinstance(getattr(self, name), str) and bool(getattr(self, name).strip()), "missing_provenance", name)
        _require(isinstance(self.partitions, FrozenPartitionSpec) and isinstance(self.target_configuration, TargetSpec)
                 and isinstance(self.code_dependency_provenance, EnvironmentSpec) and bool(self.source_artifacts)
                 and all(isinstance(ref, ResearchArtifactRef) for ref in self.source_artifacts), "invalid_receipt", "contracts")
        _require(bool(self.selected_features) and len(set(self.selected_features)) == len(self.selected_features)
                 and set(self.state_digests) == {"selector", "scaler", "estimator", "preprocessing"}, "invalid_receipt", "state_digests")
        _require(set(self.boosting_metadata) == {"rounds", "best_iteration", "best_score", "uses_early_stopping"}
                 and type(self.boosting_metadata["rounds"]) is int
                 and 0 < self.boosting_metadata["rounds"] <= self.effective_parameters["n_estimators"]
                 and type(self.boosting_metadata["uses_early_stopping"]) is bool,
                 "boosting_metadata_mismatch", "rounds")
        best = self.boosting_metadata["best_iteration"]
        _require((self.boosting_metadata["uses_early_stopping"] and type(best) is int and 0 <= best < self.boosting_metadata["rounds"])
                 or (not self.boosting_metadata["uses_early_stopping"] and best is None), "boosting_metadata_mismatch", "best_iteration")
        for value in tuple(self.state_digests.values()) + (self.prediction_capture_fingerprint, self.receipt_implementation_sha256):
            _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid_digest", "state_digests")
        descriptor = _decode(json.loads(self.preprocessing_binding_json))
        _require(_sha(self.preprocessing_binding_json.encode()) == self.state_digests["preprocessing"]
                 and descriptor["training_input_id"] == self.training_input_id
                 and tuple(descriptor["selected_features"]) == self.selected_features
                 and descriptor["selector"] == self.state_digests["selector"]
                 and descriptor["scaler"] == self.state_digests["scaler"], "fitted_state_mismatch", "preprocessing_binding")
        recipe = _decode(descriptor["recipe"])
        _require(_encode(self.feature_selection_settings) == _encode({"correlation": recipe.correlation, "importance": recipe.feature_selection})
                 and self.implementation_reference == recipe.implementation_reference,
                 "fitted_state_mismatch", "selection_recipe")
        _require(_utc_instant(self.actual_prediction_generated_at) >= _utc_instant(self.actual_fit_completed_at),
                 "invalid_completion_chronology", "timestamps")
        _require(self.temporal_manifest_id == "apf-temporal-v1-" + _sha(self.temporal_manifest_json.encode()),
                 "temporal_identity_mismatch", "temporal_manifest_id")
        model_id = "apf-fitted-model-v1-" + binding_digest((self.experiment_id, self.selected_features,
            self.target_configuration.task_type, self.target_configuration.horizon, self.target_configuration.threshold,
            ("selector", "scaler"), self.boosting_metadata["uses_early_stopping"],
            self.state_digests["estimator"], self.state_digests["preprocessing"]))
        _require(model_id == self.fitted_model_id, "fitted_state_mismatch", "fitted_model_id")

    def canonical_json(self):
        return _json({"fields": {item.name: _encode(getattr(self, item.name)) for item in fields(self)}})

    @property
    def receipt_id(self):
        return "apf-research-fit-receipt-v1-" + _sha(self.canonical_json().encode())

    @classmethod
    def from_json(cls, text):
        """Check canonical content only; actual objects must be inspected again."""
        try:
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    _require(key not in result, "invalid_serialization", "duplicate_fields")
                    result[key] = value
                return result
            payload = json.loads(text, object_pairs_hook=unique)
            _require(set(payload) == {"fields"} and set(payload["fields"]) == {item.name for item in fields(cls)},
                     "invalid_serialization", "fields")
            decoded = {key: _decode(value) for key, value in payload["fields"].items()}
            _require(all(decoded[name] is False for name in ("historical_point_in_time_verified", "execution_eligible", "independently_witnessed_training")),
                     "unsupported_certification", "limitations")
            result = cls(**{item.name: decoded[item.name] for item in fields(cls) if item.init})
            _require(result.canonical_json() == text, "noncanonical_serialization", "receipt")
            return result
        except ResearchReceiptError:
            raise
        except Exception as exc:
            _fail("invalid_serialization", type(exc).__name__)


def _verify_prepared(protocol, prepared):
    validate_research_protocol(protocol).require_valid()
    _check_recipe(protocol)
    _require(isinstance(prepared, PreparedResearchInputs), "unsupported_prepared_inputs", "prepared")
    _require(prepared.protocol_id == protocol.protocol_id and prepared.source_artifacts == protocol.artifacts,
             "protocol_mismatch", "prepared")
    _require(prepared.frozen_partitions == protocol.partitions and prepared.maturity_mapping == protocol.target.maturity,
             "membership_mismatch", "prepared")
    _require(prepared.source_training_input_id == protocol.training_input_id, "training_input_mismatch", "raw_context")
    identifier = "apf-prepared-training-v1-" + _digest((protocol.training_input_id, _view_digest(prepared.fitting), _view_digest(prepared.validation)))
    _require(identifier == prepared.training_input_id, "training_input_mismatch", "prepared_values")
    for name, partition_name in (("fitting", "fitting"), ("inner_purged", "inner_purge"), ("validation", "validation"),
                                  ("outer_purged", "outer_purge"), ("held_out", "held_out")):
        part = getattr(prepared, name)
        _require(_ids(part.row_ids) == _ids(getattr(protocol.partitions, partition_name))
                 and tuple(part.features.columns) == protocol.transformations.feature_order,
                 "membership_mismatch", name)
    _require(_ids(row.row_id for row in prepared.row_bindings) == _ids(row.row_id for row in protocol.universe.engineered_rows),
             "membership_mismatch", "engineered_rows")
    observations = {_key(row.source_id): row for row in protocol.universe.observations}
    ends = {_key(row): end for row, end in protocol.target.maturity}
    for binding, original in zip(prepared.row_bindings, protocol.universe.engineered_rows):
        end = ends[_key(original.row_id)]
        _require(_key(binding.source_id) == _key(original.source_id)
                 and _key(binding.target_end_source_id) == _key(end)
                 and _utc_instant(binding.observation_at) == _utc_instant(observations[_key(original.source_id)].observation_at)
                 and _utc_instant(binding.target_end_timestamp) == _utc_instant(observations[_key(end)].observation_at),
                 "maturity_mismatch", "prepared_bindings")
    for name in ("fitting", "inner_purged", "validation", "outer_purged", "held_out"):
        part = getattr(prepared, name)
        _require(_instants(part.features.index) == _instants(row.observation_at for row in part.row_bindings)
                 and part.labels.index.equals(part.features.index), "membership_mismatch", name)
        original = {_key(row.row_id): row for row in prepared.row_bindings}
        for row in part.row_bindings:
            expected = original[_key(row.row_id)]
            _require(_key(row.source_id) == _key(expected.source_id)
                     and _key(row.target_end_source_id) == _key(expected.target_end_source_id)
                     and _utc_instant(row.target_end_timestamp) == _utc_instant(expected.target_end_timestamp),
                     "maturity_mismatch", name)


def inspect_research_fit(protocol: ResearchProtocolV1, prepared: PreparedResearchInputs, *,
                         experiment_id: str, selector, scaler, estimator, selected_features) -> InspectedResearchFit:
    """Inspect actual supported states without fitting, transforming or predicting.

    This checks local state and deterministic scaling consistency. It does not
    witness the original fitting call or prove which rows reached an estimator.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import RobustScaler
        from sklearn.preprocessing._data import _handle_zeros_in_scale
        import sklearn
        import xgboost as xgb
        from src.features.selection import remove_high_correlation
        _verify_prepared(protocol, prepared)
        _require(isinstance(experiment_id, str) and bool(experiment_id.strip()), "missing_provenance", "experiment_id")
        _require(protocol.target.task_type == "5d_class", "unsupported_fit_type", "target")
        _require(type(selector) is RandomForestClassifier, "missing_or_unsupported_selector", "selector")
        _require(type(scaler) is RobustScaler, "missing_or_unsupported_scaler", "scaler")
        _require(type(estimator) is xgb.XGBClassifier, "unsupported_estimator", "estimator")
        _require(all(protocol.environment.package_versions.get(name) == version for name, version in
                     (("scikit-learn", sklearn.__version__), ("xgboost", xgb.__version__))), "fitting_environment_mismatch", "versions")
        fitting = prepared.fitting.features
        fitting["target"] = prepared.fitting.labels
        correlation_names = remove_high_correlation(fitting, threshold=protocol.transformations.correlation["threshold"])
        _require(tuple(selector.feature_names_in_) == tuple(correlation_names)
                 and selector.n_features_in_ == len(correlation_names), "feature_order_mismatch", "selector")
        selection = protocol.transformations.feature_selection
        actual = selector.get_params(deep=False)
        expected_selector = RandomForestClassifier(**{key: selection[key] for key in ("n_estimators", "max_depth", "random_state", "n_jobs")},
                                                  class_weight=selection["classification_class_weight"])
        _require(_encode(actual) == _encode(expected_selector.get_params(deep=False))
                 and len(selector.estimators_) == selection["n_estimators"]
                 and tuple(selector.classes_) == (0, 1, 2), "selector_configuration_mismatch", "selector")
        names = tuple(pd.Series(selector.feature_importances_, index=correlation_names).nlargest(selection["n_features"]).index)
        selected_features = tuple(selected_features)
        _require(names == selected_features, "feature_order_mismatch", "selected_features")
        _require(tuple(scaler.feature_names_in_) == names and tuple(estimator.feature_names_in_) == names,
                 "feature_order_mismatch", "fitted_states")
        scaler_params = {key: value for key, value in protocol.transformations.scaler.items() if key not in {"class", "scope", "stage"}}
        _require(_encode(scaler.get_params()) == _encode(scaler_params), "scaler_configuration_mismatch", "scaler")
        values = prepared.fitting.features[list(names)].to_numpy()
        center = np.nanmedian(values, axis=0)
        quantiles = np.nanpercentile(values, scaler.quantile_range, axis=0)
        scale = _handle_zeros_in_scale(quantiles[1] - quantiles[0], copy=True)
        _require(scaler.n_features_in_ == len(names) and np.array_equal(scaler.center_, center)
                 and np.array_equal(scaler.scale_, scale), "scaler_state_mismatch", "scaler")
        parameters = estimator.get_params(deep=False)
        expected_estimator = xgb.XGBClassifier(**dict(protocol.learning.effective_parameters))
        _require(_encode(parameters) == _encode(expected_estimator.get_params(deep=False)),
                 "model_parameters_mismatch", "estimator")
        _require(tuple(estimator.classes_) == (0, 1, 2), "class_order_mismatch", "estimator")
        booster = estimator.get_booster()
        _require(tuple(booster.feature_names) == names, "feature_order_mismatch", "booster")
        rounds = booster.num_boosted_rounds()
        early = protocol.learning.early_stopping["enabled"]
        _require(0 < rounds <= parameters["n_estimators"] and bool(parameters.get("early_stopping_rounds")) == early,
                 "boosting_metadata_mismatch", "rounds")
        _require(protocol.learning.model_type == "xgboost" and protocol.learning.training_policy == "fitting_only"
                 and protocol.learning.validation_policy == "fixed_purged_validation"
                 and protocol.learning.random_seeds.get("selector") == selection["random_state"]
                 and protocol.learning.random_seeds.get("estimator") == parameters["random_state"]
                 and (not early or protocol.learning.early_stopping["rounds"] == parameters["early_stopping_rounds"]),
                 "model_parameters_mismatch", "learning_policy")
        configuration = json.loads(booster.save_config())
        _require(configuration["learner"]["objective"]["name"] == parameters["objective"]
                 and int(configuration["learner"]["learner_model_param"]["num_class"]) == 3,
                 "model_parameters_mismatch", "booster_configuration")
        best = int(booster.attr("best_iteration")) if booster.attr("best_iteration") is not None else None
        _require((early and best is not None and 0 <= best < rounds) or (not early and best is None),
                 "boosting_metadata_mismatch", "best_iteration")
        selector_state = _bytes({"class": "sklearn.RandomForestClassifier", "parameters": actual,
            "features": tuple(selector.feature_names_in_), "classes": selector.classes_, "n_outputs": selector.n_outputs_,
            "trees": tuple({"parameters": tree.get_params(deep=False), "classes": tree.classes_,
                            "state": tree.tree_.__getstate__()} for tree in selector.estimators_)})
        scaler_state = _bytes({"class": "sklearn.RobustScaler", "parameters": scaler.get_params(),
                               "learned": {key: value for key, value in vars(scaler).items() if key.endswith("_")}})
        native_booster_json = bytes(booster.save_raw(raw_format="json"))
        estimator_state = _bytes({"format": "xgboost-json-v1", "parameters": parameters,
            "booster": json.loads(native_booster_json), "configuration": configuration})
        preprocessing = _bytes({"training_input_id": prepared.training_input_id, "selected_features": names,
            "selector": _sha(selector_state), "scaler": _sha(scaler_state), "recipe": _encode(protocol.transformations)})
        binding = FittedModelBinding(experiment_id, names,
            TargetConfiguration(protocol.target.task_type, protocol.target.horizon, protocol.target.threshold),
            ("selector", "scaler"), early, estimator_state, preprocessing)
        return InspectedResearchFit(binding, _sha(selector_state), _sha(scaler_state), _sha(estimator_state),
            _sha(preprocessing), _json(_encode(protocol.learning.effective_parameters)), rounds, best, booster.attr("best_score"),
            selector_state, scaler_state, native_booster_json)
    except ResearchReceiptError:
        raise
    except ResearchProtocolError as exc:
        raise ResearchReceiptError(exc.diagnostics) from exc
    except Exception as exc:
        _fail("unsupported_fitted_state", type(exc).__name__)


def create_research_fit_receipt(protocol: ResearchProtocolV1, prepared: PreparedResearchInputs, *,
        attempt: ResearchFitAttempt, selector, scaler, estimator, selected_features,
        temporal_manifest: FrozenModelManifest, capture, fresh_capture_receipt: CaptureReceipt) -> ResearchFitReceiptV1:
    """Reinspect supplied objects and bind an existing capture; no inference occurs."""
    try:
        _require(isinstance(attempt, ResearchFitAttempt), "missing_provenance", "attempt")
        state = inspect_research_fit(protocol, prepared, experiment_id=attempt.experiment_id, selector=selector, scaler=scaler,
                                     estimator=estimator, selected_features=selected_features)
        _require(isinstance(attempt, ResearchFitAttempt) and bool(attempt.attempt_id.strip()), "missing_provenance", "attempt")
        _require(attempt.protocol_id == protocol.protocol_id and attempt.training_input_id == prepared.training_input_id,
                 "training_input_mismatch", "attempt")
        _require(attempt.asset_id == protocol.asset_id and capture.asset_id == protocol.asset_id
                 and attempt.ticker == capture.ticker, "asset_mismatch", "capture")
        _require(isinstance(temporal_manifest, FrozenModelManifest), "invalid_temporal_manifest", "manifest")
        manifest = replace(temporal_manifest)
        _require(attempt.temporal_manifest_id == manifest.split_id, "temporal_identity_mismatch", "manifest")
        _require(manifest.model_id == state.binding.model_id and manifest.experiment_id == attempt.experiment_id
                 and manifest.asset_id == protocol.asset_id, "fitted_state_mismatch", "manifest")
        _require((manifest.target_type, manifest.target_horizon, manifest.target_threshold) ==
                 (protocol.target.task_type, protocol.target.horizon, protocol.target.threshold)
                 and manifest.feature_names == state.binding.feature_names
                 and manifest.uses_early_stopping == protocol.learning.early_stopping["enabled"], "target_or_features_mismatch", "manifest")
        groups = (("fitting", "fitting_ids"), ("inner_purged", "inner_purged_ids"), ("validation", "validation_ids"),
                  ("outer_purged", "outer_purged_ids"), ("held_out", "held_out_ids"))
        for prepared_name, manifest_name in groups:
            _require(_instants(row.observation_at for row in getattr(prepared, prepared_name).row_bindings)
                     == _instants(getattr(manifest, manifest_name)), "membership_mismatch", manifest_name)
        _require(_instants(row.row_id for row in manifest.source_rows) == _instants(row.observation_at for row in prepared.row_bindings),
                 "membership_mismatch", "source_rows")
        for row, binding in zip(manifest.source_rows, prepared.row_bindings):
            _require(_utc_instant(row.observation_at) == _utc_instant(binding.observation_at)
                     and row.target_end_at is not None and _utc_instant(row.target_end_at) == _utc_instant(binding.target_end_timestamp),
                     "maturity_mismatch", "source_rows")
        _require("scaler" in {item.name for item in manifest.fitted_states}, "missing_provenance", "scaler_availability")
        split = attempt.temporal_split
        allowed = set(_instants(manifest.fitting_ids + manifest.inner_purged_ids + manifest.validation_ids))
        source = tuple(row.row_id for row in manifest.source_rows if _utc_instant(row.row_id).isoformat() in allowed)
        _require(_instants(split.source_index) == _instants(source)
                 and _instants(split.fitting_index) == _instants(manifest.fitting_ids)
                 and _instants(split.validation_index) == _instants(manifest.validation_ids)
                 and split.purged_count == len(manifest.inner_purged_ids), "membership_mismatch", "actual_split")
        for name, expected in (("training_source_index", source), ("fitting_index", manifest.fitting_ids),
                               ("validation_index", manifest.validation_ids), ("held_out_index", manifest.held_out_ids)):
            _require(_instants(getattr(capture, name)) == _instants(expected), "capture_membership_mismatch", name)
        _require(capture.feature_names == state.binding.feature_names and capture.estimator_classes == tuple(estimator.classes_),
                 "class_or_feature_order_mismatch", "capture")
        _require(capture.estimator_type == f"{type(estimator).__module__}.{type(estimator).__qualname__}",
                 "unsupported_estimator", "capture")
        _require(all(type(label) is int for label in capture.estimator_classes), "class_order_mismatch", "capture")
        _require(capture.model_id in {None, state.binding.model_id} and capture.split_id in {None, manifest.split_id},
                 "stale_capture", "capture_identity")
        _require(len(capture.records) == len(manifest.held_out_ids)
                 and _instants(row.feature_row_id for row in capture.records) == _instants(manifest.held_out_ids),
                 "capture_membership_mismatch", "records")
        for row in capture.records:
            _require(type(row.predicted_class) is int and row.predicted_class in {0, 1, 2}, "invalid_predictions", "classes")
            _require(set(row.probabilities) == {0, 1, 2} and all(type(label) is int for label in row.probabilities),
                     "invalid_predictions", "probability_classes")
            ProbabilityRow("receipt-probability-row", tuple(row.probabilities[label] for label in capture.estimator_classes))
        fingerprint = capture_fingerprint(capture)
        _require(isinstance(fresh_capture_receipt, CaptureReceipt) and fresh_capture_receipt.run_id == attempt.attempt_id
                 and fresh_capture_receipt.capture_fingerprint == fingerprint
                 and fresh_capture_receipt.fitted_model_id == state.binding.model_id
                 and fresh_capture_receipt.stable_asset_id == protocol.asset_id, "stale_capture_or_state", "fresh_capture_receipt")
        fitting = _utc_instant(attempt.actual_fit_completed_at)
        generated = _utc_instant(capture.actual_prediction_generated_at)
        _require(manifest.actual_fit_completed_at is not None and fitting == _utc_instant(manifest.actual_fit_completed_at)
                 and fitting >= _utc_instant(manifest.model_information_cutoff) and generated >= fitting
                 and generated >= max(_utc_instant(row.features_available_at) for row in manifest.source_rows),
                 "invalid_completion_chronology", "timestamps")
        return ResearchFitReceiptV1(protocol.protocol_id, prepared.training_input_id, prepared.source_artifacts,
            protocol.partitions, tuple((row.row_id, row.source_id, row.observation_at, row.target_end_source_id) for row in prepared.row_bindings),
            state.binding.feature_names, {"correlation": protocol.transformations.correlation, "importance": protocol.transformations.feature_selection},
            {"selector": state.selector_state_sha256, "scaler": state.scaler_state_sha256,
             "estimator": state.estimator_state_sha256, "preprocessing": state.preprocessing_state_sha256},
            state.binding.preprocessing_state.decode("utf-8"), state.binding.model_id, protocol.target, protocol.learning.effective_parameters,
            {"rounds": state.boosting_rounds, "best_iteration": state.best_iteration, "best_score": state.best_score,
             "uses_early_stopping": state.binding.uses_early_stopping}, attempt.attempt_id, attempt.ticker, protocol.asset_id,
            attempt.experiment_id, manifest.split_id, manifest.canonical_json(), fingerprint, fitting, generated,
            protocol.environment, protocol.transformations.implementation_reference,
            _sha(Path(__file__).read_bytes()), _LIMITATIONS)
    except ResearchReceiptError:
        raise
    except ResearchProtocolError as exc:
        raise ResearchReceiptError(exc.diagnostics) from exc
    except Exception as exc:
        _fail("invalid_receipt_inputs", type(exc).__name__)

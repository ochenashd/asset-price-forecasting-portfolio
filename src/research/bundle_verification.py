"""Independent byte-only research verification; never loads fitted models.

Hashes establish content consistency, not publisher authentication, historical
availability, actual training, runtime compatibility or execution eligibility.
"""
from dataclasses import asdict, dataclass, fields, replace
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .bundle import BundleLimits, BundleValidationError, open_research_bundle


@dataclass(frozen=True)
class VerificationLimits:
    archive: BundleLimits = BundleLimits()
    max_json_depth: int = 64
    max_json_nodes: int = 500000

    def __post_init__(self):
        if not isinstance(self.archive, BundleLimits) or any(type(v) is not int or v <= 0
                for v in (self.max_json_depth, self.max_json_nodes)):
            raise ValueError("Positive verification limits are required.")


@dataclass(frozen=True)
class VerificationCheck:
    status: str
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputDependency:
    artifact_id: str
    relative_path: str
    sha256: str
    byte_length: int
    external_reference_json: str | None
    independently_checked: bool = False


@dataclass(frozen=True)
class BundleVerificationReport:
    expected_bundle_id: str
    checks: tuple[tuple[str, VerificationCheck], ...]
    input_dependencies: tuple[InputDependency, ...] = ()
    provenance_origin: str = "unknown"
    historical_pit_status: str = "unknown"
    execution_eligibility_status: str = "unassessed"
    limitations: tuple[str, ...] = (
        "Content hashes do not authenticate a publisher or witness training.",
        "Historical replay clocks and provenance remain declarations.",
        "No model loading, inference or functional reproduction was performed.",
        "Historical PIT evidence is unknown; execution eligibility is unassessed.",
    )

    def check(self, name):
        return dict(self.checks)[name]

    @property
    def status(self):
        statuses = {item.status for _, item in self.checks}
        for status in ("failed", "operational_error", "incomplete", "unsupported"):
            if status in statuses:
                return status
        return "verified" if self.check("scientific_consistency").status == "verified" else "incomplete"

    def to_dict(self):
        return {"expected_bundle_id": self.expected_bundle_id, "status": self.status,
                "checks": {key: asdict(value) for key, value in self.checks},
                "input_dependencies": [asdict(value) for value in self.input_dependencies],
                "provenance_origin": self.provenance_origin,
                "historical_pit_status": self.historical_pit_status,
                "execution_eligibility_status": self.execution_eligibility_status,
                "limitations": list(self.limitations)}


_NAMES = ("artifact_integrity", "bundle_manifest", "artifact_schemas", "protocol_identity", "receipt_identity",
          "reference_consistency", "scientific_consistency", "input_closure", "training_inputs",
          "state_envelope_hashes", "state_envelope_schema", "temporal_provenance",
          "prediction_capture", "prediction_decision_chronology", "runtime_model_loading",
          "functional_reproduction", "historical_point_in_time", "execution_eligibility")


class _Mismatch(ValueError):
    """Controlled diagnostic code, never a payload representation."""


def _parse(data, limits):
    """Bound nesting before JSON parsing, then bound the decoded node count."""
    depth = 0
    quoted = escaped = False
    for char in data.decode("utf-8"):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > limits.max_json_depth:
                raise ValueError("json_depth_limit")
        elif char in "]}":
            depth -= 1
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite_json")
    result = json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
    stack, count = [result], 0
    while stack:
        value = stack.pop()
        count += 1
        if count > limits.max_json_nodes:
            raise ValueError("json_node_limit")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite_json")
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return result


def _temporal(data, limits):
    """Decode only the existing version-one data-only temporal representation."""
    import pandas as pd
    from src.evaluation.execution.model_provenance import FrozenModelManifest, SourceRow, FittedStateAvailability
    raw = _parse(data, limits)
    if set(raw) != {"schema_version", "manifest"} or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise NotImplementedError("temporal_schema")
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"datetime"}:
                return pd.Timestamp(value["datetime"])
            if set(value) == {"microseconds"}:
                if type(value["microseconds"]) is not int:
                    raise ValueError("invalid_duration")
                return timedelta(microseconds=value["microseconds"])
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return tuple(decode(item) for item in value)
        return value
    values = decode(raw["manifest"])
    if set(values) != {item.name for item in fields(FrozenModelManifest)}:
        raise ValueError("temporal_fields")
    values["source_rows"] = tuple(SourceRow(**row) for row in values["source_rows"])
    values["fitted_states"] = tuple(FittedStateAvailability(**row) for row in values["fitted_states"])
    result = FrozenModelManifest(**values)
    if result.canonical_json().encode() != data:
        raise ValueError("noncanonical_temporal")
    return result


def verify_research_bundle(bundle_directory, expected_bundle_id, *, limits=VerificationLimits()):
    """Verify owned artifact images; external dependencies are never fetched.

    Supported metadata artifacts use format_version "1". Protocol/receipt use
    their canonical contract JSON. Input references, selected features, effective
    parameters and environment use the protocol typed JSON codec and protocol
    schema identity. Temporal data uses the existing canonical version-one
    manifest, with schema identity "apf-temporal-manifest-v1". Preprocessing
    binding bytes must match the receipt exactly. No state/capture codec is
    inferred from arbitrary JSON or from an artifact's claimed schema label.

    Source images are materialized only in a disposable private directory for
    the established secure Parquet verifier. Publication paths are never reopened
    after the inventory reader has returned owned verified byte images.
    """
    checks = {name: VerificationCheck("not_assessed") for name in _NAMES}
    dependencies = ()
    origin = "unknown"
    def put(name, status, *codes):
        checks[name] = VerificationCheck(status, tuple(codes))
    def report():
        return BundleVerificationReport(expected_bundle_id, tuple(checks.items()), dependencies, origin)
    try:
        bundle = open_research_bundle(bundle_directory, expected_bundle_id=expected_bundle_id, limits=limits.archive)
    except BundleValidationError as error:
        codes = tuple(item.code for item in error.diagnostics)
        status = "unsupported" if any("unsupported_schema" in code for code in codes) else "failed"
        put("artifact_integrity", status, *codes)
        put("bundle_manifest", status, *codes)
        return report()
    except OSError as error:
        put("artifact_integrity", "operational_error", type(error).__name__)
        return report()
    except (ValueError, TypeError, RecursionError):
        put("artifact_integrity", "failed", "invalid_or_excessively_nested_archive")
        return report()
    put("artifact_integrity", "verified")
    put("bundle_manifest", "verified")
    artifacts = {item.role: item for item in bundle.artifacts if item.role != "source_input"}
    meta = bundle.manifest.metadata
    def require(value, code):
        if not value:
            raise _Mismatch(code)
    def run(name, operation):
        try:
            operation()
            put(name, "verified")
        except NotImplementedError:
            put(name, "unsupported", "unsupported_version_or_codec")
        except (OSError, ImportError) as error:
            put(name, "operational_error", type(error).__name__)
        except Exception as error:
            diagnostics = getattr(error, "diagnostics", ())
            codes = (str(error),) if isinstance(error, _Mismatch) else tuple(item.code for item in diagnostics)
            # Never disclose exception text, file contents or fitted states.
            put(name, "failed", *(codes or ("invalid_or_inconsistent_" + name,)))
    schema_names = {"protocol": "apf-research-protocol-v1", "receipt": "apf-research-fit-receipt-v1",
        "temporal_manifest": "apf-temporal-manifest-v1", "preprocessing_binding": "apf-research-fit-receipt-v1",
        **{name: "apf-research-protocol-v1" for name in ("input_references", "selected_features",
            "effective_training_configuration", "environment")}}
    def schemas():
        if "native_booster" in artifacts and artifacts["estimator_state"].schema_identity == "apf-fitted-state-artifact-v1":
            if artifacts["native_booster"].schema_identity != "xgboost-native-json-v1":
                raise NotImplementedError()
        _parse(bundle.manifest.canonical_json().encode(), limits)
        for item in bundle.artifacts:
            if item.format_version != "1":
                raise NotImplementedError()
            if item.format == "json":
                _parse(item.data, limits)
        for role, schema in schema_names.items():
            accepted = {schema}
            if role == "preprocessing_binding":
                accepted.add("apf-fitted-state-artifact-v1")
            if artifacts[role].format_version != "1" or artifacts[role].schema_identity not in accepted:
                raise NotImplementedError()
    run("artifact_schemas", schemas)
    def payload(role):
        item = artifacts[role]
        if item.format_version != "1":
            raise NotImplementedError()
        _parse(item.data, limits)
        return item.data.decode("utf-8")
    try:
        from .protocol import ResearchProtocolV1, validate_research_protocol, _encode, _decode, _json, _key
        from .receipt import ResearchFitReceiptV1
    except ImportError as error:
        put("scientific_consistency", "operational_error", type(error).__name__)
        return report()
    protocol = receipt = temporal = None
    def protocol_check():
        nonlocal protocol, origin
        protocol = ResearchProtocolV1.from_json(payload("protocol"))
        validate_research_protocol(protocol).require_valid()
        require(protocol.protocol_id == meta.protocol_id, "protocol_id")
        origin = protocol.provenance.evidence_origin
    def receipt_check():
        nonlocal receipt
        text = payload("receipt")
        raw = _parse(text.encode(), limits)
        for name in ("preprocessing_binding_json", "temporal_manifest_json"):
            embedded = _decode(raw["fields"][name])
            _parse(embedded.encode(), limits)
        receipt = ResearchFitReceiptV1.from_json(text)
        require(receipt.receipt_id == meta.receipt_id, "receipt_id")
    run("protocol_identity", protocol_check)
    run("receipt_identity", receipt_check)
    # A failed identity may decode structurally; never use it as a trusted parent.
    if checks["protocol_identity"].status != "verified" or checks["receipt_identity"].status != "verified":
        statuses = {checks[key].status for key in ("protocol_identity", "receipt_identity")}
        status = next((item for item in ("failed", "operational_error", "unsupported") if item in statuses), "incomplete")
        put("scientific_consistency", status, "invalid_or_unsupported_scientific_parent")
        return report()
    def typed(role, expected):
        text = payload(role)
        value = _decode(_parse(text.encode(), limits))
        require(_json(_encode(value)) == text and _encode(value) == _encode(expected), role)
    def references():
        for name in ("protocol_id", "training_input_id", "fitted_model_id", "temporal_manifest_id",
                     "prediction_capture_fingerprint", "asset_id", "attempt_id", "experiment_id"):
            require(getattr(receipt, name) == getattr(meta, name), name)
        for left, right in ((receipt.protocol_id, protocol.protocol_id), (receipt.asset_id, protocol.asset_id),
                            (receipt.partitions, protocol.partitions), (receipt.source_artifacts, protocol.artifacts),
                            (receipt.target_configuration, protocol.target),
                            (receipt.effective_parameters, protocol.learning.effective_parameters),
                            (receipt.code_dependency_provenance, protocol.environment)):
            require(_encode(left) == _encode(right), "protocol_receipt")
        require(receipt.implementation_reference == protocol.transformations.implementation_reference, "implementation_reference")
        require(_encode(receipt.feature_selection_settings) == _encode({"correlation": protocol.transformations.correlation,
            "importance": protocol.transformations.feature_selection}), "feature_selection_settings")
        descriptor = _decode(_parse(receipt.preprocessing_binding_json.encode(), limits))
        require(_encode(_decode(descriptor["recipe"])) == _encode(protocol.transformations), "preprocessing_recipe")
        require(receipt.boosting_metadata["uses_early_stopping"] == protocol.learning.early_stopping["enabled"], "early_stopping_usage")
        selected = receipt.selected_features
        require(set(selected).issubset(protocol.transformations.feature_order), "unknown_selected_feature")
        typed("selected_features", selected)
        typed("effective_training_configuration", protocol.learning.effective_parameters)
        typed("environment", protocol.environment)
        typed("input_references", protocol.artifacts)
        require(tuple((item.artifact_id, item.protocol_relative_path, item.sha256, item.byte_length, item.schema_identity)
                      for item in meta.input_closure.sources) ==
                tuple((item.artifact_id, item.relative_path, item.sha256, item.byte_length, item.schema_sha256) for item in protocol.artifacts), "source_references")
        observations = {_key(row.source_id): row for row in protocol.universe.observations}
        maturity = {_key(row): end for row, end in protocol.target.maturity}
        expected = tuple((row.row_id, row.source_id, observations[_key(row.source_id)].observation_at,
                          maturity[_key(row.row_id)]) for row in protocol.universe.engineered_rows)
        require(_encode(receipt.row_identity_bindings) == _encode(expected), "row_maturity_bindings")
    run("reference_consistency", references)
    def states():
        for name in ("selector", "scaler", "estimator"):
            require(hashlib.sha256(artifacts[name + "_state"].data).hexdigest() == receipt.state_digests[name], name)
        require(artifacts["preprocessing_binding"].data == receipt.preprocessing_binding_json.encode(), "preprocessing")
    state_roles = {"selector": "selector_state", "scaler": "scaler_state", "estimator": "estimator_state", "preprocessing": "preprocessing_binding"}
    wrapped = tuple(artifacts[role].schema_identity == "apf-fitted-state-artifact-v1" for role in state_roles.values())
    if any(wrapped):
        if not all(wrapped):
            put("state_envelope_hashes", "failed", "mixed_state_envelope_schemas")
            put("state_envelope_schema", "failed", "mixed_state_envelope_schemas")
        else:
            from .state_artifacts import FittedStateArtifact, verify_fitted_state_artifacts
            images = {name: FittedStateArtifact(name, artifacts[role].data, artifacts[role].entry().sha256,
                artifacts[role].format, artifacts[role].format_version) for name, role in state_roles.items()}
            if "native_booster" in artifacts:
                item = artifacts["native_booster"]
                images["native_booster"] = FittedStateArtifact("native_booster", item.data, item.entry().sha256, item.format, item.format_version)
            result = verify_fitted_state_artifacts(images, receipt, limits=limits)
            codes = tuple(item.code for item in result.diagnostics)
            put("state_envelope_hashes", result.status, *codes)
            put("state_envelope_schema", result.status, *codes)
    else:
        run("state_envelope_hashes", states)
        put("state_envelope_schema", "unsupported", "no_public_safe_selector_scaler_estimator_envelope_codec_for_supplied_schema")
    prediction = artifacts["prediction_capture"]
    if prediction.schema_identity == "apf-keyed-prediction-artifact-v1" and prediction.format_version == "1":
        from .prediction_artifacts import PredictionCaptureArtifact, PredictionArtifactLimits, verify_prediction_capture_artifact
        image = PredictionCaptureArtifact(prediction.data, prediction.entry().sha256, meta.prediction_capture_fingerprint)
        result = verify_prediction_capture_artifact(image, receipt, limits=PredictionArtifactLimits(verification=limits))
        put("prediction_capture", result.status, *(item.code for item in result.diagnostics))
        put("prediction_decision_chronology", result.decision_chronology if result.status == "verified" else "not_assessed",
            "capture_does_not_retain_feature_availability_or_decision_clocks")
    else:
        put("prediction_capture", "unsupported", "no_roundtrip_keyed_capture_codec_index_dtype_timezone_name_frequency_probabilities")
        put("prediction_decision_chronology", "not_assessed", "capture_decision_timestamps_unavailable")
    def timing():
        nonlocal temporal
        data = artifacts["temporal_manifest"].data
        require(data == receipt.temporal_manifest_json.encode(), "temporal_receipt_bytes")
        temporal = _temporal(data, limits)
        require(temporal.split_id == receipt.temporal_manifest_id, "temporal_id")
        require((temporal.asset_id, temporal.model_id, temporal.experiment_id, temporal.feature_names) ==
                (receipt.asset_id, receipt.fitted_model_id, receipt.experiment_id, receipt.selected_features), "temporal_context")
        from src.evaluation.execution.prediction_export import _utc_instant
        clocks = {_key(row): clock for row, _, clock, _ in receipt.row_identity_bindings}
        for name, group in (("fitting_ids", "fitting"), ("inner_purged_ids", "inner_purge"),
                            ("validation_ids", "validation"), ("outer_purged_ids", "outer_purge"), ("held_out_ids", "held_out")):
            require(tuple(_utc_instant(x) for x in getattr(temporal, name)) ==
                    tuple(_utc_instant(clocks[_key(x)]) for x in getattr(protocol.partitions, group)), "temporal_membership")
        require((temporal.target_type, temporal.target_horizon, temporal.target_threshold) ==
                (protocol.target.task_type, protocol.target.horizon, protocol.target.threshold), "temporal_target")
        observations = {_key(row.source_id): row.observation_at for row in protocol.universe.observations}
        require(len(temporal.source_rows) == len(receipt.row_identity_bindings), "temporal_row_count")
        for row, (_, _, clock, end) in zip(temporal.source_rows, receipt.row_identity_bindings):
            require(_utc_instant(row.observation_at) == _utc_instant(clock)
                    and row.target_end_at is not None
                    and _utc_instant(row.target_end_at) == _utc_instant(observations[_key(end)]), "temporal_maturity")
        require(temporal.uses_early_stopping == receipt.boosting_metadata["uses_early_stopping"], "temporal_early_stopping")
        require(_utc_instant(receipt.actual_prediction_generated_at) >=
                max(_utc_instant(row.features_available_at) for row in temporal.source_rows), "prediction_generation_chronology")
        require(temporal.actual_fit_completed_at is not None and
                _utc_instant(temporal.actual_fit_completed_at) == _utc_instant(receipt.actual_fit_completed_at), "fit_completion")
    run("temporal_provenance", timing)
    closure = meta.input_closure
    dependencies = tuple(InputDependency(item.artifact_id, item.protocol_relative_path, item.sha256, item.byte_length,
        json.dumps(asdict(item.external), sort_keys=True) if item.external else None) for item in closure.sources)
    if closure.mode != "bundled":
        put("input_closure", "incomplete", "external_declarations_not_independently_verified" if closure.mode == "verified_external" else "source_bytes_unavailable")
        put("training_inputs", "incomplete", "source_bytes_required")
    else:
        def source_check():
            from .protocol import validate_research_artifacts
            from .input_adapter import prepare_research_inputs
            with tempfile.TemporaryDirectory(prefix="apf-research-verify-") as directory:
                root = Path(directory)
                for ref in protocol.artifacts:
                    destination = root / ref.relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(bundle.artifact_bytes("inputs-root/" + ref.relative_path))
                validate_research_artifacts(protocol, root).require_valid()
                put("input_closure", "verified")
                try:
                    prepared = prepare_research_inputs(protocol, root)
                except Exception as error:
                    codes = tuple(item.code for item in getattr(error, "diagnostics", ()))
                    if any("unsupported" in code or "environment" in code or "implementation" in code for code in codes):
                        put("training_inputs", "unsupported", *codes)
                        return
                    raise
                if prepared.training_input_id != receipt.training_input_id:
                    put("training_inputs", "failed", "prepared_training_identity_mismatch")
                else:
                    put("training_inputs", "verified")
        run("input_closure", source_check)
        if checks["input_closure"].status == "failed":
            put("training_inputs", "failed", "source_or_prepared_training_validation_failed")
    if checks["input_closure"].status == "verified":
        dependencies = tuple(replace(item, independently_checked=True) for item in dependencies)
    required = ("artifact_schemas", "reference_consistency", "input_closure", "training_inputs", "state_envelope_hashes", "state_envelope_schema", "prediction_capture", "temporal_provenance")
    status = "verified" if all(checks[key].status == "verified" for key in required) else "unsupported"
    if any(checks[key].status == "failed" for key in required):
        status = "failed"
    elif any(checks[key].status == "operational_error" for key in required):
        status = "operational_error"
    elif any(checks[key].status == "incomplete" for key in required):
        status = "incomplete"
    if status == "verified" and checks["prediction_decision_chronology"].status != "verified":
        status = "incomplete"
    put("scientific_consistency", status, "full_temporal_verification_requires_explicit_prediction_decision_clocks")
    return report()

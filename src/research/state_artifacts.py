"""Exact receipt-bound state export and data-only envelope verification.

No model reconstruction is provided. Receipt identity is SHA-256 of the exact
legacy canonical state bytes; an outer versioned artifact has a distinct file
hash, held outside its own bytes to avoid a self-referential digest.
"""
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from .bundle_verification import VerificationLimits, _parse
from .bundle import SENSITIVE_FIELDS
from .protocol import ProtocolDiagnostic, ResearchProtocolError, _encode, _decode, _json
from .receipt import ResearchFitReceiptV1, inspect_research_fit, _bytes

SCHEMA_VERSION = "apf-fitted-state-artifact-v1"
ROLES = ("selector", "scaler", "estimator", "preprocessing")


class StateArtifactError(ResearchProtocolError):
    """Structured codes only; no fitted state or sensitive payload in errors."""


def _require(value, code, role="states"):
    if not value:
        raise StateArtifactError((ProtocolDiagnostic(code, role, "Fitted-state artifact validation failed."),))


def _integer_text(value):
    _require(type(value) is str and value.isascii() and value.isdecimal(), "invalid_native_integer")
    return int(value)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FittedStateArtifact:
    role: str
    data: bytes = field(repr=False)
    sha256: str
    format: str = "json"
    format_version: str = "1"


@dataclass(frozen=True)
class FittedStateArtifacts:
    artifacts: tuple[FittedStateArtifact, ...]

    def by_role(self):
        return MappingProxyType({item.role: item for item in self.artifacts})


@dataclass(frozen=True)
class StateVerificationReport:
    status: str
    verified_roles: tuple[str, ...] = ()
    receipt_id: str | None = None
    diagnostics: tuple[ProtocolDiagnostic, ...] = ()
    runtime_loading: str = "not_assessed"
    functional_reproduction: str = "not_assessed"

    def require_valid(self):
        if self.status != "verified":
            raise StateArtifactError(self.diagnostics)
        return self


def _read_state(text, limits):
    """Decode only primitive receipt-state tags, never research/model classes."""
    raw = _parse(text.encode("utf-8"), limits)
    def allow(value):
        _require(type(value) is list and bool(value), "invalid_state_tag")
        tag = value[0]
        _require(tag in {"none", "bool", "int", "float", "str", "numpy", "tuple", "map"}, "unsupported_state_tag")
        if tag == "map":
            _require(len(value) == 2 and type(value[1]) is list, "invalid_state_map")
            for pair in value[1]:
                _require(type(pair) is list and len(pair) == 2, "invalid_state_map")
                allow(pair[0])
                allow(pair[1])
        elif tag == "tuple":
            _require(len(value) == 2 and type(value[1]) is list, "invalid_state_tuple")
            for item in value[1]:
                allow(item)
        elif tag == "numpy":
            import numpy as np
            _require(len(value) == 3 and type(value[1]) is str, "invalid_numpy_scalar")
            dtype = np.dtype(value[1])
            _require(dtype.kind in "biuf" and not dtype.hasobject and dtype.itemsize <= 8, "unsupported_numpy_scalar")
            allow(value[2])
        else:
            _require(len(value) == (1 if tag == "none" else 2), "invalid_state_scalar")
            if tag in {"float", "int", "str"}:
                _require(type(value[1]) is str, "invalid_state_scalar")
            elif tag == "bool":
                _require(type(value[1]) is bool, "invalid_state_scalar")
    allow(raw)
    decoded = _decode(raw)
    def inspect_keys(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str):
                    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
                    _require(normalized not in SENSITIVE_FIELDS and not normalized.endswith(("_password", "_secret", "_token", "_api_key")), "sensitive_state_field")
                inspect_keys(item)
        elif isinstance(value, tuple):
            for item in value:
                inspect_keys(item)
    inspect_keys(decoded)
    _require(_bytes(decoded).decode() == text, "noncanonical_state_payload")
    return decoded


def _array(value, *, dimensions=None):
    """Validate fixed primitive arrays without allocating from declared shapes."""
    import numpy as np
    _require(isinstance(value, dict) and {"dtype", "shape"}.issubset(value), "invalid_array_metadata")
    shape = value["shape"]
    _require(type(shape) is tuple and all(type(x) is int and x >= 0 for x in shape), "invalid_array_shape")
    _require(dimensions is None or len(shape) == dimensions, "invalid_array_dimension")
    _require(type(value["dtype"]) is str, "unsupported_array_dtype")
    dtype = np.dtype(value["dtype"])
    _require(dtype.kind in "biufUO" and 0 < dtype.itemsize <= 65536, "unsupported_array_dtype")
    size = 1
    for dimension in shape:
        size *= dimension
    if "strings" in value:
        _require(set(value) == {"dtype", "shape", "strings"} and dtype.kind in "UO"
                 and len(value["strings"]) == size and all(type(x) is str and (dtype.kind == "O" or len(x) <= dtype.itemsize // 4) for x in value["strings"]), "invalid_string_array")
        return tuple(value["strings"]), shape
    _require(set(value) == {"dtype", "shape", "data_hex"} and dtype.kind in "biuf"
             and type(value["data_hex"]) is str and len(value["data_hex"]) == size * dtype.itemsize * 2, "invalid_array_length")
    data = bytes.fromhex(value["data_hex"])
    _require(data.hex() == value["data_hex"], "noncanonical_array_hex")
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    _require(np.isfinite(array).all(), "nonfinite_array")
    return array, shape


def _validate_states(states, receipt, context):
    import numpy as np
    selector, scaler, estimator, descriptor = (states[key] for key in ROLES)
    _require(set(selector) == {"class", "parameters", "features", "classes", "n_outputs", "trees"}
             and selector["class"] == "sklearn.RandomForestClassifier", "unsupported_selector_schema")
    candidates = selector["features"]
    _require(type(candidates) is tuple and bool(candidates) and all(type(x) is str for x in candidates)
             and len(set(candidates)) == len(candidates) and set(receipt.selected_features).issubset(candidates), "selector_features")
    labels, shape = _array(selector["classes"], dimensions=1)
    _require(shape == (3,) and tuple(labels) == (0, 1, 2) and selector["n_outputs"] == 1, "selector_classes")
    trees = selector["trees"]
    _require(type(trees) is tuple and bool(trees) and len(trees) == selector["parameters"]["n_estimators"], "selector_tree_count")
    for tree in trees:
        _require(set(tree) == {"parameters", "classes", "state"}, "unsupported_tree_schema")
        classes, tree_shape = _array(tree["classes"], dimensions=1)
        _require(tree_shape == (3,) and tuple(classes) == (0, 1, 2), "tree_classes")
        state = tree["state"]
        _require(set(state) == {"max_depth", "node_count", "nodes", "values"}, "unsupported_tree_state_schema")
        count = state["node_count"]
        _require(type(count) is int and count > 0 and type(state["max_depth"]) is int and state["max_depth"] >= 0, "tree_dimensions")
        nodes = state["nodes"]
        _require(set(nodes) == {"dtype", "shape", "fields"} and nodes["shape"] == (count,), "tree_node_shape")
        declared = nodes["dtype"]
        _require(type(declared) is tuple and all(type(item) is tuple and len(item) == 2
                 and all(type(x) is str for x in item) for item in declared), "tree_node_dtype")
        required = {"left_child", "right_child", "feature", "threshold", "impurity", "n_node_samples", "weighted_n_node_samples"}
        names = tuple(item[0] for item in declared if item[0])
        _require(len(set(names)) == len(names) and set(names) in (required, required | {"missing_go_to_left"})
                 and set(nodes["fields"]) == set(names), "unsupported_tree_fields")
        columns = {}
        for name, dtype in declared:
            if not name:
                padding = np.dtype(dtype)
                _require(padding.kind == "V" and 0 < padding.itemsize <= 64, "unsupported_tree_padding")
                continue
            array, node_shape = _array(nodes["fields"][name], dimensions=1)
            _require(node_shape == (count,) and array.dtype == np.dtype(dtype), "tree_field_dimension")
            columns[name] = array
        for side in ("left_child", "right_child"):
            _require(columns[side].dtype.kind in "iu" and ((columns[side] == -1) | ((columns[side] >= 0) & (columns[side] < count))).all(), "tree_child_index")
        feature = columns["feature"]
        _require(feature.dtype.kind in "iu" and ((feature == -2) | ((feature >= 0) & (feature < len(candidates)))).all(), "tree_feature_index")
        _, value_shape = _array(state["values"], dimensions=3)
        _require(value_shape == (count, 1, 3), "tree_value_shape")
    _require(set(scaler) == {"class", "parameters", "learned"} and scaler["class"] == "sklearn.RobustScaler", "unsupported_scaler_schema")
    learned = scaler["learned"]
    _require(set(learned) == {"feature_names_in_", "n_features_in_", "center_", "scale_"}, "unsupported_scaler_learned_schema")
    features, shape = _array(learned["feature_names_in_"], dimensions=1)
    n = len(receipt.selected_features)
    _require(features == receipt.selected_features and shape == (n,) and learned["n_features_in_"] == n, "scaler_feature_order")
    for name in ("center_", "scale_"):
        values, shape = _array(learned[name], dimensions=1)
        _require(shape == (n,) and (name != "scale_" or (values > 0).all()), "scaler_dimensions")
    _require(set(estimator) == {"format", "parameters", "booster", "configuration"}
             and estimator["format"] == "xgboost-json-v1", "unsupported_estimator_schema")
    version = estimator["booster"]["version"]
    _require(type(version) is tuple and len(version) == 3 and all(type(x) is int and x >= 0 for x in version)
             and version[0] == 3, "unsupported_native_model_version")
    declared_version = context["dependencies"].get("xgboost")
    _require(type(declared_version) is str and declared_version.split(".")[:3] == [str(x) for x in version], "native_dependency_version_mismatch")
    config = estimator["configuration"]["learner"]
    booster = estimator["booster"]["learner"]
    _require(tuple(booster["feature_names"]) == receipt.selected_features
             and _integer_text(config["learner_model_param"]["num_feature"]) == n
             and _integer_text(config["learner_model_param"]["num_class"]) == 3
             and _integer_text(booster["learner_model_param"]["num_feature"]) == n
             and _integer_text(booster["learner_model_param"]["num_class"]) == 3, "estimator_dimensions_or_classes")
    _require(estimator["parameters"].get("num_parallel_tree") in {None, 1}
             and estimator["parameters"].get("multi_strategy") in {None, "one_output_per_tree"}, "unsupported_parallel_or_vector_trees")
    _require(config["objective"]["name"] == "multi:softprob" and estimator["parameters"]["objective"] == "multi:softprob", "unsupported_estimator_objective")
    for name, value in receipt.effective_parameters.items():
        _require(name in estimator["parameters"] and _encode(estimator["parameters"][name]) == _encode(value), "estimator_parameters")
    attributes = booster["attributes"]
    metadata = receipt.boosting_metadata
    _require((_integer_text(attributes["best_iteration"]) if "best_iteration" in attributes else None) == metadata["best_iteration"]
             and attributes.get("best_score") == metadata["best_score"], "boosting_metadata")
    model = booster["gradient_booster"]
    _require(model["name"] == "gbtree", "unsupported_booster_type")
    tree_model = model["model"]
    _require(_integer_text(tree_model["gbtree_model_param"]["num_trees"]) == len(tree_model["trees"])
             and len(tree_model["trees"]) == metadata["rounds"] * 3, "boosting_rounds")
    for tree in tree_model["trees"]:
        parameters = tree["tree_param"]
        count = _integer_text(parameters["num_nodes"])
        _require(count > 0 and _integer_text(parameters["num_feature"]) == n and _integer_text(parameters["size_leaf_vector"]) in {0, 1}, "native_tree_dimensions")
        for name in ("base_weights", "left_children", "right_children", "parents", "split_conditions", "split_indices", "default_left", "loss_changes", "sum_hessian", "split_type"):
            values = tree[name]
            _require(type(values) is tuple and len(values) == count, "native_tree_array_length")
        for name in ("left_children", "right_children"):
            _require(all(type(x) is int and (x == -1 or 0 <= x < count) for x in tree[name]), "native_tree_child_index")
        _require(all(type(x) is int and 0 <= x < n for x in tree["split_indices"]), "native_tree_feature_index")
    _require(len(booster["feature_types"]) in {0, n}, "native_feature_types")
    _require(_sha(receipt.preprocessing_binding_json.encode()) == receipt.state_digests["preprocessing"]
             and _bytes(descriptor).decode() == receipt.preprocessing_binding_json, "preprocessing_binding")
    _require(context["selected_features"] == list(receipt.selected_features) and context["estimator_classes"] == [0, 1, 2], "class_or_feature_order")
    _require(context["selector_features"] == list(candidates) and context["input_dimension"] == n, "input_dimension")
    _require(context["boosting_metadata"] == dict(receipt.boosting_metadata), "boosting_metadata")


def _context(receipt, selector_features, native_sha):
    return {"receipt_id": receipt.receipt_id, "protocol_id": receipt.protocol_id, "attempt_id": receipt.attempt_id,
        "asset_id": receipt.asset_id, "experiment_id": receipt.experiment_id, "fitted_model_id": receipt.fitted_model_id,
        "training_input_id": receipt.training_input_id, "selected_features": list(receipt.selected_features),
        "selector_features": list(selector_features), "estimator_classes": [0, 1, 2], "input_dimension": len(receipt.selected_features),
        "boosting_metadata": dict(receipt.boosting_metadata),
        "dependencies": dict(receipt.code_dependency_provenance.package_versions), "python_version": receipt.code_dependency_provenance.python_version,
        "compatibility": "exact-state-envelope-only; runtime-loading-not-assessed", "native_booster_sha256": native_sha}


def verify_fitted_state_artifacts(artifact_bytes, receipt, *, limits=VerificationLimits()):
    """Verify supplied immutable artifact images without importing model runtimes.

    Mapping values must be FittedStateArtifact records with an independently
    supplied expected file digest (normally from the research bundle manifest).
    Hash consistency does not authenticate their publisher or witness fitting.
    """
    try:
        _require(isinstance(receipt, ResearchFitReceiptV1), "invalid_receipt")
        receipt = ResearchFitReceiptV1.from_json(receipt.canonical_json())
        _require(isinstance(artifact_bytes, Mapping) and set(artifact_bytes) in (set(ROLES), set(ROLES) | {"native_booster"}), "missing_or_extra_state_artifacts")
        _require(len(artifact_bytes) <= limits.archive.max_artifacts, "artifact_count_limit")
        states, contexts = {}, []
        total = 0
        for role, artifact in artifact_bytes.items():
            _require(type(artifact) is FittedStateArtifact and artifact.role == role and type(artifact.data) is bytes
                     and artifact.format == "json" and artifact.format_version == "1", "unsupported_artifact_format", role)
            total += len(artifact.data)
            _require(len(artifact.data) <= limits.archive.max_artifact_bytes and total <= limits.archive.max_total_bytes, "artifact_size_limit", role)
            _require(_sha(artifact.data) == artifact.sha256, "artifact_digest_mismatch", role)
            raw = _parse(artifact.data, limits)
            if role == "native_booster":
                continue
            _require(type(raw) is dict and set(raw) == {"schema_version", "role", "serialization", "receipt_state_identity", "state_envelope_sha256", "canonical_state_payload", "context"}, "invalid_envelope_fields", role)
            _require(raw["schema_version"] == SCHEMA_VERSION and raw["serialization"] == {"format": "apf-receipt-canonical-json", "version": 1}, "unsupported_state_schema", role)
            _require(_json(raw).encode() == artifact.data and raw["role"] == role, "noncanonical_artifact", role)
            payload = raw["canonical_state_payload"]
            _require(type(payload) is str and _sha(payload.encode()) == raw["receipt_state_identity"] == raw["state_envelope_sha256"] == receipt.state_digests[role], "receipt_state_mismatch", role)
            states[role] = _read_state(payload, limits)
            contexts.append(raw["context"])
        _require(all(_json(value) == _json(contexts[0]) for value in contexts), "conflicting_artifact_context")
        context = contexts[0]
        _require(_json(context) == _json(_context(receipt, states["selector"]["features"], context["native_booster_sha256"])), "receipt_context_mismatch")
        _validate_states(states, receipt, context)
        native = artifact_bytes.get("native_booster")
        if native is None:
            _require(context["native_booster_sha256"] is None, "missing_native_booster")
        else:
            _require(native.sha256 == context["native_booster_sha256"], "native_digest_mismatch")
            _require(_encode(_parse(native.data, limits)) == _encode(states["estimator"]["booster"]), "native_state_mismatch")
        return StateVerificationReport("verified", tuple(sorted(artifact_bytes)), receipt.receipt_id)
    except StateArtifactError as error:
        codes = {item.code for item in error.diagnostics}
        status = "unsupported" if any(code.startswith("unsupported_") for code in codes) else "failed"
        return StateVerificationReport(status, diagnostics=error.diagnostics)
    except Exception:
        return StateVerificationReport("failed", diagnostics=(ProtocolDiagnostic("malformed_state_artifact", "states", "State structure is invalid."),))


def export_fitted_state_artifacts(protocol, prepared, receipt, *, attempt_id, selector, scaler, estimator,
                                  selected_features, limits=VerificationLimits()):
    """Export owned immutable images from the receipt's shared inspector.

    Returns bytes and digests; filesystem publication remains the bundle
    publisher's responsibility. The attempt association is caller-declared,
    checked against the receipt, and is not an independent training witness.
    """
    try:
        _require(isinstance(receipt, ResearchFitReceiptV1), "invalid_receipt")
        receipt = ResearchFitReceiptV1.from_json(receipt.canonical_json())
        _require(attempt_id == receipt.attempt_id and protocol.protocol_id == receipt.protocol_id
                 and prepared.training_input_id == receipt.training_input_id and protocol.asset_id == receipt.asset_id, "fitting_attempt_mismatch")
        _require(_encode(receipt.partitions) == _encode(protocol.partitions)
                 and _encode(receipt.source_artifacts) == _encode(protocol.artifacts)
                 and _encode(receipt.target_configuration) == _encode(protocol.target)
                 and _encode(receipt.code_dependency_provenance) == _encode(protocol.environment), "receipt_protocol_mismatch")
        bindings = tuple((row.row_id, row.source_id, row.observation_at, row.target_end_source_id) for row in prepared.row_bindings)
        _require(_encode(bindings) == _encode(receipt.row_identity_bindings), "receipt_membership_mismatch")
        # Unsupported objects must not receive any native serialization calls.
        from xgboost import XGBClassifier
        _require(type(estimator) is XGBClassifier, "unsupported_estimator_class")
        # Snapshot before serialization to reject stateful native serialization.
        booster = estimator.get_booster()
        before = (booster.save_config(), booster.attributes(), tuple(booster.feature_names or ()), tuple(booster.feature_types or ()))
        inspected = inspect_research_fit(protocol, prepared, experiment_id=receipt.experiment_id,
            selector=selector, scaler=scaler, estimator=estimator, selected_features=selected_features)
        after = (booster.save_config(), booster.attributes(), tuple(booster.feature_names or ()), tuple(booster.feature_types or ()))
        _require(before == after and bytes(booster.save_raw(raw_format="json")) == inspected.native_booster_json
                 and booster.save_config() == before[0], "native_serialization_changed_state")
        _require(inspected.binding.model_id == receipt.fitted_model_id
                 and inspected.binding.feature_names == receipt.selected_features
                 and inspected.binding.preprocessing_state.decode() == receipt.preprocessing_binding_json, "fitted_binding_mismatch")
        _require((inspected.boosting_rounds, inspected.best_iteration, inspected.best_score) ==
                 (receipt.boosting_metadata["rounds"], receipt.boosting_metadata["best_iteration"], receipt.boosting_metadata["best_score"]), "boosting_metadata_mismatch")
        images = {"selector": inspected.selector_state, "scaler": inspected.scaler_state,
                  "estimator": inspected.binding.estimator_state, "preprocessing": inspected.binding.preprocessing_state}
        _require(all(_sha(value) == receipt.state_digests[role] for role, value in images.items()), "receipt_state_mismatch")
        context = _context(receipt, tuple(selector.feature_names_in_), _sha(inspected.native_booster_json))
        artifacts = []
        for role, data in images.items():
            envelope = {"schema_version": SCHEMA_VERSION, "role": role,
                "serialization": {"format": "apf-receipt-canonical-json", "version": 1},
                "receipt_state_identity": receipt.state_digests[role], "state_envelope_sha256": _sha(data),
                "canonical_state_payload": data.decode("utf-8"), "context": context}
            serialized = _json(envelope).encode()
            artifacts.append(FittedStateArtifact(role, serialized, _sha(serialized)))
        native = inspected.native_booster_json
        artifacts.append(FittedStateArtifact("native_booster", native, _sha(native)))
        result = FittedStateArtifacts(tuple(artifacts))
        verify_fitted_state_artifacts(result.by_role(), receipt, limits=limits).require_valid()
        return result
    except StateArtifactError:
        raise
    except ResearchProtocolError as error:
        raise StateArtifactError(error.diagnostics) from None
    except Exception:
        raise StateArtifactError((ProtocolDiagnostic("unsupported_fitted_state", "export", "State export failed closed."),)) from None

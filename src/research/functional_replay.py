"""Explicit offline inference from receipt-bound research bundles.

This is functional prediction replay, not a training witness or a financial
backtest. Native JSON is still untrusted input to XGBoost's native parser.
Allowlisting and bounds reduce exposure; they are not a native-code sandbox or
publisher authentication. No replay result is written into the publication.
"""
from dataclasses import dataclass
import importlib.metadata
import math
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd

from .bundle import open_research_bundle
from .bundle_verification import VerificationLimits, VerificationCheck, _parse, verify_research_bundle
from .protocol import ResearchProtocolV1, _encode
from .receipt import ResearchFitReceiptV1
from .input_adapter import prepare_research_inputs
from .state_artifacts import _read_state, _array
from .prediction_artifacts import deserialize_prediction_capture, PredictionCaptureArtifact


@dataclass(frozen=True)
class ReplayLimits:
    verification: VerificationLimits = VerificationLimits()
    max_native_bytes: int = 8 * 1024 * 1024
    max_rows: int = 100000
    max_features: int = 256
    max_trees: int = 10000
    max_tree_nodes: int = 200000
    max_tree_depth: int = 64

    def __post_init__(self):
        if type(self.verification) is not VerificationLimits or any(type(x) is not int or x <= 0 for x in (
                self.max_native_bytes, self.max_rows, self.max_features, self.max_trees, self.max_tree_nodes, self.max_tree_depth)):
            raise ValueError("Positive replay limits are required.")


@dataclass(frozen=True)
class ReplayReport:
    status: str
    checks: tuple[tuple[str, VerificationCheck], ...]
    scope: str = "offline_functional_prediction_equivalence"
    protocol_id: str | None = None
    receipt_id: str | None = None
    fitted_model_id: str | None = None
    reference_capture_fingerprint: str | None = None
    iteration_range: tuple[int, int] | None = None
    retained_boosting_rounds: int | None = None
    compared_rows: int = 0
    maximum_absolute_error: float | None = None
    probability_mismatched_rows: int | None = None
    class_mismatched_rows: int | None = None
    bit_exact_probabilities: bool | None = None
    probability_atol: float = 0.0
    historical_pit_status: str = "unknown"
    execution_eligibility_status: str = "unassessed"
    independently_witnessed_training: bool = False

    def check(self, name):
        return dict(self.checks)[name]


class _Rejected(ValueError):
    def __init__(self, code, status="failed"):
        self.code, self.status = code, status
        super().__init__(code)


def _require(value, code, status="failed"):
    if not value:
        raise _Rejected(code, status)


def _runtime_versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
        for name in ("xgboost", "numpy", "pandas", "pyarrow", "scikit-learn")}}


def _check_runtime(receipt):
    current = _runtime_versions()
    declared = receipt.code_dependency_provenance
    _require(current["xgboost"] == "3.2.0" and declared.package_versions.get("xgboost") == "3.2.0",
             "unsupported_xgboost_runtime", "unsupported")
    _require(current["python"] == declared.python_version and all(current[name] == declared.package_versions.get(name)
             for name in ("numpy", "pandas", "pyarrow", "scikit-learn")), "source_runtime_version_mismatch", "unsupported")


def _native_preflight(data, receipt, limits):
    """Bound and validate the audited numerical, scalar-leaf XGBoost 3.2 graph."""
    _require(type(data) is bytes and 0 < len(data) <= limits.max_native_bytes, "native_json_size_limit")
    raw = _parse(data, limits.verification)
    _require(set(raw) == {"learner", "version"} and raw["version"] == [3, 2, 0]
             and all(type(x) is int for x in raw["version"]), "unsupported_native_json_version", "unsupported")
    learner = raw["learner"]
    _require(set(learner) == {"attributes", "feature_names", "feature_types", "gradient_booster", "learner_model_param", "objective"}, "unsupported_native_learner_schema", "unsupported")
    names = receipt.selected_features
    _require(0 < len(names) <= limits.max_features and learner["feature_names"] == list(names)
             and len(learner["feature_types"]) == len(names) and all(x in {"float", "int"} for x in learner["feature_types"]), "unsupported_native_features", "unsupported")
    _require(set(learner["objective"]) == {"name", "softmax_multiclass_param"}
             and learner["objective"]["name"] == "multi:softprob" and learner["objective"]["softmax_multiclass_param"] == {"num_class": "3"},
             "unsupported_native_objective", "unsupported")
    parameters = learner["learner_model_param"]
    _require(set(parameters) == {"base_score", "boost_from_average", "num_class", "num_feature", "num_target"}
             and parameters["num_class"] == "3" and parameters["num_feature"] == str(len(names))
             and parameters["num_target"] == "1" and parameters["boost_from_average"] in {"0", "1"}, "unsupported_native_dimensions", "unsupported")
    base = _parse(parameters["base_score"].encode(), limits.verification)
    base = base if type(base) is list else [base]
    _require(len(base) in {1, 3} and all(type(x) in {int, float} and math.isfinite(x) for x in base), "invalid_native_base_score")
    _require(set(learner["attributes"]).issubset({"best_iteration", "best_score"}), "unsupported_native_attributes", "unsupported")
    booster = learner["gradient_booster"]
    _require(set(booster) == {"name", "model"} and booster["name"] == "gbtree", "unsupported_booster_type", "unsupported")
    model = booster["model"]
    _require(set(model) == {"cats", "gbtree_model_param", "iteration_indptr", "tree_info", "trees"}
             and model["cats"] == {"enc": [], "feature_segments": [], "sorted_idx": []}, "unsupported_native_tree_schema", "unsupported")
    rounds = receipt.boosting_metadata["rounds"]
    count = rounds * 3
    _require(type(rounds) is int and 0 < count <= limits.max_trees and len(model["trees"]) == count
             and model["gbtree_model_param"] == {"num_parallel_tree": "1", "num_trees": str(count)}
             and model["iteration_indptr"] == list(range(0, count + 1, 3))
             and model["tree_info"] == [i % 3 for i in range(count)]
             and all(type(x) is int for x in model["iteration_indptr"] + model["tree_info"]), "invalid_native_round_structure")
    arrays = {"base_weights", "default_left", "left_children", "right_children", "loss_changes", "parents",
              "split_conditions", "split_indices", "split_type", "sum_hessian"}
    categories = {"categories", "categories_nodes", "categories_segments", "categories_sizes"}
    total = 0
    for tree_id, tree in enumerate(model["trees"]):
        _require(set(tree) == arrays | categories | {"id", "tree_param"} and type(tree["id"]) is int and tree["id"] == tree_id, "unsupported_native_node_schema", "unsupported")
        params = tree["tree_param"]
        _require(set(params) == {"num_deleted", "num_feature", "num_nodes", "size_leaf_vector"}
                 and params["num_deleted"] == "0" and params["num_feature"] == str(len(names))
                 and params["size_leaf_vector"] in {"0", "1"}
                 and type(params["num_nodes"]) is str and params["num_nodes"].isascii() and params["num_nodes"].isdecimal(), "unsupported_native_node_parameters", "unsupported")
        nodes = int(params["num_nodes"])
        total += nodes
        _require(0 < nodes <= limits.max_tree_nodes and total <= limits.max_tree_nodes, "native_tree_node_limit")
        _require(all(type(tree[key]) is list and len(tree[key]) == nodes for key in arrays)
                 and all(tree[key] == [] for key in categories), "unsupported_categorical_or_malformed_tree", "unsupported")
        for key in ("base_weights", "loss_changes", "split_conditions", "sum_hessian"):
            _require(all(type(x) in {int, float} and math.isfinite(x) for x in tree[key]), "nonfinite_native_tree")
        _require(all(type(x) is int and x in {0, 1} for x in tree["default_left"])
                 and all(type(x) is int and x == 0 for x in tree["split_type"])
                 and all(type(x) is int and 0 <= x < len(names) for x in tree["split_indices"]), "invalid_native_split_metadata")
        incoming, depths = [0] * nodes, [0] * nodes
        _require(tree["parents"][0] in {-1, 2147483647, 4294967295}, "invalid_native_root")
        for node, (left, right) in enumerate(zip(tree["left_children"], tree["right_children"])):
            _require(type(left) is int and type(right) is int and ((left == -1 and right == -1)
                     or (node < left < nodes and node < right < nodes and left != right)), "cyclic_or_invalid_native_tree")
            for child in (left, right):
                if child != -1:
                    incoming[child] += 1
                    depths[child] = depths[node] + 1
                    parent = tree["parents"][child]
                    _require(type(parent) is int and 0 <= parent <= 4294967295 and (parent & 2147483647) == node,
                             "native_parent_mismatch")
                    _require(depths[child] <= limits.max_tree_depth, "native_tree_depth_limit")
        _require(incoming[0] == 0 and all(x == 1 for x in incoming[1:]), "unreachable_or_shared_native_nodes")
    return raw


def _restore_scaling(protocol, receipt, image, frame, limits):
    envelope = _parse(image, limits.verification)
    scaler = _read_state(envelope["canonical_state_payload"], limits.verification)
    expected = {key: value for key, value in protocol.transformations.scaler.items() if key not in {"class", "scope", "stage"}}
    _require(_encode(scaler["parameters"]) == _encode(expected) and expected["copy"] is True
             and expected["with_centering"] is True and expected["with_scaling"] is True
             and expected["unit_variance"] is False and tuple(expected["quantile_range"]) == (25., 75.), "unsupported_scaler_configuration", "unsupported")
    learned = scaler["learned"]
    names, _ = _array(learned["feature_names_in_"], dimensions=1)
    center, center_shape = _array(learned["center_"], dimensions=1)
    scale, scale_shape = _array(learned["scale_"], dimensions=1)
    n = len(receipt.selected_features)
    _require(names == receipt.selected_features and learned["n_features_in_"] == n and center_shape == scale_shape == (n,)
             and center.dtype == scale.dtype == np.dtype("float64") and (scale > 0).all(), "unsupported_scaler_state", "unsupported")
    _require(tuple(frame.columns) == names and all(dtype.kind in "fiu" for dtype in frame.dtypes), "invalid_frozen_feature_schema")
    values = frame.to_numpy(dtype=np.float64, copy=True)
    _require(np.isfinite(values).all(), "nonfinite_frozen_features")
    # Exact dense RobustScaler.transform order, on an owned float64 copy.
    values -= center
    values /= scale
    _require(np.isfinite(values).all(), "nonfinite_scaled_features")
    return pd.DataFrame(values, index=frame.index.copy(deep=True), columns=list(names))


def _iteration_range(receipt):
    metadata = receipt.boosting_metadata
    rounds, best = metadata["rounds"], metadata["best_iteration"]
    if metadata["uses_early_stopping"]:
        _require(type(best) is int and 0 <= best < rounds, "invalid_best_iteration")
        return (0, best + 1)
    _require(best is None, "unexpected_best_iteration")
    return (0, rounds)


def _load_and_predict(native, scaled, receipt, iterations):
    """Only the audited fixed native class is constructed, after all checks."""
    import xgboost as xgb
    _require(xgb.__version__ == "3.2.0", "unsupported_xgboost_runtime", "unsupported")
    model = xgb.Booster(params={"device": "cpu", "nthread": 1})
    model.load_model(bytearray(native))
    model.set_param({"device": "cpu", "nthread": 1})
    _require(tuple(model.feature_names) == receipt.selected_features and model.num_features() == len(receipt.selected_features)
             and model.num_boosted_rounds() == receipt.boosting_metadata["rounds"], "loaded_model_metadata_mismatch", "unsupported")
    best = model.attr("best_iteration")
    _require((int(best) if best is not None else None) == receipt.boosting_metadata["best_iteration"], "loaded_iteration_metadata_mismatch", "unsupported")
    config = _parse(model.save_config().encode(), VerificationLimits())["learner"]
    _require(config["objective"]["name"] == "multi:softprob" and config["learner_model_param"]["num_class"] == "3", "loaded_objective_mismatch", "unsupported")
    return model.inplace_predict(scaled, iteration_range=iterations, predict_type="value", missing=np.nan, validate_features=True)


def replay_research_bundle(bundle_directory, expected_bundle_id, *, enabled=False, probability_atol=0.0, limits=ReplayLimits()):
    """Opt-in replay from a complete independently verified bundled input closure.

    Caller matrices and external-only input declarations are not accepted. The
    original capture remains the reference; no replacement capture, generation
    timestamp or fingerprint is manufactured. Default probability comparison is
    bit-exact float32. An explicitly requested absolute tolerance up to 1e-7 is
    supported with rtol=0 and exact class labels. No probabilities are repaired.
    """
    names = ("activation", "artifact_integrity", "receipt_state_consistency", "runtime_model_loading",
             "frozen_input_closure", "functional_replay", "temporal_decision_provenance")
    checks = {name: VerificationCheck("not_assessed") for name in names}
    result = {"probability_atol": probability_atol}
    stage = "activation"
    def mark(name, status, *codes):
        checks[name] = VerificationCheck(status, tuple(codes))
    def report(status):
        return ReplayReport(status, tuple(checks.items()), **result)
    if type(enabled) is not bool:
        mark("activation", "failed", "explicit_boolean_activation_required")
        return report("failed")
    if not enabled:
        mark("activation", "disabled")
        return report("disabled")
    mark("activation", "verified")
    try:
        _require(type(limits) is ReplayLimits and type(probability_atol) is float and math.isfinite(probability_atol)
                 and 0 <= probability_atol <= 1e-7, "unsupported_comparison_policy", "unsupported")
        stage = "artifact_integrity"
        audit = verify_research_bundle(bundle_directory, expected_bundle_id, limits=limits.verification)
        for key in ("artifact_integrity", "bundle_manifest", "artifact_schemas", "protocol_identity", "receipt_identity"):
            item = audit.check(key)
            _require(item.status == "verified", "bundle_" + key + "_" + item.status,
                     item.status if item.status in {"unsupported", "operational_error"} else "failed")
        mark(stage, "verified")
        stage = "receipt_state_consistency"
        for key in ("reference_consistency", "state_envelope_hashes", "state_envelope_schema", "prediction_capture", "temporal_provenance"):
            item = audit.check(key)
            _require(item.status == "verified", "bundle_" + key + "_" + item.status,
                     item.status if item.status in {"unsupported", "incomplete", "operational_error"} else "failed")
        mark(stage, "verified")
        mark("temporal_decision_provenance", audit.check("prediction_decision_chronology").status,
             "capture_decision_clocks_not_retained")
        # Reopening re-verifies every image against the SAME expected manifest
        # identity. No unhashed native/source path is opened after verification.
        owned = open_research_bundle(bundle_directory, expected_bundle_id=expected_bundle_id, limits=limits.verification.archive)
        by_role = {item.role: item for item in owned.artifacts if item.role != "source_input"}
        protocol = ResearchProtocolV1.from_json(by_role["protocol"].data.decode())
        receipt = ResearchFitReceiptV1.from_json(by_role["receipt"].data.decode())
        result.update(protocol_id=protocol.protocol_id, receipt_id=receipt.receipt_id, fitted_model_id=receipt.fitted_model_id,
                      reference_capture_fingerprint=receipt.prediction_capture_fingerprint,
                      retained_boosting_rounds=receipt.boosting_metadata["rounds"])
        stage = "runtime_model_loading"
        _check_runtime(receipt)
        _require(receipt.target_configuration.task_type == "5d_class"
                 and _encode(receipt.target_configuration.class_encoding) == _encode(((0, "Down"), (1, "Neutral"), (2, "Up"))), "unsupported_target_class_mapping", "unsupported")
        _require("native_booster" in by_role, "missing_native_model", "incomplete")
        native = by_role["native_booster"].data
        _native_preflight(native, receipt, limits)
        estimator = _read_state(_parse(by_role["estimator_state"].data, limits.verification)["canonical_state_payload"], limits.verification)
        parameters = estimator["parameters"]
        _require(parameters.get("device") in {None, "cpu"} and parameters.get("enable_categorical") in {None, False}
                 and parameters.get("base_margin") is None and type(parameters["missing"]) is float
                 and math.isnan(parameters["missing"]), "unsupported_inference_configuration", "unsupported")
        stage = "frozen_input_closure"
        _require(owned.manifest.metadata.input_closure.mode == "bundled" and audit.check("input_closure").status == "verified",
                 "complete_bundled_inputs_required", "incomplete")
        _require(audit.check("training_inputs").status == "verified", "frozen_recipe_runtime_unsupported", "unsupported")
        with tempfile.TemporaryDirectory(prefix="apf-functional-replay-") as directory:
            root = Path(directory)
            for artifact in protocol.artifacts:
                path = root / artifact.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(owned.artifact_bytes("inputs-root/" + artifact.relative_path))
            prepared = prepare_research_inputs(protocol, root)
        _require(prepared.protocol_id == protocol.protocol_id == receipt.protocol_id
                 and prepared.training_input_id == receipt.training_input_id and prepared.frozen_partitions == receipt.partitions,
                 "prepared_input_identity_mismatch")
        capture = deserialize_prediction_capture(PredictionCaptureArtifact(by_role["prediction_capture"].data,
            by_role["prediction_capture"].entry().sha256, receipt.prediction_capture_fingerprint)).capture
        frame = prepared.held_out.features
        _require(frame.index.identical(capture.held_out_index) and prepared.held_out.row_ids == receipt.partitions.held_out
                 and 0 < len(frame) <= limits.max_rows and len(receipt.selected_features) <= limits.max_features,
                 "held_out_identity_or_size_mismatch")
        _require(capture.feature_names == receipt.selected_features and capture.estimator_classes == (0, 1, 2)
                 and capture.probability_dtype == "float32", "unsupported_capture_inference_contract", "unsupported")
        scaled = _restore_scaling(protocol, receipt, by_role["scaler_state"].data, frame.loc[:, list(receipt.selected_features)], limits)
        mark(stage, "verified")
        iterations = _iteration_range(receipt)
        result["iteration_range"] = iterations
        stage = "runtime_model_loading"
        try:
            probabilities = _load_and_predict(native, scaled, receipt, iterations)
        except _Rejected:
            raise
        except Exception as error:
            raise _Rejected("native_runtime_load_or_prediction_" + type(error).__name__, "unsupported") from None
        mark(stage, "verified")
        stage = "functional_replay"
        reference = np.asarray([[row.probabilities[label] for label in capture.estimator_classes] for row in capture.records], dtype=np.float32)
        _require(type(probabilities) is np.ndarray and probabilities.dtype == np.dtype("float32") and probabilities.shape == reference.shape
                 and np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), "invalid_replay_probability_output")
        errors = np.abs(probabilities.astype(np.float64) - reference.astype(np.float64))
        bits = probabilities.view(np.uint32) == reference.view(np.uint32)
        row_matches = bits.all(axis=1) if probability_atol == 0 else (errors <= probability_atol).all(axis=1)
        labels = np.argmax(probabilities, axis=1)
        class_errors = labels != np.asarray([row.predicted_class for row in capture.records], dtype=np.int64)
        result.update(compared_rows=len(reference), maximum_absolute_error=float(errors.max()),
            probability_mismatched_rows=int((~row_matches).sum()), class_mismatched_rows=int(class_errors.sum()),
            bit_exact_probabilities=bool(bits.all()))
        _require(row_matches.all() and not class_errors.any(), "prediction_replay_mismatch")
        mark(stage, "verified")
        return report("verified")
    except _Rejected as error:
        mark(stage, error.status, error.code)
        return report(error.status)
    except (OSError, ImportError) as error:
        status = "unsupported" if stage == "runtime_model_loading" else "operational_error"
        mark(stage, status, type(error).__name__)
        return report(status)
    except Exception as error:
        mark(stage, "failed", "replay_validation_" + type(error).__name__)
        return report("failed")

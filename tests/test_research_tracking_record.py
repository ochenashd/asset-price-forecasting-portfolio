"""Data-only synthetic graphs: no estimator is fitted, loaded or executed.

One-leaf state arrays and illustrative probabilities are explicit fixture
claims, not witnessed training, functional reproduction or provider evidence.
All positive bundles pass the real independent byte/scientific verifier.
"""
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_research_input_adapter import inputs as source_inputs, research, FEATURES
from test_research_bundle_verification import scientific_bundle
from test_research_bundle import snapshot
from src.tracking import research_record as api
from src.research import bundle as archive
from src.research import protocol as contract
from src.research import receipt as receipts
from src.research import state_artifacts as state_api
from src.research.bundle_verification import verify_research_bundle, _temporal
from src.research.bundle_publication import ResearchPublicationRecord, RESEARCH_ARCHIVE_POLICY, ACCEPTED, accepts_research_archive
from src.research.prediction_artifacts import serialize_prediction_capture
from src.evaluation.execution.xgboost_bridge import capture_xgboost_predictions
from src.evaluation.execution.pipeline_handoff import CaptureReceipt, FittedModelBinding, TargetConfiguration, capture_fingerprint


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def inputs(source_inputs):
    p, root, source, frame = source_inputs
    p = replace(p, environment=replace(p.environment, package_versions={**p.environment.package_versions, "xgboost": "3.2.0"}))
    return p, root, source, frame


@pytest.fixture
def accepted(scientific_bundle):
    root, template, old_meta, p, r = scientific_bundle
    n = len(FEATURES)
    classes = np.array([0, 1, 2], dtype="int64")
    node_dtype = [("left_child", "<i8"), ("right_child", "<i8"), ("feature", "<i8"), ("threshold", "<f8"),
                  ("impurity", "<f8"), ("n_node_samples", "<i8"), ("weighted_n_node_samples", "<f8")]
    nodes = np.array([(-1, -1, -2, -2., .5, 60, 60.)], dtype=node_dtype)
    selector = {"class": "sklearn.RandomForestClassifier", "parameters": {"n_estimators": 1}, "features": FEATURES,
        "classes": classes, "n_outputs": 1, "trees": ({"parameters": {}, "classes": classes,
            "state": {"max_depth": 0, "node_count": 1, "nodes": nodes, "values": np.array([[[20.,20.,20.]]])}},)}
    scaler = {"class": "sklearn.RobustScaler", "parameters": {}, "learned": {
        "feature_names_in_": np.array(FEATURES, dtype=object), "n_features_in_": n, "center_": np.zeros(n), "scale_": np.ones(n)}}
    tree = {"tree_param": {"num_nodes": "1", "num_feature": str(n), "size_leaf_vector": "1"},
        "base_weights": [0.], "left_children": [-1], "right_children": [-1], "parents": [2147483647],
        "split_conditions": [0.], "split_indices": [0], "default_left": [0], "loss_changes": [0.], "sum_hessian": [1.], "split_type": [0]}
    booster = {"version": [3,2,0], "learner": {"feature_names": list(FEATURES), "feature_types": ["float"] * n,
        "learner_model_param": {"num_feature": str(n), "num_class": "3"},
        "attributes": {"best_iteration": "1", "best_score": "synthetic"},
        "gradient_booster": {"name": "gbtree", "model": {"gbtree_model_param": {"num_trees": "9"}, "trees": [tree] * 9}}}}
    estimator = {"format": "xgboost-json-v1", "parameters": {**p.learning.effective_parameters, "objective": "multi:softprob"},
        "booster": booster, "configuration": {"learner": {"learner_model_param": {"num_feature": str(n), "num_class": "3"}, "objective": {"name": "multi:softprob"}}}}
    images = {"selector": receipts._bytes(selector), "scaler": receipts._bytes(scaler), "estimator": receipts._bytes(estimator)}
    descriptor = receipts._bytes({"training_input_id": r.training_input_id, "selected_features": FEATURES,
        "selector": sha(images["selector"]), "scaler": sha(images["scaler"]), "recipe": contract._encode(p.transformations)})
    images["preprocessing"] = descriptor
    binding = FittedModelBinding(r.experiment_id, FEATURES, TargetConfiguration("5d_class",5,.005), ("selector","scaler"), True, images["estimator"], descriptor)
    temporal = replace(_temporal(r.temporal_manifest_json.encode(), api.VerificationLimits()), model_id=binding.model_id)
    allowed = set(temporal.fitting_ids + temporal.inner_purged_ids + temporal.validation_ids)
    split = SimpleNamespace(source_index=pd.DatetimeIndex([row.row_id for row in temporal.source_rows if row.row_id in allowed]),
        fitting_index=pd.DatetimeIndex(temporal.fitting_ids), validation_index=pd.DatetimeIndex(temporal.validation_ids))
    index = pd.DatetimeIndex(temporal.held_out_ids)
    capture = capture_xgboost_predictions(test_index=index, feature_names=FEATURES, model_feature_names=FEATURES,
        predictions=np.full(len(index), 2), probabilities=np.tile([.2,.3,.5], (len(index),1)),
        estimator=SimpleNamespace(classes_=classes, feature_names_in_=FEATURES), ticker=r.ticker, asset_id=r.asset_id,
        temporal_split=split, model_id=binding.model_id, split_id=temporal.split_id, actual_prediction_generated_at=r.actual_prediction_generated_at)
    capture = replace(capture, estimator_type="xgboost.sklearn.XGBClassifier")
    fingerprint = capture_fingerprint(capture)
    r = replace(r, fitted_model_id=binding.model_id, state_digests={name:sha(data) for name,data in images.items()},
        preprocessing_binding_json=descriptor.decode(), temporal_manifest_id=temporal.split_id,
        temporal_manifest_json=temporal.canonical_json(), prediction_capture_fingerprint=fingerprint)
    prediction = serialize_prediction_capture(capture, fitting_attempt=CaptureReceipt(r.attempt_id, fingerprint, binding.model_id, r.asset_id))
    native = contract._json(booster).encode()
    context = state_api._context(r, FEATURES, sha(native))
    packages = {}
    for role,data in images.items():
        envelope = {"schema_version": state_api.SCHEMA_VERSION, "role": role,
            "serialization": {"format": "apf-receipt-canonical-json", "version": 1}, "receipt_state_identity": sha(data),
            "state_envelope_sha256": sha(data), "canonical_state_payload": data.decode(), "context": context}
        image = contract._json(envelope).encode()
        packages[role] = state_api.FittedStateArtifact(role,image,sha(image))
    packages["native_booster"] = state_api.FittedStateArtifact("native_booster", native, sha(native))
    checked = state_api.verify_fitted_state_artifacts(packages, r)
    assert checked.status == "verified", checked.diagnostics
    meta = replace(old_meta, fitted_model_id=r.fitted_model_id, receipt_id=r.receipt_id,
        temporal_manifest_id=r.temporal_manifest_id, prediction_capture_fingerprint=fingerprint)
    role_map = {"selector_state":"selector", "scaler_state":"scaler", "estimator_state":"estimator", "preprocessing_binding":"preprocessing"}
    artifacts=[]
    for item in template:
        if item.role in role_map:
            item=replace(item,data=packages[role_map[item.role]].data,schema_identity=state_api.SCHEMA_VERSION)
        elif item.role=="receipt": item=replace(item,data=r.canonical_json().encode())
        elif item.role=="temporal_manifest": item=replace(item,data=temporal.canonical_json().encode())
        elif item.role=="prediction_capture": item=replace(item,data=prediction.data,schema_identity="apf-keyed-prediction-artifact-v1")
        if item.role in archive.ROLE_SPECS: item=replace(item,related_identity=getattr(meta,archive.ROLE_SPECS[item.role][2]))
        artifacts.append(item)
    artifacts.append(archive.ResearchArtifactInput("native_booster","models/booster.json",native,"json","1","xgboost-native-json-v1",r.fitted_model_id))
    root.mkdir()
    published=archive.publish_research_bundle(root,tuple(artifacts),meta)
    assert published.status=="published", published.diagnostics
    report=verify_research_bundle(published.directory,published.bundle_id)
    assert accepts_research_archive(report,published.bundle_id), report.to_dict()
    publication=ResearchPublicationRecord(r.ticker,"synthetic-pipeline-run",r.attempt_id,p.protocol_id,r.receipt_id,
        published.bundle_id,root,"published",ACCEPTED,published.directory,published.directory,report)
    metadata=dict(ticker=r.ticker,asset_id=r.asset_id,fitting_attempt_id=r.attempt_id,pipeline_run_id=publication.run_id,
        archive_alias="synthetic_archive",tracking_destination_id="research_index",training_status="success",reporting_status="success")
    return SimpleNamespace(publication=publication,published=published,metadata=metadata,receipt=r,protocol=p,artifacts=tuple(artifacts))


def build(case, **changes):
    args=dict(publication=case.publication,bundle_directory=case.published.directory,runtime_metadata=case.metadata)
    args.update(changes)
    return api.build_research_tracking_record(**args)


def test_real_data_only_verification_and_record_roundtrip(accepted):
    c=accepted
    before=snapshot(c.published.directory)
    record=build(c)
    assert record.receipt_id==c.receipt.receipt_id and record.bundle_id==c.published.bundle_id
    assert record.training_input_id==c.receipt.training_input_id
    assert record.research_verification_status==record.decision_time_evidence=="incomplete"
    assert record.historical_pit_status=="unknown" and record.execution_eligibility_status=="unassessed"
    assert record.functional_replay_status=="not_assessed"
    assert api.ResearchTrackingRecordV1.from_json(record.canonical_json())==record
    assert build(c).record_id==record.record_id
    assert snapshot(c.published.directory)==before


def rejected(case, code=None, **changes):
    with pytest.raises(api.ResearchTrackingError) as caught:
        build(case,**changes)
    if code:
        assert caught.value.diagnostics[0].code==code
    return caught.value


@pytest.mark.parametrize("field", api.IDENTITIES)
def test_every_expected_scientific_identity_is_cross_checked(accepted,field):
    record=build(accepted)
    value=getattr(record,field)
    wrong=value[:-1]+("0" if value[-1]!="0" else "1") if field in api.PREFIXES else value+"-different"
    rejected(accepted,"scientific_identity_mismatch",expected_identities={field:wrong})
    assert build(accepted,expected_identities={field:getattr(record,field)})==record


@pytest.mark.parametrize("field", ["protocol_id","receipt_id","attempt_id","ticker","bundle_id","run_id"])
def test_publication_identity_mismatches_fail_closed(accepted,field):
    value=getattr(accepted.publication,field)
    value=value[:-1]+("0" if value[-1]!="0" else "1") if field in api.PREFIXES else value+"-wrong"
    wrong=replace(accepted.publication,**{field:value})
    rejected(accepted,publication=wrong)


@pytest.mark.parametrize("field", ["ticker","asset_id","fitting_attempt_id","pipeline_run_id"])
def test_runtime_owner_is_not_inferred_or_coerced(accepted,field):
    rejected(accepted,runtime_metadata={**accepted.metadata,field:accepted.metadata[field]+"-wrong"})


@pytest.mark.parametrize("outcome", ["conflict","publication_busy","not_published","published_durability_uncertain"])
def test_failed_or_uncertain_publication_is_never_a_durable_reference(accepted,outcome):
    publication=replace(accepted.publication,publisher_outcome=outcome)
    rejected(accepted,"durability_uncertainty" if outcome=="published_durability_uncertain" else "publication_not_accepted",publication=publication)


@pytest.mark.parametrize("change", ["acceptance","policy","path","closure","pit","execution","temporal","report"])
def test_publication_acceptance_and_report_must_remain_supported(accepted,change):
    options={"acceptance":{"archive_status":"not_accepted"}, "policy":{"acceptance_policy":"another-policy"},
        "path":{"published_path":None}, "closure":{"input_closure_status":"incomplete"},
        "pit":{"pit_evidence_status":"verified"}, "execution":{"execution_eligibility_status":"eligible"},
        "temporal":{"temporal_evidence_completeness":"verified"}, "report":{"verification":None}}[change]
    rejected(accepted,publication=replace(accepted.publication,**options))


def test_matching_already_exists_preserves_outcome_and_linkage_identity(accepted):
    original=build(accepted)
    outcome=archive.publish_research_bundle(accepted.publication.archive_root,accepted.artifacts,accepted.published.bundle.manifest.metadata)
    assert outcome.status=="already_exists"
    second=build(accepted,publication=replace(accepted.publication,publisher_outcome=outcome.status))
    assert second.publisher_outcome=="already_exists"
    assert second.record_id!=original.record_id and second.linkage_key==original.linkage_key


def test_relocation_and_loss_of_original_sources_do_not_change_identity(accepted,tmp_path):
    record=build(accepted)
    target_root=(tmp_path/"relocated").resolve()
    target_root.mkdir()
    target=target_root/accepted.published.bundle_id
    shutil.copytree(accepted.published.directory,target)
    publication=replace(accepted.publication,archive_root=target_root,published_path=target,observed_path=target)
    moved=build(accepted,publication=publication,bundle_directory=target)
    assert moved==record and moved.record_id==record.record_id and moved.linkage_key==record.linkage_key
    assert str(target_root) not in moved.canonical_json()


@pytest.mark.parametrize("change", ["publication","bundle","missing_path","tampering"])
def test_missing_or_corrupt_inputs_return_structured_errors(accepted,change):
    if change=="publication":rejected(accepted,"missing_or_invalid_publication",publication=None)
    elif change=="bundle":rejected(accepted,"missing_bundle",bundle_directory=None)
    elif change=="missing_path":
        p=accepted.publication
        missing=p.archive_root/"missing"/p.bundle_id
        rejected(accepted,"missing_bundle",publication=replace(p,archive_root=missing.parent,published_path=missing),bundle_directory=missing)
    else:
        source=accepted.published.directory/"selected-features.json"
        source.chmod(0o600)
        source.write_bytes(b"Tampered synthetic artifact")
        rejected(accepted,"invalid_bundle_integrity")


@pytest.mark.parametrize("key", ["api_token","password","hmac_key","database_url","configuration","predictions","probabilities","model_binary","mlflow_run_id","unexpected"])
def test_unknown_or_sensitive_runtime_keys_never_enter_records_or_errors(accepted,key):
    secret="sensitive-fixture-value-never-echo"
    error=rejected(accepted,"unexpected_or_sensitive_metadata",runtime_metadata={**accepted.metadata,key:secret})
    assert secret not in str(error) and secret not in repr(error.diagnostics)


@pytest.mark.parametrize("field", ["archive_alias","tracking_destination_id"])
@pytest.mark.parametrize("value", ["/Users/private/archive","../archive","C:\\private\\archive","https://example.invalid/?token=private", "postgresql://user:password@host/db", "api_token_private", "x"*65, "", 1])
def test_logical_locators_cannot_be_paths_urls_or_credentials(accepted,field,value):
    rejected(accepted,runtime_metadata={**accepted.metadata,field:value})


@pytest.mark.parametrize("field", ["training_status","reporting_status"])
@pytest.mark.parametrize("value", ["failed","partial_success","running"])
def test_nonfinal_training_or_reporting_is_rejected(accepted,field,value):
    rejected(accepted,"successful_training_required",runtime_metadata={**accepted.metadata,field:value})


def test_identity_content_and_destination_have_documented_separate_effects(accepted):
    original=build(accepted)
    assert original.record_id==api.SCHEMA_VERSION+"-"+sha(original.canonical_json().encode())
    different_alias=replace(original,archive_alias="another_archive")
    assert different_alias.record_id!=original.record_id and different_alias.linkage_key==original.linkage_key
    different_destination=replace(original,tracking_destination_id="another_index")
    assert different_destination.record_id!=original.record_id and different_destination.linkage_key!=original.linkage_key
    assert "record_id" not in json.loads(original.canonical_json()) and "linkage_key" not in json.loads(original.canonical_json())
    with pytest.raises(FrozenInstanceError): original.asset_id="changed"
    with pytest.raises(TypeError):original.verification_summary[0]="changed"


@pytest.mark.parametrize("change", ["unknown","schema","mlflow_id","typed_integer","objects","deep","oversize","duplicates","bad_json","claim","profile","limitations"])
def test_decoder_is_bounded_allowlisted_and_never_instantiates_objects(accepted,change):
    record=build(accepted)
    raw=json.loads(record.canonical_json())
    if change=="unknown":raw["unexpected"]="not-allowed"
    elif change=="schema":raw["schema_version"]="unsupported-version"
    elif change=="mlflow_id":raw["fields"]["mlflow_run_id"]=["str","server-id"]
    elif change=="typed_integer":raw["fields"]["asset_id"]=["int","1"]
    elif change=="objects":raw["fields"]["ticker"]={"type":"ArbitraryClass","fields":{}}
    elif change=="deep":
        val=["str","x"]
        for _ in range(100):val=["tuple",[val]]
        raw["fields"]["ticker"]=val
    elif change=="claim":raw["fields"]["functional_replay_status"]=["str","verified"]
    elif change=="profile":raw["fields"]["estimator_version"]=["str","99.0.0"]
    elif change=="limitations":raw["fields"]["provenance_limitations"]=["tuple",[]]
    value=contract._json(raw)
    if change=="oversize":value=" "*(api.MAX_RECORD_BYTES+1)
    elif change=="duplicates":value='{"schema_version":"x","schema_version":"y","fields":{}}'
    elif change=="bad_json":value='not JSON'
    with pytest.raises(api.ResearchTrackingError):api.ResearchTrackingRecordV1.from_json(value)


def test_pure_mlflow_tags_are_immutable_small_and_have_no_mlflow_run_id(accepted):
    record=build(accepted)
    tags=api.to_mlflow_metadata(record)
    assert tags["apf.record_id"]==record.record_id and tags["apf.linkage_key"]==record.linkage_key
    assert tags["apf.archive_alias"]=="synthetic_archive"
    assert tags["apf.functional_replay_status"]=="not_assessed"
    assert tags["apf.historical_pit_status"]=="unknown"
    assert all(type(k) is str and type(v) is str and k.startswith("apf.") for k,v in tags.items())
    assert not any("mlflow_run_id" in key or "metric" in key for key in tags)
    assert str(accepted.publication.archive_root) not in contract._json(dict(tags))
    assert len(contract._json(dict(tags)).encode()) < api.MAX_RECORD_BYTES
    with pytest.raises(TypeError):tags["apf.asset_id"]="changed"


def test_fresh_verifier_failure_is_not_bypassed_by_previous_accepted_report(accepted,monkeypatch):
    from src.research.bundle_verification import VerificationCheck
    original=api.verify_research_bundle
    def failed(*args):
        report=original(*args)
        return replace(report,checks=tuple((name,VerificationCheck("failed")) if name=="temporal_provenance" else (name,check)
                                           for name,check in report.checks))
    monkeypatch.setattr(api,"verify_research_bundle",failed)
    rejected(accepted,"bundle_verification_rejected")


def test_unsupported_scientific_schema_is_a_structured_blocker(accepted,monkeypatch):
    from src.research.bundle_verification import VerificationCheck
    original=api.verify_research_bundle
    def unsupported(*args):
        report=original(*args)
        return replace(report,checks=tuple((name,VerificationCheck("unsupported")) if name=="state_envelope_schema" else (name,check)
                                           for name,check in report.checks))
    monkeypatch.setattr(api,"verify_research_bundle",unsupported)
    rejected(accepted,"unsupported_scientific_schema")


def test_unexpected_programming_errors_are_not_converted_to_validation_success(accepted,monkeypatch):
    def broken(*args):raise RuntimeError("Synthetic programming defect")
    monkeypatch.setattr(api,"verify_research_bundle",broken)
    with pytest.raises(RuntimeError,match="Synthetic programming defect"):build(accepted)


def test_unsupported_model_profiles_are_not_guessed_from_names(accepted):
    record=build(accepted)
    for changes in ({"model_type":"lightgbm"},{"estimator_version":"3.1.0"},{"model_profile":"unregistered-profile"},
                    {"estimator_class":"arbitrary.Class"}):
        with pytest.raises(api.ResearchTrackingError) as caught:replace(record,**changes)
        assert caught.value.diagnostics[0].code=="unsupported_model_profile"


def test_builder_never_calls_mlflow_models_services_execution_or_publication(accepted,monkeypatch):
    import builtins
    import socket
    import sqlite3
    import sys
    from src.evaluation.execution.engine import ExecutionEngineV1
    original_import=builtins.__import__
    def forbidden(*args,**kwargs):pytest.fail("Tracking contract must be data-only")
    def checked_import(name,*args,**kwargs):
        if name.split('.')[0] in {"mlflow","sklearn","xgboost","sqlalchemy","psycopg2","yfinance"}:forbidden()
        return original_import(name,*args,**kwargs)
    for method in ("create_connection","getaddrinfo"):monkeypatch.setattr(socket,method,forbidden)
    monkeypatch.setattr(socket.socket,"connect",forbidden)
    monkeypatch.setattr(sqlite3,"connect",forbidden)
    monkeypatch.setattr(ExecutionEngineV1,"run",forbidden)
    monkeypatch.setattr(archive,"publish_research_bundle",forbidden)
    if "mlflow" in sys.modules:
        module=sys.modules["mlflow"]
        for method in ("active_run","start_run","set_experiment","log_param","log_params","log_metric","log_metrics","log_artifact","log_dict","MlflowClient"):
            if hasattr(module,method):monkeypatch.setattr(module,method,forbidden)
    monkeypatch.setattr(builtins,"__import__",checked_import)
    before=snapshot(accepted.published.directory)
    record=build(accepted)
    assert api.to_mlflow_metadata(record)["apf.bundle_id"]==accepted.published.bundle_id
    assert api.ResearchTrackingRecordV1.from_json(record.canonical_json())==record
    assert snapshot(accepted.published.directory)==before


def test_expected_identity_assertions_cannot_supply_unknown_metadata(accepted):
    rejected(accepted,"unexpected_or_sensitive_metadata",expected_identities={"mlflow_run_id":"not-authoritative"})
    rejected(accepted,"scientific_identity_mismatch",expected_identities={"asset_id":1})


def test_publication_with_missing_authoritative_identity_is_a_blocker(accepted):
    rejected(accepted,publication=replace(accepted.publication,protocol_id=""))


@pytest.mark.parametrize("field,value", [("archive_root",None),("bundle_id",None),("attempt_id",None),("run_id",None)])
def test_malformed_typed_publication_is_a_validation_error(accepted,field,value):
    rejected(accepted,publication=replace(accepted.publication,**{field:value}))


@pytest.mark.parametrize("field,value", [("ticker",object()),("verification_summary",((object(),"verified"),)),
    ("provenance_limitations",(object(),)),("training_status",1)])
def test_direct_record_construction_rejects_nonprimitive_values(accepted,field,value):
    record=build(accepted)
    with pytest.raises(api.ResearchTrackingError):replace(record,**{field:value})

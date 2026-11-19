"""Synthetic archive verification without fitting, inference or simulations.

Dummy state envelopes deliberately have no model codec. Their hashes can be
verified, but they must never produce a complete scientific success claim.
"""
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

from test_research_input_adapter import inputs, research, FEATURES
from test_research_bundle import snapshot
from src.research import bundle as archive
from src.research import protocol as contract
from src.research import receipt as receipts
from src.research.input_adapter import prepare_research_inputs
from src.research.bundle_verification import VerificationLimits, verify_research_bundle, _parse
from src.evaluation.execution.model_provenance import FrozenModelManifest, SourceRow, FittedStateAvailability
from src.evaluation.execution.pipeline_handoff import FittedModelBinding, TargetConfiguration


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def scientific_bundle(inputs, tmp_path):
    """All clocks and state claims are synthetic declarations, not witnesses."""
    p, root, source, _ = inputs
    prepared = prepare_research_inputs(p, root)
    states = {name: json.dumps({"synthetic_state": name}).encode() for name in ("selector", "scaler", "estimator")}
    descriptor = receipts._bytes({"training_input_id": prepared.training_input_id, "selected_features": FEATURES,
        "selector": sha(states["selector"]), "scaler": sha(states["scaler"]), "recipe": contract._encode(p.transformations)})
    exp = "synthetic-verifier-experiment"
    binding = FittedModelBinding(exp, FEATURES, TargetConfiguration("5d_class", 5, .005),
        ("selector", "scaler"), True, states["estimator"], descriptor)
    held = set(p.partitions.held_out)
    rows = tuple(SourceRow(row.observation_at, row.observation_at, row.observation_at + timedelta(minutes=1),
        row.target_end_timestamp, None if row.row_id in held else row.target_end_timestamp + timedelta(minutes=2))
        for row in prepared.row_bindings)
    selection = max(row.labels_available_at for row in rows if row.row_id in prepared.fitting.features.index) + timedelta(minutes=1)
    early = max(row.labels_available_at for row in rows if row.row_id in prepared.validation.features.index) + timedelta(minutes=1)
    completed = datetime(2025, 1, 1, tzinfo=timezone.utc)
    temporal = FrozenModelManifest(p.asset_id, binding.model_id, exp, rows,
        tuple(prepared.fitting.features.index), tuple(prepared.inner_purged.features.index), tuple(prepared.outer_purged.features.index),
        tuple(prepared.validation.features.index), tuple(prepared.held_out.features.index), rows[0].observation_at,
        prepared.validation.features.index[0], prepared.held_out.features.index[0], "5d_class", 5, .005, FEATURES,
        selection, (FittedStateAvailability("selector", selection), FittedStateAvailability("scaler", selection)),
        True, early, timedelta(minutes=10), timedelta(minutes=1), completed)
    r = receipts.ResearchFitReceiptV1(p.protocol_id, prepared.training_input_id, p.artifacts, p.partitions,
        tuple((row.row_id, row.source_id, row.observation_at, row.target_end_source_id) for row in prepared.row_bindings), FEATURES,
        {"correlation": p.transformations.correlation, "importance": p.transformations.feature_selection},
        {**{name: sha(value) for name, value in states.items()}, "preprocessing": sha(descriptor)}, descriptor.decode(), binding.model_id,
        p.target, p.learning.effective_parameters, {"rounds": 3, "best_iteration": 1, "best_score": "synthetic", "uses_early_stopping": True},
        "synthetic-attempt", "SYNTH", p.asset_id, exp, temporal.split_id, temporal.canonical_json(), sha(b"synthetic-capture"),
        completed, completed + timedelta(minutes=1), p.environment, p.transformations.implementation_reference, "0" * 64, receipts._LIMITATIONS)
    sources = tuple(archive.InputSourceReferenceV1(ref.artifact_id, ref.relative_path, ref.byte_length, ref.sha256, ref.schema_sha256) for ref in p.artifacts)
    meta = archive.ResearchBundleMetadataV1(p.protocol_id, prepared.training_input_id, r.receipt_id, binding.model_id,
        temporal.split_id, r.prediction_capture_fingerprint, p.asset_id, r.attempt_id, exp, "success",
        archive.InputClosureV1("bundled", sources), "unknown", ("Synthetic provenance only.",), "unassessed", ("No market certification.",))
    typed = lambda value: contract._json(contract._encode(value)).encode()
    payloads = {"protocol": p.canonical_json().encode(), "receipt": r.canonical_json().encode(),
        "input_references": typed(p.artifacts), "selected_features": typed(FEATURES),
        "effective_training_configuration": typed(p.learning.effective_parameters), "environment": typed(p.environment),
        "temporal_manifest": temporal.canonical_json().encode(), "preprocessing_binding": descriptor,
        "prediction_capture": b'{"codec":"unimplemented-synthetic-capture"}',
        **{name + "_state": value for name, value in states.items()}}
    artifacts = tuple(archive.ResearchArtifactInput(role, spec[0], payloads[role], spec[1], "1",
        ({"receipt": "apf-research-fit-receipt-v1", "preprocessing_binding": "apf-research-fit-receipt-v1",
         "temporal_manifest": "apf-temporal-manifest-v1"}.get(role, "apf-research-protocol-v1")
         if role not in {"selector_state", "scaler_state", "estimator_state", "prediction_capture"} else "unsupported-synthetic-codec"), getattr(meta, spec[2])) for role, spec in archive.ROLE_SPECS.items() if role in archive.REQUIRED_ROLES)
    artifacts += (archive.ResearchArtifactInput("source_input", "inputs-root/" + p.artifacts[0].relative_path,
        source.read_bytes(), "parquet", "1", sources[0].schema_identity, sources[0].artifact_id, "input-references.json"),)
    return tmp_path / "published", artifacts, meta, p, r


def publish(case, *, artifacts=None, metadata=None):
    root, original, meta, *_ = case
    root.mkdir(exist_ok=True)
    result = archive.publish_research_bundle(root, original if artifacts is None else artifacts, metadata or meta)
    assert result.status in {"published", "already_exists"}, result.diagnostics
    return result


def verify(result, **kwargs):
    return verify_research_bundle(result.directory, result.bundle_id, **kwargs)


def edit(case, role, data):
    return tuple(replace(item, data=data) if item.role == role else item for item in case[1])


def test_supported_components_and_precise_unsupported_codecs(scientific_bundle):
    result = publish(scientific_bundle)
    report = verify(result)
    for name in ("artifact_integrity", "bundle_manifest", "protocol_identity", "receipt_identity", "reference_consistency",
                 "state_envelope_hashes", "temporal_provenance", "input_closure", "training_inputs"):
        assert report.check(name).status == "verified", (name, report.to_dict())
    assert report.status == "unsupported"
    assert report.provenance_origin == "synthetic_fixture"
    assert report.historical_pit_status == "unknown"
    assert report.execution_eligibility_status == "unassessed"
    assert report.check("scientific_consistency").status == "unsupported"
    assert "no_public_safe" in report.check("state_envelope_schema").diagnostics[0]
    assert "no_roundtrip" in report.check("prediction_capture").diagnostics[0]
    for name in ("runtime_model_loading", "functional_reproduction", "historical_point_in_time", "execution_eligibility", "prediction_decision_chronology"):
        assert report.check(name).status == "not_assessed"
    assert report.input_dependencies[0].independently_checked is True
    with pytest.raises(FrozenInstanceError):
        report.expected_bundle_id = "changed"


@pytest.mark.parametrize("change", ["bytes", "manifest", "missing", "extra", "symlink", "hardlink", "fifo", "wrong_id", "schema"])
def test_integrity_rejects_corruption_and_unsafe_inventory(scientific_bundle, change, tmp_path):
    result = publish(scientific_bundle)
    path = result.directory / "selected-features.json"
    result.directory.chmod(0o700)
    if change == "bytes":
        path.chmod(0o600)
        path.write_bytes(b"{}")
    elif change in {"manifest", "schema"}:
        path = result.directory / archive.MANIFEST_NAME
        path.chmod(0o600)
        text = path.read_text()
        path.write_text(text.replace('apf-research-bundle-v1', 'apf-research-bundle-v9', 1) if change == "schema" else text + " ")
    elif change == "extra":
        (result.directory / "unexpected.json").write_text("{}")
    elif change == "wrong_id":
        report = verify_research_bundle(result.directory, "apf-research-bundle-v1-" + "0" * 64)
        assert report.status == "failed"
        return
    else:
        path.unlink()
        if change == "symlink":
            path.symlink_to(tmp_path / "outside")
        elif change == "hardlink":
            other = tmp_path / "outside"
            other.write_bytes(b"{}")
            os.link(other, path)
        elif change == "fifo":
            os.mkfifo(path)
    assert verify(result).status in {"failed", "unsupported"}


@pytest.mark.parametrize("role,check", [("protocol", "protocol_identity"), ("receipt", "receipt_identity"),
    ("selected_features", "reference_consistency"), ("effective_training_configuration", "reference_consistency"),
    ("environment", "reference_consistency"), ("input_references", "reference_consistency"),
    ("temporal_manifest", "temporal_provenance"), ("selector_state", "state_envelope_hashes"),
    ("scaler_state", "state_envelope_hashes"), ("estimator_state", "state_envelope_hashes"),
    ("preprocessing_binding", "state_envelope_hashes")])
def test_valid_inventory_does_not_mask_scientific_mismatch(scientific_bundle, role, check):
    result = publish(scientific_bundle, artifacts=edit(scientific_bundle, role, b'{"changed":true}'))
    report = verify(result)
    assert report.check("artifact_integrity").status == "verified"
    assert report.check(check).status == "failed"
    assert report.status == "failed"


@pytest.mark.parametrize("field", ["training_input_id", "prediction_capture_fingerprint", "attempt_id", "asset_id"])
def test_cross_identity_mismatches(scientific_bundle, field):
    meta = scientific_bundle[2]
    original = getattr(meta, field)
    value = (original[:-64] + "0" * 64) if field in {"training_input_id", "prediction_capture_fingerprint"} else "changed"
    meta = replace(meta, **{field: value})
    artifacts = tuple(replace(item, related_identity=value) if item.role in archive.ROLE_SPECS
        and archive.ROLE_SPECS[item.role][2] == field else item for item in scientific_bundle[1])
    report = verify(publish(scientific_bundle, artifacts=artifacts, metadata=meta))
    assert report.check("reference_consistency").status == "failed"


@pytest.mark.parametrize("change", ["partitions", "maturity", "feature_order"])
def test_receipt_membership_maturity_and_feature_consistency(scientific_bundle, change):
    r = scientific_bundle[4]
    if change == "partitions":
        r = replace(r, partitions=replace(r.partitions, fitting=tuple(reversed(r.partitions.fitting))))
    elif change == "maturity":
        first, *tail = r.row_identity_bindings
        r = replace(r, row_identity_bindings=((first[0], first[1], first[2], "r37"), *tail))
    else:
        # A valid typed feature archive with a different order remains inconsistent.
        data = contract._json(contract._encode(tuple(reversed(r.selected_features)))).encode()
        report = verify(publish(scientific_bundle, artifacts=edit(scientific_bundle, "selected_features", data)))
        assert report.check("reference_consistency").status == "failed"
        return
    meta = replace(scientific_bundle[2], receipt_id=r.receipt_id)
    artifacts = tuple(replace(item, data=r.canonical_json().encode(), related_identity=r.receipt_id)
        if item.role == "receipt" else item for item in scientific_bundle[1])
    assert verify(publish(scientific_bundle, artifacts=artifacts, metadata=meta)).check("reference_consistency").status == "failed"


@pytest.mark.parametrize("mode", ["references_only", "verified_external"])
def test_external_closure_is_never_self_certified(scientific_bundle, mode):
    meta = scientific_bundle[2]
    refs = meta.input_closure.sources
    if mode == "verified_external":
        refs = tuple(replace(item, external=archive.ExternalInputReferenceV1("synthetic-store", "content-key", "declared-proof", True)) for item in refs)
    meta = replace(meta, input_closure=archive.InputClosureV1(mode, refs))
    artifacts = tuple(item for item in scientific_bundle[1] if item.role != "source_input")
    report = verify(publish(scientific_bundle, artifacts=artifacts, metadata=meta))
    assert report.status == "incomplete"
    assert report.check("input_closure").status == "incomplete"
    assert not report.input_dependencies[0].independently_checked


def test_readonly_determinism_relocation_and_no_side_effects(scientific_bundle, tmp_path, monkeypatch):
    result = publish(scientific_bundle)
    before = snapshot(result.directory)
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"sklearn", "xgboost", "mlflow", "psycopg2", "pickle", "joblib"}:
            raise AssertionError("Forbidden runtime import")
        return original_import(name, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access forbidden")
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded)
        guard.setattr(socket, "socket", forbidden)
        guard.setattr(socket, "create_connection", forbidden)
        first, second = verify(result), verify(result)
    assert first == second and first.status == "unsupported"
    assert snapshot(result.directory) == before
    relocated = tmp_path / "relocated" / result.bundle_id
    shutil.copytree(result.directory, relocated)
    assert verify_research_bundle(relocated, result.bundle_id) == first


@pytest.mark.parametrize("case", ["depth", "nodes", "duplicate", "nonfinite", "overflow"])
def test_bounded_json(case):
    limits = VerificationLimits(max_json_depth=4, max_json_nodes=5)
    data = {"depth": b'[[[[[0]]]]]', "nodes": b'[0,0,0,0,0]', "duplicate": b'{"a":1,"a":2}', "nonfinite": b'[NaN]', "overflow": b'[1e999]'}[case]
    with pytest.raises(ValueError):
        _parse(data, limits)


def test_resource_limits_and_unsupported_version(scientific_bundle):
    result = publish(scientific_bundle)
    limits = VerificationLimits(archive.BundleLimits(max_artifact_bytes=10))
    assert verify(result, limits=limits).status == "failed"
    artifacts = tuple(replace(item, format_version="999") if item.role == "protocol" else item for item in scientific_bundle[1])
    assert verify(publish(scientific_bundle, artifacts=artifacts)).check("protocol_identity").status == "unsupported"


@pytest.mark.parametrize("mode,code", [("unsupported", 4), ("invalid", 2), ("incomplete", 3), ("output", 4), ("overwrite", 5)])
def test_cli_readonly_explicit_output_and_exit_status(scientific_bundle, tmp_path, mode, code):
    if mode == "incomplete":
        meta = replace(scientific_bundle[2], input_closure=replace(scientific_bundle[2].input_closure, mode="references_only"))
        result = publish(scientific_bundle, metadata=meta, artifacts=tuple(x for x in scientific_bundle[1] if x.role != "source_input"))
    else:
        result = publish(scientific_bundle)
    before = snapshot(result.directory)
    root = Path(__file__).resolve().parents[1]
    bootstrap = "import sys,runpy;sys.path.extend(" + repr(list(sys.path)) + ");p=sys.argv.pop(1);runpy.run_path(p,run_name='__main__')"
    args = [sys.executable, "-B", "-c", bootstrap, str(root / "scripts/verify_research_bundle.py"), "--bundle-dir", str(result.directory),
            "--expected-bundle-id", result.bundle_id if mode != "invalid" else "wrong"]
    output = tmp_path / "report.json"
    if mode in {"output", "overwrite"}:
        args += ["--json-report", str(output)]
    if mode == "overwrite":
        output.write_text("preserve")
    completed = subprocess.run(args, cwd=root, capture_output=True, text=True)
    assert completed.returncode == code, completed.stderr + completed.stdout
    assert json.loads(completed.stdout)["status"] in {"unsupported", "failed", "incomplete", "operational_error"}
    assert snapshot(result.directory) == before
    if mode == "output":
        assert json.loads(output.read_text())["status"] == "unsupported"
    elif mode == "overwrite":
        assert output.read_text() == "preserve"
    else:
        assert not output.exists()


@pytest.mark.parametrize("role", ["protocol", "receipt"])
def test_canonical_scientific_id_is_recomputed(scientific_bundle, role):
    p, r = scientific_bundle[3:]
    value = replace(p, experiment_label="another-synthetic-experiment") if role == "protocol" else replace(r, attempt_id="another-attempt")
    result = publish(scientific_bundle, artifacts=edit(scientific_bundle, role, value.canonical_json().encode()))
    assert verify(result).check(role + "_identity").status == "failed"


@pytest.mark.parametrize("change", ["purge", "label_availability", "selection_cutoff", "early_stopping", "unknown_version"])
def test_temporal_semantics_not_just_temporal_hash(scientific_bundle, change):
    r = scientific_bundle[4]
    raw = json.loads(r.temporal_manifest_json)
    temporal = raw["manifest"]
    if change == "purge":
        temporal["fitting_ids"].append(temporal["inner_purged_ids"].pop(0))
    elif change == "label_availability":
        temporal["source_rows"][0]["labels_available_at"] = temporal["source_rows"][0]["observation_at"]
    elif change == "selection_cutoff":
        temporal["feature_selection_available_at"] = temporal["training_start"]
    elif change == "early_stopping":
        temporal["early_stopping_available_at"] = temporal["validation_start"]
    else:
        raw["schema_version"] = 999
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    identity = "apf-temporal-v1-" + sha(text.encode())
    r = replace(r, temporal_manifest_id=identity, temporal_manifest_json=text)
    meta = replace(scientific_bundle[2], receipt_id=r.receipt_id, temporal_manifest_id=identity)
    artifacts = tuple(replace(item, data=r.canonical_json().encode(), related_identity=r.receipt_id) if item.role == "receipt"
        else replace(item, data=text.encode(), related_identity=identity) if item.role == "temporal_manifest" else item
        for item in scientific_bundle[1])
    report = verify(publish(scientific_bundle, artifacts=artifacts, metadata=meta))
    assert report.check("receipt_identity").status == "verified"
    assert report.check("temporal_provenance").status == ("unsupported" if change == "unknown_version" else "failed")


def test_unknown_declared_artifact_schema_is_not_silently_accepted(scientific_bundle):
    artifacts = tuple(replace(item, schema_identity="unknown-v99") if item.role == "protocol" else item for item in scientific_bundle[1])
    assert verify(publish(scientific_bundle, artifacts=artifacts)).check("artifact_schemas").status == "unsupported"


def test_bundled_input_dependency_reports_original_schema(scientific_bundle):
    meta = scientific_bundle[2]
    refs = tuple(replace(item, schema_identity="unknown-arrow-schema") for item in meta.input_closure.sources)
    meta = replace(meta, input_closure=replace(meta.input_closure, sources=refs))
    artifacts = tuple(replace(item, schema_identity="unknown-arrow-schema") if item.role == "source_input" else item for item in scientific_bundle[1])
    assert verify(publish(scientific_bundle, artifacts=artifacts, metadata=meta)).check("reference_consistency").status == "failed"


def test_cli_refuses_report_inside_publication(scientific_bundle):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("research_verifier_cli", Path(__file__).resolve().parents[1] / "scripts/verify_research_bundle.py")
    cli = module_from_spec(spec)
    spec.loader.exec_module(cli)
    result = publish(scientific_bundle)
    before = snapshot(result.directory)
    assert cli.main(["--bundle-dir", str(result.directory), "--expected-bundle-id", result.bundle_id,
                     "--json-report", str(result.directory / "report.json")]) == 5
    assert snapshot(result.directory) == before


def test_cli_imports_no_model_runtimes_and_makes_no_network_requests(scientific_bundle):
    result = publish(scientific_bundle)
    script = Path(__file__).resolve().parents[1] / "scripts/verify_research_bundle.py"
    bootstrap = '''import sys, builtins, runpy
sys.path.extend(PATHS)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'sklearn', 'xgboost', 'mlflow', 'psycopg2', 'joblib'}:
        raise AssertionError('Model or service import forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def audit(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.bind'}:
        raise AssertionError('Network access forbidden')
sys.addaudithook(audit)
script = sys.argv.pop(1)
runpy.run_path(script, run_name='__main__')
'''.replace('PATHS', repr(list(sys.path)))
    process = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(script), "--bundle-dir", str(result.directory),
        "--expected-bundle-id", result.bundle_id], capture_output=True, text=True)
    assert process.returncode == 4, process.stderr + process.stdout
    assert json.loads(process.stdout)["checks"]["training_inputs"]["status"] == "verified"


def test_missing_root_is_an_operational_error(tmp_path):
    identity = "apf-research-bundle-v1-" + "0" * 64
    report = verify_research_bundle(tmp_path / identity, identity)
    assert report.status == "operational_error"
    assert report.check("protocol_identity").status == "not_assessed"


def test_selected_predictor_order_may_differ_from_candidate_order(scientific_bundle):
    p, r = scientific_bundle[3:]
    selected = tuple(reversed(r.selected_features))
    descriptor = receipts._bytes({"training_input_id": r.training_input_id, "selected_features": selected,
        "selector": r.state_digests["selector"], "scaler": r.state_digests["scaler"], "recipe": contract._encode(p.transformations)})
    estimator = next(x.data for x in scientific_bundle[1] if x.role == "estimator_state")
    binding = FittedModelBinding(r.experiment_id, selected, TargetConfiguration("5d_class", 5, .005),
        ("selector", "scaler"), True, estimator, descriptor)
    raw = json.loads(r.temporal_manifest_json)
    raw["manifest"]["model_id"] = binding.model_id
    raw["manifest"]["feature_names"] = list(selected)
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    temporal_id = "apf-temporal-v1-" + sha(text.encode())
    r = replace(r, selected_features=selected, preprocessing_binding_json=descriptor.decode(),
        state_digests={**r.state_digests, "preprocessing": sha(descriptor)}, fitted_model_id=binding.model_id,
        temporal_manifest_id=temporal_id, temporal_manifest_json=text)
    meta = replace(scientific_bundle[2], receipt_id=r.receipt_id, fitted_model_id=binding.model_id, temporal_manifest_id=temporal_id)
    values = {"receipt": r.canonical_json().encode(), "preprocessing_binding": descriptor, "temporal_manifest": text.encode(),
              "selected_features": contract._json(contract._encode(selected)).encode()}
    artifacts = tuple(replace(x, data=values.get(x.role, x.data), related_identity=getattr(meta, archive.ROLE_SPECS[x.role][2]))
        if x.role != "source_input" else x for x in scientific_bundle[1])
    report = verify(publish(scientific_bundle, artifacts=artifacts, metadata=meta))
    assert report.check("reference_consistency").status == "verified"
    assert report.check("temporal_provenance").status == "verified"
    assert report.status == "unsupported"


def test_unknown_state_payloads_still_obey_json_limits(scientific_bundle):
    data = b"[" * 70 + b"0" + b"]" * 70
    result = publish(scientific_bundle, artifacts=edit(scientific_bundle, "selector_state", data))
    assert verify(result).check("artifact_schemas").status == "failed"

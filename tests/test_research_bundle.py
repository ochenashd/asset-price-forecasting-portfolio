"""Byte-only publication fixtures; no real model, provider or scientific claims."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import threading

import pytest

from src.research import bundle as api


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(prefix, value):
    return prefix + digest(value.encode())


@pytest.fixture
def research_bundle(tmp_path):
    root = tmp_path.resolve() / "archive"
    root.mkdir(mode=0o700)
    raw = b"PAR1synthetic-opaque-parquet-fixturePAR1"
    source = api.InputSourceReferenceV1("synthetic-input", "inputs/prices.parquet", len(raw), digest(raw), "synthetic-arrow-schema-v1")
    metadata = api.ResearchBundleMetadataV1(
        identity("apf-research-v1-", "protocol"), identity("apf-prepared-training-v1-", "training"),
        identity("apf-research-fit-receipt-v1-", "receipt"), identity("apf-fitted-model-v1-", "model"),
        identity("apf-temporal-v1-", "temporal"), digest(b"capture"), "research:SYNTH", "attempt:001", "experiment:001",
        "success", api.InputClosureV1("bundled", (source,)), "unknown", ("Synthetic byte fixtures, not real provider data.",),
        "unassessed", ("No execution evidence was supplied.",))
    artifacts = tuple(api.ResearchArtifactInput(role, spec[0], json.dumps({"fixture": role}).encode(),
        spec[1], "1", "synthetic-json-schema-v1", getattr(metadata, spec[2])) for role, spec in api.ROLE_SPECS.items() if role in api.REQUIRED_ROLES)
    artifacts += (api.ResearchArtifactInput("source_input", "inputs-root/inputs/prices.parquet", raw,
        "parquet", "1", source.schema_identity, source.artifact_id, "input-references.json"),)
    return root, artifacts, metadata


def publish(case, **kwargs):
    root, artifacts, metadata = case
    return api.publish_research_bundle(root, artifacts, metadata, **kwargs)


def snapshot(directory):
    return {str(item.relative_to(directory)): (item.read_bytes(), item.stat().st_mode, item.stat().st_mtime_ns)
            for item in directory.rglob("*") if item.is_file()}


def test_success_and_readonly_reopening_preserve_exact_supplied_bytes(research_bundle):
    root, artifacts, metadata = research_bundle
    result = publish(research_bundle)
    assert result.status == "published" and not result.diagnostics and not result.cleanup_diagnostics
    expected = api.SCHEMA_VERSION + "-" + digest(result.bundle.manifest.canonical_core().encode())
    assert result.bundle_id == expected and result.directory == root / expected
    assert result.bundle.manifest.metadata == metadata
    assert result.bundle.manifest.metadata.input_closure.status == "complete_local_bytes"
    assert result.bundle.manifest.metadata.pit_evidence_status == "unknown"
    assert result.bundle.manifest.metadata.execution_eligibility_status == "unassessed"
    before = snapshot(result.directory)
    reopened = api.open_research_bundle(result.directory, expected_bundle_id=expected)
    for item in artifacts:
        assert reopened.artifact_bytes(item.relative_path) == item.data
    assert snapshot(result.directory) == before
    assert all(not facts[1] & 0o222 for facts in before.values())
    assert not result.directory.stat().st_mode & 0o222
    assert [item.name for item in root.iterdir()] == [expected]


def test_determinism_roundtrip_inventory_order_and_mutability(research_bundle):
    _, artifacts, metadata = research_bundle
    a = api.build_research_manifest(artifacts, metadata)
    b = api.build_research_manifest(tuple(reversed(artifacts)), metadata)
    assert a.bundle_id == b.bundle_id and a.canonical_json() == b.canonical_json()
    assert api.ResearchBundleManifestV1.from_json(a.canonical_json()) == a
    assert "bundle_id" not in json.loads(a.canonical_core())
    assert api.MANIFEST_NAME not in [item.relative_path for item in a.artifacts]
    with pytest.raises(FrozenInstanceError):
        a.schema_version = "changed"


def test_actual_byte_change_and_declared_digest_change_both_change_id(research_bundle):
    _, artifacts, metadata = research_bundle
    first = api.build_research_manifest(artifacts, metadata)
    changed = (replace(artifacts[0], data=b'{"fixture":"changed"}'),) + artifacts[1:]
    assert first.bundle_id != api.build_research_manifest(changed, metadata).bundle_id
    entries = tuple(replace(item, sha256="0" * 64) if item.role == "protocol" else item for item in first.artifacts)
    assert replace(first, artifacts=entries).bundle_id != first.bundle_id


@pytest.mark.parametrize("field", ["protocol_id", "training_input_id", "receipt_id", "fitted_model_id", "temporal_manifest_id", "prediction_capture_fingerprint", "attempt_id"])
def test_scientific_identity_is_part_of_core(research_bundle, field):
    _, artifacts, metadata = research_bundle
    original = getattr(metadata, field)
    changed = (original[:-64] + digest(b"changed")) if field != "attempt_id" else "attempt:002"
    updated = replace(metadata, **{field: changed})
    inputs = tuple(replace(item, related_identity=changed) if item.role in api.ROLE_SPECS
                   and api.ROLE_SPECS[item.role][2] == field else item for item in artifacts)
    assert api.build_research_manifest(inputs, updated).bundle_id != api.build_research_manifest(artifacts, metadata).bundle_id


@pytest.mark.parametrize("path", ["../x.parquet", "/tmp/x.parquet", "inputs/../x.parquet", "inputs//x.parquet", "inputs/./x.parquet", "C:\\x.parquet", "//server/x.parquet", "https://host/x.parquet", "inputs/%2e%2e/x.parquet"])
def test_unsafe_source_paths_are_rejected(research_bundle, path):
    reference = research_bundle[2].input_closure.sources[0]
    with pytest.raises(api.BundleValidationError):
        replace(reference, protocol_relative_path=path)


@pytest.mark.parametrize("change", ["duplicate_path", "duplicate_role", "missing", "unknown_role", "executable_format", "wrong_path", "parent_cycle", "missing_parent"])
def test_inventory_and_role_allowlist_are_strict(research_bundle, change):
    _, artifacts, metadata = research_bundle
    with pytest.raises(api.BundleValidationError):
        if change in {"duplicate_path", "duplicate_role"}:
            api.build_research_manifest(artifacts + (artifacts[0],), metadata)
        elif change == "missing":
            api.build_research_manifest(artifacts[1:], metadata)
        elif change == "unknown_role":
            replace(artifacts[0], role="arbitrary_model")
        elif change == "executable_format":
            replace(artifacts[0], format="pickle")
        elif change == "wrong_path":
            replace(artifacts[0], relative_path="scripts/model.py")
        elif change == "parent_cycle":
            api.build_research_manifest((replace(artifacts[0], parent_artifact=artifacts[0].relative_path),) + artifacts[1:], metadata)
        else:
            api.build_research_manifest((replace(artifacts[0], parent_artifact="absent.json"),) + artifacts[1:], metadata)


def test_refs_only_closure_is_incomplete_and_never_fetches(research_bundle):
    root, artifacts, metadata = research_bundle
    metadata = replace(metadata, input_closure=replace(metadata.input_closure, mode="references_only"))
    artifacts = tuple(item for item in artifacts if item.role != "source_input")
    result = api.publish_research_bundle(root, artifacts, metadata)
    assert result.status == "published"
    assert result.bundle.manifest.metadata.input_closure.status == "incomplete"
    assert not (result.directory / "inputs-root").exists()


def test_external_verification_is_an_explicit_caller_declaration(research_bundle):
    root, artifacts, metadata = research_bundle
    ref = metadata.input_closure.sources[0]
    external = api.ExternalInputReferenceV1("store:fixture", "sha256/" + ref.sha256, "verification:fixture", True)
    metadata = replace(metadata, input_closure=api.InputClosureV1("verified_external", (replace(ref, external=external),)))
    result = api.publish_research_bundle(root, tuple(item for item in artifacts if item.role != "source_input"), metadata)
    assert result.status == "published"
    assert result.bundle.manifest.metadata.input_closure.status == "external_verification_declared"
    assert result.bundle.manifest.metadata.execution_eligibility_status == "unassessed"


@pytest.mark.parametrize("change", ["missing_source", "wrong_digest", "wrong_path", "wrong_schema", "undeclared_extra"])
def test_included_source_bytes_must_match_declared_original_paths(research_bundle, change):
    _, artifacts, metadata = research_bundle
    source = next(item for item in artifacts if item.role == "source_input")
    if change == "missing_source":
        artifacts = tuple(item for item in artifacts if item is not source)
    else:
        updated = replace(source, **{"wrong_digest": {"data": source.data + b"x"},
            "wrong_path": {"relative_path": "inputs-root/other.parquet"},
            "wrong_schema": {"schema_identity": "other-schema"},
            "undeclared_extra": {"relative_path": "inputs-root/extra.parquet"}}[change])
        artifacts = artifacts + (updated,) if change == "undeclared_extra" else tuple(updated if item is source else item for item in artifacts)
    with pytest.raises(api.BundleValidationError):
        api.build_research_manifest(artifacts, metadata)


def test_identical_second_publication_is_verified_without_overwrite(research_bundle):
    first = publish(research_bundle)
    before = snapshot(first.directory)
    second = publish(research_bundle)
    assert second.status == "already_exists" and second.bundle.manifest.bundle_id == first.bundle_id
    assert snapshot(first.directory) == before


@pytest.mark.parametrize("kind", ["empty_directory", "incomplete_directory", "file", "symlink"])
def test_exclusive_kernel_publish_never_replaces_conflicting_destination(research_bundle, kind, tmp_path):
    root, artifacts, metadata = research_bundle
    identifier = api.build_research_manifest(artifacts, metadata).bundle_id
    destination = root / identifier
    if kind in {"empty_directory", "incomplete_directory"}:
        destination.mkdir()
        if kind == "incomplete_directory":
            (destination / "unrelated.txt").write_bytes(b"preserve")
    elif kind == "file":
        destination.write_bytes(b"preserve")
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        destination.symlink_to(outside, target_is_directory=True)
    facts = destination.lstat()
    result = publish(research_bundle)
    assert result.status == "conflict"
    assert destination.lstat().st_ino == facts.st_ino
    if kind == "incomplete_directory":
        assert (destination / "unrelated.txt").read_bytes() == b"preserve"


def test_concurrent_publication_uses_explicit_ownership_not_timing(research_bundle, monkeypatch):
    original = api._rename_noreplace
    entered, release = threading.Event(), threading.Event()
    def hold(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)
    monkeypatch.setattr(api, "_rename_noreplace", hold)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(publish, research_bundle)
        try:
            assert entered.wait(10)
            second = pool.submit(publish, research_bundle).result(timeout=10)
            assert second.status == "publication_busy"
        finally:
            release.set()
        assert first.result(timeout=10).status == "published"
    assert publish(research_bundle).status == "already_exists"


def test_stale_lock_is_never_deleted(research_bundle):
    root, artifacts, metadata = research_bundle
    manifest = api.build_research_manifest(artifacts, metadata)
    lock = root / (api.LOCK_PREFIX + manifest.bundle_id)
    lock.mkdir()
    assert publish(research_bundle).status == "publication_busy"
    assert list(root.iterdir()) == [lock]


@pytest.mark.parametrize("phase", ["artifact_write", "manifest_write", "file_sync", "directory_sync", "rename"])
def test_failure_before_commit_leaves_no_publication(research_bundle, monkeypatch, phase):
    original = api._write_exclusive
    def fail(*args):
        raise OSError("Injected write or sync error")
    if phase in {"artifact_write", "manifest_write"}:
        def write(fd, path, data):
            if (path == api.MANIFEST_NAME) == (phase == "manifest_write"):
                fail()
            return original(fd, path, data)
        monkeypatch.setattr(api, "_write_exclusive", write)
    else:
        monkeypatch.setattr(api, {"file_sync": "_sync_file", "directory_sync": "_sync_directory", "rename": "_rename_noreplace"}[phase], fail)
    result = publish(research_bundle)
    assert result.status == "not_published" and result.diagnostics
    assert list(research_bundle[0].iterdir()) == []


def test_staged_hash_mismatch_blocks_commit(research_bundle, monkeypatch):
    original = api._write_exclusive
    def tamper(fd, path, data):
        if path == "protocol.json":
            data = data.replace(b"fixture", b"Fixture")
        return original(fd, path, data)
    monkeypatch.setattr(api, "_write_exclusive", tamper)
    result = publish(research_bundle)
    assert result.status == "not_published"
    assert result.diagnostics[0].code == "artifact_digest_mismatch"
    assert not list(research_bundle[0].iterdir())


def test_post_rename_sync_failure_is_visible_uncertain_and_never_retried(research_bundle, monkeypatch):
    rename, sync = api._rename_noreplace, api._sync_directory
    committed = []
    def commit(*args):
        rename(*args)
        committed.append(True)
    def fail_after(fd):
        if committed:
            raise OSError("Injected post-publication sync failure")
        return sync(fd)
    monkeypatch.setattr(api, "_rename_noreplace", commit)
    monkeypatch.setattr(api, "_sync_directory", fail_after)
    result = publish(research_bundle)
    assert result.status == "published_durability_uncertain" and result.directory.is_dir()
    assert len(committed) == 1
    assert api.open_research_bundle(result.directory, expected_bundle_id=result.bundle_id).manifest.bundle_id == result.bundle_id


def test_cleanup_error_does_not_mask_primary_failure(research_bundle, monkeypatch):
    def fail_write(*args):
        raise OSError("Primary write failure")
    def fail_cleanup(*args):
        raise PermissionError("Cleanup failure")
    monkeypatch.setattr(api, "_write_exclusive", fail_write)
    monkeypatch.setattr(api, "_cleanup_stage", fail_cleanup)
    result = publish(research_bundle)
    assert result.status == "not_published" and result.diagnostics[0].error_type == "OSError"
    assert result.cleanup_diagnostics[0].error_type == "PermissionError"


def test_staging_remnants_are_not_publications(research_bundle):
    root, _, _ = research_bundle
    remnant = root / (api.STAGING_PREFIX + "crashed")
    remnant.mkdir()
    (remnant / "partial.json").write_bytes(b"{}")
    result = publish(research_bundle)
    assert result.status == "published" and remnant.is_dir()
    with pytest.raises(api.BundleValidationError):
        api.open_research_bundle(remnant, expected_bundle_id=result.bundle_id)


def test_relocation_preserves_identity_and_read_images_are_owned(research_bundle, tmp_path):
    result = publish(research_bundle)
    other = tmp_path / "relocated" / result.bundle_id
    shutil.copytree(result.directory, other)
    read = api.open_research_bundle(other, expected_bundle_id=result.bundle_id)
    assert read.manifest.canonical_json() == result.bundle.manifest.canonical_json()
    original = read.artifact_bytes("protocol.json")
    path = other / "protocol.json"
    path.chmod(0o600)
    path.write_bytes(b"changed")
    assert read.artifact_bytes("protocol.json") == original
    with pytest.raises(api.BundleValidationError):
        api.open_research_bundle(other, expected_bundle_id=result.bundle_id)


@pytest.mark.parametrize("limit", ["max_artifacts", "max_artifact_bytes", "max_total_bytes", "max_manifest_bytes"])
def test_resource_limits_reject_before_writing(research_bundle, limit):
    result = publish(research_bundle, limits=replace(api.BundleLimits(), **{limit: 1}))
    assert result.status == "not_published" and not list(research_bundle[0].iterdir())


@pytest.mark.parametrize("change", ["version", "secret", "extra_core", "wrong_id", "whitespace", "nonfinite", "duplicate_key"])
def test_strict_manifest_parsing_and_structured_metadata(research_bundle, change):
    _, artifacts, metadata = research_bundle
    text = api.build_research_manifest(artifacts, metadata).canonical_json()
    data = json.loads(text)
    if change == "version":
        data["core"]["schema_version"] = "apf-research-bundle-v99"
    elif change == "secret":
        data["core"]["metadata"]["trusted_keys"] = {"do-not-render": "test-secret-value"}
    elif change == "extra_core":
        data["core"]["publication_timestamp"] = "unstable"
    elif change == "wrong_id":
        data["bundle_id"] = "wrong"
    elif change == "whitespace":
        text += "\n"
    elif change == "nonfinite":
        text = '{"core":NaN}'
    else:
        text = '{"core":{},"core":{}}'
    if change in {"version", "secret", "extra_core", "wrong_id"}:
        text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    with pytest.raises(api.BundleValidationError) as error:
        api.ResearchBundleManifestV1.from_json(text)
    assert "test-secret-value" not in str(error.value)


def test_archive_and_staging_symlink_escapes_are_rejected(research_bundle, tmp_path, monkeypatch):
    root, artifacts, metadata = research_bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)
    assert api.publish_research_bundle(linked, artifacts, metadata).status == "not_published"
    write = api._write_exclusive
    def escape(fd, path, data):
        if path == "states/selector.state.json":
            os.symlink(outside, "states", target_is_directory=True, dir_fd=fd)
        return write(fd, path, data)
    monkeypatch.setattr(api, "_write_exclusive", escape)
    assert publish(research_bundle).status == "not_published"
    assert not list(outside.iterdir()) and not list(root.iterdir())


def test_unsupported_exclusive_publication_fails_closed(research_bundle, monkeypatch):
    monkeypatch.setattr(api.sys, "platform", "unsupported")
    result = publish(research_bundle)
    assert result.status == "not_published" and result.diagnostics[0].code == "unsupported_noreplace"
    assert not list(research_bundle[0].iterdir())


def test_no_model_serialization_loading_fitting_or_external_operations(research_bundle, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("pickle", "joblib", "sklearn", "xgboost", "pandas", "mlflow", "sqlalchemy", "psycopg", "src.models", "src.pipeline"))
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("Network forbidden"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network forbidden"))
    result = publish(research_bundle)
    assert result.status == "published"
    assert api.open_research_bundle(result.directory, expected_bundle_id=result.bundle_id).manifest == result.bundle.manifest


def test_non_bytes_inputs_and_unearned_certification_are_rejected(research_bundle):
    _, artifacts, metadata = research_bundle
    with pytest.raises(api.BundleValidationError):
        replace(artifacts[0], data=bytearray(b"{}"))
    with pytest.raises(api.BundleValidationError):
        replace(metadata, execution_eligibility_status="eligible")
    with pytest.raises(api.BundleValidationError):
        replace(metadata, pit_evidence_status="verified")


@pytest.mark.parametrize("field", ["password", "hmac_key", "database_url", "Authorization", "tracking_uri"])
def test_known_structured_secret_keys_in_json_are_rejected_without_echo(research_bundle, field):
    root, artifacts, metadata = research_bundle
    secret = "test-private-value-not-for-diagnostics"
    changed = (replace(artifacts[0], data=json.dumps({field: secret}).encode()),) + artifacts[1:]
    result = api.publish_research_bundle(root, changed, metadata)
    assert result.status == "not_published" and result.diagnostics[0].code == "sensitive_metadata_field"
    assert secret not in str(result.diagnostics) and not list(root.iterdir())


@pytest.mark.parametrize("change", ["hash", "missing", "extra", "symlink", "hardlink"])
def test_existing_bundle_is_fully_checked_before_already_exists(research_bundle, tmp_path, change):
    first = publish(research_bundle)
    directory = first.directory
    target = directory / "protocol.json"
    directory.chmod(0o700)
    if change == "hash":
        target.chmod(0o600)
        target.write_bytes(target.read_bytes().replace(b"fixture", b"Fixture"))
    elif change == "missing":
        target.unlink()
    elif change == "extra":
        (directory / "unexpected-empty-directory").mkdir()
    else:
        external = tmp_path / "outside.json"
        external.write_bytes(target.read_bytes())
        target.unlink()
        if change == "symlink":
            target.symlink_to(external)
        else:
            os.link(external, target)
    directory.chmod(0o500)
    with pytest.raises((api.BundleValidationError, OSError)):
        api.open_research_bundle(directory, expected_bundle_id=first.bundle_id)
    assert publish(research_bundle).status == "conflict"


def test_manifest_parser_limits_are_enforced_without_model_parsing(research_bundle):
    _, artifacts, metadata = research_bundle
    text = api.build_research_manifest(artifacts, metadata).canonical_json()
    with pytest.raises(api.BundleValidationError):
        api.ResearchBundleManifestV1.from_json(text, limits=api.BundleLimits(max_manifest_bytes=1))
    with pytest.raises(api.BundleValidationError):
        api.ResearchBundleManifestV1.from_json(text, limits=api.BundleLimits(max_artifacts=1))


def test_cleanup_lock_error_cannot_turn_a_complete_publication_into_failure(research_bundle, monkeypatch):
    remove = os.rmdir
    def cleanup_error(path, *args, **kwargs):
        if str(path).startswith(api.LOCK_PREFIX):
            raise PermissionError("Injected lock cleanup error")
        return remove(path, *args, **kwargs)
    monkeypatch.setattr(os, "rmdir", cleanup_error)
    result = publish(research_bundle)
    assert result.status == "published" and result.bundle is not None
    assert result.cleanup_diagnostics[0].code == "lock_cleanup_failed"


def test_non_byte_object_is_never_implicitly_pickled(research_bundle):
    class UnsafeObject:
        def __reduce__(self):
            pytest.fail("Object serialization is forbidden")
    with pytest.raises(api.BundleValidationError):
        replace(research_bundle[1][0], data=UnsafeObject())


@pytest.mark.parametrize("field,value", [("run_status", []), ("pit_evidence_status", {}), ("execution_eligibility_status", True)])
def test_ambiguous_typed_metadata_is_rejected(research_bundle, field, value):
    with pytest.raises(api.BundleValidationError):
        replace(research_bundle[2], **{field: value})


def test_incomplete_status_cannot_be_overridden_in_serialized_core(research_bundle):
    _, artifacts, metadata = research_bundle
    metadata = replace(metadata, input_closure=replace(metadata.input_closure, mode="references_only"))
    manifest = api.build_research_manifest(tuple(item for item in artifacts if item.role != "source_input"), metadata)
    value = json.loads(manifest.canonical_json())
    value["core"]["input_closure_status"] = "complete_local_bytes"
    with pytest.raises(api.BundleValidationError):
        api.ResearchBundleManifestV1.from_json(json.dumps(value, sort_keys=True, separators=(",", ":")))


def test_commit_then_error_is_uncertain_not_falsely_not_published(research_bundle, monkeypatch):
    commit = api._rename_noreplace
    calls = []
    def uncertain(*args):
        commit(*args)
        calls.append(True)
        raise OSError("Injected error after actual kernel commit")
    monkeypatch.setattr(api, "_rename_noreplace", uncertain)
    result = publish(research_bundle)
    assert result.status == "published_durability_uncertain" and len(calls) == 1
    assert api.open_research_bundle(result.directory, expected_bundle_id=result.bundle_id).manifest.bundle_id == result.bundle_id


def test_post_commit_verification_failure_never_rolls_back(research_bundle, monkeypatch):
    read = api._read_bundle
    def unavailable(fd, directory, expected, limits):
        if Path(directory).name == expected:
            raise api.BundleValidationError("final_verification_unavailable", "publication")
        return read(fd, directory, expected, limits)
    monkeypatch.setattr(api, "_read_bundle", unavailable)
    result = publish(research_bundle)
    assert result.status == "published_durability_uncertain" and result.directory.is_dir()
    assert result.diagnostics[0].code == "final_verification_unavailable"


def test_replaced_lock_is_not_removed_by_former_owner(research_bundle, monkeypatch):
    cleanup = api._cleanup_stage
    def replaced(root_fd, name, identity):
        cleanup(root_fd, name, identity)
        lock = next(item for item in os.listdir(root_fd) if item.startswith(api.LOCK_PREFIX))
        os.mkdir("other-owner", dir_fd=root_fd)
        os.rmdir(lock, dir_fd=root_fd)
        os.rename("other-owner", lock, src_dir_fd=root_fd, dst_dir_fd=root_fd)
    monkeypatch.setattr(api, "_cleanup_stage", replaced)
    result = publish(research_bundle)
    assert result.status == "published" and result.cleanup_diagnostics[0].code == "lock_cleanup_failed"
    assert any(item.name.startswith(api.LOCK_PREFIX) for item in research_bundle[0].iterdir())


def test_caller_inventory_mutation_does_not_change_owned_publication(research_bundle, monkeypatch):
    root, artifacts, metadata = research_bundle
    caller_inventory = list(artifacts)
    build = api.build_research_manifest
    def snapshot(*args, **kwargs):
        result = build(*args, **kwargs)
        caller_inventory.clear()
        return result
    monkeypatch.setattr(api, "build_research_manifest", snapshot)
    result = api.publish_research_bundle(root, caller_inventory, metadata)
    assert result.status == "published" and not caller_inventory
    assert len(result.bundle.artifacts) == len(artifacts)

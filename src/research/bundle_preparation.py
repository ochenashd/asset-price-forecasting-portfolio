"""Owned current-attempt bytes, followed by full-success-only eligibility.

Preparation never fits, infers, loads models or publishes to a durable archive.
The existing publisher is used only inside a private temporary verification root.
Decision-time binding, functional replay and execution certification remain separate.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import tempfile

from . import bundle as archive
from .bundle_verification import BundleVerificationReport, verify_research_bundle
from .prediction_artifacts import serialize_prediction_capture, SCHEMA_VERSION as CAPTURE_SCHEMA
from .protocol import _encode, _json, _open_contained, validate_research_artifacts
from .state_artifacts import export_fitted_state_artifacts, SCHEMA_VERSION as STATE_SCHEMA


@dataclass(frozen=True)
class CapturedResearchBundle:
    """Internal owned bytes; not a publication candidate while a run is pending."""
    ticker: str
    run_id: str
    identities: tuple[tuple[str, str], ...]
    input_closure: archive.InputClosureV1
    artifacts: tuple[archive.ResearchArtifactInput, ...] = field(repr=False)


@dataclass(frozen=True)
class PreparedResearchBundle:
    """Full-success candidate with explicit incomplete scientific dimensions.

    Eligibility applies only to this successful run, not to historical PIT or
    execution certification. Publishing and functional replay are never implicit.
    """
    captured: CapturedResearchBundle
    metadata: archive.ResearchBundleMetadataV1
    manifest: archive.ResearchBundleManifestV1
    verification: BundleVerificationReport

    @property
    def artifacts(self):
        return self.captured.artifacts

    @property
    def bundle_id(self):
        return self.manifest.bundle_id


class BundlePreparationError(ValueError):
    """Diagnostic codes only; never expose source contents or credentials."""


def _require(condition, code):
    if not condition:
        raise BundlePreparationError(code)


def capture_research_bundle(protocol, prepared, receipt, *, ticker, run_id,
                            research_root, selector, scaler, estimator, capture,
                            fresh_capture_receipt):
    """Serialize actual live objects and reverify each captured source byte image."""
    _require(type(ticker) is str and bool(ticker) and type(run_id) is str and bool(run_id), "missing_attempt_owner")
    _require(receipt.ticker == ticker and capture.ticker == ticker, "capture_ticker_mismatch")
    _require(receipt.attempt_id == fresh_capture_receipt.run_id, "capture_attempt_mismatch")
    states = export_fitted_state_artifacts(protocol, prepared, receipt, attempt_id=receipt.attempt_id,
        selector=selector, scaler=scaler, estimator=estimator, selected_features=receipt.selected_features).by_role()
    prediction = serialize_prediction_capture(capture, fitting_attempt=fresh_capture_receipt)
    _require(prediction.fingerprint == receipt.prediction_capture_fingerprint, "capture_fingerprint_mismatch")
    limits = archive.BundleLimits()
    _require(len(protocol.artifacts) + len(archive.ROLE_SPECS) <= limits.max_artifacts, "source_count_limit")
    root = Path(research_root).resolve(strict=True)
    sources, references, total = [], [], 0
    for ref in protocol.artifacts:
        _require(ref.byte_length <= limits.max_artifact_bytes, "source_size_limit")
        # Open through the established no-symlink contract; validate the bytes
        # actually captured, not a prior validation of this path's contents.
        with _open_contained(root, ref.relative_path) as source:
            data = source.read(ref.byte_length + 1)
        _require(len(data) == ref.byte_length and hashlib.sha256(data).hexdigest() == ref.sha256,
                 "captured_source_digest_mismatch")
        total += len(data)
        _require(total <= limits.max_total_bytes, "source_total_limit")
        for column in ref.columns:
            name = column.name.lower().replace("-", "_")
            _require(name not in archive.SENSITIVE_FIELDS and not name.endswith(("_password", "_secret", "_token", "_api_key")),
                     "sensitive_source_column")
        references.append(archive.InputSourceReferenceV1(ref.artifact_id, ref.relative_path, ref.byte_length, ref.sha256, ref.schema_sha256))
        sources.append(archive.ResearchArtifactInput("source_input", "inputs-root/" + ref.relative_path,
            data, "parquet", "1", ref.schema_sha256, ref.artifact_id, "input-references.json"))
    # Reuse schema, row identity, ordering, maturity and footprint validation on
    # these owned images. No subsequent access to an original source is needed.
    with tempfile.TemporaryDirectory(prefix="apf-preparation-inputs-") as temporary:
        for source, ref in zip(sources, protocol.artifacts):
            target = Path(temporary) / ref.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.data)
        validate_research_artifacts(protocol, temporary).require_valid()
    identities = dict(protocol_id=protocol.protocol_id, training_input_id=receipt.training_input_id,
        receipt_id=receipt.receipt_id, fitted_model_id=receipt.fitted_model_id,
        temporal_manifest_id=receipt.temporal_manifest_id, prediction_capture_fingerprint=receipt.prediction_capture_fingerprint,
        asset_id=protocol.asset_id, attempt_id=receipt.attempt_id, experiment_id=receipt.experiment_id)
    typed = lambda value: _json(_encode(value)).encode()
    images = {"protocol": protocol.canonical_json().encode(), "receipt": receipt.canonical_json().encode(),
        "input_references": typed(protocol.artifacts), "selected_features": typed(receipt.selected_features),
        "effective_training_configuration": typed(protocol.learning.effective_parameters), "environment": typed(protocol.environment),
        "temporal_manifest": receipt.temporal_manifest_json.encode(), "prediction_capture": prediction.data}
    state_roles = {"selector_state": "selector", "scaler_state": "scaler", "estimator_state": "estimator",
                   "preprocessing_binding": "preprocessing", "native_booster": "native_booster"}
    artifacts = list(sources)
    for role, (path, format_name, identity, *_) in archive.ROLE_SPECS.items():
        if role in state_roles:
            data = states[state_roles[role]].data
            schema = "xgboost-native-json-v1" if role == "native_booster" else STATE_SCHEMA
        else:
            data = images[role]
            schema = {"receipt": "apf-research-fit-receipt-v1", "temporal_manifest": "apf-temporal-manifest-v1",
                      "prediction_capture": CAPTURE_SCHEMA}.get(role, "apf-research-protocol-v1")
        artifacts.append(archive.ResearchArtifactInput(role, path, data, format_name, "1", schema, identities[identity]))
    return CapturedResearchBundle(ticker, run_id, tuple(identities.items()), archive.InputClosureV1("bundled", tuple(references)), tuple(artifacts))


def finalize_research_bundle(captured, *, ticker, run_id, attempt_id, receipt_id, run_status):
    """Verify owned images only after full success; reject other run outcomes."""
    _require(type(captured) is CapturedResearchBundle and captured.ticker == ticker and captured.run_id == run_id,
             "stale_bundle_owner")
    identities = dict(captured.identities)
    _require(identities["attempt_id"] == attempt_id and identities["receipt_id"] == receipt_id, "stale_bundle_receipt")
    _require(run_status == "success", "full_run_success_required")
    metadata = archive.ResearchBundleMetadataV1(**identities, run_status=run_status, input_closure=captured.input_closure,
        pit_evidence_status="unknown", pit_limitations=("Historical availability is caller-declared, not independently witnessed.",
            "The prediction capture codec does not bind exact session/decision timestamps."),
        execution_eligibility_status="unassessed", execution_limitations=("No authenticated market eligibility is granted by this package.",))
    manifest = archive.build_research_manifest(captured.artifacts, metadata)
    with tempfile.TemporaryDirectory(prefix="apf-preparation-verification-") as temporary:
        published = archive.publish_research_bundle(Path(temporary).resolve(strict=True), captured.artifacts, metadata)
        _require(published.status == "published" and published.bundle_id == manifest.bundle_id, "temporary_verification_publication_failed")
        report = verify_research_bundle(published.directory, manifest.bundle_id)
    required = ("artifact_integrity", "bundle_manifest", "artifact_schemas", "protocol_identity", "receipt_identity",
                "reference_consistency", "input_closure", "training_inputs", "state_envelope_hashes", "state_envelope_schema",
                "temporal_provenance", "prediction_capture")
    _require(all(report.check(name).status == "verified" for name in required), "prepared_content_verification_failed")
    _require(not any(check.status in {"failed", "operational_error", "unsupported"} for _, check in report.checks),
             "prepared_scientific_contract_failed")
    return PreparedResearchBundle(captured, metadata, manifest, report)

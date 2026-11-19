"""Bounded tracking metadata, never an MLflow client or scientific witness.

The builder reopens and verifies an accepted archive. JSON decoding validates
content only; it cannot authenticate publication or attest to actual training.
Every declared field affects record_id. linkage_key binds schema, scientific
identities, model profile and logical destination; it excludes pipeline run ID,
archive alias, publication outcome and verification presentation. Neither digest
contains a filesystem root, timestamps, network state or an MLflow-generated ID.
"""
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from src.research import bundle as archive
from src.research.bundle_publication import ResearchPublicationRecord, RESEARCH_ARCHIVE_POLICY, ACCEPTED, accepts_research_archive
from src.research.bundle_verification import VerificationLimits, _parse, verify_research_bundle, _temporal
from src.research.protocol import ResearchProtocolV1, ProtocolDiagnostic, ResearchProtocolError, _encode, _decode, _json
from src.research.receipt import ResearchFitReceiptV1

SCHEMA_VERSION = "apf-research-tracking-record-v1"
MODEL_PROFILE = "apf-xgboost-three-class-native-json-v1"
MAX_RECORD_BYTES = 32768
LIMITS = VerificationLimits(max_json_depth=12, max_json_nodes=2048)
IDENTITIES = ("protocol_id", "training_input_id", "receipt_id", "fitted_model_id", "temporal_manifest_id",
              "prediction_capture_fingerprint", "bundle_id", "asset_id", "fitting_attempt_id", "research_experiment_id")
PREFIXES = {"protocol_id": "apf-research-v1-", "training_input_id": "apf-prepared-training-v1-",
            "receipt_id": "apf-research-fit-receipt-v1-", "fitted_model_id": "apf-fitted-model-v1-",
            "temporal_manifest_id": "apf-temporal-v1-", "bundle_id": "apf-research-bundle-v1-",
            "prediction_capture_fingerprint": ""}
RUNTIME_FIELDS = frozenset({"ticker", "asset_id", "fitting_attempt_id", "pipeline_run_id", "archive_alias",
                            "tracking_destination_id", "training_status", "reporting_status"})
VERIFIED = ("artifact_integrity", "bundle_manifest", "artifact_schemas", "protocol_identity", "receipt_identity",
            "reference_consistency", "input_closure", "training_inputs", "state_envelope_hashes", "state_envelope_schema",
            "temporal_provenance", "prediction_capture")
CHECKS = tuple(sorted([(key, "verified") for key in VERIFIED] + [
    ("scientific_consistency", "incomplete"), ("prediction_decision_chronology", "incomplete"),
    ("runtime_model_loading", "not_assessed"), ("functional_reproduction", "not_assessed"),
    ("historical_point_in_time", "not_assessed"), ("execution_eligibility", "not_assessed")]))
LIMITATIONS = ("content_hashes_are_not_publisher_authentication", "historical_availability_is_caller_declared",
               "prediction_decision_clock_binding_incomplete", "training_not_independently_witnessed",
               "execution_eligibility_unassessed", "functional_replay_not_assessed")


class ResearchTrackingError(ResearchProtocolError):
    """Structured fixed codes and allowlisted field names; no input values."""


def _require(condition, code, field="record"):
    if not condition:
        raise ResearchTrackingError((ProtocolDiagnostic(code, field, "Research tracking record validation failed."),)) from None


def _identifier(value):
    return (type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}", value) is not None
            and not any(word in value.lower() for word in ("password", "secret", "token", "credential", "hmac")))


def _alias(value):
    return _identifier(value) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value) is not None


@dataclass(frozen=True)
class ResearchTrackingRecordV1:
    protocol_id: str
    training_input_id: str
    receipt_id: str
    fitted_model_id: str
    temporal_manifest_id: str
    prediction_capture_fingerprint: str
    bundle_id: str
    asset_id: str
    fitting_attempt_id: str
    research_experiment_id: str
    ticker: str
    pipeline_run_id: str
    archive_alias: str
    tracking_destination_id: str
    model_type: str
    estimator_class: str
    estimator_version: str
    model_profile: str
    training_status: str
    reporting_status: str
    publisher_outcome: str
    archive_acceptance_policy: str
    archive_status: str
    research_verification_status: str
    decision_time_evidence: str
    historical_pit_status: str
    execution_eligibility_status: str
    functional_replay_status: str
    provenance_origin: str
    verification_summary: tuple[tuple[str, str], ...]
    provenance_limitations: tuple[str, ...] = LIMITATIONS
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        for item in fields(self):
            if item.name not in {"verification_summary", "provenance_limitations"}:
                value = getattr(self, item.name)
                _require(type(value) is str and len(value) <= 192, "invalid_record_field", item.name)
        _require(self.schema_version == SCHEMA_VERSION, "unsupported_scientific_schema", "schema_version")
        for name, prefix in PREFIXES.items():
            value = getattr(self, name)
            _require(type(value) is str and re.fullmatch(re.escape(prefix) + r"[0-9a-f]{64}", value) is not None,
                     "incomplete_required_identity", name)
        for name in ("asset_id", "fitting_attempt_id", "research_experiment_id", "ticker", "pipeline_run_id"):
            _require(_identifier(getattr(self, name)), "invalid_or_sensitive_identity", name)
        for name in ("archive_alias", "tracking_destination_id"):
            _require(_alias(getattr(self, name)), "invalid_logical_locator", name)
        _require((self.model_type, self.estimator_class, self.estimator_version, self.model_profile) ==
                 ("xgboost", "xgboost.sklearn.XGBClassifier", "3.2.0", MODEL_PROFILE), "unsupported_model_profile")
        _require(self.training_status == self.reporting_status == "success", "successful_training_required")
        _require(self.publisher_outcome in {"published", "already_exists"}, "publication_not_accepted")
        _require(self.archive_acceptance_policy == RESEARCH_ARCHIVE_POLICY and self.archive_status == ACCEPTED,
                 "publication_not_accepted")
        _require((self.research_verification_status, self.decision_time_evidence, self.historical_pit_status,
                  self.execution_eligibility_status, self.functional_replay_status) ==
                 ("incomplete", "incomplete", "unknown", "unassessed", "not_assessed"), "unsupported_scientific_claim")
        _require(self.provenance_origin in {"unknown", "synthetic_fixture", "historical_provider"}, "unsupported_provenance_origin")
        _require(type(self.verification_summary) is tuple and len(self.verification_summary) == len(CHECKS)
                 and all(type(item) is tuple and len(item) == 2 and all(type(value) is str for value in item)
                         for item in self.verification_summary) and self.verification_summary == CHECKS, "invalid_verification_summary")
        _require(type(self.provenance_limitations) is tuple and all(type(item) is str for item in self.provenance_limitations)
                 and self.provenance_limitations == LIMITATIONS,
                 "unsupported_provenance_limitations")

    def canonical_json(self):
        return _json({"schema_version": self.schema_version, "fields": {
            f.name: _encode(getattr(self, f.name)) for f in fields(self) if f.name != "schema_version"}})

    @property
    def record_id(self):
        return SCHEMA_VERSION + "-" + hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def linkage_key(self):
        content = {name: _encode(getattr(self, name)) for name in IDENTITIES + ("model_profile", "tracking_destination_id")}
        return "apf-research-linkage-v1-" + hashlib.sha256(_json({"schema_version": SCHEMA_VERSION, "fields": content}).encode()).hexdigest()

    @classmethod
    def from_json(cls, value):
        _require(type(value) in (str, bytes) and len(value) <= MAX_RECORD_BYTES, "record_size_limit")
        data = value.encode() if type(value) is str else value
        _require(len(data) <= MAX_RECORD_BYTES, "record_size_limit")
        try:
            raw = _parse(data, LIMITS)
        except (ValueError, UnicodeError, RecursionError):
            _require(False, "invalid_record_json")
        _require(type(raw) is dict and set(raw) == {"schema_version", "fields"}, "unexpected_or_sensitive_metadata")
        _require(raw["schema_version"] == SCHEMA_VERSION, "unsupported_scientific_schema", "schema_version")
        names = {f.name for f in fields(cls)} - {"schema_version"}
        _require(type(raw["fields"]) is dict and set(raw["fields"]) == names, "unexpected_or_sensitive_metadata")
        def primitive(node):
            _require(type(node) is list and len(node) == 2, "unsupported_record_encoding")
            tag, val = node
            if tag == "str":
                _require(type(val) is str and len(val) <= 192, "record_field_limit")
            else:
                _require(tag == "tuple" and type(val) is list and len(val) <= 32, "unsupported_record_encoding")
                for item in val:
                    primitive(item)
        for node in raw["fields"].values():
            primitive(node)
        decoded = {name: _decode(node) for name, node in raw["fields"].items()}
        return cls(**decoded)


def build_research_tracking_record(publication, bundle_directory, *, runtime_metadata: Mapping[str, str],
                                   expected_identities: Mapping[str, str] | None = None):
    """Validate publication declarations and independently verify local bytes.

    Runtime metadata must contain exactly RUNTIME_FIELDS. Optional expected
    identities are assertions only; no missing identity is populated from them.
    No archive root, timestamps or runtime context is retained in the result.
    """
    _require(type(publication) is ResearchPublicationRecord, "missing_or_invalid_publication")
    for name in ("ticker", "run_id", "attempt_id"):
        _require(_identifier(getattr(publication, name)), "incomplete_required_identity", name)
    for name in ("protocol_id", "receipt_id", "bundle_id"):
        value = getattr(publication, name)
        _require(type(value) is str and re.fullmatch(re.escape(PREFIXES[name]) + r"[0-9a-f]{64}", value) is not None,
                 "incomplete_required_identity", name)
    for name in ("publisher_outcome", "archive_status", "acceptance_policy", "input_closure_status",
                 "temporal_evidence_completeness", "pit_evidence_status", "execution_eligibility_status"):
        value = getattr(publication, name)
        _require(type(value) is str and len(value) <= 192, "invalid_publication_metadata", name)
    _require(publication.publisher_outcome != "published_durability_uncertain", "durability_uncertainty")
    _require(publication.publisher_outcome in {"published", "already_exists"} and publication.archive_status == ACCEPTED
             and publication.acceptance_policy == RESEARCH_ARCHIVE_POLICY and publication.published_path is not None,
             "publication_not_accepted")
    _require(isinstance(publication.archive_root, Path) and isinstance(publication.published_path, Path)
             and publication.archive_root.is_absolute(), "invalid_publication_reference")
    _require(type(runtime_metadata) in (dict, MappingProxyType) and len(runtime_metadata) == len(RUNTIME_FIELDS)
             and set(runtime_metadata) == RUNTIME_FIELDS, "unexpected_or_sensitive_metadata", "runtime_metadata")
    for name, value in runtime_metadata.items():
        _require(_alias(value) if name in {"archive_alias", "tracking_destination_id"} else _identifier(value), "invalid_or_sensitive_metadata", name)
    _require(runtime_metadata["training_status"] == runtime_metadata["reporting_status"] == "success", "successful_training_required")
    _require(runtime_metadata["pipeline_run_id"] == publication.run_id and runtime_metadata["fitting_attempt_id"] == publication.attempt_id,
             "cross_attempt_mismatch")
    _require(runtime_metadata["ticker"] == publication.ticker, "cross_asset_mismatch")
    _require(type(bundle_directory) is str or isinstance(bundle_directory, Path), "missing_bundle")
    path = Path(bundle_directory)
    _require(path == publication.published_path and path == publication.archive_root / publication.bundle_id, "bundle_locator_mismatch")
    try:
        owned = archive.open_research_bundle(path, expected_bundle_id=publication.bundle_id)
    except FileNotFoundError:
        _require(False, "missing_bundle")
    except (OSError, archive.BundleValidationError):
        _require(False, "invalid_bundle_integrity")
    report = verify_research_bundle(path, publication.bundle_id)
    _require(not any(check.status == "unsupported" for _, check in report.checks), "unsupported_scientific_schema")
    _require(accepts_research_archive(report, publication.bundle_id)
             and accepts_research_archive(publication.verification, publication.bundle_id), "bundle_verification_rejected")
    _require(publication.input_closure_status == "complete_local_bytes" and publication.temporal_evidence_completeness == "incomplete"
             and publication.pit_evidence_status == "unknown" and publication.execution_eligibility_status == "unassessed",
             "unsupported_scientific_claim")
    # Parse only fixed canonical scientific contracts, never runtime objects.
    protocol = ResearchProtocolV1.from_json(owned.artifact_bytes("protocol.json").decode())
    receipt = ResearchFitReceiptV1.from_json(owned.artifact_bytes("receipt.json").decode())
    meta = owned.manifest.metadata
    temporal = _temporal(owned.artifact_bytes("temporal-manifest.json"), VerificationLimits())
    identities = {name: getattr(meta, "attempt_id" if name == "fitting_attempt_id" else "experiment_id" if name == "research_experiment_id" else name)
                  for name in IDENTITIES if name != "bundle_id"}
    identities["bundle_id"] = owned.manifest.bundle_id
    for name in IDENTITIES:
        if name == "bundle_id":
            continue
        other = getattr(receipt, "attempt_id" if name == "fitting_attempt_id" else "experiment_id" if name == "research_experiment_id" else name)
        _require(other == identities[name], "scientific_identity_mismatch", name)
    _require(protocol.protocol_id == identities["protocol_id"] == publication.protocol_id
             and receipt.receipt_id == publication.receipt_id and receipt.fitted_model_id == temporal.model_id
             and temporal.split_id == receipt.temporal_manifest_id, "scientific_identity_mismatch")
    _require(receipt.ticker == publication.ticker and receipt.asset_id == runtime_metadata["asset_id"], "cross_asset_mismatch")
    _require(receipt.attempt_id == publication.attempt_id, "cross_attempt_mismatch")
    _require(meta.run_status == "success", "successful_training_required")
    if expected_identities is not None:
        _require(type(expected_identities) in (dict, MappingProxyType) and len(expected_identities) <= len(IDENTITIES)
                 and set(expected_identities).issubset(IDENTITIES), "unexpected_or_sensitive_metadata", "expected_identities")
        for name, value in expected_identities.items():
            _require(type(value) is str and value == identities[name], "scientific_identity_mismatch", name)
    _require(protocol.learning.model_type == "xgboost" and receipt.code_dependency_provenance.package_versions.get("xgboost") == "3.2.0",
             "unsupported_model_profile")
    return ResearchTrackingRecordV1(**identities, ticker=receipt.ticker, pipeline_run_id=publication.run_id,
        archive_alias=runtime_metadata["archive_alias"], tracking_destination_id=runtime_metadata["tracking_destination_id"],
        model_type="xgboost", estimator_class="xgboost.sklearn.XGBClassifier", estimator_version="3.2.0", model_profile=MODEL_PROFILE,
        training_status="success", reporting_status="success", publisher_outcome=publication.publisher_outcome,
        archive_acceptance_policy=publication.acceptance_policy, archive_status=publication.archive_status,
        research_verification_status=report.status, decision_time_evidence=report.check("prediction_decision_chronology").status,
        historical_pit_status=report.historical_pit_status, execution_eligibility_status=report.execution_eligibility_status,
        functional_replay_status=report.check("functional_reproduction").status, provenance_origin=report.provenance_origin,
        verification_summary=tuple(sorted((name, check.status) for name, check in report.checks)))


def to_mlflow_metadata(record):
    """Pure immutable tag payload; mutable MLflow tags are a secondary index.

    No client, artifacts, metrics, model objects or authentication data is used.
    """
    _require(type(record) is ResearchTrackingRecordV1, "invalid_tracking_record")
    tags = {"apf." + f.name: getattr(record, f.name) for f in fields(record)
            if f.name not in {"verification_summary", "provenance_limitations"}}
    tags.update({"apf.record_id": record.record_id, "apf.linkage_key": record.linkage_key})
    tags.update({"apf.verification." + name: status for name, status in record.verification_summary})
    tags["apf.provenance_limitations"] = ",".join(record.provenance_limitations)
    return MappingProxyType(tags)

"""Explicit research archival, with no inference or scientific certification.

Callers supply an existing trusted archive directory and accept the named policy.
No root is created, no outcome is retried and no published artifact is modified.
The publisher's platform guarantees are preserved; Linux remains unverified here.
"""
from dataclasses import dataclass, replace
from pathlib import Path

from . import bundle as archive
from .bundle_preparation import PreparedResearchBundle
from .bundle_verification import BundleVerificationReport, verify_research_bundle

RESEARCH_ARCHIVE_POLICY = "research_archive_with_declared_limitations_v1"
ACCEPTED = "accepted_research_archive_incomplete"


@dataclass(frozen=True)
class ResearchPublicationContext:
    """Runtime-only trusted root and explicit policy; never infer either.

    Supply through ForecastingPipeline(research_publication_contexts={ticker:
    ResearchPublicationContext(absolute_existing_root, RESEARCH_ARCHIVE_POLICY)}).
    The root must already exist and satisfy the existing publisher's secure-open
    contract. This declaration does not authenticate historical market evidence.
    """
    archive_root: str | Path
    acceptance_policy: str


@dataclass(frozen=True)
class ResearchPublicationRecord:
    ticker: str
    run_id: str
    attempt_id: str
    protocol_id: str
    receipt_id: str
    bundle_id: str
    archive_root: Path
    publisher_outcome: str
    archive_status: str
    observed_path: Path | None = None
    published_path: Path | None = None
    verification: BundleVerificationReport | None = None
    input_closure_status: str = "complete_local_bytes"
    temporal_evidence_completeness: str = "incomplete"
    pit_evidence_status: str = "unknown"
    execution_eligibility_status: str = "unassessed"
    diagnostics: tuple[archive.BundleDiagnostic, ...] = ()
    cleanup_diagnostics: tuple[archive.BundleDiagnostic, ...] = ()
    acceptance_policy: str = RESEARCH_ARCHIVE_POLICY


class PublicationEligibilityError(ValueError):
    """Controlled blocker codes, without payloads or runtime contexts."""


class PublicationVerificationError(RuntimeError):
    """Unexpected post-publication error with the factual publisher outcome."""
    def __init__(self, record):
        self.record = record
        super().__init__("unexpected_publication_verification_error")


def _require(condition, code):
    if not condition:
        raise PublicationEligibilityError(code)


def accepts_research_archive(report, expected_bundle_id):
    """Accept only the documented gaps, without rewriting verifier outcomes."""
    verified = ("artifact_integrity", "bundle_manifest", "artifact_schemas", "protocol_identity", "receipt_identity",
                "reference_consistency", "input_closure", "training_inputs", "state_envelope_hashes", "state_envelope_schema",
                "temporal_provenance", "prediction_capture")
    allowed = {name: {"verified"} for name in verified}
    allowed.update(scientific_consistency={"incomplete"}, prediction_decision_chronology={"incomplete"},
                   runtime_model_loading={"not_assessed"}, functional_reproduction={"not_assessed"},
                   historical_point_in_time={"not_assessed"}, execution_eligibility={"not_assessed"})
    return (type(report) is BundleVerificationReport and report.expected_bundle_id == expected_bundle_id
            and len(report.checks) == len(allowed) and set(dict(report.checks)) == set(allowed)
            and all(check.status in allowed[name] for name, check in report.checks)
            and report.historical_pit_status == "unknown" and report.execution_eligibility_status == "unassessed")


def publish_prepared_research_bundle(candidate, context, *, ticker, run_id, attempt_id, receipt_id, run_status):
    """Publish one eligible owned image, then independently check the exact ID.

    run_status describes finalized fitting and required reporting, before the
    separate archival outcome. Incomplete scientific verification is accepted
    only by the named policy, never relabelled as verified. Functional replay is
    not invoked. A returned path is an accepted archive reference only when
    published_path is non-null; uncertain outcomes retain only observed_path.
    """
    _require(type(context) is ResearchPublicationContext, "missing_archive_context")
    _require(context.acceptance_policy == RESEARCH_ARCHIVE_POLICY, "research_archive_policy_required")
    _require(type(context.archive_root) is str or isinstance(context.archive_root, Path), "invalid_archive_root")
    root = Path(context.archive_root)
    _require(root.is_absolute() and ".." not in root.parts and root.anchor != "//", "explicit_absolute_archive_root_required")
    _require(type(candidate) is PreparedResearchBundle and run_status == "success", "successful_prepared_candidate_required")
    meta = candidate.metadata
    _require(candidate.captured.ticker == ticker and candidate.captured.run_id == run_id
             and meta.attempt_id == attempt_id and meta.receipt_id == receipt_id and meta.run_status == "success",
             "stale_publication_candidate")
    _require(meta.input_closure.status == "complete_local_bytes" and meta.pit_evidence_status == "unknown"
             and meta.execution_eligibility_status == "unassessed", "unsupported_candidate_evidence")
    try:
        rebuilt = archive.build_research_manifest(candidate.artifacts, meta)
    except archive.BundleValidationError as error:
        raise PublicationEligibilityError("candidate_inventory_rejected") from error
    _require(rebuilt.canonical_json() == candidate.manifest.canonical_json(), "candidate_bytes_changed")
    _require(accepts_research_archive(candidate.verification, candidate.bundle_id), "candidate_verification_rejected")
    outcome = archive.publish_research_bundle(root, candidate.artifacts, meta)
    statuses = {"published", "already_exists", "conflict", "publication_busy", "not_published", "published_durability_uncertain"}
    _require(outcome.status in statuses, "unknown_publisher_outcome")
    record = ResearchPublicationRecord(ticker, run_id, attempt_id, meta.protocol_id, receipt_id, candidate.bundle_id,
        root, outcome.status, "uncertain" if outcome.status == "published_durability_uncertain" else "not_accepted",
        observed_path=outcome.directory if outcome.status in {"published", "already_exists", "published_durability_uncertain"} else None,
        diagnostics=outcome.diagnostics, cleanup_diagnostics=outcome.cleanup_diagnostics)
    if outcome.status not in {"published", "already_exists", "published_durability_uncertain"}:
        return record
    expected_path = root / candidate.bundle_id
    if outcome.bundle_id != candidate.bundle_id or outcome.directory != expected_path:
        return replace(record, archive_status="verification_failed", diagnostics=record.diagnostics + (
            archive.BundleDiagnostic("published_identity_mismatch", "verification"),))
    # Reopen independently of the publisher's returned in-memory bundle. Even an
    # already-existing bundle must pass all supported scientific checks again.
    try:
        reopened = archive.open_research_bundle(expected_path, expected_bundle_id=candidate.bundle_id)
        _require(reopened.manifest.canonical_json() == candidate.manifest.canonical_json(), "published_manifest_mismatch")
        report = verify_research_bundle(expected_path, candidate.bundle_id)
    except (OSError, archive.BundleValidationError, PublicationEligibilityError) as error:
        return replace(record, archive_status="uncertain" if outcome.status == "published_durability_uncertain" else "verification_failed",
            diagnostics=record.diagnostics + (archive.BundleDiagnostic("published_reopen_failed", "verification", type(error).__name__),))
    except Exception as error:
        raise PublicationVerificationError(replace(record, archive_status="error", diagnostics=record.diagnostics + (
            archive.BundleDiagnostic("unexpected_verification_error", "verification", type(error).__name__),))) from error
    record = replace(record, verification=report)
    if outcome.status == "published_durability_uncertain":
        return record
    if not accepts_research_archive(report, candidate.bundle_id):
        return replace(record, archive_status="verification_failed", diagnostics=record.diagnostics + (
            archive.BundleDiagnostic("archive_verification_rejected", "verification"),))
    return replace(record, archive_status=ACCEPTED, published_path=expected_path)

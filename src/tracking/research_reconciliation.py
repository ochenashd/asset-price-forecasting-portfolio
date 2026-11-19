"""Bounded read-only reconciliation; never repairs, creates runs or locks.

Archive presence is not publication history. Journal records are durable caller
claims, not publisher authentication. Remote tags are mutable observations.
The existing verifier may use disposable private scratch files; inspected roots
are never written. Concurrent stability checks are not a global transaction.
"""
from dataclasses import asdict, dataclass, field, replace
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace
from typing import Protocol

from src.research.bundle import _open_root, BundleValidationError, open_research_bundle
from src.research.bundle_verification import VerificationLimits, verify_research_bundle, _parse
from src.research.bundle_publication import accepts_research_archive
from src.research.receipt import ResearchFitReceiptV1
from .research_record import ResearchTrackingRecordV1, IDENTITIES, MODEL_PROFILE, _alias
from .research_linkage_journal import LinkageIntent, MAX_BYTES, metadata_payload
from .research_mlflow import BoundMlflowClient, TrackingDestination

BUNDLE = re.compile(r"apf-research-bundle-v1-[0-9a-f]{64}\Z")
KEY = re.compile(r"apf-research-linkage-v1-[0-9a-f]{64}\.json\Z")
RUN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class ResearchReferenceReader(Protocol):
    """The only remote capabilities used by reconciliation."""
    def search_runs(self, **kwargs): ...
    def get_run(self, run_id): ...


@dataclass(frozen=True)
class ReconciliationLimits:
    max_inventory_entries: int = 256
    max_bundles: int = 32
    max_journals: int = 128
    max_search_pages: int = 128
    max_remote_runs: int = 256
    verification: VerificationLimits = VerificationLimits()

    def __post_init__(self):
        if any(type(getattr(self, name)) is not int or getattr(self, name) <= 0 for name in (
                "max_inventory_entries", "max_bundles", "max_journals", "max_search_pages", "max_remote_runs")):
            raise ValueError("positive_reconciliation_limits_required")
        if type(self.verification) is not VerificationLimits:
            raise ValueError("invalid_verification_limits")


@dataclass(frozen=True)
class Finding:
    status: str
    code: str


@dataclass(frozen=True)
class JournalSnapshot:
    schema_version: str
    revision: int
    destination_id: str
    experiment_id: str
    record_id: str
    linkage_key: str
    bundle_id: str
    fitting_attempt_id: str
    status: str
    phase: str
    mlflow_run_id: str | None
    create_may_exist: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class ReconciledReference:
    bundle_id: str | None
    linkage_key: str | None
    record_id: str | None
    journal: JournalSnapshot | None
    remote_run_ids: tuple[str, ...]
    scientific_checks: tuple[tuple[str, str], ...]
    publication_history: str
    publisher_outcome: str | None
    findings: tuple[Finding, ...]
    recommendation: str = "Inspect evidence; recovery requires separate authorization. Never infer permission to create a run."


@dataclass(frozen=True)
class ReconciliationReport:
    archive_alias: str
    destination_id: str
    experiment_id: str
    remote_evidence: str
    references: tuple[ReconciledReference, ...]
    scope_findings: tuple[Finding, ...] = ()
    limitations: tuple[str, ...] = (
        "Hashes do not authenticate publishers, witness training or prove publication fsync history.",
        "Journal absence does not prove tracking was never attempted.",
        "A valid journal snapshot cannot establish whether the last replacement directory fsync completed.",
        "Read-only stability checks cannot provide a globally atomic archive/journal/remote snapshot.",
        "Remote tags and offline snapshots are mutable or caller-supplied evidence.",
        "Historical PIT remains unknown; execution eligibility remains unassessed.",
        "No automatic recovery, remote creation, journal update, model loading or inference is performed.",
    )

    @property
    def exit_code(self):
        states = {v.status for v in self.scope_findings}
        states.update(v.status for row in self.references for v in row.findings)
        if "operational_error" in states:
            return 4
        if states & {"missing_linkage_intent", "pending", "linkage_uncertain", "duplicate", "conflict",
                     "stale_linkage", "invalid_archive", "durability_uncertain"}:
            return 2
        if "unsupported" in states:
            return 3
        return 0

    def to_dict(self):
        return {**asdict(self), "exit_code": self.exit_code, "schema_version": "apf-research-reconciliation-v1"}


class _InspectionError(ValueError):
    pass


def _facts(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _inventory(root, pattern, limits):
    fd = _open_root(Path(root))
    try:
        found = {}
        with os.scandir(fd) as entries:
            for i, item in enumerate(entries):
                if i >= limits.max_inventory_entries:
                    raise _InspectionError("inventory_limit")
                facts = os.stat(item.name, dir_fd=fd, follow_symlinks=False)
                candidate = (pattern.fullmatch(item.name) or pattern is BUNDLE and item.name.startswith("apf-research-bundle-")
                    or pattern is KEY and item.name.startswith("apf-research-linkage-") and item.name.endswith(".json"))
                if candidate:
                    found[item.name] = _facts(facts)
                elif not (stat.S_ISREG(facts.st_mode) or stat.S_ISDIR(facts.st_mode)):
                    raise _InspectionError("unsafe_scope_entry")
        return found
    finally:
        os.close(fd)


def _read_entry(root, name, maximum):
    """No lock creation; owned bounded bytes plus before/after inode checks."""
    root_fd = _open_root(Path(root))
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
                raise _InspectionError("unsafe_or_oversized_journal")
            data = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if len(data) > maximum or _facts(before) != _facts(after) or _facts(after) != _facts(named):
            raise _InspectionError("unstable_journal_snapshot")
        return data, _facts(after)
    finally:
        os.close(root_fd)


def read_offline_remote_snapshot(path, *, maximum_bytes=2 * 1024 * 1024):
    """Explicit data-only snapshot; never authenticated/current remote evidence.

    Schema: {schema_version: 'apf-research-remote-snapshot-v1', destination_id,
    experiment_id, runs: [{run_id, experiment_id, lifecycle_stage, tags}]}.
    Non-APF tags are ignored and never emitted. No credentials/URI are accepted.
    """
    path = Path(path)
    data, _ = _read_entry(path.parent, path.name, maximum_bytes)
    raw = _parse(data, VerificationLimits(max_json_depth=12, max_json_nodes=20000))
    if (type(raw) is not dict or set(raw) != {"schema_version", "destination_id", "experiment_id", "runs"}
            or raw["schema_version"] != "apf-research-remote-snapshot-v1" or type(raw["runs"]) is not list
            or len(raw["runs"]) > 256):
        raise ValueError("unsupported_remote_snapshot")
    destination = TrackingDestination(raw["destination_id"], raw["experiment_id"])
    runs = []
    for item in raw["runs"]:
        if type(item) is not dict or set(item) != {"run_id", "experiment_id", "lifecycle_stage", "tags"}:
            raise ValueError("invalid_remote_snapshot")
        row = SimpleNamespace(info=SimpleNamespace(**{k: item[k] for k in ("run_id", "experiment_id", "lifecycle_stage")}),
                              data=SimpleNamespace(tags=item["tags"]))
        _remote_metadata(row, destination)
        runs.append(row)
    if len({v.info.run_id for v in runs}) != len(runs):
        raise ValueError("duplicate_snapshot_identity")
    return OfflineResearchSnapshot(destination, tuple(runs))


@dataclass(frozen=True)
class OfflineResearchSnapshot:
    destination: TrackingDestination
    runs: tuple = field(repr=False)


def _remote_metadata(run, destination):
    info, data = getattr(run, "info", None), getattr(run, "data", None)
    identity = getattr(info, "run_id", None)
    tags = getattr(data, "tags", None)
    if (type(identity) is not str or RUN.fullmatch(identity) is None
            or getattr(info, "experiment_id", None) != destination.experiment_id
            or getattr(info, "lifecycle_stage", None) not in {"active", "deleted"}
            or type(tags) is not dict or len(tags) > 512
            or any(type(k) is not str or len(k) > 192 or type(v) is not str or len(v) > 4096 for k, v in tags.items())):
        raise _InspectionError("invalid_remote_response")
    return identity, {k: v for k, v in tags.items() if k.startswith("apf.")}, info.lifecycle_stage


class _RemoteInspection:
    def __init__(self, client, destination, offline, limits):
        self.client, self.destination, self.offline, self.limits = client, destination, offline, limits
        self.ids, self.pages, self.errors, self.reads, self.missing = set(), 0, [], {}, set()

    def search(self, field, value):
        if self.offline is not None:
            return {run.info.run_id for run in self.offline.runs if run.data.tags.get("apf." + field) == value}
        if self.client is None:
            return set()
        ids, tokens, token = set(), set(), None
        try:
            while True:
                self.pages += 1
                if self.pages > self.limits.max_search_pages:
                    raise _InspectionError("remote_page_limit")
                page = self.client.client.search_runs(experiment_ids=[self.destination.experiment_id],
                    filter_string=f"tags.`apf.{field}` = '{value}'", run_view_type=3,
                    max_results=100, page_token=token)
                if not isinstance(page, (list, tuple)) or len(page) > 100:
                    raise _InspectionError("invalid_remote_page")
                for run in page:
                    identity, _, _ = _remote_metadata(run, self.destination)
                    ids.add(identity)
                    self.ids.add(identity)
                if len(self.ids) > self.limits.max_remote_runs:
                    raise _InspectionError("remote_run_limit")
                token = getattr(page, "token", None)
                if token is None:
                    return ids
                if type(token) is not str or not 0 < len(token) <= 4096 or token in tokens:
                    raise _InspectionError("invalid_remote_page_token")
                tokens.add(token)
        except Exception:
            self.errors.append(Finding("operational_error", "remote_search_unavailable_or_limited"))
            return ids

    def read(self, identity):
        if identity in self.reads:
            return self.reads[identity]
        if len(self.reads) >= self.limits.max_remote_runs:
            self.errors.append(Finding("operational_error", "remote_readback_limit"))
            return None
        self.reads[identity] = None
        try:
            if self.offline is not None:
                run = next(v for v in self.offline.runs if v.info.run_id == identity)
            else:
                run = self.client.client.get_run(identity)
            self.reads[identity] = _remote_metadata(run, self.destination)
        except (FileNotFoundError, KeyError, StopIteration):
            self.missing.add(identity)
        except Exception:
            self.errors.append(Finding("operational_error", "remote_readback_unavailable"))
        return self.reads[identity]


def _linkage_key(identities, destination):
    # Project verified references into the EXISTING pure property. This is not
    # a ResearchTrackingRecordV1 and makes no accepted-publication declaration.
    projection = SimpleNamespace(**identities, model_profile=MODEL_PROFILE,
                                 tracking_destination_id=destination.destination_id)
    return ResearchTrackingRecordV1.linkage_key.fget(projection)


def reconcile_research_tracking(*, archive_root, archive_alias, journal_root, destination,
        client=None, offline_snapshot=None, publication_outcomes=(), expected_revisions=(),
        limits=ReconciliationLimits()):
    """Inspect only explicit scope, with no repair or implicit remote destination.

    publication_outcomes is optional (bundle_id, outcome) caller-declared history,
    never a fabricated ResearchPublicationRecord. Unknown history stays unknown.
    expected_revisions supplies optional trusted prior revision assertions.
    """
    if type(destination) is not TrackingDestination or not _alias(archive_alias):
        raise ValueError("explicit_reconciliation_scope_required")
    if (client is not None and (type(client) is not BoundMlflowClient or client.destination != destination)
            or offline_snapshot is not None and (type(offline_snapshot) is not OfflineResearchSnapshot
            or offline_snapshot.destination != destination) or client is not None and offline_snapshot is not None):
        raise ValueError("read_only_destination_binding_mismatch")
    if type(limits) is not ReconciliationLimits:
        raise ValueError("invalid_reconciliation_limits")
    outcomes, revisions = dict(publication_outcomes), dict(expected_revisions)
    if (len(outcomes) > limits.max_bundles or any(type(k) is not str or not BUNDLE.fullmatch(k)
            or v not in {"published", "already_exists", "published_durability_uncertain", "unknown"} for k, v in outcomes.items())
            or len(revisions) > limits.max_journals or any(type(k) is not str or not KEY.fullmatch(k + ".json")
            or type(v) is not int or v < 1 for k, v in revisions.items())):
        raise ValueError("invalid_reconciliation_assertions")
    remote_mode = "injected_read_only_client" if client is not None else "offline_snapshot" if offline_snapshot is not None else "unavailable"
    scope, rows = [], []
    def report():
        return ReconciliationReport(archive_alias, destination.destination_id, destination.experiment_id, remote_mode,
            tuple(sorted(rows, key=lambda v: (v.bundle_id or "", v.linkage_key or ""))), tuple(sorted(set(scope), key=lambda v: (v.status, v.code))))
    try:
        archive_root, journal_root = Path(archive_root), Path(journal_root)
        before = _inventory(archive_root, BUNDLE, limits)
        journal_before = _inventory(journal_root, KEY, limits)
        if len(before) > limits.max_bundles or len(journal_before) > limits.max_journals:
            raise _InspectionError("inventory_limit")
    except (OSError, ValueError, TypeError):
        scope.append(Finding("operational_error", "unsafe_unavailable_or_limited_scope"))
        return report()
    journals, journal_facts = {}, {}
    for name in sorted(journal_before):
        if not KEY.fullmatch(name):
            scope.append(Finding("unsupported", "unsupported_journal_filename"))
            continue
        try:
            data, facts = _read_entry(journal_root, name, MAX_BYTES)
            raw = _parse(data, VerificationLimits(max_json_depth=12, max_json_nodes=4096))
            if type(raw) is not dict or raw.get("schema_version") != "apf-research-linkage-journal-v1":
                scope.append(Finding("unsupported", "unsupported_journal_schema"))
                continue
            entry = LinkageIntent.from_json(data)
            if name != entry.linkage_key + ".json":
                raise _InspectionError("journal_filename_mismatch")
            if entry.linkage_key in revisions and entry.revision != revisions[entry.linkage_key]:
                scope.append(Finding("conflict", "journal_revision_assertion_failed"))
            journals.setdefault(entry.bundle_id, []).append(entry)
            journal_facts[name] = facts
        except (OSError, ValueError, RuntimeError):
            scope.append(Finding("operational_error", "invalid_or_unstable_journal_snapshot"))
    remote = _RemoteInspection(client, destination, offline_snapshot, limits)
    if remote_mode != "injected_read_only_client":
        scope.append(Finding("unsupported", "current_remote_verification_unavailable"))
    alias_ids = remote.search("archive_alias", archive_alias)
    seen_remote = set()
    bundle_ids = sorted(set(before) | set(journals) | set(outcomes))
    for bundle_id in bundle_ids:
        if not BUNDLE.fullmatch(bundle_id):
            scope.append(Finding("unsupported", "unsupported_archive_candidate"))
            continue
        finds, checks, identities, receipt = [], (), None, None
        entries = sorted(journals.get(bundle_id, ()), key=lambda v: v.linkage_key)
        if not entries:
            finds.append(Finding("missing_linkage_intent", "no_matching_local_linkage_evidence"))
        if len(entries) > 1:
            finds.append(Finding("conflict", "multiple_local_records_for_bundle"))
        entry = entries[0] if entries else None
        key, record = None, None
        if bundle_id not in before:
            finds.append(Finding("stale_linkage", "expected_bundle_missing"))
        elif not stat.S_ISDIR(before[bundle_id][2]):
            finds.append(Finding("invalid_archive", "unsafe_bundle_path"))
        else:
            try:
                verified = verify_research_bundle(archive_root / bundle_id, bundle_id, limits=limits.verification)
                checks = tuple((name, check.status) for name, check in verified.checks)
                if not accepts_research_archive(verified, bundle_id):
                    status = "unsupported" if any(c.status == "unsupported" for _, c in verified.checks) else "operational_error" if verified.status == "operational_error" else "invalid_archive"
                    finds.append(Finding(status, "archive_verification_not_accepted"))
                else:
                    owned = open_research_bundle(archive_root / bundle_id, expected_bundle_id=bundle_id, limits=limits.verification.archive)
                    receipt = ResearchFitReceiptV1.from_json(owned.artifact_bytes("receipt.json").decode())
                    meta = owned.manifest.metadata
                    identities = {name: getattr(meta, "attempt_id" if name == "fitting_attempt_id" else "experiment_id" if name == "research_experiment_id" else name)
                                  for name in IDENTITIES if name != "bundle_id"}
                    identities["bundle_id"] = bundle_id
                    key = _linkage_key(identities, destination)
            except (OSError, ValueError):
                finds.append(Finding("invalid_archive", "invalid_archive_identity"))
        if entry is not None:
            record = ResearchTrackingRecordV1.from_json(entry.record_json)
            if (entry.destination_id != destination.destination_id or entry.experiment_id != destination.experiment_id
                    or record.archive_alias != archive_alias):
                finds.append(Finding("conflict", "journal_scope_mismatch"))
            if identities is not None and (any(getattr(record, name) != value for name, value in identities.items())
                    or record.ticker != receipt.ticker or entry.linkage_key != key
                    or record.verification_summary != tuple(sorted(checks))):
                finds.append(Finding("conflict", "journal_scientific_identity_mismatch"))
            if entry.create_may_exist and entry.status != "linked":
                finds.append(Finding("linkage_uncertain", "creation_may_have_occurred"))
            elif entry.status != "linked":
                finds.append(Finding("pending", "journal_not_linked"))
        history = "journal_declared_accepted" if entry is not None else "unknown"
        outcome = record.publisher_outcome if record is not None else None
        if bundle_id in outcomes:
            history, outcome = "caller_declared", outcomes[bundle_id]
        if history == "unknown" or outcome == "unknown":
            finds.append(Finding("unsupported", "historical_publication_outcome_unknown"))
        if outcome == "published_durability_uncertain":
            finds.append(Finding("durability_uncertain", "publisher_durability_not_resolved"))
        ids = remote.search("bundle_id", bundle_id)
        if key is not None:
            ids |= remote.search("linkage_key", key)
        if not ids and entry is not None and entry.create_may_exist and remote_mode != "unavailable":
            finds.append(Finding("linkage_uncertain", "zero_search_results_do_not_authorize_creation"))
        if (entry is not None and remote_mode != "unavailable" and entry.mlflow_run_id is not None
                and entry.destination_id == destination.destination_id and entry.experiment_id == destination.experiment_id):
            ids.add(entry.mlflow_run_id)
        if len(ids) > limits.max_remote_runs:
            scope.append(Finding("operational_error", "remote_run_limit"))
            ids = set(sorted(ids)[:limits.max_remote_runs])
        seen_remote.update(ids)
        if len(ids) > 1:
            finds.append(Finding("duplicate", "multiple_remote_references"))
        remote_exact = False
        for identity in sorted(ids):
            read = remote.read(identity)
            if read is None:
                finds.append(Finding("stale_linkage", "remote_reference_missing") if identity in remote.missing
                             else Finding("operational_error", "remote_reference_unassessed"))
                continue
            _, tags, lifecycle = read
            if lifecycle != "active":
                finds.append(Finding("stale_linkage", "remote_reference_inactive"))
            if record is not None:
                if tags != dict(metadata_payload(record, destination.experiment_id)):
                    finds.append(Finding("conflict", "remote_record_metadata_mismatch"))
                else:
                    remote_exact = lifecycle == "active"
            elif identities is not None:
                expected = {**identities, "ticker": receipt.ticker, "archive_alias": archive_alias,
                    "tracking_destination_id": destination.destination_id,
                    "tracking_experiment_id": destination.experiment_id,
                    "archive_acceptance_policy": "research_archive_with_declared_limitations_v1",
                    "historical_pit_status": "unknown", "execution_eligibility_status": "unassessed",
                    "decision_time_evidence": "incomplete", "functional_replay_status": "not_assessed"}
                if (any(tags.get("apf." + name) != value for name, value in expected.items())
                        or any(tags.get("apf.verification." + name) != value for name, value in checks)):
                    finds.append(Finding("conflict", "remote_scientific_identity_mismatch"))
            if key is not None and tags.get("apf.linkage_key") != key:
                finds.append(Finding("conflict", "remote_linkage_key_mismatch"))
        if not ids and remote_mode != "unavailable":
            if entry is not None and entry.create_may_exist:
                finds.append(Finding("linkage_uncertain", "zero_search_results_do_not_authorize_creation"))
            elif entry is not None and entry.status == "linked":
                finds.append(Finding("stale_linkage", "linked_remote_record_not_found"))
        if entry is None and ids:
            finds.append(Finding("missing_linkage_intent", "remote_reference_without_local_intent"))
        if (entry is not None and entry.status == "linked" and len(ids) == 1 and remote_exact
                and entry.mlflow_run_id in ids and identities is not None and not finds
                and remote_mode == "injected_read_only_client"):
            finds.append(Finding("linked_consistent", "supported_references_match"))
        if entry is not None and entry.status == "linked" and not finds and not remote_exact:
            finds.append(Finding("unsupported", "linkage_not_fully_assessed"))
        view = None if entry is None else JournalSnapshot(entry.schema_version, entry.revision, entry.destination_id,
            entry.experiment_id, entry.record_id, entry.linkage_key, entry.bundle_id, entry.fitting_attempt_id,
            entry.status, entry.phase, entry.mlflow_run_id, entry.create_may_exist, entry.diagnostics)
        rows.append(ReconciledReference(bundle_id, key or (entry.linkage_key if entry else None),
            entry.record_id if entry else None, view, tuple(sorted(ids)), checks, history, outcome,
            tuple(sorted(set(finds), key=lambda v: (v.status, v.code)))))
    for identity in sorted(alias_ids - seen_remote):
        read = remote.read(identity)
        if read is not None:
            _, tags, _ = read
            claimed = tags.get("apf.bundle_id")
            rows.append(ReconciledReference(claimed if type(claimed) is str and BUNDLE.fullmatch(claimed) else None,
                None, None, None, (identity,), (), "unknown", None,
                (Finding("stale_linkage", "orphan_remote_archive_reference"),)))
    scope.extend(remote.errors)
    try:
        if _inventory(journal_root, KEY, limits) != journal_before or any(journal_before[n] != f for n, f in journal_facts.items()):
            scope.append(Finding("operational_error", "unstable_journal_snapshot"))
        if _inventory(archive_root, BUNDLE, limits) != before:
            scope.append(Finding("operational_error", "unstable_archive_inventory"))
    except (OSError, ValueError):
        scope.append(Finding("operational_error", "scope_stability_unavailable"))
    if scope:
        rows = [replace(row, findings=tuple(v for v in row.findings if v.status != "linked_consistent")) for row in rows]
    return report()

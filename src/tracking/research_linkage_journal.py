"""Versioned local intent with POSIX locks and atomic, fsynced replacement.

The trusted absolute root must already exist. A persistent lock inode is never
removed or taken over: nonblocking flock serializes cooperating writers and the
OS releases locks on process exit. No sleeps or stale-lock heuristics are used.
File fsync precedes replace and directory fsync follows it. Success relies on
local filesystem/fsync semantics, not a claim of hardware power-loss immunity.
A failure after replacement is explicitly durability-uncertain; never rollback.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import errno
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from src.research.bundle import _open_root, BundleValidationError
from src.research.bundle_verification import _parse, VerificationLimits
from src.research.protocol import _json
from .research_record import ResearchTrackingRecordV1, to_mlflow_metadata, _alias

SCHEMA_VERSION = "apf-research-linkage-journal-v1"
MAX_BYTES = 131072
STATES = frozenset({"pending", "searching", "creating", "linkage_uncertain", "linked", "duplicate", "conflict", "failed"})
PHASES = frozenset({"intent", "search", "create", "readback", "complete"})
CODES = frozenset({"remote_search_failed", "remote_create_ambiguous", "remote_readback_failed", "remote_timeout",
    "remote_metadata_conflict", "remote_metadata_incomplete", "duplicate_remote_runs", "create_visibility_uncertain",
    "invalid_remote_response", "remote_run_unavailable"})


class JournalError(RuntimeError):
    def __init__(self, code, *, uncertain=False):
        self.code, self.uncertain = code, uncertain
        super().__init__(code)


def _require(value, code):
    if not value:
        raise JournalError(code) from None


def opaque_id(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None


def metadata_payload(record, experiment_id):
    payload = dict(to_mlflow_metadata(record))
    payload.update({"apf.linkage_schema": SCHEMA_VERSION, "apf.tracking_experiment_id": experiment_id,
                    "apf.record_sha256": hashlib.sha256(record.canonical_json().encode()).hexdigest()})
    return tuple(sorted(payload.items()))


@dataclass(frozen=True)
class LinkageIntent:
    destination_id: str
    experiment_id: str
    linkage_key: str
    record_id: str
    record_sha256: str
    bundle_id: str
    fitting_attempt_id: str
    record_json: str
    metadata_payload: tuple[tuple[str, str], ...]
    payload_sha256: str
    status: str = "pending"
    phase: str = "intent"
    mlflow_run_id: str | None = None
    create_may_exist: bool = False
    revision: int = 1
    diagnostics: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        _require(self.schema_version == SCHEMA_VERSION and _alias(self.destination_id) and opaque_id(self.experiment_id), "invalid_journal_binding")
        _require(type(self.record_json) is str and len(self.record_json) <= 32768, "journal_size_limit")
        record = ResearchTrackingRecordV1.from_json(self.record_json)
        _require(self.record_json == record.canonical_json() and self.destination_id == record.tracking_destination_id,
                 "journal_record_mismatch")
        for name in ("linkage_key", "record_id", "bundle_id", "fitting_attempt_id"):
            _require(type(getattr(self, name)) is str and getattr(self, name) == getattr(record, name), "journal_record_mismatch")
        _require(self.record_sha256 == hashlib.sha256(self.record_json.encode()).hexdigest(), "journal_record_digest_mismatch")
        _require(type(self.metadata_payload) is tuple and all(type(pair) is tuple and len(pair) == 2
                 and all(type(v) is str for v in pair) for pair in self.metadata_payload), "invalid_journal_payload")
        _require(self.metadata_payload == metadata_payload(record, self.experiment_id)
                 and self.payload_sha256 == hashlib.sha256(_json(self.metadata_payload).encode()).hexdigest(), "journal_payload_mismatch")
        _require(type(self.status) is str and self.status in STATES and type(self.phase) is str and self.phase in PHASES,
                 "invalid_journal_state")
        _require(type(self.create_may_exist) is bool and type(self.revision) is int and 0 < self.revision < 2**63,
                 "invalid_journal_revision")
        _require(self.mlflow_run_id is None or opaque_id(self.mlflow_run_id), "invalid_remote_run_identity")
        _require(self.status != "creating" or self.create_may_exist, "invalid_creation_state")
        _require(self.mlflow_run_id is None or self.create_may_exist, "invalid_creation_state")
        _require(self.status != "linked" or self.mlflow_run_id is not None, "missing_remote_run_identity")
        _require(type(self.diagnostics) is tuple and len(self.diagnostics) <= 8
                 and all(type(code) is str and code in CODES for code in self.diagnostics), "invalid_journal_diagnostics")

    def canonical_json(self):
        return _json(asdict(self))

    @classmethod
    def from_json(cls, data):
        _require(type(data) is bytes and len(data) <= MAX_BYTES, "journal_size_limit")
        try:
            raw = _parse(data, VerificationLimits(max_json_depth=12, max_json_nodes=4096))
            _require(type(raw) is dict and set(raw) == set(cls.__dataclass_fields__), "unexpected_journal_fields")
            _require(type(raw["metadata_payload"]) is list and len(raw["metadata_payload"]) <= 128
                     and all(type(p) is list and len(p) == 2 for p in raw["metadata_payload"]), "invalid_journal_payload")
            _require(type(raw["diagnostics"]) is list, "invalid_journal_diagnostics")
            raw["metadata_payload"] = tuple(tuple(pair) for pair in raw["metadata_payload"])
            raw["diagnostics"] = tuple(raw["diagnostics"])
            return cls(**raw)
        except JournalError:
            raise
        except (ValueError, TypeError, KeyError, RecursionError):
            raise JournalError("invalid_journal_content") from None


def new_intent(record, experiment_id):
    payload = metadata_payload(record, experiment_id)
    return LinkageIntent(record.tracking_destination_id, experiment_id, record.linkage_key, record.record_id,
        hashlib.sha256(record.canonical_json().encode()).hexdigest(), record.bundle_id, record.fitting_attempt_id,
        record.canonical_json(), payload, hashlib.sha256(_json(payload).encode()).hexdigest())


class LinkageJournal:
    """Explicit existing root; no implicit creation, database or archive writes."""
    def __init__(self, root):
        _require(type(root) is str or isinstance(root, Path), "explicit_journal_root_required")
        self.root = Path(root)
        _require(self.root.is_absolute() and ".." not in self.root.parts, "explicit_journal_root_required")
        _require(not any(re.fullmatch(r"apf-(?:research-)?bundle-v1-[0-9a-f]{64}", p) for p in self.root.parts), "immutable_archive_journal_forbidden")

    @contextmanager
    def transaction(self, linkage_key):
        _require(type(linkage_key) is str and re.fullmatch(r"apf-research-linkage-v1-[0-9a-f]{64}", linkage_key), "invalid_journal_key")
        root_fd = lock_fd = None
        try:
            import fcntl
            root_fd = _open_root(self.root)
            lock_fd = os.open(linkage_key + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=root_fd)
            facts = os.fstat(lock_fd)
            _require(stat.S_ISREG(facts.st_mode) and facts.st_nlink == 1, "unsafe_journal_lock")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise JournalError("journal_busy") from None
            yield _Transaction(root_fd, linkage_key)
        except (OSError, BundleValidationError, ImportError):
            raise JournalError("journal_unavailable") from None
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            if root_fd is not None:
                os.close(root_fd)


class _Transaction:
    def __init__(self, root_fd, key):
        self.root_fd, self.key = root_fd, key

    def read(self):
        try:
            fd = os.open(self.key + ".json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.root_fd)
        except FileNotFoundError:
            return None
        except OSError:
            raise JournalError("unsafe_journal_entry") from None
        with os.fdopen(fd, "rb") as stream:
            facts = os.fstat(stream.fileno())
            _require(stat.S_ISREG(facts.st_mode) and facts.st_nlink == 1 and facts.st_size <= MAX_BYTES, "unsafe_journal_entry")
            return LinkageIntent.from_json(stream.read(MAX_BYTES + 1))

    def write(self, entry, *, expected_revision):
        _require(type(entry) is LinkageIntent and entry.linkage_key == self.key, "journal_key_mismatch")
        current = self.read()
        _require((0 if current is None else current.revision) == expected_revision
                 and entry.revision == expected_revision + 1, "journal_revision_conflict")
        if current is not None:
            _require((entry.record_id, entry.destination_id, entry.experiment_id, entry.record_json) ==
                     (current.record_id, current.destination_id, current.experiment_id, current.record_json), "journal_intent_conflict")
            _require(not current.create_may_exist or entry.create_may_exist, "creation_history_cannot_be_reset")
            _require(current.mlflow_run_id is None or entry.mlflow_run_id == current.mlflow_run_id, "remote_identity_cannot_be_replaced")
        data = entry.canonical_json().encode()
        _require(len(data) <= MAX_BYTES, "journal_size_limit")
        stage = ".linkage-stage-" + uuid4().hex
        committed = replace_started = False
        operation = "write"
        try:
            fd = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.root_fd)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                operation = "file_sync"
                os.fsync(stream.fileno())
            operation = "replace"
            replace_started = True
            os.replace(stage, self.key + ".json", src_dir_fd=self.root_fd, dst_dir_fd=self.root_fd)
            committed = True
            operation = "directory_sync"
            os.fsync(self.root_fd)
        except OSError as error:
            uncertain = committed
            if replace_started and not committed:
                try:
                    os.stat(stage, dir_fd=self.root_fd, follow_symlinks=False)
                except OSError:
                    uncertain = True
            code = "journal_durability_uncertain" if uncertain else "journal_write_failed"
            # Capability diagnostics do not erase post-replacement uncertainty.
            if operation in {"file_sync", "directory_sync"} and error.errno in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                code = "journal_sync_unsupported"
            raise JournalError(code, uncertain=uncertain) from None
        finally:
            if not committed:
                try:
                    os.unlink(stage, dir_fd=self.root_fd)
                except OSError:
                    pass  # Owned temporary cleanup never hides the primary error.
        return entry

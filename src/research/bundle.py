"""Byte-only Research Bundle V1 and POSIX local, exclusive publication.

No research/model object is serialized or loaded. Schema identities and external
verification references are declarations, not independent certification. Readers
check byte integrity and declared inventory, not scientific artifact semantics.

Publication requires an existing symlink-free, application-controlled archive.
Darwin renameatx_np(RENAME_EXCL) and Linux renameat2(RENAME_NOREPLACE) are the only
commit mechanisms. Unsupported platforms/filesystems fail closed. Same-filesystem
rename provides atomic visibility; fsync durability depends on the filesystem and
hardware. Read-only owner permissions are not WORM or hostile-writer protection.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass, field, fields
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType
from uuid import uuid4


SCHEMA_VERSION = "apf-research-bundle-v1"
STAGING_PREFIX = ".research-stage-"
LOCK_PREFIX = ".research-publish-lock-"
MANIFEST_NAME = "manifest.json"
# role: (exact path, declared format, scientific identity field)
ROLE_SPECS = MappingProxyType({
    "protocol": ("protocol.json", "json", "protocol_id"),
    "receipt": ("receipt.json", "json", "receipt_id"),
    "input_references": ("input-references.json", "json", "protocol_id"),
    "selector_state": ("states/selector.state.json", "json", "fitted_model_id"),
    "scaler_state": ("states/scaler.state.json", "json", "fitted_model_id"),
    "estimator_state": ("states/estimator.state.json", "json", "fitted_model_id"),
    "preprocessing_binding": ("states/preprocessing-binding.json", "json", "fitted_model_id"),
    "selected_features": ("selected-features.json", "json", "fitted_model_id"),
    "temporal_manifest": ("temporal-manifest.json", "json", "temporal_manifest_id"),
    "prediction_capture": ("prediction-capture.json", "json", "prediction_capture_fingerprint"),
    "effective_training_configuration": ("effective-training-config.json", "json", "protocol_id"),
    "environment": ("environment.json", "json", "protocol_id"),
    "native_booster": ("models/booster.json", "json", "fitted_model_id"),
})
REQUIRED_ROLES = frozenset(ROLE_SPECS) - {"native_booster"}
SENSITIVE_FIELDS = frozenset({"password", "secret", "token", "api_key", "hmac_key", "trusted_keys", "private_key",
                              "connection_string", "database_url", "tracking_uri", "access_key", "secret_key",
                              "authorization", "credentials"})


@dataclass(frozen=True)
class BundleDiagnostic:
    code: str
    field: str
    error_type: str | None = None


class BundleValidationError(ValueError):
    def __init__(self, code, field):
        self.diagnostics = (BundleDiagnostic(code, field),)
        super().__init__(f"{code}: {field}")


def _require(value, code, field):
    if not value:
        raise BundleValidationError(code, field)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identifier(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value) is not None


def _path(value):
    _require(type(value) is str and 0 < len(value) <= 1024 and len(value.split("/")) <= 16
             and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", part) for part in value.split("/")),
             "unsafe_path", "artifact_path")
    return value


def _load_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_json_key", "json")
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            _require(normalized not in SENSITIVE_FIELDS and not normalized.endswith(("_password", "_secret", "_token", "_api_key")),
                     "sensitive_metadata_field", "json")
            result[key] = value
        return result
    def constant(value):
        raise BundleValidationError("nonfinite_json", "json")
    try:
        return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
    except BundleValidationError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise BundleValidationError("invalid_json", "json") from None


def _construct(cls, value, **overrides):
    _require(type(value) is dict and set(value) == {item.name for item in fields(cls)}, "invalid_fields", cls.__name__)
    return cls(**{**value, **overrides})


@dataclass(frozen=True)
class BundleLimits:
    max_artifacts: int = 128
    max_artifact_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_manifest_bytes: int = 2 * 1024 * 1024

    def __post_init__(self):
        _require(all(type(getattr(self, item.name)) is int and getattr(self, item.name) > 0 for item in fields(self)),
                 "invalid_limits", "limits")


@dataclass(frozen=True)
class ExternalInputReferenceV1:
    """Caller-supplied immutable-store verification reference; never fetched."""
    store_id: str
    content_key: str
    verification_reference: str
    immutability_declared: bool

    def __post_init__(self):
        _require(_identifier(self.store_id) and _identifier(self.verification_reference)
                 and self.immutability_declared is True, "invalid_external_reference", "input_closure")
        _path(self.content_key)


@dataclass(frozen=True)
class InputSourceReferenceV1:
    artifact_id: str
    protocol_relative_path: str
    byte_length: int
    sha256: str
    schema_identity: str
    external: ExternalInputReferenceV1 | None = None

    def __post_init__(self):
        _require(_identifier(self.artifact_id) and _identifier(self.schema_identity) and _hash(self.sha256)
                 and type(self.byte_length) is int and self.byte_length > 0, "invalid_source_reference", "input_closure")
        _path(self.protocol_relative_path)
        _require(self.protocol_relative_path.endswith(".parquet"), "unsupported_source_format", "input_closure")
        _require(self.external is None or type(self.external) is ExternalInputReferenceV1, "invalid_external_reference", "input_closure")


@dataclass(frozen=True)
class InputClosureV1:
    mode: str
    sources: tuple[InputSourceReferenceV1, ...]

    def __post_init__(self):
        _require(type(self.sources) in (tuple, list) and self.sources and len(self.sources) <= 128
                 and all(type(item) is InputSourceReferenceV1 for item in self.sources), "invalid_input_closure", "sources")
        object.__setattr__(self, "sources", tuple(sorted(self.sources, key=lambda item: item.protocol_relative_path)))
        _require(len({item.artifact_id for item in self.sources}) == len(self.sources)
                 and len({item.protocol_relative_path for item in self.sources}) == len(self.sources), "duplicate_source", "input_closure")
        _require(type(self.mode) is str and self.mode in {"bundled", "verified_external", "references_only"}, "invalid_input_closure", "mode")
        _require(all((item.external is not None) == (self.mode == "verified_external") for item in self.sources),
                 "inconsistent_external_reference", "input_closure")

    @property
    def status(self):
        return {"bundled": "complete_local_bytes", "verified_external": "external_verification_declared",
                "references_only": "incomplete"}[self.mode]


@dataclass(frozen=True)
class ResearchBundleMetadataV1:
    protocol_id: str
    training_input_id: str
    receipt_id: str
    fitted_model_id: str
    temporal_manifest_id: str
    prediction_capture_fingerprint: str
    asset_id: str
    attempt_id: str
    experiment_id: str
    run_status: str
    input_closure: InputClosureV1
    pit_evidence_status: str
    pit_limitations: tuple[str, ...]
    execution_eligibility_status: str
    execution_limitations: tuple[str, ...]

    def __post_init__(self):
        for name, prefix in (("protocol_id", "apf-research-v1-"), ("training_input_id", "apf-prepared-training-v1-"),
                             ("receipt_id", "apf-research-fit-receipt-v1-"), ("fitted_model_id", "apf-fitted-model-v1-"),
                             ("temporal_manifest_id", "apf-temporal-v1-")):
            value = getattr(self, name)
            _require(type(value) is str and value.startswith(prefix) and _hash(value[len(prefix):]), "invalid_identity", name)
        _require(_hash(self.prediction_capture_fingerprint) and all(_identifier(getattr(self, name)) for name in
                 ("asset_id", "attempt_id", "experiment_id")), "invalid_identity", "metadata")
        _require(type(self.input_closure) is InputClosureV1 and type(self.run_status) is str and self.run_status in {"success", "partial_success", "failed"},
                 "invalid_status", "metadata")
        _require(type(self.pit_evidence_status) is str and self.pit_evidence_status in {"unknown", "caller_declared"}
                 and type(self.execution_eligibility_status) is str and self.execution_eligibility_status == "unassessed",
                 "unsupported_certification", "metadata")
        for name in ("pit_limitations", "execution_limitations"):
            value = getattr(self, name)
            _require(type(value) in (tuple, list) and value and len(value) <= 32
                     and all(type(item) is str and 0 < len(item) <= 4096 and "\x00" not in item for item in value),
                     "missing_limitations", name)
            object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class ResearchArtifactEntryV1:
    role: str
    relative_path: str
    byte_length: int
    sha256: str
    format: str
    format_version: str
    schema_identity: str
    related_identity: str
    parent_artifact: str | None = None

    def __post_init__(self):
        _path(self.relative_path)
        _require(type(self.role) is str and type(self.format) is str, "invalid_role_path", "artifact")
        _require(type(self.byte_length) is int and self.byte_length > 0 and _hash(self.sha256), "invalid_artifact_digest", "artifact")
        _require(all(_identifier(getattr(self, name)) for name in ("format_version", "schema_identity", "related_identity")),
                 "invalid_artifact_metadata", "artifact")
        if self.role == "source_input":
            _require(self.format == "parquet" and self.relative_path.startswith("inputs-root/")
                     and self.relative_path.endswith(".parquet"), "invalid_role_path", "source_input")
        else:
            _require(self.role in ROLE_SPECS and (self.relative_path, self.format) == ROLE_SPECS[self.role][:2],
                     "invalid_role_path", "artifact")
        if self.parent_artifact is not None:
            _path(self.parent_artifact)


@dataclass(frozen=True)
class ResearchArtifactInput:
    """Immutable bytes only; no streams, callables or model objects.

    Known sensitive JSON keys are rejected. No generic scan of opaque bytes or
    arbitrary string values can certify the absence of secrets.
    """
    role: str
    relative_path: str
    data: bytes = field(repr=False)
    format: str
    format_version: str
    schema_identity: str
    related_identity: str
    parent_artifact: str | None = None

    def __post_init__(self):
        _require(type(self.data) is bytes, "unsupported_artifact_source", "data")
        self.entry()

    def entry(self):
        return ResearchArtifactEntryV1(self.role, self.relative_path, len(self.data), _sha(self.data), self.format,
            self.format_version, self.schema_identity, self.related_identity, self.parent_artifact)


@dataclass(frozen=True)
class ResearchBundleManifestV1:
    metadata: ResearchBundleMetadataV1
    artifacts: tuple[ResearchArtifactEntryV1, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        _require(type(self.schema_version) is str and self.schema_version == SCHEMA_VERSION, "unsupported_schema", "schema_version")
        _require(type(self.metadata) is ResearchBundleMetadataV1 and type(self.artifacts) in (tuple, list)
                 and all(type(item) is ResearchArtifactEntryV1 for item in self.artifacts), "invalid_manifest", "contracts")
        object.__setattr__(self, "artifacts", tuple(sorted(self.artifacts, key=lambda item: item.relative_path)))
        paths = [item.relative_path for item in self.artifacts]
        roles = [item.role for item in self.artifacts if item.role != "source_input"]
        _require(len(set(paths)) == len(paths) and len(set(roles)) == len(roles), "duplicate_artifact", "inventory")
        _require(REQUIRED_ROLES.issubset(roles), "missing_required_artifact", "inventory")
        keyed = {item.relative_path: item for item in self.artifacts}
        for item in self.artifacts:
            if item.role != "source_input":
                _require(item.related_identity == getattr(self.metadata, ROLE_SPECS[item.role][2]), "scientific_identity_mismatch", item.role)
            seen, current = {item.relative_path}, item
            while current.parent_artifact is not None:
                _require(current.parent_artifact in keyed and current.parent_artifact not in seen, "invalid_parent_reference", "inventory")
                seen.add(current.parent_artifact)
                current = keyed[current.parent_artifact]
        source_entries = {item.relative_path: item for item in self.artifacts if item.role == "source_input"}
        closure = self.metadata.input_closure
        expected = {"inputs-root/" + item.protocol_relative_path: item for item in closure.sources} if closure.mode == "bundled" else {}
        _require(set(source_entries) == set(expected), "input_closure_mismatch", "source_inputs")
        for path, reference in expected.items():
            actual = source_entries[path]
            _require((actual.sha256, actual.byte_length, actual.schema_identity, actual.related_identity) ==
                     (reference.sha256, reference.byte_length, reference.schema_identity, reference.artifact_id),
                     "input_closure_mismatch", "source_inputs")

    def canonical_core(self):
        return _json({"schema_version": self.schema_version, "metadata": asdict(self.metadata),
            "input_closure_status": self.metadata.input_closure.status,
            "verification_scope": "byte_integrity_and_declared_inventory",
            "artifacts": [asdict(item) for item in self.artifacts]})

    @property
    def bundle_id(self):
        return SCHEMA_VERSION + "-" + _sha(self.canonical_core().encode())

    def canonical_json(self):
        return _json({"bundle_id": self.bundle_id, "core": _load_json(self.canonical_core())})

    @classmethod
    def from_json(cls, text, *, limits=BundleLimits()):
        _require(type(limits) is BundleLimits and type(text) is str, "invalid_manifest_input", "manifest")
        _require(len(text.encode("utf-8")) <= limits.max_manifest_bytes, "resource_limit", "manifest")
        value = _load_json(text)
        _require(type(value) is dict and set(value) == {"bundle_id", "core"}, "invalid_fields", "manifest")
        core = value["core"]
        _require(type(core) is dict and set(core) == {"schema_version", "metadata", "artifacts", "input_closure_status", "verification_scope"},
                 "invalid_fields", "core")
        _require(core["schema_version"] == SCHEMA_VERSION, "unsupported_schema", "schema_version")
        md = core["metadata"]
        _require(type(md) is dict and type(md.get("input_closure")) is dict, "invalid_fields", "metadata")
        closure = md["input_closure"]
        _require(type(closure.get("sources")) is list and type(core["artifacts"]) is list, "invalid_fields", "inventory")
        _require(len(core["artifacts"]) <= limits.max_artifacts and len(closure["sources"]) <= 128, "resource_limit", "inventory")
        refs = []
        for item in closure["sources"]:
            _require(type(item) is dict, "invalid_fields", "sources")
            ext = item.get("external")
            refs.append(_construct(InputSourceReferenceV1, item, external=None if ext is None else _construct(ExternalInputReferenceV1, ext)))
        metadata = _construct(ResearchBundleMetadataV1, md, input_closure=_construct(InputClosureV1, closure, sources=tuple(refs)))
        result = cls(metadata, tuple(_construct(ResearchArtifactEntryV1, item) for item in core["artifacts"]), core["schema_version"])
        _require(result.canonical_json() == text, "noncanonical_or_identity_mismatch", "manifest")
        return result


def _check_limits(entries, manifest_bytes, limits):
    _require(type(limits) is BundleLimits, "invalid_limits", "limits")
    _require(len(entries) <= limits.max_artifacts and all(item.byte_length <= limits.max_artifact_bytes for item in entries)
             and sum(item.byte_length for item in entries) <= limits.max_total_bytes
             and len(manifest_bytes) <= limits.max_manifest_bytes, "resource_limit", "inventory")


def build_research_manifest(artifact_inputs, metadata, *, limits=BundleLimits()):
    _require(type(limits) is BundleLimits, "invalid_limits", "limits")
    _require(type(artifact_inputs) in (tuple, list)
             and all(type(item) is ResearchArtifactInput for item in artifact_inputs), "invalid_artifact_inputs", "inventory")
    _require(len(artifact_inputs) <= limits.max_artifacts, "resource_limit", "inventory")
    artifact_inputs = tuple(artifact_inputs)
    entries = tuple(item.entry() for item in artifact_inputs)
    manifest = ResearchBundleManifestV1(metadata, entries)
    _check_limits(entries, manifest.canonical_json().encode(), limits)
    return manifest


def _open_root(path):
    _require(type(path) is str or isinstance(path, Path), "invalid_archive_root", "archive_root")
    path = Path(path)
    _require(path.is_absolute() and path.anchor != "//" and ".." not in path.parts, "invalid_archive_root", "archive_root")
    _require(os.open in getattr(os, "supports_dir_fd", ()) and all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")),
             "unsupported_platform", "secure_open")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _directory(fd, parts, *, create=False):
    current = os.dup(fd)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        yield current
    finally:
        os.close(current)


def _sync_file(fd):
    os.fsync(fd)


def _sync_directory(fd):
    try:
        os.fsync(fd)
    except OSError as error:
        # These errors on an opened directory identify an unavailable barrier.
        if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
            raise BundleValidationError("unsupported_directory_sync", "filesystem") from None
        raise


def _write_exclusive(stage_fd, path, data):
    parts = path.split("/")
    with _directory(stage_fd, parts[:-1], create=True) as parent:
        fd = os.open(parts[-1], os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), 0o400)
            _sync_file(handle.fileno())


def _read_exact(directory_fd, path, maximum):
    parts = path.split("/")
    with _directory(directory_fd, parts[:-1]) as parent:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            facts = os.fstat(handle.fileno())
            _require(stat.S_ISREG(facts.st_mode) and facts.st_nlink == 1, "unsafe_artifact", "artifact")
            _require(facts.st_size <= maximum, "resource_limit", "artifact")
            data = handle.read(maximum + 1)
            _require(len(data) == facts.st_size and len(data) <= maximum, "artifact_length_mismatch", "artifact")
            return data


def _inventory(directory_fd, *, maximum_entries, prefix="", output=None):
    output = set() if output is None else output
    for name in os.listdir(directory_fd):
        path = prefix + name
        output.add(path)
        _require(len(output) <= maximum_entries, "resource_limit", "directory_entries")
        facts = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(facts.st_mode):
            _require(len(path.split("/")) <= 16, "resource_limit", "directory_depth")
            with _directory(directory_fd, [name]) as child:
                _inventory(child, maximum_entries=maximum_entries, prefix=path + "/", output=output)
        else:
            _require(stat.S_ISREG(facts.st_mode), "unsafe_artifact", "inventory")
    return output


def _payload_syntax(item, data):
    if item.format == "json":
        _load_json(data)
    # Parquet is opaque in this stage; no Arrow or model parser is invoked.


@dataclass(frozen=True)
class VerifiedResearchBundle:
    """Owned byte images only; declared scientific schemas are not certified."""
    directory: Path
    manifest: ResearchBundleManifestV1
    artifacts: tuple[ResearchArtifactInput, ...] = field(repr=False)

    def artifact_bytes(self, relative_path):
        return next(item.data for item in self.artifacts if item.relative_path == relative_path)


def _read_bundle(directory_fd, directory, expected_bundle_id, limits):
    text = _read_exact(directory_fd, MANIFEST_NAME, limits.max_manifest_bytes).decode("utf-8")
    manifest = ResearchBundleManifestV1.from_json(text, limits=limits)
    _require(manifest.bundle_id == expected_bundle_id, "bundle_identity_mismatch", "bundle_id")
    _check_limits(manifest.artifacts, text.encode(), limits)
    expected_paths = {MANIFEST_NAME}
    for item in manifest.artifacts:
        parts = item.relative_path.split("/")
        expected_paths.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    _require(_inventory(directory_fd, maximum_entries=limits.max_artifacts * 16 + 1) == expected_paths,
             "inventory_mismatch", "artifacts")
    images = []
    for item in manifest.artifacts:
        data = _read_exact(directory_fd, item.relative_path, min(item.byte_length, limits.max_artifact_bytes))
        _require(len(data) == item.byte_length and _sha(data) == item.sha256, "artifact_digest_mismatch", item.role)
        _payload_syntax(item, data)
        images.append(ResearchArtifactInput(item.role, item.relative_path, data, item.format, item.format_version,
                                           item.schema_identity, item.related_identity, item.parent_artifact))
    return VerifiedResearchBundle(Path(directory), manifest, tuple(images))


def open_research_bundle(bundle_directory, *, expected_bundle_id, limits=BundleLimits()):
    """Read-only inventory/byte verification, without model loading or fallback."""
    path = Path(bundle_directory)
    _require(type(limits) is BundleLimits, "invalid_limits", "limits")
    _require(path.name == expected_bundle_id and not path.name.startswith(STAGING_PREFIX), "unpublished_or_wrong_directory", "bundle_directory")
    fd = _open_root(path)
    try:
        return _read_bundle(fd, path, expected_bundle_id, limits)
    finally:
        os.close(fd)


def _seal_and_sync(fd):
    for name in os.listdir(fd):
        facts = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(facts.st_mode):
            with _directory(fd, [name]) as child:
                _seal_and_sync(child)
    os.fchmod(fd, 0o500)
    _sync_directory(fd)


def _rename_noreplace(root_fd, source_name, destination_name):
    """Commit via a kernel-enforced no-replacement operation, never plain rename."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        symbol, flag = "renameatx_np", 0x00000004  # Darwin SDK RENAME_EXCL.
    elif sys.platform.startswith("linux"):
        symbol, flag = "renameat2", 1  # Linux RENAME_NOREPLACE.
    else:
        raise BundleValidationError("unsupported_noreplace", "filesystem")
    operation = getattr(library, symbol, None)
    _require(operation is not None, "unsupported_noreplace", "filesystem")
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    if operation(root_fd, os.fsencode(source_name), root_fd, os.fsencode(destination_name), flag) != 0:
        number = ctypes.get_errno()
        if number in {errno.ENOTSUP, errno.EINVAL, errno.ENOSYS}:
            raise BundleValidationError("unsupported_noreplace", "filesystem")
        raise OSError(number, "Exclusive publication failed.")


def _remove_contents(fd):
    os.fchmod(fd, 0o700)
    for name in os.listdir(fd):
        facts = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(facts.st_mode):
            with _directory(fd, [name]) as child:
                _remove_contents(child)
            os.rmdir(name, dir_fd=fd)
        else:
            os.unlink(name, dir_fd=fd)


def _cleanup_stage(root_fd, name, identity):
    try:
        facts = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _require((facts.st_dev, facts.st_ino) == identity, "stage_identity_changed", "cleanup")
    with _directory(root_fd, [name]) as child:
        _remove_contents(child)
    os.rmdir(name, dir_fd=root_fd)


def _stage_still_owned(root_fd, name, identity):
    """A missing/replaced source after a commit error makes the result uncertain."""
    try:
        facts = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        return stat.S_ISDIR(facts.st_mode) and (facts.st_dev, facts.st_ino) == identity
    except OSError:
        return False


@dataclass(frozen=True)
class ResearchPublicationOutcome:
    status: str
    bundle_id: str | None
    directory: Path | None
    bundle: VerifiedResearchBundle | None
    diagnostics: tuple[BundleDiagnostic, ...] = ()
    cleanup_diagnostics: tuple[BundleDiagnostic, ...] = ()


def publish_research_bundle(archive_root, artifact_inputs, metadata, *, limits=BundleLimits()):
    """Publish supplied bytes once; no serializers, clocks, providers or retries.

    The archive root must already exist. External content-store verification is
    caller-declared and never fetched. Outcomes after a successful rename are
    never rolled back; any subsequent failure is explicitly uncertain.
    """
    status, manifest, bundle, diagnostics = "not_published", None, None, ()
    root_fd, stage, stage_identity, lock, lock_identity = None, None, None, None, None
    owns_lock, published, commit_started = False, False, False
    cleanup = []
    directory = None
    try:
        _require(type(artifact_inputs) in (tuple, list), "invalid_artifact_inputs", "inventory")
        artifact_inputs = tuple(artifact_inputs)
        manifest = build_research_manifest(artifact_inputs, metadata, limits=limits)
        for item in artifact_inputs:
            _payload_syntax(item, item.data)
        root_fd = _open_root(archive_root)
        stage = STAGING_PREFIX + uuid4().hex
        os.mkdir(stage, mode=0o700, dir_fd=root_fd)
        facts = os.stat(stage, dir_fd=root_fd, follow_symlinks=False)
        stage_identity = facts.st_dev, facts.st_ino
        with _directory(root_fd, [stage]) as stage_fd:
            for item in artifact_inputs:
                _write_exclusive(stage_fd, item.relative_path, item.data)
            _write_exclusive(stage_fd, MANIFEST_NAME, manifest.canonical_json().encode())
            _read_bundle(stage_fd, Path(archive_root) / stage, manifest.bundle_id, limits)
            _seal_and_sync(stage_fd)
        lock = LOCK_PREFIX + manifest.bundle_id
        directory = Path(archive_root) / manifest.bundle_id
        try:
            os.mkdir(lock, mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            status = "publication_busy"
            diagnostics = (BundleDiagnostic("publication_lock_exists", "publication"),)
        else:
            owns_lock = True
            lock_facts = os.stat(lock, dir_fd=root_fd, follow_symlinks=False)
            lock_identity = lock_facts.st_dev, lock_facts.st_ino
            _sync_directory(root_fd)
            try:
                commit_started = True
                _rename_noreplace(root_fd, stage, manifest.bundle_id)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                try:
                    with _directory(root_fd, [manifest.bundle_id]) as existing_fd:
                        existing = _read_bundle(existing_fd, directory, manifest.bundle_id, limits)
                    _require(existing.manifest.canonical_json() == manifest.canonical_json(), "existing_bundle_conflict", "publication")
                    bundle, status = existing, "already_exists"
                except (OSError, ValueError, TypeError, KeyError):
                    status = "conflict"
                    diagnostics = (BundleDiagnostic("existing_bundle_conflict", "publication"),)
            else:
                published = True
                _sync_directory(root_fd)
                with _directory(root_fd, [manifest.bundle_id]) as final_fd:
                    bundle = _read_bundle(final_fd, directory, manifest.bundle_id, limits)
                status = "published"
    except Exception as error:
        uncertain = published or (commit_started and not _stage_still_owned(root_fd, stage, stage_identity))
        status = "published_durability_uncertain" if uncertain else "not_published"
        diagnostics = error.diagnostics if isinstance(error, BundleValidationError) else (BundleDiagnostic("publication_error", "publication", type(error).__name__),)
    finally:
        if root_fd is not None:
            if stage is not None and stage_identity is not None:
                try:
                    _cleanup_stage(root_fd, stage, stage_identity)
                except Exception as error:
                    cleanup.append(BundleDiagnostic("stage_cleanup_failed", "cleanup", type(error).__name__))
            if owns_lock:
                try:
                    facts = os.stat(lock, dir_fd=root_fd, follow_symlinks=False)
                    _require(stat.S_ISDIR(facts.st_mode) and (facts.st_dev, facts.st_ino) == lock_identity,
                             "lock_identity_changed", "cleanup")
                    os.rmdir(lock, dir_fd=root_fd)
                    _sync_directory(root_fd)
                except Exception as error:
                    cleanup.append(BundleDiagnostic("lock_cleanup_failed", "cleanup", type(error).__name__))
            try:
                os.close(root_fd)
            except Exception as error:
                cleanup.append(BundleDiagnostic("descriptor_cleanup_failed", "cleanup", type(error).__name__))
    return ResearchPublicationOutcome(status, manifest.bundle_id if manifest is not None else None,
                                      directory, bundle, diagnostics, tuple(cleanup))

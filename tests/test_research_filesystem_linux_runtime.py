"""Portable preflight plus explicitly gated, unqualified native Linux cases.

On another OS run only ``-k portable``. Native cases fail rather than skip when
Linux is unavailable. Process death never establishes power-loss persistence.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

import test_research_filesystem_crash_safety as behavior
from test_research_bundle import research_bundle
from test_research_mlflow_linkage import record
from src.research import bundle as archive
from src.tracking import research_linkage_journal as journal

MARKER = ".apf-linux-filesystem-test-root"


def runtime_metadata(path):
    """Read host facts; do not label an unknown physical host as verified."""
    result = {"system": platform.system(), "kernel": platform.release(),
        "architecture": platform.machine(), "python": platform.python_version(),
        "native_linux": sys.platform.startswith("linux") and platform.system() == "Linux",
        "filesystem": "unknown", "mount_options": [], "physical_host": "unknown",
        "container_hint": Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()}
    symbol = "renameat2" if result["native_linux"] else "renameatx_np" if sys.platform == "darwin" else None
    result["rename_symbol"] = symbol
    result["rename_symbol_available"] = bool(symbol and hasattr(ctypes.CDLL(None), symbol))
    if result["native_linux"]:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
        matches = []
        for line in lines:
            left, right = line.split(" - ", 1)
            fields = left.split()
            mount = fields[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
            if Path(path).resolve().is_relative_to(Path(mount)):
                matches.append((len(mount), right.split()[0], fields[5].split(","), right.split()[2].split(",")))
        if matches:
            _, result["filesystem"], mount_options, super_options = max(matches)
            result["mount_options"] = sorted(set(mount_options + super_options))
    return result


def contained(path, approved):
    """Validate existing/missing child names before writes or worker dispatch."""
    path, approved = Path(path), Path(approved)
    temporary = (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())
    if (not approved.is_absolute() or approved != approved.resolve()
            or not any(approved.is_relative_to(base) and approved != base for base in temporary)
            or approved.is_relative_to(ROOT) or ROOT.is_relative_to(approved)):
        raise ValueError("unapproved_test_root")
    marker = approved / MARKER
    if marker.is_symlink() or not marker.is_file() or marker.read_text() != "apf-isolated-filesystem-v1":
        raise ValueError("missing_test_root_authorization")
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(approved):
        raise ValueError("test_path_escape")
    for part in (path, *path.parents):
        if part == approved:
            break
        if part.is_symlink():
            raise ValueError("test_symlink_escape")
    resolved = path.resolve()
    if not resolved.is_relative_to(approved) or resolved.is_relative_to(ROOT):
        raise ValueError("test_path_escape")
    if any(part in {"mlflow.db", "mlflow.db.backup", "mlartifacts"} for part in path.relative_to(approved).parts):
        raise ValueError("protected_tracking_path")
    return resolved


@pytest.fixture
def approved(tmp_path):
    root = tmp_path.resolve()
    assert not root.is_relative_to(ROOT)
    (root / MARKER).write_text("apf-isolated-filesystem-v1")
    return root


@pytest.fixture
def native_linux(approved):
    facts = runtime_metadata(approved)
    if not facts["native_linux"]:
        pytest.fail("Native Linux unavailable: select -k portable on this host; no Linux result can be claimed")
    assert facts["rename_symbol"] == "renameat2" and facts["rename_symbol_available"]
    contained(approved / "runtime.json", approved).write_text(json.dumps(facts, indent=2))
    return facts


def checked_request(approved, *, case=None, entry=None, record=None):
    if case:
        contained(case[0], approved)
    if entry:
        contained(entry[0], approved)
    request = behavior.request_file(approved, case=case, entry=entry, record=record)
    data = json.loads(request.read_text())
    data["approved_root"] = str(approved)
    for key in ("root", "journal", "remote"):
        if key in data:
            contained(data[key], approved)
    contained(request, approved).write_text(json.dumps(data))
    return request


@contextmanager
def checked_worker(request, mode):
    approved = request.parent
    contained(request, approved)
    child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(request), mode],
        cwd=approved, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=15)


def inspect_fresh(request):
    with checked_worker(request, "inspect") as child:
        value = behavior.message(child)
        assert child.wait(timeout=15) == 0
    return value


def test_portable_metadata_describes_actual_host(approved):
    facts = runtime_metadata(approved)
    assert facts["system"] == platform.system()
    assert facts["python"] == platform.python_version()
    assert facts["native_linux"] == (sys.platform.startswith("linux") and platform.system() == "Linux")
    assert facts["physical_host"] == "unknown"


def test_portable_allows_only_owned_temporary_child(approved):
    assert contained(approved / "new" / "journal", approved) == approved / "new" / "journal"


@pytest.mark.parametrize("kind", ["outside", "relative", "parent", "repository", "symlink", "tracking"])
def test_portable_containment_rejects_unsafe_paths(approved, kind):
    paths = {"outside": approved.parent / "outside", "relative": Path("journal"),
             "parent": approved / "a" / ".." / "journal", "repository": ROOT,
             "tracking": approved / "mlartifacts"}
    if kind == "symlink":
        alias = approved / "alias"
        alias.symlink_to(approved.parent, target_is_directory=True)
        paths[kind] = alias / "journal"
    with pytest.raises(ValueError):
        contained(paths[kind], approved)


def test_portable_missing_root_marker_rejects_dispatch(tmp_path):
    with pytest.raises(ValueError, match="missing_test_root_authorization"):
        contained(tmp_path.resolve() / "journal", tmp_path.resolve())


def test_portable_request_preserves_bytes_and_provenance(approved, research_bundle):
    request = checked_request(approved, case=research_bundle)
    data = json.loads(request.read_text())
    expected = archive.build_research_manifest(research_bundle[1], research_bundle[2])
    assert data["manifest"] == expected.canonical_json()
    assert [bytes.fromhex(a["data_hex"]) for a in data["artifacts"]] == [a.data for a in research_bundle[1]]
    assert data["approved_root"] == str(approved)
    assert not list(research_bundle[0].iterdir())


def test_native_linux_real_noreplace_and_directory_sync(native_linux, approved, research_bundle):
    root, artifacts, metadata = research_bundle
    contained(root, approved)
    result = archive.publish_research_bundle(root, artifacts, metadata)
    assert result.status == "published", result.diagnostics
    before = behavior.snapshot(result.directory)
    assert archive.publish_research_bundle(root, artifacts, metadata).status == "already_exists"
    assert behavior.snapshot(result.directory) == before
    fd = archive._open_root(root)
    try:
        archive._sync_directory(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("mode,visible,locked", [("staging", False, False),
    ("before_rename", False, True), ("after_rename", True, True), ("after_publish", True, False)])
def test_native_linux_kill_then_independent_recovery(native_linux, approved, research_bundle, mode, visible, locked):
    request = checked_request(approved, case=research_bundle)
    with checked_worker(request, mode) as child:
        assert behavior.message(child)["paused"] == mode
        child.kill()
        assert child.wait(timeout=15) < 0
    first, second = inspect_fresh(request), inspect_fresh(request)
    assert first == second
    assert first["bundle_valid"] is visible and bool(first["locks"]) is locked
    assert bool(first["stages"]) is (not visible)
    if locked:
        with checked_worker(request, "publish") as child:
            assert behavior.message(child)["status"] == "publication_busy"
            assert child.wait(timeout=15) == 0
        assert inspect_fresh(request) == first


def test_native_linux_bounded_publish_contention(native_linux, approved, research_bundle):
    root, artifacts, metadata = research_bundle
    # Three deterministic rounds, four independent publishers per round.
    for iteration in range(3):
        case = (root, artifacts, replace(metadata, attempt_id=f"stress-{iteration}"))
        request = checked_request(approved, case=case)
        with checked_worker(request, "before_rename") as winner:
            assert behavior.message(winner)["paused"] == "before_rename"
            assert inspect_fresh(request)["bundle_valid"] is False
            with ExitStack() as stack:
                others = [stack.enter_context(checked_worker(request, "publish")) for _ in range(3)]
                for child in others:
                    assert behavior.message(child)["status"] == "publication_busy"
                    assert child.wait(timeout=15) == 0
            behavior.release(winner)
            assert behavior.message(winner)["status"] == "published"
            assert winner.wait(timeout=15) == 0
        assert inspect_fresh(request)["bundle_valid"]
        result = archive.open_research_bundle(root / archive.build_research_manifest(artifacts, case[2]).bundle_id,
            expected_bundle_id=archive.build_research_manifest(artifacts, case[2]).bundle_id)
        assert all(result.artifact_bytes(a.relative_path) == a.data for a in artifacts)


@pytest.mark.parametrize("mode", ["journal_hold", "journal_after_replace"])
def test_native_linux_journal_kill_revision_and_recovery(native_linux, approved, record, mode):
    root = contained(approved / "journal", approved)
    root.mkdir()
    initial = journal.new_intent(record, "experiment-1")
    request = checked_request(approved, entry=(root, initial))
    with checked_worker(request, mode) as owner:
        assert behavior.message(owner)["paused"] == mode
        with checked_worker(request, "journal_advance") as competitor:
            assert behavior.message(competitor)["error"] == "journal_busy"
            assert competitor.wait(timeout=15) == 0
        owner.kill()
        assert owner.wait(timeout=15) < 0
    evidence = inspect_fresh(request)
    assert evidence["journal"]["create_may_exist"] is True
    with checked_worker(request, "journal_advance") as next_writer:
        assert behavior.message(next_writer)["revision"] == evidence["journal"]["revision"] + 1
        assert next_writer.wait(timeout=15) == 0


def child_main(request_path, mode):
    request_path = Path(request_path)
    approved = request_path.parent
    contained(request_path, approved)
    data = json.loads(request_path.read_text())
    assert data["approved_root"] == str(approved)
    assert runtime_metadata(approved)["native_linux"], "Native Linux is required before worker dispatch"
    for key in ("root", "journal", "remote"):
        if key in data:
            contained(data[key], approved)
    if mode == "inspect":
        output = {}
        if "root" in data:
            root = Path(data["root"])
            manifest = archive.ResearchBundleManifestV1.from_json(data["manifest"])
            destination = contained(root / manifest.bundle_id, approved)
            output.update(bundle_valid=False, stages=sorted(p.name for p in root.glob(archive.STAGING_PREFIX + "*")),
                          locks=sorted(p.name for p in root.glob(archive.LOCK_PREFIX + "*")))
            if destination.exists():
                archive.open_research_bundle(destination, expected_bundle_id=manifest.bundle_id)
                output["bundle_valid"] = True
        if "journal" in data:
            initial = journal.LinkageIntent.from_json(data["intent"].encode())
            entry = contained(Path(data["journal"]) / (initial.linkage_key + ".json"), approved)
            current = journal.LinkageIntent.from_json(entry.read_bytes()) if entry.exists() else None
            output["journal"] = None if current is None else {"revision": current.revision,
                "status": current.status, "create_may_exist": current.create_may_exist}
        print(json.dumps(output), flush=True)
        return
    try:
        behavior.worker_main(str(request_path), mode)
    except journal.JournalError as error:
        print(json.dumps({"error": error.code}), flush=True)


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] == "--worker"
    child_main(sys.argv[2], sys.argv[3])

"""Disposable byte-only filesystem tests; process death is not power loss."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import errno
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from test_research_bundle import research_bundle, publish, snapshot
from test_research_mlflow_linkage import record, FakeClient
from src.research import bundle as archive
from src.tracking import research_linkage_journal as journal
from src.tracking import research_mlflow as linkage
from src.tracking import research_reconciliation as reconciliation
from src.tracking.research_record import ResearchTrackingRecordV1


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Never use a repository root or a real client in this module."""
    assert not tmp_path.resolve().is_relative_to(ROOT)
    import socket
    def denied(*args, **kwargs):
        raise AssertionError("Network access is forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.mark.parametrize("number", [errno.EINVAL, errno.ENOTSUP, errno.ENOSYS])
def test_unsupported_directory_sync_is_explicit(research_bundle, monkeypatch, number):
    original = archive.os.fsync
    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(number, "Unsupported test directory barrier")
        return original(fd)
    monkeypatch.setattr(archive.os, "fsync", sync)
    outcome = publish(research_bundle)
    assert outcome.status == "not_published"
    assert outcome.diagnostics[0].code == "unsupported_directory_sync"
    assert list(research_bundle[0].iterdir()) == []


@pytest.mark.parametrize("phase,visible", [
    ("before_stage", False), ("partial_write", False), ("before_file_sync", False),
    ("after_file_sync", False), ("before_install", False), ("install_error", False),
    ("after_install", True), ("parent_sync", True), ("final_read", True)])
def test_publication_fault_boundaries_preserve_original_archive(research_bundle, monkeypatch, phase, visible):
    root, artifacts, metadata = research_bundle
    original_archive = publish(research_bundle)
    before = snapshot(original_archive.directory)
    changed = (root, artifacts, replace(metadata, attempt_id="attempt:002"))
    identifier = archive.build_research_manifest(artifacts, changed[2]).bundle_id
    write, sync_file, sync_dir, rename, read, mkdir = (archive._write_exclusive,
        archive._sync_file, archive._sync_directory, archive._rename_noreplace,
        archive._read_bundle, os.mkdir)
    installed = []
    def fail():
        raise OSError(errno.EIO, "Deterministic test boundary")
    def make(name, *args, **kwargs):
        if str(name).startswith(archive.STAGING_PREFIX):
            fail()
        return mkdir(name, *args, **kwargs)
    def partial(fd, path, data):
        with archive._directory(fd, path.split("/")[:-1], create=True) as parent:
            f = os.open(path.split("/")[-1], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=parent)
            try:
                os.write(f, data[:max(1, len(data) // 2)])
            finally:
                os.close(f)
        fail()
    def file_barrier(fd):
        if phase == "before_file_sync":
            fail()
        sync_file(fd)
        fail()
    def install(*args):
        if phase in {"before_install", "install_error"}:
            fail()
        rename(*args)
        installed.append(True)
        if phase == "after_install":
            fail()
    def parent(fd):
        if installed:
            fail()
        return sync_dir(fd)
    def verify(fd, directory, expected, limits):
        if installed:
            fail()
        return read(fd, directory, expected, limits)
    with monkeypatch.context() as patch:
        if phase == "before_stage":
            patch.setattr(archive.os, "mkdir", make)
        elif phase == "partial_write":
            patch.setattr(archive, "_write_exclusive", partial)
        elif phase in {"before_file_sync", "after_file_sync"}:
            patch.setattr(archive, "_sync_file", file_barrier)
        if phase in {"before_install", "install_error", "after_install", "parent_sync", "final_read"}:
            patch.setattr(archive, "_rename_noreplace", install)
        if phase == "parent_sync":
            patch.setattr(archive, "_sync_directory", parent)
        if phase == "final_read":
            patch.setattr(archive, "_read_bundle", verify)
        outcome = publish(changed)
    assert outcome.status == ("published_durability_uncertain" if visible else "not_published")
    assert (root / identifier).exists() is visible
    assert snapshot(original_archive.directory) == before
    assert not any(p.name.startswith((archive.STAGING_PREFIX, archive.LOCK_PREFIX)) for p in root.iterdir())
    if visible:
        # Inspect only: a visible directory must not authorize automatic retry.
        a = archive.open_research_bundle(root / identifier, expected_bundle_id=identifier)
        b = archive.open_research_bundle(root / identifier, expected_bundle_id=identifier)
        assert a.manifest.canonical_json() == b.manifest.canonical_json()
        assert outcome.status == "published_durability_uncertain"


@pytest.mark.parametrize("number", [errno.EXDEV, errno.EACCES, errno.EIO])
def test_install_operational_errors_never_fall_back(research_bundle, monkeypatch, number):
    def fail(*args):
        raise OSError(number, "Test install failure")
    monkeypatch.setattr(archive, "_rename_noreplace", fail)
    result = publish(research_bundle)
    assert result.status == "not_published" and result.diagnostics
    assert not list(research_bundle[0].iterdir())


def test_missing_native_symbol_fails_closed(research_bundle, monkeypatch):
    class Library:
        pass
    monkeypatch.setattr(archive.ctypes, "CDLL", lambda *a, **k: Library())
    result = publish(research_bundle)
    assert result.status == "not_published"
    assert result.diagnostics[0].code == "unsupported_noreplace"


def test_cleanup_error_preserves_published_destination(research_bundle, monkeypatch):
    def fail(*args):
        raise OSError(errno.EACCES, "Test cleanup failure")
    monkeypatch.setattr(archive, "_cleanup_stage", fail)
    result = publish(research_bundle)
    assert result.status == "published" and result.cleanup_diagnostics
    assert archive.open_research_bundle(result.directory, expected_bundle_id=result.bundle_id)


def test_symlink_root_is_rejected_without_publication(research_bundle, tmp_path):
    root, artifacts, metadata = research_bundle
    alias = tmp_path / "unsafe-root"
    alias.symlink_to(root, target_is_directory=True)
    result = archive.publish_research_bundle(alias, artifacts, metadata)
    assert result.status == "not_published" and not list(root.iterdir())


@pytest.fixture
def intent(record, tmp_path):
    root = tmp_path.resolve() / "journal"
    root.mkdir(mode=0o700)
    return root, journal.new_intent(record, "experiment-1")


@pytest.mark.parametrize("phase", ["file_sync", "replace_before", "replace_after", "directory_sync"])
def test_journal_ambiguous_replace_preserves_creation_history(intent, monkeypatch, phase):
    root, initial = intent
    with journal.LinkageJournal(root).transaction(initial.linkage_key) as tx:
        tx.write(initial, expected_revision=0)
    original = (root / (initial.linkage_key + ".json")).read_bytes()
    entry = replace(initial, revision=2, status="creating", phase="create", create_may_exist=True)
    sync, rename = os.fsync, os.replace
    def barrier(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if directory == (phase == "directory_sync"):
            raise OSError(errno.EIO, "Test sync failure")
        sync(fd)
    def install(*args, **kwargs):
        if phase == "replace_before":
            raise OSError(errno.EIO, "Test precommit failure")
        rename(*args, **kwargs)
        raise OSError(errno.EIO, "Test lost replace response")
    with journal.LinkageJournal(root).transaction(initial.linkage_key) as tx:
        with monkeypatch.context() as patch:
            if phase in {"file_sync", "directory_sync"}:
                patch.setattr(journal.os, "fsync", barrier)
            else:
                patch.setattr(journal.os, "replace", install)
            with pytest.raises(journal.JournalError) as caught:
                tx.write(entry, expected_revision=1)
        uncertain = phase in {"replace_after", "directory_sync"}
        assert caught.value.uncertain is uncertain
        read = tx.read()
        assert read == (entry if uncertain else initial)
        if uncertain:
            assert read.create_may_exist
        else:
            assert (root / (initial.linkage_key + ".json")).read_bytes() == original
    assert not list(root.glob(".linkage-stage-*"))


@pytest.mark.parametrize("directory", [False, True])
def test_journal_unsupported_sync_is_distinct_and_preserves_uncertainty(intent, monkeypatch, directory):
    root, initial = intent
    sync = os.fsync
    def unsupported(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) is directory:
            raise OSError(errno.ENOTSUP, "Unsupported test sync")
        return sync(fd)
    with journal.LinkageJournal(root).transaction(initial.linkage_key) as tx:
        with monkeypatch.context() as patch:
            patch.setattr(journal.os, "fsync", unsupported)
            with pytest.raises(journal.JournalError) as caught:
                tx.write(initial, expected_revision=0)
        assert caught.value.code == "journal_sync_unsupported"
        assert caught.value.uncertain is directory
        assert tx.read() == (initial if directory else None)


def test_torn_journal_is_rejected_not_replaced(intent):
    root, initial = intent
    path = root / (initial.linkage_key + ".json")
    path.write_bytes(initial.canonical_json().encode()[:50])
    before = path.read_bytes()
    with journal.LinkageJournal(root).transaction(initial.linkage_key) as tx:
        with pytest.raises(journal.JournalError):
            tx.write(initial, expected_revision=0)
    assert path.read_bytes() == before


def request_file(tmp_path, case=None, entry=None, record=None):
    request = {}
    if case is not None:
        root, artifacts, metadata = case
        request.update(root=str(root), manifest=archive.build_research_manifest(artifacts, metadata).canonical_json(),
            artifacts=[{**{k: v for k, v in asdict(a).items() if k != "data"}, "data_hex": a.data.hex()} for a in artifacts])
    if entry is not None:
        root, value = entry
        request.update(journal=str(root), intent=value.canonical_json())
    if record is not None:
        request["record"] = record.canonical_json()
        request["remote"] = str(tmp_path / "synthetic-remote.json")
    path = tmp_path / "worker-request.json"
    path.write_text(json.dumps(request))
    return path


@contextmanager
def worker(request, mode):
    process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(request), mode],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=request.parent)
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=15)


def message(process):
    assert select.select([process.stdout], [], [], 15)[0], "Worker failed to reach deterministic boundary"
    line = process.stdout.readline()
    if not line:
        raise AssertionError(process.stderr.read())
    return json.loads(line)


def release(process):
    process.stdin.write("continue\n")
    process.stdin.flush()


@pytest.mark.parametrize("mode,visible,locked", [("staging", False, False),
    ("before_rename", False, True), ("after_rename", True, True), ("after_publish", True, False)])
def test_killed_publisher_preserves_forensic_state(research_bundle, tmp_path, mode, visible, locked):
    request = request_file(tmp_path, case=research_bundle)
    root, artifacts, metadata = research_bundle
    identifier = archive.build_research_manifest(artifacts, metadata).bundle_id
    with worker(request, mode) as child:
        assert message(child)["paused"] == mode
        child.kill()
        assert child.wait(timeout=15) < 0
    assert (root / identifier).exists() is visible
    assert (root / (archive.LOCK_PREFIX + identifier)).exists() is locked
    assert bool(list(root.glob(archive.STAGING_PREFIX + "*"))) is (not visible)
    if visible:
        before = snapshot(root / identifier)
        for _ in range(2):
            archive.open_research_bundle(root / identifier, expected_bundle_id=identifier)
        assert snapshot(root / identifier) == before
    if locked:
        assert publish(research_bundle).status == "publication_busy"
        assert (root / (archive.LOCK_PREFIX + identifier)).is_dir()


def test_independent_publishers_do_not_overwrite(research_bundle, tmp_path):
    request = request_file(tmp_path, case=research_bundle)
    with worker(request, "before_rename") as first:
        assert message(first)["paused"] == "before_rename"
        with worker(request, "publish") as second:
            assert message(second)["status"] == "publication_busy"
            assert second.wait(timeout=15) == 0
        release(first)
        assert message(first)["status"] == "published"
        assert first.wait(timeout=15) == 0
    before = snapshot(research_bundle[0])
    with worker(request, "publish") as later:
        assert message(later)["status"] == "already_exists"
        assert later.wait(timeout=15) == 0
    assert snapshot(research_bundle[0]) == before


def test_process_conflicting_destination_is_preserved(research_bundle, tmp_path):
    root, artifacts, metadata = research_bundle
    identifier = archive.build_research_manifest(artifacts, metadata).bundle_id
    destination = root / identifier
    destination.mkdir()
    (destination / "original").write_bytes(b"preserve")
    before = snapshot(destination)
    with worker(request_file(tmp_path, case=research_bundle), "publish") as child:
        assert message(child)["status"] == "conflict"
        assert child.wait(timeout=15) == 0
    assert snapshot(destination) == before


@pytest.mark.parametrize("mode", ["journal_hold", "journal_after_replace"])
def test_killed_journal_writer_releases_flock_and_preserves_creating(intent, tmp_path, mode):
    request = request_file(tmp_path, entry=intent)
    root, initial = intent
    with worker(request, mode) as child:
        assert message(child)["paused"] == mode
        with pytest.raises(journal.JournalError, match="journal_busy"):
            with journal.LinkageJournal(root).transaction(initial.linkage_key):
                pass
        child.kill()
        assert child.wait(timeout=15) < 0
    with journal.LinkageJournal(root).transaction(initial.linkage_key) as tx:
        current = tx.read()
        assert current.create_may_exist and current.status == "creating"
    lock_inode = (root / (initial.linkage_key + ".lock")).stat().st_ino
    with worker(request, "journal_advance") as child:
        assert message(child)["revision"] == current.revision + 1
        assert child.wait(timeout=15) == 0
    assert (root / (initial.linkage_key + ".lock")).stat().st_ino == lock_inode


@pytest.mark.parametrize("mode", ["lost_response", "after_response"])
def test_killed_linkage_recovers_without_second_remote_create(intent, record, tmp_path, mode):
    root, initial = intent
    request = request_file(tmp_path, entry=intent, record=record)
    with worker(request, mode) as child:
        assert message(child)["paused"] == mode
        child.kill()
        assert child.wait(timeout=15) < 0
    fake = FakeClient(json.loads((tmp_path / "synthetic-remote.json").read_text()))
    fake.hidden = True
    destination = linkage.TrackingDestination("research_index", "experiment-1")
    args = dict(record=record, client=linkage.BoundMlflowClient(destination, fake),
                destination=destination, journal_root=root, enabled=True)
    assert linkage.link_research_record(**args).status == "linkage_uncertain"
    assert "create" not in fake.calls
    fake.hidden = False
    assert linkage.link_research_record(**args).status == "linked"
    assert "create" not in fake.calls and len(fake.store) == 1


def test_process_journal_change_is_detected_during_readonly_reconciliation(intent, tmp_path):
    root, initial = intent
    with journal.LinkageJournal(root).transaction(initial.linkage_key) as tx:
        tx.write(initial, expected_revision=0)
    empty_archive = tmp_path / "empty-archive"
    empty_archive.mkdir()
    fake = FakeClient()
    with worker(request_file(tmp_path, entry=intent), "journal_wait") as child:
        assert message(child)["paused"] == "journal_wait"
        changed = []
        def observe(name):
            if not changed:
                release(child)
                changed.append(message(child)["revision"])
                assert child.wait(timeout=15) == 0
        fake.observe = observe
        destination = linkage.TrackingDestination("research_index", "experiment-1")
        result = reconciliation.reconcile_research_tracking(archive_root=empty_archive,
            archive_alias="synthetic_archive", journal_root=root, destination=destination,
            client=linkage.BoundMlflowClient(destination, fake))
    assert changed == [2]
    assert "unstable_journal_snapshot" in {v.code for v in result.scope_findings}
    assert not any(v.status == "linked_consistent" for row in result.references for v in row.findings)


def worker_main(request_path, mode):
    """Test-only child; JSON transport and fake remote, never model loading."""
    request = json.loads(Path(request_path).read_text())
    def paused(tag):
        print(json.dumps({"paused": tag}), flush=True)
        assert sys.stdin.readline().strip() == "continue"
    if mode in {"staging", "before_rename", "after_rename", "after_publish", "publish"}:
        root = Path(request["root"]).resolve()
        assert not root.is_relative_to(ROOT)
        manifest = archive.ResearchBundleManifestV1.from_json(request["manifest"])
        artifacts = tuple(archive.ResearchArtifactInput(**{k: v for k, v in a.items() if k != "data_hex"},
                          data=bytes.fromhex(a["data_hex"])) for a in request["artifacts"])
        if mode == "staging":
            original = archive._write_exclusive
            def write(*args):
                paused(mode)
                return original(*args)
            archive._write_exclusive = write
        if mode in {"before_rename", "after_rename"}:
            original = archive._rename_noreplace
            def install(*args):
                if mode == "before_rename":
                    paused(mode)
                original(*args)
                if mode == "after_rename":
                    paused(mode)
            archive._rename_noreplace = install
        outcome = archive.publish_research_bundle(root, artifacts, manifest.metadata)
        if mode == "after_publish":
            assert outcome.status == "published"
            paused(mode)
        print(json.dumps({"status": outcome.status}), flush=True)
        return
    root = Path(request["journal"]).resolve()
    assert not root.is_relative_to(ROOT)
    initial = journal.LinkageIntent.from_json(request["intent"].encode())
    if mode in {"lost_response", "after_response"}:
        class Remote(FakeClient):
            def create_run(self, **kwargs):
                value = super().create_run(**kwargs)
                Path(request["remote"]).write_text(json.dumps(self.store))
                if mode == "lost_response":
                    paused(mode)
                return value
        if mode == "after_response":
            original = journal._Transaction.write
            def write(tx, entry, **kwargs):
                if entry.create_may_exist and entry.phase == "search":
                    paused(mode)
                return original(tx, entry, **kwargs)
            journal._Transaction.write = write
        destination = linkage.TrackingDestination("research_index", "experiment-1")
        outcome = linkage.link_research_record(ResearchTrackingRecordV1.from_json(request["record"]),
            client=linkage.BoundMlflowClient(destination, Remote()), destination=destination,
            journal_root=root, enabled=True)
        print(json.dumps({"status": outcome.status}), flush=True)
        return
    if mode == "journal_wait":
        paused(mode)
    with journal.LinkageJournal(root).transaction(initial.linkage_key) as tx:
        current = tx.read()
        revision = current.revision if current else 0
        creating = replace(current or initial, revision=revision + 1,
            status="creating", phase="create", create_may_exist=True)
        if mode == "journal_after_replace":
            original = journal.os.replace
            def install(*args, **kwargs):
                original(*args, **kwargs)
                paused(mode)
            journal.os.replace = install
        tx.write(creating, expected_revision=revision)
        if mode == "journal_hold":
            paused(mode)
    print(json.dumps({"revision": creating.revision}), flush=True)


if __name__ == "__main__":
    assert sys.argv[1] == "--worker" and len(sys.argv) == 4
    worker_main(sys.argv[2], sys.argv[3])

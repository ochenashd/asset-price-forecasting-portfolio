"""Static source contracts only; no publisher, journal or model is imported/run.

These checks identify intended barriers and fail-closed branches. They cannot
establish syscall behavior, multiprocess serialization or crash durability.
"""
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def tree(path):
    return ast.parse((ROOT / path).read_text())


def function(path, name):
    matches = [node for node in ast.walk(tree(path))
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]


def calls(node, name):
    return sorted((value for value in ast.walk(node)
                   if isinstance(value, ast.Call) and ast.unparse(value.func) == name),
                  key=lambda value: (value.lineno, value.col_offset))


BUNDLE = "src/research/bundle.py"
JOURNAL = "src/tracking/research_linkage_journal.py"
LINKAGE = "src/tracking/research_mlflow.py"
RECONCILIATION = "src/tracking/research_reconciliation.py"


def test_kernel_noreplace_platform_selection_has_no_plain_rename_fallback():
    node = function(BUNDLE, "_rename_noreplace")
    text = ast.unparse(node)
    assert "renameatx_np" in text and "renameat2" in text
    assert "0x00000004" in (ROOT / BUNDLE).read_text()
    assert "symbol, flag = ('renameat2', 1)" in text
    assert not calls(node, "os.rename") and not calls(node, "os.replace")
    assert "unsupported_noreplace" in text
    assert all(value in text for value in ("errno.ENOTSUP", "errno.EINVAL", "errno.ENOSYS"))


def test_root_open_walks_components_without_following_symlinks():
    node = function(BUNDLE, "_open_root")
    text = ast.unparse(node)
    assert "is_absolute" in text and "supports_dir_fd" in text
    assert all(flag in text for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
    assert any(any(k.arg == "dir_fd" for k in call.keywords) for call in calls(node, "os.open"))


@pytest.mark.parametrize("path,name", [(BUNDLE, "_read_exact"), (JOURNAL, "read"),
                                        (RECONCILIATION, "_read_entry")])
def test_reads_require_regular_single_link_objects(path, name):
    text = ast.unparse(function(path, name))
    assert "stat.S_ISREG" in text and "st_nlink" in text
    assert "O_NOFOLLOW" in text and "O_NONBLOCK" in text


def test_artifact_write_flushes_and_fsyncs_before_close():
    node = function(BUNDLE, "_write_exclusive")
    text = ast.unparse(node)
    assert all(flag in text for flag in ("O_EXCL", "O_CREAT", "O_NOFOLLOW"))
    assert calls(node, "handle.write")[0].lineno < calls(node, "handle.flush")[0].lineno
    assert calls(node, "handle.flush")[0].lineno < calls(node, "_sync_file")[0].lineno
    assert calls(node, "os.fchmod")[0].lineno < calls(node, "_sync_file")[0].lineno


@pytest.mark.parametrize("name", ["_sync_file", "_sync_directory"])
def test_sync_helpers_request_fsync_without_hardware_durability_claim(name):
    assert len(calls(function(BUNDLE, name), "os.fsync")) == 1


def test_sealing_recurses_before_directory_sync():
    node = function(BUNDLE, "_seal_and_sync")
    assert calls(node, "_seal_and_sync")[0].lineno < calls(node, "os.fchmod")[0].lineno
    assert calls(node, "os.fchmod")[0].lineno < calls(node, "_sync_directory")[0].lineno


def test_publisher_has_staging_verification_and_parent_barriers():
    node = function(BUNDLE, "publish_research_bundle")
    rename = calls(node, "_rename_noreplace")[0].lineno
    assert calls(node, "_write_exclusive")[-1].lineno < calls(node, "_read_bundle")[0].lineno
    assert calls(node, "_read_bundle")[0].lineno < calls(node, "_seal_and_sync")[0].lineno < rename
    syncs = calls(node, "_sync_directory")
    assert syncs[0].lineno < rename < syncs[1].lineno < calls(node, "_read_bundle")[-1].lineno


def test_publisher_reports_uncertainty_and_identity_checked_cleanup():
    text = ast.unparse(function(BUNDLE, "publish_research_bundle"))
    assert "published_durability_uncertain" in text and "_stage_still_owned" in text
    assert "lock_identity" in text and "cleanup_diagnostics" not in text
    cleanup = ast.unparse(function(BUNDLE, "_cleanup_stage"))
    assert "st_dev" in cleanup and "st_ino" in cleanup and "identity" in cleanup


def test_publication_acceptance_does_not_upgrade_explicit_uncertainty():
    node = function("src/research/bundle_publication.py", "publish_prepared_research_bundle")
    branches = [part for part in ast.walk(node) if isinstance(part, ast.If)
                and ast.unparse(part.test) == "outcome.status == 'published_durability_uncertain'"]
    assert len(branches) == 1
    assert isinstance(branches[0].body[0], ast.Return)
    assert branches[0].lineno < calls(node, "accepts_research_archive")[-1].lineno


def test_journal_lock_is_nonblocking_persistent_and_not_a_stale_file_lease():
    node = function(JOURNAL, "transaction")
    text = ast.unparse(node)
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in text and "journal_busy" in text
    assert "st_nlink == 1" in text and "O_NOFOLLOW" in text
    assert not calls(node, "os.unlink") and not calls(node, "os.remove")
    assert calls(node, "os.close")


def test_journal_cas_and_creation_history_precede_temporary_write():
    node = function(JOURNAL, "write")
    text = ast.unparse(node)
    assert "expected_revision + 1" in text and "journal_revision_conflict" in text
    assert "creation_history_cannot_be_reset" in text and "remote_identity_cannot_be_replaced" in text
    assert calls(node, "self.read")[0].lineno < calls(node, "os.open")[0].lineno


def test_journal_file_sync_replace_and_directory_sync_order():
    node = function(JOURNAL, "write")
    syncs = calls(node, "os.fsync")
    assert len(syncs) == 2
    assert calls(node, "stream.flush")[0].lineno < syncs[0].lineno
    assert syncs[0].lineno < calls(node, "os.replace")[0].lineno < syncs[1].lineno
    assert {k.arg for k in calls(node, "os.replace")[0].keywords} == {"src_dir_fd", "dst_dir_fd"}
    assert "journal_durability_uncertain" in ast.unparse(node)


def test_remote_creation_has_a_persisted_creating_intent():
    node = function(LINKAGE, "link_research_record")
    create = calls(node, "client.client.create_run")
    assert len(create) == 1
    advances = calls(node, "advance")
    creating = [call for call in advances if call.args and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "creating"]
    assert len(creating) == 1 and creating[0].lineno < create[0].lineno
    assert any(k.arg == "create_may_exist" and ast.literal_eval(k.value) is True for k in creating[0].keywords)
    assert calls(function(LINKAGE, "advance"), "transaction.write")


def test_reconciliation_snapshot_checks_do_not_mutate_files_or_remote_metadata():
    node = tree(RECONCILIATION)
    forbidden = ("os.replace", "os.rename", "os.unlink", "os.mkdir", "os.fsync",
                 "os.remove", "fcntl.flock")
    assert all(not calls(node, name) for name in forbidden)
    attrs = {call.func.attr for call in ast.walk(node)
             if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)}
    assert not attrs.intersection({"create_run", "set_tag", "log_artifact", "transaction"})
    read = ast.unparse(function(RECONCILIATION, "_read_entry"))
    assert "_facts(before) != _facts(after)" in read and "_facts(after) != _facts(named)" in read
    text = ast.unparse(node)
    assert "unstable_journal_snapshot" in text and "publisher_durability_not_resolved" in text

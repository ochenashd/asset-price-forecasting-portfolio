"""Data-only synthetic archives, journal fixtures and fake read-only clients.

Illustrative state bytes/probabilities are not fitted models or predictions
created by inference. Test setup publishes synthetic artifacts and writes
fixture journals BEFORE reconciliation; the tool must never mutate them.
"""
from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from test_research_tracking_record import accepted, inputs, source_inputs, research, scientific_bundle, build
from test_research_bundle import snapshot
from src.tracking import research_reconciliation as api
from src.tracking import research_linkage_journal as journal
from src.tracking.research_mlflow import BoundMlflowClient, TrackingDestination
from src.research import bundle as archive


class Page(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class ReadOnlyClient:
    def __init__(self):
        self.rows, self.calls = {}, []
        self.page_size = 100
        self.on_search = None
        self.failure = None
        self.cycle = False
    def search_runs(self, *, experiment_ids, filter_string, run_view_type, max_results, page_token):
        self.calls.append("search")
        assert experiment_ids == ["experiment-1"] and run_view_type == 3 and max_results == 100
        if self.on_search:
            self.on_search()
        if self.failure == "search":
            raise TimeoutError("private-token-value-never-persist")
        name, value = filter_string.split("`")[1].split(".", 1)[1], filter_string.split("'")[1]
        matches = sorted(k for k, v in self.rows.items() if v.data.tags.get("apf." + name) == value)
        start = int(page_token or 0)
        ids = matches[start:start + self.page_size]
        token = str(start + self.page_size) if start + self.page_size < len(matches) else None
        if self.cycle:
            token = "0"
        return Page([self.rows[k] for k in ids], token)
    def get_run(self, identity):
        self.calls.append("read")
        if self.failure == "read":
            raise ConnectionError("private-token-value-never-persist")
        return self.rows[identity]
    def __getattr__(self, name):
        if name in {"create_run", "set_tag", "log_metric", "log_artifact", "delete_run", "set_terminated", "create_experiment"}:
            pytest.fail("Reconciliation accessed a mutation capability")
        raise AttributeError(name)


@pytest.fixture
def case(accepted, tmp_path):
    c = accepted
    c.record = build(c)
    c.journal_root = tmp_path / "explicit-journal"
    c.journal_root.mkdir()
    c.destination = TrackingDestination("research_index", "experiment-1")
    c.fake = ReadOnlyClient()
    return c


def write_intent(c, *, status="linked", create=True, remote_id="run-1", **changes):
    intent = replace(journal.new_intent(c.record, "experiment-1"), status=status,
        phase="complete" if status == "linked" else "create" if status == "creating" else "intent",
        mlflow_run_id=remote_id, create_may_exist=create, **changes)
    (c.journal_root / (intent.linkage_key + ".json")).write_text(intent.canonical_json())
    return intent


def seed(c, identity="run-1", **changes):
    tags = dict(journal.metadata_payload(c.record, "experiment-1"))
    tags.update(changes)
    c.fake.rows[identity] = SimpleNamespace(info=SimpleNamespace(run_id=identity, experiment_id="experiment-1", lifecycle_stage="active"),
                                          data=SimpleNamespace(tags=tags))


def reconcile(c, **changes):
    args = dict(archive_root=c.publication.archive_root, archive_alias="synthetic_archive", journal_root=c.journal_root,
                destination=c.destination, client=BoundMlflowClient(c.destination, c.fake))
    args.update(changes)
    return api.reconcile_research_tracking(**args)


def statuses(report):
    return {v.status for v in report.scope_findings} | {v.status for row in report.references for v in row.findings}


def test_fully_consistent_references_are_immutable_and_read_only(case, monkeypatch):
    c = case
    write_intent(c)
    seed(c)
    before = snapshot(c.publication.archive_root), snapshot(c.journal_root)
    def forbidden(*a, **kw):
        pytest.fail("Read-only reconciliation tried to publish or acquire a journal transaction")
    monkeypatch.setattr(archive, "publish_research_bundle", forbidden)
    monkeypatch.setattr(journal.LinkageJournal, "transaction", forbidden)
    result = reconcile(c)
    assert result.exit_code == 0 and statuses(result) == {"linked_consistent"}
    row = result.references[0]
    assert row.linkage_key == c.record.linkage_key and row.record_id == c.record.record_id
    assert row.journal.revision == 1 and row.journal.schema_version == journal.SCHEMA_VERSION
    assert row.journal.fitting_attempt_id == c.receipt.attempt_id and row.remote_run_ids == ("run-1",)
    assert dict(row.scientific_checks)["prediction_decision_chronology"] == "incomplete"
    assert dict(row.scientific_checks)["historical_point_in_time"] == "not_assessed"
    assert dict(row.scientific_checks)["execution_eligibility"] == "not_assessed"
    assert dict(row.scientific_checks)["functional_reproduction"] == "not_assessed"
    assert snapshot(c.publication.archive_root) == before[0] and snapshot(c.journal_root) == before[1]
    assert not any(v.name.endswith(".lock") for v in c.journal_root.iterdir())
    with pytest.raises(FrozenInstanceError):
        row.publisher_outcome = "different"
    assert set(c.fake.calls) == {"search", "read"}
    assert reconcile(c) == result


def test_archive_without_intent_and_remote_reports_crash_gap_not_never_attempted(case):
    report = reconcile(case)
    assert {"missing_linkage_intent", "unsupported"} <= statuses(report)
    row = report.references[0]
    assert row.publication_history == "unknown" and row.publisher_outcome is None
    assert "historical_publication_outcome_unknown" in {v.code for v in row.findings}
    assert "separate authorization" in row.recommendation
    assert not list(case.journal_root.iterdir())


@pytest.mark.parametrize("status,create,expected", [("pending", False, "pending"), ("creating", True, "linkage_uncertain"),
    ("linkage_uncertain", True, "linkage_uncertain"), ("linked", True, "linkage_uncertain")])
def test_zero_remote_matches_never_authorize_creation(case, status, create, expected):
    write_intent(case, status=status, create=create, remote_id=None if status != "linked" else "run-1")
    before = snapshot(case.journal_root)
    report = reconcile(case)
    assert expected in statuses(report) and "linked_consistent" not in statuses(report)
    assert snapshot(case.journal_root) == before


def test_linked_journal_with_deleted_remote_run_is_stale(case):
    write_intent(case)
    report = reconcile(case)
    assert "stale_linkage" in statuses(report) and report.exit_code == 2


def test_remote_without_local_intent_is_not_adopted(case):
    seed(case)
    result = reconcile(case)
    assert "missing_linkage_intent" in statuses(result)
    assert "remote_reference_without_local_intent" in {v.code for v in result.references[0].findings}
    assert result.references[0].record_id is None and not list(case.journal_root.iterdir())


def test_duplicate_remote_pages_are_all_inspected(case):
    write_intent(case)
    seed(case)
    seed(case, "run-2")
    case.fake.page_size = 1
    result = reconcile(case)
    assert "duplicate" in statuses(result) and "linked_consistent" not in statuses(result)
    assert result.references[0].remote_run_ids == ("run-1", "run-2")
    assert case.fake.calls.count("read") == 2


@pytest.mark.parametrize("field", ["apf.record_id", "apf.linkage_key", "apf.bundle_id", "apf.protocol_id", "apf.training_input_id",
    "apf.receipt_id", "apf.fitted_model_id", "apf.temporal_manifest_id", "apf.prediction_capture_fingerprint", "apf.ticker",
    "apf.fitting_attempt_id", "apf.archive_acceptance_policy", "apf.historical_pit_status", "apf.archive_alias",
    "apf.tracking_destination_id", "apf.tracking_experiment_id"])
def test_any_remote_identity_or_limitation_conflict_rejects_success(case, field):
    write_intent(case)
    seed(case, **{field: "private-token-value-never-persist"})
    result = reconcile(case)
    assert "conflict" in statuses(result) and "linked_consistent" not in statuses(result)
    text = json.dumps(result.to_dict())
    assert "private-token-value-never-persist" not in text


@pytest.mark.parametrize("fault", ["corrupt", "missing", "symlink", "special"])
def test_invalid_or_missing_archives_never_produce_consistent_linkage(case, tmp_path, fault):
    c = case
    write_intent(c)
    seed(c)
    path = c.published.directory
    # Fault setup changes only this test's disposable synthetic archive.
    c.publication.archive_root.chmod(0o700)
    for directory in [path] + [v for v in path.rglob("*") if v.is_dir()]:
        directory.chmod(0o700)
    if fault == "corrupt":
        (path / "receipt.json").chmod(0o600)
        (path / "receipt.json").write_bytes(b"corrupt")
    elif fault == "missing":
        shutil.rmtree(path)
    elif fault == "symlink":
        moved = tmp_path / "outside-archive"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=True)
    else:
        shutil.rmtree(path)
        os.mkfifo(path)
    result = reconcile(c)
    assert statuses(result) & {"invalid_archive", "stale_linkage"}
    assert "linked_consistent" not in statuses(result)


@pytest.mark.parametrize("fault", ["version", "revision", "symlink", "filename", "expected_revision"])
def test_journal_validation_and_revision_assertions_are_read_only(case, tmp_path, fault):
    c = case
    entry = write_intent(c)
    seed(c)
    path = c.journal_root / (entry.linkage_key + ".json")
    raw = json.loads(path.read_text())
    args = {}
    if fault in {"version", "revision"}:
        raw["schema_version" if fault == "version" else "revision"] = "unsupported" if fault == "version" else 0
        path.write_text(json.dumps(raw))
    elif fault == "symlink":
        outside = tmp_path / "outside-journal"
        path.rename(outside)
        path.symlink_to(outside)
    elif fault == "filename":
        path.rename(c.journal_root / ("apf-research-linkage-v1-" + "0" * 64 + ".json"))
    else:
        args["expected_revisions"] = ((entry.linkage_key, 2),)
    before = {v.name: v.read_bytes() for v in c.journal_root.iterdir()}
    report = reconcile(c, **args)
    assert statuses(report) & {"unsupported", "conflict", "operational_error"}
    assert "linked_consistent" not in statuses(report)
    assert {v.name: v.read_bytes() for v in c.journal_root.iterdir()} == before


def test_concurrent_atomic_journal_replacement_marks_snapshot_unstable(case):
    c = case
    entry = write_intent(c)
    seed(c)
    changed = []
    def writer():
        if not changed:
            replacement = replace(entry, revision=2)
            stage = c.journal_root / "writer-owned-stage"
            stage.write_text(replacement.canonical_json())
            stage.replace(c.journal_root / (entry.linkage_key + ".json"))
            changed.append(True)
    c.fake.on_search = writer
    report = reconcile(c)
    assert "operational_error" in statuses(report) and "linked_consistent" not in statuses(report)
    assert "unstable_journal_snapshot" in {v.code for v in report.scope_findings}


@pytest.mark.parametrize("fault", ["destination", "experiment", "alias"])
def test_wrong_scope_cannot_be_classified_consistent(case, fault):
    c = case
    write_intent(c)
    seed(c)
    if fault == "alias":
        result = reconcile(c, archive_alias="different_archive")
    else:
        destination = TrackingDestination("different_index" if fault == "destination" else "research_index",
                                          "different-experiment" if fault == "experiment" else "experiment-1")
        result = reconcile(c, destination=destination, client=None)
    assert "conflict" in statuses(result) and "linked_consistent" not in statuses(result)


def test_relocation_preserves_references_without_fabricating_publication(case, tmp_path):
    c = case
    write_intent(c)
    seed(c)
    first = reconcile(c)
    relocated = tmp_path / "relocated-root"
    c.publication.archive_root.rename(relocated)
    second = reconcile(c, archive_root=relocated)
    assert first == second and first.exit_code == 0


def test_explicit_uncertain_publisher_history_is_never_resolved_by_checksums(case):
    write_intent(case)
    seed(case)
    result = reconcile(case, publication_outcomes=((case.record.bundle_id, "published_durability_uncertain"),))
    assert "durability_uncertain" in statuses(result) and "linked_consistent" not in statuses(result)
    assert result.references[0].publication_history == "caller_declared"


@pytest.mark.parametrize("boundary", ["search", "read"])
def test_remote_operational_failures_do_not_expose_messages(case, boundary):
    write_intent(case)
    seed(case)
    case.fake.failure = boundary
    report = reconcile(case)
    assert "operational_error" in statuses(report) and report.exit_code == 4
    text = json.dumps(report.to_dict())
    assert "private-token-value" not in text and str(case.publication.archive_root) not in text
    assert str(case.journal_root) not in text


@pytest.mark.parametrize("limit", ["pages", "runs", "inventory", "bundles", "journals"])
def test_resource_limits_fail_closed(case, limit):
    c = case
    write_intent(c)
    seed(c)
    seed(c, "run-2")
    kwargs = {}
    if limit == "pages":
        c.fake.page_size = 1
        c.fake.cycle = True
        kwargs["max_search_pages"] = 1
    elif limit == "runs":
        kwargs["max_remote_runs"] = 1
    elif limit == "inventory":
        kwargs["max_inventory_entries"] = 1
        (c.journal_root / "unrelated-explicit-root-entry").write_text("ignored")
    elif limit == "bundles":
        kwargs["max_bundles"] = 1
        (c.publication.archive_root / ("apf-research-bundle-v1-" + "0" * 64)).mkdir()
    else:
        kwargs["max_journals"] = 1
        (c.journal_root / ("apf-research-linkage-v1-" + "0" * 64 + ".json")).write_text("{}")
    result = reconcile(c, limits=api.ReconciliationLimits(**kwargs))
    assert "operational_error" in statuses(result) and result.exit_code == 4


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts/reconcile_research_tracking.py"
    spec = importlib.util.spec_from_file_location("apf_reconciliation_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_args(c):
    return ["--archive-root", str(c.publication.archive_root), "--archive-alias", "synthetic_archive",
            "--journal-root", str(c.journal_root), "--tracking-destination", "research_index", "--experiment-id", "experiment-1"]


def test_cli_offline_is_explicit_and_read_only(case, capsys):
    write_intent(case)
    before = snapshot(case.publication.archive_root), snapshot(case.journal_root)
    code = cli_module().main(cli_args(case))
    result = json.loads(capsys.readouterr().out)
    assert code == 3 and result["remote_evidence"] == "unavailable"
    assert statuses(reconcile(case, client=None)) == {"unsupported"}
    assert snapshot(case.publication.archive_root) == before[0] and snapshot(case.journal_root) == before[1]


def test_cli_snapshot_is_not_current_remote_certification(case, tmp_path, capsys):
    c = case
    write_intent(c)
    seed(c)
    path = tmp_path / "explicit-remote-snapshot.json"
    row = c.fake.rows["run-1"]
    path.write_text(json.dumps({"schema_version": "apf-research-remote-snapshot-v1", "destination_id": "research_index",
        "experiment_id": "experiment-1", "runs": [{"run_id": "run-1", "experiment_id": "experiment-1", "lifecycle_stage": "active", "tags": row.data.tags}]}))
    output = tmp_path / "explicit-report.json"
    before = snapshot(c.publication.archive_root), snapshot(c.journal_root)
    assert cli_module().main(cli_args(c) + ["--remote-snapshot", str(path), "--json-report", str(output)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text()) and result["remote_evidence"] == "offline_snapshot"
    assert "linked_consistent" not in {v["status"] for r in result["references"] for v in r["findings"]}
    assert cli_module().main(cli_args(c) + ["--json-report", str(output)]) == 4
    capsys.readouterr()
    assert snapshot(c.publication.archive_root) == before[0] and snapshot(c.journal_root) == before[1]


def test_cli_exit_codes_and_report_destination_protection(case, capsys):
    c = case
    assert cli_module().main(cli_args(c)) == 2
    capsys.readouterr()
    assert cli_module().main(cli_args(c) + ["--json-report", str(c.journal_root / "report.json")]) == 4
    capsys.readouterr()
    assert not (c.journal_root / "report.json").exists()
    assert cli_module().main(cli_args(c) + ["--archive-root", "relative"]) == 4
    assert "relative" not in capsys.readouterr().out


def test_empty_explicit_scope_and_stale_remote_reference_are_deterministic(case):
    c = case
    write_intent(c)
    seed(c)
    seed(c, "orphan", **{"apf.bundle_id": "apf-research-bundle-v1-" + "0" * 64, "apf.linkage_key": "unrelated"})
    result = reconcile(c)
    assert "stale_linkage" in statuses(result)
    assert result == reconcile(c)
    assert [r.bundle_id for r in result.references] == sorted(r.bundle_id for r in result.references)

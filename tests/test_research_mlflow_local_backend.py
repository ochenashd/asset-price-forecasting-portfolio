"""Real MLflow, exclusively new local SQLite/artifact stores and data-only archives.

No fluent tracking, model loading, training, inference or strategy execution.
Fault wrappers delegate normal requests to the real client without patching its
methods. Mutations seed test-local backend faults only; reconciliation is read-only.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
from types import SimpleNamespace

import mlflow
from mlflow.tracking import MlflowClient
import pytest
import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import make_url

from test_research_tracking_record import accepted, inputs, source_inputs, research, scientific_bundle, build
from test_research_bundle import snapshot
from src.tracking import research_mlflow as linkage
from src.tracking import research_linkage_journal as journal
from src.tracking import research_reconciliation as reconciliation
from src.tracking.research_mlflow_client import ResearchMlflowClient

REPOSITORY = Path(__file__).resolve().parents[1]


def outside_repository(path):
    path = Path(os.fsdecode(path)).resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise AssertionError("Backend test paths must be outside the repository")
    return path


@contextmanager
def isolated_local_backend(root):
    """Own one empty store and restore guards/state on every exit path.

    MLflow 3.14 thread-local state is swapped, not ended against an ambient
    backend. Python audit hooks cannot be removed; each hook becomes inert in
    finally. Only the newly owned engine is disposed/removed from its cache.
    """
    import mlflow.tracking.fluent as fluent
    import mlflow.tracking._tracking_service.utils as tracking_state
    import mlflow.tracking._model_registry.utils as registry_state
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
    from mlflow.utils.thread_utils import ThreadLocalVariable
    from src.evaluation.execution.engine import ExecutionEngineV1
    from src.evaluation.backtest import Backtester
    import xgboost
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import RobustScaler

    root = outside_repository(root)
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("Backend root must be new and empty.")
    database = root / "backend.sqlite"
    artifacts = root / "artifacts"
    artifacts.mkdir()
    uri = "sqlite:///" + str(database)
    active = {"guard": True}
    engine = None
    connect = sqlite3.connect
    create_engine = sqlalchemy.create_engine

    def forbidden(*args, **kwargs):
        pytest.fail("Forbidden global tracking, network or nonlocal database access")

    def audit_event(name, args):
        if not active["guard"]:
            return
        if name == "sqlite3.connect":
            assert outside_repository(args[0]) == database, "Unexpected SQLite connection"
        if name in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
            forbidden()
        if name == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            target = Path(os.fsdecode(args[0])).resolve()
            historical = target.name in {"mlflow.db", "mlflow.db.backup"} or "mlartifacts" in target.parts
            if historical and root not in target.parents:
                pytest.fail("Historical MLflow file access is forbidden")

    def confined_connect(target, *args, **kwargs):
        assert outside_repository(target) == database, "Unexpected SQLite store"
        return connect(target, *args, **kwargs)

    def confined_engine(url, *args, **kwargs):
        parsed = make_url(url)
        assert parsed.get_backend_name() == "sqlite" and outside_repository(parsed.database) == database
        return create_engine(url, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        try:
            patch.chdir(root)
            for key, value in {
                "MLFLOW_DISABLE_TELEMETRY": "true", "MLFLOW_ENABLE_WORKSPACES": "false",
                "MLFLOW_TRACKING_URI": uri, "MLFLOW_REGISTRY_URI": uri,
                "_MLFLOW_SERVER_ARTIFACT_ROOT": artifacts.as_uri(),
                "MPLCONFIGDIR": str(root / "matplotlib-cache"),
            }.items():
                patch.setenv(key, value)
            for key in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME"):
                # Always record the original value, including absence, before
                # clearing it. This also restores direct mutations by a test.
                patch.setenv(key, "")
                os.environ.pop(key, None)
            patch.setattr(tracking_state, "_tracking_uri", uri)
            patch.setattr(registry_state, "_registry_uri", uri)
            patch.setattr(fluent, "_active_experiment_id", None)
            patch.setattr(fluent, "_active_run_stack", ThreadLocalVariable(lambda: []))
            patch.setattr(fluent, "_last_active_run_id", ThreadLocalVariable(lambda: None))
            patch.setattr(fluent, "_last_logged_model_id", ThreadLocalVariable(lambda: None))
            for name in ("start_run", "end_run", "active_run", "set_tracking_uri", "get_tracking_uri", "set_experiment",
                         "create_experiment", "log_param", "log_params", "log_metrics", "log_metric", "log_artifact", "log_figure"):
                patch.setattr(mlflow, name, forbidden)
            patch.setattr(socket.socket, "connect", forbidden)
            patch.setattr(socket, "create_connection", forbidden)
            patch.setattr(socket, "getaddrinfo", forbidden)
            patch.setattr(sqlite3, "connect", confined_connect)
            patch.setattr(sqlite3.dbapi2, "connect", confined_connect)
            patch.setattr(sqlalchemy, "create_engine", confined_engine)
            patch.setattr(ExecutionEngineV1, "run", forbidden)
            patch.setattr(Backtester, "run_backtest", forbidden)
            patch.setattr(Backtester, "run_5d_class_backtest", forbidden)
            if sys.version_info[:2] == (3, 11):
                for cls in (xgboost.XGBClassifier, xgboost.XGBRegressor, RandomForestClassifier, RandomForestRegressor):
                    for name in ("fit", "predict", "predict_proba"):
                        if hasattr(cls, name):
                            patch.setattr(cls, name, forbidden)
                patch.setattr(RobustScaler, "fit", forbidden)
                patch.setattr(xgboost.Booster, "load_model", forbidden)
                patch.setattr(xgboost.Booster, "predict", forbidden)
                patch.setattr(xgboost.Booster, "inplace_predict", forbidden)
            sys.addaudithook(audit_event)
            real = MlflowClient(tracking_uri=uri, registry_uri=uri, workspace_store_uri=uri)
            engine = real._tracking_client.store.engine
            assert real.tracking_uri == uri and real._registry_uri == uri
            assert real._tracking_client.store.artifact_root_uri == artifacts.as_uri()
            yield SimpleNamespace(root=root, database=database, artifacts=artifacts, uri=uri,
                                  real=real, audit_state=active)
        finally:
            owned_engine = engine if engine is not None else SqlAlchemyStore._engine_map.get(uri)
            try:
                if owned_engine is not None:
                    owned_engine.dispose()
            finally:
                active["guard"] = False
                if SqlAlchemyStore._engine_map.get(uri) is owned_engine:
                    SqlAlchemyStore._engine_map.pop(uri, None)
                # MonkeyPatch.context restores all functions, cwd, environment
                # and thread-local/global URI objects, even if disposal raises.


@pytest.fixture
def local_backend(tmp_path_factory):
    """Function-scoped guards and one fresh backend per guarded test."""
    root = outside_repository(tmp_path_factory.mktemp("apf-real-mlflow"))
    with isolated_local_backend(root) as backend:
        yield backend

@pytest.fixture(autouse=True)
def safe_test_root(tmp_path):
    outside_repository(tmp_path)


@pytest.fixture
def case(local_backend, accepted, tmp_path):
    b, c = local_backend, accepted
    for path in (c.publication.archive_root, c.published.directory, tmp_path):
        outside_repository(path)
    artifact_path = outside_repository(b.artifacts / tmp_path.name)
    artifact_path.mkdir()
    experiment_id = b.real.create_experiment("apf-local-" + tmp_path.name, artifact_location=artifact_path.as_uri())
    assert b.real.get_experiment(experiment_id).artifact_location == artifact_path.as_uri()
    c.backend = b
    c.record = build(c)
    c.real = b.real
    c.client = ResearchMlflowClient(c.real)
    c.destination = linkage.TrackingDestination("research_index", experiment_id)
    c.journal_root = tmp_path / "linkage-journal"
    c.journal_root.mkdir()
    return c


def link(c, client=None, **changes):
    args = dict(client=linkage.BoundMlflowClient(c.destination, c.client if client is None else client),
                destination=c.destination, journal_root=c.journal_root, enabled=True)
    args.update(changes)
    return linkage.link_research_record(c.record, **args)


def reconcile(c, client=None, **changes):
    args = dict(archive_root=c.publication.archive_root, archive_alias=c.record.archive_alias,
                journal_root=c.journal_root, destination=c.destination,
                client=linkage.BoundMlflowClient(c.destination, c.client if client is None else client))
    args.update(changes)
    return reconciliation.reconcile_research_tracking(**args)


def tags(c):
    return dict(journal.metadata_payload(c.record, c.destination.experiment_id))


def create(c, **changes):
    return c.real.create_run(c.destination.experiment_id, tags={**tags(c), **changes})


def states(report):
    return {v.status for v in report.scope_findings} | {v.status for row in report.references for v in row.findings}


def test_direct_real_client_linkage_readback_and_technical_status(case):
    c = case
    before = snapshot(c.publication.archive_root)
    result = link(c, client=c.real)
    assert result.status == "linked", result
    run = c.real.get_run(result.intent.mlflow_run_id)
    assert run.info.experiment_id == c.destination.experiment_id
    assert run.info.run_id == result.intent.mlflow_run_id and run.info.lifecycle_stage == "active"
    assert run.info.status == "RUNNING" and run.info.end_time is None
    assert {k: v for k, v in run.data.tags.items() if k.startswith("apf.")} == tags(c)
    assert run.data.params == {} and run.data.metrics == {}
    assert result.intent.record_id == c.record.record_id and result.intent.linkage_key == c.record.linkage_key
    assert snapshot(c.publication.archive_root) == before
    assert link(c).intent.mlflow_run_id == run.info.run_id
    assert len(c.real.search_runs([c.destination.experiment_id], run_view_type=3)) == 1
    assert reconcile(c).exit_code == 0


def test_real_sql_pagination_is_consumed_without_losing_duplicate_matches(case):
    c = case
    for _ in range(101):
        create(c)
    page = c.real.search_runs([c.destination.experiment_id], filter_string=f"tags.`apf.linkage_key` = '{c.record.linkage_key}'",
                             run_view_type=3, max_results=100)
    assert isinstance(page, list) and len(page) == 100 and page.token is not None
    next_page = c.real.search_runs([c.destination.experiment_id], filter_string=f"tags.`apf.linkage_key` = '{c.record.linkage_key}'",
                                  run_view_type=3, max_results=100, page_token=page.token)
    assert len(next_page) == 1 and len({r.info.run_id for r in page + next_page}) == 101
    result = link(c)
    assert result.status == "duplicate", (type(page.token).__name__, result)
    assert "duplicate" in states(reconcile(c))


def test_real_missing_run_is_classified_stale_not_unavailable(case):
    c = case
    entry = replace(journal.new_intent(c.record, c.destination.experiment_id), status="linked", phase="complete",
                    create_may_exist=True, mlflow_run_id="0" * 32)
    (c.journal_root / (entry.linkage_key + ".json")).write_text(entry.canonical_json())
    result = reconcile(c)
    assert "stale_linkage" in states(result) and "operational_error" not in states(result), result


class FaultBoundary:
    """Fault injection around unmodified real methods; no retry algorithm."""
    def __init__(self, client, fault=None, hidden=False):
        self.client, self.fault, self.hidden = client, fault, hidden
        self.calls = []
    def search_runs(self, **kwargs):
        self.calls.append("search")
        if self.fault == "search":
            raise TimeoutError("synthetic-private-error-never-persist")
        page = self.client.search_runs(**kwargs)
        return [] if self.hidden else page
    def create_run(self, **kwargs):
        self.calls.append("create")
        if self.fault == "precreate":
            raise ConnectionError("synthetic-private-error-never-persist")
        run = self.client.create_run(**kwargs)
        if self.fault == "lost_response":
            raise TimeoutError("synthetic-private-error-never-persist")
        return run
    def get_run(self, identity):
        self.calls.append("read")
        run = self.client.get_run(identity)
        if self.fault == "read":
            raise TimeoutError("synthetic-private-error-never-persist")
        return run


def saved(c):
    return journal.LinkageIntent.from_json((c.journal_root / (c.record.linkage_key + ".json")).read_bytes())


def remote_snapshot(c):
    return {r.info.run_id: {"info": {**{name: getattr(r.info, name) for name in
                ("run_id", "experiment_id", "status", "start_time", "end_time", "lifecycle_stage", "artifact_uri", "run_name")},
                "protobuf_sha256": hashlib.sha256(r.info.to_proto().SerializeToString(deterministic=True)).hexdigest()},
                           "tags": dict(r.data.tags),
                           "params": dict(r.data.params), "metrics": dict(r.data.metrics)}
            for r in c.real.search_runs([c.destination.experiment_id], run_view_type=3, max_results=100)}


def artifact_hashes(root):
    return {name: (hashlib.sha256(data).hexdigest(), mode, modified)
            for name, (data, mode, modified) in snapshot(root).items()}


def assert_reconciliation_read_only(c, **kwargs):
    before = artifact_hashes(c.publication.archive_root), artifact_hashes(c.journal_root), remote_snapshot(c)
    database_hash = hashlib.sha256(c.backend.database.read_bytes()).hexdigest()
    statements = []
    def observed(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lstrip().split(None, 1)[0].upper())
    engine = c.real._tracking_client.store.engine
    event.listen(engine, "before_cursor_execute", observed)
    try:
        result = reconcile(c, **kwargs)
    finally:
        event.remove(engine, "before_cursor_execute", observed)
    assert not (set(statements) & {"INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "ALTER", "DROP", "VACUUM"})
    assert artifact_hashes(c.publication.archive_root) == before[0]
    assert artifact_hashes(c.journal_root) == before[1]
    assert remote_snapshot(c) == before[2]
    assert hashlib.sha256(c.backend.database.read_bytes()).hexdigest() == database_hash
    return result


def test_independent_client_and_journal_restart_reuse_exact_run(case):
    c = case
    first = link(c)
    assert first.status == "linked" and first.intent == saved(c)
    other = MlflowClient(tracking_uri=c.backend.uri, registry_uri=c.backend.uri, workspace_store_uri=c.backend.uri)
    second = link(c, client=ResearchMlflowClient(other))
    assert second.status == "linked" and second.intent.mlflow_run_id == first.intent.mlflow_run_id
    assert second.intent.revision > first.intent.revision
    assert len(other.search_runs([c.destination.experiment_id], run_view_type=3)) == 1
    assert assert_reconciliation_read_only(c, client=ResearchMlflowClient(other)).exit_code == 0


def test_exact_destination_and_experiment_binding_prevents_requests(case):
    c = case
    wrong = linkage.TrackingDestination("research_index", "incorrect-experiment")
    wrapped = FaultBoundary(c.client)
    result = link(c, client=wrapped, destination=wrong)
    assert result.status == "rejected" and not wrapped.calls and not list(c.journal_root.iterdir())
    assert c.real.search_runs([c.destination.experiment_id]) == []
    different = replace(c.record, tracking_destination_id="another_index")
    assert linkage.link_research_record(different, client=linkage.BoundMlflowClient(c.destination, c.client),
        destination=c.destination, journal_root=c.journal_root, enabled=True).status == "rejected"


@pytest.mark.parametrize("field", ["apf.record_id", "apf.fitted_model_id", "apf.archive_alias", "apf.unknown"])
def test_real_backend_conflicting_metadata_is_detected_not_repaired(case, field):
    c = case
    run = create(c, **{field: "synthetic-conflict"})
    before = remote_snapshot(c)
    outcome = link(c)
    assert outcome.status == "conflict" and outcome.intent.mlflow_run_id is None
    assert remote_snapshot(c) == before and len(before) == 1
    assert c.real.get_run(run.info.run_id).data.tags[field] == "synthetic-conflict"
    assert "conflict" in states(assert_reconciliation_read_only(c))


def test_real_tag_absence_and_tampering_are_not_silently_repaired(case):
    c = case
    run = create(c)
    c.real.delete_tag(run.info.run_id, "apf.verification.temporal_provenance")
    assert link(c).status == "failed"
    assert "apf.verification.temporal_provenance" not in c.real.get_run(run.info.run_id).data.tags
    assert "conflict" in states(assert_reconciliation_read_only(c))
    c.real.set_tag(run.info.run_id, "apf.historical_pit_status", "unsupported-claim", synchronous=True)
    assert link(c).status == "conflict"
    assert c.real.get_run(run.info.run_id).data.tags["apf.historical_pit_status"] == "unsupported-claim"


@pytest.mark.parametrize("fault", ["search", "precreate", "lost_response", "read"])
def test_real_client_faults_preserve_uncertainty_without_second_creation(case, fault):
    c = case
    boundary = FaultBoundary(c.client, fault=fault)
    result = link(c, client=boundary)
    expected = "failed" if fault == "search" else "linkage_uncertain"
    assert result.status == expected and result.intent == saved(c)
    assert "synthetic-private-error" not in saved(c).canonical_json()
    count = len(c.real.search_runs([c.destination.experiment_id], run_view_type=3))
    assert count == (1 if fault in {"lost_response", "read"} else 0)
    boundary.fault = None
    repeated = link(c, client=boundary)
    assert repeated.status == ("linkage_uncertain" if fault == "precreate" else "linked")
    assert boundary.calls.count("create") == 1
    assert len(c.real.search_runs([c.destination.experiment_id], run_view_type=3)) == 1 - int(fault == "precreate")
    assert c.record.record_id == saved(c).record_id and c.record.linkage_key == saved(c).linkage_key


def test_lost_create_response_zero_visibility_then_later_discovery(case):
    c = case
    boundary = FaultBoundary(c.client, fault="lost_response")
    assert link(c, client=boundary).status == "linkage_uncertain"
    original_id = next(iter(remote_snapshot(c)))
    boundary.fault, boundary.hidden = None, True
    assert link(c, client=boundary).status == "linkage_uncertain"
    assert saved(c).create_may_exist and saved(c).mlflow_run_id is None
    report = assert_reconciliation_read_only(c, client=boundary)
    assert "linkage_uncertain" in states(report) and "linked_consistent" not in states(report)
    boundary.hidden = False
    recovered = link(c, client=boundary)
    assert recovered.status == "linked" and recovered.intent.mlflow_run_id == original_id
    assert boundary.calls.count("create") == 1 and len(remote_snapshot(c)) == 1


def test_missing_journal_with_real_remote_reference_is_not_automatically_adopted(case):
    c = case
    create(c)
    assert list(c.journal_root.iterdir()) == []
    result = assert_reconciliation_read_only(c)
    assert "missing_linkage_intent" in states(result) and "linked_consistent" not in states(result)
    assert list(c.journal_root.iterdir()) == []


def test_real_remote_missing_intent_and_unknown_publication_history(case):
    c = case
    result = assert_reconciliation_read_only(c)
    assert {"missing_linkage_intent", "unsupported"} <= states(result)
    assert result.references[0].publication_history == "unknown"
    assert result.references[0].publisher_outcome is None


def test_pending_local_intent_stays_pending_with_real_empty_search(case):
    c = case
    pending = journal.new_intent(c.record, c.destination.experiment_id)
    (c.journal_root / (pending.linkage_key + ".json")).write_text(pending.canonical_json())
    result = assert_reconciliation_read_only(c)
    assert "pending" in states(result) and saved(c) == pending
    assert len(remote_snapshot(c)) == 0


def test_real_duplicate_runs_are_reported_without_mutating_remote_state(case):
    c = case
    create(c)
    create(c)
    before = remote_snapshot(c)
    assert link(c).status == "duplicate"
    assert "duplicate" in states(assert_reconciliation_read_only(c))
    assert remote_snapshot(c) == before
    assert all(r["info"]["status"] == "RUNNING" for r in before.values())


def test_deleted_real_run_is_stale_without_recreating_or_terminating(case):
    c = case
    original = link(c)
    assert original.status == "linked"
    c.real.delete_run(original.intent.mlflow_run_id)
    result = assert_reconciliation_read_only(c)
    assert "stale_linkage" in states(result) and "linked_consistent" not in states(result)
    assert len(c.real.search_runs([c.destination.experiment_id], run_view_type=3)) == 1
    assert c.real.get_run(original.intent.mlflow_run_id).info.status == "RUNNING"


def test_invalid_synthetic_archive_is_rejected_without_repair(case):
    c = case
    assert link(c).status == "linked"
    receipt = c.published.directory / "receipt.json"
    receipt.chmod(0o600)
    receipt.write_bytes(b"deliberately-corrupt-synthetic-fixture")
    result = assert_reconciliation_read_only(c)
    assert "invalid_archive" in states(result) and "linked_consistent" not in states(result)
    assert receipt.read_bytes() == b"deliberately-corrupt-synthetic-fixture"


def test_real_backend_artifact_and_scope_configuration_stays_test_local(case):
    c = case
    result = link(c)
    assert result.status == "linked"
    run = c.real.get_run(result.intent.mlflow_run_id)
    from urllib.parse import urlparse, unquote
    artifact_path = outside_repository(unquote(urlparse(run.info.artifact_uri).path))
    assert artifact_path.is_relative_to(c.backend.artifacts)
    assert c.backend.database.is_file() and c.real.tracking_uri == c.backend.uri
    assert all(key.startswith("apf.") or key.startswith("mlflow.") for key in run.data.tags)
    for field, value in {"apf.decision_time_evidence": "incomplete", "apf.historical_pit_status": "unknown",
                         "apf.execution_eligibility_status": "unassessed", "apf.functional_replay_status": "not_assessed"}.items():
        assert run.data.tags[field] == value
    assert run.info.status == "RUNNING" and run.info.end_time is None
    assert not any(path.is_file() for path in c.backend.artifacts.rglob("*"))


@pytest.mark.parametrize("token", [b"base64-token", "already-text", None])
def test_adapter_only_changes_token_representation(token):
    class Page(list):
        pass
    run = object()
    page = Page([run])
    page.token = token
    client = SimpleNamespace(search_runs=lambda **kw: page)
    adapted = ResearchMlflowClient(client).search_runs(experiment_ids=["explicit"])
    assert adapted[0] is run and page.token == token
    assert adapted.token == (token.decode("ascii") if type(token) is bytes else token)


@pytest.mark.parametrize("token", [b"\xff", b"", "", 123, "x" * 4097])
def test_adapter_rejects_unsupported_tokens_without_guessing(token):
    class Page(list):
        pass
    page = Page()
    page.token = token
    with pytest.raises(ValueError, match="unsupported_mlflow_page_token"):
        ResearchMlflowClient(SimpleNamespace(search_runs=lambda **kw: page)).search_runs()


@pytest.mark.parametrize("code", ["RESOURCE_DOES_NOT_EXIST", "PERMISSION_DENIED", None])
def test_adapter_only_translates_structured_missing_run_errors(code):
    class BoundaryError(Exception):
        pass
    error = BoundaryError("synthetic-private-message")
    error.error_code = code
    def failed(identity):
        raise error
    adapted = ResearchMlflowClient(SimpleNamespace(get_run=failed))
    expected = FileNotFoundError if code == "RESOURCE_DOES_NOT_EXIST" else BoundaryError
    with pytest.raises(expected) as caught:
        adapted.get_run("explicit-run")
    if code == "RESOURCE_DOES_NOT_EXIST":
        assert str(caught.value) == "research_remote_run_not_found"
    else:
        assert caught.value is error

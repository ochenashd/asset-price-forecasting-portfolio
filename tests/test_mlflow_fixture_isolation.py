"""Fixture isolation without new fitting, inference or historical store access."""
import os
from pathlib import Path
import socket
import sqlite3
import sys
from types import SimpleNamespace

import mlflow
import mlflow.tracking.fluent as fluent
import mlflow.tracking._tracking_service.utils as tracking_state
import mlflow.tracking._model_registry.utils as registry_state
from mlflow.utils.thread_utils import ThreadLocalVariable
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
import pytest
import sqlalchemy
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

import test_research_mlflow_local_backend as guarded
from test_research_tracking_record import accepted
from test_research_bundle_verification import scientific_bundle
from test_research_input_adapter import inputs, research
from src.research.bundle import build_research_manifest
from src.evaluation.execution.engine import ExecutionEngineV1

ENVIRONMENT = ("MLFLOW_DISABLE_TELEMETRY", "MLFLOW_ENABLE_WORKSPACES", "MLFLOW_TRACKING_URI",
    "MLFLOW_REGISTRY_URI", "MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME",
    "_MLFLOW_SERVER_ARTIFACT_ROOT", "MPLCONFIGDIR")


def fresh(tmp_path, name="backend"):
    root = tmp_path / name
    root.mkdir()
    return root


def state():
    """Keep original values in memory only; never display ambient URI values."""
    return dict(cwd=Path.cwd(), env={key: os.environ.get(key) for key in ENVIRONMENT},
        objects=(mlflow.start_run, mlflow.end_run, mlflow.active_run, mlflow.set_tracking_uri,
            socket.create_connection, socket.getaddrinfo, socket.socket.connect,
            sqlite3.connect, sqlalchemy.create_engine, RandomForestClassifier.fit,
            RobustScaler.fit, xgboost.XGBClassifier.fit, xgboost.XGBClassifier.predict,
            xgboost.Booster.load_model, ExecutionEngineV1.run, fluent._active_run_stack,
            fluent._last_active_run_id, fluent._last_logged_model_id),
        tracking=tracking_state._tracking_uri, registry=registry_state._registry_uri,
        experiment=fluent._active_experiment_id)


def restored(before):
    after = state()
    if after["cwd"] != before["cwd"]:
        pytest.fail("Working directory not restored", pytrace=False)
    if not all(a is b for a, b in zip(after["objects"], before["objects"])):
        pytest.fail("Patched function or state object not restored", pytrace=False)
    for key in ENVIRONMENT:
        if after["env"][key] != before["env"][key]:
            pytest.fail("Environment not restored: " + key, pytrace=False)
    for key in ("tracking", "registry", "experiment"):
        if after[key] != before[key]:
            pytest.fail("MLflow global state not restored: " + key, pytrace=False)


def test_fixture_scope_is_function():
    assert guarded.local_backend._pytestfixturefunction.scope == "function"


@pytest.mark.parametrize("failure", [False, True])
def test_normal_and_exceptional_exit_restore_all_state(tmp_path, failure):
    before = state()
    uri = None
    class InjectedFailure(Exception):
        pass
    try:
        with guarded.isolated_local_backend(fresh(tmp_path)) as backend:
            uri = backend.uri
            assert backend.audit_state["guard"]
            assert Path.cwd() == backend.root
            assert fluent._active_run_stack.get() == []
            os.environ["MLFLOW_TRACKING_URI"] = "https://synthetic.invalid"
            os.environ["MLFLOW_RUN_ID"] = "synthetic-only"
            tracking_state._tracking_uri = "https://synthetic.invalid"
            registry_state._registry_uri = "https://synthetic.invalid"
            fluent._active_run_stack.get().append(object())
            fluent._last_active_run_id.set("synthetic-only")
            fluent._last_logged_model_id.set("synthetic-only")
            if failure:
                raise InjectedFailure()
    except InjectedFailure:
        assert failure
    assert not backend.audit_state["guard"]
    assert uri not in SqlAlchemyStore._engine_map
    restored(before)
    # Exercise the retired audit hook without making a real DNS/socket request.
    sys.audit("socket.getaddrinfo", "synthetic.invalid", 443, 0, 0, 0)


def test_setup_exception_restores_guards_and_environment(tmp_path, monkeypatch):
    before = state()
    def failed_client(**kwargs):
        assert kwargs["tracking_uri"].startswith("sqlite:///")
        raise RuntimeError("injected client setup failure")
    monkeypatch.setattr(guarded, "MlflowClient", failed_client)
    with pytest.raises(RuntimeError, match="injected client setup failure"):
        with guarded.isolated_local_backend(fresh(tmp_path)):
            pytest.fail("Setup failure must prevent yield")
    restored(before)
    sys.audit("socket.getaddrinfo", "synthetic.invalid", 443, 0, 0, 0)


def test_disposal_exception_still_restores_state(tmp_path, monkeypatch):
    before = state()
    with pytest.raises(RuntimeError, match="injected disposal failure"):
        with guarded.isolated_local_backend(fresh(tmp_path)) as backend:
            engine = backend.real._tracking_client.store.engine
            dispose = engine.dispose
            def failed_dispose():
                dispose()
                raise RuntimeError("injected disposal failure")
            monkeypatch.setattr(engine, "dispose", failed_dispose)
    assert not backend.audit_state["guard"]
    restored(before)


def test_ambient_uri_and_active_run_are_not_used_or_ended(tmp_path, monkeypatch):
    ambient = "https://synthetic.invalid"
    sentinel = object()
    stack = ThreadLocalVariable(lambda: [sentinel])
    monkeypatch.setattr(fluent, "_active_run_stack", stack)
    monkeypatch.setattr(tracking_state, "_tracking_uri", ambient)
    monkeypatch.setattr(registry_state, "_registry_uri", ambient)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", ambient)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", ambient)
    before = state()
    with guarded.isolated_local_backend(fresh(tmp_path)) as backend:
        assert backend.real.tracking_uri == backend.real._registry_uri == backend.uri
        assert tracking_state._tracking_uri == registry_state._registry_uri == backend.uri
        assert fluent._active_run_stack is not stack and fluent._active_run_stack.get() == []
    restored(before)
    assert fluent._active_run_stack is stack and stack.get()[0] is sentinel


def test_guard_blocks_training_inference_network_and_fluent_calls(tmp_path, monkeypatch):
    counts = dict(fit=0, predict=0, load=0)
    def unexpected(name):
        def call(*args, **kwargs):
            counts[name] += 1
            pytest.fail("Underlying model method was called")
        return call
    monkeypatch.setattr(RandomForestClassifier, "fit", unexpected("fit"))
    monkeypatch.setattr(xgboost.XGBClassifier, "predict", unexpected("predict"))
    monkeypatch.setattr(xgboost.Booster, "load_model", unexpected("load"))
    before = state()
    with guarded.isolated_local_backend(fresh(tmp_path)):
        for operation in (lambda: RandomForestClassifier.fit(None, None, None),
            lambda: xgboost.XGBClassifier.predict(None, None),
            lambda: xgboost.Booster.load_model(None, b"not a model"),
            lambda: ExecutionEngineV1.run(None, [], []),
            lambda: mlflow.start_run(), lambda: mlflow.set_tracking_uri("https://synthetic.invalid"),
            lambda: socket.create_connection(("synthetic.invalid", 443))):
            with pytest.raises(pytest.fail.Exception, match="Forbidden"):
                operation()
    assert counts == dict(fit=0, predict=0, load=0)
    restored(before)


def test_historical_database_and_artifact_reads_are_rejected(tmp_path):
    history = tmp_path / "historical"
    artifacts = history / "mlartifacts"
    artifacts.mkdir(parents=True)
    database = history / "mlflow.db"
    backup = history / "mlflow.db.backup"
    payload = artifacts / "opaque.bin"
    for p in (database, backup, payload):
        p.write_bytes(b"synthetic sentinel")
    with guarded.isolated_local_backend(fresh(tmp_path)):
        for target in (database, backup, payload):
            with pytest.raises(pytest.fail.Exception, match="Historical"):
                target.read_bytes()
        with pytest.raises(AssertionError, match="Unexpected SQLite"):
            sqlite3.connect(database)
        with pytest.raises(AssertionError):
            sqlalchemy.create_engine("sqlite:///" + str(database))
        with pytest.raises(AssertionError):
            sqlalchemy.create_engine("postgresql://synthetic.invalid/test")
    for p in (database, backup, payload):
        assert p.read_bytes() == b"synthetic sentinel"


def test_existing_backend_is_rejected_before_client_construction(tmp_path, monkeypatch):
    root = fresh(tmp_path)
    (root / "backend.sqlite").write_bytes(b"synthetic sentinel")
    def forbidden(**kwargs):
        pytest.fail("An existing store must never be bound")
    monkeypatch.setattr(guarded, "MlflowClient", forbidden)
    with pytest.raises(ValueError, match="new and empty"):
        with guarded.isolated_local_backend(root):
            pytest.fail("Existing backend accepted")
    assert (root / "backend.sqlite").read_bytes() == b"synthetic sentinel"


def test_consecutive_backends_are_distinct_and_have_own_artifact_roots(tmp_path):
    before = state()
    with guarded.isolated_local_backend(fresh(tmp_path, "first")) as first:
        first_uri = first.uri
        experiment = first.real.create_experiment("synthetic-first", artifact_location=first.artifacts.as_uri())
        assert first.real.get_experiment(experiment).artifact_location == first.artifacts.as_uri()
    restored(before)
    with guarded.isolated_local_backend(fresh(tmp_path, "second")) as second:
        assert second.uri != first_uri
        assert second.real._tracking_client.store.artifact_root_uri == second.artifacts.as_uri()
    restored(before)


def test_receipt_and_bundle_identities_do_not_depend_on_guard_order(scientific_bundle, tmp_path):
    _, artifacts, metadata, protocol, receipt = scientific_bundle
    before = (protocol.protocol_id, receipt.receipt_id, metadata.fitted_model_id,
              build_research_manifest(artifacts, metadata).bundle_id)
    with guarded.isolated_local_backend(fresh(tmp_path)):
        during = (protocol.protocol_id, receipt.receipt_id, metadata.fitted_model_id,
                  build_research_manifest(artifacts, metadata).bundle_id)
    after = (protocol.protocol_id, receipt.receipt_id, metadata.fitted_model_id,
             build_research_manifest(artifacts, metadata).bundle_id)
    assert before == during == after


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("prior_present", [False, True])
def test_demo_fixture_restores_artifact_environment_without_extra_fits(tmp_path, monkeypatch, failure, prior_present):
    import test_synthetic_research as demo_tests
    key = "_MLFLOW_SERVER_ARTIFACT_ROOT"
    if prior_present:
        monkeypatch.setenv(key, "synthetic-original")
    else:
        monkeypatch.delenv(key, raising=False)
    prior = os.environ.get(key)
    calls = []
    def synthetic_run(*args, **kwargs):
        calls.append(1)
        os.environ[key] = "synthetic-temporary"
        if failure:
            raise RuntimeError("injected demo failure")
        return {"synthetic": True}
    monkeypatch.setattr(demo_tests, "run_experiment", synthetic_run)
    factory = SimpleNamespace(mktemp=lambda _: tmp_path)
    fixture = demo_tests.completed.__wrapped__(factory)
    if failure:
        with pytest.raises(RuntimeError, match="injected demo failure"):
            next(fixture)
    else:
        assert next(fixture) == {"synthetic": True}
        fixture.close()
    if os.environ.get(key) != prior:
        pytest.fail("Demo fixture did not restore artifact environment", pytrace=False)
    assert calls == [1]

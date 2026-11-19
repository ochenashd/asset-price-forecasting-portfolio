"""Synthetic credential tests; no driver, database, provider or real MLflow use."""
import ast
import importlib.util
import logging
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import traceback

from omegaconf import OmegaConf
import pytest
import sqlalchemy
from sqlalchemy.engine import Engine, URL, make_url

from src.database import connection as api

ROOT = Path(__file__).resolve().parents[1]
SECRET = "synthetic-fixture-only:@/ ?#%+value"
ENV_KEYS = ("HOST", "PORT", "NAME", "USER", "PASSWORD", "ALLOW_PASSWORDLESS")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv("APF_DB_" + key, raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Real engine/network operations are forbidden")
    monkeypatch.setattr(sqlalchemy, "create_engine", forbidden)
    monkeypatch.setattr(api, "create_engine", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_missing_credentials_fail_before_engine_creation():
    with pytest.raises(api.DatabaseConfigurationError, match="password is required"):
        api.create_database_engine()


def test_explicit_environment_credentials(monkeypatch):
    values = {"HOST": "localhost", "PORT": "5433", "NAME": "synthetic_db",
              "USER": "synthetic_user", "PASSWORD": SECRET}
    for key, value in values.items():
        monkeypatch.setenv("APF_DB_" + key, value)
    config = api.DatabaseConfig.from_env()
    url = config.to_url()
    assert config.port == 5433 and url.host == "localhost"
    assert url.username == "synthetic_user" and url.database == "synthetic_db"
    assert url.password == SECRET


@pytest.mark.parametrize("value", [SECRET, "synthetic-only-space value", "synthetic-only:at@percent%slash/plus+"])
def test_reserved_password_characters_roundtrip_without_interpolation(value):
    config = api.DatabaseConfig(user="synthetic_user", password=value)
    url = config.to_url()
    assert isinstance(url, URL)
    decoded = make_url(url.render_as_string(hide_password=False))
    assert decoded.password == value and decoded.username == "synthetic_user"
    assert value not in str(url) and value not in repr(url) and value not in repr(config)
    assert "***" in str(url)


def test_engine_receives_url_object_and_silent_diagnostics(monkeypatch, caplog, capsys):
    captured = []
    def factory(url, **kwargs):
        captured.append((url, kwargs))
        return SimpleNamespace(url=url)
    monkeypatch.setattr(api, "create_engine", factory)
    config = api.DatabaseConfig(password=SECRET)
    with caplog.at_level(logging.DEBUG):
        engine = api.create_database_engine(config)
        logging.getLogger("synthetic-credential-check").debug("%s %r %s", config, engine, engine.url)
    url, kwargs = captured[0]
    assert isinstance(url, URL) and url.password == SECRET and kwargs["echo"] is False
    assert SECRET not in caplog.text and SECRET not in repr(engine)
    assert SECRET not in Engine.__repr__(engine)
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err


def test_engine_error_text_and_traceback_are_sanitized(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(api, "create_engine", failed)
    with pytest.raises(api.DatabaseConfigurationError) as caught:
        api.create_database_engine(api.DatabaseConfig(password=SECRET))
    text = "".join(traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__))
    assert SECRET not in str(caught.value) and SECRET not in text


@pytest.mark.parametrize("setting,value", [("port", SECRET), ("password", ""),
    ("allow_passwordless", SECRET), ("user", "")])
def test_invalid_configuration_never_echoes_values(setting, value):
    cfg = OmegaConf.create({"database": {"password": SECRET, setting: value}})
    with pytest.raises(api.DatabaseConfigurationError) as caught:
        api.create_engine_from_config(cfg)
    assert SECRET not in str(caught.value)


def test_explicit_passwordless_mode_and_no_implicit_fallback(monkeypatch):
    with pytest.raises(api.DatabaseConfigurationError):
        api.DatabaseConfig().to_url()
    monkeypatch.setenv("APF_DB_ALLOW_PASSWORDLESS", "true")
    config = api.DatabaseConfig.from_env()
    assert config.allow_passwordless is True and config.to_url().password is None
    monkeypatch.setenv("APF_DB_PASSWORD", "")
    with pytest.raises(api.DatabaseConfigurationError):
        api.DatabaseConfig.from_env().to_url()


def test_documented_config_precedence_and_null_environment_fallback(monkeypatch):
    monkeypatch.setenv("APF_DB_PASSWORD", SECRET)
    monkeypatch.setenv("APF_DB_USER", "environment_user")
    config = api.database_config_from_omegaconf(OmegaConf.create({"database": {
        "user": "explicit_user", "password": None, "allow_passwordless": False}}))
    assert config.user == "explicit_user" and config.password == SECRET
    explicit = api.database_config_from_omegaconf(OmegaConf.create({"database": {"password": "synthetic-explicit"}}))
    assert explicit.password == "synthetic-explicit"


def test_module_import_has_no_engine_or_network_side_effect(monkeypatch, capsys):
    name = "credential_import_probe"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/database/connection.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.DatabaseConfig().password is None
    assert capsys.readouterr().out == ""


def test_no_embedded_password_default_and_manual_cli_uses_environment():
    tree = ast.parse((ROOT / "src/database/connection.py").read_text())
    fields = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name) and n.target.id == "password"]
    assert len(fields) == 1
    assert isinstance(fields[0].value, ast.Call)
    assert any(k.arg == "default" and isinstance(k.value, ast.Constant) and k.value.value is None
               for k in fields[0].value.keywords)
    script = ast.parse((ROOT / "scripts/test_database.py").read_text())
    assert any(isinstance(n, ast.Call) and ast.unparse(n.func) == "DatabaseConfig.from_env" for n in ast.walk(script))


def test_mlflow_model_helper_never_logs_database_branch(monkeypatch, tmp_path, caplog):
    calls = []
    fake = SimpleNamespace(active_run=lambda: object(),
        log_param=lambda *a, **k: calls.append((a, k)),
        log_artifact=lambda *a, **k: calls.append((a, k)))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    spec = importlib.util.spec_from_file_location("credential_mlflow_probe", ROOT / "src/tracking/mlflow_utils.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cfg = OmegaConf.create({"database": {"password": SECRET}, "models": {"xgboost": {"max_depth": 3}}})
    helper.log_model_params_to_mlflow(cfg, "xgboost", output_dir=str(tmp_path))
    assert calls and SECRET not in repr(calls) + caplog.text
    assert SECRET not in (tmp_path / "xgboost_params.yaml").read_text()

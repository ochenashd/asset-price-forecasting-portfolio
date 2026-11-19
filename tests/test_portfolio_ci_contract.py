"""Static and bounded local CI contracts; no Linux execution or new model fit."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portfolio_ci_contract_driver", ROOT / "scripts/portfolio_ci.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def synthetic_state(kind):
    nodes = set(driver.NATIVE_CASES)
    nodes.update(module + "::test_synthetic_contract" for module in driver.REQUIRED_MODULES)
    nodes.update(driver.CRITICAL_PREFIXES)
    deselected = set(driver.NATIVE_CASES) if kind == "macos" else set()
    selected = sorted(nodes - deselected)
    state = dict(selected=selected, deselected=sorted(deselected), passed=selected.copy(), failed=[], errors=[], skipped=[], xfail=[], exit_code=0)
    return state, dict(tests=len(selected), failures=0, errors=0, skipped=0)


@pytest.mark.parametrize("kind", ["macos", "linux"])
def test_exact_coverage_contract_accepts_only_complete_synthetic_metadata(kind):
    state, junit = synthetic_state(kind)
    assert driver.acceptance(state, kind, junit) == []
    # This is a pure metadata check, not simulated/native Linux execution.


@pytest.mark.parametrize("fault", ["zero", "missing_module", "missing_critical", "missing_native", "deselected_native",
    "skipped", "xfail", "failed", "error", "nonzero", "incomplete_pass", "junit", "duplicate", "unclassified"])
def test_fail_closed_coverage_rejects_faults(fault):
    state, junit = synthetic_state("linux")
    if fault == "zero": state.update(selected=[], passed=[])
    elif fault == "missing_module":
        module = "tests/test_research_bundle.py"
        state["selected"] = [n for n in state["selected"] if not n.startswith(module+"::")]
        state["passed"] = state["selected"].copy()
    elif fault == "missing_critical":
        node = driver.CRITICAL_PREFIXES[0]
        state["selected"].remove(node); state["passed"].remove(node)
    elif fault == "missing_native":
        node = next(iter(driver.NATIVE_CASES)); state["selected"].remove(node); state["passed"].remove(node)
    elif fault == "deselected_native":
        node = next(iter(driver.NATIVE_CASES)); state["selected"].remove(node); state["passed"].remove(node); state["deselected"]=[node]
    elif fault in {"skipped", "xfail", "failed"}: state[fault]=[state["selected"][0]]
    elif fault == "error": state["errors"]=[state["selected"][0]]
    elif fault == "nonzero": state["exit_code"]=1
    elif fault == "incomplete_pass": state["passed"].pop()
    elif fault == "junit": junit["tests"]=0
    elif fault == "duplicate": state["selected"].append(state["selected"][0])
    elif fault == "unclassified": state["selected"].append("tests/unreviewed.py::test_example")
    assert driver.acceptance(state,"linux",junit)


def test_macos_may_deselect_only_the_eight_real_native_cases():
    state, junit = synthetic_state("macos")
    assert len(state["deselected"]) == 8
    state["deselected"].append(state["selected"][0])
    assert "unexpected_deselection" in driver.acceptance(state,"macos",junit)


def fake_layout(root):
    for name in driver.REQUIRED_MODULES + ("src/public_defaults/environment.yaml",):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# synthetic contract\n")
    return root


def test_layout_requires_every_approved_module_and_source_reference(tmp_path):
    root=fake_layout(tmp_path)
    driver.validate_layout(root)
    (root/driver.REQUIRED_MODULES[0]).unlink()
    with pytest.raises(ValueError,match="required_file"):driver.validate_layout(root)


@pytest.mark.parametrize("name", ["mlflow.db","mlflow.db.backup","mlartifacts","data","gt.py","tests/test_backtest_execution.py",".env"])
def test_sensitive_or_historical_checkout_inputs_fail_preflight(tmp_path,name):
    root=fake_layout(tmp_path)
    file=root/name;file.parent.mkdir(exist_ok=True,parents=True);file.write_text("synthetic sentinel")
    with pytest.raises(ValueError,match="historical_or_sensitive"):driver.validate_layout(root)


def test_runtime_root_must_be_owned_temporary_empty_and_outside_checkout(tmp_path):
    root=tmp_path/"run";root.mkdir()
    assert driver.checked_root(root,ROOT)==root.resolve()
    (root/"sentinel").write_text("preserve")
    with pytest.raises(ValueError,match="nonempty"):driver.checked_root(root,ROOT)
    alias=tmp_path/"alias";alias.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError):driver.checked_root(alias,ROOT)
    with pytest.raises(ValueError):driver.checked_root(ROOT,ROOT)


def test_worker_environment_discards_credentials_and_ambient_destinations(tmp_path):
    source=dict(PATH="/usr/bin",HOME="/synthetic/home",GITHUB_TOKEN="synthetic-only",APF_DB_PASSWORD="synthetic-only",
        MLFLOW_TRACKING_URI="https://synthetic.invalid",HTTP_PROXY="https://synthetic.invalid",PYTHONPATH="/synthetic/source")
    env=driver.worker_environment(source,tmp_path)
    assert not set(source).intersection(env)-{"PATH","HOME"}
    assert env["HOME"]==str(tmp_path/"home")
    assert env["MLFLOW_DISABLE_TELEMETRY"]=="true" and env["PIP_NO_INDEX"]=="1"


def test_runtime_guards_block_actual_apis_but_not_retired_audit_notifications(tmp_path):
    import mlflow
    import sqlalchemy
    before=(socket.create_connection,sqlite3.connect,sqlalchemy.create_engine,mlflow.start_run)
    with driver.runtime_guards(tmp_path):
        sys.audit("socket.getaddrinfo","synthetic.invalid",443,0,0,0)
        with pytest.raises(RuntimeError,match="Network access"):socket.create_connection(("synthetic.invalid",443))
        with pytest.raises(RuntimeError,match="Global MLflow"):mlflow.start_run()
        with pytest.raises(AssertionError,match="owned temporary"):sqlite3.connect(tmp_path.parent/"synthetic-outside.sqlite")
        with pytest.raises(AssertionError,match="SQLite"):sqlalchemy.create_engine("postgresql://synthetic.invalid/example")
    assert (socket.create_connection,sqlite3.connect,sqlalchemy.create_engine,mlflow.start_run)==before


def test_sanitized_uploads_never_include_case_payloads_paths_or_logs(tmp_path):
    raw=tmp_path/"raw.xml";clean=tmp_path/"junit.xml"
    raw.write_text('<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0" hostname="private-host">'
        '<testcase classname="private" name="synthetic-secret" file="/private/source">'
        '<failure message="synthetic-secret">source payload</failure><system-out>private log</system-out>'
        '</testcase></testsuite></testsuites>')
    assert driver.sanitized_junit(raw,clean)["failures"]==1
    result=clean.read_text()
    assert all(text not in result for text in ["private","synthetic-secret","source payload","hostname"])
    assert ET.parse(clean).find('.//failure') is not None


def test_artifact_size_limits_are_enforced_before_upload(tmp_path):
    with pytest.raises(ValueError,match="evidence_size_limit"):
        driver.bounded_json(tmp_path/"summary.json",{"large":"x"*driver.MAX_FILE_BYTES})
    assert not (tmp_path/"summary.json").exists()
    (tmp_path/"junit.xml").write_bytes(b"x"*(driver.MAX_FILE_BYTES+1))
    with pytest.raises(ValueError,match="evidence_size_limit"):driver.validate_uploads(tmp_path)


def workflow(name):
    return yaml.load((ROOT/".github/workflows"/name).read_text(),Loader=yaml.BaseLoader)


@pytest.mark.parametrize("name", ["portfolio-tests.yml","research-filesystem-linux.yml"])
def test_workflows_are_manual_readonly_pinned_and_bounded(name):
    data=workflow(name)
    assert set(data["on"])=={"workflow_dispatch"}
    assert data["permissions"]=={"contents":"read"}
    assert "secrets." not in (ROOT/".github/workflows"/name).read_text()
    for job in data["jobs"].values():
        assert job["runs-on"]=="ubuntu-24.04" and int(job["timeout-minutes"])<=35
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r'actions/(checkout|setup-python|upload-artifact)@[0-9a-f]{40}',step["uses"])
            if "run" in step:
                assert "timeout-minutes" in step
                result=subprocess.run(["bash","-n"],input=step["run"],capture_output=True,text=True)
                assert result.returncode==0
    setup=[step for job in data["jobs"].values() for step in job["steps"] if step.get("uses","").startswith("actions/setup-python@")]
    assert all(step["with"]["python-version"]=="3.11.5" for step in setup)


def test_research_ci_runs_all_modules_and_uploads_only_sanitized_reports():
    data=workflow("portfolio-tests.yml");steps=data["jobs"]["research"]["steps"]
    command="\n".join(step.get("run","") for step in steps)
    assert "scripts/portfolio_ci.py" in command and "--platform linux" in command
    assert "--collect-only" not in command and "not native_linux" not in command
    assert "--no-deps" not in command
    assert len(driver.REQUIRED_MODULES)==32 and len(driver.NATIVE_CASES)==8
    assert "tests/test_portfolio_lock_contract.py" in driver.REQUIRED_MODULES
    assert all((ROOT/name).is_file() for name in driver.REQUIRED_MODULES)
    upload=steps[-1]["with"]
    assert set(Path(x).name for x in upload["path"].splitlines())=={"summary.json","diagnostics.json","junit.xml"}
    assert int(upload["retention-days"])<=3 and "*" not in upload["path"]


def test_filesystem_ci_remains_minimal_and_separate_from_model_stack():
    command="\n".join(step.get("run","") for step in workflow("research-filesystem-linux.yml")["jobs"]["filesystem"]["steps"])
    assert "scripts/research_filesystem_ci.py" in command
    assert "scikit-learn" not in command and "xgboost" not in command and "mlflow==" not in command


def test_installed_import_probe_contains_no_fit_inference_or_model_loading():
    tree=ast.parse(driver.PROBE)
    forbidden={"fit","predict","predict_proba","load_model"}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    assert "-I" in (ROOT/"scripts/portfolio_ci.py").read_text()


def test_internal_worker_requires_parent_authorization_and_new_basetemp(tmp_path):
    root=tmp_path/"worker";root.mkdir()
    with pytest.raises(ValueError,match="authorization"):
        driver.checked_worker_root(root,ROOT)
    for name in ("home","tmp","cache","matplotlib","raw","evidence","probe"):
        (root/name).mkdir()
    marker=root/".portfolio-ci-owned.json"
    marker.write_text(json.dumps({"schema":"apf-portfolio-ci-owned-v1","checkout":"synthetic-wrong","parent_pid":os.getppid()}))
    with pytest.raises(ValueError,match="authorization"):
        driver.checked_worker_root(root,ROOT)
    marker.write_text(json.dumps({"schema":"apf-portfolio-ci-owned-v1","checkout":str(ROOT),"parent_pid":os.getppid()}))
    assert driver.checked_worker_root(root,ROOT)==root.resolve()
    (root/"tmp/pytest").mkdir()
    with pytest.raises(ValueError,match="basetemp_must_be_new"):
        driver.checked_worker_root(root,ROOT)


def test_real_checkout_classifies_every_approved_test_module():
    """Catch omissions hidden by synthetic layouts derived from the registry."""
    actual = {"tests/" + p.name for p in (ROOT / "tests").glob("test_*.py")}
    assert actual == set(driver.REQUIRED_MODULES)
    assert len(actual) == 32
    assert "tests/test_portfolio_documentation.py" in actual
    driver.validate_layout(ROOT)


def test_new_unclassified_module_is_rejected(tmp_path):
    root = fake_layout(tmp_path)
    (root / "tests/test_unreviewed_extension.py").write_text("# synthetic unclassified module\n")
    with pytest.raises(ValueError, match="unclassified_test_module"):
        driver.validate_layout(root)


def test_registered_documentation_module_cannot_be_omitted(tmp_path):
    root = fake_layout(tmp_path)
    (root / "tests/test_portfolio_documentation.py").unlink()
    with pytest.raises(ValueError, match="missing_or_unsafe_required_file"):
        driver.validate_layout(root)


def test_documentation_module_is_portable_and_has_no_model_or_backend_import():
    tree = ast.parse((ROOT / "tests/test_portfolio_documentation.py").read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    imports.update(node.module.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert imports == {"ast", "pathlib", "re", "tomllib", "pytest"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {"fit", "predict", "predict_proba", "connect", "create_engine"}
                   for node in ast.walk(tree))
    assert "native_linux" not in {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def test_full_selection_requires_the_documentation_module_too():
    state, junit = synthetic_state("macos")
    node = "tests/test_portfolio_documentation.py::test_synthetic_contract"
    assert node in state["selected"]
    state["selected"].remove(node)
    state["passed"].remove(node)
    junit["tests"] -= 1
    assert "missing_or_unclassified_modules" in driver.acceptance(state, "macos", junit)


@pytest.mark.parametrize("kind", ["macos", "linux"])
def test_worker_intended_selection_and_guards_are_preserved(tmp_path, monkeypatch, kind):
    """Inspect worker arguments under real guards; never simulate native execution."""
    import mlflow
    root = tmp_path / "worker-probe"
    for name in ("tmp", "raw", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    original_connect = socket.socket.connect
    original_active = mlflow.active_run
    calls = []

    def observe(arguments, *, plugins):
        calls.append(arguments)
        assert len(plugins) == 1
        with pytest.raises(RuntimeError, match="Network access forbidden"):
            socket.create_connection(("synthetic.invalid", 443))
        with pytest.raises(RuntimeError, match="Global MLflow logging forbidden"):
            mlflow.active_run()
        with pytest.raises(AssertionError, match="owned temporary root"):
            sqlite3.connect(tmp_path / "forbidden.sqlite")
        return 0

    monkeypatch.setattr(pytest, "main", observe)
    assert driver.worker(root, ROOT, kind) == 0
    assert len(calls) == 1 and calls[0][0] == "tests"
    assert ("-k" in calls[0]) is (kind == "macos")
    if kind == "macos":
        assert calls[0][-2:] == ["-k", "not native_linux"]
    assert (root / "raw/state.json").is_file()
    assert not (tmp_path / "forbidden.sqlite").exists()
    assert socket.socket.connect is original_connect
    assert mlflow.active_run is original_active

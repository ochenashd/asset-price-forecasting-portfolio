"""Static/manual-CI evidence checks only; no Linux test or workflow dispatch."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/research-filesystem-linux.yml"
spec = importlib.util.spec_from_file_location("fs_ci_contract_driver", ROOT / "scripts/research_filesystem_ci.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def workflow():
    # BaseLoader keeps the YAML 1.2 workflow key 'on' from YAML 1.1 bool coercion.
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def valid_state():
    nodes = [driver.TESTS[2] + f"::test_native_linux_case[{i}]" for i in range(8)]
    nodes += [path + "::test_placeholder" for path in driver.TESTS if path != driver.TESTS[2]]
    nodes += [driver.TESTS[0] + f"::test_static[{i}]" for i in range(driver.EXPECTED_TOTAL - len(nodes))]
    return {"collected": nodes, "native_passed": nodes[:8], "deselected": 0, "exit_code": 0}


def test_manual_trigger_readonly_permissions_and_pinned_official_actions():
    data = workflow()
    assert set(data["on"]) == {"workflow_dispatch"}
    assert data["permissions"] == {"contents": "read"}
    job = data["jobs"]["filesystem"]
    assert job["runs-on"] == "ubuntu-24.04" and int(job["timeout-minutes"]) <= 30
    for step in job["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"actions/(checkout|setup-python|upload-artifact)@[0-9a-f]{40}", step["uses"])
    assert "secrets." not in WORKFLOW.read_text() and "inputs." not in WORKFLOW.read_text()


def test_shell_syntax_and_command_timeouts_are_static_only():
    for step in workflow()["jobs"]["filesystem"]["steps"]:
        if "run" in step:
            assert "timeout-minutes" in step and "timeout " in step["run"]
            result = subprocess.run(["bash", "-n"], input=step["run"], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert not re.search(r"\b(sudo|docker|curl|wget|git push)\b", step["run"])


def test_sparse_checkout_and_artifact_allowlist():
    steps = workflow()["jobs"]["filesystem"]["steps"]
    checkout = steps[0]["with"]
    assert checkout["persist-credentials"] == "false"
    assert "data" not in checkout["sparse-checkout"] and "mlartifacts" not in checkout["sparse-checkout"]
    upload = steps[-1]
    assert "always()" in upload["if"] and int(upload["with"]["retention-days"]) <= 3
    paths = upload["with"]["path"].splitlines()
    assert {Path(p).name for p in paths} == {"junit.xml", "summary.json", "diagnostics.json"}
    assert all("/evidence/" in p and "*" not in p for p in paths)


def test_explicit_selection_contains_no_training_or_protected_module():
    assert len(driver.TESTS) == 5
    assert all((ROOT / p).is_file() for p in driver.TESTS)
    assert "tests/test_backtest_execution.py" not in driver.TESTS
    assert driver.EXPECTED_NATIVE == 8 and driver.EXPECTED_TOTAL == 211


@pytest.mark.parametrize("fault", ["zero", "native_zero", "missing_native", "skipped", "deselected", "exit", "wrong_file"])
def test_success_gate_rejects_incomplete_or_skipped_native_evidence(fault):
    state = valid_state()
    counts = {"tests": 211, "failures": 0, "errors": 0, "skipped": 0}
    assert driver.acceptance(state, counts)
    if fault == "zero":
        state["collected"] = []
    elif fault == "native_zero":
        state["native_passed"] = []
    elif fault == "missing_native":
        state["native_passed"] = state["native_passed"][:-1]
    elif fault == "skipped":
        counts["skipped"] = 1
    elif fault == "deselected":
        state["deselected"] = 1
    elif fault == "exit":
        state["exit_code"] = 2
    else:
        state["collected"][-1] = "tests/unapproved.py::test_case"
    assert not driver.acceptance(state, counts)


def test_junit_sanitization_removes_payloads_paths_and_logs(tmp_path):
    raw, clean = tmp_path / "raw.xml", tmp_path / "clean.xml"
    raw.write_text('<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0" hostname="private">'
        '<testcase name="test_fixture" classname="fixture" file="/private/path">'
        '<failure message="sensitive">payload-secret</failure><system-out>captured-secret</system-out>'
        '</testcase></testsuite></testsuites>')
    assert driver.sanitized_junit(raw, clean)["failures"] == 1
    result = clean.read_text()
    assert all(s not in result for s in ("private", "sensitive", "payload-secret", "captured-secret"))
    assert ET.parse(clean).find(".//failure") is not None


def test_evidence_size_is_bounded(tmp_path):
    with pytest.raises(ValueError, match="evidence_size_limit"):
        driver.write_json(tmp_path / "large.json", {"value": "x" * driver.MAX_REPORT})


def test_ci_root_rejects_source_and_unapproved_paths(tmp_path):
    root = tmp_path.resolve()
    with pytest.raises(ValueError):
        driver.checked_root(ROOT, root, ROOT)
    child = root / "child"
    child.mkdir()
    assert driver.checked_root(child, root, ROOT) == child
    alias = root / "alias"
    alias.symlink_to(child, target_is_directory=True)
    with pytest.raises(ValueError):
        driver.checked_root(alias, root, ROOT)

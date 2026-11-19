"""Manual CI driver and pytest observer; never a publication/recovery service."""
import argparse
import ctypes
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

TESTS = (
    "tests/test_research_filesystem_contracts.py",
    "tests/test_research_filesystem_crash_safety.py",
    "tests/test_research_filesystem_linux_runtime.py",
    "tests/test_research_bundle.py",
    "tests/test_research_mlflow_linkage.py",
)
EXPECTED_TOTAL = 211
EXPECTED_NATIVE = 8
MAX_REPORT = 262144
STATE = {"collected": [], "deselected": 0, "native_passed": [], "passed": [], "failed": [], "warnings": 0}
WINDOW_RULES = {
    1: ("test_publication_fault_boundaries", "before_stage", "injected mkdir failure"),
    2: ("test_publication_fault_boundaries", "partial_write", "injected partial-write failure"),
    3: ("test_publication_fault_boundaries", "after_file_sync", "real file sync then injected failure"),
    4: ("test_killed_publisher", "before_rename", "process kill before installation"),
    5: ("test_killed_publisher", "after_rename", "process kill after installation"),
    6: ("test_publication_fault_boundaries", "parent_sync", "injected parent-sync failure"),
    7: ("test_killed_publisher", "after_publish", "process kill after low-level publisher return"),
    8: ("test_journal_unsupported_sync", "False", "initial journal file-sync failure"),
    9: ("test_killed_journal_writer", "journal_hold", "process kill with creating intent"),
    10: ("test_killed_linkage", "lost_response", "fake remote response loss plus process kill"),
    11: ("test_killed_linkage", "after_response", "process kill before journal confirmation"),
    12: ("test_process_journal_change", "", "independent writer during reconciliation"),
}


def write_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True).encode()
    if len(data) > MAX_REPORT:
        raise ValueError("evidence_size_limit")
    Path(path).write_bytes(data)


def native(node):
    return node.startswith(TESTS[2] + "::test_native_linux_")


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sklearn", "xgboost", "torch", "mlflow", "yfinance"}:
            raise RuntimeError("forbidden_model_or_provider_runtime")


def install_guard():
    sys.meta_path.insert(0, NoModels())
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "sqlite3.connect", "pickle.find_class"}:
            raise RuntimeError("forbidden_external_or_unsafe_operation")
    sys.addaudithook(audit)


def pytest_configure(config):
    install_guard()


def pytest_collection_finish(session):
    STATE["collected"] = [item.nodeid for item in session.items]


def pytest_deselected(items):
    STATE["deselected"] += len(items)


def pytest_runtest_logreport(report):
    if report.when == "call" and report.passed:
        STATE["passed"].append(report.nodeid)
    if report.when == "call" and report.passed and native(report.nodeid):
        STATE["native_passed"].append(report.nodeid)
    if report.failed:
        STATE["failed"].append({"test": report.nodeid, "phase": report.when})


def pytest_warning_recorded(warning_message, when, nodeid, location):
    STATE["warnings"] += 1


def pytest_sessionfinish(session, exitstatus):
    destination = os.environ.get("APF_FS_STATE")
    if destination:
        write_json(destination, {**STATE, "exit_code": int(exitstatus)})


def evidence_counts(xml_path):
    if not Path(xml_path).is_file() or Path(xml_path).stat().st_size > 4 * 1024 * 1024:
        raise ValueError("missing_or_oversized_junit")
    suites = ET.parse(xml_path).getroot().findall("testsuite")
    if not suites:
        raise ValueError("missing_junit_suite")
    return {key: sum(int(s.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")}


def acceptance(state, counts):
    nodes = state["collected"]
    return (state["exit_code"] == 0 and len(nodes) == len(set(nodes)) == EXPECTED_TOTAL
        and {n.split("::", 1)[0] for n in nodes} == set(TESTS)
        and sum(native(n) for n in nodes) == EXPECTED_NATIVE
        and len(set(state["native_passed"])) == EXPECTED_NATIVE
        and set(state["native_passed"]) == {n for n in nodes if native(n)}
        and state["deselected"] == 0
        and counts == {"tests": EXPECTED_TOTAL, "failures": 0, "errors": 0, "skipped": 0})


def sanitized_junit(raw_path, target):
    """Allow only test identity/count attributes; never upload captured payloads."""
    counts = evidence_counts(raw_path)
    source = ET.parse(raw_path).getroot()
    output = ET.Element("testsuites")
    for suite in source.findall("testsuite"):
        clean = ET.SubElement(output, "testsuite", {k: suite.attrib[k] for k in
            ("tests", "failures", "errors", "skipped", "time") if k in suite.attrib})
        for case in suite.findall("testcase"):
            item = ET.SubElement(clean, "testcase", {k: case.attrib[k][:512] for k in
                ("name", "classname", "time") if k in case.attrib})
            for kind in ("failure", "error", "skipped"):
                if case.find(kind) is not None:
                    ET.SubElement(item, kind).text = "Raw diagnostics withheld; see bounded phase/identity summary."
    data = ET.tostring(output, encoding="utf-8", xml_declaration=True)
    if len(data) > MAX_REPORT:
        raise ValueError("sanitized_junit_size_limit")
    Path(target).write_bytes(data)
    return counts


def checked_root(value, runner_temp, checkout):
    root, base, checkout = Path(value), Path(runner_temp).resolve(), Path(checkout).resolve()
    if (not root.is_absolute() or root != root.resolve() or not root.is_dir()
            or root == base or not root.is_relative_to(base) or root.is_relative_to(checkout)
            or checkout.is_relative_to(root) or ".." in root.parts):
        raise ValueError("unsafe_ci_root")
    for path in (root, *root.parents):
        if path == base:
            break
        if path.is_symlink():
            raise ValueError("symlink_ci_root")
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    root = checked_root(args.root, os.environ["RUNNER_TEMP"], checkout)
    evidence = root / "evidence"
    evidence.mkdir(mode=0o700)  # Fail rather than reuse a pre-existing evidence tree.
    summary = {"run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "unknown"),
        "revision": os.environ.get("GITHUB_SHA", "unknown"),
        "tests": list(TESTS), "status": "incomplete", "python": platform.python_version(),
        "limitations": ["No power-loss or physical-host durability certification.",
            "Finite process tests do not establish universal race freedom or distributed exactly-once.",
            "Crash-window mappings distinguish injected exceptions from actual process termination."]}
    diagnostics = []
    write_json(evidence / "summary.json", summary)
    try:
        if platform.system() != "Linux" or not sys.platform.startswith("linux"):
            raise ValueError("native_linux_required")
        if sys.version_info[:2] != (3, 11):
            raise ValueError("python_311_required")
        if any((checkout / p).exists() for p in ("mlflow.db", "mlflow.db.backup", "mlartifacts", "data")):
            raise ValueError("historical_stores_must_not_be_checked_out")
        if any(not (checkout / p).is_file() for p in TESTS):
            raise ValueError("missing_selected_test_file")
        if not hasattr(ctypes.CDLL(None), "renameat2"):
            raise ValueError("renameat2_symbol_missing")
        probe = root / "directory-fsync-probe"
        probe.mkdir(mode=0o700)
        fd = os.open(probe, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # Required host evidence, scoped to the explicit temporary directory.
        for command in (["uname", "-a"], [sys.executable, "-m", "pytest", "--version"]):
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
            print(result.stdout[:4096].strip(), flush=True)
        mount = subprocess.run(["findmnt", "--target", str(root), "--output", "FSTYPE,OPTIONS", "--json"],
            capture_output=True, text=True, timeout=15, check=True)
        mount_data = json.loads(mount.stdout)["filesystems"]
        safe_mount = [{"fstype": item["fstype"], "option_names": sorted({option.split("=", 1)[0]
            for option in item["options"].split(",")})} for item in mount_data]
        print(json.dumps({"mount": safe_mount}), flush=True)
        summary.update(kernel=platform.release(), architecture=platform.machine(), system=platform.system(),
            mount=safe_mount, container_hint=Path("/.dockerenv").exists(),
            physical_host="unknown", renameat2_symbol=True, directory_fsync_probe="succeeded",
            dependencies={n: importlib.metadata.version(n) for n in ("pytest", "numpy", "pandas", "pyarrow", "omegaconf")})
        print(json.dumps({k: summary[k] for k in ("python", "architecture", "kernel", "renameat2_symbol", "directory_fsync_probe")}), flush=True)
        guard = root / "guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text("from scripts.research_filesystem_ci import install_guard\ninstall_guard()\n")
        env = {**os.environ, "PYTHONPATH": str(guard) + os.pathsep + str(checkout),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TMPDIR": str(root), "APF_FS_STATE": str(root / "state.json")}
        command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-p", "scripts.research_filesystem_ci", *TESTS, "--basetemp=" + str(root / "cases"),
            "--junitxml=" + str(root / "raw-junit.xml")]
        with (root / "raw-pytest.log").open("wb") as log:
            completed = subprocess.run(command, cwd=checkout, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=600)
        state = json.loads((root / "state.json").read_text())
        counts = sanitized_junit(root / "raw-junit.xml", evidence / "junit.xml")
        success = completed.returncode == 0 and acceptance(state, counts)
        summary.update(status="passed" if success else "failed", counts={**counts,
            "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
            "deselected": state["deselected"], "warnings": state["warnings"]}, collected=len(state["collected"]),
            native_collected=sum(native(n) for n in state["collected"]), native_passed=len(set(state["native_passed"])),
            pytest_exit=completed.returncode,
            coverage={"publisher_cases": sum(n.startswith(TESTS[3]) for n in state["collected"]),
                "journal_cases": sum(n.startswith(TESTS[4]) for n in state["collected"]),
                "termination_test_functions": ["test_killed_publisher_preserves_forensic_state",
                    "test_killed_journal_writer_releases_flock_and_preserves_creating",
                    "test_killed_linkage_recovers_without_second_remote_create",
                    "test_native_linux_kill_then_independent_recovery",
                    "test_native_linux_journal_kill_revision_and_recovery"],
                "windows": [{"id": key, "mechanism": rule[2], "passed_cases": [n for n in state["passed"]
                    if rule[0] in n.split("::", 1)[-1] and rule[1] in n.split("::", 1)[-1]],
                    "limit": "Scoped boundary evidence only; see committed Stage 5B/5C matrix, not power-loss validation."}
                    for key, rule in WINDOW_RULES.items()]})
        diagnostics = state["failed"][:32]
        if not success:
            diagnostics.append({"code": "pytest_or_required_collection_gate_failed"})
        code = 0 if success else completed.returncode or 1
    except Exception as error:
        summary["status"] = "failed"
        # Exception type only: never persist arbitrary provider/path/payload text.
        known = {"native_linux_required", "python_311_required", "historical_stores_must_not_be_checked_out",
                 "missing_selected_test_file", "renameat2_symbol_missing", "missing_or_oversized_junit",
                 "missing_junit_suite", "sanitized_junit_size_limit", "evidence_size_limit"}
        diagnostics.append({"code": str(error) if type(error) is ValueError and str(error) in known
                            else "validation_driver_failed", "exception": type(error).__name__})
        code = 124 if isinstance(error, subprocess.TimeoutExpired) else 1
    summary["driver_exit"] = code
    write_json(evidence / "summary.json", summary)
    write_json(evidence / "diagnostics.json", diagnostics)
    print(json.dumps({"status": summary["status"], "native_passed": summary.get("native_passed", 0)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

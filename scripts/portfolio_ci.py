"""Fail-closed synthetic research CI with bounded, sanitized evidence.

No downloads, fitting code, execution engine or remote tracking implementation
is duplicated here. One pytest worker runs the existing suite. Python API guards
are cooperative controls, not an OS sandbox against hostile dependencies.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REQUIRED_MODULES = tuple("tests/" + name + ".py" for name in (
    "test_database_credential_security", "test_execution_adapter", "test_execution_engine_v1",
    "test_execution_metrics", "test_execution_model_provenance", "test_execution_prediction_export",
    "test_execution_reversals", "test_execution_short", "test_execution_slippage",
    "test_execution_terminal_policy", "test_execution_trade_ledger", "test_market_eligibility",
    "test_market_snapshot", "test_mlflow_fixture_isolation", "test_portfolio_ci_contract", "test_portfolio_lock_contract",
    "test_portfolio_documentation", "test_preprocessing_causality", "test_public_source_manifest", "test_research_bundle",
    "test_research_bundle_verification", "test_research_filesystem_ci_contract",
    "test_research_filesystem_contracts", "test_research_filesystem_crash_safety",
    "test_research_filesystem_linux_runtime", "test_research_input_adapter", "test_research_mlflow_linkage",
    "test_research_mlflow_local_backend", "test_research_protocol", "test_research_tracking_reconciliation",
    "test_research_tracking_record", "test_synthetic_research",
))
NATIVE_MODULE = "tests/test_research_filesystem_linux_runtime.py"
NATIVE_CASES = frozenset(NATIVE_MODULE + "::" + name for name in (
    "test_native_linux_real_noreplace_and_directory_sync",
    "test_native_linux_kill_then_independent_recovery[staging-False-False]",
    "test_native_linux_kill_then_independent_recovery[before_rename-False-True]",
    "test_native_linux_kill_then_independent_recovery[after_rename-True-True]",
    "test_native_linux_kill_then_independent_recovery[after_publish-True-False]",
    "test_native_linux_bounded_publish_contention",
    "test_native_linux_journal_kill_revision_and_recovery[journal_hold]",
    "test_native_linux_journal_kill_revision_and_recovery[journal_after_replace]",
))
CRITICAL_PREFIXES = (
    "tests/test_public_source_manifest.py::test_exact_public_source_and_resource_hashes",
    "tests/test_public_source_manifest.py::test_safe_defaults_are_packaged_and_offline",
    "tests/test_mlflow_fixture_isolation.py::test_fixture_scope_is_function",
    "tests/test_mlflow_fixture_isolation.py::test_normal_and_exceptional_exit_restore_all_state",
    "tests/test_mlflow_fixture_isolation.py::test_setup_exception_restores_guards_and_environment",
    "tests/test_mlflow_fixture_isolation.py::test_disposal_exception_still_restores_state",
    "tests/test_mlflow_fixture_isolation.py::test_ambient_uri_and_active_run_are_not_used_or_ended",
    "tests/test_mlflow_fixture_isolation.py::test_guard_blocks_training_inference_network_and_fluent_calls",
    "tests/test_mlflow_fixture_isolation.py::test_historical_database_and_artifact_reads_are_rejected",
    "tests/test_mlflow_fixture_isolation.py::test_existing_backend_is_rejected_before_client_construction",
    "tests/test_mlflow_fixture_isolation.py::test_consecutive_backends_are_distinct_and_have_own_artifact_roots",
    "tests/test_mlflow_fixture_isolation.py::test_receipt_and_bundle_identities_do_not_depend_on_guard_order",
    "tests/test_mlflow_fixture_isolation.py::test_demo_fixture_restores_artifact_environment_without_extra_fits",
    "tests/test_research_mlflow_local_backend.py::test_direct_real_client_linkage_readback_and_technical_status",
    "tests/test_synthetic_research.py::test_real_training_receipt_bundle_replay_and_local_tracking",
    "tests/test_synthetic_research.py::test_selection_scaling_and_fit_receive_only_authorized_rows",
)
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_RAW_BYTES = 32 * 1024 * 1024
PROBE = r'''
from pathlib import Path
import hashlib,json,sys
from importlib import resources
import src
import examples.synthetic_research as demo
import yaml
root=Path(src.__file__).resolve().parent.parent
assert 'site-packages' in str(root) and 'site-packages' in demo.__file__
checkout=Path(sys.argv[1]).resolve()
assert root != checkout and checkout not in root.parents
profile=yaml.safe_load(resources.files('src').joinpath('public_defaults/environment.yaml').read_text())
for name,expected in profile['sources'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==expected
assert (root/'src/public_defaults/environment.yaml').read_bytes()==(checkout/'src/public_defaults/environment.yaml').read_bytes()
print(json.dumps({'status':'verified','source_reference_files':len(profile['sources']),'installed_origin_outside_checkout':True,'fitting_or_inference_called':False}))
'''


def actual_kind():
    if sys.platform.startswith("linux") and platform.system() == "Linux":
        return "linux"
    if sys.platform == "darwin" and platform.system() == "Darwin":
        return "macos"
    raise ValueError("unsupported_host")


def validate_layout(checkout):
    for name in REQUIRED_MODULES + ("src/public_defaults/environment.yaml",):
        file = checkout / name
        if not file.is_file() or file.is_symlink():
            raise ValueError("missing_or_unsafe_required_file")
    for name in ("gt.py", "tests/test_backtest_execution.py", "mlflow.db", "mlflow.db.backup",
                 "mlartifacts", "mlruns", "data", "outputs", ".env"):
        if os.path.lexists(checkout / name):
            raise ValueError("historical_or_sensitive_input_present")
    actual = {"tests/" + p.name for p in (checkout / "tests").glob("test_*.py")}
    if actual != set(REQUIRED_MODULES):
        raise ValueError("unclassified_test_module")


def checked_root(root, checkout):
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("explicit_empty_temporary_root_required")
    root = root.resolve(strict=True)
    bases = [Path(__import__('tempfile').gettempdir()).resolve()]
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("RUNNER_TEMP"):
        bases.append(Path(os.environ["RUNNER_TEMP"]).resolve())
    if (root == checkout or root in checkout.parents or checkout in root.parents
            or not any(base in root.parents for base in bases)
            or root.stat().st_uid != os.getuid() or any(root.iterdir())):
        raise ValueError("unapproved_or_nonempty_test_root")
    return root


def checked_worker_root(root, checkout):
    """Do not let the internal worker flag bypass temporary-root authorization."""
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("unapproved_worker_root")
    root = root.resolve(strict=True)
    bases = [Path(os.environ.get("APF_CI_PARENT_TEMP_BASE", __import__('tempfile').gettempdir())).resolve()]
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("RUNNER_TEMP"):
        bases.append(Path(os.environ["RUNNER_TEMP"]).resolve())
    if (not any(base in root.parents for base in bases) or root == checkout
            or root in checkout.parents or checkout in root.parents or root.stat().st_uid != os.getuid()):
        raise ValueError("unapproved_worker_root")
    marker = root / ".portfolio-ci-owned.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("missing_worker_authorization")
    expected = {"schema": "apf-portfolio-ci-owned-v1", "checkout": str(checkout), "parent_pid": os.getppid()}
    if json.loads(marker.read_text()) != expected:
        raise ValueError("invalid_worker_authorization")
    for name in ("home", "tmp", "cache", "matplotlib", "raw", "evidence", "probe"):
        if (root/name).is_symlink() or not (root/name).is_dir():
            raise ValueError("unsafe_worker_subdirectory")
    if os.path.lexists(root/"tmp/pytest"):
        raise ValueError("worker_basetemp_must_be_new")
    return root



def dependency_preflight(checkout, kind):
    """Do not run a lock-enforced job with partial evidence or network fallback."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("portfolio_dependency_preflight", checkout / "scripts/verify_portfolio_environment.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    data = verifier.read_manifest(checkout / "requirements/locks/manifest.json")
    verifier.verify_checkout(data, checkout)
    target = {"macos": "macos-arm64-cp3115", "linux": "linux-x86_64-cp3115"}[kind]
    verifier.require_complete(data, target, "test")
    verifier.require_complete(data, target, "build")


def worker_environment(original, root):
    env = {key: original[key] for key in ("PATH", "LANG", "SYSTEMROOT") if key in original}
    env.update(HOME=str(root / "home"), TMPDIR=str(root / "tmp"), APF_CI_PARENT_TEMP_BASE=str(__import__("tempfile").gettempdir()),
        XDG_CACHE_HOME=str(root / "cache"), MPLCONFIGDIR=str(root / "matplotlib"),
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PIP_CONFIG_FILE="/dev/null",
        PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", MLFLOW_DISABLE_TELEMETRY="true",
        MLFLOW_ENABLE_WORKSPACES="false")
    if original.get("GITHUB_ACTIONS") == "true" and original.get("RUNNER_TEMP"):
        env.update(GITHUB_ACTIONS="true", RUNNER_TEMP=original["RUNNER_TEMP"])
    return env


@contextmanager
def runtime_guards(root):
    """Block real Python network/global logging and confine database APIs.

    Manual sys.audit calls in lifecycle tests remain distinguishable from actual
    socket calls. Existing fixture guards may safely nest inside this context.
    """
    import pytest
    import sqlalchemy
    from sqlalchemy.engine import make_url
    with pytest.MonkeyPatch.context() as patch:
        def no_network(*args, **kwargs):
            raise RuntimeError("Network access forbidden during portfolio CI tests")
        patch.setattr(socket.socket, "connect", no_network)
        patch.setattr(socket.socket, "connect_ex", no_network)
        patch.setattr(socket, "create_connection", no_network)
        patch.setattr(socket, "getaddrinfo", no_network)
        patch.setattr(socket.socket, "bind", no_network)
        import mlflow
        def no_logging(*args, **kwargs):
            raise RuntimeError("Global MLflow logging forbidden during portfolio CI tests")
        for name in ("start_run", "end_run", "active_run", "set_tracking_uri", "get_tracking_uri",
                     "set_experiment", "create_experiment", "log_param", "log_params", "log_metric",
                     "log_metrics", "log_artifact", "log_artifacts", "log_figure"):
            patch.setattr(mlflow, name, no_logging)
        original_connect = sqlite3.connect
        original_engine = sqlalchemy.create_engine
        def confined_path(value):
            if value in (None, "", ":memory:"):
                return
            if root.resolve() not in Path(os.fsdecode(value)).resolve().parents:
                raise AssertionError("CI database path must stay inside the owned temporary root")
        def connect(value, *args, **kwargs):
            confined_path(value)
            return original_connect(value, *args, **kwargs)
        def engine(value, *args, **kwargs):
            parsed = make_url(value)
            if parsed.get_backend_name() != "sqlite":
                raise AssertionError("Only owned temporary SQLite backends are permitted")
            confined_path(parsed.database)
            return original_engine(value, *args, **kwargs)
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(sqlite3.dbapi2, "connect", connect)
        patch.setattr(sqlalchemy, "create_engine", engine)
        yield


def acceptance(state, kind, junit):
    """Return diagnostic codes, never hide missing/failed/omitted coverage."""
    selected = list(state.get("selected", [])); deselected = list(state.get("deselected", []))
    all_nodes = selected + deselected; codes = []
    if kind not in {"macos", "linux"}: codes.append("unsupported_platform")
    if not selected: codes.append("zero_selected_tests")
    if len(all_nodes) != len(set(all_nodes)): codes.append("duplicate_collection")
    modules = {n.split("::", 1)[0] for n in all_nodes}
    if modules != set(REQUIRED_MODULES): codes.append("missing_or_unclassified_modules")
    for prefix in CRITICAL_PREFIXES:
        if not any(n == prefix or n.startswith(prefix + "[") for n in selected):
            codes.append("critical_coverage_omitted"); break
    native = {n for n in all_nodes if n.startswith(NATIVE_MODULE + "::test_native_linux_")}
    if native != NATIVE_CASES: codes.append("native_collection_mismatch")
    expected_deselected = NATIVE_CASES if kind == "macos" else set()
    if set(deselected) != expected_deselected: codes.append("unexpected_deselection")
    passed = list(state.get("passed", []))
    if len(passed) != len(set(passed)) or set(passed) != set(selected): codes.append("incomplete_pass_coverage")
    if state.get("failed") or state.get("errors"): codes.append("pytest_failures_or_errors")
    if state.get("skipped") or state.get("xfail"): codes.append("unexpected_skip_or_xfail")
    if state.get("exit_code") != 0: codes.append("pytest_nonzero_exit")
    if (junit.get("tests") != len(selected) or any(junit.get(k) for k in ("failures", "errors", "skipped"))):
        codes.append("junit_incomplete_or_failed")
    return sorted(set(codes))


def bounded_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("evidence_size_limit")
    Path(path).write_bytes(data)


def sanitized_junit(source, destination):
    if source.stat().st_size > MAX_RAW_BYTES:
        raise ValueError("raw_junit_size_limit")
    original = ET.parse(source).getroot()
    suites = original.findall("testsuite") if original.tag == "testsuites" else [original]
    totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    result = ET.Element("testsuites"); suite = ET.SubElement(result, "testsuite", {k: str(v) for k, v in totals.items()})
    for item in original.iter("testcase"):
        key = hashlib.sha256((item.get("classname", "") + "::" + item.get("name", "")).encode()).hexdigest()
        case = ET.SubElement(suite, "testcase", {"name": "case-" + key[:20]})
        for kind in ("failure", "error", "skipped"):
            if item.find(kind) is not None:
                ET.SubElement(case, kind, {"message": "details_not_uploaded"})
    data = ET.tostring(result, encoding="utf-8", xml_declaration=True)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("evidence_size_limit")
    destination.write_bytes(data)
    return totals


def validate_uploads(directory):
    paths = [directory / name for name in ("summary.json", "diagnostics.json", "junit.xml")]
    if any(p.is_symlink() for p in paths): raise ValueError("unsafe_report_path")
    sizes = [p.stat().st_size for p in paths if p.is_file()]
    if any(n > MAX_FILE_BYTES for n in sizes) or sum(sizes) > MAX_TOTAL_BYTES:
        raise ValueError("evidence_size_limit")
    return sum(sizes)


def worker(root, checkout, kind):
    import pytest
    sys.path.insert(0, str(checkout))
    class Observer:
        def __init__(self):
            self.state = {"selected": [], "deselected": [], "passed": [], "failed": [],
                          "errors": [], "skipped": [], "xfail": [], "warnings": 0}
        def pytest_collection_finish(self, session):
            self.state["selected"] = [x.nodeid for x in session.items]
        def pytest_deselected(self, items):
            self.state["deselected"].extend(x.nodeid for x in items)
        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed: self.state["passed"].append(report.nodeid)
            if report.failed: self.state["failed" if report.when == "call" else "errors"].append(report.nodeid)
            if report.skipped: self.state["skipped"].append(report.nodeid)
            if getattr(report, "wasxfail", None): self.state["xfail"].append(report.nodeid)
        def pytest_warning_recorded(self, warning_message, when, nodeid, location):
            self.state["warnings"] += 1
    observer = Observer()
    args = ["tests", "--basetemp=" + str(root / "tmp/pytest"), "--junitxml=" + str(root / "raw/junit.xml"),
            "-o", "cache_dir=" + str(root / "cache/pytest"), "--tb=short"]
    if kind == "macos": args += ["-k", "not native_linux"]
    started = time.monotonic()
    with runtime_guards(root):
        code = int(pytest.main(args, plugins=[observer]))
    observer.state.update(exit_code=code, runtime_seconds=time.monotonic() - started)
    (root / "raw/state.json").write_text(json.dumps(observer.state))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=("macos", "linux"))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    if sys.version_info[:3] != (3, 11, 5):
        raise ValueError("declared_python_profile_required")
    if actual_kind() != args.platform:
        raise ValueError("requested_platform_is_not_actual_host")
    try:
        dependency_preflight(checkout, args.platform)
    except ValueError:
        print(json.dumps({"status": "rejected", "codes": ["dependency_lock_incomplete_or_invalid"], "publication_approved": False}))
        return 2
    if args.worker:
        return worker(checked_worker_root(args.root, checkout), checkout, args.platform)
    if not 60 <= args.timeout_seconds <= 1200: parser.error("timeout must be between 60 and 1200 seconds")
    if actual_kind() != args.platform: raise ValueError("requested_platform_is_not_actual_host")
    validate_layout(checkout)
    root = checked_root(args.root, checkout)
    for name in ("home", "tmp", "cache", "matplotlib", "raw", "evidence", "probe"):
        (root / name).mkdir(mode=0o700)
    (root/".portfolio-ci-owned.json").write_text(json.dumps({"schema": "apf-portfolio-ci-owned-v1", "checkout": str(checkout), "parent_pid": os.getpid()}))
    env = worker_environment(os.environ, root)
    try:
        probe = subprocess.run([sys.executable, "-I", "-B", "-c", PROBE, str(checkout)],
            cwd=root / "probe", env=env, capture_output=True, text=True, timeout=60)
        if probe.returncode:
            raise ValueError("installed_wheel_probe_failed")
        proof = json.loads(probe.stdout)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--root", str(root), "--platform", args.platform]
        with (root / "raw/pytest.log").open("w") as log:
            child = subprocess.Popen(command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.timeout_seconds
            while child.poll() is None:
                reason = ("raw_log_size_limit" if (root / "raw/pytest.log").stat().st_size > MAX_RAW_BYTES
                          else "pytest_timeout" if time.monotonic() > deadline else None)
                if reason:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=20)
                    raise ValueError(reason)
                time.sleep(0.2)
        if (root / "raw/state.json").stat().st_size > MAX_RAW_BYTES:
            raise ValueError("raw_state_size_limit")
        state = json.loads((root / "raw/state.json").read_text())
        totals = sanitized_junit(root / "raw/junit.xml", root / "evidence/junit.xml")
        codes = acceptance(state, args.platform, totals)
        module_counts = {name: sum(x.startswith(name + "::") for x in state["selected"]) for name in REQUIRED_MODULES}
        summary = dict(status="accepted" if not codes else "rejected", platform=args.platform,
            python=sys.version.split()[0], selected=len(state["selected"]), collected=len(state["selected"]) + len(state["deselected"]),
            passed=len(state["passed"]), failed=len(state["failed"]), errors=len(state["errors"]),
            skipped=len(state["skipped"]), deselected=len(state["deselected"]), warnings=state["warnings"],
            native_passed=len(set(state["passed"]) & NATIVE_CASES), runtime_seconds=state["runtime_seconds"],
            one_pytest_process=True, installed_wheel_probe=proof, module_counts=module_counts,
            linux_native_validation=args.platform == "linux" and not codes, publication_approved=False)
        bounded_json(root / "evidence/summary.json", summary)
        bounded_json(root / "evidence/diagnostics.json", {"codes": codes, "raw_details_uploaded": False})
        validate_uploads(root / "evidence")
        print(json.dumps(summary, sort_keys=True))
        return 0 if not codes else 1
    except Exception as error:
        known = str(error) if isinstance(error, ValueError) else "operational_error"
        codes = [known] if known in {"installed_wheel_probe_failed", "pytest_timeout", "evidence_size_limit", "raw_junit_size_limit", "raw_log_size_limit", "raw_state_size_limit"} else ["operational_error"]
        bounded_json(root / "evidence/summary.json", {"status": "rejected", "platform": args.platform, "publication_approved": False})
        bounded_json(root / "evidence/diagnostics.json", {"codes": codes, "raw_details_uploaded": False})
        print(json.dumps({"status": "rejected", "codes": codes}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

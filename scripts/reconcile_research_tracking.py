"""Read-only local reconciliation; no live client is implicitly constructed.

Optional --remote-snapshot accepts the data-only schema documented by
read_offline_remote_snapshot. Snapshot data never certifies current remote
state. Live callers must inject a BoundMlflowClient with search/get capabilities
through the Python API; this CLI accepts no credentials or implicit URI.
Exit codes: 0 consistent scope, 2 actionable findings, 3 incomplete evidence,
4 operational failure. Reports never authorize recovery or run creation.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os

from src.research.bundle import _open_root
from src.tracking.research_mlflow import TrackingDestination
from src.tracking.research_reconciliation import reconcile_research_tracking, read_offline_remote_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--archive-alias", required=True)
    parser.add_argument("--journal-root", required=True, type=Path)
    parser.add_argument("--tracking-destination", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--remote-snapshot", type=Path)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)
    try:
        destination = TrackingDestination(args.tracking_destination, args.experiment_id)
        snapshot = None if args.remote_snapshot is None else read_offline_remote_snapshot(args.remote_snapshot.absolute())
        report = reconcile_research_tracking(archive_root=args.archive_root, archive_alias=args.archive_alias,
            journal_root=args.journal_root, destination=destination, offline_snapshot=snapshot)
        text = json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.json_report is not None:
            output = args.json_report.absolute()
            if any(output.is_relative_to(p.absolute()) for p in (args.archive_root, args.journal_root)):
                raise ValueError("report_must_be_outside_inspected_roots")
            fd = _open_root(output.parent)
            try:
                target = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
                with os.fdopen(target, "w", encoding="utf-8") as stream:
                    stream.write(text)
            finally:
                os.close(fd)
        print(text, end="")
        return report.exit_code
    except Exception:
        print(json.dumps({"schema_version": "apf-research-reconciliation-v1", "exit_code": 4,
                          "scope_findings": [{"status": "operational_error", "code": "invalid_or_unavailable_reconciliation_scope"}]}))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())

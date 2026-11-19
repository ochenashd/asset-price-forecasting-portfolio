"""Read-only scientific archive checks; no model loading or certification."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os

from src.research.bundle import BundleLimits
from src.research.bundle_verification import VerificationLimits, verify_research_bundle

EXIT_CODES = {"verified": 0, "failed": 2, "incomplete": 3, "unsupported": 4, "operational_error": 5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--max-artifact-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-manifest-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-artifacts", type=int, default=128)
    parser.add_argument("--max-json-depth", type=int, default=64)
    parser.add_argument("--max-json-nodes", type=int, default=500000)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)
    try:
        limits = VerificationLimits(BundleLimits(args.max_artifacts, args.max_artifact_bytes,
            args.max_total_bytes, args.max_manifest_bytes), args.max_json_depth, args.max_json_nodes)
        report = verify_research_bundle(args.bundle_dir, args.expected_bundle_id, limits=limits)
        text = json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.json_report is not None:
            # Explicit output only: no overwrites, symlink traversal or writes in
            # the publication. Open parents with the publisher's secure walker.
            from src.research.bundle import _open_root
            output = args.json_report.absolute()
            bundle = args.bundle_dir.absolute()
            if output == bundle or bundle in output.parents:
                raise ValueError("Report must be outside the published bundle.")
            fd = _open_root(output.parent)
            try:
                target = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
                with os.fdopen(target, "w", encoding="utf-8") as stream:
                    stream.write(text)
            finally:
                os.close(fd)
        print(text, end="")
        return EXIT_CODES[report.status]
    except Exception as error:
        print(json.dumps({"status": "operational_error", "diagnostic": type(error).__name__}))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())

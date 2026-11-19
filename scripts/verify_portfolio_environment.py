"""Offline dependency evidence contracts. This module never installs or fetches.

Schema v1 represents P4-C offline evidence only. It cannot promote a partial lock
to verified installation/build evidence by editing a status or a boolean.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import tomllib

TARGETS = {"macos-arm64-cp3115": ("Darwin", "arm64"), "linux-x86_64-cp3115": ("Linux", "x86_64")}
PROFILES = {"core", "tracking", "test", "build"}
ORIGINS = {"p4c_resolver_archive", "p4c_retained_sdist_build", "interpreter_ensurepip_bundle"}
LIMITS = ["no_clean_hash_enforced_install", "no_repeated_build", "bootstrap_closure_unverified", "vulnerability_status_unverified", "license_evidence_incomplete", "linux_runtime_unverified"]


class EvidenceError(ValueError):
    """An explicit code, without private paths or arbitrary exception payloads."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def exact(value, keys):
    require(type(value) is dict and set(value) == set(keys), "invalid_schema")


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid_hash")


def name(value):
    require(type(value) is str and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value), "invalid_package_name")


def version(value):
    # Deliberately bounded to the exact P4-C version syntax, not a PEP 440 resolver.
    require(type(value) is str and re.fullmatch(r"\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?", value), "invalid_version")


def relative(value):
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9_./-]+", value) and not value.startswith("/")
            and all(x not in {"", ".", ".."} for x in value.split("/")), "unsafe_relative_path")


def unique_strings(values):
    require(type(values) is list and all(type(v) is str for v in values) and len(values) == len(set(values)), "duplicate_or_invalid_records")


def wheel_tags(artifact):
    filename = artifact["filename"]
    require(type(filename) is str and re.fullmatch(r"[A-Za-z0-9_.-]+\.whl", filename), "invalid_filename")
    pieces = filename[:-4].split("-")
    require(len(pieces) == 5, "unsupported_wheel_filename")
    normalized = re.sub(r"[-_.]+", "-", pieces[0]).lower()
    require(normalized == artifact["name"] and pieces[1] == artifact["version"], "filename_identity_mismatch")
    tags = sorted(f"{py}-{abi}-{plat}" for py in pieces[2].split(".") for abi in pieces[3].split(".") for plat in pieces[4].split("."))
    require(artifact["tags"] == tags, "filename_tag_mismatch")
    return tags


def compatible(artifact, target):
    for tag in artifact["tags"]:
        py, abi, plat = tag.split("-", 2)
        python_ok = py in {"py3", "cp311"} or (abi == "abi3" and re.fullmatch(r"cp3\d+", py) and int(py[3:]) <= 11)
        if not python_ok:
            continue
        if plat == "any" and abi == "none":
            return True
        if target.startswith("macos") and plat.startswith("macosx_") and plat.endswith(("_arm64", "_universal2")):
            return True
        if target.startswith("linux") and plat.endswith("_x86_64") and plat.startswith(("manylinux", "linux_")):
            return True
    return False


def _validate_manifest(data):
    exact(data, {"schema", "candidate", "python", "marker_contexts", "direct_requirements", "build_system", "generation", "artifacts", "profiles", "auxiliary_locks", "ci_blockers"})
    require(data["schema"] == "apf-dependency-evidence-v1", "invalid_schema")
    exact(data["candidate"], {"id", "v3_inventory_sha256", "scientific_sources"})
    require(data["candidate"]["id"] == "apf-portfolio-candidate-v4-work", "invalid_candidate")
    digest(data["candidate"]["v3_inventory_sha256"])
    sources = data["candidate"]["scientific_sources"]
    require(type(sources) is dict and len(sources) == 50, "scientific_reference_missing")
    for path, value in sources.items():
        relative(path); digest(value)
        require(path.startswith(("src/", "examples/")), "scientific_reference_invalid")
    require(data["python"] == {"implementation": "CPython", "requires": ">=3.11,<3.12", "locked_version": "3.11.5"}, "unsupported_python")
    require(data["marker_contexts"] == {target: {"sys_platform": "darwin" if target.startswith("macos") else "linux", "platform_system": pair[0], "platform_machine": pair[1], "python_version": "3.11", "python_full_version": "3.11.5", "implementation_name": "cpython", "implementation_version": "3.11.5", "platform_python_implementation": "CPython", "os_name": "posix"} for target, pair in TARGETS.items()}, "unsupported_marker_context")
    require(data["build_system"] == {"backend": "setuptools.build_meta", "requirements": ["setuptools==68.0.0", "wheel==0.38.4"], "closure_verified": False}, "invalid_build_system")
    require(data["build_system"]["closure_verified"] is False, "invalid_build_system")
    require(type(data["direct_requirements"]) is dict and set(data["direct_requirements"]) == PROFILES, "invalid_direct_requirements")
    direct_names = {}
    for role, requirements in data["direct_requirements"].items():
        unique_strings(requirements); direct_names[role] = set()
        for requirement in requirements:
            require(re.fullmatch(r"[A-Za-z0-9_.-]+(?:==|>=|<=|!=|~=|>|<)[0-9A-Za-z.]+(?:,(?:==|>=|<=|!=|~=|>|<)[0-9A-Za-z.]+)*", requirement), "invalid_direct_requirements")
            raw_name = re.match(r"[A-Za-z0-9_.-]+", requirement)[0]
            normalized = re.sub(r"[-_.]+", "-", raw_name).lower()
            require(normalized not in direct_names[role], "duplicate_direct_requirement")
            direct_names[role].add(normalized)
    exact(data["generation"], {"method", "p4c_report_sha256", "limitations", "offline_only"})
    require(data["generation"]["method"] == "p4c_frozen_installed_metadata_and_local_archives" and data["generation"]["offline_only"] is True
            and data["generation"]["limitations"] == LIMITS, "unsupported_evidence_claim")
    digest(data["generation"]["p4c_report_sha256"])
    require(type(data["artifacts"]) is list and data["artifacts"], "missing_artifacts")
    artifacts = {}
    for artifact in data["artifacts"]:
        exact(artifact, {"name", "version", "filename", "sha256", "tags", "origin", "source_sha256", "metadata_sha256", "requires_dist", "requires_python"})
        name(artifact["name"]); version(artifact["version"]); digest(artifact["sha256"]); digest(artifact["metadata_sha256"])
        require(artifact["name"] not in artifacts, "duplicate_conflicting_package")
        require(artifact["origin"] in ORIGINS, "unsupported_provenance")
        require(type(artifact["requires_dist"]) is list and all(type(x) is str and "\n" not in x and "://" not in x for x in artifact["requires_dist"]), "invalid_requirement_metadata")
        require(artifact["requires_python"] is None or (type(artifact["requires_python"]) is str and re.fullmatch(r"[0-9a-zA-Z.,<>=!~* +_-]+", artifact["requires_python"])), "invalid_requirement_metadata")
        if artifact["origin"] == "p4c_retained_sdist_build":
            digest(artifact["source_sha256"])
        else:
            require(artifact["source_sha256"] is None, "invalid_source_evidence")
        wheel_tags(artifact)
        artifacts[artifact["name"]] = artifact
    require({"setuptools", "wheel"} <= set(artifacts), "missing_build_dependency")
    require(artifacts["setuptools"]["version"] == "68.0.0" and artifacts["wheel"]["version"] == "0.38.4", "missing_build_dependency")
    require(type(data["profiles"]) is list, "invalid_schema")
    profiles = {}
    for record in data["profiles"]:
        exact(record, {"target", "profile", "status", "lock_file", "lock_sha256", "packages", "artifacts", "roots", "edges", "unresolved"})
        target, profile = record["target"], record["profile"]
        require(target in TARGETS and profile in PROFILES, "unsupported_target_or_profile")
        require((target, profile) not in profiles, "duplicate_profile")
        expected_status = "LOCK_PARTIAL" if target.startswith("macos") else "LOCK_UNAVAILABLE"
        require(record["status"] == expected_status, "partial_evidence_labeled_complete")
        for field in ("packages", "artifacts", "roots"):
            unique_strings(record[field])
            for value in record[field]: name(value)
        packages = set(record["packages"])
        require(set(record["roots"]) == direct_names[profile], "direct_requirement_mismatch")
        require(packages and set(record["roots"]) <= packages and packages <= set(artifacts), "missing_transitive_dependency")
        require(set(record["artifacts"]) <= packages, "missing_transitive_dependency")
        for package in record["artifacts"]:
            require(compatible(artifacts[package], target), "incompatible_target_artifact")
        unresolved_names = set()
        require(type(record["unresolved"]) is list, "invalid_schema")
        for item in record["unresolved"]:
            exact(item, {"name", "version", "reason"}); name(item["name"])
            require(item["name"] not in unresolved_names, "duplicate_unresolved")
            unresolved_names.add(item["name"])
            require(item["reason"] in {"missing_target_wheel", "missing_distribution_metadata"}, "invalid_unresolved_reason")
            if item["reason"] == "missing_distribution_metadata":
                require(item["version"] is None and item["name"] not in packages, "invented_dependency_version")
            else:
                require(item["name"] in packages and item["name"] not in record["artifacts"] and item["version"] == artifacts[item["name"]]["version"], "invalid_missing_artifact")
        require(packages - set(record["artifacts"]) <= unresolved_names, "missing_transitive_dependency")
        require(type(record["edges"]) is list, "invalid_schema")
        reachable = set(record["roots"]); pairs = set()
        for edge in record["edges"]:
            exact(edge, {"parent", "child"}); name(edge["parent"]); name(edge["child"])
            require((edge["parent"], edge["child"]) not in pairs, "duplicate_dependency_edge")
            pairs.add((edge["parent"], edge["child"]))
            require(edge["parent"] in packages and edge["child"] in packages | unresolved_names, "missing_transitive_dependency")
        while True:
            expanded = reachable | {child for parent, child in pairs if parent in reachable}
            if expanded == reachable: break
            reachable = expanded
        require(reachable == packages | unresolved_names, "missing_transitive_dependency")
        if profile == "build":
            require(packages == {"setuptools", "wheel"} and set(record["roots"]) == packages, "missing_build_dependency")
        if target.startswith("macos"):
            require(not record["unresolved"] and set(record["artifacts"]) == packages, "missing_transitive_dependency")
            relative(record["lock_file"]); digest(record["lock_sha256"])
        else:
            require(record["lock_file"] is None and record["lock_sha256"] is None, "misleading_linux_lock")
        profiles[target, profile] = record
    require(set(profiles) == {(t, p) for t in TARGETS for p in PROFILES}, "missing_required_profile")
    for target in TARGETS:
        core = set(profiles[target, "core"]["packages"]); tracking = set(profiles[target, "tracking"]["packages"]); tests = set(profiles[target, "test"]["packages"])
        require(core < tracking < tests and "mlflow" not in core and "pytest" not in tracking and {"mlflow", "pytest"} <= tests, "profile_separation_invalid")
        require(not {"lightgbm", "psycopg2", "psycopg2-binary"} & tests, "unsupported_legacy_dependency")
    require(type(data["auxiliary_locks"]) is list and len(data["auxiliary_locks"]) == 2, "invalid_auxiliary_locks")
    for item in data["auxiliary_locks"]:
        exact(item, {"role", "lock_file", "lock_sha256", "artifacts", "status"})
        require(item["role"] in {"core-pytest-only", "bootstrap-input-only"} and item["status"] == "LOCK_PARTIAL", "invalid_auxiliary_locks")
        relative(item["lock_file"]); digest(item["lock_sha256"]); unique_strings(item["artifacts"])
        require(set(item["artifacts"]) <= set(artifacts), "missing_auxiliary_dependency")
    require({x["role"] for x in data["auxiliary_locks"]} == {"core-pytest-only", "bootstrap-input-only"}, "invalid_auxiliary_locks")
    lock_paths = [x["lock_file"] for x in data["profiles"] if x["lock_file"]] + [x["lock_file"] for x in data["auxiliary_locks"]]
    require(len(lock_paths) == len(set(lock_paths)) and all(x.startswith("requirements/locks/") and x.endswith(".partial.txt") for x in lock_paths), "invalid_lock_path")
    require(data["ci_blockers"] == {"filesystem": "separate_linux_filesystem_profile_unresolved_omegaconf_2_3_0"}, "invalid_ci_blockers")
    return data


def validate_manifest(data):
    try:
        return _validate_manifest(data)
    except EvidenceError:
        raise
    except (KeyError, TypeError, AttributeError):
        raise EvidenceError("invalid_schema") from None


def canonical_bytes(data):
    validate_manifest(data)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key"); result[key] = value
    return result


def read_manifest(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 4 * 1024 * 1024, "unsafe_manifest_file")
    try:
        data = json.loads(path.read_text(), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError):
        raise EvidenceError("invalid_json") from None
    return validate_manifest(data)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def select_profile(data, target, profile):
    require(target in TARGETS, "unsupported_target_or_profile")
    if profile == "filesystem":
        raise EvidenceError("filesystem_lock_unavailable")
    require(profile in PROFILES, "unsupported_target_or_profile")
    return next(x for x in data["profiles"] if x["target"] == target and x["profile"] == profile)


def require_complete(data, target, profile):
    validate_manifest(data)
    record = select_profile(data, target, profile)
    require(record["status"] == "LOCK_VERIFIED", "required_lock_incomplete")


def verify_checkout(data, checkout):
    validate_manifest(data)
    checkout = Path(checkout).resolve()
    config = tomllib.loads((checkout/"pyproject.toml").read_text())
    core = config["project"]["dependencies"]; extras = config["project"]["optional-dependencies"]
    require(data["direct_requirements"] == {"core": core, "tracking": core + extras["tracking"], "test": core + extras["tracking"] + extras["test"], "build": config["build-system"]["requires"]}, "project_requirements_changed")
    observed = {str(p.relative_to(checkout)) for sub in ("src", "examples") for p in (checkout/sub).rglob("*") if p.is_file()}
    require(observed == set(data["candidate"]["scientific_sources"]), "scientific_source_membership_changed")
    def file(rel):
        path = checkout/rel
        require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(checkout), "unsafe_checkout_file")
        return path
    for rel, expected in data["candidate"]["scientific_sources"].items():
        require(file_sha256(file(rel)) == expected, "scientific_source_hash_changed")
    catalog = {x["name"]: x for x in data["artifacts"]}
    records = [x for x in data["profiles"] if x["lock_file"]] + data["auxiliary_locks"]
    for item in records:
        path = file(item["lock_file"])
        require(file_sha256(path) == item["lock_sha256"], "lock_file_hash_mismatch")
        entries = {}
        for line in path.read_text().splitlines():
            if not line or line.startswith("#"): continue
            match = re.fullmatch(r"([a-z0-9-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})", line)
            require(match is not None and match[1] not in entries, "invalid_lock_entry")
            entries[match[1]] = (match[2], match[3])
        expected = {n: (catalog[n]["version"], catalog[n]["sha256"]) for n in item["artifacts"]}
        require(entries == expected, "lock_manifest_mismatch")


def verify_hashes(data, target, profile, artifact_roots):
    validate_manifest(data)
    record = select_profile(data, target, profile)
    require(not record["unresolved"], "lock_has_unresolved_dependencies")
    require(artifact_roots, "explicit_artifact_roots_required")
    roots = [Path(x) for x in artifact_roots]
    require(all(x.is_dir() and not x.is_symlink() for x in roots), "unsafe_artifact_root")
    catalog = {x["name"]: x for x in data["artifacts"]}; verified = []
    for n in record["artifacts"]:
        item = catalog[n]; found = [root/item["filename"] for root in roots if (root/item["filename"]).exists() or (root/item["filename"]).is_symlink()]
        require(found, "missing_archive")
        for path in found:
            require(path.is_file() and not path.is_symlink(), "unsafe_archive")
            require(file_sha256(path) == item["sha256"], "archive_hash_mismatch")
        verified.append(n)
    return {"verified_archives": len(verified), "lock_status": record["status"], "installation_verified": False}


def actual_target():
    require(platform.python_implementation() == "CPython" and platform.python_version() == "3.11.5", "unsupported_python")
    pair = (platform.uname().system, platform.machine())
    require(pair in TARGETS.values(), "unsupported_target_or_profile")
    return next(k for k, v in TARGETS.items() if v == pair)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", choices=("inspect", "check-hashes", "require-complete"), default="inspect")
    parser.add_argument("--artifact-root", action="append", type=Path, default=[])
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args(argv)
    try:
        data = read_manifest(args.manifest)
        if args.checkout: verify_checkout(data, args.checkout)
        record = select_profile(data, args.target, args.profile)
        result = {"target": args.target, "profile": args.profile, "lock_status": record["status"], "packages": len(record["packages"]), "unresolved": len(record["unresolved"]), "manifest_sha256": hashlib.sha256(canonical_bytes(data)).hexdigest(), "installation_verified": False}
        if args.mode == "require-complete":
            require(actual_target() == args.target, "requested_target_is_not_host")
            require_complete(data, args.target, args.profile)
        elif args.mode == "check-hashes":
            result.update(verify_hashes(data, args.target, args.profile, args.artifact_root))
        print(json.dumps(result, sort_keys=True)); return 0
    except EvidenceError as error:
        print(json.dumps({"status": "rejected", "code": error.code, "installation_verified": False})); return 2
    except (OSError, ValueError, TypeError):
        print(json.dumps({"status": "rejected", "code": "invalid_or_unreadable_evidence", "installation_verified": False})); return 2


if __name__ == "__main__":
    raise SystemExit(main())

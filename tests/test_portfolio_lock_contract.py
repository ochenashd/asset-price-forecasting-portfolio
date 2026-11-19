"""Offline dependency contracts; synthetic bytes are not provider certification."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("portfolio_lock_verifier_tests", ROOT/"scripts/verify_portfolio_environment.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
MANIFEST = ROOT/"requirements/locks/manifest.json"
MAC = "macos-arm64-cp3115"
LINUX = "linux-x86_64-cp3115"


@pytest.fixture
def evidence():
    return verifier.read_manifest(MANIFEST)


def synthetic_wheelhouse(tmp_path, evidence):
    # Only the hashing contract is under test; these are not installable wheels.
    for item in evidence["artifacts"]:
        path = tmp_path/item["filename"]
        path.write_bytes(b"synthetic-hash-contract:"+item["filename"].encode())
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tmp_path


def artifact(evidence, name):
    return next(x for x in evidence["artifacts"] if x["name"] == name)


def profile(evidence, target=MAC, role="core"):
    return verifier.select_profile(evidence, target, role)


def test_current_manifest_and_scientific_source_integrity(evidence):
    assert verifier.validate_manifest(evidence) is evidence
    verifier.verify_checkout(evidence, ROOT)
    assert len(evidence["candidate"]["scientific_sources"]) == 50
    assert len(evidence["artifacts"]) == 97
    assert all(x["status"] != "LOCK_VERIFIED" for x in evidence["profiles"])


@pytest.mark.parametrize("role,count", [("core",16),("tracking",91),("test",94),("build",2)])
def test_macos_partial_profiles_are_explicit(evidence,role,count):
    item=profile(evidence,role=role)
    assert item["status"] == "LOCK_PARTIAL" and len(item["packages"]) == count
    assert not item["unresolved"]
    with pytest.raises(verifier.EvidenceError,match="required_lock_incomplete"):
        verifier.require_complete(evidence,MAC,role)


def test_core_tracking_full_test_and_build_scopes(evidence):
    core=set(profile(evidence)["packages"])
    tracking=set(profile(evidence,role="tracking")["packages"])
    tests=set(profile(evidence,role="test")["packages"])
    assert core < tracking < tests
    assert not {"mlflow","pytest","matplotlib"} & core
    assert {"mlflow","matplotlib"} <= tracking and "pytest" not in tracking
    assert tests-tracking == {"pytest","pluggy","iniconfig"}
    assert set(profile(evidence,role="build")["packages"]) == {"setuptools","wheel"}
    assert not {"lightgbm","psycopg2","psycopg2-binary"} & tests


@pytest.mark.parametrize("role,binary,metadata", [("core",8,2),("tracking",25,2),("test",25,2),("build",0,0)])
def test_linux_missing_wheels_are_profile_specific(evidence,role,binary,metadata):
    item=profile(evidence,LINUX,role)
    assert item["status"] == "LOCK_UNAVAILABLE" and item["lock_file"] is None
    assert sum(x["reason"]=="missing_target_wheel" for x in item["unresolved"]) == binary
    unknown=[x for x in item["unresolved"] if x["reason"]=="missing_distribution_metadata"]
    assert len(unknown)==metadata
    if metadata:
        assert {x["name"] for x in unknown} == {"greenlet","nvidia-nccl-cu12"}
        assert all(x["version"] is None for x in unknown)
    with pytest.raises(verifier.EvidenceError,match="required_lock_incomplete"):
        verifier.require_complete(evidence,LINUX,role)


def test_correct_hash_contract_preserves_partial_classification(tmp_path,evidence):
    synthetic_wheelhouse(tmp_path,evidence)
    result=verifier.verify_hashes(evidence,MAC,"core",[tmp_path])
    assert result == {"verified_archives":16,"lock_status":"LOCK_PARTIAL","installation_verified":False}


@pytest.mark.parametrize("fault,code", [("corrupt","archive_hash_mismatch"),("missing","missing_archive"),("symlink","unsafe_archive"),("wrong_filename","missing_archive")])
def test_invalid_archive_bytes_and_missing_files_reject(tmp_path,evidence,fault,code):
    synthetic_wheelhouse(tmp_path,evidence)
    item=artifact(evidence,"numpy");path=tmp_path/item["filename"]
    if fault=="corrupt":path.write_bytes(b"tampered synthetic bytes")
    elif fault=="missing":path.unlink()
    elif fault=="wrong_filename":path.rename(tmp_path/"wrong-name.whl")
    else:
        other=tmp_path/"owned-sentinel";path.rename(other);path.symlink_to(other)
    with pytest.raises(verifier.EvidenceError,match=code):verifier.verify_hashes(evidence,MAC,"core",[tmp_path])


def test_conflicting_same_filename_in_another_root_is_not_silently_ignored(tmp_path,evidence):
    left=tmp_path/"left";right=tmp_path/"right";left.mkdir();right.mkdir()
    synthetic_wheelhouse(left,evidence)
    (right/artifact(evidence,"numpy")["filename"]).write_bytes(b"conflicting synthetic bytes")
    with pytest.raises(verifier.EvidenceError,match="archive_hash_mismatch"):
        verifier.verify_hashes(evidence,MAC,"core",[left,right])


@pytest.mark.parametrize("fault,code", [
    ("missing_hash","invalid_schema"),("bad_hash","invalid_hash"),("bad_filename","filename_identity_mismatch"),
    ("duplicate_package","duplicate_conflicting_package"),("invalid_name","invalid_package_name"),
    ("invalid_version","invalid_version"),("invalid_schema","invalid_schema"),("unknown_key","invalid_schema"),
    ("unsupported_target","unsupported_target_or_profile"),("marker_context","unsupported_marker_context"),
    ("missing_transitive","missing_transitive_dependency"),("missing_build","missing_build_dependency"),
    ("complete_claim","partial_evidence_labeled_complete"),("network_fallback","invalid_schema"),
    ("unsafe_lock_path","unsafe_relative_path"),("invented_linux_version","invented_dependency_version"),
    ("incompatible_tag","filename_tag_mismatch"),("duplicate_profile","duplicate_profile"),
    ("missing_profile","missing_required_profile"),("build_verified","invalid_build_system"),
    ("metadata_url","invalid_requirement_metadata"),("linux_install_file","misleading_linux_lock"),
])
def test_schema_and_integrity_mutations_fail_closed(evidence,fault,code):
    if fault=="missing_hash":del artifact(evidence,"numpy")["sha256"]
    elif fault=="bad_hash":artifact(evidence,"numpy")["sha256"]="not-a-hash"
    elif fault=="bad_filename":artifact(evidence,"numpy")["filename"]="other-1.0-py3-none-any.whl"
    elif fault=="duplicate_package":evidence["artifacts"].append(copy.deepcopy(artifact(evidence,"numpy")))
    elif fault=="invalid_name":artifact(evidence,"numpy")["name"]="../numpy"
    elif fault=="invalid_version":artifact(evidence,"numpy")["version"]="latest"
    elif fault=="invalid_schema":evidence["schema"]="unknown-v2"
    elif fault=="unknown_key":evidence["unreviewed"]={}
    elif fault=="unsupported_target":evidence["profiles"][0]["target"]="windows-x86_64"
    elif fault=="marker_context":evidence["marker_contexts"][MAC]["python_full_version"]="3.12.0"
    elif fault=="missing_transitive":profile(evidence)["packages"].remove("six")
    elif fault=="missing_build":evidence["artifacts"]=[x for x in evidence["artifacts"] if x["name"]!="setuptools"]
    elif fault=="complete_claim":profile(evidence)["status"]="LOCK_VERIFIED"
    elif fault=="network_fallback":evidence["generation"]["network_fallback"]=True
    elif fault=="unsafe_lock_path":profile(evidence)["lock_file"]="../private.txt"
    elif fault=="invented_linux_version":
        next(x for x in profile(evidence,LINUX)["unresolved"] if x["name"]=="greenlet")["version"]="1.0"
    elif fault=="incompatible_tag":artifact(evidence,"numpy")["tags"]=["cp311-cp311-manylinux_2_17_x86_64"]
    elif fault=="duplicate_profile":evidence["profiles"].append(copy.deepcopy(evidence["profiles"][0]))
    elif fault=="missing_profile":evidence["profiles"].pop()
    elif fault=="build_verified":evidence["build_system"]["closure_verified"]=True
    elif fault=="metadata_url":artifact(evidence,"numpy")["requires_dist"].append("private @ https://synthetic.invalid/package")
    elif fault=="linux_install_file":profile(evidence,LINUX)["lock_file"]="requirements/locks/fake.txt"
    with pytest.raises(verifier.EvidenceError,match=code):verifier.validate_manifest(evidence)


def test_deterministic_canonical_serialization_without_input_mutation(evidence):
    before=copy.deepcopy(evidence)
    reordered=json.loads(json.dumps(evidence,sort_keys=True))
    assert verifier.canonical_bytes(evidence)==verifier.canonical_bytes(reordered)
    assert evidence==before
    companion=MANIFEST.with_suffix('.sha256').read_text().split()[0]
    assert companion==hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def test_duplicate_json_keys_are_rejected_before_schema_interpretation(tmp_path):
    file=tmp_path/"duplicate.json";file.write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(verifier.EvidenceError,match="duplicate_json_key"):verifier.read_manifest(file)


def test_inspection_and_rejection_never_access_network_or_mutate_files(monkeypatch,evidence,capsys):
    def denied(*args,**kwargs):raise AssertionError("No network call is permitted")
    monkeypatch.setattr(socket,"create_connection",denied)
    monkeypatch.setattr(socket,"getaddrinfo",denied)
    before=MANIFEST.read_bytes()
    assert verifier.main(["--manifest",str(MANIFEST),"--target",MAC,"--profile","core","--mode","inspect"])==0
    output=json.loads(capsys.readouterr().out)
    assert output["lock_status"]=="LOCK_PARTIAL" and not output["installation_verified"]
    assert verifier.main(["--manifest",str(MANIFEST),"--target",MAC,"--profile","test","--mode","require-complete"])==2
    assert json.loads(capsys.readouterr().out)["status"]=="rejected"
    assert MANIFEST.read_bytes()==before


def test_verifier_is_stdlib_only_and_has_no_install_fit_or_inference_calls():
    tree=ast.parse((ROOT/"scripts/verify_portfolio_environment.py").read_text())
    imports={n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import)}
    imports|={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imports <= set(sys.stdlib_module_names)
    assert not imports & {"socket","subprocess","pip","urllib","http"}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {"fit","predict","predict_proba","load_model","urlopen","connect","system"} for n in ast.walk(tree))


def test_manifest_contains_no_private_paths_credentials_or_source_payloads(evidence):
    raw=verifier.canonical_bytes(evidence).decode()
    assert all(text not in raw for text in ["/Users/","/private/","file://","password","GITHUB_TOKEN","BEGIN PRIVATE"])
    assert set(evidence["candidate"]["scientific_sources"])=={str(p.relative_to(ROOT)) for sub in ("src","examples") for p in (ROOT/sub).rglob('*') if p.is_file()}
    assert all(len(x)==64 for x in evidence["candidate"]["scientific_sources"].values())


@pytest.mark.parametrize("filename,role",[("portfolio-tests.yml","test"),("research-filesystem-linux.yml","filesystem")])
def test_manual_workflows_block_before_any_dependency_install(filename,role):
    workflow=yaml.load((ROOT/".github/workflows"/filename).read_text(),Loader=yaml.BaseLoader)
    assert set(workflow["on"])=={"workflow_dispatch"} and workflow["permissions"]=={"contents":"read"}
    steps=next(iter(workflow["jobs"].values()))["steps"]
    gates=[i for i,x in enumerate(steps) if "--mode require-complete" in x.get("run","")]
    installs=[i for i,x in enumerate(steps) if "pip install" in x.get("run","") or "pip wheel" in x.get("run","")]
    assert len(gates)==1 and gates[0]<min(installs)
    assert "timeout --kill-after=10s 60s" in steps[gates[0]]["run"]
    assert f'--profile {role}' in steps[gates[0]]["run"]
    assert "--checkout ." in steps[gates[0]]["run"] and "--target linux-x86_64-cp3115" in steps[gates[0]]["run"]
    assert "continue-on-error" not in steps[gates[0]]
    for x in steps:
        if "run" in x:assert subprocess.run(["bash","-n"],input=x["run"],capture_output=True,text=True).returncode==0


def test_driver_retains_every_v3_mandatory_module_and_all_native_cases():
    spec=importlib.util.spec_from_file_location("p4d1_full_ci_driver",ROOT/"scripts/portfolio_ci.py")
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    assert len(driver.REQUIRED_MODULES)==32 and len(driver.NATIVE_CASES)==8
    assert "tests/test_portfolio_lock_contract.py" in driver.REQUIRED_MODULES
    assert "tests/test_portfolio_documentation.py" in driver.REQUIRED_MODULES
    baseline=sorted(set(driver.REQUIRED_MODULES)-{"tests/test_portfolio_lock_contract.py", "tests/test_portfolio_documentation.py"})
    assert hashlib.sha256(json.dumps(baseline).encode()).hexdigest()=="f03bd2d00837e2ff6be173421b0c0008419f5750e990a2634a29adb899d5d188"
    assert all((ROOT/x).is_file() for x in driver.REQUIRED_MODULES)
    for kind in ("macos","linux"):
        with pytest.raises(ValueError,match="required_lock_incomplete"):driver.dependency_preflight(ROOT,kind)
    text=(ROOT/"scripts/portfolio_ci.py").read_text()
    assert text.index('dependency_preflight(checkout, args.platform)') < text.index('if args.worker:')
    assert 'args = ["tests"' in text and 'if kind == "macos": args += ["-k", "not native_linux"]' in text


def test_docs_keep_historical_evidence_and_do_not_offer_partial_lock_install():
    text=(ROOT/"docs/dependency_locking.md").read_text()
    assert all(x in text for x in ['LOCK_PARTIAL','LOCK_UNAVAILABLE','VULNERABILITY_STATUS_UNVERIFIED','No clean installation','eight','greenlet','nvidia-nccl-cu12'])
    assert 'pip install -r requirements/locks/' not in text
    assert "python -m pytest tests -k 'not native_linux'" in (ROOT/"docs/installation.md").read_text()


@pytest.mark.parametrize("fault,code",[("duplicate","duplicate_or_invalid_records"),("root","direct_requirement_mismatch"),("url","invalid_direct_requirements")])
def test_direct_requirements_are_validated_separately_from_metadata_projection(evidence,fault,code):
    if fault=="duplicate":evidence["direct_requirements"]["core"].append(evidence["direct_requirements"]["core"][0])
    elif fault=="root":profile(evidence)["roots"].remove("numpy")
    else:evidence["direct_requirements"]["core"][0]="numpy @ https://synthetic.invalid/distribution"
    with pytest.raises(verifier.EvidenceError,match=code):verifier.validate_manifest(evidence)


def test_missing_archive_never_becomes_verified_via_empty_or_remote_root(evidence,tmp_path):
    with pytest.raises(verifier.EvidenceError,match="explicit_artifact_roots_required"):
        verifier.verify_hashes(evidence,MAC,"core",[])
    with pytest.raises(verifier.EvidenceError,match="missing_archive"):
        verifier.verify_hashes(evidence,MAC,"core",[tmp_path])
    with pytest.raises(verifier.EvidenceError,match="lock_has_unresolved_dependencies"):
        verifier.verify_hashes(evidence,LINUX,"core",[tmp_path])


def test_filesystem_has_a_separate_unavailable_gate(evidence):
    with pytest.raises(verifier.EvidenceError,match="filesystem_lock_unavailable"):
        verifier.require_complete(evidence,LINUX,"filesystem")
    assert "omegaconf_2_3_0" in evidence["ci_blockers"]["filesystem"]


def test_direct_completeness_api_cannot_bypass_schema_by_changing_status(evidence):
    profile(evidence)["status"]="LOCK_VERIFIED"
    with pytest.raises(verifier.EvidenceError,match="partial_evidence_labeled_complete"):
        verifier.require_complete(evidence,MAC,"core")

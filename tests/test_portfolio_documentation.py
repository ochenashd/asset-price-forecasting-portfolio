"""Static documentation contracts; no fitting, inference or service access."""
import ast
from pathlib import Path
import re
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "docs/architecture.md", "docs/modeling.md",
             "docs/installation.md", "docs/reproducibility.md", "docs/limitations.md",
             "LICENSE", "pyproject.toml")


@pytest.mark.parametrize("name", DOCUMENTS)
def test_documentation_has_no_personal_paths_or_credential_urls(name):
    text = (ROOT / name).read_text()
    assert not re.search(r"/Users/|/home/[^\s/]+/|[A-Za-z]:\\Users\\", text)
    assert not re.search(r"\w+://[^\s/]+:[^\s/@]+@", text)
    assert not re.search(r"\b(?:ghp_|github_pat_|AKIA)[A-Za-z0-9_]{12,}", text)


@pytest.mark.parametrize("name", DOCUMENTS)
def test_documentation_links_resolve_inside_candidate(name):
    page = ROOT / name
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", page.read_text()):
        if target.startswith(("https://", "http://", "#")):
            continue
        destination = (page.parent / target.split("#", 1)[0]).resolve()
        assert destination.is_relative_to(ROOT)
        assert destination.exists(), (name, target)


def test_documented_cli_matches_packaging_and_argparse():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["scripts"]["apf-synthetic"] == "examples.synthetic_research:main"
    tree = ast.parse((ROOT / "examples/synthetic_research.py").read_text())
    arguments = {node.args[0].value: node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "add_argument" and node.args
                 and isinstance(node.args[0], ast.Constant)}
    assert set(arguments) == {"--output", "--track-local"}
    assert any(k.arg == "required" and ast.literal_eval(k.value) is True
               for k in arguments["--output"].keywords)
    readme = (ROOT / "README.md").read_text()
    assert "python -I -B -m examples.synthetic_research --output /tmp/apf-demo-new" in readme
    assert "--track-local" in readme
    assert "[project.optional-dependencies]" in (ROOT / "pyproject.toml").read_text()


def test_mermaid_has_declared_nodes_and_separate_post_publication_execution():
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
    assert len(blocks) == 1 and blocks[0].startswith("flowchart TD\n")
    declared = set(re.findall(r"\b([A-Z]+)\[", blocks[0]))
    edges = []
    for line in blocks[0].splitlines()[1:]:
        match = re.fullmatch(r"\s*([A-Z]+)(?:\[[^\]]+\])?\s+(?:-->|-\. [^\n]+ \.->)\s+([A-Z]+)(?:\[[^\]]+\])?\s*", line)
        assert match, line
        edges.append(match.groups())
    assert all(a in declared and b in declared for a, b in edges)
    assert {("G", "M"), ("K", "D"), ("D", "A"), ("M", "A"),
            ("A", "O"), ("R", "I"), ("R", "Q"), ("R", "T")} <= set(edges)
    assert ("O", "R") not in edges and ("A", "R") not in edges
    assert "after publication and are not inside the immutable research bundle" in text


def test_evidence_is_scoped_and_limitations_remain_explicit():
    readme = (ROOT / "README.md").read_text()
    for marker in ("P3-B / v2", "1,335 passed", "P4-B / v3", "1,371 passed",
                   "P4-D1 / v4", "356 passed", "17 verified wheel inputs",
                   "maximum probability difference 0.0", "Zero of eight mandatory native cases executed"):
        assert marker in readme
    assert "not added together" in readme
    for marker in ("incomplete for decision-time evidence", "No general reuse, modification or redistribution permission is granted",
                   "not production trading software", "No financial-performance headline"):
        assert marker in readme
    limitations = (ROOT / "docs/limitations.md").read_text()
    for marker in ("LOCK_PARTIAL", "LOCK_UNAVAILABLE", "VULNERABILITY_STATUS_UNVERIFIED",
                   "BLOCKED_BY_IP_OR_LICENSE", "not a physical"):
        if marker == "not a physical":
            assert "power-loss durability" in limitations
        else:
            assert marker in limitations


def test_model_parameters_and_declared_partition_match_documentation():
    tree = ast.parse((ROOT / "examples/synthetic_research.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "PARAMETERS" for t in node.targets))
    params = {k.arg: ast.literal_eval(k.value) for k in assignment.value.keywords}
    assert (params["n_estimators"], params["max_depth"], params["random_state"],
            params["n_jobs"], params["early_stopping_rounds"]) == (24, 2, 42, 1, 4)
    assert "60 fitting / 5 first purge / 19 validation / 5 second purge / 55 held-out" in (ROOT / "README.md").read_text()


def test_rights_reserved_notice_is_consistent_and_finalized():
    notice = (ROOT / "LICENSE").read_text()
    readme = (ROOT / "README.md").read_text()
    assert notice.startswith("APF Portfolio — Rights Reservation Notice")
    expected_notice = "Copyright (c) 2026 Ochenash. All Rights Reserved."
    assert expected_notice in notice
    assert expected_notice in readme
    assert expected_notice in (ROOT / "docs/limitations.md").read_text()
    assert "TO_BE_CONFIRMED" not in notice
    assert "DRAFT" not in notice
    for text in (notice, readme):
        assert "All Rights Reserved" in text
        assert "not an open-source release" in text
        assert "applicable law" in text
        assert "platform terms" in text
        assert "Third-party components" in text
        assert "written authorization" in text
        assert "local execution" in text
        assert "MIT License" not in text
        assert "licensed under the MIT" not in text
    assert "does not override" in notice
    assert "technically disable viewing, cloning or forking" in notice
    assert "does not authorize publication" in notice
    assert "prior written authorization" in notice
    assert "Public viewing does not automatically authorize local execution" in readme


def test_firstparty_metadata_is_proprietary_without_claiming_osi_or_spdx_grant():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["license"] == {"file": "LICENSE"}
    assert metadata["project"]["classifiers"] == ["License :: Other/Proprietary License"]
    assert not any("OSI Approved" in value or "MIT" in value
                   for value in metadata["project"]["classifiers"])
    assert metadata["build-system"]["requires"] == ["setuptools==68.0.0", "wheel==0.38.4"]
    assert "No general permission" in (ROOT / "LICENSE").read_text()

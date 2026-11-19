"""Exact current public source binding; never rewrites historical identities."""
import hashlib
from importlib import resources
import json
from pathlib import Path
import yaml
import src
import examples.synthetic_research as demo


def test_exact_public_source_and_resource_hashes():
    profile=yaml.safe_load(resources.files("src").joinpath("public_defaults/environment.yaml").read_text())
    assert profile["schema"]=="apf-public-reference-v1"
    assert profile["sources"]
    root=Path(src.__file__).resolve().parent.parent
    for name,expected in profile["sources"].items():
        assert name.startswith(("src/","examples/")) and ".." not in Path(name).parts
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    assert "src/research/bundle.py" in profile["sources"]
    expected=hashlib.sha256(json.dumps(profile["sources"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert profile["source_digest"]==expected


def test_safe_defaults_are_packaged_and_offline():
    config=yaml.safe_load(resources.files("src").joinpath("public_defaults/synthetic.yaml").read_text())
    tracking=yaml.safe_load(resources.files("src").joinpath("public_defaults/tracking.yaml").read_text())
    assert config["database"]["enabled"] is False
    assert config["external_network"] is False
    assert tracking["enabled"] is False
    assert tracking["backend"]=="fresh_local_sqlite_only"

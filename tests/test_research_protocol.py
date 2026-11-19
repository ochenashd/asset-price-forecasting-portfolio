"""Deterministic contract fixtures; no training, execution or provider evidence."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pytest

from src.research.protocol import (
    ColumnSpec, ComputationWindow, EngineeredRow, EnvironmentSpec, EvaluationSpec,
    FrozenPartitionSpec, LearningSpec, ObservationUniverse, ProvenanceLimitations,
    ResearchArtifactRef, ResearchProtocolError, ResearchProtocolV1, SourceObservation,
    TargetSpec, TrainingInputSpec, TransformationSpec, hash_training_rows,
    ProtocolValidationReport, validate_research_artifacts, validate_research_protocol,
)


UTC = timezone.utc
NY = ZoneInfo("America/New_York")


def artifact(path, ids):
    schema = pq.read_schema(path)
    return ResearchArtifactRef("inputs/prices.parquet", "synthetic-prices", "training_ohlcv",
        hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size,
        tuple(ColumnSpec(field.name, str(field.type), field.nullable) for field in schema),
        hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(), "source_id", tuple(ids), label_column="date_label")


def training_hash(frame):
    return hash_training_rows((row.source_id, (row.close, row.volume)) for row in frame.iloc[:25].itertuples())


@pytest.fixture
def research(tmp_path):
    ids = tuple(f"r{i}" for i in range(36))
    directory = tmp_path / "inputs"
    directory.mkdir()
    path = directory / "prices.parquet"
    frame = pd.DataFrame({"source_id": ids, "close": [100.0 + i for i in range(36)],
                          "volume": [1000 + i for i in range(36)], "heldout_label": [i % 3 for i in range(36)],
                          "date_label": [date(2024, 10, 1) + timedelta(days=i) for i in range(36)]})
    frame.to_parquet(path, index=False)
    clocks = tuple(datetime(2024, 10, 1, 16, tzinfo=NY) + timedelta(days=i) for i in range(36))
    observations = tuple(SourceObservation(key, "synthetic-prices", date(2024, 10, 1) + timedelta(days=i),
                          "date32[day]", clocks[i], f"session-{i}", i) for i, key in enumerate(ids))
    partitions = FrozenPartitionSpec(ids[3:12], ids[12:17], ids[17:20], ids[20:25], ids[25:31], ids[3], ids[17], ids[25])
    transformations = TransformationSpec(
        {"columns": "lowercase", "implementation": "synthetic-contract-v1"}, "stable_source_order", "reject",
        {"price": "causal_ffill", "leading_missing": "preserve", "volume": "preserve"},
        {"ema": {"span": 7, "adjust": False, "seed": "first_prefix_observation"}},
        {"threshold": 0.9, "scope": "fitting"},
        {"method": "random_forest", "n_estimators": 300, "max_depth": 6, "random_state": 42, "scope": "fitting"},
        {"class": "RobustScaler", "scope": "fitting", "quantile_range": [25.0, 75.0]},
        ("close", "volume"), ("ema_7", "volume_change"), "src/features/engineering.py@synthetic-revision", "America/New_York")
    evaluation = EvaluationSpec(ids[25], ids[30], clocks[25] - timedelta(hours=7), clocks[35], "session-35",
        "liquidate_at_final_close", 5, "sessions", True, ("synthetic daily marking; no intraday drawdown",),
        {"position_policy": "fixed_horizon", "initial_capital": 10000.0, "allocation_fraction": 0.5,
         "fee_rate": 0.001, "slippage_rate": 0.0005, "allow_short": True, "initial_margin_fraction": 0.5,
         "maintenance_margin_fraction": 0.3, "max_short_exposure": 1.0},
        {"periods_per_year": 252, "annual_risk_free_rate": 0.02, "calendar_days_per_year": 365.25},
        calendar_reference="synthetic-calendar-declaration-v1")
    protocol = ResearchProtocolV1("research:SYNTH", "Synthetic frozen research fixture", (artifact(path, ids),),
        ObservationUniverse(observations, tuple(EngineeredRow(key, key) for key in ids[3:31]),
                            tuple((key, "warm_up" if i < 3 else "label_tail") for i, key in enumerate(ids) if i < 3 or i > 30)),
        ComputationWindow(ids[:31], ids[30], ids[35], ids[25], ids[35]), transformations,
        TargetSpec("5d_class", 5, "observations", tuple((ids[i], ids[i + 5]) for i in range(3, 31))),
        partitions, evaluation,
        LearningSpec("xgboost", {"n_estimators": 40, "tree_method": "hist", "max_depth": 2, "n_jobs": 1},
                     {"estimator": 42, "selector": 42}, "fitting_only", "fixed_purged_validation",
                     {"enabled": True, "rounds": 5, "information_scope": "validation"}, {"n_estimators": 1000}),
        EnvironmentSpec("synthetic-source-revision", False, "reject", None, "3.11.5",
                        {"pandas": pd.__version__, "xgboost": "declared-fixture-version", "scipy": None}, None,
                        {"architecture": "declared-fixture", "xgboost_workers": 1, "blas_threads": None}),
        ProvenanceLimitations("declared_pending_verification", "unknown", ("synthetic calendar and data only",),
                              ("historical_vintage", "vendor_revisions"), evidence_origin="synthetic_fixture"),
        TrainingInputSpec(ids[:25], training_hash(frame)))
    return protocol, tmp_path, path, frame


def codes(report):
    return {item.code for item in report.diagnostics}


def test_round_trip_stable_identity_and_explicit_schema(research):
    protocol, root, _, _ = research
    assert validate_research_protocol(protocol).valid
    text = protocol.canonical_json()
    rebuilt = ResearchProtocolV1.from_json(text)
    assert rebuilt.canonical_json() == text
    assert rebuilt.protocol_id == protocol.protocol_id == replace(protocol).protocol_id
    assert protocol.protocol_id not in text
    assert validate_research_artifacts(protocol, root).verified_artifacts == ("synthetic-prices",)


def test_relocation_and_modification_time_do_not_change_identity(research, tmp_path):
    protocol, root, path, _ = research
    copied = tmp_path / "relocated"
    copied.mkdir()
    shutil.copytree(root / "inputs", copied / "inputs")
    identifier = protocol.protocol_id
    path.touch()
    assert validate_research_artifacts(protocol, root).valid
    assert validate_research_artifacts(protocol, copied).valid
    assert protocol.protocol_id == identifier


@pytest.mark.parametrize("change,expected", [("modify", "sha256_mismatch"), ("append", "byte_length_mismatch"),
                                            ("missing", "missing_artifact")])
def test_changed_artifact_is_rejected_before_decode(research, monkeypatch, change, expected):
    protocol, root, path, _ = research
    if change == "missing":
        path.unlink()
    elif change == "append":
        with path.open("ab") as handle:
            handle.write(b"appended")
    else:
        with path.open("r+b") as handle:
            handle.seek(12)
            byte = handle.read(1)
            handle.seek(12)
            handle.write(bytes([byte[0] ^ 1]))
    def forbidden(*args, **kwargs):
        pytest.fail("Unverified bytes must never reach Parquet decoding")
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    report = validate_research_artifacts(protocol, root)
    assert not report.valid and expected in codes(report)


@pytest.mark.parametrize("locator", ["../prices.parquet", "/tmp/prices.parquet", "inputs/../prices.parquet",
                                     "inputs//prices.parquet", "C:\\prices.parquet"])
def test_traversal_and_ambiguous_locators_are_rejected(research, locator):
    protocol, root, _, _ = research
    bad = replace(protocol, artifacts=(replace(protocol.artifacts[0], relative_path=locator),))
    assert "invalid_artifact" in codes(validate_research_artifacts(bad, root))


@pytest.mark.parametrize("directory_link", [False, True])
def test_symlink_components_cannot_escape_root(research, tmp_path, directory_link):
    protocol, _, path, _ = research
    other = tmp_path / "trusted-root"
    other.mkdir()
    if directory_link:
        (other / "inputs").symlink_to(path.parent, target_is_directory=True)
    else:
        (other / "inputs").mkdir()
        (other / "inputs" / path.name).symlink_to(path)
    report = validate_research_artifacts(protocol, other)
    assert not report.valid and "unsafe_or_unreadable_artifact" in codes(report)


def test_schema_type_and_original_row_order_are_verified(research):
    protocol, root, _, _ = research
    first = protocol.artifacts[0]
    columns = tuple(replace(column, arrow_type="int64") if column.name == "close" else column for column in first.columns)
    assert "schema_mismatch" in codes(validate_research_artifacts(replace(protocol, artifacts=(replace(first, columns=columns),)), root))
    rows = list(protocol.universe.observations)
    rows[0], rows[1] = rows[1], rows[0]
    changed = replace(protocol, universe=replace(protocol.universe, observations=tuple(rows)))
    assert "source_order_mismatch" in codes(validate_research_protocol(changed))


def test_actual_artifact_rows_cannot_be_relabelled(research):
    protocol, root, _, frame = research
    renamed = frame.source_id.tolist()
    renamed[0] = "renamed-first-source"
    first = replace(protocol.artifacts[0], original_row_ids=tuple(renamed))
    universe = replace(protocol.universe,
        observations=(replace(protocol.universe.observations[0], source_id=renamed[0]),) + protocol.universe.observations[1:],
        excluded_rows=((renamed[0], "warm_up"),) + protocol.universe.excluded_rows[1:])
    computation = replace(protocol.computation, historical_prefix_ids=tuple(renamed[:31]))
    training = replace(protocol.training_inputs, source_ids=tuple(renamed[:25]))
    changed = replace(protocol, artifacts=(first,), universe=universe, computation=computation, training_inputs=training)
    assert validate_research_protocol(changed).valid
    assert "source_order_mismatch" in codes(validate_research_artifacts(changed, root))


def test_duplicate_source_ids_and_incomplete_partitions_fail(research):
    protocol, _, _, _ = research
    rows = protocol.universe.observations
    bad = replace(protocol, universe=replace(protocol.universe, observations=(rows[0], rows[0]) + rows[2:]))
    assert "duplicate_identity" in codes(validate_research_protocol(bad))
    p = protocol.partitions
    assert "invalid_partitions" in codes(validate_research_protocol(replace(protocol, partitions=replace(p, fitting=p.fitting[:-1]))))
    assert "invalid_partitions" in codes(validate_research_protocol(replace(protocol, partitions=replace(p, validation=p.validation + p.fitting[:1]))))


@pytest.mark.parametrize("name", ["inner_purge", "outer_purge"])
def test_maturity_boundary_rows_cannot_be_admitted_to_fitting(research, name):
    protocol, _, _, _ = research
    p = protocol.partitions
    row = getattr(p, name)[0]
    fitting = tuple(x.row_id for x in protocol.universe.engineered_rows if x.row_id in p.fitting + (row,))
    changed = replace(p, fitting=fitting, **{name: getattr(p, name)[1:]})
    assert "maturity_partition_mismatch" in codes(validate_research_protocol(replace(protocol, partitions=changed)))


def test_reordered_members_and_changed_boundaries_fail(research):
    protocol, _, _, _ = research
    p = protocol.partitions
    assert "partition_order_mismatch" in codes(validate_research_protocol(replace(protocol, partitions=replace(p, fitting=p.fitting[::-1]))))
    assert "boundary_mismatch" in codes(validate_research_protocol(replace(protocol, partitions=replace(p, test_start=p.held_out[1]))))


def test_five_observation_maturity_survives_missing_feature_rows(research):
    protocol, _, _, _ = research
    assert dict(protocol.target.maturity)["r7"] == "r12"
    universe = replace(protocol.universe, engineered_rows=tuple(row for row in protocol.universe.engineered_rows if row.row_id != "r8"),
        excluded_rows=protocol.universe.excluded_rows[:3] + (("r8", "missing_features"),) + protocol.universe.excluded_rows[3:])
    changed = replace(protocol, universe=universe,
        target=replace(protocol.target, maturity=tuple(pair for pair in protocol.target.maturity if pair[0] != "r8")),
        partitions=replace(protocol.partitions, fitting=tuple(x for x in protocol.partitions.fitting if x != "r8")))
    assert validate_research_protocol(changed).valid
    wrong = tuple((row, "r13" if row == "r7" else end) for row, end in changed.target.maturity)
    assert "invalid_maturity" in codes(validate_research_protocol(replace(changed, target=replace(changed.target, maturity=wrong))))


def test_recursive_prefix_and_label_tail_cannot_be_silently_shortened(research):
    protocol, _, _, _ = research
    w = protocol.computation
    assert "invalid_window" in codes(validate_research_protocol(replace(protocol, computation=replace(w, historical_prefix_ids=w.historical_prefix_ids[1:]))))
    assert "invalid_window" in codes(validate_research_protocol(replace(protocol, computation=replace(w, label_tail_endpoint="r34"))))
    # A declared EMA seed change is part of research identity, even with equal windows.
    changed = replace(protocol, transformations=replace(protocol.transformations, indicators={"ema": {"span": 7, "adjust": False, "seed": "other"}}))
    assert changed.protocol_id != protocol.protocol_id


def test_recipe_target_and_effective_configuration_changes_are_bound(research):
    protocol, _, _, _ = research
    assert replace(protocol, target=replace(protocol.target, threshold=0.01)).protocol_id != protocol.protocol_id
    assert "invalid_maturity" in codes(validate_research_protocol(replace(protocol, target=replace(protocol.target, horizon=4))))
    for changed in (replace(protocol.transformations, feature_order=protocol.transformations.feature_order[::-1]),
                    replace(protocol.transformations, correlation={"threshold": 0.8, "scope": "fitting"})):
        assert replace(protocol, transformations=changed).protocol_id != protocol.protocol_id
    assert protocol.learning.effective_parameters["n_estimators"] == 40
    assert protocol.learning.declared_parameters["n_estimators"] == 1000
    assert replace(protocol, learning=replace(protocol.learning, effective_parameters={"n_estimators": 41})).protocol_id != protocol.protocol_id
    assert replace(protocol, learning=replace(protocol.learning, early_stopping={"enabled": False})).protocol_id != protocol.protocol_id


def test_unknown_vintage_and_execution_status_are_not_certified(research):
    protocol, root, _, _ = research
    report = validate_research_artifacts(protocol, root)
    assert report.valid and not report.execution_eligible and not report.historical_point_in_time_verified
    assert protocol.provenance.point_in_time_status == "unknown"
    assert "historical_vintage" in protocol.provenance.missing_evidence
    with pytest.raises(TypeError):
        ProtocolValidationReport(execution_eligible=True)
    for changes in ({"point_in_time_status": "verified"}, {"execution_status": "eligible"}):
        assert "unsupported_provenance_claim" in codes(validate_research_protocol(replace(protocol, provenance=replace(protocol.provenance, **changes))))


def test_session_equivalence_must_be_explicit(research):
    protocol, _, _, _ = research
    observations = tuple(replace(row, session_ordinal=None) for row in protocol.universe.observations)
    assert "unknown_session_equivalence" in codes(validate_research_protocol(replace(protocol, universe=replace(protocol.universe, observations=observations))))
    assert "horizon_unit_mismatch" in codes(validate_research_protocol(replace(protocol, evaluation=replace(protocol.evaluation, holding_horizon=4))))


def test_historical_evidence_references_are_declarations_not_authentication(research):
    protocol, root, _, _ = research
    unknown = replace(protocol, provenance=replace(protocol.provenance, point_in_time_status="evidence_referenced"))
    assert "unsupported_provenance_claim" in codes(validate_research_protocol(unknown))
    declared = replace(unknown, provenance=replace(unknown.provenance, historical_evidence_references=("synthetic-vintage-reference",)))
    report = validate_research_artifacts(declared, root)
    assert report.valid and not report.historical_point_in_time_verified and not report.execution_eligible


def test_physical_clocks_normalize_but_original_labels_remain_opaque(research):
    protocol, _, _, _ = research
    rows = tuple(replace(row, observation_at=row.observation_at.astimezone(UTC)) for row in protocol.universe.observations)
    evaluation = replace(protocol.evaluation, execution_start=protocol.evaluation.execution_start.astimezone(UTC),
                         execution_end=protocol.evaluation.execution_end.astimezone(UTC))
    changed = replace(protocol, universe=replace(protocol.universe, observations=rows), evaluation=evaluation)
    assert changed.protocol_id == protocol.protocol_id
    a = datetime(2024, 11, 3, 1, 30, tzinfo=NY, fold=0)
    c = a.replace(fold=1)
    first = protocol.universe.observations[0]
    def labelled(label):
        return replace(protocol, universe=replace(protocol.universe, observations=(replace(first, original_label=label),) + protocol.universe.observations[1:]))
    assert labelled(a).protocol_id != labelled(c).protocol_id
    assert labelled(1).protocol_id != labelled("1").protocol_id
    assert labelled(np.int64(1)).protocol_id != labelled(1).protocol_id
    assert ResearchProtocolV1.from_json(labelled(a).canonical_json()).canonical_json() == labelled(a).canonical_json()


def test_missing_values_and_numeric_types_round_trip_without_coercion(research):
    protocol, _, _, _ = research
    params = dict(protocol.learning.effective_parameters, null=None, missing=pd.NA, nan=float("nan"), negative_zero=-0.0)
    changed = replace(protocol, learning=replace(protocol.learning, effective_parameters=params))
    assert ResearchProtocolV1.from_json(changed.canonical_json()).canonical_json() == changed.canonical_json()
    assert replace(protocol, learning=replace(protocol.learning, effective_parameters={"number": 1})).protocol_id != replace(
        protocol, learning=replace(protocol.learning, effective_parameters={"number": 1.0})).protocol_id
    weights = {0: 1.0, 1: 2.0, 2: 3.0, "0": 4.0}
    weighted = replace(protocol, learning=replace(protocol.learning, effective_parameters={"class_weights": weights}))
    decoded = ResearchProtocolV1.from_json(weighted.canonical_json())
    assert dict(decoded.learning.effective_parameters["class_weights"]) == weights
    assert len(decoded.learning.effective_parameters["class_weights"]) == 4


def test_held_out_labels_do_not_enter_training_input_identity(research):
    protocol, root, path, frame = research
    frame.loc[25:, "heldout_label"] = (frame.loc[25:, "heldout_label"] + 1) % 3
    frame.to_parquet(path, index=False)
    changed = replace(protocol, artifacts=(artifact(path, frame.source_id),),
                      evaluation=replace(protocol.evaluation, held_out_labels_sha256="a" * 64))
    assert changed.protocol_id != protocol.protocol_id
    assert changed.training_input_id == protocol.training_input_id
    assert validate_research_artifacts(changed, root).valid
    frame.loc[0, "close"] += 1
    frame.to_parquet(path, index=False)
    changed = replace(changed, artifacts=(artifact(path, frame.source_id),))
    assert "training_values_mismatch" in codes(validate_research_artifacts(changed, root))
    changed = replace(changed, training_inputs=replace(changed.training_inputs, values_sha256=training_hash(frame)))
    assert changed.training_input_id != protocol.training_input_id
    assert validate_research_artifacts(changed, root).valid


def test_immutable_nested_inputs_and_stable_diagnostics(research):
    protocol, root, _, _ = research
    params = {"nested": {"list": [1, 2]}}
    changed = replace(protocol, learning=replace(protocol.learning, effective_parameters=params))
    identifier = changed.protocol_id
    params["nested"]["list"].append(3)
    assert changed.protocol_id == identifier
    with pytest.raises(TypeError):
        changed.learning.effective_parameters["new"] = 1
    with pytest.raises(FrozenInstanceError):
        changed.asset_id = "other"
    missing = replace(protocol, artifacts=(replace(protocol.artifacts[0], relative_path="missing.parquet"),))
    assert validate_research_artifacts(missing, root).diagnostics == validate_research_artifacts(missing, root).diagnostics


def test_secrets_unsupported_values_and_noncanonical_json_fail(research):
    protocol, _, _, _ = research
    for values in ({"trusted_keys": {"issuer": b"not-a-real-key"}}, {"unsupported": object()},
                   {"heldout_labels": [0, 1, 2]}, {"unsupported_precision": np.longdouble("1.1")}):
        with pytest.raises(ResearchProtocolError):
            replace(protocol.learning, effective_parameters=values)
    with pytest.raises(ResearchProtocolError):
        ResearchProtocolV1.from_json(json.dumps(json.loads(protocol.canonical_json()), indent=2))
    with pytest.raises(ResearchProtocolError):
        ResearchProtocolV1.from_json('{"type":"ResearchProtocolV1","type":"ResearchProtocolV1","fields":{}}')


def test_original_labels_and_types_cannot_be_falsified(research):
    protocol, root, _, _ = research
    first = protocol.universe.observations[0]
    changed = replace(protocol, universe=replace(protocol.universe, observations=(replace(first, original_label="invented"),) + protocol.universe.observations[1:]))
    assert "source_label_mismatch" in codes(validate_research_artifacts(changed, root))
    changed = replace(protocol, universe=replace(protocol.universe, observations=(replace(first, index_dtype="string"),) + protocol.universe.observations[1:]))
    assert "source_label_mismatch" in codes(validate_research_artifacts(changed, root))


def test_scalar_and_early_stopping_specification_errors_fail(research):
    protocol, _, _, _ = research
    changed = replace(protocol, learning=replace(protocol.learning, early_stopping={"enabled": "yes"}))
    assert "invalid_learning" in codes(validate_research_protocol(changed))
    naive = protocol.universe.observations[0].observation_at.replace(tzinfo=None)
    with pytest.raises(ResearchProtocolError) as error:
        replace(protocol, universe=replace(protocol.universe, observations=(replace(protocol.universe.observations[0], observation_at=naive),) + protocol.universe.observations[1:]))
    assert "invalid_timestamp" in {item.code for item in error.value.diagnostics}
    changed = replace(protocol, evaluation=replace(protocol.evaluation, execution_parameters=dict(protocol.evaluation.execution_parameters, fee_rate=-0.01)))
    assert "invalid_evaluation" in codes(validate_research_protocol(changed))
    changed = replace(protocol, environment=replace(protocol.environment, dirty_tree=True))
    assert "invalid_environment" in codes(validate_research_protocol(changed))


def test_source_clock_and_calendar_recipe_are_part_of_training_identity(research):
    protocol, _, _, _ = research
    old = protocol.universe.observations[0]
    changed = replace(protocol, universe=replace(protocol.universe, observations=(replace(old, observation_at=old.observation_at - timedelta(minutes=1)),) + protocol.universe.observations[1:]))
    assert changed.training_input_id != protocol.training_input_id
    assert replace(protocol, transformations=replace(protocol.transformations, calendar_timezone="UTC")).training_input_id != protocol.training_input_id


def test_artifact_hashing_uses_bounded_reads_and_verified_read_image(research, monkeypatch):
    protocol, root, path, _ = research
    import src.research.protocol as module
    original = module._open_contained
    reads = []
    class Reader:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.wrapped.close()
        def read(self, size):
            assert size == 1024 * 1024
            reads.append(size)
            return self.wrapped.read(size)
    monkeypatch.setattr(module, "_open_contained", lambda *args: Reader(original(*args)))
    original_decode = pq.ParquetFile
    def decode_verified_image(image, *args, **kwargs):
        # A later source change cannot replace the bytes already verified for this call.
        path.write_bytes(b"changed after verified copy")
        return original_decode(image, *args, **kwargs)
    monkeypatch.setattr(pq, "ParquetFile", decode_verified_image)
    assert validate_research_artifacts(protocol, root).valid
    assert reads


def test_no_network_database_mlflow_training_or_source_mutation(research, monkeypatch):
    protocol, root, path, _ = research
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Contract validation must not access external services or experiment engines")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    import builtins
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"xgboost", "sklearn", "mlflow", "sqlalchemy", "psycopg2", "yfinance"}:
            forbidden()
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert validate_research_artifacts(protocol, root).valid
    assert ResearchProtocolV1.from_json(protocol.canonical_json()).protocol_id == protocol.protocol_id
    assert path.read_bytes() == before


def test_training_row_digest_has_an_independent_canonical_reference():
    line = '[["str","row"],["tuple",[["float","0x1.0000000000000p+0"],["none"]]]]\n'
    assert hash_training_rows([("row", (1.0, None))]) == hashlib.sha256(line.encode()).hexdigest()


def test_absolute_configuration_paths_and_nonfitting_selection_are_rejected(research):
    protocol, _, _, _ = research
    with pytest.raises(ResearchProtocolError):
        replace(protocol.learning, effective_parameters={"local_path": "/private/machine-specific/model"})
    bad = replace(protocol, transformations=replace(protocol.transformations, feature_selection={"scope": "validation"}))
    assert "invalid_transformations" in codes(validate_research_protocol(bad))
    bad = replace(protocol, learning=replace(protocol.learning, early_stopping={"enabled": True, "rounds": 0, "information_scope": "validation"}))
    assert "invalid_learning" in codes(validate_research_protocol(bad))

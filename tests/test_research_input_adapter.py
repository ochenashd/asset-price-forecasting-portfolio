"""Local synthetic research inputs; no fitting, simulation or external evidence."""

from dataclasses import replace
from datetime import date, datetime, timedelta
import hashlib
import os
import platform
import shutil
import socket
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import pytest

from test_research_protocol import research, artifact
from src.research import protocol as contract
from src.research.input_adapter import (
    ResearchInputError, apf_research_input_recipe, prepare_research_inputs,
)


COLUMNS = ("open", "high", "low", "close", "volume")
FEATURES = ("close", "return_lag_1", "ema_7", "vol_30", "volume_change", "dayofweek")
NY = ZoneInfo("America/New_York")


def raw_hash(frame):
    """Hash physical Arrow values, preserving nulls rather than pandas sentinels."""
    return contract.hash_training_rows((row["source_id"], tuple(row[name] for name in COLUMNS))
        for row in pyarrow.Table.from_pandas(frame, preserve_index=False).to_pylist()[:120])


def rewrite(protocol, path, frame, *, update_training=False):
    frame.to_parquet(path, index=False)
    reference = artifact(path, tuple(frame.source_id))
    inputs = replace(protocol.training_inputs, values_sha256=raw_hash(frame)) if update_training else protocol.training_inputs
    return replace(protocol, artifacts=(reference,), training_inputs=inputs)


@pytest.fixture
def inputs(research):
    base, root, path, _ = research
    n = 180
    t = np.arange(n)
    close = 100 * np.exp(.01 * np.sin(.41 * t) + .003 * np.sin(1.13 * t) + .0002 * t)
    opening = close * (1 + .001 * np.sin(.23 * t))
    dates = [date(2024, 1, 1) + timedelta(days=int(i)) for i in t]
    clocks = [datetime.combine(day, datetime.min.time()).replace(hour=16, tzinfo=NY) for day in dates]
    frame = pd.DataFrame({"source_id": [f"r{i}" for i in t], "date_label": dates,
        "open": opening, "high": np.maximum(opening, close) * 1.003,
        "low": np.minimum(opening, close) * .997, "close": close, "volume": 1000 + t % 17})
    ids = tuple(frame.source_id)
    e = lambda start, stop: tuple(f"e{i}" for i in range(start, stop))
    observations = tuple(contract.SourceObservation(key, "synthetic-prices", dates[i], "date32[day]",
        clocks[i], f"session-{i}", i) for i, key in enumerate(ids))
    universe = contract.ObservationUniverse(observations,
        tuple(contract.EngineeredRow(f"e{i}", f"r{i}") for i in range(31, 175)),
        tuple((key, "warm_up" if i < 31 else "label_tail") for i, key in enumerate(ids) if i < 31 or i >= 175))
    p = replace(base, universe=universe,
        computation=contract.ComputationWindow(ids[:175], ids[174], ids[179], ids[120], ids[179]),
        transformations=apf_research_input_recipe(FEATURES),
        target=contract.TargetSpec("5d_class", 5, "observations", tuple((f"e{i}", f"r{i+5}") for i in range(31, 175))),
        partitions=contract.FrozenPartitionSpec(e(31, 91), e(91, 96), e(96, 115), e(115, 120), e(120, 175), "e31", "e96", "e120"),
        evaluation=replace(base.evaluation, prediction_start="e120", prediction_end="e174",
            execution_start=clocks[120] - timedelta(hours=7), execution_end=clocks[179], terminal_session="session-179"),
        environment=replace(base.environment, python_version=platform.python_version(),
            package_versions={"pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__}),
        training_inputs=contract.TrainingInputSpec(ids[:120], raw_hash(frame)))
    p = rewrite(p, path, frame)
    contract.validate_research_protocol(p).require_valid()
    return p, root, path, frame


def rejected(protocol, root, code=None):
    with pytest.raises(ResearchInputError) as caught:
        prepare_research_inputs(protocol, root)
    codes = {item.code for item in caught.value.diagnostics}
    if code:
        assert code in codes
    return codes


def test_verified_frozen_inputs_and_independent_numerical_references(inputs):
    p, root, _, frame = inputs
    result = prepare_research_inputs(p, root)
    assert (len(result.fitting.row_ids), len(result.validation.row_ids), len(result.held_out.row_ids)) == (60, 19, 55)
    assert len(result.inner_purged.row_ids) == len(result.outer_purged.row_ids) == 5
    assert result.fitting.row_ids == p.partitions.fitting
    assert result.validation.row_ids == p.partitions.validation
    assert result.held_out.row_ids == p.partitions.held_out
    assert result.source_artifacts == p.artifacts
    assert result.asset_id == p.asset_id
    assert tuple(result.fitting.features.columns) == FEATURES
    assert result.row_bindings[0].row_id == "e31" and result.row_bindings[0].source_id == "r31"
    assert result.row_bindings[0].target_end_source_id == "r36"
    expected_ema = frame.close.iloc[0]
    for price in frame.close.iloc[1:32]:
        expected_ema = .25 * price + .75 * expected_ema
    assert result.fitting.features.ema_7.iloc[0] == pytest.approx(expected_ema)
    expected_returns = frame.close.iloc[36:96].to_numpy() / frame.close.iloc[31:91].to_numpy() - 1
    np.testing.assert_allclose(result.fitting.target_returns, expected_returns)
    np.testing.assert_array_equal(result.fitting.labels, np.where(expected_returns > .005, 2, np.where(expected_returns < -.005, 0, 1)))
    assert result.original_observations.columns.tolist() == frame.columns.tolist()
    assert str(result.original_observations.volume.dtype) == "int64[pyarrow]"
    assert result.selection_status == result.scaling_status == "deferred_fitting_only"
    assert not result.execution_eligible and not result.historical_point_in_time_verified
    assert result.provenance_limitations.point_in_time_status == "unknown"


@pytest.mark.parametrize("change,code", [("mutate", "sha256_mismatch"), ("append", "byte_length_mismatch"), ("missing", "missing_artifact")])
def test_artifact_integrity_fails_before_decoding(inputs, monkeypatch, change, code):
    p, root, path, _ = inputs
    if change == "missing":
        path.unlink()
    elif change == "append":
        with path.open("ab") as handle:
            handle.write(b"future-data")
    else:
        data = bytearray(path.read_bytes())
        data[12] ^= 1
        path.write_bytes(data)
    monkeypatch.setattr(pq, "ParquetFile", lambda *a, **k: pytest.fail("Unverified decode"))
    rejected(p, root, code)


def test_replacement_after_verification_cannot_change_consumed_image(inputs, monkeypatch):
    p, root, path, frame = inputs
    original = contract.VerifiedResearchArtifact.to_pandas
    opens = []
    opener = contract._open_contained
    def tracked(*args):
        opens.append(args)
        return opener(*args)
    def replaced(self):
        changed = frame.copy()
        changed["close"] *= 2
        changed.to_parquet(path, index=False)
        return original(self)
    monkeypatch.setattr(contract, "_open_contained", tracked)
    monkeypatch.setattr(contract.VerifiedResearchArtifact, "to_pandas", replaced)
    result = prepare_research_inputs(p, root)
    assert len(opens) == 1
    assert result.original_observations.close.iloc[31] == frame.close.iloc[31]
    assert hashlib.sha256(path.read_bytes()).hexdigest() != p.artifacts[0].sha256


def test_mtime_and_relocation_are_not_selection_criteria(inputs, tmp_path):
    p, root, path, _ = inputs
    before = prepare_research_inputs(p, root)
    os.utime(path, (1, 1))
    relocated = tmp_path / "copy"
    shutil.copytree(root / "inputs", relocated / "inputs")
    after = prepare_research_inputs(p, relocated)
    assert before.protocol_id == after.protocol_id
    assert before.training_input_id == after.training_input_id


@pytest.mark.parametrize("locator", ["../prices.parquet", "/tmp/prices.parquet", "inputs/../prices.parquet"])
def test_unsafe_locators_fail_closed(inputs, locator):
    p, root, _, _ = inputs
    rejected(replace(p, artifacts=(replace(p.artifacts[0], relative_path=locator),)), root, "invalid_artifact")


def test_symlink_is_not_followed(inputs, tmp_path):
    p, _, path, _ = inputs
    other = tmp_path / "linked"
    (other / "inputs").mkdir(parents=True)
    (other / "inputs" / path.name).symlink_to(path)
    rejected(p, other, "unsafe_or_unreadable_artifact")


@pytest.mark.parametrize("change", ["reorder", "missing", "duplicate", "dtype", "label"])
def test_original_observation_mismatches_are_rejected(inputs, change):
    p, root, path, frame = inputs
    if change == "reorder":
        frame = frame.iloc[::-1]
    elif change == "missing":
        frame = frame.iloc[:-1]
    elif change == "duplicate":
        frame.loc[41, "source_id"] = frame.loc[40, "source_id"]
    elif change == "dtype":
        frame["volume"] = frame.volume.astype(float)
    else:
        frame.loc[41, "date_label"] += timedelta(days=1)
    frame.to_parquet(path, index=False)
    ref = replace(p.artifacts[0], sha256=hashlib.sha256(path.read_bytes()).hexdigest(), byte_length=path.stat().st_size)
    codes = rejected(replace(p, artifacts=(ref,)), root)
    assert codes & {"source_order_mismatch", "row_count_mismatch", "schema_mismatch", "source_label_mismatch"}


def test_structurally_valid_changed_values_need_new_training_digest(inputs):
    p, root, path, frame = inputs
    frame.loc[50, "close"] *= 1.001
    changed = rewrite(p, path, frame)
    rejected(changed, root, "training_values_mismatch")
    changed = rewrite(p, path, frame, update_training=True)
    assert prepare_research_inputs(changed, root).source_training_input_id != p.training_input_id


@pytest.mark.parametrize("policy", ["keep_last", "keep_first"])
def test_deduplication_is_not_silently_applied(inputs, policy):
    p, root, _, _ = inputs
    rejected(replace(p, transformations=replace(p.transformations, duplicate_policy=policy)), root)


def test_missing_price_is_causally_filled_but_raw_is_preserved(inputs):
    p, root, path, frame = inputs
    frame.loc[70, "close"] = np.nan
    p = rewrite(p, path, frame, update_training=True)
    result = prepare_research_inputs(p, root)
    assert pd.isna(result.original_observations.close.iloc[70])
    assert result.cleaned_observations.close.iloc[70] == frame.close.iloc[69]
    assert result.cleaned_observations.close.iloc[70] != frame.close.iloc[71]


def test_missing_volume_excludes_only_declared_source_without_shifting_maturity(inputs):
    p, root, path, frame = inputs
    frame.loc[80, "volume"] = np.nan
    p = rewrite(p, path, frame, update_training=True)
    p = replace(p, universe=replace(p.universe,
        engineered_rows=tuple(row for row in p.universe.engineered_rows if row.row_id != "e80"),
        excluded_rows=p.universe.excluded_rows[:31] + (("r80", "missing_volume"),) + p.universe.excluded_rows[31:]),
        partitions=replace(p.partitions, fitting=tuple(key for key in p.partitions.fitting if key != "e80")),
        target=replace(p.target, maturity=tuple(item for item in p.target.maturity if item[0] != "e80")))
    result = prepare_research_inputs(p, root)
    assert pd.isna(result.cleaned_observations.volume.iloc[80])
    row = next(row for row in result.row_bindings if row.row_id == "e79")
    assert row.target_end_source_id == "r84"
    assert len(result.fitting.row_ids) == 59


def test_future_tail_changes_labels_but_never_features_or_training_identity(inputs):
    p, root, path, frame = inputs
    first = prepare_research_inputs(p, root)
    frame.loc[175:, COLUMNS[:4]] *= 1.02
    changed = rewrite(p, path, frame)
    second = prepare_research_inputs(changed, root)
    pd.testing.assert_frame_equal(first.held_out.features, second.held_out.features)
    assert not first.held_out.labels.equals(second.held_out.labels)
    assert first.training_input_id == second.training_input_id
    assert first.protocol_id != second.protocol_id


def test_validation_label_values_belong_to_training_context(inputs):
    p, root, path, frame = inputs
    first = prepare_research_inputs(p, root)
    frame.loc[119, list(COLUMNS[:4])] *= 1.02
    p = rewrite(p, path, frame, update_training=True)
    second = prepare_research_inputs(p, root)
    assert not first.validation.labels.equals(second.validation.labels)
    assert first.training_input_id != second.training_input_id


@pytest.mark.parametrize("subset,changes_identity", [("validation", True), ("held_out", False)])
def test_prepared_identity_binds_actual_labels_not_just_raw_digest(inputs, monkeypatch, subset, changes_identity):
    from src.research.input_adapter import FeatureEngineer
    p, root, _, _ = inputs
    before = prepare_research_inputs(p, root)
    derive = FeatureEngineer.add_target
    clocks = {p.universe.observations[int(key[1:])].observation_at for key in getattr(p.partitions, subset)}
    def changed_labels(self, data):
        result = derive(self, data)
        mask = result.index.isin(clocks)
        result.loc[mask, "target"] = (result.loc[mask, "target"] + 1) % 3
        return result
    monkeypatch.setattr(FeatureEngineer, "add_target", changed_labels)
    after = prepare_research_inputs(p, root)
    assert before.source_training_input_id == after.source_training_input_id
    assert (before.training_input_id != after.training_input_id) is changes_identity


def test_frozen_partitions_do_not_call_fraction_splitters(inputs, monkeypatch):
    from src.research.input_adapter import DataPreprocessor
    monkeypatch.setattr(DataPreprocessor, "split_data", lambda *a, **k: pytest.fail("Frozen memberships cannot be split again"))
    p, root, _, _ = inputs
    result = prepare_research_inputs(p, root)
    assert result.frozen_partitions == p.partitions


def test_preserve_missing_policy_does_not_silently_fill_to_match_universe(inputs):
    p, root, path, frame = inputs
    frame.loc[70, "close"] = np.nan
    p = rewrite(p, path, frame, update_training=True)
    p = replace(p, transformations=apf_research_input_recipe(FEATURES, price_policy="preserve"))
    rejected(p, root, "engineered_universe_mismatch")


def test_complete_recursive_prefix_affects_ema_not_only_warmup_window(inputs):
    p, root, path, frame = inputs
    before = prepare_research_inputs(p, root)
    frame.loc[0, "close"] *= 1.02
    p = rewrite(p, path, frame, update_training=True)
    after = prepare_research_inputs(p, root)
    assert after.fitting.features.ema_7.iloc[0] - before.fitting.features.ema_7.iloc[0] == pytest.approx(.75 ** 31 * .02 * frame.close.iloc[0] / 1.02, abs=1e-13)


def test_wrong_engineered_universe_is_not_adjusted_to_match_data(inputs):
    p, root, _, _ = inputs
    p = replace(p, universe=replace(p.universe,
        engineered_rows=p.universe.engineered_rows[1:],
        excluded_rows=p.universe.excluded_rows[:31] + (("r31", "declared_exclusion"),) + p.universe.excluded_rows[31:]),
        partitions=replace(p.partitions, fitting=p.partitions.fitting[1:], training_start="e32"),
        target=replace(p.target, maturity=p.target.maturity[1:]))
    rejected(p, root, "engineered_universe_mismatch")


def test_feature_order_cannot_be_silently_rearranged(inputs):
    p, root, _, _ = inputs
    rejected(replace(p, transformations=apf_research_input_recipe(FEATURES[::-1])), root, "feature_order_mismatch")


@pytest.mark.parametrize("change", ["maturity", "inner", "outer", "overlap", "prefix"])
def test_invalid_frozen_boundaries_rejected_without_fraction_recomputation(inputs, change):
    p, root, _, _ = inputs
    if change == "maturity":
        p = replace(p, target=replace(p.target, maturity=(("e31", "r37"),) + p.target.maturity[1:]))
    elif change == "inner":
        p = replace(p, partitions=replace(p.partitions, fitting=p.partitions.fitting + ("e91",), inner_purge=p.partitions.inner_purge[1:]))
    elif change == "outer":
        p = replace(p, partitions=replace(p.partitions, validation=p.partitions.validation + ("e115",), outer_purge=p.partitions.outer_purge[1:]))
    elif change == "overlap":
        p = replace(p, partitions=replace(p.partitions, held_out=("e31",) + p.partitions.held_out))
    else:
        p = replace(p, computation=replace(p.computation, historical_prefix_ids=p.computation.historical_prefix_ids[1:]))
    rejected(p, root)


def test_both_purge_boundaries_are_strict_and_original(inputs):
    p, root, _, _ = inputs
    result = prepare_research_inputs(p, root)
    validation_start = result.validation.row_bindings[0].observation_at
    test_start = result.held_out.row_bindings[0].observation_at
    assert all(row.target_end_timestamp < validation_start for row in result.fitting.row_bindings)
    assert result.inner_purged.row_bindings[0].target_end_timestamp == validation_start
    assert all(row.target_end_timestamp < test_start for row in result.validation.row_bindings)
    assert result.outer_purged.row_bindings[0].target_end_timestamp == test_start


def test_returned_pandas_views_are_owned_defensive_copies(inputs):
    p, root, path, _ = inputs
    before = p.canonical_json(), path.read_bytes()
    result = prepare_research_inputs(p, root)
    for accessor in (lambda: result.original_observations, lambda: result.cleaned_observations,
                     lambda: result.engineered, lambda: result.fitting.features):
        view = accessor()
        original = view.copy(deep=True)
        view.iloc[0, 0] = None
        pd.testing.assert_frame_equal(accessor(), original)
    labels = result.validation.labels
    labels.iloc[0] = (int(labels.iloc[0]) + 1) % 3
    assert not labels.equals(result.validation.labels)
    assert before == (p.canonical_json(), path.read_bytes())


def test_temporary_images_close_on_success_and_transformation_failure(inputs, monkeypatch):
    p, root, _, _ = inputs
    handles = []
    create = contract.tempfile.TemporaryFile
    def tracked(*args, **kwargs):
        image = create(*args, **kwargs)
        handles.append(image)
        return image
    monkeypatch.setattr(contract.tempfile, "TemporaryFile", tracked)
    prepare_research_inputs(p, root)
    rejected(replace(p, transformations=apf_research_input_recipe(FEATURES[::-1])), root)
    assert len(handles) == 2 and all(handle.closed for handle in handles)


def test_no_fitting_simulation_network_database_or_tracking_imports(inputs, monkeypatch):
    import builtins
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("src.models", "src.pipeline", "xgboost", "sklearn", "yfinance", "mlflow", "sqlalchemy", "psycopg"))
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network forbidden"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("Network forbidden"))
    from src.evaluation.execution.engine import ExecutionEngineV1
    monkeypatch.setattr(ExecutionEngineV1, "run", lambda *a, **k: pytest.fail("Execution forbidden"))
    p, root, _, _ = inputs
    prepare_research_inputs(p, root)


def test_regression_remains_continuous(inputs):
    p, root, _, _ = inputs
    p = replace(p, target=replace(p.target, task_type="5d_return", threshold=None, class_encoding=()))
    result = prepare_research_inputs(p, root)
    pd.testing.assert_series_equal(result.fitting.labels, result.fitting.target_returns, check_names=False)
    assert (result.fitting.labels > 0).any() and (result.fitting.labels < 0).any()


def test_unknown_recipe_is_a_structured_blocker(inputs):
    p, root, _, _ = inputs
    changed = replace(p.transformations, indicators={**p.transformations.indicators, "ewm_adjust": True})
    rejected(replace(p, transformations=changed), root, "unsupported_recipe")


def test_changed_runtime_declaration_is_a_structured_blocker(inputs):
    p, root, _, _ = inputs
    rejected(replace(p, environment=replace(p.environment, python_version="0.0.0")), root, "environment_mismatch")


@pytest.mark.parametrize("asset", ["research:SYNTH", "research:OTHER", None])
def test_explicit_source_asset_column_cannot_contradict_protocol(inputs, asset):
    p, root, path, frame = inputs
    frame["asset_id"] = asset
    p = rewrite(p, path, frame)
    if asset == p.asset_id:
        assert prepare_research_inputs(p, root).asset_id == asset
    else:
        rejected(p, root, "asset_identity_mismatch")

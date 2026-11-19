"""Temporary synthetic artifacts; no provider, database, training or certification."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.market_snapshot import (
    AcquisitionMetadata, AssetMetadata, Declaration, SnapshotDeclarations, read_market_snapshot,
)


def claim(value, reference=None):
    return Declaration(value, "caller_declared", reference)


def bars():
    return pd.DataFrame({
        "Open": [100.0, -37.63, 110.0], "High": [102.0, -35.0, 112.0],
        "Low": [99.0, -40.0, 109.0], "Close": [101.0, -36.0, 111.0],
        "Volume": [1000, 2000, 3000], "Repaired?": [False, True, False],
        "Dividends": [0.0, 0.25, 0.0], "Stock Splits": [0.0, 0.0, 2.0],
    }, index=pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"], name="Date"))


def write(tmp_path, frame=None, name="explicit.parquet"):
    path = tmp_path / name
    (bars() if frame is None else frame).to_parquet(path, engine="pyarrow")
    return path


def read(path, **changes):
    return read_market_snapshot(path, artifact_id="source-artifact-v1", **changes)


def codes(snapshot):
    return [diagnostic.code for diagnostic in snapshot.diagnostics]


def decoded_values(snapshot):
    return [[value.to_python() for value in row.values] for row in snapshot.rows]


def test_explicit_artifact_is_selected_even_when_other_files_are_newer(tmp_path):
    selected = write(tmp_path)
    other = bars()
    other["Close"] = [999.0] * 3
    newer = write(tmp_path, other, "latest.parquet")
    os.utime(selected, (100, 100))
    os.utime(newer, (200, 200))
    result = read(selected)
    assert result.rows[0].values[3].to_python() == 101.0
    assert result.source_path == str(selected.absolute())
    assert result.artifact.artifact_id == "source-artifact-v1"


def test_sha256_is_of_original_file_bytes_and_optional_expected_hash_is_checked(tmp_path):
    path = write(tmp_path)
    original = path.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    result = read(path, expected_sha256=expected.upper())
    assert result.artifact.sha256 == expected
    assert result.artifact.byte_count == len(original)
    assert result.artifact.evidence_status == "artifact_verified"
    assert result.artifact.artifact_id_evidence_status == "caller_declared"
    assert path.read_bytes() == original
    assert read(path).artifact.sha256 == expected


def test_tampered_file_is_rejected_before_parquet_decoding(tmp_path, monkeypatch):
    path = write(tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b"tampered")

    def forbidden(*args, **kwargs):
        raise AssertionError("A hash mismatch must fail before decoding.")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read(path, expected_sha256=expected)


@pytest.mark.parametrize("digest", ["", "f" * 63, "x" * 64, 123])
def test_invalid_expected_hash_is_rejected(tmp_path, digest):
    with pytest.raises(ValueError, match="64 hexadecimal"):
        read(write(tmp_path), expected_sha256=digest)


def test_exact_row_order_labels_dtypes_values_and_action_columns_survive(tmp_path):
    frame = bars()
    result = read(write(tmp_path, frame))
    assert result.artifact.row_count == 3
    assert result.artifact.file_format == "parquet"
    assert result.artifact.parquet_version
    assert result.artifact.arrow_schema
    assert [column.label.to_python() for column in result.artifact.columns] == list(frame.columns)
    assert [column.dtype for column in result.artifact.columns] == [str(dtype) for dtype in frame.dtypes]
    assert [row.session_label.to_python() for row in result.rows] == list(frame.index)
    assert [row.ordinal for row in result.rows] == [0, 1, 2]
    assert len({row.source_row_id for row in result.rows}) == 3
    assert all(row.source_row_id == f"{result.artifact.sha256}:{row.ordinal}" for row in result.rows)
    for row in result.rows:
        for column, value in enumerate(row.values):
            assert value.to_python() == frame.iat[row.ordinal, column]
    assert result.rows[1].values[0].to_python() == -37.63
    assert result.rows[1].values[6].to_python() == 0.25
    assert result.rows[2].values[7].to_python() == 2.0


def test_dates_stay_naive_midnight_labels_not_market_close_clocks(tmp_path):
    result = read(write(tmp_path))
    assert result.artifact.index_schema.dtype == "datetime64[ns]"
    assert result.artifact.index_schema.names[0].to_python() == "Date"
    for row in result.rows:
        label = row.session_label.to_python()
        assert label.tz is None and label.hour == 0
        assert not hasattr(row, "close_timestamp")
    assert not result.execution_ready


def test_date_column_and_integer_index_are_not_reinterpreted(tmp_path):
    frame = bars().reset_index()
    result = read(write(tmp_path, frame))
    assert [row.session_label.to_python() for row in result.rows] == [0, 1, 2]
    assert result.artifact.columns[0].label.to_python() == "Date"
    assert result.rows[0].values[0].to_python() == pd.Timestamp("2024-01-03")


def test_timezone_and_nanosecond_labels_are_preserved(tmp_path):
    frame = bars()
    frame.index = pd.DatetimeIndex([pd.Timestamp("2024-03-11 09:30", tz="America/New_York")
                                    + pd.Timedelta(i, unit="ns") for i in range(3)], name="SourceTime")
    result = read(write(tmp_path, frame))
    assert "America/New_York" in result.artifact.index_schema.dtype
    assert [row.session_label.to_python() for row in result.rows] == list(frame.index)
    assert str(result.rows[0].session_label.to_python().tz) == "America/New_York"


def test_fixed_offset_timezone_labels_remain_decodable(tmp_path):
    frame = bars()
    frame.index = frame.index.tz_localize(timezone(timedelta(hours=2)))
    result = read(write(tmp_path, frame))
    assert [row.session_label.to_python() for row in result.rows] == list(frame.index)
    assert result.rows[0].session_label.to_python().utcoffset() == timedelta(hours=2)


@pytest.mark.parametrize("conflicting", [False, True])
def test_identical_and_conflicting_duplicates_are_retained_with_diagnostics(tmp_path, conflicting):
    frame = bars().iloc[[0, 0, 2]].copy()
    if conflicting:
        frame.iloc[1, 3] = 250.0
    result = read(write(tmp_path, frame))
    assert len(result.rows) == 3
    assert result.rows[0].session_label == result.rows[1].session_label
    assert result.rows[0].source_row_id != result.rows[1].source_row_id
    assert "duplicate_session_labels" in codes(result)
    assert ("conflicting_duplicate_observations" in codes(result)) == conflicting
    diagnostic = next(item for item in result.diagnostics if item.code == "duplicate_session_labels")
    assert diagnostic.row_ordinals == (0, 1)


def test_missing_ohlcv_values_are_not_filled_and_missing_columns_are_reported(tmp_path):
    frame = bars().drop(columns="Volume")
    frame.iloc[0, 0] = np.nan
    frame.iloc[2, 3] = np.nan
    result = read(write(tmp_path, frame))
    assert np.isnan(result.rows[0].values[0].to_python())
    assert np.isnan(result.rows[2].values[3].to_python())
    assert "missing_ohlcv_fields" in codes(result)
    affected = {(item.row_ordinals, item.column_positions) for item in result.diagnostics if item.code == "missing_ohlcv"}
    assert affected == {((0,), (0,)), ((2,), (3,))}
    assert len(result.rows) == 3


def test_zero_volume_negative_prices_and_extremes_are_preserved(tmp_path):
    frame = bars()
    frame.loc[frame.index[0], "Volume"] = 0
    frame.loc[frame.index[2], "High"] = 1_000_000.0
    result = read(write(tmp_path, frame))
    assert result.rows[0].values[4].to_python() == 0
    assert result.rows[1].values[0].to_python() == -37.63
    assert result.rows[2].values[1].to_python() == 1_000_000.0
    assert "missing_ohlcv" not in codes(result)


def test_repair_flags_are_preserved_without_certifying_false_flags(tmp_path):
    frame = bars()
    frame["Repaired?"] = pd.array([False, True, pd.NA], dtype="boolean")
    result = read(write(tmp_path, frame))
    assert result.rows[0].values[5].to_python() == False
    assert result.rows[1].values[5].to_python() == True
    assert result.rows[2].values[5].to_python() is pd.NA
    repaired = next(item for item in result.diagnostics if item.code == "repaired_observations")
    unknown = next(item for item in result.diagnostics if item.code == "unknown_repair_status")
    assert repaired.row_ordinals == (1,)
    assert unknown.row_ordinals == (2,)
    assert result.provenance_independently_verified is False


def test_absent_repair_flags_remain_unknown(tmp_path):
    result = read(write(tmp_path, bars().drop(columns="Repaired?")))
    assert "unknown_repair_status" in codes(result)


def test_multiindex_columns_preserve_ticker_levels_and_multiple_assets(tmp_path):
    frame = pd.concat({"AAA": bars(), "BBB": bars() * 1}, axis=1)
    frame.columns.names = ["Ticker", "Price"]
    path = write(tmp_path, frame)
    result = read(path)
    assert len(result.artifact.columns) == 16
    assert [column.label.to_python() for column in result.artifact.columns] == list(frame.columns)
    assert [name.to_python() for name in result.artifact.column_axis_schema.names] == ["Ticker", "Price"]
    assert len(result.artifact.column_axis_schema.levels) == 2
    # Parquet reconstructs internal level dictionaries/codes. The artifact,
    # rather than the pre-serialization Python object, is the source of truth.
    decoded_source = pd.read_parquet(path)
    assert result.artifact.column_axis_schema.codes == tuple(
        tuple(int(i) for i in code) for code in decoded_source.columns.codes)
    assert [[value.to_python() for value in level] for level in result.artifact.column_axis_schema.levels] == [
        list(level) for level in decoded_source.columns.levels]
    assert "ambiguous_ticker_identities" in codes(result)
    assert len(result.rows[0].values) == 16


def test_single_ticker_multiindex_is_preserved_and_declared_mismatch_is_diagnostic(tmp_path):
    frame = bars()
    frame.columns = pd.MultiIndex.from_tuples([(name, "AAA") for name in frame.columns], names=["Price", "Ticker"])
    path = write(tmp_path, frame)
    original = read(path)
    assert "ambiguous_ticker_identities" not in codes(original)
    declared = SnapshotDeclarations(asset=AssetMetadata(symbol=claim("BBB")))
    changed = read(path, declarations=declared)
    assert "declared_symbol_mismatch" in codes(changed)
    assert original.rows == changed.rows
    assert [column.label.to_python() for column in original.artifact.columns] == list(frame.columns)


def test_unnamed_multiindex_is_not_silently_flattened_or_guessed(tmp_path):
    frame = bars()
    frame.columns = pd.MultiIndex.from_tuples([(name, "AAA") for name in frame.columns])
    result = read(write(tmp_path, frame))
    assert "ambiguous_ticker_identities" in codes(result)
    assert result.artifact.columns[0].label.to_python() == ("Open", "AAA")
    assert len(result.rows[0].values) == 8


def test_missing_acquisition_fields_are_unknown_not_inferred_from_filename(tmp_path):
    path = write(tmp_path, name="AAA_1d_20240101_1200.parquet")
    result = read(path)
    acquisition = result.declarations.acquisition
    for name in ("provider", "acquired_at", "auto_adjust", "repair", "interval"):
        value = getattr(acquisition, name)
        assert value.evidence_status == "unknown" and value.value.to_python() is None
    assert codes(result).count("missing_acquisition_metadata") == 10
    assert codes(result).count("unknown_adjustment_basis") == 2
    assert "missing_corporate_action_evidence" in codes(result)


def test_price_and_volume_adjustment_claims_are_independent_and_never_verified(tmp_path):
    declarations = SnapshotDeclarations(
        price_adjustment_basis=claim("auto_adjusted", "caller-sidecar"),
        volume_adjustment_basis=claim("provider_reported"),
        acquisition=AcquisitionMetadata(provider=claim("synthetic-provider"), auto_adjust=claim(True),
                                        repair=claim(False), acquired_at=claim(datetime(2024, 2, 1, tzinfo=timezone.utc))))
    result = read(write(tmp_path), declarations=declarations)
    assert result.declarations.price_adjustment_basis.value.to_python() == "auto_adjusted"
    assert result.declarations.volume_adjustment_basis.value.to_python() == "provider_reported"
    assert result.declarations.acquisition.auto_adjust.value.to_python() is True
    assert result.declarations.acquisition.repair.value.to_python() is False
    assert result.declarations.acquisition.provider.evidence_status == "caller_declared"
    assert "unknown_adjustment_basis" not in codes(result)
    assert result.execution_ready is False


def test_corporate_action_columns_and_caller_claims_do_not_certify_eligibility(tmp_path):
    declarations = SnapshotDeclarations(corporate_action_evidence=claim("supplied-action-file"),
                                        corporate_action_eligibility=claim("eligible"))
    result = read(write(tmp_path), declarations=declarations)
    assert result.rows[1].values[6].to_python() == 0.25
    assert result.rows[2].values[7].to_python() == 2
    assert result.declarations.corporate_action_eligibility.evidence_status == "caller_declared"
    assert result.execution_ready is False
    assert result.provenance_independently_verified is False


@pytest.mark.parametrize("factory", [
    lambda: Declaration("provider", "artifact_verified"), lambda: Declaration("provider"),
    lambda: Declaration(None, "caller_declared"),
    lambda: AcquisitionMetadata(acquired_at=claim(datetime(2024, 1, 1))),
    lambda: AcquisitionMetadata(acquired_at=claim("2024-01-01")),
    lambda: AcquisitionMetadata(repair=claim("False")), lambda: AssetMetadata(symbol="AAA"),
    lambda: SnapshotDeclarations(price_adjustment_basis="unadjusted"),
])
def test_invalid_or_self_verified_metadata_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_snapshot_is_deeply_immutable_and_decoding_cannot_mutate_it(tmp_path):
    metadata = {"note": ["original"]}
    declarations = SnapshotDeclarations(acquisition=AcquisitionMetadata(request_parameters=claim(metadata)))
    result = read(write(tmp_path), declarations=declarations)
    identifier = result.snapshot_id
    metadata["note"].append("changed")
    decoded = result.declarations.acquisition.request_parameters.value.to_python()
    decoded["note"].append("changed-again")
    assert result.snapshot_id == identifier
    assert result.declarations.acquisition.request_parameters.value.to_python() == {"note": ["original"]}
    with pytest.raises(FrozenInstanceError):
        result.execution_ready = True
    with pytest.raises(FrozenInstanceError):
        result.rows[0].ordinal = 9
    with pytest.raises(FrozenInstanceError):
        result.rows[0].values[0].text = "changed"
    with pytest.raises(TypeError):
        result.rows[0].values[0] = result.rows[0].values[1]
    with pytest.raises(FrozenInstanceError):
        result.declarations.acquisition.repair.evidence_status = "artifact_verified"


def test_identical_reads_have_identical_canonical_identity(tmp_path):
    path = write(tmp_path)
    first, second = read(path), read(path)
    assert first.canonical_json() == second.canonical_json()
    assert first.snapshot_id == second.snapshot_id
    assert json.loads(first.canonical_json())["schema_version"] == 1
    assert len(first.snapshot_id.removeprefix("apf-market-v1-")) == 64
    assert str(tmp_path) not in first.canonical_json()


def test_identical_bytes_survive_rename_copy_and_mtime_changes(tmp_path):
    path = write(tmp_path)
    first = read(path)
    copied = tmp_path / "different-name.parquet"
    shutil.copyfile(path, copied)
    os.utime(copied, (100, 100))
    assert read(copied).snapshot_id == first.snapshot_id
    renamed = path.rename(tmp_path / "renamed.parquet")
    os.utime(renamed, (200, 200))
    assert read(renamed).snapshot_id == first.snapshot_id


def test_changed_source_bytes_change_identity_even_with_identical_observations(tmp_path):
    frame = bars()
    first_path = write(tmp_path, frame)
    second_path = tmp_path / "other-encoding.parquet"
    frame.to_parquet(second_path, compression=None)
    first, second = read(first_path), read(second_path)
    assert decoded_values(first) == decoded_values(second)
    assert first.artifact.sha256 != second.artifact.sha256
    assert first.snapshot_id != second.snapshot_id


def test_changed_observations_change_identity(tmp_path):
    frame = bars()
    first = read(write(tmp_path, frame))
    frame.iloc[0, 0] = 123.0
    second = read(write(tmp_path, frame, "changed.parquet"))
    assert first.rows[0].values[0] != second.rows[0].values[0]
    assert first.snapshot_id != second.snapshot_id


@pytest.mark.parametrize("changes", [
    {"acquisition": AcquisitionMetadata(provider=claim("provider"))},
    {"acquisition": AcquisitionMetadata(acquired_at=claim(datetime(2024, 1, 1, tzinfo=timezone.utc)))},
    {"price_adjustment_basis": claim("unadjusted")}, {"volume_adjustment_basis": claim("provider_reported")},
    {"asset": AssetMetadata(asset_id=claim("asset-v1"), instrument_type=claim("future"))},
    {"corporate_action_evidence": claim("evidence", "reference-v1")},
    {"corporate_action_eligibility": claim("unknown")},
])
def test_declarations_and_acquisition_metadata_change_snapshot_id(tmp_path, changes):
    path = write(tmp_path)
    assert read(path).snapshot_id != read(path, declarations=SnapshotDeclarations(**changes)).snapshot_id


def test_evidence_reference_is_part_of_identity_but_never_opened(tmp_path):
    path = write(tmp_path)
    first = read(path, declarations=SnapshotDeclarations(price_adjustment_basis=claim("unadjusted", "ref-1")))
    second = read(path, declarations=SnapshotDeclarations(price_adjustment_basis=claim("unadjusted", "ref-2")))
    assert first.snapshot_id != second.snapshot_id


def test_no_database_network_or_loader_access_and_no_loader_source_change(tmp_path, monkeypatch):
    path = write(tmp_path)
    loader_path = Path(__file__).resolve().parents[1] / "src/data/loader.py"
    loader_before = loader_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("The snapshot reader cannot access network, databases or the legacy loader.")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setitem(sys.modules, "src.data.loader", SimpleNamespace(DataLoader=forbidden))
    monkeypatch.setitem(sys.modules, "src.database.connection", SimpleNamespace(create_engine_from_config=forbidden))
    result = read(path)
    assert result.artifact.row_count == 3
    assert loader_path.read_bytes() == loader_before


def test_file_is_read_once_and_decoding_uses_hashed_buffer(tmp_path, monkeypatch):
    path = write(tmp_path)
    original = path.read_bytes()
    original_reader = Path.read_bytes
    calls = []

    def reader(selected):
        calls.append(selected)
        result = original_reader(selected)
        if len(calls) > 1:
            raise AssertionError("Do not reopen the source for decoding.")
        return result

    monkeypatch.setattr(Path, "read_bytes", reader)
    result = read(path)
    assert len(calls) == 1
    assert result.artifact.sha256 == hashlib.sha256(original).hexdigest()


def test_actual_hash_matches_even_if_artifact_changes_after_the_single_read(tmp_path, monkeypatch):
    path = write(tmp_path)
    original = path.read_bytes()
    original_reader = Path.read_bytes

    def reader(selected):
        payload = original_reader(selected)
        # Simulate an external writer after the reader has captured its buffer.
        selected.write_bytes(b"changed externally after capture")
        return payload

    monkeypatch.setattr(Path, "read_bytes", reader)
    result = read(path)
    assert result.artifact.sha256 == hashlib.sha256(original).hexdigest()
    assert result.rows[0].values[3].to_python() == 101.0


def test_empty_artifact_preserves_schema_and_does_not_invent_rows(tmp_path):
    result = read(write(tmp_path, bars().iloc[:0]))
    assert result.rows == () and result.artifact.row_count == 0
    assert len(result.artifact.columns) == 8
    assert not result.execution_ready


def test_additional_source_types_and_nullable_values_survive(tmp_path):
    frame = bars()
    frame["Note"] = ["a", None, "c"]
    frame["Nullable"] = pd.array([1, pd.NA, 3], dtype="Int64")
    frame["Decimal"] = [Decimal("1.230"), None, Decimal("-2.500")]
    frame["Category"] = pd.Categorical(["x", "y", "x"], categories=["x", "y", "unused"], ordered=True)
    result = read(write(tmp_path, frame))
    assert result.rows[0].values[8].to_python() == "a"
    assert result.rows[1].values[8].to_python() is None
    assert result.rows[1].values[9].to_python() is pd.NA
    assert result.rows[0].values[10].to_python() == Decimal("1.230")
    assert [value.to_python() for value in result.artifact.columns[11].categories] == ["x", "y", "unused"]
    assert result.artifact.columns[11].categories_ordered is True


def test_date_objects_and_missing_session_labels_are_preserved(tmp_path):
    frame = bars()
    frame.index = pd.Index([date(2024, 1, 3), None, date(2024, 1, 3)], name="SessionDate")
    result = read(write(tmp_path, frame))
    assert result.rows[0].session_label.to_python() == date(2024, 1, 3)
    assert result.rows[1].session_label.to_python() is None
    assert "missing_session_labels" in codes(result)
    assert "duplicate_session_labels" in codes(result)


def test_non_parquet_and_missing_artifacts_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "missing.parquet")
    invalid = tmp_path / "invalid.parquet"
    invalid.write_bytes(b"not a parquet file")
    with pytest.raises(Exception, match="Parquet|parquet"):
        read(invalid)

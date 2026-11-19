"""Synthetic trust boundary tests, not certification of real provider data.

The fixture assumes its synthetic issuer is trusted to attest asset identity,
acquisition history, the entire exchange schedule, observation lineage and action
coverage. The separately configured HMAC key authenticates that fixture statement
only. No real provider key, signature, calendar or external evidence is used.
Naive source dates are mapped explicitly by this fixture issuer, never used as
exchange-close timestamps by the production validator.
"""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
import hashlib
import hmac
import shutil
import socket
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.data.market_snapshot import (
    AcquisitionMetadata, AssetMetadata, Declaration, SnapshotDeclarations, read_market_snapshot,
)
from src.data.market_eligibility import (
    CorporateAction, CorporateActionCoverage, EvidenceAttestation, FieldObservation,
    MarketEligibilityEvidence, RowObservation, attestation_message, validate_market_eligibility,
)
from src.evaluation.execution.adapter import (
    AssetSpec, ObservedMarketBar, SessionCalendar, TradingSession, adapt_execution_inputs,
)
from src.evaluation.execution.contracts import MarketBar


NY = ZoneInfo("America/New_York")
KEY = b"synthetic-fixture-only-no-production-trust-key"
TRUST = {"synthetic-issuer": KEY}
ASSET = AssetSpec("synthetic-equity-1", "SYNTH", "equity", "XNYS-calendar-v1", "USD")
NAMES = ("open", "high", "low", "close", "volume")


def claim(value):
    return Declaration(value, "caller_declared")


def schedule(holiday=False):
    definitions = (("2020-11-25", 16, "2020-11-27"), ("2020-11-27", 13, "2020-11-30"),
                   ("2020-11-30", 16, "2020-12-01")) if holiday else (
                       ("2024-03-08", 16, "2024-03-11"), ("2024-03-11", 16, "2024-03-12"),
                       ("2024-03-12", 16, "2024-03-13"))
    sessions = []
    for day, hour, following in definitions:
        opened = datetime.fromisoformat(day).replace(hour=9, minute=30, tzinfo=NY)
        closed = opened.replace(hour=hour, minute=0)
        next_open = datetime.fromisoformat(following).replace(hour=9, minute=30, tzinfo=NY)
        sessions.append(TradingSession(day, opened, closed, next_open))
    return SessionCalendar(ASSET.calendar_id, "synthetic-calendar-source", tuple(sessions), True)


def acquisition(calendar):
    return AcquisitionMetadata(
        provider=claim("synthetic-provider"), acquired_at=claim(calendar.sessions[-1].close_timestamp + timedelta(days=1)),
        requested_start=claim(calendar.sessions[0].session_id),
        requested_end=claim(calendar.sessions[-1].next_open_timestamp.date().isoformat()),
        interval=claim("1d"), library_version=claim("fixture-v1"), auto_adjust=claim(False),
        repair=claim(False), actions_requested=claim(True),
        request_parameters=claim({"keepna": True, "back_adjust": False}))


def declarations(calendar):
    return SnapshotDeclarations(
        asset=AssetMetadata(asset_id=claim(ASSET.asset_id), symbol=claim(ASSET.symbol),
                            instrument_type=claim("equity"), exchange=claim("XNYS"),
                            timezone=claim("America/New_York"), currency=claim("USD")),
        acquisition=acquisition(calendar), price_adjustment_basis=claim("unadjusted"),
        volume_adjustment_basis=claim("unadjusted_share_count"),
        corporate_action_evidence=claim("synthetic-action-reference"), corporate_action_eligibility=claim("eligible"))


def frame(calendar):
    return pd.DataFrame({"Open": [100.0, 101.0, 102.0], "High": [105.0, 106.0, 107.0],
                         "Low": [98.0, 99.0, 100.0], "Close": [101.0, 102.0, 103.0],
                         "Volume": [1000, 2000, 3000], "Repaired?": [False, False, False],
                         "Dividends": [0.0, 0.0, 0.0], "Stock Splits": [0.0, 0.0, 0.0]},
                        index=pd.DatetimeIndex([item.session_id for item in calendar.sessions], name="Date"))


def fixture(tmp_path, *, data=None, calendar=None, source_declarations=None):
    calendar = schedule() if calendar is None else calendar
    data = frame(calendar) if data is None else data
    path = tmp_path / "source.parquet"
    data.to_parquet(path)
    snapshot = read_market_snapshot(path, artifact_id="synthetic-artifact-v1",
                                    declarations=declarations(calendar) if source_declarations is None else source_declarations)
    rows = tuple(RowObservation(row.source_row_id, row.session_label.to_python().date().isoformat(),
                               tuple(FieldObservation(name, i, True, False, False) for i, name in enumerate(NAMES)))
                 for row in snapshot.rows)
    coverage = CorporateActionCoverage(ASSET.asset_id, "synthetic-actions-source", "synthetic-action-reference",
                                       calendar.sessions[0].open_timestamp, calendar.sessions[-1].close_timestamp, True)
    evidence = MarketEligibilityEvidence(
        snapshot.artifact.artifact_id, snapshot.artifact.sha256, snapshot.snapshot_id, "synthetic-provider",
        ASSET, "XNYS", "America/New_York", acquisition(calendar), calendar, "XNYS",
        calendar.sessions[0].session_id, calendar.sessions[-1].session_id, rows,
        "unadjusted", "unadjusted_share_count", coverage)
    return snapshot, evidence


def sign(evidence, *, issuer="synthetic-issuer", reference="fixture-statement-v1", key=KEY):
    payload = attestation_message(evidence, issuer_id=issuer, reference_id=reference)
    return EvidenceAttestation(issuer, reference, evidence, hmac.new(key, payload, hashlib.sha256).hexdigest())


def validate(snapshot, evidence, **changes):
    return validate_market_eligibility(snapshot, sign(evidence), **({"trusted_keys": TRUST} | changes))


def codes(report):
    return {reason.code for reason in report.reasons}


def assert_blocked(report, status, code):
    assert report.status == status
    assert code in codes(report)
    assert report.market == report.bars == ()


def test_fully_supported_fixture_is_eligible_under_explicit_synthetic_trust(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    report = validate(snapshot, evidence)
    assert report.status == "eligible" and report.reasons == ()
    assert report.authenticated_external_evidence
    assert len(report.market) == len(report.bars) == 3
    assert report.snapshot is snapshot
    assert snapshot.execution_ready is False
    assert snapshot.declarations.price_adjustment_basis.evidence_status == "caller_declared"
    assert {item.evidence_status for item in report.evidence_references} == {"artifact_verified", "authenticated_external"}


def test_conversion_reuses_exact_existing_contracts_and_calendar_clocks(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    report = validate(snapshot, evidence)
    for index, observed in enumerate(report.market):
        assert type(observed) is ObservedMarketBar
        assert type(observed.bar) is MarketBar
        expected = frame(evidence.calendar).iloc[index]
        bar = observed.bar
        assert observed.asset_id == ASSET.asset_id and bar.symbol == ASSET.symbol
        assert bar.session_id == evidence.calendar.sessions[index].session_id
        assert bar.open_timestamp == evidence.calendar.sessions[index].open_timestamp
        assert bar.close_timestamp == evidence.calendar.sessions[index].close_timestamp
        assert (bar.open_price, bar.high, bar.low, bar.close_price, bar.volume) == tuple(expected.iloc[:5])
        assert observed.provenance.source_id == "synthetic-provider"
        assert observed.provenance.snapshot_id == snapshot.snapshot_id
    # This calls only the offline adapter, never the execution engine.
    adapted = adapt_execution_inputs(ASSET, report.calendar, report.market, [],
                                     feature_availability={}, signal_mode="long_only")
    assert adapted.bars == report.bars and adapted.signals == ()


def test_missing_expected_market_session_is_ineligible(tmp_path):
    data = frame(schedule()).iloc[[0, 2]]
    snapshot, evidence = fixture(tmp_path, data=data)
    assert_blocked(validate(snapshot, evidence), "ineligible", "missing_expected_session")


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_source_dates_are_not_deduplicated(tmp_path, conflicting):
    data = pd.concat([frame(schedule()), frame(schedule()).iloc[[0]]])
    if conflicting:
        data.iloc[-1, 3] = 500.0
    snapshot, evidence = fixture(tmp_path, data=data)
    original = snapshot.canonical_json()
    report = validate(snapshot, evidence)
    assert_blocked(report, "ineligible", "duplicate_session_labels")
    if conflicting:
        assert "conflicting_duplicate_observations" in codes(report)
    assert len(snapshot.rows) == 4 and snapshot.canonical_json() == original


@pytest.mark.parametrize("name,flag,value", [("open", "observed", False), ("close", "filled", True),
                                            ("high", "repaired", True)])
def test_signed_unobserved_filled_or_repaired_fields_are_ineligible(tmp_path, name, flag, value):
    snapshot, evidence = fixture(tmp_path)
    row = evidence.observations[0]
    row = replace(row, fields=tuple(replace(item, **{flag: value}) if item.name == name else item for item in row.fields))
    evidence = replace(evidence, observations=(row,) + evidence.observations[1:])
    assert_blocked(validate(snapshot, evidence), "ineligible", "unobserved_or_repaired_field")


@pytest.mark.parametrize("column", ["Repaired?", "Open_Filled", "Close_Repaired", "High_Observed"])
def test_source_flags_override_conflicting_clean_lineage_claims(tmp_path, column):
    data = frame(schedule())
    data[column] = [False if column.endswith("Observed") else True] * 3
    snapshot, evidence = fixture(tmp_path, data=data)
    report = validate(snapshot, evidence)
    assert report.status == "ineligible" and report.market == ()
    assert {"repaired_observations", "source_quality_flag"} & codes(report)


def test_missing_per_field_lineage_is_unknown(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    row = replace(evidence.observations[0], fields=evidence.observations[0].fields[:-1])
    assert_blocked(validate(snapshot, replace(evidence, observations=(row,) + evidence.observations[1:])),
                   "unknown", "missing_observation_flags")


def test_unknown_observation_flags_do_not_become_true(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    first = evidence.observations[0]
    row = replace(first, fields=(replace(first.fields[0], observed=None),) + first.fields[1:])
    assert_blocked(validate(snapshot, replace(evidence, observations=(row,) + evidence.observations[1:])),
                   "unknown", "unknown_observation_status")


def test_missing_acquisition_metadata_returns_unknown(tmp_path):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    report = validate(snapshot, replace(evidence, acquisition=None))
    assert_blocked(report, "unknown", "missing_acquisition_evidence")
    unknown = validate(snapshot, replace(evidence, acquisition=AcquisitionMetadata()))
    assert_blocked(unknown, "unknown", "missing_acquisition_metadata")
    assert len([reason for reason in unknown.reasons if reason.code == "missing_acquisition_metadata"]) == 10


def test_explicit_authenticated_metadata_can_fill_unknowns_without_rewriting_snapshot(tmp_path):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    report = validate(snapshot, evidence)
    assert report.status == "eligible"
    assert snapshot.declarations.acquisition.provider.evidence_status == "unknown"
    assert snapshot.execution_ready is False


@pytest.mark.parametrize("name", ["price_adjustment_basis", "volume_adjustment_basis"])
@pytest.mark.parametrize("value", [None, "unknown"])
def test_unknown_adjustment_basis_is_unknown(tmp_path, name, value):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    assert_blocked(validate(snapshot, replace(evidence, **{name: value})), "unknown", "unknown_adjustment_basis")


@pytest.mark.parametrize("name,value", [("price_adjustment_basis", "adjusted"),
                                       ("price_adjustment_basis", "split_adjusted"),
                                       ("volume_adjustment_basis", "split_adjusted")])
def test_adjusted_prices_or_incompatible_volume_are_ineligible(tmp_path, name, value):
    snapshot, evidence = fixture(tmp_path)
    assert_blocked(validate(snapshot, replace(evidence, **{name: value})), "ineligible", "incompatible_adjustment_basis")


def test_auto_adjusted_acquisition_cannot_claim_unadjusted_fills(tmp_path):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    evidence = replace(evidence, acquisition=replace(evidence.acquisition, auto_adjust=claim(True)))
    assert_blocked(validate(snapshot, evidence), "ineligible", "adjusted_acquisition")


@pytest.mark.parametrize("kind", ["dividend", "split", "spinoff"])
def test_actions_within_evaluation_window_are_ineligible(tmp_path, kind):
    snapshot, evidence = fixture(tmp_path)
    actions = replace(evidence.corporate_actions, events=(CorporateAction(kind, evidence.calendar.sessions[1].open_timestamp),))
    assert_blocked(validate(snapshot, replace(evidence, corporate_actions=actions)), "ineligible", "unsupported_corporate_action")


@pytest.mark.parametrize("column", ["Dividends", "Stock Splits", "Capital Gains"])
def test_source_actions_cannot_be_hidden_by_empty_external_event_list(tmp_path, column):
    data = frame(schedule())
    data[column] = [0.0, 1.0, 0.0]
    snapshot, evidence = fixture(tmp_path, data=data)
    assert_blocked(validate(snapshot, evidence), "ineligible", "source_corporate_action")


@pytest.mark.parametrize("change", [{"complete": False}, {"complete": None},
                                   {"covered_from": schedule().sessions[0].open_timestamp + timedelta(seconds=1)},
                                   {"covered_through": schedule().sessions[-1].close_timestamp - timedelta(seconds=1)}])
def test_incomplete_action_coverage_is_unknown(tmp_path, change):
    snapshot, evidence = fixture(tmp_path)
    evidence = replace(evidence, corporate_actions=replace(evidence.corporate_actions, **change))
    assert_blocked(validate(snapshot, evidence), "unknown", "incomplete_corporate_action_coverage")


def test_zero_action_columns_do_not_replace_external_coverage(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    assert_blocked(validate(snapshot, replace(evidence, corporate_actions=None)), "unknown", "missing_corporate_action_evidence")


def test_action_evidence_for_another_asset_is_ineligible(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    actions = replace(evidence.corporate_actions, asset_id="another-asset")
    assert_blocked(validate(snapshot, replace(evidence, corporate_actions=actions)), "ineligible", "action_asset_mismatch")


@pytest.mark.parametrize("column", ["Repaired?", "Open_Filled"])
def test_missing_source_quality_flags_remain_unknown(tmp_path, column):
    data = frame(schedule())
    data[column] = pd.array([False, pd.NA, False], dtype="boolean")
    snapshot, evidence = fixture(tmp_path, data=data)
    assert_blocked(validate(snapshot, evidence), "unknown", "unknown_source_quality_flag")


def test_generic_filled_source_flags_are_ineligible(tmp_path):
    data = frame(schedule())
    data["Filled?"] = [False, True, False]
    snapshot, evidence = fixture(tmp_path, data=data)
    assert_blocked(validate(snapshot, evidence), "ineligible", "source_quality_flag")


def test_missing_source_action_values_remain_unknown(tmp_path):
    data = frame(schedule())
    data.loc[data.index[0], "Dividends"] = np.nan
    snapshot, evidence = fixture(tmp_path, data=data)
    assert_blocked(validate(snapshot, evidence), "unknown", "unknown_source_action_value")


@pytest.mark.parametrize("changes,code", [
    ({"provider_id": "different-provider"}, "provider_mismatch"),
    ({"listing_exchange": "XNAS"}, "declaration_conflict"),
    ({"calendar_exchange": "XNAS"}, "calendar_exchange_mismatch"),
    ({"source_sha256": "0" * 64}, "source_identity_mismatch"),
    ({"artifact_id": "other-artifact"}, "source_identity_mismatch"),
    ({"snapshot_id": "other-snapshot"}, "snapshot_identity_mismatch"),
    ({"exchange_timezone": "Not/A_Zone"}, "invalid_exchange_timezone"),
])
def test_evidence_identity_and_source_inconsistencies_are_ineligible(tmp_path, changes, code):
    snapshot, evidence = fixture(tmp_path)
    assert_blocked(validate(snapshot, replace(evidence, **changes)), "ineligible", code)


def test_unsupported_futures_preserve_negative_source_prices_but_produce_no_bars(tmp_path):
    data = frame(schedule())
    data["Open"] = [-37.63, -36, -35]
    snapshot, evidence = fixture(tmp_path, data=data, source_declarations=SnapshotDeclarations())
    evidence = replace(evidence, asset=replace(ASSET, instrument_type="future"))
    assert_blocked(validate(snapshot, evidence), "ineligible", "unsupported_instrument")
    assert snapshot.rows[0].values[0].to_python() == -37.63


@pytest.mark.parametrize("column,value,code", [("Open", 0, "invalid_ohlcv"), ("Close", -1, "invalid_ohlcv"),
                                               ("Volume", -1, "invalid_ohlcv"), ("High", np.inf, "invalid_ohlcv"),
                                               ("Low", 500, "invalid_ohlc_relationship"),
                                               ("Close", np.nan, "missing_ohlcv")])
def test_invalid_prices_volume_and_relationships_are_ineligible(tmp_path, column, value, code):
    data = frame(schedule())
    data.loc[data.index[0], column] = value
    snapshot, evidence = fixture(tmp_path, data=data)
    assert_blocked(validate(snapshot, evidence), "ineligible", code)


def test_large_integer_volume_is_preserved_without_float_rounding(tmp_path):
    data = frame(schedule())
    data["Volume"] = [2**53 + 1, 0, 3000]
    snapshot, evidence = fixture(tmp_path, data=data)
    report = validate(snapshot, evidence)
    assert report.status == "eligible"
    assert report.bars[0].volume == 2**53 + 1 and type(report.bars[0].volume) is int
    assert report.bars[1].volume == 0


def test_single_ticker_multiindex_preserves_values_and_identity(tmp_path):
    data = frame(schedule())
    data.columns = pd.MultiIndex.from_tuples([(name, ASSET.symbol) for name in data.columns], names=["Price", "Ticker"])
    snapshot, evidence = fixture(tmp_path, data=data)
    report = validate(snapshot, evidence)
    assert report.status == "eligible"
    assert report.bars[0].open_price == 100
    assert snapshot.artifact.columns[0].label.to_python() == ("Open", "SYNTH")


def test_etf_identity_is_supported(tmp_path):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    report = validate(snapshot, replace(evidence, asset=replace(ASSET, instrument_type="etf")))
    assert report.status == "eligible" and report.asset.instrument_type == "etf"


def test_dst_schedule_uses_supplied_aware_clocks_and_keeps_naive_source_labels(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    report = validate(snapshot, evidence)
    assert report.status == "eligible"
    assert report.bars[0].open_timestamp.utcoffset() == timedelta(hours=-5)
    assert report.bars[1].open_timestamp.utcoffset() == timedelta(hours=-4)
    assert snapshot.rows[0].session_label.to_python().tz is None
    assert snapshot.rows[0].session_label.to_python().hour == 0
    assert report.bars[0].close_timestamp.hour == 16


def test_holiday_and_early_close_are_supplied_not_inferred(tmp_path):
    calendar = schedule(holiday=True)
    snapshot, evidence = fixture(tmp_path, calendar=calendar)
    report = validate(snapshot, evidence)
    assert report.status == "eligible"
    assert [bar.session_id for bar in report.bars] == ["2020-11-25", "2020-11-27", "2020-11-30"]
    assert report.bars[1].close_timestamp.hour == 13


def test_missing_session_in_calendar_is_detected_by_next_open_adjacency(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    calendar = replace(evidence.calendar, sessions=(evidence.calendar.sessions[0], evidence.calendar.sessions[2]))
    assert_blocked(validate(snapshot, replace(evidence, calendar=calendar)), "ineligible", "calendar_gap_or_overlap")


def test_naive_calendar_timestamp_cannot_become_an_execution_clock(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    first = evidence.calendar.sessions[0]
    calendar = replace(evidence.calendar, sessions=(replace(first, open_timestamp=first.open_timestamp.replace(tzinfo=None)),)
                       + evidence.calendar.sessions[1:])
    assert_blocked(validate(snapshot, replace(evidence, calendar=calendar)), "ineligible", "invalid_calendar_timestamp")


def test_final_next_open_cutoff_is_required_even_without_a_future_price_bar(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    sessions = evidence.calendar.sessions
    calendar = replace(evidence.calendar, sessions=sessions[:-1] + (replace(sessions[-1], next_open_timestamp=None),))
    assert_blocked(validate(snapshot, replace(evidence, calendar=calendar)), "ineligible", "invalid_calendar_timestamp")


def test_nonexistent_dst_local_time_is_rejected(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    first = evidence.calendar.sessions[0]
    invalid_open = datetime(2024, 3, 10, 2, 30, tzinfo=NY)
    calendar = replace(evidence.calendar, sessions=(replace(first, open_timestamp=invalid_open,
                                                          close_timestamp=invalid_open.replace(hour=16)),)
                       + evidence.calendar.sessions[1:])
    assert_blocked(validate(snapshot, replace(evidence, calendar=calendar)), "ineligible", "nonexistent_local_time")


def test_invalid_calendar_order_and_duplicate_ids_are_ineligible(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    first = evidence.calendar.sessions[0]
    calendar = replace(evidence.calendar, sessions=(replace(first, close_timestamp=first.open_timestamp),)
                       + evidence.calendar.sessions[1:] + (first,))
    report = validate(snapshot, replace(evidence, calendar=calendar))
    assert_blocked(report, "ineligible", "invalid_calendar_timing")
    assert "duplicate_calendar_session" in codes(report)


@pytest.mark.parametrize("name", ["asset", "listing_exchange", "exchange_timezone", "calendar", "calendar_exchange",
                                  "start_session_id", "end_session_id"])
def test_missing_required_provenance_returns_unknown(tmp_path, name):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    report = validate(snapshot, replace(evidence, **{name: None}))
    assert report.status == "unknown" and report.market == ()
    assert report.reasons


@pytest.mark.parametrize("name", ["asset_id", "symbol", "calendar_id", "quote_currency", "instrument_type"])
def test_missing_fields_within_asset_identity_are_unknown(tmp_path, name):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    report = validate(snapshot, replace(evidence, asset=replace(ASSET, **{name: None})))
    assert report.status == "unknown" and report.bars == ()


def test_blank_acquisition_fields_are_missing_evidence_not_eligible(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    evidence = replace(evidence, acquisition=replace(evidence.acquisition, requested_start=claim("")))
    assert_blocked(validate(snapshot, evidence), "unknown", "missing_acquisition_metadata")


def test_acquisition_cannot_predate_final_observed_close(tmp_path):
    snapshot, evidence = fixture(tmp_path, source_declarations=SnapshotDeclarations())
    acquisition_info = replace(evidence.acquisition, acquired_at=claim(evidence.calendar.sessions[0].close_timestamp))
    assert_blocked(validate(snapshot, replace(evidence, acquisition=acquisition_info)), "ineligible", "acquisition_precedes_observations")


def test_duplicate_row_to_session_alignment_is_rejected(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    repeated = replace(evidence.observations[1], session_id=evidence.observations[0].session_id)
    evidence = replace(evidence, observations=(evidence.observations[0], repeated, evidence.observations[2]))
    assert_blocked(validate(snapshot, evidence), "ineligible", "duplicate_session_alignment")


def test_wrong_column_positions_cannot_change_price_meanings(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    first = evidence.observations[0]
    row = replace(first, fields=(replace(first.fields[0], column_position=1),) + first.fields[1:])
    evidence = replace(evidence, observations=(row,) + evidence.observations[1:])
    assert_blocked(validate(snapshot, evidence), "ineligible", "invalid_column_mapping")


def test_unsigned_and_untrusted_claims_never_become_authenticated(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    unsigned = EvidenceAttestation("synthetic-issuer", "unsigned", evidence)
    assert_blocked(validate_market_eligibility(snapshot, unsigned, trusted_keys=TRUST), "unknown", "missing_authentication")
    assert_blocked(validate(snapshot, evidence, trusted_keys={}), "unknown", "untrusted_issuer")
    assert_blocked(validate_market_eligibility(snapshot), "unknown", "missing_observations_evidence")
    assert not validate_market_eligibility(snapshot).authenticated_external_evidence


def test_invalid_signature_and_payload_tampering_are_rejected(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    attestation = sign(evidence)
    invalid = replace(attestation, signature="0" * 64)
    assert_blocked(validate_market_eligibility(snapshot, invalid, trusted_keys=TRUST), "ineligible", "invalid_authentication")
    tampered = replace(attestation, evidence=replace(evidence, provider_id="tampered"))
    assert_blocked(validate_market_eligibility(snapshot, tampered, trusted_keys=TRUST), "ineligible", "invalid_authentication")
    relabelled = replace(attestation, reference_id="another-reference")
    assert_blocked(validate_market_eligibility(snapshot, relabelled, trusted_keys=TRUST), "ineligible", "invalid_authentication")


def test_changed_snapshot_contents_do_not_pass_old_authentication(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    changed_row = replace(snapshot.rows[0], values=(snapshot.rows[1].values[0],) + snapshot.rows[0].values[1:])
    changed = replace(snapshot, rows=(changed_row,) + snapshot.rows[1:])
    assert_blocked(validate(changed, evidence), "ineligible", "snapshot_identity_mismatch")


def test_file_rename_does_not_change_report_or_authentication(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    new_path = tmp_path / "renamed.parquet"
    shutil.copyfile(snapshot.source_path, new_path)
    renamed = read_market_snapshot(new_path, artifact_id=snapshot.artifact.artifact_id, declarations=snapshot.declarations)
    first, second = validate(snapshot, evidence), validate(renamed, evidence)
    assert first.status == second.status == "eligible"
    assert first.market == second.market
    assert first.evidence_references == second.evidence_references
    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id


def test_shuffled_source_and_evidence_order_align_by_explicit_keys(tmp_path):
    data = frame(schedule()).iloc[[2, 0, 1]]
    snapshot, evidence = fixture(tmp_path, data=data)
    evidence = replace(evidence, observations=tuple(reversed(evidence.observations)))
    report = validate(snapshot, evidence)
    assert report.status == "eligible"
    assert [bar.open_price for bar in report.bars] == [100, 101, 102]
    assert snapshot.rows[0].values[0].to_python() == 102


def test_immutable_inputs_diagnostics_and_reports_are_preserved(tmp_path):
    snapshot, evidence = fixture(tmp_path)
    original = snapshot.canonical_json()
    report = validate(snapshot, evidence)
    assert report.snapshot_diagnostics is snapshot.diagnostics
    assert snapshot.canonical_json() == original
    with pytest.raises(FrozenInstanceError):
        report.status = "eligible"
    with pytest.raises(FrozenInstanceError):
        evidence.observations[0].fields[0].observed = False
    with pytest.raises(FrozenInstanceError):
        report.market[0].bar.open_price = 999
    assert KEY.hex() not in repr(report)
    assert KEY.decode() not in repr(report)


def test_no_network_file_database_or_engine_operations(tmp_path, monkeypatch):
    snapshot, evidence = fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Eligibility validation must be pure and offline.")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(type(tmp_path), "read_bytes", forbidden)
    assert validate(snapshot, evidence).status == "eligible"

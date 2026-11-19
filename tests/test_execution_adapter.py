"""Deterministic offline calendar, observation, provenance and signal alignment."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, MarketBar, Signal
from src.evaluation.execution.adapter import (
    AdapterValidationError, AssetSpec, ObservedMarketBar, PriceProvenance,
    SessionCalendar, TradingSession, adapt_execution_inputs,
)
from src.evaluation.execution.prediction_export import (
    ExportedPrediction, ModelProvenance, PredictionIdentity, PredictiveTargetMetadata,
)


NY = ZoneInfo("America/New_York")
ASSET = AssetSpec("asset-1", "SYNTH", "equity", "synthetic-exchange", "USD")
SOURCE = PriceProvenance("synthetic-observed-source", "immutable-snapshot-1", "unadjusted", True, True)
TARGET = PredictiveTargetMetadata("5d_class", 5, 0.005)
PAST = datetime(2019, 12, 1, tzinfo=timezone.utc)
PROVENANCE = ModelProvenance("model-1", "split-1", PAST, PAST, PAST, PAST, False, False)


def calendar():
    """Declare every session and boundary explicitly, including the DST weekend."""
    dates = [(2020, 3, 6), (2020, 3, 9), (2020, 3, 10), (2020, 3, 11), (2020, 3, 12), (2020, 3, 13)]
    opens = [datetime(*date, 9, 30, tzinfo=NY) for date in dates]
    rows = [TradingSession(f"session-{i}", opens[i], datetime(*date, 16, tzinfo=NY), opens[i + 1])
            for i, date in enumerate(dates[:-1])]
    return SessionCalendar(ASSET.calendar_id, "synthetic-calendar-v1", tuple(rows), True)


def market(schedule=None):
    schedule = calendar() if schedule is None else schedule
    return [ObservedMarketBar(ASSET.asset_id, MarketBar(
        ASSET.symbol, session.session_id, session.open_timestamp, session.close_timestamp,
        100, 100, high=101, low=99, volume=1000), SOURCE) for session in schedule.sessions]


def prediction(session, label=2, **changes):
    identity = PredictionIdentity(ASSET.asset_id, f"prediction-{session.session_id}", session.session_id,
                                  f"source-row-{session.session_id}", session.close_timestamp,
                                  session.close_timestamp + timedelta(minutes=5))
    options = dict(identity=identity, predicted_class=label, probabilities={0: 0.2, 1: 0.3, 2: 0.5},
                   provenance=PROVENANCE, target_metadata=TARGET, mode="historical_replay")
    return ExportedPrediction(**(options | changes))


def adapt(schedule=None, observed=None, predictions=None, availability=None, **changes):
    schedule = calendar() if schedule is None else schedule
    observed = market(schedule) if observed is None else observed
    predictions = [prediction(schedule.sessions[0], 2), prediction(schedule.sessions[2], 1)] if predictions is None else predictions
    availability = {p.identity.prediction_id: p.identity.feature_observation_timestamp + timedelta(minutes=1)
                    for p in predictions} if availability is None else availability
    options = dict(asset=ASSET, calendar=schedule, market=observed, predictions=predictions,
                   feature_availability=availability, signal_mode="short_enabled")
    return adapt_execution_inputs(**(options | changes))


def assert_rejection(code, **changes):
    with pytest.raises(AdapterValidationError) as caught:
        adapt(**changes)
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == code and diagnostic.message
    assert isinstance(caught.value.diagnostics, tuple)
    return diagnostic


def test_exact_keyed_alignment_preserves_all_market_sessions_and_prediction_provenance():
    schedule = calendar()
    observed = market(schedule)
    predictions = [prediction(schedule.sessions[0], 2), prediction(schedule.sessions[2], 1)]
    result = adapt(schedule, observed, predictions)
    assert len(result.bars) == 5 and len(result.signals) == 2
    assert all(isinstance(bar, MarketBar) for bar in result.bars)
    assert all(isinstance(signal, Signal) for signal in result.signals)
    assert result.bars == tuple(row.bar for row in observed)
    assert result.bars[0] is observed[0].bar
    assert [signal.session_id for signal in result.signals] == ["session-0", "session-2"]
    assert [entry.prediction for entry in result.manifest.alignments] == predictions
    assert result.manifest.alignments[0].prediction.provenance is PROVENANCE
    assert result.manifest.market == tuple(observed)
    assert not result.manifest.provenance_independently_verified
    assert result.rejection_diagnostics == ()


def test_shuffling_market_calendar_and_prediction_order_does_not_change_alignment():
    schedule = calendar()
    observed = market(schedule)
    predictions = [prediction(schedule.sessions[i], label) for i, label in [(0, 2), (2, 1), (4, 0)]]
    first = adapt(schedule, observed, predictions)
    shuffled = adapt(replace(schedule, sessions=tuple(reversed(schedule.sessions))),
                     [observed[i] for i in [3, 0, 4, 2, 1]], list(reversed(predictions)))
    assert first == shuffled


def test_absent_prediction_is_not_an_explicit_flat_instruction():
    result = adapt()
    assert result.signals[1].direction == "flat"
    assert result.signals[1].session_id == "session-2"
    assert not any(signal.session_id == "session-1" for signal in result.signals)
    assert result.bars[1].session_id == "session-1"
    empty = adapt(predictions=[], availability={})
    assert empty.signals == () and len(empty.bars) == 5
    assert empty.manifest.alignments == ()


@pytest.mark.parametrize("mode,label,direction", [
    ("short_enabled", 0, "short"), ("short_enabled", 1, "flat"), ("short_enabled", 2, "long"),
    ("long_only", 0, "flat"), ("long_only", 1, "flat"), ("long_only", 2, "long"),
])
def test_explicit_signal_modes_preserve_class_mapping(mode, label, direction):
    result = adapt(predictions=[prediction(calendar().sessions[0], label)], signal_mode=mode)
    assert result.signals[0].direction == direction
    assert result.manifest.signal_mode == mode
    assert result.manifest.alignments[0].prediction.predicted_class == label


def test_signal_mode_must_be_explicit_and_supported():
    assert_rejection("invalid_signal_mode", signal_mode="guess_from_labels")
    with pytest.raises(TypeError, match="signal_mode"):
        adapt_execution_inputs(ASSET, calendar(), market(), [], feature_availability={})


@pytest.mark.parametrize("where", ["calendar", "market", "prediction"])
def test_duplicate_sessions_and_predictions_are_rejected(where):
    schedule = calendar()
    if where == "calendar":
        assert_rejection("duplicate_session", schedule=replace(schedule, sessions=schedule.sessions + (schedule.sessions[0],)))
    elif where == "market":
        rows = market(schedule)
        assert_rejection("duplicate_session", observed=rows + [rows[0]])
    else:
        first = prediction(schedule.sessions[0])
        assert_rejection("duplicate_prediction", predictions=[first, first])


@pytest.mark.parametrize("key", ["prediction_id", "feature_row_id"])
def test_duplicate_source_or_prediction_ids_across_sessions_are_rejected(key):
    schedule = calendar()
    first, second = prediction(schedule.sessions[0]), prediction(schedule.sessions[1])
    second = replace(second, identity=replace(second.identity, **{key: getattr(first.identity, key)}))
    assert_rejection("duplicate_prediction", predictions=[first, second])


@pytest.mark.parametrize("field", ["asset_id", "symbol", "calendar_id", "quote_currency"])
def test_missing_asset_identifiers_are_rejected(field):
    assert_rejection("missing_identifier", asset=replace(ASSET, **{field: ""}))


def test_missing_market_session_id_is_rejected():
    rows = market()
    rows[0] = replace(rows[0], bar=replace(rows[0].bar, session_id=""))
    assert_rejection("missing_identifier", observed=rows)


@pytest.mark.parametrize("missing_index", [0, 1, 4])
def test_missing_expected_price_session_is_not_silently_skipped(missing_index):
    rows = market()
    rows.pop(missing_index)
    diagnostic = assert_rejection("missing_market_sessions", observed=rows)
    assert diagnostic.session_id == f"session-{missing_index}"


def test_calendar_gap_is_detected_even_if_the_same_market_row_is_missing():
    schedule = calendar()
    incomplete = replace(schedule, sessions=schedule.sessions[:1] + schedule.sessions[2:])
    assert_rejection("calendar_gap_or_overlap", schedule=incomplete,
                     observed=market(schedule)[:1] + market(schedule)[2:], predictions=[])


def test_calendar_must_be_independently_declared_complete():
    assert_rejection("incomplete_calendar", schedule=replace(calendar(), declared_complete=False))
    assert_rejection("incomplete_calendar", schedule=replace(calendar(), sessions=()), predictions=[])
    assert_rejection("calendar_mismatch", schedule=replace(calendar(), calendar_id="other-calendar"))


def test_extra_market_session_is_not_silently_discarded():
    rows = market()
    extra = replace(rows[-1], bar=replace(rows[-1].bar, session_id="unexpected-session"))
    assert_rejection("unexpected_market_session", observed=rows + [extra])


@pytest.mark.parametrize("where", ["market", "prediction"])
def test_one_asset_is_enforced_on_both_inputs(where):
    if where == "market":
        rows = market()
        rows[1] = replace(rows[1], asset_id="asset-2")
        assert_rejection("asset_mismatch", observed=rows)
    else:
        first = prediction(calendar().sessions[0])
        foreign = replace(first, identity=replace(first.identity, asset_id="asset-2"))
        assert_rejection("asset_mismatch", predictions=[foreign])


def test_prediction_session_must_exist_without_positional_fallback():
    first = prediction(calendar().sessions[0])
    unknown = replace(first, identity=replace(first.identity, session_id="unknown"))
    assert_rejection("unknown_prediction_session", predictions=[unknown])


@pytest.mark.parametrize("field", ["open_timestamp", "close_timestamp", "next_open_timestamp"])
@pytest.mark.parametrize("bad", [None, datetime(2020, 3, 6), pd.NaT, "2020-03-06"])
def test_calendar_clocks_must_be_explicit_and_timezone_aware(field, bad):
    schedule = calendar()
    bad_session = replace(schedule.sessions[0], **{field: bad})
    assert_rejection("invalid_timestamp", schedule=replace(schedule, sessions=(bad_session,) + schedule.sessions[1:]),
                     predictions=[])


def test_market_clocks_must_match_independent_calendar():
    rows = market()
    rows[0] = replace(rows[0], bar=replace(rows[0].bar, open_timestamp=rows[0].bar.open_timestamp + timedelta(minutes=1)))
    assert_rejection("market_calendar_timing_mismatch", observed=rows)


def test_feature_observation_must_be_session_close_not_a_date_label():
    first = prediction(calendar().sessions[0])
    row = replace(first, identity=replace(first.identity,
                  feature_observation_timestamp=first.identity.feature_observation_timestamp.replace(hour=0)))
    diagnostic = assert_rejection("feature_session_mismatch", predictions=[row])
    assert diagnostic.prediction_id == row.identity.prediction_id


@pytest.mark.parametrize("position", ["before_close", "at_decision", "after_decision"])
def test_feature_availability_is_not_assumed_from_observation_time(position):
    first = prediction(calendar().sessions[0])
    available = {"before_close": first.identity.feature_observation_timestamp - timedelta(seconds=1),
                 "at_decision": first.identity.historical_decision_eligibility_timestamp,
                 "after_decision": first.identity.historical_decision_eligibility_timestamp + timedelta(seconds=1)}[position]
    assert_rejection("feature_not_available", predictions=[first], availability={first.identity.prediction_id: available})


@pytest.mark.parametrize("value", [None, datetime(2020, 3, 6), pd.NaT])
def test_missing_or_naive_feature_availability_fails(value):
    first = prediction(calendar().sessions[0])
    assert_rejection("invalid_timestamp", predictions=[first], availability={first.identity.prediction_id: value})


@pytest.mark.parametrize("mapping", [{}, {"wrong-prediction": datetime(2020, 3, 6, tzinfo=timezone.utc)}])
def test_feature_availability_keys_must_match_exactly(mapping):
    assert_rejection("feature_availability_keys", predictions=[prediction(calendar().sessions[0])], availability=mapping)


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=1)])
def test_decision_must_strictly_precede_next_open(offset):
    session = calendar().sessions[0]
    first = prediction(session)
    late = replace(first, identity=replace(first.identity,
                   historical_decision_eligibility_timestamp=session.next_open_timestamp + offset))
    assert_rejection("missed_next_open", predictions=[late])


def test_final_session_requires_explicit_next_open_cutoff_without_a_future_price_bar():
    schedule = calendar()
    final = prediction(schedule.sessions[-1])
    result = adapt(predictions=[final])
    assert result.signals[0].session_id == schedule.sessions[-1].session_id
    assert result.manifest.alignments[0].next_open_timestamp == schedule.sessions[-1].next_open_timestamp
    bad = replace(schedule.sessions[-1], next_open_timestamp=None)
    assert_rejection("invalid_timestamp", schedule=replace(schedule, sessions=schedule.sessions[:-1] + (bad,)), predictions=[])


def test_model_information_cutoff_must_be_strictly_before_decision():
    first = prediction(calendar().sessions[0])
    decision_time = first.identity.historical_decision_eligibility_timestamp
    equal = replace(PROVENANCE, model_information_cutoff=decision_time,
                    historical_model_available_at=decision_time)
    # The export contract permits equality; this stricter adapter requires prior information.
    first = replace(first, provenance=equal)
    assert_rejection("model_not_available", predictions=[first])


def test_historical_generation_time_is_not_misrepresented_as_a_historical_decision():
    first = prediction(calendar().sessions[0], actual_prediction_generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    result = adapt(predictions=[first])
    assert result.signals[0].decision_timestamp == first.identity.historical_decision_eligibility_timestamp
    assert result.manifest.alignments[0].prediction.actual_prediction_generated_at.year == 2026


def test_live_generation_uses_actual_time_and_rejects_a_missed_open():
    session = calendar().sessions[0]
    provenance = replace(PROVENANCE, actual_fit_completed_at=PAST)
    actual = session.close_timestamp + timedelta(minutes=10)
    first = prediction(session, provenance=provenance, mode="live_generation", actual_prediction_generated_at=actual)
    result = adapt(predictions=[first])
    assert result.signals[0].decision_timestamp == actual
    late = replace(first, actual_prediction_generated_at=session.next_open_timestamp + timedelta(seconds=1))
    assert_rejection("missed_next_open", predictions=[late])


def test_dst_weekend_uses_supplied_schedule_not_a_fixed_utc_hour():
    result = adapt()
    assert result.bars[0].open_timestamp.astimezone(timezone.utc).hour == 14
    assert result.bars[1].open_timestamp.astimezone(timezone.utc).hour == 13
    assert result.manifest.alignments[0].next_open_timestamp == result.bars[1].open_timestamp
    assert (result.bars[1].open_timestamp.astimezone(timezone.utc)
            - result.bars[0].close_timestamp.astimezone(timezone.utc)).total_seconds() == 64.5 * 3600


def test_holiday_and_early_close_are_exactly_the_declared_sessions():
    # Synthetic Thanksgiving closure and early-close schedule, supplied explicitly.
    opened = datetime(2020, 11, 25, 9, 30, tzinfo=NY)
    next_open = datetime(2020, 11, 27, 9, 30, tzinfo=NY)
    monday = datetime(2020, 11, 30, 9, 30, tzinfo=NY)
    sessions = (TradingSession("pre-holiday", opened, opened.replace(hour=16, minute=0), next_open),
                TradingSession("early-close", next_open, next_open.replace(hour=13, minute=0), monday))
    schedule = SessionCalendar(ASSET.calendar_id, "synthetic-holiday-schedule", sessions, True)
    first = prediction(sessions[1])
    result = adapt(schedule, market(schedule), [first])
    assert len(result.bars) == 2
    assert result.signals[0].decision_timestamp.hour == 13
    assert result.manifest.alignments[0].next_open_timestamp == monday


@pytest.mark.parametrize("field", ["open_observed", "close_observed"])
def test_filled_execution_price_flags_are_rejected(field):
    rows = market()
    rows[1] = replace(rows[1], bar=replace(rows[1].bar, **{field: False}))
    assert_rejection("synthetic_execution_price", observed=rows)


def test_ohlcv_observation_provenance_is_required_even_when_prices_are_numeric():
    rows = market()
    rows[1] = replace(rows[1], provenance=replace(SOURCE, all_ohlcv_observed=False))
    assert_rejection("synthetic_execution_price", observed=rows)


@pytest.mark.parametrize("basis", ["unknown", "adjusted", "split_adjusted", "total_return_adjusted", None])
def test_unknown_or_unsupported_adjustment_bases_are_rejected(basis):
    rows = market()
    rows[1] = replace(rows[1], provenance=replace(SOURCE, adjustment_basis=basis))
    assert_rejection("unsupported_adjustment_basis", observed=rows)


@pytest.mark.parametrize("field", ["source_id", "snapshot_id"])
def test_market_source_provenance_must_be_explicit(field):
    rows = market()
    rows[1] = replace(rows[1], provenance=replace(SOURCE, **{field: ""}))
    assert_rejection("missing_identifier", observed=rows)


def test_corporate_action_windows_are_not_silently_treated_as_cash_equities():
    rows = market()
    rows[1] = replace(rows[1], provenance=replace(SOURCE, corporate_actions_free=False))
    assert_rejection("unsupported_corporate_actions", observed=rows)


@pytest.mark.parametrize("instrument", ["future", "index", "crypto", "unknown"])
def test_unsupported_instruments_are_rejected(instrument):
    assert_rejection("unsupported_instrument", asset=replace(ASSET, instrument_type=instrument))


def test_etf_is_supported_and_zero_volume_is_not_a_price_error():
    rows = market()
    rows[0] = replace(rows[0], bar=replace(rows[0].bar, volume=0))
    result = adapt(observed=rows, asset=replace(ASSET, instrument_type="etf"))
    assert result.bars[0].volume == 0


@pytest.mark.parametrize("field,value", [("open_price", None), ("close_price", float("nan")),
                                          ("high", None), ("low", -1), ("volume", -1),
                                          ("open_price", "100"), ("close_price", True)])
def test_invalid_ohlcv_is_never_repaired_or_dropped(field, value):
    rows = market()
    rows[1] = replace(rows[1], bar=replace(rows[1].bar, **{field: value}))
    assert_rejection("invalid_ohlcv", observed=rows)


def test_economically_inconsistent_ohlc_is_rejected_without_outlier_replacement():
    rows = market()
    rows[1] = replace(rows[1], bar=replace(rows[1].bar, high=90))
    assert_rejection("inconsistent_ohlc", observed=rows)
    rows[1] = replace(rows[1], bar=replace(rows[1].bar, high=10000))
    assert adapt(observed=rows).bars[1].high == 10000


def test_input_containers_and_manifest_are_immutable_without_changing_source_objects():
    schedule = calendar()
    rows = market(schedule)
    first = prediction(schedule.sessions[0])
    predictions = [first]
    availability = {first.identity.prediction_id: first.identity.feature_observation_timestamp + timedelta(minutes=1)}
    rows_before = list(rows)
    result = adapt(schedule, rows, predictions, availability)
    assert rows == rows_before and predictions == [first]
    availability[first.identity.prediction_id] = first.identity.historical_decision_eligibility_timestamp
    rows.clear()
    predictions.clear()
    assert result.manifest.alignments[0].feature_available_at < result.signals[0].decision_timestamp
    assert len(result.bars) == len(result.manifest.market) == 5
    with pytest.raises(FrozenInstanceError):
        result.manifest.signal_mode = "long_only"
    with pytest.raises(FrozenInstanceError):
        result.manifest.alignments[0].signal.direction = "short"
    with pytest.raises(TypeError):
        result.manifest.alignments[0].prediction.probabilities[0] = 0.9


def test_future_price_changes_do_not_modify_any_signal_or_earlier_bars():
    rows = market()
    original = adapt(observed=rows)
    changed = [replace(row, bar=replace(row.bar, open_price=120, close_price=120, high=121, low=119))
               if i > 2 else row for i, row in enumerate(rows)]
    future = adapt(observed=changed)
    assert original.signals == future.signals
    assert original.manifest.alignments == future.manifest.alignments
    assert original.bars[:3] == future.bars[:3]


def test_appending_sessions_preserves_previous_boundary_prediction_eligibility():
    schedule = calendar()
    rows = market(schedule)
    forecasts = [prediction(session) for session in schedule.sessions]
    prefix_schedule = replace(schedule, sessions=schedule.sessions[:3])
    prefix = adapt(prefix_schedule, rows[:3], forecasts[:3])
    extended = adapt(schedule, rows, forecasts)
    assert prefix.bars == extended.bars[:3]
    assert prefix.signals == extended.signals[:3]
    assert prefix.manifest.alignments == extended.manifest.alignments[:3]
    assert prefix.manifest.alignments[-1].next_open_timestamp == extended.bars[3].open_timestamp


def test_outputs_work_with_existing_engine_and_preserve_non_prediction_sessions():
    result = adapt()
    engine = ExecutionEngineV1(ExecutionConfig(ASSET.symbol, initial_capital=1000,
                                              fee_rate=0, holding_sessions=2, allow_short=True))
    execution = engine.run(result.bars, result.signals)
    assert len(execution.ledger) == len(result.bars) + 1
    assert execution.fills.iloc[0].timestamp == result.bars[1].open_timestamp
    assert execution.fills.iloc[1].timestamp == result.bars[2].close_timestamp
    assert execution.run_status == "completed"


def test_adapter_does_not_run_engine_or_generate_predictions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Adapter must not execute trading.")
    monkeypatch.setattr(ExecutionEngineV1, "run", forbidden)
    assert len(adapt().bars) == 5

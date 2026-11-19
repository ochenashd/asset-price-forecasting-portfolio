"""Synthetic reference ledgers for the independent Stage 1 execution engine."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, MarketBar, Signal


def market(closes=None, opens=None):
    """Create observed OHLCV bars on an explicit synthetic UTC session calendar."""
    closes = [100, 105, 90, 110, 120, 100, 115, 100] if closes is None else closes
    opens = [100] * len(closes) if opens is None else opens
    days = pd.bdate_range("2020-01-06", periods=len(closes), tz="UTC")
    return [MarketBar(
        symbol="SYNTH", session_id=f"session-{i}",
        open_timestamp=day + pd.Timedelta(hours=9, minutes=30),
        close_timestamp=day + pd.Timedelta(hours=16),
        open_price=opening, close_price=close,
        high=max(opening, close) + 1, low=min(opening, close) - 1, volume=1000 + i,
    ) for i, (day, opening, close) in enumerate(zip(days, opens, closes))]


def signal(bar, direction="long"):
    """Make a decision strictly after its observed feature-session close."""
    return Signal(bar.session_id, bar.close_timestamp + timedelta(minutes=1), direction)


def engine(**kwargs):
    """Use small exact-capital examples with explicit fee and horizon defaults."""
    return ExecutionEngineV1(ExecutionConfig(
        symbol="SYNTH", initial_capital=1000, holding_sessions=kwargs.pop("holding_sessions", 3),
        fee_rate=kwargs.pop("fee_rate", 0.0), **kwargs,
    ))


def test_entry_occurs_at_next_open_and_quantity_uses_signal_close():
    bars = market()
    result = engine().run(bars, [signal(bars[0])])
    entry = result.fills.iloc[0]
    assert entry["timestamp"] == bars[1].open_timestamp
    assert entry["signal_timestamp"] == signal(bars[0]).decision_timestamp
    assert entry["quantity"] == 10
    assert entry["price"] == 100
    assert entry["timestamp"] > entry["signal_timestamp"]


def test_overnight_gap_before_execution_is_excluded():
    bars = market([100, 110, 110, 110], [100, 110, 110, 110])
    result = engine(allocation_fraction=0.5, holding_sessions=2).run(bars, [signal(bars[0])])
    assert result.fills.iloc[0]["quantity"] == 5
    assert result.fills.iloc[0]["price"] == 110
    np.testing.assert_allclose(result.ledger["equity"], 1000)
    np.testing.assert_allclose(result.ledger["daily_net_return"], 0)


def test_horizon_exit_is_pre_scheduled_for_the_correct_session_close():
    bars = market()
    result = engine(holding_sessions=3).run(bars, [signal(bars[0])])
    exit_fill = result.fills.iloc[1]
    assert exit_fill["side"] == "sell"
    assert exit_fill["session_id"] == bars[3].session_id
    assert exit_fill["timestamp"] == bars[3].close_timestamp
    assert exit_fill["price"] == 110
    assert result.events.query("event == 'entry_filled'").iloc[0]["expiry_session_ordinal"] == 3


def test_daily_cash_shares_equity_and_returns_match_independent_reference():
    bars = market()[:6]
    result = engine().run(bars, [signal(bars[0])])
    # Ten shares are bought for 1000, marked daily, and sold for 1100.
    np.testing.assert_allclose(result.ledger["cash"], [1000, 1000, 0, 0, 1100, 1100, 1100])
    np.testing.assert_array_equal(result.ledger["position_quantity"], [0, 0, 10, 10, 0, 0, 0])
    expected_equity = np.array([1000, 1000, 1050, 900, 1100, 1100, 1100])
    np.testing.assert_allclose(result.ledger["equity"], expected_equity)
    np.testing.assert_allclose(result.ledger["daily_net_return"],
                               np.r_[0, expected_equity[1:] / expected_equity[:-1] - 1])
    assert result.ledger.iloc[3]["unrealized_pnl"] == -100
    assert result.ledger.iloc[4]["realized_pnl"] == 100


def test_initial_capital_is_a_distinct_reference_before_all_sessions():
    bars = market()
    result = engine().run(bars, [])
    first = result.ledger.iloc[0]
    assert first["is_initial"]
    assert first["session_id"] is None
    assert first["timestamp"] < bars[0].open_timestamp
    assert first["cash"] == first["equity"] == 1000
    assert len(result.ledger) == len(bars) + 1
    assert result.ledger["timestamp"].iloc[1:].tolist() == [bar.close_timestamp for bar in bars]


def test_fees_are_charged_on_actual_entry_and_exit_notional():
    bars = market([100, 100, 110, 110])
    result = engine(fee_rate=0.01, holding_sessions=2).run(bars, [signal(bars[0])])
    # floor(1000 / 101) = 9 shares; buy costs 900+9, sell credits 990-9.9.
    np.testing.assert_allclose(result.fills["quantity"], [9, 9])
    np.testing.assert_allclose(result.fills["fee"], [9, 9.9])
    np.testing.assert_allclose(result.ledger["cash"], [1000, 1000, 91, 1071.1, 1071.1])
    np.testing.assert_allclose(result.ledger["equity"], [1000, 1000, 991, 1071.1, 1071.1])
    assert result.ledger.iloc[2]["unrealized_pnl"] == -9
    assert result.ledger.iloc[3]["realized_pnl"] == pytest.approx(71.1)


def test_entry_fee_uses_actual_open_even_when_it_differs_from_sizing_price():
    bars = market([100] * 4, [100, 90, 100, 100])
    result = engine(fee_rate=0.01, holding_sessions=2).run(bars, [signal(bars[0])])
    # Quantity remains nine; actual entry is 810 plus 8.10, not 900 plus 9.
    assert result.fills.iloc[0]["quantity"] == 9
    assert result.fills.iloc[0]["fee"] == pytest.approx(8.1)
    assert result.ledger.iloc[2]["cash"] == pytest.approx(181.9)
    assert result.ledger.iloc[-1]["equity"] == pytest.approx(1072.9)


def test_scheduled_exit_occurs_even_when_its_realized_price_creates_a_loss():
    bars = market()
    bars[3] = replace(bars[3], close_price=50.0)
    result = engine().run(bars, [signal(bars[0])])
    assert result.fills.iloc[-1]["timestamp"] == bars[3].close_timestamp
    assert result.fills.iloc[-1]["quantity"] == 10
    assert result.ledger.iloc[4]["equity"] == 500
    assert result.ledger.iloc[4]["realized_pnl"] == -500
    assert result.open_position is None


def test_flat_signals_preserve_cash_without_fees():
    bars = market()
    result = engine(fee_rate=0.01).run(bars, [signal(bar, "flat") for bar in bars])
    assert result.fills.empty
    assert result.open_position is None
    np.testing.assert_allclose(result.ledger["equity"], 1000)
    np.testing.assert_allclose(result.ledger["transaction_fees"], 0)


@pytest.mark.parametrize("bad_open", [None, np.nan, np.inf, 0.0, -1.0])
def test_invalid_execution_open_rejects_order_without_filling_or_retrying(bad_open):
    bars = market()
    bars[1] = replace(bars[1], open_price=bad_open)
    result = engine().run(bars, [signal(bars[0])])
    assert result.fills.empty
    rejection = result.events.query("event == 'entry_rejected'").iloc[0]
    assert rejection["reason"] == "invalid_or_unobserved_open"
    np.testing.assert_allclose(result.ledger["equity"], 1000)


def test_filled_open_price_is_not_accepted_as_observed_execution():
    bars = market()
    bars[1] = replace(bars[1], open_observed=False)
    result = engine().run(bars, [signal(bars[0])])
    assert result.fills.empty
    assert "invalid_or_unobserved_open" in result.events["reason"].values


def test_missing_open_without_an_order_does_not_invent_an_execution():
    bars = market()
    bars[1] = replace(bars[1], open_price=None)
    result = engine().run(bars, [])
    assert result.fills.empty
    np.testing.assert_allclose(result.ledger["equity"], 1000)


@pytest.mark.parametrize("bad_close", [None, np.nan, np.inf, 0.0, -1.0])
def test_invalid_daily_close_fails_explicitly_even_when_flat(bad_close):
    bars = market()
    bars[2] = replace(bars[2], close_price=bad_close)
    with pytest.raises(ValueError, match="valuation close"):
        engine().run(bars, [])


def test_missing_scheduled_exit_close_cannot_be_replaced_with_a_future_price():
    bars = market()
    bars[3] = replace(bars[3], close_price=None)
    with pytest.raises(ValueError, match="session-3.*valuation close"):
        engine().run(bars, [signal(bars[0])])


def test_filled_daily_close_is_rejected_as_a_valuation_price():
    bars = market()
    bars[2] = replace(bars[2], close_observed=False)
    with pytest.raises(ValueError, match="valuation close"):
        engine().run(bars, [])


def test_future_price_changes_do_not_change_previous_equity_or_fills():
    bars = market()
    changed = bars.copy()
    for i in range(4, len(changed)):
        changed[i] = replace(changed[i], open_price=300.0, close_price=50.0)
    signals = [signal(bars[0]), signal(bars[4])]
    first = engine().run(bars, signals)
    second = engine().run(changed, signals)
    pd.testing.assert_frame_equal(first.ledger.iloc[:5], second.ledger.iloc[:5])
    cutoff = bars[3].close_timestamp
    pd.testing.assert_frame_equal(first.fills[first.fills.timestamp <= cutoff],
                                   second.fills[second.fills.timestamp <= cutoff])


def test_appending_future_sessions_does_not_change_previous_daily_records():
    bars = market()
    signals = [signal(bars[0])]
    short = engine(holding_sessions=5).run(bars[:3], signals)
    full = engine(holding_sessions=5).run(bars, signals)
    pd.testing.assert_frame_equal(short.ledger, full.ledger.iloc[:len(short.ledger)])
    cutoff = bars[2].close_timestamp
    pd.testing.assert_frame_equal(short.fills, full.fills[full.fills.timestamp <= cutoff])


def test_cash_equity_and_net_pnl_reconcile_for_every_session():
    bars = market()
    result = engine(fee_rate=0.01).run(bars, [signal(bars[0])])
    daily = result.ledger.iloc[1:]
    np.testing.assert_allclose(daily.equity, daily.cash + daily.position_quantity * daily.close_price)
    np.testing.assert_allclose(daily.equity,
                               1000 + daily.cumulative_realized_pnl + daily.unrealized_pnl)
    assert (daily.cash >= 0).all()
    assert (daily.position_quantity >= 0).all()


def test_end_of_data_retains_open_position_when_expiry_is_unavailable():
    bars = market()[:3]
    result = engine(holding_sessions=5).run(bars, [signal(bars[0])])
    assert result.open_position is not None
    assert result.open_position.quantity == 10
    assert result.open_position.entry_session_id == bars[1].session_id
    assert result.open_position.holding_sessions_remaining == 3
    assert result.fills["side"].tolist() == ["buy"]
    assert result.ledger.iloc[-1]["equity"] == 900
    assert result.ledger.iloc[-1]["realized_pnl"] == 0


def test_final_after_close_signal_remains_unfilled():
    bars = market()[:1]
    result = engine().run(bars, [signal(bars[0])])
    assert result.fills.empty
    assert "entry_unfilled_at_end" in result.events.event.values
    assert result.ledger.iloc[-1]["equity"] == 1000


def test_signals_during_holding_are_ignored_and_fresh_post_exit_signal_can_enter():
    bars = market([100] * 8)
    decisions = [signal(bars[0]), signal(bars[1], "flat"), signal(bars[2]), signal(bars[3])]
    result = engine().run(bars, decisions)
    assert result.fills.session_id.tolist() == ["session-1", "session-3", "session-4", "session-6"]
    assert result.events.query("event == 'signal_ignored'").signal_session_id.tolist() == [
        "session-1", "session-2"]


def test_unaffordable_opening_gap_rejects_frozen_quantity_instead_of_resizing():
    bars = market()
    bars[1] = replace(bars[1], open_price=110.0)
    result = engine().run(bars, [signal(bars[0])])
    assert result.fills.empty
    assert "unaffordable_open" in result.events.reason.values
    np.testing.assert_allclose(result.ledger.cash, 1000)


def test_one_session_horizon_executes_open_entry_and_pre_scheduled_close_exit():
    bars = market()
    result = engine(holding_sessions=1).run(bars, [signal(bars[0])])
    assert result.fills.timestamp.tolist() == [bars[1].open_timestamp, bars[1].close_timestamp]
    assert result.ledger.iloc[2]["equity"] == 1050
    assert result.open_position is None


def test_signal_identifiers_join_correctly_without_signal_row_order_assumptions():
    bars = market([100] * 8)
    decisions = [signal(bars[3]), signal(bars[0])]
    original = bars.copy()
    result = engine().run(bars, decisions)
    assert result.fills.session_id.tolist() == ["session-1", "session-3", "session-4", "session-6"]
    assert bars == original
    assert decisions[0].session_id == "session-3"


@pytest.mark.parametrize("case", ["unknown_id", "duplicate", "at_close", "at_open", "naive", "short"])
def test_invalid_signal_contracts_are_rejected(case):
    bars = market()
    decision = signal(bars[0])
    if case == "unknown_id":
        decisions = [replace(decision, session_id="absent")]
    elif case == "duplicate":
        decisions = [decision, decision]
    elif case == "at_close":
        decisions = [replace(decision, decision_timestamp=bars[0].close_timestamp)]
    elif case == "at_open":
        decisions = [replace(decision, decision_timestamp=bars[1].open_timestamp)]
    elif case == "naive":
        decisions = [replace(decision, decision_timestamp=decision.decision_timestamp.replace(tzinfo=None))]
    else:
        decisions = [replace(decision, direction="short")]
    with pytest.raises(ValueError, match="Signals"):
        engine().run(bars, decisions)


@pytest.mark.parametrize("case", ["empty", "duplicate", "out_of_order", "multiple_assets"])
def test_invalid_market_session_contracts_are_rejected(case):
    bars = market()
    if case == "empty":
        bars = []
    elif case == "duplicate":
        bars[1] = replace(bars[1], session_id=bars[0].session_id)
    elif case == "out_of_order":
        bars = bars[::-1]
    else:
        bars[1] = replace(bars[1], symbol="OTHER")
    with pytest.raises(ValueError):
        engine().run(bars, [])

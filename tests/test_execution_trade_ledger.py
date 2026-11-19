"""Independent completed/open trade references for Stage 2A cash accounting."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, MarketBar, Signal


def bars(closes=None, opens=None):
    """Create one observed equity stream on a deterministic synthetic UTC calendar."""
    closes = [100, 100, 110, 110, 90, 90, 100, 100] if closes is None else closes
    opens = closes if opens is None else opens
    days = pd.bdate_range("2020-01-06", periods=len(closes), tz="UTC")
    return [MarketBar(
        "SYNTH", f"day-{i}", day + pd.Timedelta(hours=9, minutes=30),
        day + pd.Timedelta(hours=16), float(opening), float(close),
        high=max(opening, close) + 1, low=min(opening, close) - 1, volume=1000 + i,
    ) for i, (day, opening, close) in enumerate(zip(days, opens, closes))]


def decision(bar, direction="long"):
    return Signal(bar.session_id, bar.close_timestamp + pd.Timedelta(minutes=1), direction)


def run(market, decisions=None, horizon=2, fee=0.01):
    config = ExecutionConfig("SYNTH", initial_capital=1000, holding_sessions=horizon, fee_rate=fee)
    return ExecutionEngineV1(config).run(market, decisions if decisions is not None else [decision(market[0])])


def test_completed_profitable_trade_matches_explicit_reference_example():
    market = bars()
    result = run(market)
    assert len(result.completed_trades) == 1
    assert result.open_trades.empty
    assert result.open_position is None
    trade = result.completed_trades.iloc[0]
    assert trade.status == "completed"
    assert trade.quantity == 9
    assert trade.entry_price == 100
    assert trade.exit_price == 110
    assert trade.entry_fee == 9
    assert trade.exit_fee == pytest.approx(9.9)
    assert trade.gross_realized_pnl == 90
    assert trade.net_realized_pnl == pytest.approx(71.1)
    assert trade.net_pnl_to_date == pytest.approx(71.1)
    assert trade.unrealized_pnl == 0
    assert trade.entry_timestamp == market[1].open_timestamp
    assert trade.exit_timestamp == market[2].close_timestamp
    assert result.ledger.iloc[-1].equity == pytest.approx(1071.1)


def test_completed_losing_trade_has_correct_gross_and_net_losses():
    result = run(bars([100, 100, 90, 90]))
    trade = result.completed_trades.iloc[0]
    assert trade.gross_realized_pnl == -90
    assert trade.entry_fee == 9
    assert trade.exit_fee == pytest.approx(8.1)
    assert trade.net_realized_pnl == pytest.approx(-107.1)
    assert result.ledger.iloc[-1].equity == pytest.approx(892.9)


def test_open_trade_has_gross_unrealized_pnl_and_separate_incurred_entry_fee():
    market = bars([100, 100, 110])
    result = run(market, horizon=5)
    assert result.completed_trades.empty
    assert len(result.open_trades) == 1
    trade = result.open_trades.iloc[0]
    assert trade.status == "open"
    assert trade.quantity == 9
    assert trade.unrealized_pnl == 90
    assert trade.entry_fee == 9
    assert trade.exit_fee == 0
    assert trade.gross_realized_pnl == 0
    assert trade.net_realized_pnl == -9
    assert trade.net_pnl_to_date == 81
    assert trade.holding_sessions == 2
    assert trade.valuation_price == 110
    assert trade.valuation_timestamp == market[-1].close_timestamp
    assert trade.exit_timestamp is None
    assert trade.exit_price is None
    assert trade.exit_fill_id is None
    assert result.fills.side.tolist() == ["buy"]
    assert result.open_position.trade_id == trade.trade_id
    assert result.ledger.iloc[-1].cash == 91
    assert result.ledger.iloc[-1].equity == 1081


def test_daily_fees_are_realized_when_incurred_without_double_counting():
    result = run(bars())
    daily = result.ledger
    entry, exit_row = daily.iloc[2], daily.iloc[3]
    assert entry.realized_costs == 9
    assert entry.net_realized_pnl == -9
    assert entry.cumulative_net_realized_pnl == -9
    assert entry.gross_unrealized_pnl == 0
    assert exit_row.realized_costs == pytest.approx(9.9)
    assert exit_row.gross_realized_pnl == 90
    assert exit_row.net_realized_pnl == pytest.approx(80.1)
    assert exit_row.cumulative_net_realized_pnl == pytest.approx(71.1)
    assert daily.realized_costs.sum() == pytest.approx(18.9)
    assert daily.net_realized_pnl.sum() == pytest.approx(71.1)
    # Original Stage 1 columns retain their documented compatibility semantics.
    assert entry.unrealized_pnl == -9
    assert exit_row.realized_pnl == pytest.approx(71.1)


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_holding_duration_counts_actual_held_sessions_inclusively(horizon):
    market = bars([100] * 8)
    trade = run(market, horizon=horizon).completed_trades.iloc[0]
    assert trade.holding_sessions == horizon
    assert trade.entry_session_id == market[1].session_id
    assert trade.exit_session_id == market[horizon].session_id


def test_fill_ids_link_each_trade_to_exactly_one_entry_and_one_exit():
    result = run(bars())
    trade = result.completed_trades.iloc[0]
    fills = result.fills.set_index("fill_id")
    assert fills.index.is_unique
    entry = fills.loc[trade.entry_fill_id]
    exit_fill = fills.loc[trade.exit_fill_id]
    assert entry.trade_id == exit_fill.trade_id == trade.trade_id
    assert entry.side == "buy" and exit_fill.side == "sell"
    assert entry.quantity == exit_fill.quantity == trade.quantity
    assert entry.price == trade.entry_price and exit_fill.price == trade.exit_price
    assert entry.timestamp == trade.entry_timestamp and exit_fill.timestamp == trade.exit_timestamp
    assert entry.fee == trade.entry_fee and exit_fill.fee == trade.exit_fee
    linked = result.events[result.events.trade_id == trade.trade_id]
    assert linked.event.tolist() == ["entry_filled", "scheduled_exit_filled"]


def test_trade_identifiers_are_stable_across_repeated_runs_and_signal_order():
    market = bars()
    signals = [decision(market[0]), decision(market[2])]
    first = run(market, signals)
    repeated = run(market, signals[::-1])
    pd.testing.assert_frame_equal(first.completed_trades, repeated.completed_trades)
    pd.testing.assert_frame_equal(first.fills, repeated.fills)
    assert first.completed_trades.trade_id.is_unique


def test_multiple_consecutive_trades_reconcile_independent_cash_reference():
    market = bars()
    result = run(market, [decision(market[0]), decision(market[2])])
    assert len(result.completed_trades) == 2
    first, second = result.completed_trades.iloc[0], result.completed_trades.iloc[1]
    assert first.net_realized_pnl == pytest.approx(71.1)
    # Second entry: 9 at 110, exit: 9 at 90; fees 9.90 and 8.10.
    assert second.quantity == 9
    assert second.gross_realized_pnl == -180
    assert second.net_realized_pnl == pytest.approx(-198)
    assert first.exit_timestamp < second.entry_timestamp
    assert result.ledger.iloc[-1].cash == pytest.approx(873.1)
    assert result.ledger.iloc[-1].equity == pytest.approx(873.1)
    assert result.fills.fee.sum() == pytest.approx(36.9)


def test_flat_periods_have_empty_trade_tables_with_explicit_schema():
    market = bars()
    result = run(market, [decision(bar, "flat") for bar in market])
    assert result.completed_trades.empty and result.open_trades.empty
    assert result.completed_trades.columns.equals(result.open_trades.columns)
    assert {"trade_id", "entry_fill_id", "exit_fill_id", "net_realized_pnl"}.issubset(
        result.completed_trades.columns)
    assert result.fills.empty
    np.testing.assert_allclose(result.ledger.equity, 1000)


def test_trade_records_fills_and_daily_accounting_reconcile_at_every_prefix():
    market = bars()
    signals = [decision(market[0]), decision(market[2])]
    for length in range(1, len(market) + 1):
        prefix = market[:length]
        available = [item for item in signals if item.session_id in {bar.session_id for bar in prefix}]
        result = run(prefix, available)
        final = result.ledger.iloc[-1]
        open_gross = result.open_trades.unrealized_pnl.sum()
        trade_net = result.completed_trades.net_realized_pnl.sum() + result.open_trades.net_realized_pnl.sum()
        assert final.equity == pytest.approx(1000 + trade_net + open_gross)
        assert final.equity == pytest.approx(final.cash + final.position_quantity * final.close_price)
        assert final.cumulative_realized_costs == pytest.approx(result.fills.fee.sum())
        assert final.cumulative_net_realized_pnl == pytest.approx(trade_net)
        assert final.gross_unrealized_pnl == pytest.approx(open_gross)
        np.testing.assert_allclose(
            result.ledger.equity,
            1000 + result.ledger.cumulative_net_realized_pnl + result.ledger.gross_unrealized_pnl,
        )


def test_future_prices_cannot_change_earlier_completed_trade_records():
    market = bars()
    changed = market.copy()
    for index in range(3, len(changed)):
        changed[index] = replace(changed[index], open_price=80.0, close_price=70.0)
    first, second = run(market), run(changed)
    pd.testing.assert_frame_equal(first.completed_trades, second.completed_trades)
    pd.testing.assert_frame_equal(first.fills, second.fills)
    pd.testing.assert_frame_equal(first.ledger.iloc[:4], second.ledger.iloc[:4])


def test_extending_data_preserves_open_entry_identity_and_earlier_records():
    market = bars([100] * 8)
    prefix = run(market[:3], horizon=5)
    extended = run(market, horizon=5)
    opened = prefix.open_trades.iloc[0]
    completed = extended.completed_trades.iloc[0]
    for name in ["trade_id", "entry_fill_id", "entry_timestamp", "entry_price", "quantity", "entry_fee"]:
        assert opened[name] == completed[name]
    assert prefix.completed_trades.empty
    assert prefix.open_trades.iloc[0].exit_timestamp is None
    pd.testing.assert_frame_equal(prefix.ledger, extended.ledger.iloc[:len(prefix.ledger)])
    pd.testing.assert_frame_equal(prefix.fills, extended.fills.iloc[:1])


@pytest.mark.parametrize("reason", ["invalid_open", "unaffordable_open", "last_signal"])
def test_unfilled_orders_never_create_trade_records(reason):
    market = bars()
    signals = [decision(market[0])]
    if reason == "invalid_open":
        market[1] = replace(market[1], open_price=None)
    elif reason == "unaffordable_open":
        market[1] = replace(market[1], open_price=1000.0)
    else:
        signals = [decision(market[-1])]
    result = run(market, signals)
    assert result.completed_trades.empty and result.open_trades.empty
    assert result.fills.empty


def test_zero_gross_pnl_trade_loses_only_its_actual_fees():
    result = run(bars([100] * 4))
    trade = result.completed_trades.iloc[0]
    assert trade.gross_realized_pnl == 0
    assert trade.net_realized_pnl == -18
    assert result.ledger.iloc[-1].equity == 982


def test_fee_free_trade_pnl_and_equity_remain_stage_one_compatible():
    result = run(bars(), fee=0)
    trade = result.completed_trades.iloc[0]
    assert trade.quantity == 10
    assert trade.gross_realized_pnl == trade.net_realized_pnl == 100
    np.testing.assert_allclose(result.ledger.equity.iloc[:4], [1000, 1000, 1000, 1100])
    assert result.ledger.iloc[-1].equity == 1100

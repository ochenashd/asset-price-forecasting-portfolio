"""Independent cash and cost references for adverse proportional slippage."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, MarketBar, Signal


def bars(closes=None, opens=None):
    closes = [100, 100, 110, 110, 90, 90, 100, 100] if closes is None else closes
    opens = closes if opens is None else opens
    days = pd.bdate_range("2020-01-06", periods=len(closes), tz="UTC")
    return [MarketBar(
        "SYNTH", f"day-{i}", day + pd.Timedelta(hours=9, minutes=30),
        day + pd.Timedelta(hours=16), float(opening), float(close),
        high=max(opening, close) + 1, low=min(opening, close) - 1, volume=1000 + i,
    ) for i, (day, opening, close) in enumerate(zip(days, opens, closes))]


def decision(bar):
    return Signal(bar.session_id, bar.close_timestamp + pd.Timedelta(minutes=1), "long")


def run(market=None, signals=None, slip=0.01, fee=0.01, horizon=2):
    market = bars() if market is None else market
    signals = [decision(market[0])] if signals is None else signals
    config = ExecutionConfig("SYNTH", initial_capital=1000, fee_rate=fee,
                             holding_sessions=horizon, slippage_rate=slip)
    return ExecutionEngineV1(config).run(market, signals)


def test_zero_slippage_reproduces_stage_two_a_cash_and_trade_reference():
    result = run(slip=0)
    trade = result.completed_trades.iloc[0]
    assert trade.quantity == 9
    assert trade.entry_price == trade.entry_reference_price == 100
    assert trade.exit_price == trade.exit_reference_price == 110
    assert trade.entry_fee == 9
    assert trade.exit_fee == pytest.approx(9.9)
    assert trade.gross_realized_pnl == trade.reference_gross_realized_pnl == 90
    assert trade.net_realized_pnl == pytest.approx(71.1)
    assert result.ledger.iloc[-1].equity == pytest.approx(1071.1)
    np.testing.assert_allclose(result.ledger.slippage_costs, 0)
    np.testing.assert_allclose(result.fills.reference_price, result.fills.executed_price)


@pytest.mark.parametrize("rate", [0.001, 0.01, 0.05])
def test_buy_and_sell_fills_are_adverse_relative_to_observed_reference_prices(rate):
    result = run(slip=rate)
    buy, sell = result.fills.iloc[0], result.fills.iloc[1]
    expected_buy = Decimal("100") * (1 + Decimal(str(rate)))
    expected_sell = Decimal("110") * (1 - Decimal(str(rate)))
    assert buy.executed_price == float(expected_buy)
    assert sell.executed_price == float(expected_sell)
    assert buy.executed_price > buy.reference_price
    assert sell.executed_price < sell.reference_price
    assert buy.price == buy.executed_price and sell.price == sell.executed_price


def test_cash_and_commissions_match_independent_decimal_financial_example():
    result = run()
    trade = result.completed_trades.iloc[0]
    # floor(1000 / (100 * 1.01 * 1.01)) = 9, fixed before the next open.
    entry_notional = Decimal(9) * Decimal("101")
    entry_fee = entry_notional * Decimal("0.01")
    entry_cash = Decimal(1000) - entry_notional - entry_fee
    exit_notional = Decimal(9) * Decimal("108.9")
    exit_fee = exit_notional * Decimal("0.01")
    final_cash = entry_cash + exit_notional - exit_fee
    assert entry_cash == Decimal("81.91")
    assert final_cash == Decimal("1052.209")
    assert trade.quantity == 9
    assert trade.entry_fee == float(entry_fee) == 9.09
    assert trade.exit_fee == float(exit_fee) == 9.801
    assert result.ledger.iloc[2].cash == float(entry_cash)
    assert result.ledger.iloc[-1].cash == result.ledger.iloc[-1].equity == float(final_cash)


def test_completed_trade_executed_and_reference_pnl_reconcile_without_double_counting():
    trade = run().completed_trades.iloc[0]
    assert trade.reference_gross_realized_pnl == 90
    assert trade.gross_realized_pnl == pytest.approx(71.1)
    assert trade.entry_slippage_cost == 9
    assert trade.exit_slippage_cost == pytest.approx(9.9)
    assert trade.slippage_costs == pytest.approx(18.9)
    assert trade.commission_costs == pytest.approx(18.891)
    assert trade.total_execution_costs == pytest.approx(37.791)
    assert trade.net_realized_pnl == pytest.approx(52.209)
    assert trade.net_realized_pnl == pytest.approx(trade.gross_realized_pnl - trade.commission_costs)
    assert trade.net_realized_pnl == pytest.approx(
        trade.reference_gross_realized_pnl - trade.commission_costs - trade.slippage_costs)
    assert trade.reference_net_realized_pnl == trade.net_realized_pnl


def test_open_trade_keeps_commissions_and_slippage_in_separate_accounting_views():
    result = run(bars([100, 100, 110]), horizon=5)
    trade = result.open_trades.iloc[0]
    assert result.completed_trades.empty
    assert trade.entry_price == 101 and trade.entry_reference_price == 100
    assert trade.unrealized_pnl == 81
    assert trade.reference_unrealized_pnl == 90
    assert trade.net_realized_pnl == -9.09
    assert trade.reference_net_realized_pnl == -18.09
    assert trade.net_pnl_to_date == pytest.approx(71.91)
    assert trade.entry_slippage_cost == trade.slippage_costs == 9
    assert trade.exit_slippage_cost == 0 and trade.exit_fee == 0
    assert trade.exit_price is None and trade.exit_reference_price is None
    assert result.open_position.entry_price == 101
    assert result.open_position.entry_reference_price == 100
    assert result.open_position.entry_slippage_cost == 9
    assert result.ledger.iloc[-1].cash == 81.91
    assert result.ledger.iloc[-1].equity == pytest.approx(1071.91)


def test_daily_equity_contains_slippage_exactly_once_through_fills():
    result = run()
    entry = result.ledger.iloc[2]
    assert entry.equity == pytest.approx(981.91)
    assert entry.cumulative_realized_costs == 9.09
    assert entry.cumulative_slippage_costs == 9
    assert entry.cumulative_net_realized_pnl == -9.09
    assert entry.gross_unrealized_pnl == -9
    assert entry.cumulative_reference_net_realized_pnl == -18.09
    assert entry.reference_unrealized_pnl == 0
    assert result.ledger.iloc[-1].equity == pytest.approx(1000 + 90 - 18.891 - 18.9)
    assert result.ledger.iloc[-1].equity != pytest.approx(1000 + 90 - 18.891 - 2 * 18.9)


def test_fills_trade_records_and_daily_cost_columns_reconcile():
    result = run()
    trade = result.completed_trades.iloc[0]
    linked = result.fills[result.fills.trade_id == trade.trade_id]
    assert linked.commission_cost.sum() == pytest.approx(trade.commission_costs)
    assert linked.slippage_cost.sum() == pytest.approx(trade.slippage_costs)
    np.testing.assert_allclose(linked.commission_cost, linked.quantity * linked.executed_price * 0.01)
    signs = np.where(linked.side == "buy", 1, -1)
    np.testing.assert_allclose(linked.slippage_cost,
                               signs * linked.quantity * (linked.executed_price - linked.reference_price))
    np.testing.assert_allclose(result.ledger.commission_costs, result.ledger.transaction_fees)
    np.testing.assert_allclose(result.ledger.total_execution_costs,
                               result.ledger.commission_costs + result.ledger.slippage_costs)
    assert result.ledger.cumulative_total_execution_costs.iloc[-1] == pytest.approx(37.791)


def test_adverse_slippage_can_make_an_otherwise_affordable_open_unaffordable():
    market = bars([100] * 5)
    market[1] = replace(market[1], open_price=110)
    result = run(market, slip=0.02, fee=0)
    scheduled = result.events.query("event == 'entry_scheduled'").iloc[0]
    assert scheduled.quantity == 9
    assert 9 * 110 <= 1000
    assert Decimal(9) * Decimal("110") * Decimal("1.02") > 1000
    assert result.fills.empty
    assert result.completed_trades.empty and result.open_trades.empty
    assert "unaffordable_open" in result.events.reason.values
    np.testing.assert_allclose(result.ledger.equity, 1000)
    np.testing.assert_allclose(result.ledger.total_execution_costs, 0)


def test_multiple_trades_reconcile_independent_cash_and_cost_totals():
    market = bars()
    result = run(market, [decision(market[0]), decision(market[2])])
    assert len(result.completed_trades) == 2
    np.testing.assert_allclose(result.completed_trades.quantity, [9, 9])
    np.testing.assert_allclose(result.completed_trades.net_realized_pnl, [52.209, -216.018])
    assert result.completed_trades.reference_gross_realized_pnl.sum() == -90
    assert result.fills.slippage_cost.sum() == pytest.approx(36.9)
    assert result.fills.commission_cost.sum() == pytest.approx(36.909)
    assert result.ledger.iloc[-1].cash == pytest.approx(836.191)
    assert result.ledger.iloc[-1].equity == pytest.approx(836.191)


def test_equity_reconciles_in_both_accounting_views_at_every_data_prefix():
    market = bars()
    signals = [decision(market[0]), decision(market[2])]
    for length in range(1, len(market) + 1):
        available = {bar.session_id for bar in market[:length]}
        result = run(market[:length], [item for item in signals if item.session_id in available])
        daily = result.ledger.iloc[1:]
        np.testing.assert_allclose(daily.equity, daily.cash + daily.position_quantity * daily.close_price)
        np.testing.assert_allclose(daily.equity,
                                   1000 + daily.cumulative_net_realized_pnl + daily.gross_unrealized_pnl)
        np.testing.assert_allclose(daily.equity,
                                   1000 + daily.cumulative_reference_net_realized_pnl + daily.reference_unrealized_pnl)
        np.testing.assert_allclose(daily.equity,
                                   1000 + daily.cumulative_reference_gross_realized_pnl
                                   + daily.reference_unrealized_pnl - daily.cumulative_realized_costs
                                   - daily.cumulative_slippage_costs)
        final = daily.iloc[-1]
        assert final.cumulative_slippage_costs == pytest.approx(result.fills.slippage_cost.sum())
        assert final.cumulative_realized_costs == pytest.approx(result.fills.fee.sum())
        trade_net = result.completed_trades.net_realized_pnl.sum() + result.open_trades.net_pnl_to_date.sum()
        assert final.equity == pytest.approx(1000 + trade_net)


def test_future_prices_leave_earlier_fills_costs_and_completed_trades_unchanged():
    market = bars()
    changed = market.copy()
    for index in range(3, len(changed)):
        changed[index] = replace(changed[index], open_price=200, close_price=50)
    first, second = run(market), run(changed)
    pd.testing.assert_frame_equal(first.fills, second.fills)
    pd.testing.assert_frame_equal(first.completed_trades, second.completed_trades)
    pd.testing.assert_frame_equal(first.ledger.iloc[:4], second.ledger.iloc[:4])


def test_appending_future_prices_does_not_rewrite_past_cost_attribution():
    market = bars()
    short = run(market[:3], horizon=5)
    full = run(market, horizon=5)
    pd.testing.assert_frame_equal(short.ledger, full.ledger.iloc[:len(short.ledger)])
    pd.testing.assert_frame_equal(short.fills, full.fills.iloc[:1])
    assert short.open_trades.iloc[0].trade_id == full.completed_trades.iloc[0].trade_id


def test_one_session_trade_charges_each_fill_once_on_the_same_ledger_day():
    result = run(bars([100, 110, 110], [100, 100, 110]), horizon=1)
    assert len(result.fills) == 2
    assert result.ledger.iloc[2].transaction_fees == pytest.approx(18.891)
    assert result.ledger.iloc[2].slippage_costs == pytest.approx(18.9)
    assert result.ledger.iloc[2].equity == pytest.approx(1052.209)


@pytest.mark.parametrize("rate", [-0.01, 1.0, 1.1, float("nan"), float("inf"), True, "0.01"])
def test_invalid_slippage_configuration_is_rejected(rate):
    with pytest.raises(ValueError, match="slippage_rate"):
        ExecutionEngineV1(ExecutionConfig("SYNTH", slippage_rate=rate))

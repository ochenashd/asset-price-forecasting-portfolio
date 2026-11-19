"""Synthetic signed-share, collateral and margin-stop accounting references."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, MarketBar, Signal
from src.evaluation.execution.portfolio import Portfolio


def bars(closes=None, opens=None):
    closes = [100, 100, 80, 80, 80, 80, 80, 80] if closes is None else closes
    opens = [100] * len(closes) if opens is None else opens
    days = pd.bdate_range("2020-01-06", periods=len(closes), tz="UTC")
    return [MarketBar(
        "SYNTH", f"day-{i}", day + pd.Timedelta(hours=9, minutes=30),
        day + pd.Timedelta(hours=16), float(opening), float(close),
        high=max(opening, close) + 1, low=min(opening, close) - 1, volume=1000 + i,
    ) for i, (day, opening, close) in enumerate(zip(days, opens, closes))]


def decision(bar, label=0):
    return Signal.from_class(bar.session_id, bar.close_timestamp + pd.Timedelta(minutes=1), label)


def run(market=None, signals=None, **kwargs):
    market = bars() if market is None else market
    config = ExecutionConfig(
        "SYNTH", initial_capital=1000, holding_sessions=kwargs.pop("holding_sessions", 2),
        fee_rate=kwargs.pop("fee_rate", 0.01), slippage_rate=kwargs.pop("slippage_rate", 0.01),
        allow_short=kwargs.pop("allow_short", True), **kwargs,
    )
    return ExecutionEngineV1(config).run(market, signals if signals is not None else [decision(market[0])])


def assert_identity(record, price_field):
    """Reconcile either a daily record or an opening risk snapshot."""
    mark = getattr(record, price_field)
    assert record.equity == pytest.approx(record.cash + record.position_quantity * mark)
    assert record.equity == pytest.approx(
        1000 + record.cumulative_net_realized_pnl + record.gross_unrealized_pnl)
    assert record.equity == pytest.approx(
        1000 + record.cumulative_reference_net_realized_pnl + record.reference_unrealized_pnl)
    assert record.unrestricted_cash == pytest.approx(record.cash - record.restricted_proceeds)


def test_profitable_short_matches_independently_calculated_reference():
    result = run()
    trade = result.completed_trades.iloc[0]
    assert result.run_status == "completed" and result.terminal_snapshot is None
    assert trade.direction == "short" and trade.quantity == -9
    assert trade.entry_reference_price == 100
    assert trade.entry_price == 99
    assert trade.exit_reference_price == 80
    assert trade.exit_price == pytest.approx(80.8)
    assert trade.entry_fee == pytest.approx(8.91)
    assert trade.exit_fee == pytest.approx(7.272)
    # Cash: 1000 + 9*99 - 8.91 - 9*80.8 - 7.272 = 1147.618.
    expected = Decimal(1000) + Decimal(9) * Decimal(99) - Decimal("8.91")
    expected -= Decimal(9) * Decimal("80.8") + Decimal("7.272")
    assert expected == Decimal("1147.618")
    assert result.ledger.iloc[-1].cash == result.ledger.iloc[-1].equity == float(expected)
    assert trade.net_realized_pnl == pytest.approx(147.618)
    assert result.open_trades.empty and result.open_position is None


def test_losing_short_pnl_uses_signed_position_and_actual_cover_cost():
    result = run(bars([100, 100, 120, 120]))
    trade = result.completed_trades.iloc[0]
    assert trade.gross_realized_pnl == pytest.approx(-199.8)
    assert trade.entry_fee == pytest.approx(8.91)
    assert trade.exit_fee == pytest.approx(10.908)
    assert trade.net_realized_pnl == pytest.approx(-219.618)
    assert result.ledger.iloc[-1].equity == pytest.approx(780.382)


def test_short_entry_sell_and_cover_buy_are_adversely_slipped():
    market = bars()
    result = run(market)
    assert result.fills.side.tolist() == ["sell", "buy"]
    np.testing.assert_array_equal(result.fills.quantity, [9, 9])
    np.testing.assert_array_equal(result.fills.signed_quantity, [-9, 9])
    assert result.fills.timestamp.tolist() == [market[1].open_timestamp, market[2].close_timestamp]
    np.testing.assert_allclose(result.fills.executed_price, [99, 80.8])
    np.testing.assert_allclose(result.fills.slippage_cost, [9, 7.2])
    np.testing.assert_allclose(result.fills.commission_cost,
                               result.fills.quantity * result.fills.executed_price * 0.01)


def test_restricted_proceeds_are_part_of_total_cash_not_extra_equity():
    entry = run().ledger.iloc[2]
    assert entry.cash == pytest.approx(1882.09)
    assert entry.restricted_proceeds == 891
    assert entry.unrestricted_cash == pytest.approx(991.09)
    assert entry.short_liability == 900
    assert entry.equity == pytest.approx(982.09)
    assert entry.initial_margin_requirement == 450
    assert entry.maintenance_requirement == 270
    assert entry.margin_excess == pytest.approx(712.09)
    assert entry.available_collateral == pytest.approx(532.09)
    assert_identity(entry, "close_price")


def test_actual_cover_releases_restricted_proceeds_only_after_the_exit_fill():
    result = run()
    assert result.ledger.iloc[2].restricted_proceeds == 891
    assert result.ledger.iloc[3].restricted_proceeds == 0
    assert result.ledger.iloc[3].position_quantity == 0
    assert result.ledger.iloc[3].short_liability == 0
    assert result.ledger.iloc[3].unrestricted_cash == pytest.approx(1147.618)


def test_restricted_proceeds_cannot_finance_an_unrelated_long_entry():
    portfolio = Portfolio(Decimal(1000))
    portfolio.enter_short(9, Decimal(99), 0.01, reference_price=100,
                          max_exposure=1, initial_margin=0.5)
    with pytest.raises(ValueError, match="flat portfolio"):
        portfolio.enter(1, 100, 0)
    with pytest.raises(ValueError, match="flat portfolio"):
        portfolio.size_short_order(100, 1, 0.01, 0.01, 1, 0.5)
    assert portfolio.restricted_proceeds == 891


def test_adverse_entry_gap_rejects_frozen_quantity_using_post_cost_exposure():
    market = bars()
    market[1] = replace(market[1], open_price=110)
    result = run(market)
    assert result.events.query("event == 'entry_scheduled'").iloc[0].quantity == 9
    assert result.fills.empty
    assert "short_exposure_cap" in result.events.reason.values
    np.testing.assert_allclose(result.ledger.equity, 1000)
    np.testing.assert_allclose(result.ledger.restricted_proceeds, 0)


def test_initial_margin_can_bind_before_the_exposure_cap():
    market = bars()
    market[1] = replace(market[1], open_price=110)
    result = run(market, initial_margin_fraction=1.5)
    assert result.events.query("event == 'entry_scheduled'").iloc[0].quantity == 6
    # At the open, liability 660 < equity 986.866, but initial margin 990 > equity.
    assert result.fills.empty
    assert "insufficient_initial_margin" in result.events.reason.values


def test_configured_initial_exposure_cap_limits_whole_share_quantity():
    result = run(max_short_exposure=0.5)
    entry = result.ledger.iloc[2]
    assert entry.position_quantity == -4
    assert entry.short_liability <= 0.5 * entry.equity


def test_opening_margin_breach_stops_before_current_close_without_cover():
    market = bars([100] * 8)
    market[2] = replace(market[2], open_price=161, close_price=10)
    result = run(market, holding_sessions=5)
    assert result.run_status == "margin_breach"
    risk = result.terminal_snapshot
    assert risk.phase == "open" and risk.timestamp == market[2].open_timestamp
    assert risk.reference_price == 161
    assert risk.equity == pytest.approx(433.09)
    assert risk.short_liability == 1449
    assert risk.maintenance_requirement == pytest.approx(434.7)
    assert risk.margin_excess == pytest.approx(-1.61)
    assert result.ledger.timestamp.iloc[-1] == market[1].close_timestamp
    assert result.fills.side.tolist() == ["sell"]
    assert result.completed_trades.empty
    assert result.open_trades.iloc[0].valuation_phase == "open"
    assert result.open_trades.iloc[0].holding_sessions == 1
    assert result.open_position.holding_sessions_remaining == 4
    assert_identity(risk, "reference_price")


def test_closing_margin_breach_preserves_daily_close_and_stops_later_events():
    market = bars([100] * 8)
    market[2] = replace(market[2], close_price=161)
    result = run(market, holding_sessions=5)
    assert result.run_status == "margin_breach"
    assert result.terminal_snapshot.phase == "close"
    assert result.ledger.timestamp.iloc[-1] == market[2].close_timestamp
    assert len(result.events.query("event == 'margin_breach'")) == 1
    assert result.events.timestamp.max() == market[2].close_timestamp
    assert result.open_trades.iloc[0].holding_sessions == 2
    assert result.open_trades.iloc[0].exit_timestamp is None
    assert result.fills.side.tolist() == ["sell"]
    assert_identity(result.ledger.iloc[-1], "close_price")


def test_exact_maintenance_boundary_is_not_a_breach():
    market = bars([100, 100, 160], [100, 100, 160])
    result = run(market, holding_sessions=5, fee_rate=0, slippage_rate=0,
                 maintenance_margin_fraction=0.25)
    final = result.ledger.iloc[-1]
    assert final.equity == final.maintenance_requirement == 400
    assert final.margin_excess == 0
    assert result.run_status == "completed" and result.terminal_snapshot is None
    assert result.open_position.quantity == -10


def test_opening_breach_on_expiry_day_does_not_invent_the_future_scheduled_cover():
    market = bars([100] * 4)
    market[2] = replace(market[2], open_price=161)
    result = run(market)
    assert result.run_status == "margin_breach"
    assert result.fills.side.tolist() == ["sell"]
    assert result.completed_trades.empty


def test_pre_scheduled_cover_executes_before_closing_margin_assessment():
    market = bars([100, 100, 161, 161])
    result = run(market)
    assert result.run_status == "completed"
    assert result.fills.side.tolist() == ["sell", "buy"]
    assert len(result.completed_trades) == 1
    assert result.open_trades.empty
    expected = Decimal("1882.09") - 9 * Decimal("162.61") * Decimal("1.01")
    assert expected == Decimal("403.9651")
    assert result.ledger.iloc[-1].equity == float(expected)


def test_actual_scheduled_cover_can_record_insolvency_without_capital_injection():
    market = bars([100, 100, 250, 100])
    result = run(market)
    assert result.run_status == "insolvent"
    assert result.fills.side.tolist() == ["sell", "buy"]
    assert len(result.completed_trades) == 1
    assert result.open_trades.empty and result.open_position is None
    assert result.ledger.iloc[-1].equity == pytest.approx(-413.135)
    assert result.ledger.iloc[-1].cash == pytest.approx(-413.135)
    assert result.ledger.iloc[-1].restricted_proceeds == 0
    assert result.terminal_snapshot.trade_id == result.completed_trades.iloc[0].trade_id
    assert_identity(result.terminal_snapshot, "reference_price")


@pytest.mark.parametrize("phase", ["open", "close"])
def test_uncovered_insolvency_preserves_the_short_liability(phase):
    market = bars([100] * 6)
    market[2] = replace(market[2], **{f"{phase}_price": 250})
    result = run(market, holding_sessions=5)
    assert result.run_status == "insolvent"
    assert result.terminal_snapshot.equity == pytest.approx(-367.91)
    assert result.terminal_snapshot.position_quantity == -9
    assert result.terminal_snapshot.restricted_proceeds == 891
    assert result.fills.side.tolist() == ["sell"]
    assert result.completed_trades.empty and len(result.open_trades) == 1
    assert_identity(result.terminal_snapshot, "reference_price")


def test_open_short_at_end_has_separate_incurred_costs_and_marked_liability():
    result = run(bars([100, 100, 80]), holding_sessions=5)
    trade = result.open_trades.iloc[0]
    assert result.completed_trades.empty and trade.quantity == -9
    assert trade.unrealized_pnl == 171
    assert trade.reference_unrealized_pnl == 180
    assert trade.net_realized_pnl == pytest.approx(-8.91)
    assert trade.reference_net_realized_pnl == pytest.approx(-17.91)
    assert trade.net_pnl_to_date == pytest.approx(162.09)
    assert trade.exit_timestamp is None and trade.exit_fee == 0
    assert result.ledger.iloc[-1].equity == pytest.approx(1162.09)
    assert result.open_position.holding_sessions_remaining == 3


def test_reference_executed_and_net_trade_pnl_reconcile_with_fill_costs():
    result = run()
    trade = result.completed_trades.iloc[0]
    assert trade.reference_gross_realized_pnl == 180
    assert trade.gross_realized_pnl == pytest.approx(163.8)
    assert trade.slippage_costs == pytest.approx(16.2)
    assert trade.commission_costs == pytest.approx(16.182)
    assert trade.net_realized_pnl == pytest.approx(
        trade.reference_gross_realized_pnl - trade.slippage_costs - trade.commission_costs)
    assert trade.net_realized_pnl == pytest.approx(trade.gross_realized_pnl - trade.commission_costs)
    np.testing.assert_allclose(result.fills.slippage_cost,
                               result.fills.signed_quantity * (result.fills.price - result.fills.reference_price))


def test_equity_and_costs_reconcile_at_every_completed_or_open_prefix():
    market = bars([100] * 8)
    signals = [decision(market[0]), decision(market[2])]
    for length in range(1, len(market) + 1):
        prefix = market[:length]
        available = {bar.session_id for bar in prefix}
        result = run(prefix, [item for item in signals if item.session_id in available])
        for _, row in result.ledger.iloc[1:].iterrows():
            assert_identity(row, "close_price")
        final = result.ledger.iloc[-1]
        trade_net = result.completed_trades.net_realized_pnl.sum() + result.open_trades.net_pnl_to_date.sum()
        assert final.equity == pytest.approx(1000 + trade_net)
        assert final.cumulative_realized_costs == pytest.approx(result.fills.commission_cost.sum())
        assert final.cumulative_slippage_costs == pytest.approx(result.fills.slippage_cost.sum())


def test_multiple_short_trades_have_distinct_ids_and_no_pyramiding():
    market = bars([100] * 8)
    result = run(market, [decision(market[0]), decision(market[2])])
    assert result.completed_trades.trade_id.is_unique
    assert len(result.completed_trades) == 2
    np.testing.assert_allclose(result.completed_trades.net_realized_pnl, [-36, -36])
    assert result.ledger.iloc[-1].equity == 928
    assert result.completed_trades.iloc[0].exit_timestamp < result.completed_trades.iloc[1].entry_timestamp


def test_long_and_flat_signals_are_ignored_during_active_short():
    market = bars([100] * 8)
    result = run(market, [decision(market[0]), decision(market[1], 2), decision(market[2], 1)],
                 holding_sessions=4)
    assert result.fills.side.tolist() == ["sell", "buy"]
    assert result.fills.timestamp.iloc[-1] == market[4].close_timestamp
    assert len(result.events.query("event == 'signal_ignored'")) == 2


def test_future_price_changes_cannot_rewrite_a_completed_short_trade():
    market = bars()
    changed = market.copy()
    for index in range(3, len(changed)):
        changed[index] = replace(changed[index], open_price=250, close_price=50)
    original, modified = run(market), run(changed)
    pd.testing.assert_frame_equal(original.fills, modified.fills)
    pd.testing.assert_frame_equal(original.completed_trades, modified.completed_trades)
    pd.testing.assert_frame_equal(original.ledger.iloc[:4], modified.ledger.iloc[:4])


def test_margin_stop_is_invariant_to_all_future_prices_including_missing_values():
    market = bars([100] * 8)
    market[2] = replace(market[2], open_price=161)
    changed = market.copy()
    for index in range(3, len(changed)):
        changed[index] = replace(changed[index], open_price=None, close_price=None)
    original, modified = run(market, holding_sessions=5), run(changed, holding_sessions=5)
    assert original.terminal_snapshot == modified.terminal_snapshot
    assert original.run_status == modified.run_status == "margin_breach"
    pd.testing.assert_frame_equal(original.ledger, modified.ledger)
    pd.testing.assert_frame_equal(original.fills, modified.fills)
    pd.testing.assert_frame_equal(original.open_trades, modified.open_trades)


@pytest.mark.parametrize("phase", ["entry", "open_risk", "close", "cover"])
def test_missing_prices_never_produce_a_synthetic_short_fill_or_mark(phase):
    market = bars([100] * 8)
    if phase == "entry":
        market[1] = replace(market[1], open_price=None)
        result = run(market)
        assert result.fills.empty and result.open_trades.empty
        np.testing.assert_allclose(result.ledger.restricted_proceeds, 0)
    else:
        index = 2
        field = "open_price" if phase == "open_risk" else "close_price"
        market[index] = replace(market[index], **{field: None})
        with pytest.raises(ValueError, match="opening risk price|valuation close"):
            run(market, holding_sessions=2 if phase == "cover" else 5)


@pytest.mark.parametrize("label,direction", [(0, "short"), (1, "flat"), (2, "long")])
def test_class_mapping_preserves_down_neutral_up_contract(label, direction):
    market = bars()
    item = decision(market[0], label)
    assert item.direction == direction
    result = run(market, [item])
    if direction == "flat":
        assert result.fills.empty
    else:
        assert result.completed_trades.iloc[0].direction == direction


def test_shorting_remains_disabled_by_default():
    market = bars()
    with pytest.raises(ValueError, match="Signals"):
        run(market, allow_short=False)


@pytest.mark.parametrize("kwargs", [
    {"allow_short": 1}, {"max_short_exposure": 0}, {"max_short_exposure": 1.1},
    {"initial_margin_fraction": 0}, {"maintenance_margin_fraction": 0},
    {"maintenance_margin_fraction": 0.6}, {"initial_margin_fraction": float("nan")},
    {"maintenance_margin_fraction": float("inf")}, {"max_short_exposure": True},
])
def test_invalid_short_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError, match="allow_short|margin|max_short_exposure"):
        ExecutionEngineV1(ExecutionConfig("SYNTH", **kwargs))

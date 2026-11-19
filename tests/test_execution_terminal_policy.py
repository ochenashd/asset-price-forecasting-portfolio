"""Predeclared terminal orders, signed accounting and fixed-endpoint causality."""

from dataclasses import replace
from decimal import Decimal

import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, MarketBar, Signal


def market(closes=None, opens=None):
    """Build a complete, explicit synthetic calendar with observed equity prices."""
    closes = [100, 100, 110, 110, 110, 110] if closes is None else closes
    opens = [100] * len(closes) if opens is None else opens
    days = pd.bdate_range("2020-01-06", periods=len(closes), tz="UTC")
    return [MarketBar("SYNTH", f"day-{i}", day + pd.Timedelta(hours=9, minutes=30),
                      day + pd.Timedelta(hours=16), opening, close)
            for i, (day, opening, close) in enumerate(zip(days, opens, closes))]


def decision(bar, direction="long"):
    return Signal(bar.session_id, bar.close_timestamp + pd.Timedelta(minutes=1), direction)


def simulate(bars, signals=None, **kwargs):
    config = ExecutionConfig(
        "SYNTH", initial_capital=1000, holding_sessions=kwargs.pop("holding_sessions", 5),
        fee_rate=kwargs.pop("fee_rate", 0.01), slippage_rate=kwargs.pop("slippage_rate", 0.01),
        allow_short=True,
        terminal_policy=kwargs.pop("terminal_policy", "liquidate_at_final_close"),
        terminal_session_id=kwargs.pop("terminal_session_id", "day-2"), **kwargs,
    )
    signals = [decision(bars[0])] if signals is None else signals
    return ExecutionEngineV1(config).run(bars, signals)


def assert_accounting(result, fee_rate=0.01):
    """Rebuild cash from signed fills and reconcile both executed/reference views."""
    cash = Decimal(1000)
    quantity = 0
    costs = Decimal(0)
    for row in result.ledger.iloc[1:].itertuples():
        fills = result.fills[result.fills.session_id == row.session_id]
        for fill in fills.itertuples():
            assert fill.fee == pytest.approx(fill.quantity * fill.executed_price * fee_rate)
            assert fill.slippage_cost == pytest.approx(
                fill.signed_quantity * (fill.executed_price - fill.reference_price))
            cash -= Decimal(fill.signed_quantity) * Decimal(str(fill.executed_price)) + Decimal(str(fill.fee))
            costs += Decimal(str(fill.fee))
            quantity += fill.signed_quantity
        assert row.cash == pytest.approx(float(cash))
        assert row.position_quantity == quantity
        assert row.equity == pytest.approx(float(cash) + quantity * row.close_price)
        assert row.equity == pytest.approx(1000 + row.cumulative_net_realized_pnl + row.gross_unrealized_pnl)
        assert row.equity == pytest.approx(
            1000 + row.cumulative_reference_net_realized_pnl + row.reference_unrealized_pnl)
        assert row.transaction_fees == pytest.approx(fills.fee.sum())
        assert row.cumulative_realized_costs == pytest.approx(float(costs))
        assert row.slippage_costs == pytest.approx(fills.slippage_cost.sum())
        trades = result.completed_trades[result.completed_trades.exit_session_id == row.session_id]
        assert row.realized_pnl == pytest.approx(trades.net_realized_pnl.sum())
    assert result.fills.fill_id.is_unique
    assert result.fills.fill_sequence.tolist() == list(range(len(result.fills)))


@pytest.mark.parametrize("direction,close,entry,entry_fee,cash,equity,unrealized,restricted", [
    ("long", 110, 101, 9.09, 81.91, 1071.91, 81, 0),
    ("short", 80, 99, 8.91, 1882.09, 1162.09, 171, 891),
])
def test_marking_preserves_actual_open_trade_without_hypothetical_exit_costs(
        direction, close, entry, entry_fee, cash, equity, unrealized, restricted):
    bars = market([100, 100, close, close])
    result = simulate(bars, [decision(bars[0], direction)], terminal_policy="mark_to_market")
    assert result.terminal_outcome == "marked" and result.terminal_order_id is None
    assert result.terminal_policy == "mark_to_market" and result.terminal_session_id == "day-2"
    assert result.effective_end_session_id == "day-2"
    assert result.terminal_close_timestamp == bars[2].close_timestamp
    assert result.run_status == "completed"
    assert len(result.fills) == 1 and result.completed_trades.empty
    trade = result.open_trades.iloc[0]
    assert trade.entry_price == entry and trade.entry_fee == entry_fee
    assert trade.unrealized_pnl == unrealized
    assert trade.net_realized_pnl == -entry_fee
    assert trade.net_pnl_to_date == pytest.approx(unrealized - entry_fee)
    assert trade.exit_timestamp is None and trade.exit_fee == 0 and trade.exit_slippage_cost == 0
    assert result.open_position is not None
    last = result.ledger.iloc[-1]
    assert last.cash == pytest.approx(cash) and last.equity == pytest.approx(equity)
    assert last.restricted_proceeds == restricted
    assert_accounting(result)


@pytest.mark.parametrize("direction,close", [("long", 110), ("short", 80)])
def test_terminal_long_and_short_match_independent_decimal_references(direction, close):
    bars = market([100, 100, close, close])
    source = decision(bars[0], direction)
    result = simulate(bars, [source])
    signed_quantity = Decimal(9 if direction == "long" else -9)
    entry = Decimal(101 if direction == "long" else 99)
    executed_exit = Decimal(close) * (Decimal("0.99") if direction == "long" else Decimal("1.01"))
    entry_fee = abs(signed_quantity) * entry * Decimal("0.01")
    exit_fee = abs(signed_quantity) * executed_exit * Decimal("0.01")
    cash = Decimal(1000) - signed_quantity * entry - entry_fee
    cash += signed_quantity * executed_exit - exit_fee
    assert cash == Decimal("1052.209" if direction == "long" else "1147.618")
    trade = result.completed_trades.iloc[0]
    assert result.terminal_outcome == "liquidated" and result.terminal_order_id
    assert result.terminal_policy == "liquidate_at_final_close"
    assert result.run_status == "completed" and result.terminal_snapshot is None
    assert trade.quantity == signed_quantity and trade.exit_reason == "terminal"
    assert trade.entry_price == entry
    assert trade.entry_fee == float(entry_fee) and trade.exit_fee == float(exit_fee)
    assert trade.exit_price == float(executed_exit)
    assert trade.gross_realized_pnl == pytest.approx(float(signed_quantity * (executed_exit - entry)))
    assert trade.net_realized_pnl == pytest.approx(float(cash - 1000))
    assert trade.signal_timestamp == source.decision_timestamp
    assert trade.exit_signal_timestamp is None and trade.exit_signal_session_id is None
    assert trade.exit_timestamp == bars[2].close_timestamp
    assert trade.holding_sessions == 2
    assert trade.holding_duration_seconds == (bars[2].close_timestamp - bars[1].open_timestamp).total_seconds()
    fill = result.fills.iloc[-1]
    assert fill.signal_timestamp == source.decision_timestamp
    assert pd.isna(fill.trigger_signal_timestamp)
    assert fill.trade_id == trade.trade_id and fill.fill_id == trade.exit_fill_id
    event = result.events.query("event == 'terminal_exit_filled'").iloc[0]
    assert pd.isna(event.signal_timestamp) and pd.isna(event.signal_session_id)
    last = result.ledger.iloc[-1]
    assert last.cash == last.equity == float(cash)
    assert last.position_quantity == last.restricted_proceeds == last.gross_unrealized_pnl == 0
    assert result.open_trades.empty and result.open_position is None
    assert_accounting(result)


@pytest.mark.parametrize("direction,close,equity", [("long", 110, 1100), ("short", 80, 1200)])
def test_zero_cost_terminal_fill_preserves_gross_pnl(direction, close, equity):
    bars = market([100, 100, close])
    result = simulate(bars, [decision(bars[0], direction)], fee_rate=0, slippage_rate=0)
    assert result.ledger.iloc[-1].equity == equity
    assert result.completed_trades.iloc[0].commission_costs == 0
    assert result.completed_trades.iloc[0].slippage_costs == 0
    assert_accounting(result, fee_rate=0)


@pytest.mark.parametrize("direction", ["long", "short"])
def test_expiry_on_terminal_close_has_priority_and_no_duplicate_exit(direction):
    bars = market([100] * 5)
    signals = [decision(bars[0], direction)]
    terminal = simulate(bars, signals, holding_sessions=2)
    marked = simulate(bars, signals, holding_sessions=2, terminal_policy="mark_to_market")
    assert terminal.completed_trades.iloc[0].exit_reason == "scheduled"
    assert len(terminal.fills) == 2 and len(terminal.completed_trades) == 1
    assert terminal.terminal_outcome == "already_flat"
    assert terminal.events.query("event == 'terminal_exit_filled'").empty
    for field in ["ledger", "fills", "completed_trades", "events"]:
        pd.testing.assert_frame_equal(getattr(terminal, field), getattr(marked, field))
    assert_accounting(terminal)


@pytest.mark.parametrize("source,target", [("long", "short"), ("short", "long")])
def test_terminal_open_reversal_precedes_close_liquidation(source, target):
    bars = market([100] * 5)
    result = simulate(bars, [decision(bars[0], source), decision(bars[1], target)],
                      position_policy="replace_next_open", allocation_fraction=0.5)
    assert result.completed_trades.exit_reason.tolist() == ["reversal", "terminal"]
    assert result.fills.timestamp.tolist() == [bars[1].open_timestamp, bars[2].open_timestamp,
                                               bars[2].open_timestamp, bars[2].close_timestamp]
    assert result.completed_trades.trade_id.is_unique
    assert result.terminal_outcome == "liquidated" and result.open_position is None
    assert result.completed_trades.holding_sessions.tolist() == [1, 1]
    assert_accounting(result)


def test_failed_terminal_day_reversal_entry_keeps_only_actual_old_exit():
    bars = market([100, 80, 90, 90], [100, 100, 90, 90])
    result = simulate(bars, [decision(bars[0], "short"), decision(bars[1], "long")],
                      position_policy="replace_next_open")
    assert result.completed_trades.exit_reason.tolist() == ["reversal"]
    assert len(result.fills) == 2 and result.terminal_outcome == "already_flat"
    assert result.events.query("reason == 'unaffordable_open'").iloc[0].quantity == 14
    assert result.events.query("event == 'terminal_exit_filled'").empty
    assert_accounting(result)


def test_terminal_day_entry_is_allowed_and_horizon_one_remains_scheduled():
    bars = market([100] * 4)
    signals = [decision(bars[1])]
    long_horizon = simulate(bars, signals)
    short_horizon = simulate(bars, signals, holding_sessions=1)
    assert long_horizon.fills.timestamp.tolist() == [bars[2].open_timestamp, bars[2].close_timestamp]
    assert long_horizon.completed_trades.iloc[0].exit_reason == "terminal"
    assert long_horizon.completed_trades.iloc[0].holding_sessions == 1
    assert short_horizon.completed_trades.iloc[0].exit_reason == "scheduled"
    assert short_horizon.terminal_outcome == "already_flat"
    assert len(short_horizon.fills) == 2
    assert_accounting(long_horizon)
    assert_accounting(short_horizon)


@pytest.mark.parametrize("policy", ["mark_to_market", "liquidate_at_final_close"])
def test_flat_endpoint_has_no_fabricated_fills_or_costs(policy):
    result = simulate(market(), [], terminal_policy=policy)
    assert result.terminal_outcome == "already_flat"
    assert result.fills.empty and result.completed_trades.empty and result.open_trades.empty
    assert (result.ledger.equity == 1000).all()
    assert (result.ledger.transaction_fees == 0).all()


@pytest.mark.parametrize("direction", ["long", "short", "flat"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0, -1, True, "110"])
def test_invalid_terminal_close_is_never_filled_or_marked(direction, bad):
    bars = market([100] * 4)
    bars[2] = replace(bars[2], close_price=bad)
    with pytest.raises(ValueError, match="observed finite positive valuation close"):
        simulate(bars, [decision(bars[0], direction)])


def test_synthetic_terminal_close_is_not_an_execution_price():
    bars = market()
    bars[2] = replace(bars[2], close_observed=False)
    with pytest.raises(ValueError, match="observed finite positive valuation close"):
        simulate(bars)


@pytest.mark.parametrize("identifier", ["missing", "day-99"])
def test_declared_session_must_exist_without_last_row_fallback(identifier):
    with pytest.raises(ValueError, match="unavailable"):
        simulate(market(), terminal_session_id=identifier)


@pytest.mark.parametrize("changes", [
    {"terminal_policy": "unknown"},
    {"terminal_policy": "liquidate_at_final_close", "terminal_session_id": None},
    {"terminal_session_id": ""}, {"terminal_session_id": " "},
    {"terminal_session_id": False}, {"terminal_session_id": 2},
])
def test_invalid_terminal_configuration_is_rejected(changes):
    with pytest.raises(ValueError, match="terminal"):
        ExecutionEngineV1(ExecutionConfig("SYNTH", **changes))


def test_terminal_close_timestamp_must_be_timezone_aware():
    bars = market()
    bars[2] = replace(bars[2], close_timestamp=bars[2].close_timestamp.tz_localize(None))
    with pytest.raises(ValueError, match="timezone-aware"):
        simulate(bars)


@pytest.mark.parametrize("phase", ["open", "close"])
def test_earlier_margin_breach_prevents_all_terminal_fills(phase):
    bars = market([100] * 6)
    bars[2] = replace(bars[2], **{f"{phase}_price": 161})
    # The later terminal close is deliberately invalid and must never be used.
    bars[4] = replace(bars[4], close_price=None)
    result = simulate(bars, [decision(bars[0], "short")], terminal_session_id="day-4")
    marked = simulate(bars, [decision(bars[0], "short")], terminal_session_id="day-4",
                      terminal_policy="mark_to_market")
    assert result.run_status == "margin_breach" and result.terminal_outcome == "not_reached"
    assert result.effective_end_session_id == "day-2"
    assert result.terminal_snapshot == marked.terminal_snapshot
    assert result.terminal_snapshot.phase == phase
    assert result.terminal_snapshot.restricted_proceeds == 891
    assert len(result.fills) == 1 and result.completed_trades.empty
    assert result.open_position.quantity == -9
    assert result.events.query("event == 'terminal_exit_filled'").empty
    for field in ["fills", "ledger", "open_trades"]:
        pd.testing.assert_frame_equal(getattr(result, field), getattr(marked, field))


def test_opening_breach_on_terminal_session_preempts_precommitted_cover():
    bars = market([100] * 5, [100, 100, 161, 100, 100])
    result = simulate(bars, [decision(bars[0], "short")])
    assert result.run_status == "margin_breach" and result.terminal_outcome == "not_reached"
    assert result.terminal_snapshot.phase == "open"
    assert len(result.fills) == 1 and result.completed_trades.empty
    assert result.terminal_snapshot.timestamp == bars[2].open_timestamp
    assert result.ledger.iloc[-1].session_id == "day-1"


def test_terminal_cover_occurs_before_closing_margin_check():
    bars = market([100, 100, 161, 161])
    result = simulate(bars, [decision(bars[0], "short")])
    assert result.run_status == "completed" and result.terminal_outcome == "liquidated"
    expected = Decimal("1882.09") - 9 * Decimal("162.61") * Decimal("1.01")
    assert result.ledger.iloc[-1].equity == float(expected)
    assert result.open_position is None
    assert_accounting(result)


def test_actual_terminal_cover_preserves_negative_cash_and_reports_insolvency():
    bars = market([100, 100, 250, 100])
    result = simulate(bars, [decision(bars[0], "short")])
    expected = Decimal("1882.09") - 9 * Decimal("252.50") * Decimal("1.01")
    assert expected == Decimal("-413.135")
    assert result.run_status == "insolvent" and result.terminal_outcome == "liquidated"
    assert result.effective_end_session_id == "day-2"
    assert len(result.fills) == 2 and result.completed_trades.iloc[0].exit_reason == "terminal"
    assert result.open_position is None and result.open_trades.empty
    assert result.terminal_snapshot.cash == result.terminal_snapshot.equity == float(expected)
    assert result.terminal_snapshot.restricted_proceeds == 0
    assert result.terminal_snapshot.position_quantity == 0
    assert_accounting(result)


@pytest.mark.parametrize("policy", ["mark_to_market", "liquidate_at_final_close"])
def test_fixed_endpoint_ignores_terminal_after_close_and_later_signals(policy):
    bars = market()
    first = decision(bars[0])
    extra = [decision(bars[2], "short"), decision(bars[3], "long"), decision(bars[4], "short")]
    baseline = simulate(bars, [first], terminal_policy=policy)
    extended_signals = simulate(bars, [first] + extra, terminal_policy=policy)
    for field in ["ledger", "fills", "completed_trades", "open_trades", "events"]:
        pd.testing.assert_frame_equal(getattr(baseline, field), getattr(extended_signals, field))
    assert extended_signals.events.timestamp.max() <= bars[2].close_timestamp
    assert extended_signals.ledger.iloc[-1].timestamp == bars[2].close_timestamp
    assert not extended_signals.events.event.str.contains("unfilled").any()


@pytest.mark.parametrize("policy", ["mark_to_market", "liquidate_at_final_close"])
def test_fixed_endpoint_is_invariant_to_appended_prices_signals_and_metadata(policy):
    bars = market()
    source = decision(bars[0], "short")
    original = simulate(bars[:3], [source], terminal_policy=policy)
    changed = [replace(bar, open_price=None, close_price=float("nan"), open_observed=False,
                       close_timestamp=None) if i > 2 else bar for i, bar in enumerate(bars)]
    future = simulate(changed, [source, decision(bars[4])], terminal_policy=policy)
    for field in ["ledger", "fills", "completed_trades", "open_trades", "events"]:
        pd.testing.assert_frame_equal(getattr(original, field), getattr(future, field))
    for field in ["terminal_order_id", "terminal_outcome", "terminal_close_timestamp",
                  "terminal_policy", "terminal_session_id", "effective_end_session_id",
                  "run_status", "open_position", "terminal_snapshot"]:
        assert getattr(original, field) == getattr(future, field)


def test_future_prices_are_not_even_read_after_declared_endpoint():
    class UnreadableFutureBar:
        """Any access proves an out-of-window bar was inspected."""

        def __getattribute__(self, name):
            raise AssertionError(f"Read future bar attribute: {name}")

    bars = market()[:3] + [UnreadableFutureBar()]
    result = simulate(bars)
    assert result.terminal_outcome == "liquidated"
    assert result.effective_end_session_id == "day-2"


def test_endpoint_does_not_legitimize_a_delayed_earlier_signal():
    bars = market()
    delayed = replace(decision(bars[0]), decision_timestamp=bars[2].close_timestamp + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="before the next open"):
        simulate(bars, [delayed])


def test_terminal_order_and_trade_ids_are_stable_and_price_independent():
    bars = market()
    signals = [decision(bars[0])]
    original = simulate(bars, signals)
    changed = simulate([replace(bar, close_price=120) if i == 2 else bar
                        for i, bar in enumerate(bars)], signals)
    assert original.terminal_order_id == changed.terminal_order_id
    assert original.fills.fill_id.tolist() == changed.fills.fill_id.tolist()
    assert original.completed_trades.trade_id.tolist() == changed.completed_trades.trade_id.tolist()


@pytest.mark.parametrize("direction", ["long", "short"])
def test_default_marking_without_endpoint_preserves_existing_unfilled_behavior(direction):
    bars = market([100] * 3)
    signals = [decision(bars[0], direction), decision(bars[-1], "flat")]
    config = ExecutionConfig("SYNTH", initial_capital=1000, allow_short=True,
                             position_policy="replace_next_open")
    default = ExecutionEngineV1(config).run(bars, signals)
    explicit = ExecutionEngineV1(replace(config, terminal_policy="mark_to_market")).run(bars, signals)
    for field in ["ledger", "fills", "events", "completed_trades", "open_trades"]:
        pd.testing.assert_frame_equal(getattr(default, field), getattr(explicit, field))
    assert default.terminal_policy == "mark_to_market" and default.terminal_session_id is None
    assert default.terminal_close_timestamp is None and default.terminal_order_id is None
    assert default.terminal_outcome == "marked"
    assert default.events.iloc[-1].event == "replacement_unfilled_at_end"
    assert default.open_position is not None and len(default.fills) == 1

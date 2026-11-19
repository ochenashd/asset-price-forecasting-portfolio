"""Deterministic next-open replacement, reversal and accounting contracts."""

from dataclasses import replace
from decimal import Decimal

import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, MarketBar, Signal
from src.evaluation.execution.portfolio import Portfolio


def market(closes=None, opens=None):
    """Use explicit identifiers and a complete synthetic exchange-session calendar."""
    closes = [100] * 9 if closes is None else closes
    opens = [100] * len(closes) if opens is None else opens
    days = pd.bdate_range("2020-01-06", periods=len(closes), tz="UTC")
    return [MarketBar("SYNTH", f"day-{i}", day + pd.Timedelta(hours=9, minutes=30),
                      day + pd.Timedelta(hours=16), opening, close)
            for i, (day, opening, close) in enumerate(zip(days, opens, closes))]


def decision(bar, direction):
    return Signal(bar.session_id, bar.close_timestamp + pd.Timedelta(minutes=1), direction)


def run(bars, signals, **kwargs):
    config = ExecutionConfig(
        "SYNTH", initial_capital=1000, holding_sessions=kwargs.pop("holding_sessions", 5),
        fee_rate=kwargs.pop("fee_rate", 0), slippage_rate=kwargs.pop("slippage_rate", 0),
        allow_short=True, position_policy=kwargs.pop("position_policy", "replace_next_open"),
        **kwargs,
    )
    return ExecutionEngineV1(config).run(bars, signals)


def reconcile(result, initial=1000):
    """Independently reconcile signed cash flows, fees, marks and both P&L views."""
    cash = Decimal(initial)
    quantity = 0
    fees = Decimal(0)
    for row in result.ledger.iloc[1:].itertuples():
        session_fills = result.fills[result.fills.session_id == row.session_id]
        for fill in session_fills.itertuples():
            cash -= Decimal(fill.signed_quantity) * Decimal(str(fill.executed_price))
            cash -= Decimal(str(fill.fee))
            fees += Decimal(str(fill.fee))
            quantity += fill.signed_quantity
            assert fill.fee == pytest.approx(fill.quantity * fill.executed_price * 0.01)
            assert fill.slippage_cost == pytest.approx(
                fill.signed_quantity * (fill.executed_price - fill.reference_price))
        assert row.cash == pytest.approx(float(cash))
        assert row.position_quantity == quantity
        assert row.equity == pytest.approx(float(cash) + quantity * row.close_price)
        assert row.equity == pytest.approx(
            initial + row.cumulative_net_realized_pnl + row.gross_unrealized_pnl)
        assert row.equity == pytest.approx(
            initial + row.cumulative_reference_net_realized_pnl + row.reference_unrealized_pnl)
        assert row.cumulative_realized_costs == pytest.approx(float(fees))
        assert row.transaction_fees == pytest.approx(session_fills.fee.sum())
        assert row.slippage_costs == pytest.approx(session_fills.slippage_cost.sum())
        closed = result.completed_trades[result.completed_trades.exit_session_id == row.session_id]
        assert row.realized_pnl == pytest.approx(closed.net_realized_pnl.sum())
        assert row.gross_realized_pnl == pytest.approx(closed.gross_realized_pnl.sum())
    assert result.fills.fill_id.is_unique
    assert result.fills.fill_sequence.tolist() == list(range(len(result.fills)))


@pytest.mark.parametrize("source,target,expected", [
    ("flat", "long", 10), ("flat", "short", -10),
    ("long", "flat", 0), ("short", "flat", 0),
    ("long", "short", -10), ("short", "long", 10),
    ("long", "long", 10), ("short", "short", -10),
])
def test_all_eight_transitions_execute_only_at_the_next_open(source, target, expected):
    bars = market()
    signals = [decision(bars[0], source), decision(bars[1], target)]
    result = run(bars, signals)
    assert result.ledger.iloc[3].position_quantity == expected
    assert result.ledger.iloc[2].position_quantity == {"flat": 0, "long": 10, "short": -10}[source]
    if source != "flat" and target != source:
        old = result.completed_trades.iloc[0]
        assert old.exit_timestamp == bars[2].open_timestamp
        assert old.exit_reason == ("signal_flat" if target == "flat" else "reversal")
        assert old.signal_timestamp == signals[0].decision_timestamp
        assert old.exit_signal_timestamp == signals[1].decision_timestamp
        assert old.exit_signal_session_id == signals[1].session_id
        assert old.holding_sessions == 1
        assert old.holding_duration_seconds == (
            bars[2].open_timestamp - bars[1].open_timestamp).total_seconds()
        if target != "flat":
            assert result.completed_trades.iloc[1].trade_id != old.trade_id
            legs = result.fills[result.fills.session_id == bars[2].session_id]
            assert legs.trade_id.tolist() == result.completed_trades.trade_id.tolist()
            assert legs.timestamp.tolist() == [bars[2].open_timestamp] * 2
            assert legs.iloc[0].trigger_signal_timestamp == signals[1].decision_timestamp


@pytest.mark.parametrize("direction,target,close,expected_equity,new_quantity", [
    ("long", "short", 110, "1032.508", -9),
    ("short", "long", 80, "1125.106", 14),
])
def test_reference_reversals_reconcile_separate_legs(direction, target, close, expected_equity, new_quantity):
    bars = market([100, close] + [close] * 7, [100, 100] + [close] * 7)
    signals = [decision(bars[0], direction), decision(bars[1], target)]
    result = run(bars, signals, fee_rate=0.01, slippage_rate=0.01)
    old = result.completed_trades.iloc[0]
    q = Decimal(9)
    entry = Decimal(101 if direction == "long" else 99)
    entry_fee = q * entry * Decimal("0.01")
    exit_price = Decimal(close) * (Decimal("0.99") if direction == "long" else Decimal("1.01"))
    exit_fee = q * exit_price * Decimal("0.01")
    sign = 1 if direction == "long" else -1
    flat_cash = Decimal(1000) + sign * q * (exit_price - entry) - entry_fee - exit_fee
    new_price = Decimal(close) * (Decimal("0.99") if target == "short" else Decimal("1.01"))
    new_fee = abs(new_quantity) * new_price * Decimal("0.01")
    cash = flat_cash - new_quantity * new_price - new_fee
    equity = cash + new_quantity * Decimal(close)
    assert equity == Decimal(expected_equity)
    row = result.ledger.iloc[3]
    assert row.cash == pytest.approx(float(cash))
    assert row.equity == pytest.approx(float(equity))
    assert row.position_quantity == new_quantity
    assert old.net_realized_pnl == pytest.approx(float(flat_cash - 1000))
    assert row.restricted_proceeds == pytest.approx(float(abs(new_quantity) * new_price) if target == "short" else 0)
    assert result.completed_trades.iloc[1].quantity == new_quantity
    reconcile(result)


@pytest.mark.parametrize("source,target,gap,reason", [
    ("short", "long", 90, "unaffordable_open"),
    ("long", "short", 300, "short_exposure_cap"),
])
def test_rejected_new_leg_preserves_actual_exit_and_stays_flat(source, target, gap, reason):
    close = 80 if source == "short" else 110
    bars = market([100, close] + [gap] * 7, [100, 100] + [gap] * 7)
    result = run(bars, [decision(bars[0], source), decision(bars[1], target)],
                 fee_rate=0.01, slippage_rate=0.01)
    assert len(result.fills) == 2
    assert len(result.completed_trades) == 1
    assert result.open_trades.empty and result.open_position is None
    assert result.completed_trades.iloc[0].exit_reason == "reversal"
    rejection = result.events.query("event == 'entry_rejected'").iloc[0]
    assert rejection.reason == reason
    assert rejection.quantity == (14 if source == "short" else 9)
    assert (result.ledger.iloc[3:].position_quantity == 0).all()
    assert (result.ledger.iloc[3:].restricted_proceeds == 0).all()
    reconcile(result)


def test_favorable_gap_does_not_increase_frozen_replacement_quantity():
    bars = market([100, 80] + [70] * 7, [100, 100] + [70] * 7)
    result = run(bars, [decision(bars[0], "short"), decision(bars[1], "long")],
                 fee_rate=0.01, slippage_rate=0.01)
    assert result.fills.iloc[2].quantity == 14
    assert result.fills.iloc[2].timestamp == bars[2].open_timestamp
    reconcile(result)


@pytest.mark.parametrize("direction", ["long", "short"])
def test_same_direction_signal_preserves_identifier_quantity_and_original_expiry(direction):
    bars = market()
    first = decision(bars[0], direction)
    baseline = run(bars, [first], holding_sessions=3)
    repeated = run(bars, [first, decision(bars[1], direction), decision(bars[2], direction)], holding_sessions=3)
    pd.testing.assert_frame_equal(baseline.fills, repeated.fills)
    pd.testing.assert_frame_equal(baseline.completed_trades, repeated.completed_trades)
    assert repeated.completed_trades.iloc[0].exit_timestamp == bars[3].close_timestamp
    assert repeated.events.query("reason == 'same_direction'").shape[0] == 2


@pytest.mark.parametrize("target", ["flat", "short"])
def test_replacement_on_old_expiry_day_cancels_only_the_old_expiry(target):
    bars = market()
    result = run(bars, [decision(bars[0], "long"), decision(bars[2], target)], holding_sessions=3)
    old = result.completed_trades.iloc[0]
    assert old.exit_timestamp == bars[3].open_timestamp
    assert len(result.fills[result.fills.trade_id == old.trade_id]) == 2
    if target == "short":
        assert result.completed_trades.iloc[1].exit_timestamp == bars[5].close_timestamp
    else:
        assert len(result.fills) == 2


def test_after_close_signal_cannot_replace_an_already_executed_expiry():
    bars = market()
    result = run(bars, [decision(bars[0], "long"), decision(bars[2], "short")], holding_sessions=2)
    assert result.completed_trades.iloc[0].exit_timestamp == bars[2].close_timestamp
    assert result.completed_trades.iloc[0].exit_reason == "scheduled"
    assert result.fills.iloc[2].timestamp == bars[3].open_timestamp


def test_horizon_one_exits_before_after_close_replacement_signal():
    bars = market()
    result = run(bars, [decision(bars[0], "long"), decision(bars[1], "short")], holding_sessions=1,
                 fee_rate=0.01, slippage_rate=0.01)
    assert result.completed_trades.exit_reason.tolist() == ["scheduled", "scheduled"]
    assert result.completed_trades.holding_sessions.tolist() == [1, 1]
    assert result.fills.timestamp.tolist() == [bars[1].open_timestamp, bars[1].close_timestamp,
                                              bars[2].open_timestamp, bars[2].close_timestamp]
    reconcile(result)


@pytest.mark.parametrize("target", ["flat", "long"])
def test_opening_margin_breach_preempts_pending_short_replacement(target):
    bars = market([100] * 5, [100, 100, 160, 100, 100])
    result = run(bars, [decision(bars[0], "short"), decision(bars[1], target)])
    assert result.run_status == "margin_breach"
    assert len(result.fills) == 1 and result.completed_trades.empty
    assert result.open_position.quantity == -10
    assert result.terminal_snapshot.timestamp == bars[2].open_timestamp
    assert result.terminal_snapshot.restricted_proceeds == 1000
    assert result.events.query("event == 'order_blocked_by_risk'").shape[0] == 1
    assert result.events.query("event == 'replacement_unfilled_at_end'").empty
    assert bars[2].session_id not in result.ledger.session_id.tolist()


@pytest.mark.parametrize("source,target", [("long", "flat"), ("long", "short"), ("short", "long")])
@pytest.mark.parametrize("bad", [None, float("nan"), 0, -1])
def test_required_replacement_open_must_be_observed_valid_price(source, target, bad):
    bars = market()
    bars[2] = replace(bars[2], open_price=bad)
    with pytest.raises(ValueError, match="observed positive.*(execution open|opening risk price)"):
        run(bars, [decision(bars[0], source), decision(bars[1], target)])


def test_synthetic_open_is_rejected_for_replacement_exit():
    bars = market()
    bars[2] = replace(bars[2], open_observed=False)
    with pytest.raises(ValueError, match="replacement execution open"):
        run(bars, [decision(bars[0], "long"), decision(bars[1], "flat")])


def test_entry_only_missing_open_rejects_without_fills_or_fees():
    bars = market()
    bars[1] = replace(bars[1], open_price=None)
    result = run(bars, [decision(bars[0], "long")], fee_rate=0.01)
    assert result.fills.empty and result.completed_trades.empty
    assert (result.ledger.equity == 1000).all()
    assert (result.ledger.transaction_fees == 0).all()
    assert result.events.query("reason == 'invalid_or_unobserved_open'").shape[0] == 1


def test_hypothetical_close_does_not_release_short_proceeds_or_mutate_portfolio():
    portfolio = Portfolio(Decimal(1000))
    portfolio.enter_short(9, Decimal(99), 0.01, reference_price=100, max_exposure=1, initial_margin=0.5)
    before = vars(portfolio).copy()
    assert portfolio.estimated_flat_cash(80, 0.01, 0.01) == Decimal("1147.618")
    assert vars(portfolio) == before
    assert portfolio.restricted_proceeds == 891


def test_zero_new_quantity_still_closes_old_position_without_phantom_entry():
    bars = market()
    result = run(bars, [decision(bars[0], "long"), decision(bars[1], "short")],
                 initial_margin_fraction=1000)
    assert len(result.fills) == 2
    assert result.open_position is None
    assert result.events.query("reason == 'zero_quantity'").shape[0] == 1
    assert result.completed_trades.iloc[0].exit_reason == "reversal"


@pytest.mark.parametrize("target", ["flat", "short"])
def test_last_session_replacement_remains_unfilled_without_fake_exit(target):
    bars = market()[:2]
    result = run(bars, [decision(bars[0], "long"), decision(bars[1], target)])
    assert len(result.fills) == 1 and result.completed_trades.empty
    assert result.open_position.quantity == 10
    assert result.events.iloc[-1].event == "replacement_unfilled_at_end"
    assert result.open_trades.iloc[0].exit_reason is None


def test_replacement_near_end_preserves_new_open_trade_without_future_exit():
    bars = market()[:3]
    result = run(bars, [decision(bars[0], "long"), decision(bars[1], "short")])
    assert len(result.completed_trades) == 1
    assert result.open_position.quantity == -10
    assert result.open_trades.iloc[0].entry_timestamp == bars[2].open_timestamp
    assert result.open_trades.iloc[0].holding_sessions == 1
    assert len(result.fills) == 3


def test_future_price_changes_and_appends_preserve_earlier_fills_and_positions():
    bars = market()
    signals = [decision(bars[0], "long"), decision(bars[1], "short"), decision(bars[2], "long")]
    original = run(bars, signals, fee_rate=0.01, slippage_rate=0.01)
    changed = [replace(bar, open_price=130, close_price=130) if i >= 5 else bar
               for i, bar in enumerate(bars)]
    future = run(changed, signals, fee_rate=0.01, slippage_rate=0.01)
    cutoff = bars[4].close_timestamp
    pd.testing.assert_frame_equal(original.ledger.query("timestamp <= @cutoff"),
                                  future.ledger.query("timestamp <= @cutoff"))
    pd.testing.assert_frame_equal(original.fills.query("timestamp <= @cutoff"),
                                  future.fills.query("timestamp <= @cutoff"))
    prefix = run(bars[:3], signals[:2], fee_rate=0.01, slippage_rate=0.01)
    pd.testing.assert_frame_equal(prefix.ledger, original.ledger.iloc[:4])
    pd.testing.assert_frame_equal(prefix.fills, original.fills.iloc[:3])
    pd.testing.assert_frame_equal(prefix.completed_trades, original.completed_trades.iloc[:1])


def test_duplicate_pending_instructions_are_rejected_instead_of_overwritten():
    bars = market()
    with pytest.raises(ValueError, match="unique feature-session"):
        run(bars, [decision(bars[0], "long"), decision(bars[1], "flat"), decision(bars[1], "short")])


def test_default_policy_matches_explicit_fixed_horizon_and_ignores_replacements():
    bars = market()
    signals = [decision(bars[0], "long"), decision(bars[1], "short"), decision(bars[2], "flat")]
    config = ExecutionConfig("SYNTH", initial_capital=1000, holding_sessions=5, fee_rate=0, allow_short=True)
    default = ExecutionEngineV1(config).run(bars, signals)
    explicit = ExecutionEngineV1(replace(config, position_policy="fixed_horizon")).run(bars, signals)
    for field in ["ledger", "fills", "completed_trades", "open_trades", "events"]:
        pd.testing.assert_frame_equal(getattr(default, field), getattr(explicit, field))
    assert default.completed_trades.iloc[0].exit_timestamp == bars[5].close_timestamp
    assert default.completed_trades.iloc[0].exit_reason == "scheduled"


def test_unknown_position_policy_is_rejected():
    with pytest.raises(ValueError, match="position_policy"):
        ExecutionEngineV1(ExecutionConfig("SYNTH", position_policy="unknown"))


def test_costs_after_actual_cover_can_stop_insolvent_before_new_entry():
    bars = market([100] * 5, [100, 100, 208, 100, 100])
    result = run(bars, [decision(bars[0], "short"), decision(bars[1], "long")],
                 fee_rate=0.01, slippage_rate=0.01,
                 initial_margin_fraction=0.001, maintenance_margin_fraction=0.0001)
    # Opening equity is positive; actual adverse cover and its fee exhaust cash.
    expected = Decimal("1882.09") - 9 * Decimal("210.08") * Decimal("1.01")
    assert expected == Decimal("-27.5372")
    assert result.run_status == "insolvent"
    assert len(result.fills) == 2 and len(result.completed_trades) == 1
    assert result.open_position is None and result.open_trades.empty
    assert result.terminal_snapshot.cash == pytest.approx(float(expected))
    assert result.terminal_snapshot.position_quantity == 0
    assert result.terminal_snapshot.restricted_proceeds == 0
    assert result.terminal_snapshot.timestamp == bars[2].open_timestamp
    assert result.terminal_snapshot.equity == pytest.approx(
        1000 + result.terminal_snapshot.cumulative_net_realized_pnl)
    assert bars[2].session_id not in result.ledger.session_id.tolist()


def test_consecutive_reversals_reconcile_and_never_duplicate_old_expiries():
    bars = market([100, 110, 90, 105, 100, 80, 90, 90, 90])
    signals = [decision(bars[i], direction) for i, direction in
               enumerate(["long", "short", "long", "flat", "short"])]
    result = run(bars, signals, holding_sessions=3, fee_rate=0.01,
                 slippage_rate=0.01, allocation_fraction=0.5)
    assert result.completed_trades.exit_reason.tolist() == ["reversal", "reversal", "signal_flat", "scheduled"]
    assert result.completed_trades.trade_id.is_unique
    assert result.completed_trades.holding_sessions.tolist() == [1, 1, 1, 3]
    assert len(result.fills) == 8
    assert all(result.fills.groupby("trade_id").size() == 2)
    reconcile(result)


def test_disabled_short_intent_is_rejected_without_executing_either_leg():
    bars = market()
    engine = ExecutionEngineV1(ExecutionConfig("SYNTH", position_policy="replace_next_open"))
    with pytest.raises(ValueError, match="enabled direction"):
        engine.run(bars, [decision(bars[0], "long"), decision(bars[1], "short")])

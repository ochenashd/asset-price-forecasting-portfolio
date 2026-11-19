"""Causal fixed-horizon or next-open replacement with collateralized shorts."""

import math
import json
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal, Sequence
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from .contracts import ExecutionConfig, ExecutionResult, MarketBar, OpenPosition, RiskSnapshot, Signal, TradeRecord
from .portfolio import Portfolio, amount, slipped_price
from .prediction_export import _utc_instant


def _positive_price(value: object) -> bool:
    """Accept finite positive numeric equity prices, never booleans or strings."""
    return (not isinstance(value, (bool, str)) and value is not None
            and isinstance(value, (int, float)) and math.isfinite(value) and value > 0)


def _aware(value: datetime) -> bool:
    """Require explicit timezone-aware event times rather than ambiguous bar labels."""
    return (isinstance(value, datetime) and not pd.isna(value)
            and value.tzinfo is not None and value.utcoffset() is not None)


def _trade_id(symbol: str, signal: Signal) -> str:
    """Derive identity from the source decision, never prices or future row counts."""
    identity = json.dumps([symbol, signal.session_id,
                           signal.decision_timestamp.astimezone(timezone.utc).isoformat()]
                          + (["short"] if signal.direction == "short" else []))
    return str(uuid5(NAMESPACE_URL, "apf:execution_trade_ledger_v1:" + identity))


@dataclass(frozen=True)
class _PendingInstruction:
    """One next-open intent with frozen sizing and optional source-trade guard."""

    signal: Signal
    quantity: int
    close_trade_id: str | None = None


class ExecutionEngineV1:
    """Run a caller-supplied complete session calendar without target-return inputs.

    Entry quantity is fixed after the signal close. Invalid/missing opens and
    unaffordable opening gaps cancel that entry without retry. Missing closes
    abort valuation. An H-session holding entered at open e exits via an assumed
    full-fill market-on-close order at close e+H-1, pre-scheduled at entry.
    No high/low information influences fills. Terminal liquidation requires a
    predeclared session; scheduled expiry precedes terminal closing fills.
    Carried shorts are checked at observed opens and closes. The first margin
    breach stops the run without a liquidation fill. A scheduled cover is filled
    before closing solvency is assessed. Daily rows always represent closes;
    an opening risk stop is returned separately in terminal_snapshot.
    Optional replacement closes the old trade before attempting the frozen new
    entry. Failed new entries leave the portfolio flat. Same-direction signals
    are no-ops. Replacement exits require observed opens and never reset an old
    expiry without an actual exit fill.
    """

    def __init__(self, config: ExecutionConfig):
        if not config.symbol or config.instrument_type not in {"equity", "etf"}:
            raise ValueError("One identified equity or ETF is required.")
        if not _positive_price(config.initial_capital):
            raise ValueError("Initial capital must be finite and positive.")
        if (isinstance(config.holding_sessions, bool)
                or not isinstance(config.holding_sessions, int) or config.holding_sessions < 1):
            raise ValueError("Holding sessions must be a positive integer.")
        if (not math.isfinite(config.fee_rate) or not 0 <= config.fee_rate < 1
                or not math.isfinite(config.allocation_fraction)
                or not 0 < config.allocation_fraction <= 1):
            raise ValueError("Require 0 <= fee_rate < 1 and 0 < allocation_fraction <= 1.")
        if (isinstance(config.slippage_rate, bool)
                or not isinstance(config.slippage_rate, (int, float))
                or not math.isfinite(config.slippage_rate) or not 0 <= config.slippage_rate < 1):
            raise ValueError("Require a finite slippage_rate with 0 <= slippage_rate < 1.")
        if not isinstance(config.allow_short, bool):
            raise ValueError("allow_short must be a boolean.")
        margin_values = (config.max_short_exposure, config.initial_margin_fraction,
                         config.maintenance_margin_fraction)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0 for value in margin_values)
                or config.max_short_exposure > 1
                or config.maintenance_margin_fraction > config.initial_margin_fraction):
            raise ValueError("Require 0 < max_short_exposure <= 1 and finite 0 < maintenance <= initial margin.")
        if config.position_policy not in {"fixed_horizon", "replace_next_open"}:
            raise ValueError("Unsupported position_policy.")
        if config.terminal_policy not in {"mark_to_market", "liquidate_at_final_close"}:
            raise ValueError("Unsupported terminal_policy.")
        if config.terminal_session_id is not None and (
                not isinstance(config.terminal_session_id, str) or not config.terminal_session_id.strip()):
            raise ValueError("terminal_session_id must be a non-empty session identifier.")
        if config.terminal_policy == "liquidate_at_final_close" and config.terminal_session_id is None:
            raise ValueError("Terminal liquidation requires an explicit terminal_session_id.")
        self.config = config

    def _terminal_window(self, bars: Sequence[MarketBar], signals: Sequence[Signal]
                         ) -> tuple[Sequence[MarketBar], Sequence[Signal]]:
        """Bound validation and execution without reading prices after the endpoint.

        Decisions after the declared final close are outside the simulation.
        Earlier-session signals retain normal timing validation; declaring an
        endpoint must not legitimize a delayed or otherwise invalid signal.
        """
        terminal_id = self.config.terminal_session_id
        if terminal_id is None:
            return bars, signals
        for index, bar in enumerate(bars):
            if bar.session_id == terminal_id:
                window = bars[:index + 1]
                cutoff = bar.close_timestamp
                if not _aware(cutoff):
                    raise ValueError("The terminal close requires a timezone-aware timestamp.")
                session_ids = {item.session_id for item in window}
                scoped_signals = [signal for signal in signals if not (
                    (signal.session_id == terminal_id or signal.session_id not in session_ids)
                    and _aware(signal.decision_timestamp) and _utc_instant(signal.decision_timestamp) > _utc_instant(cutoff))]
                return window, scoped_signals
        raise ValueError("Declared terminal_session_id is unavailable in the supplied calendar.")

    def _validate(self, bars: Sequence[MarketBar], signals: Sequence[Signal]) -> dict[str, Signal]:
        """Validate identifiers/timing; defer price use to the actual session event."""
        if not bars:
            raise ValueError("A non-empty complete market-session sequence is required.")
        indices: dict[str, int] = {}
        previous_close = None
        for index, bar in enumerate(bars):
            if bar.symbol != self.config.symbol or not bar.session_id or bar.session_id in indices:
                raise ValueError("Market bars require one symbol and unique non-empty session identifiers.")
            if (not _aware(bar.open_timestamp) or not _aware(bar.close_timestamp)
                    or _utc_instant(bar.open_timestamp) >= _utc_instant(bar.close_timestamp)
                    or (previous_close is not None and _utc_instant(previous_close) >= _utc_instant(bar.open_timestamp))):
                raise ValueError("Market sessions require chronological, non-overlapping aware timestamps.")
            indices[bar.session_id] = index
            previous_close = bar.close_timestamp
        by_session: dict[str, Signal] = {}
        for signal in signals:
            if signal.session_id not in indices or signal.session_id in by_session:
                raise ValueError("Signals require an existing unique feature-session identifier.")
            index = indices[signal.session_id]
            directions = {"long", "flat", "short"} if self.config.allow_short else {"long", "flat"}
            if (signal.direction not in directions or not _aware(signal.decision_timestamp)
                    or _utc_instant(signal.decision_timestamp) <= _utc_instant(bars[index].close_timestamp)
                    or (index + 1 < len(bars)
                        and _utc_instant(signal.decision_timestamp) >= _utc_instant(bars[index + 1].open_timestamp))):
                raise ValueError("Signals must have an enabled direction, after their close and before the next open.")
            by_session[signal.session_id] = signal
        return by_session

    def run(self, bars: Sequence[MarketBar], signals: Sequence[Signal]) -> ExecutionResult:
        """Return one initial reference and one closing ledger row per market session."""
        bars, signals = self._terminal_window(bars, signals)
        by_session = self._validate(bars, signals)
        terminal_close_timestamp = (bars[-1].close_timestamp
                                    if self.config.terminal_session_id is not None else None)
        terminal_order_id = None
        if self.config.terminal_policy == "liquidate_at_final_close":
            identity = json.dumps([self.config.symbol, self.config.terminal_session_id,
                                   terminal_close_timestamp.astimezone(timezone.utc).isoformat()])
            terminal_order_id = str(uuid5(NAMESPACE_URL, "apf:terminal_close_v1:" + identity))
        terminal_outcome = "not_reached"
        portfolio = Portfolio(amount(self.config.initial_capital))
        previous_equity = portfolio.cash
        ledger: list[dict] = []
        events: list[dict] = []
        fills: list[dict] = []
        pending: _PendingInstruction | None = None
        active: tuple[Signal, MarketBar, int] | None = None
        active_trade: TradeRecord | None = None
        scheduled_exit: tuple[str, int] | None = None
        completed_trades: list[TradeRecord] = []
        run_status = "completed"
        terminal_snapshot = None
        last_close_index = -1

        ledger.append({
            "session_id": None, "timestamp": bars[0].open_timestamp - timedelta(microseconds=1),
            "cash": float(portfolio.cash), "position_quantity": 0, "close_price": None,
            "equity": float(portfolio.cash), "daily_net_return": 0.0, "transaction_fees": 0.0,
            "realized_pnl": 0.0, "cumulative_realized_pnl": 0.0, "unrealized_pnl": 0.0,
            "is_initial": True,
            "gross_unrealized_pnl": 0.0, "gross_realized_pnl": 0.0,
            "cumulative_gross_realized_pnl": 0.0, "realized_costs": 0.0,
            "cumulative_realized_costs": 0.0, "net_realized_pnl": 0.0,
            "cumulative_net_realized_pnl": 0.0, "active_trade_id": None,
            "commission_costs": 0.0, "slippage_costs": 0.0, "cumulative_slippage_costs": 0.0,
            "total_execution_costs": 0.0, "cumulative_total_execution_costs": 0.0,
            "reference_gross_realized_pnl": 0.0, "cumulative_reference_gross_realized_pnl": 0.0,
            "reference_unrealized_pnl": 0.0, "reference_net_realized_pnl": 0.0,
            "cumulative_reference_net_realized_pnl": 0.0,
            "restricted_proceeds": 0.0, "unrestricted_cash": float(portfolio.cash),
            "short_liability": 0.0, "initial_margin_requirement": 0.0,
            "maintenance_requirement": 0.0, "margin_excess": float(portfolio.cash),
            "available_collateral": float(portfolio.cash),
        })

        def event(kind: str, timestamp: datetime, bar: MarketBar, signal: Signal | None, **details) -> None:
            events.append({"event": kind, "timestamp": timestamp, "session_id": bar.session_id,
                           "signal_session_id": signal.session_id if signal else None,
                           "signal_timestamp": signal.decision_timestamp if signal else None, **details})

        def mark_trade(price: float, timestamp: datetime, held_sessions: int,
                       phase: Literal["open", "close"]) -> None:
            nonlocal active_trade
            if active_trade is not None:
                active_trade = replace(
                    active_trade, unrealized_pnl=float(portfolio.gross_unrealized(price)),
                    net_pnl_to_date=float(portfolio.gross_unrealized(price) - portfolio.entry_fee),
                    holding_sessions=held_sessions, valuation_timestamp=timestamp, valuation_price=price,
                    reference_unrealized_pnl=float(portfolio.reference_unrealized(price)),
                    valuation_phase=phase,
                    holding_duration_seconds=(timestamp.astimezone(timezone.utc)
                                              - active_trade.entry_timestamp.astimezone(timezone.utc)).total_seconds(),
                )

        def snapshot(bar: MarketBar, price: float, timestamp: datetime,
                     phase: Literal["open", "close"]) -> RiskSnapshot:
            margin = portfolio.margin_values(price, self.config.initial_margin_fraction,
                                               self.config.maintenance_margin_fraction)
            return RiskSnapshot(
                session_id=bar.session_id, timestamp=timestamp, phase=phase, reference_price=price,
                position_quantity=portfolio.quantity, cash=float(portfolio.cash),
                restricted_proceeds=float(portfolio.restricted_proceeds),
                unrestricted_cash=float(portfolio.unrestricted_cash), short_liability=float(margin["short_liability"]),
                equity=float(portfolio.equity(price)), maintenance_requirement=float(margin["maintenance_requirement"]),
                margin_excess=float(margin["margin_excess"]),
                gross_unrealized_pnl=float(portfolio.gross_unrealized(price)),
                cumulative_net_realized_pnl=float(portfolio.net_realized_pnl),
                reference_unrealized_pnl=float(portfolio.reference_unrealized(price)),
                cumulative_reference_net_realized_pnl=float(portfolio.reference_net_realized_pnl),
                trade_id=(active_trade.trade_id if active_trade is not None
                          else completed_trades[-1].trade_id if completed_trades else None),
            )

        def close_trade(bar: MarketBar, index: int, reference: float, timestamp: datetime,
                        phase: Literal["open", "close"],
                        reason: Literal["scheduled", "signal_flat", "reversal", "terminal"],
                        trigger: Signal | None = None) -> None:
            """Close only the currently active trade through a separately charged fill."""
            nonlocal active, active_trade, scheduled_exit
            nonlocal fees, realized, gross_realized, reference_gross_realized
            if active is None or active_trade is None:
                raise RuntimeError("An actual exit requires its active entry trade record.")
            entry_signal, entry_bar, expiry = active
            quantity = portfolio.quantity
            exit_side = "buy" if quantity < 0 else "sell"
            exit_price = slipped_price(reference, self.config.slippage_rate, exit_side)
            trade_gross = quantity * (exit_price - portfolio.entry_price)
            gross_realized += trade_gross
            trade_reference_gross = portfolio.reference_unrealized(reference)
            reference_gross_realized += trade_reference_gross
            entry_fee = portfolio.entry_fee
            entry_slippage = portfolio.entry_slippage_cost
            exit_slippage = quantity * (amount(reference) - exit_price)
            fee, trade_realized = portfolio.exit(exit_price, self.config.fee_rate,
                                           reference_price=reference)
            fees += fee
            realized += trade_realized
            exit_fill_id = f"{active_trade.trade_id}:exit"
            fills.append({"side": exit_side, "session_id": bar.session_id,
                          "timestamp": timestamp, "quantity": abs(quantity),
                          "signed_quantity": -quantity,
                          "price": float(exit_price), "fee": float(fee),
                          "reference_price": reference,
                          "executed_price": float(exit_price), "commission_cost": float(fee),
                          "slippage_cost": float(exit_slippage),
                          "signal_timestamp": entry_signal.decision_timestamp,
                          "trigger_signal_timestamp": trigger.decision_timestamp if trigger else None,
                          "exit_reason": reason, "fill_sequence": len(fills),
                          "trade_id": active_trade.trade_id, "fill_id": exit_fill_id})
            completed_trades.append(replace(
                active_trade, status="completed", exit_session_id=bar.session_id,
                exit_timestamp=timestamp, exit_fill_id=exit_fill_id,
                exit_price=float(exit_price), exit_fee=float(fee),
                gross_realized_pnl=float(trade_gross), net_realized_pnl=float(trade_realized),
                unrealized_pnl=0.0, net_pnl_to_date=float(trade_realized),
                holding_sessions=index - (expiry - self.config.holding_sessions + 1) + (phase == "close"),
                holding_duration_seconds=(timestamp.astimezone(timezone.utc)
                                          - entry_bar.open_timestamp.astimezone(timezone.utc)).total_seconds(),
                valuation_phase=phase, exit_reason=reason,
                exit_signal_timestamp=trigger.decision_timestamp if trigger else None,
                exit_signal_session_id=trigger.session_id if trigger else None,
                valuation_timestamp=timestamp, valuation_price=reference,
                exit_reference_price=reference, exit_slippage_cost=float(exit_slippage),
                commission_costs=float(entry_fee + fee),
                slippage_costs=float(entry_slippage + exit_slippage),
                total_execution_costs=float(entry_fee + fee + entry_slippage + exit_slippage),
                reference_gross_realized_pnl=float(trade_reference_gross),
                reference_unrealized_pnl=0.0, reference_net_realized_pnl=float(trade_realized),
            ))
            kind = ("scheduled_exit_filled" if reason == "scheduled" else
                    "terminal_exit_filled" if reason == "terminal" else "replacement_exit_filled")
            event(kind, timestamp, bar, None if reason == "terminal" else trigger or entry_signal,
                  quantity=quantity,
                  trade_id=active_trade.trade_id,
                  **({"reason": reason} if reason != "scheduled" else {}))
            active = None
            active_trade = None
            scheduled_exit = None

        for index, bar in enumerate(bars):
            if portfolio.quantity < 0:
                if not bar.open_observed or not _positive_price(bar.open_price):
                    raise ValueError(f"Session {bar.session_id}: an observed positive opening risk price is required.")
                opening_margin = portfolio.margin_values(bar.open_price, self.config.initial_margin_fraction,
                                                          self.config.maintenance_margin_fraction)
                if opening_margin["margin_excess"] < 0 or portfolio.equity(bar.open_price) <= 0:
                    run_status = "insolvent" if portfolio.equity(bar.open_price) <= 0 else "margin_breach"
                    mark_trade(bar.open_price, bar.open_timestamp,
                               self.config.holding_sessions - (active[2] - index) - 1, "open")
                    terminal_snapshot = snapshot(bar, bar.open_price, bar.open_timestamp, "open")
                    event("margin_breach", bar.open_timestamp, bar, active[0], reason=run_status,
                          trade_id=active_trade.trade_id)
                    break
            fees = Decimal(0)
            realized = Decimal(0)
            gross_realized = Decimal(0)
            previous_net_realized = portfolio.net_realized_pnl
            previous_reference_net = portfolio.reference_net_realized_pnl
            previous_slippage = portfolio.slippage_costs
            reference_gross_realized = Decimal(0)
            if pending is not None:
                instruction = pending
                signal, quantity = instruction.signal, instruction.quantity
                pending = None
                if instruction.close_trade_id is not None:
                    if active_trade is None or active_trade.trade_id != instruction.close_trade_id:
                        raise RuntimeError("Pending replacement conflicts with the active trade.")
                    if not bar.open_observed or not _positive_price(bar.open_price):
                        raise ValueError(f"Session {bar.session_id}: an observed positive replacement execution open is required.")
                    close_trade(bar, index, bar.open_price, bar.open_timestamp, "open",
                                "signal_flat" if signal.direction == "flat" else "reversal", signal)
                    if portfolio.cash <= 0:
                        run_status = "insolvent"
                        terminal_snapshot = snapshot(bar, bar.open_price, bar.open_timestamp, "open")
                        event("insolvency", bar.open_timestamp, bar, signal,
                              reason="insolvent", trade_id=instruction.close_trade_id)
                        break
                if signal.direction == "flat":
                    quantity = 0
                entry_price = None
                if signal.direction == "flat":
                    pass
                elif quantity <= 0:
                    event("entry_rejected", bar.open_timestamp, bar, signal, reason="zero_quantity", quantity=quantity)
                elif not bar.open_observed or not _positive_price(bar.open_price):
                    event("entry_rejected", bar.open_timestamp, bar, signal,
                          reason="invalid_or_unobserved_open", quantity=quantity)
                else:
                    is_short = signal.direction == "short"
                    entry_side = "sell" if is_short else "buy"
                    entry_price = slipped_price(bar.open_price, self.config.slippage_rate, entry_side)
                    error = (portfolio.short_entry_error(
                        quantity, bar.open_price, entry_price, self.config.fee_rate,
                        self.config.max_short_exposure, self.config.initial_margin_fraction)
                        if is_short else None)
                    if error is not None or (not is_short and not portfolio.can_afford(
                            quantity, entry_price, self.config.fee_rate)):
                        event("entry_rejected", bar.open_timestamp, bar, signal,
                              reason=error or "unaffordable_open", quantity=quantity)
                        entry_price = None
                if entry_price is not None:
                    if is_short:
                        fee = portfolio.enter_short(
                            quantity, entry_price, self.config.fee_rate, reference_price=bar.open_price,
                            max_exposure=self.config.max_short_exposure,
                            initial_margin=self.config.initial_margin_fraction)
                    else:
                        fee = portfolio.enter(quantity, entry_price, self.config.fee_rate,
                                              reference_price=bar.open_price)
                    fees += fee
                    active = (signal, bar, index + self.config.holding_sessions - 1)
                    trade_id = _trade_id(self.config.symbol, signal)
                    scheduled_exit = (trade_id, active[2])
                    active_trade = TradeRecord(
                        trade_id=trade_id, symbol=self.config.symbol,
                        signal_session_id=signal.session_id, signal_timestamp=signal.decision_timestamp,
                        entry_session_id=bar.session_id, entry_timestamp=bar.open_timestamp,
                        entry_fill_id=f"{trade_id}:entry", quantity=portfolio.quantity,
                        entry_price=float(entry_price), entry_fee=float(fee),
                        net_realized_pnl=-float(fee), net_pnl_to_date=-float(fee),
                        entry_reference_price=bar.open_price,
                        entry_slippage_cost=float(portfolio.entry_slippage_cost),
                        commission_costs=float(fee), slippage_costs=float(portfolio.entry_slippage_cost),
                        total_execution_costs=float(fee + portfolio.entry_slippage_cost),
                        reference_net_realized_pnl=float(-fee - portfolio.entry_slippage_cost),
                        direction="short" if is_short else "long",
                    )
                    fills.append({"side": entry_side, "session_id": bar.session_id,
                                  "timestamp": bar.open_timestamp, "quantity": quantity,
                                  "signed_quantity": portfolio.quantity,
                                  "price": float(entry_price), "fee": float(fee),
                                  "reference_price": bar.open_price,
                                  "executed_price": float(entry_price), "commission_cost": float(fee),
                                  "slippage_cost": float(portfolio.entry_slippage_cost),
                                  "signal_timestamp": signal.decision_timestamp,
                                  "trade_id": trade_id, "fill_id": active_trade.entry_fill_id,
                                  "trigger_signal_timestamp": signal.decision_timestamp,
                                  "exit_reason": None, "fill_sequence": len(fills)})
                    event("entry_filled", bar.open_timestamp, bar, signal, quantity=quantity,
                          expiry_session_ordinal=active[2], trade_id=trade_id)

            if not bar.close_observed or not _positive_price(bar.close_price):
                raise ValueError(f"Session {bar.session_id}: an observed finite positive valuation close is required.")
            short_at_close = portfolio.quantity < 0
            if scheduled_exit is not None and index == scheduled_exit[1]:
                if active_trade is None or scheduled_exit[0] != active_trade.trade_id:
                    raise RuntimeError("Scheduled expiry conflicts with the active trade.")
                signal = active[0]
                close_trade(bar, index, bar.close_price, bar.close_timestamp, "close", "scheduled")

            if (index == len(bars) - 1 and active is not None
                    and self.config.terminal_policy == "liquidate_at_final_close"):
                signal = active[0]
                close_trade(bar, index, bar.close_price, bar.close_timestamp, "close", "terminal")
                terminal_outcome = "liquidated"

            equity = portfolio.equity(bar.close_price)
            if not equity.is_finite() or (equity <= 0 and not short_at_close):
                raise ValueError("Non-positive or non-finite portfolio equity is unsupported.")
            if active_trade is not None and active is not None:
                mark_trade(bar.close_price, bar.close_timestamp,
                           self.config.holding_sessions - (active[2] - index), "close")
            margin = portfolio.margin_values(bar.close_price, self.config.initial_margin_fraction,
                                              self.config.maintenance_margin_fraction)
            ledger.append({
                "session_id": bar.session_id, "timestamp": bar.close_timestamp,
                "cash": float(portfolio.cash), "position_quantity": portfolio.quantity,
                "close_price": bar.close_price, "equity": float(equity),
                "daily_net_return": float(equity / previous_equity - 1),
                "transaction_fees": float(fees), "realized_pnl": float(realized),
                "cumulative_realized_pnl": float(portfolio.realized_pnl),
                "unrealized_pnl": float(portfolio.unrealized(bar.close_price)), "is_initial": False,
                "gross_unrealized_pnl": float(portfolio.gross_unrealized(bar.close_price)),
                "gross_realized_pnl": float(gross_realized),
                "cumulative_gross_realized_pnl": float(portfolio.gross_realized_pnl),
                "realized_costs": float(fees), "cumulative_realized_costs": float(portfolio.realized_costs),
                "net_realized_pnl": float(portfolio.net_realized_pnl - previous_net_realized),
                "cumulative_net_realized_pnl": float(portfolio.net_realized_pnl),
                "active_trade_id": active_trade.trade_id if active_trade is not None else None,
                "commission_costs": float(fees),
                "slippage_costs": float(portfolio.slippage_costs - previous_slippage),
                "cumulative_slippage_costs": float(portfolio.slippage_costs),
                "total_execution_costs": float(fees + portfolio.slippage_costs - previous_slippage),
                "cumulative_total_execution_costs": float(portfolio.realized_costs + portfolio.slippage_costs),
                "reference_gross_realized_pnl": float(reference_gross_realized),
                "cumulative_reference_gross_realized_pnl": float(portfolio.reference_gross_realized_pnl),
                "reference_unrealized_pnl": float(portfolio.reference_unrealized(bar.close_price)),
                "reference_net_realized_pnl": float(portfolio.reference_net_realized_pnl - previous_reference_net),
                "cumulative_reference_net_realized_pnl": float(portfolio.reference_net_realized_pnl),
                **{key: float(value) for key, value in margin.items()},
            })
            last_close_index = index
            previous_equity = equity
            if index == len(bars) - 1 and terminal_outcome != "liquidated":
                terminal_outcome = "marked" if active is not None else "already_flat"
            if short_at_close and (equity <= 0 or (portfolio.quantity < 0 and margin["margin_excess"] < 0)):
                run_status = "insolvent" if equity <= 0 else "margin_breach"
                terminal_snapshot = snapshot(bar, bar.close_price, bar.close_timestamp, "close")
                event("insolvency" if equity <= 0 else "margin_breach", bar.close_timestamp,
                      bar, signal if active is None else active[0], reason=run_status,
                      trade_id=active_trade.trade_id if active_trade is not None else completed_trades[-1].trade_id)
                break
            if self.config.terminal_session_id is not None and index == len(bars) - 1:
                break
            signal = by_session.get(bar.session_id)
            if signal is not None:
                if pending is not None:
                    raise ValueError("Conflicting pending instructions are unsupported.")
                if active is not None and self.config.position_policy == "fixed_horizon":
                    event("signal_ignored", signal.decision_timestamp, bar, signal, reason="fixed_horizon_active")
                elif active_trade is not None and signal.direction == active_trade.direction:
                    event("signal_ignored", signal.decision_timestamp, bar, signal, reason="same_direction")
                elif active is None and signal.direction == "flat":
                    event("signal_flat", signal.decision_timestamp, bar, signal)
                else:
                    sizing = Portfolio(portfolio.estimated_flat_cash(
                        bar.close_price, self.config.fee_rate, self.config.slippage_rate))
                    quantity = 0
                    if signal.direction != "flat" and sizing.cash > 0:
                        if signal.direction == "short":
                            quantity = sizing.size_short_order(
                                bar.close_price, self.config.allocation_fraction, self.config.fee_rate,
                                self.config.slippage_rate, self.config.max_short_exposure,
                                self.config.initial_margin_fraction)
                        else:
                            quantity = sizing.size_order(bar.close_price, self.config.allocation_fraction,
                                                         self.config.fee_rate, self.config.slippage_rate)
                    if quantity == 0 and active is None:
                        event("entry_rejected", signal.decision_timestamp, bar, signal, reason="zero_quantity")
                    else:
                        pending = _PendingInstruction(signal, quantity,
                                                      active_trade.trade_id if active_trade else None)
                        event("replacement_scheduled" if active is not None else "entry_scheduled",
                              signal.decision_timestamp, bar, signal, quantity=quantity,
                              **({"trade_id": pending.close_trade_id} if active is not None else {}))

        if pending is not None:
            signal, quantity = pending.signal, pending.quantity
            if run_status != "completed":
                event("order_blocked_by_risk", terminal_snapshot.timestamp,
                      bars[last_close_index + 1], signal, quantity=quantity, reason=run_status)
            else:
                event("replacement_unfilled_at_end" if pending.close_trade_id else "entry_unfilled_at_end",
                      signal.decision_timestamp, bars[-1], signal, quantity=quantity)
        open_position = None
        if active is not None:
            signal, entry_bar, expiry = active
            open_position = OpenPosition(
                portfolio.quantity, entry_bar.session_id, entry_bar.open_timestamp,
                float(portfolio.entry_price), signal.decision_timestamp, expiry - last_close_index,
                trade_id=active_trade.trade_id if active_trade is not None else None,
                entry_reference_price=float(portfolio.entry_reference_price),
                entry_slippage_cost=float(portfolio.entry_slippage_cost),
            )
        return ExecutionResult(
            pd.DataFrame(ledger),
            pd.DataFrame(events, columns=["event", "timestamp", "session_id", "signal_session_id",
                                         "signal_timestamp", "reason", "quantity", "expiry_session_ordinal",
                                         "trade_id"]),
            pd.DataFrame(fills, columns=["side", "session_id", "timestamp", "quantity", "price",
                                        "fee", "signal_timestamp", "trade_id", "fill_id",
                                        "reference_price", "executed_price", "commission_cost", "slippage_cost",
                                        "signed_quantity", "trigger_signal_timestamp", "exit_reason",
                                        "fill_sequence"]),
            open_position,
            completed_trades=pd.DataFrame([asdict(trade) for trade in completed_trades],
                                          columns=[field.name for field in fields(TradeRecord)]),
            open_trades=pd.DataFrame([asdict(active_trade)] if active_trade is not None else [],
                                     columns=[field.name for field in fields(TradeRecord)]),
            run_status=run_status, terminal_snapshot=terminal_snapshot,
            terminal_policy=self.config.terminal_policy,
            terminal_session_id=self.config.terminal_session_id,
            effective_end_session_id=(terminal_snapshot.session_id if terminal_snapshot else
                                      bars[last_close_index].session_id),
            terminal_close_timestamp=terminal_close_timestamp,
            terminal_outcome=terminal_outcome, terminal_order_id=terminal_order_id,
            requested_end_session_id=bars[-1].session_id,
        )

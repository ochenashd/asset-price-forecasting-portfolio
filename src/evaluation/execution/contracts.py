"""Explicit contracts for a single-asset, observed-price execution simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    """Whole-share orders use the signal close, configured slippage and commissions.

    Inputs must cover consecutive sessions from the caller's exchange calendar.
    This engine supports corporate-action-free equity/ETF prices in one currency.
    No adjusted/filled price reconstruction or automatic calendar inference occurs.
    Shorts are disabled by default. Opting in assumes stock can be borrowed at
    zero borrow/funding cost, with no dividends or corporate actions. The short
    exposure cap applies to post-cost equity at entry, not subsequent price moves.
    Initial/maintenance fractions are explicit simulation parameters.
    replace_next_open closes an opposite/Flat position at the following open,
    then independently attempts a decision-sized entry. Same-direction signals
    never resize positions or extend expiry. Reversal legs assume sequential full
    fills at the same opening reference, with separate fees and slippage.
    Terminal liquidation is precommitted before simulation and requires an
    explicit session identifier. A declared endpoint also bounds marking runs;
    eligible entries at its open remain allowed. No endpoint means the existing
    supplied-window marking behavior, never an inferred liquidation date.
    """

    symbol: str
    initial_capital: float = 100_000.0
    holding_sessions: int = 5
    fee_rate: float = 0.001
    allocation_fraction: float = 1.0
    instrument_type: Literal["equity", "etf"] = "equity"
    slippage_rate: float = 0.0
    allow_short: bool = False
    max_short_exposure: float = 1.0
    initial_margin_fraction: float = 0.50
    maintenance_margin_fraction: float = 0.30
    position_policy: Literal["fixed_horizon", "replace_next_open"] = "fixed_horizon"
    terminal_policy: Literal["mark_to_market", "liquidate_at_final_close"] = "mark_to_market"
    terminal_session_id: Optional[str] = None


@dataclass(frozen=True)
class MarketBar:
    """An observed session with timezone-aware opening and closing timestamps.

    Missing/invalid opens cancel orders only when an opening execution is needed.
    A carried short also requires an observed opening risk mark. Every close
    reached by the simulation must be observed, finite and positive, including
    flat sessions. No later prices are inspected after a margin stop.
    Provenance flags must come from the source; numeric prices alone cannot prove
    that upstream data has not been filled. High/low/volume are not execution inputs.
    """

    symbol: str
    session_id: str
    open_timestamp: datetime
    close_timestamp: datetime
    open_price: Optional[float]
    close_price: Optional[float]
    open_observed: bool = True
    close_observed: bool = True
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None


@dataclass(frozen=True)
class Signal:
    """An after-close decision referencing its feature session by identifier.

    Accepts one signal per session, strictly after its close and before
    the following open. Delayed/intraday signals and stale order retries are not
    supported. Flat signals do not interrupt an active fixed-horizon position;
    under replace_next_open they request an actual exit at the following open.
    """

    session_id: str
    decision_timestamp: datetime
    direction: Literal["long", "flat", "short"]

    @classmethod
    def from_class(cls, session_id: str, decision_timestamp: datetime, label: int) -> "Signal":
        """Map integer class 0 to Short, 1 to Flat and 2 to Long; shorting is opt-in."""
        from numbers import Integral

        if isinstance(label, bool) or not isinstance(label, Integral) or label not in {0, 1, 2}:
            raise ValueError("Class labels must be integers 0, 1 or 2.")
        return cls(session_id, decision_timestamp, {0: "short", 1: "flat", 2: "long"}[label])


@dataclass(frozen=True)
class OpenPosition:
    """A position retained at end of data without fabricated liquidation prices."""

    quantity: int
    entry_session_id: str
    entry_timestamp: datetime
    entry_price: float
    signal_timestamp: datetime
    holding_sessions_remaining: int
    trade_id: Optional[str] = None
    entry_reference_price: Optional[float] = None
    entry_slippage_cost: float = 0.0


@dataclass(frozen=True)
class TradeRecord:
    """One actual entry and optional actual exit, with separately allocated fees.

    Entry/exit prices and gross_realized_pnl use actual executed prices.
    Open unrealized_pnl is the executed-price mark; net_realized_pnl is minus
    incurred entry commissions. Reference-price P&L excludes slippage and is
    reconciled through separately reported analytical slippage costs. These costs
    are already embedded in fills and must never be debited from cash again.
    net_pnl_to_date always sums executed realized and unrealized P&L.
    holding_sessions counts held session closes: an opening exit excludes that
    session, while a closing exit includes it. holding_duration_seconds reports
    actual elapsed time, including nights and non-trading days.
    """

    trade_id: str
    symbol: str
    signal_session_id: str
    signal_timestamp: datetime
    entry_session_id: str
    entry_timestamp: datetime
    entry_fill_id: str
    quantity: int
    entry_price: float
    entry_fee: float
    status: Literal["open", "completed"] = "open"
    exit_session_id: Optional[str] = None
    exit_timestamp: Optional[datetime] = None
    exit_fill_id: Optional[str] = None
    exit_price: Optional[float] = None
    exit_fee: float = 0.0
    gross_realized_pnl: float = 0.0
    net_realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    net_pnl_to_date: float = 0.0
    holding_sessions: int = 0
    valuation_timestamp: Optional[datetime] = None
    valuation_price: Optional[float] = None
    entry_reference_price: Optional[float] = None
    exit_reference_price: Optional[float] = None
    entry_slippage_cost: float = 0.0
    exit_slippage_cost: float = 0.0
    commission_costs: float = 0.0
    slippage_costs: float = 0.0
    total_execution_costs: float = 0.0
    reference_gross_realized_pnl: float = 0.0
    reference_unrealized_pnl: float = 0.0
    reference_net_realized_pnl: float = 0.0
    direction: Literal["long", "short"] = "long"
    valuation_phase: Literal["open", "close"] = "close"
    exit_reason: Optional[Literal["scheduled", "signal_flat", "reversal", "terminal"]] = None
    exit_signal_timestamp: Optional[datetime] = None
    exit_signal_session_id: Optional[str] = None
    holding_duration_seconds: float = 0.0


@dataclass(frozen=True)
class RiskSnapshot:
    """Terminal observed mark; an opening snapshot is not a daily closing record."""

    session_id: str
    timestamp: datetime
    phase: Literal["open", "close"]
    reference_price: float
    position_quantity: int
    cash: float
    restricted_proceeds: float
    unrestricted_cash: float
    short_liability: float
    equity: float
    maintenance_requirement: float
    margin_excess: float
    gross_unrealized_pnl: float
    cumulative_net_realized_pnl: float
    reference_unrealized_pnl: float
    cumulative_reference_net_realized_pnl: float
    trade_id: Optional[str]


@dataclass(frozen=True)
class ExecutionResult:
    """Daily records and trades; no legacy performance metrics.

    Stage 1 ledger columns are preserved: unrealized_pnl is the net open-trade
    mark, and realized_pnl/cumulative_realized_pnl recognize net completed trades.
    The explicit Stage 2A columns recognize fees immediately: gross_unrealized_pnl,
    gross_realized_pnl, realized_costs, net_realized_pnl and their cumulative fields.
    Equity = initial capital + cumulative_net_realized_pnl + gross_unrealized_pnl.
    Trade-table unrealized_pnl is gross, not the Stage 1 compatibility column.
    Existing P&L fields use executed entry/exit prices and realized_costs remains
    commissions only. Analytical slippage_costs is never another cash debit.
    The reference view also reconciles equity: initial + cumulative_reference_net_
    realized_pnl + reference_unrealized_pnl. Its realized costs include slippage
    at each fill, while its marks exclude entry slippage. total_execution_costs
    sums commissions and analytical slippage for attribution, not cash accounting.
    run_status="completed" means the supplied window finished, not that every
    trade closed. A risk stop is explicitly incomplete: opening terminal_snapshot
    equity is the latest mark and is not appended as a fictitious closing row.
    Insolvency preserves non-positive equity or a scheduled-cover cash deficit;
    there is no capital injection, liquidation assumption or subsequent financing.
    Terminal metadata records the precommitted policy separately from run_status:
    liquidated means an actual terminal fill, which can still leave insolvency.
    not_reached means the window's closing endpoint was not reached before a
    risk stop. effective_end_session_id includes an opening risk-stop session;
    terminal_close_timestamp resolves only an explicitly declared endpoint.
    """

    ledger: pd.DataFrame
    events: pd.DataFrame
    fills: pd.DataFrame
    open_position: Optional[OpenPosition]
    engine_version: str = "execution_daily_v1"
    completed_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    open_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_ledger_version: str = "execution_trade_ledger_v1"
    cost_model_version: str = "adverse_proportional_slippage_v1"
    run_status: Literal["completed", "margin_breach", "insolvent"] = "completed"
    terminal_snapshot: Optional[RiskSnapshot] = None
    margin_model_version: str = "short_collateral_v1"
    terminal_policy: Literal["mark_to_market", "liquidate_at_final_close"] = "mark_to_market"
    terminal_session_id: Optional[str] = None
    effective_end_session_id: Optional[str] = None
    terminal_close_timestamp: Optional[datetime] = None
    terminal_outcome: Literal["marked", "liquidated", "already_flat", "not_reached"] = "not_reached"
    terminal_order_id: Optional[str] = None
    requested_end_session_id: Optional[str] = None

"""Read-only reporting from executed daily equity, completed trades and fills.

Returns include incurred commissions and slippage through equity; neither cost
is subtracted again. Initial capital participates in return and drawdown math,
but its synthetic zero return is not a daily return observation. Calendar CAGR
uses the actual timestamps of that initial reference and the last daily close.
Daily closing marks cannot measure intraday drawdown or intraday peak exposure.

The caller declares equal-session annualization; exchange holidays/weekends do
not become extra return observations. Square-root annualization assumes regular
session sampling and does not adjust for serial dependence. No trade/target
return series, benchmark or artificial denominator epsilon is used.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .contracts import ExecutionResult, RiskSnapshot


@dataclass(frozen=True)
class PerformanceConfig:
    """Explicit session frequency, effective annual risk-free rate and year length.

    For example, periods_per_year=252 declares one observation per trading day.
    calendar_days_per_year=365.25 is a calendar-year convention, not a substitute
    for the supplied timestamps. The engine relies on the caller's full calendar;
    this reporter cannot independently certify its completeness or frequency.
    """

    periods_per_year: float
    annual_risk_free_rate: float = 0.0
    calendar_days_per_year: float = 365.25

    def __post_init__(self) -> None:
        for name in ("periods_per_year", "annual_risk_free_rate", "calendar_days_per_year"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite numeric value.")
        if self.periods_per_year <= 0 or self.calendar_days_per_year <= 0:
            raise ValueError("Annualization frequency and calendar year length must be positive.")
        if self.annual_risk_free_rate <= -1:
            raise ValueError("The effective annual risk-free rate must exceed -1.")


@dataclass(frozen=True)
class RunCompleteness:
    """Window eligibility is descriptive, not evidence of statistical significance.

    Daily portfolio metrics end at valuation_end_timestamp. An opening risk
    snapshot is preserved separately and never inserted as a daily close.
    Turnover includes all actual fills, even exits after the last daily valuation
    during a risk stop. valuation_observations includes initial capital; daily_
    return_observations excludes it. A failed run remains ineligible for full-
    window comparison even if insolvency occurs on the requested final close.
    """

    run_status: str
    requested_end_session_id: str | None
    effective_end_session_id: str | None
    valuation_end_session_id: str | None
    valuation_end_timestamp: pd.Timestamp
    requested_end_reached: bool
    requested_window_completed: bool
    full_window_comparison_eligible: bool
    descriptive_only: bool
    valuation_observations: int
    daily_return_observations: int
    terminal_policy: str
    terminal_outcome: str
    terminal_risk_snapshot: RiskSnapshot | None
    terminal_risk_total_return: float
    fills_after_last_valuation: int


@dataclass(frozen=True)
class ExecutionPerformanceReport:
    """Standalone report; execution results and their status fields are unmodified."""

    portfolio_metrics: dict[str, float]
    trade_metrics: dict[str, float | int]
    turnover_metrics: dict[str, float]
    completeness: RunCompleteness
    assumptions: PerformanceConfig


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Require finite numeric inputs instead of silently dropping observations."""
    if column not in frame or not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
        raise ValueError(f"{column} must contain finite numeric observations.")
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite numeric observations.")
    return values


def _timestamps(frame: pd.DataFrame, column: str) -> list[pd.Timestamp]:
    """Reject missing or naive timestamps; never silently assign a timezone."""
    if column not in frame:
        raise ValueError(f"Missing {column} timestamps.")
    values = [pd.Timestamp(value) for value in frame[column]]
    if any(pd.isna(value) or value.tzinfo is None for value in values):
        raise ValueError(f"{column} must contain timezone-aware timestamps.")
    return values


def _trade_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    """Gross profit/loss aggregate positive/negative NET completed-trade P&L.

    Gross_Loss is a positive loss magnitude. Breakeven trades count in the win-
    rate denominator but are not wins. No-loss Profit Factor is undefined (NaN),
    including when profitable trades exist. Open trades never enter this table.
    Average duration is elapsed seconds; held session closes are also reported.
    """
    count = len(trades)
    metrics = {"Number_of_Completed_Trades": count, "Trade_Win_Rate": math.nan,
               "Gross_Profit": 0.0, "Gross_Loss": 0.0, "Profit_Factor": math.nan,
               "Average_Net_Trade_PnL": math.nan, "Average_Holding_Duration_Seconds": math.nan,
               "Average_Holding_Sessions": math.nan}
    if not count:
        return metrics
    if "status" not in trades or not trades["status"].eq("completed").all():
        raise ValueError("Completed-trade reporting requires only completed records.")
    pnl = _numeric(trades, "net_realized_pnl")
    sessions = _numeric(trades, "holding_sessions")
    entries = _timestamps(trades, "entry_timestamp")
    exits = _timestamps(trades, "exit_timestamp")
    durations = [(exit_time - entry).total_seconds() for entry, exit_time in zip(entries, exits)]
    if any(value < 0 for value in durations) or (sessions < 0).any() or (sessions != np.floor(sessions)).any():
        raise ValueError("Trade holding durations and session counts must be non-negative.")
    profit = math.fsum(pnl[pnl > 0])
    loss = -math.fsum(pnl[pnl < 0])
    metrics.update({"Trade_Win_Rate": float(np.count_nonzero(pnl > 0) / count),
                    "Gross_Profit": profit, "Gross_Loss": loss,
                    "Profit_Factor": profit / loss if loss > 0 else math.nan,
                    "Average_Net_Trade_PnL": math.fsum(pnl) / count,
                    "Average_Holding_Duration_Seconds": math.fsum(durations) / count,
                    "Average_Holding_Sessions": math.fsum(sessions) / count})
    return metrics


def calculate_performance(result: ExecutionResult, config: PerformanceConfig) -> ExecutionPerformanceReport:
    """Describe daily valuations without altering any input or executing orders.

    Average_Gross_Exposure is mean abs(quantity)*close/equity over daily closes,
    excluding the initial reference; any non-positive daily equity makes it
    undefined. Time_in_Market is the fraction of VALUED sessions with carried
    exposure, closing exposure or an actual fill. Thus H=1 trades count even
    when flat at the close. This is session activity, not elapsed invested hours.

    Turnover is two-sided cumulative reference notional / INITIAL equity, with
    no division by two or annualization. An entry and its exit count separately;
    reversal legs are never netted. No rejected order contributes. An opening
    risk mark is disclosed separately, not blended into daily risk statistics.
    """
    ledger = result.ledger
    if ledger.empty or "is_initial" not in ledger or "session_id" not in ledger:
        raise ValueError("A daily ledger with its initial-capital reference is required.")
    if (not pd.api.types.is_bool_dtype(ledger["is_initial"])
            or not ledger["is_initial"].iloc[0] or ledger["is_initial"].iloc[1:].any()):
        raise ValueError("Exactly one initial-capital reference must be the first row.")
    timestamps = _timestamps(ledger, "timestamp")
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Valuation timestamps must be strictly increasing.")
    equity = _numeric(ledger, "equity")
    quantities = _numeric(ledger, "position_quantity")
    if equity[0] <= 0 or (equity[:-1] <= 0).any() or quantities[0] != 0:
        raise ValueError("Initial and prior valuation equity must be positive, with an initial flat position.")
    if (quantities != np.floor(quantities)).any():
        raise ValueError("Portfolio quantities must be signed whole shares.")
    daily = ledger.iloc[1:]
    if daily["session_id"].isna().any() or daily["session_id"].duplicated().any():
        raise ValueError("Daily valuations require unique session identifiers.")
    count = len(daily)
    returns = equity[1:] / equity[:-1] - 1
    if not np.isfinite(returns).all():
        raise ValueError("Consecutive equity observations produce non-finite returns.")
    volatility = sharpe = sortino = math.nan
    if count:
        periodic_rf = math.expm1(math.log1p(config.annual_risk_free_rate) / config.periods_per_year)
        excess = returns - periodic_rf
        mean_excess = math.fsum(excess) / count
        mean_return = math.fsum(returns) / count
        # Identical values have exactly zero variance, even if summing/dividing
        # their floating representation rounds the computed mean. No tolerance
        # or epsilon is applied to genuinely varying observations.
        variance = (0.0 if np.all(returns == returns[0]) else
                    math.fsum((value - mean_return) ** 2 for value in returns) / count)
        excess_variance = (0.0 if np.all(excess == excess[0]) else
                           math.fsum((value - mean_excess) ** 2 for value in excess) / count)
        downside = math.sqrt(math.fsum(min(value, 0.0) ** 2 for value in excess) / count)
        volatility = math.sqrt(variance * config.periods_per_year)
        if excess_variance > 0:
            sharpe = mean_excess * math.sqrt(config.periods_per_year / excess_variance)
        if downside > 0:
            sortino = mean_excess * math.sqrt(config.periods_per_year) / downside
    elapsed_years = (timestamps[-1] - timestamps[0]).total_seconds() / (86400 * config.calendar_days_per_year)
    cagr = math.nan
    if elapsed_years > 0 and equity[-1] >= 0:
        if equity[-1] == 0:
            cagr = -1.0
        else:
            try:
                cagr = math.expm1(math.log(equity[-1] / equity[0]) / elapsed_years)
            except OverflowError:
                cagr = math.nan
    exposure = time_in_market = math.nan
    fills = result.fills
    fill_sessions: set[str] = set()
    traded_notional = 0.0
    fills_after_close = 0
    if not fills.empty:
        magnitude = _numeric(fills, "quantity")
        reference = _numeric(fills, "reference_price")
        if (magnitude <= 0).any() or (reference <= 0).any() or (magnitude != np.floor(magnitude)).any():
            raise ValueError("Fills require positive whole-share quantities and reference prices.")
        fill_times = _timestamps(fills, "timestamp")
        if any(timestamp <= timestamps[0] for timestamp in fill_times):
            raise ValueError("Fills must occur after the initial-capital reference.")
        traded_notional = math.fsum(magnitude * reference)
        fill_sessions = set(fills["session_id"])
        fills_after_close = sum(timestamp > timestamps[-1] for timestamp in fill_times)
    if count:
        closes = _numeric(daily, "close_price")
        if (closes <= 0).any():
            raise ValueError("Daily reference closes must be positive.")
        if (equity[1:] > 0).all():
            exposure = math.fsum(np.abs(quantities[1:]) * closes / equity[1:]) / count
        active = [(quantities[i] != 0 or quantities[i + 1] != 0 or session in fill_sessions)
                  for i, session in enumerate(daily["session_id"])]
        time_in_market = sum(active) / count
    portfolio_metrics = {
        "Total_Return": float(equity[-1] / equity[0] - 1), "CAGR": cagr,
        "Annualized_Volatility": volatility, "Sharpe_Ratio": sharpe, "Sortino_Ratio": sortino,
        "Max_Drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "Average_Gross_Exposure": exposure, "Time_in_Market": time_in_market,
    }
    valuation_session = daily["session_id"].iloc[-1] if count else None
    requested = result.requested_end_session_id or result.terminal_session_id
    if requested is None and result.run_status == "completed":
        requested = valuation_session
    end_reached = requested is not None and valuation_session == requested
    completed = bool(end_reached and result.run_status == "completed")
    eligible = bool(completed and (equity > 0).all())
    snapshot = result.terminal_snapshot
    completeness = RunCompleteness(
        run_status=result.run_status, requested_end_session_id=requested,
        effective_end_session_id=result.effective_end_session_id,
        valuation_end_session_id=valuation_session, valuation_end_timestamp=timestamps[-1],
        requested_end_reached=bool(end_reached), requested_window_completed=completed,
        full_window_comparison_eligible=eligible, descriptive_only=not eligible,
        valuation_observations=len(ledger), daily_return_observations=count,
        terminal_policy=result.terminal_policy, terminal_outcome=result.terminal_outcome,
        terminal_risk_snapshot=snapshot,
        terminal_risk_total_return=(snapshot.equity / equity[0] - 1 if snapshot else math.nan),
        fills_after_last_valuation=fills_after_close,
    )
    return ExecutionPerformanceReport(
        portfolio_metrics=portfolio_metrics, trade_metrics=_trade_metrics(result.completed_trades),
        turnover_metrics={"Traded_Reference_Notional": traded_notional,
                          "Turnover": traded_notional / equity[0],
                          "Turnover_Normalization_Equity": float(equity[0])},
        completeness=completeness, assumptions=config,
    )

"""Independent arithmetic references for daily, trade and run-level reporting."""

from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal
import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.execution import ExecutionConfig, ExecutionEngineV1, ExecutionResult, MarketBar, Signal
from src.evaluation.execution.metrics import PerformanceConfig, calculate_performance


DAILY = PerformanceConfig(periods_per_year=252)


def ledger_result(returns, *, start="2020-01-01", step_days=1):
    """Build a valuation contract from independently compounded Decimal wealth."""
    wealth = Decimal(1024)
    values = [float(wealth)]
    for change in returns:
        wealth *= 1 + Decimal(str(change))
        values.append(float(wealth))
    dates = pd.date_range(start, periods=len(values), freq=f"{step_days}D", tz="UTC")
    sessions = [None] + [f"session-{i}" for i in range(len(returns))]
    ledger = pd.DataFrame({"session_id": sessions, "timestamp": dates,
                           "equity": values, "position_quantity": [0] * len(values),
                           "close_price": [np.nan] + [100.0] * len(returns),
                           "is_initial": [True] + [False] * len(returns)})
    return ExecutionResult(ledger=ledger, events=pd.DataFrame(), fills=pd.DataFrame(),
                           open_position=None, requested_end_session_id=sessions[-1],
                           effective_end_session_id=sessions[-1])


def reference_risk(returns, frequency, annual_rf):
    """Scalar reference using arithmetic returns and all observations for downside."""
    periodic_rf = math.expm1(math.log1p(annual_rf) / frequency)
    excess = [r - periodic_rf for r in returns]
    average = sum(excess) / len(excess)
    standard_deviation = float(np.std(excess, ddof=0))
    downside = math.sqrt(sum(min(r, 0) ** 2 for r in excess) / len(excess))
    return (average / standard_deviation * math.sqrt(frequency) if standard_deviation else math.nan,
            average / downside * math.sqrt(frequency) if downside else math.nan)


def assert_number(actual, expected):
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("returns", [
    [0.0] * 4,
    [0.25] * 4,
    [-0.5] * 4,
    [0.125, -0.0625] * 3,
    [0.0625, -0.125, 0.25, -0.0625],
])
@pytest.mark.parametrize("rf", [0.0, 0.05, -0.01])
def test_daily_risk_metrics_match_independent_reference(returns, rf):
    result = ledger_result(returns)
    config = PerformanceConfig(periods_per_year=252, annual_risk_free_rate=rf)
    metrics = calculate_performance(result, config).portfolio_metrics
    sharpe, sortino = reference_risk(returns, 252, rf)
    assert_number(metrics["Sharpe_Ratio"], sharpe)
    assert_number(metrics["Sortino_Ratio"], sortino)
    assert metrics["Annualized_Volatility"] == pytest.approx(np.std(returns, ddof=0) * math.sqrt(252))
    expected = math.prod(1 + r for r in returns) - 1
    assert metrics["Total_Return"] == pytest.approx(expected)


@pytest.mark.parametrize("returns,expected", [
    ([-0.1], -0.1), ([-0.1, -0.1, -0.1], -0.271),
    ([-0.5, 1.0], -0.5), ([0.25, -0.5, 1.0], -0.5),
    ([0, 0, 0], 0), ([0.125] * 3, 0),
])
def test_drawdown_includes_initial_wealth_and_recovery(returns, expected):
    metrics = calculate_performance(ledger_result(returns), DAILY).portfolio_metrics
    assert metrics["Max_Drawdown"] == pytest.approx(expected)


def test_cagr_uses_actual_elapsed_calendar_time_including_irregular_gaps():
    result = ledger_result([0.25, -0.1, 0.2])
    result.ledger["timestamp"] = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-06-01", "2021-01-01"], utc=True)
    config = PerformanceConfig(252, calendar_days_per_year=365.25)
    metrics = calculate_performance(result, config).portfolio_metrics
    expected_ratio = 1.25 * 0.9 * 1.2
    assert metrics["CAGR"] == pytest.approx(expected_ratio ** (365.25 / 366) - 1)
    assert metrics["CAGR"] != pytest.approx(expected_ratio ** (252 / 3) - 1)


def test_annualization_is_required_and_changes_risk_metrics_not_total_return():
    result = ledger_result([0.125, -0.0625] * 3)
    with pytest.raises(TypeError):
        PerformanceConfig()
    daily = calculate_performance(result, PerformanceConfig(252))
    lower = calculate_performance(result, PerformanceConfig(126))
    for name in ["Annualized_Volatility", "Sharpe_Ratio", "Sortino_Ratio"]:
        assert daily.portfolio_metrics[name] == pytest.approx(lower.portfolio_metrics[name] * math.sqrt(2))
    assert daily.portfolio_metrics["Total_Return"] == lower.portfolio_metrics["Total_Return"]
    assert daily.portfolio_metrics["CAGR"] == lower.portfolio_metrics["CAGR"]


def test_returns_are_recomputed_from_equity_not_target_or_cached_return_columns():
    result = ledger_result([0.125, -0.0625, 0])
    clean = calculate_performance(result, DAILY)
    result.ledger["daily_net_return"] = 999.0
    result.ledger["target_return"] = -999.0
    altered = calculate_performance(result, DAILY)
    assert altered.portfolio_metrics == clean.portfolio_metrics
    assert altered.completeness.valuation_observations == 4
    assert altered.completeness.daily_return_observations == 3


def test_initial_reference_does_not_dilute_constant_return_volatility():
    result = ledger_result([0.25] * 4)
    metrics = calculate_performance(result, DAILY).portfolio_metrics
    assert metrics["Annualized_Volatility"] == 0
    assert math.isnan(metrics["Sharpe_Ratio"]) and math.isnan(metrics["Sortino_Ratio"])


def test_no_downside_and_no_completed_trades_are_explicitly_undefined():
    report = calculate_performance(ledger_result([0.125, 0.25, 0.0625]), DAILY)
    assert math.isnan(report.portfolio_metrics["Sortino_Ratio"])
    assert report.trade_metrics["Number_of_Completed_Trades"] == 0
    assert report.trade_metrics["Gross_Profit"] == report.trade_metrics["Gross_Loss"] == 0
    for name in ["Trade_Win_Rate", "Profit_Factor", "Average_Net_Trade_PnL", "Average_Holding_Duration_Seconds"]:
        assert math.isnan(report.trade_metrics[name])


def completed_trades(pnls):
    """Net profits are inputs; gross prices intentionally cannot define wins."""
    entry = pd.Timestamp("2020-01-01 09:30", tz="UTC")
    return pd.DataFrame({
        "status": ["completed"] * len(pnls), "net_realized_pnl": pnls,
        "gross_realized_pnl": [999.0] * len(pnls),
        "entry_timestamp": [entry] * len(pnls),
        "exit_timestamp": [entry + pd.Timedelta(hours=i + 1) for i in range(len(pnls))],
        "holding_sessions": [i + 1 for i in range(len(pnls))],
    })


@pytest.mark.parametrize("pnls,win_rate,profit,loss,factor,average", [
    ([20, -5, 0, 10], 0.5, 30, 5, 6, 6.25),
    ([-5, -15], 0, 0, 20, 0, -10),
    ([10, 20], 1, 30, 0, math.nan, 15),
    ([0, 0], 0, 0, 0, math.nan, 0),
])
def test_trade_statistics_use_completed_net_pnl(pnls, win_rate, profit, loss, factor, average):
    result = replace(ledger_result([0, 0]), completed_trades=completed_trades(pnls))
    metrics = calculate_performance(result, DAILY).trade_metrics
    assert metrics["Number_of_Completed_Trades"] == len(pnls)
    assert metrics["Trade_Win_Rate"] == win_rate
    assert metrics["Gross_Profit"] == profit and metrics["Gross_Loss"] == loss
    assert_number(metrics["Profit_Factor"], factor)
    assert metrics["Average_Net_Trade_PnL"] == average
    assert metrics["Average_Holding_Duration_Seconds"] == 3600 * (len(pnls) + 1) / 2
    assert metrics["Average_Holding_Sessions"] == (len(pnls) + 1) / 2


def market(closes=None, opens=None):
    closes = [100, 100, 110, 110, 110] if closes is None else closes
    opens = [100] * len(closes) if opens is None else opens
    days = pd.bdate_range("2020-01-06", periods=len(closes), tz="UTC")
    return [MarketBar("SYNTH", f"day-{i}", day + pd.Timedelta(hours=9, minutes=30),
                      day + pd.Timedelta(hours=16), opening, close)
            for i, (day, opening, close) in enumerate(zip(days, opens, closes))]


def decision(bar, direction="long"):
    return Signal(bar.session_id, bar.close_timestamp + pd.Timedelta(minutes=1), direction)


def execute(bars, signals=None, **kwargs):
    config = ExecutionConfig(
        "SYNTH", initial_capital=1000, holding_sessions=kwargs.pop("holding_sessions", 5),
        fee_rate=kwargs.pop("fee_rate", 0.01), slippage_rate=kwargs.pop("slippage_rate", 0.01),
        allow_short=True, **kwargs,
    )
    return ExecutionEngineV1(config).run(bars, [decision(bars[0])] if signals is None else signals)


def test_flat_portfolio_metrics_and_completeness():
    result = execute(market(), [])
    report = calculate_performance(result, DAILY)
    assert report.portfolio_metrics["Total_Return"] == report.portfolio_metrics["CAGR"] == 0
    assert report.portfolio_metrics["Annualized_Volatility"] == 0
    assert report.portfolio_metrics["Max_Drawdown"] == 0
    assert report.portfolio_metrics["Average_Gross_Exposure"] == report.portfolio_metrics["Time_in_Market"] == 0
    assert report.turnover_metrics["Turnover"] == 0
    assert report.completeness.requested_end_session_id == "day-4"
    assert report.completeness.effective_end_session_id == "day-4"
    assert report.completeness.requested_window_completed
    assert report.completeness.full_window_comparison_eligible
    assert not report.completeness.descriptive_only
    assert report.completeness.valuation_observations == 6
    assert report.completeness.daily_return_observations == 5


def test_open_unrealized_profit_is_in_portfolio_equity_but_not_trade_win_rate():
    bars = market()[:3]
    result = execute(bars)
    report = calculate_performance(result, DAILY)
    assert report.portfolio_metrics["Total_Return"] == pytest.approx(0.07191)
    assert report.trade_metrics["Number_of_Completed_Trades"] == 0
    assert math.isnan(report.trade_metrics["Trade_Win_Rate"])
    assert math.isnan(report.trade_metrics["Profit_Factor"])
    assert report.turnover_metrics["Traded_Reference_Notional"] == 900
    assert report.turnover_metrics["Turnover"] == 0.9
    expected = (0 + 9 * 100 / (81.91 + 900) + 9 * 110 / 1071.91) / 3
    assert report.portfolio_metrics["Average_Gross_Exposure"] == pytest.approx(expected)
    assert report.portfolio_metrics["Time_in_Market"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("direction,close,final,net_pnl", [
    ("long", 110, "1052.209", "52.209"), ("short", 80, "1147.618", "147.618"),
])
def test_terminal_liquidation_includes_all_costs_once_and_reference_turnover(direction, close, final, net_pnl):
    bars = market([100, 100, close, close])
    result = execute(bars, [decision(bars[0], direction)], terminal_policy="liquidate_at_final_close",
                     terminal_session_id="day-2")
    report = calculate_performance(result, DAILY)
    assert report.portfolio_metrics["Total_Return"] == pytest.approx(float(Decimal(final) / 1000 - 1))
    assert report.trade_metrics["Average_Net_Trade_PnL"] == float(Decimal(net_pnl))
    assert report.trade_metrics["Trade_Win_Rate"] == 1
    assert math.isnan(report.trade_metrics["Profit_Factor"])
    assert report.turnover_metrics["Traded_Reference_Notional"] == 9 * (100 + close)
    assert report.turnover_metrics["Turnover"] == 9 * (100 + close) / 1000
    assert report.completeness.requested_window_completed
    assert report.completeness.terminal_outcome == "liquidated"
    assert report.portfolio_metrics["Time_in_Market"] == pytest.approx(2 / 3)


def test_marking_and_liquidation_have_distinct_realized_costs_and_trade_statistics():
    bars = market()
    marked = execute(bars, terminal_session_id="day-2")
    liquidated = execute(bars, terminal_session_id="day-2", terminal_policy="liquidate_at_final_close")
    mark_report = calculate_performance(marked, DAILY)
    exit_report = calculate_performance(liquidated, DAILY)
    assert mark_report.portfolio_metrics["Total_Return"] == pytest.approx(0.07191)
    assert exit_report.portfolio_metrics["Total_Return"] == pytest.approx(0.052209)
    assert mark_report.trade_metrics["Number_of_Completed_Trades"] == 0
    assert exit_report.trade_metrics["Number_of_Completed_Trades"] == 1
    assert mark_report.completeness.requested_end_session_id == exit_report.completeness.requested_end_session_id == "day-2"


def test_reversal_turnover_counts_both_legs_without_netting_or_counting_rejections():
    bars = market([100] * 4)
    signals = [decision(bars[0]), decision(bars[1], "short")]
    result = execute(bars, signals, position_policy="replace_next_open", fee_rate=0, slippage_rate=0)
    report = calculate_performance(result, DAILY)
    # Buy 10, sell 10 to close, sell 10 to short: 3000 of reference notional.
    assert report.turnover_metrics["Traded_Reference_Notional"] == 3000
    assert report.turnover_metrics["Turnover"] == 3
    assert report.trade_metrics["Number_of_Completed_Trades"] == 1
    gap = market([100, 200, 200], [100, 200, 200])
    rejected = calculate_performance(execute(gap), DAILY)
    assert rejected.turnover_metrics["Turnover"] == 0
    assert rejected.trade_metrics["Number_of_Completed_Trades"] == 0


def test_same_day_trade_counts_time_in_market_but_not_closing_exposure():
    bars = market([100] * 3)
    report = calculate_performance(execute(bars, holding_sessions=1), DAILY)
    assert report.portfolio_metrics["Time_in_Market"] == pytest.approx(1 / 3)
    assert report.portfolio_metrics["Average_Gross_Exposure"] == 0
    assert report.trade_metrics["Average_Holding_Duration_Seconds"] == 6.5 * 3600
    assert report.trade_metrics["Average_Holding_Sessions"] == 1


@pytest.mark.parametrize("phase", ["open", "close"])
def test_margin_stop_is_descriptive_and_preserves_actual_requested_window(phase):
    bars = market([100] * 6)
    bars[2] = replace(bars[2], **{f"{phase}_price": 161})
    result = execute(bars, [decision(bars[0], "short")])
    report = calculate_performance(result, DAILY)
    meta = report.completeness
    assert meta.run_status == "margin_breach"
    assert meta.requested_end_session_id == "day-5" and meta.effective_end_session_id == "day-2"
    assert not meta.requested_end_reached and not meta.requested_window_completed
    assert not meta.full_window_comparison_eligible and meta.descriptive_only
    assert meta.terminal_risk_snapshot is result.terminal_snapshot
    assert meta.terminal_risk_total_return == pytest.approx(433.09 / 1000 - 1)
    assert meta.daily_return_observations == (2 if phase == "open" else 3)
    if phase == "open":
        assert meta.valuation_end_session_id == "day-1"
        assert report.portfolio_metrics["Total_Return"] == pytest.approx(982.09 / 1000 - 1)
        assert meta.valuation_end_timestamp < result.terminal_snapshot.timestamp
    else:
        assert meta.valuation_end_session_id == "day-2"
        assert report.portfolio_metrics["Total_Return"] == pytest.approx(433.09 / 1000 - 1)


def test_opening_insolvency_after_cover_keeps_fills_separate_from_daily_valuation():
    bars = market([100] * 5, [100, 100, 208, 100, 100])
    result = execute(bars, [decision(bars[0], "short"), decision(bars[1], "long")],
                     position_policy="replace_next_open", initial_margin_fraction=0.001,
                     maintenance_margin_fraction=0.0001)
    report = calculate_performance(result, DAILY)
    assert report.completeness.run_status == "insolvent" and report.completeness.descriptive_only
    assert report.completeness.fills_after_last_valuation == 1
    assert report.turnover_metrics["Traded_Reference_Notional"] == 9 * (100 + 208)
    assert report.portfolio_metrics["Total_Return"] == pytest.approx(-0.01791)
    assert report.completeness.terminal_risk_total_return == pytest.approx(-27.5372 / 1000 - 1)


def test_terminal_insolvency_retains_negative_equity_and_undefined_cagr():
    bars = market([100, 100, 250, 250])
    result = execute(bars, [decision(bars[0], "short")], terminal_session_id="day-2",
                     terminal_policy="liquidate_at_final_close")
    report = calculate_performance(result, DAILY)
    assert report.completeness.requested_end_reached
    assert not report.completeness.requested_window_completed
    assert not report.completeness.full_window_comparison_eligible
    assert report.completeness.terminal_outcome == "liquidated"
    assert report.portfolio_metrics["Total_Return"] == pytest.approx(-1.413135)
    assert report.portfolio_metrics["Max_Drawdown"] == pytest.approx(-1.413135)
    assert math.isnan(report.portfolio_metrics["CAGR"])
    assert math.isnan(report.portfolio_metrics["Average_Gross_Exposure"])


def test_fixed_endpoint_report_is_unchanged_by_future_data_and_signals():
    bars = market()
    signals = [decision(bars[0])]
    result = execute(bars[:3], signals, terminal_session_id="day-2", terminal_policy="liquidate_at_final_close")
    future = [replace(bar, close_price=None, open_price=None) if i > 2 else bar
              for i, bar in enumerate(bars)]
    extended = execute(future, signals + [decision(bars[4], "short")],
                       terminal_session_id="day-2", terminal_policy="liquidate_at_final_close")
    first = calculate_performance(result, DAILY)
    second = calculate_performance(extended, DAILY)
    for name in ["portfolio_metrics", "trade_metrics", "turnover_metrics"]:
        left, right = getattr(first, name), getattr(second, name)
        assert left.keys() == right.keys()
        for metric in left:
            assert_number(left[metric], right[metric])
    left, right = asdict(first.completeness), asdict(second.completeness)
    assert_number(left.pop("terminal_risk_total_return"), right.pop("terminal_risk_total_return"))
    assert left == right


def test_reporting_does_not_mutate_or_attach_fields_to_execution_result():
    result = execute(market())
    before = deepcopy(result)
    calculate_performance(result, DAILY)
    for name in ["ledger", "fills", "events", "completed_trades", "open_trades"]:
        pd.testing.assert_frame_equal(getattr(before, name), getattr(result, name))
    for name in ["run_status", "terminal_snapshot", "terminal_outcome", "open_position", "requested_end_session_id"]:
        assert getattr(before, name) == getattr(result, name)


@pytest.mark.parametrize("kwargs", [
    {"periods_per_year": 0}, {"periods_per_year": -1}, {"periods_per_year": float("nan")},
    {"periods_per_year": True}, {"periods_per_year": "252"},
    {"periods_per_year": 252, "annual_risk_free_rate": -1},
    {"periods_per_year": 252, "annual_risk_free_rate": float("inf")},
    {"periods_per_year": 252, "calendar_days_per_year": 0},
])
def test_invalid_annualization_assumptions_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        PerformanceConfig(**kwargs)


@pytest.mark.parametrize("column,value", [("equity", float("nan")), ("equity", float("inf")),
                                            ("position_quantity", 0.5), ("close_price", 0)])
def test_invalid_daily_observations_are_not_silently_dropped(column, value):
    result = ledger_result([0, 0])
    result.ledger[column] = result.ledger[column].astype(float)
    result.ledger.loc[1, column] = value
    with pytest.raises(ValueError):
        calculate_performance(result, DAILY)


def test_missing_initial_reference_is_rejected():
    result = ledger_result([0, 0])
    result.ledger.loc[0, "is_initial"] = False
    with pytest.raises(ValueError, match="initial-capital"):
        calculate_performance(result, DAILY)


def test_initial_reference_alone_does_not_invent_returns_or_exposure():
    report = calculate_performance(ledger_result([]), DAILY)
    assert report.portfolio_metrics["Total_Return"] == report.portfolio_metrics["Max_Drawdown"] == 0
    for name in ["CAGR", "Sharpe_Ratio", "Sortino_Ratio", "Annualized_Volatility", "Time_in_Market"]:
        assert math.isnan(report.portfolio_metrics[name])
    assert report.completeness.daily_return_observations == 0
    assert report.completeness.valuation_observations == 1
    assert report.completeness.descriptive_only


def test_old_risk_result_without_requested_endpoint_is_not_claimed_complete():
    bars = market([100] * 5, [100, 100, 161, 100, 100])
    result = execute(bars, [decision(bars[0], "short")])
    old_contract = replace(result, requested_end_session_id=None)
    report = calculate_performance(old_contract, DAILY)
    assert report.completeness.requested_end_session_id is None
    assert not report.completeness.requested_window_completed
    assert report.completeness.descriptive_only


@pytest.mark.parametrize("length", [3, 5, 7])
def test_identical_excess_returns_have_zero_variance_without_epsilon(length):
    report = calculate_performance(ledger_result([0.25] * length),
                                   PerformanceConfig(252, annual_risk_free_rate=0.05))
    assert report.portfolio_metrics["Annualized_Volatility"] == 0
    assert math.isnan(report.portfolio_metrics["Sharpe_Ratio"])


def test_genuinely_small_nonzero_variance_is_not_clamped_to_zero():
    report = calculate_performance(ledger_result([0.000001, 0.000002, 0.000001]), DAILY)
    assert report.portfolio_metrics["Annualized_Volatility"] > 0
    assert math.isfinite(report.portfolio_metrics["Sharpe_Ratio"])


def test_turnover_excludes_restricted_proceeds_as_additional_capital():
    bars = market([100] * 3)
    result = execute(bars, [decision(bars[0], "short")])
    report = calculate_performance(result, DAILY)
    assert result.ledger.iloc[-1].restricted_proceeds == 891
    assert report.turnover_metrics["Turnover_Normalization_Equity"] == 1000
    assert report.turnover_metrics["Turnover"] == 0.9


def test_unprofitable_net_trade_is_not_a_win_despite_positive_gross_price_profit():
    bars = market([100, 100, 101])
    result = execute(bars, holding_sessions=2, slippage_rate=0, fee_rate=0.01)
    assert result.completed_trades.iloc[0].gross_realized_pnl == 9
    report = calculate_performance(result, DAILY)
    assert report.trade_metrics["Trade_Win_Rate"] == 0
    assert report.trade_metrics["Gross_Profit"] == 0
    assert report.trade_metrics["Gross_Loss"] == pytest.approx(9.09)

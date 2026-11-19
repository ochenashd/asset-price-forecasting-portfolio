"""
Backtesting module for classification-based trading strategies.
"""

from pathlib import Path
from typing import Dict, Optional

import logging
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Backtester:
    """Backtester for non-overlapping 5-day classification targets."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        periods_per_year: int = 252,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.periods_per_year = periods_per_year
        self.logger = logging.getLogger(__name__)

    def run_5d_class_backtest(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        ticker: str = "Asset",
        model_name: str = "model",
        strategy_name: str = "predict",
        rebalance_step: int = 5,
        long_only: bool = True,
        use_proba_filter: bool = False,
        proba_up: Optional[np.ndarray] = None,
        proba_down: Optional[np.ndarray] = None,
        min_up_proba: float = 0.60,
        min_down_proba: float = 0.60,
        min_probability_edge: float = 0.0,
        target_return_col: str = "target_return",
        log_to_mlflow: bool = True,
        dynamic_thresholds: bool = False,
        dynamic_threshold_method: str = "rolling_quantile",
        dynamic_threshold_window: int = 20,
        dynamic_threshold_quantile: float = 0.70,
        dynamic_threshold_std_multiplier: float = 0.5,
        risk_free_rate: float = 0.0,
    ) -> Dict:
        """Run a non-overlapping 5-day backtest for class predictions.

        Class mapping:
        - 0 = Down
        - 1 = Neutral
        - 2 = Up

        When use_proba_filter=False, positions come from predicted classes.
        When use_proba_filter=True, positions come from predicted probabilities.
        risk_free_rate is an effective annual rate. An optional aligned
        holding_period_sessions column declares actual trading-session durations.
        Without that metadata, annualization assumes the configured rebalance
        cadence and reports Holding_Duration_Validated=0; timestamps alone do not
        establish label holding periods or an exchange trading calendar.
        """
        if target_return_col not in test_df.columns:
            raise ValueError(
                f"Column '{target_return_col}' is missing from test_df. "
                "For a 5-day classification target, test_df must contain the true 5-day return."
            )

        if len(test_df) == 0:
            raise ValueError("test_df is empty.")

        predictions = np.asarray(predictions).flatten()
        min_len = min(len(test_df), len(predictions))

        test_df = test_df.iloc[:min_len].copy()
        predictions = predictions[:min_len]

        # Use one signal every rebalance_step rows to avoid overlapping 5-day returns.
        test_bt = test_df.iloc[::rebalance_step].copy()
        pred_bt = predictions[::rebalance_step]
        ret_bt = test_bt[target_return_col].astype(float).values

        p_up_bt = None
        p_down_bt = None

        # if use_proba_filter:
        #     if proba_up is None:
        #         raise ValueError("proba_up must be provided when use_proba_filter=True.")
        #
        #     proba_up = np.asarray(proba_up).flatten()[:min_len]
        #
        #     if proba_down is None:
        #         proba_down = np.zeros_like(proba_up)
        #     else:
        #         proba_down = np.asarray(proba_down).flatten()[:min_len]

        if use_proba_filter:
            if proba_up is None or proba_down is None:
                raise ValueError(
                    "proba_up and proba_down must be provided when use_proba_filter=True."
                )

            proba_up = np.asarray(proba_up).flatten()[:min_len]
            proba_down = np.asarray(proba_down).flatten()[:min_len]

            p_up_bt = proba_up[::rebalance_step]
            p_down_bt = proba_down[::rebalance_step]

            edge_up = p_up_bt - p_down_bt
            edge_down = p_down_bt - p_up_bt

            if dynamic_thresholds:
                up_threshold_bt = self._compute_dynamic_thresholds(
                    proba=p_up_bt,
                    method=dynamic_threshold_method,
                    window=dynamic_threshold_window,
                    quantile=dynamic_threshold_quantile,
                    std_multiplier=dynamic_threshold_std_multiplier,
                    min_threshold=0.0,
                    max_threshold=1.0,
                )

                down_threshold_bt = self._compute_dynamic_thresholds(
                    proba=p_down_bt,
                    method=dynamic_threshold_method,
                    window=dynamic_threshold_window,
                    quantile=dynamic_threshold_quantile,
                    std_multiplier=dynamic_threshold_std_multiplier,
                    min_threshold=0.0,
                    max_threshold=1.0,
                )
            else:
                up_threshold_bt = np.full_like(p_up_bt, fill_value=min_up_proba, dtype=float)
                down_threshold_bt = np.full_like(p_down_bt, fill_value=min_down_proba, dtype=float)

            if long_only:
                positions = np.where(
                    (p_up_bt >= up_threshold_bt) & (edge_up >= min_probability_edge),
                    1.0,
                    0.0,
                )
            else:
                positions = np.where(
                    (p_up_bt >= up_threshold_bt) & (edge_up >= min_probability_edge),
                    1.0,
                    np.where(
                        (p_down_bt >= down_threshold_bt) & (edge_down >= min_probability_edge),
                        -1.0,
                        0.0,
                    ),
                )
            if dynamic_thresholds:
                # Long/short decisions require both thresholds; long-only uses up.
                thresholds_ready = np.isfinite(up_threshold_bt)
                if not long_only:
                    thresholds_ready &= np.isfinite(down_threshold_bt)
                positions = np.where(thresholds_ready, positions, 0.0)
        else:
            if long_only:
                positions = np.where(pred_bt == 2, 1.0, 0.0)
            else:
                positions = np.where(
                    pred_bt == 2,
                    1.0,
                    np.where(pred_bt == 0, -1.0, 0.0),
                )

        position_changes = np.abs(np.diff(positions, prepend=0.0))
        strategy_returns = positions * ret_bt - position_changes * self.transaction_cost

        equity_curve = self.initial_capital * np.cumprod(1.0 + strategy_returns)

        benchmark_returns = ret_bt
        benchmark_equity = self.initial_capital * np.cumprod(1.0 + benchmark_returns)

        strategy_returns_series = pd.Series(strategy_returns, index=test_bt.index, name="strategy_return")
        benchmark_returns_series = pd.Series(benchmark_returns, index=test_bt.index, name="benchmark_return")
        positions_series = pd.Series(positions, index=test_bt.index, name="position")
        predictions_series = pd.Series(pred_bt, index=test_bt.index, name="prediction")

        periods_per_year = self.periods_per_year / max(rebalance_step, 1)
        metrics = self._calculate_metrics(
            strategy_returns=strategy_returns_series,
            benchmark_returns=benchmark_returns_series,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            holding_period_sessions=test_bt.get("holding_period_sessions"),
        )

        metrics.update(
            {
                "Initial_Capital": float(self.initial_capital),
                "Final_Capital": float(equity_curve[-1]) if len(equity_curve) else float(self.initial_capital),
                "Number_of_Trades": int(position_changes.sum()),
                "Number_of_Rebalances": int(len(test_bt)),
                "Rebalance_Step": int(rebalance_step),
                "Long_Only": int(long_only),
                "Average_Position": float(np.mean(positions)) if len(positions) else 0.0,
                "Exposure": float(np.mean(np.abs(positions) > 0)) if len(positions) else 0.0,
                "Long_Signals": int(np.sum(positions > 0)),
                "Short_Signals": int(np.sum(positions < 0)),
                "Flat_Signals": int(np.sum(positions == 0)),
            }
        )

        results = {
            "metrics": metrics,
            "strategy_returns": strategy_returns_series,
            "benchmark_returns": benchmark_returns_series,
            "equity_curve": pd.Series(equity_curve, index=test_bt.index, name="equity_curve"),
            "benchmark_equity": pd.Series(benchmark_equity, index=test_bt.index, name="benchmark_equity"),
            "positions": positions_series,
            "predictions": predictions_series,
            "test_bt": test_bt,
            "strategy_name": strategy_name,
        }

        if use_proba_filter:
            results["proba_up"] = pd.Series(p_up_bt, index=test_bt.index, name="proba_up")
            results["proba_down"] = pd.Series(p_down_bt, index=test_bt.index, name="proba_down")
            results["up_threshold"] = pd.Series(up_threshold_bt, index=test_bt.index, name="up_threshold")
            results["down_threshold"] = pd.Series(down_threshold_bt, index=test_bt.index, name="down_threshold")
            results["edge_up"] = pd.Series(edge_up, index=test_bt.index, name="edge_up")
            results["edge_down"] = pd.Series(edge_down, index=test_bt.index, name="edge_down")

            metrics.update(
                {
                    "Mean_Proba_Up": float(np.mean(p_up_bt)),
                    "Max_Proba_Up": float(np.max(p_up_bt)),
                    "Min_Proba_Up": float(np.min(p_up_bt)),
                    "Mean_Proba_Down": float(np.mean(p_down_bt)),
                    "Max_Proba_Down": float(np.max(p_down_bt)),
                    "Min_Proba_Down": float(np.min(p_down_bt)),
                    "Mean_Edge_Up": float(np.mean(edge_up)),
                    "Max_Edge_Up": float(np.max(edge_up)),
                    "Min_Edge_Up": float(np.min(edge_up)),
                    "Dynamic_Thresholds": int(dynamic_thresholds),
                    "Mean_Up_Threshold": float(np.mean(up_threshold_bt)),
                    "Mean_Down_Threshold": float(np.mean(down_threshold_bt)),
                    "Max_Up_Threshold": float(np.max(up_threshold_bt)),
                    "Min_Up_Threshold": float(np.min(up_threshold_bt)),
                    "Max_Down_Threshold": float(np.max(down_threshold_bt)),
                    "Min_Down_Threshold": float(np.min(down_threshold_bt)),
                }
            )

        if log_to_mlflow and mlflow.active_run() is not None:
            self._log_backtest_artifacts(
                results=results,
                ticker=ticker,
                model_name=model_name,
                strategy_name=strategy_name,
            )

        self.logger.info(
            f"Backtest completed for {ticker} | model={model_name} | strategy={strategy_name} | "
            f"Sharpe={metrics.get('Sharpe_Ratio', 0.0):.3f} | "
            f"Exposure={metrics.get('Exposure', 0.0):.2%} | Trades={metrics.get('Number_of_Trades', 0)}"
        )

        return results

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        actual_returns: Optional[np.ndarray] = None,
        ticker: str = "Asset",
    ) -> Dict:
        """Backward-compatible wrapper for old pipeline calls."""
        if "target_return" not in test_df.columns:
            if actual_returns is None:
                raise ValueError("test_df must contain 'target_return' or actual_returns must be provided.")
            test_df = test_df.copy()
            test_df["target_return"] = np.asarray(actual_returns).flatten()[: len(test_df)]

        return self.run_5d_class_backtest(
            test_df=test_df,
            predictions=predictions,
            ticker=ticker,
            model_name="legacy",
            strategy_name="predict",
            rebalance_step=5,
            long_only=True,
            target_return_col="target_return",
            log_to_mlflow=True,
        )

    def _calculate_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        periods_per_year: float = 252,
        risk_free_rate: float = 0.0,
        *,
        holding_period_sessions: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculate metrics for a non-empty, finite, non-negative wealth path.

        Missing returns cannot be dropped without changing the equity timeline.
        Zero wealth is valid bankruptcy; returns below -100% are unsupported.
        Equal holding durations, when declared, must match periods_per_year using
        this backtester's annual session count. Irregular durations are unsupported.
        Without durations, the caller's frequency is an unvalidated assumption.
        Sharpe uses population volatility of arithmetic excess returns. Sortino
        uses full-sample downside deviation relative to the periodic risk-free
        rate. Zero denominators produce NaN, not an epsilon-adjusted ratio.
        """
        returns = pd.Series(strategy_returns)
        if (
            returns.empty
            or not pd.api.types.is_numeric_dtype(returns.dtype)
            or pd.api.types.is_bool_dtype(returns.dtype)
            or pd.api.types.is_complex_dtype(returns.dtype)
            or returns.isna().any()
            or not np.isfinite(returns.to_numpy(dtype=float)).all()
        ):
            raise ValueError("Strategy returns must be non-empty, finite real numeric observations.")
        if (returns < -1.0).any():
            raise ValueError("Strategy returns below -100% produce an unsupported equity path.")
        try:
            with np.errstate(over="raise", invalid="raise"):
                cumulative_returns = (1.0 + returns).cumprod()
        except FloatingPointError as exc:
            raise ValueError("Compounded equity must remain finite and non-negative.") from exc
        if not np.isfinite(cumulative_returns.to_numpy(dtype=float)).all():
            raise ValueError("Compounded equity must remain finite and non-negative.")

        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be finite and positive.")
        if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
            raise ValueError("The effective annual risk-free rate must be finite and greater than -1.")
        duration_validated = holding_period_sessions is not None
        if duration_validated:
            durations = holding_period_sessions
            if (
                not isinstance(durations, pd.Series)
                or not durations.index.equals(returns.index)
                or not pd.api.types.is_numeric_dtype(durations.dtype)
                or pd.api.types.is_bool_dtype(durations.dtype)
                or pd.api.types.is_complex_dtype(durations.dtype)
                or durations.isna().any()
                or not np.isfinite(durations.to_numpy(dtype=float)).all()
                or (durations <= 0).any()
            ):
                raise ValueError("Holding durations require aligned, finite, positive session counts.")
            if durations.nunique() != 1:
                raise ValueError("Irregular holding durations require an unsupported annualization policy.")
            expected_frequency = self.periods_per_year / float(durations.iloc[0])
            if not np.isclose(periods_per_year, expected_frequency):
                raise ValueError("Annualization frequency does not match the declared holding duration.")
        else:
            self.logger.warning(
                "Holding-duration validation is unavailable; assuming %.6g observations per year.",
                periods_per_year,
            )

        total_return = float((1.0 + returns).prod() - 1.0)
        annualized_return = float((1.0 + total_return) ** (periods_per_year / len(returns)) - 1.0)
        annualized_volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
        periodic_risk_free = np.expm1(np.log1p(risk_free_rate) / periods_per_year)
        excess_returns = returns.astype(float) - periodic_risk_free
        mean_excess = float(excess_returns.mean())
        # Avoid floating-point residual variance for exactly constant observations.
        excess_std = float(excess_returns.std(ddof=0)) if excess_returns.nunique() > 1 else 0.0
        downside = np.minimum(excess_returns.to_numpy(), 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside ** 2)))
        sharpe_ratio = (float(np.sqrt(periods_per_year) * mean_excess / excess_std)
                        if excess_std > 0 else float("nan"))
        sortino_ratio = (float(np.sqrt(periods_per_year) * mean_excess / downside_deviation)
                         if downside_deviation > 0 else float("nan"))

        # Initial normalized wealth is a peak even before the first return.
        rolling_max = cumulative_returns.cummax().clip(lower=1.0)
        drawdown = cumulative_returns / rolling_max - 1.0
        max_drawdown = float(drawdown.min())
        win_rate = float((returns > 0).mean() * 100.0)

        metrics = {
            "Total_Return": total_return,
            "Annualized_Return": annualized_return,
            "Annualized_Volatility": annualized_volatility,
            "Sharpe_Ratio": sharpe_ratio,
            "Sortino_Ratio": sortino_ratio,
            "Holding_Duration_Validated": int(duration_validated),
            "Max_Drawdown": max_drawdown,
            "Win_Rate": win_rate,
        }

        if benchmark_returns is not None:
            benchmark_returns = pd.Series(benchmark_returns).dropna().reindex(returns.index).dropna()
            aligned_strategy = returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_strategy.index).dropna()

            if len(aligned_strategy) > 0:
                benchmark_total_return = float((1.0 + aligned_benchmark).prod() - 1.0)
                benchmark_annualized_return = float(
                    (1.0 + benchmark_total_return) ** (periods_per_year / len(aligned_benchmark)) - 1.0
                )
                metrics["Benchmark_Total_Return"] = benchmark_total_return
                metrics["Benchmark_Annualized_Return"] = benchmark_annualized_return
                metrics["Alpha"] = float(annualized_return - benchmark_annualized_return)

        return metrics

    def _log_backtest_artifacts(
        self,
        results: Dict,
        ticker: str,
        model_name: str,
        strategy_name: str,
        output_dir: str | Path = "results/reports/backtests",
    ) -> None:
        """Save and log backtest artifacts to MLflow."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "strategy_return": results["strategy_returns"],
            "benchmark_return": results["benchmark_returns"],
            "position": results["positions"],
            "prediction": results["predictions"],
            "equity_curve": results["equity_curve"],
            "benchmark_equity": results["benchmark_equity"],
        }

        backtest_df = pd.DataFrame(data)

        if "proba_up" in results:
            backtest_df["proba_up"] = results["proba_up"]
            backtest_df["up_threshold"] = results["up_threshold"]
            backtest_df["edge_up"] = results["edge_up"]
        if "proba_down" in results:
            backtest_df["proba_down"] = results["proba_down"]
            backtest_df["down_threshold"] = results["down_threshold"]
            backtest_df["edge_down"] = results["edge_down"]

        safe_model_name = str(model_name).replace("/", "_")
        safe_strategy_name = str(strategy_name).replace("/", "_")

        csv_path = output_dir / f"{ticker}_{safe_model_name}_{safe_strategy_name}_5d_backtest.csv"
        backtest_df.to_csv(csv_path)

        fig_path = output_dir / f"{ticker}_{safe_model_name}_{safe_strategy_name}_5d_backtest_equity.png"
        fig = self.plot_backtest_results(
            backtest_results=results,
            ticker=ticker,
            model_name=model_name,
            strategy_name=strategy_name,
            save_path=fig_path,
        )
        plt.close(fig)

        mlflow.log_artifact(str(csv_path), artifact_path=f"backtests/{safe_strategy_name}")
        mlflow.log_artifact(str(fig_path), artifact_path=f"backtests/{safe_strategy_name}")

    def plot_backtest_results(
        self,
        backtest_results: Dict,
        ticker: str,
        model_name: str,
        strategy_name: str,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot strategy equity and benchmark equity."""
        equity = backtest_results["equity_curve"]
        benchmark_equity = backtest_results["benchmark_equity"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity.index, equity.values, label="Strategy Equity", linewidth=2)
        ax.plot(benchmark_equity.index, benchmark_equity.values, label="Benchmark Equity", linewidth=2)

        ax.set_title(f"{model_name} {strategy_name} 5D Class Backtest - {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _compute_dynamic_thresholds(
            self,
            proba: np.ndarray,
            method: str = "rolling_quantile",
            window: int = 20,
            quantile: float = 0.70,
            std_multiplier: float = 0.5,
            min_threshold: float = 0.0,
            max_threshold: float = 1.0,
    ) -> np.ndarray:
        """Compute thresholds from strictly preceding rebalance probabilities.

        Every method requires max(5, window // 3) non-missing prior values.
        Rolling windows count rebalance observations, including missing entries,
        rather than original bars. Expanding methods use all preceding history.
        Mean/std methods retain pandas' sample standard deviation (ddof=1).
        Insufficient history leaves NaN thresholds, including after rolling gaps;
        callers must remain flat until their required thresholds are defined.
        """
        proba_series = pd.Series(proba).astype(float)

        # Shift by 1 to avoid using the current probability to define its own threshold.
        past_proba = proba_series.shift(1)

        if method == "rolling_quantile":
            thresholds = past_proba.rolling(
                window=window,
                min_periods=max(5, window // 3),
            ).quantile(quantile)

        elif method == "expanding_quantile":
            thresholds = past_proba.expanding(
                min_periods=max(5, window // 3),
            ).quantile(quantile)

        elif method == "rolling_mean_std":
            rolling_mean = past_proba.rolling(
                window=window,
                min_periods=max(5, window // 3),
            ).mean()

            rolling_std = past_proba.rolling(
                window=window,
                min_periods=max(5, window // 3),
            ).std()

            thresholds = rolling_mean + std_multiplier * rolling_std

        elif method == "expanding_mean_std":
            expanding_mean = past_proba.expanding(
                min_periods=max(5, window // 3),
            ).mean()

            expanding_std = past_proba.expanding(
                min_periods=max(5, window // 3),
            ).std()

            thresholds = expanding_mean + std_multiplier * expanding_std

        else:
            raise ValueError(f"Unknown dynamic threshold method: {method}")

        thresholds = thresholds.clip(lower=min_threshold, upper=max_threshold)

        return thresholds.values

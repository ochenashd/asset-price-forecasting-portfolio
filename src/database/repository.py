import json
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.database import queries

class AssetRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def upsert_asset(
        self,
        symbol: str,
        asset_name: Optional[str] = None,
        asset_type: Optional[str] = None,
        exchange: Optional[str] = None,
        quote_currency: Optional[str] = None,
    ) -> int:
        params = {
            "symbol": symbol,
            "asset_name": asset_name,
            "asset_type": asset_type,
            "exchange": exchange,
            "quote_currency": quote_currency,
        }

        with self.engine.begin() as connection:
            result = connection.execute(text(queries.INSERT_ASSET), params)
            asset_id = result.scalar_one()

        return int(asset_id)

    def get_asset_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            result = connection.execute(
                text(queries.GET_ASSET_BY_SYMBOL),
                {"symbol": symbol},
            )
            row = result.mappings().first()

        return dict(row) if row else None


class MarketPriceRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def upsert_price(
        self,
        asset_id: int,
        timestamp: Any,
        timeframe: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        params = {
            "asset_id": asset_id,
            "timestamp": timestamp,
            "timeframe": timeframe,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "source": source,
        }

        with self.engine.begin() as connection:
            connection.execute(text(queries.INSERT_MARKET_PRICE), params)

    def upsert_prices_from_dataframe(
        self,
        asset_id: int,
        df: pd.DataFrame,
        timeframe: str,
        source: Optional[str] = None,
    ) -> None:
        required_columns = {"timestamp", "open", "high", "low", "close"}

        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        records = []

        for _, row in df.iterrows():
            records.append(
                {
                    "asset_id": asset_id,
                    "timestamp": row["timestamp"],
                    "timeframe": timeframe,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": (
                        float(row["volume"])
                        if "volume" in df.columns and pd.notna(row["volume"])
                        else None
                    ),
                    "source": source,
                }
            )

        with self.engine.begin() as connection:
            connection.execute(text(queries.INSERT_MARKET_PRICE), records)

    def get_prices(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
        }

        with self.engine.connect() as connection:
            df = pd.read_sql_query(
                sql=text(queries.GET_MARKET_PRICES),
                con=connection,
                params=params,
            )

        return df

class TechnicalFeatureRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def upsert_features_from_dataframe(
        self,
        asset_id: int,
        df: pd.DataFrame,
        timeframe: str,
    ) -> None:
        """Save engineered technical features into PostgreSQL."""
        if df.empty:
            return

        db_df = df.copy()

        if "timestamp" not in db_df.columns:
            db_df = db_df.reset_index()

        if "timestamp" not in db_df.columns:
            first_col = db_df.columns[0]
            db_df = db_df.rename(columns={first_col: "timestamp"})

        records = []

        for _, row in db_df.iterrows():
            records.append(
                {
                    "asset_id": asset_id,
                    "timestamp": row["timestamp"],
                    "timeframe": timeframe,

                    "return_1d": self._get_first_available_float(
                        row,
                        ["return_1d", "return_lag_1", "close_change_lag_1"],
                    ),
                    "return_7d": self._get_first_available_float(
                        row,
                        ["return_7d", "momentum_7", "close_change_lag_7"],
                    ),
                    "return_14d": self._get_first_available_float(
                        row,
                        ["return_14d", "momentum_14", "close_change_lag_14"],
                    ),
                    "return_30d": self._get_first_available_float(
                        row,
                        ["return_30d", "momentum_30", "close_change_lag_30"],
                    ),

                    "volatility_7d": self._get_first_available_float(
                        row,
                        ["volatility_7d", "vol_7"],
                    ),
                    "volatility_30d": self._get_first_available_float(
                        row,
                        ["volatility_30d", "vol_30"],
                    ),

                    "rsi_14": self._get_optional_float(row, "rsi_14"),
                    "macd": self._get_optional_float(row, "macd"),
                    "macd_signal": self._get_optional_float(row, "macd_signal"),
                    "macd_hist": self._get_optional_float(row, "macd_hist"),

                    "bb_upper": self._get_optional_float(row, "bb_upper"),
                    "bb_middle": self._get_first_available_float(
                        row,
                        ["bb_middle", "bb_mid", "ma_20"],
                    ),
                    "bb_lower": self._get_optional_float(row, "bb_lower"),
                    "bb_width": self._get_optional_float(row, "bb_width"),

                    "sma_7": self._get_first_available_float(row, ["sma_7", "ma_7"]),
                    "sma_14": self._get_first_available_float(row, ["sma_14", "ma_14"]),
                    "sma_30": self._get_first_available_float(row, ["sma_30", "ma_30"]),
                    "sma_60": self._get_first_available_float(row, ["sma_60", "ma_60"]),

                    "ema_7": self._get_optional_float(row, "ema_7"),
                    "ema_14": self._get_optional_float(row, "ema_14"),
                    "ema_30": self._get_optional_float(row, "ema_30"),
                    "ema_60": self._get_optional_float(row, "ema_60"),

                    "volume_change": self._get_first_available_float(
                        row,
                        ["volume_change", "volume_pct_change", "volume_change_lag_1"],
                    ),

                    "target": self._get_optional_float(row, "target"),
                    "target_return": self._get_optional_float(row, "target_return"),
                    "features_json": self._build_features_json(row),
                }
            )

        with self.engine.begin() as connection:
            connection.execute(text(queries.INSERT_TECHNICAL_FEATURE), records)
    def get_features(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Load technical features from PostgreSQL."""
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
        }

        with self.engine.connect() as connection:
            df = pd.read_sql_query(
                sql=text(queries.GET_TECHNICAL_FEATURES),
                con=connection,
                params=params,
            )

        return df

    @staticmethod
    def _get_optional_float(row: pd.Series, column: str) -> Optional[float]:
        if column not in row:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return float(value)

    @classmethod
    def _get_first_available_float(
        cls,
        row: pd.Series,
        columns: list[str],
    ) -> Optional[float]:
        """Return the first valid value found from a list of possible column names."""
        for column in columns:
            value = cls._get_optional_float(row, column)

            if value is not None:
                return value

        return None

    @staticmethod
    def _build_features_json(row: pd.Series) -> str:
        excluded_columns = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "target",
            "target_return",
        }

        payload = {}

        for column, value in row.items():
            if column in excluded_columns:
                continue

            if pd.isna(value):
                continue

            if isinstance(value, (int, float, np.integer, np.floating)):
                payload[column] = float(value)
            else:
                payload[column] = str(value)

        return json.dumps(payload)


class ModelRunRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create_run(
        self,
        model_name: str,
        asset_id: int,
        target_name: str,
        timeframe: str,
        train_start: Optional[Any] = None,
        train_end: Optional[Any] = None,
        test_start: Optional[Any] = None,
        test_end: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> int:
        params = {
            "model_name": model_name,
            "asset_id": asset_id,
            "target_name": target_name,
            "timeframe": timeframe,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "parameters_json": json.dumps(parameters or {}, default=str),
            "metrics_json": json.dumps(metrics or {}, default=str),
            "mlflow_run_id": mlflow_run_id,
        }

        with self.engine.begin() as connection:
            result = connection.execute(text(queries.INSERT_MODEL_RUN), params)
            run_id = result.scalar_one()

        return int(run_id)


class PredictionRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def upsert_prediction(
        self,
        run_id: int,
        asset_id: int,
        timestamp: Any,
        y_true: Optional[float] = None,
        y_pred: Optional[float] = None,
        y_true_class: Optional[int] = None,
        predicted_class: Optional[int] = None,
        proba_down: Optional[float] = None,
        proba_neutral: Optional[float] = None,
        proba_up: Optional[float] = None,
    ) -> None:
        params = {
            "run_id": run_id,
            "asset_id": asset_id,
            "timestamp": timestamp,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_true_class": y_true_class,
            "predicted_class": predicted_class,
            "proba_down": proba_down,
            "proba_neutral": proba_neutral,
            "proba_up": proba_up,
        }

        with self.engine.begin() as connection:
            connection.execute(text(queries.INSERT_PREDICTION), params)

    def upsert_predictions_from_dataframe(
        self,
        run_id: int,
        asset_id: int,
        df: pd.DataFrame,
    ) -> None:
        required_columns = {"timestamp"}

        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        records = []

        for _, row in df.iterrows():
            records.append(
                {
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "timestamp": row["timestamp"],
                    "y_true": self._get_optional_float(row, "y_true"),
                    "y_pred": self._get_optional_float(row, "y_pred"),
                    "y_true_class": self._get_optional_int(row, "y_true_class"),
                    "predicted_class": self._get_optional_int(row, "predicted_class"),
                    "proba_down": self._get_optional_float(row, "proba_down"),
                    "proba_neutral": self._get_optional_float(row, "proba_neutral"),
                    "proba_up": self._get_optional_float(row, "proba_up"),
                }
            )

        with self.engine.begin() as connection:
            connection.execute(text(queries.INSERT_PREDICTION), records)

    @staticmethod
    def _get_optional_float(row: pd.Series, column: str) -> Optional[float]:
        if column not in row or pd.isna(row[column]):
            return None
        return float(row[column])

    @staticmethod
    def _get_optional_int(row: pd.Series, column: str) -> Optional[int]:
        if column not in row or pd.isna(row[column]):
            return None
        return int(row[column])

class BacktestRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save_result(
        self,
        run_id: int,
        asset_id: int,
        strategy_name: str,
        initial_capital: Optional[float] = None,
        final_capital: Optional[float] = None,
        total_return: Optional[float] = None,
        annualized_return: Optional[float] = None,
        benchmark_return: Optional[float] = None,
        strategy_return: Optional[float] = None,
        volatility: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        sortino_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        win_rate: Optional[float] = None,
        number_of_trades: Optional[int] = None,
        number_of_rebalances: Optional[int] = None,
        long_only: Optional[bool] = None,
        exposure: Optional[float] = None,
        long_signals: Optional[int] = None,
        short_signals: Optional[int] = None,
        flat_signals: Optional[int] = None,
    ) -> int:
        params = {
            "run_id": run_id,
            "asset_id": asset_id,
            "strategy_name": strategy_name,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "benchmark_return": benchmark_return,
            "strategy_return": strategy_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "number_of_trades": number_of_trades,
            "number_of_rebalances": number_of_rebalances,
            "long_only": long_only,
            "exposure": exposure,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "flat_signals": flat_signals,
        }

        with self.engine.begin() as connection:
            result = connection.execute(text(queries.INSERT_BACKTEST_RESULT), params)
            backtest_id = result.scalar_one()

        return int(backtest_id)

    def get_leaderboard(self) -> pd.DataFrame:
        with self.engine.connect() as connection:
            df = pd.read_sql_query(
                sql=text(queries.GET_MODEL_LEADERBOARD),
                con=connection,
            )

        return df

class TradingSignalRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def upsert_signals_from_dataframe(
        self,
        run_id: int,
        asset_id: int,
        df: pd.DataFrame,
    ) -> None:
        """Save trading signals generated from backtest positions."""
        if df.empty:
            return

        required_columns = {
            "timestamp",
            "strategy_name",
            "signal",
            "position",
        }

        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        records = []

        for _, row in df.iterrows():
            records.append(
                {
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "timestamp": row["timestamp"],
                    "strategy_name": row["strategy_name"],
                    "signal": row["signal"],
                    "position": self._get_optional_int(row, "position"),
                    "predicted_class": self._get_optional_int(row, "predicted_class"),
                    "proba_down": self._get_optional_float(row, "proba_down"),
                    "proba_neutral": self._get_optional_float(row, "proba_neutral"),
                    "proba_up": self._get_optional_float(row, "proba_up"),
                    "target_return": self._get_optional_float(row, "target_return"),
                    "min_up_proba": self._get_optional_float(row, "min_up_proba"),
                    "min_down_proba": self._get_optional_float(row, "min_down_proba"),
                    "up_threshold": self._get_optional_float(row, "up_threshold"),
                    "down_threshold": self._get_optional_float(row, "down_threshold"),
                    "threshold_method": row.get("threshold_method", None),
                }
            )

        with self.engine.begin() as connection:
            connection.execute(text(queries.INSERT_TRADING_SIGNAL), records)

    @staticmethod
    def _get_optional_float(row: pd.Series, column: str) -> Optional[float]:
        if column not in row or pd.isna(row[column]):
            return None
        return float(row[column])

    @staticmethod
    def _get_optional_int(row: pd.Series, column: str) -> Optional[int]:
        if column not in row or pd.isna(row[column]):
            return None
        return int(row[column])
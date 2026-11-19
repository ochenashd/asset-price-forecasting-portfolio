"""
Feature engineering module for asset price forecasting.
Implements technical indicators, lagged features, volatility, calendar features, and trading targets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def validate_target_end_timestamps(index: pd.Index, timestamps: pd.Series) -> None:
    """Reject missing, misaligned or non-future label maturity metadata."""
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Observations require unique, chronological datetime timestamps.")
    if not isinstance(timestamps, pd.Series) or not timestamps.index.equals(index):
        raise ValueError("target_end_timestamps must be a Series aligned exactly with observations.")
    if not pd.api.types.is_datetime64_any_dtype(timestamps.dtype) or timestamps.isna().any():
        raise ValueError("target_end_timestamps must contain valid, non-missing datetimes.")
    try:
        future = timestamps > index
    except TypeError as exc:
        raise ValueError("Observation and target-end timezones must be compatible.") from exc
    if not future.all():
        raise ValueError("Every target end must be strictly after its observation timestamp.")


class FeatureEngineer:
    """Handles all feature engineering for time series price data."""

    def __init__(self, config: Dict):
        self.config = config
        self.window_sizes: List[int] = config.get("features", {}).get("window_sizes", [7, 14, 30, 60])
        self.technical_indicators: bool = config.get("features", {}).get("technical_indicators", True)
        self.target_config = config.get("target", {})
        self.target_type = self.target_config.get("type", "5d_class")
        self.horizon = int(self.target_config.get("horizon", 5))
        self.threshold = float(self.target_config.get("threshold", 0.005))

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standard technical indicators and lagged return features."""
        df = df.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Base returns are used to build lagged features and targets.
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        # Lagged returns are known at prediction time and are safe features.
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)
            df[f"log_return_lag_{lag}"] = df["log_returns"].shift(lag)

        # Moving averages, volatility, and relative price features.
        for w in self.window_sizes:
            ma = close.rolling(w).mean()
            ema = close.ewm(span=w, adjust=False).mean()
            vol = df["returns"].rolling(w).std().shift(1)

            df[f"ma_{w}"] = ma
            df[f"ema_{w}"] = ema
            df[f"vol_{w}"] = vol
            df[f"close_to_ma_{w}"] = close / ma - 1
            df[f"close_to_ema_{w}"] = close / ema - 1
            df[f"momentum_{w}"] = close.pct_change(w).shift(1)

        # RSI.
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD.
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands.
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_middle"] = bb_mid
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        if "volume" in df.columns:
            df["volume_change"] = df["volume"].pct_change().shift(1)
            df["volume_change_7d"] = df["volume"].pct_change(7).shift(1)

        # Lagged prices are safe, but relative features are usually easier for tree models.
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"close_lag_{lag}"] = close.shift(lag)
            df[f"close_change_lag_{lag}"] = close / close.shift(lag) - 1

        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based calendar features."""
        df = df.copy()
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_month_end"] = df.index.is_month_end.astype(int)
        return df

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add targets and maturity on the original chronological observation index."""
        df = df.copy()
        if (
            not isinstance(df.index, pd.DatetimeIndex)
            or df.index.hasnans
            or not df.index.is_unique
            or not df.index.is_monotonic_increasing
        ):
            raise ValueError("Targets require unique, chronological datetime timestamps.")
        horizon = 1 if self.target_type == "1d_return" else self.horizon
        if horizon < 1:
            raise ValueError("Target horizon must be a positive observation count.")
        df["target_end_timestamp"] = pd.Series(df.index, index=df.index).shift(-horizon)
        future_return = df["close"].shift(-horizon) / df["close"] - 1
        df["target_return"] = future_return

        if self.target_type == "5d_class":
            df["target"] = 1
            df.loc[future_return > self.threshold, "target"] = 2
            df.loc[future_return < -self.threshold, "target"] = 0
            df["target"] = df["target"].astype("Int64")
        elif self.target_type == "5d_return":
            df["target"] = future_return
        elif self.target_type == "1d_return":
            df["target"] = df["returns"].shift(-1)
            df["target_return"] = df["target"]
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")

        return df

    def create_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Create model-ready features and target columns."""
        logger.info(f"Engineering features for {ticker} | Shape: {df.shape}")

        df = self.add_technical_indicators(df)
        df = self.add_calendar_features(df)
        df = self.add_target(df)

        # Raw returns are removed to avoid same-day leakage. Lagged versions are kept.
        leak_columns = ["returns", "log_returns", "daily_return", "weekly_return"]
        df = df.drop(columns=[col for col in leak_columns if col in df.columns], errors="ignore")

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        validate_target_end_timestamps(df.index, df["target_end_timestamp"])

        logger.info(f"Final engineered shape: {df.shape} | Features: {df.columns.tolist()}")
        return df

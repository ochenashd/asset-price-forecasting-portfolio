
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging
from omegaconf import DictConfig


class DataPreprocessor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processed_path = Path(config.paths.processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, ticker: str) -> pd.DataFrame:
        from src.data.loader import DataLoader
        loader = DataLoader(self.config)
        return loader.load_symbol(ticker)


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Copy and order observations without repairing observed market prices.

        Forward-fill normalized price columns from past observations only. This
        provides continuity for model inputs, not evidence of an executable price.
        Leading price gaps, missing volume, and other missing fields remain missing.
        Observed extremes, zero prices, and negative futures prices are preserved.
        """
        if df.empty:
            return df.copy()
        df = df.copy().sort_index()
        df = df[~df.index.duplicated(keep='last')]

        price_columns = [
            column for column in ("open", "high", "low", "close", "adj_close")
            if column in df.columns
        ]
        df[price_columns] = df[price_columns].ffill()

        self.logger.info(f"Cleaned: {df.shape} | Columns: {df.columns.tolist()}")
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if len(df) < 100:
            return df, pd.DataFrame(), pd.DataFrame()
        test_len = int(len(df) * 0.15)
        val_len = int(len(df) * 0.1)
        train = df.iloc[:-test_len-val_len].copy()
        val = df.iloc[-test_len-val_len:-test_len].copy()
        test = df.iloc[-test_len:].copy()
        self.logger.info(f"Split -> Train:{len(train)} Val:{len(val)} Test:{len(test)}")
        return train, val, test

    def save_processed(self, df: pd.DataFrame, ticker: str, dataset_type: str):
        clean_ticker = ticker.replace("^", "").replace("/", "-")
        filename = f"{clean_ticker}_{dataset_type}.parquet"
        path = self.processed_path / filename
        df.to_parquet(path, compression='gzip')
        self.logger.info(f"Saved {dataset_type}: {filename} | {df.shape}")

    def detect_and_clean_outliers(
        self,
        df: pd.DataFrame,
        method: str = "rolling_mad",
        threshold: float = 5.0,
        **kwargs,
    ) -> Tuple[pd.DataFrame, int]:
        """Preserve the legacy interface without modifying observed prices.

        Outlier detection and replacement are disabled. The method, threshold,
        and extra options remain accepted for existing callers but are not used.
        Return an independent copy and a replacement count of zero.
        """
        self.logger.warning(
            "Outlier replacement is disabled; returning unchanged observations."
        )
        return df.copy(), 0


"""
Data loader module for raw Parquet files from yfinance.
Handles MultiIndex columns properly.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Union

import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig
from src.database.connection import create_engine_from_config, test_connection
from src.database.repository import AssetRepository, MarketPriceRepository

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and basic validation of raw market data files."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.raw_data_dir = Path(cfg.paths.raw_path)
        self.processed_dir = Path(cfg.paths.processed_path)

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.database_enabled = bool(cfg.get("database", {}).get("enabled", False))
        self.read_market_data_from_db = bool(cfg.get("database", {}).get("read_market_data", False))
        self.write_market_data_to_db = bool(cfg.get("database", {}).get("write_market_data", False))
        self.database_timeframe = str(cfg.get("database", {}).get("timeframe", "1d"))
        self.database_source = str(cfg.get("database", {}).get("source", "yfinance"))

        self.engine = None
        self.asset_repo = None
        self.price_repo = None

        if self.database_enabled:
            self.engine = create_engine_from_config(cfg)

            if test_connection(self.engine):
                self.asset_repo = AssetRepository(self.engine)
                self.price_repo = MarketPriceRepository(self.engine)
                logger.info("Database connection initialized in DataLoader.")
            else:
                logger.warning("Database connection failed. DataLoader will use files only.")
                self.engine = None

        logger.info(f"DataLoader initialized. Raw dir: {self.raw_data_dir}")

    def list_files(self, symbol: Optional[str] = None, pattern: str = "*.parquet") -> List[Path]:
        """List raw Parquet files, most recent first."""
        if symbol:
            clean_symbol = symbol.replace("^", "").replace("/", "-")
            files = list(self.raw_data_dir.glob(f"{clean_symbol}_*.parquet"))
        else:
            files = list(self.raw_data_dir.glob(pattern))
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

    def flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust handling of yfinance MultiIndex columns (Price, Ticker)."""
        if isinstance(df.columns, pd.MultiIndex):
            # Extract only the Price level (first level)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Standardize column names
        df.columns = [str(col).lower().strip() for col in df.columns]

        rename_map = {
            'adj close': 'adj_close',
            'adjclose': 'adj_close',
            'repaired?': 'repaired'
        }
        df = df.rename(columns=rename_map)

        # Keep only relevant columns
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        available_cols = [col for col in keep_cols if col in df.columns]
        if available_cols:
            df = df[available_cols]

        logger.info(f"Flattened columns: {df.columns.tolist()}")
        return df

    def load_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a single Parquet file with robust column handling."""
        file_path = Path(file_path)
        logger.info(f"Loading file: {file_path.name}")

        try:
            df = pd.read_parquet(file_path)

            # Flatten MultiIndex
            df = self.flatten_columns(df)

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]

            logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns | Columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
            return pd.DataFrame()

    def load_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a single Parquet file with robust column handling."""
        file_path = Path(file_path)
        logger.info(f"Loading file: {file_path.name}")

        try:
            df = pd.read_parquet(file_path)
            df = self.flatten_columns(df)

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df = df.set_index('date')
                df.index = pd.to_datetime(df.index)

            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]

            logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return pd.DataFrame()

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load latest data for a symbol from PostgreSQL or Parquet."""
        if self.database_enabled and self.read_market_data_from_db:
            df_db = self.load_symbol_from_database(symbol)
            if not df_db.empty:
                return df_db

        files = self.list_files(symbol=symbol)

        if not files:
            logger.warning(f"No raw files found for {symbol}")
            return pd.DataFrame()

        latest_file = files[0]
        df = self.load_single_file(latest_file)

        if self.database_enabled and self.write_market_data_to_db:
            self.save_symbol_to_database(symbol, df)

        return df

    def download_all(self):
        """Use existing raw data (can be extended later)."""
        logger.info("✅ Using existing raw data files.")
        # Return dict of loaded dataframes
        data = {}
        for ticker in self.cfg.tickers:
            data[ticker] = self.load_symbol(ticker)
        return data

    def load_symbol_from_database(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data for one symbol from PostgreSQL."""
        if self.price_repo is None:
            return pd.DataFrame()

        df = self.price_repo.get_prices(
            symbol=symbol,
            timeframe=self.database_timeframe,
            start_date=self.cfg.data.get("start_date"),
            end_date=self.cfg.data.get("end_date"),
        )

        if df.empty:
            logger.info(f"No database market data found for {symbol}.")
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        logger.info(f"Loaded {len(df):,} rows for {symbol} from PostgreSQL.")
        return df

    def save_symbol_to_database(self, symbol: str, df: pd.DataFrame) -> None:
        """Save OHLCV data for one symbol into PostgreSQL."""
        if self.asset_repo is None or self.price_repo is None:
            return

        if df.empty:
            return

        required_columns = {"open", "high", "low", "close"}
        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            logger.warning(f"Cannot save {symbol} to database. Missing columns: {missing_columns}")
            return

        asset_id = self.asset_repo.upsert_asset(
            symbol=symbol,
            asset_name=symbol,
            asset_type="unknown",
            exchange="unknown",
            quote_currency=None,
        )

        db_df = df.copy()
        db_df = db_df.reset_index()

        if "timestamp" not in db_df.columns:
            first_col = db_df.columns[0]
            db_df = db_df.rename(columns={first_col: "timestamp"})

        self.price_repo.upsert_prices_from_dataframe(
            asset_id=asset_id,
            df=db_df,
            timeframe=self.database_timeframe,
            source=self.database_source,
        )

        logger.info(f"Saved {len(db_df):,} OHLCV rows for {symbol} to PostgreSQL.")

"""Diagnostic preprocessing contracts: known bugs fail without skips or xfail marks.

The causal contracts and price-preservation contracts expose current defects.
The raw-input immutability and ordinary forward-fill controls currently pass.
All inputs are synthetic; no data loaders, models, or external services are used.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.preprocessor import DataPreprocessor


def make_ohlcv(closes):
    """Build internally consistent synthetic daily candles, including negative prices."""
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(len(closes), 1000.0),
        },
        index=pd.date_range("2020-01-01", periods=len(closes), freq="D"),
    )


@pytest.fixture
def preprocessor(tmp_path):
    """Use only a temporary output directory and never load market data."""
    config = OmegaConf.create({"paths": {"processed_path": str(tmp_path / "processed")}})
    return DataPreprocessor(config)


def test_appending_future_prices_does_not_change_cleaned_prefix(preprocessor):
    """Known bug: global IQR bounds change the historical outlier decision."""
    full = make_ohlcv([100.0] * 99 + [300.0] + [1000.0] * 200)
    prefix = full.iloc[:100].copy()

    cleaned_prefix = preprocessor.clean_data(prefix)
    cleaned_with_future = preprocessor.clean_data(full).loc[prefix.index]

    pd.testing.assert_frame_equal(cleaned_prefix, cleaned_with_future)


def test_leading_missing_prices_are_not_filled_from_future(preprocessor):
    """Known bug: backward filling creates a price before its first observation."""
    raw = make_ohlcv([np.nan] * 5 + [100.0] * 40)

    cleaned = preprocessor.clean_data(raw).reindex(raw.index)

    assert cleaned.iloc[:5][["open", "high", "low", "close"]].isna().all().all(), (
        "Leading prices must remain unavailable or their rows must be removed; "
        "future prices must not be backfilled."
    )


def test_appending_first_valid_price_does_not_fill_missing_history(preprocessor):
    """Known bug: adding future observations changes an all-missing prefix."""
    full = make_ohlcv([np.nan] * 10 + [100.0] * 40)
    prefix = full.iloc[:10].copy()

    cleaned_prefix = preprocessor.clean_data(prefix).reindex(prefix.index)
    cleaned_with_future = preprocessor.clean_data(full).reindex(prefix.index)

    pd.testing.assert_frame_equal(cleaned_prefix, cleaned_with_future)


def test_legitimate_extreme_price_movement_is_preserved(preprocessor):
    """Known bug: a valid large move is replaced by a rolling median."""
    raw = make_ohlcv([100.0] * 99 + [300.0])

    cleaned = preprocessor.clean_data(raw)

    assert cleaned.loc[raw.index[-1], "close"] == raw.iloc[-1]["close"], (
        "A valid synthetic price jump must not be silently classified as bad data."
    )


@pytest.mark.parametrize(
    "closes",
    [[100.0] * 99 + [-37.63], [-37.63] * 40],
    ids=["negative-shock", "sustained-negative-prices"],
)
def test_negative_crude_oil_futures_price_is_preserved(preprocessor, closes):
    """Known bug: generic OHLC names do not activate the crude-oil-specific bounds."""
    raw = make_ohlcv(closes)
    raw.attrs["symbol"] = "CL=F"

    cleaned = preprocessor.clean_data(raw).reindex(raw.index)

    assert cleaned.loc[raw.index[-1], "close"] == raw.iloc[-1]["close"], (
        "A legitimate negative futures price must not be clipped, replaced, or removed."
    )


@pytest.mark.parametrize(
    "closes",
    [
        [100.0] * 40,
        [np.nan] * 5 + [100.0] * 40,
        [100.0] * 99 + [300.0],
        [100.0] * 99 + [-37.63],
    ],
    ids=["ordinary", "missing", "extreme", "negative-futures"],
)
def test_cleaning_does_not_mutate_raw_input(preprocessor, closes):
    """Passing control: cleaning must preserve the caller's original observations."""
    raw = make_ohlcv(closes)
    original = raw.copy(deep=True)

    preprocessor.clean_data(raw)

    pd.testing.assert_frame_equal(raw, original)


def test_interior_missing_prices_can_use_past_observations(preprocessor):
    """Passing control: ordinary forward filling uses an already observed price."""
    raw = make_ohlcv([100.0] * 20 + [np.nan] + [100.0] * 20)

    cleaned = preprocessor.clean_data(raw)

    assert cleaned.loc[raw.index[20], "close"] == raw.loc[raw.index[19], "close"]

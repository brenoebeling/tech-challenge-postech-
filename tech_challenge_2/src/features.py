"""Feature engineering utilities for IBOVESPA modeling."""

from __future__ import annotations

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate required features for the IBOVESPA dataset."""
    data = df.copy()

    # Daily return (percentage change)
    data["daily_return"] = data["Close"].pct_change()

    # Price differences
    data["close_open_diff"] = data["Close"] - data["Open"]
    data["high_low_diff"] = data["High"] - data["Low"]

    # Moving averages of Close
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_10"] = data["Close"].rolling(window=10).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()

    # Rolling volatility (5-day standard deviation)
    data["volatility_5"] = data["Close"].rolling(window=5).std()

    # Close lags
    data["close_lag_1"] = data["Close"].shift(1)
    data["close_lag_2"] = data["Close"].shift(2)

    return data

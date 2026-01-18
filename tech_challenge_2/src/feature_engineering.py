"""Feature engineering for IBOVESPA daily historical data."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and target for IBOVESPA classification."""
    data = df.copy()

    # Daily return (percentage)
    data["daily_return"] = (data["Close"] - data["Close"].shift(1)) / data[
        "Close"
    ].shift(1)

    # Moving averages of Close
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_10"] = data["Close"].rolling(window=10).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()

    # Volatility: rolling std of daily returns (10 days)
    data["volatility_10"] = data["daily_return"].rolling(window=10).std()

    # Daily range features
    data["close_open"] = data["Close"] - data["Open"]
    data["high_low"] = data["High"] - data["Low"]

    # Close lags
    data["close_lag_1"] = data["Close"].shift(1)
    data["close_lag_2"] = data["Close"].shift(2)

    # Target: next day close higher than current close
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # Remove rows with nulls from rolling/shift operations
    data = data.dropna().reset_index(drop=True)

    # Keep only numeric columns and target for supervised modeling
    numeric_columns = data.select_dtypes(include="number").columns
    data = data[numeric_columns]

    return data


if __name__ == "__main__":
    # Example usage: assumes df is already loaded, datetime parsed, and sorted.
    df = pd.read_csv("data/ibovespa_preprocessed.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df_features = build_features(df)

    print(df_features.head())
    print(df_features.info())

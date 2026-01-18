"""Train an IBOVESPA next-day direction classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load IBOVESPA daily data and ensure datetime ordering."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create required features and target for modeling."""
    data = df.copy()

    # Target definition: next day close higher than current close
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # Feature engineering
    data["daily_return"] = (data["Close"] - data["Open"]) / data["Open"]
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_10"] = data["Close"].rolling(window=10).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()
    data["volatility_5"] = data["Close"].rolling(window=5).std()
    data["daily_range"] = data["High"] - data["Low"]
    data["close_lag_1"] = data["Close"].shift(1)
    data["close_lag_2"] = data["Close"].shift(2)

    # Remove rows without full feature availability
    data = data.dropna().reset_index(drop=True)

    # Remove last row after target creation (already NaN on target shift)
    data = data.iloc[:-1].reset_index(drop=True)

    return data


def temporal_train_test_split(data: pd.DataFrame, test_size: int = 30):
    """Split dataset into train and test sets preserving time order."""
    if len(data) <= test_size:
        raise ValueError("Dataset must have more rows than the test size.")

    train_data = data.iloc[:-test_size].reset_index(drop=True)
    test_data = data.iloc[-test_size:].reset_index(drop=True)
    return train_data, test_data


def train_model(train_df: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression classifier."""
    feature_cols = [
        "daily_return",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_5",
        "daily_range",
        "close_lag_1",
        "close_lag_2",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, test_df: pd.DataFrame) -> None:
    """Evaluate model and print metrics."""
    feature_cols = [
        "daily_return",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_5",
        "daily_range",
        "close_lag_1",
        "close_lag_2",
    ]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    confusion = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(confusion)


def save_model(model: LogisticRegression, output_path: Path | str) -> None:
    """Persist the trained model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "ibovespa.csv"

    df = load_data(data_path)
    df_features = create_features(df)

    train_df, test_df = temporal_train_test_split(df_features, test_size=30)

    trained_model = train_model(train_df)
    evaluate_model(trained_model, test_df)

    model_path = base_dir / "model" / "model.pkl"
    save_model(trained_model, model_path)

"""Train an IBOVESPA next-day direction classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from features import create_features


def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load IBOVESPA daily data and ensure datetime ordering."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target for next-day direction and drop last row."""
    data = df.copy()
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.iloc[:-1].reset_index(drop=True)
    return data


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and drop rows with NaNs."""
    data = create_features(df)
    data = data.dropna().reset_index(drop=True)
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
        "close_open_diff",
        "high_low_diff",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_5",
        "close_lag_1",
        "close_lag_2",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, test_df: pd.DataFrame) -> float:
    """Evaluate model and print metrics."""
    feature_cols = [
        "daily_return",
        "close_open_diff",
        "high_low_diff",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_5",
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

    return accuracy


def save_model(model: LogisticRegression, output_path: Path | str) -> None:
    """Persist the trained model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "ibovespa.csv"

    df_raw = load_data(data_path)
    df_target = add_target(df_raw)
    df_features = prepare_dataset(df_target)

    train_df, test_df = temporal_train_test_split(df_features, test_size=30)

    trained_model = train_model(train_df)
    test_accuracy = evaluate_model(trained_model, test_df)

    if test_accuracy < 0.75:
        print("Warning: test accuracy below 75% target.")

    model_path = base_dir / "model" / "model.pkl"
    save_model(trained_model, model_path)

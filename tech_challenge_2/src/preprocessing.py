"""Preprocess IBOVESPA data for binary classification."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_RENAME_MAP = {
    "Data": "date",
    "Último": "close",
    "Abertura": "open",
    "Máxima": "high",
    "Mínima": "low",
    "Vol.": "volume",
    "Var%": "variation_pct",
}

VOLUME_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
}


def parse_brazilian_float(value: str | float | int) -> float:
    """Convert Brazilian-formatted numeric strings to float."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    normalized = str(value).strip().replace(".", "").replace(",", ".")
    return float(normalized)


def parse_variation_pct(value: str | float | int) -> float:
    """Parse percentage strings like '1,23%' into float values."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    normalized = str(value).strip().replace("%", "").replace(",", ".")
    return float(normalized)


def parse_volume(value: str | float | int) -> float:
    """Convert volume strings with K/M/B suffix into absolute numbers."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().upper()
    match = re.fullmatch(r"([\d.,]+)([KMB]?)", text)
    if not match:
        return np.nan

    number_part, suffix = match.groups()
    number_value = parse_brazilian_float(number_part)
    multiplier = VOLUME_MULTIPLIERS.get(suffix, 1)
    return number_value * multiplier


def preprocess_ibovespa(
    input_path: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Load, preprocess, and persist the IBOVESPA dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    df = df.rename(columns=COLUMN_RENAME_MAP)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    df = df.sort_values("date").reset_index(drop=True)

    for column in ["close", "open", "high", "low"]:
        df[column] = df[column].apply(parse_brazilian_float)

    df["variation_pct"] = df["variation_pct"].apply(parse_variation_pct)
    df["volume"] = df["volume"].apply(parse_volume)

    df = df.ffill()

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.iloc[:-1].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    input_file = base_dir / "data" / "ibovespa.csv"
    output_file = base_dir / "data" / "ibovespa_preprocessed.csv"
    preprocess_ibovespa(input_file, output_file)

"""Data loading utilities for the Alternative Credit Scoring project."""

import pandas as pd

RAW_DATA_PATH = "german_credit_data.csv"


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw German Credit Data CSV."""
    return pd.read_csv(path, index_col=0)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and percentages for columns that have any."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    pct = (missing / len(df) * 100).round(1)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct})

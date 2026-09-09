"""Data loading, time-series splitting, and rolling-origin CV fold generation."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd
from src.config import DATA_PROCESSED, DATA_RAW, H


def load_clean_sales(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load cleaned sales time series, ensuring date parsing and ascending order."""
    file_path = Path(path) if path else DATA_PROCESSED / "sales_clean.csv"
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_promotions(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load promotional campaigns calendar with start and end dates parsed."""
    file_path = Path(path) if path else DATA_RAW / "promotions.csv"
    return pd.read_csv(file_path, parse_dates=["start_date", "end_date"])


def load_features(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load engineered features matrix (H=28, version 2)."""
    file_path = Path(path) if path else DATA_PROCESSED / f"features_h{H}_v2.csv"
    return pd.read_csv(file_path, parse_dates=["date"])


def split_by_date(
    df: pd.DataFrame,
    train_end: Union[str, pd.Timestamp],
    valid_start: Union[str, pd.Timestamp],
    valid_end: Union[str, pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict temporal split between training and validation/test windows.

    Guarantees no future leakage across the partition boundary.
    """
    train = df[df.date <= pd.Timestamp(train_end)].copy()
    valid = df[(df.date >= pd.Timestamp(valid_start)) & (df.date <= pd.Timestamp(valid_end))].copy()
    return train, valid


def generate_rolling_folds(
    start: str = "2016-07-01",
    cutoff: str = "2022-05-01",
    months: int = 6,
    buffer_days: int = 31,
) -> List[Tuple[str, str, str, str]]:
    """Generate sequential, non-overlapping 6-month validation folds with lag buffer.

    Delineates rolling-origin windows stopping before cutoff so as not to touch
    the official TEST_FOLD (valid_start: 2022-07-01).

    Returns:
        List of tuples: (fold_label, train_end_str, valid_start_str, valid_end_str)
    """
    folds = []
    cur = pd.Timestamp(start)
    cutoff_ts = pd.Timestamp(cutoff)
    idx = 1

    while cur < cutoff_ts:
        valid_start = cur
        valid_end = cur + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        train_end = valid_start - pd.Timedelta(days=buffer_days)
        folds.append((
            f"Fold {idx}",
            str(train_end.date()),
            str(valid_start.date()),
            str(valid_end.date()),
        ))
        cur = cur + pd.DateOffset(months=months)
        idx += 1

    return folds

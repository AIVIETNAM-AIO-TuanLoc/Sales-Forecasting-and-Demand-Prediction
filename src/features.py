"""Leak-free feature engineering for sales forecasting at horizon H=28."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import H, RETAIL_SPECIAL_DAYS, TARGET, TET_DATES


def add_lag_features(df: pd.DataFrame, col: str = TARGET, h: int = H) -> pd.DataFrame:
    """Derive historical revenue lag and rolling summary features strictly >= h days ago."""
    y = df[col]
    safe = y.shift(h)

    for lag in (28, 35, 42, 56, 91, 182, 364, 371):
        df[f"lag_{lag}"] = y.shift(lag)

    for w in (7, 28, 91, 364):
        df[f"roll_mean_{w}"] = safe.rolling(w, min_periods=max(2, w // 4)).mean()
        df[f"roll_std_{w}"] = safe.rolling(w, min_periods=max(2, w // 4)).std()

    df["roll_min_91"] = safe.rolling(91, min_periods=20).min()
    df["roll_max_91"] = safe.rolling(91, min_periods=20).max()

    # Centered-like seasonal smoothers around lag 364 (1 year prior)
    df["lag364_smooth7"] = y.shift(364 - 3).rolling(7, min_periods=3).mean()
    df["lag364_smooth28"] = y.shift(364 - 14).rolling(28, min_periods=10).mean()

    df["ratio_28_364"] = df.roll_mean_28 / df.roll_mean_364
    df["ratio_91_364"] = df.roll_mean_91 / df.roll_mean_364
    df["trend_28_91"] = df.roll_mean_28 - df.roll_mean_91
    df["cv_91"] = df.roll_std_91 / df.roll_mean_91
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar and cyclical Fourier features."""
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["dom"] = d.dt.day
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["doy"] = d.dt.dayofyear
    df["days_in_month"] = d.dt.days_in_month
    df["dom_reverse"] = df.days_in_month - df.dom
    df["is_weekend"] = (df.dow >= 5).astype("int8")
    df["is_eom"] = (df.dom_reverse <= 3).astype("int8")
    df["is_som"] = (df.dom <= 2).astype("int8")
    df["is_odd_year"] = (d.dt.year % 2).astype("int8")

    for k in (1, 2, 3, 4):
        df[f"fourier_sin{k}"] = np.sin(2 * np.pi * k * df.doy / 365.25)
        df[f"fourier_cos{k}"] = np.cos(2 * np.pi * k * df.doy / 365.25)
    return df


def add_tet_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Vietnamese Lunar New Year (Tết) proximity counters and holiday flags."""
    d = df["date"]
    tet = pd.to_datetime(pd.Series(TET_DATES))
    diff = np.stack([(d - t).dt.days.values for t in tet])
    df["days_to_tet"] = diff[np.abs(diff).argmin(axis=0), np.arange(len(df))]
    df["tet_before"] = df.days_to_tet.between(-30, -1).astype("int8")
    df["tet_after"] = df.days_to_tet.between(0, 14).astype("int8")

    special_set = set(RETAIL_SPECIAL_DAYS)
    df["is_special_day"] = np.array(
        [(m, dd) in special_set for m, dd in zip(d.dt.month, d.dt.day)],
        dtype="int8",
    )
    return df


def add_promo_features(df: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    """Derive baseline promotional count, max discount, and category flags."""
    df["promo_n"], df["promo_disc_max"], df["promo_day_idx"] = 0, 0.0, -1
    df["promo_streetwear"], df["promo_outdoor"] = 0, 0

    for _, p in promotions.iterrows():
        m = (df.date >= p.start_date) & (df.date <= p.end_date)
        if not m.any():
            continue
        df.loc[m, "promo_n"] += 1
        df.loc[m, "promo_disc_max"] = np.maximum(
            df.loc[m, "promo_disc_max"],
            float(p.discount_value),
        )
        idx = (df.loc[m, "date"] - p.start_date).dt.days
        cur = df.loc[m, "promo_day_idx"]
        df.loc[m, "promo_day_idx"] = np.where(cur < 0, idx, np.minimum(cur, idx))

        if p.applicable_category == "Streetwear":
            df.loc[m, "promo_streetwear"] = 1
        elif p.applicable_category == "Outdoor":
            df.loc[m, "promo_outdoor"] = 1

    df["promo_any"] = (df.promo_n > 0).astype("int8")
    return df


def add_promo_features_v2(df: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    """Derive extended promotional attributes including channels, minimum orders, and duration."""
    df = add_promo_features(df, promotions)
    channels = sorted(promotions.promo_channel.dropna().unique())

    df["promo_is_percentage"] = 0
    df["stackable_flag"] = 0
    df["min_order_value"] = 0.0
    df["promo_days_left"] = -1
    for ch in channels:
        df[f"promo_ch_{ch}"] = 0

    best_discount = pd.Series(-1.0, index=df.index)
    for _, p in promotions.iterrows():
        m = (df.date >= p.start_date) & (df.date <= p.end_date)
        if not m.any():
            continue
        take = m & (p.discount_value > best_discount)
        df.loc[take, "promo_is_percentage"] = int(p.promo_type == "percentage")
        df.loc[take, "stackable_flag"] = int(p.stackable_flag)
        df.loc[take, "min_order_value"] = float(p.min_order_value)
        df.loc[take, "promo_days_left"] = (p.end_date - df.loc[take, "date"]).dt.days
        best_discount.loc[take] = p.discount_value

        df.loc[m, f"promo_ch_{p.promo_channel}"] = 1

    return df


def add_cogs_lag_features(df: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """Derive COGS past historical lags and past profit margin (leak-free)."""
    c = df["cogs"]
    y = df[TARGET]
    safe = c.shift(h)

    for lag in (28, 91, 364):
        df[f"cogs_lag_{lag}"] = c.shift(lag)

    df["cogs_roll_mean_28"] = safe.rolling(28, min_periods=7).mean()
    df["cogs_roll_mean_91"] = safe.rolling(91, min_periods=20).mean()
    df["margin_lag_28"] = 1 - (c.shift(28) / y.shift(28))
    return df


def add_specific_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive countdown distances to 11/11 and 12/12 shopping festivals."""
    d = df["date"]
    for name, month, day in [("1111", 11, 11), ("1212", 12, 12)]:
        marks = pd.to_datetime([
            f"{y}-{month:02d}-{day:02d}"
            for y in range(d.dt.year.min() - 1, d.dt.year.max() + 2)
        ])
        diff = np.stack([(d - m).dt.days.values for m in marks])
        df[f"days_to_{name}"] = diff[np.abs(diff).argmin(axis=0), np.arange(len(df))]
    return df


def build_feature_pipeline(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame,
    include_extended_promo: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Execute end-to-end feature pipeline producing the optimal 62-feature matrix."""
    df = sales_df.copy().sort_values("date").reset_index(drop=True)
    df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_tet_features(df)

    if include_extended_promo:
        df = add_promo_features_v2(df, promotions_df)
    else:
        df = add_promo_features(df, promotions_df)

    feature_cols = [c for c in df.columns if c not in ("date", TARGET, "cogs")]
    ready_df = df.dropna(subset=["lag_371", "roll_mean_364"]).reset_index(drop=True)
    return ready_df, feature_cols


def leakage_probe_cogs(raw_df: pd.DataFrame, n_checks: int = 8, h: int = H) -> Tuple[float, int, int]:
    """Test data leakage by corrupting the most recent h target and cogs values.

    Returns max difference observed across lagged feature columns (must be 0).
    """
    fa = add_cogs_lag_features(add_lag_features(raw_df.copy()))
    cols = [
        c for c in fa.columns
        if c.startswith(("lag", "roll", "ratio", "trend", "cv_", "cogs_lag", "cogs_roll", "margin_lag"))
    ]
    rows = np.linspace(max(h, 400), len(raw_df) - 1, n_checks, dtype=int)
    worst = 0.0

    for row in rows:
        b = raw_df.copy()
        block = slice(row - h + 1, row + 1)
        b.loc[b.index[block], TARGET] *= 9.3
        b.loc[b.index[block], "cogs"] *= 9.3
        fb = add_cogs_lag_features(add_lag_features(b))
        worst = max(
            worst,
            float((fa[cols].iloc[row].fillna(-1) - fb[cols].iloc[row].fillna(-1)).abs().max()),
        )

    return worst, len(rows), len(cols)

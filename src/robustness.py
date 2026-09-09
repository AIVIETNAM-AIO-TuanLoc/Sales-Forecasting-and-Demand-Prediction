"""Robustness testing, error segment analysis, and retrain backtesting."""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.config import TARGET
from src.data import split_by_date
from src.metrics import bias, wape
from src.models import fit_lightgbm


def segment_wape(detail_df: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    """Compute WAPE and forecast bias segmented across categorical or discretized slices."""
    rows = []
    for value, g in detail_df.groupby(key, observed=True):
        rows.append({
            label: value,
            "n_days": len(g),
            "WAPE": wape(g[TARGET].values, g["pred"].values),
            "bias": bias(g[TARGET].values, g["pred"].values),
        })
    return pd.DataFrame(rows).set_index(label)


def backtest_retrain_strategy(
    df: pd.DataFrame,
    features: List[str],
    demo_fold_train_end: str = "2021-12-01",
    demo_valid_start: str = "2022-01-01",
    demo_valid_end: str = "2022-06-30",
    params: Any = None,
) -> Dict[str, Any]:
    """Compare a static single-fit model against monthly retraining across a 6-month period."""
    # Strategy A: Single fit
    train_a, valid_a = split_by_date(df, demo_fold_train_end, demo_valid_start, demo_valid_end)
    model_a = fit_lightgbm(train_a, features, params=params)
    pred_a = np.expm1(model_a.predict(valid_a[features]))
    strategy_a = {
        "WAPE": wape(valid_a[TARGET].values, pred_a),
        "bias": bias(valid_a[TARGET].values, pred_a),
    }

    # Strategy B: Monthly retrain
    months = pd.date_range(demo_valid_start, demo_valid_end, freq="MS")
    records_b = []
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        train_end_b = month_start - pd.Timedelta(days=31)
        train_b, valid_b = split_by_date(
            df, str(train_end_b.date()), str(month_start.date()), str(month_end.date())
        )
        model_b = fit_lightgbm(train_b, features, params=params)
        pred_b = np.expm1(model_b.predict(valid_b[features]))
        records_b.append(valid_b[["date", TARGET]].assign(pred=pred_b))

    pred_b_all = pd.concat(records_b, ignore_index=True)
    strategy_b = {
        "WAPE": wape(pred_b_all[TARGET].values, pred_b_all["pred"].values),
        "bias": bias(pred_b_all[TARGET].values, pred_b_all["pred"].values),
    }

    bias_reduced = abs(strategy_b["bias"]) < abs(strategy_a["bias"])
    wape_reduced = strategy_b["WAPE"] < strategy_a["WAPE"]

    return {
        "strategy_a_single_fit": strategy_a,
        "strategy_b_monthly_retrain": strategy_b,
        "bias_reduced": bool(bias_reduced),
        "wape_reduced": bool(wape_reduced),
    }

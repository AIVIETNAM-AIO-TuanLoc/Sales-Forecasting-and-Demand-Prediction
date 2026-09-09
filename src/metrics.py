"""Evaluation metrics and statistical hypothesis testing for time-series forecasting."""

from typing import Any, Dict, Tuple, Union
import numpy as np
from scipy import stats
from src.config import H


def wape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Weighted Absolute Percentage Error: sum(|y - p|) / sum(y)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    denom = np.sum(y)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y - p)) / denom)


def bias(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Mean Percentage Forecast Bias: (mean(p) - mean(y)) / mean(y)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_mean = y.mean()
    if y_mean == 0:
        return 0.0
    return float((p.mean() - y_mean) / y_mean)


def r2_score_custom(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Coefficient of Determination (R^2)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_res = np.sum((y - p) ** 2)
    return float(1 - ss_res / ss_tot)


def evaluate_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Calculate standard benchmark metrics: WAPE, bias, and R^2."""
    return {
        "WAPE": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "R2": r2_score_custom(y_true, y_pred),
    }


def diebold_mariano(
    e1: Union[np.ndarray, list],
    e2: Union[np.ndarray, list],
    h: int = H,
    loss: str = "absolute",
) -> Tuple[float, float]:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample adjustment.

    Args:
        e1: Error sequence for model 1 (actual - pred1).
        e2: Error sequence for model 2 (actual - pred2).
        h: Forecast horizon (default H=28).
        loss: 'absolute' (L1) or 'squared' (L2).

    Returns:
        Tuple of (DM_statistic, p_value).
        A negative DM_statistic indicates model 1 has lower expected loss than model 2.
    """
    e1_arr, e2_arr = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    if loss == "squared":
        d = e1_arr ** 2 - e2_arr ** 2
    else:
        d = np.abs(e1_arr) - np.abs(e2_arr)

    T = len(d)
    d_mean = d.mean()

    # Autocovariance up to lag h - 1
    var_d = np.var(d, ddof=0)
    for lag in range(1, min(h, T)):
        cov = np.mean((d[:-lag] - d_mean) * (d[lag:] - d_mean))
        var_d += 2 * cov
    var_d = max(var_d, 1e-12)

    dm_stat = d_mean / np.sqrt(var_d / T)
    # Harvey-Leybourne-Newbold small sample correction factor
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat *= correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=T - 1))
    return float(dm_stat), float(p_value)

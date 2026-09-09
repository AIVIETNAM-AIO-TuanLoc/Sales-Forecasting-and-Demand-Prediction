"""Exploratory data analysis, changepoint detection, and non-parametric statistical tests."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy import stats
from src.config import TARGET


def kruskal_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: Any,
    group_name: str,
) -> Dict[str, float]:
    """Perform Kruskal-Wallis non-parametric ANOVA and compute epsilon-squared effect size."""
    groups = [g[value_col].values for _, g in df.groupby(group_col)]
    stat, p = stats.kruskal(*groups)
    n_groups = len(groups)
    n = len(df)
    eta_sq = (stat - n_groups + 1) / (n - n_groups)
    return {
        "H": float(stat),
        "p": float(p),
        "effect_size": float(eta_sq),
    }


def detect_changepoints(
    series: np.ndarray,
    penalty: float = 15.0,
    min_size: int = 60,
) -> List[int]:
    """Detect structural regime shifts using ruptures Pelt with an RBF kernel."""
    import ruptures as rpt
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(series)
    breakpoints = algo.predict(pen=penalty)
    return [b for b in breakpoints if b < len(series)]


def promo_flags_for_day(sales: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    """Label promo_type, promo_channel, and stackable status per day."""
    out = sales[["date", TARGET]].copy()
    out["promo_type"] = "none"
    out["promo_channel"] = "none"
    out["stackable_flag"] = 0
    out["has_promo"] = 0
    best_discount = pd.Series(-1.0, index=out.index)

    for _, p in promotions.iterrows():
        m = (out.date >= p.start_date) & (out.date <= p.end_date)
        if not m.any():
            continue
        take = m & (p.discount_value > best_discount)
        out.loc[take, "promo_type"] = p.promo_type
        out.loc[take, "promo_channel"] = p.promo_channel
        out.loc[take, "stackable_flag"] = int(p.stackable_flag)
        out.loc[m, "has_promo"] = 1
        best_discount.loc[take] = p.discount_value
    return out


def compare_vs_baseline(
    flagged: pd.DataFrame,
    group_col: str,
    min_count: int = 15,
) -> pd.DataFrame:
    """Compare normalized revenue of promo groups against non-promotional days via Mann-Whitney U."""
    baseline = flagged.loc[flagged.has_promo == 0, "year_ratio"]
    rows = []
    for value, g in flagged[flagged.has_promo == 1].groupby(group_col):
        if len(g) < min_count:
            continue
        stat, p = stats.mannwhitneyu(g.year_ratio, baseline, alternative="two-sided")
        rows.append({
            group_col: value,
            "n_days": len(g),
            "mean_ratio": g.year_ratio.mean(),
            "lift_pct": 100 * (g.year_ratio.mean() / baseline.mean() - 1),
            "p_value": p,
        })
    return pd.DataFrame(rows).sort_values("lift_pct", ascending=False)

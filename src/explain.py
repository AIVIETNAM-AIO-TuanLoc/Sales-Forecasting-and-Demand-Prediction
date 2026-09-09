"""Model interpretability, TreeSHAP attributions, and interaction analysis."""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_shap_explanations(model: Any, X: pd.DataFrame) -> Tuple[Any, np.ndarray, float, float]:
    """Compute TreeSHAP values, expected baseline, and verify additivity error."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = float(np.ravel(explainer.expected_value)[0])

    prediction = model.predict(X)
    reconstructed = base_value + shap_values.sum(axis=1)
    additivity_error = float(np.abs(reconstructed - prediction).max())
    return explainer, shap_values, base_value, additivity_error


def compute_group_shap(
    mean_abs_series: pd.Series,
    groups: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate global SHAP importance into logical business feature groups."""
    total_abs = float(mean_abs_series.sum())
    group_shap = {}
    for name, cols in groups.items():
        s = float(mean_abs_series[cols].sum())
        group_shap[name] = {
            "n_columns": len(cols),
            "total": s,
            "share_pct": 100 * s / total_abs,
            "mean_per_column": s / len(cols) if cols else 0.0,
        }
    return group_shap


def compute_top_interactions(
    explainer: Any,
    X: pd.DataFrame,
    features: List[str],
    top_n: int = 12,
) -> List[Dict[str, Any]]:
    """Compute pairwise SHAP interaction values and return the top interactive feature pairs."""
    interaction_values = explainer.shap_interaction_values(X)
    inter_mean_abs = np.abs(interaction_values).mean(axis=0)
    np.fill_diagonal(inter_mean_abs, 0)

    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append({
                "feature_1": features[i],
                "feature_2": features[j],
                "mean_abs_interaction": float(inter_mean_abs[i, j]),
            })
    pairs.sort(key=lambda x: -x["mean_abs_interaction"])
    return pairs[:top_n]

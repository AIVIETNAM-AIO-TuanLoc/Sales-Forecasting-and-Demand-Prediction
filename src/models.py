"""Model construction, Optuna hyperparameter optimization, and ensemble routines."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import SEED, TARGET
from src.data import split_by_date
from src.metrics import wape


def build_model(name: str, params: Dict[str, Any], seed: int = SEED) -> Any:
    """Instantiate a regressor with the given hyperparameter dictionary."""
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **params)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=seed, **params)
    if name == "DecisionTree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=seed, **params)
    raise ValueError(f"Unknown model name: {name}")


def get_baseline_models(seed: int = SEED) -> Dict[str, Any]:
    """Get the 5 baseline models initialized with default settings."""
    boosting = dict(
        n_estimators=700, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=1.0, n_jobs=-1, random_state=seed,
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=15, random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.85, random_state=seed),
        "XGBoost": xgb.XGBRegressor(max_depth=5, tree_method="hist", **boosting),
        "LightGBM": lgb.LGBMRegressor(num_leaves=31, min_child_samples=20, subsample_freq=5, verbose=-1, **boosting),
    }


def default_params(name: str) -> Dict[str, Any]:
    """Retrieve default baseline parameters for top models."""
    boosting = dict(
        n_estimators=700, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=1.0,
    )
    defaults = {
        "LightGBM": dict(num_leaves=31, min_child_samples=20, subsample_freq=5, **boosting),
        "XGBoost": dict(max_depth=5, **boosting),
        "RandomForest": dict(n_estimators=400, max_depth=14, min_samples_leaf=3),
        "GradientBoosting": dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.85),
    }
    if name not in defaults:
        raise ValueError(f"No default params for: {name}")
    return defaults[name]


def suggest_params(trial: Any, name: str) -> Dict[str, Any]:
    """Define Optuna hyperparameter search spaces."""
    if name == "LightGBM":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        )
    if name == "XGBoost":
        return dict(
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        )
    if name == "RandomForest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600),
            max_depth=trial.suggest_int("max_depth", 6, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        )
    if name == "GradientBoosting":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
        )
    raise ValueError(f"Unsupported model for tuning: {name}")


def make_objective(
    name: str,
    folds: List[Tuple[str, str, str, str]],
    df: pd.DataFrame,
    features: List[str],
) -> Callable[[Any], float]:
    """Generate an Optuna objective function evaluating cross-fold mean WAPE."""
    def objective(trial: Any) -> float:
        params = suggest_params(trial, name)
        scores = []
        for _, train_end, valid_start, valid_end in folds:
            train, valid = split_by_date(df, train_end, valid_start, valid_end)
            model = build_model(name, params)
            model.fit(train[features], np.log1p(train[TARGET].values))
            pred = np.expm1(model.predict(valid[features]))
            scores.append(wape(valid[TARGET].values, pred))
        return float(np.mean(scores))
    return objective


def evaluate_holt_winters(
    df: pd.DataFrame,
    folds: List[Tuple[str, str, str, str]],
) -> List[float]:
    """Evaluate classical Holt-Winters triple exponential smoothing on validation folds."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    scores = []
    for _, train_end, valid_start, valid_end in folds:
        train, valid = split_by_date(df, train_end, valid_start, valid_end)
        y_train = train.set_index("date")[TARGET].asfreq("D")
        fit = ExponentialSmoothing(
            y_train, trend=None, seasonal="add",
            seasonal_periods=7, initialization_method="estimated",
        ).fit()
        pred = fit.forecast(len(valid))
        scores.append(wape(valid[TARGET].values, pred.values))
    return scores


def fit_lightgbm(
    train_df: pd.DataFrame,
    features: List[str],
    params: Optional[Dict[str, Any]] = None,
    seed: int = SEED,
) -> Any:
    """Helper to fit a LightGBM regressor on log1p-transformed target."""
    import lightgbm as lgb
    p = params or default_params("LightGBM")
    model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, subsample_freq=1, **p)
    model.fit(train_df[features], np.log1p(train_df[TARGET].values))
    return model


def compute_ensemble_weights(val_wapes: Dict[str, float]) -> Dict[str, float]:
    """Calculate inverse-error weights for model ensembling."""
    inv_errors = {m: 1.0 / val_wapes[m] for m in val_wapes}
    total_inv = sum(inv_errors.values())
    return {m: inv_errors[m] / total_inv for m in inv_errors}


def train_quantile_models(
    train_df: pd.DataFrame,
    features: List[str],
    params: Dict[str, Any],
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    seed: int = SEED,
) -> Dict[float, Any]:
    """Fit quantile regression LightGBM models for prediction intervals."""
    import lightgbm as lgb
    clean_params = {k: v for k, v in params.items() if k != "subsample_freq"}
    fitted = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q, random_state=seed,
            n_jobs=-1, verbose=-1, subsample_freq=1, **clean_params,
        )
        m.fit(train_df[features], np.log1p(train_df[TARGET].values))
        fitted[q] = m
    return fitted


# ---------------------------------------------------------------------------
# Fold evaluation helpers (previously inline in notebooks 02–04)
# ---------------------------------------------------------------------------

def evaluate_fold(
    df: pd.DataFrame,
    fold: Tuple[str, str, str, str],
    columns: List[str],
    seed: int = SEED,
) -> float:
    """Fit LightGBM on a single fold and return the validation WAPE."""
    train, valid = split_by_date(df, fold[1], fold[2], fold[3])
    model = fit_lightgbm(train, columns, seed=seed)
    pred = np.expm1(model.predict(valid[columns]))
    return wape(valid[TARGET].values, pred)


def cross_validate(
    df: pd.DataFrame,
    columns: List[str],
    folds: Optional[List[Tuple[str, str, str, str]]] = None,
    cv_folds: Optional[List[str]] = None,
    seed: int = SEED,
) -> Dict[str, float]:
    """Run LightGBM cross-validation across all folds; report per-fold and mean CV WAPE."""
    from src.config import FOLDS as DEFAULT_FOLDS, CV_FOLDS as DEFAULT_CV_FOLDS
    _folds = folds if folds is not None else DEFAULT_FOLDS
    _cv_folds = cv_folds if cv_folds is not None else DEFAULT_CV_FOLDS
    scores = {f[0]: evaluate_fold(df, f, columns, seed=seed) for f in _folds}
    scores["CV"] = float(np.mean([scores[f] for f in _cv_folds]))
    return scores


def evaluate_fold_predictions(
    df: pd.DataFrame,
    fold: Tuple[str, str, str, str],
    columns: List[str],
    model_name: str,
    seed: int = SEED,
) -> pd.DataFrame:
    """Fit a named baseline model on a fold and return a predictions DataFrame."""
    _, train_end, valid_start, valid_end = fold
    train, valid = split_by_date(df, train_end, valid_start, valid_end)
    model = get_baseline_models(seed=seed)[model_name]
    model.fit(train[columns], np.log1p(train[TARGET].values))
    pred = np.expm1(model.predict(valid[columns]))
    return valid[["date", TARGET]].assign(model=model_name, fold=fold[0], pred=pred)


def evaluate_full(
    name: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    features: List[str],
    folds: List[Tuple[str, str, str, str]],
    seed: int = SEED,
) -> List[float]:
    """Evaluate a named model with given params across all provided folds; return WAPE scores."""
    scores = []
    for _, train_end, valid_start, valid_end in folds:
        train, valid = split_by_date(df, train_end, valid_start, valid_end)
        model = build_model(name, params, seed=seed)
        model.fit(train[features], np.log1p(train[TARGET].values))
        pred = np.expm1(model.predict(valid[features]))
        scores.append(wape(valid[TARGET].values, pred))
    return scores


def collect_predictions(
    name: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    features: List[str],
    folds: List[Tuple[str, str, str, str]],
    seed: int = SEED,
) -> pd.DataFrame:
    """Collect out-of-fold predictions from a named model across all provided folds."""
    records = []
    for fold in folds:
        _, train_end, valid_start, valid_end = fold
        train, valid = split_by_date(df, train_end, valid_start, valid_end)
        model = build_model(name, params, seed=seed)
        model.fit(train[features], np.log1p(train[TARGET].values))
        pred = np.expm1(model.predict(valid[features]))
        records.append(valid[["date", TARGET]].assign(model=name, fold=fold[0], pred=pred))
    return pd.concat(records, ignore_index=True)

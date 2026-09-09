# Sales Forecasting and Demand Prediction (Horizon $H=28$)

An enterprise-grade, leak-free time-series forecasting and demand prediction pipeline with a forecasting horizon of **$H = 28$ days** (4 weeks ahead), built with Python, LightGBM, XGBoost, Optuna, and TreeSHAP.

---

## 1. Overview & Methodological Disciplines

Forecasting daily sales across retail operations requires strict adherence to time-series modeling principles to prevent data leakage and ensure real-world generalization:

- **Strict Leakage Prevention**: All features (lags, rolling statistics, indicators) use information strictly available at or prior to $t - H$ ($t - 28$ days). An automated temporal corruption test verifies that changes in future values do not alter historical feature matrices.
- **12-Fold Rolling-Origin Cross-Validation**: Replaces fragile small-window validation with 12 non-overlapping 6-month validation windows (with 31-day lag buffers) spanning from mid-2016 through mid-2022.
- **Diebold-Mariano Hypothesis Testing**: Pairwise statistical testing with Harvey-Leybourne-Newbold small-sample adjustment determines whether WAPE differences between models are statistically distinguishable.
- **Single-Shot Test Discipline**: The official test set (`TEST_FOLD`: 2022-07-01 to 2022-12-31, 184 days) is unblinded strictly once for final model evaluation.
- **Bayesian Optimization & Probabilistic Ensembling**: Hyperparameter optimization via Optuna TPE over recent folds, combined with inverse-variance weighted blending and quantile regression ($Q_{10}, Q_{90}$) for prediction intervals.
- **Explainable AI (XAI)**: Exact TreeSHAP feature attributions, logical group contributions, and pairwise feature interaction analysis.

---

## 2. Key Performance Milestones

| Metric / Stage | Baseline Benchmark | Upgraded Architecture | Gain / Significance |
|---|---|---|---|
| **Feature Set Size** | 53 features | **62 features** (+promo channels, min order, stackable) | +9 features, -0.35% CV WAPE |
| **Validation Framework** | 3 folds (2 CV points) | **12 rolling-origin folds + Diebold-Mariano tests** | Proven statistical indistinguishability |
| **Hyperparameter Tuning** | Default parameters | **Optuna Bayesian HPO across 6 folds** | +4.19% gain on 12-fold LightGBM |
| **Ensemble Support** | Single model | **4-model weighted ensemble** | +0.44% lower WAPE vs single tree |
| **Test Set WAPE** | 0.2038 | **0.20308** | **+0.35% improvement** |
| **Test Set $R^2$** | 0.6398 | **0.6461** | **+0.98% improvement** |
| **Statistical Baseline** | N/A | **Holt-Winters Exp. Smoothing (WAPE 1.6927)** | Trees beat Holt-Winters by 84.4% |
| **Uncertainty Bounds** | Point forecast only | **Quantile bounds ($Q_{10} - Q_{90}$)** | 53.04% empirical coverage |

---

## 3. Repository Structure

```
Sales-Forecasting-and-Demand-Prediction/
├── src/                                # Reusable modular Python package
│   ├── __init__.py                     # Package metadata
│   ├── config.py                       # Centralized paths, constants, and parameters
│   ├── data.py                         # Data loading and 12-fold rolling CV generator
│   ├── eda.py                          # Kruskal-Wallis tests, changepoints, and promo lift
│   ├── explain.py                      # TreeSHAP, group SHAP, and interaction helpers
│   ├── features.py                     # Leak-free feature transforms (H >= 28)
│   ├── metrics.py                      # WAPE, bias, R^2, and Diebold-Mariano tests
│   ├── models.py                       # Model builders, Optuna HPO, and quantile regressors
│   └── robustness.py                   # Segmented error slicing and retrain backtesting
├── notebooks/                          # Step-by-step reproducible experiment notebooks
│   ├── 01_deep_eda.ipynb               # Phase 1: STL decomposition & changepoints
│   ├── 02_feature_engineering.ipynb    # Phase 2: Feature creation & ablation study
│   ├── 03_validation.ipynb             # Phase 3: 12-fold rolling CV & Diebold-Mariano
│   ├── 04_modeling.ipynb               # Phase 4: Optuna HPO & probabilistic ensembling
│   ├── 05_explain.ipynb                # Phase 5: Single test unblinding & TreeSHAP
│   ├── 06_robustness.ipynb             # Phase 6: Subgroup slicing & retrain backtest
│   └── 07_report.ipynb                 # Phase 7: Synthesis & final report generation
├── tests/                              # Automated unit and regression test suite
│   └── test_pipeline.py                # Tests for metrics, data, features, and models
├── data/
│   ├── raw/                            # sales.csv, promotions.csv
│   └── processed/                      # sales_clean.csv, features_h28_v2.csv
├── results/                            # Machine-readable JSON milestone artifacts (01 to 07)
├── images/                             # Generated figures and evaluation plots
├── requirements.txt                    # Project dependencies
├── ARCHITECTURE.md                     # Detailed architecture and flow diagrams
└── README.md                           # This documentation
```

---

## 4. Setup and Installation

### Prerequisites
- Python 3.10+
- Recommended: [`uv`](https://github.com/astral-sh/uv) or standard `venv`

### Setup with `uv` (Recommended)
```bash
# Create virtual environment
uv venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Setup with standard `pip`
```bash
python -m venv .venv
# Activate virtual environment
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Unix/macOS

pip install -r requirements.txt
```

---

## 5. Running the Tests & Verification

Run the automated test suite to verify metric correctness, leak-free feature engineering, and model instantiations:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 6. Execution Workflow

The notebooks in `notebooks/` are designed to run sequentially:

1. **`01_deep_eda.ipynb`**: Performs STL seasonal-trend decomposition, Kruskal-Wallis ANOVA tests for day-of-week/month seasonality, and ruptures changepoint detection.
2. **`02_feature_engineering.ipynb`**: Computes lag ($H \ge 28$) and rolling features, cyclical Fourier terms, Tết holiday distance, and runs a 5-way ablation experiment to select the optimal 62-feature matrix (`features_h28_v2.csv`).
3. **`03_validation.ipynb`**: Executes 12-fold rolling-origin backtesting across 5 model families and computes pairwise Diebold-Mariano test matrices.
4. **`04_modeling.ipynb`**: Runs Optuna Bayesian hyperparameter optimization, evaluates against Holt-Winters exponential smoothing, builds weighted ensembles, and trains quantile regression bounds ($Q_{10}, Q_{90}$).
5. **`05_explain.ipynb`**: Unblinds the official `TEST_FOLD` once, benchmarks upgraded models, and runs TreeSHAP global, group, and local waterfall explanations.
6. **`06_robustness.ipynb`**: Evaluates errors across months, promotion conditions, and sales quintiles, and backtests periodic retraining strategies.
7. **`07_report.ipynb`**: Aggregates all JSON milestone artifacts and exports the comprehensive project synthesis.

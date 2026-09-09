"""Unit and regression tests for the sales forecasting pipeline."""

import json
import unittest
import numpy as np
import pandas as pd
from src.config import DATA_PROCESSED, H, RESULTS, TARGET
from src.data import generate_rolling_folds, load_clean_sales, load_features, split_by_date
from src.features import add_lag_features, leakage_probe_cogs
from src.metrics import bias, diebold_mariano, evaluate_metrics, wape
from src.models import build_model, default_params, get_baseline_models


class TestMetrics(unittest.TestCase):
    def test_wape_perfect(self):
        y = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(wape(y, y), 0.0)

    def test_wape_known(self):
        y = np.array([100.0, 200.0])
        p = np.array([110.0, 180.0])
        # |100-110| + |200-180| = 10 + 20 = 30. sum(y) = 300. wape = 30/300 = 0.1
        self.assertAlmostEqual(wape(y, p), 0.1)

    def test_bias_known(self):
        y = np.array([100.0, 200.0])
        p = np.array([110.0, 220.0])
        # mean(y) = 150, mean(p) = 165 -> (165 - 150)/150 = 0.1
        self.assertAlmostEqual(bias(y, p), 0.1)

    def test_evaluate_metrics_keys(self):
        y = np.array([100.0, 200.0, 300.0])
        p = np.array([105.0, 195.0, 310.0])
        res = evaluate_metrics(y, p)
        self.assertIn("WAPE", res)
        self.assertIn("bias", res)
        self.assertIn("R2", res)

    def test_diebold_mariano(self):
        np.random.seed(42)
        e1 = np.random.normal(0, 1, 100)
        e2 = np.random.normal(0, 2, 100)
        stat, p = diebold_mariano(e1, e2, h=28)
        self.assertIsInstance(stat, float)
        self.assertIsInstance(p, float)
        self.assertTrue(stat < 0)  # e1 has smaller errors than e2


class TestDataAndFolds(unittest.TestCase):
    def test_rolling_folds_count(self):
        folds = generate_rolling_folds("2016-07-01", "2022-05-01")
        self.assertEqual(len(folds), 12)
        # Check buffer is 31 days
        for _, train_end, valid_start, valid_end in folds:
            te = pd.Timestamp(train_end)
            vs = pd.Timestamp(valid_start)
            ve = pd.Timestamp(valid_end)
            self.assertEqual((vs - te).days, 31)
            self.assertTrue(vs < ve)

    def test_clean_sales_loading(self):
        sales = load_clean_sales()
        self.assertGreater(len(sales), 3000)
        self.assertIn("date", sales.columns)
        self.assertIn(TARGET, sales.columns)


class TestFeatureIntegrity(unittest.TestCase):
    def test_features_shape_and_columns(self):
        features_df = load_features()
        meta_path = RESULTS / "02_features.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_cols = meta["final_features"]
            self.assertEqual(len(expected_cols), 62)
            for col in expected_cols:
                self.assertIn(col, features_df.columns)

    def test_no_data_leakage(self):
        raw = load_clean_sales()
        worst, n_checkpoints, n_cols = leakage_probe_cogs(raw, n_checks=4)
        self.assertLess(worst, 1e-9)


class TestModels(unittest.TestCase):
    def test_build_all_baseline_models(self):
        models = get_baseline_models(seed=42)
        expected = ["DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
        for name in expected:
            self.assertIn(name, models)

    def test_default_params(self):
        for name in ["LightGBM", "XGBoost", "RandomForest", "GradientBoosting"]:
            p = default_params(name)
            self.assertIsInstance(p, dict)
            m = build_model(name, p)
            self.assertIsNotNone(m)


if __name__ == "__main__":
    unittest.main()

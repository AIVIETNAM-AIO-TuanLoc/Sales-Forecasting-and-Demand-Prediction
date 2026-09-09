"""Central configuration for paths, parameters, and domain constants."""

from pathlib import Path

# Repository root and key directory paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
REPORTS = ROOT / "reports"

for p in (RESULTS, IMAGES, REPORTS, DATA_PROCESSED):
    p.mkdir(exist_ok=True, parents=True)

# Forecasting task definition
H = 28
TARGET = "revenue"
SEED = 42

# Cross-validation folds (legacy 3-fold split)
FOLDS = [
    ("Fold 1", "2019-05-31", "2019-07-01", "2019-12-31"),
    ("Fold 2", "2020-05-31", "2020-07-01", "2020-12-31"),
    ("Fold 3", "2021-05-31", "2021-07-01", "2021-12-31"),
]
CV_FOLDS = ["Fold 2", "Fold 3"]

# Official single-shot test fold
TEST_FOLD = ("TEST", "2022-05-31", "2022-07-01", "2022-12-31")

# Vietnamese Lunar New Year (Tết Nguyên Đán) reference dates
TET_DATES = [
    "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08",
    "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25", "2021-02-12",
    "2022-02-01", "2023-01-22",
]

# Vietnamese retail recurring special days (month, day)
RETAIL_SPECIAL_DAYS = [
    (1, 1), (3, 8), (4, 30), (5, 1), (9, 2),
    (10, 20), (11, 11), (12, 12), (12, 24), (12, 25),
]

# Baseline feature group classification prefixes
BASE_GROUP_PREFIXES = {
    "lịch sử doanh thu": ("lag", "roll", "ratio", "trend", "cv_"),
    "lịch": (
        "dow", "dom", "month", "quarter", "doy", "days_in",
        "is_week", "is_eom", "is_som", "is_odd", "fourier",
    ),
    "Tết và ngày đặc biệt": ("days_to_tet", "tet_", "is_special"),
    "khuyến mãi": ("promo",),
}

# Core model families
TOP_MODELS = ["XGBoost", "GradientBoosting", "RandomForest", "LightGBM"]

# Baseline document benchmarks
ORIGINAL_LIGHTGBM = {"WAPE": 0.2038, "bias": -0.1181, "R2": 0.6398}
ORIGINAL_BASELINE_WAPE = 0.2941

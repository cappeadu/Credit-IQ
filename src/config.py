from pathlib import Path

categorical_columns = ["edu", "marital_status"]
ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "credit-card-risk"
FEATURE_VERSION = "feature-engineering-v1"

THRESHOLDS = {"lower_threshold": 0.12, "upper_threshold": 0.3}
THRESHOLD_LOWER_CANDIDATES = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
THRESHOLD_UPPER_CANDIDATES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
THRESHOLD_SELECTION_METRIC = "recall"
MAX_FALSE_APPROVAL_RATE = 0.05

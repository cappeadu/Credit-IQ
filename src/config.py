from pathlib import Path

categorical_columns = ["edu", "marital_status"]
ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "credit-card-risk"
FEATURE_VERSION = "feature-engineering-v1"

THRESHOLDS = {"lower_threshold": 0.12, "upper_threshold": 0.3}

from pathlib import Path

categorical_columns = ["edu", "marital_status"]
ROOT = Path(__name__).parent.parent

THRESHOLDS = {"lower_threshold": 0.12, "upper_threshold": 0.3}

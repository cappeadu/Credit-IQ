import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

from src.config import ROOT
from src.data import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_engineered_schema,
)
from src.evaluation import evaluate_binary_predictions
from src.utils import loan_decision

app = typer.Typer()


def load_test_set(test_set_path: str | Path) -> pd.DataFrame:
    """Load and validate the feature-engineered labelled test set."""
    test_file = Path(test_set_path) / "test_only.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"Test set was not found: {test_file}")

    test_data = pd.read_csv(test_file)
    validate_feature_engineered_schema(test_data, require_target=True)
    return test_data


def score_dataframe(
    prediction_model,
    feature_engineered_data: pd.DataFrame,
    *,
    threshold: float = 0.30,
    lower_threshold: float = 0.12,
    upper_threshold: float = 0.30,
) -> pd.DataFrame:
    """Score a validated feature-engineered frame without writing files."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if not 0 <= lower_threshold < upper_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy 0 <= lower_threshold < upper_threshold <= 1."
        )

    validate_feature_engineered_schema(
        feature_engineered_data,
        require_target=TARGET_COLUMN in feature_engineered_data.columns,
    )
    model_features = feature_engineered_data.drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )
    if not np.isfinite(
        model_features[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError("Model input contains non-finite values.")

    scored_data = feature_engineered_data.copy()
    scored_data["probability"] = prediction_model.predict_proba(
        model_features
    )[:, 1]
    scored_data["prediction_raw"] = prediction_model.predict(model_features)
    scored_data["prediction_at_threshold"] = (
        scored_data["probability"] >= threshold
    ).astype(int)
    scored_data["decision"] = scored_data["probability"].apply(
        lambda probability: loan_decision(
            probability,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
    )
    return scored_data


def evaluate_predictions(
    scored_data: pd.DataFrame,
    *,
    prediction_column: str = "prediction_at_threshold",
) -> dict[str, float]:
    """Evaluate scored labelled data using one explicit prediction column."""
    if TARGET_COLUMN not in scored_data:
        raise ValueError("Evaluation requires a target column.")
    if prediction_column not in scored_data:
        raise ValueError(f"Prediction column was not found: {prediction_column}")

    return evaluate_binary_predictions(
        scored_data[TARGET_COLUMN],
        scored_data["probability"],
        scored_data[prediction_column],
    )


@app.command()
def test_predictions(
    test_set_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
    model_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
):
    model_file = Path(model_path) / "calibrated_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Calibrated model was not found: {model_file}")

    test_data = load_test_set(test_set_path)
    calibrated_model = joblib.load(model_file)
    scored_data = score_dataframe(calibrated_model, test_data)

    test_metrics = {
        "default_threshold(0.5)": evaluate_predictions(
            score_dataframe(calibrated_model, test_data, threshold=0.5),
            prediction_column="prediction_at_threshold",
        ),
        "operational_threshold(0.3)": evaluate_predictions(scored_data),
    }

    all_metrics = json.dumps(test_metrics, indent=2)
    all_metrics_path = ROOT / "metrics/test_set.json"
    with all_metrics_path.open("w") as f:
        f.write(all_metrics)

    # save test csv
    test_set_with_prob_path = ROOT / "artifacts/scored_customers.csv"
    scored_data.to_csv(test_set_with_prob_path, index=False)
    return all_metrics


if __name__ == "__main__":
    app()

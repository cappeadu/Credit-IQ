import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from typing_extensions import Annotated

from src.config import ROOT
from src.data import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_engineered_schema,
)
from src.utils import loan_decision

app = typer.Typer()


def load_test_set(test_set_path: str | Path) -> pd.DataFrame:
    """Load and validate the feature-engineered labelled test set."""
    test_file = Path(test_set_path) / "test_only.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"Test set was not found: {test_file}")

    df = pd.read_csv(test_file)
    validate_feature_engineered_schema(df, require_target=True)
    return df


def score_dataframe(
    model,
    df: pd.DataFrame,
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
        df,
        require_target=TARGET_COLUMN in df.columns,
    )
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    if not np.isfinite(X[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Model input contains non-finite values.")

    scored = df.copy()
    scored["probability"] = model.predict_proba(X)[:, 1]
    scored["prediction_raw"] = model.predict(X)
    scored["prediction_at_threshold"] = (
        scored["probability"] >= threshold
    ).astype(int)
    scored["decision"] = scored["probability"].apply(
        lambda probability: loan_decision(
            probability,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
    )
    return scored


def evaluate_predictions(
    scored_df: pd.DataFrame,
    *,
    prediction_column: str = "prediction_at_threshold",
) -> dict[str, float]:
    """Evaluate scored labelled data using one explicit prediction column."""
    if TARGET_COLUMN not in scored_df:
        raise ValueError("Evaluation requires a target column.")
    if prediction_column not in scored_df:
        raise ValueError(f"Prediction column was not found: {prediction_column}")

    target = scored_df[TARGET_COLUMN]
    probabilities = scored_df["probability"]
    predictions = scored_df[prediction_column]
    return {
        "roc_auc_score": round(roc_auc_score(target, probabilities), 4),
        "recall": round(recall_score(target, predictions, zero_division=0), 4),
        "precision": round(
            precision_score(target, predictions, zero_division=0), 4
        ),
        "f1_score": round(f1_score(target, predictions, zero_division=0), 4),
    }


@app.command()
def test_predictions(
    test_set_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
    model_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
):
    model_file = Path(model_path) / "calibrated_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Calibrated model was not found: {model_file}")

    df = load_test_set(test_set_path)
    calibrated_model = joblib.load(model_file)
    scored = score_dataframe(calibrated_model, df)

    test_metrics = {
        "default_threshold(0.5)": evaluate_predictions(
            score_dataframe(calibrated_model, df, threshold=0.5),
            prediction_column="prediction_at_threshold",
        ),
        "operational_threshold(0.3)": evaluate_predictions(scored),
    }

    all_metrics = json.dumps(test_metrics, indent=2)
    all_metrics_path = ROOT / "metrics/test_set.json"
    with all_metrics_path.open("w") as f:
        f.write(all_metrics)

    # save test csv
    test_set_with_prob_path = ROOT / "artifacts/scored_customers.csv"
    scored.to_csv(test_set_with_prob_path, index=False)
    return all_metrics


if __name__ == "__main__":
    app()

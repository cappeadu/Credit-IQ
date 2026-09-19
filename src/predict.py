import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

from src.config import ROOT, THRESHOLDS
from src.data import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_schema,
    validate_feature_engineered_schema,
)
from src.evaluation import (
    analyze_calibration,
    evaluate_binary_predictions,
    evaluate_threshold_policy,
)
from src.utils import loan_decision

app = typer.Typer()


def _resolve_dataset_file(dataset_path: str | Path) -> Path:
    """Resolve a CSV file or a directory containing exactly one CSV file."""
    path = Path(dataset_path)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV dataset: {path}")
        return path
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if len(csv_files) == 1:
            return csv_files[0]
        if not csv_files:
            raise FileNotFoundError(f"No CSV dataset was found in: {path}")
        raise ValueError(
            f"Expected one CSV dataset in {path}, found {len(csv_files)}. "
            "Pass the dataset file directly."
        )
    raise FileNotFoundError(f"Dataset path was not found: {path}")


def load_raw_dataset(
    dataset_path: str | Path,
    *,
    require_target: bool = False,
) -> pd.DataFrame:
    """Load and validate a cleaned, pre-feature-engineered CSV dataset.

    Labelled datasets are required for evaluation; unlabelled datasets can be
    loaded for prediction only.
    """
    dataset_file = _resolve_dataset_file(dataset_path)
    dataset = pd.read_csv(dataset_file)
    validate_schema(
        dataset,
        require_target=require_target or TARGET_COLUMN in dataset.columns,
    )
    return dataset


def transform_for_prediction(
    raw_dataset: pd.DataFrame,
    feature_engineering,
    *,
    require_target: bool = False,
) -> pd.DataFrame:
    """Apply the saved feature-engineering object to raw/cleaned data."""
    validate_schema(
        raw_dataset,
        require_target=require_target or TARGET_COLUMN in raw_dataset.columns,
    )
    transformed_dataset = feature_engineering.transform(raw_dataset)
    validate_feature_engineered_schema(
        transformed_dataset,
        require_target=TARGET_COLUMN in transformed_dataset.columns,
    )
    return transformed_dataset


def load_thresholds(model_path: str | Path) -> dict[str, float]:
    """Load and validate thresholds frozen during validation."""
    thresholds_file = Path(model_path) / "thresholds.json"
    if not thresholds_file.exists():
        raise FileNotFoundError(
            f"Threshold configuration was not found: {thresholds_file}"
        )

    with thresholds_file.open() as file:
        threshold_configuration = json.load(file)

    try:
        lower_threshold = float(threshold_configuration["lower_threshold"])
        upper_threshold = float(threshold_configuration["upper_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Threshold configuration must contain numeric lower_threshold and "
            "upper_threshold values."
        ) from exc

    if not 0 <= lower_threshold < upper_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy 0 <= lower_threshold < upper_threshold <= 1."
        )
    return {
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
    }


def score_dataframe(
    prediction_model,
    feature_engineered_data: pd.DataFrame,
    *,
    threshold: float | None = None,
    lower_threshold: float = THRESHOLDS["lower_threshold"],
    upper_threshold: float = THRESHOLDS["upper_threshold"],
) -> pd.DataFrame:
    """Score a validated feature-engineered frame without writing files."""
    if threshold is None:
        threshold = upper_threshold
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
    scored_data["probability"] = prediction_model.predict_proba(model_features)[:, 1]
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


def evaluate_labelled_dataset(
    scored_data: pd.DataFrame,
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[str, object]:
    """Evaluate any scored labelled dataset with the frozen policy."""
    if TARGET_COLUMN not in scored_data:
        raise ValueError("Dataset evaluation requires a target column.")
    if "probability" not in scored_data:
        raise ValueError("Dataset evaluation requires a probability column.")

    return {
        "thresholds": {
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
        },
        "operational_policy": evaluate_threshold_policy(
            scored_data[TARGET_COLUMN],
            scored_data["probability"],
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        ),
        "calibration": analyze_calibration(
            scored_data[TARGET_COLUMN],
            scored_data["probability"],
        ),
    }


def _load_prediction_model(model_path: str | Path):
    model_file = Path(model_path) / "calibrated_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Calibrated model was not found: {model_file}")
    return joblib.load(model_file)


def _load_feature_engineering(model_path: str | Path):
    feature_engineering_file = Path(model_path) / "feature_engineering.joblib"
    if not feature_engineering_file.exists():
        raise FileNotFoundError(
            f"Feature-engineering artifact was not found: {feature_engineering_file}"
        )
    return joblib.load(feature_engineering_file)


def _write_scored_dataset(
    scored_data: pd.DataFrame,
    *,
    output_path: str | Path | None,
    dataset_name: str,
) -> Path:
    scored_file = (
        Path(output_path)
        if output_path
        else ROOT / "artifacts" / f"scored_{dataset_name}.csv"
    )
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    scored_data.to_csv(scored_file, index=False)
    return scored_file


@app.command("score")
def score_dataset(
    data_path: Annotated[
        str, typer.Option(help="CSV dataset path or directory")
    ] = None,
    model_path: Annotated[str, typer.Option(help="model artifact directory")] = None,
    output_path: Annotated[str, typer.Option(help="output CSV path")] = None,
    dataset_name: Annotated[
        str, typer.Option(help="name used for the default output filename")
    ] = "dataset",
):
    raw_prediction_data = load_raw_dataset(data_path, require_target=False)
    feature_engineering = _load_feature_engineering(model_path)
    prediction_data = transform_for_prediction(
        raw_prediction_data,
        feature_engineering,
        require_target=False,
    )
    calibrated_model = _load_prediction_model(model_path)
    thresholds = load_thresholds(model_path)
    scored_data = score_dataframe(
        calibrated_model,
        prediction_data,
        threshold=thresholds["upper_threshold"],
        lower_threshold=thresholds["lower_threshold"],
        upper_threshold=thresholds["upper_threshold"],
    )
    _write_scored_dataset(
        scored_data,
        output_path=output_path,
        dataset_name=dataset_name,
    )
    return scored_data.to_json(orient="records", indent=2)


@app.command("evaluate")
def evaluate_dataset_command(
    data_path: Annotated[
        str, typer.Option(help="labelled CSV dataset path or directory")
    ] = None,
    model_path: Annotated[str, typer.Option(help="model artifact directory")] = None,
    output_path: Annotated[str, typer.Option(help="output metrics JSON path")] = None,
    dataset_name: Annotated[
        str, typer.Option(help="dataset name used for default output paths")
    ] = "dataset",
):
    raw_evaluation_data = load_raw_dataset(data_path, require_target=True)
    feature_engineering = _load_feature_engineering(model_path)
    evaluation_data = transform_for_prediction(
        raw_evaluation_data,
        feature_engineering,
        require_target=True,
    )
    calibrated_model = _load_prediction_model(model_path)
    thresholds = load_thresholds(model_path)
    scored_data = score_dataframe(
        calibrated_model,
        evaluation_data,
        threshold=thresholds["upper_threshold"],
        lower_threshold=thresholds["lower_threshold"],
        upper_threshold=thresholds["upper_threshold"],
    )

    evaluation_metrics = evaluate_labelled_dataset(
        scored_data,
        lower_threshold=thresholds["lower_threshold"],
        upper_threshold=thresholds["upper_threshold"],
    )
    metrics_file = (
        Path(output_path) if output_path else ROOT / "metrics" / f"{dataset_name}.json"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(evaluation_metrics, indent=2))
    _write_scored_dataset(
        scored_data,
        output_path=None,
        dataset_name=dataset_name,
    )
    return json.dumps(evaluation_metrics, indent=2)


if __name__ == "__main__":
    app()

import json

import joblib
import typer
from sklearn.calibration import CalibratedClassifierCV
from typing_extensions import Annotated

from src.config import ROOT, THRESHOLDS
from src.data import (
    clean_cols,
    load_dataset,
    validate_feature_engineered_schema,
    validate_numeric_values,
    validate_schema,
)
from src.modeling import (
    DEFAULT_SELECTION_METRICS,
    build_candidate_models,
    comparison_table,
    evaluate_candidate_models,
    select_best_model,
)
from src.tracking import (
    configure_mlflow,
    log_baseline_comparison_run,
)
from src.utils import FeatureEngineering, split_dataset

app = typer.Typer()


@app.command()
def train(
    data_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
    path_to_save_val_test: Annotated[
        str, typer.Option(help="path to save val and test sets")
    ] = None,
    path_to_save_test_only: Annotated[
        str, typer.Option(help="path to save test set only")
    ] = None,
):
    configure_mlflow()

    # 1. load dataset
    raw_dataset = load_dataset(data_path=data_path, read_excel=True)

    # 2. # clean columns and properly categorize columns
    cleaned_dataset = clean_cols(raw_dataset)
    validate_schema(cleaned_dataset, require_target=True)
    validate_numeric_values(cleaned_dataset, include_target=True)

    # 3. split and save dataset
    training_data, holdout_data = split_dataset(
        dataset=cleaned_dataset,
        save_dataset=True,
        path_to_save=path_to_save_val_test,
    )
    validation_data, test_data = split_dataset(
        dataset=holdout_data,
        test_size=0.5,
    )

    # feature engineering
    feature_engineering = FeatureEngineering()
    training_features = feature_engineering.fit_transform(training_data)
    validation_features = feature_engineering.transform(validation_data)
    test_features = feature_engineering.transform(test_data)
    for split_name, split_features in {
        "train": training_features,
        "validation": validation_features,
        "test": test_features,
    }.items():
        try:
            validate_feature_engineered_schema(
                split_features,
                require_target=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid feature-engineered {split_name} split: {exc}"
            ) from exc
    test_features.to_csv(
        f"{path_to_save_test_only}/test_only.csv",
        index=False,
    )

    training_features, training_target = (
        training_features.drop("target", axis=1),
        training_features["target"],
    )
    validation_features, validation_target = (
        validation_features.drop("target", axis=1),
        validation_features["target"],
    )

    candidate_models = build_candidate_models(
        training_features,
        training_target,
    )
    fitted_candidate_models, comparison_results = evaluate_candidate_models(
        candidate_models,
        training_features,
        training_target,
        validation_features,
        validation_target,
    )
    print(comparison_table(comparison_results).to_string(index=False))

    # Keep the current selection policy explicit; later evaluation stages can
    # replace this with a documented multi-metric policy.
    best_model_name = select_best_model(comparison_results)
    print(f"Best model: {best_model_name}")
    selected_model = fitted_candidate_models[best_model_name]

    # calibrated model
    calibrated_model = CalibratedClassifierCV(
        estimator=selected_model,
        method="isotonic",
        cv=5,
    )
    calibrated_model.fit(training_features, training_target)
    calibrated_model_path = ROOT / "artifacts/calibrated_model.joblib"
    joblib.dump(calibrated_model, calibrated_model_path)

    # save underlying model for shap analysis
    selected_model_path = ROOT / f"artifacts/{best_model_name}.joblib"
    joblib.dump(selected_model, selected_model_path)

    # save feature engineering object
    feature_engineering_path = ROOT / "artifacts/feature_engineering.joblib"
    joblib.dump(feature_engineering, feature_engineering_path)

    # save thresholds per eda notebook
    thresholds = json.dumps(THRESHOLDS, indent=2)
    thresholds_path = ROOT / "artifacts/thresholds.json"
    with thresholds_path.open("w") as f:
        f.write(thresholds)

    # save metrics
    results_json = json.dumps(comparison_results, indent=2)
    metrics_path = ROOT / "metrics/val_set.json"
    with metrics_path.open("w") as f:
        f.write(results_json)

    mlflow_run_id = log_baseline_comparison_run(
        fitted_candidate_models=fitted_candidate_models,
        comparison_results=comparison_results,
        selected_model_name=best_model_name,
        selection_metrics=DEFAULT_SELECTION_METRICS,
        training_parameters={
            "data_path": data_path,
            "random_state": 42,
            "validation_split": 0.1,
            "test_split": 0.1,
            "calibration_method": "isotonic",
            "calibration_cv": 5,
        },
        artifact_paths={
            "metrics": metrics_path,
            "model": calibrated_model_path,
            "selected_estimator": selected_model_path,
            "preprocessing": feature_engineering_path,
            "configuration": thresholds_path,
        },
    )
    print(f"MLflow run ID: {mlflow_run_id}")

    return results_json


if __name__ == "__main__":
    app()

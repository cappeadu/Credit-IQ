import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import joblib
import typer
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    FEATURE_VERSION,
    MAX_FALSE_APPROVAL_RATE,
    ROOT,
    THRESHOLD_LOWER_CANDIDATES,
    THRESHOLD_SELECTION_METRIC,
    THRESHOLD_UPPER_CANDIDATES,
)
from src.data import (
    MODEL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    clean_cols,
    load_dataset,
    validate_feature_engineered_schema,
    validate_numeric_values,
    validate_schema,
)
from src.evaluation import (
    analyze_calibration,
    evaluate_threshold_policy,
    select_thresholds,
)
from src.model_package import create_model_package
from src.modeling import (
    DEFAULT_SELECTION_METRICS,
    build_candidate_models,
    comparison_table,
    evaluate_candidate_models,
    select_best_model,
)
from src.splitting import create_or_load_splits
from src.tracking import (
    configure_mlflow,
    fingerprint_dataframe,
    get_repository_revision,
    log_baseline_comparison_run,
)
from src.utils import FeatureEngineering

app = typer.Typer()


@app.command()
def train(
    data_path: Annotated[str | None, typer.Option(help="dataset path or link")] = None,
    path_to_save_val_test: Annotated[
        str | None, typer.Option(help="path to save val and test sets")
    ] = None,
    path_to_save_test_only: Annotated[
        str | None, typer.Option(help="path to save test set only")
    ] = None,
    split_manifest_path: Annotated[
        str | None, typer.Option(help="path to the persisted split manifest")
    ] = None,
):
    configure_mlflow()

    # 1. load dataset
    raw_dataset = load_dataset(data_path=data_path, read_excel=True)

    # 2. # clean columns and properly categorize columns
    cleaned_dataset = clean_cols(raw_dataset)
    validate_schema(cleaned_dataset, require_target=True)
    validate_numeric_values(cleaned_dataset, include_target=True)
    dataset_fingerprint = fingerprint_dataframe(cleaned_dataset)

    # 3. split and save dataset
    split_output_dir = (
        ROOT / "data/val_test"
        if path_to_save_val_test is None
        else Path(path_to_save_val_test)
    )
    manifest_file = (
        split_output_dir / "split_manifest.json"
        if split_manifest_path is None
        else Path(split_manifest_path)
    )
    training_data, validation_data, test_data, split_manifest = create_or_load_splits(
        dataset=cleaned_dataset,
        manifest_path=manifest_file,
        dataset_fingerprint=dataset_fingerprint,
        random_state=42,
    )
    split_output_dir.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(split_output_dir / "train_set.csv", index=False)
    validation_data.to_csv(split_output_dir / "validation_set.csv", index=False)

    # Persist the raw/cleaned validation and test splits. Feature engineering
    # is applied later by the same artifact used during prediction.
    validation_output_dir = split_output_dir
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    validation_data.to_csv(validation_output_dir / "validation_set.csv", index=False)
    test_output_dir = (
        ROOT / "data/test_only"
        if path_to_save_test_only is None
        else Path(path_to_save_test_only)
    )
    test_output_dir.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(test_output_dir / "test_only.csv", index=False)

    # feature engineering
    feature_engineering = FeatureEngineering()
    training_features = feature_engineering.fit_transform(training_data)
    validation_features = feature_engineering.transform(validation_data)
    for split_name, split_features in {
        "train": training_features,
        "validation": validation_features,
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
    staging_directory = TemporaryDirectory(prefix="credit_card_model_")
    staging_path = Path(staging_directory.name)
    calibrated_model = CalibratedClassifierCV(
        estimator=selected_model,
        method="isotonic",
        cv=5,
    )
    calibrated_model.fit(training_features, training_target)
    validation_probabilities = calibrated_model.predict_proba(validation_features)[:, 1]
    threshold_selection = select_thresholds(
        validation_target,
        validation_probabilities,
        lower_thresholds=THRESHOLD_LOWER_CANDIDATES,
        upper_thresholds=THRESHOLD_UPPER_CANDIDATES,
        selection_metric=THRESHOLD_SELECTION_METRIC,
        maximum_false_approval_rate=MAX_FALSE_APPROVAL_RATE,
    )
    selected_thresholds = threshold_selection["selected_thresholds"]
    calibrated_model_path = staging_path / "calibrated_model.joblib"
    joblib.dump(calibrated_model, calibrated_model_path)

    # save underlying model for shap analysis
    selected_model_path = staging_path / "selected_estimator.joblib"
    joblib.dump(selected_model, selected_model_path)

    # save feature engineering object
    feature_engineering_path = staging_path / "feature_engineering.joblib"
    joblib.dump(feature_engineering, feature_engineering_path)

    feature_schema = {
        "feature_version": FEATURE_VERSION,
        "raw_features": RAW_FEATURE_COLUMNS,
        "model_features": MODEL_FEATURE_COLUMNS,
        "target": "target",
    }
    feature_schema_path = staging_path / "feature_schema.json"
    feature_schema_path.write_text(json.dumps(feature_schema, indent=2))

    # Save thresholds selected from calibrated validation probabilities.
    thresholds = json.dumps(
        {
            **selected_thresholds,
            "selection_metric": threshold_selection["selection_metric"],
            "maximum_false_approval_rate": threshold_selection[
                "maximum_false_approval_rate"
            ],
        },
        indent=2,
    )
    thresholds_path = staging_path / "thresholds.json"
    with thresholds_path.open("w") as f:
        f.write(thresholds)

    # save metrics
    calibrated_validation_evaluation = {
        "thresholds": selected_thresholds,
        "operational_policy": evaluate_threshold_policy(
            validation_target,
            validation_probabilities,
            lower_threshold=selected_thresholds["lower_threshold"],
            upper_threshold=selected_thresholds["upper_threshold"],
        ),
        "calibration": analyze_calibration(
            validation_target,
            validation_probabilities,
        ),
    }
    validation_report = {
        "model_comparison": comparison_results,
        "threshold_selection": threshold_selection,
        "calibrated_validation_evaluation": calibrated_validation_evaluation,
        "split_manifest": split_manifest,
    }
    results_json = json.dumps(validation_report, indent=2)
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
        reproducibility_metadata={
            "dataset_fingerprint": dataset_fingerprint,
            "feature_version": FEATURE_VERSION,
            "repository_revision": get_repository_revision(ROOT),
            "cleaned_row_count": len(cleaned_dataset),
            "training_row_count": len(training_data),
            "validation_row_count": len(validation_data),
            "test_row_count": len(test_data),
            "split_manifest_path": manifest_file,
        },
        feature_schema=feature_schema,
        validation_report=validation_report,
        artifact_paths={
            "metrics": metrics_path,
            "model": calibrated_model_path,
            "selected_estimator": selected_model_path,
            "preprocessing": feature_engineering_path,
            "configuration": thresholds_path,
            "schema": feature_schema_path,
        },
    )
    print(f"MLflow run ID: {mlflow_run_id}")

    model_package_path = create_model_package(
        package_directory=ROOT / "artifacts" / f"model_package_{mlflow_run_id}",
        calibrated_model_path=calibrated_model_path,
        selected_estimator_path=selected_model_path,
        feature_engineering_path=feature_engineering_path,
        thresholds_path=thresholds_path,
        feature_schema_path=feature_schema_path,
        split_manifest_path=manifest_file,
        metadata={
            "mlflow_run_id": mlflow_run_id,
            "selected_model_name": best_model_name,
            "feature_version": FEATURE_VERSION,
            "dataset_fingerprint": dataset_fingerprint,
            "split_manifest_path": str(manifest_file),
        },
    )
    print(f"Model package: {model_package_path}")
    staging_directory.cleanup()

    return results_json


if __name__ == "__main__":
    app()

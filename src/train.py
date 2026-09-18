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
    build_candidate_models,
    evaluate_candidate_models,
    select_best_model,
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
    # 1. load dataset
    df = load_dataset(data_path=data_path, read_excel=True)

    # 2. # clean columns and properly categorize columns
    df = clean_cols(df)
    validate_schema(df, require_target=True)
    validate_numeric_values(df, include_target=True)

    # 3. split and save dataset
    train_df, test_df = split_dataset(
        df=df, save_dataset=True, path_to_save=path_to_save_val_test
    )
    val_df, test_df = split_dataset(df=test_df, test_size=0.5)

    # feature engineering
    feature_engineering = FeatureEngineering()
    train_df_fe = feature_engineering.fit_transform(train_df)
    val_df_fe = feature_engineering.transform(val_df)
    test_df_fe = feature_engineering.transform(test_df)
    for split_name, split_df in {
        "train": train_df_fe,
        "validation": val_df_fe,
        "test": test_df_fe,
    }.items():
        try:
            validate_feature_engineered_schema(split_df, require_target=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid feature-engineered {split_name} split: {exc}"
            ) from exc
    test_df_fe.to_csv(f"{path_to_save_test_only}/test_only.csv", index=False)

    X_train_fe, y_train_fe = train_df_fe.drop("target", axis=1), train_df_fe["target"]
    X_val_fe, y_val_fe = val_df_fe.drop("target", axis=1), val_df_fe["target"]
    # X_test_fe, y_test_fe = test_df_fe.drop("target", axis=1), test_df_fe["target"]

    models = build_candidate_models(X_train_fe, y_train_fe)
    models_feature_eng, results = evaluate_candidate_models(
        models,
        X_train_fe,
        y_train_fe,
        X_val_fe,
        y_val_fe,
    )

    # Keep the current selection policy explicit; later evaluation stages can
    # replace this with a documented multi-metric policy.
    best_model_name = select_best_model(results, metric="recall")
    print(f"Best model: {best_model_name}")
    best_model = models_feature_eng[best_model_name]

    # calibrated model
    final_model = CalibratedClassifierCV(estimator=best_model, method="isotonic", cv=5)
    final_model.fit(X_train_fe, y_train_fe)
    file_name = ROOT / "artifacts/calibrated_model.joblib"
    joblib.dump(final_model, file_name)

    # save underlying model for shap analysis
    best_model_file_name = ROOT / f"artifacts/{best_model_name}.joblib"
    joblib.dump(best_model, best_model_file_name)

    # save feature engineering object
    feat_eng_path = ROOT / "artifacts/feature_engineering.joblib"
    joblib.dump(feature_engineering, feat_eng_path)

    # save thresholds per eda notebook
    thresholds = json.dumps(THRESHOLDS, indent=2)
    thresholds_path = ROOT / "artifacts/thresholds.json"
    with thresholds_path.open("w") as f:
        f.write(thresholds)

    # save metrics
    results_json = json.dumps(results, indent=2)
    metrics_path = ROOT / "metrics/val_set.json"
    with metrics_path.open("w") as f:
        f.write(results_json)

    return results_json


if __name__ == "__main__":
    app()

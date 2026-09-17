import json

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


@app.command()
def test_predictions(
    test_set_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
    model_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
):
    df = pd.read_csv(f"{test_set_path}/test_only.csv")
    validate_feature_engineered_schema(df, require_target=True)
    if not np.isfinite(df[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Test data contains non-finite model features.")

    X_test_fe, y_test_fe = df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]

    calibrated_model = joblib.load(f"{model_path}/calibrated_model.joblib")

    cal_test_probs = calibrated_model.predict_proba(X_test_fe)[:, 1]
    df["probability"] = cal_test_probs
    df["prediction_raw"] = calibrated_model.predict(X_test_fe)
    df["prediction_thres_0_3"] = (df["probability"] >= 0.3).astype(int)
    df["decision"] = df["probability"].apply(loan_decision)

    test_metrics = {}
    test_metrics["default_thres(0.5)"] = {
        "roc_auc_score": round(roc_auc_score(y_test_fe, cal_test_probs), 4),
        "recall": round(recall_score(y_test_fe, df["prediction_raw"]), 4),
        "precision": round(precision_score(y_test_fe, df["prediction_raw"]), 4),
        "f1_score": round(f1_score(y_test_fe, df["prediction_raw"]), 4),
    }

    test_metrics["adjusted_thres(0.3)"] = {
        "roc_auc_score": round(roc_auc_score(y_test_fe, cal_test_probs), 4),
        "recall": round(recall_score(y_test_fe, df["prediction_thres_0_3"]), 4),
        "precision": round(precision_score(y_test_fe, df["prediction_thres_0_3"]), 4),
        "f1_score": round(f1_score(y_test_fe, df["prediction_thres_0_3"]), 4),
    }

    all_metrics = json.dumps(test_metrics, indent=2)
    all_metrics_path = ROOT / "metrics/test_set.json"
    with all_metrics_path.open("w") as f:
        f.write(all_metrics)

    # save test csv
    test_set_with_prob_path = ROOT / "artifacts/scored_customers.csv"
    df.to_csv(test_set_with_prob_path, index=False)
    return all_metrics


if __name__ == "__main__":
    app()

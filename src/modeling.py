"""Reusable model construction and validation helpers.

This module intentionally does not perform hyperparameter tuning. Candidate
parameters can be supplied through model_params so a future Optuna study can
provide tuned values without changing the training orchestration.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import cat_cols

DEFAULT_RANDOM_STATE = 42
MODEL_NAMES = ("Logistic Regression", "Random Forest", "XGBoost")


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing used by the linear candidate model."""
    categorical_columns = [*cat_cols, "gender"]
    numeric_columns = [
        column for column in X_train.columns if column not in categorical_columns
    ]

    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                cat_cols,
            ),
            (
                "gender",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                ["gender"],
            ),
        ]
    )


def build_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Build baseline candidate estimators.

    model_params is keyed by the public model names. It is optional so
    baseline training remains simple while future tuning tools can inject
    trial parameters here.
    """
    model_params = model_params or {}
    default_params = {
        "Logistic Regression": {
            "random_state": random_state,
            "class_weight": "balanced",
            "max_iter": 1000,
        },
        "Random Forest": {
            "n_estimators": 50,
            "max_depth": 6,
            "random_state": random_state,
            "class_weight": "balanced",
        },
        "XGBoost": {
            "random_state": random_state,
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "scale_pos_weight": _positive_class_weight(y_train),
        },
    }

    resolved_params = {
        name: {**default_params[name], **dict(model_params.get(name, {}))}
        for name in MODEL_NAMES
    }

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    LogisticRegression(**resolved_params["Logistic Regression"]),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(**resolved_params["Random Forest"]),
        "XGBoost": XGBClassifier(**resolved_params["XGBoost"]),
    }


def _positive_class_weight(y_train: pd.Series) -> float:
    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))
    if positive_count == 0:
        raise ValueError("Training data must contain at least one positive target.")
    return negative_count / positive_count


def evaluate_candidate_models(
    models: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit candidates and return fitted models plus comparable metrics."""
    fitted_models = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        fitted_models[name] = model
        results[name] = {
            "roc_auc_score": round(roc_auc_score(y_validation, probabilities), 4),
            "recall": round(
                recall_score(y_validation, predictions, zero_division=0), 4
            ),
            "precision": round(
                precision_score(y_validation, predictions, zero_division=0), 4
            ),
            "f1_score": round(f1_score(y_validation, predictions, zero_division=0), 4),
            "model_name": name,
        }

    return fitted_models, results


def select_best_model(
    results: Mapping[str, Mapping[str, Any]],
    *,
    metric: str = "recall",
) -> str:
    """Select the highest-scoring candidate using an explicit metric."""
    if not results:
        raise ValueError("Cannot select a model from empty results.")
    if any(metric not in result for result in results.values()):
        raise ValueError(f"Selection metric is missing from model results: {metric}")

    return max(results, key=lambda name: (results[name][metric], name))

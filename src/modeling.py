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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import categorical_columns

DEFAULT_RANDOM_STATE = 42
MODEL_NAMES = ("Logistic Regression", "Random Forest", "XGBoost")


def build_preprocessor(training_features: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing used by the linear candidate model."""
    model_categorical_columns = [*categorical_columns, "gender"]
    numeric_columns = [
        column
        for column in training_features.columns
        if column not in model_categorical_columns
    ]

    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "gender",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                ["gender"],
            ),
        ]
    )


def build_candidate_models(
    training_features: pd.DataFrame,
    training_target: pd.Series,
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
            "scale_pos_weight": _positive_class_weight(training_target),
        },
    }

    resolved_params = {
        model_name: {
            **default_params[model_name],
            **dict(model_params.get(model_name, {})),
        }
        for model_name in MODEL_NAMES
    }

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(training_features)),
                (
                    "model",
                    LogisticRegression(**resolved_params["Logistic Regression"]),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(**resolved_params["Random Forest"]),
        "XGBoost": XGBClassifier(**resolved_params["XGBoost"]),
    }


def _positive_class_weight(training_target: pd.Series) -> float:
    positive_count = int(np.sum(training_target == 1))
    negative_count = int(np.sum(training_target == 0))
    if positive_count == 0:
        raise ValueError("Training data must contain at least one positive target.")
    return negative_count / positive_count


def evaluate_candidate_models(
    candidate_models: Mapping[str, Any],
    training_features: pd.DataFrame,
    training_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit candidates and return fitted models plus comparable metrics."""
    fitted_candidate_models = {}
    comparison_results = {}

    for model_name, candidate_model in candidate_models.items():
        candidate_model.fit(training_features, training_target)
        validation_predictions = candidate_model.predict(validation_features)
        validation_probabilities = candidate_model.predict_proba(validation_features)[
            :, 1
        ]
        true_negatives, false_positives, _, _ = confusion_matrix(
            validation_target,
            validation_predictions,
            labels=[0, 1],
        ).ravel()
        specificity_denominator = true_negatives + false_positives
        specificity = (
            true_negatives / specificity_denominator if specificity_denominator else 0.0
        )

        fitted_candidate_models[model_name] = candidate_model
        comparison_results[model_name] = {
            "roc_auc_score": round(
                roc_auc_score(validation_target, validation_probabilities),
                4,
            ),
            "pr_auc_score": round(
                average_precision_score(
                    validation_target,
                    validation_probabilities,
                ),
                4,
            ),
            "recall": round(
                recall_score(
                    validation_target,
                    validation_predictions,
                    zero_division=0,
                ),
                4,
            ),
            "precision": round(
                precision_score(
                    validation_target,
                    validation_predictions,
                    zero_division=0,
                ),
                4,
            ),
            "f1_score": round(
                f1_score(
                    validation_target,
                    validation_predictions,
                    zero_division=0,
                ),
                4,
            ),
            "specificity": round(specificity, 4),
            "model_name": model_name,
        }

    return fitted_candidate_models, comparison_results


def comparison_table(
    comparison_results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Return baseline results as a readable, consistently ordered table."""
    if not comparison_results:
        raise ValueError("Cannot create a comparison table from empty results.")

    table = pd.DataFrame.from_dict(comparison_results, orient="index")
    metric_columns = [
        "model_name",
        "roc_auc_score",
        "pr_auc_score",
        "recall",
        "precision",
        "f1_score",
        "specificity",
    ]
    return (
        table[metric_columns]
        .sort_values(
            by=["recall", "pr_auc_score"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def select_best_model(
    comparison_results: Mapping[str, Mapping[str, Any]],
    *,
    metric: str = "recall",
) -> str:
    """Select the highest-scoring candidate using an explicit metric."""
    if not comparison_results:
        raise ValueError("Cannot select a model from empty results.")
    if any(metric not in result for result in comparison_results.values()):
        raise ValueError(f"Selection metric is missing from model results: {metric}")

    return max(
        comparison_results,
        key=lambda model_name: (
            comparison_results[model_name][metric],
            model_name,
        ),
    )

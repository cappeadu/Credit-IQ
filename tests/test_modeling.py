import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.modeling import (
    build_candidate_models,
    comparison_table,
    evaluate_candidate_models,
    select_best_model,
)
from src.predict import score_dataframe
from src.utils import FeatureEngineering
from tests.test_data import make_customer_frame


def make_training_frame(row_count=8):
    rows = []
    for index in range(row_count):
        row = make_customer_frame().iloc[0].copy()
        row["target"] = index % 2
        row["limit"] = 100000 + index
        rows.append(row)
    return pd.DataFrame(rows)


class ModelingTests(unittest.TestCase):
    def test_candidate_registry_supports_model_specific_parameters(self):
        training_data = make_training_frame()
        training_features = training_data.drop(columns=["target"])
        training_target = training_data["target"]

        models = build_candidate_models(
            training_features,
            training_target,
            model_params={"Random Forest": {"n_estimators": 7}},
        )

        self.assertEqual(
            set(models),
            {"Logistic Regression", "Random Forest", "XGBoost"},
        )
        self.assertEqual(models["Random Forest"].n_estimators, 7)

    def test_candidate_evaluation_returns_fitted_models_and_metrics(self):
        training_data = make_training_frame()
        training_features = training_data.drop(columns=["target"])
        training_target = training_data["target"]
        candidate_models = {
            "first": DummyClassifier(strategy="prior"),
            "second": DummyClassifier(strategy="uniform", random_state=42),
        }

        fitted_models, comparison_results = evaluate_candidate_models(
            candidate_models,
            training_features,
            training_target,
            training_features,
            training_target,
        )

        self.assertEqual(set(fitted_models), set(candidate_models))
        self.assertEqual(set(comparison_results), set(candidate_models))
        self.assertIn("pr_auc_score", comparison_results["first"])
        self.assertIn("specificity", comparison_results["first"])
        self.assertEqual(
            list(comparison_table(comparison_results)["model_name"]),
            ["second", "first"],
        )
        selected_model_name = select_best_model(
            {
                "model_a": {
                    "pr_auc_score": 0.40,
                    "recall": 0.90,
                    "f1_score": 0.60,
                    "specificity": 0.80,
                    "roc_auc_score": 0.70,
                },
                "model_b": {
                    "pr_auc_score": 0.45,
                    "recall": 0.50,
                    "f1_score": 0.55,
                    "specificity": 0.85,
                    "roc_auc_score": 0.75,
                },
            }
        )
        self.assertEqual(selected_model_name, "model_b")

    def test_score_dataframe_returns_probability_and_decision(self):
        training_data = make_training_frame(row_count=2)
        feature_engineering = FeatureEngineering()
        feature_engineered_data = feature_engineering.fit_transform(training_data)

        class FixedModel:
            def predict_proba(self, X):
                return np.tile([[0.2, 0.8]], (len(X), 1))

            def predict(self, X):
                return np.ones(len(X), dtype=int)

        scored_data = score_dataframe(FixedModel(), feature_engineered_data)

        self.assertTrue(np.allclose(scored_data["probability"], 0.8))
        self.assertTrue((scored_data["prediction_at_threshold"] == 1).all())
        self.assertTrue((scored_data["decision"] == "REJECT").all())


if __name__ == "__main__":
    unittest.main()

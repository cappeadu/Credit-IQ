import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.modeling import (
    build_candidate_models,
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
        frame = make_training_frame()
        X = frame.drop(columns=["target"])
        y = frame["target"]

        models = build_candidate_models(
            X,
            y,
            model_params={"Random Forest": {"n_estimators": 7}},
        )

        self.assertEqual(
            set(models),
            {"Logistic Regression", "Random Forest", "XGBoost"},
        )
        self.assertEqual(models["Random Forest"].n_estimators, 7)

    def test_candidate_evaluation_returns_fitted_models_and_metrics(self):
        frame = make_training_frame()
        X = frame.drop(columns=["target"])
        y = frame["target"]
        models = {
            "first": DummyClassifier(strategy="prior"),
            "second": DummyClassifier(strategy="uniform", random_state=42),
        }

        fitted, results = evaluate_candidate_models(models, X, y, X, y)

        self.assertEqual(set(fitted), set(models))
        self.assertEqual(set(results), set(models))
        self.assertIn("roc_auc_score", results["first"])
        self.assertEqual(select_best_model(results, metric="recall"), "second")

    def test_score_dataframe_returns_probability_and_decision(self):
        frame = make_training_frame(row_count=2)
        feature_engineering = FeatureEngineering()
        feature_engineered = feature_engineering.fit_transform(frame)

        class FixedModel:
            def predict_proba(self, X):
                return np.tile([[0.2, 0.8]], (len(X), 1))

            def predict(self, X):
                return np.ones(len(X), dtype=int)

        scored = score_dataframe(FixedModel(), feature_engineered)

        self.assertTrue(np.allclose(scored["probability"], 0.8))
        self.assertTrue((scored["prediction_at_threshold"] == 1).all())
        self.assertTrue((scored["decision"] == "REJECT").all())


if __name__ == "__main__":
    unittest.main()

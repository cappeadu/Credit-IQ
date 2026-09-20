import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app, resolve_model_package_path
from src.utils import FeatureEngineering
from tests.test_api_schemas import make_customer_payload


class FakePredictionModel:
    def predict_proba(self, features):
        probabilities = [0.2 + 0.6 * (index % 2) for index in range(len(features))]
        return np.array(
            [[1 - probability, probability] for probability in probabilities]
        )

    def predict(self, features):
        return [int(probability >= 0.5) for probability in self._probabilities(features)]

    def _probabilities(self, features):
        return [0.2 + 0.6 * (index % 2) for index in range(len(features))]


class ApiStartupTests(unittest.TestCase):
    def _prediction_package(self):
        return {
            "metadata": {
                "package_version": 1,
                "mlflow_run_id": "run-123",
                "selected_model_name": "Logistic Regression",
                "feature_version": "v1",
            },
            "thresholds": {"lower_threshold": 0.3, "upper_threshold": 0.7},
            "feature_engineering": FeatureEngineering(),
            "calibrated_model": FakePredictionModel(),
            "selected_estimator": FakePredictionModel(),
        }

    def test_explicit_relative_package_path_is_resolved_from_repository_root(self):
        resolved_path = resolve_model_package_path("artifacts")

        self.assertEqual(
            resolved_path,
            Path(__file__).resolve().parent.parent / "artifacts",
        )

    def test_api_loads_the_validated_package_during_startup(self):
        fake_package = {
            "metadata": {
                "package_version": 1,
                "mlflow_run_id": "run-123",
                "selected_model_name": "Logistic Regression",
                "feature_version": "v1",
            },
            "thresholds": {"lower_threshold": 0.3, "upper_threshold": 0.7},
        }
        application = create_app("artifacts")

        with patch("api.main.load_model_package", return_value=fake_package):
            with TestClient(application):
                self.assertEqual(application.state.model_package, fake_package)

    def test_health_reports_loaded_model(self):
        fake_package = {"metadata": {"mlflow_run_id": "run-123"}}
        application = create_app("artifacts")

        with patch("api.main.load_model_package", return_value=fake_package):
            with TestClient(application) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_loaded": True})

    def test_model_info_returns_loaded_package_metadata_and_thresholds(self):
        fake_package = {
            "metadata": {
                "package_version": 1,
                "mlflow_run_id": "run-123",
                "selected_model_name": "Logistic Regression",
                "feature_version": "v1",
            },
            "thresholds": {"lower_threshold": 0.3, "upper_threshold": 0.7},
        }
        application = create_app("artifacts")

        with patch("api.main.load_model_package", return_value=fake_package):
            with TestClient(application) as client:
                response = client.get("/model-info")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "package_version": 1,
                "mlflow_run_id": "run-123",
                "selected_model_name": "Logistic Regression",
                "feature_version": "v1",
                "lower_threshold": 0.3,
                "upper_threshold": 0.7,
            },
        )

    def test_model_info_returns_service_unavailable_when_package_is_not_loaded(self):
        application = create_app("artifacts")

        with patch("api.main.load_model_package", return_value={}):
            with TestClient(application) as client:
                application.state.model_package = None
                response = client.get("/model-info")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "The model package is not loaded.")

    def test_prediction_returns_unprocessable_entity_when_scoring_fails(self):
        application = create_app("artifacts")

        with patch(
            "api.main.load_model_package",
            return_value={"thresholds": {"lower_threshold": 0.3, "upper_threshold": 0.7}},
        ):
            with TestClient(application) as client:
                response = client.post("/predict", json=make_customer_payload())

        self.assertEqual(response.status_code, 422)
        self.assertIn("Customer could not be scored", response.json()["detail"])

    def test_single_prediction_transforms_and_scores_customer(self):
        application = create_app("artifacts")

        with patch(
            "api.main.load_model_package",
            return_value=self._prediction_package(),
        ), patch(
            "api.main.build_shap_explanations",
            return_value=[
                {
                    "base_value": 0.1,
                    "output_space": "model_output",
                    "contributions": [],
                }
            ],
        ):
            with TestClient(application) as client:
                response = client.post("/predict", json=make_customer_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["probability"], 0.2)
        self.assertEqual(response.json()["decision"], "APPROVE")
        self.assertIn("explanation", response.json())

    def test_batch_prediction_returns_one_result_per_customer(self):
        application = create_app("artifacts")
        request_body = {
            "customers": [make_customer_payload(), make_customer_payload()]
        }

        with patch(
            "api.main.load_model_package",
            return_value=self._prediction_package(),
        ), patch(
            "api.main.build_shap_explanations",
            return_value=[
                {
                    "base_value": 0.1,
                    "output_space": "model_output",
                    "contributions": [],
                },
                {
                    "base_value": 0.1,
                    "output_space": "model_output",
                    "contributions": [],
                },
            ],
        ):
            with TestClient(application) as client:
                response = client.post("/predict/batch", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "predictions": [
                    {
                        "probability": 0.2,
                        "decision": "APPROVE",
                        "explanation": {
                            "base_value": 0.1,
                            "output_space": "model_output",
                            "contributions": [],
                        },
                    },
                    {
                        "probability": 0.8,
                        "decision": "REJECT",
                        "explanation": {
                            "base_value": 0.1,
                            "output_space": "model_output",
                            "contributions": [],
                        },
                    },
                ]
            },
        )

    def test_api_startup_fails_without_a_package_path(self):
        application = create_app()

        with self.assertRaises(RuntimeError):
            with TestClient(application):
                pass


if __name__ == "__main__":
    unittest.main()

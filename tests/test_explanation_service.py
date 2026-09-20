import unittest

from api.schemas import ExplanationRequest
from src.explanation_service import generate_deterministic_explanation


def make_explanation_request():
    return ExplanationRequest.model_validate(
        {
            "prediction": {
                "probability": 0.72,
                "decision": "REJECT",
                "explanation": {
                    "base_value": -1.4,
                    "output_space": "model_output",
                    "contributions": [
                        {
                            "feature_name": "rep_status_mth_6",
                            "feature_value": 2,
                            "shap_value": 0.8,
                            "direction": "increases_risk",
                        },
                        {
                            "feature_name": "pay_streak",
                            "feature_value": 4,
                            "shap_value": -0.7,
                            "direction": "decreases_risk",
                        },
                        {
                            "feature_name": "small_effect",
                            "feature_value": 1,
                            "shap_value": 0.005,
                            "direction": "increases_risk",
                        },
                    ],
                },
            },
            "model_info": {
                "package_version": 1,
                "mlflow_run_id": "run-123",
                "selected_model_name": "XGBoost",
                "feature_version": "feature-engineering-v1",
                "lower_threshold": 0.3,
                "upper_threshold": 0.7,
            },
        }
    )


class ExplanationServiceTests(unittest.TestCase):
    def test_service_returns_ranked_deterministic_response(self):
        response = generate_deterministic_explanation(make_explanation_request())

        self.assertIn("XGBoost model assigned REJECT", response.summary)
        self.assertEqual(len(response.increasing_contributors), 1)
        self.assertIn("rep_status_mth_6", response.increasing_contributors[0])
        self.assertIn("pay_streak", response.decreasing_contributors[0])
        self.assertTrue(all("small_effect" not in item for item in response.increasing_contributors))
        self.assertIn("do not prove causation", response.limitations[0])

    def test_service_requires_shap_context(self):
        request = make_explanation_request()
        request.prediction.explanation = None

        with self.assertRaisesRegex(ValueError, "explanation context is required"):
            generate_deterministic_explanation(request)


if __name__ == "__main__":
    unittest.main()

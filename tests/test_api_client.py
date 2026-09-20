import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.api_client import CreditRiskApiClient, CreditRiskApiError
from tests.test_api_schemas import make_customer_payload


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ApiClientTests(unittest.TestCase):
    def test_explain_prediction_sends_prediction_and_model_context(self):
        client = CreditRiskApiClient("http://example.test")
        prediction = {
            "probability": 0.72,
            "decision": "REJECT",
            "explanation": {
                "base_value": 0.1,
                "output_space": "model_output",
                "contributions": [],
            },
        }
        model_info = {
            "package_version": 1,
            "selected_model_name": "XGBoost",
        }

        with patch(
            "src.api_client.urlopen",
            return_value=FakeHttpResponse(
                {"summary": "Deterministic explanation"}
            ),
        ) as urlopen:
            response = client.explain_prediction(prediction, model_info)

        request = urlopen.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(response["summary"], "Deterministic explanation")
        self.assertEqual(request.full_url, "http://example.test/explain")
        self.assertEqual(request_body["prediction"], prediction)
        self.assertEqual(request_body["model_info"], model_info)

    def test_model_info_is_requested_from_configured_api(self):
        client = CreditRiskApiClient("http://example.test")

        with patch(
            "src.api_client.urlopen",
            return_value=FakeHttpResponse({"selected_model_name": "XGBoost"}),
        ) as urlopen:
            response = client.model_info()

        self.assertEqual(response["selected_model_name"], "XGBoost")
        self.assertEqual(urlopen.call_args.args[0].full_url, "http://example.test/model-info")

    def test_dataframe_prediction_sends_only_raw_feature_columns(self):
        client = CreditRiskApiClient("http://example.test")
        raw_data = pd.DataFrame([make_customer_payload()])
        raw_data["target"] = 0
        raw_data["unexpected_column"] = "ignored"

        with patch(
            "src.api_client.urlopen",
            return_value=FakeHttpResponse(
                {"predictions": [{"probability": 0.2, "decision": "APPROVE"}]}
            ),
        ) as urlopen:
            predictions = client.predict_dataframe(raw_data)

        request_body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(predictions[0]["decision"], "APPROVE")
        self.assertNotIn("target", request_body["customers"][0])
        self.assertNotIn("unexpected_column", request_body["customers"][0])

    def test_connection_failure_is_wrapped_in_api_error(self):
        client = CreditRiskApiClient("http://example.test")

        with patch("src.api_client.urlopen", side_effect=OSError("offline")):
            with self.assertRaises(CreditRiskApiError):
                client.health()

    def test_large_dataframe_is_sent_in_api_sized_batches(self):
        client = CreditRiskApiClient("http://example.test")
        raw_data = pd.DataFrame([make_customer_payload()] * 1001)

        responses = [
            FakeHttpResponse(
                {
                    "predictions": [
                        {"probability": 0.2, "decision": "APPROVE"}
                    ]
                    * 1000
                }
            ),
            FakeHttpResponse(
                {"predictions": [{"probability": 0.8, "decision": "REJECT"}]}
            ),
        ]
        with patch("src.api_client.urlopen", side_effect=responses) as urlopen:
            predictions = client.predict_dataframe(raw_data)

        self.assertEqual(len(predictions), 1001)
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()

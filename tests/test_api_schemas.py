import unittest

from pydantic import ValidationError

from api.schemas import (
    BatchPredictionRequest,
    CustomerRecord,
    DeterministicExplanationResponse,
    ExplanationRequest,
)


def make_customer_payload():
    return {
        "limit": 100000,
        "gender": 1,
        "edu": 2,
        "marital_status": 1,
        "age": 35,
        "rep_status_mth_6": 0,
        "rep_status_mth_5": 0,
        "rep_status_mth_4": 0,
        "rep_status_mth_3": 0,
        "rep_status_mth_2": 0,
        "rep_status_mth_1": 0,
        "bill_amt_mth_6": 50000,
        "bill_amt_mth_5": 50000,
        "bill_amt_mth_4": 50000,
        "bill_amt_mth_3": 50000,
        "bill_amt_mth_2": 50000,
        "bill_amt_mth_1": 50000,
        "pmt_amt_mth_6": 10000,
        "pmt_amt_mth_5": 10000,
        "pmt_amt_mth_4": 10000,
        "pmt_amt_mth_3": 10000,
        "pmt_amt_mth_2": 10000,
        "pmt_amt_mth_1": 10000,
    }


class ApiSchemaTests(unittest.TestCase):
    def test_customer_record_accepts_valid_cleaned_input(self):
        customer = CustomerRecord.model_validate(make_customer_payload())
        self.assertEqual(customer.limit, 100000)

    def test_customer_record_rejects_extra_columns(self):
        payload = make_customer_payload()
        payload["target"] = 0
        with self.assertRaises(ValidationError):
            CustomerRecord.model_validate(payload)

    def test_customer_record_rejects_non_positive_limit(self):
        payload = make_customer_payload()
        payload["limit"] = 0
        with self.assertRaises(ValidationError):
            CustomerRecord.model_validate(payload)

    def test_batch_request_requires_at_least_one_customer(self):
        with self.assertRaises(ValidationError):
            BatchPredictionRequest.model_validate({"customers": []})

    def test_explanation_request_contains_prediction_and_model_context(self):
        request = ExplanationRequest.model_validate(
            {
                "prediction": {
                    "probability": 0.72,
                    "decision": "REJECT",
                    "explanation": {
                        "base_value": -1.4,
                        "output_space": "model_output",
                        "contributions": [
                            {
                                "feature_name": "utilization_rate",
                                "feature_value": 0.9,
                                "shap_value": 0.6,
                                "direction": "increases_risk",
                            }
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
        self.assertEqual(request.prediction.decision, "REJECT")
        self.assertEqual(request.model_info.selected_model_name, "XGBoost")

    def test_deterministic_explanation_response_requires_a_limitation(self):
        with self.assertRaises(ValidationError):
            DeterministicExplanationResponse.model_validate(
                {
                    "summary": "The model assigned REVIEW.",
                    "increasing_contributors": [],
                    "decreasing_contributors": [],
                    "limitations": [],
                }
            )

    def test_deterministic_explanation_response_schema_forbids_additional_properties(self):
        response_schema = DeterministicExplanationResponse.model_json_schema()

        self.assertFalse(response_schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()

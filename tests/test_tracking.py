import unittest

import pandas as pd

from src.tracking import (
    fingerprint_dataframe,
    flatten_comparison_metrics,
    flatten_numeric_metrics,
)


class TrackingTests(unittest.TestCase):
    def test_comparison_metrics_are_flattened_for_mlflow(self):
        flattened_metrics = flatten_comparison_metrics(
            {
                "Random Forest": {
                    "pr_auc_score": 0.42,
                    "recall": 0.61,
                    "model_name": "Random Forest",
                }
            }
        )

        self.assertEqual(
            flattened_metrics,
            {
                "random_forest.pr_auc_score": 0.42,
                "random_forest.recall": 0.61,
            },
        )

    def test_dataset_fingerprint_is_stable_for_the_same_frame(self):
        data_frame = pd.DataFrame({"age": [25, 40], "limit": [1000, 2000]})

        self.assertEqual(
            fingerprint_dataframe(data_frame),
            fingerprint_dataframe(data_frame.copy()),
        )

    def test_dataset_fingerprint_changes_when_data_changes(self):
        original_data = pd.DataFrame({"age": [25, 40]})
        changed_data = pd.DataFrame({"age": [25, 41]})

        self.assertNotEqual(
            fingerprint_dataframe(original_data),
            fingerprint_dataframe(changed_data),
        )

    def test_nested_evaluation_report_metrics_are_flattened(self):
        self.assertEqual(
            flatten_numeric_metrics(
                {
                    "operational_policy": {
                        "binary_metrics": {"recall": 0.8},
                        "decision_rates": {"REJECT": 0.2},
                    },
                    "calibration": {"brier_score": 0.1},
                    "calibration_bins": [{"sample_count": 5}],
                }
            ),
            {
                "operational_policy.binary_metrics.recall": 0.8,
                "operational_policy.decision_rates.REJECT": 0.2,
                "calibration.brier_score": 0.1,
            },
        )


if __name__ == "__main__":
    unittest.main()

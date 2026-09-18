import unittest

from src.tracking import flatten_comparison_metrics


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


if __name__ == "__main__":
    unittest.main()

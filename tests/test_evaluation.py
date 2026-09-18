import unittest

from src.evaluation import evaluate_binary_predictions


class EvaluationTests(unittest.TestCase):
    def test_binary_evaluation_returns_ranking_classification_and_calibration_metrics(
        self,
    ):
        metrics = evaluate_binary_predictions(
            target_values=[0, 0, 1, 1],
            predicted_probabilities=[0.1, 0.2, 0.8, 0.9],
            predicted_classes=[0, 0, 1, 1],
        )

        self.assertEqual(metrics["roc_auc_score"], 1.0)
        self.assertEqual(metrics["pr_auc_score"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["true_positives"], 2.0)
        self.assertEqual(metrics["true_negatives"], 2.0)
        self.assertIn("brier_score", metrics)

    def test_binary_evaluation_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            evaluate_binary_predictions(
                target_values=[0, 1],
                predicted_probabilities=[0.2],
                predicted_classes=[0, 1],
            )

    def test_binary_evaluation_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            evaluate_binary_predictions(
                target_values=[0, 1],
                predicted_probabilities=[-0.1, 1.1],
                predicted_classes=[0, 1],
            )


if __name__ == "__main__":
    unittest.main()

import unittest

from src.evaluation import (
    analyze_calibration,
    evaluate_binary_predictions,
    evaluate_threshold_policy,
    select_thresholds,
)


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

    def test_calibration_analysis_returns_bins_and_brier_score(self):
        calibration = analyze_calibration(
            target_values=[0, 0, 1, 1],
            predicted_probabilities=[0.1, 0.2, 0.8, 0.9],
            number_of_bins=2,
        )

        self.assertEqual(calibration["brier_score"], 0.025)
        self.assertEqual(calibration["number_of_bins"], 2)
        self.assertEqual(len(calibration["calibration_bins"]), 2)
        self.assertEqual(calibration["calibration_bins"][0]["sample_count"], 2.0)

    def test_calibration_analysis_rejects_invalid_bin_count(self):
        with self.assertRaises(ValueError):
            analyze_calibration(
                target_values=[0, 1],
                predicted_probabilities=[0.2, 0.8],
                number_of_bins=1,
            )

    def test_threshold_policy_reports_decision_rates_and_error_rates(self):
        policy = evaluate_threshold_policy(
            target_values=[0, 0, 1, 1],
            predicted_probabilities=[0.05, 0.2, 0.6, 0.9],
            lower_threshold=0.1,
            upper_threshold=0.5,
        )

        self.assertEqual(policy["decision_counts"], {
            "APPROVE": 1,
            "REVIEW": 1,
            "REJECT": 2,
        })
        self.assertEqual(policy["false_approval_rate"], 0.0)
        self.assertEqual(policy["false_rejection_rate"], 0.0)

    def test_threshold_selection_returns_the_best_documented_candidate(self):
        selection = select_thresholds(
            target_values=[0, 0, 1, 1],
            predicted_probabilities=[0.05, 0.2, 0.6, 0.9],
            lower_thresholds=[0.1, 0.3],
            upper_thresholds=[0.5, 0.7],
        )

        self.assertEqual(selection["selection_metric"], "recall")
        self.assertIn("candidate_evaluations", selection)
        self.assertEqual(
            selection["selected_thresholds"],
            {"lower_threshold": 0.1, "upper_threshold": 0.5},
        )


if __name__ == "__main__":
    unittest.main()

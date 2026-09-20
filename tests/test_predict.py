import unittest

import pandas as pd

from src.config import ROOT
from src.predict import (
    _load_runtime_artifacts,
    evaluate_labelled_dataset,
    transform_for_prediction,
)
from src.utils import FeatureEngineering
from tests.test_data import make_customer_frame


class PredictionDatasetTests(unittest.TestCase):
    def test_prediction_transforms_raw_labelled_data_using_saved_style_transformer(self):
        raw_data = make_customer_frame()
        feature_engineering = FeatureEngineering()

        transformed_data = transform_for_prediction(
            raw_data,
            feature_engineering,
            require_target=True,
        )

        self.assertIn("pay_streak", transformed_data.columns)
        self.assertIn("target", transformed_data.columns)

    def test_labelled_evaluation_returns_generic_policy_results(self):
        labelled_data = pd.concat(
            [make_customer_frame(), make_customer_frame()],
            ignore_index=True,
        )
        labelled_data.loc[1, "target"] = 1
        labelled_data = FeatureEngineering().fit_transform(labelled_data)
        labelled_data["probability"] = [0.05, 0.75]

        evaluation = evaluate_labelled_dataset(
            labelled_data,
            lower_threshold=0.12,
            upper_threshold=0.30,
        )

        self.assertIn("operational_policy", evaluation)
        self.assertIn("calibration", evaluation)
        self.assertEqual(
            evaluation["thresholds"],
            {"lower_threshold": 0.12, "upper_threshold": 0.30},
        )

    def test_loose_artifact_directory_is_rejected_as_runtime_input(self):
        with self.assertRaises(ValueError):
            _load_runtime_artifacts(ROOT / "artifacts")

if __name__ == "__main__":
    unittest.main()

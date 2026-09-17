import unittest

import pandas as pd

from src.data import (
    DERIVED_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_engineered_schema,
    validate_schema,
)


def make_customer_frame(include_target=True):
    values = {
        column: [0] for column in RAW_FEATURE_COLUMNS
    }
    values["limit"] = [100000]
    values["bill_amt_mth_6"] = [50000]
    values["bill_amt_mth_1"] = [45000]
    values["pmt_amt_mth_6"] = [10000]
    if include_target:
        values[TARGET_COLUMN] = [0]
    return pd.DataFrame(values)


class SchemaValidationTests(unittest.TestCase):
    def test_prediction_schema_accepts_required_features_without_target(self):
        validate_schema(make_customer_frame(include_target=False))

    def test_training_schema_requires_target(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns: target"):
            validate_schema(make_customer_frame(include_target=False), require_target=True)

    def test_unexpected_columns_are_rejected(self):
        frame = make_customer_frame()
        frame["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "Unexpected columns"):
            validate_schema(frame, require_target=True)

    def test_feature_engineered_schema_requires_derived_features(self):
        frame = make_customer_frame()
        for column in DERIVED_FEATURE_COLUMNS:
            frame[column] = 0

        validate_feature_engineered_schema(frame, require_target=True)
        self.assertEqual(len(MODEL_FEATURE_COLUMNS), len(RAW_FEATURE_COLUMNS) + 7)

    def test_feature_engineered_schema_rejects_missing_derived_feature(self):
        frame = make_customer_frame()
        for column in DERIVED_FEATURE_COLUMNS[:-1]:
            frame[column] = 0

        with self.assertRaisesRegex(ValueError, "Missing feature-engineered columns"):
            validate_feature_engineered_schema(frame, require_target=True)


if __name__ == "__main__":
    unittest.main()

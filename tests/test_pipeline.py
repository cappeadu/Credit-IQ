import unittest

import pandas as pd

from src.data import (
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_cols,
    validate_feature_engineered_schema,
    validate_schema,
)
from src.utils import FeatureEngineering, split_dataset
from tests.test_data import make_customer_frame


def make_legacy_frame(row_count=20):
    rows = []
    for index in range(row_count):
        customer = make_customer_frame().iloc[0].copy()
        customer["target"] = index % 2
        customer["limit"] = 100000 + index
        customer["bill_amt_mth_6"] = 50000 + index
        customer["bill_amt_mth_1"] = 45000 + index
        customer["pmt_amt_mth_6"] = 10000 + index
        rows.append(customer)

    canonical = pd.DataFrame(rows)
    source_header = ["ID", *canonical.columns.tolist()]
    source_rows = [
        [index + 1, *row.tolist()]
        for index, (_, row) in enumerate(canonical.iterrows())
    ]
    return pd.DataFrame([source_header, *source_rows])


class EndToEndDataPipelineTests(unittest.TestCase):
    def test_legacy_data_reaches_valid_model_features(self):
        cleaned = clean_cols(make_legacy_frame())
        validate_schema(cleaned, require_target=True)

        train_df, holdout_df = split_dataset(cleaned, test_size=0.2)
        validation_df, test_df = split_dataset(holdout_df, test_size=0.5)

        feature_engineering = FeatureEngineering()
        train_fe = feature_engineering.fit_transform(train_df)
        validation_fe = feature_engineering.transform(validation_df)
        test_fe = feature_engineering.transform(test_df)

        for split in (train_fe, validation_fe, test_fe):
            validate_feature_engineered_schema(split, require_target=True)

        self.assertEqual(len(train_df) + len(validation_df) + len(test_df), 20)
        self.assertEqual(list(train_fe.columns[: len(RAW_FEATURE_COLUMNS)]), RAW_FEATURE_COLUMNS)
        self.assertIn(TARGET_COLUMN, test_fe.columns)


if __name__ == "__main__":
    unittest.main()

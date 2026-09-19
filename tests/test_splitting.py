import unittest
from pathlib import Path

import pandas as pd

from src.splitting import build_stable_row_ids, create_or_load_splits


def make_split_data(row_count=30):
    return pd.DataFrame(
        {
            "limit": [1000 + index for index in range(row_count)],
            "gender": [1] * row_count,
            "edu": [1] * row_count,
            "marital_status": [1] * row_count,
            "age": [30] * row_count,
            "rep_status_mth_6": [0] * row_count,
            "rep_status_mth_5": [0] * row_count,
            "rep_status_mth_4": [0] * row_count,
            "rep_status_mth_3": [0] * row_count,
            "rep_status_mth_2": [0] * row_count,
            "rep_status_mth_1": [0] * row_count,
            "bill_amt_mth_6": [100] * row_count,
            "bill_amt_mth_5": [100] * row_count,
            "bill_amt_mth_4": [100] * row_count,
            "bill_amt_mth_3": [100] * row_count,
            "bill_amt_mth_2": [100] * row_count,
            "bill_amt_mth_1": [100] * row_count,
            "pmt_amt_mth_6": [10] * row_count,
            "pmt_amt_mth_5": [10] * row_count,
            "pmt_amt_mth_4": [10] * row_count,
            "pmt_amt_mth_3": [10] * row_count,
            "pmt_amt_mth_2": [10] * row_count,
            "pmt_amt_mth_1": [10] * row_count,
            "target": [index % 2 for index in range(row_count)],
        }
    )


class SplittingTests(unittest.TestCase):
    def test_row_ids_are_stable(self):
        dataset = make_split_data()
        changed_target_dataset = dataset.copy()
        changed_target_dataset.loc[0, "target"] = 1 - changed_target_dataset.loc[0, "target"]

        self.assertEqual(
            build_stable_row_ids(dataset),
            build_stable_row_ids(changed_target_dataset),
        )

    def test_existing_manifest_keeps_old_members_and_assigns_new_rows_to_train(self):
        original_data = make_split_data()
        expanded_data = pd.concat(
            [original_data, make_split_data(row_count=2)],
            ignore_index=True,
        )
        new_row_start = len(original_data)
        expanded_data.loc[new_row_start:, "limit"] = [90001, 90002]
        expanded_data.loc[new_row_start:, "target"] = [0, 1]

        manifest_path = Path(__file__).parent / "split_manifest_test.json"
        manifest_path.unlink(missing_ok=True)
        try:
            original_splits = create_or_load_splits(
                original_data,
                manifest_path=manifest_path,
                dataset_fingerprint="original",
            )
            expanded_splits = create_or_load_splits(
                expanded_data,
                manifest_path=manifest_path,
                dataset_fingerprint="expanded",
            )
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertEqual(len(original_splits[1]), len(expanded_splits[1]))
        self.assertEqual(len(original_splits[2]), len(expanded_splits[2]))
        self.assertEqual(len(expanded_splits[0]), len(original_splits[0]) + 2)


if __name__ == "__main__":
    unittest.main()

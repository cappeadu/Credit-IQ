import unittest

import numpy as np

from src.utils import FeatureEngineering
from tests.test_data import make_customer_frame


class FeatureEngineeringTests(unittest.TestCase):
    def test_derived_features_are_created(self):
        frame = make_customer_frame()
        frame.loc[0, "rep_status_mth_6"] = 2
        frame.loc[0, "rep_status_mth_5"] = 1

        transformed = FeatureEngineering().fit_transform(frame)

        self.assertEqual(transformed.loc[0, "pay_streak"], 2)
        self.assertAlmostEqual(transformed.loc[0, "avg_payment_delay"], 0.5)
        self.assertAlmostEqual(transformed.loc[0, "utilization_rate"], 0.5)
        self.assertAlmostEqual(transformed.loc[0, "pmt_ratio"], 0.2)
        self.assertTrue(np.isfinite(transformed.select_dtypes(include="number")).all().all())

    def test_zero_latest_bill_uses_zero_payment_ratio(self):
        frame = make_customer_frame()
        frame.loc[0, "bill_amt_mth_6"] = 0
        frame.loc[0, "pmt_amt_mth_6"] = 100

        transformed = FeatureEngineering().fit_transform(frame)

        self.assertEqual(transformed.loc[0, "pmt_ratio"], 0)

    def test_zero_credit_limit_is_rejected(self):
        frame = make_customer_frame()
        frame.loc[0, "limit"] = 0

        with self.assertRaisesRegex(ValueError, "Credit limit must be greater than zero"):
            FeatureEngineering().fit_transform(frame)


if __name__ == "__main__":
    unittest.main()

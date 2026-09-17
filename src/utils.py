import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.data import (
    DERIVED_FEATURE_COLUMNS,
    validate_numeric_values,
    validate_schema,
)


# split dataset function
def split_dataset(
    df, test_size=0.2, random_state=42, save_dataset=None, path_to_save=None
):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["target"], random_state=random_state
    )
    if save_dataset:
        train_df.to_csv(f"{path_to_save}/train_set.csv", index=False)
        test_df.to_csv(f"{path_to_save}/val_test_set.csv", index=False)
    return train_df, test_df


#### create features
class FeatureEngineering(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Create deterministic risk features from a cleaned customer frame.

        The transformer validates its input before doing arithmetic.  A zero
        credit limit is rejected because utilization cannot be defined for
        that record.  When the latest bill is zero, payment ratio is defined
        as zero rather than dividing by an arbitrary fallback denominator.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        df = X.copy()
        has_target = "target" in df.columns
        validate_schema(df, require_target=has_target)
        validate_numeric_values(df, include_target=has_target)

        if (df["limit"] <= 0).any():
            raise ValueError("Credit limit must be greater than zero.")

        # pay streak
        df["pay_streak"] = (
            df[
                [
                    "rep_status_mth_6",
                    "rep_status_mth_5",
                    "rep_status_mth_4",
                    "rep_status_mth_3",
                    "rep_status_mth_2",
                    "rep_status_mth_1",
                ]
            ]
            > 0
        ).sum(axis=1)

        # avg payment delay
        df["avg_payment_delay"] = (
            df[
                [
                    "rep_status_mth_6",
                    "rep_status_mth_5",
                    "rep_status_mth_4",
                    "rep_status_mth_3",
                    "rep_status_mth_2",
                    "rep_status_mth_1",
                ]
            ]
        ).mean(axis=1)

        # utilization_rate. Upper limit reduced to 1.
        df["utilization_rate"] = np.clip(df["bill_amt_mth_6"] / df["limit"], None, 1)

        # Payment ratio. A zero bill has no meaningful payment denominator, so
        # a zero ratio is used for that case instead of adding an arbitrary 1.
        pmt_ratio = np.divide(
            df["pmt_amt_mth_6"],
            df["bill_amt_mth_6"],
            out=np.zeros(len(df), dtype=float),
            where=df["bill_amt_mth_6"].to_numpy() != 0,
        )
        df["pmt_ratio"] = np.clip(pmt_ratio, None, 1)

        # bill trend
        df["bill_trend"] = df["bill_amt_mth_6"] - df["bill_amt_mth_1"]

        # avg utlization rate
        avg_utilization_rate = (
            df[
                [
                    "bill_amt_mth_6",
                    "bill_amt_mth_5",
                    "bill_amt_mth_4",
                    "bill_amt_mth_3",
                    "bill_amt_mth_2",
                    "bill_amt_mth_1",
                ]
            ]
        ).sum(axis=1) / (6 * df["limit"])
        df["avg_utilization"] = np.clip(avg_utilization_rate, None, 1)

        df["worst_pmt_delay"] = np.max(
            df[
                [
                    "rep_status_mth_6",
                    "rep_status_mth_5",
                    "rep_status_mth_4",
                    "rep_status_mth_3",
                    "rep_status_mth_2",
                    "rep_status_mth_1",
                ]
            ],
            axis=1,
        )

        if not np.isfinite(df[DERIVED_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError("Feature engineering produced non-finite values.")

        return df


def loan_decision(prob, lower_threshold=0.12, upper_threshold=0.30):

    # 12% just to give an allowance
    if prob < lower_threshold:
        return "APPROVE"

    elif prob >= upper_threshold:
        return "REJECT"

    else:
        return "REVIEW"

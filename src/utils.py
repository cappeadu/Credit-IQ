import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


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
    def fit(self, X):
        return self

    def correcting_zero_div(self, row):
        if row["bill_amt_mth_6"] == 0:
            return row["pmt_amt_mth_6"] / (row["bill_amt_mth_6"] + 1)
        else:
            return row["pmt_amt_mth_6"] / row["bill_amt_mth_6"]

    def transform(self, X, y=None):
        df = X.copy()

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

        # utilization_rate. upper limit reduced to 1
        df["utilization_rate"] = np.clip(df["bill_amt_mth_6"] / df["limit"], None, 1)

        # payment ratio
        pmt_ratio = df.apply(self.correcting_zero_div, axis=1)
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
        return df


def loan_decision(prob, lower_threshold=0.12, upper_threshold=0.30):

    # 12% just to give an allowance
    if prob < lower_threshold:
        return "APPROVE"

    elif prob >= upper_threshold:
        return "REJECT"

    else:
        return "REVIEW"

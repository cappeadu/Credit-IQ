from pathlib import Path

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
    dataset,
    test_size=0.2,
    random_state=42,
    save_dataset=None,
    path_to_save=None,
):
    """Create a reproducible stratified split from a labelled clean dataset."""
    validate_schema(dataset, require_target=True)
    validate_numeric_values(dataset, include_target=True)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    if save_dataset and not path_to_save:
        raise ValueError("path_to_save is required when save_dataset is True.")

    training_data, holdout_data = train_test_split(
        dataset,
        test_size=test_size,
        stratify=dataset["target"],
        random_state=random_state,
    )
    if save_dataset:
        output_dir = Path(path_to_save)
        output_dir.mkdir(parents=True, exist_ok=True)
        training_data.to_csv(output_dir / "train_set.csv", index=False)
        holdout_data.to_csv(output_dir / "val_test_set.csv", index=False)
    return training_data, holdout_data


#### create features
class FeatureEngineering(TransformerMixin, BaseEstimator):
    def fit(self, training_data, target=None):
        return self

    def transform(self, customer_data, target=None):
        """Create deterministic risk features from a cleaned customer frame.

        The transformer validates its input before doing arithmetic.  A zero
        credit limit is rejected because utilization cannot be defined for
        that record.  When the latest bill is zero, payment ratio is defined
        as zero rather than dividing by an arbitrary fallback denominator.
        """
        if not isinstance(customer_data, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        engineered_data = customer_data.copy()
        has_target = "target" in engineered_data.columns
        validate_schema(engineered_data, require_target=has_target)
        validate_numeric_values(engineered_data, include_target=has_target)

        if (engineered_data["limit"] <= 0).any():
            raise ValueError("Credit limit must be greater than zero.")

        # pay streak
        engineered_data["pay_streak"] = (
            engineered_data[
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
        engineered_data["avg_payment_delay"] = (
            engineered_data[
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
        engineered_data["utilization_rate"] = np.clip(
            engineered_data["bill_amt_mth_6"] / engineered_data["limit"],
            None,
            1,
        )

        # Payment ratio. A zero bill has no meaningful payment denominator, so
        # a zero ratio is used for that case instead of adding an arbitrary 1.
        payment_ratio = np.divide(
            engineered_data["pmt_amt_mth_6"],
            engineered_data["bill_amt_mth_6"],
            out=np.zeros(len(engineered_data), dtype=float),
            where=engineered_data["bill_amt_mth_6"].to_numpy() != 0,
        )
        engineered_data["pmt_ratio"] = np.clip(payment_ratio, None, 1)

        # bill trend
        engineered_data["bill_trend"] = (
            engineered_data["bill_amt_mth_6"]
            - engineered_data["bill_amt_mth_1"]
        )

        # avg utlization rate
        avg_utilization_rate = (
            engineered_data[
                [
                    "bill_amt_mth_6",
                    "bill_amt_mth_5",
                    "bill_amt_mth_4",
                    "bill_amt_mth_3",
                    "bill_amt_mth_2",
                    "bill_amt_mth_1",
                ]
            ]
        ).sum(axis=1) / (6 * engineered_data["limit"])
        engineered_data["avg_utilization"] = np.clip(
            avg_utilization_rate,
            None,
            1,
        )

        engineered_data["worst_pmt_delay"] = np.max(
            engineered_data[
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

        if not np.isfinite(
            engineered_data[DERIVED_FEATURE_COLUMNS].to_numpy(dtype=float)
        ).all():
            raise ValueError("Feature engineering produced non-finite values.")

        return engineered_data


def loan_decision(prob, lower_threshold=0.12, upper_threshold=0.30):

    # 12% just to give an allowance
    if prob < lower_threshold:
        return "APPROVE"

    elif prob >= upper_threshold:
        return "REJECT"

    else:
        return "REVIEW"

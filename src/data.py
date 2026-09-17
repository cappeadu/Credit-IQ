import numpy as np
import pandas as pd

# Canonical schema used after the source dataset has been cleaned.  Keeping the
# schema here gives training, batch prediction, and the future API one source of
# truth for the columns a customer record must contain.
RAW_FEATURE_COLUMNS = [
    "limit",
    "gender",
    "edu",
    "marital_status",
    "age",
    "rep_status_mth_6",
    "rep_status_mth_5",
    "rep_status_mth_4",
    "rep_status_mth_3",
    "rep_status_mth_2",
    "rep_status_mth_1",
    "bill_amt_mth_6",
    "bill_amt_mth_5",
    "bill_amt_mth_4",
    "bill_amt_mth_3",
    "bill_amt_mth_2",
    "bill_amt_mth_1",
    "pmt_amt_mth_6",
    "pmt_amt_mth_5",
    "pmt_amt_mth_4",
    "pmt_amt_mth_3",
    "pmt_amt_mth_2",
    "pmt_amt_mth_1",
]

TARGET_COLUMN = "target"


def validate_schema(
    df: pd.DataFrame,
    *,
    require_target: bool = False,
    allow_extra_columns: bool = False,
) -> None:
    """Validate the canonical cleaned customer-data schema.

    This function deliberately validates without changing the input frame.  It
    is intended to be called before cleaning, feature engineering, or model
    prediction so bad input fails with a useful message at the boundary.

    Args:
        df: DataFrame using the project's cleaned column names.
        require_target: Require the labelled target column, as during training
            and test evaluation.
        allow_extra_columns: Permit additional columns, such as an ID supplied
            by a caller.  Production prediction paths should normally leave
            this disabled.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If required columns are missing or disallowed columns are
            present.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    required_columns = RAW_FEATURE_COLUMNS.copy()
    if require_target:
        required_columns.append(TARGET_COLUMN)

    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    if not allow_extra_columns:
        allowed_columns = set(required_columns)
        extra_columns = [
            column for column in df.columns if column not in allowed_columns
        ]
        if extra_columns:
            raise ValueError(
                "Unexpected columns: " + ", ".join(map(str, extra_columns))
            )


def validate_numeric_values(
    df: pd.DataFrame,
    *,
    include_target: bool = False,
) -> None:
    """Validate that canonical fields contain finite numeric values.

    Type conversion is intentionally not performed here.  A later preparation
    stage will define the project's explicit conversion policy; this utility
    only reports invalid input at the boundary.
    """
    columns = RAW_FEATURE_COLUMNS.copy()
    if include_target:
        columns.append(TARGET_COLUMN)

    non_numeric_columns = [
        column for column in columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            "Columns must contain numeric values: " + ", ".join(non_numeric_columns)
        )

    non_finite_columns = [
        column
        for column in columns
        if df[column].isna().any()
        or not np.isfinite(df[column].to_numpy(dtype=float)).all()
    ]
    if non_finite_columns:
        raise ValueError(
            "Columns contain missing or non-finite values: "
            + ", ".join(non_finite_columns)
        )


# load dataset
def load_dataset(data_path, frac=None, read_excel=None, read_csv=None):
    if read_excel:
        df = pd.read_excel(data_path)
    if read_csv:
        df = pd.read_csv(data_path)

    df = df.sample(frac=frac) if frac else df
    return df


def categorize_features(df):
    df = df.astype("int")  # change to integers

    # education
    # 1: Graduate School
    # 2: Undergrad/University
    # 3: High School
    # 4: Others
    # use the categories above
    edu_cat = [1, 2, 3]
    df["edu"] = df["edu"].apply(lambda x: x if x in edu_cat else 4)

    # marital status
    # 1: Married
    # 2: Single
    # 3: Others
    # use the categories above
    marital_cat = [1, 2]
    df["marital_status"] = df["marital_status"].apply(
        lambda x: x if x in marital_cat else 3
    )
    return df


# clean columns function
def clean_cols(df):
    # create dataset with readable column names
    custom_col_names = [
        "limit",
        "gender",
        "edu",
        "marital_status",
        "age",
        "rep_status_mth_6",
        "rep_status_mth_5",
        "rep_status_mth_4",
        "rep_status_mth_3",
        "rep_status_mth_2",
        "rep_status_mth_1",
        "bill_amt_mth_6",
        "bill_amt_mth_5",
        "bill_amt_mth_4",
        "bill_amt_mth_3",
        "bill_amt_mth_2",
        "bill_amt_mth_1",
        "pmt_amt_mth_6",
        "pmt_amt_mth_5",
        "pmt_amt_mth_4",
        "pmt_amt_mth_3",
        "pmt_amt_mth_2",
        "pmt_amt_mth_1",
        "target",
    ]

    # change column names to names on first row
    df.columns = df.iloc[0].values  # first row becomes the columns
    df = (
        df.iloc[1:].reset_index(drop=True).drop("ID", axis=1)
    )  # drop ID column. not useful
    df.columns = custom_col_names  # replace columns with custom names

    df = categorize_features(df)
    return df

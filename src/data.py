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
def load_dataset(data_path, frac=None, read_excel=False, read_csv=False):
    """Load exactly one supported dataset format.

    The previous implementation silently allowed both format flags to be
    false, which left ``df`` undefined.  It also sampled without a fixed seed,
    making a sampled dataset difficult to reproduce.
    """
    if read_excel == read_csv:
        raise ValueError("Set exactly one of read_excel or read_csv to True.")

    if read_excel:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    if frac is not None:
        if not 0 < frac <= 1:
            raise ValueError("frac must be greater than 0 and no greater than 1.")
        df = df.sample(frac=frac, random_state=42).reset_index(drop=True)

    return df


def categorize_features(df):
    df = df.copy().astype("int")  # change to integers

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
    """Convert the source dataset into the canonical cleaned schema.

    The source workbook currently contains a header row that is promoted from
    the first data row by the original project workflow.  That behaviour is
    retained for compatibility, but the resulting frame is now validated
    before it is returned to the training pipeline.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if df.empty:
        raise ValueError("The input dataset is empty.")

    df = df.copy()

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

    # Cleaned CSV splits already use the canonical schema.  Make cleaning
    # idempotent so those files can safely pass through the same boundary.
    if set(df.columns) == set(custom_col_names):
        df = df[custom_col_names]
        df = categorize_features(df)
        validate_schema(df, require_target=True)
        validate_numeric_values(df, include_target=True)
        return df

    # Change column names to names on the first row.  This is the format used
    # by the current source workbook and is kept until the source-data format
    # is standardised in a later stage.
    first_row_columns = df.iloc[0].tolist()
    if len(first_row_columns) != len(custom_col_names) + 1:
        raise ValueError(
            "Unexpected source schema: expected an ID column plus "
            f"{len(custom_col_names)} data columns, got {len(first_row_columns)}."
        )

    df.columns = first_row_columns  # first row becomes the columns
    if "ID" not in df.columns:
        raise ValueError("Unexpected source schema: missing ID column.")

    df = df.iloc[1:].reset_index(drop=True).drop("ID", axis=1)
    df.columns = custom_col_names  # replace columns with custom names

    df = categorize_features(df)
    validate_schema(df, require_target=True)
    validate_numeric_values(df, include_target=True)
    return df

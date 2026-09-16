import pandas as pd


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

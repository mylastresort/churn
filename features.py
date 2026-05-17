import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

SAMPLERS = {
    "smote":       lambda: SMOTE(random_state=42),
    "oversample":  lambda: RandomOverSampler(random_state=42),
    "undersample": lambda: RandomUnderSampler(random_state=42),
    "none":        None,
}

def get_train_test_data(path: str):
    train_df = pd.read_csv(path)
    df = train_df.drop(columns=["ID"])

    # Drop columns with any nulls
    df = df[[col for col in df.columns if df[col].isnull().sum() == 0]]

    num_cols = [col for col in df.columns
                if df[col].dtype in [np.float64, np.int64] and col != "TARGET"]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    x = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerics — fit on train only
    scaler = StandardScaler()
    x_train_num = pd.DataFrame(
        scaler.fit_transform(x_train[num_cols]),
        columns=num_cols,
        index=x_train.index,
    )
    x_test_num = pd.DataFrame(
        scaler.transform(x_test[num_cols]),
        columns=num_cols,
        index=x_test.index,
    )

    # OHE categoricals — fit on train only
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_train = ohe.fit_transform(x_train[cat_cols])
        ohe_test  = ohe.transform(x_test[cat_cols])

        ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()

        ohe_train_df = pd.DataFrame(ohe_train, columns=ohe_cols, index=x_train.index)
        ohe_test_df  = pd.DataFrame(ohe_test,  columns=ohe_cols, index=x_test.index)

        x_train = pd.concat([x_train_num, ohe_train_df], axis=1)
        x_test  = pd.concat([x_test_num,  ohe_test_df],  axis=1)
    else:
        x_train = x_train_num
        x_test  = x_test_num

    return x_train, x_test, y_train, y_test, scaler, ohe, num_cols, cat_cols


def get_test_data(path: str, features: list, scaler: StandardScaler, 
                  ohe: OneHotEncoder = None, num_cols: list = None, cat_cols: list = None):
    test_df = pd.read_csv(path)
    df = test_df.drop(columns=["ID"])

    # Scale numerics — transform only, never fit
    num_df = pd.DataFrame(
        scaler.transform(df[num_cols]),
        columns=num_cols,
    )

    # OHE categoricals — transform only, never fit
    if ohe is not None and cat_cols:
        ohe_arr = ohe.transform(df[cat_cols])
        ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(cat_cols).tolist())
        result  = pd.concat([num_df, ohe_df], axis=1)
    else:
        result = num_df

    result = result[features]        # keep only columns x_train used
    result["ID"] = test_df["ID"].values

    return result
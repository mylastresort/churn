import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# ── Samplers ──────────────────────────────────────────────────────────────────

SAMPLERS = {
    "smote":       lambda: SMOTE(random_state=42),
    "oversample":  lambda: RandomOverSampler(random_state=42),
    "undersample": lambda: RandomUnderSampler(random_state=42),
    "none":        None,
}


# ── Return object ─────────────────────────────────────────────────────────────

class DataSplit:
    """
    Holds all processed splits and the fitted preprocessors.

    Attributes
    ----------
    x_train, y_train  : training data (resampled if sampler != 'none')
    x_val,   y_val    : validation data (only if validation_size > 0, else None)
    x_test,  y_test   : held-out test data
    scaler            : fitted StandardScaler
    ohe               : fitted OneHotEncoder (None if no cat cols)
    num_cols          : list of numeric column names
    cat_cols          : list of categorical column names
    features          : all output column names (num + ohe expanded)
    """

    def __init__(self, x_train, y_train, x_val, y_val,
                 x_test, y_test, scaler, ohe, num_cols, cat_cols, features):
        self.x_train  = x_train
        self.y_train  = y_train
        self.x_val    = x_val
        self.y_val    = y_val
        self.x_test   = x_test
        self.y_test   = y_test
        self.scaler   = scaler
        self.ohe      = ohe
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.features = features

    def __repr__(self):
        val_shape = self.x_val.shape if self.x_val is not None else None
        return (
            f"DataSplit(\n"
            f"  x_train : {self.x_train.shape}  y_train : {self.y_train.shape}\n"
            f"  x_val   : {val_shape}  y_val   : {self.y_val.shape if self.y_val is not None else None}\n"
            f"  x_test  : {self.x_test.shape}  y_test  : {self.y_test.shape}\n"
            f"  features: {len(self.features)}  num: {len(self.num_cols)}  cat: {len(self.cat_cols)}\n"
            f")"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _process(x_tr_raw, x_other_raw, num_cols, cat_cols):
    """
    Fit scaler + OHE on x_tr_raw, transform both.
    Returns processed DataFrames and fitted transformers.
    """
    # Scale numerics — fit on train only
    scaler = StandardScaler()
    x_tr_num = pd.DataFrame(
        scaler.fit_transform(x_tr_raw[num_cols]),
        columns=num_cols,
        index=x_tr_raw.index,
    )
    x_other_num = pd.DataFrame(
        scaler.transform(x_other_raw[num_cols]),
        columns=num_cols,
        index=x_other_raw.index,
    )

    # OHE categoricals — fit on train only
    ohe = None
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_tr    = ohe.fit_transform(x_tr_raw[cat_cols])
        ohe_other = ohe.transform(x_other_raw[cat_cols])

        ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()

        x_tr_cat    = pd.DataFrame(ohe_tr,    columns=ohe_cols, index=x_tr_raw.index)
        x_other_cat = pd.DataFrame(ohe_other, columns=ohe_cols, index=x_other_raw.index)

        x_tr    = pd.concat([x_tr_num,    x_tr_cat],    axis=1)
        x_other = pd.concat([x_other_num, x_other_cat], axis=1)
    else:
        x_tr    = x_tr_num
        x_other = x_other_num

    return x_tr, x_other, scaler, ohe


def _resample(x, y, sampler: str):
    """Apply resampling — returns DataFrame + Series."""
    if sampler == "none" or sampler not in SAMPLERS:
        return x, y
    cols = x.columns.tolist()
    x_res, y_res = SAMPLERS[sampler]().fit_resample(x, y)
    return pd.DataFrame(x_res, columns=cols), pd.Series(y_res, name=y.name)


# ── Public API ────────────────────────────────────────────────────────────────

def get_train_test_data(
    path: str,
    sampler: str = "none",
    validation_size: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """
    Load, preprocess, and split training data.

    Parameters
    ----------
    path            : path to the training CSV
    sampler         : 'none' | 'smote' | 'oversample' | 'undersample'
    validation_size : fraction of training data to hold out as validation (0 = no val set)
    test_size       : fraction of full data to hold out as test
    random_state    : random seed

    Returns
    -------
    DataSplit object with x_train, y_train, x_val, y_val, x_test, y_test,
    scaler, ohe, num_cols, cat_cols, features
    """
    assert sampler in SAMPLERS, f"Unknown sampler '{sampler}'. Choose from {list(SAMPLERS)}"

    df = pd.read_csv(path).drop(columns=["ID"])

    # Drop columns with any nulls
    df = df[[col for col in df.columns if df[col].isnull().sum() == 0]]

    num_cols = [c for c in df.select_dtypes(include=[np.float64, np.int64]).columns
                if c != "TARGET"]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    x = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # ── 1. Train / test split (raw) ───────────────────────────────────────────
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 2. Optional train / val split (raw) ──────────────────────────────────
    if validation_size > 0:
        x_tr_raw, x_val_raw, y_tr, y_val = train_test_split(
            x_train_raw, y_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train,
        )
    else:
        x_tr_raw, x_val_raw = x_train_raw, None
        y_tr,     y_val     = y_train,     None

    # ── 3. Fit scaler + OHE on x_tr only, transform test ────────────────────
    #       We process train vs test here so scaler never sees val or test rows
    x_tr_scaled, x_test_scaled, scaler, ohe = _process(
        x_tr_raw, x_test_raw, num_cols, cat_cols
    )

    # ── 4. Transform val using the same fitted scaler + OHE ──────────────────
    if x_val_raw is not None:
        x_val_num = pd.DataFrame(
            scaler.transform(x_val_raw[num_cols]),
            columns=num_cols,
            index=x_val_raw.index,
        )
        if ohe is not None and cat_cols:
            ohe_val  = ohe.transform(x_val_raw[cat_cols])
            ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()
            x_val_cat = pd.DataFrame(ohe_val, columns=ohe_cols, index=x_val_raw.index)
            x_val_scaled = pd.concat([x_val_num, x_val_cat], axis=1)
        else:
            x_val_scaled = x_val_num
    else:
        x_val_scaled = None

    # ── 5. Resample train only — never val or test ───────────────────────────
    x_train_final, y_train_final = _resample(x_tr_scaled, y_tr, sampler)

    features = x_train_final.columns.tolist()

    return DataSplit(
        x_train  = x_train_final,
        y_train  = y_train_final,
        x_val    = x_val_scaled,
        y_val    = y_val,
        x_test   = x_test_scaled,
        y_test   = y_test,
        scaler   = scaler,
        ohe      = ohe,
        num_cols = num_cols,
        cat_cols = cat_cols,
        features = features,
    )


def get_test_data(path: str, data: DataSplit) -> pd.DataFrame:
    """
    Load and preprocess the holdout test file using the fitted preprocessors
    stored in the DataSplit object.

    Returns pd.DataFrame with processed features + original ID column.
    """
    test_df = pd.read_csv(path)
    ids     = test_df["ID"].values
    df      = test_df.drop(columns=["ID"])

    # Scale numerics
    num_df = pd.DataFrame(
        data.scaler.transform(df[data.num_cols]),
        columns=data.num_cols,
    )

    # OHE categoricals
    if data.ohe is not None and data.cat_cols:
        ohe_arr  = data.ohe.transform(df[data.cat_cols])
        ohe_cols = data.ohe.get_feature_names_out(data.cat_cols).tolist()
        ohe_df   = pd.DataFrame(ohe_arr, columns=ohe_cols)
        result   = pd.concat([num_df, ohe_df], axis=1)
    else:
        result = num_df

    result = result[data.features]
    result.insert(0, "ID", ids)
    return result
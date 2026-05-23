from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

NUM_IMPUTE_STRATEGIES = ("mean", "median", "drop")


def _impute_numerics(x_tr_raw, x_other_raw, num_cols, strategy):
    """
    Fit a SimpleImputer on x_tr_raw[num_cols], transform both splits.
    Returns imputed DataFrames and the fitted imputer.
    strategy: 'mean' | 'median'
    """
    imputer = SimpleImputer(strategy=strategy)

    x_tr_imp = pd.DataFrame(
        imputer.fit_transform(x_tr_raw[num_cols]),
        columns=num_cols,
        index=x_tr_raw.index,
    )
    x_other_imp = pd.DataFrame(
        imputer.transform(x_other_raw[num_cols]),
        columns=num_cols,
        index=x_other_raw.index,
    )
    return x_tr_imp, x_other_imp, imputer


def _process(x_tr_raw, x_other_raw, y_tr, num_cols, cat_cols, num_impute):
    imputer = None
    ohe = None

    # ── Impute missing numerics (fit on train only) ───────────────────────────
    if num_impute in ("mean", "median"):
        x_tr_num_raw, x_other_num_raw, imputer = _impute_numerics(
            x_tr_raw, x_other_raw, num_cols, strategy=num_impute
        )
    else:
        x_tr_num_raw = x_tr_raw[num_cols].copy()
        x_other_num_raw = x_other_raw[num_cols].copy()

    # ── Signed log1p transform — compress heavy tails (skewness up to 187) ───
    # Applied after imputation so no NaNs remain; skips binary null-indicator cols
    x_tr_num_raw = x_tr_num_raw.copy()
    x_other_num_raw = x_other_num_raw.copy()
    x_tr_num_raw[num_cols] = np.sign(x_tr_num_raw[num_cols].values) * np.log1p(
        np.abs(x_tr_num_raw[num_cols].values)
    )
    x_other_num_raw[num_cols] = np.sign(x_other_num_raw[num_cols].values) * np.log1p(
        np.abs(x_other_num_raw[num_cols].values)
    )

    # ── Scale numerics (fit on train only) ────────────────────────────────────
    scaler = StandardScaler()
    x_tr_num = pd.DataFrame(
        scaler.fit_transform(x_tr_num_raw),
        columns=num_cols,
        index=x_tr_raw.index,
    )
    x_other_num = pd.DataFrame(
        scaler.transform(x_other_num_raw),
        columns=num_cols,
        index=x_other_raw.index,
    )

    # ── Encode categoricals (fit on train only) ───────────────────────────────
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_tr = ohe.fit_transform(x_tr_raw[cat_cols])
        ohe_other = ohe.transform(x_other_raw[cat_cols])

        ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()

        x_tr_cat = pd.DataFrame(ohe_tr, columns=ohe_cols, index=x_tr_raw.index)
        x_other_cat = pd.DataFrame(ohe_other, columns=ohe_cols, index=x_other_raw.index)

        x_tr = pd.concat([x_tr_num, x_tr_cat], axis=1)
        x_other = pd.concat([x_other_num, x_other_cat], axis=1)
    else:
        x_tr = x_tr_num
        x_other = x_other_num

    return x_tr, x_other, scaler, imputer, ohe


def _apply_num_transform(x_raw, num_cols, cat_cols, scaler, imputer, ohe, index):
    """
    Apply fitted imputer → scaler → OHE (or target encoding) to a raw split (val or holdout).
    """
    # Impute
    num_data = x_raw[num_cols].copy()
    if imputer is not None:
        num_data = pd.DataFrame(
            imputer.transform(num_data),
            columns=num_cols,
            index=index,
        )

    # Signed log1p transform (must match _process)
    num_data = num_data.copy()
    num_data[num_cols] = np.sign(num_data[num_cols].values) * np.log1p(
        np.abs(num_data[num_cols].values)
    )

    # Scale numerics
    x_num = pd.DataFrame(
        scaler.transform(num_data),
        columns=num_cols,
        index=index,
    )

    # Categoricals
    if cat_cols and ohe is not None:
        ohe_arr = ohe.transform(x_raw[cat_cols])
        ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()
        x_cat = pd.DataFrame(ohe_arr, columns=ohe_cols, index=index)
        return pd.concat([x_num, x_cat], axis=1)

    return x_num

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from DataSplit import DataSplit
from utils import (
    NUM_IMPUTE_STRATEGIES,
    _process,
    _apply_num_transform,
)


def get_train_test_data(
    path: str,
    num_impute: str = "median",  # 'mean' | 'median' | 'none'
    validation_size: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    assert num_impute in (
        *NUM_IMPUTE_STRATEGIES,
        "none",
    ), f"Unknown num_impute '{num_impute}'. Choose from {(*NUM_IMPUTE_STRATEGIES, 'none')}"

    df = pd.read_csv(path).drop(columns=["ID", "CLNT_JOB_POSITION"])

    num_cols = [
        c
        for c in df.select_dtypes(include=[np.float64, np.int64]).columns
        if c != "TARGET"
    ]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Drop categorical columns with any missing values ─────────────────────
    null_cat_cols = [c for c in cat_cols if df[c].isnull().sum() > 0]
    df = df.drop(columns=null_cat_cols)
    cat_cols = [c for c in cat_cols if c in df.columns]

    # ── Handle numeric nulls ──────────────────────────────────────────────────
    if num_impute in ("none", "drop"):
        null_num_cols = [c for c in num_cols if df[c].isnull().sum() > 0]
        df = df.drop(columns=null_num_cols)
        num_cols = [c for c in num_cols if c in df.columns]

    x = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # ── 1. Train / test split (raw) ───────────────────────────────────────────
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 2. Optional train / val split (raw) ──────────────────────────────────
    if validation_size > 0:
        x_tr_raw, x_val_raw, y_tr, y_val = train_test_split(
            x_train_raw,
            y_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train,
        )
    else:
        x_tr_raw, x_val_raw = x_train_raw, None
        y_tr, y_val = y_train, None

    # ── 4. Fit imputer + scaler + encoder on x_tr only, transform test ────────
    x_tr_scaled, x_test_scaled, scaler, imputer, ohe = _process(
        x_tr_raw, x_test_raw, y_tr, num_cols, cat_cols, num_impute
    )

    # ── 5. Transform val using the same fitted transformers ───────────────────
    if x_val_raw is not None:
        x_val_scaled = _apply_num_transform(
            x_val_raw,
            num_cols,
            cat_cols,
            scaler,
            imputer,
            ohe,
            index=x_val_raw.index,
        )
    else:
        x_val_scaled = None

    # ── 6. Compute balanced class weights (fit on train labels only) ──────────
    labels = np.unique(y_tr)
    weights = compute_class_weight(class_weight="balanced", classes=labels, y=y_tr)
    class_weight = dict(zip(labels.tolist(), weights.tolist()))

    features = x_tr_scaled.columns.tolist()

    return DataSplit(
        x_train=x_tr_scaled,
        y_train=y_tr,
        x_val=x_val_scaled,
        y_val=y_val,
        x_test=x_test_scaled,
        y_test=y_test,
        scaler=scaler,
        imputer=imputer,
        ohe=ohe,
        num_cols=num_cols,
        cat_cols=cat_cols,
        features=features,
        class_weight=class_weight,
    )


def get_test_data(path: str, data: DataSplit) -> pd.DataFrame:
    """
    Load and preprocess the holdout test file using the fitted preprocessors
    stored in the DataSplit object.

    Returns pd.DataFrame with processed features + original ID column.
    """
    test_df = pd.read_csv(path)
    ids = test_df["ID"].values
    df = test_df.drop(columns=["ID"])

    result = _apply_num_transform(
        df,
        data.num_cols,
        data.cat_cols,
        data.scaler,
        data.imputer,
        data.ohe,
        index=df.index,
    )

    result = result[data.features]
    result.insert(0, "ID", ids)
    return result

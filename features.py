import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, chi2

# ── Samplers ──────────────────────────────────────────────────────────────────

SAMPLERS = {
    "smote": lambda: SMOTE(random_state=42),
    "oversample": lambda: RandomOverSampler(random_state=42),
    "undersample": lambda: RandomUnderSampler(random_state=42),
    "none": None,
}

NUM_IMPUTE_STRATEGIES = ("mean", "median", "drop")
CAT_HANDLE_STRATEGIES = ("drop", "none", "most_frequent", "constant")


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
    imputer           : fitted SimpleImputer for numeric columns (None if num_impute='none')
    ohe               : fitted OneHotEncoder (None if no cat cols)
    num_cols          : list of numeric column names
    cat_cols          : list of categorical column names
    features          : all output column names (num + ohe expanded)
    """

    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        scaler,
        imputer,
        ohe,
        num_cols,
        cat_cols,
        features,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler
        self.imputer = imputer
        self.ohe = ohe
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.features = features

    def __repr__(self):
        val_shape = self.x_val.shape if self.x_val is not None else None
        imputer_str = self.imputer.strategy if self.imputer is not None else "none"
        return (
            f"DataSplit(\n"
            f"  x_train : {self.x_train.shape}  y_train : {self.y_train.shape}\n"
            f"  x_val   : {val_shape}  y_val   : {self.y_val.shape if self.y_val is not None else None}\n"
            f"  x_test  : {self.x_test.shape}  y_test  : {self.y_test.shape}\n"
            f"  features: {len(self.features)}  num: {len(self.num_cols)}  cat: {len(self.cat_cols)}\n"
            f"  imputer : {imputer_str}\n"
            f")"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────


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


def _process(x_tr_raw, x_other_raw, num_cols, cat_cols, num_impute):
    imputer = None

    # ── Impute missing numerics (fit on train only) ───────────────────────────
    if num_impute in (
        "mean",
        "median",
    ):  # ← was NUM_IMPUTE_STRATEGIES which now includes "drop"
        x_tr_num_raw, x_other_num_raw, imputer = _impute_numerics(
            x_tr_raw, x_other_raw, num_cols, strategy=num_impute
        )
    else:
        # "drop" | "none" → pass through as-is, nulls already removed upstream
        x_tr_num_raw = x_tr_raw[num_cols].copy()
        x_other_num_raw = x_other_raw[num_cols].copy()

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

    # ── OHE categoricals (fit on train only) ─────────────────────────────────
    ohe = None
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
    Apply fitted imputer → scaler → OHE to a raw split (val or holdout).
    """
    # Impute
    num_data = x_raw[num_cols].copy()
    if imputer is not None:
        num_data = pd.DataFrame(
            imputer.transform(num_data),
            columns=num_cols,
            index=index,
        )

    # Scale
    x_num = pd.DataFrame(
        scaler.transform(num_data),
        columns=num_cols,
        index=index,
    )

    # OHE
    if ohe is not None and cat_cols:
        ohe_arr = ohe.transform(x_raw[cat_cols])
        ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()
        x_cat = pd.DataFrame(ohe_arr, columns=ohe_cols, index=index)
        return pd.concat([x_num, x_cat], axis=1)

    return x_num


def _resample(x, y, sampler: str):
    """Apply resampling — returns DataFrame + Series."""
    if sampler == "none" or sampler not in SAMPLERS:
        return x, y
    cols = x.columns.tolist()
    x_res, y_res = SAMPLERS[sampler]().fit_resample(x, y)
    return pd.DataFrame(x_res, columns=cols), pd.Series(y_res, name=y.name)


# ── Public API ────────────────────────────────────────────────────────────────
# CAT_HANDLE_STRATEGIES = ("drop", "none")
def get_train_test_data(
    path: str,
    sampler: str = "none",
    threshold_missing_data: float = 0,
    num_impute: str = "median",  # 'mean' | 'median' | 'none'
    cat_handle: str = "drop",  # 'drop' | 'none'
    validation_size: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
    cat_k: int | None = None,  # if not None, SelectKBest chi2 on cat cols to keep top k
    num_k: (
        int | None
    ) = None,  # if not None, SelectKBest f_classif on num cols to keep top k
) -> DataSplit:
    assert (
        sampler in SAMPLERS
    ), f"Unknown sampler '{sampler}'. Choose from {list(SAMPLERS)}"
    assert num_impute in (
        *NUM_IMPUTE_STRATEGIES,
        "none",
    ), f"Unknown num_impute '{num_impute}'. Choose from {(*NUM_IMPUTE_STRATEGIES, 'none')}"
    assert (
        cat_handle in CAT_HANDLE_STRATEGIES
    ), f"Unknown cat_handle '{cat_handle}'. Choose from {list(CAT_HANDLE_STRATEGIES)}"

    df = pd.read_csv(path).drop(columns=["ID", "CLNT_JOB_POSITION"])

    if threshold_missing_data > 0:
        # Drop the raw columns that are >80% missing — keep only their null flags
        drop_high_missing = [
            c
            for c in df.columns
            if c not in ("ID", "TARGET")
            and df[c].isnull().mean() > threshold_missing_data
            and not c.endswith("_WAS_NULL")
        ]

        df = df.drop(columns=drop_high_missing)

    num_cols = [
        c
        for c in df.select_dtypes(include=[np.float64, np.int64]).columns
        if c != "TARGET"
    ]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Handle categorical nulls ──────────────────────────────────────────────
    if cat_handle == "drop":
        null_cat_cols = [c for c in cat_cols if df[c].isnull().sum() > 0]
        df = df.drop(columns=null_cat_cols)
        cat_cols = [c for c in cat_cols if c in df.columns]
    # 'none' → leave cat nulls as-is (OHE will raise if nulls exist, user's responsibility)

    # ── Handle numeric nulls ──────────────────────────────────────────────────
    if num_impute in ("none", "drop"):
        null_num_cols = [c for c in num_cols if df[c].isnull().sum() > 0]
        df = df.drop(columns=null_num_cols)
        num_cols = [c for c in num_cols if c in df.columns]
    # 'mean' | 'median' → nulls kept here, SimpleImputer handles them later

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

    # ── 3. SelectKBest on cat_cols (fit on train only) ────────────────────────
    if cat_cols and cat_k is not None:
        cat_k = min(cat_k, len(cat_cols))

        _cat_imputer = SimpleImputer(strategy="most_frequent")
        x_tr_cat_imp = _cat_imputer.fit_transform(x_tr_raw[cat_cols])

        _ord_enc = OrdinalEncoder()
        x_tr_cat_enc = _ord_enc.fit_transform(x_tr_cat_imp)

        selector = SelectKBest(chi2, k=cat_k)
        selector.fit(x_tr_cat_enc, y_tr)

        selected_idx = selector.get_support(indices=True)
        scores = pd.Series(selector.scores_, index=cat_cols).sort_values(
            ascending=False
        )
        cat_cols = [cat_cols[i] for i in selected_idx]

        print(f"[SelectKBest chi2] keeping {len(cat_cols)}/{len(scores)} cat cols")
        print(scores.to_string(), "\n")
        print(f"Selected: {cat_cols}")

    # ── SelectKBest on num_cols (fit on train only) ───────────────────────────
    if num_cols and num_k is not None:
        from sklearn.feature_selection import f_classif

        num_k = min(num_k, len(num_cols))

        x_tr_num_imp = x_tr_raw[num_cols].fillna(x_tr_raw[num_cols].median())

        selector_num = SelectKBest(f_classif, k=num_k)
        selector_num.fit(x_tr_num_imp, y_tr)

        selected_idx_num = selector_num.get_support(indices=True)
        scores_num = pd.Series(selector_num.scores_, index=num_cols).sort_values(
            ascending=False
        )
        num_cols = [num_cols[i] for i in selected_idx_num]

        print(
            f"[SelectKBest f_classif] keeping {len(num_cols)}/{len(scores_num)} num cols"
        )
        print(scores_num.to_string(), "\n")
        print(f"Selected: {num_cols}")

    # ── 4. Fit imputer + scaler + OHE on x_tr only, transform test ───────────
    x_tr_scaled, x_test_scaled, scaler, imputer, ohe = _process(
        x_tr_raw, x_test_raw, num_cols, cat_cols, num_impute
    )

    # ── 5. Transform val using the same fitted transformers ───────────────────
    if x_val_raw is not None:
        x_val_scaled = _apply_num_transform(
            x_val_raw, num_cols, cat_cols, scaler, imputer, ohe, index=x_val_raw.index
        )
    else:
        x_val_scaled = None

    # ── 5. Resample train only — never val or test ───────────────────────────
    x_train_final, y_train_final = _resample(x_tr_scaled, y_tr, sampler)

    features = x_train_final.columns.tolist()

    return DataSplit(
        x_train=x_train_final,
        y_train=y_train_final,
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

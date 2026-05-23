class DataSplit:
    """
    Holds all processed splits and the fitted preprocessors.

    Attributes
    ----------
    x_train, y_train  : training data
    x_val,   y_val    : validation data (only if validation_size > 0, else None)
    x_test,  y_test   : held-out test data
    scaler            : fitted StandardScaler
    imputer           : fitted SimpleImputer for numeric columns (None if num_impute='none')
    ohe               : fitted OneHotEncoder (None if no cat cols)
    num_cols          : list of numeric column names
    cat_cols          : list of categorical column names
    features          : all output column names (num + ohe expanded or target encoded)
    class_weight      : dict mapping class label -> balanced weight (for model.fit)
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
        class_weight,
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
        self.class_weight = class_weight

    def __repr__(self):
        val_shape = self.x_val.shape if self.x_val is not None else None
        imputer_str = self.imputer.strategy if self.imputer is not None else "none"
        enc_str = "ohe" if self.ohe is not None else "none"
        return (
            f"DataSplit(\n"
            f"  x_train : {self.x_train.shape}  y_train : {self.y_train.shape}\n"
            f"  x_val   : {val_shape}  y_val   : {self.y_val.shape if self.y_val is not None else None}\n"
            f"  x_test  : {self.x_test.shape}  y_test  : {self.y_test.shape}\n"
            f"  features: {len(self.features)}  num: {len(self.num_cols)}  cat: {len(self.cat_cols)}\n"
            f"  imputer : {imputer_str}  cat_enc: {enc_str}\n"
            f")"
        )

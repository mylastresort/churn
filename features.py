import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def get_train_test_data(path: str):
    train_df = pd.read_csv(path)
    df = train_df.drop(columns=["ID"])

    allowed_features = [
        col for col in df.columns if df[col].dtype in [np.float64, np.int64]
    ]
    features = [col for col in allowed_features if df[col].isnull().sum() == 0]
    df = df[features]

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["TARGET"]),
        df["TARGET"],
        test_size=0.2,
        random_state=42,
        stratify=df["TARGET"],
    )

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    return x_train, x_test, y_train, y_test

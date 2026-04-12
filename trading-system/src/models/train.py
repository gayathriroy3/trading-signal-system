import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from src.data import split

def train_model(df, features):
    X = df[features].shift(1)
    y = df['target']

    df_model = pd.concat([X, y, df['close'], df['date']], axis=1).dropna()

    X = df_model[features]
    y = df_model['target']

    X_train,y_train,X_test,y_test,split_idx=split(df_model,X,y)

    tscv = TimeSeriesSplit(n_splits=3)

    base_model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model = CalibratedClassifierCV(base_model, method='isotonic', cv=tscv)
    model.fit(X_train, y_train)

    return model, X_test, y_test, df_model.iloc[split_idx:]
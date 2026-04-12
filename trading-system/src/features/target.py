import numpy as np

def create_binary_target(df, horizon=15):
    df["future_close"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
    df["vol"] = df["close"].pct_change().rolling(50).std()

    up_th = 0.0025 + 0.5 * df['vol']
    down_th = 0.0020 + 0.5 * df['vol']

    df["target"] = np.where(df["future_return"] > up_th, 1,
                            np.where(df["future_return"] < -down_th, 0,
                                     np.nan))
    return df.dropna()
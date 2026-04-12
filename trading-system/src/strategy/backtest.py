import pandas as pd

def generate_signals(model, X_test, df_test):
    df_test = df_test.copy()

    probs = model.predict_proba(X_test)[:, 1]
    df_test['prob'] = probs

    df_test['strength'] = abs(df_test['prob'] - 0.5)

    def get_position(prob):
        if prob > 0.60:
            return 1
        elif prob < 0.38:
            return -1
        else:
            return 0

    df_test['position'] = df_test['prob'].apply(get_position)
    return df_test

def backtest(df, cost=0.0002, sl=-0.002, tp=0.004):
    df = df.copy()
    df['return'] = df['close'].pct_change()

    trades = []
    position = 0
    entry_price = 0

    for i in range(1, len(df)):
        row = df.iloc[i]

        if position == 0:
            if row['position'] != 0:
                position = row['position']
                entry_price = row['close']
                entry_idx = i

        else:
            current_price = row['close']
            pnl = (current_price - entry_price) / entry_price * position

            if pnl <= sl or pnl >= tp or row['position'] == -position:
                trades.append(pnl - 2 * cost)
                position = 0

    return pd.Series(trades)
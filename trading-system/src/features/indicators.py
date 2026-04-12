import pandas as pd

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha=1/window)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stochastic(df, k_window=14):
    low_min = df["low"].rolling(k_window).min()
    high_max = df["high"].rolling(k_window).max()

    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0,1e-9)
    return df

def compute_breakout(df):
  df["rolling_high"]=df["high"].rolling(20).max().shift(1)
  df["rolling_low"]=df["low"].rolling(20).min().shift(1)
  df["high_break"]=(df["close"]-df["rolling_high"])/df["rolling_high"].replace(0, 1e-9)
  df["low_break"]=(df["rolling_low"]-df["close"])/df["rolling_low"].replace(0, 1e-9)
  df["high_break_vol"]=df["high_break"]/df["volatility"].replace(0, 1e-9)
  df["low_break_vol"]=df["low_break"]/df["volatility"].replace(0, 1e-9)
  return df

def getTrendFeatures(df):
  df["ema_9"]=df["close"].ewm(span=19).mean()
  df["ema_diff"]=(df["close"]-df["ema_9"])/df["ema_9"].replace(0, 1e-9)
  df['ema_fast']=df["close"].ewm(span=10).mean()
  df['ema_slow']=df["close"].ewm(span=30).mean()
  df['trend']=df['ema_fast']-df['ema_slow']
  df["below_ema"]=(df["close"]<df["ema_9"]).astype(int)
  df["below_ema_count"]=df["below_ema"].rolling(5).sum()

  df["above_ema"]=(df["close"]>df["ema_9"]).astype(int)
  df["above_ema_count"]=df["above_ema"].rolling(5).sum()
  df['nifty_return']=df['close'].pct_change(1)
  df['nifty_return_10']=df['close'].pct_change(10)
  df["volatility"]=df["nifty_return"].rolling(20).std()
  df["nifty_ema20"] = df["close"].ewm(span=20).mean()
  df["nifty_ema9"] = df["close"].ewm(span=9).mean()
  df["ema_diff"]=df["nifty_ema9"]-df["nifty_ema20"]
  df["range_ratio"]=(df["high"]-df["low"])/df["close"].replace(0, 1e-9)

  df["nifty_trend"] = (df["close"] - df["nifty_ema20"]) / df["nifty_ema20"].replace(0, 1e-9)
  df["nifty_rsi"] = compute_rsi(df["close"])
  df["nifty_range"] = (df["high"] - df["low"]) / df["close"].replace(0, 1e-9)
  return df

def rule_features(df):
  df["below_ema"]=(df["close"]<df["nifty_ema9"]).astype(int)
  df["trend_below_ema"]=df["below_ema"].rolling(5).mean()
  df["gap_down"]=(df["open"]<df["low"].shift(1)).astype(int)
  df["stoch_overbought"]=(df["stoch_k"]>80).astype(int)
  df["stoch_oversold"]=(df["stoch_k"]<60).astype(int)
  return df

def getBreakFeatures(df):
  df["prev_close"]=df["close"].shift(1)
  df["breakdown"]=(df["open"]<df["prev_close"]).astype(int)
  df["breakout"]=(df["open"]>df["prev_close"]).astype(int)
  return df

def getStochFeatures(df):
  low_min=df["low"].rolling(18).min()
  high_max=df["high"].rolling(18).max()

  df["stoch_k"]=100*(df["close"]-low_min)/(high_max-low_min+1e-9)

  df["stoch_overbought"]=((df["stoch_k"]>60) & (df["stoch_k"]<80)).astype(int)
  df["stoch_oversold"]=((df["stoch_k"]>20) & (df["stoch_k"]<40)).astype(int)
  return df

def getStrengthFeatures(df):
  df["range"]=df["high"]-df["low"]
  df["body"]=abs(df["close"]-df["open"])
  df["candle_strength"]=df["body"]/(df["range"].replace(0, 1e-9))

  df["bearish_strength"]=((df["close"]<df["open"])*df["candle_strength"])
  df["bullish_strength"]=((df["close"]>df["open"])*df["candle_strength"])
  return df

def getMomentumFeatures(df):
  df['nifty_return']=df['close'].pct_change(1)
  df['nifty_return_5']=df['close'].pct_change(5)
  df["ret_1"]=df["close"].pct_change(3)
  df["ret_3"]=df["close"].pct_change(3)
  df["ret_5"]=df["close"].pct_change(5)
  df["ret_10"]=df["close"].pct_change(10)

  df["down_momentum"]=(df["ret_5"]<0).astype(int)
  df["up_momentum"]=(df["ret_5"]>0).astype(int)
  df['momentum']=df['ret_3']+df['ret_5']

  df['hour']=df.index.hour
  df['minute']=df.index.minute

  df["session"] = df["hour"].apply(
      lambda x: "morning" if x < 11 else "midday" if x < 14 else "closing"
  )
  session_dummies=pd.get_dummies(df["session"],prefix="session",dtype=int)
  df=pd.concat([df,session_dummies],axis=1)
  df.drop(columns=["session"],inplace=True)
  return df

def getVolatilityFeatures(df):
  df["volatility"]=df["ret_1"].rolling(50).std()
  df['vol_regime']=df['volatility']/df['volatility'].rolling(100).mean().replace(0, 1e-9)
  df["range_expansion"]=df["range"]/df["range"].rolling(10).mean().replace(0, 1e-9)
  return df

def getTraderSetupFeatures(df):
  df['sell_setup']=((df['below_ema_count']>=4) &
   (df['breakdown']==1) & (df['stoch_overbought']==1)).astype(int)
  df["sell_strength"]=(df['below_ema_count']/5+df['bearish_strength']+abs(df['ret_5']))

  df['buy_setup']=((df['above_ema_count']>=4) & (df['breakout']==1) & (df['stoch_oversold']==1)).astype(int)
  df['buy_strength']=(df['above_ema_count']/5+df['bullish_strength']+abs(df['ret_5']))
  return df
from datetime import date


NIFTY = "^NSEI"
DEFAULT_INTERVAL = "5m"

# Features
FEATURES = [
        'nifty_return',
        'nifty_return_5',
        'nifty_return_10',
        'volatility',
        'nifty_ema20',
        'nifty_ema9',
        'ema_diff',
        'range_ratio',
        'nifty_trend',
        'nifty_rsi',
        'stoch_k',
        'rolling_high',
        'rolling_low',
        'high_break',
        'low_break',
        'high_break_vol',
        'low_break_vol',
        'below_ema',
        'trend_below_ema',
        'gap_down',
        'stoch_overbought',
        'stoch_oversold',
        'hour',
        'session_closing',
        'session_midday',
        'session_morning',
        'ema_fast',
        'ema_slow',
        'trend',
        'below_ema_count',
        'above_ema_count',
        'breakdown',
        'breakout',
        'candle_strength',
        'bearish_strength',
        'bullish_strength',
        'ret_1',
        'ret_3',
        'ret_5',
        'ret_10',
        'down_momentum',
        'up_momentum',
        'momentum',
        'vol_regime',
        'range_expansion',
        'sell_setup',
        'sell_strength',
        'buy_setup',
        'buy_strength'
    ]

# -------------------------
# Targets
# -------------------------
HORIZONS = [3, 6, 9, 12]  # 15m, 30m, 45m, 60m
BASE_THRESHOLD = 0.003
TARGETS=["target_15min","target_30min","target_45min","target_60min"]

# -------------------------
# Model
# -------------------------
MODEL_PATH='models/model_v1.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42

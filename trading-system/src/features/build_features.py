import pandas as pd
from .indicators import *
from data.preprocess import clean_nifty_data

def get_and_process_data(df, selected_date=None):
    df = clean_nifty_data(df.copy(), selected_date=selected_date)

    # Step 1: Calculate basic features that don't depend on too many others yet
    df = getMomentumFeatures(df)

    # Step 2: Calculate trend, RSI, and initial volatility components
    df = getTrendFeatures(df)

    # Step 3: Calculate Stochastics
    df = getStochFeatures(df)

    # Step 4: Calculate Break features
    df = getBreakFeatures(df)

    # Step 5: Calculate Strength features
    df = getStrengthFeatures(df)

    # Step 6: Calculate advanced volatility features (depends on earlier volatility/range)
    df = getVolatilityFeatures(df)

    # Step 7: Calculate breakout features (depends on rolling high/low and volatility)
    df = compute_breakout(df)

    # Step 8: Apply rule-based features (depends on EMA, stochastic, previous low/open)
    df = rule_features(df)

    # Step 9: Calculate Trader Setup features (depends on many prior features)
    df = getTraderSetupFeatures(df)

    return df
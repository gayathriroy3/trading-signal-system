import pandas as pd
import yfinance as yf
import joblib
import os
from src.features import get_and_process_data
from src.config.config import FEATURES,MODEL_PATH
from src.data import load_data
from src.features.target import create_binary_target
from src.models.train import train_model
from src.strategy.backtest import generate_signals, backtest
from src.models.evaluate import evaluate

def main():
    # Load NIFTY 50 data
    nifty_df=load_data('NIFTY 50_5minute.csv')

    # Clean and process data (using the combined get_and_process_data from feature_engineering)
    print("Cleaning and processing data...")
    nifty_df = get_and_process_data(nifty_df)
    nifty_df = nifty_df.dropna()

    # Create binary target
    print("Creating binary target...")
    nifty_df = create_binary_target(nifty_df)

    # Train model
    print("Training model...")
    model, X_test, y_test, df_test = train_model(nifty_df, FEATURES)

    # Generate signals
    print("Generating signals...")
    df_test = generate_signals(model, X_test, df_test)

    # Backtest and evaluate
    print("Backtesting and evaluating...")
    trades = backtest(df_test)
    evaluate(trades)

    # Save the trained model
    print("Saving model...")
    joblib.dump(model, MODEL_PATH)
    print(f"Model {MODEL_PATH} saved successfully.")

if __name__ == "__main__":

    os.makedirs('src', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    main()
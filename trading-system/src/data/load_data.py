import pandas as pd

def load_data(filename):
    print("Loading NIFTY 50 data...")
    nifty_df = pd.read_csv(filename)
    nifty_df = nifty_df.rename(columns={'date': 'datetime'})
    nifty_df["datetime"] = pd.to_datetime(nifty_df["datetime"])
    nifty_df["datetime"] = nifty_df["datetime"].dt.tz_localize("Asia/Kolkata")
    nifty_df.drop(columns=['volume'], inplace=True)
    nifty_df.set_index("datetime", inplace=True)
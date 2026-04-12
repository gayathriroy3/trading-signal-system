import pandas as pd


def clean_nifty_data(df, selected_date=None):
    df_copy = df.copy()
    df_copy["date"] = df_copy.index.date

    # Use selected_date if provided, otherwise default to the latest date in the dataframe
    current_trading_date = selected_date if selected_date else df_copy.index.date.max()

    df_copy['temp_datetime_for_diff'] = df_copy.index
    df_copy["time_diff"] = df_copy.groupby("date")["temp_datetime_for_diff"].diff()
    df_copy.drop(columns=['temp_datetime_for_diff'], inplace=True)

    counts_per_day = df_copy.groupby("date").size().reset_index(name="count")
    low_entry_count_days = counts_per_day[counts_per_day["count"] < 60][["date"]]

    inconsistent_time_diff_days = df_copy[df_copy["time_diff"] > pd.Timedelta(minutes=5)][["date"]]

    bad_days_nifty_all = pd.concat([inconsistent_time_diff_days, low_entry_count_days]).drop_duplicates().sort_values('date').reset_index(drop=True)

    if not bad_days_nifty_all.empty:
        bad_days_nifty_all = bad_days_nifty_all[bad_days_nifty_all['date'] != current_trading_date]

    df_cleaned = df_copy[~df_copy["date"].isin(bad_days_nifty_all['date'])].copy()
    return df_cleaned
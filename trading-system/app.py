import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib

from src.features.build_features import get_and_process_data
from src.features.indicators import *
from src.config.config import NIFTY,MODEL_PATH,FEATURES
from src.data.refresh_data import load_predict
import datetime


# Load the pre-trained model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file {MODEL_PATH} not found. Please train the model using main.py first.")
    st.stop()

# 1. SETUP: Page configuration & Auto-refresh (every 300,000ms = 5 minutes)
st.set_page_config(page_title="Nifty 50 Intraday Signal Dashboard", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

st.title("📊 Nifty 50 Real-Time Signal Dashboard")
st.markdown("This dashboard updates automatically every 5 minutes to scan for your trader's setup.")
st.info(f"Last dashboard refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 

nifty_predict,start_date_for_yf,end_date_for_yf,selected_date=load_predict()

if nifty_predict.empty:
    st.warning(f"No data fetched from Yahoo Finance for the period {start_date_for_yf.strftime('%Y-%m-%d')} to {end_date_for_yf.strftime('%Y-%m-%d')}. Please check the selected date and market availability.")
    st.stop()

nifty_predict.index = nifty_predict.index.tz_convert("Asia/Kolkata")

nifty_predict.drop(columns=['Volume'],inplace=True)
nifty_predict.columns = [col[0].lower() for col in nifty_predict.columns]
nifty_predict.rename_axis('datetime', inplace=True)

# 2. DATA PROCESSING FUNCTION
df=get_and_process_data(nifty_predict, selected_date=selected_date)

current_day_data = df[df.index.date == selected_date].copy()

if current_day_data.empty:
    st.warning("No Nifty 50 data available for the selected trading day after feature calculation. This might indicate an issue with the data source, market conditions, or the selected date.")
    st.stop() 


for feature in FEATURES:
    if feature not in current_day_data.columns:
        current_day_data[feature] = 0.0 

X_live = current_day_data[FEATURES]

X_live = X_live.fillna(0)

if X_live.empty:
    st.warning("No data remaining after feature selection and NaN imputation. This might indicate an issue with feature calculation or insufficient historical data provided by Yahoo Finance for the selected date.")
    st.stop()

# Ensure all feature columns exist in the DataFrame after processing and before selecting for X_live
# Handle missing feature columns by adding them with default values (e.g., 0) if they don't exist
for feature in FEATURES:
    if feature not in current_day_data.columns:
        current_day_data[feature] = 0.0 

X_live = current_day_data[FEATURES]

# Fill any remaining NaNs in X_live that might exist due to lookback requirements for early candles
X_live = X_live.fillna(0) # CRITICAL FIX: Impute NaNs in X_live before prediction

# Check if X_live is empty after feature selection and NaN imputation
if X_live.empty:
    st.warning("No data remaining after feature selection and NaN imputation. This might indicate an issue with feature calculation or insufficient historical data provided by Yahoo Finance for the selected date.")
    st.stop()

probs = model.predict_proba(X_live)[:, 1]
current_day_data['probs'] = probs

current_day_data['buy_signal'] = (current_day_data['probs'] > 0.60)
current_day_data['sell_signal'] = (current_day_data['probs'] < 0.38)

# 3. UI RENDERING
latest_row = current_day_data.iloc[-1]

# Top Metrics Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nifty 50 Close", f"{latest_row['close']:.2f}")
col2.metric("Stochastic K", f"{latest_row['stoch_k']:.1f}%")
col3.metric("EMA 9", f"{latest_row['ema_9']:.2f}")
col4.metric("Last Update", latest_row.name.strftime("%H:%M"))

# 4. INTERACTIVE CHARTING (Plotly)
fig = go.Figure()

# Add Candlesticks
fig.add_trace(go.Candlestick(
    x=current_day_data.index, open=current_day_data['open'], high=current_day_data['high'],
    low=current_day_data['low'], close=current_day_data['close'], name="Nifty 50"
))

# Add EMA 9 Line
fig.add_trace(go.Scatter(x=current_day_data.index, y=current_day_data['ema_9'], line=dict(color='orange', width=1.5), name="9 EMA"))

# Add Buy Signals (Green Arrows)
buys = current_day_data[current_day_data['buy_signal']]
fig.add_trace(go.Scatter(
    x=buys.index, y=buys['low'] * 0.999,
    mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name="BUY"
))

# Add Sell Signals (Red Arrows)
sells = current_day_data[current_day_data['sell_signal']]
fig.add_trace(go.Scatter(
    x=sells.index, y=sells['high'] * 1.001,
    mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name="SELL"
))

fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Latest Alerts Section
st.subheader("🔔 Recent Trade Alerts")
alerts = current_day_data[current_day_data['buy_signal'] | current_day_data['sell_signal']]

# Ensure 'datetime' is a column for the table, not just the index
alerts_display = alerts.reset_index().rename(columns={'index': 'datetime'})

if not alerts_display.empty:
    # Display only the last 5 alerts
    st.table(alerts_display.tail(5)[['datetime', 'close', 'stoch_k', 'buy_signal', 'sell_signal']])
else:
    st.info("No active signals found.")
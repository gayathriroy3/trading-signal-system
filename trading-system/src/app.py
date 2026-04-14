import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import pytz
import glob

from features.build_features import get_and_process_data
from features.indicators import *
from config.config import MODEL_PATH,FEATURES
from data.refresh_data import load_predict
from datetime import datetime,date

# 1. SETUP: Page configuration & Auto-refresh (every 300,000ms = 5 minutes)
st.set_page_config(page_title="Nifty 50 Intraday Signal Dashboard", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

st.title("📊 Nifty 50 Real-Time Signal Dashboard")
st.markdown("This dashboard updates automatically every 5 minutes to scan for your trader's setup.")
st.info(f"Last dashboard refresh: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}") 

# Dynamic date selection
date_option = st.radio(
    "Select date for analysis:",
    ('Today', 'Custom Date'),
    index=0 # Default to 'Today'
)

if date_option == 'Today':
    selected_date = date.today()
else:
    selected_date = st.date_input("Choose a date:", value=date.today(), max_value=date.today()) # Added max_value to restrict future dates

# Model Selection
model_files = glob.glob('*.joblib') # Get all .joblib files
if not model_files:
    st.error("No .joblib model files found in the directory. Please train and save a model first.")
    st.stop()

selected_model_file = st.selectbox("Select a model to use:", model_files)
try:
    model = joblib.load(MODEL_PATH+selected_model_file)
except FileNotFoundError:
    st.error(f"Model file {MODEL_PATH+selected_model_file} not found. Please train the model using main.py first.")
    st.stop()


nifty_predict,start_date_for_yf,end_date_for_yf=load_predict(selected_date)

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

X_live = current_day_data[FEATURES]

X_live = X_live.fillna(0)

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
col2.metric("Stochastic K", f"{latest_row['stoch_k']:.1f}")
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

# Add Stoch K line
fig_stoch=go.Figure()
fig_stoch.add_trace(go.Scatter(x=current_day_data.index,y=current_day_data['stoch_k'],line=dict(color='grey',width=1.5),name='Stoch K'))
fig_stoch.update_layout(height=300,template='plotly_dark',title='Stochastic Osillator')
st.plotly_chart(fig_stoch,use_container_width=True)


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
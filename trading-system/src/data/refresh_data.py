from src.config.config import NIFTY,MODEL_PATH,FEATURES,REALTIME_PREDICT_DAY
from pandas import pd
import yfinance as yf
from datetime import timedelta,date

def load_predict():
    selected_date = date.today()
    end_date_for_yf = pd.Timestamp(selected_date) + timedelta(days=1)
    start_date_for_yf = end_date_for_yf - timedelta(days=7) 
    nifty_predict = yf.download(NIFTY, start=start_date_for_yf, end=end_date_for_yf, interval="5m")
    return nifty_predict,start_date_for_yf,selected_date
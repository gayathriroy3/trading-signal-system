# Nifty 50 Intraday Trading Signal Dashboard

This project provides a Streamlit-based dashboard for generating real-time intraday trading signals for Nifty 50, based on technical indicators and a trained machine learning model.

## Project Structure

```
my_trading_project/
├── requirements.txt
├── README.md
├── main.py
├── app.py
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── models/
│   └── model_2.joblib # This will be generated after running main.py
├── NIFTY 50_5minute.csv # Placeholder for historical data
```

## Setup and Installation

1.  **Clone the repository (if on your local machine):**
    ```bash
    git clone <repository_url>
    cd my_trading_project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place your historical data:**
    Ensure your historical NIFTY 50 5-minute data (e.g., `NIFTY 50_5minute.csv`) is in the project root directory.

4.  **Train the model:**
    Run the `main.py` script to preprocess data, train the model, and save it to the `models/` directory.
    ```bash
    python main.py
    ```

## Running the Streamlit Dashboard

After training the model, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will open the dashboard in your web browser, displaying real-time Nifty 50 signals for the hardcoded date (currently April 10, 2026).

## Modules Overview

-   **`main.py`**: Orchestrates the data loading, preprocessing, model training, and evaluation pipeline.
-   **`app.py`**: The Streamlit application that fetches live data, applies feature engineering, generates signals using the trained model, and visualizes them.
-   **`src/data_processing.py`**: Contains functions for cleaning and preparing the raw Nifty 50 data.
-   **`src/feature_engineering.py`**: Houses functions to compute various technical indicators and custom features from the raw data.
-   **`src/model_training.py`**: Includes functions for creating binary targets, training the XGBoost model, generating trading signals, and backtesting strategies.
-   **`src/utils.py`**: Contains utility functions like `compute_rsi` and `compute_stochastic` that are used across different modules.
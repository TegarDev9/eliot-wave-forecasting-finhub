# Elliott Wave Forecasting & FinHub Analytics

This Streamlit application combines a rule-based Elliott Wave detector with a LightGBM quantile regression forecaster. It lets traders and analysts explore hybrid technical/statistical insights for assets fetched from Yahoo Finance.

## Features

- **Elliott Wave Scanner** – Automatically finds pivot structures, scores candidate patterns, and visualizes wave counts on an interactive candlestick chart.
- **Machine Learning Forecasts** – Uses quantile regression with LightGBM to predict median, lower, and upper bounds for future prices.
- **Risk Management Toolkit** – Converts detected wave geometry into trade ideas with entry, stop-loss, take-profit, and position sizing guidance.
- **Macroeconomic Context** – Incorporates DXY, VIX, and TNX to enrich the predictive features beyond price action.

## Getting Started

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run eliotwave_ml_app.py
   ```

3. Open the provided local URL in your browser to interact with the dashboard.

## Disclaimer

This project is for educational and analytical purposes only. It does **not** constitute financial advice. Always perform independent research and manage risk appropriately.

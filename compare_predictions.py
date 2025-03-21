import joblib
import numpy as np
import yfinance as yf
import pandas as pd

#  Model & Scaler Load Karo
model = joblib.load("stock_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

#  Stocks List 
stock_symbols = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "NVDA", "META", "NFLX", "AMD", "IBM",
                 "INTC", "ORCL", "CSCO", "PYPL", "QCOM", "TXN", "ADBE", "AVGO", "CRM", "BA",
                 "NKE", "MCD", "V", "MA", "DIS", "JPM", "GS", "MS", "BAC", "C", "WMT",
                 "HD", "LOW", "TGT", "COST", "XOM", "CVX", "PEP", "KO", "ABBV", "JNJ",
                 "PFE", "UNH", "TMO", "LIN", "DHR", "BMY", "RTX", "LMT"]

#  Result Store
results = []

for stock_symbol in stock_symbols:
    try:
        #  Latest Real Data Fetch 
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="2d")  # Latest 2 days ka data le rahe hain
        latest_data = hist.iloc[-1]  # Most Recent Day ka Data

        #  Real Close Price
        real_price = round(float(latest_data["Close"]), 2)

        #  Model  Features Prepare
        features = np.array([
            latest_data["Open"],
            latest_data["High"],
            latest_data["Low"],
            latest_data["Close"],
            latest_data["Volume"],
            latest_data["Close"] - latest_data["Open"],  # Daily_Change
            hist["Close"].rolling(10).mean().iloc[-1]  # MA_10
        ]).reshape(1, -1)

        # Scale Features
        features_scaled = scaler.transform(features)

        # Prediction Lo
        predicted_price = round(float(model.predict(features_scaled)[0]), 2)

        #  Error Calculate 
        error = round(abs(predicted_price - real_price), 2)

        # Results Store 
        results.append([stock_symbol, real_price, predicted_price, error])

        print(f"{stock_symbol} - Real: ${real_price} | Predicted: ${predicted_price} | Error: ${error}")

    except Exception as e:
        print(f"⚠️ Error fetching data for {stock_symbol}: {e}")

# DataFrame 
df_results = pd.DataFrame(results, columns=["Stock", "Real Price", "Predicted Price", "Error"])

#  Results 
df_results.to_csv("comparison_results.csv", index=False)

print("\nComparison Completed! Results saved in 'comparison_results.csv'")

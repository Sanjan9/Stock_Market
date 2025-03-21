import joblib
import numpy as np
import yfinance as yf
import pandas as pd

# Load Model
model = joblib.load("stock_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

stocks = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"]

results = []

for stock_symbol in stocks:
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="2d")

    if len(hist) < 2:
        print(f"⚠ Not enough data for {stock_symbol}")
        continue

    latest_data = hist.iloc[-2]
    actual_price = hist.iloc[-1]["Close"]

    features = np.array([
        latest_data["Open"], latest_data["High"], latest_data["Low"], latest_data["Close"], latest_data["Volume"],
        latest_data["Close"] - latest_data["Open"]
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    predicted_price = round(float(model.predict(features_scaled)[0]), 2)

    results.append({"Stock": stock_symbol, "Real_Close": actual_price, "Predicted_Close": predicted_price})

df = pd.DataFrame(results)
df.to_csv("backtest_results.csv", index=False)
print("\n Backtest Completed! Results saved in 'backtest_results.csv'")

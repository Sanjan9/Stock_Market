import joblib
import numpy as np
import yfinance as yf
import pandas as pd

# Load the trained model & scalers
model = joblib.load("stock_model.pkl")  
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# List of 50 Stock Symbols (as per your project)
stock_symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "IBM",
    "INTC", "ORCL", "CSCO", "PYPL", "QCOM", "TXN", "ADBE", "AVGO", "CRM", "BA", "NKE",
    "MCD", "V", "MA", "DIS", "JPM", "GS", "MS", "BAC", "C", "WMT", "HD", "LOW", "TGT",
    "COST", "XOM", "CVX", "PEP", "KO", "ABBV", "JNJ", "PFE", "UNH", "TMO", "LIN",
    "DHR", "BMY", "RTX", "LMT"
]

# Define timeframe for backtesting (past 30 days)
timeframe = "30d"

# Create a DataFrame to store results
all_results = []

# Loop through each stock to fetch data & predict
for stock_symbol in stock_symbols:
    print(f"Fetching data & predicting for {stock_symbol}...")

    try:
        # Fetch historical data from Yahoo Finance
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period=timeframe)

        # Ensure enough data
        if len(hist) < 10:
            print(f"Not enough data for {stock_symbol}, skipping...")
            continue

        # Store results
        dates, actual_prices, predicted_prices, symbols = [], [], [], []

        # Predict for each day
        for i in range(len(hist) - 1):  # Predict next day
            today_data = hist.iloc[i]

            # Prepare input features (Open, High, Low, Close, Volume)
            features_today = np.array([[today_data["Open"], today_data["High"], today_data["Low"], today_data["Close"], today_data["Volume"]]])
            features_scaled = feature_scaler.transform(features_today)

            # Model Prediction
            predicted_price_scaled = model.predict(features_scaled)[0]
            predicted_price = target_scaler.inverse_transform([[predicted_price_scaled]])[0][0]

            # Store results
            actual_prices.append(hist.iloc[i + 1]["Close"])
            predicted_prices.append(predicted_price)
            dates.append(hist.index[i + 1].strftime("%Y-%m-%d"))
            symbols.append(stock_symbol)

        # Append results to main list
        stock_results = pd.DataFrame({
            "Stock": symbols,
            "Date": dates,
            "Actual Price": actual_prices,
            "Predicted Price": predicted_prices
        })
        all_results.append(stock_results)

    except Exception as e:
        print(f"Error processing {stock_symbol}: {e}")

# Combine results for all stocks into a single DataFrame
final_results = pd.concat(all_results, ignore_index=True)

# Save to CSV
final_results.to_csv("backtest_50_companies.csv", index=False)

# Save to Excel
final_results.to_excel("backtest_50_companies.xlsx", index=False)

print("\nBacktesting Completed for 50 Stocks.")
print(final_results.head())
print("\nResults saved to 'backtest_50_companies.csv' and 'backtest_50_companies.xlsx'")

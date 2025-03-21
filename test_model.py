import joblib
import numpy as np
import yfinance as yf

# Load trained model and scalers
model = joblib.load("stock_model.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

stocks = ["AAPL", "TSLA"]

for stock in stocks:
    hist = yf.Ticker(stock).history(period="10d")
    if hist.empty:
        print(f"⚠️ No data found for {stock}")
        continue

    today_data = hist.iloc[-2]  # Yesterday's data for prediction
    actual_price = hist.iloc[-1]["Close"]

    # Feature Extraction
    features_today = np.array([[today_data["Open"], today_data["High"], today_data["Low"], today_data["Close"], today_data["Volume"]]])
    features_scaled = feature_scaler.transform(features_today)

    # Model Prediction
    predicted_scaled = model.predict(features_scaled)[0]
    predicted_price = target_scaler.inverse_transform([[predicted_scaled]])[0][0]

    print(f"{stock} - Actual: ${actual_price:.2f} | Predicted: ${predicted_price:.2f}")

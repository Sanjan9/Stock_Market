import joblib
import numpy as np
import yfinance as yf

# ✅ Load the trained model
model_path = "stock_rf_model.pkl"

try:
    loaded_model = joblib.load(model_path)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 🚨 Skipping Scaler Since It Doesn't Exist
scaler = None
print("⚠️ Warning: No scaler found. Proceeding without scaling.")

# ✅ Get recent stock data for AAPL
stock_symbol = "AAPL"
stock = yf.Ticker(stock_symbol)
hist = stock.history(period="5d")  # Get last 5 days of stock data

if hist.empty:
    print("❌ Error: Stock data not found!")
    exit()

# ✅ Use the most recent stock data
latest_data = hist.iloc[-1]
print(f"✅ Latest Stock Data:\n{latest_data}")

# ✅ Extract features for prediction
features = np.array([
    latest_data["Open"],   # Opening price
    latest_data["High"],   # Highest price of the day
    latest_data["Low"],    # Lowest price of the day
    latest_data["Close"],  # Closing price
    latest_data["Volume"]  # Trading volume
]).reshape(1, -1)

print(f"📊 Raw Features Before Scaling: {features}")

# 🚨 Skipping Scaling Step
# if scaler:
#     features = scaler.transform(features)
#     print(f"📊 Scaled Features: {features}")

# ✅ Make the prediction
predicted_price = loaded_model.predict(features)[0]
predicted_price = round(float(predicted_price), 2)

print(f"📈 Predicted Next Day Close Price for {stock_symbol}: ${predicted_price}")

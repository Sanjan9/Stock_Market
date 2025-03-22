from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import yfinance as yf
import os
import urllib.request

app = Flask(__name__)

# ✅ GitHub Releases Direct Download Link for the Model
model_url = "https://github.com/Sanjan9/Stock_Market/releases/download/v1.0/stock_model.pkl"
model_path = "stock_model.pkl"

# ✅ Automatically Download Model if It's Missing
if not os.path.exists(model_path):
    print("🔽 Downloading stock_model.pkl from GitHub Releases...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("✅ Model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

# ✅ Load Model & Scalers (Your Original Code)
model = joblib.load("stock_model.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

print("✅ Model and Scalers Loaded Successfully!")

# ✅ Serve HTML Page
@app.route("/")
def home():
    return send_file("index.html")

# ✅ Stock Prediction API (POST)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        stock_symbol = data.get("symbol", "").upper()

        if not stock_symbol:
            return jsonify({"error": "Stock symbol is required!"}), 400

        # Fetch Stock Data
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="10d")

        if hist.empty:
            return jsonify({"error": f"No stock history data for {stock_symbol}!"})

        today_data = hist.iloc[-2]
        actual_price = hist.iloc[-1]["Close"]

        # ✅ Keep Your Original Feature Selection
        features_today = np.array([[today_data["Open"], today_data["High"], today_data["Low"], today_data["Close"], today_data["Volume"]]])

        # Scale input data
        features_scaled = feature_scaler.transform(features_today)

        # Model Prediction
        predicted_price_scaled = model.predict(features_scaled)[0]
        predicted_price = target_scaler.inverse_transform([[predicted_price_scaled]])[0][0]

        return jsonify({
            "symbol": stock_symbol,
            "actual_price": round(float(actual_price), 2),
            "predicted_price": round(float(predicted_price), 2)
        })

    except Exception as e:
        return jsonify({"error": f"Internal error: {e}"}), 500

# ✅ Stock History API
@app.route("/history/<symbol>")
def history(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="30d")

        if hist.empty:
            return jsonify({"error": f"No stock history data for {symbol}!"})

        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist["Close"].tolist()

        return jsonify({"dates": dates, "prices": prices})

    except Exception as e:
        return jsonify({"error": f"Internal error: {e}"}), 500

# ✅ Correct Port Handling for Render Deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port)

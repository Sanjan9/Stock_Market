import yfinance as yf
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Fetch stock data
def get_stock_data(symbol, period="5y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)

    if hist.empty:
        print(f"⚠️ No data found for {symbol}")
        return None

    hist["Next_Day_Close"] = hist["Close"].shift(-1)  # Target variable
    hist.dropna(inplace=True)

    return hist[["Open", "High", "Low", "Close", "Volume", "Next_Day_Close"]]

# ✅ Stock List
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

# Fetch and merge data
data_frames = [get_stock_data(symbol) for symbol in tickers if get_stock_data(symbol) is not None]
if not data_frames:
    print("❌ No valid stock data found! Exiting.")
    exit()

df = pd.concat(data_frames).dropna()

# ✅ Define Features & Target
features = df[["Open", "High", "Low", "Close", "Volume"]]
target = df["Next_Day_Close"]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# ✅ Feature & Target Scaling
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# ✅ Train Model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

# ✅ Save Model & Scalers
joblib.dump(model, "stock_model.pkl")
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# ✅ Evaluate Model
y_pred_scaled = model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print("\n📊 Model Evaluation:")
print(f"🔹 MAE: {mae:.2f}")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")
print("✅ Model Training Completed Successfully!")

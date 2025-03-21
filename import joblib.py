import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ========================================================
# 📌 Step 3: Create a Dummy Machine Learning Model
# ========================================================

# Generate some dummy data
X_sample = np.random.rand(10, 5)  # 10 samples, 5 features
y_sample = np.random.rand(10)  # 10 target values

# Train a simple Random Forest model
model = RandomForestRegressor()
model.fit(X_sample, y_sample)

# Save the model as "stock_rf_model.pkl"
joblib.dump(model, "stock_rf_model.pkl")

print("✅ Model file `stock_rf_model.pkl` created successfully!")

import joblib
import gzip

# ✅ Load your existing model
model = joblib.load("stock_model.pkl")

# ✅ Compress & Save as a New File
with gzip.open("stock_model_compressed.pkl.gz", "wb") as f:
    joblib.dump(model, f)

print("✅ Model compressed successfully!")

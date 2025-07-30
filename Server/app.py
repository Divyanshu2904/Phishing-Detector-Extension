from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from feature_extractor import extract_features_from_url  # Your 48-feature extractor

app = Flask(__name__)
CORS(app)

# ── File paths ─────────────────────────────────────────
MODEL_PATH   = "trained_model.pkl"
DATASET_PATH = "Phishing_Legitimate_full.csv"

# ── Train Model using real-time extractor ─────────────
def train_model():
    print("📊 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()

    # Ensure required columns
    if 'url' not in df.columns or 'ClassLabel' not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise ValueError("CSV must contain 'url' and 'ClassLabel' columns")

    # ✅ Filter labels: 0 = legit, 1 = phishing
    df = df[df['ClassLabel'].isin([0, 1])]
    df = df.dropna(subset=["url"])
    df = df.head(201361)  # Limit to 201361 rows

    # Extract features from URLs
    print("🧠 Extracting features from URLs...")
    X = df['url'].apply(extract_features_from_url).tolist()
    X = pd.DataFrame(X)

    # Ensure labels are int
    y = df['ClassLabel'].astype(int)

    # Sanity check
    assert X.shape[1] == 48, f"Expected 48 features, got {X.shape[1]}"
    print(f"🎯 Training on {len(X)} samples with {X.shape[1]} features")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump((model, list(X.columns)), MODEL_PATH)
    print("✅ Model trained and saved to:", MODEL_PATH)

# ── Train if model doesn't exist ───────────────────────
if not os.path.exists(MODEL_PATH):
    train_model()

# ── Load model ─────────────────────────────────────────
model, feature_names = joblib.load(MODEL_PATH)

# ── Prediction Endpoint ────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extract same 48 features
    feats = extract_features_from_url(url)
    if len(feats) != len(feature_names):
        return jsonify({"error": f"Feature count mismatch: got {len(feats)}, expected {len(feature_names)}"}), 400

    X_test = pd.DataFrame([feats], columns=feature_names)
    pred   = model.predict(X_test)[0]
    proba  = model.predict_proba(X_test)[0]
    label  = "phishing🔴" if pred == 1 else "legit🟢"
    conf   = round(max(proba) * 100, 2)

    return jsonify({"prediction": label, "confidence": f"{conf}%"})

# ── Run App ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

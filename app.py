from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os
from flask_cors import CORS

# ---------------------------
#  PATH CONFIG
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ---------------------------
#  LOAD MODEL + ENCODERS
# ---------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.joblib")
feature_columns = joblib.load("feature_columns.joblib")

# ---------------------------
#  BLOCKED IP STORAGE (RAM)
# ---------------------------
blocked_ips = set()

# ---------------------------
#  SAFE ENCODER FUNCTION
# ---------------------------
def safe_transform(encoder, value):
    """Avoid crashes when unseen categorical values appear."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1   # unseen → treat as unknown class


# ---------------------------
#  SERVE FRONTEND
# ---------------------------
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------
#  PREDICTION API
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # IP check
    ip = data.get("ip")
    if not ip:
        return jsonify({"error": "IP is required"}), 400

    if ip in blocked_ips:
        return jsonify({"status": "blocked", "reason": "IP already blocked"}), 403

    # Create dataframe
    df = pd.DataFrame([data])

    # Encode categorical columns safely
    for col in encoders:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_transform(encoders[col], x))
        else:
            return jsonify({"error": f"Missing column {col}"}), 400

    # Apply one-hot alignment
    df = pd.get_dummies(df)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # Predict
    features = df.values
    prediction = model.predict(features)[0]

    # Attack detected
    if prediction == 1:
        blocked_ips.add(ip)
        return jsonify({"prediction": "attack", "action": "IP blocked"})

    # Normal traffic
    return jsonify({"prediction": "normal", "action": "allowed"})


# ---------------------------
#  CHECK BLOCKED IPs
# ---------------------------
@app.route("/blocked", methods=["GET"])
def blocked():
    return jsonify(list(blocked_ips))


# ---------------------------
#  RUN SERVER
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

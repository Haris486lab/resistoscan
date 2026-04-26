from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

from model import predict_environment, predict_iti

app = Flask(__name__)
CORS(app)

# -------------------------
# Home route
# -------------------------
@app.route("/")
def home():
    return "ResistoScan Backend Running 🚀"

# -------------------------
# Prediction route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        env_pred = predict_environment(df)
        iti_score = predict_iti(df)

        return jsonify({
            "environment": int(env_pred),
            "iti_score": float(iti_score)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------
# Run server (Render compatible)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, jsonify
import pandas as pd
from model import predict_environment, predict_iti

app = Flask(__name__)

@app.route("/")
def home():
    return "ResistoScan Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)

    env_pred = predict_environment(df)
    iti_score = predict_iti(df)

    return jsonify({
        "environment": int(env_pred),
        "iti_score": float(iti_score)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

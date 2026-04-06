from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# ✅ HOME ROUTE (Frontend UI)
@app.route("/")
def home():
    return render_template("index.html")

# ✅ UPLOAD API (Your ML logic)
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    df = pd.read_csv(file)

    # Example ITI calculation (replace with your logic if needed)
    iti_score = df.sum().sum()

    if iti_score > 40000:
        risk = "Critical"
    elif iti_score > 20000:
        risk = "High"
    else:
        risk = "Moderate"

    return jsonify({
        "ITI": round(iti_score, 2),
        "Risk": risk,
        "Samples": len(df)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

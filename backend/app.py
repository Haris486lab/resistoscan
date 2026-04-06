from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# ✅ DEFINE APP FIRST (VERY IMPORTANT)
app = Flask(__name__, template_folder="templates")


# ✅ ITI FUNCTION
def calculate_iti(df):
    resistome_density = df.sum().sum()
    pathogen_load = (df > 0).sum().sum() / df.size
    iti = resistome_density * (1 + pathogen_load)

    if iti < 100:
        risk = "Low"
    elif iti < 300:
        risk = "Moderate"
    elif iti < 700:
        risk = "High"
    else:
        risk = "Critical"

    return iti, risk


# ✅ PLOT FUNCTION
def generate_plots(df):
    images = {}

    # Heatmap
    plt.figure(figsize=(6,4))
    sns.heatmap(df.iloc[:10, :10], cmap="viridis")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    images["heatmap"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Bar plot
    plt.figure(figsize=(6,4))
    df.sum().plot(kind="bar")
    plt.xticks(rotation=90)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    images["barplot"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return images


# ✅ HOME ROUTE
@app.route("/")
def home():
    return render_template("index.html")


# ✅ UPLOAD ROUTE
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    df = pd.read_csv(file)

    iti, risk = calculate_iti(df)
    plots = generate_plots(df)

    return jsonify({
        "ITI": round(iti, 2),
        "Risk": risk,
        "Samples": len(df),
        "heatmap": plots["heatmap"],
        "barplot": plots["barplot"]
    })


# ✅ RUN
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

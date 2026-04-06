from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

# ✅ FIRST CREATE APP
app = Flask(__name__, template_folder="templates")


# ✅ HOME ROUTE
@app.route("/")
def home():
    return render_template("index.html")


# ✅ SIMPLE TEST UPLOAD (no ML for now)
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    df = pd.read_csv(file)

    return jsonify({
        "message": "File received",
        "rows": len(df)
    })


# ✅ RUN
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

return jsonify({
    "ITI": round(iti, 2),
    "Risk": risk,
    "Samples": len(df)
})

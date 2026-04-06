from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

# Create app FIRST
app = Flask(__name__, template_folder="templates")


# Home route
@app.route("/")
def home():
    return render_template("index.html")


# Upload route
@app.route("/upload", methods=["POST"])
def upload():

    def calculate(df):
        iti = df.sum().sum()
        if iti < 100:
            risk = "Low"
        elif iti < 300:
            risk = "Moderate"
        elif iti < 700:
            risk = "High"
        else:
            risk = "Critical"
        return iti, risk

    file1 = request.files.get("file1")
    file2 = request.files.get("file2")

    df1 = pd.read_csv(file1)
    iti1, risk1 = calculate(df1)

    # If second dataset exists → compare
    if file2 and file2.filename != "":
        df2 = pd.read_csv(file2)
        iti2, risk2 = calculate(df2)

        return jsonify({
            "comparison": True,
            "d1": {"iti": round(iti1,2), "risk": risk1},
            "d2": {"iti": round(iti2,2), "risk": risk2}
        })

    # Otherwise single dataset
    return jsonify({
        "comparison": False,
        "iti": round(iti1,2),
        "risk": risk1
    })
   

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

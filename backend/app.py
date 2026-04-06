@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    iti, risk = calculate_iti(df)

    return jsonify({
        "ITI": round(iti, 2),
        "Risk": risk,
        "Samples": len(df)
    })
